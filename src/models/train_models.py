"""
train_models.py
Обучает выбранную модель (сейчас только SARIMAX) и сохраняет:
  • models/<model>.pkl
  • reports/metrics_<model>.csv
  • reports/preds_<model>.csv
Запуск:
  python src/models/train_models.py --model sarimax
  python src/models/train_models.py --model sarimax --params '{"order":[2,1,0]}'
"""

import argparse, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ──────────────────────────── пути ────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "clean"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"
MODEL_DIR.mkdir(exist_ok=True)   # если каталогов нет — создаём
REPORT_DIR.mkdir(exist_ok=True)

# ──────────────────────────── данные ──────────────────────────────────────
births = pd.read_parquet(DATA / "births_total.parquet")
train = births[births.Year <= 2021].set_index("Year")
test  = births[births.Year > 2021].set_index("Year")
y_train, y_test = train.Births_total_year, test.Births_total_year

# ────────────────────────── модели ────────────────────────────────────────
def train_sarimax(params: dict):
    order = tuple(params.get("order", (1, 1, 1)))  # p,d,q
    mdl = SARIMAX(
        y_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    pred = mdl.forecast(len(y_test))
    pred.index = y_test.index
    return mdl, pred


def train_prophet(params):
    df = y_train.reset_index().rename(columns={"Year":"ds","Births_total_year":"y"})
    m = Prophet(**params).fit(df)
    fut = pd.DataFrame({"ds": y_test.index})
    pred = pd.Series(m.predict(fut)["yhat"].values, index=y_test.index)
    return m, pred

def _make_ts_features(n):                # simple “time index” feature
    return np.arange(n).reshape(-1,1)

def train_xgb(params):
    Xtr, Xte = _make_ts_features(len(y_train)), _make_ts_features(len(y_train)+len(y_test))[len(y_train):]
    mdl = XGBRegressor(**params).fit(Xtr, y_train)
    pred = pd.Series(mdl.predict(Xte), index=y_test.index)
    return mdl, pred

def train_cat(params):
    Xtr, Xte = _make_ts_features(len(y_train)), _make_ts_features(len(y_train)+len(y_test))[len(y_train):]
    mdl = CatBoostRegressor(verbose=0, **params).fit(Xtr, y_train)
    pred = pd.Series(mdl.predict(Xte), index=y_test.index)
    return mdl, pred

TRAINERS = {
    "sarimax": train_sarimax,
    "prophet": train_prophet,
    "xgb":     train_xgb,
    "cat":     train_cat,
}

# ──────────────────────── метрики ─────────────────────────────────────────
def evaluate(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "MSE":  mean_squared_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "R2":   r2_score(y_true, y_pred),
    }

# ──────────────────────────── CLI ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=TRAINERS.keys())
    parser.add_argument("--params", default="{}",
                        help="JSON-строка с гиперпараметрами модели")
    args = parser.parse_args()
    params = json.loads(args.params)

    model, y_pred = TRAINERS[args.model](params)

    # метрики
    metrics = evaluate(y_test, y_pred)
    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / f"metrics_{args.model}.csv", index=False)

    # предсказания для графика
    pd.DataFrame({
        "Year": y_test.index,
        "y_true": y_test.values,
        "y_pred": y_pred.values,
    }).to_csv(REPORT_DIR / f"preds_{args.model}.csv", index=False)

    # модель
    joblib.dump(model, MODEL_DIR / f"{args.model}.pkl")

    print(f"[OK] {args.model} saved, metrics:", metrics)
