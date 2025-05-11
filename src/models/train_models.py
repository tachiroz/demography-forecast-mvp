"""
train_models.py
Запуск:
  python src/models/train_models.py --model sarimax
  python src/models/train_models.py --model sarimax_pop
  python src/models/train_models.py --model prophet --params '{"seasonality_mode":"multiplicative"}'
"""

from pathlib import Path
import argparse, json, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ──────────────────────────── пути ────────────────────────────
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "clean"
MODEL_DIR, REPORT_DIR = BASE / "models", BASE / "reports"
MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)

# ──────────────────── Births (34 строки) ──────────────────────
births = pd.read_parquet(DATA / "births_total.parquet")
birth_col = "Births_total_year" if "Births_total_year" in births.columns else "Birth"

b_tr = births[births.Year <= 2021].set_index("Year")
b_ts = births[births.Year >= 2022].set_index("Year")
y_b_tr, y_b_ts = b_tr[birth_col], b_ts[birth_col]

# ────────────────── Population + экзогены ─────────────────────
dem = pd.read_parquet(DATA / "demography.parquet").fillna(0)

pop_birth_col = "Birth" if "Birth" in dem.columns else "Births_total_year"

p_tr = dem[dem.Year <= 2021].set_index("Year")
p_ts = dem[dem.Year >= 2022].set_index("Year")

y_p_tr, y_p_ts = p_tr.Population, p_ts.Population
X_p_tr, X_p_ts = (
    p_tr[[pop_birth_col, "Death", "Migration"]].fillna(0),
    p_ts[[pop_birth_col, "Death", "Migration"]].fillna(0),
)

# ───────────────────────── train-функции ──────────────────────
def train_sarimax(params):
    order = tuple(params.get("order", (1, 1, 1)))
    mdl = SARIMAX(y_b_tr, order=order,
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    pred = mdl.forecast(len(y_b_ts)); pred.index = y_b_ts.index
    return mdl, pred, y_b_ts

def train_prophet(params):
    df = y_b_tr.reset_index().rename(columns={"Year": "ds", birth_col: "y"})
    mdl = Prophet(**params).fit(df)
    fut = pd.DataFrame({"ds": y_b_ts.index})
    pred = pd.Series(mdl.predict(fut)["yhat"].values, index=y_b_ts.index)
    return mdl, pred, y_b_ts

def _make_ts_features(n: int) -> np.ndarray:
    """простой индекс времени 0..n-1"""
    return np.arange(n).reshape(-1, 1)

def train_xgb(params):
    Xtr, Xte = _make_ts_features(len(y_b_tr)), _make_ts_features(len(y_b_tr) + len(y_b_ts))[len(y_b_tr):]
    mdl = XGBRegressor(**params).fit(Xtr, y_b_tr)
    pred = pd.Series(mdl.predict(Xte), index=y_b_ts.index)
    return mdl, pred, y_b_ts

def train_cat(params):
    Xtr, Xte = _make_ts_features(len(y_b_tr)), _make_ts_features(len(y_b_tr) + len(y_b_ts))[len(y_b_tr):]
    mdl = CatBoostRegressor(verbose=0, **params).fit(Xtr, y_b_tr)
    pred = pd.Series(mdl.predict(Xte), index=y_b_ts.index)
    return mdl, pred, y_b_ts

def train_sarimax_pop(params):
    order = tuple(params.get("order", (1, 1, 1)))
    seas  = tuple(params.get("seas",  (0, 0, 0, 0)))

    mdl = SARIMAX(
        y_p_tr,
        exog=X_p_tr,
        order=order,
        seasonal_order=seas,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    pred = mdl.predict(
        start=len(y_p_tr),
        end=len(y_p_tr) + len(y_p_ts) - 1,
        exog=X_p_ts
    )
    pred.index = y_p_ts.index
    return mdl, pred, y_p_ts

# ─────────────────────── словарь моделей ──────────────────────
TRAINERS = {
    "sarimax":      train_sarimax,
    "sarimax_pop":  train_sarimax_pop,
    "prophet":      train_prophet,
    "xgb":          train_xgb,
    "cat":          train_cat,
}

# ────────────────────────── метрики ───────────────────────────
def evaluate(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "MSE":  mean_squared_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "R2":   r2_score(y_true, y_pred),
    }

# ─────────────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=TRAINERS.keys())
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    params = json.loads(args.params)

    mdl, y_pred, y_true = TRAINERS[args.model](params)

    # ── метрики ───────────────────────────────────────────────
    target = "Population" if args.model.endswith("_pop") else "Births"
    
    metrics = evaluate(y_true, y_pred)
    metrics["Target"] = target                # ещё один столбец
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / f"metrics_{args.model}.csv", index=False)

    # ── полный исторический ряд + прогноз ─────────────────────
    hist_series = y_b_tr if args.model != "sarimax_pop" else y_p_tr

    # все годы, отсортированные
    hist_years = sorted(set(hist_series.index) | set(y_true.index))

    # объединяем Series через pd.concat (append в pandas ≥2.0 удалён)
    hist_vals = (
        pd.concat([hist_series, y_true])
          .groupby(level=0).first()          # оставляем единственное значение на год
          .reindex(hist_years)               # гарантируем полный диапазон
    )

    pd.DataFrame({
        "Year":   hist_years,
        "y_hist": hist_vals.values,               # вся история (1990-2023)
        "y_true": y_true.reindex(hist_years).values,  # test-часть
        "y_pred": y_pred.reindex(hist_years).values,  # только прогноз
    }).to_csv(REPORT_DIR / f"preds_{args.model}.csv", index=False)

    # ── сохраняем модель ──────────────────────────────────────
    joblib.dump(mdl, MODEL_DIR / f"{args.model}.pkl")
    print(f"[OK] {args.model} saved, metrics:", metrics)
