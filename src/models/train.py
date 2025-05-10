"""
train.py
Запуск:  python src/models/train.py
Обучает SARIMAX на Births_total_year, сохраняет модель и таблицу метрик.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── пути ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "clean"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ── читаем данные ──────────────────────────────────────────────────────────
births = pd.read_parquet(DATA_DIR / "births_total.parquet")

# ── train / test split ─────────────────────────────────────────────────────
train = births[births["Year"] <= 2021].set_index("Year")
test  = births[births["Year"] > 2021].set_index("Year")

y_train = train["Births_total_year"]
y_test  = test["Births_total_year"]

# ── SARIMAX (первые «разумные» параметры) ──────────────────────────────────
model = SARIMAX(
    y_train,
    order=(1, 1, 1),       # p, d, q
    seasonal_order=(0, 0, 0, 0),  # сезонность не берём пока
    enforce_stationarity=False,
    enforce_invertibility=False,
)
model_fit = model.fit(disp=False)

# ── прогноз для test ───────────────────────────────────────────────────────
y_pred = model_fit.forecast(steps=len(y_test))
y_pred.index = y_test.index

# ── метрики ────────────────────────────────────────────────────────────────
metrics = {
    "MAE" : mean_absolute_error(y_test, y_pred),
    "MSE" : mean_squared_error(y_test, y_pred),
    "MAPE": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
    "R2"  : r2_score(y_test, y_pred),
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(REPORT_DIR / "metrics_baseline.csv", index=False)
print("[OK] metrics_baseline.csv →", metrics)

# ── сохраняем модель ───────────────────────────────────────────────────────
import joblib
joblib.dump(model_fit, MODEL_DIR / "baseline_sarimax.pkl")
print("[OK] baseline_sarimax.pkl saved")

