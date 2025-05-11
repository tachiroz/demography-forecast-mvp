# backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, json, pathlib, uuid, sys   # ← добавили sys
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse   # вверху файла
from fastapi.encoders import jsonable_encoder
from fastapi import Query


BASE = pathlib.Path(__file__).resolve().parents[2]
app = FastAPI()

class TrainRequest(BaseModel):
    model: str
    params: dict | None = {}

@app.post("/train")
def train(req: TrainRequest):
    run_id = uuid.uuid4().hex[:8]
    cmd = [
        sys.executable,                       # ← используем тот Python,
        str(BASE / "src/models/train_models.py"),  #   в котором крутится FastAPI
        "--model", req.model,
        "--params", json.dumps(req.params),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"status": "error", "detail": proc.stderr}
    return {"status": "ok", "run_id": run_id}

@app.get("/metrics/{model}")
def metrics(model: str):
    """Вернёт словарь {MAE, MSE, MAPE, R2} для выбранной модели."""
    path = BASE / "reports" / f"metrics_{model}.csv"
    if not path.exists():
        return {"status": "error", "detail": "metrics file not found"}
    return pd.read_csv(path).iloc[0].to_dict()

@app.get("/preds/{model}")
def preds(model: str):
    path = BASE / "reports" / f"preds_{model}.csv"
    if not path.exists():
        return {"status": "error", "detail": "preds file not found"}

    df = pd.read_csv(path)

    # NaN/±Inf  →  None (чтобы json.dumps не упал)
    clean = df.replace([np.inf, -np.inf], np.nan).to_dict(orient="list")
    for k, v in clean.items():
        clean[k] = [None if (isinstance(x, float) and np.isnan(x)) else x for x in v]

    # jsonable_encoder гарантированно делает всё JSON-совместимым
    return JSONResponse(content=jsonable_encoder(clean))

@app.get("/forecast/{model}")
def forecast(model: str,
             years: int = Query(5, ge=1, le=22)):   # макс до 2046 (2024+22)
    """
    Возвращает прогноз ещё `years` лет после 2023-го.
    Для *_pop моделей — Population, иначе Births.
    """
    import joblib, numpy as np, pandas as pd

    mdl_path = BASE / "models" / f"{model}.pkl"
    preds_path = BASE / "reports" / f"preds_{model}.csv"
    if not (mdl_path.exists() and preds_path.exists()):
        return {"status": "error", "detail": "model not trained"}

    mdl = joblib.load(mdl_path)
    hist = pd.read_csv(preds_path)        # берём последние известные точки
    last_year = int(hist["Year"].max())
    future_years = list(range(last_year + 1, last_year + years + 1))

    # --- варианты по модели ---
    if model == "sarimax_pop":
        # exog = предположим «нулевая миграция + усред. Birth/Death»
        avg = hist[["y_hist"]].tail(3).mean().values[0]
        exog_future = pd.DataFrame({
            "Birth":      [avg]*years,
            "Death":      [avg]*years,
            "Migration":  [0]*years,
        })
        forecast = mdl.forecast(steps=years, exog=exog_future)
    elif model == "sarimax":
        forecast = mdl.forecast(steps=years)
    else:          # prophet, xgb, cat  работают на Births
        steps = np.arange(len(hist)+1, len(hist)+years+1).reshape(-1,1)
        if model == "prophet":
            df = pd.DataFrame({"ds": future_years})
            forecast = mdl.predict(df)["yhat"].values
        else:      # xgb / cat
            forecast = mdl.predict(steps)

    return {
        "Year": future_years,
        "y_pred": forecast.tolist(),
    }