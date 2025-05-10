# backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, json, pathlib, uuid, sys   # ← добавили sys
import pandas as pd


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
    """Вернёт Year-массив и массивы фактов/прогноза для графика."""
    path = BASE / "reports" / f"preds_{model}.csv"
    if not path.exists():
        return {"status": "error", "detail": "preds file not found"}
    df = pd.read_csv(path)
    return {
        "Year":   df["Year"].tolist(),
        "y_true": df["y_true"].tolist(),
        "y_pred": df["y_pred"].tolist(),
    }