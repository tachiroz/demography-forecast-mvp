from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd, numpy as np, io
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = FastAPI(title="Demography MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # dev-режим
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────── helpers ───────────────────────────
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def mape(a, b): return float(np.mean(np.abs((a - b) / a)) * 100)

# ───────────────────────── /upload-train ─────────────────────
def _load(file: UploadFile, target: str) -> pd.DataFrame:
    """
    • target  – имя числового столбца, который нужен ('Birth','Death',...)
    • в файле может быть любой заголовок,
      0-я колонка = Year, target-колонка = нужное значение
    """
    df = pd.read_csv(file.file)

    # убираем кавычки и пробелы
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.applymap(lambda x: str(x).strip().strip('"'))

    # оставляем Year и target
    df = df[["Year", target]].astype(int)
    return df

@app.post("/upload-train/")
async def upload_and_train(
    births: UploadFile = File(...),
    deaths: UploadFile = File(...),
    population: UploadFile = File(...),
    migration: UploadFile = File(...),
    model: str = Form("sarimax_pop"),
):
    b = _load(births,     "Birth")
    d = _load(deaths,     "Death")
    p = _load(population, "Population")
    m = _load(migration,  "M_come").merge(_load(migration, "M_out"), on="Year")
    m["Migration"] = m["M_come"] - m["M_out"]
    m = m[["Year", "Migration"]]

    # агрегируем по годам
    df = (
        p.groupby("Year",as_index=False)["Population"].sum()
         .merge(b.groupby("Year",as_index=False)["Birth"].sum(),      on="Year")
         .merge(d.groupby("Year",as_index=False)["Death"].sum(),      on="Year")
         .merge(m.groupby("Year",as_index=False)["Migration"].sum(),  on="Year")
         .sort_values("Year")
    )


# ───────────────────────── /forecast ─────────────────────────
@app.post("/forecast/")
async def forecast(
    file: UploadFile = File(...),
    model: str = "sarimax_pop",
    years: int = Query(5, ge=1, le=22),     # ≤22 лет (до 2046)
):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content)).fillna(0)

    train = df[df.Year <= 2023].set_index("Year")   # используем всё до 2023
    last_year = 2023
    future_years = list(range(last_year + 1, last_year + years + 1))

    if model == "sarimax_pop":
        y_tr = train.Population
        X_tr = train[["Birth", "Death", "Migration"]]

        last_exog = X_tr.tail(1).to_numpy().repeat(years, axis=0)
        X_future  = pd.DataFrame(last_exog, columns=X_tr.columns)

        mdl = SARIMAX(
            y_tr, exog=X_tr,
            order=(1,1,1), seasonal_order=(0,1,1,1),
            enforce_invertibility=False
        ).fit(disp=False)
        y_pred = mdl.forecast(steps=years, exog=X_future)
    else:  # sarimax (Births)
        y_tr = train.Birth
        mdl = SARIMAX(y_tr, order=(1,1,1), seasonal_order=(0,1,1,1)).fit(disp=False)
        y_pred = mdl.forecast(steps=years)

    return JSONResponse({
        "Year": future_years,
        "y_pred": y_pred.tolist(),
    })
