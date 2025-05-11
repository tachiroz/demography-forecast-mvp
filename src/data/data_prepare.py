# ──────────────────────────────────────────────────────────────
# data_prepare.py  ―  очистка CSV Росстата, формирование parquet
# Запускать из корня репо:  python src/data/data_prepare.py
# ──────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
RAW  = BASE / "data" / "raw"
CLN  = BASE / "data" / "clean"
CLN.mkdir(parents=True, exist_ok=True)

# ────────────────────────── универсальная функция ─────────────
def load_and_clean(fname: str, cols: list[str]) -> pd.DataFrame:
    """
    • строки в CSV заключены в двойные кавычки => убираем их
    • разбиваем по запятой, назначаем имена столбцов
    • всё (кроме потенциальных строковых) → int
    """
    raw = pd.read_csv(RAW / fname, header=None, names=["raw"])
    df  = (
        raw["raw"]
        .str.strip('"')
        .str.split(",", expand=True)
    )
    df.columns = cols

    if df.loc[0, cols[0]] == cols[0]:      # убираем строку с заголовком
        df = df.iloc[1:]

    df = df.astype(int)
    return df

# ─────────────────────────── загрузка датасетов ───────────────
population = load_and_clean(
    "Population.csv", ["Year", "Age", "ID_sex", "ID_np", "Population"]
)
births = load_and_clean(
    "Births.csv", ["Year", "Age", "ID_sex", "ID_np", "Birth"]
)
deaths = load_and_clean(
    "Deaths.csv", ["Year", "Age", "ID_sex", "ID_np", "Death"]
)

# migration может отсутствовать
try:
    migration = load_and_clean(
        "Migration.csv",
        ["Year", "Age", "ID_sex", "ID_np", "M_come", "M_out"]
    )
    migration["Migration"] = migration["M_come"] - migration["M_out"]
    migration = migration.groupby("Year", as_index=False)["Migration"].sum()
except FileNotFoundError:
    print("[WARN] Migration.csv not found – заполняю Migration=0")
    migration = pd.DataFrame({
        "Year": births["Year"].unique(),
        "Migration": 0
    })

# ───────────────────── агрегируем по годам ────────────────────
births_tot = births.groupby("Year", as_index=False)["Birth"].sum()
births_tot.rename(columns={"Birth": "Births_total_year"}, inplace=True)
deaths_tot = deaths.groupby("Year", as_index=False)["Death"].sum()
pop_tot    = population.groupby("Year", as_index=False)["Population"].sum()

df = (
    pop_tot
    .merge(births_tot, on="Year", how="left")
    .merge(deaths_tot, on="Year", how="left")
    .merge(migration,  on="Year", how="left")
    .sort_values("Year")
)

# ─────────────────────────── сохранение ───────────────────────
df.to_parquet(CLN / "demography.parquet", index=False)
births_tot.to_parquet(CLN / "births_total.parquet", index=False)

print("[OK] Сохранил:")
print(" • data/clean/demography.parquet  →", len(df), "строк")
print(" • data/clean/births_total.parquet →", len(births_tot), "строк")
