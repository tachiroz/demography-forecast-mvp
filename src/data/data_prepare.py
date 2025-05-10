from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR  = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean(filename: str, col_names: list[str]) -> pd.DataFrame:
    df_raw = pd.read_csv(RAW_DIR / filename, header=None, names=["raw"])
    df = (
        df_raw["raw"]
        .str.strip('"')            # убираем кавычки
        .str.split(",", expand=True)
    )
    df.columns = col_names

    # ── убираем строку-заголовок ──────────────────────────────────────────────
    if df.loc[0, "Year"] == "Year":
        df = df.iloc[1:]

    # ── превращаем строки в числа ────────────────────────────────────────────
    df = df.astype(int)

    return df

population = load_and_clean(
    "Population.csv", ["Year", "Age", "ID_sex", "ID_np", "Population"]
)
births = load_and_clean(
    "Births.csv", ["Year", "Age", "ID_sex", "ID_np", "Birth"]
)
deaths = load_and_clean(
    "Deaths.csv", ["Year", "Age", "ID_sex", "ID_np", "Died"]
)

# ── агрегируем: общее число рождений в году ──────────────────────────────────
births_total = births.groupby("Year", as_index=False)["Birth"].sum()
births_total.rename(columns={"Birth": "Births_total_year"}, inplace=True)

out_file = CLEAN_DIR / "births_total.parquet"
births_total.to_parquet(out_file, index=False)
print(f"[OK] Сохранил {out_file.relative_to(BASE_DIR)} ({len(births_total)} строк)")
