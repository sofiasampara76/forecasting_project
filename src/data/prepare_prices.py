from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH = PROJECT_ROOT / "data" / "raw" / "136.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "avg_prices_monthly.csv"

# Індекси потрібних колонок (колонку 20 з довгими примітками навмисно пропускаємо)
USECOLS = [0, 4, 5, 6, 7, 8, 10, 11, 13, 21]
NAMES = ["indicator", "region_code", "region", "good_code", "good",
         "freq", "period", "value", "unit", "obs_note"]

chunks = []
total_rows = 0

reader = pd.read_csv(
    PATH,
    header=0,            # перший рядок із дубльованими назвами замінюємо своїми
    names=NAMES,
    usecols=USECOLS,
    dtype=str,
    encoding="utf-8-sig",
    chunksize=100_000,
)

for chunk in reader:
    total_rows += len(chunk)
    # лишаємо тільки середні ціни, місячні
    chunk = chunk[(chunk["indicator"] == "AVG_CONS_PRCS") & (chunk["freq"] == "M")]
    chunks.append(chunk)

p = pd.concat(chunks, ignore_index=True)
print(f"Прочитано рядків усього: {total_rows:,}")
print(f"Лишилось (AVG_CONS_PRCS, місячні): {len(p):,}")

# типи
p["value"] = pd.to_numeric(p["value"], errors="coerce")
p["date"] = pd.to_datetime(p["period"].str.replace("-M", "-", regex=False) + "-01")

# зберігаємо компактну версію, далі працюємо тільки з нею
p.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

# ---------- профайлінг ----------
print("\nПеріод:", p["date"].min().date(), "→", p["date"].max().date())
print("Регіонів:", p["region"].nunique(), "| товарів:", p["good"].nunique())

g = (p.groupby(["region", "good"])
       .agg(n=("value", "size"),
            n_missing=("value", lambda s: s.isna().sum()),
            start=("date", "min"),
            end=("date", "max"),
            unit=("unit", "first")))
print("\nРозподіл довжини рядів:")
print(g["n"].describe())

print("\nТоп-40 товарів по Україні за довжиною ряду:")
print(g.loc["Україна"].sort_values("n", ascending=False).head(40).to_string())

print("\nПрапорці приміток до спостережень:")
print(p["obs_note"].value_counts(dropna=False))