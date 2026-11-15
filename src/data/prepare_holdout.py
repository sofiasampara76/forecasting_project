"""
Final holdout: monthly average prices that the newer XLSX release adds after the CSV snapshot.

The CSV (data/raw/136.csv) ends in 2025-06; the XLSX published later on the same data.gov.ua
page continues the same series. No model or design decision used these months, so they are a
genuine out-of-sample check. Run after src/data/prepare_prices.py.

Output: data/processed/avg_prices_holdout.csv (same columns as avg_prices_monthly.csv).
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
XLSX = PROJECT_ROOT / "data" / "raw" / "136-indeksi-spozhivchikh-tsin.xlsx"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "avg_prices_monthly.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "avg_prices_holdout.csv"

# Same source column positions as prepare_prices.py; the XLSX has 22 columns, like the CSV.
USECOLS = [0, 4, 5, 6, 7, 8, 10, 11, 13, 21]
NAMES = ["indicator", "region_code", "region", "good_code", "good",
         "freq", "period", "value", "unit", "obs_note"]

x = pd.read_excel(XLSX, header=0, names=NAMES, usecols=USECOLS, dtype=str)
x = x[(x["indicator"] == "AVG_CONS_PRCS") & (x["freq"] == "M")].copy()
x["value"] = pd.to_numeric(x["value"], errors="coerce")
x["date"] = pd.to_datetime(x["period"].str.replace("-M", "-", regex=False) + "-01")

csv = pd.read_csv(PROCESSED, parse_dates=["date"])
csv_end = csv["date"].max()

# The two releases must agree where they overlap, otherwise the holdout is not the same series.
key = ["region", "good", "date"]
overlap = x[x["date"] <= csv_end]
unmatched = len(csv.merge(overlap[key], on=key, how="left", indicator=True).query("_merge == 'left_only'"))
if unmatched:
    raise ValueError(f"{unmatched:,} CSV (region, good, month) keys are missing from the XLSX; "
                     "names or coverage changed, so the releases cannot be compared")
both = csv.merge(overlap, on=key, suffixes=("_csv", "_xlsx"))
diff = (both["value_csv"] - both["value_xlsx"]).abs()
mismatch = int((diff > 1e-9).sum() + (both["value_csv"].isna() != both["value_xlsx"].isna()).sum())
print(f"Overlap rows compared: {len(both):,} | value mismatches: {mismatch}")
if mismatch:
    raise ValueError("CSV and XLSX disagree on overlapping months; the holdout would not be comparable")

holdout = x[x["date"] > csv_end]
holdout.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Holdout months: {holdout['date'].min().date()} to {holdout['date'].max().date()} "
      f"| rows: {len(holdout):,} | saved to {OUTPUT}")
