import os
import warnings
from pathlib import Path

# Keep numerical libraries from allocating a worker per CPU core.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERIES = ["Картопля", "Морква", "Яйця",
          "Молоко пастеризоване жирністю до 2,6% включно",
          "Бензин А-95", "Олія соняшникова"]
H, M, INIT, STEP = 12, 12, 36, 6     # горизонт, сезонність, стартовий train, крок фолдів
WAR = pd.Timestamp("2022-01-01")

chunks = []
for chunk in pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "avg_prices_monthly.csv",
        usecols=["date", "region", "good", "value"],
        chunksize=1_000):
    selected = chunk[(chunk["region"] == "Україна") & chunk["good"].isin(SERIES)]
    if not selected.empty:
        chunks.append(selected)

if not chunks:
    raise ValueError("No rows found for Ukraine and the selected series")

p = pd.concat(chunks, ignore_index=True)
p["date"] = pd.to_datetime(p["date"])
y = (p.pivot(index="date", columns="good", values="value")
       .asfreq("MS")[SERIES].astype(float))

# ---------- моделі: приймають train (Series) і h, повертають масив прогнозів ----------
def f_naive(tr, h):
    return np.repeat(tr.iloc[-1], h)

def f_snaive(tr, h):
    return np.array([tr.iloc[-M + (i % M)] for i in range(h)])

def f_snaive_yoy(tr, h):            # сезонний наївний + річний приріст останнього року
    yoy = tr.iloc[-M:].mean() - tr.iloc[-2 * M:-M].mean()
    return f_snaive(tr, h) + yoy

def f_drift(tr, h):
    slope = (tr.iloc[-1] - tr.iloc[0]) / (len(tr) - 1)
    return tr.iloc[-1] + slope * np.arange(1, h + 1)

def f_ets(tr, h):                   # на логарифмах, демпфований тренд + адитивна сезонність
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    m = ExponentialSmoothing(np.log(tr.values), trend="add", damped_trend=True,
                             seasonal="add", seasonal_periods=M,
                             initialization_method="estimated").fit()
    return np.exp(m.forecast(h))

def f_sarima(tr, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    m = SARIMAX(np.log(tr.values), order=(1, 1, 1), seasonal_order=(0, 1, 1, M),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return np.exp(m.forecast(h))

MODELS = {"naive": f_naive, "seasonal_naive": f_snaive, "snaive+yoy": f_snaive_yoy,
          "drift": f_drift, "ETS": f_ets, "SARIMA": f_sarima}

# ---------- rolling-origin (expanding window) ----------
rows = []
for good in SERIES:
    s = y[good]
    for t in range(INIT, len(s) - H + 1, STEP):
        tr, te = s.iloc[:t], s.iloc[t:t + H]
        scale = np.mean(np.abs(tr.values[M:] - tr.values[:-M]))   # сезонний MASE-масштаб, тільки train
        for name, f in MODELS.items():
            try:
                fc = np.asarray(f(tr, H), dtype=float)
                if not np.all(np.isfinite(fc)):
                    raise ValueError
            except Exception:
                fc = f_naive(tr, H)
                print(f"[fallback to naive] {good} | {name} | origin {s.index[t].date()}")
            for h in range(H):
                rows.append((good, name, s.index[t], h + 1, te.index[h], te.iloc[h], fc[h], scale))

res = pd.DataFrame(rows, columns=["good", "model", "origin", "h", "date", "y", "yhat", "scale"])
res["ae"] = (res.y - res.yhat).abs()
res["se"] = (res.y - res.yhat) ** 2
res["ase"] = res.ae / res.scale
res["smape"] = 200 * res.ae / (res.y.abs() + res.yhat.abs())

# режим: тест повністю до 2022, перехідний, повністю після
last = res.groupby(["good", "origin"])["date"].transform("max")
res["regime"] = np.where(last < WAR, "до 2022",
                 np.where(res.origin >= WAR, "після 2022", "перехід"))

res.to_csv(PROJECT_ROOT / "results" / "backtest_results.csv", index=False, encoding="utf-8-sig")

def summ(df, by):
    return (df.groupby(by).agg(MAE=("ae", "mean"),
                               RMSE=("se", lambda x: np.sqrt(x.mean())),
                               MASE=("ase", "mean"), sMAPE=("smape", "mean")).round(3))

order = ["naive", "drift", "seasonal_naive", "snaive+yoy", "ETS", "SARIMA"]
print("Фолдів на ряд:", res.groupby("good")["origin"].nunique().iloc[0])
print("\n=== Усі товари, h=1..12 ===");  print(summ(res, "model").loc[order])
print("\n=== Усі товари, h=1..3 ===");   print(summ(res[res.h <= 3], "model").loc[order])
print("\n=== MASE за товаром, h=1..12 ===")
print(res.pivot_table(index="good", columns="model", values="ase").loc[:, order].round(2))
print("\n=== MASE за режимом, h=1..12 ===")
print(res.pivot_table(index="regime", columns="model", values="ase").loc[:, order].round(2))