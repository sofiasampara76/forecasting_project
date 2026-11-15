from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
p = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "avg_prices_monthly.csv", parse_dates=["date"])

# 1. Усі товари (щоб побачити решту 15)
print(sorted(p["good"].unique()))

# 2. Де пропуски і примітки
p["is_na"] = p["value"].isna()
print("\nЧастка пропусків за регіоном:")
print(p.groupby("region")["is_na"].mean().sort_values(ascending=False).round(3).to_string())

print("\nПримітки за регіоном (топ):")
print(p[p["obs_note"].notna()].groupby(["obs_note", "region"]).size()
        .sort_values(ascending=False).head(20).to_string())

print("\n'Явищ не було' за товаром (топ):")
print(p[p["obs_note"].str.contains("Явищ", na=False)].groupby("good").size()
        .sort_values(ascending=False).head(15).to_string())

print("\nПропуски за роками:")
print(p.groupby(p["date"].dt.year)["is_na"].sum().to_string())

# 3. Сила сезонності й тренду для рядів по Україні
ua = (p[p["region"] == "Україна"]
        .pivot(index="date", columns="good", values="value")
        .asfreq("MS"))

rows = []
for good in ua.columns:
    s = ua[good].astype(float)
    if s.isna().any() or (s <= 0).any():
        rows.append({"good": good, "n_missing": int(s.isna().sum())})
        continue
    res = STL(np.log(s), period=12, robust=True).fit()
    var_r = res.resid.var()
    f_seas = max(0.0, 1 - var_r / (res.seasonal + res.resid).var())
    f_trend = max(0.0, 1 - var_r / (res.trend + res.resid).var())
    by_month = res.seasonal.groupby(res.seasonal.index.month).mean()
    rows.append({
        "good": good, "n_missing": 0,
        "F_seasonal": round(f_seas, 2), "F_trend": round(f_trend, 2),
        "peak_month": int(by_month.idxmax()),
        "trough_month": int(by_month.idxmin()),
        "amplitude_%": round(100 * (np.exp(by_month.max() - by_month.min()) - 1), 1),
    })

rank = pd.DataFrame(rows).sort_values("F_seasonal", ascending=False)
print("\nСезонність і тренд (по Україні):")
print(rank.to_string(index=False))
rank.to_csv(PROJECT_ROOT / "results" / "seasonality_rank.csv", index=False, encoding="utf-8-sig")