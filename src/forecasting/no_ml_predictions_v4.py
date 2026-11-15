"""
Rolling-origin бектест (v4) бейзлайнів без ML для середніх споживчих цін (Держстат).
Вхід:  data/processed/avg_prices_monthly.csv  (колонки: region, good, date, value, ...)
Вихід: results/backtest_results.csv, sarima_convergence.csv, fallbacks.csv + зведені таблиці в консоль

Модуль можна імпортувати (ноутбуки беруть звідси MODELS, load_targets, guarded_forecast);
бектест запускається лише як скрипт.
"""
import os
# Обмежуємо потоки ДО імпорту numpy: на Windows OpenBLAS резервує буфер пам'яті під кожне ядро
# і падає з "Memory allocation still failed", якщо вільної RAM мало.
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERIES = ["Картопля", "Морква", "Яйця",
          "Молоко пастеризоване жирністю до 2,6% включно",
          "Бензин А-95", "Олія соняшникова"]
H, M, INIT, STEP = 12, 12, 36, 6      # горизонт, сезонний період, стартовий train, крок фолдів
WAR = pd.Timestamp("2022-01-01")


def load_targets(path=PROJECT_ROOT / "data" / "processed" / "avg_prices_monthly.csv"):
    """Панель date × good для шести цільових рядів по Україні."""
    p = pd.read_csv(path, parse_dates=["date"])
    return (p[p["region"] == "Україна"]
              .pivot(index="date", columns="good", values="value")
              .asfreq("MS")[SERIES].astype(float))


# ---------- моделі: (train Series, h) -> масив прогнозів ----------
def f_naive(tr, h):
    return np.repeat(tr.iloc[-1], h)

def f_snaive(tr, h):
    return np.array([tr.iloc[-M + (i % M)] for i in range(h)])

def f_snaive_yoy(tr, h):               # сезонний наївний + річний приріст останнього року
    yoy = tr.iloc[-M:].mean() - tr.iloc[-2 * M:-M].mean()
    return f_snaive(tr, h) + yoy

def f_drift(tr, h):
    slope = (tr.iloc[-1] - tr.iloc[0]) / (len(tr) - 1)
    return tr.iloc[-1] + slope * np.arange(1, h + 1)

def f_ets(tr, h):                      # логарифми, демпфований тренд + адитивна сезонність
    m = ExponentialSmoothing(np.log(tr.values), trend="add", damped_trend=True,
                             seasonal="add", seasonal_periods=M,
                             initialization_method="estimated").fit()
    return np.exp(m.forecast(h))

def f_sarima(tr, h):                   # "airline model" (0,1,1)(0,1,1)12: стабільніша на коротких рядах
    m = SARIMAX(np.log(tr.values), order=(0, 1, 1), seasonal_order=(0, 1, 1, M),
                enforce_stationarity=True, enforce_invertibility=True
                ).fit(disp=False, maxiter=200)
    f_sarima.converged = bool(m.mle_retvals.get("converged", True))   # читає цикл бектесту
    return np.exp(m.forecast(h))

def f_stl_drift(tr, h):                # STL-сезонність + random walk з дрейфом на скоригованому ряді
    mod = STLForecast(np.log(tr), ARIMA, model_kwargs=dict(order=(0, 1, 0), trend="t"),
                      period=M, robust=True).fit()
    return np.exp(np.asarray(mod.forecast(h)))

MODELS = {"naive": f_naive, "drift": f_drift, "seasonal_naive": f_snaive,
          "snaive+yoy": f_snaive_yoy, "STL+drift": f_stl_drift,
          "ETS": f_ets, "SARIMA": f_sarima}
ORDER = list(MODELS)

# Запобіжник ловить лише божевільні прогнози (вибухи, від'ємні ціни). Коридор широкий навмисно:
# сезонні овочі (морква, буряк) законно змінюють ціну в 3-6 разів між мінімумом і піком.
LOW, HIGH = 0.1, 10.0


def guarded_forecast(f, tr, h):
    """Прогноз моделі f, або ValueError, якщо він нескінченний, від'ємний чи поза коридором."""
    fc = np.asarray(f(tr, h), dtype=float)
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite forecast")
    ratio = fc / tr.iloc[-1]
    if (fc <= 0).any() or ratio.min() < LOW or ratio.max() > HIGH:
        raise ValueError(f"implausible forecast (x{ratio.min():.2f}..x{ratio.max():.2f} of last price)")
    return fc


# ---------- rolling-origin (expanding window) ----------
def run_backtest(y):
    """Повертає (res, conv, fb): прогнози по фолдах, збіжність SARIMA, відкати на naive."""
    rows, sarima_flags, fallbacks = [], [], []
    for good in SERIES:
        s = y[good]
        for t in range(INIT, len(s) - H + 1, STEP):
            tr, te = s.iloc[:t], s.iloc[t:t + H]
            origin = s.index[t]
            scale = np.mean(np.abs(tr.values[M:] - tr.values[:-M]))   # MASE-масштаб лише з train
            for name, f in MODELS.items():
                f_sarima.converged = None          # лишається None, якщо фіт SARIMA впав
                try:
                    fc = guarded_forecast(f, tr, H)
                except Exception as e:
                    fc = f_naive(tr, H)
                    fallbacks.append((good, name, origin, str(e)))
                    print(f"[fallback to naive] {good} | {name} | {origin.date()} | {e}")
                if name == "SARIMA" and f_sarima.converged is not None:
                    sarima_flags.append((good, origin, f_sarima.converged))
                for h in range(H):
                    rows.append((good, name, origin, h + 1, te.index[h], te.iloc[h], fc[h], scale))

    res = pd.DataFrame(rows, columns=["good", "model", "origin", "h", "date", "y", "yhat", "scale"])
    res["ae"] = (res.y - res.yhat).abs()
    res["se"] = (res.y - res.yhat) ** 2
    res["ase"] = res.ae / res.scale
    res["smape"] = 200 * res.ae / (res.y.abs() + res.yhat.abs())

    last = res.groupby(["good", "origin"])["date"].transform("max")
    res["regime"] = np.where(last < WAR, "до 2022",
                     np.where(res.origin >= WAR, "після 2022", "перехід"))

    conv = pd.DataFrame(sarima_flags, columns=["good", "origin", "converged"])
    res = res.merge(conv.assign(sarima_converged=conv.converged.astype(float))
                        [["good", "origin", "sarima_converged"]],
                    on=["good", "origin"], how="left")
    res.loc[res.model != "SARIMA", "sarima_converged"] = np.nan
    fb = pd.DataFrame(fallbacks, columns=["good", "model", "origin", "reason"])
    return res, conv, fb


# ---------- звіт ----------
def summ(df, by):                       # зведені метрики БЕЗ MAE/RMSE у гривнях
    return df.groupby(by).agg(MASE=("ase", "mean"), sMAPE=("smape", "mean")).round(3)


def report(res, conv, fb):
    print("\n=== Відкати на naive (від'ємний, нескінченний або божевільний прогноз) ===")
    print(fb.groupby("model").size().to_string() if len(fb) else "жодного")
    n_folds = res.groupby("good")["origin"].nunique().iloc[0]
    print(f"\nФолдів на ряд: {n_folds}")
    print(f"SARIMA не збіглась у {(~conv.converged).sum()} із {len(conv)} фітів "
          f"({100 * (~conv.converged).mean():.0f}%)")

    print("\n=== Зведено по всіх товарах, h=1..12 (MASE, sMAPE) ===")
    print(summ(res, "model").loc[ORDER])
    print("\n=== Зведено по всіх товарах, h=1..3 ===")
    print(summ(res[res.h <= 3], "model").loc[ORDER])

    print("\n=== MASE за товаром, h=1..12 ===")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print(res.pivot_table(index="good", columns="model", values="ase").loc[:, ORDER].round(2))

    print("\n=== MAE (грн) за товаром, h=1..12 (порівнювати лише в межах рядка) ===")
    print(res.pivot_table(index="good", columns="model", values="ae").loc[:, ORDER].round(2))
    print("\n=== RMSE (грн) за товаром, h=1..12 ===")
    rm = res.groupby(["good", "model"])["se"].mean().pow(0.5).unstack()[ORDER]
    print(rm.round(2))

    print("\n=== MASE за режимом (порівнювати моделі в межах рядка, не режими між собою) ===")
    cnt = res.groupby("regime")["origin"].nunique()
    tab = res.pivot_table(index="regime", columns="model", values="ase").loc[:, ORDER].round(2)
    tab["фолдів"] = cnt
    print(tab.drop(index="перехід", errors="ignore"))
    print("(режим 'перехід' — 1 фолд, виключено зі звіту)")

    if len(conv) and (~conv.converged).any():
        s_all = res[res.model == "SARIMA"]
        print("\n=== SARIMA: усі фолди vs лише збіглі, MASE h=1..12 ===")
        print(pd.Series({"усі": s_all.ase.mean(),
                         "лише збіглі": s_all[s_all.sarima_converged == 1.0].ase.mean()}).round(3))

    # чесне порівняння: усі моделі лише на фолдах, де SARIMA збіглась
    ok = conv[conv.converged][["good", "origin"]]
    sub = res.merge(ok, on=["good", "origin"])
    print(f"\n=== Усі моделі лише на фолдах, де SARIMA збіглась ({len(ok)} із {len(conv)}), h=1..12 ===")
    print(summ(sub, "model").loc[ORDER])

    print("\n=== Де SARIMA не збігається: за товаром та за початком фолда ===")
    print(conv.assign(not_conv=~conv.converged).groupby("good")["not_conv"].sum().to_string())
    print(conv.assign(not_conv=~conv.converged).groupby("origin")["not_conv"].sum().to_string())


if __name__ == "__main__":
    # Збіжність SARIMA відстежуємо явно (mle_retvals), тому в консольному прогоні всі попередження глушимо.
    # При імпорті (ноутбуки) фільтр не ставиться: там кожен сам вирішує, що бачити.
    warnings.filterwarnings("ignore")
    res, conv, fb = run_backtest(load_targets())
    out = PROJECT_ROOT / "results"
    res.to_csv(out / "backtest_results.csv", index=False, encoding="utf-8-sig")
    conv.to_csv(out / "sarima_convergence.csv", index=False, encoding="utf-8-sig")
    fb.to_csv(out / "fallbacks.csv", index=False, encoding="utf-8-sig")
    report(res, conv, fb)
