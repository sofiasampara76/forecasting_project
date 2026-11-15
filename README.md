# Consumer Price Forecasting

A forecasting-course project using the State Statistics Service of Ukraine's published monthly average consumer prices. The primary source is [`data/raw/136.csv`](data/raw/136.csv), available from the [data.gov.ua dataset](https://data.gov.ua/dataset/c1cd8f39-c2ee-4f20-8c8a-6cdde6e6f343).

## Project question

How well can simple, non-machine-learning time-series baselines forecast the next 12 months of selected official average prices in Ukraine, and how do forecast errors change across the 2022 regime break?

The target is a published **price level**, not a consumer price index. The selected baseline series are potatoes, carrots, eggs, pasteurized milk, A-95 gasoline, and sunflower oil. Each product is forecast separately in its published unit.

## Current data snapshot

The current processed data contains 145,860 monthly average-price records for 55 products across 26 territorial categories, from January 2017 through June 2025 (102 months per region/product series). It contains 9,001 missing price values: 6,349 marked “Not available” and 2,652 marked “Not observed.” Missingness rises from 2022 onward, alongside source-documented changes in territorial coverage.

The six selected Ukraine-level baseline series each have 102 observations and no missing prices. The processed file retains missing values and source observation notes; it does not interpolate them. The 2022 coverage and price-regime changes should be considered when interpreting both seasonality and model performance.

## Repository structure

```text
data/
  raw/                 Original source files; do not edit
  processed/           Filtered monthly average-price data
notebooks/
  00_dataset_selection.ipynb
  01_data_preparation.ipynb
  02_eda_seasonality.ipynb
  03_forecasting_results.ipynb
  04_optimization_bridge.ipynb
results/                Seasonality and backtest CSV outputs
  figures/              PNG charts written by notebooks 03 and 04 (used in the presentation)
src/
  data/                 Data extraction scripts (main CSV, final holdout from the XLSX)
  eda/                  Supporting exploratory-analysis scripts
  forecasting/          Four successive baseline script versions (v4 is current and importable)
  reporting/            Shared chart labels and style for notebooks 03 and 04
README.md
requirements.txt
```

Both raw files come from the [data.gov.ua dataset page](https://data.gov.ua/dataset/c1cd8f39-c2ee-4f20-8c8a-6cdde6e6f343) and go in `data/raw/` (git-ignored): `136. Індекси споживчих цін.csv` saved as `136.csv`, and `136. Індекси споживчих цін.xlsx` saved as `136-indeksi-spozhivchikh-tsin.xlsx`. The CSV is the modelling snapshot (to June 2025). The XLSX was published later and continues the same series; `src/data/prepare_holdout.py` takes only the months it adds after the CSV (July–September 2025 at the time of writing) as a final out-of-sample check, after verifying that the two files agree on every overlapping value.

## Setup

Use Python 3.11 or a compatible version supported by the installed scientific packages. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, the existing local environment can be used with `.\venv\Scripts\python.exe`. Launch Jupyter from the project root so notebooks can locate the `data/` and `results/` folders:

```powershell
jupyter lab
```

The raw `136.csv` is approximately 520 MB. Ensure sufficient memory and disk space before preparing or reloading it.

## Reproducible workflow

Run the notebooks in order:

1. **Dataset context and selection** — [`notebooks/00_dataset_selection.ipynb`](notebooks/00_dataset_selection.ipynb) profiles the primary CSV, its indicators, frequencies, coverage, products, regions, units, and observation notes, then documents the six selected targets.
2. **Preparation and quality checks** — [`notebooks/01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb) extracts monthly `AVG_CONS_PRCS` records, writes `data/processed/avg_prices_monthly.csv`, checks coverage and missingness, builds the national target panel, and flags extreme monthly changes for review. Flags do not automatically remove observations.
3. **Seasonality and trend EDA** — [`notebooks/02_eda_seasonality.ipynb`](notebooks/02_eda_seasonality.ipynb) performs robust STL decomposition on log prices, compares annual profiles, creates diagnostic charts, and saves `results/seasonality_rank.csv`.
4. **Baseline forecasts (Block 3)** — [`notebooks/03_forecasting_results.ipynb`](notebooks/03_forecasting_results.ipynb) explains the models and metrics, ranks the models overall, by horizon, product, and regime, checks SARIMA convergence and fallbacks, and refits every model for the final July–September 2025 holdout. It needs the backtest CSVs and the holdout file (below).
5. **Bridge to optimization (Block 4)** — [`notebooks/04_optimization_bridge.ipynb`](notebooks/04_optimization_bridge.ipynb) states the procurement-planning linear program the forecasts feed, solves an illustrative version for carrots and potatoes with the backtest forecasts, and lists the current limitations and next steps.

Before notebooks 03 and 04, run from the project root (Windows paths shown; on macOS/Linux use `.venv/bin/python src/...`):

```powershell
.\venv\Scripts\python.exe .\src\data\prepare_prices.py                 # data/processed/avg_prices_monthly.csv
.\venv\Scripts\python.exe .\src\data\prepare_holdout.py                # data/processed/avg_prices_holdout.csv (~1 min, reads the XLSX)
.\venv\Scripts\python.exe .\src\forecasting\no_ml_predictions_v4.py    # results/backtest_results.csv etc. (~20 s)
.\venv\Scripts\python.exe .\src\eda\check_seasonality.py               # optional: results/seasonality_rank.csv
```

## Preliminary EDA findings

The current STL analysis uses a 12-month seasonal period and robust decomposition of log prices. Its variance-based strength measures are descriptive statistics, not p-values or tests of statistical significance.

- Carrots and eggs show comparatively strong seasonal components, with estimated seasonal amplitudes of about 95% and 64%, respectively. Potatoes have an estimated amplitude of about 63%.
- Eggs have a different estimated peak/trough pattern from the selected vegetables. Year-by-year profiles are included to check whether pooled seasonal patterns repeat consistently.
- Pasteurized milk has a relatively high seasonal-strength measure but a small estimated amplitude (about 3%). Strength and economic magnitude therefore should not be treated as interchangeable.
- A-95 gasoline and sunflower oil have weak seasonal-strength estimates in this analysis.
- Beet, cabbage, and bulb onion are additional seasonal-procurement candidates. Cabbage and onion have large estimated amplitudes but weak seasonal-strength estimates, which may indicate unstable year-to-year patterns or shocks.
- Trend strength is high in most series. The 2022 regime change and changes in territorial coverage limit simple extrapolation and motivate period-specific backtest reporting.

These are exploratory findings on the current data snapshot, not causal conclusions.

## Baseline forecasting and validation

The current version is [`src/forecasting/no_ml_predictions_v4.py`](src/forecasting/no_ml_predictions_v4.py). It evaluates seven non-ML methods:

- Naive and drift
- Seasonal naive and seasonal naive with a year-over-year adjustment
- STL plus drift
- Damped-trend ETS
- SARIMA

Validation uses an expanding-window rolling-origin design: 36 initial training months, a 12-month test horizon, and a 6-month step. For each 102-month target series, this creates 10 forecast origins. At every origin, the model is fit on past observations only; the next 12 months are held out for scoring. Results distinguish pre-2022, transition, and post-2022 forecast periods.

The script reports MAE, RMSE, MASE, and sMAPE, tracks SARIMA convergence, and records fallbacks to the naive forecast. Check all three outputs after each run:

- `results/backtest_results.csv`
- `results/sarima_convergence.csv`
- `results/fallbacks.csv`

The forecast scripts `no_ml_predictions.py`, `no_ml_predictions_v2.py`, and `no_ml_predictions_v3.py` are earlier experiment versions. Use v4 for the current workflow; earlier versions may overwrite result files.

Importing the module (as notebook 03 does) exposes `MODELS`, `load_targets`, and `guarded_forecast` without running the backtest or silencing warnings; both happen only when the file is executed as a script. SARIMA and ETS are fitted by numerical optimisers, so their forecasts can differ slightly between machines and library versions; the four naive-family benchmarks are exact.

**Main baseline findings** (full tables and charts in notebook 03; numbers from the run of 2026-11-15):

- Naive is a hard benchmark. ETS is best overall but only about 7% better than naive over 12 months; averaged over 1–3 months ahead, naive, ETS, and SARIMA are nearly tied.
- Pure seasonal naive is the worst model: the price level moves too much for last year's value to be a good guess. ETS and SARIMA, which model level and seasonality together, beat naive by 10–24% at 3–7 months ahead; from 9 months drift is best.
- The best model differs by product (ETS: potato, eggs, sunflower oil; SARIMA: carrot; drift: milk; naive: gasoline).
- On the July–September 2025 holdout, ETS and SARIMA have the lowest mean MASE because no product goes badly wrong for them.

## Optimization bridge, limitations, and next steps

Notebook 04 formulates a procurement-planning linear program: a buyer with a fixed monthly need chooses monthly purchases and storage, given forecast prices, spoilage, storage cost, and capacity. Only prices come from the data; demand and the cost parameters are explicitly labelled assumptions. Plans are made with each model's backtest forecast and scored at actual prices. For carrots (stable seasonality) the forecast-based plans recover about half of the saving that perfect foresight would give; for potatoes (unstable seasonality) they cost slightly more than buying every month. Forecast accuracy (MASE) and decision value can therefore disagree.

The source contains retail prices but no quantities, demand, or wholesale prices, so price elasticity and selling-side decisions cannot be estimated from it. Notebook 04 lists the data, forecasting, and optimization limitations and the planned next steps: prediction intervals feeding stochastic or robust optimization, monthly re-planning, model selection by decision cost, and ML models with exogenous drivers benchmarked against naive.

## Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt). Notebooks and documentation are in English; original product and region labels are kept as published in the source data.
