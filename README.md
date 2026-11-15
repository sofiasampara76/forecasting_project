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
results/                Seasonality and backtest CSV outputs
src/
  data/                 Data extraction script
  eda/                  Supporting exploratory-analysis scripts
  forecasting/          Four successive baseline script versions
README.md
requirements.txt
```

`data/raw/136-indeksi-spozhivchikh-tsin.xlsx` is an additional source file retained for reference; the main forecasting workflow uses `136.csv`.

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
4. **Forecast-result review** — [`notebooks/03_forecasting_results.ipynb`](notebooks/03_forecasting_results.ipynb) summarizes saved backtest scores, convergence, and fallback records. It does not fit forecasting models.

The extraction and seasonality calculations can also be run from the project root:

```powershell
.\venv\Scripts\python.exe .\src\data\prepare_prices.py
.\venv\Scripts\python.exe .\src\eda\check_seasonality.py
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

Run the current baseline from the project root:

```powershell
.\venv\Scripts\python.exe .\src\forecasting\no_ml_predictions_v4.py
```

## Optimization, limitations, and next steps

Strongly seasonal produce may motivate a future storage or procurement-planning scenario. Price forecasts alone are not sufficient for an optimization model: demand, purchase and selling prices, storage capacity, spoilage, operating costs, lead times, and budget constraints are also needed.

The source contains prices but not product-level sales quantities or demand. Price elasticity and demand response therefore cannot be identified from this dataset alone. Any future optimization exercise that uses synthetic inputs should label assumptions and generated data explicitly.

Recommended next steps:

1. Review model scores by product, forecast horizon, and regime, including SARIMA convergence and fallback cases.
2. Use the annual seasonal-profile charts to assess whether the observed vegetable and egg patterns recur across years.
3. Prepare the course progress presentation: explain the data and target, EDA, rolling-origin validation, baseline results, limitations, and planned next steps.
4. If pursuing optimization, collect additional operational/demand data or define and clearly label a synthetic scenario.

## Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt). Notebooks and documentation are in English; original product and region labels are kept as published in the source data.
