# Ethereum Price Prediction Model

A clean, reproducible Ethereum (`ETH-USD`) time-series forecasting project built around [Prophet](https://facebook.github.io/prophet/) and Yahoo Finance price history.

The original notebook is still included for exploration, and the production-ready workflow now lives in `src/ethereum_prediction_model.py` so the model can be run, evaluated, and regenerated from the command line.

## What this project does

- Downloads daily Ethereum price history from Yahoo Finance with `yfinance`.
- Converts raw market data into Prophet's required `ds` / `y` format.
- Trains a Prophet model with weekly and yearly seasonality.
- Uses a final holdout window to report MAE, RMSE, and MAPE before fitting the final forecast model.
- Exports forecast data, metrics, and optional charts into an `outputs/` directory.

> **Important:** This project is for education and research. Crypto assets are highly volatile, and this model should not be treated as financial advice or as a guarantee of future price movement.

## Repository layout

```text
.
├── ETH_Price_Prediction_with_fbProphet.ipynb  # Original exploratory notebook
├── Python_for_DataSciences.ipynb              # General Python/data-science practice notebook
├── src/
│   └── ethereum_prediction_model.py           # Reusable forecasting CLI and pipeline
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation
```

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Ethereum forecast

```bash
python src/ethereum_prediction_model.py
```

By default, the script:

- forecasts `ETH-USD`,
- starts from `2017-11-09`,
- validates on the latest 60 observed days,
- forecasts 60 future days, and
- writes artifacts to `outputs/`.

## Common commands

Run a 90-day future forecast with a 90-day validation window:

```bash
python src/ethereum_prediction_model.py --horizon 90 --forecast-days 90
```

Use a custom date range:

```bash
python src/ethereum_prediction_model.py --start 2020-01-01 --end 2026-01-01
```

Skip PNG chart generation, which is useful in minimal CI environments:

```bash
python src/ethereum_prediction_model.py --no-plots
```

Save results to a custom directory:

```bash
python src/ethereum_prediction_model.py --output-dir experiment_outputs
```

## Output files

After a successful run, the output directory contains:

| File | Purpose |
| --- | --- |
| `eth_forecast.csv` | Prophet forecast with `ds`, `yhat`, `yhat_lower`, `yhat_upper`, and `trend`. |
| `metrics.json` | Run configuration plus holdout metrics: MAE, RMSE, and MAPE. |
| `eth_forecast.png` | Forecast plot with historical observations and prediction intervals. |
| `eth_forecast_components.png` | Prophet trend and seasonality component plots. |

## Methodology

1. **Data ingestion:** `fetch_price_history()` downloads daily close prices for `ETH-USD` and cleans them into the two columns Prophet requires.
2. **Validation:** `split_train_test()` reserves the most recent observations as a holdout period, so the model is evaluated on data it did not train on.
3. **Forecasting:** Prophet models trend changes plus weekly and yearly seasonal effects, then produces point forecasts and uncertainty intervals.
4. **Reporting:** `evaluate_forecast()` calculates MAE, RMSE, and MAPE to make model quality visible instead of only plotting a line chart.

## How to improve the model further

Potential next steps for a stronger research-grade model:

- Add exogenous features such as BTC price, ETH volume, gas fees, staking yields, macro rates, or market sentiment.
- Compare Prophet against ARIMA/SARIMAX, XGBoost, LightGBM, LSTM/GRU, and transformer-based time-series models.
- Add walk-forward validation instead of a single holdout period.
- Tune Prophet hyperparameters with cross-validation.
- Track experiments with MLflow, Weights & Biases, or a simple results table.
- Add automated tests around data preparation, metrics, and CLI argument validation.

## Notebook note

`ETH_Price_Prediction_with_fbProphet.ipynb` is useful for learning and visual exploration. For repeatable runs, prefer the command-line pipeline in `src/ethereum_prediction_model.py` because it includes input validation, holdout evaluation, saved artifacts, and configurable parameters.

## Troubleshooting

- If `prophet` fails to install, upgrade packaging tools first:

  ```bash
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  ```

- If Yahoo Finance returns no rows, retry later or specify a different `--start` / `--end` window.
- If charts do not render on a headless machine, run with `--no-plots`.
