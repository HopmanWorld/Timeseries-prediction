"""Ethereum price forecasting with Prophet.

This module provides a small, reusable pipeline for downloading ETH-USD
historical prices, training a Prophet time-series model, evaluating a holdout
window, and exporting future forecasts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def ensure_dependencies(packages: list[str]) -> None:
    """Fail fast with an actionable install message when dependencies are missing."""

    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required package(s): {joined}. "
            "Install project dependencies with: pip install -r requirements.txt"
        )


DEFAULT_TICKER = "ETH-USD"
DEFAULT_START_DATE = "2017-11-09"  # Earliest broadly available ETH-USD data in Yahoo Finance.


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for an Ethereum forecasting run."""

    ticker: str = DEFAULT_TICKER
    start: str = DEFAULT_START_DATE
    end: str | None = None
    horizon: int = 60
    forecast_days: int = 60
    output_dir: str = "outputs"
    interval_width: float = 0.8


def fetch_price_history(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily close prices and return Prophet-ready columns.

    Prophet expects two columns:
    - ``ds``: datestamp
    - ``y``: numeric target value
    """

    ensure_dependencies(["pandas", "yfinance"])

    import pandas as pd
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"No price history returned for {ticker!r} from {start!r} to {end!r}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if "Close" not in raw.columns:
        raise ValueError("Downloaded data did not include a 'Close' price column.")

    history = raw.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    history["ds"] = pd.to_datetime(history["ds"]).dt.tz_localize(None)
    history["y"] = pd.to_numeric(history["y"], errors="coerce")
    history = history.dropna().sort_values("ds").reset_index(drop=True)

    if len(history) < 90:
        raise ValueError("At least 90 daily observations are recommended for a meaningful forecast.")

    return history


def split_train_test(history: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series into train and final holdout windows."""

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if len(history) <= horizon + 30:
        raise ValueError("Not enough observations for the requested holdout horizon.")

    return history.iloc[:-horizon].copy(), history.iloc[-horizon:].copy()


def build_model(interval_width: float):
    """Create a Prophet model tuned for daily crypto prices."""

    ensure_dependencies(["prophet"])

    from prophet import Prophet

    model = Prophet(
        interval_width=interval_width,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_mode="multiplicative",
    )
    return model


def evaluate_forecast(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, float]:
    """Calculate common forecast accuracy metrics for the holdout period."""

    merged = actual.merge(predicted[["ds", "yhat"]], on="ds", how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between actuals and predictions.")

    errors = merged["y"] - merged["yhat"]
    mae = float(errors.abs().mean())
    rmse = float(math.sqrt((errors**2).mean()))
    non_zero = merged["y"] != 0
    mape = float((errors[non_zero].abs() / merged.loc[non_zero, "y"]).mean() * 100)
    return {"mae": mae, "rmse": rmse, "mape_percent": mape}


def run_forecast(config: ForecastConfig, write_plots: bool = True) -> dict[str, Any]:
    """Run the complete forecasting workflow and write artifacts to disk."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = fetch_price_history(config.ticker, config.start, config.end)
    train, test = split_train_test(history, config.horizon)

    validation_model = build_model(config.interval_width)
    validation_model.fit(train)
    validation_future = validation_model.make_future_dataframe(periods=config.horizon, freq="D")
    validation_forecast = validation_model.predict(validation_future)
    metrics = evaluate_forecast(test, validation_forecast)

    final_model = build_model(config.interval_width)
    final_model.fit(history)
    future = final_model.make_future_dataframe(periods=config.forecast_days, freq="D")
    forecast = final_model.predict(future)

    forecast_columns = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
    forecast_path = output_dir / "eth_forecast.csv"
    forecast[forecast_columns].to_csv(forecast_path, index=False)

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {"config": asdict(config), "metrics": metrics}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    plot_paths: list[str] = []
    if write_plots:
        ensure_dependencies(["matplotlib"])

        import matplotlib.pyplot as plt

        fig = final_model.plot(forecast)
        ax = fig.gca()
        ax.scatter(history["ds"], history["y"], s=8, color="black", alpha=0.35, label="Actual close")
        ax.set_title(f"{config.ticker} close-price forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("USD")
        ax.legend()
        forecast_plot_path = output_dir / "eth_forecast.png"
        fig.savefig(forecast_plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(forecast_plot_path))

        components_fig = final_model.plot_components(forecast)
        components_path = output_dir / "eth_forecast_components.png"
        components_fig.savefig(components_path, dpi=160, bbox_inches="tight")
        plt.close(components_fig)
        plot_paths.append(str(components_path))

    return {
        "forecast_path": str(forecast_path),
        "metrics_path": str(metrics_path),
        "plot_paths": plot_paths,
        "metrics": metrics,
        "rows": len(history),
        "last_observed_date": history["ds"].max().date().isoformat(),
    }


def positive_int(value: str) -> int:
    """Argparse type for positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Forecast Ethereum (ETH-USD) daily close prices with Prophet.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Yahoo Finance ticker to forecast. Default: ETH-USD")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Start date for history download (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Optional exclusive end date for history download (YYYY-MM-DD).")
    parser.add_argument("--horizon", type=positive_int, default=60, help="Holdout days used for validation metrics.")
    parser.add_argument("--forecast-days", type=positive_int, default=60, help="Future days to forecast after the last observation.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV, metrics, and plot artifacts.")
    parser.add_argument("--interval-width", type=float, default=0.8, help="Prediction interval width between 0 and 1.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    if not 0 < args.interval_width < 1:
        raise SystemExit("--interval-width must be between 0 and 1.")

    config = ForecastConfig(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        forecast_days=args.forecast_days,
        output_dir=args.output_dir,
        interval_width=args.interval_width,
    )
    result = run_forecast(config, write_plots=not args.no_plots)

    print("Ethereum forecast complete")
    print(f"Rows used: {result['rows']}")
    print(f"Last observed date: {result['last_observed_date']}")
    print(f"Forecast: {result['forecast_path']}")
    print(f"Metrics: {result['metrics_path']}")
    for key, value in result["metrics"].items():
        print(f"{key}: {value:,.4f}")
    for plot_path in result["plot_paths"]:
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
