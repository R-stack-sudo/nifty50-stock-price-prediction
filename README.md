# Nifty-50 Stock Price Prediction

Time-series forecasting of the NIFTY-50 index (`^NSEI`) closing price, comparing an ARIMA baseline against LSTM, Bi-LSTM, GRU, and Deep GRU neural network models.

## Overview

The notebook pulls daily NIFTY-50 close prices (2015-01-01 to 2025-07-31) via `yfinance`, then:

1. **Stationarity analysis** — ADF and KPSS tests on the log-close price and its first difference.
2. **ARIMA baseline** — fits `ARIMA(1,1,0)` on the differenced series to model the linear component; residuals are combined back with the price series before scaling.
3. **Deep learning models** — the ARIMA-adjusted, MinMax-scaled series is windowed (60-step lookback) and fed into four recurrent architectures:
   - LSTM
   - Bidirectional LSTM
   - GRU
   - Deep (stacked) GRU
4. **Evaluation** — each model is scored on the held-out test split with MAPE, RMSE, and MASE, and predictions are plotted against the actual (inverse-transformed) closing price.

## Requirements

```
pandas
numpy
matplotlib
yfinance
statsmodels
scikit-learn
tensorflow
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Open and run [Nifty50_StockPricePrediction.ipynb](Nifty50_StockPricePrediction.ipynb) top to bottom, e.g.:

```bash
jupyter notebook Nifty50_StockPricePrediction.ipynb
```

or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute Nifty50_StockPricePrediction.ipynb
```

Internet access is required for the `yfinance` download step.

## Notes

- All four neural network models are trained for a small number of epochs for demonstration; increase `epochs` in the notebook for better convergence.
- Metrics (MAPE / RMSE / MASE) are printed after each model's training cell so results can be compared directly.

## Limitations

- **The ARIMA step has no seasonal component.** It is `ARIMA(1,1,0)`: one autoregressive term on the first-differenced series. It models short-term linear dependence, not trend and seasonality as separate components.
- **Metrics are computed on MinMax-scaled values.** MAPE, RMSE and MASE are printed on the scaled, ARIMA-adjusted series, so they do not read as index points or rupee/percentage errors on the actual NIFTY-50 close. Do not quote them as price-level accuracy.
- **MASE above 1 means no gain over a naive forecast.** MASE compares the model's error with a naive "repeat the last value" forecast. A value above 1 on the test split means the model did not beat that baseline. Check the printed MASE for each model before claiming an improvement.
- **No stand-alone baselines.** Naive and ARIMA-only forecasts are not scored separately on the same test split, so the deep models' gain over simple methods is not demonstrated here.
- **Single split.** There is one train/test split and no walk-forward (rolling-origin) validation, and the networks are trained for few epochs, so the results are one run, not a stable estimate.
