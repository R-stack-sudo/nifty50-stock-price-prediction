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
4. **Evaluation** — each model is scored on the held-out test split with MAPE, RMSE, and MASE on the scaled series (printed after each training cell), and again in index points against two baselines (naive previous close, and ARIMA(1,1,0) alone) in the last section of the notebook.

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

The first run downloads the prices with `yfinance` and caches them in `nifty50.csv` (not committed); later runs read that file, so they work offline and use identical data.

## Notes

- All four neural network models are trained for a small number of epochs for demonstration; increase `epochs` in the notebook for better convergence.
- Random seeds are fixed (`random`, `numpy`, `tensorflow`), but results from a 5-epoch run still move noticeably between runs and library versions, so treat them as one run.

## Results

Test period 2022-08-30 to 2025-07-29 (720 trading days), forecasts converted back to closing prices:

| Model | MAE (index pts) | RMSE (index pts) | MAPE (%) | MASE | Direction correct (%) |
|---|---|---|---|---|---|
| Naive (previous close) | 124.2 | 174.8 | 0.58 | 1.48 | n/a |
| ARIMA(1,1,0) only | 124.3 | 174.6 | 0.58 | 1.48 | 46.0 |
| ARIMA-LSTM | 495.3 | 545.2 | 2.32 | 5.89 | 46.1 |
| ARIMA-Bi-LSTM | 609.1 | 716.9 | 2.79 | 7.25 | 45.0 |
| ARIMA-GRU | 188.6 | 239.1 | 0.83 | 2.25 | 57.9 |
| ARIMA-Deep GRU | 1220.1 | 1301.3 | 5.56 | 14.52 | 45.7 |

MASE here divides by the training-period mean absolute one-day change, so values above 1 are worse than a one-day-ahead naive forecast on that scale.

**Reading the table:** none of the hybrid models beats the naive "tomorrow equals today" forecast in index points. The GRU comes closest (0.83% MAPE against 0.58%) and is the only model with a direction hit rate noticeably above 50%, but with 720 test days that is weak evidence on its own. The five-epoch networks are probably under-trained, and the LSTM, Bi-LSTM and Deep GRU show a systematic offset from the actual price.

The scaled-series metrics printed in the training cells are on a different scale and flatter the models: GRU 0.012 MAPE / 1.93 MASE, LSTM 0.025 / 3.87, Bi-LSTM 0.029 / 4.54, Deep GRU 0.053 / 6.22 (the Deep GRU cell uses a training-set MASE denominator, the others a test-set one, so their MASE values are not directly comparable).

## Limitations

- **The ARIMA step has no seasonal component.** It is `ARIMA(1,1,0)` applied to the already-differenced log close, so it models short-term linear dependence only, not trend and seasonality as separate components.
- **The target series is a constructed hybrid.** The networks predict the ARIMA residual series (cumulative sum, plus the previous day's log close), not the raw close. The residual series is built from a model fitted on the full sample, including the test period, so it is not a clean out-of-sample setup. The results above suggest it does not help anyway, but a proper design would fit the ARIMA on the training period only.
- **Scaled-series metrics are not price errors.** The MAPE, RMSE and MASE printed in the training cells use MinMax-scaled values. Use the index-point table above for any claim about price accuracy.
- **Single split, few epochs, one seed.** There is no walk-forward validation, and the networks are trained for 5 epochs, so these numbers are not a stable estimate and a longer training run would change them.
- **Baselines are simple.** Only the naive and ARIMA-only forecasts are compared. Stronger baselines (for example ARIMA with tuned order, or a plain LSTM on returns) are not included.
