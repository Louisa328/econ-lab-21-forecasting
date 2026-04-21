# Time Series Forecasting — ARIMA, GARCH & Bootstrap

## Objective
Diagnose and correct a misspecified ARIMA pipeline on U.S. CPI data, extend the analysis to conditional volatility modeling of S&P 500 returns using GARCH(1,1), and implement distribution-free forecast uncertainty quantification via block bootstrap resampling.

## Methodology
- **Error Diagnosis:** Identified three deliberate modeling errors in a broken ARIMA pipeline — unit root violation (d=0 on non-stationary CPI), seasonality omission (ARIMA instead of SARIMA on monthly data), and missing residual diagnostics (no Ljung-Box test prior to forecasting)
- **SARIMA Correction:** Refitted the pipeline using `auto_arima` with seasonal order (2,0,0,12) and non-seasonal order (3,1,0); verified white-noise residuals via Ljung-Box test at lags 12 and 24 (p-values = 1.0), confirming model adequacy before generating a 24-month CPI forecast
- **GARCH(1,1) Volatility Modeling:** Fitted a GARCH(1,1) model to S&P 500 daily log returns (2000–2024) using the `arch` library; extracted conditional volatility series and annotated major crisis episodes (Lehman 2008, COVID 2020, 2022 Bear Market)
- **Forecast Evaluation Module:** Developed a production-grade `forecast_evaluation.py` module implementing `compute_mase()` (Mean Absolute Scaled Error relative to a seasonal naive benchmark) and `backtest_expanding_window()` (expanding-window walk-forward validation with per-horizon error logging)
- **Block Bootstrap Intervals:** Implemented a moving block bootstrap (block size = 6, B = 500 replications) to generate distribution-free 95% forecast intervals robust to residual non-normality and heteroskedasticity

## Key Findings
- SARIMA residuals passed the Ljung-Box diagnostic at both lag 12 and lag 24 after correcting all three pipeline errors, confirming the model fully captured the seasonal and trend structure of CPI
- GARCH(1,1) estimated α = 0.1197, β = 0.8629, yielding α + β = 0.9826 < 1 (variance stationarity confirmed); the implied half-life of volatility shocks is **39.5 days**, consistent with near-integrated GARCH behavior and slow mean reversion in equity volatility
- GARCH long-run volatility (1.16%) closely matched the sample standard deviation (1.22%), indicating a well-calibrated model; peak conditional volatility of 6.85% was recorded on March 17, 2020 (COVID crash)
- Block bootstrap CI width remained stable across forecast horizons (2.82 at h=12, 2.54 at h=24), reflecting low and homoskedastic residual variance in the corrected SARIMA model

## Tools & Libraries
`Python` · `statsmodels` · `pmdarima` · `arch` · `yfinance` · `fredapi` · `pandas` · `numpy` · `matplotlib`
