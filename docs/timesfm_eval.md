# TimesFM-3 on Korean large caps — walk-forward evaluation

All numbers below were produced by one run of `models/eval_timesfm.py` on 2026-09-27/28
(Apple M4 Max, MLX backend). Raw results: [`timesfm_eval_results.json`](timesfm_eval_results.json),
flat table: [`timesfm_eval_results.csv`](timesfm_eval_results.csv). Tables were rendered from the
JSON with `--report`; nothing was typed in by hand.

## TL;DR

- **Returns: TimesFM-3 zero-shot did not beat the naive baselines.** On MAE and CRPS it is worse
  than simply forecasting a 0 return at 1, 5 and 15 days (the paired 95% CIs exclude 0 for MAE;
  at 1 day the CRPS difference is inside noise). Directional accuracy is 49.9–50.9%, and the
  cross-sectional rank IC is ~0.01–0.02 at best. Adding covariates (KOSPI, volume) changes nothing
  that matters.
- **3-class triple-barrier labels:** every model lands at macro-F1 0.34–0.37 with overlapping CIs.
  The repo's LightGBM (0.350) and LightGBM + TimesFM features (0.355) do not beat a calibrated
  historical-drift baseline (0.369).
- **Cost-aware long-only backtest:** no TimesFM strategy beat the equal-weight universe. Daily
  rebalancing loses heavily to costs for every signal. With 5-day rebalancing, TimesFM top-10 got
  Sharpe 0.39 against 0.81 for equal weight, and LightGBM + TimesFM (0.42) did worse than
  LightGBM alone (0.58).
- **Volatility is where TimesFM helps a little:** for 5-day realized vol it has the lowest MAE of all
  models, beating HAR-RV by a significant but small margin (−0.014 pp). Its median forecast is
  biased low, so QLIKE is worse than EWMA/HAR. Taking the quantile mean closes that gap to a tie
  with HAR. At 20 days, HAR-RV is as good or better.
- Conclusion: in this repo, treat TimesFM-3 as an optional **volatility / uncertainty feature**
  (position sizing, barrier width), not as a return or direction signal. Its weights are
  non-commercial, so it stays in research and paper trading only.

## Setup

| item | value |
|---|---|
| data | FinanceDataReader daily OHLCV (split-adjusted), fetched 2026-09-27; source returns at most ~3000 rows, so history starts 2014-07-04 |
| universe | 50 current KOSPI large caps: `config.KOSPI200_TOP` (40) + 10 more (see `EXTRA_TICKERS`); tickers listed later (e.g. 373220, 259960, 352820) enter once they have 537 days of history |
| index | KOSPI (`KS11`), inner-joined on dates; rows with volume 0 (halts) dropped |
| forecast origins | every trading day t with ≥537 days of history and t+15 available: 113,919 (ticker, day) origins, 2016-09-07 … 2026-08-27 |
| test windows | 8 calendar years 2019 … 2026 (2026 = Jan–Aug); **89,642 test origins** |
| TimesFM | `google/timesfm-3.0-pytorch` (330M), `timesfm==3.0.2`, MLX backend, context 512 days, zero-shot, 9 quantiles q10…q90 |
| label (3-class) | the repo's own `FeatureEngine.get_target_labels` (±1.5·ATR14 barriers, 15-day timeout), called through a small in-memory DB adapter |
| costs (backtest) | from `nn_config.py`: commission 0.015% per side, tax 0.18% on sells, slippage 5 bp per side |

### What goes into each model (no look-ahead)

- **TimesFM `timesfm`** gets the log price relative to the last close, `log C[t-511..t] − log C[t]`, and
  forecasts 15 steps. Step k gives the quantiles of the k-day cumulative log return
  `log C[t+k]/C[t]`, so a single call covers the 1/3/5/15-day horizons.
- **`timesfm_cov`** uses the same target plus two *past-only* covariates: KOSPI log level relative to
  t, and z-scored log volume. Both use data up to t only.
- **`timesfm_retseries`** (ablation) takes daily log returns ×100 as the context. Its point forecast is
  the sum of the per-step medians, so it has no CRPS.
- **Volatility:** the context is the trailing-5 (or 20) day RV series. Step 5 (or 20) of the forecast
  is exactly the RV over t+1…t+5 (or t+20).
- **Naive baselines** use only the trailing 250 days:
  - `zero`: 0 return, Gaussian quantiles with σ_250·√h
  - `last_return`: the last h-day return
  - `hist_drift`: μ_250·h
  - `always_up`: direction only
- **Vol baselines:** persistence (trailing RV_h), trailing 60-day RV, EWMA λ=0.94, and HAR-RV
  (`RV ~ |r_t| + RV5 + RV22`, pooled OLS).
- **LightGBM** is the repo's `LGBMModel` with `nn_config.LGBM_PARAMS` (`n_jobs` set to 12). It is
  trained on a **reduced feature set** of 56 features: the repo's `compute_technical_features` on the
  last 150 bars, plus 5 KOSPI context features. Investor-flow, order-book, US-market and FX features
  are not available from public data. `pykrx` currently fails without KRX credentials, so flows were skipped. Sample
  weights use the repo's class-balance (uniqueness) term only; the regret and time-decay terms need
  the private DB.
- **LightGBM + TimesFM** adds 19 `tfm_*` features. For each h ∈ {1,3,5,15} there are four:
  P(up), median, q90−q10 and skew. On top of those come the barrier P(down)/P(up) and the RV5
  median.
- **TimesFM → 3-class.** `timesfm_barrier_raw` uses the reflection principle on the marginal quantiles:
  P(touch) ≈ max(2·P(X_15 ≥ b), max_k P(X_k ≥ b)), with the remainder assigned to flat.
  `*_calibrated` fits a class-balanced logistic regression on the barrier probabilities plus the
  15-day P(up) and spread, on the training fold only. `hist_drift_*` applies the same mapping to the
  Gaussian historical-drift forecast, so any difference isolates TimesFM's information.
- **Walk-forward.** There is one fold per test year, and every learned model (LightGBM, the
  calibration LR, HAR) is refit each year. A sample is used for training only if its label window
  (t+15 for classes, t+20 for RV20) ends before 1 January of the test year (purging). The last 60
  training dates serve as the LightGBM early-stopping set.
- **Backtest** (`backtest()` in the script):
  - long-only, top 10 by score, 1/10 each, rebalanced every 1 or 5 trading days
  - the signal is taken at the close of t and filled at the **open of t+1**; returns are open-to-open,
    and costs are charged on turnover
  - "rank" variants are always fully invested; `_gated` variants hold only names whose score > 0 and
    keep the rest in cash
  - there is no risk manager, sector cap or Kelly sizing, so this is a signal test, not a replay of
    `PaperEngine`

### Metrics

- **Directional accuracy:** sign agreement, with y = 0 excluded.
- **AUC:** of P(up), or of the score, against up/down.
- **Rank IC:** per-date Spearman correlation across the ~50 names, averaged.
- **MAE:** of the point forecast (median).
- **CRPS:** approximated as 2·mean pinball loss over the 9 deciles. Baselines get Gaussian deciles,
  so the comparison is like for like.
- **Volatility:** MAE, RMSE, QLIKE (on variance) and Mincer–Zarnowitz R².
- **3-class:** macro-F1 over down/flat/up.
- **Backtest:** Sharpe (rf 3.5%, as in `Evaluator`), CAGR and max drawdown.
- **95% CIs:** moving-block bootstrap over test dates (block = 20 days, 1000 reps). For macro-F1 the
  bootstrap resamples 3-month blocks (300 reps). Paired "A − B" rows bootstrap the per-date loss
  difference.

## Results

#### Returns, horizon 1d (test origins 89,642)

| model | dir. acc [95% CI] | AUC | rank IC [95% CI] | MAE ×100 | CRPS ×100 |
|---|---|---|---|---|---|
| zero | — | — | — | 1.7820 | 1.4529 |
| last_return | 0.4628 [0.4558, 0.4689] | 0.4797 | -0.0400 [-0.0500, -0.0287] | 2.6219 | 2.1115 |
| hist_drift | 0.4985 [0.4921, 0.5049] | 0.5002 | +0.0047 [-0.0063, +0.0158] | 1.7894 | 1.4558 |
| timesfm | 0.5063 [0.5019, 0.5107] | 0.5063 | +0.0125 [+0.0051, +0.0198] | 1.8120 | 1.4552 |
| timesfm_cov | 0.5091 [0.5046, 0.5130] | 0.5110 | +0.0148 [+0.0071, +0.0218] | 1.8108 | 1.4533 |
| timesfm_retseries | 0.5054 [0.5006, 0.5110] | 0.5043 | +0.0142 [+0.0052, +0.0238] | 1.7962 | — |
| lgbm | 0.5026 [0.4958, 0.5089] | 0.5031 | +0.0084 [+0.0004, +0.0159] | — | — |
| lgbm_plus_timesfm | 0.5035 [0.4967, 0.5092] | 0.5041 | +0.0035 [-0.0063, +0.0114] | — | — |
| always_up | 0.4952  | — | — | — | — |

`timesfm_minus_zero` (per-date paired, ×100; negative = TimesFM better): MAE +0.0300 [+0.0253, +0.0345], CRPS +0.0024 [-0.0043, +0.0072]

`timesfm_cov_minus_zero` (per-date paired, ×100; negative = TimesFM better): MAE +0.0288 [+0.0245, +0.0331], CRPS +0.0005 [-0.0071, +0.0063]

#### Returns, horizon 5d (test origins 89,642)

| model | dir. acc [95% CI] | AUC | rank IC [95% CI] | MAE ×100 | CRPS ×100 |
|---|---|---|---|---|---|
| zero | — | — | — | 4.0869 | 3.2831 |
| last_return | 0.4848 [0.4750, 0.4955] | 0.4886 | -0.0193 [-0.0358, +0.0012] | 5.9054 | 4.7420 |
| hist_drift | 0.4973 [0.4838, 0.5106] | 0.4975 | +0.0036 [-0.0193, +0.0280] | 4.1392 | 3.3137 |
| timesfm | 0.5054 [0.4981, 0.5111] | 0.5068 | +0.0025 [-0.0114, +0.0163] | 4.2066 | 3.3463 |
| timesfm_cov | 0.5041 [0.4951, 0.5115] | 0.5052 | +0.0027 [-0.0122, +0.0161] | 4.2146 | 3.3522 |
| timesfm_retseries | 0.5047 [0.4949, 0.5155] | 0.5034 | +0.0220 [+0.0056, +0.0418] | 4.1757 | — |
| lgbm | 0.5034 [0.4933, 0.5138] | 0.5071 | +0.0030 [-0.0123, +0.0170] | — | — |
| lgbm_plus_timesfm | 0.5083 [0.4983, 0.5183] | 0.5088 | -0.0024 [-0.0179, +0.0113] | — | — |
| always_up | 0.5067  | — | — | — | — |

`timesfm_minus_zero` (per-date paired, ×100; negative = TimesFM better): MAE +0.1197 [+0.0969, +0.1459], CRPS +0.0633 [+0.0472, +0.0815]

`timesfm_cov_minus_zero` (per-date paired, ×100; negative = TimesFM better): MAE +0.1277 [+0.1044, +0.1542], CRPS +0.0692 [+0.0498, +0.0888]

#### Returns, horizon 15d (test origins 89,642)

| model | dir. acc [95% CI] | AUC | rank IC [95% CI] | MAE ×100 | CRPS ×100 |
|---|---|---|---|---|---|
| zero | — | — | — | 7.1255 | 5.6960 |
| last_return | 0.4839 [0.4691, 0.5011] | 0.4828 | -0.0239 [-0.0535, +0.0066] | 10.3248 | 8.2521 |
| hist_drift | 0.4988 [0.4788, 0.5184] | 0.4972 | +0.0095 [-0.0254, +0.0472] | 7.3782 | 5.8486 |
| timesfm | 0.5074 [0.4969, 0.5163] | 0.5087 | -0.0015 [-0.0235, +0.0203] | 7.4861 | 5.9637 |
| timesfm_cov | 0.4987 [0.4845, 0.5108] | 0.4980 | -0.0077 [-0.0301, +0.0133] | 7.5538 | 6.0043 |
| lgbm | 0.5031 [0.4889, 0.5177] | 0.5032 | -0.0140 [-0.0380, +0.0066] | — | — |
| lgbm_plus_timesfm | 0.5076 [0.4929, 0.5227] | 0.5080 | -0.0145 [-0.0378, +0.0052] | — | — |
| always_up | 0.5144  | — | — | — | — |

`timesfm_minus_zero` (per-date paired, ×100; negative = TimesFM better): MAE +0.3607 [+0.2848, +0.4444], CRPS +0.2677 [+0.2090, +0.3303]

`timesfm_cov_minus_zero` (per-date paired, ×100; negative = TimesFM better): MAE +0.4283 [+0.3280, +0.5184], CRPS +0.3083 [+0.2349, +0.3779]

#### Direction accuracy by test year (horizon 5d)

| model | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| last_return | 0.487 | 0.485 | 0.468 | 0.493 | 0.483 | 0.484 | 0.496 | 0.481 |
| hist_drift | 0.470 | 0.510 | 0.485 | 0.497 | 0.487 | 0.510 | 0.510 | 0.513 |
| timesfm | 0.506 | 0.482 | 0.498 | 0.506 | 0.506 | 0.515 | 0.517 | 0.514 |
| timesfm_cov | 0.508 | 0.484 | 0.497 | 0.504 | 0.502 | 0.514 | 0.506 | 0.521 |
| timesfm_retseries | 0.490 | 0.519 | 0.502 | 0.505 | 0.492 | 0.514 | 0.513 | 0.500 |
| lgbm | 0.495 | 0.503 | 0.499 | 0.505 | 0.513 | 0.509 | 0.495 | 0.511 |
| lgbm_plus_timesfm | 0.503 | 0.515 | 0.508 | 0.501 | 0.509 | 0.512 | 0.504 | 0.517 |

#### Rank IC by test year (horizon 5d)

| model | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| last_return | -0.029 | -0.027 | -0.009 | +0.010 | -0.024 | -0.037 | -0.018 | -0.021 |
| hist_drift | -0.040 | +0.043 | -0.043 | -0.014 | -0.005 | +0.050 | +0.027 | +0.016 |
| timesfm | +0.014 | -0.055 | +0.003 | +0.001 | -0.006 | +0.010 | +0.036 | +0.028 |
| timesfm_cov | +0.019 | -0.046 | +0.012 | -0.001 | -0.004 | +0.012 | +0.014 | +0.021 |
| timesfm_retseries | -0.027 | +0.043 | +0.003 | +0.002 | +0.014 | +0.065 | +0.053 | +0.025 |
| lgbm | +0.030 | -0.013 | +0.008 | -0.001 | -0.006 | +0.016 | -0.015 | +0.003 |
| lgbm_plus_timesfm | +0.004 | +0.011 | +0.016 | +0.011 | -0.021 | +0.003 | -0.033 | -0.017 |

#### 3-class triple barrier (±1.5·ATR14, 15d) — test class counts down/flat/up = [38263, 7701, 43678]

| model | macro-F1 [95% CI] | accuracy | predicted down/flat/up |
|---|---|---|---|
| majority_class | 0.2184 [0.2106, 0.2247] | 0.4872 | [0, 0, 89642] |
| hist_drift_barrier | 0.3423 [0.3246, 0.3629] | 0.4615 | [39602, 1153, 48887] |
| hist_drift_calibrated | 0.3686 [0.3485, 0.3835] | 0.4189 | [41005, 13405, 35232] |
| timesfm_barrier_raw | 0.3378 [0.3272, 0.3486] | 0.4605 | [47739, 681, 41222] |
| timesfm_calibrated | 0.3586 [0.3416, 0.3728] | 0.3941 | [29177, 23769, 36696] |
| lgbm | 0.3504 [0.3387, 0.3600] | 0.3887 | [30987, 22441, 36214] |
| lgbm_plus_timesfm | 0.3545 [0.3434, 0.3644] | 0.3946 | [31227, 21892, 36523] |

macro-F1 by test year:

| model | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| majority_class | 0.208 | 0.231 | 0.210 | 0.202 | 0.220 | 0.217 | 0.233 | 0.227 |
| hist_drift_barrier | 0.280 | 0.372 | 0.267 | 0.297 | 0.322 | 0.355 | 0.318 | 0.345 |
| hist_drift_calibrated | 0.304 | 0.357 | 0.369 | 0.372 | 0.369 | 0.344 | 0.364 | 0.358 |
| timesfm_barrier_raw | 0.329 | 0.322 | 0.329 | 0.343 | 0.327 | 0.326 | 0.335 | 0.371 |
| timesfm_calibrated | 0.308 | 0.373 | 0.363 | 0.366 | 0.370 | 0.334 | 0.349 | 0.235 |
| lgbm | 0.338 | 0.334 | 0.363 | 0.349 | 0.370 | 0.330 | 0.332 | 0.329 |
| lgbm_plus_timesfm | 0.343 | 0.349 | 0.364 | 0.355 | 0.385 | 0.329 | 0.318 | 0.339 |

#### Volatility, 5-day realized vol (daily units)

| model | MAE ×100 [95% CI] | RMSE ×100 | QLIKE [95% CI] | MZ R² |
|---|---|---|---|---|
| persistence | 1.0668 [0.9937, 1.1406] | 1.5786 | 1.3402 [1.2488, 1.4544] | 0.2027 |
| hist_60d | 0.9515 [0.8808, 1.0213] | 1.3780 | 0.5650 [0.5050, 0.6435] | 0.2288 |
| ewma_094 | 0.9374 [0.8660, 1.0076] | 1.3561 | 0.5432 [0.5027, 0.5925] | 0.2622 |
| har_rv | 0.8629 [0.7986, 0.9286] | 1.3136 | 0.6082 [0.5476, 0.6829] | 0.2462 |
| timesfm | 0.8486 [0.7801, 0.9182] | 1.3189 | 0.6715 [0.6008, 0.7598] | 0.2555 |
| timesfm_qmean | 0.8540 [0.7872, 0.9218] | 1.3088 | 0.6178 [0.5551, 0.6967] | 0.2600 |

Paired per-date loss differences (negative = TimesFM better):

| comparison | mean | 95% CI |
|---|---|---|
| timesfm_minus_har_rv_mae | -0.01439 | [-0.02441, -0.00498] |
| timesfm_minus_har_rv_qlike | +0.06330 | [+0.04222, +0.08963] |
| timesfm_minus_ewma_094_mae | -0.08888 | [-0.11581, -0.06057] |
| timesfm_minus_ewma_094_qlike | +0.12832 | [+0.09048, +0.17303] |
| timesfm_minus_hist_60d_mae | -0.10292 | [-0.13522, -0.07182] |
| timesfm_minus_hist_60d_qlike | +0.10650 | [+0.07638, +0.14262] |
| timesfm_qmean_minus_har_rv_mae | -0.00891 | [-0.01870, -0.00009] |
| timesfm_qmean_minus_har_rv_qlike | +0.00957 | [-0.00652, +0.02869] |
| timesfm_qmean_minus_ewma_094_mae | -0.08340 | [-0.10900, -0.05798] |
| timesfm_qmean_minus_ewma_094_qlike | +0.07460 | [+0.04557, +0.10987] |
| timesfm_qmean_minus_hist_60d_mae | -0.09744 | [-0.12891, -0.06887] |
| timesfm_qmean_minus_hist_60d_qlike | +0.05277 | [+0.02889, +0.07995] |

#### Volatility, 20-day realized vol (daily units)

| model | MAE ×100 [95% CI] | RMSE ×100 | QLIKE [95% CI] | MZ R² |
|---|---|---|---|---|
| persistence | 0.7906 [0.7098, 0.8758] | 1.1841 | 0.4390 [0.3717, 0.5215] | 0.2913 |
| hist_60d | 0.7175 [0.6405, 0.7878] | 1.0659 | 0.3395 [0.2565, 0.4498] | 0.3313 |
| ewma_094 | 0.7256 [0.6498, 0.7974] | 1.0849 | 0.3392 [0.2754, 0.4203] | 0.3401 |
| har_rv | 0.6883 [0.6142, 0.7675] | 1.0468 | 0.3547 [0.2838, 0.4477] | 0.3056 |
| timesfm | 0.7016 [0.6206, 0.7836] | 1.0855 | 0.3710 [0.2906, 0.4694] | 0.3085 |
| timesfm_qmean | 0.6982 [0.6181, 0.7765] | 1.0709 | 0.3464 [0.2702, 0.4438] | 0.3186 |

Paired per-date loss differences (negative = TimesFM better):

| comparison | mean | 95% CI |
|---|---|---|
| timesfm_minus_har_rv_mae | +0.01334 | [-0.01472, +0.04350] |
| timesfm_minus_har_rv_qlike | +0.01630 | [-0.00449, +0.03874] |
| timesfm_minus_ewma_094_mae | -0.02398 | [-0.04031, -0.00673] |
| timesfm_minus_ewma_094_qlike | +0.03181 | [+0.01146, +0.05657] |
| timesfm_minus_hist_60d_mae | -0.01590 | [-0.04151, +0.02082] |
| timesfm_minus_hist_60d_qlike | +0.03152 | [+0.01053, +0.05571] |
| timesfm_qmean_minus_har_rv_mae | +0.00992 | [-0.01984, +0.03870] |
| timesfm_qmean_minus_har_rv_qlike | -0.00829 | [-0.02840, +0.01228] |
| timesfm_qmean_minus_ewma_094_mae | -0.02740 | [-0.04176, -0.01247] |
| timesfm_qmean_minus_ewma_094_qlike | +0.00721 | [-0.00990, +0.02805] |
| timesfm_qmean_minus_hist_60d_mae | -0.01932 | [-0.04346, +0.01408] |
| timesfm_qmean_minus_hist_60d_qlike | +0.00692 | [-0.01539, +0.02940] |

#### Backtest, rebalance every 1d, top-10 equal weight, costs on

| strategy | Sharpe [95% CI] | CAGR % | MDD % | total % |
|---|---|---|---|---|
| equal_weight_universe | 0.82 [0.01, 1.55] | 20.5 | -39.0 | 302 |
| timesfm | -0.92 [-1.65, -0.30] | -20.8 | -86.5 | -82 |
| timesfm_gated | -0.88 [-1.62, -0.27] | -19.6 | -85.0 | -80 |
| hist_drift | 0.80 [0.08, 1.49] | 26.9 | -41.0 | 489 |
| hist_drift_gated | 0.80 [0.06, 1.49] | 26.2 | -41.0 | 468 |
| last_return | -2.88 [-3.73, -2.08] | -54.6 | -99.7 | -100 |
| last_return_gated | -2.86 [-3.69, -2.04] | -51.7 | -99.6 | -100 |
| lgbm | -0.18 [-0.84, 0.42] | -5.5 | -57.5 | -34 |
| lgbm_gated | -0.19 [-0.88, 0.44] | -5.4 | -56.9 | -34 |
| lgbm_plus_timesfm | -0.20 [-0.93, 0.40] | -6.1 | -63.8 | -38 |
| lgbm_plus_timesfm_gated | -0.15 [-0.87, 0.48] | -4.4 | -60.0 | -28 |
| timesfm_cov | -0.75 [-1.50, -0.12] | -17.6 | -86.1 | -76 |
| timesfm_cov_gated | -0.73 [-1.49, -0.10] | -17.0 | -85.2 | -75 |

#### Backtest, rebalance every 5d, top-10 equal weight, costs on

| strategy | Sharpe [95% CI] | CAGR % | MDD % | total % |
|---|---|---|---|---|
| equal_weight_universe | 0.81 [-0.00, 1.54] | 20.2 | -39.1 | 294 |
| timesfm | 0.39 [-0.35, 1.07] | 10.9 | -52.8 | 116 |
| timesfm_gated | 0.39 [-0.35, 1.07] | 10.9 | -52.8 | 116 |
| hist_drift | 0.93 [0.21, 1.62] | 32.1 | -41.3 | 698 |
| hist_drift_gated | 0.91 [0.18, 1.59] | 30.9 | -41.3 | 643 |
| last_return | 0.25 [-0.48, 0.94] | 6.7 | -46.8 | 63 |
| last_return_gated | 0.27 [-0.44, 0.96] | 7.3 | -45.3 | 69 |
| lgbm | 0.58 [-0.16, 1.21] | 17.0 | -46.2 | 223 |
| lgbm_gated | 0.54 [-0.17, 1.17] | 15.5 | -44.4 | 193 |
| lgbm_plus_timesfm | 0.42 [-0.32, 1.04] | 11.9 | -46.6 | 131 |
| lgbm_plus_timesfm_gated | 0.43 [-0.28, 1.05] | 12.1 | -43.4 | 135 |
| timesfm_cov | 0.23 [-0.46, 0.90] | 6.3 | -52.6 | 58 |
| timesfm_cov_gated | 0.23 [-0.46, 0.89] | 6.2 | -52.6 | 56 |

KOSPI index (close-to-close, no costs, same period): Sharpe 0.63, MDD -38.6%, total 244%

Sharpe by test year (rebalance 5d, rank variant):

| strategy | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| timesfm | 0.16 | 0.30 | 0.49 | -0.71 | 0.76 | -0.51 | 1.69 | 0.94 |
| hist_drift | -0.20 | 1.83 | 0.49 | -1.24 | 0.94 | 0.68 | 2.13 | 1.61 |
| last_return | -1.04 | 0.88 | 0.43 | -0.92 | -0.06 | -0.01 | 1.65 | 0.19 |
| lgbm | 0.59 | 0.62 | 0.76 | -0.83 | 0.69 | -0.13 | 1.72 | 1.31 |
| lgbm_plus_timesfm | 0.19 | 0.76 | 0.12 | -0.19 | 0.28 | 0.32 | 1.19 | 0.52 |
| timesfm_cov | -0.08 | 0.32 | 0.49 | -1.07 | 0.04 | -0.55 | 1.87 | 0.69 |
| equal_weight_universe | 0.16 | 1.06 | 0.63 | -0.75 | 1.30 | 0.24 | 2.75 | 1.20 |


## Reading the results honestly

- **Return forecasts sit at the noise floor.** Beating a zero forecast on MAE for daily or weekly
  large-cap returns is hard, and TimesFM does not do it. Its median wanders away from 0, which costs
  MAE. The largest directional edge is +0.9 pp at 1 day (`timesfm_cov`, 50.9%). A 1-day rank IC of
  0.013–0.015 is statistically nonzero over ~1,900 test dates, but it is far too small to survive the
  ~0.31% round-trip cost. The daily-rebalance backtest shows exactly that.
- **`timesfm_retseries` at 5 days** has a rank IC of +0.022 with a CI that excludes 0. It is one of
  dozens of IC estimates in these tables and its sign flips across years (2019 −0.027, 2024 +0.065), so
  treat it as a multiple-testing candidate, not a finding.
- **TimesFM features did not help LightGBM here.** For 3-class macro-F1 the difference is +0.004,
  inside the CI. In the 5-day backtest LightGBM + TimesFM did worse than LightGBM alone. The
  LightGBM here uses a reduced feature set, so this does not prove anything about the full 93-feature
  model on the private DB.
- **Momentum baseline.** `hist_drift` is the 250-day mean return, effectively 12-month momentum. It
  had the best backtest Sharpe (0.93 at 5 days). That is almost certainly inflated by survivorship
  bias: the universe is *today's* large caps, which by construction were winners.
- **Volatility.** TimesFM's 5-day vol median has the lowest MAE of all models, and the paired CI
  against HAR-RV excludes 0. The margin is small (0.863 → 0.849 ×100). Under QLIKE, which penalizes
  under-prediction, the median is worse than EWMA/HAR. Using the mean of the 9 quantiles makes it
  statistically tied with HAR. At 20 days HAR-RV is at least as good. A zero-shot model matching a
  fitted HAR is a respectable result; clearly beating it would
  likely need fine-tuning, which was not tried.

## Caveats

- **Survivorship bias:** the universe is current large caps. Absolute backtest returns
  (equal-weight +294%) are inflated. Only the comparison between strategies on the same universe
  is meaningful.
- **Data:** FinanceDataReader's default source for Korean tickers. Prices are split-adjusted
  (Samsung's 2018 50:1 split is back-adjusted), and this script applies no dividend adjustment.
  There are no investor flows (pykrx currently fails without KRX credentials, `KRX_ID`/`KRX_PW`)
  and no delisted names.
- **Pretraining overlap:** per the model card, TimesFM-3 was pretrained on GiftEvalPretrain
  (excluding fev-bench overlaps), Wikipedia pageviews (cutoff Nov 2023), Google Trends (cutoff 2022)
  and synthetic data. Korean single-stock prices are unlikely to be in it, but that cannot be
  verified. The 2024–2026 windows fall after the stated cutoffs and show no
  consistent return edge either.
- **Approximations:** the CRPS uses 9 quantiles only, and the tails beyond q10/q90 are ignored. The
  barrier-touch probability comes from marginal quantiles via the reflection principle, not from
  sampled paths.
- **Backtest scope:** a simple top-10 selector. There are no position limits, no daily loss limit,
  and no ML-based exits from `PaperEngine`.
- **Repo observation (not changed):** `Trainer._compute_sample_weights` applies time decay by row
  index. `get_feature_matrix` orders rows by ticker and then date, so the decay favours the
  last-listed tickers rather than recent dates. This eval does not use that code path.
- **License:** TimesFM 3.0 weights are `timesfm-non-commercial-license-v1.0`, for non-commercial and
  non-production use only. `trading/live_engine.py` refuses to load TimesFM or run models with
  `tfm_*` features.

## Runtime (M4 Max 16-core CPU / 40-core GPU, 64 GB, MLX)

| step | origins | time |
|---|---|---|
| data fetch + features (56 × 113,919) | — | 53 s |
| TimesFM `ret` (univariate, h=15) | 113,919 | 219 s (≈520/s, batch 256) |
| TimesFM `rv5` | 113,919 | 208 s |
| TimesFM `rv20` | 89,642 | 145 s |
| TimesFM `retseries` | 89,642 | 139 s |
| TimesFM `ret_cov` (covariates, per-series path) | 89,642 | 1,532 s (≈58/s) |
| walk-forward fits (8 folds × LightGBM ×2, LR, HAR) | — | 95 s |
| **total** | | **2,418 s (40 min)** |

The first model load downloads the weights (~1.3 GB) into the Hugging Face cache. The covariate
path in `timesfm3.mlx` runs one series at a time, which is why it is 10× slower. Nothing was
subsampled: all 50 tickers and every trading-day origin were used.

## Reproduce

```bash
brew install uv libomp            # libomp: LightGBM on macOS
uv venv --python 3.12 .venv       # git-ignored; system python 3.14 is too new for the stack
uv pip install --python .venv/bin/python -r requirements-timesfm.txt
.venv/bin/python -m models.eval_timesfm            # fetch → cache → evaluate → docs/timesfm_eval_results.{json,csv}
.venv/bin/python -m models.eval_timesfm --report   # print the tables in this document
.venv/bin/python -m models.eval_timesfm --tickers 8 --no-cov   # quick check (a few minutes)
```

Caches (prices, samples, forecasts) go to `data/public/`, which is git-ignored.
Use `--refresh` to re-download and recompute.

Versions from this run: Python 3.12.13 (uv), timesfm 3.0.2, mlx 0.32.2, lightgbm 4.7.0,
numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, finance-datareader 0.9.202, macOS 26.5 arm64.
