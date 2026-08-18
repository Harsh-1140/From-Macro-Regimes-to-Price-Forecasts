# From Macro Regimes to Price Forecasts: A Two-Stage Statistical Pipeline for Crude-Oil Prediction

[![R](https://img.shields.io/badge/Language-R_4.x-blue.svg)](https://www.r-project.org/)
[![Course](https://img.shields.io/badge/IIT_Kanpur-MTH443-orange.svg)](https://www.iitk.ac.in/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

> **Course Project — MTH443**  
> **Department of Statistics and Data Science, Indian Institute of Technology Kanpur (IIT Kanpur)**  

This repository contains the complete statistical pipeline and modeling framework for the course project of **MTH443**. The objective is to endogenously classify macroeconomic regimes from multi-source financial and policy uncertainty indicators, and condition a 9-stage stacked ensemble pipeline (**HYDRA**) on late-cycle market environments to accurately forecast daily WTI crude oil prices.

---

## Problem Overview & Dataset

* **Test Benchmark**: Held-out out-of-sample window of **December 2024 (19 trading days)**.
* **Macroeconomic Panel (2000–2025, 6,783 trading days)**:
  * **CBOE Volatility Index (`^VIX`)**: Implied 30-day S&P 500 volatility from Yahoo Finance.
  * **10Y & 2Y Treasury Constant Maturity Yields (`DGS10`, `DGS2`)**: Federal Reserve Economic Data (FRED) API to compute term spread.
  * **Economic Policy Uncertainty Index (`EPU`)**: News-based policy uncertainty index from Baker, Bloom, & Davis.
  * **Geopolitical Risk Daily Index (`GPRD`)**: Newspaper-based geopolitical risk metric from Caldara & Iacoviello.
  * **Cross-Asset Reference Series**: Front-month WTI Crude (`CL=F`), COMEX Gold (`GC=F`), S&P 500 (`^GSPC`), and US Dollar Index (`DX-Y.NYB`).
* **Commodity Panel**: Daily closing prices for WTI crude oil, gas oil, heating oil, and natural gas from Investing.com.
* **Evaluation Metric**: Root Mean Squared Error (**RMSE**), Mean Absolute Error (**MAE**), and out-of-sample $R^2$ score.

---

## Methodology & Pipeline Architecture

### Stage 1: Macro-Regime Discovery (2000–2025)
Using $K$-Means clustering ($K=4$, validated via the Elbow method) on five standardized macro-financial features (VIX, 10Y–2Y yield spread, EPU index, GPRD index, and gold-to-oil ratio), we endogenously recover four persistent economic states without arbitrary *ex-ante* cutoffs:
1. **$R3$ (Green) — Calm Expansion (3,204 days)**: Low volatility, steep yield curve, subdued policy uncertainty.
2. **$R2$ (Blue) — Acute Crisis (638 days)**: Extreme VIX and GPRD, rapid Fed easing (2001, 2008 GFC, 2020 COVID crash).
3. **$R1$ (Red) — Late-Cycle / Moderate Stress (2,443 days)**: Flattened yield curve, moderate VIX, elevated policy uncertainty.
4. **$R4$ (Purple) — Policy-Uncertainty Dominated (437 days)**: Extremely high EPU index with compressed cyclical commodity demand (2020–2022).

### Stage 2: The 9-Stage HYDRA Forecasting Pipeline
Conditioning exclusively on the $R1$ (Red) late-cycle sub-sample (2,443 trading days):
1. **Preprocessing & Lag Engineering**: Constructs target lags (lags 1, 2, 3, 5) and includes S&P 500 and stress indices.
2. **Stress Feature PCA**: Extracts PC1 and PC2 from standardized stress features, explaining **81.8%** of the stress space variance.
3. **$K$-Means Micro-Regimes ($K=3$)**: Discovers *Low-Stress*, *Med-Stress*, and *High-Stress* states and calculates soft-membership weights via normalized negative distance softmax.
4. **Fisher Linear Discriminant Analysis (LDA)**: Supervised projection maximizing micro-cluster separation along two discriminant coordinates.
5. **Gaussian Naive Bayes**: Computes calibrated posterior class probabilities across the micro-regimes as additional meta-features.
6. **Regime-Expert Regressors**: Fits specialized learners: SVR ($\text{cost}=200, \epsilon=0.3$) for High-Stress, GBM (150 trees, depth 4) for Med-Stress, and MLP (8 hidden units) for Low-Stress, blended via soft-membership weights.
7. **Global Regressors**: Trains unconstrained global models (SVR, MLP, Random Forest, GBM) evaluated with 5-fold rolling-origin time-series cross-validation.
8. **Level-2 Meta-Stacker**: A Gradient-Boosted Decision Tree model (100 trees, depth 3, $\eta=0.05$) trained on Level-1 predictions to dynamically combine expert and global signals.
9. **Test Evaluation**: Final performance assessment against the December 2024 held-out test series.

---

## Model Performance Benchmark

Evaluated on the December 2024 holdout set (19 trading days):

| Model | Test RMSE (USD) | Test MAE (USD) | Test $R^2$ Score | Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **Global Random Forest** | **$0.775** | **$0.624** | **0.372** | Best single model; effectively leverages AR lag-1 correlation ($r = 0.99$) |
| **Global Gradient Boosting** | $0.810 | $0.686 | 0.314 | Forward stagewise additive regression tree ensemble |
| **HYDRA (Meta-Stacker)** | **$0.813** | **$0.627** | **0.309** | 9-stage stacked ensemble combining expert & global learners |
| **Global SVR** | $1.038 | $0.880 | -0.127 | Radial basis kernel $\epsilon$-insensitive support vector regressor |
| **Regime-Expert Blend** | $2.503 | $1.968 | -5.549 | Soft-weighted blend of 3 sub-regime expert models |
| **Global MLP (Neural Net)** | $4.476 | $4.361 | -19.952 | Failed to capture tabular linear/autoregressive persistence |

> **Key Takeaway**: While tree ensembles dominate autoregressive prediction, **HYDRA provides robust insurance against base-learner failure**. Despite the failure of the standalone Global MLP ($RMSE = \$4.48$), the meta-stacker automatically discounted erroneous signals and tied the best tree models ($RMSE = \$0.81$).

---

## Pipeline Execution & Verification

To execute the entire statistical pipeline from scratch, run the scripts in sequential order:

```R
# 1. Fetch external data and compile the 2000–2025 macro panel
source("Data Scraping.R")

# 2. Perform Stage 1 Macro-Regime K-Means clustering & extract the Red Regime
source("Macro_regimes.R")

# 3. Generate exploratory statistical plots, distributions, and correlation matrices
source("01_EDA.R")

# 4. Train the 9-Stage HYDRA Pipeline & evaluate on Dec 2024 test data
source("02_HYDRA_Pipeline.R")

# 5. Output publication-ready diagnostic charts and residual plots
source("03_Results_Plots.R")
```
## Repository Structure
``` text
├── Data Scraping.R             # Script 0: Automated data ingestion (FRED, Yahoo, EPU, GPRD)
├── Macro_regimes.R             # Script 1: Stage 1 Macro-regime K-Means clustering & visualizations
├── 01_EDA.R                    # Script 2: Exploratory data analysis, distributions & correlation matrix
├── 02_HYDRA_Pipeline.R         # Script 3: Complete 9-stage HYDRA modeling & stacking engine
├── 03_Results_Plots.R          # Script 4: Generates evaluation charts, biplots & residual diagnostics
│
├── regime_data_2000_2025.csv   # Unified multi-source macroeconomic panel (2000–2025)
├── combined_prices.csv         # Daily energy commodity panel (Investing.com)
│
├── outputs/                    # Exported figures, tables & model objects
│   ├── eda_01_crude_oil_timeseries.png
│   ├── eda_02_distributions.png
│   ├── eda_03_correlation_matrix.png
│   ├── eda_04_stress_indicators.png
│   ├── eda_05_scatter_predictors.png
│   ├── results_01_actual_vs_predicted.png
│   ├── results_02_error_bar.png
│   ├── results_03_model_comparison.png
│   ├── results_04_pca_biplot.png
│   ├── results_05_regime_over_time.png
│   ├── results_06_all_models_faceted.png
│   ├── eda_summary_stats.csv
│   ├── hydra_model_comparison.csv
│   ├── hydra_predictions_dec2024.csv
│   └── hydra_objects.rds
│
├── 443_Report.pdf              # Comprehensive academic research report
└── README.md                   # Project documentation
```
### Authors 
Akshat Saxena (230099)

Aryan Deo (230213)

Harsh Agrawalla (230443)

Rupant Dixit (230883)

Utkarsh Kesharwani (231108)


### References
Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.

Caldara, D., & Iacoviello, M. (2022). Measuring geopolitical risk. American Economic Review, 112(4), 1194–1225.

Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. Quarterly Journal of Economics, 131(4), 1593–1636.

Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
