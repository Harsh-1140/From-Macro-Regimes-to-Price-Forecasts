```markdown
# From Macro Regimes to Price Forecasts: A Two-Stage Statistical Pipeline for Crude-Oil Prediction

[![R](https://img.shields.io/badge/Language-R_4.x-blue.svg)](https://www.r-project.org/)
[![Course](https://img.shields.io/badge/IIT_Kanpur-MTH443-orange.svg)](https://www.iitk.ac.in/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

This repository contains the complete statistical pipeline, data scraping workflows, exploratory analysis, and predictive modeling for the **MTH443 Course Project** at **Indian Institute of Technology Kanpur (IIT Kanpur)**.

---

## Executive Summary

Crude oil prices are governed by a complex mixture of demand fundamentals, supply elasticity, and macro-financial risk factors. Standard global models treating all market days uniformly suffer from parameter instability due to unobserved business cycle shifts.

We implement a **two-stage statistical framework**:
1. **Stage 1: Macro-Regime Identification (2000–2025)**: Endogenous classification using $K$-Means clustering over five standardized macro-financial indicators (VIX, 10Y–2Y Treasury spread, Economic Policy Uncertainty, Geopolitical Risk Daily, and Gold-to-Oil ratio). Four distinct, highly persistent regimes emerge: *Calm Expansion*, *Late-Cycle / Moderate Stress*, *Acute Crisis*, and *Policy Uncertainty-Dominated*.
2. **Stage 2: Conditional Forecasting with HYDRA**: Isolating the late-cycle regime (the "red" regime, 2,443 trading days) to fit **HYDRA**—a 9-stage hybrid pipeline combining PCA, K-Means micro-clustering, Fisher LDA, Naive Bayes, regime-specialized expert regressors (SVR, GBM, MLP), four global learners, and a gradient-boosted meta-stacker.

---

## Model Performance Benchmark

Evaluated on the held-out test window of **December 2024 (19 trading days)**:

| Model | Test RMSE (USD) | Test MAE (USD) | $R^2$ Score | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Global Random Forest** | **$0.775** | **$0.624** | **0.372** | Bagged tree ensemble exploiting autoregressive features |
| **Global Gradient Boosting** | $0.810 | $0.686 | 0.314 | Forward stagewise additive gradient boosting |
| **HYDRA (Meta-Stacker)** | **$0.813** | **$0.627** | **0.309** | 9-stage stacked ensemble combining expert & global learners |
| **Global SVR** | $1.038 | $0.880 | -0.127 | Radial basis kernel Support Vector Regression |
| **Regime-Expert Blend** | $2.503 | $1.968 | -5.549 | Soft-weighted blend of 3 sub-regime expert models |
| **Global MLP** | $4.476 | $4.361 | -19.952 | Single-hidden-layer neural network |

> **Key Finding**: While pure tree ensembles dominate short-horizon autoregressive forecasting, **HYDRA provides robust insurance against base-learner failure**—safely recovering from the failure of the Global MLP ($RMSE = \$4.48$) and Expert Blend ($RMSE = \$2.50$) to tie the best-performing tree models ($RMSE = \$0.81$).

---

## System Architecture


```

# ========================================================================================
STAGE 1: MACRO-REGIME CLASSIFICATION (2000–2025)
Features: VIX | 10Y-2Y Spread | EPU Index | GPRD Index | Gold/Oil Ratio (Standardized)

```
                                       │
                          [K-Means Clustering: K = 4]
                                       │
  ┌────────────────────┬───────────────┴───────────────┬───────────────────┐
  ▼                    ▼                               ▼                   ▼

```

# R3 (Green)           R2 (Blue)                       R1 (Red)            R4 (Purple)
Calm Expansion       Acute Crisis                  Late-Cycle / Stress   Policy Uncertainty
3,204 days           638 days                        2,443 days             437 days
(2003-07, 2013-15) (2008 GFC, 2020 Crash)         (Post-2015, 2023-24)      (2020–2022)
│
[Extract Red Regime Dataset]
│
=================================▼==
STAGE 2: THE HYDRA FORECASTING PIPELINE

# [1] Preprocessing: Autoregressive lags (1, 2, 3, 5 days) + S&P 500 & Stress Features
[2] PCA on Stress Space: PC1 and PC2 explain 81.8% of stress-feature variance
[3] K-Means Micro-Regimes (K = 3): Low-Stress, Med-Stress, High-Stress + Softmax Weights
[4] Fisher Linear Discriminant Analysis: Supervised projection via micro-regime labels
[5] Naive Bayes: Multivariate class-posterior generation as meta-features
[6] Regime-Expert Regressors:
• High-Stress Expert: SVR (RBF Kernel, C=200, eps=0.3)
• Med-Stress Expert : GBM (150 trees, depth=4, lr=0.08)
• Low-Stress Expert : MLP (8 hidden units, decay=1e-3)
→ Blended via soft membership weights
[7] Global Regressors: SVR, MLP, Random Forest, GBM (5-Fold Rolling TS-CV)
[8] Meta-Stacker: Gradient Boosted Trees trained on Level-1 predictions
[9] Test Evaluation: December 2024 Out-of-Sample Performance

```

---

## Repository Structure


```

├── Data Scraping.R             # Script 0: Automated data retrieval (FRED, Yahoo, EPU, GPRD)
├── Macro_regimes.R             # Script 1: Stage 1 Macro-regime K-Means & barcode visualization
├── 01_EDA.R                    # Script 2: Exploratory data analysis, distributions & correlation matrix
├── 02_HYDRA_Pipeline.R         # Script 3: Complete 9-stage HYDRA pipeline and modeling engine
├── 03_Results_Plots.R          # Script 4: Diagnostic visual generation & error analysis
│
├── regime_data_2000_2025.csv   # Aligned macro panel (2000–2025)
├── combined_prices.csv         # Daily commodity price dataset (Investing.com)
├── merged_market_data.csv      # Merged cross-asset master dataset
├── merged_red_regime_data.csv  # Filtered R1 (Late-Cycle) dataset
│
├── outputs/                    # Output directory for exported figures, tables & RDS objects
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
├── 443_Report.pdf              # Comprehensive project report
└── README.md                   # Project documentation

```

---

## Installation & Setup

Ensure you have R installed ($\ge 4.2$). Install the required CRAN packages:

```R
install.packages(c(
  "quantmod", "fredr", "readr", "dplyr", "lubridate", "tidyr", 
  "zoo", "httr", "readxl", "roll", "ggplot2", "factoextra", 
  "cluster", "corrplot", "gridExtra", "scales", "MASS", 
  "e1071", "nnet", "randomForest", "gbm", "caret"
))

```

---

## Pipeline Execution

Run the scripts in sequential order:

```R
# Step 1: Collect macro-financial indicators and build the 2000–2025 master panel
source("Data Scraping.R")

# Step 2: Run Stage 1 Macro-Regime K-Means clustering and filter to Red Regime
source("Macro_regimes.R")

# Step 3: Run Exploratory Data Analysis & generate distribution/correlation plots
source("01_EDA.R")

# Step 4: Run the 9-Stage HYDRA Pipeline & evaluate on Dec 2024 test data
source("02_HYDRA_Pipeline.R")

# Step 5: Generate evaluation figures and comparison charts
source("03_Results_Plots.R")

```

---

## Authors

* **Akshat Saxena** (230099)
* **Aryan Deo** (230213)
* **Harsh Agrawalla** (230443)
* **Rupant Dixit** (230883)
* **Utkarsh Kesharwani** (231108)

*Department of Mathematics and Statistics, Indian Institute of Technology Kanpur*

---

## References

* **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5–32.
* **Caldara, D., & Iacoviello, M. (2022).** Measuring geopolitical risk. *American Economic Review*, 112(4), 1194–1225.
* **Baker, S. R., Bloom, N., & Davis, S. J. (2016).** Measuring economic policy uncertainty. *Quarterly Journal of Economics*, 131(4), 1593–1636.
* **Wolpert, D. H. (1992).** Stacked generalization. *Neural Networks*, 5(2), 241–259.

```

```
