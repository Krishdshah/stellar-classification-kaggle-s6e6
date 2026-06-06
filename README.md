# Stellar Classification - Kaggle Season 6 Episode 6

Repository for the Stellar Classification Kaggle competition (tabular dataset containing astronomical spectroscopic details of stars, galaxies, and quasars).

## Directory Structure
```
stellar-classification-kaggle-s6e6/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda_feature_engineering.ipynb
│   ├── 02_catboost_gpu_baseline.ipynb
│   ├── 03_lgbm_xgb_models.ipynb
│   └── 04_ensemble_submission.ipynb
│
├── submissions/
│   ├── submission_catboost.csv
│   ├── submission_lgbm.csv
│   ├── submission_xgb.csv
│   ├── submission_ensemble.csv
│   └── leaderboard_tracking.csv
│
├── models/
│   ├── catboost_fold1.cbm
│   ├── catboost_fold2.cbm
│   ├── catboost_fold3.cbm
│   ├── catboost_fold4.cbm
│   ├── catboost_fold5.cbm
│   └── .gitkeep
│
├── outputs/
│   ├── feature_importance.csv
│   ├── cv_scores.csv
│   ├── confusion_matrix.png
│   ├── target_distribution.png
│   ├── redshift_distribution.png
│   └── feature_correlation.png
│
├── reports/
│   ├── eda_report.csv
│   └── experiment_log.md
│
└── src/
    ├── features.py
    ├── train.py
    ├── inference.py
    └── utils.py
```

## Get Started
1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download Datasets**: Place `train.csv` and `test.csv` inside the `data/` directory.
3. **Run EDA**: Work through the notebooks starting with `notebooks/01_eda_feature_engineering.ipynb`.
