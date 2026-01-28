"""
Feature Engineering & Selection Pipeline for Regression (Numerical Only)

Author: ---
Use case: Banking / Deposit Outflow Regression
"""

import numpy as np
import pandas as pd

from scipy.stats import skew
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer

from lightgbm import LGBMRegressor
from arfs.feature_selection import GrootCV


def split_identifier_columns(X: pd.DataFrame, id_cols: list):
    """
    Separates identifier columns from feature columns.

    Returns:
    - X_id: identifier dataframe
    - X_feat: feature dataframe
    """
    X_id = X[id_cols].copy()
    X_feat = X.drop(columns=id_cols)
    return X_id, X_feat


# ============================================================
# 1. Drop categorical (string) columns
# ============================================================

def drop_string_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drops all string/object dtype columns.
    """
    string_cols = X.select_dtypes(include=["object", "string"]).columns
    return X.drop(columns=string_cols)


# ============================================================
# 2. Outlier treatment (1st–99th percentile capping)
# ============================================================

def cap_outliers(
    X: pd.DataFrame,
    q_low: float = 0.01,
    q_high: float = 0.99
) -> pd.DataFrame:
    """
    Caps numerical features at given quantiles.
    """
    X_out = X.copy()
    for col in X_out.columns:
        if X_out[col].nunique() > 10:
            lower = X_out[col].quantile(q_low)
            upper = X_out[col].quantile(q_high)
            X_out[col] = X_out[col].clip(lower, upper)
    return X_out


# ============================================================
# 3. Missing value imputation (Excel-driven)
# ============================================================

def impute_missing_from_config(
    X: pd.DataFrame,
    config_path: str,
    sheet_name: str = "missing_strategy"
) -> pd.DataFrame:
    """
    Imputes missing values using an Excel configuration file.

    Expected Excel columns:
    - feature
    - strategy  (median / constant)
    - fill_value (optional, used if strategy == constant)
    """
    config = pd.read_excel(config_path, sheet_name=sheet_name)
    X_imp = X.copy()

    for _, row in config.iterrows():
        col = row["feature"]
        if col not in X_imp.columns:
            continue

        if row["strategy"] == "median":
            X_imp[col] = X_imp[col].fillna(X_imp[col].median())

        elif row["strategy"] == "constant":
            X_imp[col] = X_imp[col].fillna(row["fill_value"])

    return X_imp


# ============================================================
# 4. Feature transformation (skewness-based)
# ============================================================

def transform_skewed_features(
    X: pd.DataFrame,
    skew_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Applies Yeo-Johnson transformation to highly skewed features.
    """
    X_trans = X.copy()
    pt = PowerTransformer(method="yeo-johnson")

    skewed_cols = [
        col for col in X_trans.columns
        if abs(skew(X_trans[col].dropna())) > skew_threshold
    ]

    if skewed_cols:
        X_trans[skewed_cols] = pt.fit_transform(X_trans[skewed_cols])

    return X_trans


# ============================================================
# 5. Missing value filter (feature-level)
# ============================================================

def drop_high_missing_features(
    X: pd.DataFrame,
    missing_threshold: float = 0.4
) -> pd.DataFrame:
    """
    Drops features with missing percentage above threshold.
    """
    missing_pct = X.isnull().mean()
    keep_cols = missing_pct[missing_pct <= missing_threshold].index
    return X[keep_cols]


# ============================================================
# 6. GrootCV (SHAP-based) Feature Selection
# ============================================================

def grootcv_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    importance_threshold: float = 0.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Applies GrootCV for regression using SHAP-based importance.
    """
    estimator = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=random_state
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    selector = GrootCV(
        estimator=estimator,
        cv=cv,
        importance_threshold=importance_threshold,
        verbose=0
    )

    selector.fit(X, y)
    selected_features = selector.get_feature_names_out()

    return X[selected_features]


# ============================================================
# 7. Correlation filter
# ============================================================

def drop_correlated_features(
    X: pd.DataFrame,
    corr_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Drops highly correlated features using absolute correlation.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > corr_threshold)
    ]

    return X.drop(columns=to_drop)


# ============================================================
# 8. Orchestrator: Full training data preparation
# ============================================================

def prepare_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    missing_config_path: str,
    id_cols: list,
    missing_threshold: float = 0.4,
    corr_threshold: float = 0.8
):
    """
    End-to-end feature engineering & selection pipeline
    with identifier preservation.
    """

    # ---- Step 0: Separate identifiers ----
    X_id, X_proc = split_identifier_columns(X, id_cols)

    # ---- Step 1: Drop categorical (string) columns ----
    X_proc = drop_string_columns(X_proc)

    # ---- Step 2: Outlier treatment ----
    X_proc = cap_outliers(X_proc)

    # ---- Step 3: Missing value imputation ----
    X_proc = impute_missing_from_config(X_proc, missing_config_path)

    # ---- Step 4: Feature transformation ----
    X_proc = transform_skewed_features(X_proc)

    # ---- Step 5: Missing value filter ----
    X_proc = drop_high_missing_features(X_proc, missing_threshold)

    # ---- Step 6: GrootCV #1 ----
    X_proc = grootcv_feature_selection(X_proc, y)

    # ---- Step 7: Correlation filter ----
    X_proc = drop_correlated_features(X_proc, corr_threshold)

    # ---- Step 8: GrootCV #2 ----
    X_proc = grootcv_feature_selection(X_proc, y)

    # ---- Step 9: Final assembly ----
    X_final = pd.concat([X_id.loc[X_proc.index], X_proc], axis=1)
    y_final = y.loc[X_proc.index]

    return X_final, y_final


id_cols = ["n_cust", "month_end"]

X_train_ready, y_train_ready = prepare_training_data(
    X=X_original,
    y=y_reg,
    missing_config_path="missing_value_strategy.xlsx",
    id_cols=id_cols
)
