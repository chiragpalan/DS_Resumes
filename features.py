"""
feature_selection_pipeline.py

End-to-end feature handling and feature selection pipeline
for regression models in banking use cases.

Key characteristics:
- Automatic inference of column types
- String columns treated as categorical
- Numeric columns treated as continuous
- Outlier treatment (1st–99th percentile)
- Excel-driven missing value treatment
- Missing value filtering
- SHAP-based GrootCV feature selection (two-stage)
- Correlation filtering (numeric only)
- Returns final processed X (model-ready)

Designed for:
- Tree-based models (LightGBM)
- SHAP explainability
- Model validation and regulatory review
"""

import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from arfs.feature_selection import GrootCV


# ============================================================
# 1. Column Type Inference
# ============================================================

def infer_column_types(X: pd.DataFrame):
    """
    Infer categorical and numerical columns automatically.

    Rules:
    - Categorical: string / object dtype only
    - Numerical: all remaining columns

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix

    Returns
    -------
    categorical_cols : list
        List of categorical (string) columns
    numerical_cols : list
        List of numerical columns
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.columns.difference(categorical_cols).tolist()
    return categorical_cols, numerical_cols


# ============================================================
# 2. Outlier Treatment (Numeric Only)
# ============================================================

def cap_outliers_quantile_numeric(
    X: pd.DataFrame,
    numerical_cols: list,
    lower_q: float = 0.01,
    upper_q: float = 0.99
):
    """
    Cap numerical variables at specified quantiles.

    Purpose:
    - Reduce impact of extreme values
    - Stabilize tree splits and SHAP values

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix
    numerical_cols : list
        List of numerical columns
    lower_q : float
        Lower quantile (default 0.01)
    upper_q : float
        Upper quantile (default 0.99)

    Returns
    -------
    X_capped : pd.DataFrame
        DataFrame with capped numerical features
    """
    X_capped = X.copy()

    for col in numerical_cols:
        if pd.api.types.is_numeric_dtype(X_capped[col]):
            lower = X_capped[col].quantile(lower_q)
            upper = X_capped[col].quantile(upper_q)
            X_capped[col] = X_capped[col].clip(lower, upper)

    return X_capped


# ============================================================
# 3. Missing Value Treatment (Excel Driven)
# ============================================================

def apply_missing_treatment(
    X: pd.DataFrame,
    config_path: str
):
    """
    Apply missing value treatment based on external Excel configuration.

    Supported strategies:
    - mean
    - median
    - mode
    - constant

    Excel format:
    ----------------
    feature_name | strategy | fill_value

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix
    config_path : str
        Path to missing value treatment Excel file

    Returns
    -------
    X_filled : pd.DataFrame
        DataFrame with missing values treated
    """
    config = pd.read_excel(config_path)
    X_filled = X.copy()

    for _, row in config.iterrows():
        col = row["feature_name"]
        strategy = row["strategy"]
        fill_value = row.get("fill_value", None)

        if col not in X_filled.columns:
            continue

        if strategy == "mean":
            X_filled[col] = X_filled[col].fillna(X_filled[col].mean())
        elif strategy == "median":
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())
        elif strategy == "mode":
            X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0])
        elif strategy == "constant":
            X_filled[col] = X_filled[col].fillna(fill_value)

    return X_filled


# ============================================================
# 4. Missing Value Filter (Feature-Level)
# ============================================================

def missing_value_filter(
    X: pd.DataFrame,
    threshold: float = 0.3
):
    """
    Remove features with excessive missing values.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix
    threshold : float
        Maximum allowed missing fraction

    Returns
    -------
    X_filtered : pd.DataFrame
        DataFrame with high-missing columns removed
    """
    missing_ratio = X.isnull().mean()
    keep_cols = missing_ratio[missing_ratio <= threshold].index.tolist()
    return X[keep_cols]


# ============================================================
# 5. Prepare Categorical Dtypes for LightGBM
# ============================================================

def prepare_string_categoricals(
    X: pd.DataFrame,
    categorical_cols: list
):
    """
    Convert string categorical columns to 'category' dtype
    for native LightGBM handling.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix
    categorical_cols : list
        List of categorical columns

    Returns
    -------
    X_prepared : pd.DataFrame
        DataFrame with categorical dtypes set
    """
    X_prepared = X.copy()
    for col in categorical_cols:
        X_prepared[col] = X_prepared[col].astype("category")
    return X_prepared


# ============================================================
# 6. GrootCV Feature Selection (SHAP-Based)
# ============================================================

def grootcv_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list,
    cv: int = 5,
    threshold: float = 0.01
):
    """
    Perform SHAP-based feature selection using GrootCV.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix
    y : pd.Series
        Target variable
    categorical_cols : list
        List of categorical columns
    cv : int
        Number of cross-validation folds
    threshold : float
        SHAP importance threshold

    Returns
    -------
    selected_features : list
        List of selected feature names
    """
    model = LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    selector = GrootCV(
        estimator=model,
        cv=cv,
        threshold=threshold,
        importance="shap",
        verbose=True
    )

    selector.fit(X, y, categorical_feature=categorical_cols)
    selected_features = X.columns[selector.support_].tolist()

    return selected_features


# ============================================================
# 7. Correlation Filter (Numeric Only)
# ============================================================

def correlation_filter_numeric(
    X: pd.DataFrame,
    numerical_cols: list,
    threshold: float = 0.75
):
    """
    Remove highly correlated numerical features.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix
    numerical_cols : list
        List of numerical columns
    threshold : float
        Correlation threshold

    Returns
    -------
    drop_cols : list
        List of columns to drop due to high correlation
    """
    if len(numerical_cols) == 0:
        return []

    corr = X[numerical_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_cols = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    return drop_cols


# ============================================================
# 8. Full Feature Selection Pipeline (Returns Final X)
# ============================================================

def feature_selection_pipeline_return_X(
    X: pd.DataFrame,
    y: pd.Series,
    missing_config_path: str,
    missing_threshold: float = 0.3,
    corr_threshold: float = 0.75,
    groot_threshold: float = 0.01
):
    """
    End-to-end feature handling and feature selection pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Raw input features
    y : pd.Series
        Target variable
    missing_config_path : str
        Path to missing value treatment Excel file
    missing_threshold : float
        Feature-level missing value threshold
    corr_threshold : float
        Correlation threshold for numeric features
    groot_threshold : float
        SHAP importance threshold for GrootCV

    Returns
    -------
    X_final : pd.DataFrame
        Final processed and selected feature matrix
    """

    X_proc = X.copy()

    # Infer column types
    categorical_cols, numerical_cols = infer_column_types(X_proc)

    # Outlier treatment
    X_proc = cap_outliers_quantile_numeric(X_proc, numerical_cols)

    # Missing value treatment
    X_proc = apply_missing_treatment(X_proc, missing_config_path)

    # Missing value filter
    X_proc = missing_value_filter(X_proc, threshold=missing_threshold)

    # Re-infer types
    categorical_cols, numerical_cols = infer_column_types(X_proc)

    # Prepare categorical dtypes
    X_proc = prepare_string_categoricals(X_proc, categorical_cols)

    # First GrootCV
    selected_1 = grootcv_feature_selection(
        X_proc, y, categorical_cols, threshold=groot_threshold
    )
    X_proc = X_proc[selected_1]

    # Re-infer types
    categorical_cols, numerical_cols = infer_column_types(X_proc)

    # Correlation filter
    drop_corr = correlation_filter_numeric(
        X_proc, numerical_cols, threshold=corr_threshold
    )
    X_proc = X_proc.drop(columns=drop_corr)

    # Re-infer types
    categorical_cols, numerical_cols = infer_column_types(X_proc)

    # Final GrootCV
    selected_final = grootcv_feature_selection(
        X_proc, y, categorical_cols, threshold=groot_threshold
    )
    X_final = X_proc[selected_final]

    return X_final
