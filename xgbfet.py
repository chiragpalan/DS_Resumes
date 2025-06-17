import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor  # Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# -------------------------------------------
# STEP 1: Define a function to create lagged features
# -------------------------------------------
def create_lagged_features(df, lags=3):
    """
    Create lagged features for each column in the DataFrame.
    """
    df_lagged = df.copy()
    for col in df.columns:
        for lag in range(1, lags + 1):
            df_lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
    df_lagged.dropna(inplace=True)
    return df_lagged

# -------------------------------------------
# STEP 2: Prepare your dataset
# -------------------------------------------
# Example: Assuming df is already loaded with 'y' and X1...X50
# df = pd.read_csv("your_timeseries_data.csv")  # Load your data

# Split target and predictors
y = df['y']
X = df.drop(columns='y')

# Create lagged features for predictors (e.g., 3 lags)
X_lagged = create_lagged_features(X, lags=3)

# Align y to match lagged features
y_lagged = y[X_lagged.index]

print("Shape of lagged feature matrix:", X_lagged.shape)

# -------------------------------------------
# STEP 3: Fit XGBoost model to all features
# -------------------------------------------
model = XGBRegressor(random_state=42, n_estimators=100)
# model = RandomForestRegressor(random_state=42, n_estimators=100)  # Optional alternative

model.fit(X_lagged, y_lagged)

# -------------------------------------------
# STEP 4: Plot top feature importances
# -------------------------------------------
importances = model.feature_importances_
feature_names = X_lagged.columns
indices = np.argsort(importances)[::-1]

top_n = 15
plt.figure(figsize=(10, 6))
plt.title("Top Feature Importances from XGBoost")
plt.barh(range(top_n), importances[indices][:top_n], align='center')
plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -------------------------------------------
# STEP 5: Select top features and prepare new dataset
# -------------------------------------------
top_features = [feature_names[i] for i in indices[:10]]  # Top 10 features
X_selected = X_lagged[top_features]

# -------------------------------------------
# STEP 6: Optional - Evaluate performance using TimeSeriesSplit
# -------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []

for train_index, test_index in tscv.split(X_selected):
    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
    y_train, y_test = y_lagged.iloc[train_index], y_lagged.iloc[test_index]

    model = XGBRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold RMSE: {rmse:.4f}")

print(f"\nAverage RMSE across folds: {np.mean(rmse_scores):.4f}")
