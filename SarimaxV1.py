import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# -----------------------------------------------
# STEP 0: Create Dummy Data (Replace with your own)
# -----------------------------------------------
np.random.seed(0)
dates = pd.date_range(start="2015-01-01", periods=60, freq="MS")
df = pd.DataFrame(index=dates)
df["y"] = np.cumsum(np.random.randn(60)) + 50
for i in range(1, 21):
    df[f"x{i}"] = np.random.randn(60)

# --------------------------------------------------------
# STEP 1: ADF Test + Differencing to Ensure Stationarity
# --------------------------------------------------------
def make_stationary(series, name):
    """Run ADF test, apply differencing if needed, and re-check."""
    result = adfuller(series)
    if result[1] > 0.05:
        print(f"{name} is NOT stationary (p={result[1]:.4f}) — applying differencing.")
        differenced = series.diff().dropna()
        second_result = adfuller(differenced)
        if second_result[1] <= 0.05:
            print(f"{name} IS stationary after differencing (p={second_result[1]:.4f})")
        else:
            print(f"{name} STILL NOT stationary after differencing (p={second_result[1]:.4f})")
        return differenced
    else:
        print(f"{name} is already stationary (p={result[1]:.4f})")
        return series

# Apply to target
y = make_stationary(df['y'], 'y')

# Apply to all X features
X_stationary = pd.DataFrame(index=y.index)
for col in df.columns.difference(['y']):
    X_stationary[col] = make_stationary(df[col], col)

X_stationary = X_stationary.loc[y.index]  # Align indices

# --------------------------
# STEP 2: Train-Test Split
# --------------------------
train_size = int(len(y) * 0.8)
X_train, X_test = X_stationary.iloc[:train_size], X_stationary.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# ------------------------------
# STEP 3: Feature Selection (RFE)
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ridge = Ridge()
rfe = RFE(estimator=ridge, n_features_to_select=4)
rfe.fit(X_train_scaled, y_train)

selected_features = X_train.columns[rfe.support_]
print("\n✅ Selected features:", selected_features.tolist())
print("❌ Dropped features:", X_train.columns[~rfe.support_].tolist())

# ------------------------------
# STEP 4: Fit SARIMAX Model
# ------------------------------
model = SARIMAX(y_train, exog=X_train[selected_features], order=(1, 1, 1),
                enforce_stationarity=True, enforce_invertibility=True)
results = model.fit(disp=False)

# ------------------------------
# STEP 5: Forecast + MAPE
# ------------------------------
forecast = results.forecast(steps=len(y_test), exog=X_test[selected_features])
mape = mean_absolute_percentage_error(y_test, forecast)
print(f"\n📉 MAPE: {mape:.4f}")

# ------------------------------
# STEP 6: Plot Forecast vs Actual
# ------------------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='True')
plt.plot(y_test.index, forecast, label='Forecast', linestyle='--')
plt.title("Forecast vs Actual")
plt.xlabel("Time")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 7: Documentation
# ------------------------------
"""
Model Summary Notes:
- Coefficients explain influence of each lag and exogenous variable.
- Low AIC/BIC = better model.
- RFE selects features by eliminating those that add least value to Ridge regression.
"""
