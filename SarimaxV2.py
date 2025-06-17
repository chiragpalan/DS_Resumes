import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_percentage_error
from pmdarima import auto_arima

# -----------------------------
# 1. Load your dataset here
# -----------------------------
# Example:
# df = pd.read_csv("your_data.csv", parse_dates=["date"], index_col="date")

# Make sure your index is datetime
df.index = pd.to_datetime(df.index)

# Specify your target and features
target_column = "y"
exogenous_columns = [col for col in df.columns if col != target_column]

y = df[target_column]
X = df[exogenous_columns]

# -----------------------------
# 2. Stationarity check + differencing
# -----------------------------
def adf_test(series, name=""):
    pval = adfuller(series.dropna())[1]
    print(f"ADF for {name}: p = {pval:.4f}")
    return pval <= 0.05

def make_stationary(df):
    stm = pd.DataFrame(index=df.index)
    diff_levels = {}
    for col in df:
        if adf_test(df[col], col):
            stm[col] = df[col]
            diff_levels[col] = 0
        else:
            d1 = df[col].diff().dropna()
            if adf_test(d1, f"{col} Δ1"):
                stm[col] = d1
                diff_levels[col] = 1
            else:
                d2 = d1.diff().dropna()
                stm[col] = d2
                diff_levels[col] = 2
    return stm.fillna(method="bfill"), diff_levels

X_stat, diff_info = make_stationary(X)

# -----------------------------
# 3. Train-test split (time-based)
# -----------------------------
train_size = int(len(df) * 0.8)
X_train, X_test = X_stat.iloc[:train_size], X_stat.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# -----------------------------
# 4. Feature Selection with LassoCV
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

lasso = LassoCV(cv=5, random_state=0).fit(X_scaled, y_train)
selector = SelectFromModel(lasso, prefit=True)
selected_features = X_train.columns[selector.get_support()].tolist()
print("\n✅ Selected Features:", selected_features)

# -----------------------------
# 5. Fit Auto-ARIMA model (quarterly seasonality)
# -----------------------------
model = auto_arima(
    y_train,
    exogenous=X_train[selected_features],
    seasonal=True,
    m=3,  # Quarterly seasonality
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)

print("\n✅ Chosen ARIMA order:", model.order)
print("✅ Chosen seasonal_order:", model.seasonal_order)

# -----------------------------
# 6. Forecast on test set
# -----------------------------
forecast = model.predict(n_periods=len(y_test), exogenous=X_test[selected_features])

# -----------------------------
# 7. Evaluate with MAPE
# -----------------------------
mape = mean_absolute_percentage_error(y_test, forecast)
print(f"\n✅ MAPE on test set: {mape:.4f}")

# -----------------------------
# 8. Plot forecast vs. actual
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual", linewidth=2)
plt.plot(y_test.index, forecast, label="Forecast", linestyle="--", linewidth=2)
plt.title("Forecast vs Actual (Auto ARIMA)")
plt.xlabel("Date")
plt.ylabel("Target Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
