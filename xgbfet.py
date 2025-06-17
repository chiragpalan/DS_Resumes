
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# ---------------------------------------------
# STEP 0: Load your DataFrame
#   - Must have a DateTime index
#   - Column "y" is your target
#   - All other columns are candidate exogenous features
# ---------------------------------------------
# df_sarimax = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# ---------------------------------------------
# STEP 1: Stationarity check and differencing
# ---------------------------------------------
def make_stationary(series, signif=0.05):
    """ADF test; difference once if non-stationary."""
    series = series.dropna()
    adf_stat, pval, *_ , crit_vals, _ = adfuller(series)
    print(f"ADF p-value for {series.name}: {pval:.4f}")
    if pval > signif:
        print(f"  → differencing {series.name}")
        return series.diff().dropna(), True
    else:
        return series, False

# 1a) Target
y_orig = df_sarimax['y']
y_stat, y_diffed = make_stationary(y_orig)

# 1b) Exogenous features
exog_cols = [c for c in df_sarimax.columns if c != 'y']
df_exog = df_sarimax[exog_cols].copy()
diffed_exogs = []
for col in exog_cols:
    ser, diffed = make_stationary(df_exog[col])
    if diffed:
        df_exog[col] = ser
        diffed_exogs.append(col)
print("Differenced exogs:", diffed_exogs)

# Align everything on common index
if y_diffed:
    df_core = pd.concat([y_stat, df_exog], axis=1).dropna()
else:
    df_core = pd.concat([y_orig, df_exog], axis=1).loc[y_stat.index].dropna()
df_core.rename(columns={y_orig.name: 'y'}, inplace=True)

# ---------------------------------------------
# STEP 2: Create lagged exogenous variables
# ---------------------------------------------
def create_lags(df, lags=3):
    """For each column in df, create lag1…lagN and drop rows with any NaN."""
    lagged = df.copy()
    for col in df.columns:
        for lag in range(1, lags+1):
            lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
    return lagged.dropna()

# Only create lags for exogenous columns
df_lags = create_lags(df_core.drop(columns='y'), lags=3)

# Align y to the lagged exog DataFrame
y_lagged = df_core['y'].loc[df_lags.index]

# ---------------------------------------------
# STEP 3: Train/Test Split (80/20 chronologically)
# ---------------------------------------------
split_at = int(len(df_lags) * 0.8)
X_train_raw = df_lags.iloc[:split_at]
X_test_raw  = df_lags.iloc[split_at:]
y_train      = y_lagged.iloc[:split_at]
y_test       = y_lagged.iloc[split_at:]

# ---------------------------------------------
# STEP 4: Scale exogenous features
# ---------------------------------------------
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train_raw),
    index=X_train_raw.index,
    columns=X_train_raw.columns
)
X_test  = pd.DataFrame(
    scaler.transform(X_test_raw),
    index=X_test_raw.index,
    columns=X_test_raw.columns
)

# ---------------------------------------------
# STEP 5: Feature Selection via XGBoost
# ---------------------------------------------
xgb = XGBRegressor(random_state=42, n_estimators=100)
xgb.fit(X_train, y_train)

importances = xgb.feature_importances_
names       = X_train.columns
idx_sorted  = np.argsort(importances)[::-1]

top_n = 10
top_feats = names[idx_sorted[:top_n]].tolist()
print("Top features:", top_feats)

X_train_sel = X_train[top_feats]
X_test_sel  = X_test[top_feats]

# ---------------------------------------------
# STEP 6: Fit auto_arima (SARIMAX) with quarterly seasonality
# ---------------------------------------------
model = auto_arima(
    y_train,
    exogenous=X_train_sel,
    seasonal=True,
    m=4,                 # quarterly
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print(model.summary())

# ---------------------------------------------
# STEP 7: Forecast & Evaluate
# ---------------------------------------------
n_periods = len(y_test)
forecast, conf_int = model.predict(
    n_periods=n_periods,
    exogenous=X_test_sel,
    return_conf_int=True
)

rmse = np.sqrt(mean_squared_error(y_test, forecast))
print(f"Test RMSE: {rmse:.4f}")

# ---------------------------------------------
# STEP 8: Plot Results
# ---------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_train.index, y_train, label='Train')
plt.plot(y_test.index,  y_test,  label='Test',   color='black')
plt.plot(y_test.index,  forecast, label='Forecast', color='red')
plt.fill_between(y_test.index,
                 conf_int[:,0], conf_int[:,1],
                 color='pink', alpha=0.3)
plt.title("SARIMAX Forecast with XGBoost-Selected Lagged Exog")
plt.legend()
plt.tight_layout()
plt.show()
