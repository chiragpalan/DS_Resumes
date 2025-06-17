
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# ---------------------------------------------
# STEP 0: Assume df_sarimax is already loaded,
# indexed by DateTime, with one column 'y'
# and other columns as potential exogenous X's.
# ---------------------------------------------
# df_sarimax = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# ---------------------------------------------
# STEP 1: Check stationarity of y and difference if needed
# ---------------------------------------------
def make_stationary(series, signif=0.05):
    """
    Perform ADF test; if p-value > signif, difference once.
    Returns stationary series and a flag whether diff was applied.
    """
    adf_stat, pval, _, _, crit_vals, _ = adfuller(series.dropna())
    print(f"ADF p-value for {series.name}: {pval:.4f}")
    if pval > signif:
        print(f" -> Differencing {series.name}")
        return series.diff().dropna(), True
    else:
        return series, False

y_orig = df_sarimax['y']
y_stationary, y_diffed = make_stationary(y_orig)

# ---------------------------------------------
# STEP 2: Align df_sarimax to y_stationary index
# ---------------------------------------------
df = df_sarimax.copy()
if y_diffed:
    df['y'] = y_stationary
else:
    df = df.loc[y_stationary.index]

# ---------------------------------------------
# STEP 3: (Optional) Check & difference exog if needed
# ---------------------------------------------
exog_cols = [c for c in df.columns if c != 'y']
df_exog = df[exog_cols].copy()
diffed_exogs = []
for col in exog_cols:
    series, did = make_stationary(df_exog[col])
    if did:
        df_exog[col] = series
        diffed_exogs.append(col)
print("Differenced exogs:", diffed_exogs)

# Re-combine y + exog into one DF
df_clean = pd.concat([y_stationary, df_exog], axis=1).dropna()

# ---------------------------------------------
# STEP 4: Train–test split (80% train / 20% test)
# ---------------------------------------------
split_idx = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_idx]
test_df  = df_clean.iloc[split_idx:]

y_train = train_df['y']
y_test  = test_df['y']
X_train = train_df.drop(columns='y')
X_test  = test_df.drop(columns='y')

# ---------------------------------------------
# STEP 5: Scale exogenous features
# ---------------------------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    index=X_train.index, columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    index=X_test.index, columns=X_test.columns
)

# ---------------------------------------------
# STEP 6: Feature Selection via XGBoost
# ---------------------------------------------
xgb = XGBRegressor(random_state=42, n_estimators=100)
xgb.fit(X_train_scaled, y_train)

importances = xgb.feature_importances_
feat_names  = X_train_scaled.columns
indices     = np.argsort(importances)[::-1]

# Pick top 10 features
top_n = 10
top_features = feat_names[indices[:top_n]].tolist()
print("Top features selected:", top_features)

X_train_sel = X_train_scaled[top_features]
X_test_sel  = X_test_scaled[top_features]

# ---------------------------------------------
# STEP 7: Fit auto_arima with quarterly seasonality
# ---------------------------------------------
arima_model = auto_arima(
    y_train,
    exogenous=X_train_sel,
    seasonal=True,     # enable SARIMAX
    m=4,               # quarterly data: 4 periods per year
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print(arima_model.summary())

# ---------------------------------------------
# STEP 8: Forecast on test set & evaluate
# ---------------------------------------------
n_test = len(y_test)
forecast, conf_int = arima_model.predict(
    n_periods=n_test,
    exogenous=X_test_sel,
    return_conf_int=True
)

rmse = np.sqrt(mean_squared_error(y_test, forecast))
print(f"Test RMSE: {rmse:.4f}")

# ---------------------------------------------
# STEP 9: Plot train, test, forecast
# ---------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_train.index, y_train, label='Train y')
plt.plot(y_test.index, y_test,   label='Test y',   color='black')
plt.plot(y_test.index, forecast, label='Forecast', color='red')
plt.fill_between(y_test.index,
                 conf_int[:, 0], conf_int[:, 1],
                 color='pink', alpha=0.3)
plt.title("Auto ARIMA Forecast with XGBoost-Selected Exog (Quarterly)")
plt.legend()
plt.tight_layout()
plt.show()
