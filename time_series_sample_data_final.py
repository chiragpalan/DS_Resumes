# %%
import numpy as np
import pandas as pd

# Create a date range for 60 months
date_range = pd.date_range(start="2020-01-01", periods=60, freq='M')
yearmonth = date_range.strftime('%Y%m')

np.random.seed(42)

# Generate synthetic features
trend = np.linspace(50, 100, 60)  # a linear trend for y
seasonal = 10 * np.sin(2 * np.pi * np.arange(60) / 4)  # 4-month seasonality
noise = np.random.normal(scale=5, size=60)  # noise
y = trend + seasonal + noise

interest_rate = np.random.normal(loc=0.04, scale=0.005, size=60)
stock_index = np.linspace(3000, 3500, 60) + np.random.normal(scale=50, size=60)
casa_balances = np.random.normal(loc=200, scale=20, size=60)
td_balances = np.random.normal(loc=150, scale=15, size=60)
wealth_balances = np.random.normal(loc=100, scale=10, size=60)
central_bank_rate = np.random.normal(loc=0.06, scale=0.002, size=60)

# Assemble DataFrame
df = pd.DataFrame({
    'yearmonth': yearmonth,
    'y_balance': y,
    'interest_rate': interest_rate,
    'stock_index': stock_index,
    'casa_balances': casa_balances,
    'td_balances': td_balances,
    'wealth_balances': wealth_balances,
    'central_bank_rate': central_bank_rate
})


# Save to CSV
df.to_csv('sample_TS_data.csv', index=False)

# %% [markdown]
# # 0. Imports & Data Load
# 

# %%

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import shap
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# assume df is your DataFrame from before, with columns:
# ['yearmonth','y_balance','interest_rate', 'stock_index',
#  'casa_balances','td_balances','wealth_balances','central_bank_rate']
df = pd.read_csv('sample_TS_data.csv')

df['date'] = pd.to_datetime(df['yearmonth'], format='%Y%m')
# df = df.sort_values('date').reset_index(drop=True)


# %% [markdown]
# # 1. Feature Engineering

# %%
# a) Lag features (lags 1–4)
for lag in range(1, 5):
    df[f'y_lag{lag}'] = df['y_balance'].shift(lag)

# b) Rolling window stats (3-month)
df['y_roll_mean_3'] = df['y_balance'].rolling(window=3).mean().shift(1)
df['y_roll_std_3']  = df['y_balance'].rolling(window=3).std().shift(1)

# c) Seasonal encoding (sin/cos for month)
df['month'] = df['date'].dt.month
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

# drop initial nulls from shifting
df = df.dropna().reset_index(drop=True)
# df.set_index('yearmonth', inplace=True)

# %% [markdown]
# # 

# %% [markdown]
# ## 1A Stationarity check

# %%


def make_stationary_df(y: pd.Series, X: pd.DataFrame, alpha: float = 0.05, max_diff: int = 5):
    """
    Make y and X stationary by differencing. Return single aligned DataFrame.

    Parameters:
        y        : Target time series (pd.Series)
        X        : Exogenous variables (pd.DataFrame)
        alpha    : Significance level for ADF test
        max_diff : Maximum differencing steps

    Returns:
        df_stationary : Combined DataFrame of differenced y and X
        d             : Order of differencing used
    """
    d = 0
    y_diff = y.copy()

    # Step 1: Difference y until stationary
    while d < max_diff:
        pval = adfuller(y_diff.dropna())[1]
        print(f"ADF p-value after {d} differencing(s): {pval:.4f}")
        if pval < alpha:
            print(f"✅ y is stationary at d = {d}")
            break
        y_diff = y_diff.diff()
        d += 1

    if d == max_diff:
        print("⚠️ Warning: y did not become stationary within max_diff limit.")

    # Step 2: Apply same differencing to X
    X_diff = X.copy()
    for _ in range(d):
        X_diff = X_diff.diff()

    # Step 3: Combine and drop missing values
    df_combined = pd.concat([y_diff.rename(y.name), X_diff], axis=1).dropna()

    return df_combined, d


# %%


# %%
# Assuming df is your full dataset and SELECTED are your X columns
y_raw = df['y_balance']
X_raw = df[[col for col in df.columns if col != 'y_balance' and col != 'yearmonth' and col != 'date']]

# Call the function
# df_stationary  = make_stationary_df(y_raw, X_raw)

# Now y_stationary and X_stationary are ready to use in SARIMAX or regression
# print(f"Final differencing order used: d = {d}")


# %%


# %% [markdown]
# # 2. Train/Test Split

# %%
# index-based split: first 80% train, last 20% test (here ~48/12)
train_size = int(len(df) * 0.9)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Define target & features
TARGET = 'y_balance'
FEATURES = [c for c in df.columns
            if c not in ['yearmonth','date','y_balance','month']]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]


# #Stationary data
# X_train, y_train = df_stationary[1].iloc[:train_size], df_stationary[2].iloc[:train_size]
# X_test, y_test = df_stationary[1].iloc[train_size:], df_stationary[].iloc[train_size:]



# %% [markdown]
# # 3. Scaling

# %%
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), index=X_train.index, columns=FEATURES
)
X_test_scaled  = pd.DataFrame(
    scaler.transform(X_test), index=X_test.index, columns=FEATURES
)

# %% [markdown]
# # 4. Feature Selection

# %%
# a) LASSO with time-series CV
tscv = TimeSeriesSplit(n_splits=5)
lasso = LassoCV(cv=tscv, random_state=42)
lasso.fit(X_train_scaled, y_train)

# coefficient magnitudes
coef_df = pd.Series(np.abs(lasso.coef_), index=FEATURES)
top_lasso = coef_df.sort_values(ascending=False).head(4).index.tolist()

# b) Random Forest importances
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
imp_df = pd.Series(rf.feature_importances_, index=FEATURES)
top_rf = imp_df.sort_values(ascending=False).head(4).index.tolist()

print("Top 4 LASSO:", top_lasso)
print("Top 4 RF   :", top_rf)

# choose intersection or union; e.g.:
SELECTED = list(set(top_lasso).intersection(top_rf))
if len(SELECTED) < 3:
    SELECTED = list(set(top_lasso).union(top_rf))[:4]
print("Final selected features:", SELECTED)


# %% [markdown]
# # 5A. SARIMAX with Exogenous X’s

# %% [markdown]
# ## Auto ARIMA

# %%
import pmdarima as pm
import matplotlib.pyplot as plt
auto_arima_model = pm.auto_arima(
    y_train,
    exogenous        = X_train_scaled[SELECTED],
    seasonal         = True,
    m                = 4,           # 4-month seasonality
    stepwise         = True,
    suppress_warnings= True,
    error_action     = 'ignore',
    trace            = False
)
print(auto_arima_model.summary())

n_test = len(test)
sarimax_pred_test = auto_arima_model.predict(
    n_periods = n_test,
    exogenous = X_test_scaled[SELECTED]
)

plt.figure(figsize=(10,4))
plt.plot(test['date'], y_test,            label='Actual')
plt.plot(test['date'], sarimax_pred_test, label='SARIMAX Forecast')
plt.title('SARIMAX: Actual vs Forecast (Test)')
plt.xlabel('Date'); plt.ylabel('y_balance')
plt.legend()
plt.show()

# Test metrics
print("SARIMAX MAE:",  mean_absolute_error(y_test, sarimax_pred_test))
print("SARIMAX RMSE:", np.sqrt(mean_squared_error(y_test, sarimax_pred_test)))




# %% [markdown]
# ## 5.B Extracting the model equation
# #### Fitting SARIMAX model to get X variable coefficient

# %%


# %%
import statsmodels.api as sm

# 1. Fit directly
model = sm.tsa.SARIMAX(
    endog            = y_train,
    exog             = X_train_scaled[SELECTED],  # your real columns
    order            = auto_arima_model.order,
    seasonal_order   = auto_arima_model.seasonal_order,
    enforce_stationarity  = False,
    enforce_invertibility = False
)
res = model.fit(disp=False)

# 2. Print the summary
print(res.summary())

# 3. Now res.params is a pandas Series indexed by names:
print("\nCoefficients:")
print(res.params)    # you’ll see entries named exactly 'interest_rate', 'stock_index', etc.


# 2. Forecast on test
# start = test.index[0]
# end   = test.index[-1]

n_train = len(y_train)
n_test = len(y_test)

sarimax_pred_test = res.predict(
    start = n_train,
    end   = n_train + n_test - 1,
    exog    = X_test_scaled[SELECTED],
    dynamic = False,     # use true y’s for lagged terms
    typ     = 'levels'   # return forecasts on original scale
)

# 3. Plot Actual vs Forecast
plt.figure(figsize=(10,4))
plt.plot(test['date'], test['y_balance'],         label='Actual', marker='o')
plt.plot(test['date'], sarimax_pred_test,         label='SARIMAX Forecast', linestyle='--')
plt.title('SARIMAX: Actual vs Forecast (Test)')
plt.xlabel('Date')
plt.ylabel('y_balance')
plt.legend()
plt.tight_layout()
plt.show()
# 4. Optional: print test metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae  = mean_absolute_error(test['y_balance'], sarimax_pred_test)
rmse = mean_squared_error(test['y_balance'], sarimax_pred_test, squared=False)
print(f"SARIMAX Test MAE:  {mae:.4f}")
print(f"SARIMAX Test RMSE: {rmse:.4f}")

# %%
# STEP 1: Extract model orders and parameters
p, d, q = res.model.order
params = res.params  # this is a pandas Series with named coefficients

# STEP 2: Start building the equation terms
terms = []

# Intercept
mu = params.get('const', 0.0)
terms.append(f"{mu:.4f}")

# AR terms: yₜ₋1, yₜ₋2, ..., yₜ₋p
for i in range(1, p + 1):
    phi = params.get(f'ar.L{i}', 0.0)
    terms.append(f"({phi:.4f})·yₜ₋{i}")

# MA terms: εₜ₋1, εₜ₋2, ..., εₜ₋q
for j in range(1, q + 1):
    theta = params.get(f'ma.L{j}', 0.0)
    terms.append(f"({theta:.4f})·εₜ₋{j}")

# Exogenous (X) variables
for var in SELECTED:
    beta = params.get(var, None)
    if beta is not None:
        terms.append(f"({beta:.4f})·{var}ₜ")

# Final equation
equation = "yₜ = " + " + ".join(terms) + " + εₜ"
print("\n==== SARIMAX Prediction Equation ====")
print(equation)


# %%
df.head()

# %% [markdown]
# 1. No manula differencing is required if using SARIMAX so try without it.
# 1.1 If doing differencing need to do on both X and y variables
# 2. When using SARIMAX algorithm, no need to create a lag feature exclusively as it will be taken care by AR terms
# 3. 
