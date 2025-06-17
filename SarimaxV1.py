
import pandas as pd import numpy as np import matplotlib.pyplot as plt from sklearn.linear_model import Ridge from sklearn.feature_selection import RFE from sklearn.preprocessing import StandardScaler from sklearn.metrics import mean_absolute_percentage_error from statsmodels.tsa.statespace.sarimax import SARIMAX from statsmodels.tsa.stattools import adfuller

---------------------------------------------------------------------------------------

STEP 0: Dummy Data Preparation for Demonstration (Replace this with your actual dataset)

---------------------------------------------------------------------------------------

np.random.seed(0) dates = pd.date_range(start="2015-01-01", periods=60, freq="MS") df = pd.DataFrame(index=dates) df["y"] = np.cumsum(np.random.randn(60)) + 50  # target with some trend for i in range(1, 21): df[f"x{i}"] = np.random.randn(60)

---------------------------------------------------------------------------------------

STEP 1: Check and Transform Stationarity for y and all X variables

---------------------------------------------------------------------------------------

def make_stationary(series, name): """Perform ADF test and apply differencing if non-stationary.""" result = adfuller(series) if result[1] > 0.05: print(f"{name} is not stationary (p={result[1]:.4f}). Differencing applied.") differenced = series.diff().dropna() second_result = adfuller(differenced) if second_result[1] <= 0.05: print(f"{name} became stationary after differencing (p={second_result[1]:.4f}).") else: print(f"{name} is still not stationary after differencing (p={second_result[1]:.4f}).") return differenced else: print(f"{name} is stationary (p={result[1]:.4f}).") return series

Make y stationary

y = make_stationary(df['y'], 'y')

Make all X variables stationary

X_stationary = pd.DataFrame(index=y.index) for col in df.columns.difference(['y']): X_stationary[col] = make_stationary(df[col], col)

Drop rows with NaN after differencing

X_stationary = X_stationary.loc[y.index]

---------------------------------------------------------------------------------------

STEP 2: Train-test split (Time series split)

---------------------------------------------------------------------------------------

train_size = int(len(y) * 0.8) X_train, X_test = X_stationary.iloc[:train_size], X_stationary.iloc[train_size:] y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

---------------------------------------------------------------------------------------

STEP 3: Feature Selection using RFE (Recursive Feature Elimination)

---------------------------------------------------------------------------------------

scaler = StandardScaler() X_train_scaled = scaler.fit_transform(X_train)

ridge = Ridge(alpha=1.0) rfe = RFE(estimator=ridge, n_features_to_select=4) rfe.fit(X_train_scaled, y_train)

selected_features = X_train.columns[rfe.support_] dropped_features = X_train.columns[~rfe.support_] print("\n✅ Selected features:", selected_features.tolist()) print("❌ Dropped features:", dropped_features.tolist())

---------------------------------------------------------------------------------------

STEP 4: Fit SARIMAX Model on Train Set

---------------------------------------------------------------------------------------

exog_train = X_train[selected_features] exog_test = X_test[selected_features]

model = SARIMAX(endog=y_train, exog=exog_train, order=(1, 1, 1), enforce_stationarity=True, enforce_invertibility=True) results = model.fit(disp=False)

---------------------------------------------------------------------------------------

STEP 5: Forecasting and Evaluation

---------------------------------------------------------------------------------------

forecast = results.forecast(steps=len(y_test), exog=exog_test) mape = mean_absolute_percentage_error(y_test, forecast) print(f"\n📉 Mean Absolute Percentage Error (MAPE): {mape:.4f}")

---------------------------------------------------------------------------------------

STEP 6: Plot Forecast vs True Values

---------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6)) plt.plot(y_test.index, y_test, label='True') plt.plot(y_test.index, forecast, label='Forecast', linestyle='--') plt.title("Forecast vs Actual") plt.xlabel("Time") plt.ylabel("y") plt.legend() plt.grid(True) plt.tight_layout() plt.show()

---------------------------------------------------------------------------------------

STEP 7: Documentation/Explanation

---------------------------------------------------------------------------------------

""" Model Summary Explanation (results.summary()):

Dep. Variable: The dependent (target) variable name

No. Observations: Number of observations in training data

Model: Type of model (SARIMAX)

Log Likelihood: Model log-likelihood value (higher = better)

AIC / BIC: Model selection criteria (lower = better)

coef: Coefficients for ARIMA terms and exogenous variables

std err: Standard error of the coefficient

z: z-statistic for hypothesis testing (coef / std err)

P>|z|: p-value for testing the null hypothesis (coef = 0)


Dropped Variables in RFE:

RFE drops features that have the least impact on reducing model error, as estimated by the base model (Ridge). This is a greedy backward elimination process based on coefficients. """


