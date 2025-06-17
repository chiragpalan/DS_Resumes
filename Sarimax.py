import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------------------------------------------------------------------
# STEP 1: Prepare Data
# ---------------------------------------------------------------------------------------

# Assume `df` is your time-ordered DataFrame that includes:
# - Target variable: 'y'
# - 20+ X variables (lagged or current)
# - Already sorted by time (important for time series!)

# Separate features (X) and target (y)
X = df.drop(columns=['y'])     # All columns except target
y = df['y']                    # Target column

# ---------------------------------------------------------------------------------------
# STEP 2: Feature Selection using RFE (Recursive Feature Elimination)
# ---------------------------------------------------------------------------------------

# Scale features for regularized model stability (important for Ridge/Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # Transforms X to have mean=0 and std=1

# Create a Ridge regression model as the estimator (handles multicollinearity well)
ridge = Ridge(alpha=1.0)

# Set up RFE to select the top 4 features based on Ridge's coefficients
rfe = RFE(estimator=ridge, n_features_to_select=4)
rfe.fit(X_scaled, y)   # Fit RFE on scaled data

# Get names of selected features (columns where support_ is True)
selected_features = X.columns[rfe.support_]
print("✅ Selected Features for Time Series Model:", selected_features.tolist())

# ---------------------------------------------------------------------------------------
# STEP 3: Build SARIMAX Model (ARIMA + selected exogenous features)
# ---------------------------------------------------------------------------------------

# Extract only the selected features for SARIMAX model
exog_selected = df[selected_features]

# Define SARIMAX order (p, d, q) — start simple: (1,1,1)
# - p: autoregressive terms
# - d: differencing (to remove trend)
# - q: moving average terms
# If needed, tune these later using AIC/BIC or ACF/PACF plots
model = SARIMAX(
    endog=y,              # Target variable (time series)
    exog=exog_selected,   # Selected X variables as exogenous regressors
    order=(1, 1, 1),      # ARIMA order: change this if needed
    enforce_stationarity=True,
    enforce_invertibility=True
)

# Fit the SARIMAX model
results = model.fit(disp=False)

# Print model summary including coefficients, AIC, etc.
print("\n📊 SARIMAX Model Summary:\n")
print(results.summary())
