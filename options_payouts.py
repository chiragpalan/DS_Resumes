import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

st.title("Options Payoff Calculator")

--- User Inputs ---

option_type = st.selectbox("Select Option Type:", ["Call", "Put"]) position_type = st.selectbox("Select Position Type:", ["Buy", "Sell"])

spot_price = st.number_input("Current Price of Underlying (Spot)", min_value=0.0, value=25500.0) strike_price = st.number_input("Strike Price", min_value=0.0, value=25500.0) premium = st.number_input("Premium per Option", min_value=0.0, value=200.0) lot_size = st.number_input("Lot Size", min_value=1, value=75)

--- Price range around spot ---

price_range = np.arange(spot_price - 500, spot_price + 501, 50)

--- Payoff Calculation ---

payoffs = [] for price in price_range: if option_type == "Call": intrinsic_value = max(price - strike_price, 0) else:  # Put intrinsic_value = max(strike_price - price, 0)

if position_type == "Buy":
    profit_per_unit = intrinsic_value - premium
else:  # Sell
    profit_per_unit = premium - intrinsic_value

total_profit = profit_per_unit * lot_size
payoffs.append(total_profit)

--- Display Table ---

df = pd.DataFrame({ "Underlying Price at Expiry": price_range, "Payout (Total for Lot)": payoffs }) st.subheader("Payoff Table") st.dataframe(df.style.format({"Underlying Price at Expiry": "{:.2f}", "Payout (Total for Lot)": "{:.2f}"}))

--- Plot Payoff Curve ---

st.subheader("Payoff Diagram") fig, ax = plt.subplots() ax.plot(price_range, payoffs, label="Payoff") ax.axhline(0, color='black', linestyle='--') ax.set_xlabel("Underlying Price at Expiry") ax.set_ylabel("Profit / Loss") ax.set_title(f"{position_type} {option_type} Payoff") ax.legend() st.pyplot(fig)

