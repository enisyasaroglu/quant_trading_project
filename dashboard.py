# dashboard.py

# Final version for cloud deployment
import os
import time
import redis
import pandas as pd
import streamlit as st

# --- Page Configuration ---
# Sets the title and layout for your web application.
st.set_page_config(
    page_title="QMind Quant Dashboard",
    layout="wide",
)


# --- Data Connection ---
@st.cache_resource
def get_redis_connection():
    """
    Establishes a connection to Redis.
    It intelligently checks for a cloud environment variable (REDIS_URL) provided by Railway
    and falls back to localhost if it's not found, allowing the script to work both locally and in the cloud.
    """
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        print("Connecting to cloud Redis...")
        return redis.from_url(redis_url, decode_responses=True)
    else:
        print("Connecting to local Redis...")
        return redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )


# Establish the connection when the script starts.
redis_client = get_redis_connection()


# --- Helper Functions to Fetch Data from Redis ---
def get_portfolio_cash():
    """Fetches the current cash value from Redis."""
    return float(redis_client.get("qmind:portfolio:cash") or 0)


def get_positions():
    """Fetches all current stock positions and returns them as a DataFrame."""
    positions = redis_client.hgetall("qmind:positions")
    if not positions:
        return pd.DataFrame(columns=["Ticker", "Quantity"])
    df = pd.DataFrame(list(positions.items()), columns=["Ticker", "Quantity"])
    df["Quantity"] = df["Quantity"].astype(float)
    return df


def get_live_prices():
    """Fetches the latest known price for each stock."""
    prices = redis_client.hgetall("qmind:live_prices")
    return {ticker: float(price) for ticker, price in prices.items()}


def get_event_log():
    """Fetches the latest system events from the Redis log."""
    return redis_client.lrange("qmind:event_log", 0, -1)


# --- Dashboard Layout ---
st.title("QMind Quant - Live Trading Dashboard")

# Fetch all data at the beginning of each page refresh.
cash = get_portfolio_cash()
positions_df = get_positions()
live_prices = get_live_prices()
event_log = get_event_log()

# Perform real-time calculations based on the latest data.
positions_df["Current Price"] = positions_df["Ticker"].map(live_prices).fillna(0)
positions_df["Market Value"] = positions_df["Quantity"] * positions_df["Current Price"]
market_value = positions_df["Market Value"].sum()
total_value = cash + market_value

# --- Display Metrics and Status ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Portfolio Overview")
    st.metric(label="Total Portfolio Value", value=f"${total_value:,.2f}")
    st.metric(label="Cash", value=f"${cash:,.2f}")
    st.metric(label="Market Value", value=f"${market_value:,.2f}")

with col2:
    st.subheader("System Status")
    status = "🟢 Running" if event_log else "🔴 Stopped"
    st.markdown(f"**Engine Status:** `{status}`")
    st.markdown(
        f"**Last Update:** `{pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S')}`"
    )

st.markdown("---")

# --- Display Positions and Logs ---
pos_col, log_col = st.columns([1, 2])
with pos_col:
    st.subheader("Current Positions")
    st.dataframe(
        positions_df[["Ticker", "Quantity", "Current Price", "Market Value"]],
        use_container_width=True,
        hide_index=True,
        column_config={"Market Value": st.column_config.NumberColumn(format="$%.2f")},
    )
with log_col:
    st.subheader("Event Log")
    st.text_area(
        "Event Log",
        value="\n".join(event_log),
        height=300,
        disabled=True,
        label_visibility="collapsed",
    )

# --- Auto-refresh ---
# This tells Streamlit to rerun the script every 2 seconds, keeping the dashboard up-to-date.
time.sleep(2)
st.rerun()
