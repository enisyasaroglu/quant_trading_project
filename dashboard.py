# dashboard.py

import os
import time
import redis
import pandas as pd
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="QMind Quant Dashboard",
    layout="wide",
)


# --- Data Connection ---
@st.cache_resource
def get_redis_connection():
    """
    Establishes a connection to Redis.
    It intelligently checks for a cloud environment variable (REDIS_URL)
    and falls back to localhost if it's not found.
    """
    # Railway provides the connection URL in the REDIS_URL environment variable.
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        print("Connecting to cloud Redis...")
        # Use the cloud URL to connect.
        return redis.from_url(redis_url, decode_responses=True)
    else:
        # Fallback to a local connection if the cloud URL isn't found.
        print("Connecting to local Redis...")
        return redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )


redis_client = get_redis_connection()


# --- Helper Functions (No changes needed here) ---
def get_portfolio_cash():
    return float(redis_client.get("qmind:portfolio:cash") or 0)


def get_positions():
    positions = redis_client.hgetall("qmind:positions")
    if not positions:
        return pd.DataFrame(columns=["Ticker", "Quantity"])
    df = pd.DataFrame(list(positions.items()), columns=["Ticker", "Quantity"])
    df["Quantity"] = df["Quantity"].astype(float)
    return df


def get_live_prices():
    prices = redis_client.hgetall("qmind:live_prices")
    return {ticker: float(price) for ticker, price in prices.items()}


def get_event_log():
    return redis_client.lrange("qmind:event_log", 0, -1)


# --- Dashboard Layout (No changes needed here) ---
st.title("QMind Quant - Live Trading Dashboard")

cash = get_portfolio_cash()
positions_df = get_positions()
live_prices = get_live_prices()
event_log = get_event_log()

positions_df["Current Price"] = positions_df["Ticker"].map(live_prices).fillna(0)
positions_df["Market Value"] = positions_df["Quantity"] * positions_df["Current Price"]
market_value = positions_df["Market Value"].sum()
total_value = cash + market_value

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

time.sleep(2)
st.rerun()
