# qmind_quant/analytics/technical_indicators.py

import pandas as pd
import numpy as np

# A. Trend Indicators
# ==============================================================================


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA).
    It's used to identify the trend direction by smoothing out price fluctuations.
    """
    return prices.rolling(window=window).mean()


def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).
    It's a moving average that gives more weight to recent prices, making it more responsive.
    """
    return prices.ewm(span=window, adjust=False).mean()


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    It shows the relationship between two moving averages to reveal changes in trend momentum.
    """
    ema_fast = calculate_ema(prices, window=fast_period)
    ema_slow = calculate_ema(prices, window=slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, window=signal_period)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculates the Average Directional Index (ADX).
    It's used to quantify the strength of a market trend, regardless of its direction.
    """
    plus_dm = high.diff()
    minus_dm = low.diff().mul(-1)

    plus_dm[plus_dm < 0] = 0
    plus_dm[minus_dm > plus_dm] = 0

    minus_dm[minus_dm < 0] = 0
    minus_dm[plus_dm > minus_dm] = 0

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / window, min_periods=window).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, min_periods=window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, min_periods=window).mean() / atr)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1 / window, min_periods=window).mean()

    return adx


# B. Momentum Indicators
# ==============================================================================


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    It's a momentum oscillator that measures the speed of price changes to identify overbought or oversold conditions.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculates the Stochastic Oscillator (%K).
    It's a momentum indicator comparing a closing price to its price range over time to find overbought and oversold signals.
    """
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()

    percent_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    return percent_k


# C. Volatility Indicators
# ==============================================================================


def calculate_bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: int = 2
) -> pd.DataFrame:
    """
    Calculates Bollinger Bands.
    They are volatility bands placed above and below a moving average, used to identify periods of high or low volatility.
    """
    middle_band = calculate_sma(prices, window)
    rolling_std = prices.rolling(window=window).std()
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    return pd.DataFrame(
        {"bb_upper": upper_band, "bb_middle": middle_band, "bb_lower": lower_band}
    )


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculates the Average True Range (ATR).
    It's a volatility indicator that shows the average size of the price range over a period.
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    return atr


# D. Volume Indicators
# ==============================================================================


def calculate_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculates the On-Balance Volume (OBV).
    It's a momentum indicator that uses volume flow to gauge buying and selling pressure.
    """
    price_change_direction = np.sign(prices.diff()).fillna(0)
    obv = (volume * price_change_direction).cumsum()
    return obv


def calculate_vwap(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculates the Volume-Weighted Average Price (VWAP).
    It provides the average price a security has traded at throughout the period, weighted by volume.
    """
    return (prices * volume).cumsum() / volume.cumsum()
