import pandas as pd
import ta # type: ignore # pip install ta

def add_all_indicators(df):
    """Adds technical indicators (RSI, MACD, SMA, EMA) to the DataFrame."""

    # Ensure required columns exist
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' and 'Volume' columns.")

    # Make sure Close is a flat 1D series
    close = df['Close']
    if hasattr(close.values, 'shape') and close.values.ndim > 1:
        close = pd.Series(close.values.squeeze(), index=close.index)

    # RSI
    rsi = ta.momentum.RSIIndicator(close=close)
    df['RSI'] = rsi.rsi()

    # MACD
    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # SMA / EMA
    sma = ta.trend.SMAIndicator(close=close, window=20)
    df['SMA_20'] = sma.sma_indicator()

    ema = ta.trend.EMAIndicator(close=close, window=20)
    df['EMA_20'] = ema.ema_indicator()

    # Drop any rows with NaNs created by indicators
    df.dropna(inplace=True)

    return df