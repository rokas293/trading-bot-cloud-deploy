import pandas as pd
import numpy as np

def strategy_supertrend(df: pd.DataFrame, **params) -> pd.Series:
    """
    SuperTrend Strategy (Kıvanç Özbilgiç)
    
    Trend-following strategy using ATR-based dynamic support/resistance bands.
    The SuperTrend indicator adapts to volatility and generates signals when
    price crosses the trend line, indicating trend changes.
    
    Entry conditions:
        - LONG: Trend flips from -1 to 1 (price crosses above down-trend line)
        - SHORT: Trend flips from 1 to -1 (price crosses below up-trend line)
    
    Exit conditions:
        - Automatic reversal when opposite signal occurs
        - LONG exits when trend flips to -1 (short signal)
        - SHORT exits when trend flips to 1 (long signal)
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        atr_period: ATR calculation period (default: 7)
        atr_multiplier: ATR multiplier for band distance (default: 2.0)
        use_atr: Use ta.atr() vs simple TR average (default: True)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry, -1 = short entry, 0 = no signal
    """
    # Extract parameters
    # Backwards-compatible param names: accept 'period' and 'multiplier'
    atr_period = params.get('atr_period', params.get('period', 7))
    atr_multiplier = params.get('atr_multiplier', params.get('multiplier', 2.0))
    use_atr = params.get('use_atr', True)
    
    # Convert to numpy arrays for speed
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(df)
    
    # === Calculate indicators ===
    
    # Source: hl2 (high + low) / 2
    src = (high + low) / 2
    
    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # ATR calculation
    if use_atr:
        # Wilder's smoothed ATR using EMA
        alpha = 1 / atr_period
        atr = np.zeros(n)
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    else:
        # Simple moving average of TR
        atr = np.convolve(tr, np.ones(atr_period)/atr_period, mode='full')[:n]
        atr[:atr_period-1] = np.nan
    
    # === Calculate SuperTrend bands (stateful, using numpy) ===
    up = np.zeros(n)
    dn = np.zeros(n)
    trend = np.ones(n, dtype=np.int32)
    
    # Initial values
    up[0] = src[0] - (atr_multiplier * atr[0])
    dn[0] = src[0] + (atr_multiplier * atr[0])
    
    for i in range(1, n):
        # Calculate basic up/dn bands
        up_basic = src[i] - (atr_multiplier * atr[i])
        dn_basic = src[i] + (atr_multiplier * atr[i])
        
        # SuperTrend up logic (trailing stop for uptrend)
        if close[i-1] > up[i-1]:
            up[i] = max(up_basic, up[i-1])
        else:
            up[i] = up_basic
        
        # SuperTrend dn logic (trailing stop for downtrend)
        if close[i-1] < dn[i-1]:
            dn[i] = min(dn_basic, dn[i-1])
        else:
            dn[i] = dn_basic
        
        # Determine trend
        if trend[i-1] == -1 and close[i] > dn[i-1]:
            trend[i] = 1  # Flip to uptrend
        elif trend[i-1] == 1 and close[i] < up[i-1]:
            trend[i] = -1  # Flip to downtrend
        else:
            trend[i] = trend[i-1]  # Continue current trend
    
    # === Generate signals ===
    signals = np.zeros(n, dtype=np.int32)
    prev_position = 0
    
    for i in range(atr_period + 1, n):
        # Buy signal: trend flips from -1 to 1
        if trend[i] == 1 and trend[i-1] == -1:
            if prev_position != 1:
                signals[i] = 1
                prev_position = 1
        
        # Sell signal: trend flips from 1 to -1
        elif trend[i] == -1 and trend[i-1] == 1:
            if prev_position != -1:
                signals[i] = -1
                prev_position = -1
    
    return pd.Series(signals, index=df.index, dtype=int)


# Example usage
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv("../data/BTC_1H_OHLCV.csv")
    
    # Default strategy
    signals_default = strategy_supertrend(df)
    print("Default SuperTrend strategy (ATR=10, Mult=3.0):")
    print(f"  Total signals: {(signals_default != 0).sum()}")
    print(f"  Long entries: {(signals_default == 1).sum()}")
    print(f"  Short entries: {(signals_default == -1).sum()}")
    print("\nSignal distribution:")
    print(signals_default.value_counts().sort_index())
    
    # More sensitive (lower multiplier)
    signals_sensitive = strategy_supertrend(df, atr_multiplier=2.0)
    print("\nSensitive (multiplier=2.0):")
    print(f"  Total signals: {(signals_sensitive != 0).sum()}")
    print(signals_sensitive.value_counts().sort_index())
    
    # Less sensitive (higher multiplier)
    signals_conservative = strategy_supertrend(df, atr_multiplier=4.0)
    print("\nConservative (multiplier=4.0):")
    print(f"  Total signals: {(signals_conservative != 0).sum()}")
    print(signals_conservative.value_counts().sort_index())
    
    # Longer ATR period (smoother)
    signals_smooth = strategy_supertrend(df, atr_period=20)
    print("\nSmooth (ATR period=20):")
    print(f"  Total signals: {(signals_smooth != 0).sum()}")
    print(signals_smooth.value_counts().sort_index())
    
    # Shorter ATR period (more responsive)
    signals_fast = strategy_supertrend(df, atr_period=5)
    print("\nFast (ATR period=5):")
    print(f"  Total signals: {(signals_fast != 0).sum()}")
    print(signals_fast.value_counts().sort_index())
    
    # Use simple TR average instead of ATR
    signals_simple = strategy_supertrend(df, use_atr=False)
    print("\nSimple TR average:")
    print(f"  Total signals: {(signals_simple != 0).sum()}")
    print(signals_simple.value_counts().sort_index())
    
    # Aggressive: short period + low multiplier
    signals_aggressive = strategy_supertrend(df, atr_period=7, atr_multiplier=2.5)
    print("\nAggressive (ATR=7, Mult=2.5):")
    print(f"  Total signals: {(signals_aggressive != 0).sum()}")
    print(signals_aggressive.value_counts().sort_index())
    
    # Very conservative: long period + high multiplier
    signals_very_conservative = strategy_supertrend(df, atr_period=14, atr_multiplier=5.0)
    print("\nVery conservative (ATR=14, Mult=5.0):")
    print(f"  Total signals: {(signals_very_conservative != 0).sum()}")
    print(signals_very_conservative.value_counts().sort_index())
