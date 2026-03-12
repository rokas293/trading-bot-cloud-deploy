import pandas as pd
import numpy as np

def strategy_chandelier_exit(df: pd.DataFrame, **params) -> pd.Series:
    """
    Chandelier Exit Strategy (Alex Orekhov / everget)
    
    Trend-following strategy using ATR-based trailing stops that "hang" from the
    highest high (for longs) or lowest low (for shorts). The stops trail price
    movement and only move in the direction of the trend.
    
    Entry conditions:
        - LONG: Direction changes from -1 to 1 (price crosses above short stop)
        - SHORT: Direction changes from 1 to -1 (price crosses below long stop)
    
    Exit conditions:
        - LONG exits when price crosses below long stop (direction flips to -1)
        - SHORT exits when price crosses above short stop (direction flips to 1)
        - Automatic reversal on opposite signal
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        atr_period: ATR calculation period (default: 22)
        atr_multiplier: ATR multiplier for stop distance (default: 3.0)
        use_close: Use close for extremums instead of high/low (default: True)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry, -1 = short entry, 0 = no signal
    """
    df = df.copy()
    
    # Extract parameters
    atr_period = params.get('atr_period', 22)
    atr_multiplier = params.get('atr_multiplier', 3.0)
    use_close = params.get('use_close', True)
    
    # === Calculate ATR ===
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's ATR (RMA in Pine Script)
    df['ATR'] = df['TR'].ewm(alpha=1/atr_period, adjust=False).mean()
    df['atr_distance'] = atr_multiplier * df['ATR']
    
    # === Calculate highest/lowest for stops ===
    if use_close:
        df['highest'] = df['Close'].rolling(window=atr_period).max()
        df['lowest'] = df['Close'].rolling(window=atr_period).min()
    else:
        df['highest'] = df['High'].rolling(window=atr_period).max()
        df['lowest'] = df['Low'].rolling(window=atr_period).min()
    
    # === Calculate Chandelier Exit stops (vectorized) ===
    # Initialize arrays
    n = len(df)
    longStop = np.zeros(n)
    shortStop = np.zeros(n)
    direction = np.ones(n, dtype=int)
    
    # Get numpy arrays for faster access
    close_arr = df['Close'].values
    highest_arr = df['highest'].fillna(0).values
    lowest_arr = df['lowest'].fillna(0).values
    atr_dist_arr = df['atr_distance'].fillna(0).values
    
    # Calculate stops iteratively (required for stateful logic)
    for i in range(1, n):
        # Long stop calculation (trails from highest high)
        longStop_basic = highest_arr[i] - atr_dist_arr[i]
        
        if close_arr[i-1] > longStop[i-1]:
            longStop[i] = max(longStop_basic, longStop[i-1])  # Trail upward
        else:
            longStop[i] = longStop_basic  # Reset
        
        # Short stop calculation (trails from lowest low)
        shortStop_basic = lowest_arr[i] + atr_dist_arr[i]
        
        if close_arr[i-1] < shortStop[i-1]:
            shortStop[i] = min(shortStop_basic, shortStop[i-1])  # Trail downward
        else:
            shortStop[i] = shortStop_basic  # Reset
        
        # Direction determination
        if close_arr[i] > shortStop[i-1]:
            direction[i] = 1  # Uptrend
        elif close_arr[i] < longStop[i-1]:
            direction[i] = -1  # Downtrend
        else:
            direction[i] = direction[i-1]  # Continue current direction
    
    # Assign back to dataframe
    df['longStop'] = longStop
    df['shortStop'] = shortStop
    df['dir'] = direction
    
    # === Generate signals (vectorized) ===
    signals = pd.Series(0, index=df.index, dtype=int)
    
    # Detect direction changes
    dir_changes = df['dir'].diff()
    
    # Long signal: direction changes from -1 to 1 (diff = 2)
    signals[dir_changes == 2] = 1
    
    # Short signal: direction changes from 1 to -1 (diff = -2)
    signals[dir_changes == -2] = -1
    
    # Zero out signals before warmup period
    signals.iloc[:atr_period + 1] = 0
    
    return signals


# Example usage
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv("../data/BTC_1H_OHLCV.csv")
    
    # Default strategy (ATR=22, Mult=3.0, use close)
    signals_default = strategy_chandelier_exit(df)
    print("Default Chandelier Exit strategy (ATR=22, Mult=3.0):")
    print(f"  Total signals: {(signals_default != 0).sum()}")
    print(f"  Long entries: {(signals_default == 1).sum()}")
    print(f"  Short entries: {(signals_default == -1).sum()}")
    print("\nSignal distribution:")
    print(signals_default.value_counts().sort_index())
    
    # More sensitive (lower multiplier)
    signals_sensitive = strategy_chandelier_exit(df, atr_multiplier=2.0)
    print("\nSensitive (multiplier=2.0):")
    print(f"  Total signals: {(signals_sensitive != 0).sum()}")
    print(signals_sensitive.value_counts().sort_index())
    
    # Less sensitive (higher multiplier)
    signals_conservative = strategy_chandelier_exit(df, atr_multiplier=4.0)
    print("\nConservative (multiplier=4.0):")
    print(f"  Total signals: {(signals_conservative != 0).sum()}")
    print(signals_conservative.value_counts().sort_index())
    
    # Use high/low instead of close for extremums
    signals_hl = strategy_chandelier_exit(df, use_close=False)
    print("\nUsing High/Low extremums:")
    print(f"  Total signals: {(signals_hl != 0).sum()}")
    print(signals_hl.value_counts().sort_index())
    
    # Shorter ATR period (more responsive)
    signals_fast = strategy_chandelier_exit(df, atr_period=10)
    print("\nFast (ATR period=10):")
    print(f"  Total signals: {(signals_fast != 0).sum()}")
    print(signals_fast.value_counts().sort_index())
    
    # Longer ATR period (smoother)
    signals_smooth = strategy_chandelier_exit(df, atr_period=30)
    print("\nSmooth (ATR period=30):")
    print(f"  Total signals: {(signals_smooth != 0).sum()}")
    print(signals_smooth.value_counts().sort_index())
    
    # Aggressive: short period + low multiplier
    signals_aggressive = strategy_chandelier_exit(df, atr_period=15, atr_multiplier=2.5)
    print("\nAggressive (ATR=15, Mult=2.5):")
    print(f"  Total signals: {(signals_aggressive != 0).sum()}")
    print(signals_aggressive.value_counts().sort_index())
    
    # Very conservative: long period + high multiplier
    signals_very_conservative = strategy_chandelier_exit(df, atr_period=30, atr_multiplier=5.0)
    print("\nVery conservative (ATR=30, Mult=5.0):")
    print(f"  Total signals: {(signals_very_conservative != 0).sum()}")
    print(signals_very_conservative.value_counts().sort_index())
    
    # Classic settings (period=22 is standard)
    signals_classic = strategy_chandelier_exit(df, atr_period=22, atr_multiplier=3.0, use_close=False)
    print("\nClassic (ATR=22, Mult=3.0, High/Low):")
    print(f"  Total signals: {(signals_classic != 0).sum()}")
    print(signals_classic.value_counts().sort_index())
