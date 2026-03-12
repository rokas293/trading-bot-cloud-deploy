import pandas as pd
import numpy as np

def strategy_ut_bot(df: pd.DataFrame, **params) -> pd.Series:
    """
    UT Bot Alerts Strategy (ATR Trailing Stop)
    
    Dynamic trend-following strategy using ATR-based trailing stops. The indicator
    adapts to volatility by adjusting stop levels based on Average True Range,
    generating signals when price crosses the trailing stop line.
    
    Entry conditions:
        - LONG: Price crosses above ATR trailing stop AND EMA crosses above trailing stop
        - SHORT: Price crosses below ATR trailing stop AND EMA crosses below trailing stop
    
    Exit conditions:
        - LONG exits when price crosses below ATR trailing stop
        - SHORT exits when price crosses above ATR trailing stop
        - Automatic reversal when opposite signal occurs
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        key_value: Sensitivity multiplier for ATR (default: 1, higher = wider stops)
        atr_period: ATR calculation period (default: 10)
        ema_period: EMA smoothing period (default: 1, set higher for smoother signals)
        use_heikin_ashi: Use Heikin Ashi candles for signals (default: False)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry/hold, -1 = short entry/hold, 0 = flat/exit
    """
    df = df.copy()
    
    # Extract parameters
    key_value = params.get('key_value', 1)
    atr_period = params.get('atr_period', 10)
    ema_period = params.get('ema_period', 1)
    use_heikin_ashi = params.get('use_heikin_ashi', False)
    
    # === Calculate indicators ===
    
    # ATR calculation
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()
    
    # Loss distance (ATR * key_value)
    df['nLoss'] = key_value * df['ATR']
    
    # Source price (Close or Heikin Ashi Close)
    if use_heikin_ashi:
        # Calculate Heikin Ashi candles (optimized)
        df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # Pre-allocate HA_Open array for speed
        ha_open = np.zeros(len(df))
        ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
        
        # Efficient loop using numpy array
        ha_close_vals = df['HA_Close'].values
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close_vals[i-1]) / 2
        
        df['HA_Open'] = ha_open
        df['src'] = df['HA_Close']
    else:
        df['src'] = df['Close']
    
    # EMA of source (configurable period)
    df['ema'] = df['src'].ewm(span=ema_period, adjust=False).mean()
    
    # === Calculate ATR Trailing Stop (stateful) ===
    df['xATRTrailingStop'] = 0.0
    
    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        
        src_curr = df.loc[idx, 'src']
        src_prev = df.loc[prev_idx, 'src']
        n_loss = df.loc[idx, 'nLoss']
        prev_stop = df.loc[prev_idx, 'xATRTrailingStop']
        
        # ATR Trailing Stop logic (Pine Script translation)
        if src_curr > prev_stop and src_prev > prev_stop:
            # Uptrend: trailing stop moves up
            df.loc[idx, 'xATRTrailingStop'] = max(prev_stop, src_curr - n_loss)
        elif src_curr < prev_stop and src_prev < prev_stop:
            # Downtrend: trailing stop moves down
            df.loc[idx, 'xATRTrailingStop'] = min(prev_stop, src_curr + n_loss)
        elif src_curr > prev_stop:
            # Just flipped to uptrend
            df.loc[idx, 'xATRTrailingStop'] = src_curr - n_loss
        else:
            # Just flipped to downtrend
            df.loc[idx, 'xATRTrailingStop'] = src_curr + n_loss
    
    # === Signal generation ===
    signals = pd.Series(0, index=df.index, dtype=int)
    position = 0
    prev_position = 0
    
    for i in range(atr_period + 1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        
        src = df.loc[idx, 'src']
        src_prev = df.loc[prev_idx, 'src']
        atr_stop = df.loc[idx, 'xATRTrailingStop']
        atr_stop_prev = df.loc[prev_idx, 'xATRTrailingStop']
        ema = df.loc[idx, 'ema']
        ema_prev = df.loc[prev_idx, 'ema']
        
        # Detect crossovers
        ema_cross_above = ema > atr_stop and ema_prev <= atr_stop_prev
        ema_cross_below = atr_stop > ema and atr_stop_prev <= ema_prev
        
        # Buy signal: price above trailing stop AND ema crosses above
        buy = src > atr_stop and ema_cross_above
        
        # Sell signal: price below trailing stop AND ema crosses below
        sell = src < atr_stop and ema_cross_below
        
        # Update position
        if buy and position != 1:
            position = 1
        elif sell and position != -1:
            position = -1
        
        # Output signal only when position changes
        if position != prev_position:
            signals.iloc[i] = position
            prev_position = position
    
    return signals


# Example usage
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv("../data/BTC_1H_OHLCV.csv")
    
    # Default strategy (EMA period = 1, essentially price crossover)
    signals_default = strategy_ut_bot(df)
    print("Default UT Bot strategy (EMA=1):")
    print(f"  Total signals: {(signals_default != 0).sum()}")
    print(f"  Long entries: {(signals_default == 1).sum()}")
    print(f"  Short entries: {(signals_default == -1).sum()}")
    print("\nSignal distribution:")
    print(signals_default.value_counts().sort_index())
    
    # Smoother signals with EMA period = 5
    signals_smooth = strategy_ut_bot(df, ema_period=5)
    print("\nSmooth EMA (ema_period=5):")
    print(f"  Total signals: {(signals_smooth != 0).sum()}")
    print(signals_smooth.value_counts().sort_index())
    
    # Smoother signals with EMA period = 10
    signals_smooth10 = strategy_ut_bot(df, ema_period=10)
    print("\nSmooth EMA (ema_period=10):")
    print(f"  Total signals: {(signals_smooth10 != 0).sum()}")
    print(signals_smooth10.value_counts().sort_index())
    
    # More sensitive (lower key_value)
    signals_sensitive = strategy_ut_bot(df, key_value=0.5)
    print("\nSensitive (key_value=0.5):")
    print(f"  Total signals: {(signals_sensitive != 0).sum()}")
    print(signals_sensitive.value_counts().sort_index())
    
    # Less sensitive (higher key_value)
    signals_conservative = strategy_ut_bot(df, key_value=2.0)
    print("\nConservative (key_value=2.0):")
    print(f"  Total signals: {(signals_conservative != 0).sum()}")
    print(signals_conservative.value_counts().sort_index())
    
    # Longer ATR period
    signals_long_atr = strategy_ut_bot(df, atr_period=20)
    print("\nLonger ATR period (20):")
    print(f"  Total signals: {(signals_long_atr != 0).sum()}")
    print(signals_long_atr.value_counts().sort_index())
    
    # With Heikin Ashi candles
    signals_ha = strategy_ut_bot(df, use_heikin_ashi=True)
    print("\nWith Heikin Ashi candles:")
    print(f"  Total signals: {(signals_ha != 0).sum()}")
    print(signals_ha.value_counts().sort_index())
    
    # Aggressive: short ATR + low key value
    signals_aggressive = strategy_ut_bot(df, key_value=0.75, atr_period=7)
    print("\nAggressive (key_value=0.75, atr_period=7):")
    print(f"  Total signals: {(signals_aggressive != 0).sum()}")
    print(signals_aggressive.value_counts().sort_index())
    
    # Smooth with higher EMA and Heikin Ashi
    signals_ultra_smooth = strategy_ut_bot(df, ema_period=8, use_heikin_ashi=True)
    print("\nUltra Smooth (EMA=8 + Heikin Ashi):")
    print(f"  Total signals: {(signals_ultra_smooth != 0).sum()}")
    print(signals_ultra_smooth.value_counts().sort_index())
