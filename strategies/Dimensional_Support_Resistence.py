import pandas as pd
import numpy as np

def strategy_dsr(df: pd.DataFrame, **params) -> pd.Series:
    """
    Dimensional Support Resistance (DSR) Strategy
    
    Detects bounces at dynamically calculated support and resistance levels
    based on pivot points with volume confirmation.
    
    Entry conditions:
        - LONG: Pivot low detected (support bounce) + bullish candle + optional volume spike
        - SHORT: Pivot high detected (resistance bounce) + bearish candle + optional volume spike
    
    Exit conditions:
        - Exit after cooldown period (default 15 bars) to allow new signals
        - Position flips when opposite signal occurs
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        lookback: How far back to look for S/R levels (default: 50)
        pivot_strength: Bars needed to confirm a pivot (default: 3)
        max_levels: Max levels each side (default: 2)
        zone_width_pct: Zone width percentage (default: 0.15)
        min_distance_pct: Min distance between levels (default: 1.0)
        vol_multiplier: Volume multiplier for strong signals (default: 1.5)
        signal_cooldown: Minimum bars between signals (default: 15)
        use_strong_only: Only trade strong (high volume) signals (default: False)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry, -1 = short entry, 0 = exit/flat
    """
    df = df.copy()
    
    # Normalize column names (handle both upper and lowercase)
    df.columns = df.columns.str.capitalize()
    
    # Extract parameters
    lookback = params.get('lookback', 50)
    pivot_strength = params.get('pivot_strength', 3)
    max_levels = params.get('max_levels', 2)
    zone_width_pct = params.get('zone_width_pct', 0.15)
    min_distance_pct = params.get('min_distance_pct', 1.0)
    vol_multiplier = params.get('vol_multiplier', 1.5)
    signal_cooldown = params.get('signal_cooldown', 15)
    use_strong_only = params.get('use_strong_only', False)
    
    # === Helper function: Find pivot highs/lows ===
    def find_pivots(series, strength, is_low=True):
        """Find pivot points in a price series."""
        pivots = pd.Series(np.nan, index=series.index)
        
        for i in range(strength, len(series) - strength):
            window_left = series.iloc[i - strength:i]
            window_right = series.iloc[i + 1:i + strength + 1]
            current = series.iloc[i]
            
            if is_low:
                # Pivot low: current is minimum in window
                if current < window_left.min() and current < window_right.min():
                    pivots.iloc[i] = current
            else:
                # Pivot high: current is maximum in window
                if current > window_left.max() and current > window_right.max():
                    pivots.iloc[i] = current
        
        return pivots
    
    # === Calculate indicators ===
    
    # Pivot detection
    pivot_lows = find_pivots(df['Low'], pivot_strength, is_low=True)
    pivot_highs = find_pivots(df['High'], pivot_strength, is_low=False)
    
    # Volume filter
    vol_sma = df['Volume'].rolling(window=20).mean()
    high_volume = df['Volume'] > (vol_sma * vol_multiplier)
    
    # Candle direction
    bullish_candle = df['Close'] > df['Open']
    bearish_candle = df['Close'] < df['Open']
    
    # === Signal generation ===
    signals = pd.Series(0, index=df.index, dtype=int)
    position = 0  # 0=flat, 1=long, -1=short
    last_signal_bar = -signal_cooldown - 1  # Allow signal on first valid bar
    
    for i in range(pivot_strength * 2, len(df)):
        idx = df.index[i]
        can_signal = (i - last_signal_bar) > signal_cooldown
        
        # Support bounce detection
        support_bounce = (not pd.isna(pivot_lows.iloc[i]) and 
                         bullish_candle.iloc[i])
        strong_support = support_bounce and high_volume.iloc[i]
        
        # Resistance bounce detection
        resist_bounce = (not pd.isna(pivot_highs.iloc[i]) and 
                        bearish_candle.iloc[i])
        strong_resist = resist_bounce and high_volume.iloc[i]
        
        # Apply signal logic
        if can_signal:
            if use_strong_only:
                # Only trade strong signals
                if strong_support and position != 1:
                    signals.iloc[i] = 1  # Long entry signal
                    position = 1
                    last_signal_bar = i
                elif strong_resist and position != -1:
                    signals.iloc[i] = -1  # Short entry signal
                    position = -1
                    last_signal_bar = i
            else:
                # Trade all bounces (strong or regular)
                if (support_bounce or strong_support) and position != 1:
                    signals.iloc[i] = 1  # Long entry signal
                    position = 1
                    last_signal_bar = i
                elif (resist_bounce or strong_resist) and position != -1:
                    signals.iloc[i] = -1  # Short entry signal
                    position = -1
                    last_signal_bar = i
    
    return signals


# Example usage
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv("../data/BTC_1H_OHLCV.csv")
    
    # Basic strategy (all bounces)
    signals_all = strategy_dsr(df)
    print("DSR Strategy - All bounces:")
    print(f"  Total signals: {(signals_all != 0).sum()}")
    print(f"  Long entries: {(signals_all == 1).sum()}")
    print(f"  Short entries: {(signals_all == -1).sum()}")
    print(f"  Signal distribution:\n{signals_all.value_counts()}")
    
    # Strong signals only (high volume)
    signals_strong = strategy_dsr(df, use_strong_only=True)
    print("\nDSR Strategy - Strong bounces only:")
    print(f"  Total signals: {(signals_strong != 0).sum()}")
    print(f"  Long entries: {(signals_strong == 1).sum()}")
    print(f"  Short entries: {(signals_strong == -1).sum()}")
    
    # Custom parameters
    signals_custom = strategy_dsr(
        df,
        lookback=100,
        pivot_strength=5,
        signal_cooldown=20,
        vol_multiplier=2.0
    )
    print("\nDSR Strategy - Custom parameters:")
    print(f"  Total signals: {(signals_custom != 0).sum()}")
