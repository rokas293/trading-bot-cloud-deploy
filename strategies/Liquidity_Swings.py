import pandas as pd
import numpy as np

def strategy_liquidity_swings_lux(df: pd.DataFrame, **params) -> pd.Series:
    """
    Liquidity Swings Strategy (LuxAlgo interpretation)
    
    Identifies swing highs/lows that act as liquidity zones where stop losses cluster.
    Trades reversals after price sweeps through these levels (wick penetration) and 
    closes back inside the range, indicating liquidity absorption.
    
    Entry conditions:
        - LONG: Price sweeps below swing low (wick breaks zone), then closes back above bottom
               AND liquidity filter is met (count/volume threshold)
        - SHORT: Price sweeps above swing high (wick breaks zone), then closes back below top
                AND liquidity filter is met (count/volume threshold)
    
    Exit conditions:
        - LONG exits when price closes below swing low zone bottom
        - SHORT exits when price closes above swing high zone top
        - Or when opposite signal occurs (reversal)
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        pivot_length: Pivot lookback period for swing detection (default: 14)
        swing_area: 'wick' or 'full' - zone definition (default: 'wick')
        filter_by: 'count' or 'volume' - filter liquidity zones (default: 'volume')
        filter_value: Minimum count/volume threshold (default: 0)
        require_sweep: Require wick to penetrate zone before signal (default: True)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry, -1 = short entry, 0 = flat/no signal
    """
    df = df.copy()
    
    # Extract parameters
    pivot_length = params.get('pivot_length', 14)
    swing_area = params.get('swing_area', 'wick')  # 'wick' or 'full'
    filter_by = params.get('filter_by', 'volume')
    filter_value = params.get('filter_value', 0)
    require_sweep = params.get('require_sweep', True)
    
    # === Helper: Detect swing highs and lows ===
    def detect_pivots(high_series, low_series, length):
        """Detect pivot highs and lows using Pine Script ta.pivothigh/pivotlow logic."""
        pivot_highs = pd.Series(np.nan, index=high_series.index)
        pivot_lows = pd.Series(np.nan, index=low_series.index)
        
        # Pine's ta.pivothigh(length, length) looks left AND right
        for i in range(length, len(high_series) - length):
            # Pivot high: current bar is highest in window
            window_high_left = high_series.iloc[i - length:i]
            window_high_right = high_series.iloc[i + 1:i + length + 1]
            current_high = high_series.iloc[i]
            
            if current_high >= window_high_left.max() and current_high >= window_high_right.max():
                pivot_highs.iloc[i] = current_high
            
            # Pivot low: current bar is lowest in window
            window_low_left = low_series.iloc[i - length:i]
            window_low_right = low_series.iloc[i + 1:i + length + 1]
            current_low = low_series.iloc[i]
            
            if current_low <= window_low_left.min() and current_low <= window_low_right.min():
                pivot_lows.iloc[i] = current_low
        
        return pivot_highs, pivot_lows
    
    # === Detect pivots ===
    pivot_highs, pivot_lows = detect_pivots(df['High'], df['Low'], pivot_length)
    
    # === Signal generation (stateful loop) ===
    signals = pd.Series(0, index=df.index, dtype=int)
    position = 0
    prev_position = 0
    
    # Active pivot tracking
    ph_top = None
    ph_btm = None
    ph_crossed = False
    ph_x1 = None
    ph_count = 0
    ph_vol = 0.0
    
    pl_top = None
    pl_btm = None
    pl_crossed = False
    pl_x1 = None
    pl_count = 0
    pl_vol = 0.0
    
    for i in range(pivot_length * 2, len(df)):
        idx = df.index[i]
        i_offset = i - pivot_length  # Pine looks back at [length] for pivot
        
        close_val = df.loc[idx, 'Close']
        open_val = df.loc[idx, 'Open']
        high_val = df.loc[idx, 'High']
        low_val = df.loc[idx, 'Low']
        vol_val = df.loc[idx, 'Volume']
        
        # === Check for new pivot high ===
        if not pd.isna(pivot_highs.iloc[i_offset]):
            ph = pivot_highs.iloc[i_offset]
            ph_idx = df.index[i_offset]
            
            # Define zone
            if swing_area == 'wick':
                ph_top = df.loc[ph_idx, 'High']
                ph_btm = max(df.loc[ph_idx, 'Close'], df.loc[ph_idx, 'Open'])
            else:  # 'full'
                ph_top = df.loc[ph_idx, 'High']
                ph_btm = df.loc[ph_idx, 'Low']
            
            ph_x1 = i_offset
            ph_crossed = False
            ph_count = 0
            ph_vol = 0.0
        
        # === Check for new pivot low ===
        if not pd.isna(pivot_lows.iloc[i_offset]):
            pl = pivot_lows.iloc[i_offset]
            pl_idx = df.index[i_offset]
            
            # Define zone
            if swing_area == 'wick':
                pl_top = min(df.loc[pl_idx, 'Close'], df.loc[pl_idx, 'Open'])
                pl_btm = df.loc[pl_idx, 'Low']
            else:  # 'full'
                pl_top = df.loc[pl_idx, 'High']
                pl_btm = df.loc[pl_idx, 'Low']
            
            pl_x1 = i_offset
            pl_crossed = False
            pl_count = 0
            pl_vol = 0.0
        
        # === Accumulate liquidity at pivot high zone ===
        if ph_top is not None and not ph_crossed:
            # Check if price is interacting with zone
            if low_val < ph_top and high_val > ph_btm:
                ph_count += 1
                ph_vol += vol_val
        
        # === Accumulate liquidity at pivot low zone ===
        if pl_top is not None and not pl_crossed:
            # Check if price is interacting with zone
            if low_val < pl_top and high_val > pl_btm:
                pl_count += 1
                pl_vol += vol_val
        
        # === Detect Bearish Liquidity Sweep (SHORT signal) ===
        if ph_top is not None and not ph_crossed:
            # Filter check
            filter_metric = ph_vol if filter_by == 'volume' else ph_count
            
            if filter_metric >= filter_value:
                # Sweep detection: wick breaks above, close returns below
                if require_sweep:
                    sweep_occurred = high_val > ph_top and close_val < ph_top
                else:
                    # Just closing below after accumulation
                    sweep_occurred = close_val < ph_top
                
                if sweep_occurred and position != -1:
                    position = -1
                    ph_crossed = True
            
            # Mark as crossed if price closes above (invalidated)
            if close_val > ph_top:
                ph_crossed = True
        
        # === Detect Bullish Liquidity Sweep (LONG signal) ===
        if pl_top is not None and not pl_crossed:
            # Filter check
            filter_metric = pl_vol if filter_by == 'volume' else pl_count
            
            if filter_metric >= filter_value:
                # Sweep detection: wick breaks below, close returns above
                if require_sweep:
                    sweep_occurred = low_val < pl_btm and close_val > pl_btm
                else:
                    # Just closing above after accumulation
                    sweep_occurred = close_val > pl_btm
                
                if sweep_occurred and position != 1:
                    position = 1
                    pl_crossed = True
            
            # Mark as crossed if price closes below (invalidated)
            if close_val < pl_btm:
                pl_crossed = True
        
        # === Exit logic ===
        # Exit long if price closes below active pivot low
        if position == 1 and pl_btm is not None:
            if close_val < pl_btm:
                position = 0
        
        # Exit short if price closes above active pivot high
        if position == -1 and ph_top is not None:
            if close_val > ph_top:
                position = 0
        
        # Output signal only when position changes
        if position != prev_position:
            signals.iloc[i] = position
            prev_position = position
    
    return signals


# Example usage
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv("../data/BTC_1H_OHLCV.csv")
    
    # Default strategy (no filter, should get more signals)
    signals_default = strategy_liquidity_swings_lux(df, filter_value=0)
    print("Default Liquidity Swings (LuxAlgo style):")
    print(f"  Total signals: {(signals_default != 0).sum()}")
    print(f"  Long entries: {(signals_default == 1).sum()}")
    print(f"  Short entries: {(signals_default == -1).sum()}")
    print("\nSignal distribution:")
    print(signals_default.value_counts().sort_index())
    
    # With volume filter (more selective)
    signals_filtered = strategy_liquidity_swings_lux(df, filter_by='volume', filter_value=500000)
    print("\nWith volume filter (>500k):")
    print(f"  Total signals: {(signals_filtered != 0).sum()}")
    print(signals_filtered.value_counts().sort_index())
    
    # Count-based filtering
    signals_count = strategy_liquidity_swings_lux(df, filter_by='count', filter_value=5)
    print("\nCount-based (min 5 touches):")
    print(f"  Total signals: {(signals_count != 0).sum()}")
    print(signals_count.value_counts().sort_index())
    
    # Full range zones
    signals_full = strategy_liquidity_swings_lux(df, swing_area='full')
    print("\nFull range zones:")
    print(f"  Total signals: {(signals_full != 0).sum()}")
    print(signals_full.value_counts().sort_index())
    
    # No sweep requirement (signal on any close through zone)
    signals_no_sweep = strategy_liquidity_swings_lux(df, require_sweep=False, filter_value=0)
    print("\nNo sweep requirement:")
    print(f"  Total signals: {(signals_no_sweep != 0).sum()}")
    print(signals_no_sweep.value_counts().sort_index())
    
    # Shorter pivot (more sensitive)
    signals_short = strategy_liquidity_swings_lux(df, pivot_length=7, filter_value=0)
    print("\nShorter pivot (7):")
    print(f"  Total signals: {(signals_short != 0).sum()}")
    print(signals_short.value_counts().sort_index())
    
    # Aggressive: short pivot + no filter + no sweep requirement
    signals_aggressive = strategy_liquidity_swings_lux(
        df, 
        pivot_length=7, 
        filter_value=0,
        require_sweep=False
    )
    print("\nAggressive (pivot=7, no filter, no sweep req):")
    print(f"  Total signals: {(signals_aggressive != 0).sum()}")
    print(signals_aggressive.value_counts().sort_index())
