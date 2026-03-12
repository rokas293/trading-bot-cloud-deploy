import pandas as pd
import numpy as np

def strategy_order_blocks(df: pd.DataFrame, **params) -> pd.Series:
    """
    Order Blocks & Breaker Blocks Strategy
    
    Identifies institutional order blocks (bullish and bearish zones) based on swing 
    detection, then trades breakouts and breaker block formations (polarity changes).
    
    Entry conditions:
        - LONG: Price breaks above swing high AND creates bullish order block
        - SHORT: Price breaks below swing low AND creates bearish order block
        - Alternative: Trade breaker block formations (when OB is violated and flips polarity)
    
    Exit conditions:
        - Exits occur automatically when opposite signal is generated (reversal-based)
        - LONG → SHORT: When bearish breaker forms
        - SHORT → LONG: When bullish breaker forms
        - Optional: Exit when price closes through active order block
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        swing_length: Swing lookback period (default: 10)
        use_body: Use candle body instead of wicks (default: False)
        max_bull_ob: Max number of bullish order blocks to track (default: 3)
        max_bear_ob: Max number of bearish order blocks to track (default: 3)
        trade_breakers: Trade breaker block polarity changes (default: True)
        trade_initial_breaks: Trade initial breakouts through swings (default: False)
        use_ob_exits: Exit when price closes through active OB (default: False)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry/hold, -1 = short entry/hold, 0 = flat/exit
    """
    df = df.copy()
    
    # Extract parameters - defaults adjusted for signal generation
    swing_length = params.get('swing_length', 5)
    use_body = params.get('use_body', False)
    max_bull_ob = params.get('max_bull_ob', 3)
    max_bear_ob = params.get('max_bear_ob', 3)
    trade_breakers = params.get('trade_breakers', True)
    trade_initial_breaks = params.get('trade_initial_breaks', True)
    use_ob_exits = params.get('use_ob_exits', False)
    
    # === Helper: Detect swing highs and lows ===
    def detect_swings(high_series, low_series, length):
        """Detect swing highs and lows using rolling window."""
        swing_highs = pd.Series(np.nan, index=high_series.index)
        swing_lows = pd.Series(np.nan, index=low_series.index)
        
        for i in range(length, len(high_series) - length):
            # Check if current bar is swing high
            window_high_left = high_series.iloc[i - length:i]
            window_high_right = high_series.iloc[i + 1:i + length + 1]
            current_high = high_series.iloc[i]
            
            if current_high > window_high_left.max() and current_high > window_high_right.max():
                swing_highs.iloc[i] = current_high
            
            # Check if current bar is swing low
            window_low_left = low_series.iloc[i - length:i]
            window_low_right = low_series.iloc[i + 1:i + length + 1]
            current_low = low_series.iloc[i]
            
            if current_low < window_low_left.min() and current_low < window_low_right.min():
                swing_lows.iloc[i] = current_low
        
        return swing_highs, swing_lows
    
    # === Calculate indicators ===
    
    # Price references (body or wick)
    if use_body:
        df['upper'] = df[['Close', 'Open']].max(axis=1)
        df['lower'] = df[['Close', 'Open']].min(axis=1)
    else:
        df['upper'] = df['High']
        df['lower'] = df['Low']
    
    # Detect swings
    swing_highs, swing_lows = detect_swings(df['High'], df['Low'], swing_length)
    df['swing_high'] = swing_highs
    df['swing_low'] = swing_lows
    
    # === Signal generation (stateful loop) ===
    signals = pd.Series(0, index=df.index, dtype=int)
    position = 0
    prev_position = 0
    
    # Track order blocks
    bullish_obs = []  # List of dicts: {'top', 'btm', 'loc', 'breaker', 'break_loc'}
    bearish_obs = []
    
    # Track last swing states
    last_swing_high = {'value': None, 'index': None, 'crossed': False}
    last_swing_low = {'value': None, 'index': None, 'crossed': False}
    
    for i in range(swing_length * 2, len(df)):
        idx = df.index[i]
        close_val = df.loc[idx, 'Close']
        open_val = df.loc[idx, 'Open']
        upper_val = df.loc[idx, 'upper']
        lower_val = df.loc[idx, 'lower']
        
        # Update last swing high (new swing detected)
        if not pd.isna(df.loc[idx, 'swing_high']):
            last_swing_high = {
                'value': df.loc[idx, 'swing_high'],
                'index': i,
                'crossed': False
            }
        
        # Update last swing low (new swing detected)
        if not pd.isna(df.loc[idx, 'swing_low']):
            last_swing_low = {
                'value': df.loc[idx, 'swing_low'],
                'index': i,
                'crossed': False
            }
        
        # === Bullish Order Block Detection ===
        if (last_swing_high['value'] is not None and 
            not last_swing_high['crossed'] and 
            close_val > last_swing_high['value']):
            
            last_swing_high['crossed'] = True
            
            # Generate LONG signal if trading initial breaks
            if trade_initial_breaks and position != 1:
                position = 1
            
            # Find the order block (candle with lowest low before swing high)
            minima = float('inf')
            maxima = None
            ob_loc = None
            
            for j in range(1, i - last_swing_high['index']):
                check_idx = i - j
                if check_idx < 0:
                    break
                
                check_lower = df['lower'].iloc[check_idx]
                check_upper = df['upper'].iloc[check_idx]
                
                if check_lower < minima:
                    minima = check_lower
                    maxima = check_upper
                    ob_loc = check_idx
            
            if ob_loc is not None:
                bullish_obs.insert(0, {
                    'top': maxima,
                    'btm': minima,
                    'loc': ob_loc,
                    'breaker': False,
                    'break_loc': None
                })
        
        # === Bearish Order Block Detection ===
        if (last_swing_low['value'] is not None and 
            not last_swing_low['crossed'] and 
            close_val < last_swing_low['value']):
            
            last_swing_low['crossed'] = True
            
            # Generate SHORT signal if trading initial breaks
            if trade_initial_breaks and position != -1:
                position = -1
            
            # Find the order block (candle with highest high before swing low)
            maxima = float('-inf')
            minima = None
            ob_loc = None
            
            for j in range(1, i - last_swing_low['index']):
                check_idx = i - j
                if check_idx < 0:
                    break
                
                check_upper = df['upper'].iloc[check_idx]
                check_lower = df['lower'].iloc[check_idx]
                
                if check_upper > maxima:
                    maxima = check_upper
                    minima = check_lower
                    ob_loc = check_idx
            
            if ob_loc is not None:
                bearish_obs.insert(0, {
                    'top': maxima,
                    'btm': minima,
                    'loc': ob_loc,
                    'breaker': False,
                    'break_loc': None
                })
        
        # === Optional: Exit Logic (closes through active OB) ===
        if use_ob_exits:
            # Exit long if price closes below active bullish OB
            if position == 1:
                for ob in bullish_obs:
                    if not ob['breaker']:  # Active OB, not breaker yet
                        if close_val < ob['btm']:
                            position = 0
                            break
            
            # Exit short if price closes above active bearish OB
            if position == -1:
                for ob in bearish_obs:
                    if not ob['breaker']:  # Active OB, not breaker yet
                        if close_val > ob['top']:
                            position = 0
                            break
        
        # === Update Bullish Order Blocks ===
        bull_break_confirmed = False
        
        for ob_idx in range(len(bullish_obs) - 1, -1, -1):
            ob = bullish_obs[ob_idx]
            
            if not ob['breaker']:
                # Check if OB is broken (price closes below)
                if min(close_val, open_val) < ob['btm']:
                    ob['breaker'] = True
                    ob['break_loc'] = i
            else:
                # OB is now a breaker block
                # Remove if price closes above top (invalidated)
                if close_val > ob['top']:
                    bullish_obs.pop(ob_idx)
                # Check for breaker block confirmation (swing high forms in breaker zone)
                elif (ob_idx < max_bull_ob and 
                      last_swing_high['value'] is not None and
                      last_swing_high['value'] < ob['top'] and 
                      last_swing_high['value'] > ob['btm']):
                    bull_break_confirmed = True
        
        # === Update Bearish Order Blocks ===
        bear_break_confirmed = False
        
        for ob_idx in range(len(bearish_obs) - 1, -1, -1):
            ob = bearish_obs[ob_idx]
            
            if not ob['breaker']:
                # Check if OB is broken (price closes above)
                if max(close_val, open_val) > ob['top']:
                    ob['breaker'] = True
                    ob['break_loc'] = i
            else:
                # OB is now a breaker block
                # Remove if price closes below bottom (invalidated)
                if close_val < ob['btm']:
                    bearish_obs.pop(ob_idx)
                # Check for breaker block confirmation (swing low forms in breaker zone)
                elif (ob_idx < max_bear_ob and 
                      last_swing_low['value'] is not None and
                      last_swing_low['value'] > ob['btm'] and 
                      last_swing_low['value'] < ob['top']):
                    bear_break_confirmed = True
        
        # === Generate Trading Signals (Breaker Blocks) ===
        if trade_breakers:
            # Trade breaker block formations (polarity changes)
            if bear_break_confirmed and position != -1:
                position = -1
            elif bull_break_confirmed and position != 1:
                position = 1
        
        # Trim order block lists to max size
        if len(bullish_obs) > max_bull_ob:
            bullish_obs = bullish_obs[:max_bull_ob]
        if len(bearish_obs) > max_bear_ob:
            bearish_obs = bearish_obs[:max_bear_ob]
        
        # Output signal only when position changes
        if position != prev_position:
            signals.iloc[i] = position
            prev_position = position
    
    return signals


# Example usage
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv("../data/BTC_1H_OHLCV.csv")
    
    # Default strategy (breaker blocks only)
    signals_breakers = strategy_order_blocks(df)
    print("Breaker Blocks strategy:")
    print(f"  Total signals: {(signals_breakers != 0).sum()}")
    print(f"  Long entries: {(signals_breakers == 1).sum()}")
    print(f"  Short entries: {(signals_breakers == -1).sum()}")
    print("\nSignal distribution:")
    print(signals_breakers.value_counts().sort_index())
    
    # Trade initial breakouts
    signals_breaks = strategy_order_blocks(df, trade_breakers=False, trade_initial_breaks=True)
    print("\nInitial breakouts strategy:")
    print(f"  Total signals: {(signals_breaks != 0).sum()}")
    print(signals_breaks.value_counts().sort_index())
    
    # Both breakers and initial breaks
    signals_both = strategy_order_blocks(df, trade_breakers=True, trade_initial_breaks=True)
    print("\nCombined strategy:")
    print(f"  Total signals: {(signals_both != 0).sum()}")
    print(signals_both.value_counts().sort_index())
    
    # With OB exit logic
    signals_exits = strategy_order_blocks(df, use_ob_exits=True)
    print("\nWith OB exits:")
    print(f"  Total signals: {(signals_exits != 0).sum()}")
    print(signals_exits.value_counts().sort_index())
    
    # Use candle bodies instead of wicks
    signals_body = strategy_order_blocks(df, use_body=True)
    print("\nBody-based order blocks:")
    print(f"  Total signals: {(signals_body != 0).sum()}")
    print(signals_body.value_counts().sort_index())
    
    # More sensitive (shorter swing length)
    signals_sensitive = strategy_order_blocks(df, swing_length=5)
    print("\nSensitive (swing_length=5):")
    print(f"  Total signals: {(signals_sensitive != 0).sum()}")
    print(signals_sensitive.value_counts().sort_index())
