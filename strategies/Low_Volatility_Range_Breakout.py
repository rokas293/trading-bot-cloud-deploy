import pandas as pd
import numpy as np

def strategy_lvrb(df: pd.DataFrame, **params) -> pd.Series:
    """
    Low Volatility Range Breakout (LVRB) Strategy
    
    Identifies consolidation periods with low volatility (low TR and small candle bodies),
    then trades breakouts from the established range with optional volume/body filters.
    
    Entry conditions:
        - LONG: Price breaks above consolidation range high (wick or close mode)
        - SHORT: Price breaks below consolidation range low (wick or close mode)
        - Must meet minimum bars in range (default 6) with sufficient low-vol fraction (0.65)
        - Optional: breakout candle body size filter and direction filter
    
    Exit conditions:
        - Position exits when opposite breakout occurs
        - Range is invalidated if height exceeds max (baseATR × 1.5) or gap limit exceeded
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        len_vol: Volatility window for SMA (default: 20)
        tr_mult: TR threshold multiplier (default: 1.0, means TR <= avgTR × 1.0)
        body_mult: Body threshold multiplier (default: 1.1)
        use_body: Use body filter for low-vol detection (default: True)
        min_bars: Minimum bars in one consolidation box (default: 6)
        min_good_frac: Minimum fraction of low-vol bars in box (default: 0.65)
        gap_max: Max consecutive non-low-vol bars allowed (default: 5)
        height_mult: Max box height = baseATR × this (default: 1.5)
        break_mode: "Wick" or "Close" breakout mode (default: "Wick")
        breakout_body_mult: Breakout candle body multiplier (default: 1.0 = OFF)
        require_candle_color: Require candle direction match (default: False)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry/hold, -1 = short entry/hold, 0 = flat/exit
    """
    df = df.copy()
    
    # Extract parameters
    len_vol = params.get('len_vol', 20)
    tr_mult = params.get('tr_mult', 1.0)
    body_mult = params.get('body_mult', 1.1)
    use_body = params.get('use_body', True)
    min_bars = params.get('min_bars', 6)
    min_good_frac = params.get('min_good_frac', 0.65)
    gap_max = params.get('gap_max', 5)
    height_mult = params.get('height_mult', 1.5)
    break_mode = params.get('break_mode', 'Wick')
    breakout_body_mult = params.get('breakout_body_mult', 1.0)
    require_candle_color = params.get('require_candle_color', False)
    
    min_good_bars = int(np.ceil(min_bars * min_good_frac))
    
    # === Calculate indicators (vectorized) ===
    
    # True Range calculation
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['avgTR'] = df['TR'].rolling(window=len_vol).mean()
    
    # Body size
    df['body'] = (df['Close'] - df['Open']).abs()
    df['avgBody'] = df['body'].rolling(window=len_vol).mean()
    
    # Low volatility conditions
    df['lvTR'] = df['TR'] <= (df['avgTR'] * tr_mult)
    if use_body:
        df['lvBody'] = df['body'] <= (df['avgBody'] * body_mult)
        df['isLowVol'] = df['lvTR'] & df['lvBody']
    else:
        df['isLowVol'] = df['lvTR']
    
    # Candle properties
    df['bullish'] = df['Close'] > df['Open']
    df['bearish'] = df['Close'] < df['Open']
    
    # === Signal generation (stateful loop) ===
    signals = pd.Series(0, index=df.index, dtype=int)
    position = 0
    prev_position = 0  # Track previous position to detect changes
    
    # State variables
    in_seq = False
    qualified = False
    start_idx = None
    total_bars = 0
    good_bars = 0
    fail_streak = 0
    rng_hi = None
    rng_lo = None
    base_atr = None
    base_body0 = None
    
    for i in range(len(df)):
        idx = df.index[i]
        
        if not in_seq:
            # Start new sequence on low-vol bar
            if df.loc[idx, 'isLowVol'] and not pd.isna(df.loc[idx, 'avgTR']):
                in_seq = True
                qualified = False
                start_idx = i
                total_bars = 1
                good_bars = 1
                fail_streak = 0
                rng_hi = df.loc[idx, 'High']
                rng_lo = df.loc[idx, 'Low']
                base_atr = df.loc[idx, 'avgTR']
                base_body0 = df.loc[idx, 'avgBody']
        else:
            # Check for breakout if qualified
            if qualified and rng_hi is not None and rng_lo is not None:
                high_val = df.loc[idx, 'High']
                low_val = df.loc[idx, 'Low']
                close_val = df.loc[idx, 'Close']
                body_now = df.loc[idx, 'body']
                
                # Breakout conditions
                up_wick_pierce = high_val > rng_hi
                dn_wick_pierce = low_val < rng_lo
                up_close_out = close_val > rng_hi
                dn_close_out = close_val < rng_lo
                
                if break_mode == "Wick":
                    up_cond = up_wick_pierce and up_close_out
                    dn_cond = dn_wick_pierce and dn_close_out
                else:  # "Close"
                    up_cond = up_close_out
                    dn_cond = dn_close_out
                
                # Body size filter
                big_ok = True
                if breakout_body_mult > 1.0 and not pd.isna(base_body0):
                    big_ok = body_now >= base_body0 * breakout_body_mult
                
                # Direction filter
                dir_ok_up = True if not require_candle_color else df.loc[idx, 'bullish']
                dir_ok_dn = True if not require_candle_color else df.loc[idx, 'bearish']
                
                bull_break = up_cond and big_ok and dir_ok_up
                bear_break = dn_cond and big_ok and dir_ok_dn
                
                if bull_break:
                    position = 1
                    # Reset sequence
                    in_seq = False
                    qualified = False
                    start_idx = None
                    total_bars = 0
                    good_bars = 0
                    fail_streak = 0
                    rng_hi = None
                    rng_lo = None
                    base_atr = None
                    base_body0 = None
                elif bear_break:
                    position = -1
                    # Reset sequence
                    in_seq = False
                    qualified = False
                    start_idx = None
                    total_bars = 0
                    good_bars = 0
                    fail_streak = 0
                    rng_hi = None
                    rng_lo = None
                    base_atr = None
                    base_body0 = None
            
            # Extend/update range if still in sequence
            if in_seq:
                if pd.isna(base_atr) or pd.isna(df.loc[idx, 'avgTR']):
                    # Reset if avgTR becomes invalid
                    in_seq = False
                    qualified = False
                    start_idx = None
                    total_bars = 0
                    good_bars = 0
                    fail_streak = 0
                    rng_hi = None
                    rng_lo = None
                    base_atr = None
                    base_body0 = None
                else:
                    close_val = df.loc[idx, 'Close']
                    high_val = df.loc[idx, 'High']
                    low_val = df.loc[idx, 'Low']
                    
                    # Check if price is outside range (not yet confirmed)
                    out_now = qualified and (close_val > rng_hi or close_val < rng_lo)
                    
                    if not out_now:
                        # Detect wick-out but re-entered
                        wick_out_but_reentered = (qualified and 
                                                 ((high_val > rng_hi) or (low_val < rng_lo)) and
                                                 (close_val <= rng_hi) and (close_val >= rng_lo))
                        
                        # Calculate next state
                        if wick_out_but_reentered:
                            next_hi = rng_hi
                            next_lo = rng_lo
                        else:
                            next_hi = max(rng_hi, high_val)
                            next_lo = min(rng_lo, low_val)
                        
                        next_total = total_bars + 1
                        is_low_vol_now = df.loc[idx, 'isLowVol']
                        next_good = good_bars + (1 if is_low_vol_now else 0)
                        next_fail = 0 if is_low_vol_now else (fail_streak + 1)
                        
                        next_frac = next_good / next_total
                        next_qual = (next_total >= min_bars and 
                                   next_good >= min_good_bars and 
                                   next_frac >= min_good_frac)
                        
                        # Check violations
                        violates = ((next_hi - next_lo) > base_atr * height_mult or 
                                  next_fail > gap_max)
                        
                        if violates:
                            # Reset and optionally start new sequence
                            in_seq = False
                            qualified = False
                            start_idx = None
                            total_bars = 0
                            good_bars = 0
                            fail_streak = 0
                            rng_hi = None
                            rng_lo = None
                            base_atr = None
                            base_body0 = None
                            
                            # Start new sequence on same bar if conditions met
                            if is_low_vol_now and not pd.isna(df.loc[idx, 'avgTR']):
                                in_seq = True
                                qualified = False
                                start_idx = i
                                total_bars = 1
                                good_bars = 1
                                fail_streak = 0
                                rng_hi = high_val
                                rng_lo = low_val
                                base_atr = df.loc[idx, 'avgTR']
                                base_body0 = df.loc[idx, 'avgBody']
                        else:
                            # Accept update
                            rng_hi = next_hi
                            rng_lo = next_lo
                            total_bars = next_total
                            good_bars = next_good
                            fail_streak = next_fail
                            
                            # Become qualified
                            if next_qual and not qualified:
                                qualified = True
        
        # CRITICAL FIX: Only signal when position changes
        if position != prev_position:
            signals.iloc[i] = position
            prev_position = position
        # else: signals.iloc[i] stays 0 (default)
    
    return signals


# Example usage
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv("../data/BTC_1H_OHLCV.csv")
    
    # Default strategy
    signals_default = strategy_lvrb(df)
    print("Default LVRB strategy:")
    print(f"  Total signals: {(signals_default != 0).sum()}")
    print(f"  Long entries: {(signals_default == 1).sum()}")
    print(f"  Short entries: {(signals_default == -1).sum()}")
    print("\nSignal distribution:")
    print(signals_default.value_counts().sort_index())
    
    # Close-based breakouts only
    signals_close = strategy_lvrb(df, break_mode='Close')
    print("\nClose breakouts only:")
    print(f"  Total signals: {(signals_close != 0).sum()}")
    print(signals_close.value_counts().sort_index())
    
    # Strict: require larger breakout body and candle color
    signals_strict = strategy_lvrb(
        df,
        breakout_body_mult=1.5,
        require_candle_color=True
    )
    print("\nStrict breakouts (large body + color):")
    print(f"  Total signals: {(signals_strict != 0).sum()}")
    print(signals_strict.value_counts().sort_index())
    
    # Tighter consolidation requirements
    signals_tight = strategy_lvrb(
        df,
        min_bars=10,
        min_good_frac=0.80,
        tr_mult=0.8,
        body_mult=0.9
    )
    print("\nTighter consolidation:")
    print(f"  Total signals: {(signals_tight != 0).sum()}")
    print(signals_tight.value_counts().sort_index())
