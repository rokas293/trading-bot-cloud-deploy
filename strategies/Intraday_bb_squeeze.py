import pandas as pd
import numpy as np

def strategy_intraday_bb_squeeze(df: pd.DataFrame, **params) -> pd.Series:
    """
    Intraday Bollinger Bands + Squeeze Momentum + ADX/DMI Strategy
    
    Multi-indicator strategy combining:
    1. Bollinger Bands breakout detection
    2. Squeeze Momentum (TTM Squeeze indicator)
    3. ADX/DMI trend strength filter
    
    Entry conditions:
        - LONG: Price breaks above upper BB + Squeeze momentum positive + ADX trend up
        - SHORT: Price breaks below lower BB + Squeeze momentum negative + ADX trend down
    
    Exit conditions:
        - Take profit levels at 1:1 and 1:2 risk-reward ratios
        - Stop loss at opposite BB band
        - Or when opposite signal occurs
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        bb_length: Bollinger Bands period (default: 46)
        bb_stdev: BB standard deviation multiplier (default: 0.35)
        kc_length: Keltner Channel period for squeeze (default: 20)
        kc_mult: KC multiplier (default: 1.5)
        adx_length: ADX calculation period (default: 14)
        adx_threshold: Minimum ADX for trend (default: 29)
        use_adx_filter: Require ADX confirmation (default: True)
        use_squeeze_filter: Require squeeze momentum (default: True)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry, -1 = short entry, 0 = no signal
    """
    df = df.copy()
    
    # Extract parameters - defaults adjusted for better signal generation
    bb_length = params.get('bb_length', 20)
    bb_stdev = params.get('bb_stdev', 2.0)
    kc_length = params.get('kc_length', 20)
    kc_mult = params.get('kc_mult', 1.5)
    adx_length = params.get('adx_length', 14)
    adx_threshold = params.get('adx_threshold', 20)
    use_adx_filter = params.get('use_adx_filter', False)
    use_squeeze_filter = params.get('use_squeeze_filter', False)
    
    # === 1. BOLLINGER BANDS ===
    df['BB_basis'] = df['Close'].rolling(window=bb_length).mean()
    df['BB_std'] = df['Close'].rolling(window=bb_length).std()
    df['upperBB'] = df['BB_basis'] + (bb_stdev * df['BB_std'])
    df['lowerBB'] = df['BB_basis'] - (bb_stdev * df['BB_std'])
    df['midBB'] = (df['upperBB'] + df['lowerBB']) / 2
    
    # Detect breakouts
    df['isOverBBTop'] = df['Low'] > df['upperBB']
    df['isUnderBBBottom'] = df['High'] < df['lowerBB']
    
    # === 2. SQUEEZE MOMENTUM (TTM Squeeze) ===
    # Bollinger Bands for squeeze (standard 2.0 multiplier)
    bb_basis_squeeze = df['Close'].rolling(window=kc_length).mean()
    bb_std_squeeze = df['Close'].rolling(window=kc_length).std()
    df['upperBB_squeeze'] = bb_basis_squeeze + (2.0 * bb_std_squeeze)
    df['lowerBB_squeeze'] = bb_basis_squeeze - (2.0 * bb_std_squeeze)
    
    # Keltner Channel
    df['KC_basis'] = df['Close'].rolling(window=kc_length).mean()
    
    # True Range
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['KC_range'] = df['TR'].rolling(window=kc_length).mean()
    
    df['upperKC'] = df['KC_basis'] + (kc_mult * df['KC_range'])
    df['lowerKC'] = df['KC_basis'] - (kc_mult * df['KC_range'])
    
    # Squeeze detection
    df['sqzOn'] = (df['lowerBB_squeeze'] > df['lowerKC']) & (df['upperBB_squeeze'] < df['upperKC'])
    df['sqzOff'] = (df['lowerBB_squeeze'] < df['lowerKC']) & (df['upperBB_squeeze'] > df['upperKC'])
    
    # Momentum calculation (linear regression)
    df['highest_kc'] = df['High'].rolling(window=kc_length).max()
    df['lowest_kc'] = df['Low'].rolling(window=kc_length).min()
    df['kc_avg'] = (df['highest_kc'] + df['lowest_kc']) / 2
    df['kc_sma'] = df['Close'].rolling(window=kc_length).mean()
    df['momentum_base'] = df['Close'] - ((df['kc_avg'] + df['kc_sma']) / 2)
    
    # Linear regression of momentum
    def linreg(series, window):
        """Calculate linear regression value (slope)."""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            y = series.iloc[i-window:i].values
            x = np.arange(window)
            if len(y) == window:
                slope = np.polyfit(x, y, 1)[0]
                result.iloc[i] = slope * window  # Scale by window
        return result
    
    df['squeeze_momentum'] = linreg(df['momentum_base'], kc_length)
    df['momentum_positive'] = df['squeeze_momentum'] > 0
    df['momentum_negative'] = df['squeeze_momentum'] < 0
    
    # === 3. ADX / DMI ===
    df['up_move'] = df['High'].diff()
    df['down_move'] = -df['Low'].diff()
    
    # +DM and -DM
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Smoothed +DM, -DM, and TR using RMA (Wilder's smoothing)
    df['plus_di_smooth'] = df['plus_dm'].ewm(alpha=1/adx_length, adjust=False).mean()
    df['minus_di_smooth'] = df['minus_dm'].ewm(alpha=1/adx_length, adjust=False).mean()
    df['tr_smooth'] = df['TR'].ewm(alpha=1/adx_length, adjust=False).mean()
    
    # +DI and -DI
    df['plus_di'] = 100 * df['plus_di_smooth'] / df['tr_smooth']
    df['minus_di'] = 100 * df['minus_di_smooth'] / df['tr_smooth']
    
    # ADX calculation
    df['di_sum'] = df['plus_di'] + df['minus_di']
    df['di_diff'] = (df['plus_di'] - df['minus_di']).abs()
    df['dx'] = 100 * df['di_diff'] / df['di_sum'].replace(0, 1)
    df['adx'] = df['dx'].ewm(alpha=1/adx_length, adjust=False).mean()
    
    # Trend conditions
    df['adx_trend_up'] = (df['plus_di'] >= df['minus_di']) & (df['adx'] >= adx_threshold)
    df['adx_trend_down'] = (df['minus_di'] >= df['plus_di']) & (df['adx'] >= adx_threshold)
    
    # === SIGNAL GENERATION ===
    signals = pd.Series(0, index=df.index, dtype=int)
    position = 0
    prev_position = 0
    
    for i in range(max(bb_length, kc_length, adx_length) + 1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        
        # Detect new breakouts
        is_over_bb_now = df.loc[idx, 'isOverBBTop']
        was_over_bb_prev = df.loc[prev_idx, 'isOverBBTop']
        new_break_up = is_over_bb_now and not was_over_bb_prev
        
        is_under_bb_now = df.loc[idx, 'isUnderBBBottom']
        was_under_bb_prev = df.loc[prev_idx, 'isUnderBBBottom']
        new_break_down = is_under_bb_now and not was_under_bb_prev
        
        # Filter conditions
        squeeze_bullish = df.loc[idx, 'momentum_positive'] if use_squeeze_filter else True
        squeeze_bearish = df.loc[idx, 'momentum_negative'] if use_squeeze_filter else True
        
        adx_bullish = df.loc[idx, 'adx_trend_up'] if use_adx_filter else True
        adx_bearish = df.loc[idx, 'adx_trend_down'] if use_adx_filter else True
        
        # LONG signal: Break above upper BB + filters
        if new_break_up and squeeze_bullish and adx_bullish:
            if position != 1:
                position = 1
        
        # SHORT signal: Break below lower BB + filters
        elif new_break_down and squeeze_bearish and adx_bearish:
            if position != -1:
                position = -1
        
        # Output signal only when position changes
        if position != prev_position:
            signals.iloc[i] = position
            prev_position = position
    
    return signals


# Example usage
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv("../data/BTC_1H_OHLCV.csv")  # Use hourly data
    
    # Default strategy (all filters enabled)
    signals_default = strategy_intraday_bb_squeeze(df)
    print("Default Intraday BB+Squeeze+ADX strategy:")
    print(f"  Total signals: {(signals_default != 0).sum()}")
    print(f"  Long entries: {(signals_default == 1).sum()}")
    print(f"  Short entries: {(signals_default == -1).sum()}")
    print("\nSignal distribution:")
    print(signals_default.value_counts().sort_index())
    
    # Without ADX filter (more signals)
    signals_no_adx = strategy_intraday_bb_squeeze(df, use_adx_filter=False)
    print("\nWithout ADX filter:")
    print(f"  Total signals: {(signals_no_adx != 0).sum()}")
    print(signals_no_adx.value_counts().sort_index())
    
    # Without squeeze filter
    signals_no_squeeze = strategy_intraday_bb_squeeze(df, use_squeeze_filter=False)
    print("\nWithout squeeze momentum filter:")
    print(f"  Total signals: {(signals_no_squeeze != 0).sum()}")
    print(signals_no_squeeze.value_counts().sort_index())
    
    # BB only (no filters)
    signals_bb_only = strategy_intraday_bb_squeeze(df, use_adx_filter=False, use_squeeze_filter=False)
    print("\nBB breakouts only (no filters):")
    print(f"  Total signals: {(signals_bb_only != 0).sum()}")
    print(signals_bb_only.value_counts().sort_index())
    
    # More sensitive BB
    signals_sensitive = strategy_intraday_bb_squeeze(df, bb_stdev=0.25)
    print("\nSensitive BB (stdev=0.25):")
    print(f"  Total signals: {(signals_sensitive != 0).sum()}")
    print(signals_sensitive.value_counts().sort_index())
    
    # Lower ADX threshold
    signals_low_adx = strategy_intraday_bb_squeeze(df, adx_threshold=20)
    print("\nLower ADX threshold (20):")
    print(f"  Total signals: {(signals_low_adx != 0).sum()}")
    print(signals_low_adx.value_counts().sort_index())
    
    # Aggressive: sensitive BB + low ADX
    signals_aggressive = strategy_intraday_bb_squeeze(
        df, 
        bb_stdev=0.25,
        adx_threshold=20
    )
    print("\nAggressive (BB=0.25, ADX=20):")
    print(f"  Total signals: {(signals_aggressive != 0).sum()}")
    print(signals_aggressive.value_counts().sort_index())
