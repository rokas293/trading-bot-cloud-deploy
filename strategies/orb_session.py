import pandas as pd
import numpy as np
from datetime import datetime, time

def strategy_orb_sessions(df: pd.DataFrame, **params) -> pd.Series:
    """
    Opening Range Breakout + Key Session Levels Strategy
    
    Trades breakouts from Opening Range (ORB) and major session highs/lows (Asian, London, NY).
    Combines price action with time-based session analysis for intraday trading.
    
    Entry conditions:
        - LONG: Break above ORB high, Asian high, London high, or NY session high
        - SHORT: Break below ORB low, Asian low, London low, or NY session low
        - Optional: Trade only during specific sessions or with HTF bias confirmation
    
    Exit conditions:
        - Exit on opposite breakout signal
        - Optional: Exit at end of trading day or session
    
    Args:
        df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
        orb_start_hour: ORB start hour in 24h format (default: 9 for 9am ET)
        orb_start_min: ORB start minute (default: 30)
        orb_duration_min: ORB duration in minutes (default: 15)
        trade_orb: Trade ORB breakouts (default: True)
        trade_asian: Trade Asian session breakouts (default: True)
        trade_london: Trade London session breakouts (default: True)
        trade_ny: Trade NY session breakouts (default: False)
        asian_start_hour: Asian session start hour (default: 20 for 8pm ET)
        asian_end_hour: Asian session end hour (default: 0 for midnight ET)
        london_start_hour: London session start hour (default: 2 for 2am ET)
        london_end_hour: London session end hour (default: 5 for 5am ET)
        ny_start_hour: NY session start hour (default: 9 for 9am ET)
        ny_end_hour: NY session end hour (default: 12 for noon ET)
        require_htf_bias: Require higher timeframe bias alignment (default: False)
        swing_lookback: Swing lookback for HTF bias detection (default: 10)
        bias_recalc_bars: Recalculate HTF bias every N bars (default: 20)
    
    Returns:
        pd.Series of int with same index as df:
        1 = long entry/hold, -1 = short entry/hold, 0 = flat/exit
    """
    df = df.copy()
    
    # Extract parameters
    orb_start_hour = params.get('orb_start_hour', 9)
    orb_start_min = params.get('orb_start_min', 30)
    orb_duration_min = params.get('orb_duration_min', 15)
    trade_orb = params.get('trade_orb', True)
    trade_asian = params.get('trade_asian', True)
    trade_london = params.get('trade_london', True)
    trade_ny = params.get('trade_ny', False)
    
    asian_start_hour = params.get('asian_start_hour', 20)
    asian_end_hour = params.get('asian_end_hour', 0)
    london_start_hour = params.get('london_start_hour', 2)
    london_end_hour = params.get('london_end_hour', 5)
    ny_start_hour = params.get('ny_start_hour', 9)
    ny_end_hour = params.get('ny_end_hour', 12)
    
    require_htf_bias = params.get('require_htf_bias', False)
    swing_lookback = params.get('swing_lookback', 10)
    bias_recalc_bars = params.get('bias_recalc_bars', 20)
    
    # Ensure timestamp column is datetime (handle both 'Timestamp' and 'timestamp')
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert the index to datetime
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("DataFrame must have datetime index or 'Timestamp'/'timestamp' column")
    
    # === Helper: Detect swing structure for HTF bias ===
    def detect_bias(high_series, low_series, lookback):
        """Detect market structure bias (bullish/bearish/neutral)."""
        swing_highs = []
        swing_lows = []
        
        if len(high_series) < lookback * 2 + 1:
            return 0  # Not enough data
        
        for i in range(lookback, len(high_series) - lookback):
            # Check if current bar is swing high
            window_high_left = high_series.iloc[i - lookback:i]
            window_high_right = high_series.iloc[i + 1:i + lookback + 1]
            current_high = high_series.iloc[i]
            
            if current_high > window_high_left.max() and current_high > window_high_right.max():
                swing_highs.append(current_high)
            
            # Check if current bar is swing low
            window_low_left = low_series.iloc[i - lookback:i]
            window_low_right = low_series.iloc[i + 1:i + lookback + 1]
            current_low = low_series.iloc[i]
            
            if current_low < window_low_left.min() and current_low < window_low_right.min():
                swing_lows.append(current_low)
        
        # Determine bias from recent swing structure
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            higher_high = swing_highs[-1] > swing_highs[-2]
            higher_low = swing_lows[-1] > swing_lows[-2]
            lower_high = swing_highs[-1] < swing_highs[-2]
            lower_low = swing_lows[-1] < swing_lows[-2]
            
            if higher_high and higher_low:
                return 1  # Bullish
            elif lower_high and lower_low:
                return -1  # Bearish
        
        return 0  # Neutral
    
    # Calculate ORB end time and check for midnight wrap
    orb_end_hour = orb_start_hour
    orb_end_min = orb_start_min + orb_duration_min
    if orb_end_min >= 60:
        orb_end_hour += orb_end_min // 60
        orb_end_min = orb_end_min % 60
    
    # Handle midnight wrap
    if orb_end_hour >= 24:
        orb_end_hour = orb_end_hour % 24
    
    orb_wraps_midnight = orb_end_hour < orb_start_hour or (orb_end_hour == orb_start_hour and orb_end_min <= orb_start_min)
    
    # Check if sessions wrap around midnight
    asian_wraps_midnight = asian_end_hour < asian_start_hour
    london_wraps_midnight = london_end_hour < london_start_hour
    ny_wraps_midnight = ny_end_hour < ny_start_hour
    
    # === Signal generation (stateful loop) ===
    signals = pd.Series(0, index=df.index, dtype=int)
    position = 0
    prev_position = 0
    
    # Track session levels
    orb_high = None
    orb_low = None
    orb_complete = False
    
    asian_high = None
    asian_low = None
    asian_complete = False
    
    london_high = None
    london_low = None
    london_complete = False
    
    ny_high = None
    ny_low = None
    ny_complete = False
    
    current_date = None
    
    # HTF bias tracking
    htf_bias = 0
    last_bias_calc = -bias_recalc_bars - 1
    
    for i in range(len(df)):
        idx = df.index[i]
        current_hour = idx.hour
        current_minute = idx.minute
        current_day = idx.date()
        
        high_val = df['High'].iloc[i]
        low_val = df['Low'].iloc[i]
        close_val = df['Close'].iloc[i]
        
        # === Calculate HTF Bias Dynamically ===
        if require_htf_bias and (i - last_bias_calc) >= bias_recalc_bars:
            if i >= swing_lookback * 2 + 10:
                # Look back at recent bars for bias calculation
                lookback_window = min(100, i)  # Use up to last 100 bars
                recent_highs = df['High'].iloc[max(0, i - lookback_window):i]
                recent_lows = df['Low'].iloc[max(0, i - lookback_window):i]
                htf_bias = detect_bias(recent_highs, recent_lows, swing_lookback)
                last_bias_calc = i
        
        # Reset on new day
        if current_date != current_day:
            current_date = current_day
            orb_high = None
            orb_low = None
            orb_complete = False
            asian_high = None
            asian_low = None
            asian_complete = False
            london_high = None
            london_low = None
            london_complete = False
            ny_high = None
            ny_low = None
            ny_complete = False
        
        # === ORB Tracking (with midnight wrap handling) ===
        if orb_wraps_midnight:
            # Session spans midnight (e.g., 23:45 to 00:15)
            in_orb_session = (
                (current_hour > orb_start_hour or 
                 (current_hour == orb_start_hour and current_minute >= orb_start_min)) or
                (current_hour < orb_end_hour or
                 (current_hour == orb_end_hour and current_minute < orb_end_min))
            )
        else:
            # Normal session (doesn't cross midnight)
            in_orb_session = (
                (current_hour > orb_start_hour or 
                 (current_hour == orb_start_hour and current_minute >= orb_start_min)) and
                (current_hour < orb_end_hour or
                 (current_hour == orb_end_hour and current_minute < orb_end_min))
            )
        
        if in_orb_session and not orb_complete:
            if orb_high is None:
                orb_high = high_val
                orb_low = low_val
            else:
                orb_high = max(orb_high, high_val)
                orb_low = min(orb_low, low_val)
        elif not in_orb_session and orb_high is not None and not orb_complete:
            orb_complete = True
        
        # === Asian Session Tracking (with midnight wrap handling) ===
        if asian_wraps_midnight:
            in_asian_session = current_hour >= asian_start_hour or current_hour < asian_end_hour
        else:
            in_asian_session = asian_start_hour <= current_hour < asian_end_hour
        
        if in_asian_session and not asian_complete:
            if asian_high is None:
                asian_high = high_val
                asian_low = low_val
            else:
                asian_high = max(asian_high, high_val)
                asian_low = min(asian_low, low_val)
        elif not in_asian_session and asian_high is not None and not asian_complete:
            asian_complete = True
        
        # === London Session Tracking (with midnight wrap handling) ===
        if london_wraps_midnight:
            in_london_session = current_hour >= london_start_hour or current_hour < london_end_hour
        else:
            in_london_session = london_start_hour <= current_hour < london_end_hour
        
        if in_london_session and not london_complete:
            if london_high is None:
                london_high = high_val
                london_low = low_val
            else:
                london_high = max(london_high, high_val)
                london_low = min(london_low, low_val)
        elif not in_london_session and london_high is not None and not london_complete:
            london_complete = True
        
        # === NY Session Tracking (with midnight wrap handling) ===
        if ny_wraps_midnight:
            in_ny_session = current_hour >= ny_start_hour or current_hour < ny_end_hour
        else:
            in_ny_session = ny_start_hour <= current_hour < ny_end_hour
        
        if in_ny_session and not ny_complete:
            if ny_high is None:
                ny_high = high_val
                ny_low = low_val
            else:
                ny_high = max(ny_high, high_val)
                ny_low = min(ny_low, low_val)
        elif not in_ny_session and ny_high is not None and not ny_complete:
            ny_complete = True
        
        # === Generate Trading Signals ===
        
        # Check HTF bias filter (uses dynamically calculated bias)
        bias_allows_long = not require_htf_bias or htf_bias >= 0
        bias_allows_short = not require_htf_bias or htf_bias <= 0
        
        # ORB Breakouts
        if trade_orb and orb_complete and orb_high is not None:
            if close_val > orb_high and position != 1 and bias_allows_long:
                position = 1
            elif close_val < orb_low and position != -1 and bias_allows_short:
                position = -1
        
        # Asian Session Breakouts
        if trade_asian and asian_complete and asian_high is not None:
            if close_val > asian_high and position != 1 and bias_allows_long:
                position = 1
            elif close_val < asian_low and position != -1 and bias_allows_short:
                position = -1
        
        # London Session Breakouts
        if trade_london and london_complete and london_high is not None:
            if close_val > london_high and position != 1 and bias_allows_long:
                position = 1
            elif close_val < london_low and position != -1 and bias_allows_short:
                position = -1
        
        # NY Session Breakouts
        if trade_ny and ny_complete and ny_high is not None:
            if close_val > ny_high and position != 1 and bias_allows_long:
                position = 1
            elif close_val < ny_low and position != -1 and bias_allows_short:
                position = -1
        
        # Output signal only when position changes
        if position != prev_position:
            signals.iloc[i] = position
            prev_position = position
    
    return signals


# Example usage
if __name__ == "__main__":
    # ====================================================================
    # INTRADAY DATA TESTING (Required for this strategy)
    # ====================================================================
    print("="*70)
    print("ORB + SESSION LEVELS STRATEGY")
    print("="*70)
    
    # This strategy REQUIRES intraday data (5min, 15min, or 1hour)
    df = pd.read_csv("../data/BTC_15M_OHLCV.csv")
    # Ensure timestamp is parsed as datetime
    if 'Timestamp' in df.columns:
        df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\nDataset: {len(df)} bars (15-minute timeframe)\n")
    
    # 1. ORB Only (Classic Opening Range Breakout)
    signals_orb = strategy_orb_sessions(
        df,
        orb_start_hour=9,
        orb_start_min=30,
        orb_duration_min=15,
        trade_orb=True,
        trade_asian=False,
        trade_london=False,
        trade_ny=False
    )
    print("1. ORB Only (9:30-9:45 ET):")
    print(f"   Total signals: {(signals_orb != 0).sum()}")
    print(f"   Long: {(signals_orb == 1).sum()} | Short: {(signals_orb == -1).sum()}")
    print(signals_orb.value_counts().sort_index())
    
    # 2. London Session Only
    signals_london = strategy_orb_sessions(
        df,
        trade_orb=False,
        trade_asian=False,
        trade_london=True,
        trade_ny=False,
        london_start_hour=2,
        london_end_hour=5
    )
    print("\n2. London Session (2am-5am ET Kill Zone):")
    print(f"   Total signals: {(signals_london != 0).sum()}")
    print(f"   Long: {(signals_london == 1).sum()} | Short: {(signals_london == -1).sum()}")
    print(signals_london.value_counts().sort_index())
    
    # 3. Asian + London Combined
    signals_asia_london = strategy_orb_sessions(
        df,
        trade_orb=False,
        trade_asian=True,
        trade_london=True,
        trade_ny=False,
        asian_start_hour=20,
        asian_end_hour=0,
        london_start_hour=2,
        london_end_hour=5
    )
    print("\n3. Asian + London Sessions:")
    print(f"   Total signals: {(signals_asia_london != 0).sum()}")
    print(f"   Long: {(signals_asia_london == 1).sum()} | Short: {(signals_asia_london == -1).sum()}")
    print(signals_asia_london.value_counts().sort_index())
    
    # 4. All Sessions Combined
    signals_all = strategy_orb_sessions(
        df,
        trade_orb=True,
        trade_asian=True,
        trade_london=True,
        trade_ny=True,
        orb_start_hour=9,
        orb_start_min=30,
        orb_duration_min=30
    )
    print("\n4. All Sessions (ORB + Asian + London + NY):")
    print(f"   Total signals: {(signals_all != 0).sum()}")
    print(f"   Long: {(signals_all == 1).sum()} | Short: {(signals_all == -1).sum()}")
    print(signals_all.value_counts().sort_index())
    
    # 5. With HTF Bias Filter (Dynamic Recalculation)
    signals_htf = strategy_orb_sessions(
        df,
        trade_orb=True,
        trade_london=True,
        require_htf_bias=True,
        swing_lookback=10,
        bias_recalc_bars=20  # Recalculate bias every 20 bars
    )
    print("\n5. ORB + London with Dynamic HTF Bias Filter:")
    print(f"   Total signals: {(signals_htf != 0).sum()}")
    print(f"   Long: {(signals_htf == 1).sum()} | Short: {(signals_htf == -1).sum()}")
    print(signals_htf.value_counts().sort_index())
    
    # 6. Edge Case: ORB that wraps midnight (for crypto/24h markets)
    signals_midnight_orb = strategy_orb_sessions(
        df,
        orb_start_hour=23,
        orb_start_min=45,
        orb_duration_min=30,  # 23:45 to 00:15
        trade_orb=True,
        trade_asian=False,
        trade_london=False,
        trade_ny=False
    )
    print("\n6. Midnight-Wrapping ORB (23:45-00:15):")
    print(f"   Total signals: {(signals_midnight_orb != 0).sum()}")
    print(f"   Long: {(signals_midnight_orb == 1).sum()} | Short: {(signals_midnight_orb == -1).sum()}")
    print(signals_midnight_orb.value_counts().sort_index())
    
    print("\n" + "="*70)
    print("NOTE: This strategy requires INTRADAY data (5m, 15m, or 1h)")
    print("Session times are in ET (Eastern Time)")
    print("HTF bias now recalculates dynamically every 20 bars")
    print("="*70)