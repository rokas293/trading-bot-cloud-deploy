"""
Supervised Local Trial Script
=============================

This script runs the trading bot in a controlled 1-hour trial with:
- Enhanced logging (to console AND file)
- Real-time status updates
- Trial mode (signals logged, orders optional)
- Graceful shutdown
- Summary report generation

Perfect for FYP report: demonstrates proper testing methodology!

Author: FYP Trading Bot Project
Date: February 2026
"""

import sys
import os
import time
import signal
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'live_trading'))

# ==============================================================================
# TRIAL CONFIGURATION
# ==============================================================================

TRIAL_DURATION_MINUTES = 60  # Run for 1 hour
STATUS_UPDATE_INTERVAL = 300  # Status update every 5 minutes
LOG_FILE = 'trial_run_log.txt'
SUMMARY_FILE = 'trial_summary.txt'

# Trial mode: True = log signals without placing orders, False = execute real orders
TRIAL_MODE = os.getenv('TRIAL_MODE', 'true').lower() == 'true'

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

class DualLogger:
    """Logger that writes to both console and file with timestamps."""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.file_handle = None
        
        # Setup file logging
        self._setup_file()
        
        # Track statistics
        self.stats = {
            'data_fetches': 0,
            'signals_generated': 0,
            'signal_breakdown': {'LONG': 0, 'SHORT': 0, 'FLAT': 0},
            'orders_would_place': 0,
            'orders_placed': 0,
            'errors': 0,
            'warnings': 0,
        }
        
    def _setup_file(self):
        """Initialize log file."""
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        self.file_handle.write("=" * 70 + "\n")
        self.file_handle.write("SUPERVISED LOCAL TRIAL - LOG FILE\n")
        self.file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("=" * 70 + "\n\n")
        self.file_handle.flush()
        
    def log(self, message: str, level: str = 'INFO'):
        """Log to both console and file with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"[{timestamp}] {message}"
        
        # Console output
        print(formatted)
        
        # File output
        if self.file_handle:
            self.file_handle.write(f"[{timestamp}] [{level}] {message}\n")
            self.file_handle.flush()
            
    def success(self, message: str):
        """Log success message."""
        self.log(f"✅ {message}", 'SUCCESS')
        
    def warning(self, message: str):
        """Log warning message."""
        self.log(f"⚠️  {message}", 'WARNING')
        self.stats['warnings'] += 1
        
    def error(self, message: str):
        """Log error message."""
        self.log(f"❌ {message}", 'ERROR')
        self.stats['errors'] += 1
        
    def info(self, message: str):
        """Log info message."""
        self.log(f"ℹ️  {message}", 'INFO')
        
    def signal(self, signal_type: str, price: float = None, action: str = None):
        """Log trading signal."""
        self.stats['signals_generated'] += 1
        
        if signal_type in ['LONG', 'SHORT', 'FLAT']:
            self.stats['signal_breakdown'][signal_type] += 1
        
        if price and action:
            self.log(f"🎯 Generated signal: {signal_type} - {action} at ${price:,.2f}", 'SIGNAL')
        else:
            self.log(f"📊 Generated signal: {signal_type}", 'SIGNAL')
            
    def data_fetch(self, bars: int):
        """Log data fetch."""
        self.stats['data_fetches'] += 1
        self.log(f"📊 Fetched {bars} bars of market data", 'DATA')
        
    def order_simulation(self, side: str, quantity: float, price: float):
        """Log simulated order (trial mode)."""
        self.stats['orders_would_place'] += 1
        value = quantity * price
        self.log(f"🔔 Would place order: {side} {quantity} BTC at ${price:,.2f} (${value:,.2f})", 'ORDER_SIM')
        self.log(f"   (Trial mode - order NOT executed)", 'ORDER_SIM')
        
    def order_placed(self, side: str, quantity: float, price: float):
        """Log actual order placed."""
        self.stats['orders_placed'] += 1
        value = quantity * price
        self.log(f"📤 Order placed: {side} {quantity} BTC at ${price:,.2f} (${value:,.2f})", 'ORDER')
        
    def status_update(self, elapsed_minutes: int, total_minutes: int):
        """Log periodic status update."""
        remaining = total_minutes - elapsed_minutes
        self.log(f"📈 Status check: Running normally ({elapsed_minutes} mins elapsed, {remaining} mins remaining)", 'STATUS')
        
    def close(self):
        """Close log file."""
        if self.file_handle:
            self.file_handle.close()

# ==============================================================================
# TRIAL BOT WRAPPER
# ==============================================================================

class TrialBotWrapper:
    """
    Wrapper for exchange bot with trial mode and enhanced logging.
    
    Features:
    - Intercepts and logs all actions
    - Optional order execution (TRIAL_MODE)
    - Time-limited execution
    - Graceful shutdown
    """
    
    def __init__(self, logger: DualLogger, trial_mode: bool = True):
        self.logger = logger
        self.trial_mode = trial_mode
        self.bot = None
        self.running = False
        self.start_time = None
        
        # Import bot class
        try:
            from binance_bot import BinanceFuturesBot
        except ImportError:
            # Try alternative import path
            from live_trading.binance_bot import BinanceFuturesBot
        
        self.BotClass = BinanceFuturesBot
        
    def initialize(self):
        """Initialize the trading bot."""
        load_dotenv()
        
        # Get API credentials
        api_key = (
            os.getenv('BYBIT_TESTNET_KEY')
            or os.getenv('BYBIT_API_KEY')
            or os.getenv('BINANCE_TESTNET_KEY')
            or os.getenv('BINANCE_TESTNET_API_KEY')
        )
        api_secret = (
            os.getenv('BYBIT_TESTNET_SECRET')
            or os.getenv('BYBIT_API_SECRET')
            or os.getenv('BINANCE_TESTNET_SECRET')
            or os.getenv('BINANCE_TESTNET_API_SECRET')
        )
        
        if not api_key or not api_secret:
            self.logger.error("API credentials not found. Set BYBIT_TESTNET_KEY and BYBIT_TESTNET_SECRET as environment variables (or in .env for local dev)")
            return False
        
        try:
            self.bot = self.BotClass(api_key, api_secret)
            self.logger.success("Bot initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {e}")
            return False
            
    def run_single_cycle(self):
        """Run a single trading cycle (fetch data, generate signal, maybe trade)."""
        try:
            # Step 1: Fetch market data
            df = self.bot.fetch_ohlcv_data()
            
            if df is None or len(df) == 0:
                self.logger.warning("No data received")
                return
            
            self.logger.data_fetch(len(df))
            current_price = df['Close'].iloc[-1]
            self.logger.info(f"Latest close price: ${current_price:,.2f}")
            
            # Step 2: Run strategy
            signal = self.bot.run_strategy(df)
            
            # Map signal to text
            signal_map = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}
            signal_text = signal_map.get(signal, 'UNKNOWN')
            
            self.logger.signal(signal_text)
            
            # Step 3: Check if trade needed
            if signal != self.bot.current_position:
                # Signal changed - would place order
                if signal == 1:
                    action = 'ENTER LONG'
                    side = 'BUY'
                elif signal == -1:
                    action = 'ENTER SHORT'
                    side = 'SELL'
                else:
                    action = 'CLOSE POSITION'
                    side = 'CLOSE'
                
                # Calculate position size
                balance = self.bot.get_balance()
                position_value = balance * self.bot.position_size
                quantity = round(position_value / current_price, 3)
                
                if self.trial_mode:
                    # Trial mode - just log
                    self.logger.order_simulation(side, quantity, current_price)
                    self.logger.info("(In trial mode - order not executed)")
                else:
                    # Real mode - execute order
                    self.logger.info(f"Executing: {side} {quantity} BTC")
                    self.bot.place_order(signal)
                    self.logger.order_placed(side, quantity, current_price)
            else:
                self.logger.info(f"No signal change (current: {signal_text})")
                
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            traceback.print_exc()
            
    def run_trial(self, duration_minutes: int, status_interval_seconds: int = 300):
        """
        Run the bot for a specified duration with status updates.
        
        Args:
            duration_minutes: How long to run (in minutes)
            status_interval_seconds: How often to print status updates
        """
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(minutes=duration_minutes)
        self.running = True
        
        # Get check interval from bot (how often to trade)
        check_interval = self.bot.check_interval
        
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log(f"Starting trial run...")
        self.logger.log(f"Duration: {duration_minutes} minutes")
        self.logger.log(f"Strategy: SuperTrend ({self.bot.timeframe} timeframe)")
        self.logger.log(f"Check interval: {check_interval} seconds")
        self.logger.log(f"Trial mode: {'ENABLED (no real orders)' if self.trial_mode else 'DISABLED (real orders!)'}")
        self.logger.log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.log("=" * 60)
        self.logger.log("")
        
        last_status_time = time.time()
        cycle_count = 0
        
        try:
            while self.running and datetime.now() < end_time:
                # Run trading cycle
                cycle_count += 1
                self.logger.log("")
                self.logger.log(f"--- Cycle {cycle_count} ---")
                self.run_single_cycle()
                
                # Calculate time until next check
                elapsed = datetime.now() - self.start_time
                elapsed_minutes = int(elapsed.total_seconds() / 60)
                
                # Status update check
                if time.time() - last_status_time >= status_interval_seconds:
                    self.logger.status_update(elapsed_minutes, duration_minutes)
                    last_status_time = time.time()
                
                # Check if we should continue
                remaining_time = end_time - datetime.now()
                if remaining_time.total_seconds() <= 0:
                    break
                
                # Wait for next cycle (but not longer than remaining time)
                wait_time = min(check_interval, remaining_time.total_seconds())
                
                if wait_time > 0:
                    self.logger.info(f"Next check in {int(wait_time)} seconds...")
                    
                    # Sleep in small chunks to allow graceful shutdown
                    sleep_chunk = 10
                    slept = 0
                    while slept < wait_time and self.running:
                        time.sleep(min(sleep_chunk, wait_time - slept))
                        slept += sleep_chunk
                        
        except Exception as e:
            self.logger.error(f"Unexpected error during trial: {e}")
            traceback.print_exc()
            
        finally:
            self.running = False
            
    def stop(self):
        """Signal the trial to stop."""
        self.logger.log("")
        self.logger.log("🛑 Stop signal received - shutting down gracefully...")
        self.running = False
        
    def get_elapsed_time(self):
        """Get elapsed time since start."""
        if self.start_time:
            return datetime.now() - self.start_time
        return timedelta(0)

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

def generate_summary(logger: DualLogger, duration: timedelta, success: bool):
    """Generate and save trial summary report."""
    
    stats = logger.stats
    
    # Build summary text
    summary_lines = [
        "",
        "=" * 70,
        "                      TRIAL SUMMARY REPORT",
        "=" * 70,
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration: {int(duration.total_seconds() // 60)} minutes {int(duration.total_seconds() % 60)} seconds",
        "",
        "--- Trading Activity ---",
        f"Data fetches:      {stats['data_fetches']}",
        f"Signals generated: {stats['signals_generated']}",
        f"  - LONG signals:  {stats['signal_breakdown']['LONG']}",
        f"  - SHORT signals: {stats['signal_breakdown']['SHORT']}",
        f"  - FLAT signals:  {stats['signal_breakdown']['FLAT']}",
        "",
        "--- Orders ---",
        f"Orders would place: {stats['orders_would_place']}",
        f"Orders executed:    {stats['orders_placed']}",
        "",
        "--- Health ---",
        f"Warnings: {stats['warnings']}",
        f"Errors:   {stats['errors']}",
        "",
        "--- Result ---",
        f"Status: {'✅ SUCCESSFUL' if success else '❌ FAILED'}",
        "",
        "=" * 70,
    ]
    
    # Print to console
    for line in summary_lines:
        print(line)
    
    # Save to file
    summary_path = Path(SUMMARY_FILE)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
        f.write('\n')
        
    print(f"\nDetailed logs saved to: {LOG_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print()
    
    # Next steps
    if success:
        print("Next step: Deploy to Oracle Cloud")
        print("See: live_trading/README.md for deployment instructions")
    else:
        print("Please review errors in the log file and fix issues before deployment.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point for supervised trial."""
    
    # Print header
    print()
    print("=" * 70)
    print("            SUPERVISED LOCAL TRIAL - 1 HOUR TEST")
    print("=" * 70)
    print()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {TRIAL_DURATION_MINUTES} minutes")
    print(f"Trial mode: {'ENABLED' if TRIAL_MODE else 'DISABLED (real orders!)'}")
    print(f"Logging to: {LOG_FILE}")
    print()
    
    # Initialize logger
    logger = DualLogger(LOG_FILE)
    
    # Initialize wrapper
    wrapper = TrialBotWrapper(logger, trial_mode=TRIAL_MODE)
    
    # Setup signal handler for Ctrl+C
    def signal_handler(signum, frame):
        logger.log("")
        logger.log("⏰ Received interrupt signal")
        wrapper.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize bot
    if not wrapper.initialize():
        logger.error("Failed to initialize bot - aborting")
        generate_summary(logger, timedelta(0), False)
        logger.close()
        return 1
    
    # Run trial
    logger.log("Starting bot...")
    
    try:
        wrapper.run_trial(
            duration_minutes=TRIAL_DURATION_MINUTES,
            status_interval_seconds=STATUS_UPDATE_INTERVAL
        )
        
        elapsed = wrapper.get_elapsed_time()
        
        logger.log("")
        logger.log(f"⏰ Trial complete - ran for {int(elapsed.total_seconds() // 60)} minutes")
        
        success = logger.stats['errors'] == 0
        
    except Exception as e:
        logger.error(f"Trial failed with exception: {e}")
        traceback.print_exc()
        elapsed = wrapper.get_elapsed_time()
        success = False
    
    finally:
        logger.close()
    
    # Generate summary
    generate_summary(logger, elapsed, success)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
