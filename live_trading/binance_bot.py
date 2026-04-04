"""
Bybit Linear Futures Testnet Trading Bot

This bot connects to Bybit testnet, fetches live BTC/USDT data,
runs the SuperTrend strategy, and executes trades automatically.

TESTNET ONLY - No real money at risk!
"""

import ccxt
import pandas as pd
import time
import sys
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import os

# Add project root to path to import our strategy
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.SuperTrend_Strategy import strategy_supertrend


class BybitFuturesBot:
    """
    Simple automated trading bot for Bybit linear futures testnet.
    
    Features:
    - Connects to Bybit testnet (paper trading)
    - Fetches live 1-hour BTC/USDT OHLCV data
    - Runs SuperTrend strategy
    - Executes trades based on signals
    - Uses 1% of balance per trade
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the bot with API credentials.
        
        Args:
            api_key: Bybit testnet API key
            api_secret: Bybit testnet API secret
        """
        print("=" * 70)
        print("🤖 BYBIT LINEAR FUTURES TESTNET BOT")
        print("=" * 70)
        
        # Runtime configuration from env (Railway-friendly defaults)
        self.testnet = os.getenv('TESTNET', 'true').strip().lower() == 'true'
        self.bybit_demo_mode = os.getenv('BYBIT_DEMO_MODE', 'false').strip().lower() == 'true'
        if self.testnet and self.bybit_demo_mode:
            # Bybit demo mode uses api-demo and should not be combined with sandbox mode.
            print("⚠️  TESTNET=true and BYBIT_DEMO_MODE=true are incompatible. Forcing TESTNET=false.")
            self.testnet = False

        raw_symbol = os.getenv('SYMBOL', 'BTCUSDT').strip().upper()
        if '/' in raw_symbol:
            # If no settle suffix provided, default to USDT-settled linear perp.
            self.symbol = raw_symbol if ':' in raw_symbol else f"{raw_symbol}:USDT"
        elif len(raw_symbol) >= 6 and raw_symbol.endswith('USDT'):
            self.symbol = f"{raw_symbol[:-4]}/USDT:USDT"
        else:
            self.symbol = 'BTC/USDT:USDT'

        try:
            self.position_size = float(os.getenv('RISK_PER_TRADE', '0.01'))
        except ValueError:
            self.position_size = 0.01

        self.bybit_account_type = os.getenv('BYBIT_ACCOUNT_TYPE', 'UNIFIED').strip().upper()
        self.balance_debug = os.getenv('BALANCE_DEBUG', 'false').strip().lower() == 'true'

        # Price source for data/ticker on Bybit: index is the most stable on testnet.
        self.price_source = os.getenv('PRICE_SOURCE', 'index').strip().lower()

        # Initialize CCXT exchange object for Bybit linear futures.
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # Respect API rate limits
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
                'recvWindow': 10000,
            }
        })
        # Bybit mode selection:
        # - TESTNET=true  => testnet.bybit.com (sandbox)
        # - BYBIT_DEMO_MODE=true => mainnet base URL with demo-trading keys
        if self.testnet and not self.bybit_demo_mode:
            self.exchange.set_sandbox_mode(True)
        elif self.bybit_demo_mode:
            self.exchange.enable_demo_trading(True)

        # Public client for market data only (no keys) so data fetches still
        # work when credentials are invalid or permissions are missing.
        self.public_exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
            }
        })
        if self.testnet and not self.bybit_demo_mode:
            self.public_exchange.set_sandbox_mode(True)
        elif self.bybit_demo_mode:
            self.public_exchange.enable_demo_trading(True)

        if self.bybit_demo_mode:
            print("Connected to Bybit demo trading mode ✓")
            print("📡 Using Bybit mainnet URL with demo API keys")
        elif self.testnet:
            print("Connected to Bybit testnet ✓")
            print("📡 Using Bybit linear futures sandbox mode")
        else:
            print("⚠️  TESTNET=false: using Bybit live environment")
        
        # =====================================================================
        # ⚙️ TRADING CONFIGURATION - EASY TO CHANGE!
        # =====================================================================
        
        # Choose your timeframe: '5m' or '1h'
        # 5-minute = faster signals, more trades
        # 1-hour = slower signals, fewer trades
        self.timeframe = '1h'  # ← CHANGE THIS: '5m' or '1h'
        
        # Trading parameters
        self.bars_to_fetch = 100   # Fetch last 100 bars for strategy
        
        # Strategy parameters - OPTIMIZED FOR EACH TIMEFRAME
        # After optimization, update these values:
        
        if self.timeframe == '5m':
            # 5-MINUTE OPTIMIZED PARAMETERS
            # Run: python examples/optimize_supertrend_5m.py
            # Then copy the best parameters here:
            self.strategy_params = {
                'atr_period': 10,       # ← Update after optimization
                'atr_multiplier': 2.0,  # ← Update after optimization
            }
            self.check_interval = 300  # Check every 5 minutes (300 seconds)
        
        elif self.timeframe == '1h':
            # 1-HOUR OPTIMIZED PARAMETERS
            # Optimized on BTC_1H_OHLCV grid: period=7, multiplier=2.5
            self.strategy_params = {
                'atr_period': 7,
                'atr_multiplier': 2.5
            }
            self.check_interval = 60  # Check every 60 seconds
        
        else:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}. Use '5m' or '1h'")
        
        # ====================================================================
        
        # Track current position
        self.current_position = 0  # 0 = flat, 1 = long, -1 = short
        
        print(f"📊 Symbol: {self.symbol}")
        print(f"⏰ Timeframe: {self.timeframe}")
        print(f"🧮 Price source: {self.price_source}")
        print(f"🏦 Bybit account type: {self.bybit_account_type}")
        print(f"🎛️  Bybit demo mode: {self.bybit_demo_mode}")
        print(f"� Strategy: SuperTrend (period={self.strategy_params['atr_period']}, mult={self.strategy_params['atr_multiplier']})")
        print(f"⏱️  Check interval: {self.check_interval} seconds")
        print(f"�💰 Position size: {self.position_size * 100}% of balance")
        print()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert value to float safely, returning None on failure."""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_usdt_balance(self, balance: Dict[str, Any]) -> float:
        """Extract free/available USDT from multiple Bybit response shapes."""
        free_map = balance.get('free') if isinstance(balance, dict) else None
        if isinstance(free_map, dict):
            v = self._safe_float(free_map.get('USDT'))
            if v is not None:
                return max(v, 0.0)

        usdt = balance.get('USDT') if isinstance(balance, dict) else None
        if isinstance(usdt, dict):
            for key in ('free', 'available', 'used', 'total'):
                v = self._safe_float(usdt.get(key))
                if v is not None and (key in ('free', 'available', 'total')):
                    return max(v, 0.0)

        info = balance.get('info') if isinstance(balance, dict) else None
        result = info.get('result') if isinstance(info, dict) else None
        lists = result.get('list') if isinstance(result, dict) else None

        if isinstance(lists, list):
            for acct in lists:
                coins = acct.get('coin') if isinstance(acct, dict) else None
                if not isinstance(coins, list):
                    continue
                for coin in coins:
                    if str(coin.get('coin', '')).upper() != 'USDT':
                        continue
                    for key in ('availableToWithdraw', 'availableBalance', 'walletBalance', 'equity'):
                        v = self._safe_float(coin.get(key))
                        if v is not None:
                            return max(v, 0.0)

        return 0.0

    def _fetch_current_price(self) -> float:
        """Fetch current reference price using configured source."""
        ticker = self.public_exchange.fetch_ticker(self.symbol, params={'category': 'linear'})
        info = ticker.get('info', {}) if isinstance(ticker, dict) else {}

        if self.price_source == 'index':
            p = self._safe_float(info.get('indexPrice'))
            if p is not None and p > 0:
                return p
        elif self.price_source == 'mark':
            p = self._safe_float(info.get('markPrice'))
            if p is not None and p > 0:
                return p

        p = self._safe_float(ticker.get('last'))
        return p if p is not None else 0.0
    
    def fetch_ohlcv_data(self):
        """
        Fetch recent OHLCV (candlestick) data from Bybit.
        
        Returns:
            pd.DataFrame with columns: Timestamp, Open, High, Low, Close, Volume
        """
        # Add simple retry logic to tolerate transient network/testnet errors
        max_attempts = 3
        backoff_secs = [1, 2, 4]

        for attempt in range(1, max_attempts + 1):
            try:
                # Fetch OHLCV data from exchange
                ohlcv = self.public_exchange.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    limit=self.bars_to_fetch,
                    params={
                        'category': 'linear',
                        'price': self.price_source,
                    }
                )

                # Convert to DataFrame with proper column names
                df = pd.DataFrame(
                    ohlcv,
                    columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
                )

                # Convert timestamp to datetime
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

                return df

            except Exception as e:
                print(f"❌ Error fetching data (attempt {attempt}/{max_attempts}): {e}")
                # If not last attempt, back off and retry
                if attempt < max_attempts:
                    sleep_for = backoff_secs[min(attempt - 1, len(backoff_secs) - 1)]
                    print(f"   Retrying in {sleep_for} seconds...")
                    time.sleep(sleep_for)
                    continue
                else:
                    print("   All attempts failed. Will return None and continue.")
                    return None
    
    def get_balance(self):
        """
        Get current USDT balance from account.
        
        Returns:
            float: Available USDT balance
        """
        preferred = {'accountType': self.bybit_account_type, 'coin': 'USDT'} if self.bybit_account_type else {'coin': 'USDT'}
        param_attempts = [
            preferred,
            {'accountType': 'UNIFIED', 'coin': 'USDT'},
            {'accountType': 'CONTRACT', 'coin': 'USDT'},
            {'type': 'swap', 'coin': 'USDT'},
            {'type': 'linear', 'coin': 'USDT'},
            {},
        ]

        last_error = None
        for params in param_attempts:
            try:
                balance = self.exchange.fetch_balance(params)
                usdt_balance = self._extract_usdt_balance(balance)
                if self.balance_debug:
                    print(f"🧪 Balance attempt {params}: USDT free={usdt_balance}")
                return usdt_balance
            except Exception as e:
                last_error = e
                if self.balance_debug:
                    print(f"🧪 Balance attempt failed {params}: {e}")

        if last_error is not None:
            print(f"❌ Error fetching balance: {last_error}")
        return 0.0
    
    def get_current_position(self):
        """
        Check if we have an open position in the configured symbol.
        
        Returns:
            float: Position size (positive = long, negative = short, 0 = flat)
        """
        try:
            positions = self.exchange.fetch_positions([self.symbol], params={'category': 'linear'})
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    contracts = float(pos.get('contracts') or 0)
                    return contracts
            return 0
        except Exception as e:
            print(f"❌ Error fetching position: {e}")
            return 0
    
    def close_position(self):
        """Close any open position."""
        try:
            position_size = self.get_current_position()
            
            if position_size == 0:
                print("ℹ️  No position to close")
                return True
            
            # Determine side for closing
            side = 'sell' if position_size > 0 else 'buy'
            amount = abs(position_size)

            # Never submit dust/zero close orders.
            market = self.exchange.market(self.symbol)
            min_amount = (market.get('limits', {})
                                .get('amount', {})
                                .get('min') or 0.0)
            if amount <= 0:
                print("ℹ️  Close skipped: position size is zero")
                self.current_position = 0
                return True
            if min_amount and amount < min_amount:
                print(
                    f"ℹ️  Close skipped: amount {amount} is below min amount {min_amount}"
                )
                self.current_position = 0
                return True
            
            print(f"🔄 Closing position: {side} {amount} {self.symbol}")
            
            order = self.exchange.create_market_order(
                self.symbol,
                side,
                amount,
                params={'reduceOnly': True, 'category': 'linear'}
            )
            
            print(f"✅ Position closed: {order['id']}")
            self.current_position = 0
            return True
            
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False
    
    def place_order(self, signal):
        """
        Place a market order based on strategy signal.
        
        Args:
            signal: 1 = go long, -1 = go short, 0 = close position
        """
        try:
            # If signal is 0, close any open position
            if signal == 0:
                if self.current_position != 0:
                    print("📉 Signal = 0: Closing position")
                    self.close_position()
                return
            
            # Check if we already have the desired position
            if signal == self.current_position:
                print(f"ℹ️  Already in desired position ({signal})")
                return
            
            # Get current balance
            balance = self.get_balance()
            print(f"💰 Current balance: {balance:.2f} USDT")
            
            # Get current BTC price
            current_price = self._fetch_current_price()
            print(f"💵 Current BTC price: ${current_price:.2f}")
            
            # Calculate position size (1% of balance)
            position_value = balance * self.position_size
            if position_value <= 0:
                print(
                    f"ℹ️  Order skipped: position value is {position_value:.2f} USDT"
                )
                return

            if current_price <= 0:
                print("ℹ️  Order skipped: invalid current price")
                return

            amount = position_value / current_price

            # Use exchange precision for the active market symbol.
            amount = float(self.exchange.amount_to_precision(self.symbol, amount))

            market = self.exchange.market(self.symbol)
            min_amount = (market.get('limits', {})
                                .get('amount', {})
                                .get('min') or 0.0)
            
            print(f"📊 Position value: {position_value:.2f} USDT")
            print(f"📦 Amount: {amount} BTC")

            if amount <= 0:
                print("ℹ️  Order skipped: computed amount is zero after precision rounding")
                return

            if min_amount and amount < min_amount:
                print(
                    f"ℹ️  Order skipped: computed amount {amount} is below min amount {min_amount}"
                )
                return
            
            # Close existing position first if we have one
            if self.current_position != 0:
                print("🔄 Closing existing position first...")
                self.close_position()
                time.sleep(1)  # Wait a moment
            
            # Determine order side
            side = 'buy' if signal == 1 else 'sell'
            
            print(f"📤 Placing {side.upper()} order for {amount} {self.symbol}")
            
            # Place market order
            order = self.exchange.create_market_order(
                self.symbol,
                side,
                amount,
                params={'category': 'linear'}
            )
            
            print(f"✅ Order executed!")
            print(f"   Order ID: {order['id']}")
            print(f"   Type: {order['type']}")
            print(f"   Side: {order['side']}")
            print(f"   Amount: {order['amount']}")
            
            # Update current position
            self.current_position = signal
            
        except Exception as e:
            print(f"❌ Error placing order: {e}")
    
    def run_strategy(self, df):
        """
        Run SuperTrend strategy on the data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            int: Latest signal (1, -1, or 0)
        """
        try:
            # Run strategy with optimized parameters
            signals = strategy_supertrend(
                df,
                period=self.strategy_params['atr_period'],
                multiplier=self.strategy_params['atr_multiplier']
            )
            
            # Get the most recent signal
            latest_signal = signals.iloc[-1]
            
            return latest_signal
            
        except Exception as e:
            print(f"❌ Error running strategy: {e}")
            return 0
    
    def run(self, check_interval=None):
        """
        Main bot loop - continuously check for signals and trade.
        
        Args:
            check_interval: Seconds between checks (default: uses self.check_interval)
        """
        # Use configured check_interval if not specified
        if check_interval is None:
            check_interval = self.check_interval
        print("🚀 Starting bot...")
        print("SuperTrend monitoring live...")
        print(f"⏰ Checking for signals every {check_interval} seconds")
        print("Press Ctrl+C to stop")
        print("=" * 70)
        print()
        
        try:
            while True:
                print(f"\n{'=' * 70}")
                print(f"🔍 Checking for trading signals... [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
                print(f"{'=' * 70}")
                
                # Step 1: Fetch latest market data
                print("\n📊 Fetching market data...")
                df = self.fetch_ohlcv_data()
                
                if df is None or len(df) == 0:
                    print("⚠️  No data received, skipping this cycle")
                    time.sleep(check_interval)
                    continue
                
                print(f"✅ Fetched {len(df)} bars")
                print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")

                # Always print account snapshot each cycle for monitoring.
                cycle_balance = self.get_balance()
                print(f"💰 Balance snapshot: {cycle_balance:.2f} USDT (free)")
                
                # Step 2: Run strategy
                print("\n🧠 Running SuperTrend strategy...")
                signal = self.run_strategy(df)
                
                signal_text = {1: "LONG 🟢", -1: "SHORT 🔴", 0: "FLAT ⚪"}
                print(f"📊 Signal: {signal_text.get(signal, 'UNKNOWN')}")
                
                # Step 3: Execute trade if signal changed
                if signal != self.current_position:
                    print(f"\n🎯 New signal detected! Current: {self.current_position} → New: {signal}")
                    latest_price = float(df['Close'].iloc[-1])
                    if signal == 1:
                        print(f"LONG signal detected @ ${latest_price:,.0f}")
                    elif signal == -1:
                        print(f"SHORT signal detected @ ${latest_price:,.0f}")
                    self.place_order(signal)
                else:
                    print(f"ℹ️  No change in signal (current position: {signal_text.get(self.current_position, 'UNKNOWN')})")
                
                # Wait before next check
                next_check = time.time() + check_interval
                print(f"\n⏳ Waiting {check_interval} seconds until next check...")
                print(f"🫀 Heartbeat: sleeping now, next wake at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_check))}")
                time.sleep(check_interval)
                print(f"🫀 Heartbeat: woke up at {time.strftime('%Y-%m-%d %H:%M:%S')}, starting next cycle")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            print("=" * 70)
            
            # Close any open positions before exiting
            print("\n🔄 Closing any open positions...")
            self.close_position()
            
            print("\n👋 Bot shutdown complete")
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("🔄 Attempting to close positions...")
            self.close_position()


def main():
    """Main entry point for the bot."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API credentials from environment
    api_key = (
        os.getenv('BYBIT_DEMO_KEY')
        or os.getenv('BYBIT_TESTNET_KEY')
        or os.getenv('BYBIT_API_KEY')
    )
    api_secret = (
        os.getenv('BYBIT_DEMO_SECRET')
        or os.getenv('BYBIT_TESTNET_SECRET')
        or os.getenv('BYBIT_API_SECRET')
    )

    # Avoid hidden newline/space issues from copied secrets.
    if isinstance(api_key, str):
        api_key = api_key.strip()
    if isinstance(api_secret, str):
        api_secret = api_secret.strip()
    
    # Validate credentials
    if not api_key or not api_secret:
        print("❌ Error: API credentials not found!")
        print("Set BYBIT_DEMO_KEY/BYBIT_DEMO_SECRET or BYBIT_TESTNET_KEY/BYBIT_TESTNET_SECRET")
        return
    
    # Create and run bot
    bot = BybitFuturesBot(api_key, api_secret)
    bot.run()  # Uses configured check_interval (5m=300s or 1h=60s)


# Backward-compatible alias to avoid breaking existing imports.
BinanceFuturesBot = BybitFuturesBot


if __name__ == '__main__':
    main()
