"""
Trading Strategies Module

Contains all converted Pine Script strategies ready for backtesting.
Each strategy follows the standard interface:
    def strategy_name(df: pd.DataFrame, **params) -> pd.Series
"""

from .Dimensional_Support_Resistence import strategy_dsr
from .Low_Volatility_Range_Breakout import strategy_lvrb
from .Order_Blocks_Breaker_Blocks import strategy_order_blocks
from .UT_Bot_Alerts import strategy_ut_bot
from .Liquidity_Swings import strategy_liquidity_swings_lux
from .SuperTrend_Strategy import strategy_supertrend
from .Chandelier_Exit import strategy_chandelier_exit
from .Intraday_bb_squeeze import strategy_intraday_bb_squeeze
from .orb_session import strategy_orb_sessions

__all__ = [
    'strategy_dsr',
    'strategy_lvrb',
    'strategy_order_blocks',
    'strategy_ut_bot',
    'strategy_liquidity_swings_lux',
    'strategy_supertrend',
    'strategy_chandelier_exit',
    'strategy_intraday_bb_squeeze',
    'strategy_orb_sessions',
]
