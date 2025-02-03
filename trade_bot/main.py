import sys
sys.path.append('/data0/tien/vnstock')

from datetime import datetime, timedelta
from trade_bot.account_manager import AccountManager
from trade_bot.data import DataFetcher
from trade_bot.simulator import Simulator
from trade_bot.strategy.lgbm.strategy import LGBMStrategy

def run():
    data_fetcher = DataFetcher(data_dir='/data0/tien/vnstock/data/stocks', symbols=['ACB', 'VND'])
    strategy = LGBMStrategy(
        retrain_period=timedelta(days=7),
        valid_period=timedelta(days=14),
        test_period=timedelta(days=14)
    )
    account_manager = AccountManager()
    simulator = Simulator(data_fetcher, strategy, account_manager)
    simulator.run(start_time=datetime(2024, 6, 1), end_time=datetime(2025, 1, 25), interval=timedelta(days=1))

if __name__ == "__main__":
    run()