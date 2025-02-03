from datetime import datetime, timedelta

from trade_bot.data import DataFetcher
from trade_bot.strategy.common import Strategy
from trade_bot.account_manager import AccountManager

class Simulator:
    def __init__(self, data_fetcher: DataFetcher, strategy: Strategy, account_manager: AccountManager):
        self.data_fetcher = data_fetcher
        self.strategy = strategy
        self.account_manager = account_manager

    def run(self, start_time: datetime, end_time: datetime, interval: timedelta):
        """
        Run the simulator from start_time to end_time.
        """
        cur_time = start_time
        while cur_time < end_time:
            data = self.data_fetcher.fetch(current_time=cur_time, fetch_from_beginning=cur_time==start_time)
            actions = self.strategy.act(data=data, current_time=cur_time)
            self.account_manager.execute(actions=actions, data=data)
            self.account_manager.report(current_time=cur_time)
            cur_time += interval
