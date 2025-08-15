import json
from datetime import datetime, timedelta

import pandas as pd

from trade_bot.data import DataFetcher
from trade_bot.strategy.common import Strategy
from trade_bot.account_manager import AccountManager

import json
import numpy as np
from datetime import datetime

# Define a custom encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):  # Convert datetime to string
            return obj.isoformat()
        elif isinstance(obj, np.float64):  # Convert np.float64 to float
            return float(obj)
        elif hasattr(obj, "__dict__"):  # Convert custom objects to dicts
            return obj.__dict__
        return super().default(obj)

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
            self.account_manager.execute_orders(current_time=cur_time, price_data=data)
            orders = self.strategy.act(data=data, current_time=cur_time)
            self.account_manager.add_orders(orders=orders)
            self.account_manager.report(current_time=cur_time, price_data=data)
            cur_time += interval
        
        print(f"Simulation completed from {start_time} to {end_time}.")
        print(f"Initial balance: {self.account_manager.initial_balance}")
        print(f"Final balance: {self.account_manager.current_all_balance}")
        print(f"Profit: {self.account_manager.current_all_balance - self.account_manager.initial_balance}")
        trackings = self.account_manager.trackings
        trackings_df = pd.DataFrame([trackings['time'], trackings['all_balance'], trackings['money_balance']]).T
        trackings_df.columns = ['time', 'all_balance', 'money_balance']
        trackings_df.to_csv('trackings.csv', index=False)

        json_data_active_orders = json.dumps(trackings['active_orders'], cls=CustomJSONEncoder, indent=4)
        json_data_take_profit_orders = json.dumps(trackings['take_profit_orders'], cls=CustomJSONEncoder, indent=4)
        with open('active_orders.json', 'w') as f:
            f.write(json_data_active_orders)
        with open('take_profit_orders.json', 'w') as f:
            f.write(json_data_take_profit_orders)
        
