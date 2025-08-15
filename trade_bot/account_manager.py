from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd

@dataclass
class Order:
    _id_counter: int = field(init=False, default=0, repr=False)  # Hidden class variable
    id: int = field(init=False)
    symbol: str
    volume: int
    direction: str
    type: str
    take_profit_rate: float = 0.05
    trigger_take_profit: bool = False
    cutloss_rate: float = 0.05
    buy_time: datetime = None
    take_profit_time: datetime = None
    status: str = 'pending'
    buy_cost: float = None
    sell_cost: float = None
    buy_price: float = None
    sell_price: float = None
    note: str = ""

    def __post_init__(self):
        self.__class__._id_counter += 1  # Increment the counter
        self.id = self.__class__._id_counter  # Assign unique ID

class AccountManager:
    TRADING_FEE = 0.0018

    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.money_balance = initial_balance
        self.current_all_balance = initial_balance
        self.active_orders = []
        self.last_price_data = {}
        self.trackings = {
            'time': [],
            'all_balance': [],
            'money_balance': [],
            'active_orders': [],
            'take_profit_orders': []
        }

    def get_active_orders(self):
        return self.active_orders

    def get_initial_balance(self):
        return self.initial_balance
    
    def get_all_balance(self):
        return self.current_all_balance
    
    def get_money_balance(self):
        return self.money_balance
    
    def find_order_by_id(self, order_id):
        for order in self.active_orders:
            if order.id == order_id:
                return order
    
    def order(self, order: Order, price_data: dict[str, pd.DataFrame], current_time: datetime):
        """
        Buy or sell a volume amount of symbol with order_type. Only support close and open order_type for now.
        """
        if order.direction not in ['buy', 'sell']:
            raise ValueError("Invalid order.direction: {}, only support buy and sell.".format(order.direction))
        if order.type not in ['close', 'open']:
            raise ValueError("Invalid order.type: {}, only support close and open.".format(order.type))

        if order.direction == 'buy':
            if current_time != order.buy_time:
                return order
            if price_data[order.symbol].shape[0] == 0:
                order.status = 'no_price_data'
                return order
            buy_price = price_data[order.symbol].iloc[-1][order.type]
            total_cost = order.volume * buy_price * (1 + self.TRADING_FEE)
            if total_cost > self.money_balance:
                print(f"Insufficient balance to buy {order.volume} {order.symbol}.")
                order.status = 'insufficient_balance'
                return order
            order.buy_cost = total_cost
            order.buy_price = buy_price
            self.money_balance -= total_cost
        else:
            if price_data[order.symbol].shape[0] == 0:
                raise ValueError(f"No price data found for symbol {order.symbol}, increase take_profit_time by 1 day.")
            sell_price = price_data[order.symbol].iloc[-1][order.type]
            total_cost = order.volume * sell_price * (1 - self.TRADING_FEE)
            order.sell_cost = total_cost
            order.sell_price = sell_price
            self.money_balance += total_cost
            self.trackings['take_profit_orders'][-1].append(order)
            self.active_orders.remove(self.find_order_by_id(order.id))
        order.status = 'triggered'
        print('Order triggered:', order)
        return order

    def remove_order(self, order: Order):
        self.active_orders.remove(order)

    def take_profit(self, order: Order, price_data: dict[str, pd.DataFrame], current_time: datetime, order_type: str = 'close'):
        """
        Take profit for an order if the current price is higher than the take_profit_price.
        """
        if order.take_profit_time is not None and price_data[order.symbol].shape[0] > 0 and current_time >= order.take_profit_time:
            sell_order = deepcopy(order)
            sell_order.direction = 'sell'
            sell_order.type = order_type or sell_order.type
            self.order(sell_order, price_data, current_time)

    def execute_orders(self, price_data: dict[str, pd.DataFrame], current_time: datetime):
        for order in self.active_orders:
            order = self.order(order, price_data, current_time)
            if order.status == 'insufficient_balance':
                self.remove_order(order)
        
        self.trackings['take_profit_orders'].append([])
        active_orders = self.active_orders.copy()
        for order in active_orders:
            self.take_profit(order, price_data, current_time)

    def add_orders(self, orders):
        self.active_orders += orders

    def report(self, current_time: datetime, price_data: dict[str, pd.DataFrame]):
        all_balance = self.money_balance
        for order in self.active_orders:
            if order.status != 'triggered':
                continue
            if price_data[order.symbol].shape[0] == 0 and order.symbol not in self.last_price_data:
                raise ValueError(f"No price data found for symbol {order.symbol}.")
            price = price_data[order.symbol].iloc[-1]['close'] if price_data[order.symbol].shape[0] > 0 else self.last_price_data[order.symbol].iloc[-1]['close']
            all_balance += order.volume * price
            print('Profit of order with ID {}: {}'.format(order.id, order.volume * price - order.buy_cost))
            if price_data[order.symbol].shape[0] > 0:
                self.last_price_data[order.symbol] = price_data[order.symbol]
        self.current_all_balance = all_balance
        self.trackings['time'].append(current_time)
        self.trackings['all_balance'].append(all_balance)
        self.trackings['money_balance'].append(self.money_balance)
        self.trackings['active_orders'].append(self.active_orders.copy())
        print(f"[{current_time}]\tAll balance: {self.current_all_balance}\n\t\t\tMoney balance: {self.money_balance}\n\t\t\tActive orders: {self.active_orders}")