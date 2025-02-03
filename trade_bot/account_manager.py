from datetime import datetime

class AccountManager:
    def __init__(self):
        self.balance = 0
        self.holdings = {}

    def order(self, symbol: str, volume: int, order_direction: str, order_type: str, data):
        """
        Buy or sell a volume amount of symbol with order_type. Only support close and open order_type for now.
        """
        if order_direction not in ['buy', 'sell']:
            raise ValueError("Invalid order_direction: {}, only support buy and sell.".format(order_direction))
        if order_type not in ['close', 'open']:
            raise ValueError("Invalid order_type: {}, only support close and open.".format(order_type))

        if order_direction == 'buy':
            if order_type == 'close':
                self.balance -= volume * data[symbol].iloc[-1]['close']
                self.balance += volume * self.holdings[symbol]
                self.holdings[symbol] = 0
            else:
                self.holdings[symbol] = volume
                self.balance -= volume
        
    def execute(self, actions, data):
        for action in actions:
            self.order(action['symbol'], action['volume'], action['order_direction'], action['order_type'], data)

    def report(self, current_time: datetime):
        print(f"[{current_time}] Balance: {self.balance}, Holdings: {self.holdings}")