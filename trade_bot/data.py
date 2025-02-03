from typing import List
import pandas as pd
from datetime import datetime

class DataFetcher:
    def __init__(
        self,
        data_dir: str,
        symbols: List[str] = None
    ):
        """
        Initialize a DataFetcher object with the data directory and symbols.

        Args:
            data_dir (str): Directory to store the data.
            symbols (List[str]): List of symbols to fetch data for.
        """

        self.data = {}
        self.data_dir = data_dir
        self.symbols = symbols
        for symbol in symbols:
            self.data[symbol] = pd.read_csv(f"{data_dir}/{symbol}.csv")
            self.data[symbol]['time'] = pd.to_datetime(self.data[symbol]['time'])
    
    def fetch(self, current_time: datetime, start_time: datetime = None, symbol: str = None, fetch_from_beginning: bool = False):
        """
        Fetch data from start_time to current_time. If start_time is None, fetch data of current_time only.
        """
        returned_data = {}
        fetched_symbol = symbol
        for symbol in self.symbols:
            if fetch_from_beginning:
                returned_data[symbol] = self.data[symbol][self.data[symbol]['time'] <= current_time].reset_index(drop=True)
            elif start_time is not None:
                returned_data[symbol] = self.data[symbol][(self.data[symbol]['time'] >= start_time) & (self.data[symbol]['time'] <= current_time)].reset_index(drop=True)
            else:
                returned_data[symbol] = self.data[symbol][self.data[symbol]['time'] == current_time].reset_index(drop=True)
        if fetched_symbol is not None:
            return returned_data[fetched_symbol]
        return returned_data