"""
Module quản lý các thông tin về giao dịch chứng khoán từ nguồn dữ liệu Fireant.
"""

from typing import List, Optional
from vnstock.core.utils.logger import get_logger
from vnai import optimize_execution

from vnstock.core.utils.websocket import WSClient
from .const import _WSS_BASE_URL
from .models import HandshakeMsg, TradeSubscriptionMsg

import json
import pandas as pd

logger = get_logger(__name__)


class Trading:
    TRADES_BOARD_COLUMNS = ["id", "date", "price", "volume", "side", "sessionDate", "symbol", "totalVolume"]

    """
    Truy xuất dữ liệu giao dịch của mã chứng khoán từ nguồn dữ liệu Fireant.
    
    Tham số:
        - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là True.
    """
    def __init__(self, show_log: Optional[bool] = True):
        """
        Khởi tạo đối tượng Trading với các tham số cho việc truy xuất dữ liệu.
        """
        self.show_log = show_log
        self.wss_base_url = _WSS_BASE_URL
        self.ws_client = WSClient(
            self.wss_base_url,
            handshake_msg=HandshakeMsg().model_dump()
        )

        if not show_log:
            logger.setLevel('CRITICAL')
        
    @optimize_execution("Fireant")
    def trades_board(self, symbol: str, to_df: Optional[bool] = True, timeout: int = 5):
        """
        Truy xuất thông tin giao dịch theo lô của các mã chứng khoán tùy chọn từ nguồn dữ liệu Fireant.
        Nguồn dữ liệu đến từ websockets của Fireant.

        Tham số:
            - symbol (str): Mã chứng khoán cần truy xuất thông tin.
            - to_df (bool): Chuyển đổi kết quả thành DataFrame hoặc không. Mặc định là True.
        
        Returns:
            Thông tin giao dịch theo lô dưới dạng DataFrame hoặc chuỗi JSON tùy theo tham số to_df.
        """
        self.ws_client.send_message(
            TradeSubscriptionMsg(arguments=[symbol]).model_dump()
        )
        data = self.ws_client.wait_until_done(timeout=timeout)
        
        result = []
        for d in data:
            for sd in d.split('\x1e'):
                if len(sd) == 0:
                    continue
                sd = json.loads(sd)
                if 'result' in sd:
                    result = sd['result']
                    break
        
        if to_df:
            df = pd.DataFrame(result, columns=self.TRADES_BOARD_COLUMNS)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            df['sessionDate'] = pd.to_datetime(df['sessionDate'], format='mixed')
            return df
        
        return result