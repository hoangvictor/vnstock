import os
from vnstock import Listing, Trading
listing = Listing()
all_symbols = listing.all_symbols()['symbol'].tolist()
trading = Trading(source="Fireant")

save_dir = "/home/hoangvictor/Documents/Work/vnstock/data/trades_board"
for symbol in all_symbols:
    print(f"Symbol: {symbol}")
    data = trading.trades_board(symbol=symbol, timeout=3)
    if data.shape[0] > 0:
        session_date = data['sessionDate'].values[0].astype('M8[D]')
        os.makedirs(f"{save_dir}/{session_date}", exist_ok=True)
        data.to_csv(f"{save_dir}/{session_date}/{symbol}.csv", index=False)