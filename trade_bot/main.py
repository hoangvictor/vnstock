import os
import shutil
import sys
sys.path.append('/remote/vast0/share-mv/tien/project/vnstock')

from datetime import datetime, timedelta
from trade_bot.account_manager import AccountManager
from trade_bot.data import DataFetcher
from trade_bot.simulator import Simulator
from trade_bot.strategy.lgbm.strategy import LGBMStrategy

def run():
    if os.path.exists('./run_data'):
        shutil.rmtree('./run_data')
    os.makedirs('./run_data', exist_ok=True)
    symbols = ['ACB', 'AGR', 'BNA', 'AAA', 'AGG', 'CSC', 'CTF', 'DRI', 'DHG', 'DLG', 'DGC', 'DPM', 'DPR', 'BWE', 'BID', 'ACV', 'EVG', 'ELC', 'AMV', 'ANV', 'FCN', 'BVB', 'CSV', 'APG', 'APH', 'FIT', 'ICT', 'FMC', 'FPT', 'CSM', 'DHA', 'DCM', 'BSI', 'BMI', 'DVP', 'CTS', 'BMP', 'HAH', 'BCG', 'BOT', 'AAS', 'CNG', 'HBC', 'C69', 'D2D', 'BFC', 'LTG', 'CII', 'HAX', 'AAV', 'ABB', 'DRC', 'ASM', 'ABI', 'CTG', 'CST', 'CMG', 'CTI', 'DRH', 'FIR', 'EVF', 'BCM', 'DBD', 'IJC', 'GEX', 'HPG', 'CCL', 'HQC', 'DXG', 'HHP', 'HT1', 'TCH', 'IDI', 'CTR', 'BVH', 'HTN', 'IDJ', 'DXS', 'DIG', 'TNH', 'SSH', 'BVS', 'DPG', 'PHP', 'ITD', 'DBC', 'CEO', 'HHS', 'EIB', 'KDC', 'PTB', 'HCM', 'KDH', 'KHG', 'BMC', 'HDC', 'HUT', 'HDG', 'C4G', 'KHP', 'GAS', 'JVC', 'KSB', 'GEG', 'GMD', 'HVH', 'KVC', 'HAP', 'SGP', 'DHC', 'LAS', 'LCG', 'LHG', 'LIG', 'IMP', 'KOS', 'LSS', 'DHT', 'MBB', 'KBC', 'DGW', 'LPB', 'GIL', 'MSN', 'MSR', 'DST', 'ITA', 'NAG', 'FTS', 'L14', 'CTD', 'MWG', 'NLG', 'NRC', 'HNG', 'PSH', 'NT2', 'BSR', 'HDB', 'NTL', 'OCB', 'NTC', 'PET', 'MCH', 'PLX', 'DCL', 'MSB', 'PC1', 'PNJ', 'GKM', 'PLC', 'DTD', 'PVB', 'PDR', 'PVD', 'LDG', 'PVT', 'PHR', 'IDV', 'QCG', 'PVI', 'POW', 'IDC', 'QNS', 'PVP', 'SAB', 'FRT', 'NAF', 'QTP', 'SBS', 'SCR', 'HAG', 'NKG', 'REE', 'NVL', 'SJD', 'SJS', 'SGN', 'TCM', 'NAB', 'MST', 'SAM', 'NHA', 'PAN', 'MIG', 'HSG', 'PPC', 'SBT', 'VHM', 'SCG', 'SCS', 'OIL', 'PVS', 'SSB', 'SHB', 'SHS', 'NTP', 'SMC', 'SZC', 'PVC', 'TIP', 'TLG', 'SMB', 'MSH', 'SLS', 'SAS', 'SIP', 'STB', 'SSI', 'SHI', 'TIG', 'VHC', 'TCB', 'TCL', 'TDC', 'VIC', 'VJC', 'THG', 'MBS', 'TNG', 'GVR', 'TPB', 'TTN', 'VPB', 'VPG', 'TTA', 'TTF', 'TV2', 'VAB', 'VCG', 'VC3', 'VEF', 'VCS', 'VEA', 'VCB', 'VCI', 'VFS', 'VGS', 'VGC', 'VIB', 'VIP', 'VIX', 'HVN', 'VND', 'VNM', 'VOS', 'VPI', 'VSC', 'VGI', 'VRE', 'VTO', 'VTP', 'YEG']
    symbols = symbols
    data_fetcher = DataFetcher(data_dir='/remote/vast0/share-mv/tien/project/vnstock/data/stocks', symbols=symbols)
    initial_balance = 1e5
    account_manager = AccountManager(initial_balance)
    strategy = LGBMStrategy(
        account_manager=account_manager,
        retrain_period=timedelta(days=30),
        test_period=timedelta(days=30)
    )
    simulator = Simulator(data_fetcher, strategy, account_manager)
    simulator.run(start_time=datetime(2024, 6, 1), end_time=datetime(2025, 1, 25), interval=timedelta(days=1))

if __name__ == "__main__":
    run()