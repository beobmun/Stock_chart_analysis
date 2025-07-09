import win32com
import pandas as pd
from cybos import CpUtil, CpSysDib, CPE_MARKET_KIND
from tqdm import tqdm
import os

print("ready to connect cybos")
cpUtil = CpUtil()
cpUtil.check_server_type()

print("Get KOSPI code list")
kospi_list = cpUtil.get_stock_list(CPE_MARKET_KIND.KOSPI)

# stock info 가져오기
stock_info = {"code": [], "name": [], "kospi200": [], "listed_date": [], "status": []}
for code in tqdm(kospi_list, desc="Get stock info"):
    try:
        (n, k, l, s) = cpUtil.get_stock_info(code)
        stock_info["code"].append(code)
        stock_info["name"].append(n)
        stock_info["kospi200"].append(k)
        stock_info["listed_date"].append(l)
        stock_info["status"].append(s)
    except:
        print(f"err get info {code}")
        
info_df = pd.DataFrame(stock_info)

os.makedirs("./data/info", exist_ok=True)
info_df.to_csv("./data/info/kospi_info.csv", sep=',', index=False)