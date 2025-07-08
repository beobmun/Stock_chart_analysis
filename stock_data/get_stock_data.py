import win32com.client
import pandas as pd
from cybos import CpUtil, CpSysDib, CPE_MARKET_KIND
from tqdm import tqdm
import os


print("Cybos 연결 준비")
cpUtil = CpUtil()
cpUtil.check_server_type()


kospi_list = cpUtil.get_stock_list(CPE_MARKET_KIND.KOSPI)
print("KOSPI 종목:")
print(kospi_list[:5])

cpSysDib = CpSysDib()

# Kospi 5분봉 데이터 불러오기 & csv 파일 저장
os.makedirs("./data/KOSPI",exist_ok=True)
for code in tqdm(kospi_list[:3], desc="Get KOSPI", position=0):
    tmp_df = cpSysDib.get_stock_data_p(code, '202006', '202506')
    tmp_df.to_csv(f"./data/KOSPI/{code}.csv", sep=',', index=False)