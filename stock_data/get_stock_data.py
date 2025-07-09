import win32com.client
import pandas as pd
from cybos import CpUtil, CpSysDib, CPE_MARKET_KIND
from tqdm import tqdm
import os


print("Cybos 연결 준비")
cpUtil = CpUtil()
cpUtil.check_server_type()

cpSysDib = CpSysDib()

# KOSPI 5분봉 데이터 불러오기 & csv 파일 저장
# 기존에 불러온 데이터 있는지 확인
if os.path.exists("./data/KOSPI"):
    existed_files = os.listdir("./data/KOSPI")
    existed_files = {filename.split('.')[0] for filename in existed_files}
else:
    os.makedirs("./data/KOSPI",exist_ok=True)
    existed_files = {}

kospi_list = cpUtil.get_stock_list(CPE_MARKET_KIND.KOSPI)
for code in tqdm(kospi_list, desc="Get KOSPI", position=0):
    if code in existed_files:
        continue
    tmp_df = cpSysDib.get_stock_data_p(code, '202006', '202506')
    tmp_df.to_csv(f"./data/KOSPI/{code}.csv", sep=',', index=False)
    existed_files.add(code)