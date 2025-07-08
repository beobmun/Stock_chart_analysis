import win32com.client
import pandas as pd
from cybos import CpUtil, CpSysDib, CPE_MARKET_KIND


print("Cybos 연결 준비")
cpUtil = CpUtil()
cpUtil.check_server_type()


kospi_list = cpUtil.get_stock_list(CPE_MARKET_KIND.KOSPI)
print("KOSPI 종목:")
print(kospi_list[:5])

cpSysDib = CpSysDib()
df = cpSysDib.get_stock_data(kospi_list[1])
print('---')
print(df)
