import win32com.client
import pandas
from cybos import CpUtil, CPE_MARKET_KIND


print("Cybos 연결 준비")
cpUtil = CpUtil()
server_type = cpUtil.CpCybos.ServerType

if server_type == 0:
    print("연결 끊김")
elif server_type == 1:
    print("cybos plus 서버 연결 완료")
elif server_type == 2:
    print("HTS 보통서버 연결 완료")
else:
    print("알 수 없음")

kospi_list = cpUtil.CpCodeMgr.GetStockListByMarket(CPE_MARKET_KIND.KOSPI)
print("KOSPI 종목:")
print(kospi_list[:5])
inStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
