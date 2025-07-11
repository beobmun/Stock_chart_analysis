import win32com.client
from enum import IntEnum
import pandas as pd
from tqdm import tqdm

class CPE_MARKET_KIND(IntEnum):
    NULL = 0
    KOSPI = 1
    KOSDAQ = 2
    FREEBOARD = 3
    KRX = 4

class CPE_KOSPI200_KIND(IntEnum):
    NONE = 0                            # Not in KOSPI200
    CONSTRUCTIONS_MACHINERY = 1         # 건설기계
    SHIPBUILDING_TRANSPORTATION = 2     # 조선운송
    STEELS_METERIALS = 3                # 철강소재
    ENERGY_CHEMICALS = 4                # 에너지화학
    IT = 5                              # IT   
    FINANCE = 6                         # 금융
    CUSTOMER_STAPLES = 7                # 필수소비재
    CUSTOMER_DISCRETIONARY = 8          # 자유소비재

class CPE_SUPERVISION_KIND(IntEnum):
    CPC_STOCK_STATUS_NORMAL = 0
    CPC_STOCK_STATUS_STOP = 1
    CPC_STOCK_STATUS_BREAK = 2

class CpUtil:
    def __init__(self):
        self.CpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
        self.CpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
        
    def check_server_type(self):
        server_type = self.CpCybos.ServerType
        
        if server_type == 0:
            print("연결 끊김")
        elif server_type == 1:
            print("cybos plus 서버 연결 완료")
        elif server_type == 2:
            print("HTS 보통서버 연결 완료")
        else:
            print("알 수 없음")
    
    def get_stock_list(self, CPE_MARKET_KIND):
        return self.CpCodeMgr.GetStockListByMarket(CPE_MARKET_KIND)

    def get_stock_info(self, code):
        name = self.CpCodeMgr.CodeToName(code)
        kospi200 = self.CpCodeMgr.GetStockKospi200Kind(code)
        listed_date = self.CpCodeMgr.GetStockListedDate(code)
        status = self.CpCodeMgr.GetStockStatusKind(code)
        return (name, kospi200, listed_date, status)

class CpSysDib:
    def __init__(self):
        self.StockChart = win32com.client.Dispatch("CpSysDib.StockChart")
        
    def get_stock_data(self, code, start, end):
        self.StockChart.SetInputValue(0, code) # 종목 선택
        self.StockChart.SetInputValue(1, ord('1')) # '1': 기간으로 요청, '2': 개수로 요청
        self.StockChart.SetInputValue(2, end) # 요청 종료일
        self.StockChart.SetInputValue(3, start) # 요청 시작일
        self.StockChart.SetInputValue(5, (0, 1, 2, 3, 4, 5, 8)) # 0:날짜, 1:시간, 2:시가, 3:고가, 4:저가, 5:종가, 8:거래량
        self.StockChart.SetInputValue(6, ord('m')) # 차트 구분: 'm': 분
        self.StockChart.SetInputValue(7, 5) # 주가
        self.StockChart.SetInputValue(9, ord('1')) # '0': 무수정 주가, '1':수정주가
        self.StockChart.SetInputValue(10, ord('3'))
        
        self.StockChart.BlockRequest()
        
        num_data = self.StockChart.GetHeaderValue(3) # 3: 수신 개수
        num_field = self.StockChart.GetHeaderValue(1) # 1: 필드 개수
        name_field = self.StockChart.GetHeaderValue(2) # 2: 필드 이름
        
        df = pd.DataFrame()
        for i in range(num_field):
            temp = []
            for j in range(num_data):
                temp.append(self.StockChart.GetDataValue(i, j))
            df[name_field[i]] = temp
        
        return df.loc[::-1].reset_index(drop=True)
    
    def _make_period(self, start, end):
        s_y = int(start[:4])
        s_m = int(start[4:])
        e_y = int(end[:4])
        e_m = int(end[4:])
        period = list()
        while s_y < e_y or s_m <= e_m:
            if s_m < 10:
                sm = '0'+str(s_m)
            else:
                sm = str(s_m)
            s = str(s_y) + sm + '01'
            e = str(s_y) + sm + '31'
            period.append((s, e))
            s_m += 1
            if s_m > 12:
                s_m = 1
                s_y += 1
        return period
            
    def get_stock_data_p(self, code, start, end):
        p = self._make_period(start, end)
        df = list()
        with tqdm(p, desc=f"get data {code}", leave=False) as pbar:
            for s, e in pbar:
                df.append(self.get_stock_data(code, s, e))
        return pd.concat(df).reset_index(drop=True)