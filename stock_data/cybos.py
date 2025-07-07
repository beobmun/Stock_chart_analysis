import win32com.client
from enum import IntEnum

class CPE_MARKET_KIND(IntEnum):
    NULL = 0
    KOSPI = 1
    KOSDAQ = 2
    FREEBOARD = 3
    KRX = 4

class CPE_KOSPI200_KIND(IntEnum):
    NONE = 0
    CONSTRUCTIONS_MACHINERY = 1
    SHIPBUILDING_TRANSPORTATION = 2
    STEELS_METERIALS = 3
    ENERGY_CHEMICALS = 4
    IT = 5
    FINANCE = 6
    CUSTOMER_STAPLES = 7
    CUSTOMER_DISCRETIONARY = 8

class CpUtil:
    def __init__(self):
        self.CpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
        self.CpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")  
