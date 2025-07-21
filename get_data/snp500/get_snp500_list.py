import requests
from bs4 import BeautifulSoup
import pandas as pd

import requests
from bs4 import BeautifulSoup
import pandas as pd

# S&P 500 종목 위키피디아 URL
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# 웹 페이지 요청
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 종목 테이블 찾기
table = soup.find('table', {'class': 'wikitable'})

# 헤더 추출
headers = [th.text.strip() for th in table.find_all('tr')[0].find_all('th')]

# 각 행에서 데이터 추출
rows = []
for row in table.find_all('tr')[1:]:  # 첫 번째 행은 헤더
    cols = row.find_all('td')
    if len(cols) >= 7:
        rows.append({
            'Symbol': cols[0].text.strip(),
            'Security': cols[1].text.strip(),
            'GICS Sector': cols[2].text.strip(),
            'GICS Sub-Industry': cols[3].text.strip(),
            'Date added': cols[5].text.strip(),
            'Founded': cols[7].text.strip() if len(cols) > 7 else ''
        })

# DataFrame으로 변환
df = pd.DataFrame(rows)

# CSV로 저장_현재 25.07.21 기준 저장됨.
df.to_csv('../data/info/snp500_info.csv', index=False, encoding='utf-8-sig')