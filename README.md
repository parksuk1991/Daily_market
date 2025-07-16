
# Daily Market 대시보드

전일 기준 글로벌 주요 시장(주식, 채권, 통화/비트코인, 섹터) 성과를 한눈에 요약해서 보여주는 Streamlit 대시보드입니다.

## 실행법

1. 패키지 설치  
   ```bash
   pip install -r requirements.txt
   ```
2. Streamlit 실행  
   ```bash
   streamlit run global_dashboard.py
   ```

## 주요 기능
- 글로벌 주요 주가지수/ETF, 채권, 섹터, 통화/비트코인 하루 수익률 요약
- 주요 주가지수(ETF 기준) Normalized 누적 수익률 그래프(6개월)
- 주요 섹터 ETF 하루 수익률 비교(막대그래프)
- S&P500 ETF 기준 뉴스 헤드라인(yfinance 제공)

---

*ETF 및 종목은 미국 상장 기준이며, 필요시 코드에서 자유롭게 수정 가능합니다.*
