import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="글로벌 시황 대시보드",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 글로벌 시황 대시보드")
st.markdown("#### 전일 시장 데이터 및 다양한 기간별 성과 확인")

# ======= 자산 정의 =======
STOCK_ETFS = {
    'S&P 500 (SPY)': 'SPY',
    'NASDAQ 100 (QQQ)': 'QQQ',
    'MSCI ACWI (ACWI)': 'ACWI',
    '유럽(Europe, VGK)': 'VGK',
    '중국(China, MCHI)': 'MCHI',
    '일본(Japan, EWJ)': 'EWJ'
}
BOND_ETFS = {
    '미국 장기국채(TLT)': 'TLT',
    '미국 단기국채(SHY)': 'SHY',
    '미국 IG회사채(LQD)': 'LQD',
    '신흥국채(EMB)': 'EMB'
}
CURRENCY = {
    'USD/KRW': 'KRW=X',
    'USD/EUR': 'EURUSD=X',
    'USD/JPY': 'JPY=X',
    '비트코인(BTC-USD)': 'BTC-USD'
}
SECTOR_ETFS = {
    'IT (XLK)': 'XLK',
    '헬스케어 (XLV)': 'XLV',
    '금융 (XLF)': 'XLF',
    '커뮤니케이션 (XLC)': 'XLC',
    '에너지 (XLE)': 'XLE',
    '산업재 (XLI)': 'XLI',
    '소재 (XLB)': 'XLB',
    '필수소비재 (XLP)': 'XLP',
    '자유소비재 (XLY)': 'XLY',
    '유틸리티 (XLU)': 'XLU',
    '부동산 (XLRE)': 'XLRE'
}

# ===== Helper: 다양한 기간 성과 =====
def get_perf_table(tickers, start, end):
    # 가져올 최소 시작일
    today = end
    first_date = today - timedelta(days=365*3+31)
    df = yf.download(list(tickers.values()), start=first_date, end=today+timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    df.columns = tickers.keys()
    last = df.index[-1]
    # 기준일 계산
    periods = {
        '1D': 1,
        '1W': 5,
        'MTD': None, # 월초
        '1M': 21,
        '3M': 63,
        '6M': 126,
        'YTD': None, # 연초
        '1Y': 252,
        '3Y': 756
    }
    results = []
    for label in df.columns:
        row = {}
        row['자산명'] = label
        row['현재값'] = df[label].iloc[-1] if not np.isnan(df[label].iloc[-1]) else None
        for k, val in periods.items():
            try:
                if k == 'MTD':
                    base = df[:last].loc[df.index.month == last.month].iloc[0][label]
                elif k == 'YTD':
                    base = df[:last].loc[df.index.year == last.year].iloc[0][label]
                else:
                    if len(df) > val:
                        base = df[label].iloc[-val-1]
                    else:
                        base = df[label].iloc[0]
                if base and row['현재값']:
                    row[k] = (row['현재값']/base - 1) * 100
                else:
                    row[k] = np.nan
            except Exception:
                row[k] = np.nan
        results.append(row)
    return pd.DataFrame(results)

# ===== Helper: Normalized 수익률 =====
def get_normalized_prices(tickers, months=6):
    end = datetime.now().date()
    start = end - timedelta(days=months*31)
    df = yf.download(list(tickers.values()), start=start, end=end + timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    df.columns = tickers.keys()
    norm_df = df / df.iloc[0] * 100
    return norm_df

# ===== 뉴스 헤드라인 (Colab 참고, yfinance v0.2 이상) =====
def get_news_headlines(tickers, limit=3):
    # 여러 티커 순회, 최근 뉴스만 추출
    news_list = []
    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)
        try:
            news = ticker.news if hasattr(ticker, "news") else ticker.get_news()
            for article in news[:limit]:
                content = article.get('content', {})
                title = article.get('title') or content.get('title')
                pubdate = article.get('providerPublishTime') or content.get('pubDate')
                if pubdate:
                    # 시간 단위별로 처리
                    if isinstance(pubdate, int):
                        date = datetime.fromtimestamp(pubdate)
                    else:
                        try:
                            date = pd.to_datetime(pubdate)
                        except Exception:
                            date = None
                else:
                    date = None
                news_list.append({
                    '티커': ticker_symbol,
                    '일자': date.strftime('%Y-%m-%d') if date else '',
                    '헤드라인': title
                })
        except Exception:
            continue
    df = pd.DataFrame(news_list)
    # 최신순 정렬
    if not df.empty:
        df = df.sort_values('일자', ascending=False)
    return df

# ==== 사용자 입력 UI ====
with st.sidebar:
    st.header("설정")
    idx_months = st.slider("주요 주가지수 Normalized 기간 (개월)", 3, 36, 6)
    sector_months = st.slider("섹터 Normalized 기간 (개월)", 3, 36, 6)
    news_cnt = st.slider("뉴스 헤드라인 개수 (티커별)", 1, 5, 3)

# ==== 버튼 및 로딩 ====
if st.button("전일 시장 Update", type="primary"):
    with st.spinner("데이터 불러오는 중..."):
        # 1. 주식/채권/통화/비트코인/섹터 성과 테이블
        st.subheader("📊 주식시장 성과 (다양한 기간)")
        stock_perf = get_perf_table(STOCK_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(stock_perf.set_index('자산명'), use_container_width=True, height=350,
            column_config={
                "현재값": st.column_config.NumberColumn("현재값", format="%.2f"),
                **{k: st.column_config.NumberColumn(f"{k} (%)", format="%.2f") for k in ['1D','1W','MTD','1M','3M','6M','YTD','1Y','3Y']}
            }
        )
        st.subheader("📊 채권시장 성과 (다양한 기간)")
        bond_perf = get_perf_table(BOND_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(bond_perf.set_index('자산명'), use_container_width=True, height=250)
        st.subheader("📊 통화 및 비트코인 성과 (다양한 기간)")
        curr_perf = get_perf_table(CURRENCY, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(curr_perf.set_index('자산명'), use_container_width=True, height=250)

        # 2. 주요 주가지수 Normalized 성과
        st.subheader(f"📈 주요 주가지수 Normalized 수익률 (최근 {idx_months}개월)")
        norm_idx = get_normalized_prices(STOCK_ETFS, months=idx_months)
        fig1 = go.Figure()
        for col in norm_idx.columns:
            fig1.add_trace(go.Scatter(x=norm_idx.index, y=norm_idx[col], mode='lines', name=col))
        fig1.update_layout(
            xaxis_title="날짜", yaxis_title="100 기준 누적수익률(%)",
            template="plotly_dark", height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 3. 섹터 Normalized 성과
        st.subheader(f"📈 섹터 ETF Normalized 수익률 (최근 {sector_months}개월)")
        norm_sector = get_normalized_prices(SECTOR_ETFS, months=sector_months)
        fig2 = go.Figure()
        for col in norm_sector.columns:
            fig2.add_trace(go.Scatter(x=norm_sector.index, y=norm_sector[col], mode='lines', name=col))
        fig2.update_layout(
            xaxis_title="날짜", yaxis_title="100 기준 누적수익률(%)",
            template="plotly_dark", height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 4. 뉴스 헤드라인
        st.subheader("📰 최근 뉴스 헤드라인 (대표 티커 위주)")
        # 주요시장 대표티커 3개(빠르게)
        headline_tickers = ['SPY', 'QQQ', 'XLK', 'XLF', 'XLV', 'BTC-USD']
        news_df = get_news_headlines(headline_tickers, news_cnt)
        if not news_df.empty:
            for _, row in news_df.iterrows():
                st.markdown(f"- **[{row['티커']}]** {row['일자']}: {row['헤드라인']}")
        else:
            st.info("뉴스 헤드라인을 가져올 수 없습니다.")
else:
    st.info("왼쪽 설정 후 '전일 시장 Update' 버튼을 눌러주세요.")
