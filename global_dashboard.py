import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="글로벌 시황 대시보드",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 글로벌 시황 대시보드 (전일 기준 요약)")

# ======= 1. 티커 사전 정의 =======

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

# ======= 2. 데이터 불러오기 함수 =======

@st.cache_data(show_spinner=False)
def get_single_day_change(tickers: dict):
    """
    각 티커별로 어제-오늘(전일 종가 대비) 수익률을 계산
    """
    today = datetime.now().date()
    weekday = today.weekday()
    if weekday == 0:
        # 월요일: 금요일/목요일/수요일 필요
        last_trading = today - timedelta(days=3)
        day_before = today - timedelta(days=4)
    elif weekday == 6:
        # 일요일: 금요일/목요일 필요
        last_trading = today - timedelta(days=2)
        day_before = today - timedelta(days=3)
    elif weekday == 5:
        # 토요일: 금요일/목요일 필요
        last_trading = today - timedelta(days=1)
        day_before = today - timedelta(days=2)
    else:
        # 평일: 어제/그제
        last_trading = today - timedelta(days=1)
        day_before = today - timedelta(days=2)

    df = yf.download(list(tickers.values()), start=day_before, end=today + timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.ffill()

    ret_dict = {}
    for label, code in tickers.items():
        try:
            closes = df[code].dropna()
            if len(closes) < 2:
                ret = np.nan
            else:
                ret = closes.iloc[-1] / closes.iloc[-2] - 1
        except Exception:
            ret = np.nan
        ret_dict[label] = ret
    return ret_dict, df

@st.cache_data(show_spinner=False)
def get_normalized_prices(tickers: dict, months=6):
    end = datetime.now()
    start = end - timedelta(days=months*31)
    df = yf.download(list(tickers.values()), start=start, end=end + timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    norm_df = df / df.iloc[0] * 100
    norm_df.columns = tickers.keys()
    return norm_df

@st.cache_data(show_spinner=False)
def get_sector_performance(sectors: dict):
    ret_dict, _ = get_single_day_change(sectors)
    return ret_dict

@st.cache_data(show_spinner=False)
def get_news_headlines(etf_ticker="SPY", limit=7):
    try:
        ticker = yf.Ticker(etf_ticker)
        news = ticker.get_news()
        news = [n for n in news if 'title' in n]
        news = news[:limit]
        headlines = [
            {
                "title": n['title'],
                "publisher": n.get('publisher', ''),
                "link": n.get('link', ''),
                "providerPublishTime": datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d')
            }
            for n in news
        ]
        return headlines
    except Exception:
        return []

# ======= 3. 데이터 취득 =======
st.subheader("📈 1. 주식시장 하루 성과 (ETF 기준)")
stock_perf, stock_df = get_single_day_change(STOCK_ETFS)
stock_perf_df = pd.DataFrame({
    'ETF': list(stock_perf.keys()),
    '전일수익률': [f"{v*100:.2f}%" if pd.notnull(v) else "N/A" for v in stock_perf.values()]
})
st.dataframe(stock_perf_df, hide_index=True, use_container_width=True)

st.subheader("💵 2. 채권시장 하루 성과 (ETF 기준)")
bond_perf, _ = get_single_day_change(BOND_ETFS)
bond_perf_df = pd.DataFrame({
    'ETF': list(bond_perf.keys()),
    '전일수익률': [f"{v*100:.2f}%" if pd.notnull(v) else "N/A" for v in bond_perf.values()]
})
st.dataframe(bond_perf_df, hide_index=True, use_container_width=True)

st.subheader("💱 3. 통화 및 비트코인 하루 성과")
curr_perf, curr_df = get_single_day_change(CURRENCY)
curr_perf_df = pd.DataFrame({
    '통화/코인': list(curr_perf.keys()),
    '전일수익률': [f"{v*100:.2f}%" if pd.notnull(v) else "N/A" for v in curr_perf.values()]
})
st.dataframe(curr_perf_df, hide_index=True, use_container_width=True)

# ======= 4. 지수 Normalized 그래프 =======
st.subheader("📊 4. 주요 주가지수 (ETF 기준) Normalized 수익률 (최근 6개월)")
norm_df = get_normalized_prices(STOCK_ETFS, months=6)
fig_idx = go.Figure()
for col in norm_df.columns:
    fig_idx.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines', name=col))
fig_idx.update_layout(
    xaxis_title="날짜", yaxis_title="100 기준 누적수익률(%)",
    template="plotly_dark", height=400,
    legend=dict(orientation='h')
)
st.plotly_chart(fig_idx, use_container_width=True)

# ======= 5. 뉴스 헤드라인 =======
st.subheader("📰 5. 오늘의 주요 뉴스 헤드라인 (S&P500 ETF 기준, yfinance 제공)")
news = get_news_headlines("SPY")
if news:
    for n in news:
        st.markdown(f"- [{n['title']}]({n['link']}) ({n['publisher']}, {n['providerPublishTime']})")
else:
    st.info("뉴스 헤드라인을 가져올 수 없습니다.")

# ======= 6. 섹터 ETF 성과 비교 =======
st.subheader("🟣 6. 주요 섹터 ETF 하루 성과 비교")
sector_perf = get_sector_performance(SECTOR_ETFS)
sector_perf_sorted = dict(sorted(sector_perf.items(), key=lambda x: x[1] if x[1] is not None else -999, reverse=True))
fig_sector = px.bar(
    x=list(sector_perf_sorted.keys()),
    y=[v*100 if pd.notnull(v) else np.nan for v in sector_perf_sorted.values()],
    labels={'x':'섹터', 'y':'전일수익률(%)'},
    color=[v if v is not None else 0 for v in sector_perf_sorted.values()],
    color_continuous_scale='RdYlGn'
)
fig_sector.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig_sector, use_container_width=True)