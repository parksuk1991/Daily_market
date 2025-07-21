import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# lxml ImportError 방지
try:
    import lxml
except ImportError:
    st.error("lxml 패키지가 필요합니다. requirements.txt에 lxml을 추가하세요.")

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
    'USD/JPY': 'JPY=X'
}
CRYPTO = {
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
STYLE_ETFS = {
    'Growth (SPYG)': 'SPYG',
    'Value (SPYV)': 'SPYV',
    'Momentum (MTUM)': 'MTUM',
    'Quality (QUAL)': 'QUAL',
    'Dividend (VIG)': 'VIG',
    'Low Volatility (USMV)': 'USMV'
}

def get_perf_table(label2ticker, start, end):
    # Download using ticker symbol, keep symbol as columns
    tickers = list(label2ticker.values())
    labels = list(label2ticker.keys())
    today = end
    first_date = today - timedelta(days=365*3+31)
    df = yf.download(tickers, start=first_date, end=today+timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    # Make sure columns are tickers (symbols)
    df = df[tickers]  # enforce order
    last = df.index[-1]
    periods = {
        '1D': 1,
        '1W': 5,
        'MTD': None,
        '1M': 21,
        '3M': 63,
        '6M': 126,
        'YTD': None,
        '1Y': 252,
        '3Y': 756
    }
    results = []
    for label, ticker in label2ticker.items():
        row = {}
        row['자산명'] = label
        try:
            series = df[ticker]
        except Exception:
            row['현재값'] = np.nan
            for k in periods: row[k] = np.nan
            results.append(row)
            continue
        row['현재값'] = series.iloc[-1] if not np.isnan(series.iloc[-1]) else None
        for k, val in periods.items():
            try:
                if k == 'MTD':
                    base = series[:last][series.index.month == last.month].iloc[0]
                elif k == 'YTD':
                    base = series[:last][series.index.year == last.year].iloc[0]
                else:
                    if len(series) > val:
                        base = series.iloc[-val-1]
                    else:
                        base = series.iloc[0]
                if base and row['현재값']:
                    row[k] = (row['현재값']/base - 1) * 100
                else:
                    row[k] = np.nan
            except Exception:
                row[k] = np.nan
        results.append(row)
    return pd.DataFrame(results)

def get_normalized_prices(label2ticker, months=6):
    # Download using ticker symbol, keep symbol as columns
    tickers = list(label2ticker.values())
    labels = list(label2ticker.keys())
    end = datetime.now().date()
    start = end - timedelta(days=months*31)
    df = yf.download(tickers, start=start, end=end + timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    df = df[tickers]
    norm_df = df / df.iloc[0] * 100
    # return DataFrame with label columns
    norm_df.columns = [k for k in label2ticker]
    return norm_df

def get_news_headlines(tickers, limit=3):
    # (동일)
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
    if not df.empty:
        df = df.sort_values('일자', ascending=False)
    return df

def get_sp500_top_bottom_movers():
    # S&P500 전체 티커/섹터/시총/볼륨/종가/변동률
    try:
        stocks = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    except Exception as e:
        st.error("S&P500 구성종목 목록을 가져오려면 인터넷 및 lxml 패키지가 필요합니다.")
        return pd.DataFrame(), pd.DataFrame()
    tickers = stocks['Symbol'].unique().tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    # 데이터 적재 (아래에서 볼륨/시가총액/변동률/종가 모두 추출)
    df = yf.download(tickers, period="5d", interval="1d", group_by="ticker", progress=False, auto_adjust=True)
    results = []
    for t in tickers:
        try:
            closes = df[t]['Close']
            vols = df[t]['Volume']
            last = closes.index[-1]
            prev = closes.index[-2]
            curr_close = closes.loc[last]
            prev_close = closes.loc[prev]
            ret = (curr_close/prev_close-1)*100 if prev_close else np.nan
            volume = vols.loc[last]
            yf_info = yf.Ticker(t).info
            mktcap = yf_info.get("marketCap", np.nan)
            results.append({
                "Ticker": t,
                "종가": curr_close,
                "전일수익률(%)": ret,
                "Volume": volume,
                "시가총액": mktcap
            })
        except Exception:
            continue
    movers = pd.DataFrame(results)
    movers = movers.dropna(subset=["전일수익률(%)"])
    top10 = movers.sort_values("전일수익률(%)", ascending=False).head(10)
    bottom10 = movers.sort_values("전일수익률(%)", ascending=True).head(10)
    return top10, bottom10

with st.sidebar:
    st.header("설정")
    idx_months = st.slider("주요 주가지수 Normalized 기간 (개월)", 3, 36, 6)
    sector_months = st.slider("섹터 Normalized 기간 (개월)", 3, 36, 6)
    style_months = st.slider("스타일 ETF Normalized 기간 (개월)", 3, 36, 6)
    news_cnt = st.slider("뉴스 헤드라인 개수 (티커별)", 1, 5, 3)

if st.button("전일 시장 Update", type="primary"):
    with st.spinner("데이터 불러오는 중..."):
        st.subheader("📊 주식시장 성과")
        stock_perf = get_perf_table(STOCK_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(stock_perf.set_index('자산명'), use_container_width=True, height=350)

        st.subheader("📊 채권시장 성과")
        bond_perf = get_perf_table(BOND_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(bond_perf.set_index('자산명'), use_container_width=True, height=250)

        st.subheader("📊 통화시장 성과")
        curr_perf = get_perf_table(CURRENCY, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(curr_perf.set_index('자산명'), use_container_width=True, height=200)

        st.subheader("📊 비트코인 성과")
        crypto_perf = get_perf_table(CRYPTO, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(crypto_perf.set_index('자산명'), use_container_width=True, height=80)

        st.subheader("📊 스타일 ETF 성과")
        style_perf = get_perf_table(STYLE_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(style_perf.set_index('자산명'), use_container_width=True, height=250)

        st.subheader(f"📈 주요 주가지수 수익률 (최근 {idx_months}개월)")
        norm_idx = get_normalized_prices(STOCK_ETFS, months=idx_months)
        fig1 = go.Figure()
        for col in norm_idx.columns:
            fig1.add_trace(go.Scatter(x=norm_idx.index, y=norm_idx[col], mode='lines', name=col))
        fig1.update_layout(
            xaxis_title="날짜", yaxis_title="100 기준 누적수익률(%)",
            template="plotly_dark", height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(f"📈 섹터 ETF 수익률 (최근 {sector_months}개월)")
        norm_sector = get_normalized_prices(SECTOR_ETFS, months=sector_months)
        fig2 = go.Figure()
        for col in norm_sector.columns:
            fig2.add_trace(go.Scatter(x=norm_sector.index, y=norm_sector[col], mode='lines', name=col))
        fig2.update_layout(
            xaxis_title="날짜", yaxis_title="100 기준 누적수익률(%)",
            template="plotly_dark", height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(f"📈 스타일 ETF 수익률 (최근 {style_months}개월)")
        norm_style = get_normalized_prices(STYLE_ETFS, months=style_months)
        fig3 = go.Figure()
        for col in norm_style.columns:
            fig3.add_trace(go.Scatter(x=norm_style.index, y=norm_style[col], mode='lines', name=col))
        fig3.update_layout(
            xaxis_title="날짜", yaxis_title="100 기준 누적수익률(%)",
            template="plotly_dark", height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("📰 최근 뉴스 헤드라인 (대표 티커 위주)")
        headline_tickers = list(STOCK_ETFS.values())[:2] + list(SECTOR_ETFS.values())[:2] + ['BTC-USD']
        news_df = get_news_headlines(headline_tickers, news_cnt)
        if not news_df.empty:
            for _, row in news_df.iterrows():
                st.markdown(f"- **[{row['티커']}]** {row['일자']}: {row['헤드라인']}")
        else:
            st.info("뉴스 헤드라인을 가져올 수 없습니다.")

        st.subheader("🚀 미국 시장 전일 상승 Top 10 / 하락 Top 10 (S&P500 기준)")
        top10, bottom10 = get_sp500_top_bottom_movers()
        if top10.empty or bottom10.empty:
            st.info("S&P500 Top/Bottom movers를 불러올 수 없습니다. 인터넷 연결 및 lxml 패키지를 확인하세요.")
        else:
            st.markdown("**Top 10 상승**")
            st.dataframe(top10.set_index('Ticker'), use_container_width=True, height=320)
            st.markdown("**Top 10 하락**")
            st.dataframe(bottom10.set_index('Ticker'), use_container_width=True, height=320)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.bar(top10, x='종가', y='전일수익률(%)', text='전일수익률(%)', hover_data=['Volume', '시가총액'],
                                       orientation='h', title="Top10 상승폭(%)", color='전일수익률(%)', color_continuous_scale='Teal'), use_container_width=True)
            with col2:
                st.plotly_chart(px.bar(bottom10, x='종가', y='전일수익률(%)', text='전일수익률(%)', hover_data=['Volume', '시가총액'],
                                       orientation='h', title="Top10 하락폭(%)", color='전일수익률(%)', color_continuous_scale='OrRd'), use_container_width=True)
else:
    st.info("왼쪽 설정 후 '전일 시장 Update' 버튼을 눌러주세요.")
