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

# ----------------- 사이드바 디자인 개선 -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/2x/search.png", width=90)
    st.markdown("<h2 style='color:#008B8B; text-align:center;'>글로벌 대시보드 설정</h2>", unsafe_allow_html=True)
    st.markdown("---")
    idx_months = st.slider("📅 주요 주가지수 Normalized 기간 (개월)", 3, 36, 6, help="주요 주가지수 Normalized 수익률의 기간입니다")
    sector_months = st.slider("🏢 섹터 Normalized 기간 (개월)", 3, 36, 6, help="섹터별 Normalized 수익률의 기간입니다")
    style_months = st.slider("🌈 스타일 ETF Normalized 기간 (개월)", 3, 36, 6, help="스타일ETF Normalized 수익률의 기간입니다")
    news_cnt = st.slider("📰 뉴스 헤드라인 개수 (티커별)", 1, 5, 3)
    st.markdown("---")
    st.markdown("<small style='color:#888'>Made by parksuk1991</small>", unsafe_allow_html=True)

st.title("🌐 글로벌 시황 대시보드")
st.markdown("#### 전일 시장 데이터 및 다양한 기간별 성과 확인")

# =========== 자산 정의 ================
STOCK_ETFS = {
    'S&P 500 (SPY)': 'SPY',
    'NASDAQ 100 (QQQ)': 'QQQ',
    'MSCI ACWI (ACWI)': 'ACWI',
    '유럽(Europe, VGK)': 'VGK',
    '중국(China, MCHI)': 'MCHI',
    '일본(Japan, EWJ)': 'EWJ',
    '한국(KOSPI, EWY)': 'EWY',
    '인도(INDIA, INDA)': 'INDA',
    '영국(UK, EWU)': 'EWU',
    '브라질(Brazil, EWZ)': 'EWZ',
    '캐나다(Canada, EWC)': 'EWC'
}
BOND_ETFS = {
    '미국 장기국채(TLT)': 'TLT',
    '미국 단기국채(SHY)': 'SHY',
    '미국 IG회사채(LQD)': 'LQD',
    '신흥국채(EMB)': 'EMB',
    '미국 하이일드(HYG)': 'HYG',
    '미국 물가연동(TIP)': 'TIP',
    '미국 단기회사채(VCSH)': 'VCSH',
    '글로벌국채(BNDX)': 'BNDX',
    '미국 국채(BND)': 'BND',
    '단기국채(SPTS)': 'SPTS'
}
CURRENCY = {
    'USD/KRW': 'KRW=X',
    'USD/EUR': 'EURUSD=X',
    'USD/JPY': 'JPY=X'
}
CRYPTO = {
    '비트코인(BTC-USD)': 'BTC-USD',
    '이더리움(ETH-USD)': 'ETH-USD',
    '솔라나(SOL-USD)': 'SOL-USD',
    '리플(XRP-USD)': 'XRP-USD',
    '폴리곤(MATIC-USD)': 'MATIC-USD'
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

# === 정확한 기준일별 수익률 계산 함수 ===
def get_perf_table_precise(label2ticker, ref_date=None):
    """
    ref_date: datetime.date. None이면 오늘 날짜 기준, 전일 종가(마지막 거래일 종가) 기준.
    """
    tickers = list(label2ticker.values())
    labels = list(label2ticker.keys())

    # 기준일 계산 (오늘이 월요일이면 직전 영업일인 금요일로 맞춤)
    if ref_date is None:
        ref_date = datetime.now().date()
    # 야후 파이낸스는 시차 때문에 실제 마지막 종가가 1~2일 전일 수도 있으므로, 넉넉히 3년+14일치 다운로드
    start = ref_date - timedelta(days=3*365+14)
    end = ref_date + timedelta(days=1)  # inclusive

    # 데이터 다운로드 및 결측치 처리
    df = yf.download(tickers, start=start, end=end, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    df = df[tickers]  # enforce order

    # 실제 마지막 거래일 찾기 (주로 ref_date 기준 직전 영업일)
    last_trade_date = df.index[-1].date()
    # 직전 영업일이 ref_date보다 크면, ref_date와 같거나 작은 마지막 거래일 사용
    if last_trade_date > ref_date:
        last_trade_date = df.index[df.index.date <= ref_date][-1].date()
    # 기준 종가
    last_idx = df.index[df.index.date == last_trade_date][0]

    # 기준일별 offset
    periods = {
        '1D': 1,
        '1W': 5,    # 1주일 전 영업일 (5 영업일 전)
        'MTD': 'mtd',  # 이번달 첫 영업일
        '1M': 21,   # 21 영업일 전
        '3M': 63,   # 63 영업일 전
        '6M': 126,  # 126 영업일 전
        'YTD': 'ytd',  # 올해 첫 영업일
        '1Y': 252,  # 252 영업일 전
        '3Y': 756   # 756 영업일 전
    }

    results = []
    for i, (label, ticker) in enumerate(label2ticker.items()):
        row = {'자산명': label}
        series = df[ticker].dropna()
        if last_idx not in series.index:
            # 해당 자산은 마지막 거래일에 데이터 없음
            row['현재값'] = np.nan
            for k in periods: row[k] = np.nan
            results.append(row)
            continue
        curr_val = series.loc[last_idx]
        row['현재값'] = curr_val
        for k, val in periods.items():
            base = None
            try:
                if val == 'mtd':
                    # 월초 첫 영업일 종가
                    this_month = last_trade_date.month
                    this_year = last_trade_date.year
                    m_idx = series.index[(series.index.month == this_month) & (series.index.year == this_year)][0]
                    base = series.loc[m_idx]
                elif val == 'ytd':
                    this_year = last_trade_date.year
                    y_idx = series.index[(series.index.year == this_year)][0]
                    base = series.loc[y_idx]
                elif k == '1D':
                    # 1 영업일 전 종가
                    idx = series.index.get_loc(last_idx)
                    if idx >= 1:
                        base = series.iloc[idx-1]
                    else:
                        base = np.nan
                else:
                    # N 영업일 전 종가
                    idx = series.index.get_loc(last_idx)
                    if idx >= val:
                        base = series.iloc[idx-val]
                    else:
                        base = series.iloc[0]
                if base is not None and not np.isnan(base) and base != 0:
                    row[k] = (curr_val/base-1)*100
                else:
                    row[k] = np.nan
            except Exception:
                row[k] = np.nan
        results.append(row)

    df_r = pd.DataFrame(results)
    # 수익률 소수점 둘째자리까지 표시
    for col in ['1D', '1W', 'MTD', '1M', '3M', '6M', 'YTD', '1Y', '3Y']:
        df_r[col] = df_r[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    df_r['현재값'] = df_r['현재값'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
    return df_r

def get_normalized_prices(label2ticker, months=6):
    tickers = list(label2ticker.values())
    end = datetime.now().date()
    start = end - timedelta(days=months*31)
    df = yf.download(tickers, start=start, end=end + timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    df = df[tickers]
    norm_df = df / df.iloc[0] * 100
    norm_df.columns = [k for k in label2ticker]
    return norm_df

def get_news_headlines(tickers, limit=3):
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

# ====== 성과 테이블 스타일링: 수익률 양수 빨간색, 음수 파란색 ======
def colorize_return(val):
    try:
        v = float(str(val).replace("%", ""))
    except Exception:
        return ""
    if v > 0:
        return "color: red;"
    elif v < 0:
        return "color: blue;"
    else:
        return ""

def style_perf_table(df, perf_cols):
    # perf_cols: ['1D', ...]
    styled = df.copy()
    return df.style.applymap(colorize_return, subset=perf_cols)

# =========== MAIN BUTTON ===========
if st.button("전일 시장 Update", type="primary"):
    with st.spinner("데이터 불러오는 중..."):
        st.subheader("📊 주식시장 성과")
        stock_perf = get_perf_table_precise(STOCK_ETFS)
        perf_cols = ['1D','1W','MTD','1M','3M','6M','YTD','1Y','3Y']
        st.dataframe(
            style_perf_table(stock_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=470
        )

        st.subheader("📊 채권시장 성과")
        bond_perf = get_perf_table_precise(BOND_ETFS)
        st.dataframe(
            style_perf_table(bond_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=420
        )

        st.subheader("📊 통화시장 성과")
        curr_perf = get_perf_table_precise(CURRENCY)
        st.dataframe(
            style_perf_table(curr_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=200
        )

        st.subheader("📊 암호화폐 성과")
        crypto_perf = get_perf_table_precise(CRYPTO)
        st.dataframe(
            style_perf_table(crypto_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=180
        )

        st.subheader("📊 스타일 ETF 성과")
        style_perf = get_perf_table_precise(STYLE_ETFS)
        st.dataframe(
            style_perf_table(style_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=250
        )

        st.subheader("📊 섹터 ETF 성과")
        sector_perf = get_perf_table_precise(SECTOR_ETFS)
        sector_height = int(43 * sector_perf.shape[0] + 42)
        st.dataframe(
            style_perf_table(sector_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=sector_height
        )

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
        headline_tickers = list(STOCK_ETFS.values())[:2] + list(SECTOR_ETFS.values())[:2] + ['BTC-USD', 'ETH-USD']
        news_df = get_news_headlines(headline_tickers, news_cnt)
        if not news_df.empty:
            for _, row in news_df.iterrows():
                st.markdown(f"- **[{row['티커']}]** {row['일자']}: {row['헤드라인']}")
        else:
            st.info("뉴스 헤드라인을 가져올 수 없습니다.")

else:
    st.info("왼쪽 설정 후 '전일 시장 Update' 버튼을 눌러주세요.")
