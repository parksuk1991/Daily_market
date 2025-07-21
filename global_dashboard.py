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
    st.image("https://img.icons8.com/color/96/000000/worldwide-location.png", width=90)
    st.markdown("<h2 style='color:#008B8B; text-align:center;'>설정</h2>", unsafe_allow_html=True)
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

def get_perf_table(label2ticker, start, end):
    tickers = list(label2ticker.values())
    labels = list(label2ticker.keys())
    today = end
    first_date = today - timedelta(days=365*3+31)
    df = yf.download(tickers, start=first_date, end=today+timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
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
                elif k == '1W':
                    # 1주일 전의 실제 영업일(5일 전이 데이터가 없을 수 있음) 보정
                    idx = series.index.get_loc(last)
                    found = False
                    for back in range(1, 8):  # 최대 7영업일 전까지 시도
                        if idx - back >= 0:
                            base = series.iloc[idx - back]
                            if not np.isnan(base):
                                found = True
                                break
                    if not found:
                        base = series.iloc[0]
                else:
                    if val is not None and len(series) > val:
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
    df_r = pd.DataFrame(results)
    # 소수점 둘째자리까지 % 형식
    for col in ['1D', '1W', 'MTD', '1M', '3M', '6M', 'YTD', '1Y', '3Y']:
        df_r[col] = df_r[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    df_r['현재값'] = df_r['현재값'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
    return df_r

def get_perf_table_numeric(label2ticker, start, end):
    tickers = list(label2ticker.values())
    today = end
    first_date = today - timedelta(days=365*3+31)
    df = yf.download(tickers, start=first_date, end=today+timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    df = df[tickers]
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
        try:
            series = df[ticker]
        except Exception:
            for k in periods: row[k] = np.nan
            results.append(row)
            continue
        curr_val = series.iloc[-1] if not np.isnan(series.iloc[-1]) else None
        for k, val in periods.items():
            try:
                if k == 'MTD':
                    base = series[:last][series.index.month == last.month].iloc[0]
                elif k == 'YTD':
                    base = series[:last][series.index.year == last.year].iloc[0]
                elif k == '1W':
                    idx = series.index.get_loc(last)
                    found = False
                    for back in range(1, 8):
                        if idx - back >= 0:
                            base = series.iloc[idx - back]
                            if not np.isnan(base):
                                found = True
                                break
                    if not found:
                        base = series.iloc[0]
                else:
                    if val is not None and len(series) > val:
                        base = series.iloc[-val-1]
                    else:
                        base = series.iloc[0]
                if base and curr_val:
                    row[k] = (curr_val/base - 1) * 100
                else:
                    row[k] = np.nan
            except Exception:
                row[k] = np.nan
        results.append(row)
    return pd.DataFrame(results, index=list(label2ticker.keys()))

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

def get_sp500_top_bottom_movers():
    try:
        stocks = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    except Exception as e:
        st.error("S&P500 구성종목 목록을 가져오려면 인터넷 및 lxml 패키지가 필요합니다.")
        return pd.DataFrame(), pd.DataFrame()
    tickers = stocks['Symbol'].unique().tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    name_map = dict(zip(stocks['Symbol'].str.replace('.', '-'), stocks['Security']))
    sector_map = dict(zip(stocks['Symbol'].str.replace('.', '-'), stocks['GICS Sector']))
    try:
        df = yf.download(tickers, period="5d", interval="1d", group_by="ticker", progress=False, auto_adjust=True)
    except Exception as e:
        st.error("야후 파이낸스에서 S&P500 가격 데이터를 읽는 데 실패했습니다.")
        return pd.DataFrame(), pd.DataFrame()
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
                "시가총액": mktcap,
                "종목명": name_map.get(t, ""),
                "섹터명": sector_map.get(t, "")
            })
        except Exception:
            continue
    movers = pd.DataFrame(results)
    if movers.empty or "전일수익률(%)" not in movers.columns:
        return pd.DataFrame(), pd.DataFrame()
    movers = movers.dropna(subset=["전일수익률(%)"])
    movers['전일수익률(%)'] = movers['전일수익률(%)'].round(2)
    top10 = movers.sort_values("전일수익률(%)", ascending=False).head(10)
    bottom10 = movers.sort_values("전일수익률(%)", ascending=True).head(10)
    return top10, bottom10

def colorize_vertical_heatmap(df, cols):
    # 각 컬럼별로 min/max 기준으로 컬러를 정하고 모든 행에 적용
    styled = df.copy()
    for col in cols:
        try:
            vals = df[col].str.replace("%", "").astype(float)
        except Exception:
            vals = pd.Series([np.nan]*len(df))
        minv = vals.min(skipna=True)
        maxv = vals.max(skipna=True)
        rng = maxv - minv
        colors = []
        for v in vals:
            if np.isnan(v):
                colors.append("")
                continue
            # 빨강(최저)~흰~초록(최고) linear mapping
            if rng == 0:
                ratio = 0.5
            else:
                ratio = (v - minv) / rng
            if v >= 0:
                r, g, b = int((1-ratio)*220), int(180 + ratio*75), int((1-ratio)*220)
            else:
                r, g, b = int(220 + ratio*35), int((1-ratio)*180), int((1-ratio)*220)
            r = min(max(r, 0), 255)
            g = min(max(g, 0), 255)
            b = min(max(b, 0), 255)
            colors.append(f"background-color: rgb({r},{g},{b},0.6)")
        styled[col] = colors
    return styled

def make_vertical_styler(df, cols):
    # streamlit styler용: 각 열별로 색상 반환
    def stylefn(x):
        result = []
        for col in x.index:
            if col not in cols:
                result.append("")
                continue
            try:
                vals = x[col].replace("%","")
                v = float(vals)
            except Exception:
                result.append("")
                continue
            # 컬럼별 min/max
            col_vals = df[col].str.replace("%","").astype(float)
            minv = col_vals.min(skipna=True)
            maxv = col_vals.max(skipna=True)
            rng = maxv - minv
            if rng == 0:
                ratio = 0.5
            else:
                ratio = (v - minv) / rng
            if v >= 0:
                r, g, b = int((1-ratio)*220), int(180 + ratio*75), int((1-ratio)*220)
            else:
                r, g, b = int(220 + ratio*35), int((1-ratio)*180), int((1-ratio)*220)
            r = min(max(r, 0), 255)
            g = min(max(g, 0), 255)
            b = min(max(b, 0), 255)
            result.append(f"background-color: rgb({r},{g},{b},0.6)")
        return result
    return stylefn

if st.button("전일 시장 Update", type="primary"):
    with st.spinner("데이터 불러오는 중..."):
        st.subheader("📊 주식시장 성과")
        stock_perf = get_perf_table(STOCK_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        perf_cols = ['1D','1W','MTD','1M','3M','6M','YTD','1Y','3Y']
        # 각 열별로 vertical heatmap 적용(styler로)
        st.dataframe(
            stock_perf.set_index('자산명').style.apply(make_vertical_styler(stock_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=470
        )

        st.subheader("📊 채권시장 성과")
        bond_perf = get_perf_table(BOND_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(
            bond_perf.set_index('자산명').style.apply(make_vertical_styler(bond_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=420
        )

        st.subheader("📊 통화시장 성과")
        curr_perf = get_perf_table(CURRENCY, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(
            curr_perf.set_index('자산명').style.apply(make_vertical_styler(curr_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=200
        )

        st.subheader("📊 암호화폐 성과")
        crypto_perf = get_perf_table(CRYPTO, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(
            crypto_perf.set_index('자산명').style.apply(make_vertical_styler(crypto_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=180
        )

        st.subheader("📊 스타일 ETF 성과")
        style_perf = get_perf_table(STYLE_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(
            style_perf.set_index('자산명').style.apply(make_vertical_styler(style_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=250
        )

        st.subheader("📊 섹터 ETF 성과")
        sector_perf = get_perf_table(SECTOR_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        # 전체 행이 항상 한 화면에 다 보이도록 height 자동 조정
        sector_height = int(43 * sector_perf.shape[0] + 42)
        st.dataframe(
            sector_perf.set_index('자산명').style.apply(make_vertical_styler(sector_perf.set_index('자산명'), perf_cols), axis=1),
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

        st.subheader("🚀 미국 시장 전일 상승 Top 10 / 하락 Top 10 (S&P500 기준)")
        top10, bottom10 = get_sp500_top_bottom_movers()
        if top10.empty or bottom10.empty:
            st.info("S&P500 Top/Bottom movers를 불러올 수 없습니다. 인터넷 연결 및 lxml 패키지를 확인하세요.")
        else:
            st.markdown("**Top 10 상승**")
            st.dataframe(top10.set_index('Ticker')[['종목명', '섹터명', '종가', '전일수익률(%)', 'Volume', '시가총액']], use_container_width=True, height=380)
            st.markdown("**Top 10 하락**")
            st.dataframe(bottom10.set_index('Ticker')[['종목명', '섹터명', '종가', '전일수익률(%)', 'Volume', '시가총액']], use_container_width=True, height=380)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_top = px.bar(top10, x='Ticker', y='전일수익률(%)', text='전일수익률(%)', color='전일수익률(%)',
                                 hover_data=['종목명', '섹터명', '종가', 'Volume', '시가총액'],
                                 title="Top10 상승폭(%)", color_continuous_scale='Teal')
                fig_top.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_top.update_layout(xaxis_title='티커', yaxis_title='전일수익률(%)', template='plotly_white', height=340)
                st.plotly_chart(fig_top, use_container_width=True)
            with col2:
                fig_bot = px.bar(bottom10, x='Ticker', y='전일수익률(%)', text='전일수익률(%)', color='전일수익률(%)',
                                 hover_data=['종목명', '섹터명', '종가', 'Volume', '시가총액'],
                                 title="Top10 하락폭(%)", color_continuous_scale='OrRd')
                fig_bot.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_bot.update_layout(xaxis_title='티커', yaxis_title='전일수익률(%)', template='plotly_white', height=340)
                st.plotly_chart(fig_bot, use_container_width=True)
else:
    st.info("왼쪽 설정 후 '전일 시장 Update' 버튼을 눌러주세요.")
