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
    results = []
    for label, ticker in label2ticker.items():
        row = {}
        row['자산명'] = label
        try:
            series = df[ticker]
        except Exception:
            row['현재값'] = np.nan
            for k in ['1D','1W','MTD','1M','3M','6M','YTD','1Y','3Y']:
                row[k] = np.nan
            results.append(row)
            continue

        row['현재값'] = series.iloc[-1] if not np.isnan(series.iloc[-1]) else None

        # 보조 함수: 기준일 찾기 (영업일 존재 보장)
        def find_nearest_index(series_idx, target_date):
            # target_date 이전 or 같은 날 중 가장 마지막 영업일
            candidates = series_idx[series_idx <= target_date]
            if len(candidates) > 0:
                return candidates[-1]
            # target_date 이후 영업일
            candidates = series_idx[series_idx >= target_date]
            if len(candidates) > 0:
                return candidates[0]
            return series_idx[0]

        # 1D: 마지막 두 영업일
        try:
            if len(series) >= 2:
                base = series.iloc[-2]
            else:
                base = series.iloc[0]
            row['1D'] = (series.iloc[-1]/base-1)*100 if base != 0 else np.nan
        except Exception:
            row['1D'] = np.nan

        # 1W: 7일 전(달력 기준) 가장 가까운 영업일
        try:
            base_date = last - pd.Timedelta(days=7)
            idx = find_nearest_index(series.index, base_date)
            base = series.loc[idx]
            row['1W'] = (series.iloc[-1]/base-1)*100 if base != 0 else np.nan
        except Exception:
            row['1W'] = np.nan

        # MTD: 월초 첫 영업일
        try:
            mtd_idx = series.index[(series.index.month == last.month) & (series.index.year == last.year)][0]
            base = series.loc[mtd_idx]
            row['MTD'] = (series.iloc[-1]/base-1)*100 if base != 0 else np.nan
        except Exception:
            row['MTD'] = np.nan

        # 1M: 1달 전(달력 기준) 가장 가까운 영업일
        try:
            base_date = last - pd.DateOffset(months=1)
            idx = find_nearest_index(series.index, base_date)
            base = series.loc[idx]
            row['1M'] = (series.iloc[-1]/base-1)*100 if base != 0 else np.nan
        except Exception:
            row['1M'] = np.nan

        # 3M, 6M, 1Y, 3Y
        for col, offset in zip(['3M','6M','1Y','3Y'], [3,6,12,36]):
            try:
                base_date = last - pd.DateOffset(months=offset)
                idx = find_nearest_index(series.index, base_date)
                base = series.loc[idx]
                row[col] = (series.iloc[-1]/base-1)*100 if base != 0 else np.nan
            except Exception:
                row[col] = np.nan

        # YTD: 연초 첫 영업일
        try:
            ytd_idx = series.index[(series.index.year == last.year)][0]
            base = series.loc[ytd_idx]
            row['YTD'] = (series.iloc[-1]/base-1)*100 if base != 0 else np.nan
        except Exception:
            row['YTD'] = np.nan

        results.append(row)

    df_r = pd.DataFrame(results)
    # 소수점 둘째자리까지 % 형식
    for col in ['1D', '1W', 'MTD', '1M', '3M', '6M', 'YTD', '1Y', '3Y']:
        df_r[col] = df_r[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    df_r['현재값'] = df_r['현재값'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
    return df_r

# 수익률 양수 빨간색, 음수 파란색 스타일러
def make_return_color_styler(df, cols):
    def stylefn(x):
        result = []
        for col in x.index:
            if col not in cols:
                result.append("")
                continue
            val = x[col]
            try:
                num = float(val.replace("%", ""))
                if num > 0:
                    result.append("color:red;")
                elif num < 0:
                    result.append("color:blue;")
                else:
                    result.append("")
            except Exception:
                result.append("")
        return result
    return stylefn

# Top/Bottom 10 함수에서 차트용 데이터 별도 추출 및 캐시 적용
@st.cache_data(ttl=3600)
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
    # 차트용 데이터 프레임도 여기서 만듦
    top10_chart = top10[['Ticker', '전일수익률(%)']].copy()
    bottom10_chart = bottom10[['Ticker', '전일수익률(%)']].copy()
    return top10, bottom10, top10_chart, bottom10_chart

def make_vertical_styler_colwise_minmax(df, cols):
    # 각 열별로 min/max 기준으로 컬러를 정하고 모든 행에 적용(열별 독립적 히트맵)
    def stylefn(x):
        result = []
        for col in x.index:
            if col not in cols:
                result.append("")
                continue
            try:
                v = float(x[col].replace("%",""))
            except Exception:
                result.append("")
                continue
            col_vals = df[col].str.replace("%","").astype(float)
            minv = col_vals.min(skipna=True)
            maxv = col_vals.max(skipna=True)
            rng = maxv - minv
            if pd.isnull(v):
                result.append("")
                continue
            # 빨강(최저)~흰~초록(최고) linear mapping
            if rng == 0:
                ratio = 0.5
            else:
                ratio = (v - minv) / rng
            # green for high, red for low, white for center
            r = int((1 - ratio) * 255 + ratio * 220)
            g = int((1 - ratio) * 255 + ratio * 255)
            b = int((1 - ratio) * 255 + ratio * 220)
            # slightly more vivid
            result.append(f"background-color: rgb({r},{g},{b},0.7)")
        return result
    return stylefn

if st.button("전일 시장 Update", type="primary"):
    with st.spinner("데이터 불러오는 중..."):
        st.subheader("📊 주식시장 성과")
        stock_perf = get_perf_table(STOCK_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        perf_cols = ['1D','1W','MTD','1M','3M','6M','YTD','1Y','3Y']
        st.dataframe(
            stock_perf.set_index('자산명').style.apply(make_return_color_styler(stock_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=470
        )

        st.subheader("📊 채권시장 성과")
        bond_perf = get_perf_table(BOND_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(
            bond_perf.set_index('자산명').style.apply(make_return_color_styler(bond_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=420
        )

        st.subheader("📊 통화시장 성과")
        curr_perf = get_perf_table(CURRENCY, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(
            curr_perf.set_index('자산명').style.apply(make_return_color_styler(curr_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=200
        )

        st.subheader("📊 암호화폐 성과")
        crypto_perf = get_perf_table(CRYPTO, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(
            crypto_perf.set_index('자산명').style.apply(make_return_color_styler(crypto_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=180
        )

        st.subheader("📊 스타일 ETF 성과")
        style_perf = get_perf_table(STYLE_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        st.dataframe(
            style_perf.set_index('자산명').style.apply(make_return_color_styler(style_perf.set_index('자산명'), perf_cols), axis=1),
            use_container_width=True, height=250
        )

        st.subheader("📊 섹터 ETF 성과")
        sector_perf = get_perf_table(SECTOR_ETFS, datetime.now().date() - timedelta(days=1100), datetime.now().date())
        sector_height = int(43 * sector_perf.shape[0] + 42)
        st.dataframe(
            sector_perf.set_index('자산명').style.apply(make_return_color_styler(sector_perf.set_index('자산명'), perf_cols), axis=1),
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
        top10, bottom10, top10_chart, bottom10_chart = get_sp500_top_bottom_movers()
        if top10.empty or bottom10.empty:
            st.info("S&P500 Top/Bottom movers를 불러올 수 없습니다. 인터넷 연결 및 lxml 패키지를 확인하세요.")
        else:
            st.markdown("**Top 10 상승**")
            st.dataframe(top10.set_index('Ticker')[['종목명', '섹터명', '종가', '전일수익률(%)', 'Volume', '시가총액']].style
                         .apply(make_return_color_styler(top10.set_index('Ticker'), ['전일수익률(%)']), axis=1),
                         use_container_width=True, height=380)
            st.markdown("**Top 10 하락**")
            st.dataframe(bottom10.set_index('Ticker')[['종목명', '섹터명', '종가', '전일수익률(%)', 'Volume', '시가총액']].style
                         .apply(make_return_color_styler(bottom10.set_index('Ticker'), ['전일수익률(%)']), axis=1),
                         use_container_width=True, height=380)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_top = px.bar(top10_chart, x='Ticker', y='전일수익률(%)', text='전일수익률(%)',
                                 title="Top10 상승폭(%)", color='전일수익률(%)', color_continuous_scale='Teal')
                fig_top.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_top.update_layout(xaxis_title='티커', yaxis_title='전일수익률(%)', template='plotly_white', height=500)
                st.plotly_chart(fig_top, use_container_width=True)
            with col2:
                fig_bot = px.bar(bottom10_chart, x='Ticker', y='전일수익률(%)', text='전일수익률(%)',
                                 title="Top10 하락폭(%)", color='전일수익률(%)', color_continuous_scale='OrRd')
                fig_bot.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_bot.update_layout(xaxis_title='티커', yaxis_title='전일수익률(%)', template='plotly_white', height=500)
                st.plotly_chart(fig_bot, use_container_width=True)
else:
    st.info("왼쪽 설정 후 '전일 시장 Update' 버튼을 눌러주세요.")
