import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import requests
from PIL import Image
from io import BytesIO

# lxml ImportError 방지
try:
    import lxml
except ImportError:
    st.error("lxml 패키지가 필요합니다. requirements.txt에 lxml을 추가하세요.")

st.set_page_config(
    page_title="글로벌 시장 대시보드",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 글로벌 시장 대시보드")

# -------------------- 상단 레이아웃: 제목+설명 / 이미지+크레딧 ---------------------
col_title, col_img = st.columns([9, 1])
with col_title:
    st.markdown("#### 전일 및 기간별 주요 시장 성과")
with col_img:
    # 닐 암스트롱 달착륙 사진(퍼블릭 도메인, NASA) - 다운로드 실패시 대체 아이콘 제공
    image_url = "https://cdn.theatlantic.com/thumbor/gjwD-uCiv0sHowRxQrQgL9b3Shk=/900x638/media/img/photo/2019/07/apollo-11-moon-landing-photos-50-ye/a01_40-5903/original.jpg"
    fallback_icon = "https://cdn-icons-png.flaticon.com/512/3211/3211357.png"  # 우주인 아이콘 (flaticon)
    img_displayed = False
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        st.image(img, width=110, caption=None)
        img_displayed = True
    except Exception:
        try:
            response = requests.get(fallback_icon, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            st.image(img, width=90, caption=None)
            img_displayed = True
        except Exception:
            st.info("이미지를 불러올 수 없습니다.")
    st.markdown("<small style='color:#888'>Made by parksuk1991</small>", unsafe_allow_html=True)

# ===================== 차트 구간 설정 및 전일 시장 업데이트 버튼 =====================
st.markdown("---")
st.markdown("##### 📈 차트 구간 설정")

# 슬라이더와 버튼, 안내문구를 수평 배치, 버튼 가로폭 좁고 세로폭은 슬라이더와 맞춤
col_slider, col_warn, col_btn = st.columns([3, 6, 1])

with col_slider:
    # 슬라이더의 세로 크기와 일치시키기 위해 placeholder 사용
    slider_placeholder = st.empty()
    normalized_months = slider_placeholder.slider(
        "차트 수익률 기간 설정 (N개월, 모든 차트에 동일 적용)",
        3, 36, 12,
        help="모든 차트에 적용될 정규화 수익률 기간입니다.",
        key="norm_months_slider"
    )


with col_btn:
    # 버튼을 슬라이더와 같은 세로 높이로 맞추기 위해 margin/padding/height 스타일 조정
    button_style = """
    <style>
    div.stButton > button {
        width: 500px !important;
        height: 120px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        white-space: pre-line !important;
        margin-top: 12px !important;
        margin-bottom: 10px !important;
        padding-top: 20px !important;
        padding-bottom: 20px !important;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    update_clicked = st.button("전일 시장\nUpdate", key="main_update_btn")

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

def get_perf_table_precise(label2ticker, ref_date=None):
    tickers = list(label2ticker.values())
    labels = list(label2ticker.keys())

    if ref_date is None:
        ref_date = datetime.now().date()
    start = ref_date - timedelta(days=3*365+14)
    end = ref_date + timedelta(days=1)  # inclusive

    df = yf.download(tickers, start=start, end=end, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()
    df = df[tickers]
    last_trade_date = df.index[-1].date()
    if last_trade_date > ref_date:
        last_trade_date = df.index[df.index.date <= ref_date][-1].date()
    last_idx = df.index[df.index.date == last_trade_date][0]

    periods = {
        '1D': 1,
        '1W': 5,
        'MTD': 'mtd',
        '1M': 21,
        '3M': 63,
        '6M': 126,
        'YTD': 'ytd',
        '1Y': 252,
        '3Y': 756
    }

    results = []
    for i, (label, ticker) in enumerate(label2ticker.items()):
        row = {'자산명': label}
        series = df[ticker].dropna()
        if last_idx not in series.index:
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
                    this_month = last_trade_date.month
                    this_year = last_trade_date.year
                    m_idx = series.index[(series.index.month == this_month) & (series.index.year == this_year)][0]
                    base = series.loc[m_idx]
                elif val == 'ytd':
                    this_year = last_trade_date.year
                    y_idx = series.index[(series.index.year == this_year)][0]
                    base = series.loc[y_idx]
                elif k == '1D':
                    idx = series.index.get_loc(last_idx)
                    if idx >= 1:
                        base = series.iloc[idx-1]
                    else:
                        base = np.nan
                else:
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
    return df.style.applymap(colorize_return, subset=perf_cols)

# =========== MAIN BUTTON ===========
if update_clicked:
    with st.spinner("데이터 불러오는 중..."):
        st.subheader("📊 주식시장")
        stock_perf = get_perf_table_precise(STOCK_ETFS)
        perf_cols = ['1D','1W','MTD','1M','3M','6M','YTD','1Y','3Y']
        st.dataframe(
            style_perf_table(stock_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=470
        )

        st.subheader("📊 채권시장")
        bond_perf = get_perf_table_precise(BOND_ETFS)
        st.dataframe(
            style_perf_table(bond_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=420
        )

        st.subheader("📊 통화")
        curr_perf = get_perf_table_precise(CURRENCY)
        st.dataframe(
            style_perf_table(curr_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=200
        )

        st.subheader("📊 암호화폐")
        crypto_perf = get_perf_table_precise(CRYPTO)
        st.dataframe(
            style_perf_table(crypto_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=180
        )

        st.subheader("📊 스타일 ETF")
        style_perf = get_perf_table_precise(STYLE_ETFS)
        st.dataframe(
            style_perf_table(style_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=250
        )

        st.subheader("📊 섹터 ETF")
        sector_perf = get_perf_table_precise(SECTOR_ETFS)
        sector_height = int(43 * sector_perf.shape[0] + 42)
        st.dataframe(
            style_perf_table(sector_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=sector_height
        )

        # ---------- Normalized 차트 구간 설정 아래에 위치 ----------
        st.subheader(f"📈 주요 주가지수 수익률 (최근 {normalized_months}개월)")
        norm_idx = get_normalized_prices(STOCK_ETFS, months=normalized_months)
        fig1 = go.Figure()
        for col in norm_idx.columns:
            fig1.add_trace(go.Scatter(x=norm_idx.index, y=norm_idx[col], mode='lines', name=col))
        fig1.update_layout(
            xaxis_title="날짜", yaxis_title="100 기준 누적수익률(%)",
            template="plotly_dark", height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(f"📈 섹터 ETF 수익률 (최근 {normalized_months}개월)")
        norm_sector = get_normalized_prices(SECTOR_ETFS, months=normalized_months)
        fig2 = go.Figure()
        for col in norm_sector.columns:
            fig2.add_trace(go.Scatter(x=norm_sector.index, y=norm_sector[col], mode='lines', name=col))
        fig2.update_layout(
            xaxis_title="날짜", yaxis_title="100 기준 누적수익률(%)",
            template="plotly_dark", height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(f"📈 스타일 ETF 수익률 (최근 {normalized_months}개월)")
        norm_style = get_normalized_prices(STYLE_ETFS, months=normalized_months)
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
        news_df = get_news_headlines(headline_tickers, 3)
        if not news_df.empty:
            for _, row in news_df.iterrows():
                st.markdown(f"- **[{row['티커']}]** {row['일자']}: {row['헤드라인']}")
        else:
            st.info("뉴스 헤드라인을 가져올 수 없습니다.")

else:
    st.warning("⚠️위에서 차트 수익률 기간 설정 후 '전일 시장 Update' 버튼을 눌러주세요!")
