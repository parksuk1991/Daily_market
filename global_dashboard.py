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

# -------------------- 상단 레이아웃: 제목+설명 / 이미지+크레딧 ---------------------
col_title, col_img_credit = st.columns([7, 1])
with col_title:
    st.title("🌐 글로벌 주요 시장 모니터링")
with col_img_credit:
    image_url = "https://cdn.theatlantic.com/thumbor/gjwD-uCiv0sHowRxQrQgL9b3Shk=/900x638/media/img/photo/2019/07/apollo-11-moon-landing-photos-50-ye/a01_40-5903/original.jpg"
    fallback_icon = "https://cdn-icons-png.flaticon.com/512/3211/3211357.png"
    img_displayed = False
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        st.image(img, width=180, caption=None)
        img_displayed = True
    except Exception:
        try:
            response = requests.get(fallback_icon, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            st.image(img, width=180, caption=None)
            img_displayed = True
        except Exception:
            st.info("이미지를 불러올 수 없습니다.")
    st.markdown(
        "<div style='margin-top: -1px; text-align:center;'>"
        "<span style='font-size:0.9rem; color:#888;'>Made by parksuk1991</span>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align: center; margin-bottom: 6px;">'
        'Data 출처: <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>'
        '</div>',
        unsafe_allow_html=True
    )

# ===================== 차트 구간 설정 및 전일 시장 업데이트 버튼 (사이드바로 이동) =====================
with st.sidebar:
    st.markdown("### ⚙️ 대시보드 설정")
    st.markdown("""
        <div style="font-size:1rem;font-weight:600;">
            차트 수익률 기간 설정
        </div>
        <div style="font-size:0.8rem; color:#888; line-height:1.2; margin-bottom:-10px;">
            (N개월, 모든 차트에 동일 적용)
        </div>
    """, unsafe_allow_html=True)
    normalized_months = st.slider(
        "",
        3, 36, 12,
        help="모든 차트에 적용될 정규화 수익률 기간입니다.",
        key="norm_months_slider"
    )
    update_clicked = st.button("Update", type="primary", use_container_width=True)
    st.markdown(
        """
        <div style='text-align:center; margin-top:20px;'>
            <span style='font-size:0.85rem; color:#d9534f; font-weight:500;'>
                ⚠️ 위에서 차트 수익률 기간 설정 후<br>'Update' 버튼을 눌러주세요!
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

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

# -------------------- 1. 자산별 성과 계산 검증/개선 --------------------
def get_perf_table_precise(label2ticker, ref_date=None):
    """
    각 자산별 성과를 정확하게 계산 (close price 기준, 1D는 전 거래일 대비, 나머지는 해당 기간 첫 거래일 대비)
    """
    tickers = list(label2ticker.values())
    labels = list(label2ticker.keys())

    if ref_date is None:
        ref_date = datetime.now().date()
    start = ref_date - timedelta(days=3*366+20)
    end = ref_date + timedelta(days=1)
    df = yf.download(tickers, start=start, end=end, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.ffill()[tickers]
    last_trade_idx = df.index[df.index.date <= ref_date][-1]
    last_trade_date = last_trade_idx.date()

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
    for label, ticker in label2ticker.items():
        row = {'자산명': label}
        series = df[ticker].dropna()
        if last_trade_idx not in series.index:
            row['현재값'] = np.nan
            for k in periods: row[k] = np.nan
            results.append(row)
            continue
        curr_val = series.loc[last_trade_idx]
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
                    idx = series.index.get_loc(last_trade_idx)
                    if idx >= 1:
                        base = series.iloc[idx-1]
                    else:
                        base = np.nan
                else:
                    idx = series.index.get_loc(last_trade_idx)
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

# -------------------- 2. 섹터 ETF별 최대 비중 종목 뉴스 --------------------
def get_etf_top_holding_stock(etf_ticker):
    """
    yfinance의 fund_holdings에서 비중 1위 종목 반환. 미지원시 info['topHoldings']/['holdings'] fallback.
    """
    try:
        etf = yf.Ticker(etf_ticker)
        try:
            h = etf.fund_holdings
            if h is not None and not h.empty:
                return h.sort_values('weight', ascending=False)['symbol'].iloc[0]
        except Exception:
            pass
        h2 = etf.info.get('topHoldings', None)
        if h2 and len(h2) > 0:
            weights = [x['holdingPercent'] for x in h2]
            max_idx = np.argmax(weights)
            return h2[max_idx]['symbol']
        h3 = etf.info.get('holdings', None)
        if h3 and len(h3) > 0:
            weights = [x['holdingPercent'] for x in h3]
            max_idx = np.argmax(weights)
            return h3[max_idx]['symbol']
    except Exception:
        pass
    return None

def get_news_headline_for_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news if hasattr(stock, "news") else stock.get_news()
        for article in news:
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
            return {
                '티커': ticker,
                '일자': date.strftime('%Y-%m-%d') if date else '',
                '헤드라인': title
            }
    except Exception:
        return None
    return None

def get_sector_top_holdings_news(sector_etfs):
    """
    섹터ETF별 최상위 비중종목별 뉴스 한개씩(중복 허용X)
    """
    stocks = []
    for etf_ticker in sector_etfs.values():
        stock = get_etf_top_holding_stock(etf_ticker)
        if stock: stocks.append(stock)
    stocks = list(dict.fromkeys(stocks))  # 중복 제거 (순서 유지)
    news_list = []
    for tk in stocks:
        news = get_news_headline_for_ticker(tk)
        if news and news['헤드라인']:
            news_list.append(news)
    return news_list

# -------------------- 3. SPY 전체보유종목 전일 Top10/Bottom10 --------------------
def get_spy_holdings():
    """
    SPY의 전체 보유종목(symbol, name, sector) DataFrame 반환. 실패시 S&P 500 위키 fallback.
    """
    try:
        etf = yf.Ticker("SPY")
        h = etf.fund_holdings
        if h is not None and not h.empty:
            h = h.rename(columns={"symbol":"Ticker", "holdingName":"Company", "sector":"Sector"})
            if "Company" not in h.columns:
                h["Company"] = h["Ticker"]
            if "Sector" not in h.columns:
                h["Sector"] = ""
            return h[["Ticker", "Company", "Sector"]]
    except Exception:
        pass
    # fallback: 위키피디아 S&P 500
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, header=0)[0]
    df = df.rename(columns={"Symbol": "Ticker", "Security": "Company", "GICS Sector": "Sector"})
    df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)
    return df[['Ticker', 'Company', 'Sector']]

def get_spy_daily_perf():
    """
    SPY 보유종목별 전일 수익률, 거래량, 시가총액(백만) 포함 DataFrame 반환
    """
    spy_df = get_spy_holdings()
    tickers = spy_df['Ticker'].tolist()
    dfs = []
    # yfinance 요청 제한 우회 위해 90개씩 분할
    for i in range(0, len(tickers), 90):
        tks = tickers[i:i+90]
        data = yf.download(tks, period="7d", interval="1d", group_by='ticker', threads=True, progress=False)
        for tk in tks:
            try:
                close = data[tk]['Close'].dropna()
                vol = data[tk]['Volume'].dropna()
                if len(close) < 2:
                    continue
                ret = (close.iloc[-1] / close.iloc[-2] - 1) * 100
                last_vol = vol.iloc[-1]
                info = yf.Ticker(tk).info
                mktcap = info.get('marketCap', np.nan)
                mktcap = mktcap/1e6 if pd.notnull(mktcap) else np.nan
                dfs.append({
                    'Ticker': tk,
                    '전일수익률': ret,
                    '거래량': int(last_vol),
                    '시가총액': mktcap,
                })
            except Exception:
                continue
    perf_df = pd.DataFrame(dfs)
    perf_df = perf_df.merge(spy_df, on='Ticker', how='left')
    return perf_df

def get_spy_top_bottom10():
    df = get_spy_daily_perf()
    df = df[df['전일수익률'].notnull()]
    top10 = df.nlargest(10, '전일수익률')
    bottom10 = df.nsmallest(10, '전일수익률')
    for d in [top10, bottom10]:
        d['거래량'] = d['거래량'].apply(lambda x: f"{x:,}")
        d['시가총액'] = d['시가총액'].apply(lambda x: f"{x:,.0f}")
        d['전일수익률'] = d['전일수익률'].apply(lambda x: f"{x:.2f}%")
    return top10, bottom10

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
    st.markdown("<br>", unsafe_allow_html=True)
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

    st.subheader(f"📈 주요 주가지수 수익률 (최근 {normalized_months}개월)")
    norm_idx = get_normalized_prices(STOCK_ETFS, months=normalized_months)
    fig1 = go.Figure()
    for col in norm_idx.columns:
        fig1.add_trace(go.Scatter(x=norm_idx.index, y=norm_idx[col], mode='lines', name=col))
    fig1.update_layout(
        yaxis_title="100 기준 누적수익률(%)",
        template="plotly_dark", height=500, legend=dict(orientation='h')
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"📈 섹터 ETF 수익률 (최근 {normalized_months}개월)")
    norm_sector = get_normalized_prices(SECTOR_ETFS, months=normalized_months)
    fig2 = go.Figure()
    for col in norm_sector.columns:
        fig2.add_trace(go.Scatter(x=norm_sector.index, y=norm_sector[col], mode='lines', name=col))
    fig2.update_layout(
        yaxis_title="100 기준 누적수익률(%)",
        template="plotly_dark", height=500, legend=dict(orientation='h')
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"📈 스타일 ETF 수익률 (최근 {normalized_months}개월)")
    norm_style = get_normalized_prices(STYLE_ETFS, months=normalized_months)
    fig3 = go.Figure()
    for col in norm_style.columns:
        fig3.add_trace(go.Scatter(x=norm_style.index, y=norm_style[col], mode='lines', name=col))
    fig3.update_layout(
        yaxis_title="100 기준 누적수익률(%)",
        template="plotly_dark", height=500, legend=dict(orientation='h')
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📰 섹터 ETF별 최대 비중 종목의 최근 뉴스 헤드라인")
    news_list = get_sector_top_holdings_news(SECTOR_ETFS)
    if news_list:
        for row in news_list:
            st.markdown(f"- **[{row['티커']}]** {row['일자']}: {row['헤드라인']}")
    else:
        st.info("뉴스 헤드라인을 가져올 수 없습니다.")

    st.subheader("🏅 SPY 보유종목 전일 성과 Top 10 / Bottom 10")
    try:
        top10, bottom10 = get_spy_top_bottom10()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top 10")
            st.dataframe(top10[['Ticker', 'Company', 'Sector', '전일수익률', '거래량', '시가총액']], use_container_width=True)
        with col2:
            st.markdown("#### Bottom 10")
            st.dataframe(bottom10[['Ticker', 'Company', 'Sector', '전일수익률', '거래량', '시가총액']], use_container_width=True)
    except Exception as e:
        st.warning(f"SPY Top/Bottom 10 데이터를 불러오는 데 실패했습니다. ({e})")
