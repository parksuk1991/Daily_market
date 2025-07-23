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
from yahooquery import Ticker  # 추가
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

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
    st.title("🌐 Global Markets Monitoring")
    #st.markdown("---", unsafe_allow_html=True)
    #st.markdown("####    주요 시장 성과", unsafe_allow_html=True)
with col_img_credit:
    # 닐 암스트롱 달착륙 사진(퍼블릭 도메인, NASA) - 다운로드 실패시 대체 아이콘 제공
    image_url = "https://cdn.theatlantic.com/thumbor/gjwD-uCiv0sHowRxQrQgL9b3Shk=/900x638/media/img/photo/2019/07/apollo-11-moon-landing-photos-50-ye/a01_40-5903/original.jpg"
    fallback_icon = "https://cdn-icons-png.flaticon.com/512/3211/3211357.png"  # 우주인 아이콘 (flaticon)
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
    # 슬라이더 타이틀: 메인, 괄호/보조설명은 하단 줄바꿈+축소
    st.markdown("""
        <div style="font-size:1rem;font-weight:600;">
            차트 수익률 기간 설정
        </div>
        <div style="font-size:0.8rem; color:#888; line-height:1.2; margin-bottom:-10px;">
            (N개월, 모든 차트에 동일 적용)
        </div>
    """, unsafe_allow_html=True)
    normalized_months = st.slider(
        "",  # 제목은 위에서 렌더링
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

def get_top_holdings(etf_ticker, n=3):
    """ETF 내 비중 상위 n개 종목의 심볼과 이름을 반환"""
    try:
        t = Ticker(etf_ticker)
        info = t.fund_holding_info or {}
        holdings = info.get(etf_ticker, {}).get('holdings', [])
        if holdings:
            holdings_sorted = sorted(holdings, key=lambda x: x.get('holdingPercent', 0), reverse=True)
            # symbol, holdingName 둘 다 반환
            return [(h['symbol'], h.get('holdingName', h['symbol'])) for h in holdings_sorted[:n]]
        else:
            return []
    except Exception:
        return []

def get_news_for_ticker(ticker_symbol, limit=1):
    y = yf.Ticker(ticker_symbol)
    try:
        news = y.news if hasattr(y, 'news') else y.get_news()
    except Exception:
        news = []
    result = []
    for art in news[:limit]:
        title = art.get('title') or art.get('content', {}).get('title')
        ts = art.get('providerPublishTime')
        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if isinstance(ts, int) else ''
        if title:
            result.append({'ticker': ticker_symbol, 'date': date, 'title': title})
    return result


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





# 감정 분류 함수
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# 뉴스 데이터 수집 및 감정 분석
@st.cache_data
def get_news_sentiment_data():
    news_list = []
    for label, etf in SECTOR_ETFS.items():
        top_holdings = get_top_holdings(etf, n=3)
        holdings_syms = [sym for sym, _ in top_holdings]
        for ticker_symbol in holdings_syms:
            try:
                ticker = yf.Ticker(ticker_symbol)
                news = ticker.news
                for article in news:
                    content = article.get('content', {})
                    news_list.append({
                        'Ticker': ticker_symbol,
                        'Date': content.get('pubDate'),
                        'Headline': content.get('title')
                    })
            except Exception as e:
                st.warning(f"{ticker_symbol} 뉴스 데이터 수집 오류: {e}")
                continue
    
    if not news_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(news_list)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 감정 분석
    sid = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Headline'].apply(
        lambda headline: sid.polarity_scores(headline)['compound'] if headline else 0
    )
    df['Sentiment_Category'] = df['Sentiment'].apply(classify_sentiment)
    
    return df

# 감정 분포 히스토그램 (Plotly)
def create_sentiment_histogram(df):
    fig = go.Figure()
    
    # 히스토그램 생성
    fig.add_trace(go.Histogram(
        x=df['Sentiment'],
        nbinsx=20,
        name='Sentiment Distribution',
        marker_color='rgba(158, 71, 99, 0.7)',
        opacity=0.8
    ))
    
    # KDE 곡선 추가 (근사)
    hist, bin_edges = np.histogram(df['Sentiment'], bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 간단한 smoothing을 위한 moving average
    from scipy import ndimage
    smoothed = ndimage.gaussian_filter1d(hist, 1)
    
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=smoothed * len(df) * (bin_edges[1] - bin_edges[0]),
        mode='lines',
        name='KDE',
        line=dict(color='crimson', width=2)
    ))
    
    fig.update_layout(
        title='감정 점수 분포',
        xaxis_title='감정 점수',
        yaxis_title='빈도',
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    return fig

# 감정 박스플롯 (Plotly)
def create_sentiment_boxplot(df):
    # 티커별 평균 감정 점수 계산
    mean_values = df.groupby('Ticker')['Sentiment'].mean().reset_index()
    
    fig = go.Figure()
    
    # 각 티커별 박스플롯 생성
    tickers = df['Ticker'].unique()
    colors = px.colors.qualitative.Set3[:len(tickers)]
    
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['Ticker'] == ticker]['Sentiment']
        fig.add_trace(go.Box(
            y=ticker_data,
            name=ticker,
            marker_color=colors[i % len(colors)],
            boxmean=True
        ))
    
    # 평균값 텍스트 추가
    for i, row in mean_values.iterrows():
        color = 'red' if row['Sentiment'] >= 0 else 'blue'
        fig.add_annotation(
            x=i,
            y=row['Sentiment'],
            text=f'{row["Sentiment"]:.2f}',
            showarrow=False,
            font=dict(color=color, size=12),
            bgcolor="rgba(255,255,255,0.8)"
        )
    
    fig.update_layout(
        title='티커별 감정 점수 분포',
        xaxis_title='종목',
        yaxis_title='감정 점수',
        template="plotly_dark",
        height=500,
        showlegend=False
    )
    
    return fig

# 감정 카테고리 카운트 플롯 (Plotly)
def create_sentiment_countplot(df):
    sentiment_counts = df['Sentiment_Category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment_Category', 'Count']
    
    # 색상 매핑
    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'gray'
    }
    
    colors = [color_map.get(cat, 'blue') for cat in sentiment_counts['Sentiment_Category']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sentiment_counts['Sentiment_Category'],
        y=sentiment_counts['Count'],
        marker_color=colors,
        text=sentiment_counts['Count'],
        textposition='inside',
        textfont=dict(color='white', size=14)
    ))
    
    fig.update_layout(
        title='포트폴리오 감정 분포',
        xaxis_title='감정 카테고리',
        yaxis_title='뉴스 개수',
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig

# Streamlit 앱 메인 부분
def show_sentiment_analysis():
    st.subheader("📰 뉴스 감정 분석")
    
    # 데이터 로딩
    with st.spinner("뉴스 데이터 수집 및 감정 분석 중..."):
        df = get_news_sentiment_data()
    
    if df.empty:
        st.warning("뉴스 데이터를 가져올 수 없습니다.")
        return
    
    # 기본 통계 정보
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 뉴스 수", len(df))
    with col2:
        st.metric("평균 감정 점수", f"{df['Sentiment'].mean():.3f}")
    with col3:
        positive_pct = (df['Sentiment_Category'] == 'Positive').sum() / len(df) * 100
        st.metric("긍정 비율", f"{positive_pct:.1f}%")
    with col4:
        negative_pct = (df['Sentiment_Category'] == 'Negative').sum() / len(df) * 100
        st.metric("부정 비율", f"{negative_pct:.1f}%")
    
    # 감정 분포 히스토그램
    st.subheader("감정 점수 분포")
    fig1 = create_sentiment_histogram(df)
    st.plotly_chart(fig1, use_container_width=True)
    
    # 티커별 감정 박스플롯
    st.subheader("종목별 감정 점수")
    fig2 = create_sentiment_boxplot(df)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 감정 카테고리 분포
    st.subheader("감정 카테고리 분포")
    fig3 = create_sentiment_countplot(df)
    st.plotly_chart(fig3, use_container_width=True)
    
    # 상세 데이터 테이블
    with st.expander("상세 뉴스 데이터 보기"):
        st.dataframe(
            df[['Ticker', 'Date', 'Headline', 'Sentiment', 'Sentiment_Category']].sort_values('Date', ascending=False),
            use_container_width=True
        )






# =========== MAIN BUTTON ===========
if update_clicked:
    # 빈 줄(공백) 추가해서 '주식시장' 부분을 조금 더 내려줌
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

    # ---------- Normalized 차트 구간 설정 아래에 위치 ----------
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

    st.subheader("📰 섹터별 주요 종목 헤드라인")
    for label, etf in SECTOR_ETFS.items():
        top_holdings = get_top_holdings(etf, n=3)
        if top_holdings:
            # 섹터명에서 괄호와 ETF코드 제거 → "IT (XLK)" → "IT섹터" 등 가공
            sector_name = label.split()[0] + " 섹터"
            holding_names = [name for _, name in top_holdings]
            holding_syms = [sym for sym, _ in top_holdings]
            st.write(f"#### {sector_name} 주요 종목: {', '.join(holding_names)}")
            for sym, name in top_holdings:
                news = get_news_for_ticker(sym, limit=1)
                if news:
                    art = news[0]
                    st.markdown(f"- **[{sym}]** {art['date']}: {art['title']}")
                else:
                    st.write(f"- [{sym}] 뉴스 없음")
        else:
            st.write(f"- {label}: 보유종목 정보 없음")
            
    # 새로운 감정 분석 섹션 추가
    show_sentiment_analysis()


# else 블록 삭제: 안내문구는 사이드바에서 항상 노출
