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
from yahooquery import Ticker
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')

try:
    import lxml
except ImportError:
    st.error("lxml 패키지가 필요합니다. requirements.txt에 lxml을 추가하세요.")

st.set_page_config(
    page_title="Global Market Monitoring",
    page_icon="🌐",
    layout="wide"
)

# -------------------- 상단 레이아웃---------------------
col_title, col_img_credit = st.columns([7, 1])
with col_title:
    st.title("🌐 Global Market Monitoring")
with col_img_credit:
    image_url = "https://amateurphotographer.com/wp-content/uploads/sites/7/2017/08/Screen-Shot-2017-08-23-at-22.29.18.png?w=600.jpg"
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        st.image(img, width=180, caption=None)
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
        "",  # 제목은 위에서 렌더링
        3, 36, 12,
        help="모든 차트에 적용될 정규화 수익률 기간을 의미",
        key="norm_months_slider"
    )
    update_clicked = st.button("Update", type="primary", use_container_width=True)
    st.markdown(
        """
        <div style='text-align:center; margin-top:20px;'>
            <span style='font-size:0.85rem; color:#d9534f; font-weight:500;'>
                ⚠️ 위에서 차트 수익률 기간 설정 후<br>'Update' 버튼 Click!
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========== 자산 정의 ================
STOCK_ETFS = {
    'S&P 500 (SPY)': 'SPY',
    'NASDAQ 100 (QQQ)': 'QQQ',
    '전세계 (ACWI)': 'ACWI',
    '선진국 (VEA)': 'VEA',
    '신흥국 (VWO)': 'VWO',
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
    '달러인덱스': 'DX-Y.NYB',
    '달러-원': 'KRW=X',
    '유로-원': 'EURKRW=X',
    '달러-엔': 'JPY=X',
    '원-엔': 'JPYKRW=X',
    '달러-유로': 'EURUSD=X',
    '달러-파운드': 'GBPUSD=X',
    '달러-위안': 'CNY=X'
}
CRYPTO = {
    '비트코인 (BTC)': 'BTC-USD',
    '이더리움 (ETH)': 'ETH-USD',
    '솔라나 (SOL)': 'SOL-USD',
    '리플 (XRP)': 'XRP-USD',
    '에이다 (ADA)': 'ADA-USD',
    '라이트코인 (LTC)': 'LTC-USD',
    '비트코인캐시 (BCH)': 'BCH-USD',
    '체인링크 (LINK)': 'LINK-USD',
    '도지코인 (DOGE)': 'DOGE-USD',
    '아발란체 (AVAX)': 'AVAX-USD',
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

# ---- 주요 데이터 함수들 ----
def get_perf_table_improved(label2ticker, ref_date=None):
    tickers = list(label2ticker.values())
    labels = list(label2ticker.keys())
    if ref_date is None:
        ref_date = datetime.now().date()
    start = ref_date - timedelta(days=4*365)
    end = ref_date + timedelta(days=1)
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.ffill().dropna(how='all')
        df = df[tickers]
    except Exception as e:
        st.error(f"데이터 다운로드 오류: {e}")
        return pd.DataFrame()
    if df.empty:
        st.warning("다운로드된 데이터가 없습니다.")
        return pd.DataFrame()
    available_dates = df.index[df.index.date <= ref_date]
    if len(available_dates) == 0:
        st.warning(f"기준일({ref_date}) 이전의 데이터가 없습니다.")
        return pd.DataFrame()
    last_trade_date = available_dates[-1].date()
    last_idx = available_dates[-1]
    periods = {
        '1D(%)': {'days': 1, 'type': 'business'},
        '1W(%)': {'days': 5, 'type': 'business'}, 
        'MTD(%)': {'type': 'month_start'},
        '1M(%)': {'days': 21, 'type': 'business'},
        '3M(%)': {'days': 63, 'type': 'business'},
        '6M(%)': {'days': 126, 'type': 'business'},
        'YTD(%)': {'type': 'year_start'},
        '1Y(%)': {'days': 252, 'type': 'business'},
        '3Y(%)': {'days': 756, 'type': 'business'}
    }
    results = []
    for label, ticker in label2ticker.items():
        row = {'자산명': label}
        series = df[ticker].dropna()
        if last_idx not in series.index or len(series) == 0:
            row['현재값'] = np.nan
            for period_key in periods.keys():
                row[period_key] = np.nan
            results.append(row)
            continue
        curr_val = series.loc[last_idx]
        row['현재값'] = curr_val
        for period_key, period_config in periods.items():
            base_val = None
            try:
                if period_config['type'] == 'month_start':
                    month_start = last_trade_date.replace(day=1)
                    month_data = series[series.index.date >= month_start]
                    if len(month_data) > 0:
                        base_val = month_data.iloc[0]
                elif period_config['type'] == 'year_start':
                    year_start = last_trade_date.replace(month=1, day=1)
                    year_data = series[series.index.date >= year_start]
                    if len(year_data) > 0:
                        base_val = year_data.iloc[0]
                elif period_config['type'] == 'business':
                    current_idx = series.index.get_loc(last_idx)
                    lookback_days = period_config['days']
                    if current_idx >= lookback_days:
                        base_val = series.iloc[current_idx - lookback_days]
                    elif current_idx > 0:
                        base_val = series.iloc[0]
                if base_val is not None and not np.isnan(base_val) and base_val != 0:
                    return_pct = (curr_val / base_val - 1) * 100
                    row[period_key] = return_pct
                else:
                    row[period_key] = np.nan
            except Exception:
                row[period_key] = np.nan
        results.append(row)
    df_result = pd.DataFrame(results)
    if '현재값' in df_result.columns:
        df_result['현재값'] = df_result['현재값'].apply(
            lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A"
        )
    return df_result

def get_sample_calculation_dates(label2ticker, ref_date=None):
    if ref_date is None:
        ref_date = datetime.now().date()
    sample_ticker = list(label2ticker.values())[0]
    sample_label = list(label2ticker.keys())[0]
    start = ref_date - timedelta(days=4*365)
    end = ref_date + timedelta(days=1)
    try:
        data = yf.download(sample_ticker, start=start, end=end, progress=False)['Close']
        data = data.dropna()
        available_dates = data.index[data.index.date <= ref_date]
        if len(available_dates) == 0:
            return None, None, None
        last_trade_date = available_dates[-1].date()
        current_idx = data.index.get_loc(available_dates[-1])
        actual_dates = {}
        periods_check = {'1D': 1, '1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '3Y': 756}
        for period, days in periods_check.items():
            if current_idx >= days:
                base_date = data.index[current_idx - days].date()
                actual_dates[period] = base_date.strftime('%Y-%m-%d')
            else:
                actual_dates[period] = f"데이터 부족 ({current_idx+1}/{days}일)"
        month_start = last_trade_date.replace(day=1)
        year_start = last_trade_date.replace(month=1, day=1)
        mtd_data = data[data.index.date >= month_start]
        ytd_data = data[data.index.date >= year_start]
        if len(mtd_data) > 0:
            actual_dates['MTD'] = mtd_data.index[0].date().strftime('%Y-%m-%d')
        if len(ytd_data) > 0:
            actual_dates['YTD'] = ytd_data.index[0].date().strftime('%Y-%m-%d')
        return sample_label, last_trade_date.strftime('%Y-%m-%d'), actual_dates
    except Exception:
        return None, None, None

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
    try:
        t = Ticker(etf_ticker)
        info = t.fund_holding_info or {}
        holdings = info.get(etf_ticker, {}).get('holdings', [])
        if holdings:
            holdings_sorted = sorted(holdings, key=lambda x: x.get('holdingPercent', 0), reverse=True)
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

def format_percentage(val):
    if pd.isna(val):
        return "N/A"
    try:
        if isinstance(val, (int, float)):
            return f"{val:.6f}"
    except:
        return "N/A"

def colorize_return(val):
    """값에 따른 색상 지정 (Streamlit pandas Styler용)"""
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except (ValueError, TypeError):
        try:
            v = float(str(val).replace('%','').replace(' ',''))
        except Exception:
            return ""
    if v > 0:
        return "color: red;"
    elif v < 0:
        return "color: blue;"
    else:
        return ""

def style_perf_table(df, perf_cols):
    styled = df.style
    for col in perf_cols:
        if col in df.columns:
            styled = styled.format({col: format_percentage}).applymap(colorize_return, subset=[col])
    return styled

def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

@st.cache_data
def get_news_sentiment_data():
    news_list = []
    all_syms = []
    for label, etf in SECTOR_ETFS.items():
        top_holdings = get_top_holdings(etf, n=3)
        holdings_syms = [sym for sym, _ in top_holdings]
        all_syms.extend(holdings_syms)
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
        return pd.DataFrame(), []
    df = pd.DataFrame(news_list)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    sid = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Headline'].apply(
        lambda headline: sid.polarity_scores(headline)['compound'] if headline else 0
    )
    df['Sentiment_Category'] = df['Sentiment'].apply(classify_sentiment)
    return df, list(set(all_syms))

def create_sentiment_histogram(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Sentiment'],
        nbinsx=20,
        name='Sentiment Distribution',
        marker_color='rgba(235, 0, 140, 0.7)',
        opacity=0.8
    ))
    hist, bin_edges = np.histogram(df['Sentiment'], bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    from scipy import ndimage
    smoothed = ndimage.gaussian_filter1d(hist, 1)
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=smoothed * len(df) * (bin_edges[1] - bin_edges[0]),
        mode='lines',
        name='KDE',
        line=dict(color='royalblue', width=2)
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

def create_sentiment_boxplot(df):
    mean_values = df.groupby('Ticker')['Sentiment'].mean().reset_index()
    fig = go.Figure()
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
        title='감정 점수 분포',
        xaxis_title='종목',
        yaxis_title='감정 점수',
        template="plotly_dark",
        height=500,
        showlegend=False
    )
    return fig

def create_sentiment_countplot(df):
    sentiment_counts = df['Sentiment_Category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment_Category', 'Count']
    color_map = {
        'Positive': 'rgba(235,0,140,0.8)',
        'Negative': 'rgba(65,105,225,0.8)',
        'Neutral': 'rgba(102,194,165,0.8)'
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
        title='감정 분포',
        xaxis_title='감정 카테고리',
        yaxis_title='뉴스 개수',
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    return fig

def get_analyst_report_data(ticker_syms):
    rows = []
    for sym in ticker_syms:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            current_price = info.get('regularMarketPrice')
            target_price = info.get('targetMeanPrice')
            name = info.get('shortName') or info.get('longName') or ''
            upside = None
            if target_price and current_price and current_price != 0:
                upside = ((target_price / current_price) - 1) * 100
            rows.append({
                'Ticker': sym,
                '종목명': name,
                '애널리스트 의견 등급 점수': info.get('recommendationMean'),
                '애널리스트 의견': info.get('recommendationKey'),
                '애널리스트 목표가(평균)': target_price,
                '현재가': current_price,
                '상승여력': upside
            })
        except Exception:
            rows.append({
                'Ticker': sym,
                '종목명': '',
                '애널리스트 의견 등급 점수': None,
                '애널리스트 의견': None,
                '애널리스트 목표가(평균)': None,
                '현재가': None,
                '상승여력': None
            })
    df = pd.DataFrame(rows)
    df = df[['Ticker', '종목명', '애널리스트 의견 등급 점수', '애널리스트 의견', '애널리스트 목표가(평균)', '현재가', '상승여력']]
    return df

def get_valuation_eps_table(ticker_syms):
    rows = []
    for sym in ticker_syms:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            name = info.get('shortName') or info.get('longName') or ''
            trailingPE = info.get('trailingPE')
            forwardPE = info.get('forwardPE')
            trailingEPS = info.get('trailingEps') or info.get('trailingEPS')
            forwardEPS = info.get('forwardEps') or info.get('forwardEPS')
            eps_growth = None
            if trailingEPS is not None and forwardEPS is not None and trailingEPS != 0:
                eps_growth = ((forwardEPS / trailingEPS) - 1) * 100
            rows.append({
                'Ticker': sym,
                '종목명': name,
                '현재 PE': trailingPE,
                '선행 PE': forwardPE,
                '현재 EPS': trailingEPS,
                '선행 EPS': forwardEPS,
                'EPS 상승률': eps_growth
            })
        except Exception:
            rows.append({
                'Ticker': sym,
                '종목명': '',
                '현재 PE': None,
                '선행 PE': None,
                '현재 EPS': None,
                '선행 EPS': None,
                'EPS 상승률': None
            })
    df = pd.DataFrame(rows)
    df = df[['Ticker', '종목명', '현재 PE', '선행 PE', '현재 EPS', '선행 EPS', 'EPS 상승률']]
    return df

def show_sentiment_analysis():
    st.subheader("✳️✴️ 주요 종목 뉴스 감정 분석")
    with st.spinner("뉴스 데이터 수집 및 감정 분석 중..."):
        df, ticker_syms = get_news_sentiment_data()
    if df.empty:
        st.warning("뉴스 데이터를 가져올 수 없습니다.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 뉴스 개수", len(df))
    with col2:
        st.metric("평균 감정 점수", f"{df['Sentiment'].mean():.3f}")
    with col3:
        positive_pct = (df['Sentiment_Category'] == 'Positive').sum() / len(df) * 100
        st.metric("긍정 비율", f"{positive_pct:.1f}%")
    with col4:
        negative_pct = (df['Sentiment_Category'] == 'Negative').sum() / len(df) * 100
        st.metric("부정 비율", f"{negative_pct:.1f}%")

    st.subheader("감정 점수 및 카테고리 분포")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig1 = create_sentiment_histogram(df)
        st.plotly_chart(fig1, use_container_width=True)
    with chart_col2:
        fig3 = create_sentiment_countplot(df)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("종목별 감정 점수")
    fig2 = create_sentiment_boxplot(df)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("상세 뉴스 데이터 보기"):
        st.dataframe(
            df[['Ticker', 'Date', 'Headline', 'Sentiment', 'Sentiment_Category']].sort_values('Date', ascending=False),
            use_container_width=True
        )

    st.markdown("---")
    # 애널리스트 리포트 요약 한 번만 출력
    st.subheader("🧑‍💼 33개 주요 종목 애널리스트 리포트 요약")
    st.caption("• 최근 3~6개월 내의 애널리스트 리포트에서 제시된 목표가(Price Target) 기준")
    analyst_df = get_analyst_report_data(ticker_syms)
    st.dataframe(
        analyst_df.style.format({
            '애널리스트 의견 등급 점수': '{:.2f}',
            '애널리스트 목표가(평균)': '{:,.2f}',
            '현재가': '{:,.2f}',
            '상승여력': '{:.1f}%'
        }).highlight_null('lightgray'),
        use_container_width=True, height=min(900, 30 + 30*len(analyst_df))
    )
    # 밸류에이션 및 EPS 추이 한 번만 출력
    st.subheader("📊 33개 주요 종목 밸류에이션 및 EPS 추이")
    st.caption("• 현재 PE, 선행 PE, 현재 EPS, 선행 EPS 및 EPS 상승률을 함께 표시")
    valuation_df = get_valuation_eps_table(ticker_syms)
    st.dataframe(
        valuation_df.style.format({
            '현재 PE': '{:.2f}',
            '선행 PE': '{:.2f}',
            '현재 EPS': '{:.2f}',
            '선행 EPS': '{:.2f}',
            'EPS 상승률': '{:.1f}%'
        }).background_gradient(subset=['EPS 상승률'], cmap='RdPu'),
        use_container_width=True, height=min(900, 30 + 30*len(valuation_df))
    )

def show_all_performance_tables():
    perf_cols = ['1D(%)','1W(%)','MTD(%)','1M(%)','3M(%)','6M(%)','YTD(%)','1Y(%)','3Y(%)']
    st.subheader("📊 주식시장")
    with st.spinner("주식시장 성과 데이터 계산 중..."):
        stock_perf = get_perf_table_improved(STOCK_ETFS)
    if not stock_perf.empty:
        st.dataframe(
            style_perf_table(stock_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=490
        )
    else:
        st.error("주식시장 성과 데이터를 계산할 수 없습니다.")
    st.subheader("🗠 채권시장")
    with st.spinner("채권시장 성과 데이터 계산 중..."):
        bond_perf = get_perf_table_improved(BOND_ETFS)
    if not bond_perf.empty:
        st.dataframe(
            style_perf_table(bond_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=385
        )
    else:
        st.error("채권시장 성과 데이터를 계산할 수 없습니다.")
    st.subheader("💹 통화")
    with st.spinner("통화 성과 데이터 계산 중..."):
        curr_perf = get_perf_table_improved(CURRENCY)
    if not curr_perf.empty:
        st.dataframe(
            style_perf_table(curr_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=315
        )
    else:
        st.error("통화 성과 데이터를 계산할 수 없습니다.")
    st.subheader("📈 암호화폐")
    with st.spinner("암호화폐 성과 데이터 계산 중..."):
        crypto_perf = get_perf_table_improved(CRYPTO)
    if not crypto_perf.empty:
        st.dataframe(
            style_perf_table(crypto_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=385
        )
    else:
        st.error("암호화폐 성과 데이터를 계산할 수 없습니다.")
    st.subheader("📕 스타일 ETF")
    with st.spinner("스타일 ETF 성과 데이터 계산 중..."):
        style_perf = get_perf_table_improved(STYLE_ETFS)
    if not style_perf.empty:
        st.dataframe(
            style_perf_table(style_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=245
        )
    else:
        st.error("스타일 ETF 성과 데이터를 계산할 수 없습니다.")
    st.subheader("📘 섹터 ETF")
    with st.spinner("섹터 ETF 성과 데이터 계산 중..."):
        sector_perf = get_perf_table_improved(SECTOR_ETFS)
    if not sector_perf.empty:
        st.dataframe(
            style_perf_table(sector_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=420
        )
    else:
        st.error("섹터 ETF 성과 데이터를 계산할 수 없습니다.")
    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.caption("📝 **성과 계산 기준**")
        st.caption("• 영업일 기준: 1D=1영업일, 1W=5영업일, 1M=21영업일, 3M=63영업일, 6M=126영업일, 1Y=252영업일, 3Y=756영업일")
        st.caption("• MTD: 해당 월 첫 영업일 기준, YTD: 해당 연도 첫 영업일 기준")
        st.caption("• 데이터 부족 시 사용 가능한 가장 오래된 데이터 기준으로 계산")
    with col2:
        with st.expander("📋 상세 계산 기준일 보기"):
            sample_label, last_date, actual_dates = get_sample_calculation_dates(STOCK_ETFS)
            if sample_label and actual_dates:
                st.caption(f"**샘플 자산:** {sample_label} | **최근 거래일:** {last_date}")
                periods_line1 = [f"{period}: {actual_dates[period]}" for period in ['1D', '1W', 'MTD', '1M'] if period in actual_dates]
                st.caption("• " + " | ".join(periods_line1))
                periods_line2 = [f"{period}: {actual_dates[period]}" for period in ['3M', '6M', 'YTD', '1Y', '3Y'] if period in actual_dates]
                st.caption("• " + " | ".join(periods_line2))
            else:
                st.caption("샘플 데이터를 불러올 수 없습니다.")

if update_clicked:
    st.markdown("<br>", unsafe_allow_html=True)
    show_all_performance_tables()
    st.subheader(f"✅ 주요 주가지수 수익률 (최근 {normalized_months}개월)")
    norm_idx = get_normalized_prices(STOCK_ETFS, months=normalized_months)
    fig1 = go.Figure()
    for col in norm_idx.columns:
        fig1.add_trace(go.Scatter(x=norm_idx.index, y=norm_idx[col], mode='lines', name=col))
    fig1.update_layout(
        yaxis_title="100 기준 누적수익률(%)",
        template="plotly_dark", height=500, legend=dict(orientation='h')
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.subheader(f"☑️ 섹터 ETF 수익률 (최근 {normalized_months}개월)")
    norm_sector = get_normalized_prices(SECTOR_ETFS, months=normalized_months)
    fig2 = go.Figure()
    for col in norm_sector.columns:
        fig2.add_trace(go.Scatter(x=norm_sector.index, y=norm_sector[col], mode='lines', name=col))
    fig2.update_layout(
        yaxis_title="100 기준 누적수익률(%)",
        template="plotly_dark", height=500, legend=dict(orientation='h')
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader(f"☑️ 스타일 ETF 수익률 (최근 {normalized_months}개월)")
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
    st.markdown("---")
    show_sentiment_analysis()
