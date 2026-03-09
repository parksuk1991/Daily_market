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
import time
import warnings
import ssl
import urllib3

warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(
    page_title="Global Market Monitoring",
    page_icon="🌐",
    layout="wide"
)

# ================== 상단 레이아웃 ==================
col_title, col_img_credit = st.columns([9, 1])
with col_title:
    st.title("🌐 Global Market Monitoring")
    update_clicked = st.button("Update", type="primary", use_container_width=False, key="main_update_btn")
with col_img_credit:
    image_url = "https://amateurphotographer.com/wp-content/uploads/sites/7/2017/08/Screen-Shot-2017-08-23-at-22.29.18.png?w=600.jpg"
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        st.image(img, width=150, caption=None)
    except Exception:
        pass
    st.markdown(
        '<div style="text-align: left; margin-bottom: 3px; font-size:0.9rem;">'
        'Data 출처: <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>'
        '</div>',
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

STYLE_ETFS = {
    'Growth (SPYG)': 'SPYG',
    'Value (SPYV)': 'SPYV',
    'Momentum (MTUM)': 'MTUM',
    'Quality (QUAL)': 'QUAL',
    'Dividend (VIG)': 'VIG',
    'Low Volatility (USMV)': 'USMV'
}

# ====== 섹터 ETF + 주요 종목 (marketmonitor 스타일) ======
SECTOR_ETFS = {
    'IT (XLK)': {'ticker': 'XLK', 'holdings': ['MSFT', 'AAPL', 'NVDA', 'META', 'GOOGL']},
    '헬스케어 (XLV)': {'ticker': 'XLV', 'holdings': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY']},
    '금융 (XLF)': {'ticker': 'XLF', 'holdings': ['JPM', 'BAC', 'WFC', 'GS', 'MS']},
    '커뮤니케이션 (XLC)': {'ticker': 'XLC', 'holdings': ['META', 'GOOGL', 'VZ', 'T', 'DIS']},
    '에너지 (XLE)': {'ticker': 'XLE', 'holdings': ['XOM', 'CVX', 'COP', 'MPC', 'PSX']},
    '산업재 (XLI)': {'ticker': 'XLI', 'holdings': ['BA', 'CAT', 'MMM', 'GE', 'RTX']},
    '소재 (XLB)': {'ticker': 'XLB', 'holdings': ['LIN', 'APD', 'SHW', 'NEM', 'DOW']},
    '필수소비재 (XLP)': {'ticker': 'XLP', 'holdings': ['PG', 'KO', 'WMT', 'MO', 'PM']},
    '자유소비재 (XLY)': {'ticker': 'XLY', 'holdings': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE']},
    '유틸리티 (XLU)': {'ticker': 'XLU', 'holdings': ['NEE', 'DUK', 'SO', 'EXC', 'AEP']},
    '부동산 (XLRE)': {'ticker': 'XLRE', 'holdings': ['PLD', 'AMT', 'CCI', 'EQIX', 'SPG']}
}

# ---- 주요 데이터 함수 ----
def get_perf_table_improved(label2ticker, ref_date=None):
    """성과 테이블 생성"""
    tickers = list(label2ticker.values())
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
        return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    available_dates = df.index[df.index.date <= ref_date]
    if len(available_dates) == 0:
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
    """계산 기준일 표시"""
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

@st.cache_data(show_spinner="차트 데이터 로딩 중...")
def get_normalized_prices(label2ticker, months=6):
    """정규화된 가격 데이터"""
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

def format_percentage(val):
    if pd.isna(val):
        return "N/A"
    try:
        if isinstance(val, (int, float)):
            return f"{val:.2f}%"
    except:
        return "N/A"

def colorize_return(val):
    if pd.isna(val):
        return ""
    try:
        v = float(val) if isinstance(val, (int, float)) else float(str(val).replace('%','').replace(' ',''))
    except (ValueError, TypeError):
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

# ====== Sentiment 분석 함수 (marketmonitor 방식) ======
def get_etf_holdings_enhanced(etf_ticker, retry=3):
    """yahooquery로 ETF 보유종목 수집 (재시도 로직)"""
    for attempt in range(retry):
        try:
            from yahooquery import Ticker
            etf = Ticker(etf_ticker)
            holdings = etf.fund_holding_info
            
            if etf_ticker in holdings and 'holdings' in holdings[etf_ticker]:
                top_holdings = holdings[etf_ticker]['holdings'][:5]
                
                result = []
                for holding in top_holdings:
                    symbol = holding.get('symbol', '')
                    weight = holding.get('holdingPercent', 0.0)
                    
                    if symbol:
                        result.append({
                            'ticker': symbol,
                            'name': holding.get('holdingName', symbol),
                            'weight': weight * 100
                        })
                
                if result:
                    return result
            
            if attempt < retry - 1:
                time.sleep(2)
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2)
            continue
    
    return []

@st.cache_data(ttl=3600)
def collect_news_for_symbols(symbols):
    """Yahoo RSS로 뉴스 수집"""
    import feedparser
    
    all_news = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    for symbol in symbols:
        if not symbol or not isinstance(symbol, str):
            continue
        
        try:
            url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:2]:  # 2개로 제한
                try:
                    title = entry.get('title', '')
                    article_url = entry.get('link', '')
                    pub_date = entry.get('published_parsed')
                    
                    if pub_date:
                        pub_dt = datetime(*pub_date[:6])
                        date_str = pub_dt.strftime('%Y-%m-%d')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    if title:
                        all_news.append({
                            'ticker': symbol,
                            'title': title[:150],
                            'url': article_url,
                            'date': date_str
                        })
                except:
                    continue
            
            time.sleep(0.3)
        except:
            continue
    
    return all_news

def analyze_sentiment_vader(text):
    """VADER 기반 감정 분석"""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    
    try:
        nltk.download('vader_lexicon', quiet=True)
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(text)
        return scores['compound']
    except:
        return 0.0

def show_sector_headlines_enhanced():
    """섹터별 뉴스 헤드라인 (개선버전)"""
    st.subheader("📰 섹터별 주요 종목 뉴스")
    
    for sector_name, sector_info in SECTOR_ETFS.items():
        holdings = sector_info.get('holdings', [])
        
        if holdings:
            st.write(f"#### {sector_name}")
            
            try:
                news_data = collect_news_for_symbols(holdings)
                
                if news_data:
                    for article in news_data[:3]:  # 섹터당 최대 3개
                        sentiment = analyze_sentiment_vader(article['title'])
                        sentiment_emoji = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
                        st.markdown(f"- {sentiment_emoji} **[{article['ticker']}]** {article['title']}")
                else:
                    st.caption(f"- {sector_name}: 뉴스 데이터 없음")
            except Exception as e:
                st.caption(f"- {sector_name}: 데이터 로드 실패")

def show_sector_sentiment_analysis():
    """섹터 감정 분석"""
    st.subheader("✳️✴️ 섹터별 종목 감정 분석")
    
    all_news_data = []
    
    with st.spinner("뉴스 데이터 수집 및 감정 분석 중..."):
        for sector_name, sector_info in SECTOR_ETFS.items():
            holdings = sector_info.get('holdings', [])
            
            try:
                news_data = collect_news_for_symbols(holdings)
                
                for article in news_data:
                    sentiment = analyze_sentiment_vader(article['title'])
                    all_news_data.append({
                        'Sector': sector_name.split()[0],
                        'Ticker': article['ticker'],
                        'Title': article['title'],
                        'Date': article['date'],
                        'Sentiment': sentiment,
                        'URL': article['url']
                    })
            except:
                continue
    
    if not all_news_data:
        st.warning("뉴스 데이터를 수집할 수 없습니다.")
        return
    
    df_sentiment = pd.DataFrame(all_news_data)
    
    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 뉴스 개수", len(df_sentiment))
    with col2:
        st.metric("평균 감정 점수", f"{df_sentiment['Sentiment'].mean():.3f}")
    with col3:
        positive_pct = (df_sentiment['Sentiment'] > 0.1).sum() / len(df_sentiment) * 100 if len(df_sentiment) > 0 else 0
        st.metric("긍정 비율", f"{positive_pct:.1f}%")
    with col4:
        negative_pct = (df_sentiment['Sentiment'] < -0.1).sum() / len(df_sentiment) * 100 if len(df_sentiment) > 0 else 0
        st.metric("부정 비율", f"{negative_pct:.1f}%")
    
    # 차트 1: 종목별 감��
    st.subheader("종목별 감정 점수")
    ticker_sentiment = df_sentiment.groupby('Ticker')['Sentiment'].mean().sort_values().tail(10)
    colors = ['#f44336' if x < -0.1 else '#4CAF50' if x > 0.1 else '#FFC107' for x in ticker_sentiment]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=ticker_sentiment.index, x=ticker_sentiment.values, orientation='h',
                         marker=dict(color=colors), text=[f"{v:.3f}" for v in ticker_sentiment.values], 
                         textposition='outside'))
    fig.update_layout(title="종목별 감정 점수", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 차트 2: 감정 분포
    sentiment_dist = pd.cut(df_sentiment['Sentiment'], bins=[-1, -0.1, 0.1, 1], 
                            labels=['Negative', 'Neutral', 'Positive']).value_counts()
    colors_dist = ['#f44336', '#FFC107', '#4CAF50']
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=sentiment_dist.index, y=sentiment_dist.values, marker_color=colors_dist,
                          text=sentiment_dist.values, textposition='inside'))
    fig2.update_layout(title="감정 분포", height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 상세 데이터
    with st.expander("상세 뉴스 데이터"):
        st.dataframe(df_sentiment.sort_values('Date', ascending=False), use_container_width=True)

def show_all_performance_tables():
    """모든 성과 테이블 표시"""
    perf_cols = ['1D(%)','1W(%)','MTD(%)','1M(%)','3M(%)','6M(%)','YTD(%)','1Y(%)','3Y(%)']
    
    st.subheader("📊 주식시장")
    with st.spinner("주식시장 성과 데이터 계산 중..."):
        stock_perf = get_perf_table_improved(STOCK_ETFS)
    if not stock_perf.empty:
        st.dataframe(
            style_perf_table(stock_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=490
        )
    
    st.subheader("🗠 채권시장")
    with st.spinner("채권시장 성과 데이터 계산 중..."):
        bond_perf = get_perf_table_improved(BOND_ETFS)
    if not bond_perf.empty:
        st.dataframe(
            style_perf_table(bond_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=385
        )
    
    st.subheader("💱 통화")
    with st.spinner("통화 성과 데이터 계산 중..."):
        curr_perf = get_perf_table_improved(CURRENCY)
    if not curr_perf.empty:
        st.dataframe(
            style_perf_table(curr_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=315
        )
    
    st.subheader("📈 암호화폐")
    with st.spinner("암호화폐 성과 데이��� 계산 중..."):
        crypto_perf = get_perf_table_improved(CRYPTO)
    if not crypto_perf.empty:
        st.dataframe(
            style_perf_table(crypto_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=385
        )
    
    st.subheader("📕 스타일 ETF")
    with st.spinner("스타일 ETF 성과 데이터 계산 중..."):
        style_perf = get_perf_table_improved(STYLE_ETFS)
    if not style_perf.empty:
        st.dataframe(
            style_perf_table(style_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=245
        )
    
    st.subheader("📘 섹터 ETF")
    with st.spinner("섹터 ETF 성과 데이터 계산 중..."):
        sector_perf = get_perf_table_improved(
            {name: info['ticker'] for name, info in SECTOR_ETFS.items()}
        )
    if not sector_perf.empty:
        st.dataframe(
            style_perf_table(sector_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=420
        )

# ---- 메인 레이아웃 ----
period_options = {
    "3개월": 3,
    "6개월": 6,
    "12개월": 12,
    "24개월": 24,
    "36개월": 36,
}

if update_clicked:
    st.session_state['updated'] = True

if st.session_state.get('updated', False):
    st.markdown("<br>", unsafe_allow_html=True)
    show_all_performance_tables()
    show_sector_headlines_enhanced()
    
    st.markdown("---")
    
    # 탭으로 차트 분리
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 주가지수 차트", "📗 섹터 차트", "📙 스타일 차트", "📰 감정 분석", "📋 상세 정보"]
    )
    
    with tab1:
        st.subheader("✅ 주요 주가지수 수익률")
        if "idx_months" not in st.session_state:
            st.session_state["idx_months"] = 6
        
        months = st.selectbox(
            "기간 선택", 
            options=list(period_options.keys()),
            index=list(period_options.values()).index(st.session_state["idx_months"]),
            key="idx_selectbox"
        )
        months_val = period_options[months]
        st.session_state["idx_months"] = months_val
        
        with st.spinner("차트 로딩 중..."):
            norm_df = get_normalized_prices(STOCK_ETFS, months=months_val)
            fig = go.Figure()
            for col in norm_df.columns:
                fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines', name=col))
            fig.update_layout(
                yaxis_title="100 기준 누적수익률(%)",
                template="plotly_dark", height=500, legend=dict(orientation='h')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("☑️ 섹터 ETF 수익률")
        if "sector_months" not in st.session_state:
            st.session_state["sector_months"] = 6
        
        months = st.selectbox(
            "기간 선택", 
            options=list(period_options.keys()),
            index=list(period_options.values()).index(st.session_state["sector_months"]),
            key="sector_selectbox"
        )
        months_val = period_options[months]
        st.session_state["sector_months"] = months_val
        
        with st.spinner("차트 로딩 중..."):
            sector_tickers = {name: info['ticker'] for name, info in SECTOR_ETFS.items()}
            norm_df = get_normalized_prices(sector_tickers, months=months_val)
            fig = go.Figure()
            for col in norm_df.columns:
                fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines', name=col))
            fig.update_layout(
                yaxis_title="100 기준 누적수익률(%)",
                template="plotly_dark", height=500, legend=dict(orientation='h')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("☑️ 스타일 ETF 수익률")
        if "style_months" not in st.session_state:
            st.session_state["style_months"] = 6
        
        months = st.selectbox(
            "기간 선택", 
            options=list(period_options.keys()),
            index=list(period_options.values()).index(st.session_state["style_months"]),
            key="style_selectbox"
        )
        months_val = period_options[months]
        st.session_state["style_months"] = months_val
        
        with st.spinner("차트 로딩 중..."):
            norm_df = get_normalized_prices(STYLE_ETFS, months=months_val)
            fig = go.Figure()
            for col in norm_df.columns:
                fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines', name=col))
            fig.update_layout(
                yaxis_title="100 기준 누적수익률(%)",
                template="plotly_dark", height=500, legend=dict(orientation='h')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        show_sector_sentiment_analysis()
    
    with tab5:
        st.subheader("📋 계산 기준일")
        sample_label, last_date, actual_dates = get_sample_calculation_dates(STOCK_ETFS)
        if sample_label and actual_dates:
            st.caption(f"**샘플 자산:** {sample_label} | **최근 거래일:** {last_date}")
            periods_line1 = [f"{period}: {actual_dates[period]}" for period in ['1D', '1W', 'MTD', '1M'] if period in actual_dates]
            st.caption("• " + " | ".join(periods_line1))
            periods_line2 = [f"{period}: {actual_dates[period]}" for period in ['3M', '6M', 'YTD', '1Y', '3Y'] if period in actual_dates]
            st.caption("• " + " | ".join(periods_line2))

else:
    st.info("상단 'Update' 버튼을 눌러주세요.")
