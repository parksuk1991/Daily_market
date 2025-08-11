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
import matplotlib.pyplot as plt

# =================== 추가: LLM & 뉴스 관련 패키지 ===================
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from transformers import pipeline, logging as hf_logging
hf_logging.set_verbosity_error()
from GoogleNews import GoogleNews
from deep_translator import GoogleTranslator

# =================== Streamlit 페이지 세팅 ===================
try:
    import lxml
except ImportError:
    st.error("lxml 패키지가 필요합니다. requirements.txt에 lxml을 추가하세요.")

st.set_page_config(
    page_title="Global Market Monitoring",
    page_icon="🌐",
    layout="wide"
)

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
        st.info("이미지를 불러올 수 없습니다.")
    st.markdown(
        '<div style="text-align: left; margin-bottom: 3px; font-size:0.9rem;">'
        'Data 출처: <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>'
        '</div>',
        unsafe_allow_html=True
    )

# =================== 자산 정의 ===================
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

# =================== 데이터/차트 함수 ===================
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

@st.cache_data(show_spinner="차트 데이터 로딩 중...")
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

def format_percentage(val):
    if pd.isna(val):
        return "N/A"
    try:
        if isinstance(val, (int, float)):
            return f"{val:.6f}"
    except:
        return "N/A"

def colorize_return(val):
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

# =================== LLM 기반 뉴스/감정/번역 함수 ===================
@st.cache_resource
def get_hf_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-xsum")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return summarizer, sentiment_analyzer

def translate_to_korean(text):
    if not text or len(text.strip()) == 0:
        return ""
    try:
        return GoogleTranslator(source='auto', target='ko').translate(text)
    except Exception:
        return "[번역 실패]"

def get_google_news(ticker):
    googlenews = GoogleNews(lang='en')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%m/%d/%Y')
    end_date = datetime.now().strftime('%m/%d/%Y')
    googlenews.set_time_range(start_date, end_date)
    googlenews.search(ticker)
    results = googlenews.result()
    news_data = []
    for r in results:
        news_data.append({
            "Ticker": ticker,
            "Date": r.get('date'),
            "Title": r.get('title'),
            "Media": r.get('media'),
            "Link": r.get('link'),
            "Description": r.get('desc')
        })
    return news_data

def analyze_news(df, summarizer, sentiment_analyzer):
    summaries, sentiments, final_scores, desc_ko, summary_ko = [], [], [], [], []
    for desc in df["Description"]:
        if not desc or len(desc.strip()) == 0:
            summaries.append("")
            sentiments.append("neutral")
            final_scores.append(0)
            desc_ko.append("")
            summary_ko.append("")
            continue
        try:
            summary = summarizer(desc, max_length=100, min_length=15, do_sample=False)[0]['summary_text']
        except:
            summary = desc[:300]
        summaries.append(summary)
        try:
            desc_sent = sentiment_analyzer(desc)[0]
            desc_label = desc_sent["label"].lower()
            desc_score = desc_sent["score"] if desc_label == "positive" else -desc_sent["score"] if desc_label == "negative" else 0
        except:
            desc_label = "neutral"
            desc_score = 0
        try:
            summ_sent = sentiment_analyzer(summary)[0]
            summ_label = summ_sent["label"].lower()
            summ_score = summ_sent["score"] if summ_label == "positive" else -summ_sent["score"] if summ_label == "negative" else 0
        except:
            summ_label = "neutral"
            summ_score = 0
        final_score = desc_score * 0.5 + summ_score * 0.5
        sentiments.append(summ_label)
        final_scores.append(final_score)
        desc_ko.append(translate_to_korean(desc))
        summary_ko.append(translate_to_korean(summary))
    df["Summary"] = summaries
    df["Sentiment"] = sentiments
    df["Sentiment_Score"] = final_scores
    df["Description_KO"] = desc_ko
    df["Summary_KO"] = summary_ko
    return df

@st.cache_data(show_spinner="뉴스 & 감정 분석 로딩 중...")
def get_sector_news_sentiment():
    summarizer, sentiment_analyzer = get_hf_pipelines()
    all_news = []
    sector_to_syms = {}
    for sector_label, etf in SECTOR_ETFS.items():
        top_holdings = get_top_holdings(etf, n=3)
        holding_syms = [sym for sym, _ in top_holdings]
        sector_to_syms[sector_label] = holding_syms
        for sym in holding_syms:
            news_list = get_google_news(sym)
            all_news.extend(news_list)
    if not all_news:
        return pd.DataFrame(), sector_to_syms
    df_news = pd.DataFrame(all_news)
    df_news = analyze_news(df_news, summarizer, sentiment_analyzer)
    return df_news, sector_to_syms

def show_sector_news_sentiment():
    st.subheader("🔍 섹터별 주요 종목 뉴스 및 감정 점수")
    with st.spinner("뉴스 및 감정 분석 중..."):
        df, sector_syms = get_sector_news_sentiment()
    if df.empty:
        st.warning("뉴스 데이터를 찾을 수 없습니다.")
        return
    st.dataframe(
        df[["Ticker", "Date", "Title", "Description", "Summary", "Sentiment", "Sentiment_Score", "Description_KO", "Summary_KO"]],
        use_container_width=True, height=min(900, 30 + 30*len(df))
    )
    st.markdown("#### 섹터별 종목별 평균 감정 점수")
    mean_scores = df.groupby("Ticker")["Sentiment_Score"].mean().reset_index()
    fig = px.bar(mean_scores, x="Ticker", y="Sentiment_Score", color="Sentiment_Score", color_continuous_scale="RdBu")
    st.plotly_chart(fig, use_container_width=True)

# =================== 기존 Sentiment 분석 & 애널리스트/EPS 등 표 함수들 유지 ===================
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

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
                '애널리스트 등급 점수': info.get('recommendationMean'),
                '애널리스트 등급': info.get('recommendationKey'),
                '애널리스트 목표가': target_price,
                '현재가': current_price,
                '상승여력': upside
            })
        except Exception:
            rows.append({
                'Ticker': sym,
                '종목명': '',
                '애널리스트 등급 점수': None,
                '애널리스트 등급': None,
                '애널리스트 목표가(평균)': None,
                '현재가': None,
                '상승여력': None
            })
    df = pd.DataFrame(rows)
    df = df[['Ticker', '종목명', '애널리스트 등급 점수', '애널리스트 등급', '애널리스트 목표가', '현재가', '상승여력']]
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

# =================== 차트 부분별 기간 선택 UI & 렌더링 ===================
period_options = {
    "3개월": 3,
    "6개월": 6,
    "12개월": 12,
    "24개월": 24,
    "36개월": 36,
}

def render_normalized_chart(title, etf_dict, key, default_val):
    st.subheader(f"{title}")
    if f"{key}_months" not in st.session_state:
        st.session_state[f"{key}_months"] = default_val
    months = st.selectbox(
        "기간 선택", options=list(period_options.keys()),
        index=list(period_options.values()).index(st.session_state[f"{key}_months"]),
        key=f"{key}_selectbox"
    )
    months_val = period_options[months]
    st.session_state[f"{key}_months"] = months_val
    if st.session_state.get('updated', False):
        norm_df = get_normalized_prices(etf_dict, months=months_val)
        fig = go.Figure()
        for col in norm_df.columns:
            fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines', name=col))
        fig.update_layout(
            yaxis_title="100 기준 누적수익률(%)",
            template="plotly_dark", height=500, legend=dict(orientation='h')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("차트 갱신을 위해 상단 'Update' 버튼을 눌러주세요.")

# =================== 성과, 뉴스 모니터링 대시보드 메인 ===================
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
    st.subheader("💱 통화")
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

# =================== 전체 대시보드 구동 ===================
if update_clicked:
    st.session_state['updated'] = True

if st.session_state.get('updated', False):
    st.markdown("<br>", unsafe_allow_html=True)
    show_all_performance_tables()
    render_normalized_chart("✅ 주요 주가지수 수익률", STOCK_ETFS, "idx", 6)
    render_normalized_chart("☑️ 섹터 ETF 수익률", SECTOR_ETFS, "sector", 6)
    render_normalized_chart("☑️ 스타일 ETF 수익률", STYLE_ETFS, "style", 6)
    st.subheader("📰 섹터별 주요 종목 헤드라인 및 감정 분석")
    show_sector_news_sentiment()
    # 아래 기존 show_sentiment_analysis() 등은 필요시 추가적으로 LLM 뉴스와 별개로 사용할 수 있음
else:
    st.info("상단 'Update' 버튼을 눌러주세요.")
