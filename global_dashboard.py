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
import feedparser
from bs4 import BeautifulSoup
import re

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
    update_clicked = st.button("Update", type="primary", width='content', key="main_update_btn")
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

# ====== 섹터 ETF (동적 holdings 로드) ======
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

# ====== ETF Collector (marketmonitor ��식) ======
class ETFCollector:
    """동적 ETF Holdings 수집"""
    
    def __init__(self):
        try:
            from curl_cffi import requests as cffi_requests
            self.session = cffi_requests.Session(impersonate="chrome")
            self.session.verify = False
        except:
            self.session = None
    
    def get_etf_holdings(self, ticker: str, retry=3):
        """Top 10 Holdings 동적 로드"""
        for attempt in range(retry):
            try:
                from yahooquery import Ticker
                
                if self.session:
                    etf = Ticker(ticker, session=self.session)
                else:
                    etf = Ticker(ticker)
                
                holdings = etf.fund_holding_info
                
                if ticker in holdings and 'holdings' in holdings[ticker]:
                    top_holdings = holdings[ticker]['holdings'][:10]
                    
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
    
    def get_etf_name(self, ticker: str):
        """ETF 이름"""
        try:
            from yahooquery import Ticker
            
            if self.session:
                etf = Ticker(ticker, session=self.session)
            else:
                etf = Ticker(ticker)
            
            quote_type = etf.quote_type
            
            if ticker in quote_type:
                return quote_type[ticker].get('longName', f'{ticker} ETF')
            
            return f'{ticker} ETF'
        except:
            return f'{ticker} ETF'

# ====== News Collector (경량 버전) ======
class NewsCollector:
    """Yahoo RSS 기반 뉴스 수집"""
    
    def __init__(self, days=3):
        self.days = days
        self.cutoff_date = datetime.now() - timedelta(days=days)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_content(self, url: str):
        """본문 추출"""
        try:
            response = requests.get(url, headers=self.headers, timeout=8)
            if response.status_code != 200:
                return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            paragraphs = soup.find_all('p')
            texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]
            
            full_text = ' '.join(texts)
            
            return full_text[:5000] if full_text else ""
        except:
            return ""
    
    def is_valid_content(self, content: str):
        """Paywall 체크"""
        if not content or len(content) < 200:
            return False
        
        paywall_words = ['sign in', 'log in', 'subscribe', 'register']
        content_lower = content.lower()
        
        for word in paywall_words:
            if word in content_lower[:500]:
                return False
        
        return True
    
    def create_summary(self, text: str):
        """추출식 요약"""
        if not text or len(text) < 20:
            return ""
        
        sentences = re.split(r'[.!?]\s+', text)
        
        summary_sentences = []
        total_len = 0
        
        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if len(sentence) > 20:
                summary_sentences.append(sentence)
                total_len += len(sentence)
                
                if total_len >= 300:
                    break
        
        summary = '. '.join(summary_sentences)
        
        if len(summary) > 300:
            summary = summary[:300] + '...'
        
        return summary if summary else text[:300] + '...'
    
    def collect_yahoo_rss(self, ticker: str):
        """Yahoo Finance RSS"""
        try:
            url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            feed = feedparser.parse(url)
            
            news = []
            for entry in feed.entries[:3]:
                try:
                    pub_date = entry.get('published_parsed')
                    if pub_date:
                        pub_dt = datetime(*pub_date[:6])
                        if pub_dt < self.cutoff_date:
                            continue
                        date_str = pub_dt.strftime('%Y-%m-%d')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    title = entry.get('title', '')
                    article_url = entry.get('link', '')
                    summary = entry.get('summary', '')
                    
                    content = self.extract_content(article_url)
                    
                    if not self.is_valid_content(content):
                        continue
                    
                    highlights = self.create_summary(content)
                    
                    news.append({
                        'ticker': ticker,
                        'title': title,
                        'url': article_url,
                        'published_at': date_str,
                        'summary': summary[:300],
                        'content': content,
                        'highlights': highlights,
                        'source': 'Yahoo Finance'
                    })
                except:
                    continue
            
            return news
        except:
            return []
    
    def collect_for_ticker(self, ticker: str, company: str):
        """티커별 뉴스"""
        all_news = []
        
        yahoo_news = self.collect_yahoo_rss(ticker)
        all_news.extend(yahoo_news)
        
        for item in all_news:
            item['company_name'] = company
        
        return all_news
    
    def collect_all(self, holdings, etf_ticker: str):
        """전체 수집"""
        all_news = []
        
        for idx, holding in enumerate(holdings):
            ticker = holding['ticker']
            company = holding['name']
            
            news = self.collect_for_ticker(ticker, company)
            
            for item in news:
                item['etf'] = etf_ticker
                item['weight'] = holding['weight']
            
            all_news.extend(news)
            time.sleep(0.3)
        
        return all_news

# ====== Sentiment Analyzer (FinBERT 경량 버전) ======
class FinBERTAnalyzer:
    """경량 FinBERT"""
    
    def __init__(self):
        self.pipe = None
        self._initialize()
    
    def _initialize(self):
        """모델 로드"""
        try:
            from transformers import pipeline
            self.pipe = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=-1,
                max_length=512,
                truncation=True
            )
        except Exception as e:
            self.pipe = None
    
    def analyze_chunk(self, text: str):
        """단일 청크 분석"""
        if not self.pipe or not text or len(text) < 10:
            return 0.0
        
        try:
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            result = self.pipe(text[:512])[0]
            
            label = result['label']
            score = result['score']
            
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:
                return 0.0
        except:
            return 0.0
    
    def analyze_text(self, text: str):
        """텍스트 분석 (최대 3개 청크)"""
        if not text or len(text) < 100:
            return 0.0
        
        chunk_size = 1000
        
        chunks = []
        for i in range(0, min(len(text), 3000), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk) > 100:
                chunks.append(chunk)
        
        scores = []
        for chunk in chunks[:3]:
            score = self.analyze_chunk(chunk)
            if score != 0.0:
                scores.append(score)
        
        if scores:
            return sum(scores) / len(scores)
        
        return 0.0
    
    def categorize(self, title: str):
        """카테고리"""
        title_lower = title.lower()
        
        if any(w in title_lower for w in ['earnings', 'revenue', 'profit']):
            return 'Earnings'
        elif any(w in title_lower for w in ['merger', 'acquisition', 'deal']):
            return 'M&A'
        elif any(w in title_lower for w in ['product', 'launch']):
            return 'Product'
        elif any(w in title_lower for w in ['regulation', 'lawsuit']):
            return 'Regulatory'
        elif any(w in title_lower for w in ['analyst', 'upgrade', 'downgrade']):
            return 'Analyst'
        else:
            return 'General'
    
    def analyze_news(self, news: dict):
        """뉴스 분석"""
        content = news.get('content', '')
        
        if not content or len(content) < 100:
            return None
        
        sentiment = self.analyze_text(content)
        
        news['sentiment_score'] = round(sentiment, 4)
        news['category'] = self.categorize(news.get('title', ''))
        
        return news
    
    def batch_analyze(self, news_list: list):
        """일괄 분석"""
        analyzed = []
        
        for idx, news in enumerate(news_list):
            if (idx + 1) % 5 == 0:
                print(f"  분석: {idx + 1}/{len(news_list)}")
            
            result = self.analyze_news(news)
            if result:
                analyzed.append(result)
        
        return analyzed

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
    """계산 기준일"""
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
    """정규화된 가격"""
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

# ====== 섹터 분석 (marketmonitor 방식) ======
@st.cache_resource
def load_analyzer():
    return FinBERTAnalyzer()

def run_sector_etf_analysis(etf_ticker: str, etf_name: str):
    """단일 섹터 ETF 분석"""
    try:
        # Holdings
        collector = ETFCollector()
        holdings = collector.get_etf_holdings(etf_ticker)
        
        if not holdings:
            return None, f"❌ {etf_name}: Holdings 없음"
        
        # 뉴스 수집
        news_collector = NewsCollector(days=3)
        all_news = news_collector.collect_all(holdings, etf_ticker)
        
        if not all_news:
            return holdings, f"⚠️ {etf_name}: 뉴스 없음"
        
        # 감정 분석
        analyzer = load_analyzer()
        analyzed = analyzer.batch_analyze(all_news)
        
        return analyzed, None
    
    except Exception as e:
        return None, f"❌ {etf_name}: {str(e)[:50]}"

def show_sector_analysis():
    """섹터 분석"""
    st.subheader("📰 섹터별 주요 종목 뉴스 & 감정 분석")
    
    sector_results = {}
    
    for sector_name, etf_ticker in SECTOR_ETFS.items():
        try:
            with st.spinner(f"{sector_name} 분석 중..."):
                analyzed_news, error = run_sector_etf_analysis(etf_ticker, sector_name)
                
                if error:
                    st.warning(error)
                elif analyzed_news:
                    sector_results[sector_name] = analyzed_news
        except Exception as e:
            st.warning(f"❌ {sector_name}: 오류 발생")
    
    if not sector_results:
        st.warning("섹터 분석 데이터를 가져올 수 없습니다.")
        return
    
    # 섹터별 표시
    for sector_name, news_list in sector_results.items():
        st.markdown(f"#### {sector_name}")
        
        if news_list:
            # 메트릭
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("뉴스 개수", len(news_list))
            with col2:
                avg_sentiment = np.mean([n.get('sentiment_score', 0) for n in news_list])
                st.metric("평균 감정", f"{avg_sentiment:.3f}")
            with col3:
                positive_count = sum(1 for n in news_list if n.get('sentiment_score', 0) > 0.1)
                st.metric("긍정 뉴스", f"{positive_count}/{len(news_list)}")
            
            # 상세 데이터
            news_df = pd.DataFrame([{
                'Ticker': n.get('ticker', ''),
                'Title': n.get('title', '')[:80],
                'Date': n.get('published_at', ''),
                'Sentiment': round(n.get('sentiment_score', 0), 3),
                'Category': n.get('category', ''),
            } for n in news_list[:10]])
            
            st.dataframe(news_df, width='stretch') #width='stretch'
        
        st.markdown("---")

def show_all_performance_tables():
    """성과 테이블"""
    perf_cols = ['1D(%)','1W(%)','MTD(%)','1M(%)','3M(%)','6M(%)','YTD(%)','1Y(%)','3Y(%)']
    
    st.subheader("📊 주식시장")
    with st.spinner("주식시장 성과 데이터 계산 중..."):
        stock_perf = get_perf_table_improved(STOCK_ETFS)
    if not stock_perf.empty:
        st.dataframe(
            style_perf_table(stock_perf.set_index('자산명'), perf_cols),
            width='stretch', height=490
        )
    
    st.subheader("🗠 채권시장")
    with st.spinner("채권시장 성과 데이터 계산 중..."):
        bond_perf = get_perf_table_improved(BOND_ETFS)
    if not bond_perf.empty:
        st.dataframe(
            style_perf_table(bond_perf.set_index('자산명'), perf_cols),
            width='stretch', height=385
        )
    
    st.subheader("💱 통화")
    with st.spinner("통화 성과 데이터 계산 중..."):
        curr_perf = get_perf_table_improved(CURRENCY)
    if not curr_perf.empty:
        st.dataframe(
            style_perf_table(curr_perf.set_index('자산명'), perf_cols),
            width='stretch', height=315
        )
    
    st.subheader("📈 암호화폐")
    with st.spinner("암호화폐 성과 데이터 계산 중..."):
        crypto_perf = get_perf_table_improved(CRYPTO)
    if not crypto_perf.empty:
        st.dataframe(
            style_perf_table(crypto_perf.set_index('자산명'), perf_cols),
            width='stretch', height=385
        )
    
    st.subheader("📕 스타일 ETF")
    with st.spinner("스타일 ETF 성과 데이터 계산 중..."):
        style_perf = get_perf_table_improved(STYLE_ETFS)
    if not style_perf.empty:
        st.dataframe(
            style_perf_table(style_perf.set_index('자산명'), perf_cols),
            width='stretch', height=245
        )
    
    st.subheader("📘 섹터 ETF")
    with st.spinner("섹터 ETF 성과 데이터 계산 중..."):
        sector_perf = get_perf_table_improved(SECTOR_ETFS)
    if not sector_perf.empty:
        st.dataframe(
            style_perf_table(sector_perf.set_index('���산명'), perf_cols),
            width='stretch', height=420
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
    
    st.markdown("---")
    
    # 탭 분리
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 주가지수 차트", "📗 섹터 차트", "📙 스타일 차트", "📰 섹터 분석", "📋 정보"]
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
            st.plotly_chart(fig, width='stretch')
    
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
            norm_df = get_normalized_prices(SECTOR_ETFS, months=months_val)
            fig = go.Figure()
            for col in norm_df.columns:
                fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines', name=col))
            fig.update_layout(
                yaxis_title="100 기준 누적수익률(%)",
                template="plotly_dark", height=500, legend=dict(orientation='h')
            )
            st.plotly_chart(fig, width='stretch')
    
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
            st.plotly_chart(fig, width='stretch')
    
    with tab4:
        show_sector_analysis()
    
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
