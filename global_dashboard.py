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
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Global Market Monitoring", page_icon="🌐", layout="wide")

# ===== 자산 정의 ===== 이건 테스트용이고 추후에 실제 펀드 보유 자산으로 동적으로 변경 필요 
STOCK_ETFS = {
    'S&P 500 (SPY)': 'SPY', 'NASDAQ 100 (QQQ)': 'QQQ', '전세계 (ACWI)': 'ACWI',
    '선진국 (VEA)': 'VEA', '신흥국 (VWO)': 'VWO', '유럽(Europe, VGK)': 'VGK',
    '중국(China, MCHI)': 'MCHI', '일본(Japan, EWJ)': 'EWJ', '한국(KOSPI, EWY)': 'EWY',
    '인도(INDIA, INDA)': 'INDA', '영국(UK, EWU)': 'EWU', '브라질(Brazil, EWZ)': 'EWZ',
    '캐나다(Canada, EWC)': 'EWC',
}
BOND_ETFS = {
    '미국 장기국채(TLT)': 'TLT', '미국 단기국채(SHY)': 'SHY', '미국 IG회사채(LQD)': 'LQD',
    '신흥국채(EMB)': 'EMB', '미국 하이일드(HYG)': 'HYG', '미국 물가연동(TIP)': 'TIP',
    '미국 단기회사채(VCSH)': 'VCSH', '글로벌국채(BNDX)': 'BNDX', '미국 국채(BND)': 'BND',
    '단기국채(SPTS)': 'SPTS',
}
CURRENCY = { 
    '달러-원': 'KRW=X', '유로-원': 'EURKRW=X',
    '달러-엔': 'JPY=X', '원-엔': 'JPYKRW=X', '달러-유로': 'EURUSD=X',
    '달러-파운드': 'GBPUSD=X', '달러-위안': 'CNY=X',
}
CRYPTO = {
    '비트코인 (BTC)': 'BTC-USD', '이더리움 (ETH)': 'ETH-USD', '솔라나 (SOL)': 'SOL-USD',
    '리플 (XRP)': 'XRP-USD', '에이다 (ADA)': 'ADA-USD', '라이트코인 (LTC)': 'LTC-USD',
    '비트코인캐시 (BCH)': 'BCH-USD', '체인링크 (LINK)': 'LINK-USD',
    '도지코인 (DOGE)': 'DOGE-USD', '아발란체 (AVAX)': 'AVAX-USD',
}
STYLE_ETFS = {
    'Growth (SPYG)': 'SPYG', 'Value (SPYV)': 'SPYV', 'Momentum (MTUM)': 'MTUM',
    'Quality (QUAL)': 'QUAL', 'Dividend (VIG)': 'VIG', 'Low Volatility (USMV)': 'USMV',
}
SECTOR_ETFS = {
    'IT (XLK)': 'XLK', '헬스케어 (XLV)': 'XLV', '금융 (XLF)': 'XLF',
    '커뮤니케이션 (XLC)': 'XLC', '에너지 (XLE)': 'XLE', '산업재 (XLI)': 'XLI',
    '소재 (XLB)': 'XLB', '필수소비재 (XLP)': 'XLP', '자유소비재 (XLY)': 'XLY',
    '유틸리티 (XLU)': 'XLU', '부동산 (XLRE)': 'XLRE',
}

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

PERIOD_OPTIONS = [("6개월", 6), ("1년", 12), ("2년", 24), ("3년", 36)]
TITLE_COLOR = "#605c4c"

def create_transparent_wistia_cmap(alpha=0.4):
    """Wistia 컬러맵에 투명도를 적용한 커스텀 컬러맵 생성"""
    wistia_cmap = cm.get_cmap('Wistia')
    colors = [wistia_cmap(i) for i in np.linspace(0, 1, wistia_cmap.N)]
    colors_with_alpha = [(r, g, b, alpha) for r, g, b, a in colors]
    return mcolors.ListedColormap(colors_with_alpha)

# ======================================================
# ETF Collector
# ======================================================
class ETFCollector:
    def __init__(self):
        self.cf_session = None
        try:
            from curl_cffi import requests as cffi_req
            self.cf_session = cffi_req.Session(impersonate="chrome")
            self.cf_session.verify = False
        except Exception:
            pass

    def _try_ssga(self, ticker: str):
        url = (
            "https://www.ssga.com/us/en/intermediary/etfs/library-content/"
            f"products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15, verify=False)
            if resp.status_code != 200:
                return []
            df = pd.read_excel(BytesIO(resp.content), skiprows=4, engine='openpyxl')
            if df.empty:
                return []
            df.columns = [str(c).strip() for c in df.columns]
            tc = next((c for c in df.columns if 'ticker' in c.lower() or 'symbol' in c.lower()), None)
            wc = next((c for c in df.columns if 'weight' in c.lower()), None)
            nc = next((c for c in df.columns if 'name' in c.lower()), tc)
            if not tc or not wc:
                return []
            df = df.dropna(subset=[tc])
            df[tc] = df[tc].astype(str).str.strip()
            df = df[df[tc].str.len() > 0]
            df = df[~df[tc].str.lower().isin(['nan', 'ticker', 'symbol', '-'])]
            df[wc] = pd.to_numeric(df[wc], errors='coerce')
            df = df.dropna(subset=[wc])
            df = df[df[wc] > 0].sort_values(wc, ascending=False).head(10)
            return [{'ticker': str(r[tc]).strip(), 'name': str(r[nc]).strip(),
                     'weight': round(float(r[wc]), 2)} for _, r in df.iterrows()]
        except Exception:
            return []

    def _try_stockanalysis(self, ticker: str):
        url = f"https://stockanalysis.com/etf/{ticker.lower()}/holdings/"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=12, verify=False)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table')
            if not table:
                return []
            result = []
            for row in table.find_all('tr')[1:]:
                cols = [td.get_text(strip=True) for td in row.find_all('td')]
                if len(cols) < 4:
                    continue
                sym = cols[1].strip()
                name = cols[2].strip() if len(cols) > 2 else sym
                try:
                    weight = float(cols[3].replace('%', '').replace(',', '').strip())
                except ValueError:
                    continue
                if sym and weight > 0:
                    result.append({'ticker': sym, 'name': name, 'weight': round(weight, 2)})
                if len(result) >= 10:
                    break
            return result
        except Exception:
            return []

    def _try_yahooquery(self, ticker: str):
        try:
            from yahooquery import Ticker
            etf = Ticker(ticker, session=self.cf_session) if self.cf_session else Ticker(ticker)
            holdings = etf.fund_holding_info
            if ticker in holdings and isinstance(holdings[ticker], dict):
                raw = holdings[ticker].get('holdings', [])
                return [{'ticker': h.get('symbol', ''), 'name': h.get('holdingName', h.get('symbol', '')),
                         'weight': round(h.get('holdingPercent', 0.0) * 100, 2)}
                        for h in raw[:10] if h.get('symbol', '')]
        except Exception:
            pass
        return []

    def _try_yfinance(self, ticker: str):
        try:
            df = yf.Ticker(ticker).fund_top_holdings
            if df is None or df.empty:
                return []
            result = []
            for _, row in df.head(10).iterrows():
                sym = str(row.get('Symbol', row.get('symbol', ''))).strip()
                name = str(row.get('Holding Name', row.get('holdingName', sym))).strip()
                pct = float(row.get('% Assets', row.get('holdingPercent', 0)) or 0)
                if pct < 1:
                    pct *= 100
                if sym and sym != 'nan':
                    result.append({'ticker': sym, 'name': name, 'weight': round(pct, 2)})
            return result
        except Exception:
            return []

    def get_etf_holdings(self, ticker: str, retry: int = 2):
        result = self._try_ssga(ticker)
        if result:
            return result
        result = self._try_stockanalysis(ticker)
        if result:
            return result
        for attempt in range(retry):
            result = self._try_yahooquery(ticker)
            if result:
                return result
            if attempt < retry - 1:
                time.sleep(1.5)
        return self._try_yfinance(ticker)


# ======================================================
# News Collector
# ======================================================
class NewsCollector:
    def __init__(self, days: int = 3):
        self.days = days
        self.cutoff_date = datetime.now() - timedelta(days=days)

    def extract_content(self, url: str) -> str:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=8)
            if resp.status_code != 200:
                return ""
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            texts = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 30]
            full = ' '.join(texts)
            return full[:5000] if full else ""
        except Exception:
            return ""

    def is_valid_content(self, content: str) -> bool:
        if not content or len(content) < 200:
            return False
        lower = content.lower()
        return not any(w in lower[:500] for w in ['sign in', 'log in', 'subscribe', 'register'])

    def is_relevant(self, title: str, ticker: str, company_name: str) -> bool:
        title_lower = title.lower()
        ticker_clean = re.sub(r'[^a-zA-Z0-9]', '', ticker).lower()
        if re.search(r'\b' + re.escape(ticker_clean) + r'\b', title_lower):
            return True
        name_lower = company_name.lower().strip()
        if name_lower and len(name_lower) >= 4:
            if name_lower in title_lower:
                return True
            first_word = name_lower.split()[0]
            if len(first_word) >= 5 and re.search(r'\b' + re.escape(first_word) + r'\b', title_lower):
                return True
        return False

    def create_summary(self, text: str) -> str:
        if not text or len(text) < 20:
            return ""
        sentences = re.split(r'[.!?]\s+', text)
        parts, total = [], 0
        for s in sentences[:5]:
            s = s.strip()
            if len(s) > 20:
                parts.append(s)
                total += len(s)
                if total >= 300:
                    break
        summary = '. '.join(parts)
        return (summary[:300] + '...') if len(summary) > 300 else (summary or text[:300] + '...')

    def collect_yahoo_rss(self, ticker: str, company_name: str) -> list:
        try:
            feed = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={ticker}")
            news = []
            for entry in feed.entries:
                try:
                    pub = entry.get('published_parsed')
                    if pub:
                        pub_dt = datetime(*pub[:6])
                        if pub_dt < self.cutoff_date:
                            continue
                        date_str = pub_dt.strftime('%Y-%m-%d')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')

                    title = entry.get('title', '')
                    article_url = entry.get('link', '')

                    if not self.is_relevant(title, ticker, company_name):
                        continue

                    content = self.extract_content(article_url)
                    if not self.is_valid_content(content):
                        continue

                    news.append({
                        'ticker': ticker,
                        'title': title,
                        'url': article_url,
                        'published_at': date_str,
                        'summary': entry.get('summary', '')[:300],
                        'content': content,
                        'highlights': self.create_summary(content),
                        'source': 'Yahoo Finance',
                    })
                except Exception:
                    continue
            return news
        except Exception:
            return []

    def collect_for_ticker(self, ticker: str, company: str) -> list:
        news = self.collect_yahoo_rss(ticker, company)
        for item in news:
            item['company_name'] = company
        return news

    def collect_all(self, holdings: list, etf_ticker: str) -> list:
        all_news = []
        for holding in holdings:
            news = self.collect_for_ticker(holding['ticker'], holding['name'])
            for item in news:
                item['etf'] = etf_ticker
                item['weight'] = holding['weight']
            all_news.extend(news)
            time.sleep(0.3)
        return all_news


# ======================================================
# FinBERT Analyzer
# ======================================================
class FinBERTAnalyzer:
    def __init__(self):
        self.pipe = None
        try:
            from transformers import pipeline as hf_pipeline
            self.pipe = hf_pipeline(
                "text-classification", model="ProsusAI/finbert",
                device=-1, max_length=512, truncation=True,
            )
        except Exception:
            pass

    def analyze_chunk(self, text: str) -> float:
        if not self.pipe or not text or len(text) < 10:
            return 0.0
        try:
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            res = self.pipe(text[:512])[0]
            label, score = res['label'], res['score']
            return score if label == 'positive' else (-score if label == 'negative' else 0.0)
        except Exception:
            return 0.0

    def analyze_text(self, text: str) -> float:
        if not text or len(text) < 100:
            return 0.0
        chunks = [text[i:i+1000] for i in range(0, min(len(text), 3000), 1000)
                  if len(text[i:i+1000]) > 100]
        scores = [s for s in (self.analyze_chunk(c) for c in chunks[:3]) if s != 0.0]
        return sum(scores) / len(scores) if scores else 0.0

    def categorize(self, title: str) -> str:
        t = title.lower()
        if any(w in t for w in ['earnings', 'revenue', 'profit']):
            return 'Earnings'
        if any(w in t for w in ['merger', 'acquisition', 'deal']):
            return 'M&A'
        if any(w in t for w in ['product', 'launch']):
            return 'Product'
        if any(w in t for w in ['regulation', 'lawsuit']):
            return 'Regulatory'
        if any(w in t for w in ['analyst', 'upgrade', 'downgrade']):
            return 'Analyst'
        return 'General'

    def analyze_news(self, news: dict) -> dict:
        content = news.get('content', '')
        news['sentiment_score'] = (
            round(self.analyze_text(content), 4) if len(content) >= 100 else 0.0
        )
        news['category'] = self.categorize(news.get('title', ''))
        return news

    def batch_analyze(self, news_list: list) -> list:
        return [self.analyze_news(n) for n in news_list]


@st.cache_resource
def load_analyzer():
    return FinBERTAnalyzer()


# ======================================================
# Sentiment Analysis Chart Functions
# ======================================================
def build_sentiment_df(news_list: list) -> pd.DataFrame:
    rows = []
    for n in news_list:
        s = n.get('sentiment_score', 0.0) or 0.0
        rows.append({
            'Ticker': n.get('ticker', ''),
            'Company': n.get('company_name', ''),
            'Date': n.get('published_at', ''),
            'Title': n.get('title', ''),
            'Headline': n.get('highlights', ''),
            'Sentiment': float(s),
            'Category': n.get('category', ''),
            'URL': n.get('url', ''),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['Sentiment_Category'] = df['Sentiment'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    return df


def render_sentiment_bar_chart(df: pd.DataFrame, sector_name: str):
    if df.empty:
        return
    agg = (df.groupby('Ticker')['Sentiment']
             .agg(['mean', 'count'])
             .reset_index()
             .rename(columns={'mean': 'Avg', 'count': 'N'})
             .sort_values('Avg', ascending=True))

    company_map = df.drop_duplicates('Ticker').set_index('Ticker')['Company'].to_dict()
    colors = ['#FFBC00' if v > 0.05 else ('#e74c3c' if v < -0.05 else '#95a5a6')
              for v in agg['Avg']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg['Avg'],
        y=agg['Ticker'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f} ({n}건)" for v, n in zip(agg['Avg'], agg['N'])],
        textposition='outside',
        customdata=[company_map.get(t, '') for t in agg['Ticker']],
        hovertemplate='<b>%{y}</b> (%{customdata})<br>Sentiment: %{x:.4f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_width=1, line_dash='dash', line_color='white', opacity=0.4)
    fig.update_layout(
        title=dict(text=f"📊 {sector_name} — 종목별 평균 감정 점수", font=dict(size=14)),
        xaxis=dict(title="Sentiment Score", range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1]),
        yaxis=dict(title=""),
        template="plotly_white",
        height=max(260, len(agg) * 44),
        margin=dict(l=70, r=130, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def create_sentiment_histogram(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Sentiment'], nbinsx=20,
        name='Sentiment Distribution',
        marker_color='rgba(255, 188, 0, 0.7)', opacity=0.8,
    ))
    hist, bin_edges = np.histogram(df['Sentiment'], bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed = ndimage.gaussian_filter1d(hist, 1)
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=smoothed * len(df) * (bin_edges[1] - bin_edges[0]),
        mode='lines', name='KDE',
        line=dict(color='royalblue', width=2),
    ))
    fig.update_layout(
        title='감정 점수 분포', xaxis_title='감정 점수', yaxis_title='빈도',
        template="plotly_white", height=380, showlegend=True,
    )
    return fig


def create_sentiment_countplot(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    counts = df['Sentiment_Category'].value_counts().reindex(
        ['Positive', 'Neutral', 'Negative'], fill_value=0
    ).reset_index()
    counts.columns = ['Category', 'Count']
    color_map = {
        'Positive': 'rgba(255,188,0,0.8)',
        'Neutral': 'rgba(102,194,165,0.8)',
        'Negative': 'rgba(230,126,34,0.8)',
    }
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts['Category'],
        y=counts['Count'],
        marker_color=[color_map.get(c, 'grey') for c in counts['Category']],
        text=counts['Count'], textposition='inside',
        textfont=dict(color='white', size=14),
    ))
    fig.update_layout(
        title='감정 분포', xaxis_title='감정 카테고리', yaxis_title='뉴스 개수',
        template="plotly_white", height=380, showlegend=False,
    )
    return fig


def create_sentiment_boxplot(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    tickers = df['Ticker'].unique()
    colors = px.colors.qualitative.Set3
    mean_df = df.groupby('Ticker')['Sentiment'].mean().reset_index()

    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['Ticker'] == ticker]['Sentiment']
        fig.add_trace(go.Box(
            y=ticker_data, name=ticker,
            marker_color=colors[i % len(colors)], boxmean=True,
        ))
    for i, row in enumerate(mean_df.itertuples()):
        color = 'red' if row.Sentiment >= 0 else 'blue'
        fig.add_annotation(
            x=i, y=row.Sentiment,
            text=f'{row.Sentiment:.2f}',
            showarrow=False,
            font=dict(color=color, size=11),
            bgcolor="rgba(255,255,255,0.8)",
        )
    fig.update_layout(
        title='종목별 감정 점수 분포 (Box Plot)',
        xaxis_title='종목', yaxis_title='감정 점수',
        template="plotly_white", height=480, showlegend=False,
    )
    return fig


def render_news_table(df: pd.DataFrame):
    if df.empty:
        st.info("관련 뉴스가 없습니다.")
        return

    display = df[['Date', 'Ticker', 'Title', 'Headline', 'Sentiment', 'Sentiment_Category', 'Category', 'URL']]\
        .copy().sort_values('Date', ascending=False).reset_index(drop=True)

    def color_sent(val):
        try:
            v = float(val)
            if v > 0.05:
                return 'color: #2ecc71; font-weight:bold'
            if v < -0.05:
                return 'color: #e74c3c; font-weight:bold'
            return 'color: #95a5a6'
        except Exception:
            return ''

    styled = (display.style
              .map(color_sent, subset=['Sentiment'])
              .format({'Sentiment': '{:.2f}'}))
    st.dataframe(
        styled,
        column_config={'URL': st.column_config.LinkColumn('URL')},
        use_container_width=True,
        height=min(600, 40 + len(display) * 35),
    )


# ======================================================
# Analyst & Valuation Functions
# ======================================================
@st.cache_data(ttl=3600)
def get_analyst_report_data(ticker_syms: list) -> pd.DataFrame:
    rows = []
    for sym in ticker_syms:
        try:
            ticker_obj = yf.Ticker(sym)
            info = ticker_obj.info or {}
            
            current_px = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('bid')
            target_px = info.get('targetMeanPrice')
            rec_mean = info.get('recommendationMean')
            rec_key = info.get('recommendationKey', 'none')
            
            upside = None
            if target_px and current_px and float(current_px) != 0:
                try:
                    upside = ((float(target_px) / float(current_px)) - 1) * 100
                except:
                    pass
            
            short_name = info.get('shortName') or info.get('longName') or info.get('symbol', '')
            
            rows.append({
                'Ticker': sym,
                '종목명': short_name,
                '등급 점수': rec_mean,
                '등급': rec_key.capitalize() if rec_key and rec_key != 'none' else 'N/A',
                '목표주가': target_px,
                '현재가': current_px,
                '상승여력(%)': upside,
            })
            time.sleep(0.1)
        except Exception as e:
            rows.append({
                'Ticker': sym,
                '종목명': 'N/A',
                '등급 점수': None,
                '등급': 'N/A',
                '목표주가': None,
                '현재가': None,
                '상승여력(%)': None,
            })
    
    df = pd.DataFrame(rows)
    return df[['Ticker', '종목명', '등급 점수', '등급', '목표주가', '현재가', '상승여력(%)']]


@st.cache_data(ttl=3600)
def get_valuation_eps_table(ticker_syms: list) -> pd.DataFrame:
    rows = []
    for sym in ticker_syms:
        try:
            ticker_obj = yf.Ticker(sym)
            info = ticker_obj.info or {}
            
            trailing_pe = info.get('trailingPE')
            forward_pe = info.get('forwardPE')
            t_eps = info.get('trailingEps') or info.get('trailingEPS')
            f_eps = info.get('forwardEps') or info.get('forwardEPS')
            
            eps_growth = None
            if t_eps and f_eps and float(t_eps) != 0:
                try:
                    eps_growth = ((float(f_eps) / float(t_eps)) - 1) * 100
                except:
                    pass
            
            short_name = info.get('shortName') or info.get('longName') or info.get('symbol', '')
            
            rows.append({
                'Ticker': sym,
                '종목명': short_name,
                'Trailing PE': trailing_pe,
                'Forward PE': forward_pe,
                'Trailing EPS': t_eps,
                'Forward EPS': f_eps,
                'EPS 상승률(%)': eps_growth,
            })
            time.sleep(0.1)
        except Exception as e:
            rows.append({
                'Ticker': sym,
                '종목명': 'N/A',
                'Trailing PE': None,
                'Forward PE': None,
                'Trailing EPS': None,
                'Forward EPS': None,
                'EPS 상승률(%)': None,
            })
    
    df = pd.DataFrame(rows)
    return df[['Ticker', '종목명', 'Trailing PE', 'Forward PE', 'Trailing EPS', 'Forward EPS', 'EPS 상승률(%)']]


# ======================================================
# Performance Table Functions
# ======================================================
def format_number(val, decimals=2):
    if pd.isna(val):
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except:
        return "N/A"


def get_perf_table_improved(label2ticker, ref_date=None):
    tickers = list(label2ticker.values())
    if ref_date is None:
        ref_date = datetime.now().date()
    start = ref_date - timedelta(days=4 * 365)
    end = ref_date + timedelta(days=1)

    try:
        raw = yf.download(tickers, start=start, end=end, progress=False)
        if isinstance(raw, pd.DataFrame):
            df = raw['Close']
        else:
            df = raw
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.ffill().dropna(how='all')[tickers]
    except:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    avail = df.index[df.index.date <= ref_date]
    if len(avail) == 0:
        return pd.DataFrame()

    last_trade = avail[-1].date()
    last_idx = avail[-1]

    periods = {
        '1D(%)': {'days': 1, 'type': 'business'},
        '1W(%)': {'days': 5, 'type': 'business'},
        'MTD(%)': {'type': 'month_start'},
        '1M(%)': {'days': 21, 'type': 'business'},
        '3M(%)': {'days': 63, 'type': 'business'},
        '6M(%)': {'days': 126, 'type': 'business'},
        'YTD(%)': {'type': 'year_start'},
        '1Y(%)': {'days': 252, 'type': 'business'},
        '3Y(%)': {'days': 756, 'type': 'business'},
    }

    results = []
    for label, ticker in label2ticker.items():
        row = {'자산명': label}
        series = df[ticker].dropna()
        if last_idx not in series.index or len(series) == 0:
            row['현재값'] = np.nan
            for pk in periods:
                row[pk] = np.nan
            results.append(row)
            continue

        curr = series.loc[last_idx]
        row['현재값'] = curr

        for pk, cfg in periods.items():
            base = None
            try:
                if cfg['type'] == 'month_start':
                    d = series[series.index.date >= last_trade.replace(day=1)]
                    base = d.iloc[0] if len(d) else None
                elif cfg['type'] == 'year_start':
                    d = series[series.index.date >= last_trade.replace(month=1, day=1)]
                    base = d.iloc[0] if len(d) else None
                else:
                    ci = series.index.get_loc(last_idx)
                    lb = cfg['days']
                    base = series.iloc[ci - lb] if ci >= lb else (series.iloc[0] if ci > 0 else None)

                row[pk] = (curr / base - 1) * 100 if (base is not None and not np.isnan(base) and base != 0) else np.nan
            except:
                row[pk] = np.nan

        results.append(row)

    df_r = pd.DataFrame(results)
    
    # 현재값만 문자열로 변환 (포맷팅)
    if '현재값' in df_r.columns:
        df_r['현재값'] = df_r['현재값'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
    
    # 성능 컬럼은 숫자 그대로 유지 (bar chart용)
    # 포맷팅은 style 함수에서 처리
    
    return df_r


def style_perf_table_with_databars(df, perf_cols):
    """성능 테이블에 데이터바 적용 (양수: 황금색, 음수: 갈색)"""
    # 인덱스를 다시 열로 변환 (안전성을 위해)
    df_work = df.reset_index()
    
    styled = df_work.style
    
    # 각 성능 컬럼에 대해 bar chart 적용
    for col in perf_cols:
        if col not in df_work.columns:
            continue
        
        try:
            # 숫자 값 추출 (NaN은 0으로 대체)
            numeric_vals = pd.to_numeric(df_work[col], errors='coerce')
            
            # NaN을 0으로 임시 대체
            df_work[col] = numeric_vals.fillna(0)
            
            # 범위 계산 (원본 데이터 기반, NaN 제외)
            valid_vals = numeric_vals.dropna()
            
            if len(valid_vals) == 0:
                continue
            
            vmin = valid_vals.min()
            vmax = valid_vals.max()
            
            # 양수/음수 대칭 범위
            range_val = max(abs(vmin), abs(vmax), 1)
            
            # bar 적용
            styled = styled.bar(
                subset=[col],
                color=['rgba(255,188,0,0.7)', 'rgba(96,88,76,0.9)'],
                vmin=-range_val,
                vmax=range_val,
                width=95,
                align='mid'
            )
        except Exception as e:
            continue
    
    # 포맷팅 (숫자를 소수점 2자리까지 표시)
    for col in perf_cols:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
    
    format_dict = {col: '{:.2f}' for col in perf_cols if col in df_work.columns}
    if format_dict:
        styled = styled.format(format_dict, na_rep='—')
    
    # 셀 스타일 (공통)
    styled = styled.set_properties(**{
        'border': '1px solid #d0d0d0',
        'text-align': 'center',
        'padding': '6px',
    })
    
    # 자산명 열 스타일 (인덱스로 돌아간 경우)
    if '자산명' in df_work.columns:
        styled = styled.set_properties(**{
            'text-align': 'left',
            'padding-left': '12px',
            'font-weight': 'bold',
        }, subset=['자산명'])
    
    # 현재값 열 스타일
    if '현재값' in df_work.columns:
        styled = styled.set_properties(**{
            'text-align': 'right',
            'padding-right': '12px',
            'font-weight': 'bold',
        }, subset=['현재값'])
    
    # 다시 인덱스로 설정
    styled = styled.hide(axis='index')
    
    return styled


# ======================================================
# Chart Functions for Page 1
# ======================================================
def plot_monthly_returns(prices_df, asset_name):
    monthly = prices_df.resample('M').last()
    returns = monthly.pct_change().dropna() * 100
    colors = ['#FFBC00' if x > 0 else '#60584c' for x in returns.iloc[:, 0]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns.index.strftime('%Y-%m'),
        y=returns.iloc[:, 0],
        marker_color=colors,
        text=[f"{v:.2f}%" for v in returns.iloc[:, 0]],
        textposition='outside',
    ))
    fig.update_layout(
        title=f'{asset_name}',
        xaxis_title='Month',
        yaxis_title='Return (%)',
        template='plotly_white',
        height=320,
        showlegend=False,
        margin=dict(t=30, b=20, l=30, r=20),
    )
    return fig


def get_distribution_stats(prices_df, asset_name):
    monthly = prices_df.resample('M').last()
    returns = monthly.pct_change().dropna() * 100
    returns_flat = returns.iloc[:, 0].values

    stats = {
        '자산': asset_name,
        '평균(%)': format_number(np.mean(returns_flat), 2),
        '표준편차(%)': format_number(np.std(returns_flat), 2),
        '최소(%)': format_number(np.min(returns_flat), 2),
        '25분위(%)': format_number(np.percentile(returns_flat, 25), 2),
        '중앙값(%)': format_number(np.median(returns_flat), 2),
        '75분위(%)': format_number(np.percentile(returns_flat, 75), 2),
        '최대(%)': format_number(np.max(returns_flat), 2),
    }
    return stats


def plot_rolling_volatility_visual(prices_df, asset_name, window=126):
    returns = prices_df.pct_change().dropna()
    rolling_vol = returns.iloc[:, 0].rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values,
        mode='lines',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)',
    ))
    fig.update_layout(
        title=f'{asset_name}',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        template='plotly_white',
        height=320,
        margin=dict(t=30, b=20, l=30, r=20),
    )
    return fig


def plot_rolling_sharpe(prices_df, asset_name, window=126, risk_free_rate=0.02):
    returns = prices_df.pct_change().dropna()
    rolling_mean = returns.iloc[:, 0].rolling(window).mean() * 252
    rolling_std = returns.iloc[:, 0].rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        line=dict(color='#f39c12', width=2),
        fill='tozeroy',
        fillcolor='rgba(243, 156, 18, 0.2)',
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig.update_layout(
        title=f'{asset_name}',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        template='plotly_white',
        height=320,
        margin=dict(t=30, b=20, l=30, r=20),
    )
    return fig


def plot_maximum_drawdown(prices_df, asset_name):
    """Maximum Drawdown 계산 및 표시"""
    cumulative_return = (1 + prices_df.iloc[:, 0].pct_change()).cumprod()
    running_max = cumulative_return.expanding().max()
    drawdown = (cumulative_return - running_max) / running_max * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        line=dict(color='#e74c3c', width=2),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)',
        name='Drawdown'
    ))
    
    fig.update_layout(
        title=f'{asset_name}',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=320,
        hovermode='x unified',
        margin=dict(t=30, b=20, l=30, r=20),
    )
    
    return fig


# ======================================================
# Page 1: Market Performance
# ======================================================
def show_page1():
    st.markdown(f'<h1 style="color: {TITLE_COLOR};">🌐 Market Performance</h1>', unsafe_allow_html=True)
    update_clicked = st.button("🔄 Update", type="primary", key="p1_update")

    if update_clicked:
        st.session_state['p1_updated'] = True

    if not st.session_state.get('p1_updated', False):
        st.info("🔄 Update 버튼을 눌러 데이터를 불러오세요.")
        return

    perf_cols = ['1D(%)', '1W(%)', 'MTD(%)', '1M(%)', '3M(%)', '6M(%)', 'YTD(%)', '1Y(%)', '3Y(%)']

    for title, label2t, h in [
        ("📊 Equity", STOCK_ETFS, 489),
        ("🗠 Bond", BOND_ETFS, 385),
        ("💱 Currency", CURRENCY, 283),
        ("📈 Crypto", CRYPTO, 385),
        ("📕 Style ETF", STYLE_ETFS, 249),
        ("📘 Sector ETF", SECTOR_ETFS, 425),
    ]:
        st.markdown(f'<h2 style="color: {TITLE_COLOR};">{title}</h2>', unsafe_allow_html=True)
        with st.spinner(f"{title} 계산 중..."):
            perf = get_perf_table_improved(label2t)
        if not perf.empty:
            # 자산명 인덱스로 설정
            perf_indexed = perf.set_index('자산명')
            
            # 스타일링 적용
            try:
                styled_df = style_perf_table_with_databars(perf_indexed, perf_cols)
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=h
                )
            except Exception as e:
                # 스타일링 실패 시 원본 데이터 표시
                st.warning(f"테이블 스타일링 중 오류 발생. 원본 데이터를 표시합니다.")
                st.dataframe(perf, use_container_width=True, height=h)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 국가", "📗 섹터", "📙 스타일"])

    with tab1:
        st.markdown(f'<h2 style="color: {TITLE_COLOR};">✅ 국가</h2>', unsafe_allow_html=True)
        render_comprehensive_chart(STOCK_ETFS, "stock_indices")

    with tab2:
        st.markdown(f'<h2 style="color: {TITLE_COLOR};">☑️ 섹터</h2>', unsafe_allow_html=True)
        render_comprehensive_chart(SECTOR_ETFS, "sector")

    with tab3:
        st.markdown(f'<h2 style="color: {TITLE_COLOR};">☑️ 스타일</h2>', unsafe_allow_html=True)
        render_comprehensive_chart(STYLE_ETFS, "style")


def render_comprehensive_chart(label2t, chart_key):
    if f"{chart_key}_period" not in st.session_state:
        st.session_state[f"{chart_key}_period"] = "3년"

    period_options = [p[0] for p in PERIOD_OPTIONS]
    current_idx = period_options.index(st.session_state[f"{chart_key}_period"])

    selected_period = st.selectbox(
        "기간 선택",
        options=period_options,
        index=current_idx,
        key=f"{chart_key}_select"
    )

    if selected_period != st.session_state[f"{chart_key}_period"]:
        st.session_state[f"{chart_key}_period"] = selected_period
        st.rerun()

    months = next(m for n, m in PERIOD_OPTIONS if n == selected_period)

    with st.spinner("데이터 분석 중..."):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 31)

        try:
            tickers = list(label2t.values())
            prices_raw = yf.download(tickers, start=start_date, end=end_date, progress=False)

            if isinstance(prices_raw, pd.DataFrame):
                prices_data = prices_raw['Close']
            else:
                prices_data = prices_raw.to_frame()

            if isinstance(prices_data, pd.Series):
                prices_data = prices_data.to_frame()

            rename_dict = {}
            for label, ticker in label2t.items():
                if ticker in prices_data.columns:
                    rename_dict[ticker] = label
            
            prices_data = prices_data.rename(columns=rename_dict)
            prices_data = prices_data.ffill().dropna()

        except Exception as e:
            st.error(f"데이터 다운로드 실패: {str(e)}")
            return

        st.markdown(f'<h2 style="color: {TITLE_COLOR};">📈 Cumulative Returns</h2>', unsafe_allow_html=True)
        norm = prices_data / prices_data.iloc[0] * 100
        fig = go.Figure()
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
        for i, col in enumerate(norm.columns):
            fig.add_trace(go.Scatter(
                x=norm.index,
                y=norm[col],
                mode='lines',
                name=col,
                line=dict(color=colors_list[i % len(colors_list)], width=2.5),
            ))
        fig.update_layout(
            yaxis_title="누적 성과",
            template="plotly_white",
            height=420,
            legend=dict(orientation='h', y=1.15, x=0),
            hovermode='x unified',
            margin=dict(b=80)
        )
        st.plotly_chart(fig, use_container_width=True)

        assets = list(prices_data.columns)

        # ===== 4개 탭: Monthly Returns, Rolling Volatility, Rolling Sharpe, Maximum Drawdown =====
        st.markdown("---")
        st.markdown(f'<h2 style="color: {TITLE_COLOR};">📊 분석 차트</h2>', unsafe_allow_html=True)
        
        tab_mr, tab_rv, tab_rs, tab_md = st.tabs(
            ["📊 Monthly Returns", "📈 Rolling Volatility", "⭐ Rolling Sharpe", "📉 Maximum Drawdown"]
        )

        # Tab 1: Monthly Returns (+Distribution of Monthly Returns)
        with tab_mr:
            st.caption("각 자산의 월별 수익률")
            for i in range(0, len(assets), 2):
                cols = st.columns(2)

                asset1 = assets[i]
                asset1_data = prices_data[[asset1]]
                try:
                    with cols[0]:
                        fig = plot_monthly_returns(asset1_data, asset1)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[0].error(f"{asset1} 실패")

                if i + 1 < len(assets):
                    asset2 = assets[i + 1]
                    asset2_data = prices_data[[asset2]]
                    try:
                        with cols[1]:
                            fig = plot_monthly_returns(asset2_data, asset2)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        cols[1].error(f"{asset2} 실패")

            # Distribution of Monthly Returns (Monthly Returns 탭에서만 표시)
            st.markdown("---")
            st.markdown(f'<h2 style="color: {TITLE_COLOR};">📉 Distribution of Monthly Returns</h2>', unsafe_allow_html=True)
            
            all_stats = []
            for asset in assets:
                asset_data = prices_data[[asset]]
                stats = get_distribution_stats(asset_data, asset)
                all_stats.append(stats)
            
            stats_df = pd.DataFrame(all_stats)
            styled = stats_df.style
            numeric_cols = [col for col in stats_df.columns if col != '자산']
            transparent_wistia = create_transparent_wistia_cmap(alpha=0.4)
            
            for col in numeric_cols:
                numeric_vals = pd.to_numeric(stats_df[col], errors='coerce')
                valid_vals = numeric_vals[numeric_vals.notna()]
                if len(valid_vals) > 0:
                    vmin = valid_vals.min()
                    vmax = valid_vals.max()
                    styled = styled.background_gradient(
                        subset=[col],
                        cmap=transparent_wistia,
                        vmin=vmin,
                        vmax=vmax,
                        low=0.3,
                        high=0.3
                    )
            
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # Tab 2: Rolling Volatility
        with tab_rv:
            st.caption("6개월 Rolling 변동성")
            for i in range(0, len(assets), 2):
                cols = st.columns(2)

                asset1 = assets[i]
                asset1_data = prices_data[[asset1]]
                try:
                    with cols[0]:
                        fig = plot_rolling_volatility_visual(asset1_data, asset1)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[0].error(f"{asset1} 실패")

                if i + 1 < len(assets):
                    asset2 = assets[i + 1]
                    asset2_data = prices_data[[asset2]]
                    try:
                        with cols[1]:
                            fig = plot_rolling_volatility_visual(asset2_data, asset2)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        cols[1].error(f"{asset2} 실패")

        # Tab 3: Rolling Sharpe
        with tab_rs:
            st.caption("6개월 Rolling Sharpe Ratio")
            for i in range(0, len(assets), 2):
                cols = st.columns(2)

                asset1 = assets[i]
                asset1_data = prices_data[[asset1]]
                try:
                    with cols[0]:
                        fig = plot_rolling_sharpe(asset1_data, asset1)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[0].error(f"{asset1} 실패")

                if i + 1 < len(assets):
                    asset2 = assets[i + 1]
                    asset2_data = prices_data[[asset2]]
                    try:
                        with cols[1]:
                            fig = plot_rolling_sharpe(asset2_data, asset2)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        cols[1].error(f"{asset2} 실패")

        # Tab 4: Maximum Drawdown
        with tab_md:
            st.caption("Maximum Drawdown")
            
            for i in range(0, len(assets), 2):
                cols = st.columns(2)

                asset1 = assets[i]
                asset1_data = prices_data[[asset1]]
                try:
                    with cols[0]:
                        fig = plot_maximum_drawdown(asset1_data, asset1)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[0].error(f"{asset1} 실패")

                if i + 1 < len(assets):
                    asset2 = assets[i + 1]
                    asset2_data = prices_data[[asset2]]
                    try:
                        with cols[1]:
                            fig = plot_maximum_drawdown(asset2_data, asset2)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        cols[1].error(f"{asset2} 실패")


# ======================================================
# Page 2: LLM Analysis
# ======================================================
def show_page2():
    st.markdown(f'<h1 style="color: {TITLE_COLOR};">🤖 LLM 분석 — 뉴스 감성 분석</h1>', unsafe_allow_html=True)
    st.caption("Yahoo Finance RSS에서 수집한 최근 3일 뉴스를 FinBERT로 감성 분석합니다.")

    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        selected = st.selectbox("섹터 선택", list(SECTOR_ETFS.keys()), key="p2_sector")
    with col2:
        run_btn = st.button("📡 분석 시작", type="primary", use_container_width=True, key="p2_run")

    etf_ticker = SECTOR_ETFS[selected]
    cache_key = f'llm_{etf_ticker}'

    if run_btn:
        progress = st.progress(0, text="Holdings 수집 중...")
        holdings = ETFCollector().get_etf_holdings(etf_ticker)
        if not holdings:
            st.error(f"❌ {selected}: Holdings 수집 실패 (4가지 방법 모두 실패)")
            return
        progress.progress(20, text=f"✅ {len(holdings)}개 종목 — 뉴스 수집 중...")

        all_news = NewsCollector(days=3).collect_all(holdings, etf_ticker)
        if not all_news:
            st.warning(f"⚠️ {selected}: 관련 뉴스를 찾지 못했습니다.")
            progress.empty()
            return
        progress.progress(60, text=f"✅ {len(all_news)}건 뉴스 — FinBERT 감성 분석 중...")

        analyzed = load_analyzer().batch_analyze(all_news)
        st.session_state[cache_key] = analyzed
        progress.progress(100, text="✅ 분석 완료!")
        time.sleep(0.5)
        progress.empty()

    if cache_key not in st.session_state:
        st.info("섹터를 선택하고 '📡 분석 시작' 버튼을 누르세요.")
        return

    news_list = st.session_state[cache_key]
    df = build_sentiment_df(news_list)
    if df.empty:
        st.warning("분석 결과가 없습니다.")
        return

    avg_s = df['Sentiment'].mean()
    pos = (df['Sentiment_Category'] == 'Positive').sum()
    neg = (df['Sentiment_Category'] == 'Negative').sum()
    neu = (df['Sentiment_Category'] == 'Neutral').sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("🟢 Positive", int(pos))
    c3.metric("🔴 Negative", int(neg))
    c4.metric("Average", f"{avg_s:+.3f}")

    st.markdown("---")

    render_sentiment_bar_chart(df, selected)

    col_h, col_c = st.columns(2)
    with col_h:
        st.plotly_chart(create_sentiment_histogram(df), use_container_width=True)
    with col_c:
        st.plotly_chart(create_sentiment_countplot(df), use_container_width=True)

    st.plotly_chart(create_sentiment_boxplot(df), use_container_width=True)

    st.markdown("---")

    st.markdown(f"##### 📋 관련 뉴스 전체 목록 ({len(df)}건)")
    render_news_table(df)


# ======================================================
# Page 3: Analyst & Valuation
# ======================================================
def show_page3():
    st.markdown(f'<h1 style="color: {TITLE_COLOR};">👨‍💼 애널리스트 & 밸류에이션</h1>', unsafe_allow_html=True)
    st.caption(
        "• 등급 점수: 1=Strong Buy  2=Buy  3=Neutral  4=Sell  5=Strong Sell\n"
        "• 목표주가: 최근 3~6개월 애널리스트 리포트 평균\n"
        "• Trailing PE/EPS: 최근 12M  |  Forward PE/EPS: 선행 12M"
    )

    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        selected = st.selectbox("섹터 선택", list(SECTOR_ETFS.keys()), key="p3_sector")
    with col2:
        run_btn = st.button("🔍 조회", type="primary", use_container_width=True, key="p3_run")

    etf_ticker = SECTOR_ETFS[selected]
    cache_key = f'analyst_{etf_ticker}'

    if run_btn:
        progress = st.progress(0, text="Holdings 수집 중...")
        holdings = ETFCollector().get_etf_holdings(etf_ticker)
        if not holdings:
            st.error(f"❌ {selected}: Holdings 수집 실패")
            return
        ticker_syms = [h['ticker'] for h in holdings]
        progress.progress(15, text=f"✅ {len(ticker_syms)}개 종목 — 애널리스트 데이터 수집 중...")

        analyst_df = get_analyst_report_data(ticker_syms)
        progress.progress(55, text="✅ 애널리스트 수집 완료 — 밸류에이션 수집 중...")
        valuation_df = get_valuation_eps_table(ticker_syms)
        progress.progress(100, text="✅ 조회 완료!")
        time.sleep(0.4)
        progress.empty()

        st.session_state[cache_key] = {
            'analyst': analyst_df,
            'valuation': valuation_df,
            'holdings': holdings,
        }

    if cache_key not in st.session_state:
        st.info("섹터를 선택하고 '🔍 조회' 버튼을 누르세요.")
        return

    data = st.session_state[cache_key]
    analyst_df = data['analyst']
    val_df = data['valuation']
    holdings = data['holdings']

    st.markdown(f'<h2 style="color: {TITLE_COLOR};">📦 {selected} — Top Holdings</h2>', unsafe_allow_html=True)
    holdings_df = pd.DataFrame(holdings)
    st.dataframe(holdings_df, use_container_width=True,
                 height=min(400, 40 + len(holdings_df) * 35))

    st.markdown("---")

    st.markdown(f'<h2 style="color: {TITLE_COLOR};">👨‍💼 애널리스트 등급 & 목표주가</h2>', unsafe_allow_html=True)
    analyst_sorted = analyst_df.sort_values('상승여력(%)', ascending=False, na_position='last')

    def color_upside(val):
        try:
            v = float(val)
            if v > 10:
                return 'color: #2ecc71; font-weight:bold'
            if v < 0:
                return 'color: #e74c3c; font-weight:bold'
            return ''
        except:
            return ''

    def color_rating(val):
        try:
            v = float(val)
            if v <= 2:
                return 'color: #2ecc71; font-weight:bold'
            if v >= 4:
                return 'color: #e74c3c; font-weight:bold'
            return 'color: #f39c12'
        except:
            return ''

    fmt = {'등급 점수': '{:.2f}', '목표주가': '{:,.2f}', '현재가': '{:,.2f}', '상승여력(%)': '{:.2f}%'}
    styled_a = (analyst_sorted.style
                .format(fmt, na_rep='N/A')
                .map(color_upside, subset=['상승여력(%)'])
                .map(color_rating, subset=['등급 점수']))
    
    upside_vals = pd.to_numeric(analyst_sorted['상승여력(%)'], errors='coerce')
    valid_upside = upside_vals[upside_vals.notna()]
    if len(valid_upside) > 0:
        transparent_wistia = create_transparent_wistia_cmap(alpha=0.4)
        styled_a = styled_a.background_gradient(
            subset=['상승여력(%)'], 
            cmap=transparent_wistia, 
            vmin=-20, 
            vmax=40, 
            low=0.3, 
            high=0.3
        )
    
    st.dataframe(styled_a, use_container_width=True,
                 height=min(500, 40 + len(analyst_sorted) * 35))

    if not analyst_sorted['상승여력(%)'].isna().all():
        fig_up = go.Figure()
        df_plot = analyst_sorted.dropna(subset=['상승여력(%)'])
        if len(df_plot) > 0:
            fig_up.add_trace(go.Bar(
                x=df_plot['Ticker'],
                y=df_plot['상승여력(%)'],
                marker_color=['#FFBC00' if v > 0 else '#e74c3c' for v in df_plot['상승여력(%)']],
                text=[f"{v:.1f}%" for v in df_plot['상승여력(%)']],
                textposition='outside',
            ))
            fig_up.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.4)
            fig_up.update_layout(
                title='종목별 애널리스트 목표주가 상승여력',
                xaxis_title='Ticker', yaxis_title='상승여력 (%)',
                template='plotly_white', height=380,
            )
            st.plotly_chart(fig_up, use_container_width=True)

    st.markdown("---")

    st.markdown(f'<h2 style="color: {TITLE_COLOR};">🔍 밸류에이션 & EPS</h2>', unsafe_allow_html=True)
    val_sorted = val_df.sort_values('EPS 상승률(%)', ascending=False, na_position='last')
    fmt_v = {'Trailing PE': '{:.1f}', 'Forward PE': '{:.1f}',
             'Trailing EPS': '{:.2f}', 'Forward EPS': '{:.2f}', 'EPS 상승률(%)': '{:.2f}%'}

    def color_eps(val):
        try:
            v = float(val)
            if v > 5:
                return 'color: #2ecc71; font-weight:bold'
            if v < 0:
                return 'color: #e74c3c; font-weight:bold'
            return ''
        except:
            return ''

    styled_v = (val_sorted.style
                .format(fmt_v, na_rep='N/A')
                .map(color_eps, subset=['EPS 상승률(%)']))
    
    eps_vals = pd.to_numeric(val_sorted['EPS 상승률(%)'], errors='coerce')
    valid_eps = eps_vals[eps_vals.notna()]
    if len(valid_eps) > 0:
        transparent_wistia = create_transparent_wistia_cmap(alpha=0.4)
        styled_v = styled_v.background_gradient(
            subset=['EPS 상승률(%)'], 
            cmap=transparent_wistia, 
            low=0.3, 
            high=0.3
        )
    
    st.dataframe(styled_v, use_container_width=True,
                 height=min(500, 40 + len(val_sorted) * 35))

    pe_df = val_sorted.dropna(subset=['Trailing PE', 'Forward PE'])
    if not pe_df.empty:
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Bar(
            x=pe_df['Ticker'], y=pe_df['Trailing PE'],
            name='Trailing PE', marker_color='rgba(96,88,76,0.9)',
        ))
        fig_pe.add_trace(go.Bar(
            x=pe_df['Ticker'], y=pe_df['Forward PE'],
            name='Forward PE', marker_color='rgba(255,188,0,0.7)',
        ))
        fig_pe.update_layout(
            title='Trailing PE vs Forward PE',
            xaxis_title='Ticker', yaxis_title='PE Ratio',
            barmode='group', template='plotly_white', height=380,
        )
        st.plotly_chart(fig_pe, use_container_width=True)


# ======================================================
# Main Navigation
# ======================================================
with st.sidebar:
    logo_url = "https://img.inhr.co.kr/static/careerlink/DSGN/250310110803368lsi.svg"
    st.markdown(
        f"""
        <div style="display: flex;">
            <img src="{logo_url}" width="200px" style="display: block;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<h1 style="color: {TITLE_COLOR};">💡 Global Market</h1>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "페이지 선택",
        ["📊 시장 성과", "🤖 LLM 분석", "👨‍💼 애널리스트"],
        key="nav_page",
    )
    st.markdown("---")
    st.markdown('<div style="font-size:0.85rem;">Data source: '
                '<a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a></div>',
                unsafe_allow_html=True)

    st.caption(f"Last visit: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.divider()
    st.caption("© 2026 KB Asset Management.")
    with st.expander("📄 MIT License Details"):
        st.markdown("""
        **MIT License**

        Copyright (c) 2026 **KB Asset Management**

        Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files...

        ---
        *The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.*

        `THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED...`
        """, help="Click to see the full license text.")


if page == "📊 시장 성과":
    show_page1()
elif page == "🤖 LLM 분석":
    show_page2()
elif page == "👨‍💼 애널리스트":
    show_page3()
