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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Global Market Monitoring", page_icon="🌐", layout="wide")

# ===== 자산 정의 =====
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

def _get_cmap(name: str):
    """matplotlib 버전과 무관하게 컬러맵을 가져온다.
    matplotlib 3.11+ 에서 cm.get_cmap 이 제거되어 AttributeError 가 나는 문제를 해결한다."""
    try:
        return matplotlib.colormaps[name]          # matplotlib >= 3.5 (권장)
    except Exception:
        pass
    try:
        return plt.get_cmap(name)                  # 폴백 1
    except Exception:
        from matplotlib import cm as _cm           # 폴백 2 (구버전)
        return _cm.get_cmap(name)


def create_transparent_YlOrBr_cmap(alpha=0.4):
    """YlOrBr 컬러맵에 투명도를 적용한 커스텀 컬러맵 생성"""
    YlOrBr_cmap = _get_cmap('YlOrBr')
    colors = [YlOrBr_cmap(i) for i in np.linspace(0, 1, YlOrBr_cmap.N)]
    colors_with_alpha = [(r, g, b, alpha) for r, g, b, a in colors]
    return mcolors.ListedColormap(colors_with_alpha)


# ======================================================
# yfinance 안정화 레이어
#  - Streamlit Cloud 공용 IP 에서 Yahoo 가 .info 호출을 봇 차단/레이트리밋 하여
#    값이 통째로 비는(N/A) 문제를 완화한다.
#  - (1) curl_cffi 세션으로 브라우저를 흉내내고,
#    (2) info / fast_info / analyst_price_targets 등 여러 엔드포인트로 보강하고,
#    (3) 결과를 캐시하여 재실행 시 재요청을 막는다.
# ======================================================
@st.cache_resource(show_spinner=False)
def _get_yf_session():
    """Yahoo 차단 완화를 위한 curl_cffi 세션(크롬 흉내). 실패하면 None."""
    try:
        from curl_cffi import requests as _curl
        s = _curl.Session(impersonate="chrome")
        try:
            s.verify = False
        except Exception:
            pass
        return s
    except Exception:
        return None


def _make_ticker(sym: str):
    """세션이 있으면 세션을 붙여 Ticker 를 생성(버전에 따라 미지원 시 자동 폴백)."""
    sess = _get_yf_session()
    if sess is not None:
        try:
            return yf.Ticker(sym, session=sess)
        except Exception:
            pass
    return yf.Ticker(sym)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(sym: str) -> dict:
    """단일 티커의 펀더멘털을 여러 Yahoo 엔드포인트에서 견고하게 수집한다.
    .info 가 레이트리밋으로 비어도 fast_info / analyst_price_targets 로 보강한다.
    (TTL 30분 캐시 → 애널리스트·밸류에이션 표가 동일 티커를 한 번만 호출)"""
    out = {
        'shortName': None, 'currentPrice': None, 'targetMeanPrice': None,
        'recommendationMean': None, 'recommendationKey': None,
        'trailingPE': None, 'forwardPE': None,
        'trailingEps': None, 'forwardEps': None,
    }
    t = _make_ticker(sym)

    # 1) info / get_info
    info = {}
    for getter in ('info', 'get_info'):
        try:
            attr = getattr(t, getter)
            data = attr() if callable(attr) else attr
            if isinstance(data, dict) and len(data) > 3:
                info = data
                break
        except Exception:
            continue

    if info:
        out['shortName'] = info.get('shortName') or info.get('longName')
        for f in ('currentPrice', 'regularMarketPrice', 'previousClose', 'open'):
            v = info.get(f)
            if isinstance(v, (int, float)) and v > 0:
                out['currentPrice'] = float(v)
                break
        for f in ('targetMeanPrice', 'targetMedianPrice', 'targetPrice'):
            v = info.get(f)
            if isinstance(v, (int, float)) and v > 0:
                out['targetMeanPrice'] = float(v)
                break
        rm = info.get('recommendationMean')
        if isinstance(rm, (int, float)):
            out['recommendationMean'] = float(rm)
        rk = info.get('recommendationKey')
        if rk and str(rk).lower() != 'none':
            out['recommendationKey'] = str(rk).capitalize()
        v = info.get('trailingPE')
        if isinstance(v, (int, float)) and 0 < v < 500:
            out['trailingPE'] = float(v)
        v = info.get('forwardPE')
        if isinstance(v, (int, float)) and 0 < v < 500:
            out['forwardPE'] = float(v)
        for f in ('trailingEps', 'epsTrailingTwelveMonths'):
            v = info.get(f)
            if isinstance(v, (int, float)):
                out['trailingEps'] = float(v)
                break
        for f in ('forwardEps', 'epsForward'):
            v = info.get(f)
            if isinstance(v, (int, float)):
                out['forwardEps'] = float(v)
                break

    # 2) fast_info 로 현재가 보강(가벼운 별도 엔드포인트)
    if out['currentPrice'] is None:
        try:
            fi = t.fast_info
            for f in ('last_price', 'lastPrice', 'previous_close', 'previousClose'):
                v = fi.get(f) if hasattr(fi, 'get') else getattr(fi, f, None)
                if isinstance(v, (int, float)) and v > 0:
                    out['currentPrice'] = float(v)
                    break
        except Exception:
            pass

    # 3) analyst_price_targets 로 목표가/현재가 보강
    if out['targetMeanPrice'] is None or out['currentPrice'] is None:
        try:
            apt = t.analyst_price_targets
            if isinstance(apt, dict):
                if out['targetMeanPrice'] is None:
                    for k in ('mean', 'median'):
                        v = apt.get(k)
                        if isinstance(v, (int, float)) and v > 0:
                            out['targetMeanPrice'] = float(v)
                            break
                if out['currentPrice'] is None:
                    v = apt.get('current')
                    if isinstance(v, (int, float)) and v > 0:
                        out['currentPrice'] = float(v)
        except Exception:
            pass

    # 4) 최후의 가격 폴백: 최근 종가
    if out['currentPrice'] is None:
        try:
            h = t.history(period='5d')
            if h is not None and not h.empty:
                out['currentPrice'] = float(h['Close'].dropna().iloc[-1])
        except Exception:
            pass

    if not out['shortName']:
        out['shortName'] = sym
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _download_close_cached(tickers_tuple: tuple, start_str: str, end_str: str) -> pd.DataFrame:
    """종가 다운로드(캐시). 날짜를 'YYYY-MM-DD' 문자열로 받아 장중 재실행에도 캐시가 적중한다."""
    tickers = list(tickers_tuple)
    sess = _get_yf_session()
    try:
        if sess is not None:
            raw = yf.download(tickers, start=start_str, end=end_str, progress=False, session=sess)
        else:
            raw = yf.download(tickers, start=start_str, end=end_str, progress=False)
    except TypeError:
        raw = yf.download(tickers, start=start_str, end=end_str, progress=False)
    except Exception:
        return pd.DataFrame()

    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw, pd.DataFrame):
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0)
            df = raw['Close'] if 'Close' in lvl0 else raw
        elif 'Close' in raw.columns:
            df = raw[['Close']].copy()
            df.columns = [tickers[0]]
        else:
            df = raw
    else:
        df = raw.to_frame()

    if isinstance(df, pd.Series):
        df = df.to_frame()

    try:
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df


def download_close_prices(tickers, start, end) -> pd.DataFrame:
    """캐시된 종가 다운로드 래퍼. start/end 는 date·datetime·문자열 모두 허용."""
    def _s(x):
        try:
            return x.strftime('%Y-%m-%d')
        except Exception:
            return str(x)[:10]
    return _download_close_cached(tuple(tickers), _s(start), _s(end))

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
        if any(w in t for w in ['earnings', 'revenue', 'profit', 'guidance']):
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

def get_analyst_report_data(ticker_syms: list) -> pd.DataFrame:
    """애널리스트 등급·목표주가 표.
    fetch_fundamentals(캐시 + curl_cffi 세션 + 다중 엔드포인트)를 사용해
    Streamlit Cloud 에서 .info 가 차단되어도 값이 채워지도록 한다."""
    rows = []
    for sym in ticker_syms:
        f = fetch_fundamentals(sym)
        cur = f.get('currentPrice')
        tgt = f.get('targetMeanPrice')
        upside = None
        if cur and tgt and cur > 0:
            try:
                upside = ((float(tgt) / float(cur)) - 1) * 100
            except Exception:
                upside = None
        rows.append({
            'Ticker': sym,
            '종목명': f.get('shortName') or sym,
            '등급 점수': f.get('recommendationMean'),
            '등급': f.get('recommendationKey') or 'N/A',
            '목표주가': tgt,
            '현재가': cur,
            '상승여력(%)': upside,
        })

    df = pd.DataFrame(rows)
    return df[['Ticker', '종목명', '등급 점수', '등급', '목표주가', '현재가', '상승여력(%)']]


def get_valuation_eps_table(ticker_syms: list) -> pd.DataFrame:
    """밸류에이션·EPS 표. fetch_fundamentals 캐시를 공유하므로
    애널리스트 표에서 이미 조회한 티커는 추가 네트워크 호출 없이 채워진다."""
    rows = []
    for sym in ticker_syms:
        f = fetch_fundamentals(sym)
        t_eps = f.get('trailingEps')
        f_eps = f.get('forwardEps')
        eps_growth = None
        if t_eps and f_eps and float(t_eps) != 0:
            try:
                eps_growth = ((float(f_eps) / float(t_eps)) - 1) * 100
            except Exception:
                eps_growth = None
        rows.append({
            'Ticker': sym,
            '종목명': f.get('shortName') or sym,
            'Trailing PE': f.get('trailingPE'),
            'Forward PE': f.get('forwardPE'),
            'Trailing EPS': t_eps,
            'Forward EPS': f_eps,
            'EPS 상승률(%)': eps_growth,
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
        df = download_close_prices(tickers, start, end)
        if df.empty:
            st.error("데이터 다운로드 실패: 가격 데이터를 가져오지 못했습니다.")
            return pd.DataFrame()
        keep = [t for t in tickers if t in df.columns]
        df = df.ffill().dropna(how='all')[keep]
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {str(e)}")
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
        if ticker not in df.columns:
            row['현재값'] = np.nan
            for pk in periods:
                row[pk] = np.nan
            results.append(row)
            continue
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
    
    if '현재값' in df_r.columns:
        df_r['현재값'] = df_r['현재값'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
    
    perf_cols = ['1D(%)', '1W(%)', 'MTD(%)', '1M(%)', '3M(%)', '6M(%)', 'YTD(%)', '1Y(%)', '3Y(%)']
    for col in perf_cols:
        if col in df_r.columns:
            df_r[col] = df_r[col].apply(lambda x: format_number(x, 2) if pd.notnull(x) else "N/A")

    return df_r


def style_perf_table_with_databars(df, perf_cols):
    """YlOrBr 색상 히트맵 적용 (투명도 조정)"""
    styled = df.copy().style
    transparent_YlOrBr = create_transparent_YlOrBr_cmap(alpha=0.4)

    for col in perf_cols:
        if col in df.columns:
            numeric_vals = pd.to_numeric(
                df[col].astype(str).str.replace('%', '').str.strip(),
                errors='coerce'
            )
            valid_vals = numeric_vals[numeric_vals.notna()]
            
            if len(valid_vals) > 0:
                vmin = valid_vals.min()
                vmax = valid_vals.max()
                
                styled = styled.background_gradient(
                    subset=[col],
                    cmap=transparent_YlOrBr,
                    vmin=vmin,
                    vmax=vmax,
                    low=0.3,
                    high=0.3
                )

    return styled


# ======================================================
# Chart Functions for Page 1
# ======================================================
def plot_monthly_returns(prices_df, asset_name):
    """월별 수익률 차트"""
    try:
        # 인덱스 timezone 제거
        if hasattr(prices_df.index, 'tz') and prices_df.index.tz is not None:
            prices_df = prices_df.copy()
            prices_df.index = prices_df.index.tz_localize(None)
        
        monthly = prices_df.resample('ME').last()
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
    except Exception as e:
        st.error(f"월별 수익률 차트 오류: {str(e)}")
        return go.Figure()


def get_distribution_stats(prices_df, asset_name):

    try:
        
        if hasattr(prices_df.index, 'tz') and prices_df.index.tz is not None:
            prices_df = prices_df.copy()
            prices_df.index = prices_df.index.tz_localize(None)
        
        monthly = prices_df.resample('ME').last()
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
    except Exception as e:
        return {}


def plot_rolling_volatility_visual(prices_df, asset_name, window=126):
   
    try:
        if hasattr(prices_df.index, 'tz') and prices_df.index.tz is not None:
            prices_df = prices_df.copy()
            prices_df.index = prices_df.index.tz_localize(None)
        
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
    except Exception as e:
        st.error(f"변동성 차트 오류")
        return go.Figure()


def plot_rolling_sharpe(prices_df, asset_name, window=126, risk_free_rate=0.02):
    
    try:
        if hasattr(prices_df.index, 'tz') and prices_df.index.tz is not None:
            prices_df = prices_df.copy()
            prices_df.index = prices_df.index.tz_localize(None)
        
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
    except Exception as e:
        st.error(f"샤프 비율 차트 오류")
        return go.Figure()


def plot_maximum_drawdown(prices_df, asset_name):
    """최대 낙폭 차트"""
    try:
        if hasattr(prices_df.index, 'tz') and prices_df.index.tz is not None:
            prices_df = prices_df.copy()
            prices_df.index = prices_df.index.tz_localize(None)
        
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
    except Exception as e:
        st.error(f"낙폭 차트 오류")
        return go.Figure()


def plot_correlation_heatmap(prices_data, period_label=""):
    """자산 간 일간 수익률 상관관계 히트맵 (리스크 분산/군집 파악용)."""
    try:
        rets = prices_data.pct_change().dropna()
        if rets.shape[1] < 2 or rets.shape[0] < 5:
            return None
        corr = rets.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            colorbar=dict(title='ρ'),
            hovertemplate='%{y} ↔ %{x}<br>상관계수: %{z:.2f}<extra></extra>',
        ))
        fig.update_layout(
            title=f'일간 수익률 상관관계 {("· " + period_label) if period_label else ""}',
            template='plotly_white',
            height=max(360, 60 + len(corr) * 34),
            margin=dict(t=50, b=40, l=40, r=20),
            xaxis=dict(tickangle=-40),
        )
        return fig
    except Exception:
        return None


def compute_relative_strength(prices_data):
    """기간 내 상대강도(정규화 누적성과) 랭킹 테이블 — 누가 강했나 한눈에."""
    try:
        norm = prices_data / prices_data.iloc[0] * 100
        last = norm.iloc[-1]
        rs = (last - 100).sort_values(ascending=False)
        out = pd.DataFrame({'자산': rs.index, '기간성과(%)': rs.values.round(2)})
        out.insert(0, '순위', range(1, len(out) + 1))
        return out
    except Exception:
        return pd.DataFrame()


# ======================================================
# Page 1: Market Performance
# ======================================================
def show_page1():
    st.markdown(f'<h1 style="color: {TITLE_COLOR};">🌐 Market Performance</h1>', unsafe_allow_html=True)
    update_clicked = st.button("🔄 Update", type="primary", key="p1_update")

    if update_clicked:
        st.session_state['p1_updated'] = True
        st.rerun()

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
            st.dataframe(
                style_perf_table_with_databars(perf.set_index('자산명'), perf_cols),
                use_container_width=True,
                height=h
            )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 국가 성과", "📗 섹터 성과", "📙 스타일 성과"])

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
    
    
    mode_tab1, mode_tab2 = st.tabs(["📅 기본 기간 설정", "🔧 사용자 정의 기간 설정"])
    
    # ===== 탭 1: 기본 기간 선택 =====
    with mode_tab1:
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

        months = next(m for n, m in PERIOD_OPTIONS if n == selected_period)

        with st.spinner("데이터 분석 중..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 31)
            
            display_chart_analysis(label2t, start_date, end_date, f"기본 기간 - {selected_period}")
    
    # ===== 탭 2: 사용자 정의 기간 선택 =====
    with mode_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            start_date_custom = st.date_input(
                "시작일",
                value=datetime.now() - timedelta(days=365),
                key=f"{chart_key}_start_date"
            )
        
        with col2:
            end_date_custom = st.date_input(
                "종료일",
                value=datetime.now(),
                key=f"{chart_key}_end_date"
            )
        
        
        if start_date_custom >= end_date_custom:
            st.error("❌ 시작일이 종료일보다 이전이어야 합니다.")
            return
        
        days_diff = (end_date_custom - start_date_custom).days
        if days_diff < 5:
            st.error("❌ 최소 5일 이상의 기간을 선택해주세요.")
            return
        
        
        st.info(f"📊 분석 기간: {start_date_custom} ~ {end_date_custom} ({days_diff}일)")
        
        if st.button("📈 분석 시작", key=f"{chart_key}_analyze_btn", type="primary"):
            with st.spinner("데이터 분석 중..."):
                display_chart_analysis(label2t, start_date_custom, end_date_custom, 
                                     f"사용자 정의 기간 - {start_date_custom} ~ {end_date_custom}")


def display_chart_analysis(label2t, start_date, end_date, period_label):
    
    
    try:
        tickers = list(label2t.values())
        prices_data = download_close_prices(tickers, start_date, end_date)
        if prices_data.empty:
            st.error("❌ 데이터 다운로드 실패: 가격 데이터를 가져오지 못했습니다.")
            return

        rename_dict = {}
        for label, ticker in label2t.items():
            if ticker in prices_data.columns:
                rename_dict[ticker] = label

        prices_data = prices_data.rename(columns=rename_dict)
        prices_data = prices_data.ffill().dropna()

        if prices_data.empty or len(prices_data) < 5:
            st.error("❌ 해당 기간에 충분한 데이터가 없습니다.")
            return

    except Exception as e:
        st.error(f"❌ 데이터 다운로드 실패: {str(e)}")
        return

    # ===== Cumulative Returns =====
    st.markdown(f'<h3 style="color: {TITLE_COLOR};">📈 누적 수익률 ({period_label})</h3>', unsafe_allow_html=True)
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
    st.markdown(f'<h3 style="color: {TITLE_COLOR};">📊 분석 차트</h3>', unsafe_allow_html=True)
    
    tab_mr, tab_rv, tab_rs, tab_md, tab_corr = st.tabs(
        ["📊 Monthly Returns", "📈 Rolling Volatility", "⭐ Rolling Sharpe", "📉 Maximum Drawdown", "🔗 상관관계·상대강도"]
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
        st.markdown(f'<h3 style="color: {TITLE_COLOR};">📉 월별 수익률 통계</h3>', unsafe_allow_html=True)
        
        all_stats = []
        for asset in assets:
            asset_data = prices_data[[asset]]
            stats = get_distribution_stats(asset_data, asset)
            if stats:
                all_stats.append(stats)
        
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            styled = stats_df.style
            numeric_cols = [col for col in stats_df.columns if col != '자산']
            transparent_YlOrBr = create_transparent_YlOrBr_cmap(alpha=0.4)
            
            for col in numeric_cols:
                numeric_vals = pd.to_numeric(stats_df[col], errors='coerce')
                valid_vals = numeric_vals[numeric_vals.notna()]
                if len(valid_vals) > 0:
                    vmin = valid_vals.min()
                    vmax = valid_vals.max()
                    styled = styled.background_gradient(
                        subset=[col],
                        cmap=transparent_YlOrBr,
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

    # Tab 5: 상관관계 + 상대강도
    with tab_corr:
        st.caption("기간 내 자산 간 일간 수익률 상관관계 및 상대강도 랭킹")
        col_l, col_r = st.columns([3, 2])
        with col_l:
            corr_fig = plot_correlation_heatmap(prices_data, period_label)
            if corr_fig is not None:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("상관관계를 계산할 데이터가 부족합니다.")
        with col_r:
            st.markdown(f'<h4 style="color: {TITLE_COLOR};">🏁 상대강도 랭킹</h4>', unsafe_allow_html=True)
            rs_df = compute_relative_strength(prices_data)
            if not rs_df.empty:
                styled_rs = rs_df.style.format({'기간성과(%)': '{:+.2f}'})
                rs_num = pd.to_numeric(rs_df['기간성과(%)'], errors='coerce')
                if rs_num.notna().any():
                    styled_rs = styled_rs.background_gradient(
                        subset=['기간성과(%)'],
                        cmap=create_transparent_YlOrBr_cmap(alpha=0.4),
                        vmin=float(rs_num.min()), vmax=float(rs_num.max()),
                        low=0.3, high=0.3,
                    )
                st.dataframe(styled_rs, use_container_width=True, hide_index=True,
                             height=min(520, 40 + len(rs_df) * 35))
            else:
                st.info("상대강도를 계산할 데이터가 부족합니다.")


# ======================================================
# Page 2: LLM Analysis
# ======================================================
def show_page2():
    st.markdown(f'<h1 style="color: {TITLE_COLOR};">🤖 LLM 분석 — 뉴스 감성 분석</h1>', unsafe_allow_html=True)
    st.caption("Yahoo Finance RSS에서 수집한 최근 3일 뉴스를 FinBERT(Financial Bidirectional Encoder Representations from Transformers)로 감성 분석합니다.")

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
    )
    st.caption(
        "• 목표주가: 최근 3~6개월 애널리스트 리포트 평균\n"
        "• Trailing PE, EPS: 최근 12M  |  Forward PE, EPS: 선행 12M(Blended Forward)"
    )
    st.caption(
        "• Data Source: Yahoo Finance 내부 Reuters 및 TipRanks 데이터"
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
        transparent_YlOrBr = create_transparent_YlOrBr_cmap(alpha=0.4)
        styled_a = styled_a.background_gradient(
            subset=['상승여력(%)'], 
            cmap=transparent_YlOrBr, 
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
        transparent_YlOrBr = create_transparent_YlOrBr_cmap(alpha=0.4)
        styled_v = styled_v.background_gradient(
            subset=['EPS 상승률(%)'], 
            cmap=transparent_YlOrBr, 
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
# 시장 국면(Regime) 데이터: 변동성·금리·시장 폭·모멘텀
#  - 모두 Yahoo Finance 무료 데이터(^VIX, ^TNX, ^IRX)와
#    이미 수집한 자산 시계열에서 파생한다(추가 비용/키 불필요).
# ======================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_regime_indicators(asof_str: str) -> dict:
    """무료 데이터에서 시장 국면 지표를 수집: VIX, 미 국채 10Y/3M, 장단기 스프레드."""
    today = datetime.now().date()
    start = today - timedelta(days=90)
    end = today + timedelta(days=1)
    out = {'vix': None, 'vix_chg': None, 'us10y': None, 'us10y_chg': None,
           'us3m': None, 'slope_10y_3m': None}
    df = download_close_prices(['^VIX', '^TNX', '^IRX'], start, end)

    def _last_prev(col):
        if df is None or df.empty or col not in df.columns:
            return None, None
        s = df[col].dropna()
        if s.empty:
            return None, None
        last = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) >= 2 else None
        return last, prev

    def _norm_yield(v):
        # 야후 ^TNX/^IRX 는 보통 퍼센트(예: 4.23). 과거 ×10 표기(예: 42.3) 방어.
        if v is None:
            return None
        return v / 10.0 if v > 30 else v

    vix, vix_p = _last_prev('^VIX')
    if vix is not None:
        out['vix'] = vix
        out['vix_chg'] = (vix - vix_p) if vix_p is not None else None

    t10, t10_p = _last_prev('^TNX')
    t10, t10_p = _norm_yield(t10), _norm_yield(t10_p)
    if t10 is not None:
        out['us10y'] = t10
        out['us10y_chg'] = (t10 - t10_p) if t10_p is not None else None

    t3, _ = _last_prev('^IRX')
    t3 = _norm_yield(t3)
    out['us3m'] = t3
    if t10 is not None and t3 is not None:
        out['slope_10y_3m'] = t10 - t3
    return out


def compute_breadth(prices_data: pd.DataFrame) -> dict:
    """포트폴리오 유니버스의 50일·200일 이동평균 상회 비율(시장 폭/Breadth)."""
    out = {'above_50': None, 'above_200': None, 'count_50': 0, 'count_200': 0,
           'n50': 0, 'n200': 0}
    try:
        c50 = c200 = n50 = n200 = 0
        for c in prices_data.columns:
            s = prices_data[c].dropna()
            if len(s) >= 50:
                ma50 = s.rolling(50).mean().iloc[-1]
                if pd.notna(ma50):
                    n50 += 1
                    if s.iloc[-1] > ma50:
                        c50 += 1
            if len(s) >= 200:
                ma200 = s.rolling(200).mean().iloc[-1]
                if pd.notna(ma200):
                    n200 += 1
                    if s.iloc[-1] > ma200:
                        c200 += 1
        out['count_50'], out['count_200'] = c50, c200
        out['n50'], out['n200'] = n50, n200
        out['above_50'] = (c50 / n50 * 100) if n50 else None
        out['above_200'] = (c200 / n200 * 100) if n200 else None
    except Exception:
        pass
    return out


def _rsi(series, period: int = 14):
    """Wilder 근사 RSI(이동평균 방식)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_momentum_table(label2ticker: dict, prices_data: pd.DataFrame) -> pd.DataFrame:
    """자산별 추세·모멘텀 점검 표: 현재가, 50/200MA 대비, RSI(14), 1M, 52주 고점대비, 추세."""
    t2label = {v: k for k, v in label2ticker.items()}
    rows = []
    for col in prices_data.columns:
        s = prices_data[col].dropna()
        if len(s) < 30:
            continue
        last = float(s.iloc[-1])
        ma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else None
        ma200 = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else None
        rsi = float(_rsi(s).iloc[-1]) if len(s) >= 15 else None
        m1 = (last / float(s.iloc[-22]) - 1) * 100 if len(s) >= 22 else None
        win = s.iloc[-252:] if len(s) >= 252 else s
        hi = float(win.max()) if len(win) else None
        from_high = (last / hi - 1) * 100 if hi else None
        above50 = (last / ma50 - 1) * 100 if ma50 else None
        above200 = (last / ma200 - 1) * 100 if ma200 else None
        if ma50 and ma200:
            trend = '강세' if (last > ma50 and last > ma200) else ('약세' if (last < ma50 and last < ma200) else '혼조')
        elif ma50:
            trend = '강세' if last > ma50 else '약세'
        else:
            trend = '—'
        rows.append({
            '자산': t2label.get(col, col),
            '현재가': round(last, 2),
            'vs50MA(%)': round(above50, 2) if above50 is not None else None,
            'vs200MA(%)': round(above200, 2) if above200 is not None else None,
            'RSI(14)': round(rsi, 1) if rsi is not None else None,
            '1M(%)': round(m1, 2) if m1 is not None else None,
            '52주고점대비(%)': round(from_high, 2) if from_high is not None else None,
            '추세': trend,
        })
    return pd.DataFrame(rows)


def _regime_to_text(reg: dict) -> str:
    """국면 지표를 LLM/요약용 텍스트로."""
    def f2(v, suf=''):
        return f"{v:.2f}{suf}" if isinstance(v, (int, float)) else "N/A"
    vix_chg = reg.get('vix_chg')
    lines = ["[시장 국면 지표]"]
    lines.append(f"- VIX(변동성): {f2(reg.get('vix'))} "
                 f"(1D {('%+.2f' % vix_chg) if isinstance(vix_chg, (int, float)) else 'N/A'})")
    lines.append(f"- 미 국채 10Y: {f2(reg.get('us10y'), '%')} / 3M: {f2(reg.get('us3m'), '%')}")
    sl = reg.get('slope_10y_3m')
    lines.append(f"- 장단기 스프레드(10Y-3M): {f2(sl, '%p')}"
                 f"{' (역전)' if isinstance(sl, (int, float)) and sl < 0 else ''}")
    return "\n".join(lines)


def _regime_comment(reg: dict, prices) -> str:
    """규칙 기반 국면 코멘트(키 불필요, 데이터로 계산)."""
    parts = []
    vix = reg.get('vix')
    if vix is not None:
        if vix < 15:
            parts.append(f"VIX {vix:.1f}로 변동성이 낮아 위험선호(리스크온)에 우호적인 환경으로 보입니다.")
        elif vix < 20:
            parts.append(f"VIX {vix:.1f}로 변동성은 보통 수준입니다.")
        elif vix < 30:
            parts.append(f"VIX {vix:.1f}로 변동성이 높아져 시장 경계감이 커진 것으로 보입니다.")
        else:
            parts.append(f"VIX {vix:.1f}로 변동성이 매우 높아 스트레스 국면 가능성이 있습니다.")
    slope = reg.get('slope_10y_3m')
    if slope is not None:
        if slope < 0:
            parts.append(f"미 국채 장단기 금리차(10Y-3M)가 {slope:+.2f}%p로 역전되어 있어 경기 둔화 신호를 모니터링할 필요가 있습니다.")
        else:
            parts.append(f"장단기 금리차(10Y-3M)는 {slope:+.2f}%p로 우상향(정상) 구간입니다.")
    if prices is not None and not getattr(prices, 'empty', True):
        a200 = compute_breadth(prices).get('above_200')
        if a200 is not None:
            if a200 >= 60:
                parts.append(f"추적 ETF의 {a200:.0f}%가 200일선 위에 있어 추세가 비교적 광범위하게 양호합니다.")
            elif a200 >= 40:
                parts.append(f"200일선 상회 비율이 {a200:.0f}%로 추세가 혼재되어 있습니다.")
            else:
                parts.append(f"200일선 상회 비율이 {a200:.0f}%에 그쳐 시장 전반의 추세가 약한 편입니다.")
    if not parts:
        return "_지표 데이터가 부족하여 국면 요약을 생성할 수 없습니다._"
    return " ".join(parts)


# ======================================================
# Page 4: AI 마켓 브리핑 (자동 생성)
#  - 크로스에셋 성과 스냅샷 + 시장 국면 지표(+선택 섹터 뉴스 감성)를 모아
#    한국어 데일리 브리핑을 자동 생성한다.
#  - 운영자가 Streamlit Secrets 에 키를 넣어두면 Claude 가 더 풍부하게 작성하고,
#    키가 없으면 동일 데이터로 규칙 기반 자동 분석을 제공한다(모든 방문자 공통).
# ======================================================
BRIEFING_GROUPS = {
    '주식': {
        'S&P 500': 'SPY', '나스닥100': 'QQQ', '선진국(VEA)': 'VEA', '신흥국(VWO)': 'VWO',
        '한국(EWY)': 'EWY', '중국(MCHI)': 'MCHI', '일본(EWJ)': 'EWJ', '유럽(VGK)': 'VGK',
    },
    '채권': {
        '미국 장기국채(TLT)': 'TLT', '미국 종합채권(BND)': 'BND', 'IG 회사채(LQD)': 'LQD',
        '하이일드(HYG)': 'HYG', '물가연동(TIP)': 'TIP',
    },
    '환율': {
        '달러-원': 'KRW=X', '달러-엔': 'JPY=X', '유로-달러': 'EURUSD=X', '달러-위안': 'CNY=X',
    },
    '원자재·대체': {
        '금(GLD)': 'GLD', 'WTI 원유(USO)': 'USO', '비트코인': 'BTC-USD', '이더리움': 'ETH-USD',
    },
    '미국 섹터': {
        'IT(XLK)': 'XLK', '헬스케어(XLV)': 'XLV', '금융(XLF)': 'XLF', '에너지(XLE)': 'XLE',
        '자유소비재(XLY)': 'XLY', '필수소비재(XLP)': 'XLP', '유틸리티(XLU)': 'XLU',
    },
}

BRIEFING_SYSTEM_PROMPT = """당신은 자산운용사의 글로벌 매크로 전략가입니다. 제공된 '시장 스냅샷'(가격 변화율), '시장 국면 지표'(VIX·미 국채 금리·장단기 스프레드), 그리고 선택적으로 제공되는 '섹터 뉴스 감성' 데이터만을 근거로, 해외 ETF 포트폴리오를 운용하는 펀드매니저를 위한 한국어 데일리 마켓 브리핑을 작성합니다.

작성 규칙:
- 반드시 제공된 수치에만 근거할 것. 데이터에 없는 구체적 뉴스·이벤트·수치는 절대 지어내지 말 것.
- 환율은 기준통화 대비 변화율이며, '달러-원/엔/위안'이 +이면 해당 통화 약세(달러 강세)를 의미함을 반영할 것.
- VIX는 변동성(공포)지수로 높을수록 위험회피 성향, 장단기 금리차(10Y-3M)가 음수이면 금리 역전(경기 둔화 신호 가능성)임을 반영할 것.
- 단정 대신 확률적·신중한 표현을 사용할 것("~로 보인다", "~가능성").
- 아래 형식을 따를 것:

## 📌 한 줄 요약
(오늘 시장을 한 문장으로)

## 🌍 자산군별 코멘트
- 주식 / 채권 / 환율 / 원자재·대체 / 섹터 순으로 각 1~2줄, 가장 두드러진 움직임 위주.

## 🌡️ 국면·리스크 지표
- VIX 수준과 변화, 미 국채 10Y 금리, 장단기 스프레드를 종합해 현재 시장 국면을 진단.

## 🔀 크로스에셋 신호
- 주식 vs 채권, 달러 방향, 안전자산(금·국채) 흐름을 종합해 리스크온/오프를 판단.

## 👀 주목 포인트
- 데이터에서 드러난 이례적/극단적 움직임(최대 상승·하락, 섹터 쏠림, 금리·변동성 급변 등) 2~3개.

## ⚠️ 리스크 체크
- 포트폴리오 관점에서 모니터링이 필요한 잠재 리스크 2~3개(데이터 기반 추론).

전체 분량은 한국어 600~900자 내외로 간결하게 작성하세요."""


def _pct_change_over(series, n):
    s = series.dropna()
    if len(s) < 2:
        return None
    if len(s) <= n:
        return (s.iloc[-1] / s.iloc[0] - 1) * 100
    return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100


def _ytd_change(series):
    s = series.dropna()
    if s.empty:
        return None
    try:
        yr = s.index[-1].year
        s_y = s[s.index >= pd.Timestamp(year=yr, month=1, day=1)]
        if len(s_y) >= 2:
            return (s_y.iloc[-1] / s_y.iloc[0] - 1) * 100
    except Exception:
        pass
    return None


@st.cache_data(ttl=900, show_spinner=False)
def collect_briefing_snapshot(asof_str: str) -> dict:
    """모든 브리핑 그룹의 1D/1W/1M/YTD 성과를 숫자로 수집(캐시). asof_str 로 일 단위 캐시."""
    today = datetime.now().date()
    start = today - timedelta(days=400)
    end = today + timedelta(days=1)
    out = {}
    for gname, gmap in BRIEFING_GROUPS.items():
        tickers = list(gmap.values())
        df = download_close_prices(tickers, start, end)
        recs = []
        for label, tk in gmap.items():
            if df is None or df.empty or tk not in df.columns:
                recs.append({'자산': label, '1D': None, '1W': None, '1M': None, 'YTD': None})
                continue
            s = df[tk].dropna()
            recs.append({
                '자산': label,
                '1D': _pct_change_over(s, 1),
                '1W': _pct_change_over(s, 5),
                '1M': _pct_change_over(s, 21),
                'YTD': _ytd_change(s),
            })
        out[gname] = recs
    return out


def _snapshot_to_text(snapshot: dict) -> str:
    asof = datetime.now().strftime('%Y-%m-%d %H:%M')
    def fmt(v):
        return f"{v:+.2f}" if isinstance(v, (int, float)) else "N/A"
    lines = [
        f"[글로벌 시장 스냅샷] 기준: {asof} KST",
        "단위: % (가격 변화율). 환율은 기준통화 대비(+면 달러 강세/해당통화 약세). N/A=데이터 없음.",
        "",
    ]
    for gname, recs in snapshot.items():
        lines.append(f"■ {gname}")
        for r in recs:
            lines.append(
                f"- {r['자산']}: 1D {fmt(r['1D'])}, 1W {fmt(r['1W'])}, 1M {fmt(r['1M'])}, YTD {fmt(r['YTD'])}"
            )
        lines.append("")
    return "\n".join(lines)


def _optional_sector_sentiment(sector_label: str):
    """선택한 섹터의 최근 3일 뉴스 감성을 FinBERT로 요약(텍스트, df 반환). 실패 시 (None, None)."""
    try:
        etf = SECTOR_ETFS[sector_label]
        holdings = ETFCollector().get_etf_holdings(etf)
        if not holdings:
            return None, None
        news = NewsCollector(days=3).collect_all(holdings, etf)
        if not news:
            return None, None
        analyzed = load_analyzer().batch_analyze(news)
        df = build_sentiment_df(analyzed)
        if df.empty:
            return None, None
        agg = (df.groupby('Ticker')['Sentiment']
                 .agg(['mean', 'count']).reset_index()
                 .sort_values('mean', ascending=False))
        lines = [f"[{sector_label} 섹터 뉴스 감성 — 최근 3일, FinBERT 점수 -1~+1]"]
        for _, r in agg.iterrows():
            lines.append(f"- {r['Ticker']}: 평균감성 {r['mean']:+.3f} ({int(r['count'])}건)")
        return "\n".join(lines), df
    except Exception:
        return None, None


def _rule_based_briefing(snapshot: dict) -> str:
    all_assets = []
    for g, recs in snapshot.items():
        for r in recs:
            if isinstance(r.get('1D'), (int, float)):
                all_assets.append((g, r['자산'], r['1D']))
    lines = ["#### 🧭 자동 요약 (규칙 기반 · API 키 없이 생성)"]
    if not all_assets:
        lines.append("- 데이터가 부족하여 요약을 생성할 수 없습니다.")
        return "\n".join(lines)

    by_1d = sorted(all_assets, key=lambda x: x[2], reverse=True)
    gainers = by_1d[:3]
    losers = list(reversed(by_1d[-3:]))
    lines.append("- **상승 상위(1D):** " + ", ".join(f"{a[1]} ({a[2]:+.2f}%)" for a in gainers))
    lines.append("- **하락 하위(1D):** " + ", ".join(f"{a[1]} ({a[2]:+.2f}%)" for a in losers))

    def grp_avg(g, key='1D'):
        vals = [r[key] for r in snapshot.get(g, []) if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    eq, bd = grp_avg('주식'), grp_avg('채권')
    if eq is not None and bd is not None:
        if eq > 0 and bd <= 0:
            risk = "주식↑·채권↓ → **리스크온(위험선호)** 성향"
        elif eq < 0 and bd >= 0:
            risk = "주식↓·채권↑ → **리스크오프(안전선호)** 성향"
        else:
            risk = "주식·채권 혼조 → 방향성 뚜렷하지 않음"
        lines.append(f"- **크로스에셋:** 주식 평균 {eq:+.2f}%, 채권 평균 {bd:+.2f}% — {risk}")

    usd_pairs = [r['1D'] for r in snapshot.get('환율', [])
                 if r['자산'] in ('달러-원', '달러-엔', '달러-위안') and isinstance(r.get('1D'), (int, float))]
    if usd_pairs:
        usd_avg = sum(usd_pairs) / len(usd_pairs)
        lines.append(f"- **달러:** 주요 통화 대비 평균 {usd_avg:+.2f}% → 달러 {'강세' if usd_avg > 0 else '약세'}")

    return "\n".join(lines)


def _get_secret(name, default=None):
    try:
        val = st.secrets.get(name, default)
        return val if val not in (None, "") else default
    except Exception:
        return default


def _claude_briefing(user_text: str):
    """Claude(Anthropic API) 호출 → (브리핑 텍스트, 오류메시지). 성공 시 오류메시지=None."""
    try:
        import anthropic
    except Exception:
        return None, "`anthropic` 패키지가 설치되어 있지 않습니다. requirements.txt 에 `anthropic` 을 추가하세요."

    api_key = _get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY 가 설정되지 않았습니다. Streamlit Secrets 에 키를 추가하세요."

    model = _get_secret("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1800,
            system=BRIEFING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        if not text.strip():
            return None, "Claude 응답이 비어 있습니다."
        return text, None
    except Exception as e:
        return None, f"Claude API 호출 실패: {e}"


def _render_snapshot_tables(snapshot: dict):
    cmap = create_transparent_YlOrBr_cmap(alpha=0.4)
    for gname, recs in snapshot.items():
        sdf = pd.DataFrame(recs)
        if sdf.empty:
            continue
        st.markdown(f"**{gname}**")
        num_cols = [c for c in ['1D', '1W', '1M', 'YTD'] if c in sdf.columns]
        styled = sdf.style.format({c: '{:+.2f}' for c in num_cols}, na_rep='N/A')
        for c in num_cols:
            vals = pd.to_numeric(sdf[c], errors='coerce')
            if vals.notna().any():
                styled = styled.background_gradient(
                    subset=[c], cmap=cmap,
                    vmin=float(vals.min()), vmax=float(vals.max()), low=0.3, high=0.3,
                )
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=min(420, 40 + len(sdf) * 35))


def show_ai_briefing():
    st.markdown(f'<h1 style="color: {TITLE_COLOR};">🧠 AI 마켓 브리핑</h1>', unsafe_allow_html=True)
    st.caption("최근 크로스에셋 성과와 시장 국면 지표(VIX·금리 등)를 종합해 한국어 데일리 마켓 브리핑을 자동 생성합니다.")

    col1, col2 = st.columns([1, 2], vertical_alignment="center")
    with col1:
        include_news = st.checkbox("📰 섹터 뉴스 감성 포함 (느림)", value=False, key="ai_inc_news")
    with col2:
        sector_for_news = st.selectbox(
            "감성 분석 섹터", list(SECTOR_ETFS.keys()),
            key="ai_news_sector", disabled=not include_news,
        )

    run = st.button("🧠 브리핑 생성", type="primary", key="ai_run")

    if run:
        today_str = datetime.now().strftime('%Y-%m-%d')
        with st.spinner("시장 스냅샷 수집 중..."):
            snapshot = collect_briefing_snapshot(today_str)

        with st.spinner("시장 국면 지표 수집 중..."):
            regime = fetch_regime_indicators(today_str)

        sentiment_text, sentiment_df = (None, None)
        if include_news:
            with st.spinner(f"{sector_for_news} 섹터 뉴스 감성 분석 중... (FinBERT)"):
                sentiment_text, sentiment_df = _optional_sector_sentiment(sector_for_news)

        user_text = _snapshot_to_text(snapshot) + "\n\n" + _regime_to_text(regime)
        if sentiment_text:
            user_text += "\n\n" + sentiment_text

        with st.spinner("브리핑 작성 중..."):
            briefing, _err = _claude_briefing(user_text)

        st.session_state['ai_briefing'] = {
            'snapshot': snapshot,
            'regime': regime,
            'briefing': briefing,
            'sentiment_df': sentiment_df,
            'asof': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }

    if 'ai_briefing' not in st.session_state:
        st.info("'🧠 브리핑 생성' 버튼을 누르면 최신 시장 데이터를 모아 브리핑을 만듭니다.")
        return

    res = st.session_state['ai_briefing']
    snapshot = res['snapshot']
    regime = res.get('regime', {})

    if res.get('briefing'):
        # 운영자가 키를 연결한 경우 — Claude 가 작성한 풍부한 브리핑
        st.markdown(res['briefing'])
        st.caption(f"🤖 AI 코멘트 · 생성: {res['asof']}")
    else:
        # 키가 없을 때 — 동일 데이터로 계산하는 규칙 기반 자동 분석
        st.markdown(_rule_based_briefing(snapshot))
        rc = _regime_comment(regime, None)
        if rc and not rc.startswith("_지표"):
            st.markdown("#### 🌡️ 시장 국면")
            st.markdown(rc)
        st.caption(f"🧭 자동 분석 · 생성: {res['asof']}")

    st.markdown("---")
    with st.expander("📊 브리핑에 사용된 시장 스냅샷 보기", expanded=False):
        _render_snapshot_tables(snapshot)

    if res.get('sentiment_df') is not None and not res['sentiment_df'].empty:
        st.markdown("---")
        st.markdown(f'<h3 style="color: {TITLE_COLOR};">📰 섹터 뉴스 감성</h3>', unsafe_allow_html=True)
        render_sentiment_bar_chart(res['sentiment_df'], st.session_state.get('ai_news_sector', ''))



# ======================================================
# Page 5: 시장 국면 & 모멘텀 (Regime Cockpit)
#  - 변동성(VIX)·미 국채 금리·시장 폭(Breadth)·추세 모멘텀을 한 화면에 모은
#    ETF 포트폴리오 운용자용 리스크 점검 대시보드.
# ======================================================
def show_regime():
    st.markdown(f'<h1 style="color: {TITLE_COLOR};">📡 시장 국면 & 모멘텀</h1>', unsafe_allow_html=True)
    st.caption("변동성(VIX)·미 국채 금리·시장 폭(Breadth)·추세 모멘텀을 한 화면에서 점검합니다. (데이터: Yahoo Finance, 무료)")

    run = st.button("📡 국면 점검 실행", type="primary", key="regime_run")
    if not (run or st.session_state.get('regime_loaded')):
        st.info("'📡 국면 점검 실행' 버튼을 누르면 변동성·금리·시장 폭·모멘텀을 한 번에 점검합니다.")
        return

    st.session_state['regime_loaded'] = True
    today = datetime.now().date()
    today_str = today.strftime('%Y-%m-%d')

    with st.spinner("국면 지표 수집 중... (VIX·금리)"):
        reg = fetch_regime_indicators(today_str)

    # 유니버스: 지역 주식 + 섹터 + 스타일
    universe = {}
    universe.update(STOCK_ETFS)
    universe.update(SECTOR_ETFS)
    universe.update(STYLE_ETFS)
    with st.spinner("자산 시계열 수집 중..."):
        prices = download_close_prices(list(universe.values()),
                                       today - timedelta(days=420), today + timedelta(days=1))
    if prices is not None and not prices.empty:
        keep = [t for t in universe.values() if t in prices.columns]
        prices = prices.ffill()[keep]

    # ---- 핵심 국면 지표 ----
    st.markdown(f'<h3 style="color: {TITLE_COLOR};">🌡️ 핵심 국면 지표</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    vix, vix_chg = reg.get('vix'), reg.get('vix_chg')
    if vix is not None:
        regime_lbl = ('낮음(안정)' if vix < 15 else '보통' if vix < 20
                      else '높음(경계)' if vix < 30 else '매우 높음(스트레스)')
        c1.metric("VIX (변동성)", f"{vix:.1f}",
                  f"{vix_chg:+.1f}" if vix_chg is not None else None, delta_color="inverse")
        c1.caption(f"국면: {regime_lbl}")
    else:
        c1.metric("VIX (변동성)", "N/A")

    t10, t10c = reg.get('us10y'), reg.get('us10y_chg')
    if t10 is not None:
        c2.metric("미 국채 10Y", f"{t10:.2f}%",
                  f"{t10c * 100:+.0f}bp" if t10c is not None else None, delta_color="off")
    else:
        c2.metric("미 국채 10Y", "N/A")

    t3 = reg.get('us3m')
    c3.metric("미 단기 (3M)", f"{t3:.2f}%" if t3 is not None else "N/A")

    slope = reg.get('slope_10y_3m')
    if slope is not None:
        c4.metric("장단기차 (10Y-3M)", f"{slope:+.2f}%p", "역전" if slope < 0 else None,
                  delta_color="inverse")
        c4.caption("⚠️ 역전 — 경기둔화 경계" if slope < 0 else "우상향(정상)")
    else:
        c4.metric("장단기차 (10Y-3M)", "N/A")

    if prices is not None and not prices.empty:
        # ---- 시장 폭 (Breadth) ----
        br = compute_breadth(prices)
        st.markdown("---")
        st.markdown(f'<h3 style="color: {TITLE_COLOR};">📈 시장 폭 (Breadth)</h3>', unsafe_allow_html=True)
        st.caption("추적 ETF 중 이동평균선을 상회하는 비율 — 높을수록 상승 추세가 광범위함")
        b1, b2 = st.columns(2)
        a50, a200 = br.get('above_50'), br.get('above_200')
        if a50 is not None:
            b1.metric(f"50일선 상회  ({br['count_50']}/{br['n50']})", f"{a50:.0f}%")
            b1.progress(min(1.0, a50 / 100))
        if a200 is not None:
            b2.metric(f"200일선 상회  ({br['count_200']}/{br['n200']})", f"{a200:.0f}%")
            b2.progress(min(1.0, a200 / 100))

        # ---- 추세·모멘텀 점검 ----
        st.markdown("---")
        st.markdown(f'<h3 style="color: {TITLE_COLOR};">🚦 추세·모멘텀 점검</h3>', unsafe_allow_html=True)
        st.caption("강세=현재가가 50·200일선 위 · 약세=둘 다 아래 · 혼조=혼재 / RSI>70 과매수·<30 과매도")
        mom = compute_momentum_table(universe, prices)
        if not mom.empty:
            num_fmt = {'현재가': '{:.2f}', 'vs50MA(%)': '{:+.2f}', 'vs200MA(%)': '{:+.2f}',
                       'RSI(14)': '{:.1f}', '1M(%)': '{:+.2f}', '52주고점대비(%)': '{:+.2f}'}
            styled = mom.style.format(num_fmt, na_rep='N/A')
            cmap = create_transparent_YlOrBr_cmap(alpha=0.4)
            for col in ['vs50MA(%)', 'vs200MA(%)', '1M(%)', '52주고점대비(%)']:
                vals = pd.to_numeric(mom[col], errors='coerce')
                if vals.notna().any():
                    styled = styled.background_gradient(
                        subset=[col], cmap=cmap,
                        vmin=float(vals.min()), vmax=float(vals.max()), low=0.3, high=0.3)

            def _rsi_color(v):
                if pd.isna(v):
                    return ''
                if v >= 70:
                    return 'background-color: rgba(231,76,60,0.25)'
                if v <= 30:
                    return 'background-color: rgba(46,134,222,0.25)'
                return ''

            def _trend_color(v):
                return {'강세': 'color:#1e8e3e; font-weight:600',
                        '약세': 'color:#e74c3c; font-weight:600',
                        '혼조': 'color:#b08900'}.get(v, '')

            styled = styled.map(_rsi_color, subset=['RSI(14)'])
            styled = styled.map(_trend_color, subset=['추세'])
            st.dataframe(styled, use_container_width=True, hide_index=True,
                         height=min(760, 45 + len(mom) * 35))
    else:
        st.warning("자산 시계열을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")

    # ---- 국면 요약 ----
    st.markdown("---")
    st.markdown(f'<h3 style="color: {TITLE_COLOR};">🧭 국면 요약</h3>', unsafe_allow_html=True)
    st.markdown(_regime_comment(reg, prices))


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
    st.caption("*펀드 보유 종목만 분석되도록 개발 예정")
    st.markdown("---")
    page = st.radio(
        "페이지 선택",
        ["📊 시장 성과", "📡 시장 국면", "🧠 AI 브리핑", "🤖 LLM 분석", "👨‍💼 애널리스트"],
        key="nav_page",
    )
    st.markdown("---")
    st.markdown('<div style="font-size:0.85rem;">Data source: '
                '<a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a></div>',
                unsafe_allow_html=True)

    st.caption(f"Last visit: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.divider()
    st.caption("© 2026 KB Asset Management.")

    with st.sidebar.expander("📄 MIT License Details"):
        license_html = """
        <div style="font-size: 0.8rem; color: #808080; line-height: 1.5;">
            MIT License<br>
            Copyright (c) 2026 KB Asset Management<br><br>
            Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
            associated documentation files...<br><br>
            The above copyright notice and this permission notice shall be included in all copies or 
            substantial portions of the Software.<br><br>
            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED...<br><br>
            <a href="https://opensource.org/license/mit" target="_blank" style="color: #606060; text-decoration: underline;">
                View Full License Page
            </a>
        </div>
        """
        st.markdown(license_html, unsafe_allow_html=True)


if page == "📊 시장 성과":
    show_page1()
elif page == "📡 시장 국면":
    show_regime()
elif page == "🧠 AI 브리핑":
    show_ai_briefing()
elif page == "🤖 LLM 분석":
    show_page2()
elif page == "👨‍💼 애널리스트":
    show_page3()
