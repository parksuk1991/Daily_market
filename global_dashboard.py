import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
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
    update_clicked = st.button("Update", type="primary", key="main_update_btn")
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
    '캐나다(Canada, EWC)': 'EWC',
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
    '단기국채(SPTS)': 'SPTS',
}

CURRENCY = {
    '달러인덱스': 'DX-Y.NYB',
    '달러-원': 'KRW=X',
    '유로-원': 'EURKRW=X',
    '달러-엔': 'JPY=X',
    '원-엔': 'JPYKRW=X',
    '달러-유로': 'EURUSD=X',
    '달러-파운드': 'GBPUSD=X',
    '달러-위안': 'CNY=X',
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
    'Low Volatility (USMV)': 'USMV',
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
    '부동산 (XLRE)': 'XLRE',
}

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


# ======================================================
# ====== ETF Collector (완전 동적 — 4가지 방법 체인) ======
# ======================================================
class ETFCollector:
    """
    ETF Top-10 Holdings 수집기.
    수집 시도 순서:
      1. SSGA 공식 XLSX  (SPDR 계열 ETF용, State Street 직접 제공)
      2. stockanalysis.com 스크래핑
      3. yahooquery
      4. yfinance fund_top_holdings
    hardcoding 없음 — 모두 실패 시 빈 리스트 반환.
    """

    def __init__(self):
        self.cf_session = None
        try:
            from curl_cffi import requests as cffi_req
            self.cf_session = cffi_req.Session(impersonate="chrome")
            self.cf_session.verify = False
        except Exception:
            pass

    # ── 방법 1 : SSGA 공식 XLSX ─────────────────────────
    def _try_ssga(self, ticker: str):
        """
        State Street (SPDR) 일별 holdings XLSX.
        URL: https://www.ssga.com/us/en/intermediary/etfs/library-content/
             products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx
        SPDR이 아닌 ETF는 404 → 빠르게 스킵.
        """
        url = (
            "https://www.ssga.com/us/en/intermediary/etfs/library-content/"
            f"products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15, verify=False)
            if resp.status_code != 200:
                return []

            # SSGA XLSX 구조:
            #   행 0-3  : 메타데이터
            #   행 4    : 컬럼 헤더 (Name, Ticker, Identifier, SEDOL, Weight, Sector, ...)
            #   행 5+   : 실제 종목
            df = pd.read_excel(BytesIO(resp.content), skiprows=4, engine='openpyxl')
            if df.empty:
                return []

            df.columns = [str(c).strip() for c in df.columns]

            ticker_col = next(
                (c for c in df.columns if 'ticker' in c.lower() or 'symbol' in c.lower()), None
            )
            weight_col = next(
                (c for c in df.columns if 'weight' in c.lower()), None
            )
            name_col = next(
                (c for c in df.columns if 'name' in c.lower()), ticker_col
            )
            if not ticker_col or not weight_col:
                return []

            df = df.dropna(subset=[ticker_col])
            df[ticker_col] = df[ticker_col].astype(str).str.strip()
            df = df[df[ticker_col].str.len() > 0]
            df = df[~df[ticker_col].str.lower().isin(['nan', 'ticker', 'symbol', '-'])]
            df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
            df = df.dropna(subset=[weight_col])
            df = df[df[weight_col] > 0]
            df = df.sort_values(weight_col, ascending=False).head(10)

            return [
                {
                    'ticker': str(row[ticker_col]).strip(),
                    'name':   str(row[name_col]).strip(),
                    'weight': round(float(row[weight_col]), 2),
                }
                for _, row in df.iterrows()
                if str(row[ticker_col]).strip()
            ]
        except Exception:
            return []

    # ── 방법 2 : stockanalysis.com 스크래핑 ──────────────
    def _try_stockanalysis(self, ticker: str):
        """
        https://stockanalysis.com/etf/{ticker}/holdings/
        테이블 컬럼: # | Symbol | Company | Weight% | Shares | Revenue%
        """
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
            for row in table.find_all('tr')[1:]:   # 헤더 스킵
                cols = [td.get_text(strip=True) for td in row.find_all('td')]
                if len(cols) < 4:
                    continue
                # cols[0]=rank, cols[1]=symbol, cols[2]=company, cols[3]=weight%
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

    # ── 방법 3 : yahooquery ──────────────────────────────
    def _try_yahooquery(self, ticker: str):
        try:
            from yahooquery import Ticker
            etf = Ticker(ticker, session=self.cf_session) if self.cf_session else Ticker(ticker)
            holdings = etf.fund_holding_info
            if ticker in holdings and isinstance(holdings[ticker], dict):
                raw = holdings[ticker].get('holdings', [])
                result = [
                    {
                        'ticker': h.get('symbol', ''),
                        'name':   h.get('holdingName', h.get('symbol', '')),
                        'weight': round(h.get('holdingPercent', 0.0) * 100, 2),
                    }
                    for h in raw[:10]
                    if h.get('symbol', '')
                ]
                return result
        except Exception:
            pass
        return []

    # ── 방법 4 : yfinance fund_top_holdings ──────────────
    def _try_yfinance(self, ticker: str):
        try:
            t = yf.Ticker(ticker)
            df = t.fund_top_holdings
            if df is None or df.empty:
                return []
            result = []
            for _, row in df.head(10).iterrows():
                sym  = str(row.get('Symbol',       row.get('symbol', ''))).strip()
                name = str(row.get('Holding Name', row.get('holdingName', sym))).strip()
                pct  = float(row.get('% Assets',   row.get('holdingPercent', 0)) or 0)
                if pct < 1:        # yfinance가 0~1 범위로 반환하는 경우
                    pct *= 100
                if sym and sym != 'nan':
                    result.append({'ticker': sym, 'name': name, 'weight': round(pct, 2)})
            return result
        except Exception:
            return []

    # ── 공개 API ──────────────────────────────────────────
    def get_etf_holdings(self, ticker: str, retry: int = 2):
        """
        4가지 방법 순서대로 시도.
        하나라도 성공하면 즉시 반환, 모두 실패하면 빈 리스트.
        """
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

    def get_etf_name(self, ticker: str):
        try:
            from yahooquery import Ticker
            etf = Ticker(ticker, session=self.cf_session) if self.cf_session else Ticker(ticker)
            qt = etf.quote_type
            if ticker in qt:
                return qt[ticker].get('longName', f'{ticker} ETF')
        except Exception:
            pass
        return f'{ticker} ETF'


# ====== News Collector ======
class NewsCollector:
    """Yahoo RSS 기반 뉴스 수집"""

    def __init__(self, days=3):
        self.days = days
        self.cutoff_date = datetime.now() - timedelta(days=days)

    def extract_content(self, url: str):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=8)
            if resp.status_code != 200:
                return ""
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            texts = [
                p.get_text(strip=True)
                for p in soup.find_all('p')
                if len(p.get_text(strip=True)) > 30
            ]
            full = ' '.join(texts)
            return full[:5000] if full else ""
        except Exception:
            return ""

    def is_valid_content(self, content: str):
        if not content or len(content) < 200:
            return False
        lower = content.lower()
        return not any(w in lower[:500] for w in ['sign in', 'log in', 'subscribe', 'register'])

    def create_summary(self, text: str):
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

    def collect_yahoo_rss(self, ticker: str):
        try:
            feed = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={ticker}")
            news = []
            for entry in feed.entries[:3]:
                try:
                    pub = entry.get('published_parsed')
                    if pub:
                        pub_dt = datetime(*pub[:6])
                        if pub_dt < self.cutoff_date:
                            continue
                        date_str = pub_dt.strftime('%Y-%m-%d')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    content = self.extract_content(entry.get('link', ''))
                    if not self.is_valid_content(content):
                        continue
                    news.append({
                        'ticker':       ticker,
                        'title':        entry.get('title', ''),
                        'url':          entry.get('link', ''),
                        'published_at': date_str,
                        'summary':      entry.get('summary', '')[:300],
                        'content':      content,
                        'highlights':   self.create_summary(content),
                        'source':       'Yahoo Finance',
                    })
                except Exception:
                    continue
            return news
        except Exception:
            return []

    def collect_for_ticker(self, ticker: str, company: str):
        news = self.collect_yahoo_rss(ticker)
        for item in news:
            item['company_name'] = company
        return news

    def collect_all(self, holdings, etf_ticker: str):
        all_news = []
        for holding in holdings:
            news = self.collect_for_ticker(holding['ticker'], holding['name'])
            for item in news:
                item['etf']    = etf_ticker
                item['weight'] = holding['weight']
            all_news.extend(news)
            time.sleep(0.3)
        return all_news


# ====== FinBERT Analyzer ======
class FinBERTAnalyzer:
    def __init__(self):
        self.pipe = None
        try:
            from transformers import pipeline as hf_pipeline
            self.pipe = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=-1,
                max_length=512,
                truncation=True,
            )
        except Exception:
            pass

    def analyze_chunk(self, text: str):
        if not self.pipe or not text or len(text) < 10:
            return 0.0
        try:
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            res   = self.pipe(text[:512])[0]
            label, score = res['label'], res['score']
            return score if label == 'positive' else (-score if label == 'negative' else 0.0)
        except Exception:
            return 0.0

    def analyze_text(self, text: str):
        if not text or len(text) < 100:
            return 0.0
        chunks = [
            text[i:i+1000]
            for i in range(0, min(len(text), 3000), 1000)
            if len(text[i:i+1000]) > 100
        ]
        scores = [s for s in (self.analyze_chunk(c) for c in chunks[:3]) if s != 0.0]
        return sum(scores) / len(scores) if scores else 0.0

    def categorize(self, title: str):
        t = title.lower()
        if any(w in t for w in ['earnings', 'revenue', 'profit']):   return 'Earnings'
        if any(w in t for w in ['merger', 'acquisition', 'deal']):   return 'M&A'
        if any(w in t for w in ['product', 'launch']):               return 'Product'
        if any(w in t for w in ['regulation', 'lawsuit']):           return 'Regulatory'
        if any(w in t for w in ['analyst', 'upgrade', 'downgrade']): return 'Analyst'
        return 'General'

    def analyze_news(self, news: dict):
        if len(news.get('content', '')) < 100:
            return None
        news['sentiment_score'] = round(self.analyze_text(news['content']), 4)
        news['category']        = self.categorize(news.get('title', ''))
        return news

    def batch_analyze(self, news_list: list):
        return [r for r in (self.analyze_news(n) for n in news_list) if r]


# ---- 주요 데이터 함수 ----
def get_perf_table_improved(label2ticker, ref_date=None):
    tickers = list(label2ticker.values())
    if ref_date is None:
        ref_date = datetime.now().date()
    start = ref_date - timedelta(days=4 * 365)
    end   = ref_date + timedelta(days=1)

    try:
        df = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.ffill().dropna(how='all')[tickers]
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    avail      = df.index[df.index.date <= ref_date]
    if len(avail) == 0:
        return pd.DataFrame()
    last_trade = avail[-1].date()
    last_idx   = avail[-1]

    periods = {
        '1D(%)':  {'days': 1,   'type': 'business'},
        '1W(%)':  {'days': 5,   'type': 'business'},
        'MTD(%)': {'type': 'month_start'},
        '1M(%)':  {'days': 21,  'type': 'business'},
        '3M(%)':  {'days': 63,  'type': 'business'},
        '6M(%)':  {'days': 126, 'type': 'business'},
        'YTD(%)': {'type': 'year_start'},
        '1Y(%)':  {'days': 252, 'type': 'business'},
        '3Y(%)':  {'days': 756, 'type': 'business'},
    }

    results = []
    for label, ticker in label2ticker.items():
        row    = {'자산명': label}
        series = df[ticker].dropna()
        if last_idx not in series.index or len(series) == 0:
            row['현재값'] = np.nan
            for pk in periods: row[pk] = np.nan
            results.append(row)
            continue

        curr        = series.loc[last_idx]
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
                    ci   = series.index.get_loc(last_idx)
                    lb   = cfg['days']
                    base = series.iloc[ci - lb] if ci >= lb else (series.iloc[0] if ci > 0 else None)
                row[pk] = (curr / base - 1) * 100 if (base is not None and not np.isnan(base) and base != 0) else np.nan
            except Exception:
                row[pk] = np.nan
        results.append(row)

    df_r = pd.DataFrame(results)
    if '현재값' in df_r.columns:
        df_r['현재값'] = df_r['현재값'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
    return df_r


def get_sample_calculation_dates(label2ticker, ref_date=None):
    if ref_date is None:
        ref_date = datetime.now().date()
    sample_ticker = list(label2ticker.values())[0]
    sample_label  = list(label2ticker.keys())[0]
    try:
        data  = yf.download(
            sample_ticker,
            start=ref_date - timedelta(days=4 * 365),
            end=ref_date + timedelta(days=1),
            progress=False,
        )['Close'].dropna()
        avail = data.index[data.index.date <= ref_date]
        if len(avail) == 0:
            return None, None, None
        last_trade = avail[-1].date()
        ci         = data.index.get_loc(avail[-1])
        actual     = {}
        for p, d in {'1D': 1, '1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '3Y': 756}.items():
            actual[p] = data.index[ci - d].date().strftime('%Y-%m-%d') if ci >= d else "데이터 부족"
        for key, dt in [('MTD', last_trade.replace(day=1)), ('YTD', last_trade.replace(month=1, day=1))]:
            d = data[data.index.date >= dt]
            if len(d): actual[key] = d.index[0].date().strftime('%Y-%m-%d')
        return sample_label, last_trade.strftime('%Y-%m-%d'), actual
    except Exception:
        return None, None, None


@st.cache_data(show_spinner="차트 데이터 로딩 중...")
def get_normalized_prices(label2ticker, months=6):
    tickers = list(label2ticker.values())
    end     = datetime.now().date()
    start   = end - timedelta(days=months * 31)
    df      = yf.download(tickers, start=start, end=end + timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df   = df.ffill()[tickers]
    norm = df / df.iloc[0] * 100
    norm.columns = list(label2ticker.keys())
    return norm


def format_percentage(val):
    if pd.isna(val): return "N/A"
    try:    return f"{float(val):.2f}%"
    except: return "N/A"


def colorize_return(val):
    if pd.isna(val): return ""
    try:
        v = float(val) if isinstance(val, (int, float)) else float(str(val).replace('%', '').strip())
    except: return ""
    return "color: red;" if v > 0 else ("color: blue;" if v < 0 else "")


def style_perf_table(df, perf_cols):
    styled = df.style
    for col in perf_cols:
        if col in df.columns:
            styled = styled.format({col: format_percentage}).applymap(colorize_return, subset=[col])
    return styled


# ====== 섹터 분석 ======
@st.cache_resource
def load_analyzer():
    return FinBERTAnalyzer()


def run_sector_etf_analysis(etf_ticker: str, etf_name: str):
    """단일 섹터 ETF 분석"""
    try:
        collector = ETFCollector()
        holdings  = collector.get_etf_holdings(etf_ticker)
        if not holdings:
            return None, f"❌ {etf_name}: Holdings 수집 실패 (4가지 방법 모두 실패)"

        all_news = NewsCollector(days=3).collect_all(holdings, etf_ticker)
        if not all_news:
            return holdings, f"⚠️ {etf_name}: Holdings {len(holdings)}개 확인, 최근 뉴스 없음"

        analyzed = load_analyzer().batch_analyze(all_news)
        return analyzed, None
    except Exception as e:
        return None, f"❌ {etf_name}: {str(e)[:80]}"


def show_sector_analysis():
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
        except Exception:
            st.warning(f"❌ {sector_name}: 오류 발생")

    if not sector_results:
        st.warning("섹터 분석 데이터를 가져올 수 없습니다.")
        return

    for sector_name, news_list in sector_results.items():
        st.markdown(f"#### {sector_name}")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("뉴스 개수", len(news_list))
        with c2:
            avg_s = np.mean([n.get('sentiment_score', 0) for n in news_list])
            st.metric("평균 감정", f"{avg_s:.3f}")
        with c3:
            pos = sum(1 for n in news_list if n.get('sentiment_score', 0) > 0.1)
            st.metric("긍정 뉴스", f"{pos}/{len(news_list)}")

        news_df = pd.DataFrame([{
            'Ticker':    n.get('ticker', ''),
            'Title':     n.get('title', '')[:80],
            'Date':      n.get('published_at', ''),
            'Sentiment': round(n.get('sentiment_score', 0), 3),
            'Category':  n.get('category', ''),
        } for n in news_list[:10]])
        st.dataframe(news_df, use_container_width=True)
        st.markdown("---")


def show_all_performance_tables():
    perf_cols = ['1D(%)', '1W(%)', 'MTD(%)', '1M(%)', '3M(%)', '6M(%)', 'YTD(%)', '1Y(%)', '3Y(%)']
    for title, label2t, h in [
        ("📊 주식시장",   STOCK_ETFS,  490),
        ("🗠 채권시장",   BOND_ETFS,   385),
        ("💱 통화",       CURRENCY,    315),
        ("📈 암호화폐",   CRYPTO,      385),
        ("📕 스타일 ETF", STYLE_ETFS,  245),
        ("📘 섹터 ETF",   SECTOR_ETFS, 420),
    ]:
        st.subheader(title)
        with st.spinner(f"{title} 계산 중..."):
            perf = get_perf_table_improved(label2t)
        if not perf.empty:
            st.dataframe(
                style_perf_table(perf.set_index('자산명'), perf_cols),
                use_container_width=True, height=h,
            )


# ---- 메인 레이아웃 ----
period_options = {"3개월": 3, "6개월": 6, "12개월": 12, "24개월": 24, "36개월": 36}

if update_clicked:
    st.session_state['updated'] = True

if st.session_state.get('updated', False):
    st.markdown("<br>", unsafe_allow_html=True)
    show_all_performance_tables()
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 주가지수 차트", "📗 섹터 차트", "📙 스타일 차트", "📰 섹터 분석", "📋 정보"]
    )

    def render_chart(label2t, session_key, select_key):
        if session_key not in st.session_state:
            st.session_state[session_key] = 6
        months = st.selectbox(
            "기간 선택",
            options=list(period_options.keys()),
            index=list(period_options.values()).index(st.session_state[session_key]),
            key=select_key,
        )
        mv = period_options[months]
        st.session_state[session_key] = mv
        with st.spinner("차트 로딩 중..."):
            norm = get_normalized_prices(label2t, months=mv)
            fig  = go.Figure()
            for col in norm.columns:
                fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode='lines', name=col))
            fig.update_layout(
                yaxis_title="100 기준 누적수익률(%)",
                template="plotly_dark", height=500,
                legend=dict(orientation='h'),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab1:
        st.subheader("✅ 주요 주가지수 수익률")
        render_chart(STOCK_ETFS, "idx_months", "idx_selectbox")

    with tab2:
        st.subheader("☑️ 섹터 ETF 수익률")
        render_chart(SECTOR_ETFS, "sector_months", "sector_selectbox")

    with tab3:
        st.subheader("☑️ 스타일 ETF 수익률")
        render_chart(STYLE_ETFS, "style_months", "style_selectbox")

    with tab4:
        show_sector_analysis()

    with tab5:
        st.subheader("📋 계산 기준일")
        sample_label, last_date, actual_dates = get_sample_calculation_dates(STOCK_ETFS)
        if sample_label and actual_dates:
            st.caption(f"**샘플 자산:** {sample_label} | **최근 거래일:** {last_date}")
            l1 = [f"{p}: {actual_dates[p]}" for p in ['1D', '1W', 'MTD', '1M'] if p in actual_dates]
            st.caption("• " + " | ".join(l1))
            l2 = [f"{p}: {actual_dates[p]}" for p in ['3M', '6M', 'YTD', '1Y', '3Y'] if p in actual_dates]
            st.caption("• " + " | ".join(l2))

else:
    st.info("상단 'Update' 버튼을 눌러주세요.")
