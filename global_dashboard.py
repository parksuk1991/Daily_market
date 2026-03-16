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

warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(
    page_title="Global Market Monitoring",
    page_icon="🌐",
    layout="wide"
)

# =========== 자산 정의 ================
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
    '달러인덱스': 'DX-Y.NYB', '달러-원': 'KRW=X', '유로-원': 'EURKRW=X',
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
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


# ======================================================
# ====== ETF Collector (4-method chain, no hardcoding) =
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
            soup  = BeautifulSoup(resp.text, 'html.parser')
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
            etf      = Ticker(ticker, session=self.cf_session) if self.cf_session else Ticker(ticker)
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
                sym  = str(row.get('Symbol',       row.get('symbol', ''))).strip()
                name = str(row.get('Holding Name', row.get('holdingName', sym))).strip()
                pct  = float(row.get('% Assets',   row.get('holdingPercent', 0)) or 0)
                if pct < 1: pct *= 100
                if sym and sym != 'nan':
                    result.append({'ticker': sym, 'name': name, 'weight': round(pct, 2)})
            return result
        except Exception:
            return []

    def get_etf_holdings(self, ticker: str, retry: int = 2):
        result = self._try_ssga(ticker)
        if result: return result
        result = self._try_stockanalysis(ticker)
        if result: return result
        for attempt in range(retry):
            result = self._try_yahooquery(ticker)
            if result: return result
            if attempt < retry - 1: time.sleep(1.5)
        return self._try_yfinance(ticker)


# ======================================================
# ====== News Collector (Title-only relevance filter) ==
# ======================================================
class NewsCollector:
    """
    관련성 필터: 뉴스 제목(title)에만 ticker 심볼 또는 기업명이 포함되어야 함.
    본문 검색 시 타 종목을 언급하는 기사가 오탐되는 문제를 원천 차단.
    """
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
            full  = ' '.join(texts)
            return full[:5000] if full else ""
        except Exception:
            return ""

    def is_valid_content(self, content: str) -> bool:
        if not content or len(content) < 200:
            return False
        lower = content.lower()
        return not any(w in lower[:500] for w in ['sign in', 'log in', 'subscribe', 'register'])

    def is_relevant(self, title: str, ticker: str, company_name: str) -> bool:
        """
        제목(title)에만 종목 심볼 또는 기업명이 있는 경우만 관련 뉴스로 판단.
        본문을 포함하면 타 종목 언급 기사가 오탐됨 — 제목 한정으로 엄격하게 필터.
        """
        title_lower = title.lower()

        # 1) ticker 심볼 (예: META, AAPL) — 단어 경계 매칭
        ticker_clean = re.sub(r'[^a-zA-Z0-9]', '', ticker).lower()
        if re.search(r'\b' + re.escape(ticker_clean) + r'\b', title_lower):
            return True

        # 2) 기업명 — 전체명 또는 첫 유의미한 단어 (5자 이상)
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
        """Yahoo Finance RSS — 제목 기반 관련성 필터, 건수 제한 없음"""
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

                    title       = entry.get('title', '')
                    article_url = entry.get('link', '')

                    # ── 제목 기반 관련성 필터 (오탐 방지 핵심) ──────
                    if not self.is_relevant(title, ticker, company_name):
                        continue

                    content = self.extract_content(article_url)
                    if not self.is_valid_content(content):
                        continue

                    news.append({
                        'ticker':       ticker,
                        'title':        title,
                        'url':          article_url,
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
                "text-classification", model="ProsusAI/finbert",
                device=-1, max_length=512, truncation=True,
            )
        except Exception:
            pass

    def analyze_chunk(self, text: str) -> float:
        if not self.pipe or not text or len(text) < 10:
            return 0.0
        try:
            text  = re.sub(r'<[^>]+>', '', text)
            text  = re.sub(r'http\S+', '', text)
            text  = re.sub(r'\s+', ' ', text).strip()
            res   = self.pipe(text[:512])[0]
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
        if any(w in t for w in ['earnings', 'revenue', 'profit']):   return 'Earnings'
        if any(w in t for w in ['merger', 'acquisition', 'deal']):   return 'M&A'
        if any(w in t for w in ['product', 'launch']):               return 'Product'
        if any(w in t for w in ['regulation', 'lawsuit']):           return 'Regulatory'
        if any(w in t for w in ['analyst', 'upgrade', 'downgrade']): return 'Analyst'
        return 'General'

    def analyze_news(self, news: dict) -> dict:
        """
        항상 news 딕셔너리를 반환 (None 반환 없음).
        본문이 짧아 분석 불가한 경우 sentiment_score = 0.0으로 설정.
        """
        content = news.get('content', '')
        news['sentiment_score'] = (
            round(self.analyze_text(content), 4) if len(content) >= 100 else 0.0
        )
        news['category'] = self.categorize(news.get('title', ''))
        return news

    def batch_analyze(self, news_list: list) -> list:
        """전체 뉴스 반환 — 건수 제한, None 필터링 없음"""
        return [self.analyze_news(n) for n in news_list]


@st.cache_resource
def load_analyzer():
    return FinBERTAnalyzer()


# ======================================================
# ====== 감성 분석 차트 함수 ===========================
# ======================================================
def build_sentiment_df(news_list: list) -> pd.DataFrame:
    rows = []
    for n in news_list:
        s = n.get('sentiment_score', 0.0) or 0.0
        rows.append({
            'Ticker':    n.get('ticker', ''),
            'Company':   n.get('company_name', ''),
            'Date':      n.get('published_at', ''),
            'Title':     n.get('title', ''),
            'Headline':  n.get('highlights', ''),
            'Sentiment': float(s),
            'Category':  n.get('category', ''),
            'URL':       n.get('url', ''),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['Sentiment_Category'] = df['Sentiment'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    return df


def render_sentiment_bar_chart(df: pd.DataFrame, sector_name: str):
    """종목별 평균 감정 가로 막대 차트"""
    if df.empty:
        return
    agg = (df.groupby('Ticker')['Sentiment']
             .agg(['mean', 'count'])
             .reset_index()
             .rename(columns={'mean': 'Avg', 'count': 'N'})
             .sort_values('Avg', ascending=True))

    company_map = df.drop_duplicates('Ticker').set_index('Ticker')['Company'].to_dict()
    colors = ['#2ecc71' if v > 0.05 else ('#e74c3c' if v < -0.05 else '#95a5a6')
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
        xaxis=dict(title="Sentiment Score", range=[-1, 1],
                   tickvals=[-1, -0.5, 0, 0.5, 1]),
        yaxis=dict(title=""),
        template="plotly_dark",
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
        marker_color='rgba(235, 0, 140, 0.7)', opacity=0.8,
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
        template="plotly_dark", height=380, showlegend=True,
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
        'Positive': 'rgba(235,0,140,0.8)',
        'Neutral':  'rgba(102,194,165,0.8)',
        'Negative': 'rgba(65,105,225,0.8)',
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
        template="plotly_dark", height=380, showlegend=False,
    )
    return fig


def create_sentiment_boxplot(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    tickers = df['Ticker'].unique()
    colors  = px.colors.qualitative.Set3
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
        template="plotly_dark", height=480, showlegend=False,
    )
    return fig


def render_news_table(df: pd.DataFrame):
    """전체 뉴스 테이블 — 긍정/부정/중립 모두 표시"""
    if df.empty:
        st.info("관련 뉴스가 없습니다.")
        return

    display = df[['Date', 'Ticker', 'Title', 'Headline', 'Sentiment', 'Sentiment_Category', 'Category', 'URL']]\
        .copy().sort_values('Date', ascending=False).reset_index(drop=True)

    def color_sent(val):
        try:
            v = float(val)
            if v > 0.05:  return 'color: #2ecc71; font-weight:bold'
            if v < -0.05: return 'color: #e74c3c; font-weight:bold'
            return 'color: #95a5a6'
        except Exception:
            return ''

    styled = (display.style
              .applymap(color_sent, subset=['Sentiment'])
              .format({'Sentiment': '{:.4f}'}))
    st.dataframe(
        styled,
        column_config={'URL': st.column_config.LinkColumn('URL')},
        use_container_width=True,
        height=min(600, 40 + len(display) * 35),
    )


# ======================================================
# ====== 애널리스트 & 밸류에이션 데이터 ================
# ======================================================
def get_analyst_report_data(ticker_syms: list) -> pd.DataFrame:
    rows = []
    for sym in ticker_syms:
        try:
            info         = yf.Ticker(sym).info or {}
            current_px   = info.get('regularMarketPrice') or info.get('currentPrice')
            target_px    = info.get('targetMeanPrice')
            upside       = ((target_px / current_px - 1) * 100
                            if target_px and current_px and current_px != 0 else None)
            rows.append({
                'Ticker':        sym,
                '종목명':        info.get('shortName') or info.get('longName') or '',
                '등급 점수':     info.get('recommendationMean'),
                '등급':          info.get('recommendationKey', '').capitalize(),
                '목표주가':      target_px,
                '현재가':        current_px,
                '상승여력(%)':   upside,
            })
        except Exception:
            rows.append({'Ticker': sym, '종목명': '', '등급 점수': None,
                         '등급': None, '목표주가': None, '현재가': None, '상승여력(%)': None})
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    return df[['Ticker', '종목명', '등급 점수', '등급', '목표주가', '현재가', '상승여력(%)']]


def get_valuation_eps_table(ticker_syms: list) -> pd.DataFrame:
    rows = []
    for sym in ticker_syms:
        try:
            info        = yf.Ticker(sym).info or {}
            trailing_pe = info.get('trailingPE')
            forward_pe  = info.get('forwardPE')
            t_eps       = info.get('trailingEps') or info.get('trailingEPS')
            f_eps       = info.get('forwardEps')  or info.get('forwardEPS')
            eps_growth  = ((f_eps / t_eps - 1) * 100
                           if t_eps and f_eps and t_eps != 0 else None)
            rows.append({
                'Ticker':     sym,
                '종목명':     info.get('shortName') or info.get('longName') or '',
                'Trailing PE': trailing_pe,
                'Forward PE':  forward_pe,
                'Trailing EPS': t_eps,
                'Forward EPS':  f_eps,
                'EPS 상승률(%)': eps_growth,
            })
        except Exception:
            rows.append({'Ticker': sym, '종목명': '', 'Trailing PE': None,
                         'Forward PE': None, 'Trailing EPS': None,
                         'Forward EPS': None, 'EPS 상승률(%)': None})
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    return df[['Ticker', '종목명', 'Trailing PE', 'Forward PE',
               'Trailing EPS', 'Forward EPS', 'EPS 상승률(%)']]


# ======================================================
# ====== 성과 테이블 / 차트 공통 함수 =================
# ======================================================
def get_perf_table_improved(label2ticker, ref_date=None):
    tickers = list(label2ticker.values())
    if ref_date is None:
        ref_date = datetime.now().date()
    start = ref_date - timedelta(days=4 * 365)
    end   = ref_date + timedelta(days=1)
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if isinstance(df, pd.Series): df = df.to_frame()
        df = df.ffill().dropna(how='all')[tickers]
    except Exception:
        return pd.DataFrame()
    if df.empty: return pd.DataFrame()

    avail      = df.index[df.index.date <= ref_date]
    if len(avail) == 0: return pd.DataFrame()
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
            results.append(row); continue
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
                    ci = series.index.get_loc(last_idx); lb = cfg['days']
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
    if ref_date is None: ref_date = datetime.now().date()
    try:
        data  = yf.download(list(label2ticker.values())[0],
                            start=ref_date - timedelta(days=4*365),
                            end=ref_date + timedelta(days=1), progress=False)['Close'].dropna()
        avail = data.index[data.index.date <= ref_date]
        if len(avail) == 0: return None, None, None
        last_trade = avail[-1].date()
        ci         = data.index.get_loc(avail[-1])
        actual     = {}
        for p, d in {'1D':1,'1W':5,'1M':21,'3M':63,'6M':126,'1Y':252,'3Y':756}.items():
            actual[p] = data.index[ci-d].date().strftime('%Y-%m-%d') if ci >= d else "데이터 부족"
        for key, dt in [('MTD', last_trade.replace(day=1)), ('YTD', last_trade.replace(month=1, day=1))]:
            d = data[data.index.date >= dt]
            if len(d): actual[key] = d.index[0].date().strftime('%Y-%m-%d')
        return list(label2ticker.keys())[0], last_trade.strftime('%Y-%m-%d'), actual
    except Exception:
        return None, None, None


@st.cache_data(show_spinner="차트 데이터 로딩 중...")
def get_normalized_prices(label2ticker, months=6):
    tickers = list(label2ticker.values())
    end     = datetime.now().date()
    start   = end - timedelta(days=months * 31)
    df      = yf.download(tickers, start=start, end=end+timedelta(days=1), progress=False)['Close']
    if isinstance(df, pd.Series): df = df.to_frame()
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
        v = float(val) if isinstance(val, (int, float)) else float(str(val).replace('%','').strip())
    except: return ""
    return "color: red;" if v > 0 else ("color: blue;" if v < 0 else "")

def style_perf_table(df, perf_cols):
    styled = df.style
    for col in perf_cols:
        if col in df.columns:
            styled = styled.format({col: format_percentage}).applymap(colorize_return, subset=[col])
    return styled


# ======================================================
# ====== 페이지 함수 ===================================
# ======================================================
period_options = {"3개월": 3, "6개월": 6, "12개월": 12, "24개월": 24, "36개월": 36}

def _render_chart(label2t, session_key, select_key):
    if session_key not in st.session_state:
        st.session_state[session_key] = 6
    months = st.selectbox(
        "기간 선택", options=list(period_options.keys()),
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


# ─────────────────────────────────────────────────────
# Page 1: 시장 성과
# ─────────────────────────────────────────────────────
def show_page1():
    st.title("🌐 Market Performance")
    update_clicked = st.button("🔄 Update", type="primary", key="p1_update")

    if update_clicked:
        st.session_state['p1_updated'] = True

    if not st.session_state.get('p1_updated', False):
        st.info("'🔄 Update' 버튼을 눌러 데이터를 불러오세요.")
        return

    # 성과 테이블
    perf_cols = ['1D(%)', '1W(%)', 'MTD(%)', '1M(%)', '3M(%)', '6M(%)', 'YTD(%)', '1Y(%)', '3Y(%)']
    for title, label2t, h in [
        ("📊 Equity",   STOCK_ETFS,  490),
        ("🗠 Bond",   BOND_ETFS,   385),
        ("💱 Currency",       CURRENCY,    315),
        ("📈 Crypto",   CRYPTO,      385),
        ("📕 Style ETF", STYLE_ETFS,  245),
        ("📘 Sector ETF",   SECTOR_ETFS, 420),
    ]:
        st.subheader(title)
        with st.spinner(f"{title} 계산 중..."):
            perf = get_perf_table_improved(label2t)
        if not perf.empty:
            st.dataframe(style_perf_table(perf.set_index('자산명'), perf_cols),
                         use_container_width=True, height=h)

    st.markdown("---")

    # 차트 탭
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 주가지수", "📗 섹터", "📙 스타일", "📋 기준일 정보"]
    )
    with tab1:
        st.subheader("✅ Performance")
        _render_chart(STOCK_ETFS, "idx_months", "idx_sel")
    with tab2:
        st.subheader("☑️ 섹터 ETF 수익률")
        _render_chart(SECTOR_ETFS, "sec_months", "sec_sel")
    with tab3:
        st.subheader("☑️ 스타일 ETF 수익률")
        _render_chart(STYLE_ETFS, "sty_months", "sty_sel")
    with tab4:
        st.subheader("📋 계산 기준일")
        lbl, last_d, adates = get_sample_calculation_dates(STOCK_ETFS)
        if lbl and adates:
            st.caption(f"**샘플 자산:** {lbl} | **최근 거래일:** {last_d}")
            l1 = [f"{p}: {adates[p]}" for p in ['1D','1W','MTD','1M'] if p in adates]
            st.caption("• " + " | ".join(l1))
            l2 = [f"{p}: {adates[p]}" for p in ['3M','6M','YTD','1Y','3Y'] if p in adates]
            st.caption("• " + " | ".join(l2))


# ─────────────────────────────────────────────────────
# Page 2: LLM 분석
# ─────────────────────────────────────────────────────
def show_page2():
    st.title("🤖 LLM 분석 — 뉴스 감성 분석")
    st.caption("Yahoo Finance RSS에서 수집한 최근 3일 뉴스를 FinBERT로 감성 분석합니다.")

    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        selected = st.selectbox("섹터 선택", list(SECTOR_ETFS.keys()), key="p2_sector")
    with col2:
        run_btn = st.button("📡 분석 시작", type="primary", use_container_width=True, key="p2_run")

    etf_ticker = SECTOR_ETFS[selected]
    cache_key  = f'llm_{etf_ticker}'

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
    df        = build_sentiment_df(news_list)
    if df.empty:
        st.warning("분석 결과가 없습니다.")
        return

    # ── 요약 메트릭 ─────────────────────────────────────
    avg_s = df['Sentiment'].mean()
    pos   = (df['Sentiment_Category'] == 'Positive').sum()
    neg   = (df['Sentiment_Category'] == 'Negative').sum()
    neu   = (df['Sentiment_Category'] == 'Neutral').sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",  len(df))
    c2.metric("🟢 Positive",  int(pos))
    c3.metric("🔴 Negative",  int(neg))
    c4.metric("Average", f"{avg_s:+.3f}")

    st.markdown("---")

    # ── 차트 1: 종목별 평균 감정 가로 막대 ──────────────
    render_sentiment_bar_chart(df, selected)

    # ── 차트 2: 히스토그램 + 카운트플롯 ────────────────
    col_h, col_c = st.columns(2)
    with col_h:
        st.plotly_chart(create_sentiment_histogram(df), use_container_width=True)
    with col_c:
        st.plotly_chart(create_sentiment_countplot(df), use_container_width=True)

    # ── 차트 3: 박스플롯 ────────────────────────────────
    st.plotly_chart(create_sentiment_boxplot(df), use_container_width=True)

    st.markdown("---")

    # ── 전체 뉴스 테이블 (긍정/부정/중립 모두 포함) ─────
    st.markdown(f"##### 📋 관련 뉴스 전체 목록 ({len(df)}건)")
    render_news_table(df)


# ─────────────────────────────────────────────────────
# Page 3: 애널리스트 & 밸류에이션
# ─────────────────────────────────────────────────────
def show_page3():
    st.title("👨‍💼 애널리스트 & 밸류에이션")
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
    cache_key  = f'analyst_{etf_ticker}'

    if run_btn:
        progress = st.progress(0, text="Holdings 수집 중...")
        holdings = ETFCollector().get_etf_holdings(etf_ticker)
        if not holdings:
            st.error(f"❌ {selected}: Holdings 수집 실패")
            return
        ticker_syms = [h['ticker'] for h in holdings]
        progress.progress(15, text=f"✅ {len(ticker_syms)}개 종목 — 애널리스트 데이터 수집 중...")

        analyst_df   = get_analyst_report_data(ticker_syms)
        progress.progress(55, text="✅ 애널리스트 수집 완료 — 밸류에이션 수집 중...")
        valuation_df = get_valuation_eps_table(ticker_syms)
        progress.progress(100, text="✅ 조회 완료!")
        time.sleep(0.4)
        progress.empty()

        st.session_state[cache_key] = {
            'analyst':   analyst_df,
            'valuation': valuation_df,
            'holdings':  holdings,
        }

    if cache_key not in st.session_state:
        st.info("섹터를 선택하고 '🔍 조회' 버튼을 누르세요.")
        return

    data       = st.session_state[cache_key]
    analyst_df = data['analyst']
    val_df     = data['valuation']
    holdings   = data['holdings']

    # ── Holdings 정보 ────────────────────────────────────
    st.subheader(f"📦 {selected} — Top Holdings")
    holdings_df = pd.DataFrame(holdings)
    st.dataframe(holdings_df, use_container_width=True,
                 height=min(400, 40 + len(holdings_df) * 35))

    st.markdown("---")

    # ── 애널리스트 등급 테이블 ────────────────────────────
    st.subheader("👨‍💼 애널리스트 등급 & 목표주가")
    analyst_sorted = analyst_df.sort_values('상승여력(%)', ascending=False, na_position='last')

    def color_upside(val):
        try:
            v = float(val)
            if v > 10:  return 'color: #2ecc71; font-weight:bold'
            if v < 0:   return 'color: #e74c3c; font-weight:bold'
            return ''
        except: return ''

    def color_rating(val):
        try:
            v = float(val)
            if v <= 2:   return 'color: #2ecc71; font-weight:bold'
            if v >= 4:   return 'color: #e74c3c; font-weight:bold'
            return 'color: #f39c12'
        except: return ''

    fmt = {'등급 점수': '{:.2f}', '목표주가': '{:,.2f}', '현재가': '{:,.2f}', '상승여력(%)': '{:.1f}%'}
    styled_a = (analyst_sorted.style
                .format(fmt, na_rep='N/A')
                .applymap(color_upside,  subset=['상승여력(%)'])
                .applymap(color_rating,  subset=['등급 점수'])
                .background_gradient(subset=['상승여력(%)'], cmap='RdYlGn', vmin=-20, vmax=40))
    st.dataframe(styled_a, use_container_width=True,
                 height=min(500, 40 + len(analyst_sorted) * 35))

    # ── 상승여력 막대 차트 ───────────────────────────────
    if not analyst_sorted['상승여력(%)'].isna().all():
        fig_up = go.Figure()
        df_plot = analyst_sorted.dropna(subset=['상승여력(%)'])
        fig_up.add_trace(go.Bar(
            x=df_plot['Ticker'],
            y=df_plot['상승여력(%)'],
            marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in df_plot['상승여력(%)']],
            text=[f"{v:.1f}%" for v in df_plot['상승여력(%)']],
            textposition='outside',
        ))
        fig_up.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
        fig_up.update_layout(
            title='종목별 애널리스트 목표주가 상승여력',
            xaxis_title='Ticker', yaxis_title='상승여력 (%)',
            template='plotly_dark', height=380,
        )
        st.plotly_chart(fig_up, use_container_width=True)

    st.markdown("---")

    # ── 밸류에이션 & EPS 테이블 ──────────────────────────
    st.subheader("🔍 밸류에이션 & EPS")
    val_sorted = val_df.sort_values('EPS 상승률(%)', ascending=False, na_position='last')
    fmt_v = {'Trailing PE': '{:.1f}', 'Forward PE': '{:.1f}',
             'Trailing EPS': '{:.2f}', 'Forward EPS': '{:.2f}', 'EPS 상승률(%)': '{:.1f}%'}

    def color_eps(val):
        try:
            v = float(val)
            if v > 5:  return 'color: #2ecc71; font-weight:bold'
            if v < 0:  return 'color: #e74c3c; font-weight:bold'
            return ''
        except: return ''

    styled_v = (val_sorted.style
                .format(fmt_v, na_rep='N/A')
                .applymap(color_eps, subset=['EPS 상승률(%)'])
                .background_gradient(subset=['EPS 상승률(%)'], cmap='RdYlGn'))
    st.dataframe(styled_v, use_container_width=True,
                 height=min(500, 40 + len(val_sorted) * 35))

    # ── PE 비교 차트 ─────────────────────────────────────
    pe_df = val_sorted.dropna(subset=['Trailing PE', 'Forward PE'])
    if not pe_df.empty:
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Bar(
            x=pe_df['Ticker'], y=pe_df['Trailing PE'],
            name='Trailing PE', marker_color='rgba(65,105,225,0.8)',
        ))
        fig_pe.add_trace(go.Bar(
            x=pe_df['Ticker'], y=pe_df['Forward PE'],
            name='Forward PE', marker_color='rgba(235,0,140,0.7)',
        ))
        fig_pe.update_layout(
            title='Trailing PE vs Forward PE',
            xaxis_title='Ticker', yaxis_title='PE Ratio',
            barmode='group', template='plotly_dark', height=380,
        )
        st.plotly_chart(fig_pe, use_container_width=True)


# ======================================================
# ====== 사이드바 네비게이션 & 메인 라우팅 =============
# ======================================================
with st.sidebar:
    # KB자산운용 로고로 변경
    # 공식 KB자산운용 로고 URL을 사용하거나, 아래의 댓글 처리된 부분을 수정하세요
    try:
        kb_logo_url = "https://img.inhr.co.kr/static/careerlink/DSGN/250310110803368lsi.svg"  # KB자산운용 공식 로고 URL
        resp = requests.get(kb_logo_url, timeout=4, verify=False)
        if resp.status_code == 200:
            st.image(Image.open(BytesIO(resp.content)), use_container_width=True)
    except Exception:
        # 로고 로드 실패 시 텍스트만 표시
        pass

    st.title("💡 Global Market")
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

if page == "📊 시장 성과":
    show_page1()
elif page == "🤖 LLM 분석":
    show_page2()
elif page == "👨‍💼 애널리스트":
    show_page3()
