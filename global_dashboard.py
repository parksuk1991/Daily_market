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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

# ── 불용어 (Wordcloud 제외 단어) ─────────────────────────
_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
    'do', 'does', 'did', 'not', 'no', 'it', 'its', 'this', 'that', 'these',
    'those', 'as', 'into', 'up', 'out', 'about', 'said', 'says', 'also',
    'he', 'she', 'they', 'we', 'you', 'i', 'their', 'his', 'her', 'our',
    'after', 'over', 'more', 'new', 'than', 'which', 'who', 'what', 'when',
    'how', 'all', 'been', 'between', 'through', 'while', 'during', 'before',
    'company', 'year', 'quarter', 'first', 'second', 'third', 'fourth',
    'percent', 'billion', 'million', 'trillion', 'based', 'reported',
}


# ======================================================
# ====== ETF Collector (완전 동적 — 4가지 방법 체인) ======
# ======================================================
class ETFCollector:
    """
    ETF Top-10 Holdings 수집기.
    수집 순서: SSGA XLSX → stockanalysis.com → yahooquery → yfinance
    hardcoding 없음.
    """

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
            ticker_col = next((c for c in df.columns if 'ticker' in c.lower() or 'symbol' in c.lower()), None)
            weight_col = next((c for c in df.columns if 'weight' in c.lower()), None)
            name_col   = next((c for c in df.columns if 'name'   in c.lower()), ticker_col)
            if not ticker_col or not weight_col:
                return []
            df = df.dropna(subset=[ticker_col])
            df[ticker_col] = df[ticker_col].astype(str).str.strip()
            df = df[df[ticker_col].str.len() > 0]
            df = df[~df[ticker_col].str.lower().isin(['nan', 'ticker', 'symbol', '-'])]
            df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
            df = df.dropna(subset=[weight_col])
            df = df[df[weight_col] > 0].sort_values(weight_col, ascending=False).head(10)
            return [
                {'ticker': str(r[ticker_col]).strip(), 'name': str(r[name_col]).strip(),
                 'weight': round(float(r[weight_col]), 2)}
                for _, r in df.iterrows() if str(r[ticker_col]).strip()
            ]
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
                return [
                    {'ticker': h.get('symbol', ''), 'name': h.get('holdingName', h.get('symbol', '')),
                     'weight': round(h.get('holdingPercent', 0.0) * 100, 2)}
                    for h in raw[:10] if h.get('symbol', '')
                ]
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


# ====== News Collector (Relevance Filter 포함) ======
class NewsCollector:
    """
    Yahoo RSS 기반 뉴스 수집.
    관련성 필터: 뉴스 제목 또는 본문에 ticker 심볼 혹은 기업명이 포함된 경우만 수집.
    """

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
            texts = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 30]
            full  = ' '.join(texts)
            return full[:5000] if full else ""
        except Exception:
            return ""

    def is_valid_content(self, content: str):
        if not content or len(content) < 200:
            return False
        lower = content.lower()
        return not any(w in lower[:500] for w in ['sign in', 'log in', 'subscribe', 'register'])

    def is_relevant(self, title: str, content: str, ticker: str, company_name: str) -> bool:
        """
        뉴스가 해당 종목과 관련 있는지 판단.
        제목 또는 본문 앞 1000자에 ticker 심볼 또는 기업명(첫 단어 이상)이 포함되어야 함.
        """
        combined = (title + ' ' + content[:1000]).lower()
        # ticker 심볼 (예: AAPL) — 단어 경계 매칭
        ticker_clean = ticker.replace('-', '').lower()
        if re.search(r'\b' + re.escape(ticker_clean) + r'\b', combined):
            return True
        # 기업명: 공백 기준 첫 단어 또는 전체명
        name_lower = company_name.lower().strip()
        if name_lower and len(name_lower) >= 3:
            # 전체 기업명
            if name_lower in combined:
                return True
            # 첫 단어 (2글자 초과인 경우)
            first_word = name_lower.split()[0]
            if len(first_word) > 3 and re.search(r'\b' + re.escape(first_word) + r'\b', combined):
                return True
        return False

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

    def collect_yahoo_rss(self, ticker: str, company_name: str):
        """Yahoo Finance RSS — 관련성 필터 적용, 건수 제한 없음"""
        try:
            feed = feedparser.parse(f"https://finance.yahoo.com/rss/headline?s={ticker}")
            news = []
            for entry in feed.entries:          # 전체 항목 처리
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
                    summary_rss = entry.get('summary', '')
                    content     = self.extract_content(article_url)

                    if not self.is_valid_content(content):
                        continue

                    # ── 관련성 필터 ──────────────────────────────
                    if not self.is_relevant(title, content, ticker, company_name):
                        continue

                    news.append({
                        'ticker':       ticker,
                        'title':        title,
                        'url':          article_url,
                        'published_at': date_str,
                        'summary':      summary_rss[:300],
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
        news = self.collect_yahoo_rss(ticker, company)
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
            text  = re.sub(r'<[^>]+>', '', text)
            text  = re.sub(r'http\S+', '', text)
            text  = re.sub(r'\s+', ' ', text).strip()
            res   = self.pipe(text[:512])[0]
            label, score = res['label'], res['score']
            return score if label == 'positive' else (-score if label == 'negative' else 0.0)
        except Exception:
            return 0.0

    def analyze_text(self, text: str):
        if not text or len(text) < 100:
            return 0.0
        chunks = [text[i:i+1000] for i in range(0, min(len(text), 3000), 1000) if len(text[i:i+1000]) > 100]
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


# ====== 시각화 헬퍼 ======
def render_sentiment_chart(news_list: list, sector_name: str):
    """종목별 평균 감정 점수 가로 막대 차트"""
    if not news_list:
        return

    # 종목별 집계
    ticker_data: dict = {}
    for n in news_list:
        t = n.get('ticker', '')
        s = n.get('sentiment_score', 0)
        if t not in ticker_data:
            ticker_data[t] = {'scores': [], 'count': 0, 'company': n.get('company_name', t)}
        ticker_data[t]['scores'].append(s)
        ticker_data[t]['count'] += 1

    rows = []
    for t, v in ticker_data.items():
        avg = np.mean(v['scores']) if v['scores'] else 0
        rows.append({'Ticker': t, 'Company': v['company'], 'Avg Sentiment': round(avg, 4), 'News Count': v['count']})

    df = pd.DataFrame(rows).sort_values('Avg Sentiment', ascending=True)

    # 색상: 양수=초록, 음수=빨강, 중립=회색
    colors = []
    for val in df['Avg Sentiment']:
        if val > 0.05:
            colors.append('#2ecc71')
        elif val < -0.05:
            colors.append('#e74c3c')
        else:
            colors.append('#95a5a6')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Avg Sentiment'],
        y=df['Ticker'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f} ({r['News Count']}건)" for v, r in zip(df['Avg Sentiment'], df.to_dict('records'))],
        textposition='outside',
        customdata=df['Company'],
        hovertemplate='<b>%{y}</b> (%{customdata})<br>Sentiment: %{x:.4f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_width=1, line_dash='dash', line_color='white', opacity=0.4)
    fig.update_layout(
        title=dict(text=f"📊 {sector_name} — 종목별 평균 감정 점수", font=dict(size=14)),
        xaxis=dict(title="Sentiment Score", range=[-1, 1], zeroline=True,
                   tickvals=[-1, -0.5, 0, 0.5, 1]),
        yaxis=dict(title="", tickfont=dict(size=11)),
        template="plotly_dark",
        height=max(250, len(df) * 42),
        margin=dict(l=70, r=120, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_wordcloud(news_list: list, sector_name: str):
    """뉴스 제목 + 본문 기반 Wordcloud"""
    if not news_list:
        return

    text = ' '.join(
        n.get('title', '') + ' ' + n.get('highlights', '')
        for n in news_list
    )
    # 특수문자 제거
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = [w for w in text.lower().split() if len(w) > 2 and w not in _STOPWORDS]
    if not words:
        return

    wc_text = ' '.join(words)
    try:
        wc = WordCloud(
            width=900, height=380,
            background_color='#0e1117',
            colormap='RdYlGn',
            max_words=80,
            max_font_size=90,
            min_font_size=10,
            prefer_horizontal=0.85,
            collocations=False,
        ).generate(wc_text)

        fig, ax = plt.subplots(figsize=(9, 3.8), facecolor='#0e1117')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"☁️ {sector_name} News Word Cloud",
                     color='white', fontsize=13, pad=8)
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception:
        pass


def render_news_table(news_list: list):
    """뉴스 제목 + 헤드라인 + 감정 전체 표시 (건수 제한 없음)"""
    if not news_list:
        st.info("관련 뉴스가 없습니다.")
        return

    rows = []
    for n in news_list:
        sentiment = n.get('sentiment_score', 0)
        if sentiment > 0.05:
            emoji = "🟢"
        elif sentiment < -0.05:
            emoji = "🔴"
        else:
            emoji = "⚪"
        rows.append({
            '📅 Date':      n.get('published_at', ''),
            '🏷 Ticker':    n.get('ticker', ''),
            '📰 Title':     n.get('title', ''),
            '🔍 Headline':  n.get('highlights', '')[:200] + ('...' if len(n.get('highlights', '')) > 200 else ''),
            f'{emoji} Sent': round(sentiment, 3),
            '🗂 Category':  n.get('category', ''),
            '🔗 URL':       n.get('url', ''),
        })

    df = pd.DataFrame(rows)

    # 색상 스타일 함수
    def color_sentiment(val):
        try:
            v = float(str(val).replace('🟢','').replace('🔴','').replace('⚪','').strip())
        except Exception:
            return ''
        if v > 0.05:   return 'color: #2ecc71; font-weight: bold'
        if v < -0.05:  return 'color: #e74c3c; font-weight: bold'
        return 'color: #95a5a6'

    styled = df.style.applymap(color_sentiment, subset=[c for c in df.columns if 'Sent' in c])
    st.dataframe(styled, use_container_width=True, height=min(600, 38 + len(rows) * 35))


# ====== 섹터 분석 ======
@st.cache_resource
def load_analyzer():
    return FinBERTAnalyzer()


def run_sector_etf_analysis(etf_ticker: str, etf_name: str):
    try:
        holdings = ETFCollector().get_etf_holdings(etf_ticker)
        if not holdings:
            return None, None, f"❌ {etf_name}: Holdings 수집 실패"

        all_news = NewsCollector(days=3).collect_all(holdings, etf_ticker)
        if not all_news:
            return holdings, None, f"⚠️ {etf_name}: Holdings {len(holdings)}개 확인, 관련 뉴스 없음"

        analyzed = load_analyzer().batch_analyze(all_news)
        return analyzed, holdings, None
    except Exception as e:
        return None, None, f"❌ {etf_name}: {str(e)[:80]}"


def show_sector_analysis():
    st.subheader("📰 섹터별 주요 종목 뉴스 & 감정 분석")

    sector_results   = {}
    sector_holdings  = {}

    for sector_name, etf_ticker in SECTOR_ETFS.items():
        try:
            with st.spinner(f"{sector_name} 분석 중..."):
                analyzed_news, holdings, error = run_sector_etf_analysis(etf_ticker, sector_name)
                if error:
                    st.warning(error)
                else:
                    if analyzed_news:
                        sector_results[sector_name]  = analyzed_news
                    if holdings:
                        sector_holdings[sector_name] = holdings
        except Exception:
            st.warning(f"❌ {sector_name}: 오류 발생")

    if not sector_results:
        st.warning("섹터 분석 데이터를 가져올 수 없습니다.")
        return

    for sector_name, news_list in sector_results.items():
        with st.expander(f"#### {sector_name}  ({len(news_list)}건 관련 뉴스)", expanded=False):

            # ── 요약 메트릭 ─────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            avg_s = np.mean([n.get('sentiment_score', 0) for n in news_list]) if news_list else 0
            pos   = sum(1 for n in news_list if n.get('sentiment_score', 0) >  0.05)
            neg   = sum(1 for n in news_list if n.get('sentiment_score', 0) < -0.05)
            neu   = len(news_list) - pos - neg
            c1.metric("총 뉴스",    len(news_list))
            c2.metric("🟢 긍정",    pos)
            c3.metric("🔴 부정",    neg)
            c4.metric("평균 감정",  f"{avg_s:+.3f}")

            st.markdown("---")

            # ── 종목별 감정 차트 ────────────────────────
            render_sentiment_chart(news_list, sector_name)

            # ── 워드 클라우드 ───────────────────────────
            render_wordcloud(news_list, sector_name)

            st.markdown("---")

            # ── 뉴스 전체 테이블 (제한 없음) ───────────
            st.markdown("##### 📋 관련 뉴스 전체 목록")
            render_news_table(news_list)


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


# ---- 주요 데이터 함수 ----
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
            except Exception:
                row[pk] = np.nan
        results.append(row)

    df_r = pd.DataFrame(results)
    if '현재값' in df_r.columns:
        df_r['현재값'] = df_r['현재값'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
    return df_r


def get_sample_calculation_dates(label2ticker, ref_date=None):
    if ref_date is None: ref_date = datetime.now().date()
    sample_ticker = list(label2ticker.values())[0]
    sample_label  = list(label2ticker.keys())[0]
    try:
        data  = yf.download(sample_ticker,
                            start=ref_date - timedelta(days=4*365),
                            end=ref_date + timedelta(days=1),
                            progress=False)['Close'].dropna()
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
        return sample_label, last_trade.strftime('%Y-%m-%d'), actual
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
            l1 = [f"{p}: {actual_dates[p]}" for p in ['1D','1W','MTD','1M'] if p in actual_dates]
            st.caption("• " + " | ".join(l1))
            l2 = [f"{p}: {actual_dates[p]}" for p in ['3M','6M','YTD','1Y','3Y'] if p in actual_dates]
            st.caption("• " + " | ".join(l2))

else:
    st.info("상단 'Update' 버튼을 눌러주세요.")
