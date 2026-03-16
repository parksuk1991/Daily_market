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

st.set_page_config(page_title="Global Market Monitoring", page_icon="🌐", layout="wide")

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
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# ===== 기간 옵션 (3년 기본값, 6개월~3년만 선택 가능) =====
PERIOD_OPTIONS = [("6개월", 6), ("1년", 12), ("2년", 24), ("3년", 36)]
DEFAULT_PERIOD = "3년"

# ======================================================
# ====== ETF Collector ======================================================
class ETFCollector:
    def __init__(self):
        self.cf_session = None
    def _try_ssga(self, ticker: str):
        try:
            url = f"https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
            resp = requests.get(url, headers=_HEADERS, timeout=15, verify=False)
            if resp.status_code != 200: return []
            df = pd.read_excel(BytesIO(resp.content), skiprows=4, engine='openpyxl')
            if df.empty: return []
            return [{'ticker': 'TEST', 'name': 'Test', 'weight': 1.0}]
        except: return []

    def get_etf_holdings(self, ticker: str, retry: int = 2):
        return [{'ticker': ticker, 'name': f'{ticker} Holding', 'weight': 10.0}]

class NewsCollector:
    def __init__(self, days: int = 3):
        self.days = days
        self.cutoff_date = datetime.now() - timedelta(days=days)
    def collect_all(self, holdings: list, etf_ticker: str) -> list:
        return []

class FinBERTAnalyzer:
    def __init__(self): pass
    def batch_analyze(self, news_list: list) -> list:
        return []

@st.cache_resource
def load_analyzer():
    return FinBERTAnalyzer()

# ======================================================
# ====== 성과 분석 함수 ===============================
# ======================================================
def format_percentage(val):
    """소수점 2자리 포맷"""
    if pd.isna(val): return "N/A"
    try: 
        return f"{float(val):.2f}%"
    except: return "N/A"

def get_perf_table_improved(label2ticker, ref_date=None):
    """성과 테이블 계산"""
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
    if '현재값' in df_r.columns:
        df_r['현재값'] = df_r['현재값'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
    
    return df_r

def style_perf_table_with_heatmap(df, perf_cols):
    """히트맵을 적용한 성과 테이블"""
    styled = df.style
    
    # 소수점 2자리로 포맷
    for col in perf_cols:
        if col in df.columns:
            styled = styled.format({col: format_percentage})
    
    # 각 컬럼별로 히트맵 적용
    for col in perf_cols:
        if col in df.columns:
            try:
                # 숫자값 추출
                numeric_vals = pd.to_numeric(
                    df[col].astype(str).str.replace('%', '').str.strip(), 
                    errors='coerce'
                )
                # 유효한 값만 필터링
                valid_vals = numeric_vals[numeric_vals.notna()]
                if len(valid_vals) > 0:
                    vmin = valid_vals.min()
                    vmax = valid_vals.max()
                    # RdYlGn 히트맵 적용 (빨강-노랑-초록)
                    styled = styled.background_gradient(
                        subset=[col], 
                        cmap='RdYlGn', 
                        vmin=vmin, 
                        vmax=vmax
                    )
            except:
                pass
    
    return styled

# ======================================================
# ====== 차트 함수 ===============================
# ======================================================
def plot_monthly_returns(prices_df, asset_name):
    """월간 수익률"""
    monthly = prices_df.resample('M').last()
    returns = monthly.pct_change().dropna() * 100
    colors = ['#FFBC00' if x > 0 else '#e74c3c' for x in returns.iloc[:, 0]]
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
        height=350,
        showlegend=False,
        margin=dict(t=30, b=20, l=30, r=20),
    )
    return fig

def plot_distribution_monthly(prices_df, asset_name):
    """월간 분포"""
    monthly = prices_df.resample('M').last()
    returns = monthly.pct_change().dropna() * 100
    returns_flat = returns.iloc[:, 0].values
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns_flat,
        nbinsx=20,
        marker_color='rgba(255, 188, 0, 0.7)',
        opacity=0.8,
    ))
    fig.update_layout(
        title=f'{asset_name}',
        xaxis_title='Monthly Return (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=350,
        showlegend=False,
        margin=dict(t=30, b=20, l=30, r=20),
    )
    return fig

def plot_rolling_volatility(prices_df, asset_name, window=126):
    """롤링 변동성"""
    returns = prices_df.pct_change().dropna()
    rolling_vol = returns.iloc[:, 0].rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values,
        mode='lines',
        line=dict(color='#2ecc71', width=2),
        fill='tozeroy',
    ))
    fig.update_layout(
        title=f'{asset_name}',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        template='plotly_white',
        height=350,
        margin=dict(t=30, b=20, l=30, r=20),
    )
    return fig

def plot_rolling_sharpe(prices_df, asset_name, window=126, risk_free_rate=0.02):
    """롤링 샤프"""
    returns = prices_df.pct_change().dropna()
    rolling_mean = returns.iloc[:, 0].rolling(window).mean() * 252
    rolling_std = returns.iloc[:, 0].rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.3)
    fig.update_layout(
        title=f'{asset_name}',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        template='plotly_white',
        height=350,
        margin=dict(t=30, b=20, l=30, r=20),
    )
    return fig

# ======================================================
# ====== 페이지 함수 ===================================
# ======================================================
def show_page1():
    st.title("🌐 Market Performance")
    update_clicked = st.button("🔄 Update", type="primary", key="p1_update")

    if update_clicked:
        st.session_state['p1_updated'] = True

    if not st.session_state.get('p1_updated', False):
        st.info("🔄 Update 버튼을 눌러 데이터를 불러오세요.")
        return

    perf_cols = ['1D(%)', '1W(%)', 'MTD(%)', '1M(%)', '3M(%)', '6M(%)', 'YTD(%)', '1Y(%)', '3Y(%)']

    for title, label2t, h in [
        ("📊 Equity", STOCK_ETFS, 490),
        ("🗠 Bond", BOND_ETFS, 385),
        ("💱 Currency", CURRENCY, 315),
        ("📈 Crypto", CRYPTO, 385),
        ("📕 Style ETF", STYLE_ETFS, 245),
        ("📘 Sector ETF", SECTOR_ETFS, 420),
    ]:
        st.subheader(title)
        with st.spinner(f"{title} 계산 중..."):
            perf = get_perf_table_improved(label2t)
        if not perf.empty:
            st.dataframe(
                style_perf_table_with_heatmap(perf.set_index('자산명'), perf_cols),
                use_container_width=True,
                height=h
            )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 주가지수", "📗 섹터", "📙 스타일", "📋 기준일 정보"])
    
    with tab1:
        st.subheader("✅ Stock Indices - Comprehensive Analysis")
        render_chart_with_all_assets(STOCK_ETFS)
    
    with tab2:
        st.subheader("☑️ Sector ETF - Comprehensive Analysis")
        render_chart_with_all_assets(SECTOR_ETFS)
    
    with tab3:
        st.subheader("☑️ Style ETF - Comprehensive Analysis")
        render_chart_with_all_assets(STYLE_ETFS)
    
    with tab4:
        st.subheader("📋 계산 기준일")
        st.write("기준일 정보")

def render_chart_with_all_assets(label2t):
    """모든 자산의 차트를 2개씩 배치하여 표시"""
    session_key = f"period_{id(label2t)}"
    select_key = f"select_{id(label2t)}"
    
    # 기간 선택 - 기본값은 "3년"
    period_idx = 3  # "3년"은 PERIOD_OPTIONS[3]
    selected_period = st.selectbox(
        "기간 선택",
        options=[p[0] for p in PERIOD_OPTIONS],
        index=period_idx,
        key=select_key
    )
    
    # 선택한 기간에 해당하는 월 수 찾기
    months = next(m for n, m in PERIOD_OPTIONS if n == selected_period)
    
    with st.spinner("데이터 분석 중..."):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 31)
        
        try:
            # 모든 자산 데이터 다운로드
            tickers = list(label2t.values())
            prices_raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            if isinstance(prices_raw, pd.DataFrame):
                prices_data = prices_raw['Close']
            else:
                prices_data = prices_raw.to_frame()
            
            if isinstance(prices_data, pd.Series):
                prices_data = prices_data.to_frame()
            
            prices_data.columns = list(label2t.keys())
            
        except Exception as e:
            st.error(f"데이터 다운로드 실패: {str(e)}")
            return
        
        # 1️⃣ 누적 수익률 (모든 자산)
        st.subheader("📈 Cumulative Returns")
        norm = prices_data / prices_data.iloc[0] * 100
        fig = go.Figure()
        colors_list = ['#FFBC00', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                      '#74B9FF', '#A29BFE', '#81ECEC', '#55EFC4', '#FD79A8', '#FDCB6E']
        for i, col in enumerate(norm.columns):
            fig.add_trace(go.Scatter(
                x=norm.index,
                y=norm[col],
                mode='lines',
                name=col,
                line=dict(color=colors_list[i % len(colors_list)], width=2),
            ))
        fig.update_layout(
            yaxis_title="100 기준 누적수익률(%)",
            template="plotly_white",
            height=400,
            legend=dict(orientation='h', y=1.15),
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 자산 목록
        assets = list(label2t.keys())
        
        # 2️⃣ 월간 수익률 - 2개씩 배치
        st.subheader("📊 Monthly Returns")
        for i in range(0, len(assets), 2):
            cols = st.columns(2)
            
            # 첫 번째
            asset1 = assets[i]
            asset1_data = prices_data[[asset1]]
            try:
                with cols[0]:
                    fig = plot_monthly_returns(asset1_data, asset1)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                cols[0].error(f"{asset1} 실패")
            
            # 두 번째
            if i + 1 < len(assets):
                asset2 = assets[i + 1]
                asset2_data = prices_data[[asset2]]
                try:
                    with cols[1]:
                        fig = plot_monthly_returns(asset2_data, asset2)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[1].error(f"{asset2} 실패")
        
        # 3️⃣ 월간 분포 - 2개씩 배치
        st.subheader("📉 Distribution of Monthly Returns")
        for i in range(0, len(assets), 2):
            cols = st.columns(2)
            
            # 첫 번째
            asset1 = assets[i]
            asset1_data = prices_data[[asset1]]
            try:
                with cols[0]:
                    fig = plot_distribution_monthly(asset1_data, asset1)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                cols[0].error(f"{asset1} 실패")
            
            # 두 번째
            if i + 1 < len(assets):
                asset2 = assets[i + 1]
                asset2_data = prices_data[[asset2]]
                try:
                    with cols[1]:
                        fig = plot_distribution_monthly(asset2_data, asset2)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[1].error(f"{asset2} 실패")
        
        # 4️⃣ 롤링 변동성 - 2개씩 배치
        st.subheader("📈 Rolling Volatility (6-Month)")
        for i in range(0, len(assets), 2):
            cols = st.columns(2)
            
            # 첫 번째
            asset1 = assets[i]
            asset1_data = prices_data[[asset1]]
            try:
                with cols[0]:
                    fig = plot_rolling_volatility(asset1_data, asset1)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                cols[0].error(f"{asset1} 실패")
            
            # 두 번째
            if i + 1 < len(assets):
                asset2 = assets[i + 1]
                asset2_data = prices_data[[asset2]]
                try:
                    with cols[1]:
                        fig = plot_rolling_volatility(asset2_data, asset2)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[1].error(f"{asset2} 실패")
        
        # 5️⃣ 롤링 샤프 - 2개씩 배치
        st.subheader("⭐ Rolling Sharpe Ratio (6-Month)")
        for i in range(0, len(assets), 2):
            cols = st.columns(2)
            
            # 첫 번째
            asset1 = assets[i]
            asset1_data = prices_data[[asset1]]
            try:
                with cols[0]:
                    fig = plot_rolling_sharpe(asset1_data, asset1)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                cols[0].error(f"{asset1} 실패")
            
            # 두 번째
            if i + 1 < len(assets):
                asset2 = assets[i + 1]
                asset2_data = prices_data[[asset2]]
                try:
                    with cols[1]:
                        fig = plot_rolling_sharpe(asset2_data, asset2)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[1].error(f"{asset2} 실패")

def show_page2():
    st.title("🤖 LLM 분석 — 뉴스 감성 분석")
    st.info("준비 중입니다.")

def show_page3():
    st.title("👨‍💼 애널리스트 & 밸류에이션")
    st.info("준비 중입니다.")

# ======================================================
# ====== 메인 라우팅 ===================================
# ======================================================
with st.sidebar:
    st.title("💡 Global Market")
    st.markdown("---")
    page = st.radio("페이지 선택",
        ["📊 시장 성과", "🤖 LLM 분석", "👨‍💼 애널리스트"], key="nav_page",)
    st.markdown("---")
    st.caption(f"Last visit: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.divider()
    st.caption(f"© 2026 KB Asset Management.")

if page == "📊 시장 성과":
    show_page1()
elif page == "🤖 LLM 분석":
    show_page2()
elif page == "👨‍💼 애널리스트":
    show_page3()
