import streamlit as st
from dashboard_utils import get_news_sentiment_data, get_analyst_report_data, get_valuation_eps_table

def show_analyst_opinion():
    st.subheader("👨‍💼🔝 주요 종목 애널리스트 의견")
    st.caption("• 애널리스트 등급 점수: 1 = Strong Buy,  2 = Buy,  3 = Neutral,  4 = Sell,  5 = Strong Sell")
    st.caption("• 애널리스트 목표가: 최근 3~6개월 내의 애널리스트 리포트에서 제시된 목표가(Price Target)의 평균")
    with st.spinner("애널리스트 등급 데이터 로딩 중..."):
        df, ticker_syms = get_news_sentiment_data()
        analyst_df = get_analyst_report_data(ticker_syms)
    analyst_df_sorted = analyst_df.sort_values('상승여력', ascending=False, na_position='last')
    st.dataframe(
        analyst_df_sorted.style.format({
            '애널리스트 등급 점수': '{:.2f}',
            '애널리스트 목표가': '{:,.2f}',
            '현재가': '{:,.2f}',
            '상승여력': '{:.1f}%'
        }).background_gradient(subset=['상승여력'], cmap='Spectral'),
        use_container_width=True, height=min(900, 30 + 30*len(analyst_df))
    )
    st.subheader("🔍 주요 종목 밸류에이션 및 주당순이익 추이")
    st.caption("• 현재 = Trailing 12M,  선행 = Blended Forward 12M")
    with st.spinner("밸류에이션 및 EPS 데이터 로딩 중..."):
        valuation_df = get_valuation_eps_table(ticker_syms)
    valuation_df_sorted = valuation_df.sort_values('EPS 상승률', ascending=False, na_position='last')
    st.dataframe(
        valuation_df_sorted.style.format({
            '현재 PE': '{:.2f}',
            '선행 PE': '{:.2f}',
            '현재 EPS': '{:.2f}',
            '선행 EPS': '{:.2f}',
            'EPS 상승률': '{:.1f}%'
        }).background_gradient(subset=['EPS 상승률'], cmap='Spectral'),
        use_container_width=True, height=min(900, 30 + 30*len(valuation_df))
    )