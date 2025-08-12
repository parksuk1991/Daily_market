import streamlit as st
from dashboard_utils import SECTOR_ETFS, get_top_holdings, get_news_for_ticker, get_news_sentiment_data, create_sentiment_histogram, create_sentiment_boxplot, create_sentiment_countplot

def show_sector_headlines():
    st.subheader("📰 섹터별 주요 종목 헤드라인")
    for label, etf in SECTOR_ETFS.items():
        top_holdings = get_top_holdings(etf, n=3)
        if top_holdings:
            sector_name = label.split()[0] + " 섹터"
            holding_names = [name for _, name in top_holdings]
            holding_syms = [sym for sym, _ in top_holdings]
            st.write(f"#### {sector_name} 주요 종목: {', '.join(holding_names)}")
            for sym, name in top_holdings:
                news = get_news_for_ticker(sym, limit=1)
                if news:
                    art = news[0]
                    st.markdown(f"- **[{sym}]** {art['date']}: {art['title']}")
                else:
                    st.write(f"- [{sym}] 뉴스 없음")
        else:
            st.write(f"- {label}: 보유종목 정보 없음")
    st.markdown("---")
    st.subheader("✳️✴️ 주요 종목 뉴스 감정 분석")
    with st.spinner("뉴스 데이터 수집 및 감정 분석 중..."):
        df, ticker_syms = get_news_sentiment_data()
    if df.empty:
        st.warning("뉴스 데이터를 가져올 수 없습니다.")
        return
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 뉴스 개수", len(df))
    with col2:
        st.metric("평균 감정 점수", f"{df['Sentiment'].mean():.3f}")
    with col3:
        positive_pct = (df['Sentiment_Category'] == 'Positive').sum() / len(df) * 100
        st.metric("긍정 비율", f"{positive_pct:.1f}%")
    with col4:
        negative_pct = (df['Sentiment_Category'] == 'Negative').sum() / len(df) * 100
        st.metric("부정 비율", f"{negative_pct:.1f}%")
    st.subheader("감정 점수 및 카테고리 분포")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig1 = create_sentiment_histogram(df)
        st.plotly_chart(fig1, use_container_width=True)
    with chart_col2:
        fig3 = create_sentiment_countplot(df)
        st.plotly_chart(fig3, use_container_width=True)
    st.subheader("종목별 감정 점수")
    fig2 = create_sentiment_boxplot(df)
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("상세 뉴스 데이터 보기"):
        st.dataframe(
            df[['Ticker', 'Date', 'Headline', 'Sentiment', 'Sentiment_Category']].sort_values('Date', ascending=False),
            use_container_width=True
        )