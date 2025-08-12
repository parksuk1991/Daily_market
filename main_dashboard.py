import streamlit as st
from dashboard_utils import (
    STOCK_ETFS, BOND_ETFS, CURRENCY, CRYPTO, STYLE_ETFS, SECTOR_ETFS,
    get_perf_table_improved, style_perf_table, get_sample_calculation_dates, get_normalized_prices
)

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
    norm_df = get_normalized_prices(etf_dict, months=months_val)
    fig = st.empty()
    fig = st.container()
    import plotly.graph_objects as go
    fig_obj = go.Figure()
    for col in norm_df.columns:
        fig_obj.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines', name=col))
    fig_obj.update_layout(
        yaxis_title="100 기준 누적수익률(%)",
        template="plotly_dark", height=500, legend=dict(orientation='h')
    )
    st.plotly_chart(fig_obj, use_container_width=True)

def show_all_performance_tables():
    perf_cols = ['1D(%)','1W(%)','MTD(%)','1M(%)','3M(%)','6M(%)','YTD(%)','1Y(%)','3Y(%)']
    st.subheader("📊 주식시장")
    stock_perf = get_perf_table_improved(STOCK_ETFS)
    if not stock_perf.empty:
        st.dataframe(
            style_perf_table(stock_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=490
        )
    st.subheader("🗠 채권시장")
    bond_perf = get_perf_table_improved(BOND_ETFS)
    if not bond_perf.empty:
        st.dataframe(
            style_perf_table(bond_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=385
        )
    st.subheader("💱 통화")
    curr_perf = get_perf_table_improved(CURRENCY)
    if not curr_perf.empty:
        st.dataframe(
            style_perf_table(curr_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=315
        )
    st.subheader("📈 암호화폐")
    crypto_perf = get_perf_table_improved(CRYPTO)
    if not crypto_perf.empty:
        st.dataframe(
            style_perf_table(crypto_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=385
        )
    st.subheader("📕 스타일 ETF")
    style_perf = get_perf_table_improved(STYLE_ETFS)
    if not style_perf.empty:
        st.dataframe(
            style_perf_table(style_perf.set_index('자산명'), perf_cols),
            use_container_width=True, height=245
        )
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

def show_main_dashboard():
    show_all_performance_tables()
    render_normalized_chart("✅ 주요 주가지수 수익률", STOCK_ETFS, "idx", 6)
    render_normalized_chart("☑️ 섹터 ETF 수익률", SECTOR_ETFS, "sector", 6)
    render_normalized_chart("☑️ 스타일 ETF 수익률", STYLE_ETFS, "style", 6)