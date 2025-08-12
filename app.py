import streamlit as st
from main_dashboard import show_main_dashboard
from sector_headlines import show_sector_headlines
from analyst_opinion import show_analyst_opinion

st.set_page_config(page_title="Global Market Monitoring", page_icon="🌐", layout="wide")

col_title, col_img_credit = st.columns([9, 1])
with col_title:
    st.title("🌐 Global Market Monitoring")
    update_clicked = st.button("Update", type="primary", use_container_width=False, key="main_update_btn")
with col_img_credit:
    image_url = "https://amateurphotographer.com/wp-content/uploads/sites/7/2017/08/Screen-Shot-2017-08-23-at-22.29.18.png?w=600.jpg"
    try:
        import requests
        from PIL import Image
        from io import BytesIO
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        st.image(img, width=150, caption=None)
    except Exception:
        st.info("이미지를 불러올 수 없습니다.")
    st.markdown(
        '<div style="text-align: left; margin-bottom: 3px; font-size:0.9rem;">'
        'Data 출처: <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>'
        '</div>',
        unsafe_allow_html=True
    )

st.sidebar.title("대시보드 구분")
section = st.sidebar.radio(
    "원하는 섹션을 선택하세요.",
    ["시장 성과", "섹터별 헤드라인&뉴스", "애널리스트 의견"]
)

if update_clicked:
    st.session_state['updated'] = True

if st.session_state.get('updated', False):
    if section == "시장 성과":
        show_main_dashboard()
    elif section == "섹터별 헤드라인&뉴스":
        show_sector_headlines()
    elif section == "애널리스트 의견":
        show_analyst_opinion()
else:
    st.info("상단 'Update' 버튼을 눌러주세요.")