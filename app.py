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

# 데이터 로딩 및 session_state 관리
if "updated" not in st.session_state:
    st.session_state["updated"] = False

if update_clicked:
    st.session_state["updated"] = True
    # 각 섹션별 데이터 초기화 및 로드
    st.session_state["main_data_loaded"] = False
    st.session_state["sector_data_loaded"] = False
    st.session_state["analyst_data_loaded"] = False

# 각 섹션별로 데이터 로드 플래그와 실제 데이터
if st.session_state["updated"]:
    if section == "시장 성과":
        if not st.session_state.get("main_data_loaded", False):
            show_main_dashboard()
            st.session_state["main_data_loaded"] = True
        else:
            # 이미 데이터가 로드되었으니 재실행하지 않고 그대로 보여줌
            show_main_dashboard()
    elif section == "섹터별 헤드라인&뉴스":
        if not st.session_state.get("sector_data_loaded", False):
            show_sector_headlines()
            st.session_state["sector_data_loaded"] = True
        else:
            show_sector_headlines()
    elif section == "애널리스트 의견":
        if not st.session_state.get("analyst_data_loaded", False):
            show_analyst_opinion()
            st.session_state["analyst_data_loaded"] = True
        else:
            show_analyst_opinion()
else:
    st.info("상단 'Update' 버튼을 눌러주세요.")
