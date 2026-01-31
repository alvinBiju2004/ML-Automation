import streamlit as st

main_page = st.Page(
    page="views/main.py",
    title="🏠 Home",
    default=True
)

more_details_page = st.Page(
    page="views/more_details.py",
    title="📋 Data Details"
)

visual_page = st.Page(
    page="views/visual.py",
    title="📊 Visualization"
)

st.sidebar.markdown("## 🤖 AutoML Trainer")
st.sidebar.caption("Train ML models from any dataset")
st.sidebar.divider()

pg = st.navigation(
    pages=[main_page, more_details_page, visual_page]
)

pg.run()
