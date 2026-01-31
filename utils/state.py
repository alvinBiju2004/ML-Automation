import streamlit as st

def store_df(df, key="df"):
    st.session_state[key] = df
