import streamlit as st
import pandas as pd
from utils.state import store_df


st.title("Know more about your dataset")

# Check if raw_df exists in session_state
if "raw_df" in st.session_state:
    raw_df = st.session_state.raw_df
    st.write("Data loaded from session_state:")
    st.dataframe(raw_df)

    if "df_shape" not in st.session_state:
        st.session_state.df_shape = raw_df.shape
    if "df_head" not in st.session_state:
        st.session_state.df_head = raw_df.head()
    if "df_tail" not in st.session_state:
        st.session_state.df_tail = raw_df.tail()
    if "df_dtypes" not in st.session_state:
        st.session_state.df_dtypes = raw_df.dtypes
    if "df_columns" not in st.session_state:
        st.session_state.df_columns = raw_df.columns.tolist()
    if "df_missing_values" not in st.session_state:
        st.session_state.df_missing_values = raw_df.isnull().sum()

    for var in ["df_shape", "df_head", "df_tail", "df_dtypes", "df_columns", "df_missing_values"]:
        if f"show_{var}" not in st.session_state:
            st.session_state[f"show_{var}"] = False
    
    def toggle_button(var_name, display_name=None):
        display_name = display_name or var_name
        state_key = f"show_{var_name}"

        label = f"Hide {display_name}" if st.session_state[state_key] else f"Show {display_name}"

        if st.button(label, key=f"btn_{var_name}"):
            st.session_state[state_key] = not st.session_state[state_key]

        if st.session_state[state_key]:
            st.write(st.session_state[var_name])

    toggle_button("df_shape", "DataFrame Shape")
    toggle_button("df_head", "DataFrame Head")
    toggle_button("df_tail", "DataFrame Tail")
    toggle_button("df_dtypes", "DataFrame Data Types")
    toggle_button("df_columns", "DataFrame Columns")
    toggle_button("df_missing_values", "DataFrame Missing Values")
    
    
    
else:
    st.write("No dataframe found. Please load it on the first page.")


