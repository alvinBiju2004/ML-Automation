import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import confusin_display,confusion_metric

st.title("📊 Visual Analytics")
st.caption("Explore relationships and model behavior")


if "raw_df" in st.session_state:
    raw_df = st.session_state.raw_df
    st.write("Data loaded from session_state:")
    st.dataframe(raw_df)

    # Determine if the file is for classification or regression
    "fileType" in st.session_state
    fileType = st.session_state.fileType
    
    for i in raw_df:
        if raw_df[i].dtypes=='O':
            if len(raw_df[i].unique())<8:
                st.write(f"Column '{i}' is Categorical")
                val_count=raw_df[i].value_counts()

                fig,ax=plt.subplots()
                ax.bar(val_count.index,val_count.values,color='r')
                ax.set_xlabel("types")
                ax.set_ylabel("count")
                ax.set_title(f"{i} {"count"}")
                st.pyplot(fig)
            else:
                st.write(f"there are {len(raw_df[i])} varieties of {raw_df[i]}, hence cant be vizualized through a graph")

    if fileType == "Regression":
        x = raw_df.iloc[:, :-1]
        y = raw_df.iloc[:, -1]

        for i in x.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots()
            sns.regplot(
                x=x[i],
                y=y,
                ax=ax,
                scatter_kws={"color": "r"},
                line_kws={"color": "blue"}
            )

            ax.set_xlabel(i)
            ax.set_ylabel(y.name)
            ax.set_title(f"{i} vs {y.name}")

            st.pyplot(fig)

    if st.session_state.flag==0:
        if fileType=="Classification":
            cm=confusion_metric(st.session_state.y_test,st.session_state.predictions)
            st.write(f"The confusion matrix is {cm}")

            cmd=confusin_display(st.session_state.y_train,cm)
            cmd.plot()
            st.pyplot(cmd.figure_)
            

else:
    st.write("No dataframe found. Please load it on the first page.")