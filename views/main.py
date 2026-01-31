from cProfile import label
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from preprocessing import fill_missing_values
from preprocessing import split_x_y_classification,split_x_y_regression
from preprocessing import label_encode_column,default_encode
from preprocessing import split_train_test
from preprocessing import normalize_data
from preprocessing import create_knn_model
from preprocessing import create_naive_bayes_model
from preprocessing import create_svm_model
from preprocessing import create_linear_regression_model
from preprocessing import accuracy
from preprocessing import meanAbsoluteError,meanAbsolutePercentageError
from preprocessing import meanSquaredError,rootMeanSquaredError,r_score
from preprocessing import create_polynomial_regression_model

st.title("🏠 Dataset Upload & Training")
st.caption("Upload your data and train models automatically")


with st.expander("Supported Models"):
    st.write("""
    • KNN  
    • Naive Bayes  
    • SVM  
    • Simple Linear Regression  
    • Multiple Linear Regression  
    • Polynomial Regression
    """)

flag=1
if "flag" not in st.session_state:
    st.session_state.flag=flag

if "df" not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("upload a file")

if uploaded_file:
    st.success("Dataset uploaded successfully ✅")

    st.divider()

if uploaded_file and st.session_state.df is None:
    st.session_state.df = pd.read_csv(uploaded_file)

if st.session_state.df is not None:
    st.dataframe(st.session_state.df)
    
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = st.session_state.df.copy(deep=True)

    

    # Determine if the file is for classification or regression

    dtypeOfLastColumn=st.session_state.df.iloc[:,-1].dtypes
    if dtypeOfLastColumn=='O' or len(st.session_state.df.iloc[:,-1].unique())<=2:
        fileType="Classification"
    else:
        fileType="Regression"
    st.write("The file type is:",fileType)

    if "fileType" not in st.session_state:
        st.session_state.fileType = fileType

    st.divider()

    #To drop columns

    x=st.session_state.df.iloc[:,:-1]
    y=st.session_state.df.iloc[:,-1]

    columnsToDrop=[]
    checkboxColumn=[]

    st.write("Select columns to drop")
    for i in x:
        checkboxColumn.append(st.checkbox(i))
    for j in range(len(checkboxColumn)):
        if checkboxColumn[j]==True:
            columnsToDrop.append(x.columns[j])                       
    st.write("Columns to drop:",columnsToDrop)

    if st.button("Drop Selected Columns"):


        st.session_state.df.drop(columnsToDrop,axis=1,inplace=True)
        st.dataframe(st.session_state.df)

    st.divider()
    
    #Handling missing values

    st.write("After filling missing values:")

      # initial dataframe
    st.session_state.df = fill_missing_values(st.session_state.df)
    st.dataframe(st.session_state.df)

    if "x" not in st.session_state or "y" not in st.session_state:
        st.session_state.x = x
        st.session_state.y = y
    st.session_state.x=st.session_state.df.iloc[:,:-1]
    st.session_state.y=st.session_state.df.iloc[:,-1]

    #Encoding 

    columns_to_encode=[]
    for i in st.session_state.x:
        if st.session_state.df[i].dtypes=='O':
            columns_to_encode.append(i)
    st.write("Columns to encode:",columns_to_encode)                    #columns name not showing
    st.session_state.x=default_encode(st.session_state.x,columns_to_encode)
    st.write("After encoding:")
    st.dataframe(st.session_state.x)
    st.caption("Encoded using One Hot Encoding")

    st.divider()

    #Splitting x and y

    if fileType=="regression":
        x,y=split_x_y_regression(st.session_state.df)

    if fileType=="Classification":
        st.session_state.x,st.session_state.y=split_x_y_classification(st.session_state.df)
    #else:
        #x,y=split_x_y_regression(df)

    st.write("x:",st.session_state.x) 
    st.write("y:",st.session_state.y)

    #Splitting training and testing data

    x_train,x_test,y_train,y_test=split_train_test(st.session_state.x,st.session_state.y,test_size=0.3)
    

    # Split only once
    if "x_train" not in st.session_state:
        (
            st.session_state.x_train,
            st.session_state.x_test,
            st.session_state.y_train,
            st.session_state.y_test,
        ) = train_test_split(st.session_state.x, st.session_state.y, test_size=0.3)

    # Initialize toggle states
    for var in ["x_train", "x_test", "y_train", "y_test"]:
        if f"show_{var}" not in st.session_state:
            st.session_state[f"show_{var}"] = False

    # Toggle button function
    def toggle_button(var_name, display_name=None):
        display_name = display_name or var_name
        state_key = f"show_{var_name}"

        label = f"Hide {display_name}" if st.session_state[state_key] else f"Show {display_name}"

        if st.button(label, key=f"btn_{var_name}"):
            st.session_state[state_key] = not st.session_state[state_key]

        if st.session_state[state_key]:
            st.write(st.session_state[var_name])

    # Buttons
    toggle_button("x_train", "X Train")
    toggle_button("x_test", "X Test")
    toggle_button("y_train", "Y Train")
    toggle_button("y_test", "Y Test")

    # Normalization 

    if fileType=="Classification":
        x_train, x_test = normalize_data(st.session_state.x_train, st.session_state.x_test)
        st.write("After normalization:")
        st.write("X Train:")
        st.write(x_train)
        st.write("X Test:")
        st.write(x_test)
    
    # Model Creation


    if fileType=="Classification":

        st.subheader("Choose Algorithm")

        if st.button("Train KNN"):
            model = create_knn_model(st.session_state.x_train, st.session_state.y_train)
            st.session_state.model = model
            st.success("KNN model trained")

        if st.button("Train Naive Bayes"):
            model= create_naive_bayes_model(
            st.session_state.x_train, st.session_state.y_train
            )
            st.session_state.model = model
            st.success("Naive Bayes trained")

        if st.button("Train SVM"):
            model = create_svm_model(st.session_state.x_train, st.session_state.y_train)
            st.session_state.model = model
            st.success("SVM model trained")

    else:

        if st.button("Train Linear Regression"):
            model = create_linear_regression_model(
            st.session_state.x_train, st.session_state.y_train
            )
            st.session_state.model = model
            st.success("Linear Regression model trained")

        if st.button("Train Polynomial Regression Model"):
            st.session_state.x1=create_polynomial_regression_model(st.session_state.x)
            st.session_state.x_train1,st.session_state.x_test1,st.session_state.y_train1,st.session_state.y_test1=split_train_test(st.session_state.x,st.session_state.y,test_size=0.3)
            model=create_linear_regression_model(
                st.session_state.x_train1, st.session_state.y_train1
            )
            st.session_state.model=model
            st.success("Polynomial Regression model trained")

            
    # Model Output

    if "model" in st.session_state:
        st.subheader("Model Predictions on Test Data")
        predictions = st.session_state.model.predict(st.session_state.x_test)
        st.session_state.predictions=predictions
        st.session_state.flag=0

        if "predictions" not in st.session_state:
            st.session_state.predictions = predictions 

        if fileType=="Classification":
            results_df = pd.DataFrame({
            "y_test": st.session_state.y_test,
            "predictions": st.session_state.predictions
            })

            score=accuracy(st.session_state.y_test,st.session_state.predictions)
            st.write(f"The accuracy score for this model is {score}")

        else:
            results_df = pd.DataFrame({
            "y_test": st.session_state.y_test,
            "predictions": st.session_state.predictions,
            "Difference": st.session_state.y_test - st.session_state.predictions
            })

            mae=meanAbsoluteError(st.session_state.y_test,st.session_state.predictions)
            mape=meanAbsolutePercentageError(st.session_state.y_test,st.session_state.predictions)
            mse=meanSquaredError(st.session_state.y_test,st.session_state.predictions)
            rmse=rootMeanSquaredError(st.session_state.y_test,st.session_state.predictions)
            r2=r_score(st.session_state.y_test,st.session_state.predictions)

            st.write(f"Mean absolute error is {mae}")
            st.write(f"Mean absolute percentage error is {mape}")
            st.write(f"Mean squared error is {mse}")
            st.write(f"Root mean squared error is {rmse}")
            st.write(f"r_2 score is {r2}")

            model_coef=st.session_state.model.coef_
            st.write(f"The model coefficient {model_coef}")

            model_intercept=st.session_state.model.intercept_
            st.write(f"The model intercept is {model_intercept}")

        st.write("Results DataFrame:")
        st.dataframe(results_df)

    



        