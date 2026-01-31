import streamlit as st
import numpy as np

# MISSING VALUES HANDLING
def fill_missing_values(df):
    for column in df.columns:
        if df[column].isnull().sum()>0:
            if df[column].dtypes=='O':
                df[column].fillna(df[column].mode()[0],inplace=True)
            else:
                if df[column].nunique()<10:
                    df[column].fillna(df[column].mode()[0],inplace=True)
                else:
                    df[column].fillna(df[column].mean(),inplace=True)
    return df


# LABEL ENCODING
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encode_column(df, columns):
    # Safety check
    if not isinstance(df, pd.DataFrame):
        raise TypeError("label_encode_column expects a pandas DataFrame")

    df = df.copy()

    for col in columns:
        if col in df.columns and df[col].dtype == "O":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df

# ONE HOT ENCODING + LABEL ENCODING BASED ON UNIQUE VALUES | DEFAULT ENCODING
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
def default_encode(x,columns_to_encode):
    flag=0
    for i in columns_to_encode:
        if len(x[i].unique())>15:
            flag=1
    if flag==1:
        lab=LabelEncoder()
        for i in columns_to_encode:
            x[i]=lab.fit_transform(x[i])
        st.write("Applied Label Encoding as one or more columns have more than 15 unique values")
        return x
    else:
        col=make_column_transformer((OneHotEncoder(handle_unknown='ignore'),columns_to_encode),remainder='passthrough')
        x=col.fit_transform(x)
        return x
    
# SPLITTING X AND Y
def split_x_y_classification(df):
    x=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    return x,y

def split_x_y_regression(df):
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    return x,y
    
# SPLITTING TRAINING AND TESTING DATA
from sklearn.model_selection import train_test_split
def split_train_test(x,y,test_size=0.3):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size)
    return x_train,x_test,y_train,y_test

# Normalization
from sklearn.preprocessing import StandardScaler
def normalize_data(x_train,x_test):
    scaler=StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    return x_train,x_test

# KNN
from sklearn.neighbors import KNeighborsClassifier
def create_knn_model(x_train,y_train,n_neighbors=7):
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train,y_train)
    return knn

# Naive Bayes
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
def create_naive_bayes_model(x_train,y_train,model_type="Gaussian"):
    if np.any(x_train<0):
        model=GaussianNB()
    elif len(np.unique(x_train)) == 2:
        model=BernoulliNB()
    else:
        model=MultinomialNB()
    model.fit(x_train,y_train)
    return model

# SVM
from sklearn.svm import SVC
def create_svm_model(x_train,y_train):
    svm=SVC()
    svm.fit(x_train,y_train)
    return svm

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
def create_linear_regression_model(x_train,y_train):
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    return lr

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
def create_polynomial_regression_model(x):
    poly=PolynomialFeatures(degree=2)
    x=poly.fit_transform(x)

# Confusin metrix
from sklearn.metrics import confusion_matrix
def confusion_metric(y_test,predictions):
    if st.session_state.flag==0:
        cm=confusion_matrix(y_test,predictions)
        return cm

# Confusin matrix display
from sklearn.metrics import ConfusionMatrixDisplay
def confusin_display(y_train,cm):
    lab=list(np.unique(y_train))
    cmd=ConfusionMatrixDisplay(cm,display_labels=lab)
    return cmd

# Accuracy Score
from sklearn.metrics import accuracy_score
def accuracy(y_test,predictions):
    score=accuracy_score(y_test,predictions)
    return score

# MAE
from sklearn.metrics import mean_absolute_error
def meanAbsoluteError(y_test,predictions):
    mae=mean_absolute_error(y_test,predictions)
    return mae

# MAPE
from sklearn.metrics import mean_absolute_percentage_error
def meanAbsolutePercentageError(y_test,predictions):
    mape=mean_absolute_percentage_error(y_test,predictions)
    return mape

# MSE
from sklearn.metrics import mean_squared_error
def meanSquaredError(y_test,predictions):
    mse=mean_squared_error(y_test,predictions)
    return mse

# RMSE
from sklearn.metrics import root_mean_squared_error
def rootMeanSquaredError(y_test,predictions):
    rmse=mean_absolute_error(y_test,predictions)
    return rmse

# r2
from sklearn.metrics import r2_score
def r_score(y_test,predictions):
    r2=r2_score(y_test,predictions)
    return r2

