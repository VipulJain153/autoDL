import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import pandas as pd,os
from sklearn.metrics import mean_squared_error,precision_score
import pickle as pk,joblib
from sklearn.datasets import fetch_california_housing
from process import preprocess
from sklearn.model_selection import cross_val_score
from VBBPS import modelSimplexRegression,modelComplexRegression,modelsimplexClassification,modelComplexClassification
import numpy as np
california_housing = fetch_california_housing(as_frame=True)
path = "source_data.csv"
with st.sidebar:
    st.header("Navigation")
    radios = ["Model Configurations", "Data Analysis and Preprocessing", "Train Best Model"]
    radio = st.radio("Select Process",radios)
if 'task' not in st.session_state:
    st.session_state.task = "Select Task"
if radio==radios[0]:
    st.header("Vipul Automatic Machine Learning")

    tasks = ["Select Task", "Regression","Regression VBBPS", "Classification", "Classification VBBPS"]
    st.session_state.task = st.selectbox("Type",tasks)
    data = st.file_uploader("Upload Data")

    if data!=None:
        df = pd.read_csv(data)
        st.dataframe(df)
        df.to_csv(path,index=False)
task=st.session_state.task
if os.path.exists(path) and task!="Select Task":    
    
    df = pd.read_csv(path)
    # df = california_housing.data
    # print(df)

    if radio==radios[1]:
        if task!="Clustering" and task!="Association":
            labels = st.multiselect("Target Varible(s)", list(df.columns))
            if labels:
                    with open("labels.pk","wb") as f:
                        pk.dump(labels,f)
        # st_profile_report(profile_report.ProfileReport(df))

    elif radio==radios[2]:
        if os.path.exists("labels.pk"):
            with open("labels.pk","rb") as f:
                target = pk.load(f)[0]
            X,y = preprocess(df,target)
            if task=="Regression":
                rnf=modelSimplexRegression
                rnf.fit(X,y)
                joblib.dump(rnf.best_estimator_, "model.joblib")
                st.download_button("Download Model", "model.joblib", file_name="model.joblib")
                msg =f'Model\'s MSE:{np.sqrt(-rnf.best_score_)}'
                st.success(body=msg,icon='😍')
            elif task=="Regression VBBPS":
                rnf=modelComplexRegression
                rnf.fit(X,y)
                joblib.dump(rnf, "model.joblib")
                st.download_button("Download Model", "model.joblib", file_name="model.joblib")
                msg =f'Model\'s MSE:{np.sqrt(mean_squared_error(rnf.predict(X),y))}'
                st.success(body=msg,icon='😍')
            elif task=="Classification":
                rnf = modelsimplexClassification
                rnf.fit(X,y)
                joblib.dump(rnf.best_estimator_, "model.joblib")
                st.download_button("Download Model", "model.joblib", file_name="model.joblib")
                msg =f'Model\'s Precision:{rnf.best_score_*100}%'
                st.success(body=msg,icon='😍')
            elif task=="Classification VBBPS":
                rnf = modelComplexClassification
                rnf.fit(X,y)
                joblib.dump(rnf, "model.joblib")
                st.download_button("Download Model", "model.joblib", file_name="model.joblib")
                msg =f'Model\'s Precision:{precision_score(rnf.predict(X),y)*100}%'
                st.success(body=msg,icon='😍')

else:
    st.warning("Please Define Task and DataFrame")