import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# logger

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Session state initialization

if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False

# Folder setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

log("Application started")
log(f"RAW_DIR: {RAW_DIR}")
log(f"CLEAN_DIR: {CLEAN_DIR}")

# page config

st.set_page_config("End-to-End SVM", layout="wide")
st.title("End-to-End SVM Platform")

# Sidebar : model settings

st.sidebar.header("Model Settings")
kernel = st.sidebar.selectbox("Select Kernel", ["linear", "poly", "rbf", "sigmoid"])
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox('Gamma', ['scale', 'auto'])

log(f"SVM Settings ---> Kernel= {kernel}, C: {C}, Gamma= {gamma}")

# step 1: Data Ingestion

st.header("Step 1: Data Ingestion")
log("Step 1: Data Ingestion")
option = st.radio("choose Data Source",["Download Dataset","Upload Dataset"])
df=None
raw_path=None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)

        raw_path = os.path.join(RAW_DIR, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)

        
        df = pd.read_csv(raw_path)
        st.success("Dataset Downloaded Successfully!")
        log(f"Iris dataset saved at {raw_path}")

if option == "upload csv":
    upload_file = st.file_uploader("Upload CSV file", type=["csv"])
    if upload_file:
        raw_path = os.path.join(RAW_DIR, upload_file.name)
        with open(raw_path, "wb") as f:
            f.write(upload_file.getbuffer())
        
        df = pd.read_csv(raw_path)
        st.success("Dataset Uploaded Successfully!")
        log(f"Uploaded dataset saved at {raw_path}")

# step 2: EDA

if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log("Step 2 started: EDA")

    st.dataframe(df.head())
    st.write("Shape",df.shape)
    st.write("Missing Values",df.isnull().sum())

    fig,ax=plt.subplots()
    sns.heatmap(df.corr(numeric_only=True),annot=True, cmap="coolwarm",ax=ax)
    st.pyplot(fig)

    log("EDA Completed")

    # step 3: Data Cleaning

    if df is not None:
        st.header("Step 3: Data Cleaning")
        
        strategy = st.selectbox("Missing values strategy", ["Mean","Median","Drop Rows"])
        df_clean = df.copy()

        if strategy == "Mean":
            df_clean = df_clean.dropna()

        else:
            for col in df_clean.select_dtypes(include=[np.number]).columns:
                if strategy == "Mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == "Median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        st.session_state.df_clean = df_clean
        st.success("Data Cleaning Completed")
    
    else:
        st.info("please complete step 1 (Data Ingestion) first... ")

# step 4: save cleaned data

if st.button("Save Cleaned Data"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data to save. Please complete Step 3 first...")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_data_{timestamp}.csv"
        clean_path = os.path.join(CLEAN_DIR, clean_filename)
        st.session_state.df_clean.to_csv(clean_path, index=False)

        st.success(" Cleaned Dataset saved ")
        st.info(f"Saved at: {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")

# step 5: Load cleaned dataset

st.header("Step 5: Load Cleaned Dataset")
clean_files = os.listdir(CLEAN_DIR)

if not clean_files:
    st.warning("No cleaned datasets found. Please save one in Step 4...")
    log("No cleaned datasets available")
else:
    selected = st.selectbox("Select Cleaned Dataset", clean_files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))

    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded cleaned dataset: {selected}")

    st.dataframe(df_model.head())


# step 6: Train SVM
st.header("Step 6: Train SVM")
log("Step 6 started: Train SVM")
target=st.selectbox("Select Target Column",df_model.columns)

y=df_model[target]

if y.dtype=="object":
    y=LabelEncoder().fit_transform(y)
    log("Target column encoded")

# Select numeric features only

x=df_model.drop(columns=[target])
x=x.select_dtypes(include=[np.number])

if x.empty:
    st.error("No numeric features available for the training..")
    st.stop()

#Scale features
scaler=StandardScaler()
x=scaler.fit_transform(x)

# Train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#svm
model=SVC(kernel=kernel,C=C,gamma=gamma)
model.fit(x_train,y_train)
#Evaluate
y_pred=model.predict(x_test)
acc=accuracy_score(y_test,y_pred)
st.success(f"Accuracy:{acc:.2f}")
log(f"SVM Trained Successfully|Accuracy:{acc:.2f}")
cm=confusion_matrix(y_test,y_pred)
fig,ax=plt.subplots()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
st.pyplot(fig)