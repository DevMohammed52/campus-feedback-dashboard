import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from transformers import pipeline
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Page configuration
st.set_page_config(
    page_title="Campus Feedback Sentiment Dashboard", 
    page_icon="📝", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load sentiment model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

sentiment_analyzer = load_sentiment_model()

# Google Sheets connection
@st.cache_resource
def init_google_sheets():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets["spreadsheet_key"]).sheet1
    return sheet

# Load data from Google Sheets
def load_data_from_sheets():
    try:
        sheet = init_google_sheets()
        data = sheet.get_all_records()
        return data if data else []
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []

# Save data to Google Sheets
def save_data_to_sheets(feedback_entry):
    try:
        sheet = init_google_sheets()
        row = [
            feedback_entry['name'],
            feedback_entry['category'],
            feedback_entry['feedback'],
            feedback_entry['sentiment'],
            str(feedback_entry['confidence']),
            feedback_entry['timestamp']
        ]
        sheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")
        return False

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = load_data_from_sheets()

if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Sidebar
with st.sidebar:
    st.title("Navigation")
    
    if st.session_state.admin_authenticated:
        default_index = 1
    else:
        default_index = 0
    
    mode = st.radio("Select View", ["Student View", "Admin View"], index=default_index)
    
    if mode == "Admin View" and not st.session_state.admin_authenticated:
        st.markdown("---")
        admin_pass = st.text_input("Admin Password", type="password", key="sidebar_password")
        if st.button("Login"):
            if admin_pass == "admin123":
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password!")
    elif mode == "Admin View" and st.session_state.admin_authenticated:
        st.success("Logged in as Admin")
        if st.button("Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()

# Title
st.markdown('<p class="main-header">Campus Feedback Sentiment Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Real-Time Sentiment Analysis for Campus Issues</p>', unsafe_allow_html=True)

# STUDENT VIEW
if mode == "Student View":
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Submit Your Feedback")
        st
