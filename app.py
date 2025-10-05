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
st.markdown('<p class="sub-header">Anonymous Feedback System for Campus Improvement</p>', unsafe_allow_html=True)

# STUDENT VIEW
if mode == "Student View":
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Submit Your Feedback")
        
        with st.form("feedback_form", clear_on_submit=True):
            name = st.text_input("Your Name (Optional)", placeholder="Anonymous")
            category = st.selectbox("Issue Category", 
                                    ["Classroom", "Infrastructure", "Food", "Library", 
                                     "Hostel", "Transportation", "Faculty", "Other"])
            feedback_text = st.text_area("Your Feedback", 
                                         placeholder="Describe your experience...", 
                                         height=180)
            
            submitted = st.form_submit_button("Submit Feedback", type="primary", use_container_width=True)
            
            if submitted:
                if feedback_text.strip():
                    with st.spinner("Analyzing sentiment..."):
                        result = sentiment_analyzer(feedback_text[:512])[0]
                        
                        label = result['label']
                        confidence = result['score']
                        
                        if label == 'POSITIVE':
                            sentiment = "Positive"
                        else:
                            sentiment = "Negative"
                        
                        if confidence < 0.70:
                            sentiment = "Neutral"
                    
                    feedback_entry = {
                        'name': name if name else "Anonymous",
                        'category': category,
                        'feedback': feedback_text,
                        'sentiment': sentiment,
                        'confidence': round(confidence * 100, 1),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if save_data_to_sheets(feedback_entry):
                        st.session_state.feedback_data.append(feedback_entry)
                        st.success("Thank you! Your feedback has been recorded.")
                    else:
                        st.error("Failed to save feedback. Please try again.")
                else:
                    st.warning("Please enter some feedback before submitting!")
    
    with col2:
        st.subheader("Community Feedback Overview")
        
        if len(st.session_state.feedback_data) > 0:
            df = pd.DataFrame(st.session_state.feedback_data)
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
            
            # Overall sentiment breakdown
            sentiment_counts = df['sentiment'].value_counts()
            
            st.markdown("### Sentiment Summary")
            sent_col1, sent_col2, sent_col3 = st.columns(3)
            
            with sent_col1:
                st.metric("Positive", sentiment_counts.get('Positive', 0), 
                         label_visibility="visible")
            with sent_col2:
                st.metric("Neutral", sentiment_counts.get('Neutral', 0))
            with sent_col3:
                st.metric("Negative", sentiment_counts.get('Negative', 0))
            
            st.markdown("---")
            
            # Category breakdown
            st.markdown("### Issues by Category")
            category_counts = df['category'].value_counts().head(5)
            
            for cat, count in category_counts.items():
                percentage = (count / len(df)) * 100
                st.markdown(f"**{cat}:** {count} feedbacks ({percentage:.1f}%)")
            
            st.markdown("---")
            
            # Simple pie chart
            colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
            color_map = [colors[sent] for sent in sentiment_counts.index]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.3,
                marker=dict(colors=color_map),
                textinfo='label+percent'
            )])
            fig_pie.update_layout(
                title="Overall Sentiment",
                showlegend=False,
                height=280,
                margin=dict(t=40, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        else:
            st.info("No feedback data yet. Be the first to share!")

st.markdown("---")
st.caption("Powered by Hugging Face DistilBERT | Data stored in Google Sheets")

