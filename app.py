import streamlit as st
from textblob import TextBlob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Campus Feedback Sentiment Dashboard", 
    page_icon="📝", 
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📝 Campus Feedback Sentiment Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Share your feedback about campus or class issues - we analyze the sentiment in real-time!</p>', unsafe_allow_html=True)

# Initialize session state to store feedback
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# Create two columns for layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📋 Submit Your Feedback")
    
    # Input fields
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your Name (Optional)", placeholder="Anonymous")
        category = st.selectbox("Issue Category", 
                                ["Classroom", "Infrastructure", "Food", "Library", 
                                 "Hostel", "Transportation", "Faculty", "Other"])
        feedback_text = st.text_area("Your Feedback", 
                                     placeholder="Share your thoughts here...", 
                                     height=150)
        
        # Submit button
        submitted = st.form_submit_button("Submit Feedback", type="primary", use_container_width=True)
        
        if submitted:
            if feedback_text.strip():
                # Analyze sentiment using TextBlob
                blob = TextBlob(feedback_text)
                polarity = blob.sentiment.polarity
                
                # Classify sentiment
                if polarity > 0.1:
                    sentiment = "Positive"
                    emoji = "😊"
                elif polarity < -0.1:
                    sentiment = "Negative"
                    emoji = "😞"
                else:
                    sentiment = "Neutral"
                    emoji = "😐"
                
                # Store feedback
                feedback_entry = {
                    'name': name if name else "Anonymous",
                    'category': category,
                    'feedback': feedback_text,
                    'sentiment': sentiment,
                    'polarity': round(polarity, 2),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.feedback_data.append(feedback_entry)
                st.success(f"✅ Feedback submitted! Sentiment: **{sentiment}** {emoji}")
            else:
                st.warning("⚠️ Please enter some feedback before submitting!")

with col2:
    st.subheader("📊 Live Sentiment Analysis")
    
    if st.session_state.feedback_data:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.feedback_data)
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        
        # Create three columns for metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Feedback", len(df))
        with metric_col2:
            st.metric("😊 Positive", sentiment_counts.get('Positive', 0))
        with metric_col3:
            st.metric("😐 Neutral", sentiment_counts.get('Neutral', 0))
        with metric_col4:
            st.metric("😞 Negative", sentiment_counts.get('Negative', 0))
        
        # Create visualizations in two columns
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Pie chart for sentiment distribution
            colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
            color_map = [colors[sent] for sent in sentiment_counts.index]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker=dict(colors=color_map),
                textinfo='label+percent',
                textfont_size=14
            )])
            fig_pie.update_layout(
                title="Sentiment Distribution",
                showlegend=True,
                height=300,
                margin=dict(t=40, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with viz_col2:
            # Bar chart for category-wise feedback count
            category_counts = df['category'].value_counts()
            fig_bar = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                labels={'x': 'Category', 'y': 'Count'},
                title="Feedback by Category",
                color=category_counts.values,
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(
                showlegend=False,
                height=300,
                margin=dict(t=40, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.info("📭 No feedback submitted yet. Be the first to share your thoughts!")

# Admin section - password protected to view detailed feedback
st.markdown("---")
with st.expander("🔐 Admin View - Detailed Feedback"):
    password = st.text_input("Enter Admin Password", type="password")
    
    if password == "admin123":  # Change this password!
        if st.session_state.feedback_data:
            df_admin = pd.DataFrame(st.session_state.feedback_data)
            
            # Show sentiment filter
            filter_sentiment = st.multiselect(
                "Filter by Sentiment",
                options=['Positive', 'Neutral', 'Negative'],
                default=['Positive', 'Neutral', 'Negative']
            )
            
            # Filter dataframe
            df_filtered = df_admin[df_admin['sentiment'].isin(filter_sentiment)]
            
            st.write(f"**Showing {len(df_filtered)} of {len(df_admin)} feedback entries**")
            
            # Display table with better formatting
            st.dataframe(
                df_filtered[['timestamp', 'name', 'category', 'sentiment', 'feedback']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No feedback data available yet.")
    elif password:
        st.error("❌ Incorrect password!")