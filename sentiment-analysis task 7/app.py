# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
nltk.download('vader_lexicon')

# Custom CSS with animations
def inject_custom_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Header animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .header-animation {
            animation: fadeIn 1s ease-out;
        }
        
        /* Card styling with hover effect */
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 20px;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Animated SVG wave footer */
        .footer {
            position: relative;
            height: 150px;
            width: 100%;
            background-color: #0e1117;
            margin-top: 50px;
        }
        
        .waves {
            position: absolute;
            top: -50px;
            left: 0;
            width: 100%;
            height: 50px;
            overflow: hidden;
        }
        
        .wave {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 200%;
            height: 100%;
            background-repeat: repeat no-repeat;
            background-position: 0 bottom;
            background-size: 50% 50px;
            animation: wave 10s linear infinite;
        }
        
        .wave1 {
            background-image: url('data:image/svg+xml;utf8,<svg viewBox="0 0 1200 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none"><path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" fill="%230e1117" opacity=".25"/></svg>');
        }
        
        .wave2 {
            background-image: url('data:image/svg+xml;utf8,<svg viewBox="0 0 1200 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none"><path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" fill="%230e1117" opacity=".5"/></svg>');
            animation-delay: -5s;
            animation-duration: 15s;
        }
        
        .wave3 {
            background-image: url('data:image/svg+xml;utf8,<svg viewBox="0 0 1200 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none"><path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" fill="%230e1117"/></svg>');
            animation-delay: -2s;
            animation-duration: 20s;
        }
        
        @keyframes wave {
            0% { transform: translateX(0); }
            50% { transform: translateX(-25%); }
            100% { transform: translateX(-50%); }
        }
        
        /* Footer content */
        .footer-content {
            position: absolute;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 20px 0;
            color: white;
        }
        
        .social-icons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .social-icon {
            color: white;
            font-size: 20px;
            transition: transform 0.3s ease;
        }
        
        .social-icon:hover {
            transform: translateY(-3px);
        }
        
        /* Pulse animation for sentiment indicators */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
    </style>
    """, unsafe_allow_html=True)

# Set up page with custom CSS
inject_custom_css()

# Header with animation
st.markdown("""
<div class="header-animation">
    <h1 style="text-align: center; color: #2c3e50; margin-bottom: 30px;">🎬 IMDb Movie Reviews Sentiment Analysis</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <p style="text-align: center; font-size: 16px; color: #555;">
        Analyzing sentiment of IMDb movie reviews with interactive visualizations
    </p>
</div>
""", unsafe_allow_html=True)

# Load data with robust error handling
try:
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv('imdb_reviews_with_sentiment.csv', encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
            continue
    
    if df is None or df.empty:
        st.error("Failed to load data or file is empty")
        st.stop()
        
    # Ensure required columns exist
    required_cols = ['review', 'sentiment', 'sentiment_score', 'cleaned_review']
    if not all(col in df.columns for col in required_cols):
        st.error("CSV file is missing required columns")
        st.stop()
        
    sia = SentimentIntensityAnalyzer()
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="card">
        <h3 style="color: #2c3e50;">Filters</h3>
    """, unsafe_allow_html=True)
    
    selected_sentiment = st.selectbox(
        "Select Sentiment",
        ['All', 'Positive', 'Neutral', 'Negative']
    )
    
    st.markdown("""
    <h3 style="color: #2c3e50;">Real-time Analysis</h3>
    """, unsafe_allow_html=True)
    
    user_review = st.text_area("Enter a movie review to analyze:", height=150)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Filter data
if selected_sentiment != 'All':
    df = df[df['sentiment'] == selected_sentiment]

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3 style="color: #2c3e50;">Sentiment Distribution</h3>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots()
    sentiment_counts = df['sentiment'].value_counts()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
           colors=['#4CAF50', '#FFC107', '#F44336'])
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3 style="color: #2c3e50;">Sentiment Scores Distribution</h3>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='sentiment_score', bins=30, kde=True, ax=ax)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# Word clouds with safe sampling
st.markdown("""
<div class="card">
    <h3 style="color: #2c3e50;">Most Common Words by Sentiment</h3>
</div>
""", unsafe_allow_html=True)

sentiments = ['Positive', 'Neutral', 'Negative']
cols = st.columns(3)

for idx, sentiment in enumerate(sentiments):
    with cols[idx]:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <h4 style="color: {'#4CAF50' if sentiment == 'Positive' else '#FFC107' if sentiment == 'Neutral' else '#F44336'}">
                {sentiment} Reviews
            </h4>
        """, unsafe_allow_html=True)
        
        # Safe sampling - won't try to sample more than available
        sentiment_df = df[df['sentiment'] == sentiment]
        sample_size = min(1000, len(sentiment_df))
        text = ' '.join(sentiment_df['cleaned_review'].sample(sample_size)) if sample_size > 0 else ''
        
        if text:
            wordcloud = WordCloud(width=300, height=200, background_color='white').generate(text)
            plt.figure(figsize=(5, 3))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.info(f"No {sentiment} reviews available")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Sample reviews with safe sampling
st.markdown("""
<div class="card">
    <h3 style="color: #2c3e50;">Sample Reviews</h3>
</div>
""", unsafe_allow_html=True)

sample_size = st.slider("Number of reviews to show", 1, 20, 5, key="sample_slider")
sample_size = min(sample_size, len(df))
if sample_size > 0:
    st.dataframe(df[['review', 'sentiment', 'sentiment_score']].sample(sample_size))
else:
    st.info("No reviews available")

# Real-time analysis
if user_review:
    st.markdown("""
    <div class="card">
        <h3 style="color: #2c3e50;">Analysis Results</h3>
    """, unsafe_allow_html=True)
    
    score = sia.polarity_scores(user_review)
    
    if score['compound'] > 0.05:
        sentiment = 'Positive'
        st.markdown(f"""
        <div class="pulse" style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            Sentiment: {sentiment} (Score: {score['compound']:.2f})
        </div>
        """, unsafe_allow_html=True)
    elif score['compound'] < -0.05:
        sentiment = 'Negative'
        st.markdown(f"""
        <div class="pulse" style="background-color: #F44336; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            Sentiment: {sentiment} (Score: {score['compound']:.2f})
        </div>
        """, unsafe_allow_html=True)
    else:
        sentiment = 'Neutral'
        st.markdown(f"""
        <div class="pulse" style="background-color: #FFC107; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            Sentiment: {sentiment} (Score: {score['compound']:.2f})
        </div>
        """, unsafe_allow_html=True)
    
    # Score breakdown
    st.markdown("""
    <h4 style="color: #2c3e50;">Score Breakdown:</h4>
    <ul style="color: #555;">
        <li>Positive: {pos:.2f}</li>
        <li>Neutral: {neu:.2f}</li>
        <li>Negative: {neg:.2f}</li>
    </ul>
    """.format(pos=score['pos'], neu=score['neu'], neg=score['neg']), unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Stylish animated footer
st.markdown("""
<div class="footer">
    <div class="waves">
        <div class="wave wave1"></div>
        <div class="wave wave2"></div>
        <div class="wave wave3"></div>
    </div>
    <div class="footer-content">
        <div class="social-icons">
            <a href="#" class="social-icon">📱</a>
            <a href="#" class="social-icon">📧</a>
            <a href="#" class="social-icon">🔗</a>
            <a href="#" class="social-icon">📞</a>
        </div>
        <p>© 2023 IMDb Sentiment Analysis Dashboard | Data source: <a href="https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews" style="color: white;">Kaggle</a></p>
    </div>
</div>
""", unsafe_allow_html=True)