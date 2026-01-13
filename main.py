import streamlit as st
import tensorflow as tf
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .fake-news {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
    }
    .true-news {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model('fake_news_detector.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Preprocessing function
@st.cache_data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Prediction function
def predict_news(text, model, tokenizer, max_len=200):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded, verbose=0)[0][0]
    
    return pred, cleaned

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("<h1>🔍 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by Deep Learning | Detect Misinformation Instantly</p>", unsafe_allow_html=True)

# Load model
model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("⚠️ Model ya tokenizer load nahi ho saka. Please check files!")
    st.info("📝 Required files: fake_news_detector.h5 aur tokenizer.pkl")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Statistics")
    
    if st.session_state.history:
        total = len(st.session_state.history)
        fake_count = sum(1 for h in st.session_state.history if h['prediction'] == 'FAKE')
        true_count = total - fake_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Checks", total)
        with col2:
            st.metric("Fake News", fake_count)
        
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Fake', 'True'],
            values=[fake_count, true_count],
            hole=.3,
            marker_colors=['#f5576c', '#00f2fe']
        )])
        fig.update_layout(
            showlegend=True,
            height=250,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet!")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This AI-powered tool uses:
    - 🧠 **Bidirectional LSTM**
    - 📝 **NLP Processing**
    - 🎯 **High Accuracy**
    
    Trained on thousands of news articles!
    """)
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Detect News", "📜 History", "📈 Analytics"])

with tab1:
    st.markdown("### Enter News Article")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["Text Input", "Sample News"], horizontal=True)
    
    if input_method == "Text Input":
        news_text = st.text_area(
            "Paste your news article here:",
            height=200,
            placeholder="Enter the news text you want to verify..."
        )
    else:
        sample_options = {
            "Sample 1 - Suspicious": "Breaking: Scientists claim they have discovered aliens living among us in secret underground bases.",
            "Sample 2 - Credible": "The Federal Reserve announced today that it will maintain current interest rates following the monthly economic review.",
            "Sample 3 - Suspicious": "Miracle cure found! This one weird trick will cure all diseases instantly, doctors hate it!",
            "Sample 4 - Credible": "Stock markets showed mixed results today with technology sector gaining 2% while energy stocks declined."
        }
        selected_sample = st.selectbox("Select a sample:", list(sample_options.keys()))
        news_text = sample_options[selected_sample]
        st.text_area("Selected news:", news_text, height=150, disabled=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        analyze_button = st.button("🔍 Analyze News", use_container_width=True)
    
    if analyze_button and news_text:
        with st.spinner("🔄 Analyzing news article..."):
            # Simulate processing time for effect
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Make prediction
            pred_score, cleaned_text = predict_news(news_text, model, tokenizer)
            
            # Determine result
            if pred_score > 0.5:
                prediction = "TRUE"
                confidence = pred_score * 100
                st.markdown(f"""
                    <div class='true-news'>
                        ✅ TRUE NEWS<br>
                        <span style='font-size: 18px;'>Confidence: {confidence:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                prediction = "FAKE"
                confidence = (1 - pred_score) * 100
                st.markdown(f"""
                    <div class='fake-news'>
                        ❌ FAKE NEWS<br>
                        <span style='font-size: 18px;'>Confidence: {confidence:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Confidence meter
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#00f2fe" if prediction == "TRUE" else "#f5576c"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info
            with st.expander("📊 Detailed Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", len(news_text.split()))
                    st.metric("Character Count", len(news_text))
                with col2:
                    st.metric("Prediction Score", f"{pred_score:.4f}")
                    st.metric("Processed Words", len(cleaned_text.split()))
                
                st.markdown("**Processed Text Preview:**")
                st.text(cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text)
            
            # Save to history
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'text': news_text[:100] + "..." if len(news_text) > 100 else news_text,
                'prediction': prediction,
                'confidence': confidence,
                'score': pred_score
            })
            
            st.success("✅ Analysis complete!")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze!")

with tab2:
    st.markdown("### 📜 Prediction History")
    
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history), 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{idx}. {item['timestamp']}**")
                    st.text(item['text'])
                
                with col2:
                    if item['prediction'] == "FAKE":
                        st.markdown("🔴 **FAKE**")
                    else:
                        st.markdown("🟢 **TRUE**")
                
                with col3:
                    st.metric("Confidence", f"{item['confidence']:.1f}%")
                
                st.markdown("---")
    else:
        st.info("No predictions yet. Start analyzing news articles!")

with tab3:
    st.markdown("### 📈 Analytics Dashboard")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fake_count = sum(1 for h in st.session_state.history if h['prediction'] == 'FAKE')
            st.metric("Total Fake News", fake_count, delta=None)
        
        with col2:
            true_count = sum(1 for h in st.session_state.history if h['prediction'] == 'TRUE')
            st.metric("Total True News", true_count, delta=None)
        
        with col3:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Timeline chart
        st.markdown("#### Prediction Timeline")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = px.scatter(df, x='timestamp', y='confidence', 
                        color='prediction',
                        color_discrete_map={'FAKE': '#f5576c', 'TRUE': '#00f2fe'},
                        size='confidence',
                        hover_data=['text'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        st.markdown("#### Confidence Distribution")
        fig = px.histogram(df, x='confidence', color='prediction',
                          color_discrete_map={'FAKE': '#f5576c', 'TRUE': '#00f2fe'},
                          nbins=20)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No data available yet. Start making predictions!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <p>Made with ❤️ using Streamlit & TensorFlow | © 2025 Fake News Detector</p>
    </div>
""", unsafe_allow_html=True)