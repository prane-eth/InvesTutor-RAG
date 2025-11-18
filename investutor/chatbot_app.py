import streamlit as st
from utils.ingestion import ingest_document
from utils.model_utils import rag_chain
from utils.news_integration import (fetch_financial_news, get_market_sentiment,
                                    summarize_news_article)

# Page configuration
st.set_page_config(
    page_title="AI Investment Tutor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .lesson-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .news-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .sentiment-bullish {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-bearish {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic" not in st.session_state:
    st.session_state.current_topic = "Introduction to Investing"
if "lesson_progress" not in st.session_state:
    st.session_state.lesson_progress = {}

# Sidebar
with st.sidebar:
    st.title("📚 AI Investment Tutor")
    
    # Topic selection
    topics = [
        "Introduction to Investing",
        "Stocks and Equities",
        "Bonds and Fixed Income",
        "Diversification",
        "Risk Management",
        "Portfolio Construction",
        "Market Analysis"
    ]
    
    selected_topic = st.selectbox("Choose a topic:", topics, index=topics.index(st.session_state.current_topic))
    if selected_topic != st.session_state.current_topic:
        st.session_state.current_topic = selected_topic
        st.rerun()
    
    # Progress tracking
    st.subheader("📊 Your Progress")
    for topic in topics:
        progress = st.session_state.lesson_progress.get(topic, 0)
        st.progress(progress, text=f"{topic}: {progress}%")
    
    # Market sentiment
    st.subheader("🌡️ Market Sentiment")
    sentiment = get_market_sentiment()
    sentiment_class = f"sentiment-{sentiment['overall']}"
    st.markdown(f"<p class='{sentiment_class}'>{sentiment['overall'].upper()}</p>", unsafe_allow_html=True)
    st.caption(f"Confidence: {sentiment['confidence']:.1%}")

# Main content
st.markdown('<h1 class="main-header">AI Investment Tutor</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📖 Lessons", "❓ Q&A", "📰 News", "⚙️ Settings"])

with tab1:
    st.header(f"📖 {st.session_state.current_topic}")
    
    # Lesson content based on topic
    lesson_content = {
        "Introduction to Investing": """
        ## What is Investing?
        
        Investing is the act of allocating resources, usually money, with the expectation of generating income or profit.
        
        ### Key Concepts:
        - **Risk vs. Reward**: Higher potential returns usually come with higher risk
        - **Time Horizon**: Your investment timeline affects strategy
        - **Diversification**: Spreading investments to reduce risk
        - **Compound Interest**: Earning interest on interest
        
        ### Getting Started:
        1. Define your financial goals
        2. Assess your risk tolerance
        3. Choose appropriate investment vehicles
        4. Monitor and adjust your portfolio
        """,
        
        "Stocks and Equities": """
        ## Understanding Stocks
        
        Stocks represent ownership in a company. When you buy a stock, you become a shareholder.
        
        ### Types of Stocks:
        - **Common Stock**: Basic ownership with voting rights
        - **Preferred Stock**: Priority claims on dividends and assets
        - **Growth Stocks**: Companies expected to grow faster than average
        - **Value Stocks**: Undervalued companies trading below intrinsic value
        
        ### Stock Market Basics:
        - **Bull Market**: Rising prices and optimism
        - **Bear Market**: Falling prices and pessimism
        - **Blue-Chip Stocks**: Large, stable companies
        - **Penny Stocks**: Low-priced, speculative stocks
        """,
        
        "Diversification": """
        ## The Power of Diversification
        
        Diversification is spreading investments across different assets to reduce risk.
        
        ### Why Diversify?
        - **Risk Reduction**: No single investment can sink your portfolio
        - **Smoother Returns**: Balance volatility across assets
        - **Opportunity Capture**: Different assets perform well at different times
        
        ### Diversification Strategies:
        - **Asset Allocation**: Mix of stocks, bonds, cash
        - **Geographic Diversification**: Invest in different countries
        - **Sector Diversification**: Spread across industries
        - **Time Diversification**: Dollar-cost averaging
        """
    }
    
    content = lesson_content.get(st.session_state.current_topic, "Lesson content coming soon...")
    st.markdown(content)
    
    # Interactive quiz
    st.subheader("🧠 Quick Quiz")
    if st.session_state.current_topic == "Introduction to Investing":
        quiz_question = st.radio("What is compound interest?", 
                                ["Interest paid on the principal only", 
                                 "Interest paid on both principal and accumulated interest",
                                 "Interest paid monthly"])
        if st.button("Check Answer"):
            if "both principal and accumulated interest" in quiz_question:
                st.success("Correct! Compound interest helps your money grow exponentially.")
                st.session_state.lesson_progress[st.session_state.current_topic] = min(100, 
                    st.session_state.lesson_progress.get(st.session_state.current_topic, 0) + 25)
            else:
                st.error("Not quite. Compound interest is interest earned on both the initial principal and the accumulated interest from previous periods.")

with tab2:
    st.header("❓ Ask Questions About Investing")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about investing..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if rag_chain:
                    response = rag_chain(prompt)
                else:
                    response = "The knowledge base is not available. Please check the configuration."
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab3:
    st.header("📰 Financial News & Insights")
    
    # News filters
    col1, col2 = st.columns(2)
    with col1:
        news_query = st.text_input("Search news:", "investments OR markets")
    with col2:
        days_back = st.slider("Days back:", 1, 7, 1)
    
    if st.button("🔍 Fetch News"):
        with st.spinner("Fetching latest news..."):
            news = fetch_financial_news(news_query, days_back)
            
            if news:
                st.success(f"Found {len(news)} articles")
                
                for article in news:
                    with st.container():
                        st.markdown('<div class="news-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(article['title'])
                            st.write(article['description'])
                            st.caption(f"Source: {article['source']} | {article['published_at'][:10]}")
                        with col2:
                            if st.button("📖 Summarize", key=f"sum_{article['url']}"):
                                with st.spinner("Summarizing..."):
                                    summary = summarize_news_article(article['url'])
                                    st.info(summary)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No news found. Check your NewsAPI key configuration.")

with tab4:
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF document", type=['pdf'])
    url_input = st.text_input("Or enter URL to ingest:")
    
    if st.button("📥 Ingest Document"):
        if uploaded_file:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            try:
                ingest_document(temp_path)
                st.success("Document ingested successfully!")
            except Exception as e:
                st.error(f"Failed to ingest document: {e}")
            finally:
                import os
                os.unlink(temp_path)
                
        elif url_input:
            try:
                ingest_document(url_input)
                st.success("URL content ingested successfully!")
            except Exception as e:
                st.error(f"Failed to ingest URL: {e}")
        else:
            st.warning("Please upload a file or enter a URL.")
    
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents in Knowledge Base", "3+")
    with col2:
        st.metric("RAG System Status", "✅ Active")

# Footer
st.markdown("---")
st.caption("AI Investment Tutor - Powered by RAG technology. Always consult with financial professionals for personalized advice.")
