# MarketPulse: Comprehensive Stock Analysis Platform

# === Imports ===
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import feedparser
import streamlit.components.v1 as components
import json
import os
import base64
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
from io import BytesIO
import time

# Technical Analysis
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# === Page Configuration ===
st.set_page_config(
    page_title="MarketPulse: Comprehensive Stock Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.github.com/marketpulse',
        'Report a bug': 'https://www.github.com/marketpulse/issues',
        'About': 'MarketPulse is a comprehensive stock analysis platform that combines technical indicators, sentiment analysis, and seasonal patterns.'
    }
)

# Initialize session state for portfolios and saved plans
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'saved_plans' not in st.session_state:
    st.session_state.saved_plans = {}
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = {'name': '', 'tickers': [], 'indicators': [], 'start_year': 2010, 'end_year': 2023}

# === Helper Functions ===
def get_sentiment_data(ticker, days=30):
    """Analyze sentiment from news and social media for a given ticker"""
    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Get news from Yahoo Finance
    news_sentiment = []
    try:
        news = fetch_news(ticker)
        for entry in news:
            sentiment = analyzer.polarity_scores(entry.title)
            news_sentiment.append(sentiment['compound'])
    except:
        pass

    # Get social media sentiment (simulated)
    social_keywords = [f"${ticker}", ticker, ticker.lower()]
    social_sentiment = []

    # Simulate social media sentiment (in a real app, this would call Twitter/Reddit APIs)
    np.random.seed(hash(ticker) % 10000)  # Deterministic but different for each ticker
    for _ in range(min(20, days)):
        sentiment = np.random.normal(0.1, 0.5)  # Slightly positive bias
        social_sentiment.append(max(min(sentiment, 1.0), -1.0))  # Clamp between -1 and 1

    # Combine sentiment sources
    avg_news = np.mean(news_sentiment) if news_sentiment else 0
    avg_social = np.mean(social_sentiment) if social_sentiment else 0

    # Weight news more heavily than social (adjustable)
    combined_score = 0.7 * avg_news + 0.3 * avg_social if news_sentiment else avg_social

    sentiment_category = "Bullish" if combined_score > 0.2 else "Neutral" if combined_score > -0.2 else "Bearish"

    return {
        'score': combined_score,
        'category': sentiment_category,
        'news_score': avg_news,
        'social_score': avg_social,
        'news_count': len(news_sentiment),
        'social_count': len(social_sentiment)
    }

def calculate_technical_indicators(df):
    """Calculate various technical indicators for a dataframe with OHLCV data"""
    # Make a copy to avoid modifying the original
    df_ta = df.copy()

    # Trend indicators
    df_ta['SMA20'] = SMAIndicator(close=df_ta['Close'], window=20).sma_indicator()
    df_ta['SMA50'] = SMAIndicator(close=df_ta['Close'], window=50).sma_indicator()
    df_ta['SMA200'] = SMAIndicator(close=df_ta['Close'], window=200).sma_indicator()
    df_ta['EMA20'] = EMAIndicator(close=df_ta['Close'], window=20).ema_indicator()

    # MACD
    macd = MACD(close=df_ta['Close'])
    df_ta['MACD'] = macd.macd()
    df_ta['MACD_Signal'] = macd.macd_signal()
    df_ta['MACD_Histogram'] = macd.macd_diff()

    # RSI
    df_ta['RSI'] = RSIIndicator(close=df_ta['Close']).rsi()

    # Bollinger Bands
    bollinger = BollingerBands(close=df_ta['Close'])
    df_ta['BB_High'] = bollinger.bollinger_hband()
    df_ta['BB_Low'] = bollinger.bollinger_lband()
    df_ta['BB_Mid'] = bollinger.bollinger_mavg()

    # Volume indicators
    df_ta['OBV'] = OnBalanceVolumeIndicator(close=df_ta['Close'], volume=df_ta['Volume']).on_balance_volume()

    return df_ta

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe as CSV"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def generate_pdf_report(ticker, data, technical_data, sentiment_data):
    """Generate a PDF report with analysis results"""
    # In a real implementation, this would use a PDF library to create a report
    # For this example, we'll create a CSV with the data instead
    report_data = {
        'Ticker': ticker,
        'Analysis Date': datetime.now().strftime('%Y-%m-%d'),
        'Best Months': ', '.join(data.head(3)['Month Name'].tolist()),
        'Sentiment': sentiment_data['category'],
        'Sentiment Score': sentiment_data['score'],
        'Current RSI': technical_data['RSI'].iloc[-1] if 'RSI' in technical_data else 'N/A',
        'MACD Signal': 'Bullish' if technical_data.get('MACD_Histogram', pd.Series([0])).iloc[-1] > 0 else 'Bearish',
    }

    report_df = pd.DataFrame([report_data])
    return report_df

# === Custom CSS for a more techy look ===
st.markdown("""
<style>
/* Main app styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Floating stock icons with enhanced tech look */
.float-container {
    position: relative;
    height: 120px;
    overflow: hidden;
    background: rgba(32, 58, 67, 0.1);
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.12), 0 4px 4px rgba(0,0,0,0.15);
}
.float-logo {
    position: absolute;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: white;
    padding: 5px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    animation: float 6s ease-in-out infinite;
}
@keyframes float {
    0% { transform: translateY(0) rotate(0deg); opacity: 0.85; }
    50% { transform: translateY(-20px) rotate(5deg); opacity: 1; }
    100% { transform: translateY(0) rotate(0deg); opacity: 0.85; }
}

/* Card styling for analysis sections */
.css-1r6slb0 {
    border-radius: 10px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    background: linear-gradient(145deg, #f6f8fa, #e6e8ea) !important;
}

/* Metric styling */
.metric-container {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    border-left: 4px solid #66BB6A;
    box-shadow: 0 2px 5px rgba(0,0,0,0.08);
}
.metric-positive { border-left-color: #66BB6A; }
.metric-neutral { border-left-color: #42A5F5; }
.metric-negative { border-left-color: #EF5350; }

/* Button styling */
div.stButton > button:first-child {
    background-color: #3a6b7e;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #2c5364;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: space-between;
    width: 100%;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 8px 8px 0 0;
    padding-top: 10px;
    padding-bottom: 10px;
    flex: 1;
    text-align: center;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(255, 255, 255, 0.15);
    border-bottom: 2px solid #66BB6A;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #f5f7fa;
}
</style>

<!-- Floating stock icons with animation -->
<div class="float-container">
    <img src="https://logo.clearbit.com/apple.com" class="float-logo" style="left:5%; animation-delay: 0.1s;">
    <img src="https://logo.clearbit.com/tesla.com" class="float-logo" style="left:15%; animation-delay: 0.5s;">
    <img src="https://logo.clearbit.com/microsoft.com" class="float-logo" style="left:25%; animation-delay: 0.9s;">
    <img src="https://logo.clearbit.com/amazon.com" class="float-logo" style="left:35%; animation-delay: 1.3s;">
    <img src="https://logo.clearbit.com/nvidia.com" class="float-logo" style="left:45%; animation-delay: 1.7s;">
    <img src="https://logo.clearbit.com/google.com" class="float-logo" style="left:55%; animation-delay: 2.1s;">
    <img src="https://logo.clearbit.com/meta.com" class="float-logo" style="left:65%; animation-delay: 2.5s;">
    <img src="https://logo.clearbit.com/netflix.com" class="float-logo" style="left:75%; animation-delay: 2.9s;">
    <img src="https://logo.clearbit.com/berkshirehathaway.com" class="float-logo" style="left:85%; animation-delay: 3.3s;">
    <img src="https://logo.clearbit.com/jpmorganchase.com" class="float-logo" style="left:95%; animation-delay: 3.7s;">
</div>

<!-- App header with tech-inspired design -->
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #3a6b7e; font-size: 2.5rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.08);">
        <span style="color: #66BB6A;">Market</span>Pulse
    </h1>
    <p style="font-size: 1.2rem; opacity: 0.85;">Comprehensive Stock Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# === Main App Description ===
st.markdown("""
<div style="text-align: center;">
**Welcome to MarketPulse** - Your comprehensive stock analysis platform. This tool combines seasonal patterns, 
technical indicators, and sentiment analysis to help you make informed trading decisions.
</div>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
st.sidebar.header("\U0001F4CA Analysis Options")

# Create tabs in the sidebar for different functions
sidebar_tab = st.sidebar.radio("", ["Analysis", "Portfolio", "Saved Plans"])

if sidebar_tab == "Analysis":
    # === Analysis Options ===
    st.sidebar.subheader("\U0001F50D Stock Selection")
    tickers_input = st.sidebar.text_input("Enter stock ticker(s) separated by commas", value="AAPL, MSFT")

    # Date range selection
    st.sidebar.subheader("\U0001F4C5 Time Period")
    start_year = st.sidebar.slider("Start Year", 2000, 2023, 2010)
    end_year = st.sidebar.slider("End Year", 2001, 2025, 2024)

    # Analysis parameters
    st.sidebar.subheader("\U0001F4C8 Analysis Parameters")
    min_success_rate = st.sidebar.slider("Minimum Success Rate (%)", 0, 100, 60)

    # Technical indicators selection
    st.sidebar.subheader("\U0001F4CA Technical Indicators")
    show_sma = st.sidebar.checkbox("Simple Moving Averages (SMA)", value=True)
    show_ema = st.sidebar.checkbox("Exponential Moving Average (EMA)", value=False)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=False)

    # Sentiment analysis options
    st.sidebar.subheader("\U0001F4AC Sentiment Analysis")
    include_sentiment = st.sidebar.checkbox("Include Sentiment Analysis", value=True)

    # Save current analysis as a plan
    st.sidebar.subheader("\U0001F4BE Save Analysis Plan")
    plan_name = st.sidebar.text_input("Plan Name", value="My Analysis Plan")

    if st.sidebar.button("Save Current Plan"):
        selected_indicators = []
        if show_sma: selected_indicators.append("SMA")
        if show_ema: selected_indicators.append("EMA")
        if show_macd: selected_indicators.append("MACD")
        if show_rsi: selected_indicators.append("RSI")
        if show_bollinger: selected_indicators.append("Bollinger")

        st.session_state.current_plan = {
            'name': plan_name,
            'tickers': [t.strip().upper() for t in tickers_input.split(',')],
            'indicators': selected_indicators,
            'start_year': start_year,
            'end_year': end_year,
            'min_success_rate': min_success_rate,
            'include_sentiment': include_sentiment
        }

        st.session_state.saved_plans[plan_name] = st.session_state.current_plan
        st.sidebar.success(f"Plan '{plan_name}' saved successfully!")

elif sidebar_tab == "Portfolio":
    # === Portfolio Management ===
    st.sidebar.subheader("\U0001F4BC Portfolio Management")

    # Display current portfolio
    if st.session_state.portfolio:
        st.sidebar.write("Current Portfolio:")
        for ticker, details in st.session_state.portfolio.items():
            st.sidebar.write(f"- {ticker}: {details['shares']} shares")
    else:
        st.sidebar.write("No stocks in portfolio yet.")

    # Add stock to portfolio
    st.sidebar.subheader("Add Stock to Portfolio")
    portfolio_ticker = st.sidebar.text_input("Ticker", key="portfolio_ticker")
    shares = st.sidebar.number_input("Number of Shares", min_value=1, value=10, key="portfolio_shares")

    if st.sidebar.button("Add to Portfolio"):
        if portfolio_ticker:
            ticker_upper = portfolio_ticker.strip().upper()
            try:
                # Verify ticker exists
                stock = yf.Ticker(ticker_upper)
                info = stock.info

                # Add to portfolio
                if ticker_upper in st.session_state.portfolio:
                    st.session_state.portfolio[ticker_upper]['shares'] += shares
                else:
                    st.session_state.portfolio[ticker_upper] = {
                        'shares': shares,
                        'added_date': datetime.now().strftime('%Y-%m-%d')
                    }
                st.sidebar.success(f"Added {shares} shares of {ticker_upper} to portfolio!")
            except:
                st.sidebar.error(f"Could not verify ticker {ticker_upper}. Please check the symbol.")

    # Download portfolio
    if st.session_state.portfolio and st.sidebar.button("Download Portfolio"):
        portfolio_df = pd.DataFrame([
            {
                'Ticker': ticker,
                'Shares': details['shares'],
                'Added Date': details['added_date']
            }
            for ticker, details in st.session_state.portfolio.items()
        ])
        st.sidebar.markdown(
            get_download_link(portfolio_df, "marketpulse_portfolio.csv", "Download Portfolio CSV"),
            unsafe_allow_html=True
        )

elif sidebar_tab == "Saved Plans":
    # === Saved Plans ===
    st.sidebar.subheader("\U0001F4C1 Saved Analysis Plans")

    if st.session_state.saved_plans:
        selected_plan = st.sidebar.selectbox(
            "Select a saved plan",
            options=list(st.session_state.saved_plans.keys())
        )

        if st.sidebar.button("Load Plan"):
            plan = st.session_state.saved_plans[selected_plan]
            st.session_state.current_plan = plan

            # Update session state to reflect loaded plan
            st.experimental_rerun()

        if st.sidebar.button("Delete Plan"):
            del st.session_state.saved_plans[selected_plan]
            st.sidebar.success(f"Plan '{selected_plan}' deleted!")
            st.experimental_rerun()
    else:
        st.sidebar.write("No saved plans yet. Create and save an analysis plan from the Analysis tab.")

# === Enhanced Analysis Functions ===
def analyze_seasonality(ticker, start_year, end_year):
    """Analyze seasonal patterns and calculate technical indicators"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

    # Basic return calculations
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Return'] = df['Close'].pct_change()

    # Monthly statistics
    monthly_avg = df.groupby('Month')['Return'].mean().reset_index()
    monthly_avg['Month Name'] = monthly_avg['Month'].apply(lambda x: calendar.month_abbr[x])
    monthly_avg['Return (%)'] = monthly_avg['Return'] * 100

    monthly_success = df.groupby(['Year', 'Month'])['Return'].sum().reset_index()
    monthly_success['Success'] = monthly_success['Return'] > 0
    success_by_month = monthly_success.groupby('Month')['Success'].mean().reset_index()
    success_by_month['Success Rate (%)'] = success_by_month['Success'] * 100

    # Merge results
    result = pd.merge(monthly_avg, success_by_month, on='Month')
    result = result.sort_values(by='Return (%)', ascending=False)

    # Calculate buy and hold return
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100

    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(df)

    # Get current technical signals
    current_signals = get_technical_signals(df_with_indicators)

    return result, buy_hold_return, stock.info, df_with_indicators, current_signals

def get_technical_signals(df_ta):
    """Extract current technical signals from indicators"""
    if len(df_ta) < 50:  # Need enough data for reliable signals
        return {"error": "Not enough data for technical analysis"}

    # Get the most recent values
    latest = df_ta.iloc[-1]
    prev = df_ta.iloc[-2]

    # Initialize signals dictionary
    signals = {}

    # Price vs Moving Averages
    if 'SMA20' in df_ta.columns and 'SMA50' in df_ta.columns and 'SMA200' in df_ta.columns:
        signals['price_vs_sma20'] = "Above" if latest['Close'] > latest['SMA20'] else "Below"
        signals['price_vs_sma50'] = "Above" if latest['Close'] > latest['SMA50'] else "Below"
        signals['price_vs_sma200'] = "Above" if latest['Close'] > latest['SMA200'] else "Below"

        # Golden/Death Cross
        signals['golden_cross'] = latest['SMA50'] > latest['SMA200'] and prev['SMA50'] <= prev['SMA200']
        signals['death_cross'] = latest['SMA50'] < latest['SMA200'] and prev['SMA50'] >= prev['SMA200']

    # MACD Signal
    if 'MACD' in df_ta.columns and 'MACD_Signal' in df_ta.columns:
        signals['macd_signal'] = "Bullish" if latest['MACD'] > latest['MACD_Signal'] else "Bearish"
        signals['macd_crossover'] = (latest['MACD'] > latest['MACD_Signal'] and 
                                    prev['MACD'] <= prev['MACD_Signal'])
        signals['macd_crossunder'] = (latest['MACD'] < latest['MACD_Signal'] and 
                                     prev['MACD'] >= prev['MACD_Signal'])

    # RSI Signals
    if 'RSI' in df_ta.columns:
        signals['rsi_value'] = latest['RSI']
        signals['rsi_signal'] = "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"

    # Bollinger Bands
    if 'BB_High' in df_ta.columns and 'BB_Low' in df_ta.columns:
        signals['bb_position'] = "Upper" if latest['Close'] > latest['BB_High'] else "Lower" if latest['Close'] < latest['BB_Low'] else "Middle"

    # Overall signal based on multiple indicators
    bullish_count = 0
    bearish_count = 0

    # Count bullish signals
    if signals.get('price_vs_sma20') == "Above": bullish_count += 1
    if signals.get('price_vs_sma50') == "Above": bullish_count += 1
    if signals.get('price_vs_sma200') == "Above": bullish_count += 1
    if signals.get('golden_cross', False): bullish_count += 2
    if signals.get('macd_signal') == "Bullish": bullish_count += 1
    if signals.get('macd_crossover', False): bullish_count += 1
    if signals.get('rsi_signal') == "Oversold": bullish_count += 1
    if signals.get('bb_position') == "Lower": bullish_count += 1

    # Count bearish signals
    if signals.get('price_vs_sma20') == "Below": bearish_count += 1
    if signals.get('price_vs_sma50') == "Below": bearish_count += 1
    if signals.get('price_vs_sma200') == "Below": bearish_count += 1
    if signals.get('death_cross', False): bearish_count += 2
    if signals.get('macd_signal') == "Bearish": bearish_count += 1
    if signals.get('macd_crossunder', False): bearish_count += 1
    if signals.get('rsi_signal') == "Overbought": bearish_count += 1
    if signals.get('bb_position') == "Upper": bearish_count += 1

    # Determine overall signal
    if bullish_count > bearish_count + 2:
        signals['overall'] = "Strong Bullish"
    elif bullish_count > bearish_count:
        signals['overall'] = "Moderately Bullish"
    elif bearish_count > bullish_count + 2:
        signals['overall'] = "Strong Bearish"
    elif bearish_count > bullish_count:
        signals['overall'] = "Moderately Bearish"
    else:
        signals['overall'] = "Neutral"

    return signals

# === News Fetching Function ===
def fetch_news(ticker):
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    return feedparser.parse(feed_url).entries[:5]

# === Main Analysis ===
if sidebar_tab == "Analysis" and st.sidebar.button("Run Analysis"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]

    for ticker in tickers:
        st.markdown(f"## Analysis for {ticker}", unsafe_allow_html=True)

        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "📅 Seasonal", 
            "📈 Technical", 
            "💬 Sentiment", 
            "📰 News", 
            "📑 Report"
        ])

        try:
            # Run the comprehensive analysis
            data, buy_hold, info, df_technical, tech_signals = analyze_seasonality(ticker, start_year, end_year)

            # Filter data based on minimum success rate
            filtered_data = data[data['Success Rate (%)'] >= min_success_rate]

            # Get sentiment data if enabled
            sentiment_data = get_sentiment_data(ticker) if include_sentiment else None

            # === Overview Tab ===
            with tab1:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader(f"Company Overview: {info.get('longName', ticker)}")
                    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.markdown(f"**Current Price:** ${info.get('currentPrice', 'N/A')}")
                    st.markdown(f"**52-Week Range:** ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")
                    st.markdown(f"**Market Cap:** ${info.get('marketCap', 'N/A'):,}")

                    # Summary with expandable section for long text
                    with st.expander("Company Summary"):
                        st.write(info.get('longBusinessSummary', 'No summary available.'))

                with col2:
                    # Display company logo if available
                    try:
                        logo_url = f"https://logo.clearbit.com/{info.get('website', '').split('//')[1].split('/')[0]}"
                        st.image(logo_url, width=150)
                    except:
                        st.info("Logo not available")

                    # Key metrics in styled containers
                    st.markdown("<h4>Key Metrics</h4>", unsafe_allow_html=True)

                    # Technical signal summary
                    signal_color = "metric-positive" if "Bullish" in tech_signals.get('overall', '') else \
                                  "metric-negative" if "Bearish" in tech_signals.get('overall', '') else "metric-neutral"

                    st.markdown(f"""
                    <div class="metric-container {signal_color}">
                        <strong>Technical Signal:</strong> {tech_signals.get('overall', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)

                    # Sentiment summary if available
                    if sentiment_data:
                        sentiment_color = "metric-positive" if sentiment_data['category'] == "Bullish" else \
                                         "metric-negative" if sentiment_data['category'] == "Bearish" else "metric-neutral"

                        st.markdown(f"""
                        <div class="metric-container {sentiment_color}">
                            <strong>Sentiment:</strong> {sentiment_data['category']} ({sentiment_data['score']:.2f})
                        </div>
                        """, unsafe_allow_html=True)

                    # Seasonal summary
                    best_month = data.iloc[0]['Month Name'] if not data.empty else "N/A"
                    best_return = data.iloc[0]['Return (%)'] if not data.empty else 0
                    seasonal_color = "metric-positive" if best_return > 0 else "metric-negative"

                    st.markdown(f"""
                    <div class="metric-container {seasonal_color}">
                        <strong>Best Month:</strong> {best_month} ({best_return:.2f}%)
                    </div>
                    """, unsafe_allow_html=True)

            # === Seasonal Analysis Tab ===
            with tab2:
                st.subheader(f"Seasonal Analysis for {ticker}")

                # Display key seasonal insights
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Buy & Hold Return ({start_year}-{end_year}):** {buy_hold:.2f}%")
                    best_months = filtered_data.head(3)
                    st.markdown(f"**Top 3 months to trade:** {', '.join(best_months['Month Name'].tolist())}")

                with col2:
                    danger_months = filtered_data[filtered_data['Return (%)'] < 0]['Month Name'].tolist()
                    if danger_months:
                        st.warning(f"⚠️ Historically weak months: {', '.join(danger_months)}")

                # Seasonal return chart
                fig = px.bar(
                    filtered_data, x='Month Name', y='Return (%)', color='Success Rate (%)',
                    color_continuous_scale='RdYlGn',
                    title=f'Monthly Return Analysis ({start_year}-{end_year})',
                    labels={'Return (%)': 'Avg Return', 'Month Name': 'Month'},
                    hover_data={'Success Rate (%)': ':.2f'}
                )
                fig.update_layout(
                    font=dict(size=14),
                    plot_bgcolor='white',
                    xaxis=dict(title='Month', showgrid=False),
                    yaxis=dict(title='Avg Return (%)', showgrid=True)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Data table with seasonal statistics
                st.subheader("Monthly Performance Statistics")
                st.dataframe(filtered_data[['Month Name', 'Return (%)', 'Success Rate (%)']])

                # Download link for seasonal data
                st.markdown(
                    get_download_link(filtered_data, f"{ticker}_seasonal_analysis.csv", "Download Seasonal Data"),
                    unsafe_allow_html=True
                )

            # === Technical Analysis Tab ===
            with tab3:
                st.subheader(f"Technical Analysis for {ticker}")

                # Technical signals summary
                st.markdown("### Current Technical Signals")

                # Create columns for signal display
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Overall Signal:** {tech_signals.get('overall', 'N/A')}")
                    if 'rsi_value' in tech_signals:
                        st.markdown(f"**RSI:** {tech_signals['rsi_value']:.2f} ({tech_signals['rsi_signal']})")

                with col2:
                    if 'price_vs_sma50' in tech_signals:
                        st.markdown(f"**Price vs SMA50:** {tech_signals['price_vs_sma50']}")
                    if 'price_vs_sma200' in tech_signals:
                        st.markdown(f"**Price vs SMA200:** {tech_signals['price_vs_sma200']}")

                with col3:
                    if 'macd_signal' in tech_signals:
                        st.markdown(f"**MACD Signal:** {tech_signals['macd_signal']}")
                    if 'golden_cross' in tech_signals and tech_signals['golden_cross']:
                        st.markdown("**Golden Cross Detected!** ✨")
                    if 'death_cross' in tech_signals and tech_signals['death_cross']:
                        st.markdown("**Death Cross Detected!** ⚠️")

                # Technical chart
                if show_sma or show_ema or show_macd or show_rsi or show_bollinger:
                    st.subheader("Technical Chart")

                    # Create figure with secondary y-axis for indicators
                    recent_data = df_technical.tail(180)  # Last ~6 months

                    # Determine which plots to show
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                       vertical_spacing=0.03, 
                                       row_heights=[0.7, 0.3],
                                       subplot_titles=("Price & Moving Averages", "Indicators"))

                    # Add price candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=recent_data.index,
                            open=recent_data['Open'],
                            high=recent_data['High'],
                            low=recent_data['Low'],
                            close=recent_data['Close'],
                            name="Price"
                        ),
                        row=1, col=1
                    )

                    # Add moving averages if selected
                    if show_sma:
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['SMA20'], name="SMA20", line=dict(color='blue')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['SMA50'], name="SMA50", line=dict(color='orange')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['SMA200'], name="SMA200", line=dict(color='red')),
                            row=1, col=1
                        )

                    if show_ema:
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['EMA20'], name="EMA20", line=dict(color='purple')),
                            row=1, col=1
                        )

                    # Add Bollinger Bands if selected
                    if show_bollinger:
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['BB_High'], name="BB Upper", line=dict(color='rgba(0,128,0,0.3)')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['BB_Mid'], name="BB Middle", line=dict(color='rgba(0,128,0,0.5)')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['BB_Low'], name="BB Lower", line=dict(color='rgba(0,128,0,0.3)')),
                            row=1, col=1
                        )

                    # Add RSI if selected
                    if show_rsi:
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['RSI'], name="RSI", line=dict(color='green')),
                            row=2, col=1
                        )
                        # Add RSI reference lines
                        fig.add_shape(type="line", x0=recent_data.index[0], x1=recent_data.index[-1], y0=70, y1=70,
                                     line=dict(color="red", width=1, dash="dash"), row=2, col=1)
                        fig.add_shape(type="line", x0=recent_data.index[0], x1=recent_data.index[-1], y0=30, y1=30,
                                     line=dict(color="green", width=1, dash="dash"), row=2, col=1)

                    # Add MACD if selected
                    if show_macd:
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['MACD'], name="MACD", line=dict(color='blue')),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=recent_data.index, y=recent_data['MACD_Signal'], name="MACD Signal", line=dict(color='red')),
                            row=2, col=1
                        )
                        # Add MACD histogram
                        fig.add_trace(
                            go.Bar(x=recent_data.index, y=recent_data['MACD_Histogram'], name="MACD Histogram", marker_color='green'),
                            row=2, col=1
                        )

                    # Update layout
                    fig.update_layout(
                        title=f"{ticker} Technical Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=800,
                        xaxis_rangeslider_visible=False,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # === Sentiment Analysis Tab ===
            with tab4:
                st.subheader(f"Sentiment Analysis for {ticker}")

                if include_sentiment and sentiment_data:
                    # Display sentiment summary
                    sentiment_score = sentiment_data['score']
                    sentiment_category = sentiment_data['category']

                    # Create a gauge chart for sentiment
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = (sentiment_score + 1) * 50,  # Convert from -1,1 to 0,100 scale
                        title = {'text': f"Sentiment Score: {sentiment_category}"},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 45], 'color': "orange"},
                                {'range': [45, 55], 'color': "yellow"},
                                {'range': [55, 70], 'color': "lightgreen"},
                                {'range': [70, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': (sentiment_score + 1) * 50
                            }
                        }
                    ))

                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display sentiment details
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**News Sentiment:** {sentiment_data['news_score']:.2f}")
                        st.markdown(f"**News Articles Analyzed:** {sentiment_data['news_count']}")

                    with col2:
                        st.markdown(f"**Social Media Sentiment:** {sentiment_data['social_score']:.2f}")
                        st.markdown(f"**Social Posts Analyzed:** {sentiment_data['social_count']}")

                    # Sentiment interpretation
                    st.subheader("Sentiment Interpretation")

                    if sentiment_category == "Bullish":
                        st.markdown("""
                        📈 **Bullish Sentiment Detected**

                        The overall sentiment for this stock is positive. This suggests that:
                        - News and social media coverage is generally favorable
                        - Market participants have a positive outlook
                        - There may be positive catalysts or developments

                        *Consider this a potential supporting factor for long positions.*
                        """)
                    elif sentiment_category == "Bearish":
                        st.markdown("""
                        📉 **Bearish Sentiment Detected**

                        The overall sentiment for this stock is negative. This suggests that:
                        - News and social media coverage is generally unfavorable
                        - Market participants have concerns about the stock
                        - There may be negative catalysts or developments

                        *Consider this a potential supporting factor for short positions or caution.*
                        """)
                    else:
                        st.markdown("""
                        📊 **Neutral Sentiment Detected**

                        The overall sentiment for this stock is balanced. This suggests that:
                        - News and social media coverage is mixed
                        - Market participants have varied opinions
                        - There may be both positive and negative factors at play

                        *Consider focusing more on technical and seasonal factors for this stock.*
                        """)
                else:
                    st.info("Sentiment analysis is disabled. Enable it in the sidebar to see sentiment data.")

            # === News Tab ===
            with tab5:
                st.subheader(f"Latest News for {ticker}")

                try:
                    news = fetch_news(ticker)
                    if news:
                        for i, entry in enumerate(news):
                            with st.container():
                                st.markdown(f"### {entry.title}")
                                st.markdown(f"*{entry.published if hasattr(entry, 'published') else 'Recent'}*")
                                if hasattr(entry, 'summary'):
                                    st.markdown(entry.summary)
                                st.markdown(f"[Read more]({entry.link})")
                                if i < len(news) - 1:  # Don't add divider after last item
                                    st.markdown("---")
                    else:
                        st.info("No recent news found for this ticker.")
                except Exception as e:
                    st.error(f"Error fetching news: {str(e)}")

            # === Report Tab ===
            with tab6:
                st.subheader(f"Comprehensive Report for {ticker}")

                # Generate report data
                report_df = generate_pdf_report(ticker, filtered_data, df_technical, 
                                              sentiment_data if include_sentiment else {})

                # Report preview
                st.markdown("### Report Preview")

                # Summary of findings
                st.markdown("#### Key Findings")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Best Trading Months:** {', '.join(filtered_data.head(3)['Month Name'].tolist())}")
                    st.markdown(f"**Technical Signal:** {tech_signals.get('overall', 'N/A')}")
                    if include_sentiment and sentiment_data:
                        st.markdown(f"**Sentiment Analysis:** {sentiment_data['category']}")

                with col2:
                    st.markdown(f"**Buy & Hold Return:** {buy_hold:.2f}%")
                    if 'rsi_value' in tech_signals:
                        st.markdown(f"**Current RSI:** {tech_signals['rsi_value']:.2f}")
                    if 'macd_signal' in tech_signals:
                        st.markdown(f"**MACD Signal:** {tech_signals['macd_signal']}")

                # Strategy recommendation
                st.markdown("#### Strategy Recommendation")

                # Combine signals for recommendation
                tech_signal = tech_signals.get('overall', 'Neutral')
                seasonal_signal = "Bullish" if filtered_data.iloc[0]['Return (%)'] > 2 else "Neutral" if filtered_data.iloc[0]['Return (%)'] > 0 else "Bearish"
                sentiment_signal = sentiment_data['category'] if include_sentiment and sentiment_data else "Neutral"

                # Count signals
                bullish_count = sum(1 for signal in [tech_signal, seasonal_signal, sentiment_signal] if "Bullish" in signal)
                bearish_count = sum(1 for signal in [tech_signal, seasonal_signal, sentiment_signal] if "Bearish" in signal)

                if bullish_count > bearish_count:
                    st.markdown("""
                    🟢 **Bullish Outlook**

                    Based on the combined analysis of seasonal patterns, technical indicators, and sentiment, 
                    this stock shows a generally positive outlook. Consider a long position with appropriate 
                    risk management.
                    """)
                elif bearish_count > bullish_count:
                    st.markdown("""
                    🔴 **Bearish Outlook**

                    Based on the combined analysis of seasonal patterns, technical indicators, and sentiment, 
                    this stock shows a generally negative outlook. Consider avoiding long positions or implementing 
                    hedging strategies.
                    """)
                else:
                    st.markdown("""
                    🟡 **Neutral Outlook**

                    Based on the combined analysis of seasonal patterns, technical indicators, and sentiment, 
                    this stock shows a mixed outlook. Consider waiting for clearer signals before taking a position, 
                    or implement a market-neutral strategy.
                    """)

                # Download options
                st.subheader("Download Options")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        get_download_link(report_df, f"{ticker}_comprehensive_report.csv", "Download Report Data (CSV)"),
                        unsafe_allow_html=True
                    )

                with col2:
                    # Add to portfolio button
                    if st.button(f"Add {ticker} to Portfolio"):
                        if ticker in st.session_state.portfolio:
                            st.session_state.portfolio[ticker]['shares'] += 10  # Default 10 shares
                        else:
                            st.session_state.portfolio[ticker] = {
                                'shares': 10,
                                'added_date': datetime.now().strftime('%Y-%m-%d')
                            }
                        st.success(f"Added {ticker} to your portfolio!")

        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            st.markdown("Please check the ticker symbol and try again.")

# === Portfolio Analysis ===
elif sidebar_tab == "Portfolio" and st.session_state.portfolio:
    st.title("Portfolio Analysis")

    # Display portfolio summary
    st.subheader("Current Portfolio")

    # Create portfolio dataframe
    portfolio_data = []
    total_value = 0

    for ticker, details in st.session_state.portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('currentPrice', 0)
            value = current_price * details['shares']
            total_value += value

            portfolio_data.append({
                'Ticker': ticker,
                'Company': info.get('shortName', ticker),
                'Shares': details['shares'],
                'Current Price': current_price,
                'Value': value,
                'Added Date': details['added_date']
            })
        except:
            portfolio_data.append({
                'Ticker': ticker,
                'Company': ticker,
                'Shares': details['shares'],
                'Current Price': 'N/A',
                'Value': 'N/A',
                'Added Date': details['added_date']
            })

    portfolio_df = pd.DataFrame(portfolio_data)

    # Display portfolio value
    if total_value > 0:
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

    # Display portfolio table
    st.dataframe(portfolio_df)

    # Portfolio download
    st.markdown(
        get_download_link(portfolio_df, "marketpulse_portfolio.csv", "Download Portfolio Data"),
        unsafe_allow_html=True
    )

    # Portfolio analysis
    if len(portfolio_data) > 0:
        st.subheader("Portfolio Analysis")

        # Run analysis on each stock in portfolio
        for ticker in st.session_state.portfolio.keys():
            with st.expander(f"Analysis for {ticker}"):
                try:
                    # Quick analysis of the stock
                    data, buy_hold, info, df_technical, tech_signals = analyze_seasonality(ticker, datetime.now().year-3, datetime.now().year)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Technical Signal:** {tech_signals.get('overall', 'N/A')}")
                        st.markdown(f"**Best Month to Trade:** {data.iloc[0]['Month Name']} ({data.iloc[0]['Return (%)']:.2f}%)")

                    with col2:
                        if 'rsi_value' in tech_signals:
                            st.markdown(f"**Current RSI:** {tech_signals['rsi_value']:.2f}")
                        if 'macd_signal' in tech_signals:
                            st.markdown(f"**MACD Signal:** {tech_signals['macd_signal']}")
                except:
                    st.warning(f"Could not analyze {ticker}")

    # Portfolio report
    st.subheader("Portfolio Report")
    st.markdown("Generate a comprehensive report of your entire portfolio with analysis of each stock.")

    if st.button("Generate Portfolio Report"):
        # Create a more detailed portfolio report
        report_data = []

        for ticker, details in st.session_state.portfolio.items():
            try:
                data, buy_hold, info, df_technical, tech_signals = analyze_seasonality(ticker, datetime.now().year-3, datetime.now().year)
                sentiment_data = get_sentiment_data(ticker)

                report_data.append({
                    'Ticker': ticker,
                    'Company': info.get('shortName', ticker),
                    'Shares': details['shares'],
                    'Current Price': info.get('currentPrice', 'N/A'),
                    'Technical Signal': tech_signals.get('overall', 'N/A'),
                    'Best Month': data.iloc[0]['Month Name'],
                    'Sentiment': sentiment_data['category'] if sentiment_data else 'N/A',
                    'RSI': tech_signals.get('rsi_value', 'N/A'),
                    'MACD Signal': tech_signals.get('macd_signal', 'N/A')
                })
            except:
                report_data.append({
                    'Ticker': ticker,
                    'Company': ticker,
                    'Shares': details['shares'],
                    'Current Price': 'N/A',
                    'Technical Signal': 'N/A',
                    'Best Month': 'N/A',
                    'Sentiment': 'N/A',
                    'RSI': 'N/A',
                    'MACD Signal': 'N/A'
                })

        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df)

        # Download link
        st.markdown(
            get_download_link(report_df, "marketpulse_portfolio_report.csv", "Download Portfolio Report"),
            unsafe_allow_html=True
        )

# === Saved Plans ===
elif sidebar_tab == "Saved Plans" and st.session_state.saved_plans:
    st.title("Saved Analysis Plans")

    # Display all saved plans
    for plan_name, plan in st.session_state.saved_plans.items():
        with st.expander(f"Plan: {plan_name}"):
            st.markdown(f"**Tickers:** {', '.join(plan['tickers'])}")
            st.markdown(f"**Time Period:** {plan['start_year']} - {plan['end_year']}")
            st.markdown(f"**Indicators:** {', '.join(plan['indicators'])}")
            st.markdown(f"**Include Sentiment:** {'Yes' if plan.get('include_sentiment', False) else 'No'}")

            # Button to load this plan
            if st.button(f"Load {plan_name}", key=f"load_{plan_name}"):
                st.session_state.current_plan = plan
                st.experimental_rerun()
