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
    height: 180px;
    overflow: hidden;
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23);
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
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    border-left: 4px solid #4CAF50;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.metric-positive { border-left-color: #4CAF50; }
.metric-neutral { border-left-color: #2196F3; }
.metric-negative { border-left-color: #F44336; }

/* Button styling */
div.stButton > button:first-child {
    background-color: #2c5364;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #203a43;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transform: translateY(-2px);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 8px 8px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(255, 255, 255, 0.1);
    border-bottom: 2px solid #4CAF50;
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
    <h1 style="color: #2c5364; font-size: 2.5rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">
        <span style="color: #4CAF50;">Market</span>Pulse
    </h1>
    <p style="font-size: 1.2rem; opacity: 0.8;">Comprehensive Stock Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# === Main App Description ===
st.markdown("""
**Welcome to MarketPulse** - Your comprehensive stock analysis platform. This tool combines seasonal patterns, 
technical indicators, and sentiment analysis to help you make informed trading decisions.
""")

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
if st.sidebar.button("Run Analysis"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    for ticker in tickers:
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analysis", "News", "Report Preview"])

        with tab1:
            st.subheader(f"\U0001F4C8 Company Overview: {ticker}")
            try:
                _, _, info = analyze_seasonality(ticker, start_year, end_year)
                st.markdown(f"**Name:** {info.get('longName', 'N/A')}")
                st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                st.markdown(f"**Summary:** {info.get('longBusinessSummary', 'No summary available.')}")
            except:
                st.warning("Unable to retrieve company overview.")

        with tab2:
            st.subheader(f"\U0001F4C9 Seasonal Analysis for {ticker}")
            try:
                data, buy_hold, _ = analyze_seasonality(ticker, start_year, end_year)
                data = data[data['Success Rate (%)'] >= min_success_rate]
                danger_months = data[data['Return (%)'] < 0]['Month Name'].tolist()
                if danger_months:
                    st.warning(f"⚠️ Historically weak months: {', '.join(danger_months)}")

                st.markdown(f"**Buy & Hold Return (Full Period):** {buy_hold:.2f}%")
                best_months = data.head(3)
                st.markdown(f"**Top 3 months to trade {ticker}:** {', '.join(best_months['Month Name'])}")

                fig = px.bar(
                    data, x='Month Name', y='Return (%)', color='Success Rate (%)',
                    color_continuous_scale='Blues',
                    title=f'Seasonal Return Analysis for {ticker}',
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
                st.dataframe(data[['Month Name', 'Return (%)', 'Success Rate (%)']])
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")

        with tab3:
            st.subheader(f"\U0001F4F0 Latest News for {ticker}")
            try:
                news = fetch_news(ticker)
                for entry in news:
                    st.markdown(f"**{entry.title}**")
                    st.markdown(f"[{entry.link}]({entry.link})")
                    st.markdown("---")
            except:
                st.info("No news or unable to fetch.")

        with tab4:
            st.subheader("\U0001F4C4 Report Preview")
            st.markdown("""
            This report will include:
            - Top 3 performing months
            - Success rate chart
            - Strategy notes
            - Visuals for print/export
            """)
