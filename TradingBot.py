# Updated MarketPulse App: Cleaned, Structured, and Branded

# === Imports ===
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import calendar
import feedparser
import streamlit.components.v1 as components

# === Page Configuration ===
st.set_page_config(page_title="MarketPulse: Seasonal Stock Advisor", layout="wide")

# === Floating Stock Icons Animation ===
st.markdown("""
<style>
.float-container {
    position: relative;
    height: 200px;
    overflow: hidden;
    background: linear-gradient(to right, #f5f7fa, #c3cfe2);
    border-radius: 12px;
    margin-bottom: 30px;
}
.float-logo {
    position: absolute;
    width: 60px;
    animation: float 6s ease-in-out infinite;
}
@keyframes float {
    0% { transform: translateY(0); opacity: 0.85; }
    50% { transform: translateY(-20px); opacity: 1; }
    100% { transform: translateY(0); opacity: 0.85; }
}
</style>
<div class="float-container">
    <img src="https://logo.clearbit.com/apple.com" class="float-logo" style="left:5%;">
    <img src="https://logo.clearbit.com/tesla.com" class="float-logo" style="left:15%;">
    <img src="https://logo.clearbit.com/microsoft.com" class="float-logo" style="left:25%;">
    <img src="https://logo.clearbit.com/amazon.com" class="float-logo" style="left:35%;">
    <img src="https://logo.clearbit.com/nvidia.com" class="float-logo" style="left:45%;">
    <img src="https://logo.clearbit.com/google.com" class="float-logo" style="left:55%;">
    <img src="https://logo.clearbit.com/meta.com" class="float-logo" style="left:65%;">
    <img src="https://logo.clearbit.com/netflix.com" class="float-logo" style="left:75%;">
    <img src="https://logo.clearbit.com/berkshirehathaway.com" class="float-logo" style="left:85%;">
    <img src="https://logo.clearbit.com/jpmorganchase.com" class="float-logo" style="left:95%;">
</div>
""", unsafe_allow_html=True)

# === Title & Description ===
st.title("\U0001F4C8 MarketPulse: Seasonal Stock Advisor")
st.markdown("""
**Welcome to MarketPulse.** This tool helps identify the best months to trade a stock, based on historical performance.
Select one or more tickers and a time range to get started.
""")

# === Sidebar Inputs ===
st.sidebar.header("\U0001F50D Stock Selection")
tickers_input = st.sidebar.text_input("Enter stock ticker(s) separated by commas", value="AAPL, MSFT")
start_year = st.sidebar.slider("Start Year", 2000, 2023, 2010)
end_year = st.sidebar.slider("End Year", 2001, 2025, 2024)
min_success_rate = st.sidebar.slider("Minimum Success Rate (%)", 0, 100, 60)

# === Seasonality Analysis Function ===
def analyze_seasonality(ticker, start_year, end_year):
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Return'] = df['Close'].pct_change()

    monthly_avg = df.groupby('Month')['Return'].mean().reset_index()
    monthly_avg['Month Name'] = monthly_avg['Month'].apply(lambda x: calendar.month_abbr[x])
    monthly_avg['Return (%)'] = monthly_avg['Return'] * 100

    monthly_success = df.groupby(['Year', 'Month'])['Return'].sum().reset_index()
    monthly_success['Success'] = monthly_success['Return'] > 0
    success_by_month = monthly_success.groupby('Month')['Success'].mean().reset_index()
    success_by_month['Success Rate (%)'] = success_by_month['Success'] * 100

    result = pd.merge(monthly_avg, success_by_month, on='Month')
    result = result.sort_values(by='Return (%)', ascending=False)
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    return result, buy_hold_return, stock.info

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
