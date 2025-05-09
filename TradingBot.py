# Install required libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import calendar
import feedparser  # For pulling recent stock news
import streamlit.components.v1 as components

# Set streamlit page configuration
st.set_page_config(page_title="Seasonal Stock Strategy", layout="wide")

# Floating logo animation at top of homepage
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

# App Title and Description
st.title("\U0001F4C8 Seasonal Stock Strategy Analyzer")
st.markdown("""
Welcome to your automated seasonal stock planner.  
Select one or more stock tickers and see when is historically the **best time to buy and sell** for maximum returns.
""")

# Sidebar for user input
st.sidebar.header("\U0001F50D Stock Selection")
tickers_input = st.sidebar.text_input("Enter stock ticker(s) separated by commas", value="AAPL, MSFT")

# Year range selection
start_year = st.sidebar.slider("Start Year", min_value=2000, max_value=2023, value=2010)
end_year = st.sidebar.slider("End Year", min_value=2001, max_value=2025, value=2024)

# Minimum success rate threshold (filter out weak months)
min_success_rate = st.sidebar.slider("Minimum Success Rate (%)", min_value=0, max_value=100, value=60)

# Define a function that analyzes seasonal performance of a given stock
def analyze_seasonality(ticker, start_year, end_year):
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Return'] = df['Close'].pct_change()

    monthly_avg_returns = df.groupby('Month')['Return'].mean().reset_index()
    monthly_avg_returns['Month Name'] = monthly_avg_returns['Month'].apply(lambda x: calendar.month_abbr[x])
    monthly_avg_returns['Return (%)'] = monthly_avg_returns['Return'] * 100

    success_rate = df.groupby(['Year', 'Month'])['Return'].sum().reset_index()
    success_rate['Success'] = success_rate['Return'] > 0
    success_by_month = success_rate.groupby('Month')['Success'].mean().reset_index()
    success_by_month['Success Rate (%)'] = success_by_month['Success'] * 100

    result = pd.merge(monthly_avg_returns, success_by_month, on='Month')
    result = result.sort_values(by='Return (%)', ascending=False)
    buy_and_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    return result, buy_and_hold_return, stock.info

# Define a function to fetch latest news headlines from Yahoo Finance RSS feed
def fetch_news(ticker):
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(feed_url)
    return feed.entries[:5]

# Submit button
if st.sidebar.button("Run Analysis"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    for ticker in tickers:
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analysis", "News", "Report Preview"])

        with tab1:
            st.subheader(f"\U0001F4C8 Company Overview: {ticker}")
            try:
                _, _, info = analyze_seasonality(ticker, start_year, end_year)
                st.markdown(f"**Company Name:** {info.get('longName', 'N/A')}")
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
                    st.warning(f"⚠️ Historically underperforming months: {', '.join(danger_months)}")
                st.markdown(f"📌 **Buy & Hold Return (Full Period):** {buy_hold:.2f}%")
                best_months = data.head(3)
                st.markdown(f"**Best months to buy {ticker}:** {', '.join(best_months['Month Name'].values)}")
                st.markdown(f"**Success rate range:** {best_months['Success Rate (%)'].min():.1f}% to {best_months['Success Rate (%)'].max():.1f}%")

                fig = px.bar(
                    data, x='Month Name', y='Return (%)', color='Success Rate (%)',
                    color_continuous_scale='Blues',
                    title=f'Seasonal Return Analysis for {ticker}',
                    labels={'Return (%)': 'Avg Monthly Return', 'Month Name': 'Month'},
                    hover_data={'Success Rate (%)': ':.2f'}
                )
                fig.update_layout(
                    title_font_size=22,
                    font=dict(size=14),
                    plot_bgcolor='white',
                    xaxis=dict(title='Month', showgrid=False),
                    yaxis=dict(title='Average Return (%)', showgrid=True),
                    coloraxis_colorbar=dict(title='Success Rate (%)')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(data[['Month Name', 'Return (%)', 'Success Rate (%)']].reset_index(drop=True))
            except Exception as e:
                st.error(f"Failed to analyze {ticker}: {str(e)}")

        with tab3:
            st.subheader(f"\U0001F4F0 Latest News for {ticker}")
            try:
                news = fetch_news(ticker)
                for entry in news:
                    st.markdown(f"**{entry.title}**")
                    st.markdown(f"[{entry.link}]({entry.link})")
                    st.markdown("---")
            except:
                st.info("No recent news available or failed to fetch news.")

        with tab4:
            st.subheader("\U0001F4C4 Report Preview")
            st.markdown("""
            This tab will soon allow you to generate a printable trading plan.
            It will include:
            - Top 3 recommended months
            - Success rate & average return
            - Visual charts
            - Trading instructions
            - Your branding & disclaimers
            """)
