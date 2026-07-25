import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="IG Beat the Street - Prediction Hub",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 IG Beat the Street: Weekly Market Prediction Engine")
st.markdown("Designed for IG's 6-question weekly prediction window (Saturday 8 AM - Tuesday 4:30 PM).")

# Preset Watchlists by IG Asset Class
ASSET_PRESETS = {
    "IG Gameweek Core": [
        "NVDA", "TSLA", "AMD", "AAPL", "AMZN", "MSFT", "META",
        "BARC.L", "RR.L", "BP.L", "SHEL.L", 
        "GC=F", "CL=F", "GBPUSD=X", "EURUSD=X", "BTC-USD", "^GSPC", "^FTSE"
    ],
    "US Volatility Equities": ["NVDA", "TSLA", "AMD", "PLTR", "AMZN", "NFLX", "COIN"],
    "UK Equities (LSE)": ["BARC.L", "RR.L", "BP.L", "SHEL.L", "LLOY.L", "AZN.L", "VOD.L"],
    "Commodities & Forex": ["GC=F", "CL=F", "NG=F", "HG=F", "GBPUSD=X", "EURUSD=X", "USDJPY=X"]
}

# Sidebar Controls
st.sidebar.header("⚙️ Prediction Settings")
preset_choice = st.sidebar.selectbox("Select Watchlist Group:", list(ASSET_PRESETS.keys()))

default_tickers = ", ".join(ASSET_PRESETS[preset_choice])
user_tickers = st.sidebar.text_area(
    "Asset Tickers (Comma separated):",
    value=default_tickers,
    help="US Equities (NVDA), UK Equities (BARC.L), Commodities (GC=F), Forex (GBPUSD=X), Crypto (BTC-USD)"
)

tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

# Quantitative Analytics Engine
@st.cache_data(ttl=1800)
def analyze_gameweek_assets(symbol_list):
    results = []
    
    for symbol in symbol_list:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if hist.empty or len(hist) < 10:
                continue
                
            price = hist['Close'].iloc[-1]
            
            # Historical Returns
            perf_1d = ((price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            perf_5d = ((price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100
            
            # Annualized 5-Day Volatility Estimate
            daily_returns = hist['Close'].pct_change().dropna()
            daily_std = daily_returns.std()
            est_5d_vol = daily_std * np.sqrt(5) * 100
            
            # Relative Strength Index (RSI - 14 Day)
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Categorize Asset Class
            if "=" in symbol:
                cat = "Commodity / Forex"
            elif "-USD" in symbol:
                cat = "Crypto"
            elif ".L" in symbol:
                cat = "UK Equity"
            elif "^" in symbol:
                cat = "Index"
            else:
                cat = "US Equity"
                
            # Directional Momentum & Reversal Bias
            if rsi < 32 and perf_5d < -3.0:
                signal = "🚀 Strong Bullish Reversal"
                score = 2
            elif rsi > 68 and perf_5d > 4.0:
                signal = "🔻 Strong Bearish Reversal"
                score = -2
            elif perf_5d > 1.5:
                signal = "🟢 Upward Momentum"
                score = 1
            elif perf_5d < -1.5:
                signal = "🔴 Downward Momentum"
                score = -1
            else:
                signal = "⚪ Neutral / Consolidation"
                score = 0
                
            results.append({
                "Ticker": symbol,
                "Category": cat,
                "Price": round(price, 2),
                "1D %": round(perf_1d, 2),
                "5D %": round(perf_5d, 2),
                "Est. 5D Volatility %": round(est_5d_vol, 2),
                "RSI (14D)": round(rsi, 1),
                "Gameweek Signal": signal,
                "Score": score,
                "History": hist
            })
        except Exception as e:
            st.warning(f"Could not load data for {symbol}: {e}")
            
    return pd.DataFrame(results)

# Run Button Trigger
if st.sidebar.button("⚡ Generate Weekly Predictions", type="primary"):
    with st.spinner("Processing market data & running volatility models..."):
        df = analyze_gameweek_assets(tickers)
        
    if not df.empty:
        # Sort by expected volatility
        df_sorted = df.sort_values(by="Est. 5D Volatility %", ascending=False)
        
        # Key Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        top_vol = df_sorted.iloc[0]
        col1.metric("Total Assets Analyzed", len(df_sorted))
        col2.metric("Highest Volatility Asset", f"{top_vol['Ticker']} ({top_vol['Est. 5D Volatility %']}%)")
        col3.metric("Bullish Signals", len(df_sorted[df_sorted['Score'] > 0]))
        col4.metric("Bearish Signals", len(df_sorted[df_sorted['Score'] < 0]))
        
        st.markdown("---")
        
        # Leaderboard Table
        st.subheader("📊 Ranked Gameweek Movers")
        st.dataframe(
            df_sorted.drop(columns=['History', 'Score']),
            column_config={
                "Est. 5D Volatility %": st.column_config.ProgressColumn(
                    "Est. 5D Volatility %",
                    format="%.2f%%",
                    min_value=0,
                    max_value=float(df_sorted['Est. 5D Volatility %'].max() or 10.0)
                )
            },
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Single Asset Interactive Charting
        st.subheader("🔍 Deep Dive Asset Inspection")
        selected_ticker = st.selectbox("Select Asset to Chart:", df_sorted['Ticker'].tolist())
        
        asset_data = df_sorted[df_sorted['Ticker'] == selected_ticker].iloc[0]
        hist = asset_data['History']
        
        # Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=selected_ticker
        )])
        
        fig.update_layout(
            title=f"{selected_ticker} - 30 Day Price & Volatility Trend",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=400
        )
        
        c_chart, c_stats = st.columns([2, 1])
        with c_chart:
            st.plotly_chart(fig, use_container_width=True)
            
        with c_stats:
            st.markdown(f"### {selected_ticker} Profile")
            st.write(f"**Asset Category:** {asset_data['Category']}")
            st.write(f"**Current Price:** {asset_data['Price']}")
            st.write(f"**5-Day Volatility Risk:** `{asset_data['Est. 5D Volatility %']}%`")
            st.write(f"**14-Day Relative Strength (RSI):** `{asset_data['RSI (14D)']}`")
            st.info(f"**Model Signal:**\n{asset_data['Gameweek Signal']}")
else:
    st.info("Select your watchlist in the sidebar and click **⚡ Generate Weekly Predictions**.")