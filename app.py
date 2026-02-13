import streamlit as st
import pandas as pd
import os

# Page Config (Title, layout, etc.)
st.set_page_config(page_title="NBA AI Bettor", layout="wide")

# Title
st.title("🏀 NBA Prop Correlation & Projection Engine")
st.write("AI-Powered bets combining Market Odds + XGBoost Projections")

# --- 1. LOAD DATA ---
# We check if the 'program_runs' folder has the latest scan
scan_file = "program_runs/correlated_plays.csv"

if os.path.exists(scan_file):
    df = pd.read_csv(scan_file)
    
    # --- 2. FILTERS (Mobile Friendly) ---
    # Create a sidebar for filters
    st.sidebar.header("Filter Plays")
    
    # Tier Filter
    if 'Confidence_Tier' in df.columns:
        tiers = df['Confidence_Tier'].unique()
        selected_tiers = st.sidebar.multiselect("Select Tiers", tiers, default=tiers)
        df = df[df['Confidence_Tier'].isin(selected_tiers)]
    
    # Bookmaker Filter
    if 'Bookmaker' in df.columns:
        books = df['Bookmaker'].unique()
        selected_books = st.sidebar.multiselect("Sportsbook", books, default=books)
        df = df[df['Bookmaker'].isin(selected_books)]

    # --- 3. DISPLAY DATA ---
    st.subheader(f"Found {len(df)} High-Value Plays")
    
    # Display the dataframe (Streamlit makes this interactive/scrollable)
    st.dataframe(df, use_container_width=True, height=600)

    # --- 4. DOWNLOAD BUTTON ---
    # Allow you to download the CSV to your phone
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Plays as CSV",
        csv,
        "nba_picks.csv",
        "text/csv",
        key='download-csv'
    )
    
else:
    st.error("⚠️ No scan data found. Please run the scanner locally and push the results.")

# --- 5. VISUALIZATIONS TAB ---
st.divider()
st.subheader("📊 Model Accuracy")

# Display the charts we generated earlier
col1, col2 = st.columns(2)

with col1:
    if os.path.exists("analysis_plots/individual_model_accuracy.png"):
        st.image("analysis_plots/individual_model_accuracy.png", caption="Model Accuracy by Stat")
    else:
        st.info("Run visualizer.py to generate accuracy charts.")

with col2:
    if os.path.exists("analysis_plots/win_rate_trend.png"):
        st.image("analysis_plots/win_rate_trend.png", caption="Win Rate Trend")