import streamlit as st
import pandas as pd
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA AI Scanner", layout="wide")

# --- HEADER ---
st.title("🏀 NBA AI Scanner")
st.markdown("XGBoost Projections vs. PrizePicks Lines")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Configuration")
scan_option = st.sidebar.radio("Select Scan Date:", ["Today's Games", "Tomorrow's Games"])

# Define file paths based on scanner.py output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TODAY_FILE = os.path.join(BASE_DIR, 'data/projections/todays_automated_analysis.csv')
TOMORROW_FILE = os.path.join(BASE_DIR, 'data/projections/tomorrows_automated_analysis.csv')

# Select the file
if scan_option == "Today's Games":
    target_file = TODAY_FILE
    st.subheader("🚀 Scan Results: Today's Games")
else:
    target_file = TOMORROW_FILE
    st.subheader("🔮 Scan Results: Tomorrow's Games")

# --- LOAD DATA ---
if os.path.exists(target_file):
    try:
        # Load CSV
        df = pd.read_csv(target_file)
        
        # --- PRE-PROCESSING ---
        # 1. Filter out rows with no line (PP = 0)
        df = df[df['PP'] > 0]
        
        # 2. Add 'Tier' Logic (Replicating scanner.py)
        elite_stats = ['PTS', 'FGM', 'PA', 'PR', 'PRA']
        strong_stats = ['FG3A', 'FGA']
        decent_stats = ['FG3M', 'FTA']

        def get_tier(row):
            stat = row['TARGET']
            if stat in elite_stats: return '⭐ ELITE'
            if stat in strong_stats: return '✓ STRONG'
            if stat in decent_stats: return '~ DECENT'
            return '⚠ WEAK'

        df['Confidence Tier'] = df.apply(get_tier, axis=1)
        
        # 3. Calculate Edge %
        df['Edge %'] = ((df['AI'] - df['PP']) / df['PP']) * 100
        
        # 4. Determine Pick Type (Over/Under)
        df['Pick'] = df.apply(lambda x: "OVER" if x['AI'] > x['PP'] else "UNDER", axis=1)

        # 5. Reorder Columns for Readability
        # Check if 'REC' column exists (from scanner.py), if not, ignore
        cols_to_show = ['NAME', 'TARGET', 'Confidence Tier', 'Pick', 'AI', 'PP', 'Edge %']
        df_display = df[cols_to_show].copy()

        # Rename for nicer display
        df_display.columns = ['Player', 'Stat', 'Tier', 'Pick', 'AI Proj', 'PP Line', 'Edge %']

        # --- FILTERS ---
        # Tier Filter
        selected_tiers = st.sidebar.multiselect(
            "Filter by Confidence:", 
            options=['⭐ ELITE', '✓ STRONG', '~ DECENT', '⚠ WEAK'],
            default=['⭐ ELITE', '✓ STRONG']
        )
        
        if selected_tiers:
            df_display = df_display[df_display['Tier'].isin(selected_tiers)]

        # --- STYLING & DISPLAY ---
        def color_tiers(val):
            color = 'white'
            if 'ELITE' in val: color = '#90ee90' # Light green
            elif 'STRONG' in val: color = '#add8e6' # Light blue
            elif 'DECENT' in val: color = '#ffffe0' # Light yellow
            elif 'WEAK' in val: color = '#ffcccb' # Light red
            return f'background-color: {color}; color: black'

        def color_pick(val):
            color = '#90ee90' if val == 'OVER' else '#ffcccb'
            return f'color: {color}; font-weight: bold'

        # Sort by absolute Edge %
        df_display['Abs Edge'] = df_display['Edge %'].abs()
        df_display = df_display.sort_values('Abs Edge', ascending=False).drop(columns=['Abs Edge'])

        # Display Dataframe with style
        st.dataframe(
            df_display.style
            .map(color_tiers, subset=['Tier'])
            .map(color_pick, subset=['Pick'])
            .format({"AI Proj": "{:.2f}", "PP Line": "{:.2f}", "Edge %": "{:.1f}%"}),
            use_container_width=True,
            height=600
        )
        
        # Last Updated info
        last_mod = os.path.getmtime(target_file)
        from datetime import datetime
        st.caption(f"Last Updated: {datetime.fromtimestamp(last_mod).strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.warning(f"⚠️ No data found at `{target_file}`.")
    st.info("Please run `python3 main.py` locally and select 'Scan Games', then push the results to GitHub.")

# --- VISUALS SECTION ---
st.divider()
st.subheader("📊 Model Performance")

viz_col1, viz_col2 = st.columns(2)
viz_path = os.path.join(BASE_DIR, 'analysis_plots')

with viz_col1:
    acc_plot = os.path.join(viz_path, 'individual_model_accuracy.png')
    if os.path.exists(acc_plot):
        st.image(acc_plot, caption="Model Accuracy by Stat")
    else:
        st.warning("No accuracy chart found.")

with viz_col2:
    trend_plot = os.path.join(viz_path, 'win_rate_trend.png')
    if os.path.exists(trend_plot):
        st.image(trend_plot, caption="Win Rate Trend")
    else:
        st.warning("No win rate chart found.")