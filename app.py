import streamlit as st
import pandas as pd
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA AI Scanner", layout="wide")
st.title("🤖 NBA AI Scanner")

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TODAY_FILE = os.path.join(BASE_DIR, 'data/projections/todays_automated_analysis.csv')
TOMORROW_FILE = os.path.join(BASE_DIR, 'data/projections/tomorrows_automated_analysis.csv')

# --- HELPER FUNCTIONS ---
def load_scan_data(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        # Add Tiers
        elite = ['PTS', 'FGM', 'PA', 'PR', 'PRA']
        strong = ['FG3A', 'FGA']
        
        def get_tier(row):
            s = row['TARGET']
            if s in elite: return '⭐ ELITE'
            if s in strong: return '✓ STRONG'
            return '~ DECENT'
            
        df['Tier'] = df.apply(get_tier, axis=1)
        df['Edge %'] = ((df['AI'] - df['PP']) / df['PP']) * 100
        return df
    except:
        return None

def show_scan_table(df):
    """Displays the standard High Value Plays table"""
    if df is None:
        st.warning("⚠️ No data available for this scan.")
        return

    # Filter for active lines only
    df_active = df[df['PP'] > 0].copy()
    
    # Sidebar Filters
    tiers = st.multiselect("Filter Tiers", options=['⭐ ELITE', '✓ STRONG', '~ DECENT'], default=['⭐ ELITE', '✓ STRONG'], key="tier_filter")
    if tiers:
        df_active = df_active[df_active['Tier'].isin(tiers)]

    # Sort by Edge
    df_active['Abs Edge'] = df_active['Edge %'].abs()
    df_active = df_active.sort_values('Abs Edge', ascending=False)
    
    # Style
    cols = ['NAME', 'TARGET', 'Tier', 'REC', 'AI', 'PP', 'Edge %', 'MATCHUP']
    st.dataframe(
        df_active[cols].style.format({"AI": "{:.2f}", "PP": "{:.2f}", "Edge %": "{:.1f}%"}),
        use_container_width=True,
        height=600
    )

def show_player_scout(df_today, df_tomorrow):
    """Interactive Player Scout Tab"""
    st.subheader("🔎 Scout Specific Player")
    
    # Combine today and tomorrow for the search box
    combined_df = pd.concat([df_today, df_tomorrow], ignore_index=True) if (df_today is not None and df_tomorrow is not None) else (df_today if df_today is not None else df_tomorrow)
    
    if combined_df is None or combined_df.empty:
        st.error("No player data loaded to scout.")
        return

    # 1. Search Box
    all_players = sorted(combined_df['NAME'].unique())
    selected_player = st.selectbox("Select Player:", all_players)

    if selected_player:
        # Get player data (prefer Today's data if available, else Tomorrow)
        player_data = combined_df[combined_df['NAME'] == selected_player]
        
        # Extract Context (Team, Injury, etc.) from the first row
        info = player_data.iloc[0]
        
        # --- PLAYER CARD ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Team", info.get('TEAM', 'N/A'))
        with col2:
            st.metric("Matchup", info.get('MATCHUP', 'N/A'))
        with col3:
            # Color code injury impact
            usage = info.get('MISSING_USAGE', 0)
            delta_color = "inverse" if usage > 15 else "off"
            st.metric("Team Injury Impact", f"{usage}% Usage Missing", delta_color=delta_color)

        st.divider()
        
        # --- PROJECTIONS TABLE ---
        st.write(f"**AI Projections for {selected_player}**")
        
        # Show all stats, even those without lines
        display_cols = ['TARGET', 'Tier', 'AI', 'PP', 'REC']
        st.dataframe(
            player_data[display_cols].sort_values('TARGET').style.format({"AI": "{:.2f}", "PP": "{:.2f}"}),
            use_container_width=True
        )

# --- MAIN LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🚀 Today's Scan", "🔮 Tomorrow's Scan", "🔎 Player Scout"])

df_today = load_scan_data(TODAY_FILE)
df_tomorrow = load_scan_data(TOMORROW_FILE)

with tab1:
    st.header("Today's High-Value Plays")
    show_scan_table(df_today)

with tab2:
    st.header("Tomorrow's Opportunities")
    show_scan_table(df_tomorrow)

with tab3:
    show_player_scout(df_today, df_tomorrow)