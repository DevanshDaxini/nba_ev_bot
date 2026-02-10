import pandas as pd
import time
import os
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players

# --- CONFIGURATION ---
SEASONS = ['2022-23', '2023-24', '2024-25']
DATA_FOLDER = 'data'
OUTPUT_FILE = f'{DATA_FOLDER}/raw_game_logs.csv'

def fetch_all_game_logs():
    """
    Downloads game logs for ALL active players for the specified seasons.
    This creates the massive dataset needed for training.
    """
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created folder: {DATA_FOLDER}")

    all_logs = []
    
    print(f"--- STARTING HISTORICAL DOWNLOAD ({len(SEASONS)} Seasons) ---")
    print("This may take a few minutes...")

    for season in SEASONS:
        print(f"Fetching logs for season: {season}...")
        try:
            # We use the bulk endpoint to get everyone at once (Much faster)
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                league_id_nullable='00' # NBA
            )
            df = logs.get_data_frames()[0]
            df['SEASON_ID'] = season # Tag the season
            all_logs.append(df)
            print(f" -> Found {len(df)} game rows for {season}")
            
            # Sleep to be nice to the API
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error fetching {season}: {e}")

    # Combine all seasons
    if all_logs:
        master_df = pd.concat(all_logs, ignore_index=True)
        
        # Save to CSV
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS: Saved {len(master_df)} total game rows to {OUTPUT_FILE}")
        print("You can now proceed to Feature Engineering.")
    else:
        print("FAILED: No data found.")

if __name__ == "__main__":
    fetch_all_game_logs()