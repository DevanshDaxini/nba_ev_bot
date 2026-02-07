import os  # <--- Need this to make folders
from datetime import datetime
from src.prizepicks import PrizePicksClient
from src.fanduel import FanDuelClient
from src.analyzer import PropsAnalyzer

def main():
    print("--- 1. Fetching PrizePicks Lines ---")
    pp = PrizePicksClient()
    pp_df = pp.fetch_board()
    print(f"Got {len(pp_df)} PrizePicks props.")

    print("\n--- 2. Fetching FanDuel Odds ---")
    fd = FanDuelClient()
    fd_df = fd.get_all_odds() 
    print(f"Got {len(fd_df)} FanDuel props.")

    print("\n--- 3. Analyzing All Lines (Scanner Mode) ---")
    analyzer = PropsAnalyzer(pp_df, fd_df)
    all_bets = analyzer.calculate_edges()

    if not all_bets.empty:
        # Sort by Win %
        sorted_bets = all_bets.sort_values(by='Implied_Win_%', ascending=False)
        
        print("\nTOP 15 HIGHEST PROBABILITY PLAYS:")
        print(sorted_bets[['Date', 'Player', 'Stat', 'Side', 'Line', 'Implied_Win_%']].head(15).to_string(index=False))
        
        # --- NEW SAVING LOGIC ---
        output_folder = "program_runs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 1. Group by the 'Date' column we just added
        # This splits the big dataframe into mini-dataframes based on game day
        unique_dates = sorted_bets['Date'].unique()
        
        print(f"\n--- Saving Data for {len(unique_dates)}"
              f" Different Game Days ---")

        for game_date in unique_dates:
            # Get only the rows for this specific date
            day_data = sorted_bets[sorted_bets['Date'] == game_date]
            
            # Save as scan_YYYY-MM-DD.csv
            filename = f"{output_folder}/scan_{game_date}.csv"
            
            # If file exists, we might want to overwrite or append. 
            # For now, let's overwrite to keep it fresh.
            day_data.to_csv(filename, index=False)
            print(f" -> Saved {len(day_data)} lines to {filename}")

    else:
        print("No matches found!")

if __name__ == "__main__":
    main()