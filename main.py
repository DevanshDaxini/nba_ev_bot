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
        print(sorted_bets[['Player', 'Stat', 'Side', 'Line', 'Implied_Win_%']].head(15).to_string(index=False))
        
        # --- FOLDER LOGIC ---
        output_folder = "program_runs"
        
        # Create the folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"\nCreated new folder: {output_folder}/")

        # Save file inside the folder
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{output_folder}/scan_{date_str}.csv"
        
        sorted_bets.to_csv(filename, index=False)
        print(f"\nSaved {len(sorted_bets)} total lines to '{filename}'")
    else:
        print("No matches found! (Check if player names match exactly)")

if __name__ == "__main__":
    main()