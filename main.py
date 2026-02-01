import sys
import pandas as pd
from src.prizepicks import PrizePicksClient
from src.fanduel import FanDuelClient
from src.analyzer import PropsAnalyzer

def main():
    print("🏀 NBA +EV Bot Initialized 🏀")
    print("-------------------------------")
    
    # The 'Game Loop' - keeps the program running until you quit
    while True:
        user_input = input("\nType 'scan' to fetch"+
                           " data, or 'q' to quit: ").strip().lower()
        
        if user_input == 'q':
            print("Exiting...")
            break
            
        if user_input == 'scan':
            print("\n1. Fetching PrizePicks Lines...")
            # TODO: Initialize PrizePicksClient and fetch_board()
            
            print("2. Fetching FanDuel Odds...")
            # TODO: Initialize FanDuelClient and get_odds()
            
            # Error Handling: Ensure both 
            # DataFrames have data before proceeding
            # if pp_df.empty or fd_df.empty:
            #     print("Error: Could not fetch data. 
            #           Check API keys/connection.")
            #     continue
            
            print("3. Analyzing Discrepancies...")
            # TODO: Initialize PropsAnalyzer with the two dataframes
            # edges_df = analyzer.calculate_edges()
            
            # TODO: Print the results
            # if not edges_df.empty:
            #     print(edges_df)
            # else:
            #     print("No +EV plays found at this time.")

if __name__ == "__main__":
    main()