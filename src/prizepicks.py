import requests
import pandas as pd
import json

class PrizePicksClient:
    def __init__(self):
        self.url = "https://api.prizepicks.com/projections"
        # We must look like a real browser, or they will block us
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"+
            " AppleWebKit/537.36 (KHTML, like Gecko)"+
            " Chrome/91.0.4472.124 Safari/537.36"
        }

    def fetch_board(self):
        """
        Fetches the current board and returns a clean DataFrame.
        """
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error connecting to PrizePicks: {e}")
            return pd.DataFrame()

        # --- YOUR HOMEWORK STARTS HERE ---
        
        # 1. Understand the Data Structure
        # data' = The Lines (e.g., "Over 24.5 Points"). 
        # But it doesn't have the player's name!
        # 'included' = The Metadata (Player Names, Team info). 
        # You have to match them using "relationships" -> "new_player" -> "id"
        
        projections_list = data['data']
        included_list = data['included']
        
        # STEP 1: Create a "Lookup Dictionary" for Player Names
        # Goal: { '12345': 'LeBron James', '67890': 'Steph Curry' }
        player_map = {}
        
        for item in included_list:
            # TODO: Write an IF statement to check if item['type'] 
            # is equal to 'new_player'
            # If yes, extract the 'id' and 'attributes' -> 
            # 'name' and save to player_map
            pass # <--- Delete this 'pass' and write your code

        # STEP 2: Parse the Projections
        clean_lines = []
        
        for proj in projections_list:
            # Only look for 'is_promo': false 
            # (unless you want Taco Tuesday discounts)
            if proj['attributes'].get('is_promo') is True:
                continue

            # TODO: Extract the Player ID from: proj['relationships']
            # ['new_player']['data']['id']
            # TODO: Use your player_map to find the actual Name.
            
            # TODO: Extract 'line_score' and 'stat_type' from proj['attributes']
            
            # Append to our list
            # clean_lines.append({
            #    'player': player_name,
            #    'team': ..., # (Optional, if you want to find team info too)
            #    'stat': stat_type,
            #    'line': line_score
            # })
            pass 

        # Return as a DataFrame
        return pd.DataFrame(clean_lines)

# --- TEST BLOCK ---
if __name__ == "__main__":
    client = PrizePicksClient()
    df = client.fetch_board()
    
    if not df.empty:
        print(f"Success! Found {len(df)} lines.")
        # Print just NBA lines to check
        # (PrizePicks usually uses league_id 7 for NBA, 
        # but let's just inspect the first few)
        print(df.head())
    else:
        print("DataFrame is empty. Did you fill in the logic?")