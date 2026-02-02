import requests
import pandas as pd
import json

class PrizePicksClient:
    def __init__(self):
        self.url = "https://api.prizepicks.com/projections"
        # We need a stronger disguise to bypass the 403 error
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://app.prizepicks.com/",
            "Origin": "https://app.prizepicks.com"
        }

    def fetch_board(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error connecting to PrizePicks: {e}")
            return pd.DataFrame()

        projections_list = data['data']
        included_list = data['included']
        
        # STEP 1: Create a "Lookup Dictionary" for Player Names
        player_map = {}
        
        for item in included_list:
            # FIX 1: Added underscore ('new_player')
            if item['type'] == 'new_player':
                p_id = str(item['id'])
                player_name = item['attributes']['name']
                # FIX 2: Changed 'player_id' to 'p_id' to match the variable above
                player_map[p_id] = player_name

        print(f"DEBUG: I learned {len(player_map)} player names.")
        
        # STEP 2: Parse the Projections
        clean_lines = []
        
        for proj in projections_list:
            if proj['attributes'].get('is_promo') is True:
                continue
            
            if proj['attributes'].get('odds_type') != 'standard':
                continue
            
            p_id = str(proj['relationships']['new_player']['data']['id'])
            
            current_name = player_map.get(p_id)

            p_line = proj['attributes']['line_score']
            p_stat = proj['attributes']['stat_type']

            my_map = {}
            my_map['ID'] = p_id
            my_map['Player Name'] = current_name
            my_map['Player Stat'] = p_stat
            my_map['Line'] = p_line

            clean_lines.append(my_map)

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