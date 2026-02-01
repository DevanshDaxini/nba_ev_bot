import requests
import pandas as pd
from src.config import ODDS_API_KEY, SPORT, REGIONS, MARKETS, ODDS_FORMAT

class FanDuelClient:
    def __init__(self):
        self.api_key = ODDS_API_KEY
        # The base endpoint for fetching odds
        self.url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'

    def get_odds(self):
        """
        Fetches odds from The Odds API, 
        filters for FanDuel, and parses the response.
        Returns a DataFrame with columns: 
        [player, market, line, over_price, under_price]
        """
        # 1. Setup Parameters
        params = {
            'apiKey': self.api_key,
            'regions': REGIONS,
            'markets': MARKETS, # e.g. 'player_points'
            'oddsFormat': ODDS_FORMAT,
            'bookmakers': 'fanduel' # STRICTLY filter for FanDuel
        }

        # 2. Make Request
        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching FanDuel odds: {e}")
            return pd.DataFrame()

        clean_data = []

        # --- YOUR HOMEWORK STARTS HERE ---
        # The JSON structure is nested 4 layers deep:
        # Games (List) -> Bookmakers (List) -> 
        # Markets (List) -> Outcomes (List)
        
        # 1. Loop through each Game in 'data'
        for game in data:
            # 2. Loop through 'bookmakers' inside the game
            # (Note: Since we filtered for 'fanduel' in params, 
            # this list should only have 1 item)
            for bookmaker in game['bookmakers']:
                
                # 3. Loop through 'markets' 
                # (e.g., player_points, player_rebounds)
                for market in bookmaker['markets']:
                    market_key = market['key'] # e.g., 'player_points'
                    
                    # 4. Loop through 'outcomes' 
                    # (The actual players and lines)
                    # The Odds API groups them differently 
                    # than you might expect.
                    # Usually, you get a list like: 
                    # {desc: 'LeBron', name: 'Over', point: 24.5}, 
                    # {desc: 'LeBron', name: 'Under', point: 24.5}
                    
                    # You need to group these pairs together so 
                    # you have ONE row per player.
                    # Hint: Use a temporary dictionary to store 
                    # the 'Over' while you wait for the 'Under'.
                    
                    outcomes = market['outcomes']
                    
                    # TODO: Write logic to iterate through 'outcomes'
                    # TODO: Match the Over outcome with the Under 
                    # outcome for the same player & line.
                    # TODO: Append a dictionary to 'clean_data' looking like:
                    # {
                    #   'player': 'LeBron James',
                    #   'market': market_key,
                    #   'line': 24.5,
                    #   'over_price': -115,
                    #   'under_price': -105
                    # }
                    pass 

        return pd.DataFrame(clean_data)

if __name__ == "__main__":
    client = FanDuelClient()
    df = client.get_odds()
    
    if not df.empty:
        print(f"Success! Found {len(df)} lines.")
        print(df.head())
    else:
        print("DataFrame is empty. Did you write the parsing logic?")