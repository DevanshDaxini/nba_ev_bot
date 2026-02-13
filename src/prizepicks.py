"""
PrizePicks API Client

Fetches current prop lines from PrizePicks' public API.
Converts stat names to abbreviations using STAT_MAP for consistency.

API Endpoint:
    https://api.prizepicks.com/projections
    
Returns:
    NBA player props with lines (e.g., LeBron James, Points, 25.5)
    
Usage:
    from src.prizepicks import fetch_current_lines_dict
    lines = fetch_current_lines_dict()
    # {'LeBron James': {'PTS': 25.5, 'REB': 7.5, 'AST': 7.5}}
    
Note:
    - Filters to 'standard' odds_type only (excludes promos)
    - NBA games only (ignores NFL, MLB, etc.)
    - Uses STAT_MAP to convert 'Points' → 'PTS'
"""

import requests
import pandas as pd
import sys
import os

# 1. LINK TO CONFIG
# Add the current directory to path so we can import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import STAT_MAP

class PrizePicksClient:
    def __init__(self):
        self.url = "https://api.prizepicks.com/projections"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://app.prizepicks.com/",
            "Origin": "https://app.prizepicks.com"
        }

    def fetch_board(self):
        """
        Fetch current PrizePicks prop offerings.
        
        Returns:
            pandas.DataFrame: Columns: Player, League, Stat, Line
            
        Filtering:
            - is_promo = False (no promotional lines)
            - odds_type = 'standard' (no demon/goblin lines)
            - League = 'NBA' only
            
        Example:
            df = client.fetch_board()
            print(df.head())
            
                 Player    League    Stat     Line
            0  LeBron James   NBA    Points    25.5
            1  LeBron James   NBA  Rebounds     7.5
            2  Luka Doncic    NBA    Points    33.5
            
        Note:
            Returns empty DataFrame if API fails (doesn't crash)
        """

        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error connecting to PrizePicks: {e}")
            return pd.DataFrame()

        projections_list = data.get('data', [])
        included_list = data.get('included', [])
        
        player_map = {}
        league_map = {}
        
        for item in included_list:
            if item['type'] == 'new_player':
                player_map[str(item['id'])] = item['attributes']['name']
            if item['type'] == 'league':
                league_map[str(item['id'])] = item['attributes']['name']
        
        clean_lines = []
        
        for proj in projections_list:
            attrs = proj['attributes']
            if attrs.get('is_promo') is True: continue
            if attrs.get('odds_type') != 'standard': continue

            # Check relationships exist before accessing
            if 'new_player' not in proj['relationships'] or 'league' not in proj['relationships']:
                continue

            p_id = str(proj['relationships']['new_player']['data']['id'])
            l_id = str(proj['relationships']['league']['data']['id'])
            
            current_league = league_map.get(l_id)
            if current_league != 'NBA': continue 

            clean_lines.append({
                'Player': player_map.get(p_id),
                'League': current_league,
                'Stat': attrs['stat_type'],
                'Line': attrs['line_score']
            })

        return pd.DataFrame(clean_lines)
    
def fetch_current_lines_dict():
    """
    Fetch PrizePicks lines as nested dictionary.
    
    Returns:
        dict: {player_name: {stat_abbr: line_value}}
              Example: {'LeBron James': {'PTS': 25.5, 'REB': 7.5}}
              
    Stat Conversion:
        'Points' → 'PTS'
        'Rebounds' → 'REB'
        'Pts+Rebs+Asts' → 'PRA'
        (Uses STAT_MAP from config.py)
        
    Usage:
        lines = fetch_current_lines_dict()
        lebron_pts = lines.get('LeBron James', {}).get('PTS')
        # 25.5
        
    Note:
        Safe navigation with .get() prevents KeyErrors
        Returns {} if API is down
    """
    
    client = PrizePicksClient()
    df = client.fetch_board()
    
    if df.empty: return {}
    
    lines_dict = {}
    
    for _, row in df.iterrows():
        player = row['Player']
        stat = row['Stat']
        line = row['Line']
        
        if player not in lines_dict:
            lines_dict[player] = {}
        
        # USE THE CONFIG MAP TO CONVERT 'Points' -> 'PTS' IMMEDIATELY
        clean_stat = STAT_MAP.get(stat, stat)
        
        lines_dict[player][clean_stat] = float(line)
        
    return lines_dict

if __name__ == "__main__":
    lines = fetch_current_lines_dict()
    print(f"Fetched {len(lines)} players.")
    # Debug check
    for p, stats in list(lines.items())[:2]:
        print(f"{p}: {stats}")
        