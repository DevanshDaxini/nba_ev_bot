# src/config.py

# 1. API Configuration
# GET THIS KEY HERE: https://the-odds-api.com/ (Click "Get API Key", it's free)
# It will be emailed to you. Paste it inside the quotes below.
# Will get API after the structure for this code is setup
ODDS_API_KEY = 'YOUR_ACTUAL_API_KEY_HERE'

# 2. Sport Constants
SPORT = 'basketball_nba'
REGIONS = 'us'

# CHANGED: We need player props, not game lines
# Note: On the free plan, you might be limited to specific props. 
# We will start with Points, Rebounds, and Assists.
MARKETS = 'player_points,player_rebounds,player_assists' 

ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# 3. PrizePicks Mathematical Hurdles (The "Breakeven" Win Rates)
# This tells the bot: "If FanDuel gives us a 56% chance, is that good enough?"
SLIP_CONFIG = {
    '2_man_power': {'hurdle': 57.74, 'min_odds': -137},
    '3_man_power': {'hurdle': 58.48, 'min_odds': -141},
    '3_man_flex':  {'hurdle': 59.80, 'min_odds': -149},
    '4_man_power': {'hurdle': 56.23, 'min_odds': -128},
    '4_man_flex':  {'hurdle': 56.90, 'min_odds': -132},
    '5_man_power': {'hurdle': 61.00, 'min_odds': -157},
    '5_man_flex':  {'hurdle': 54.25, 'min_odds': -119}, # Best Value usually
    '6_man_power': {'hurdle': 65.00, 'min_odds': -186},
    '6_man_flex':  {'hurdle': 54.21, 'min_odds': -118}, # Best Value usually
}