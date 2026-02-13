"""
Configuration Constants and API Keys

Central configuration file for the NBA props betting system.
Stores API keys, sport mappings, stat name conversions, and 
PrizePicks breakeven rates.

Key Constants:
    ODDS_API_KEY - The Odds API key (loaded from .env file)
    SPORT_MAP - Maps league names to API sport keys
    STAT_MAP - Converts display names ('Points') to abbreviations ('PTS')
    SLIP_CONFIG - PrizePicks breakeven win rates by parlay type
    
Environment Variables:
    .env file must contain: ODDS_API_KEY=your_key_here
    
SLIP_CONFIG Explanation:
    Each parlay type has a 'hurdle' (required win rate to break even)
    Example: 5_man_flex requires 54.25% to be profitable
    
    Higher hurdle = higher payout but need more confident picks
    Lower hurdle = safer but lower payout
    
Usage:
    from src.config import STAT_MAP, SLIP_CONFIG
"""

import os
from dotenv import load_dotenv

# Load secrets from the .env file
load_dotenv()

# 1. API Configuration
# We now get the key safely from the environment variable
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

# If the key is missing, warn the user (helpful for debugging)
if not ODDS_API_KEY:
    raise ValueError("API Key not found! Make sure you have a .env file with ODDS_API_KEY inside.")

# ... (Rest of your file stays the same: SPORT_MAP, MARKETS, etc.)

# 2. Sport Constants

SPORT_MAP = {
    'NBA': 'basketball_nba',
}

# Looking for all of these markets will make lose API keys fast
"""
SPORT_MAP = {
    'NBA': 'basketball_nba',
    'NHL': 'icehockey_nhl',
    'NFL': 'americanfootball_nfl',
}
"""

REGIONS = 'us'

# Updated and conservative market
MARKETS = (
        'player_points,player_rebounds,'
        'player_assists,player_threes,player_blocks,'
        'player_steals,player_blocks_steals,player_turnovers,'
        'player_points_rebounds_assists,player_points_rebounds,'
        'player_points_assists,player_rebounds_assists,'
        'player_field_goals,player_frees_made,player_frees_attempts,'
)

# Looking for all of these markets will make lose API keys fast
"""
MARKETS = (
    # NBA (Basketball)


    # NHL (Hockey)
    'player_power_play_points,player_blocked_shots,'
    'player_goals,player_total_saves,player_shots_on_goal,'
    
    # NFL (Football)
    'player_pass_attempts,player_pass_completions,player_pass_interceptions,'
    'player_pass_longest_completion,player_pass_rush_yds,player_pass_tds,'
    'player_pass_yds,player_receptions,player_reception_longest,'
    'player_reception_tds,player_reception_yds,player_rush_attempts,'
    'player_rush_longest,player_rush_reception_yds,player_rush_tds,'
    'player_rush_yds,player_sacks,player_solo_tackles,player_tackles_assists,'
    'player_pass_rush_reception_tds,player_pass_rush_reception_yds'
)
"""


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


STAT_MAP = {
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Assists': 'AST',
    '3-PT Made': 'FG3M',
    '3-PT Attempted': 'FG3A',
    'Blocked Shots': 'BLK',
    'Steals': 'STL',
    'Turnovers': 'TOV',
    'FG Made': 'FGM',
    'FG Attempted': 'FGA',
    'Free Throws Made': 'FTM',
    'Free Throws Attempted': 'FTA',
    'Pts+Rebs+Asts': 'PRA',
    'Pts+Rebs': 'PR',
    'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA',
    'Blks+Stls': 'SB'
}

# 4. MODEL QUALITY TIERS (Based on Actual Directional Accuracy)
# Updated: 2026-02-13 from your training results
MODEL_TIERS = {
    'ELITE': {
        'models': ['PTS', 'FGM', 'PA', 'PR', 'PRA'],
        'accuracy_range': '85-90%',
        'edge_threshold': 1.5,  # Minimum edge to recommend
        'description': '⭐ Highest confidence - bet heavily',
        'emoji': '⭐'
    },
    'STRONG': {
        'models': ['FG3A', 'FGA'],
        'accuracy_range': '80-85%',
        'edge_threshold': 2.0,
        'description': '✓ Good confidence - bet selectively',
        'emoji': '✓'
    },
    'DECENT': {
        'models': ['FG3M', 'FTA', 'RA', 'FTM', 'REB', 'AST'],  # Added REB & AST
        'accuracy_range': '72-80%',
        'edge_threshold': 2.5,  # Higher threshold = only show when edge is big
        'description': '~ Moderate confidence - bet carefully',
        'emoji': '~'
    },
    'RISKY': {
        'models': ['STL', 'TOV'],  # Still profitable but volatile
        'accuracy_range': '61-71%',
        'edge_threshold': 3.0,  # Need BIG edge to justify
        'description': '⚠️ High variance - bet only large edges',
        'emoji': '⚠️'
    },
    'AVOID': {
        'models': ['BLK', 'SB'],  # These are genuinely unpredictable
        'accuracy_range': '35-53%',
        'edge_threshold': 10.0,  # Effectively never show unless massive edge
        'description': '❌ Too random - avoid unless huge edge',
        'emoji': '❌'
    }
}

# Create lookup dict for quick tier checking
MODEL_QUALITY = {}
for tier, data in MODEL_TIERS.items():
    for model in data['models']:
        MODEL_QUALITY[model] = {
            'tier': tier,
            'threshold': data['edge_threshold'],
            'emoji': data['emoji']
        }

# 5. SCANNING MODE
# Controls which tiers to scan
# Options: 'ELITE_ONLY', 'SAFE', 'BALANCED', 'AGGRESSIVE', 'ALL'
SCANNING_MODE = 'BALANCED'

SCANNING_MODES = {
    'ELITE_ONLY': MODEL_TIERS['ELITE']['models'],  # Just 5 models
    'SAFE': (MODEL_TIERS['ELITE']['models'] + 
             MODEL_TIERS['STRONG']['models']),  # 7 models
    'BALANCED': (MODEL_TIERS['ELITE']['models'] + 
                 MODEL_TIERS['STRONG']['models'] + 
                 MODEL_TIERS['DECENT']['models']),  # 13 models (includes REB/AST)
    'AGGRESSIVE': (MODEL_TIERS['ELITE']['models'] + 
                   MODEL_TIERS['STRONG']['models'] + 
                   MODEL_TIERS['DECENT']['models'] +
                   MODEL_TIERS['RISKY']['models']),  # 15 models (adds STL/TOV)
    'ALL': list(MODEL_QUALITY.keys())  # All 17 models (includes BLK/SB)
}

# Build active targets based on mode
ACTIVE_TARGETS = SCANNING_MODES.get(SCANNING_MODE, SCANNING_MODES['BALANCED'])

# Print mode info
mode_descriptions = {
    'ELITE_ONLY': "⭐ ELITE ONLY - 5 models (85%+), max accuracy",
    'SAFE': "✓ SAFE MODE - 7 models (80%+), high confidence",
    'BALANCED': "📊 BALANCED - 13 models (72%+), includes REB/AST",
    'AGGRESSIVE': "⚡ AGGRESSIVE - 15 models (61%+), includes STL/TOV", 
    'ALL': "🎲 ALL MODELS - 17 models (includes BLK/SB)"
}

print(f"⚙️  {mode_descriptions.get(SCANNING_MODE, 'UNKNOWN MODE')}")
print(f"   Scanning: {', '.join(ACTIVE_TARGETS)}")
if SCANNING_MODE == 'BALANCED':
    print(f"   Excluded: {', '.join(MODEL_TIERS['RISKY']['models'] + MODEL_TIERS['AVOID']['models'])}")