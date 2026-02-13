"""
Quick diagnostic to check what went wrong with features
"""
import pandas as pd

# Load the training data
df = pd.read_csv('data/training_dataset.csv')

print(f"Total features: {len(df.columns)}")
print(f"Total samples: {len(df)}")
print()

# Check for weak model features
weak_model_features = {
    'BLK': ['OPP_RIM_ATTEMPT_RATE', 'FOUL_TROUBLE_RATE', 'POSITION_BLOCK_BASELINE', 'BLOCK_SKILL_ADVANTAGE'],
    'STL': ['OPP_TOV_RATE', 'OPP_TOV_PER_100', 'STEAL_ATTEMPT_RATE', 'POSITION_STEAL_BASELINE'],
    'TOV': ['OPP_PRESSURE_RATE', 'USAGE_SPIKE', 'AST_TO_TOV_SKILL', 'GAME_SCRIPT_RISK'],
    'REB': ['TEAM_OREB_EMPHASIS', 'OPP_REB_WEAKNESS', 'REBOUND_OPPORTUNITY', 'POSITION_REB_BASELINE'],
    'AST': ['TEAMMATE_SHOOTING_L10', 'PLAYMAKER_ROLE', 'ASSIST_OPPORTUNITY', 'POSITION_AST_BASELINE']
}

print("WEAK MODEL FEATURE CHECK:")
print("="*60)
for stat, features in weak_model_features.items():
    print(f"\n{stat} Features:")
    found = 0
    for feat in features:
        if feat in df.columns:
            print(f"  ✅ {feat}")
            found += 1
        else:
            print(f"  ❌ {feat} - MISSING")
    print(f"  Found: {found}/{len(features)}")

# Check rookie features
print("\n" + "="*60)
print("ROOKIE FEATURES:")
rookie_features = ['CAREER_GAMES', 'IS_ROOKIE', 'ROOKIE_VOLATILITY', 'IS_EARLY_SEASON']
for feat in rookie_features:
    if feat in df.columns:
        print(f"  ✅ {feat}")
    else:
        print(f"  ❌ {feat} - MISSING")

# List all columns
print("\n" + "="*60)
print("ALL FEATURES (first 50):")
for i, col in enumerate(df.columns[:50], 1):
    print(f"{i}. {col}")

print(f"\n... and {len(df.columns) - 50} more")