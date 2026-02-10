import pandas as pd
import os
import sys

# --- CONFIGURATION ---
PROJECTIONS_FILE = 'data/todays_projections.csv'

def load_projections():
    if not os.path.exists(PROJECTIONS_FILE):
        print("ERROR: Projections not found. Run scanner.py first.")
        return None
    return pd.read_csv(PROJECTIONS_FILE)

def find_edge():
    df = load_projections()
    if df is None: return

    print("\n🏀 --- NBA AI EDGE FINDER --- 🏀")
    print("Type a player's name to check their projection.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input(">> Enter Player Name (e.g. 'LeBron'): ").strip().lower()
        
        if query == 'exit':
            break
            
        # Fuzzy Search: Find players that contain the search string
        matches = df[df['NAME'].str.lower().str.contains(query)]
        
        if matches.empty:
            print("❌ Player not found in tonight's projections.")
            continue
            
        # If multiple matches (e.g. "Jalen"), ask to clarify or show all
        for _, player in matches.iterrows():
            print(f"\n🔎 FOUND: {player['NAME']} ({player['TEAM']})")
            print("-" * 40)
            
            # Ask for the Line to calculate Edge
            try:
                stat_type = input("   Which stat? (PTS, REB, AST, PRA, SB): ").upper()
                if stat_type not in player:
                    print(f"   ⚠️ Model didn't predict {stat_type}.")
                    continue
                    
                line = float(input(f"   Enter PrizePicks Line for {stat_type}: "))
                
                projection = player[stat_type]
                diff = projection - line
                
                print(f"\n   🔮 Model Projection: {projection}")
                print(f"   📊 Vegas Line:      {line}")
                print(f"   📈 Diff:            {diff:+.1f}")
                
                # Recommendation Logic
                if diff > 1.5:
                    print(f"   ✅ RECOMMENDATION: HAMMER THE OVER (Edge: {diff:.1f})")
                elif diff < -1.5:
                    print(f"   ✅ RECOMMENDATION: HAMMER THE UNDER (Edge: {diff:.1f})")
                else:
                    print(f"   ⚠️ RECOMMENDATION: STAY AWAY (Edge too small)")
                    
            except ValueError:
                print("   ❌ Invalid input. Please enter a number for the line.")
            print("-" * 40)

if __name__ == "__main__":
    find_edge()