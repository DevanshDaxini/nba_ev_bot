import pandas as pd
import os
from datetime import datetime
from nba_api.stats.endpoints import playergamelogs
from src.config import STAT_MAP

# PrizePicks Name -> NBA API Column Name
NBA_STAT_MAP = {
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Assists': 'AST',
    '3-PT Made': 'FG3M',
    'Blocks': 'BLK',
    'Steals': 'STL',
    'Turnovers': 'TOV',
    'Pts+Rebs+Asts': 'PRA'
}

def get_user_date():
    """Asks the user for a date and validates it."""
    while True:
        date_str = input("\nEnter the date to grade (YYYY-MM-DD) or press Enter for Yesterday: ")
        
        # Default to Yesterday if empty
        if not date_str.strip():
            from datetime import timedelta
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Validate format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            print("Invalid format! Please use YYYY-MM-DD (e.g., 2026-02-05)")

def update_history_file(date_str, wins, losses, total, win_rate):
    """Appends the day's results to a master CSV file."""
    history_file = "program_runs/win_rate_history.csv"
    
    # Prepare the new row
    new_row = {
        "Date": date_str,
        "Total_Bets": total,
        "Wins": wins,
        "Losses": losses,
        "Win_Rate": f"{win_rate:.2f}%",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    df_new = pd.DataFrame([new_row])

    if not os.path.exists(history_file):
        # Create new file with headers
        df_new.to_csv(history_file, index=False)
        print(f"Created new history log: {history_file}")
    else:
        # Append to existing file without writing headers again
        df_new.to_csv(history_file, mode='a', header=False, index=False)
        print(f"Added entry to: {history_file}")

def grade_bets():
    # 1. Ask User for Date
    target_date = get_user_date()
    filename = f"program_runs/scan_{target_date}.csv"
    
    print(f"\n--- GRADING BETS FOR {target_date} ---")
    
    if not os.path.exists(filename):
        print(f"ERROR: No file found at {filename}")
        print("Did you run the scanner on that day?")
        return

    df = pd.read_csv(filename)

    # 2. Get ACTUAL Stats from NBA API
    print("Fetching actual game results from NBA API...")
    # Using '2024-25' season (Change this if grading historical data from older seasons)
    logs = playergamelogs.PlayerGameLogs(season_nullable='2025-26', 
                                         date_from_nullable=target_date, 
                                         date_to_nullable=target_date)
    
    # Check if API returned data
    frames = logs.get_data_frames()
    if not frames:
        print("NBA API returned no data. (Are games finished? Is the date correct?)")
        return
        
    box_scores = frames[0]
    
    # Create look-up dict
    player_stats = {}
    for _, row in box_scores.iterrows():
        name = row['PLAYER_NAME']
        player_stats[name] = row.to_dict()
        player_stats[name]['PRA'] = row['PTS'] + row['REB'] + row['AST']

    print(f"Found stats for {len(player_stats)} players.")

    # 3. Grade the Bets
    wins = 0
    losses = 0
    total_graded = 0
    results = []
    actuals = []

    for index, row in df.iterrows():
        player = row['Player']
        prop = row['Stat']
        line = row['Line']
        side = row['Side']
        
        # Handle Missing Players
        if player not in player_stats:
            results.append("DNP/Unknown")
            actuals.append(0)
            continue
            
        # Get Actual Score
        nba_col = NBA_STAT_MAP.get(prop)
        if not nba_col:
            results.append("Unsupported Stat")
            actuals.append(0)
            continue
            
        actual_val = player_stats[player].get(nba_col, 0)
        actuals.append(actual_val)
        
        # Determine Outcome
        if (side == 'Over' and actual_val > line) or \
           (side == 'Under' and actual_val < line):
            results.append("WIN")
            wins += 1
            total_graded += 1
        elif actual_val == line:
            results.append("Push") # Tie (doesn't count as win or loss)
        else:
            results.append("LOSS")
            losses += 1
            total_graded += 1

    # 4. Save the "Graded" detailed file (Overwrites the daily scan)
    df['Result'] = results
    df['Actual'] = actuals
    df.to_csv(filename, index=False)
    print(f"Updated daily file with results: {filename}")
    
    # 5. Save the "Summary" to history log
    if total_graded > 0:
        win_rate = (wins / total_graded) * 100
        
        print(f"\n--- REPORT CARD ({target_date}) ---")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"WIN RATE: {win_rate:.2f}%")
        
        update_history_file(target_date, wins, losses, total_graded, win_rate)
    else:
        print("No bets could be graded (Check if players played).")

if __name__ == "__main__":
    grade_bets()

# Run this by doing: python3 -m src.grader