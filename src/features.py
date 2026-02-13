"""
NBA Statistical Feature Engineering Pipeline

Transforms raw game logs into a rich feature set for machine learning models.
Creates 80+ predictive features including rolling averages, defensive matchups,
fatigue indicators, pace adjustments, and teammate injury impact.

Feature Categories:
    1. Rolling Averages (L5, L20, Season) - Recent performance trends
    2. Advanced Stats (TS%, Usage Rate, Game Score) - Efficiency metrics
    3. Context (Home/Away, Rest Days, B2B flags) - Game circumstances
    4. Defense vs Position (OPP_PTS_ALLOWED, etc.) - Matchup difficulty
    5. Injury Impact (MISSING_USAGE) - Teammate availability
    6. Schedule Density (GAMES_7D, IS_4_IN_6) - Fatigue indicators
    7. Pace (PACE_ROLLING) - Team tempo adjustments
    8. Efficiency Signals (FGA_PER_MIN, TS_EFFICIENCY_GAP) - Skill vs luck

Pipeline:
    1. load_and_merge_data() - Load raw logs + positions
    2. add_advanced_stats() - Calculate TS%, Usage, Game Score
    3. add_context_features() - Home/Away, Rest, Opponent
    4. add_missing_player_context() - Injury simulation
    5. add_schedule_density() - Fatigue tracking
    6. add_pace_features() - Tempo adjustments
    7. ensure_combo_stats() - Create PRA, PR, PA, RA, SB
    8. add_rolling_features() - Historical averages
    9. add_efficiency_signals() - Regression indicators
    10. add_defense_vs_position() - Matchup difficulty
    11. add_usage_vacuum_features() - Opportunity creation

Output:
    data/training_dataset.csv - Ready for XGBoost training

Usage:
    $ python3 -m src.features
"""

import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
LOGS_FILE = 'data/raw_game_logs.csv'
POS_FILE = 'data/player_positions.csv'
OUTPUT_FILE = 'data/training_dataset.csv'

# The specific stats we want to predict (Target Variables)
TARGET_STATS = ['PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'STL', 
                'BLK', 'TOV', 'FGM', 'FGA', 'FTM', 'FTA']


def load_and_merge_data():
    """
    Load raw game logs and merge with player position data.
    
    Steps:
        1. Read data/raw_game_logs.csv (from builder.py)
        2. Read data/player_positions.csv
        3. Merge on PLAYER_ID (left join to keep all games)
        4. Fill missing positions with 'Unknown'
        5. Drop games with missing MATCHUP or GAME_DATE
        6. Sort by PLAYER_ID and GAME_DATE ascending
        
    Returns:
        pandas.DataFrame: Merged dataset with POSITION column added,
                         or None if files don't exist
                         
    Raises:
        FileNotFoundError: If raw_game_logs.csv or player_positions.csv missing
        
    Note:
        Only merges PLAYER_ID and POSITION to avoid duplicate columns
        (prevents PLAYER_NAME_x and PLAYER_NAME_y issues)
    """

    print("...Loading and Merging Data")
    
    if not os.path.exists(POS_FILE) or not os.path.exists(LOGS_FILE):
        print("Error: Data files not found please run builder.py first")
        return None
    
    df_logs = pd.read_csv(LOGS_FILE)
    df_pos = pd.read_csv(POS_FILE)

    # FIX: Only merge the columns we need. 
    # If we merge the whole file, we get duplicate columns like 
    # 'PLAYER_NAME_x' and 'PLAYER_NAME_y'
    df = pd.merge(df_logs, df_pos[['PLAYER_ID', 'POSITION']], 
                  on='PLAYER_ID', how='left')

    # Fill missing values
    df['POSITION'] = df['POSITION'].fillna('Unknown')
    
    # Drop rows where vital data (like MATCHUP) is missing
    df = df.dropna(subset=['MATCHUP', 'GAME_DATE'])

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'], ascending=True)
    
    return df

def add_advanced_stats(df):
    """
    Calculate advanced efficiency metrics.
    
    Formulas:
        TS% (True Shooting %): PTS / (2 * (FGA + 0.44 * FTA))
        Usage Rate: 100 * ((FGA + 0.44*FTA + TOV) / (MIN + 0.1))
        Game Score: Complex formula weighing all box score stats
        
    Args:
        df (pandas.DataFrame): Raw game logs
        
    Returns:
        pandas.DataFrame: Input df with 3 new columns added:
            - TS_PCT: Shooting efficiency (0 to ~0.75)
            - USAGE_RATE: Possession usage % (0 to ~50)
            - GAME_SCORE: Overall performance metric
            
    Note:
        Fills NaN values with 0 (handles division by zero cases)
    """

    print("...Calculating Advanced Stats")
    
    # Existing metrics
    df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    df['TS_PCT'] = df['TS_PCT'].fillna(0)

    df['USAGE_RATE'] = 100 * ((df['FGA'] + 0.44 * 
                               df['FTA'] + df['TOV'])) / (df['MIN'] + 0.1)
    df['USAGE_RATE'] = df['USAGE_RATE'].fillna(0)

    # Existing Game Score
    df['GAME_SCORE'] = (df['PTS'] + (0.4 * df['FGM']) - (0.7 * df['FGA']) - 
                        (0.4 * (df['FTA'] - df['FTM'])) + (0.7 * df['OREB']) + 
                        (0.3 * df['DREB']) + df['STL'] + (0.7 * df['AST']) + 
                        (0.7 * df['BLK']) - (0.4 * df['PF']) - df['TOV'])
    df['GAME_SCORE'] = df['GAME_SCORE'].fillna(0)

    return df

def add_rolling_features(df):
    """
    Docstring for add_rolling_features
    
    :param df: Description
    :return: DataFrame with rolling features added
    :rtype: pandas.DataFrame
    
    Note:
        - Creates L5, L20, and Season rolling averages for key stats
        - Uses groupby + transform to avoid fragmentation and performance issues
        - Shifts by 1 to prevent data leakage (only uses past games)
    """

    print("...Calculating Rolling Averages")
    df = df.copy() # Prevent fragmentation
    grouped = df.groupby('PLAYER_ID')
    
    # FIX #1: Only create rolling features for base stats that drive predictions
    # Don't create rolling features for FG3A, FGA, FTM, FTA (they are targets)
    base_stats = ['PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV', 
                  'FGM', 'FTM']
    stats_to_roll = base_stats + ['MIN', 'GAME_SCORE', 'USAGE_RATE']
    
    # Add combo stats if they exist (created before this function)
    combo_stats = ['PRA', 'PR', 'PA', 'RA', 'SB']
    for combo in combo_stats:
        if combo in df.columns:
            stats_to_roll.append(combo)
    
    rolling_data = {}
    for stat in stats_to_roll:
        rolling_data[f'{stat}_L5'] = grouped[stat].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        rolling_data[f'{stat}_L20'] = grouped[stat].transform(
            lambda x: x.shift(1).rolling(20, min_periods=1).mean())
        rolling_data[f'{stat}_Season'] = grouped[stat].transform(
            lambda x: x.shift(1).expanding().mean())

    df = pd.concat([df, pd.DataFrame(rolling_data, index=df.index)], axis=1)
    return df

def add_context_features(df):
    """
    Docstring for add_context_features
    
    :param df: Description
    :return: Description
    :rtype: Any
    
    Note:
        - Adds Home/Away flag based on 'vs.' in MATCHUP
        - Extracts Opponent team code from MATCHUP
        - Calculates Rest Days and flags for Back-to-Back and Freshness
    """

    print("...Adding Context Features")

    # 1. Home/Away
    # Ensure it's a string before checking 'vs.'
    df['IS_HOME'] = df['MATCHUP'].astype(str).apply(
        lambda x: 1 if 'vs.' in x else 0)
    
    # 2. Opponent Code
    # FIX: Convert to string first so 'NaN' doesn't crash the .split() method
    df['OPPONENT'] = df['MATCHUP'].astype(str).apply(lambda x: x.split(' ')[-1])
    
    # 3. Rest Days calculation
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(upper=7)
    
    # NEW: Categorical "Rest" flags
    # Is it the second night of a back-to-back? (0 or 1)
    df['IS_B2B'] = (df['DAYS_REST'] == 1).astype(int)
    
    # Is it "heavy rest" (3+ days)? (0 or 1)
    df['IS_FRESH'] = (df['DAYS_REST'] >= 3).astype(int)
    
    return df

def add_defense_vs_position(df):
    """
    Docstring for add_defense_vs_position
    
    :param df: Description
    :return: Description
    :rtype: Any

    Note:
        - Calculates opponent's average allowed stats to that player's position
        - Uses expanding mean to prevent data leakage (only past games)
        - Fills early season gaps with global position averages
        - Adds defensive features for combo stats like PRA, PR, etc.
    """

    print("...Calculating Defense vs. Position")
    # De-fragment the frame immediately
    df = df.copy() 
    
    defense_group = df.groupby(['OPPONENT', 'POSITION'])
    new_def_cols = {} # Temporary storage for new columns
    
    for stat in TARGET_STATS:
        col_name = f'OPP_{stat}_ALLOWED'
        # Calculate expanding mean
        new_def_cols[col_name] = defense_group[stat].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        
    # Join all defensive columns at once to prevent fragmentation
    df = pd.concat([df, pd.DataFrame(new_def_cols, index=df.index)], axis=1)

    # Fill gaps with global position average
    for stat in TARGET_STATS:
        col_name = f'OPP_{stat}_ALLOWED'
        global_pos_avg = df.groupby('POSITION')[stat].transform('mean')
        df[col_name] = df[col_name].fillna(global_pos_avg)
    
    # FIX #3: Add defensive features for combo stats
    if 'OPP_PTS_ALLOWED' in df.columns and 'OPP_REB_ALLOWED' in df.columns:
        df['OPP_PRA_ALLOWED'] = df['OPP_PTS_ALLOWED'] + df['OPP_REB_ALLOWED'] + df['OPP_AST_ALLOWED']
        df['OPP_PR_ALLOWED'] = df['OPP_PTS_ALLOWED'] + df['OPP_REB_ALLOWED']
        df['OPP_PA_ALLOWED'] = df['OPP_PTS_ALLOWED'] + df['OPP_AST_ALLOWED']
        df['OPP_RA_ALLOWED'] = df['OPP_REB_ALLOWED'] + df['OPP_AST_ALLOWED']
        df['OPP_SB_ALLOWED'] = df['OPP_STL_ALLOWED'] + df['OPP_BLK_ALLOWED']
    
    return df

def add_usage_vacuum_features(df):
    """
    Docstring for add_usage_vacuum_features
    
    :param df: Description
    :return: Description
    :rtype: Any
    Note:
        - Calculates the "usage vacuum" by identifying games 
            where key players are absent
        - Uses season average usage to identify star players (>18% usage)
        - Sums the missing usage for absent stars to create 
            MISSING_USAGE feature
        - This simulates the opportunity created by injuries to key teammates
    """

    print("...Calculating Usage Vacuum")
    # De-fragment to resolve PerformanceWarning on line 139
    df = df.copy() 

    # Identify stars based on Season average usage
    usage_col = 'USAGE_RATE_Season' if 'USAGE_RATE_Season' in df.columns else 'USAGE_RATE'
    stars = df[df[usage_col] > 28][['PLAYER_ID', 'GAME_ID', 'TEAM_ID']].copy()
    
    star_games = stars.groupby(['GAME_ID', 
                                'TEAM_ID'])['PLAYER_ID'].count().reset_index()
    star_games.columns = ['GAME_ID', 'TEAM_ID', 'STAR_COUNT']

    df = df.merge(star_games, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['STAR_COUNT'] = df['STAR_COUNT'].fillna(0)

    team_avg_stars = df.groupby('TEAM_ID')['STAR_COUNT'].transform('mean')
    df['USAGE_VACUUM'] = (team_avg_stars - df['STAR_COUNT']).clip(lower=0)
    
    return df

def add_injury_context(df):
    """
    Docstring for add_injury_context
    
    :param df: Description
    :return: Description
    :rtype: Any
    
    Note:
        - Detects players returning from an absence 
            (e.g., more than 10 days since last game)
        - Captures the 'rust' factor of coming back from an injury
        - Tracks performance in the first 3 games back 
            (Optional: requires a rolling count of games played)
    """

    # Detect players returning from an absence 
    # (e.g., more than 10 days since last game)
    # This captures the 'rust' factor of coming back from an injury
    df['IS_RETURNING_FROM_INJURY'] = (df['DAYS_REST'] > 10).astype(int)
    
    # Track performance in the first 3 games back
    # (Optional: requires a rolling count of games played)
    return df

def add_missing_player_context(df):
    """
    Docstring for add_missing_player_context
    
    :param df: Description
    :return: Description
    :rtype: Any

    Note:
        - Simulates the impact of missing key players (e.g., due to injury)
        - Identifies "star" players based on season average usage (>18%)
        - Sums the usage of absent stars for each game to create 
            MISSING_USAGE feature
        - This feature captures the opportunity created for remaining 
            players when stars are out
    """

    print("...Calculating Missing Player Impact (Injury Simulation)")
    df = df.copy()

    # 1. Identify Key Players (Usage > 18%) based on Season Average
    season_stats = df.groupby(['SEASON_ID', 
                               'TEAM_ID', 
                               'PLAYER_ID'])['USAGE_RATE'].mean().reset_index()
    key_players = season_stats[season_stats['USAGE_RATE'] > 18.0]

    # 2. Build 'Expected Roster' (All games * All key players for that team)
    team_games = df[['SEASON_ID', 'TEAM_ID', 'GAME_ID']].drop_duplicates()
    expected = team_games.merge(key_players, on=[
                'SEASON_ID', 'TEAM_ID'], how='left')

    # 3. Build 'Actual Roster' (Who actually logged minutes)
    actual = df[['GAME_ID', 'PLAYER_ID']].drop_duplicates()
    actual['PLAYED'] = True

    # 4. Find Missing Stars
    merged = expected.merge(actual, on=['GAME_ID', 'PLAYER_ID'], how='left')
    missing = merged[merged['PLAYED'].isna()]

    # 5. Sum the missing usage per game
    missing_usage = missing.groupby(['GAME_ID', 
                                     'TEAM_ID'])['USAGE_RATE'].sum().reset_index()
    missing_usage.rename(columns={'USAGE_RATE': 'MISSING_USAGE'}, inplace=True)

    # 6. Merge back
    df = df.merge(missing_usage, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['MISSING_USAGE'] = df['MISSING_USAGE'].fillna(0)
    
    return df

def add_schedule_stress(df):
    """
    Docstring for add_schedule_stress
    
    :param df: Description
    :return: Description
    :rtype: Any

    Note:
        - Captures the impact of a dense schedule on player performance
        - Uses a rolling 7-day window to count games played
        - Flags players with 4+ games in 7 days as "Heavy Schedule"
        - This feature helps the model understand fatigue effects from 
            tight scheduling
    """

    df = df.copy()
    grouped = df.groupby('PLAYER_ID')
    
    # Games in the last 7 days
    # (Note: This requires your dataframe to have every day, 
    # or you can use a rolling sum of 'count' over 7-day windows)
    df['GAMES_LAST_7D'] = grouped['GAME_DATE'].transform(
        lambda x: x.rolling('7D', on=x).count()
    )
    
    # Binary flag for heavy schedule (4+ games in a week)
    df['HEAVY_SCHEDULE'] = (df['GAMES_LAST_7D'] >= 4).astype(int)
    
    return df

def add_schedule_density(df):
    """
    Docstring for add_schedule_density
    
    :param df: Description
    :return: Description
    :rtype: Any

    Note:
        - Captures the impact of a dense schedule on player performance
        - Uses a rolling 7-day window to count games played
        - Flags players with 4+ games in 7 days as "IS_4_IN_6"
        - This feature helps the model understand fatigue effects from 
            tight scheduling
    """

    print("...Calculating Schedule Density")
    df = df.copy()
    
    # Ensure date is datetime and sorted
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    def get_rolling_count(group):
        # We need the date as the index for rolling('7D') to work
        temp_series = pd.Series(1, index=group['GAME_DATE'])
        return temp_series.rolling('7D').count().values

    # Apply the logic to each player
    df['GAMES_7D'] = df.groupby('PLAYER_ID', 
                                group_keys=False).apply(
                                    get_rolling_count).explode().values
    
    # Cast to float/int to ensure it's usable for training
    df['GAMES_7D'] = df['GAMES_7D'].astype(float)
    
    # 4-in-6 nights flag (Fatigue 'Red Alert')
    df['IS_4_IN_6'] = (df['GAMES_7D'] >= 4).astype(int)
    
    return df

def add_pace_features(df):
    """
    Docstring for add_pace_features
    
    :param df: Description
    :return: Description
    :rtype: Any

    Note:
        - Calculates team pace based on estimated possessions
        - Uses a rolling 10-game window to smooth pace
        - This feature helps the model understand the tempo of the game
          and adjust predictions accordingly (e.g., 
          more possessions = more stats)
    """
    print("...Calculating Team Pace")
    # Estimated possessions: FGA + 0.44*FTA - OREB + TOV
    df['POSS_EST'] = df['FGA'] + (0.44 * df['FTA']) - df['OREB'] + df['TOV']
    
    # Rolling 10-game team pace
    team_pace = df.groupby(['TEAM_ID', 'GAME_ID'])['POSS_EST'].sum().reset_index()
    team_pace['PACE_ROLLING'] = team_pace.groupby('TEAM_ID')['POSS_EST'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    
    # Merge back to main dataframe
    df = df.merge(team_pace[['GAME_ID', 'TEAM_ID', 'PACE_ROLLING']], 
                  on=['GAME_ID', 'TEAM_ID'], how='left')
    return df

def add_efficiency_signals(df):
    """
    Docstring for add_efficiency_signals
    
    :param df: Description
    :return: Description
    :rtype: Any

    Note:
        - Creates signals that indicate whether a player's performance is
          above or below their season average (e.g., TS_EFFICIENCY_GAP)
        - Calculates volume-based signals like FGA_PER_MIN to identify
          players taking more shots than usual (potentially due to injuries)
        - TOV_PER_USAGE captures if a player is turning the ball over more
          relative to their usage, which can indicate risk or opportunity
    """

    print("...Calculating Efficiency Signals")
    # Volume: Field Goal Attempts per Minute
    df['FGA_PER_MIN'] = df['FGA'] / (df['MIN'] + 0.1)
    
    # Signal: Difference from season average efficiency
    if 'TS_PCT_Season' in df.columns:
        df['TS_EFFICIENCY_GAP'] = df['TS_PCT'] - df['TS_PCT_Season']
    
    # TOV Rate: Turnovers relative to Usage
    df['TOV_PER_USAGE'] = df['TOV'] / (df['USAGE_RATE'] + 0.1)
    
    return df

def ensure_combo_stats(df):
    """
    Docstring for ensure_combo_stats
    
    :param df: Description
    :return: Description
    :rtype: Any
    
    Note:
        - Ensures that combo stats like PRA, PR, PA, RA, SB are calculated
          before rolling features are created
        - This prevents the "SettingWithCopyWarning" by creating new columns
          in a single step and avoiding chained assignments
    """

    df = df.copy()
    if 'PRA' not in df.columns: df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    if 'PR' not in df.columns: df['PR'] = df['PTS'] + df['REB']
    if 'PA' not in df.columns: df['PA'] = df['PTS'] + df['AST']
    if 'RA' not in df.columns: df['RA'] = df['REB'] + df['AST']
    if 'SB' not in df.columns: df['SB'] = df['STL'] + df['BLK']
    return df

def main():
    """
    1. Loads raw game logs and merges with player positions.
    2. Engineers 80+ features including advanced stats, rolling averages,
       context features, defensive matchups, and injury impact.
    3. Cleans the dataset by filtering out low-minute games and dropping NaNs.
    4. Saves the final training dataset to data/training_dataset.csv 
        for model training. 
    
    The resulting dataset is ready for use in XGBoost or other 
        machine learning models.
    
    It contains a rich set of features designed to help the model learn 
    complex patterns in player performance based on historical data, 
    game context, and opponent matchups.
    """

    # 1. Load
    df = load_and_merge_data()
    if df is None: return

    # 2. Engineer
    df = add_advanced_stats(df)
    df = add_context_features(df)
    df = add_missing_player_context(df)

    df = add_schedule_density(df)
    df = add_pace_features(df)

    # FIX #2: Create combo stats BEFORE rolling averages so PRA_L5, PR_L5, etc. are created
    df = ensure_combo_stats(df)
    
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    df = add_rolling_features(df)
    df = add_efficiency_signals(df)

    df = add_defense_vs_position(df)
    df = add_usage_vacuum_features(df)
    
    # 3. Clean
    # FIX #6: Filter out low-minute games (garbage time / DNPs)
    df = df[df['MIN'] >= 10]
    
    # Drop rows that have NaNs (usually the first few games of a season where L5 isn't ready)
    df = df.dropna()

    # 4. Save
    print(f"\nSuccess! Saving {len(df)} training rows to {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()