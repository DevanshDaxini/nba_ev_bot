"""
Model Performance Visualizations

Creates charts showing:
    1. Feature Importance - Which features models rely on most
    2. Win Rate Trend - Accuracy over time vs breakeven rate
    
Output Files:
    feature_importance.png - Horizontal bar chart
    win_rate_trend.png - Line chart with breakeven threshold
    
Usage:
    $ python3 -m src.visualizer
    
Requirements:
    - matplotlib
    - Trained models in models/ folder
    - win_rate_history.csv (from grader.py)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import matplotlib.ticker as mtick

def plot_feature_importance():
    """
    Visualize which features PTS model uses most.
    
    Loads:
        models/PTS_model.json
        
    Generates:
        Horizontal bar chart showing top features by F-score
        (F-score = number of times feature was used for splits)
        
    Example Output:
        PTS_L5         ████████████ 450
        PTS_Season     ██████████ 380
        MISSING_USAGE  ████████ 290
        OPP_PTS_ALLOWED ████ 180
        
    Note:
        If new features (GAMES_7D, PACE_ROLLING) aren't showing,
        they weren't important for predictions (model ignored them)
        
    Saves:
        feature_importance.png in current directory
    """
    
    # CORRECTED: Point explicitly to the .json file
    model_path = 'models/PTS_model.json'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find {model_path}")
        print("   -> Check if your folder is named 'models' and contains .json files")
        return

    # Load the model using XGBoost's native JSON loader
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # Get importance
    importance = model.get_booster().get_score(importance_type='weight')
    importance = pd.Series(importance).sort_values(ascending=True)

    # Plot
    plt.figure(figsize=(10, 8))
    importance.plot(kind='barh', color='skyblue')
    plt.title('Feature Importance: What is the Model Weighting?')
    plt.xlabel('Weight (F-Score)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("✅ Saved: feature_importance.png")

def plot_win_rate():
    """
    Plot accuracy trend vs PrizePicks breakeven rate.
    
    Loads:
        program_runs/win_rate_history.csv
        
    Generates:
        Line chart with:
            - Green line: Your win rate over time
            - Red dashed line: 54.1% breakeven (5-man flex)
            - Y-axis: 40% to 60% (zoomed for clarity)
            
    Data Cleaning:
        - Handles "50%", "0.5", and "50.00" formats
        - Removes duplicate header rows if present
        - Auto-scales decimal (0.5) to percentage (50.0)
        
    Example Output:
        [Chart showing win rate fluctuating 52-58% over 10 days]
        
    Interpretation:
        Above red line = Profitable
        Below red line = Losing money
        
    Saves:
        win_rate_trend.png in current directory
    """
    
    history_file = "program_runs/win_rate_history.csv"
    if not os.path.exists(history_file):
        print("⚠️ No history file found yet.")
        return

    df = pd.read_csv(history_file)
    
    # --- CRITICAL FIX: FORCE NUMERIC CONVERSION ---
    # 1. Ensure it's a string, strip '%', and coerce errors to NaN
    # This handles "50%", "0.5", and even garbage text like "Win_Rate" in the middle of the file
    df['Win_Rate'] = pd.to_numeric(
        df['Win_Rate'].astype(str).str.replace('%', ''), 
        errors='coerce'
    )
    
    # 2. Drop any rows that couldn't be converted (like repeated headers)
    df = df.dropna(subset=['Win_Rate'])
    
    # 3. Auto-Scale: If data is 0.51, convert to 51.0
    if df['Win_Rate'].mean() < 1.0:
        print("ℹ️ Detected decimal format (e.g., 0.51). Converting to percentage.")
        df['Win_Rate'] = df['Win_Rate'] * 100
        
    # ----------------------------------------------

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Plot Green Line
    plt.plot(df['Date'], df['Win_Rate'], marker='o', markersize=8, 
             linestyle='-', color='green', linewidth=3, label='Model Accuracy')
    
    # Plot Red Breakeven Line
    plt.axhline(y=54.1, color='red', linestyle='--', linewidth=2, label='PrizePicks Breakeven (54.1%)')
    
    # Y-Axis Settings (40% to 60%)
    plt.ylim(40, 60)
    plt.yticks(range(40, 61, 2)) # Ticks every 2%
    
    plt.title('NBA Bot Accuracy Tracker', fontsize=16, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('win_rate_trend.png')
    print("✅ Saved: win_rate_trend.png (Cleaned & Scaled)")

if __name__ == "__main__":
    print("📊 Generating Visualizations...")
    plot_feature_importance()
    plot_win_rate()
    print("🚀 Done! Check your folder for .png files.")
    
#To run: python3 -m src.visualizer