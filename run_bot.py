"""
Headless Runner for GitHub Actions
Runs the daily scan without asking for user input.
"""
# CORRECTED IMPORT: load_data and load_models come from scanner.py
from src.scanner import scan_all, load_data, load_models

def run_headless():
    print("🤖 Starting Headless Scan...")
    
    # 1. Load Data & Models
    # The functions are imported directly from scanner
    df = load_data()
    models = load_models()
    
    if df is None or not models:
        print("❌ Critical Error: Data or Models missing.")
        return

    # 2. Run the Scan (Mode: Today)
    print("🚀 Scanning Today's Games...")
    # We call scan_all directly since we imported it
    scan_all(df, models, is_tomorrow=False)
    
    # 3. Run the Scan (Mode: Tomorrow)
    print("🔮 Scanning Tomorrow's Games...")
    scan_all(df, models, is_tomorrow=True)
    
    print("✅ Headless Scan Complete.")

if __name__ == "__main__":
    run_headless()