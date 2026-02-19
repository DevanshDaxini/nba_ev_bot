"""
Tennis XGBoost Model Training Pipeline

Trains separate regression models for 7 tennis markets using time-series
split validation. Mirrors the NBA train.py structure exactly.

Targets:
    total_games, games_won, total_sets, aces,
    bp_won, total_tiebreaks, double_faults

Output:
    models/tennis/{TARGET}_model.json
    models/tennis/model_metrics.csv

Usage:
    $ python3 -m src.sports.tennis.train
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import csv
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FILE  = os.path.join(BASE_DIR, 'data',   'tennis', 'processed', 'training_dataset.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models', 'tennis')

# Hold out the last 10% of matches (by date) as the test set.
# This is robust regardless of when the data ends — no hardcoded dates.
TEST_FRACTION = 0.10

TARGETS = [
    'total_games',
    'games_won',
    'total_sets',
    'aces',
    'bp_won',
    'total_tiebreaks',
    'double_faults',
]

# Core features used for every model
FEATURES = [
    # --- Surface ---
    'surface_hard', 'surface_clay', 'surface_grass', 'surface_carpet',
    'is_best_of_5', 'round_ordinal', 'is_atp',

    # --- Ranking ---
    'player_rank', 'opp_rank', 'rank_delta', 'rank_ratio',
    'log_rank', 'log_opp_rank',

    # --- Fatigue / Schedule ---
    'days_rest', 'matches_L14D', 'is_b2b',

    # --- H2H ---
    'h2h_win_rate', 'h2h_avg_games',

    # --- Opponent Surface Limits ---
    # (Dynamically added below, but listed here for clarity)

    # --- Vs Rank Brackets ---
    'career_win_pct_vs_top20', 'career_win_pct_vs_top50', 'career_win_pct_vs_top100',

    # --- Career ---
    'career_matches', 'is_early_career',
]

# Add Opponent Surface-Specific Rolling Stats to features
for stat in ['aces', 'double_faults', 'total_games', 'games_won', 'bp_won']:
    for surf in ['hard', 'clay', 'grass']:
        FEATURES.append(f'opp_{stat}_{surf}_L10')

for stat in TARGETS + ['won_match', 'sets_won', 'bp_faced', 'svpt', 'svc_games']:
    for window in ['L5', 'L20', 'Season']:
        FEATURES.append(f'{stat}_{window}')

for stat in ['aces', 'double_faults', 'total_games', 'games_won', 'bp_won']:
    for surf in ['hard', 'clay', 'grass']:
        FEATURES.append(f'{stat}_{surf}_L10')

for stat in ['total_games', 'games_won', 'aces', 'double_faults', 'bp_won', 'bp_faced']:
    for window in ['L5', 'L20']:
        FEATURES.append(f'opp_{stat}_{window}')


def train_and_evaluate():
    print("=" * 55)
    print("   🎾 TENNIS MODEL TRAINING")
    print("=" * 55)

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Training data not found at {DATA_FILE}")
        print("       Run features.py first.")
        return

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    df = df.dropna(subset=['tourney_date'])
    df = df.sort_values('tourney_date').reset_index(drop=True)

    # --- Dynamic time-based split (last 10% of rows by date) ---
    split_idx      = int(len(df) * (1 - TEST_FRACTION))
    split_date     = df.iloc[split_idx]['tourney_date']
    test_start_str = split_date.strftime('%Y-%m-%d')

    print(f"Loaded {len(df):,} rows for training")
    print(f"Train: before {test_start_str}  ({split_idx:,} rows)")
    print(f"Test:  {test_start_str} → {df['tourney_date'].max().date()}  ({len(df) - split_idx:,} rows)\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.csv')
    all_metrics  = []

    for target in TARGETS:
        if target not in df.columns:
            print(f"⚠️  Target '{target}' not found in dataset. Skipping.")
            continue

        print(f"--- Training: {target.upper()} ---")

        df_model = df[df[target].notna()].copy()
        df_model = df_model.sort_values('tourney_date').reset_index(drop=True)

        available_features = [f for f in FEATURES if f in df_model.columns]
        missing = len(FEATURES) - len(available_features)
        if missing > 0:
            print(f"   ℹ️  {missing} features absent — using {len(available_features)} available")

        X = df_model[available_features].fillna(0)
        y = df_model[target]

        # Re-compute split index per-target (some targets have fewer rows)
        t_split    = int(len(df_model) * (1 - TEST_FRACTION))
        train_mask = df_model.index < t_split
        test_mask  = df_model.index >= t_split

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(X_train) < 100:
            print(f"   ⚠️  Too few training rows ({len(X_train)}). Skipping.")
            continue

        if len(X_test) == 0:
            print(f"   ⚠️  Empty test set. Skipping.")
            continue

        print(f"   Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

        model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)

        # Directional accuracy vs line (over/under the line value)
        # Use test-set mean as proxy for "the line"
        target_mean = float(y_train.mean())
        dir_correct = (
            np.sum((preds > target_mean) == (y_test.values > target_mean))
            / len(y_test) * 100
        )

        print(f"   MAE: {mae:.3f} | R²: {r2:.3f} | Dir Acc: {dir_correct:.1f}%")

        model_path = os.path.join(MODEL_DIR, f'{target}_model.json')
        model.save_model(model_path)
        print(f"   ✅  Saved → {model_path}")

        all_metrics.append({
            'target':       target,
            'train_rows':   len(X_train),
            'test_rows':    len(X_test),
            'mae':          round(mae, 4),
            'r2':           round(r2, 4),
            'dir_accuracy': round(dir_correct, 2),
            'trained_at':   datetime.now().strftime('%Y-%m-%d %H:%M'),
        })

    if all_metrics:
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n📊 Metrics saved → {metrics_path}")

    print("\n" + "=" * 55)
    print("   MODEL SUMMARY")
    print("=" * 55)
    print(f"{'TARGET':<22} {'MAE':>6} {'R²':>6} {'DIR%':>7}")
    print("-" * 45)
    for m in all_metrics:
        print(f"{m['target']:<22} {m['mae']:>6.3f} {m['r2']:>6.3f} {m['dir_accuracy']:>6.1f}%")
    print("=" * 55)
    print("✅  TRAINING COMPLETE")
    print("   Next step: Run scanner.py to make predictions.")


if __name__ == "__main__":
    train_and_evaluate()