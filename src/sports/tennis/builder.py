"""
Historical Tennis Data Collection

Downloads raw match data from Jeff Sackmann's open-source tennis databases
(github.com/JeffSackmann) for both ATP (men's) and WTA (women's) tours.
Parses score strings to extract sets, tiebreaks, and per-player match stats.

Data Sources:
    - github.com/JeffSackmann/tennis_atp  - ATP match CSVs (annual + current year)
    - github.com/JeffSackmann/tennis_wta  - WTA match CSVs (annual + current year)

    For the current year, Sackmann publishes matches in files named:
        atp_matches_YYYY.csv       (pushed mid/end of season)
    If that file returns 404, we fall back to:
        atp_matches_activity.csv   (rolling year file, updated frequently)
        atp_matches_current.csv    (alternative rolling name)

Output Files:
    data/tennis/raw/atp_raw_matches.csv   - ATP player-level match rows
    data/tennis/raw/wta_raw_matches.csv   - WTA player-level match rows

Usage:
    $ python3 -m src.sports.tennis.builder

Performance:
    Takes ~1-2 minutes (GitHub downloads, no rate limiting needed)
"""

import pandas as pd
import os
import re
import time
import urllib.request
from datetime import datetime

# --- CONFIGURATION ---
CURRENT_YEAR  = datetime.now().year
FALLBACK_FROM = 2025                            # try fallback URLs from this year onward
YEARS         = range(2019, CURRENT_YEAR + 1)  # always includes current year

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_FOLDER = os.path.join(BASE_DIR, 'data', 'tennis', 'raw')

ATP_OUTPUT = os.path.join(RAW_FOLDER, 'atp_raw_matches.csv')
WTA_OUTPUT = os.path.join(RAW_FOLDER, 'wta_raw_matches.csv')

# Primary annual file URL pattern
ATP_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
WTA_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"

# Fallback URLs tried in order for the current year if annual file is 404
ATP_FALLBACKS = [
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_activity.csv",
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_current.csv",
]
WTA_FALLBACKS = [
    "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
    "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_activity.csv",
    "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_current.csv",
]

# Sackmann columns we care about
KEEP_COLS = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_date',
    'match_num', 'round',
    'winner_id', 'winner_name', 'winner_rank',
    'loser_id',  'loser_name',  'loser_rank',
    'score',
    'w_ace',  'w_df',  'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
    'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace',  'l_df',  'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
    'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    'best_of',
]


# ---------------------------------------------------------------------------
# SCORE PARSING
# ---------------------------------------------------------------------------

def parse_score(score_str):
    result = {
        'total_games': 0, 'total_sets': 0, 'total_tiebreaks': 0,
        'w_games_won': 0, 'l_games_won': 0, 'retired': 0,
    }
    if not isinstance(score_str, str) or not score_str.strip():
        return result

    if any(t in score_str.upper() for t in ['RET', 'W/O', 'DEF', 'ABD']):
        result['retired'] = 1

    for w_str, l_str in re.findall(r'(\d+)-(\d+)(?:\(\d+\))?', score_str):
        w_g, l_g = int(w_str), int(l_str)
        result['w_games_won'] += w_g
        result['l_games_won'] += l_g
        result['total_sets']  += 1
        if (w_g == 7 and l_g == 6) or (w_g == 6 and l_g == 7):
            result['total_tiebreaks'] += 1

    result['total_games'] = result['w_games_won'] + result['l_games_won']
    return result


def _count_sets_won(score_str, winner=True):
    if not isinstance(score_str, str):
        return 0
    count = 0
    for w_str, l_str in re.findall(r'(\d+)-(\d+)(?:\(\d+\))?', score_str):
        w_g, l_g = int(w_str), int(l_str)
        count += 1 if (winner and w_g > l_g) or (not winner and l_g > w_g) else 0
    return count


def _safe_int(val):
    try:    return int(float(val))
    except: return 0

def _safe_float(val):
    try:    return float(val)
    except: return None


def expand_to_player_rows(df, tour):
    """Convert match rows → 2 player-level rows per match."""
    rows = []
    for _, m in df.iterrows():
        sd        = parse_score(m.get('score', ''))
        w_bpFaced = _safe_int(m.get('w_bpFaced'))
        w_bpSaved = _safe_int(m.get('w_bpSaved'))
        l_bpFaced = _safe_int(m.get('l_bpFaced'))
        l_bpSaved = _safe_int(m.get('l_bpSaved'))

        base = {
            'tourney_id':      m.get('tourney_id'),
            'tourney_name':    m.get('tourney_name'),
            'surface':         m.get('surface', 'Unknown'),
            'tourney_date':    m.get('tourney_date'),
            'round':           m.get('round'),
            'best_of':         _safe_int(m.get('best_of', 3)),
            'score':           m.get('score'),
            'tour':            tour,
            'retired':         sd['retired'],
            'total_games':     sd['total_games'],
            'total_sets':      sd['total_sets'],
            'total_tiebreaks': sd['total_tiebreaks'],
        }

        rows.append({**base,
            'player_id':     m.get('winner_id'),    'player_name':  m.get('winner_name'),
            'player_rank':   _safe_float(m.get('winner_rank')),
            'opp_id':        m.get('loser_id'),     'opp_name':     m.get('loser_name'),
            'opp_rank':      _safe_float(m.get('loser_rank')),
            'won_match':     1,
            'games_won':     sd['w_games_won'],     'games_lost':   sd['l_games_won'],
            'sets_won':      _count_sets_won(m.get('score', ''), winner=True),
            'aces':          _safe_int(m.get('w_ace')),
            'double_faults': _safe_int(m.get('w_df')),
            'bp_won':        max(0, l_bpFaced - l_bpSaved),
            'bp_faced':      w_bpFaced, 'bp_saved': w_bpSaved, 'bp_faced_opp': l_bpFaced,
            'svpt':          _safe_int(m.get('w_svpt')),
            'first_in':      _safe_int(m.get('w_1stIn')),
            'first_won':     _safe_int(m.get('w_1stWon')),
            'second_won':    _safe_int(m.get('w_2ndWon')),
            'svc_games':     _safe_int(m.get('w_SvGms')),
        })

        rows.append({**base,
            'player_id':     m.get('loser_id'),     'player_name':  m.get('loser_name'),
            'player_rank':   _safe_float(m.get('loser_rank')),
            'opp_id':        m.get('winner_id'),    'opp_name':     m.get('winner_name'),
            'opp_rank':      _safe_float(m.get('winner_rank')),
            'won_match':     0,
            'games_won':     sd['l_games_won'],     'games_lost':   sd['w_games_won'],
            'sets_won':      _count_sets_won(m.get('score', ''), winner=False),
            'aces':          _safe_int(m.get('l_ace')),
            'double_faults': _safe_int(m.get('l_df')),
            'bp_won':        max(0, w_bpFaced - w_bpSaved),
            'bp_faced':      l_bpFaced, 'bp_saved': l_bpSaved, 'bp_faced_opp': w_bpFaced,
            'svpt':          _safe_int(m.get('l_svpt')),
            'first_in':      _safe_int(m.get('l_1stIn')),
            'first_won':     _safe_int(m.get('l_1stWon')),
            'second_won':    _safe_int(m.get('l_2ndWon')),
            'svc_games':     _safe_int(m.get('l_SvGms')),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------------------------------

def download_csv(url):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            content = r.read().decode('utf-8')
        from io import StringIO
        return pd.read_csv(StringIO(content), low_memory=False)
    except Exception:
        return None


def download_with_fallbacks(year, fallback_urls):
    """Try each URL in order, return first successful DataFrame + source name."""
    for url_template in fallback_urls:
        url = url_template.format(year=year)
        df  = download_csv(url)
        if df is not None and not df.empty:
            return df, url.split('/')[-1]
    return None, None


# ---------------------------------------------------------------------------
# MAIN FETCH
# ---------------------------------------------------------------------------

def fetch_tour_data(tour, base_url, fallback_urls, output_path):
    os.makedirs(RAW_FOLDER, exist_ok=True)
    all_dfs = []

    print(f"\n--- DOWNLOADING {tour.upper()} DATA ({min(YEARS)}-{max(YEARS)}) ---")

    for year in YEARS:
        use_fallback = (year >= FALLBACK_FROM)

        if use_fallback:
            label = "current" if year == CURRENT_YEAR else f"{year}"
            print(f"  Fetching {year} ({label} — trying fallbacks)...", end=' ')
            df, source = download_with_fallbacks(year, fallback_urls)
            if df is None:
                print(f"⚠️  Not available yet (Sackmann hasn't pushed {year} data)")
                continue
            print(f"✅  {len(df):,} matches  [{source}]")
        else:
            url = base_url.format(year=year)
            print(f"  Fetching {year}...", end=' ')
            df = download_csv(url)
            if df is None or df.empty:
                print("SKIPPED (no data)")
                continue
            print(f"✅  {len(df):,} matches")

        available = [c for c in KEEP_COLS if c in df.columns]
        df = df[available].copy()
        df['year'] = year

        df = df[df['score'].notna()]
        df = df[~df['score'].astype(str).str.upper().str.contains('W/O')]

        all_dfs.append(df)
        time.sleep(0.3)

    if not all_dfs:
        print(f"FAILED: No {tour.upper()} data downloaded.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate in case activity file overlaps with prior annual file
    combined = combined.drop_duplicates(
        subset=['tourney_id', 'winner_id', 'loser_id', 'round'],
        keep='last'
    )

    print(f"\n  Parsing scores and expanding to player rows...")
    player_df = expand_to_player_rows(combined, tour=tour)

    player_df['tourney_date'] = pd.to_datetime(
        player_df['tourney_date'].astype(str), format='%Y%m%d', errors='coerce'
    )
    player_df = player_df.sort_values(['player_id', 'tourney_date']).reset_index(drop=True)

    player_df.to_csv(output_path, index=False)
    print(f"  ✅  Saved {len(player_df):,} player-match rows → {output_path}")
    print(f"      Players:    {player_df['player_name'].nunique():,}")
    print(f"      Date range: {player_df['tourney_date'].min().date()} → {player_df['tourney_date'].max().date()}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("   🎾 TENNIS DATA BUILDER")
    print("=" * 55)
    print(f"Years: {min(YEARS)} – {max(YEARS)}  (current year: {CURRENT_YEAR})")
    print(f"Output: {RAW_FOLDER}")

    fetch_tour_data('atp', ATP_BASE_URL, ATP_FALLBACKS, ATP_OUTPUT)
    fetch_tour_data('wta', WTA_BASE_URL, WTA_FALLBACKS, WTA_OUTPUT)

    print("\n" + "=" * 55)
    print("✅  BUILD COMPLETE")
    print("   Next step: Run features.py to engineer training data.")
    print("=" * 55)