"""
Tennis Rankings & Tournament Context

Pulls live ATP/WTA rankings from Jeff Sackmann's GitHub (free, no API key).
Same source as training data. Updated weekly by Sackmann.

Sources:
    github.com/JeffSackmann/tennis_atp  - atp_rankings_current.csv + atp_players.csv
    github.com/JeffSackmann/tennis_wta  - wta_rankings_current.csv + wta_players.csv

Usage:
    from src.sports.tennis.rankings import TennisRankings
    r = TennisRankings()
    r.load()
    rank = r.get_rank('Carlos Alcaraz')     # → 3.0
    surf = r.get_surface('Australian Open') # → 'Hard'
"""

import pandas as pd
import urllib.request
import json
import os
import time
import re
import unicodedata
from io import StringIO

# --- PATHS ---
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE_DIR   = os.path.join(BASE_DIR, 'data', 'tennis', 'rankings_cache')
ATP_CACHE   = os.path.join(CACHE_DIR, 'atp_rankings.json')
WTA_CACHE   = os.path.join(CACHE_DIR, 'wta_rankings.json')
CACHE_HOURS = 24

ATP_RANKINGS_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv"
WTA_RANKINGS_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv"
ATP_PLAYERS_URL  = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"
WTA_PLAYERS_URL  = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_players.csv"

# ---------------------------------------------------------------------------
# TOURNAMENT → SURFACE MAP
# ---------------------------------------------------------------------------
TOURNAMENT_SURFACE_MAP = {
    # Grand Slams
    'australian open': 'Hard', 'roland garros': 'Clay', 'french open': 'Clay',
    'wimbledon': 'Grass', 'us open': 'Hard',
    # ATP Masters 1000
    'indian wells': 'Hard', 'miami open': 'Hard', 'miami': 'Hard',
    'monte carlo': 'Clay', 'madrid open': 'Clay', 'madrid': 'Clay',
    'italian open': 'Clay', 'rome': 'Clay', 'internazionali': 'Clay',
    'canadian open': 'Hard', 'toronto': 'Hard', 'montreal': 'Hard',
    'cincinnati': 'Hard', 'western & southern': 'Hard',
    'shanghai': 'Hard', 'paris masters': 'Hard', 'paris': 'Hard',
    # ATP 500
    'dubai': 'Hard', 'acapulco': 'Hard', 'rio': 'Clay',
    'barcelona': 'Clay', 'hamburg': 'Clay', 'washington': 'Hard',
    'beijing': 'Hard', 'china open': 'Hard', 'tokyo': 'Hard',
    'japan open': 'Hard', 'basel': 'Hard', 'vienna': 'Hard',
    'halle': 'Grass', "queen's club": 'Grass', 'queens club': 'Grass',
    # ATP 250 hard
    'brisbane': 'Hard', 'auckland': 'Hard', 'adelaide': 'Hard',
    'doha': 'Hard', 'marseille': 'Hard', 'rotterdam': 'Hard',
    'delray beach': 'Hard', 'dallas': 'Hard', 'los cabos': 'Hard',
    'winston-salem': 'Hard', 'metz': 'Hard', 'astana': 'Hard',
    'st. petersburg': 'Hard', 'sofia': 'Hard', 'antwerp': 'Hard',
    'stockholm': 'Hard', 'moscow': 'Hard', 'pune': 'Hard',
    # ATP 250 clay
    'geneva': 'Clay', 'lyon': 'Clay', 'munich': 'Clay', 'estoril': 'Clay',
    'bucharest': 'Clay', 'marrakech': 'Clay', 'gstaad': 'Clay',
    'umag': 'Clay', 'kitzbuhel': 'Clay', 'bastad': 'Clay',
    # ATP 250 grass
    'newport': 'Grass', 'eastbourne': 'Grass', 'stuttgart': 'Grass',
    'mallorca': 'Grass', 'nottingham': 'Grass', 's-hertogenbosch': 'Grass',
    # WTA
    'berlin': 'Clay', 'bad homburg': 'Grass', 'birmingham': 'Grass',
    'san jose': 'Hard', 'abu dhabi': 'Hard', 'wuhan': 'Hard', 'guangzhou': 'Hard',
    # Catch-all keywords
    'clay': 'Clay', 'grass': 'Grass', 'indoor': 'Hard',
}

SLAM_NAMES = ['australian open', 'roland garros', 'french open', 'wimbledon', 'us open']


def _norm(name: str) -> str:
    if not isinstance(name, str):
        return ''
    n = unicodedata.normalize('NFD', name.lower().strip())
    n = ''.join(c for c in n if unicodedata.category(c) != 'Mn')
    n = re.sub(r"[-'.,]+", ' ', n)
    return re.sub(r'\s+', ' ', n).strip()


def _cache_fresh(path):
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) / 3600 < CACHE_HOURS


def _load_cache(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def _fetch_raw(url: str) -> str | None:
    """Download URL and return raw text."""
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.read().decode('utf-8')
    except Exception as e:
        print(f"   ⚠️  Download failed ({url.split('/')[-1]}): {e}")
        return None


def _load_csv_autodetect(raw: str, fallback_names: list) -> pd.DataFrame:
    """
    Read a CSV that may or may not have a header row.
    Strategy: read first line. If ALL fields look like column labels (non-numeric),
    treat as header. Otherwise assign fallback_names positionally.
    """
    first_line = raw.split('\n')[0].strip()
    fields     = [f.strip().strip('"') for f in first_line.split(',')]

    # If every field in the first row is non-numeric → it's a header
    has_header = all(not f.replace('.', '').replace('-', '').isdigit() for f in fields if f)

    if has_header:
        df = pd.read_csv(StringIO(raw), dtype=str)
        # Normalise column names
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    else:
        df = pd.read_csv(StringIO(raw), header=None, dtype=str)
        n  = min(len(df.columns), len(fallback_names))
        df.columns = list(fallback_names[:n]) + [f'col{i}' for i in range(n, len(df.columns))]

    return df


def _build_rankings(rankings_url: str, players_url: str, tour: str) -> dict:
    """
    Download ranking + player CSVs, merge them, return {normalized_name: rank_int}.
    Handles both headered and headerless files robustly.
    """
    raw_rank = _fetch_raw(rankings_url)
    raw_play = _fetch_raw(players_url)
    if not raw_rank or not raw_play:
        return {}

    try:
        # Rankings columns: ranking_date, rank, player_id, points
        rank_df = _load_csv_autodetect(raw_rank, ['ranking_date', 'rank', 'player_id', 'points'])

        # Players columns: player_id, name_first, name_last, hand, dob, country
        play_df = _load_csv_autodetect(raw_play, ['player_id', 'name_first', 'name_last', 'hand', 'dob', 'country'])

        # Normalise column name aliases
        for alias, canon in [('ranking', 'rank'), ('id', 'player_id'), ('player', 'player_id'),
                              ('first_name', 'name_first'), ('last_name', 'name_last')]:
            if alias in rank_df.columns and canon not in rank_df.columns:
                rank_df.rename(columns={alias: canon}, inplace=True)
            if alias in play_df.columns and canon not in play_df.columns:
                play_df.rename(columns={alias: canon}, inplace=True)

        # Validate required columns exist
        for col in ['rank', 'player_id']:
            if col not in rank_df.columns:
                print(f"   ⚠️  Rankings file missing '{col}'. Columns: {rank_df.columns.tolist()}")
                return {}
        for col in ['player_id', 'name_last']:
            if col not in play_df.columns:
                print(f"   ⚠️  Players file missing '{col}'. Columns: {play_df.columns.tolist()}")
                return {}

        # Clean up
        rank_df['player_id'] = rank_df['player_id'].astype(str).str.strip()
        play_df['player_id'] = play_df['player_id'].astype(str).str.strip()
        rank_df['rank']      = pd.to_numeric(rank_df['rank'], errors='coerce')
        rank_df = rank_df.dropna(subset=['rank'])

        # Most recent ranking per player
        date_col = next((c for c in rank_df.columns if 'date' in c), None)
        if date_col:
            rank_df = rank_df.sort_values(date_col, ascending=False)
        rank_df = rank_df.drop_duplicates('player_id', keep='first')

        # Merge
        name_cols = ['player_id', 'name_last']
        if 'name_first' in play_df.columns:
            name_cols.append('name_first')
        merged = rank_df.merge(play_df[name_cols], on='player_id', how='left')

        result = {}
        for _, row in merged.iterrows():
            last  = str(row.get('name_last',  '') or '').strip()
            first = str(row.get('name_first', '') or '').strip()
            if not last or last.lower() == 'nan':
                continue
            full  = _norm(f"{first} {last}".strip())
            result[full] = int(row['rank'])

        return result

    except Exception as e:
        print(f"   ⚠️  {tour} rankings error: {e}")
        import traceback; traceback.print_exc()
        return {}


# ---------------------------------------------------------------------------
# PUBLIC CLASS
# ---------------------------------------------------------------------------

class TennisRankings:

    def __init__(self):
        self.atp_ranks: dict[str, int] = {}
        self.wta_ranks: dict[str, int] = {}

    def load(self, force_refresh=False):
        """Load ATP + WTA rankings (24-hour disk cache)."""
        # ATP
        if not force_refresh and _cache_fresh(ATP_CACHE):
            d = _load_cache(ATP_CACHE)
            if d:
                self.atp_ranks = d
                print(f"   ♻️  ATP rankings from cache ({len(d)} players)")
            else:
                self._load_atp()
        else:
            self._load_atp()

        # WTA
        if not force_refresh and _cache_fresh(WTA_CACHE):
            d = _load_cache(WTA_CACHE)
            if d:
                self.wta_ranks = d
                print(f"   ♻️  WTA rankings from cache ({len(d)} players)")
            else:
                self._load_wta()
        else:
            self._load_wta()

        total = len(self.atp_ranks) + len(self.wta_ranks)
        if total > 0:
            print(f"   ✅  Rankings: {len(self.atp_ranks)} ATP + {len(self.wta_ranks)} WTA players")
        else:
            print("   ⚠️  Rankings unavailable — predictions will use rank=50 default")

    def _load_atp(self):
        print("   📡 Fetching ATP rankings...")
        result = _build_rankings(ATP_RANKINGS_URL, ATP_PLAYERS_URL, 'ATP')
        if result:
            self.atp_ranks = result
            _save_cache(ATP_CACHE, result)
            print(f"   ✅  ATP: {len(result)} players")

    def _load_wta(self):
        print("   📡 Fetching WTA rankings...")
        result = _build_rankings(WTA_RANKINGS_URL, WTA_PLAYERS_URL, 'WTA')
        if result:
            self.wta_ranks = result
            _save_cache(WTA_CACHE, result)
            print(f"   ✅  WTA: {len(result)} players")

    def get_rank(self, player_name: str, default: float = 50.0) -> float:
        norm = _norm(player_name)
        # Exact match
        rank = self.atp_ranks.get(norm) or self.wta_ranks.get(norm)
        if rank:
            return float(rank)
        # Last-name fallback
        last = norm.split()[-1] if norm else ''
        if len(last) > 3:
            for d in [self.atp_ranks, self.wta_ranks]:
                for name, r in d.items():
                    if name.split()[-1] == last:
                        return float(r)
        return default

    def get_tour(self, player_name: str) -> str:
        norm = _norm(player_name)
        if norm in self.atp_ranks:
            return 'atp'
        if norm in self.wta_ranks:
            return 'wta'
        last = norm.split()[-1] if norm else ''
        for name in self.atp_ranks:
            if name.split()[-1] == last:
                return 'atp'
        for name in self.wta_ranks:
            if name.split()[-1] == last:
                return 'wta'
        return 'atp'

    @staticmethod
    def get_surface(tournament_name: str) -> str:
        if not tournament_name:
            return 'Hard'
        norm = _norm(tournament_name)
        if norm in TOURNAMENT_SURFACE_MAP:
            return TOURNAMENT_SURFACE_MAP[norm]
        for key, surface in TOURNAMENT_SURFACE_MAP.items():
            if key in norm or norm in key:
                return surface
        return 'Hard'

    @staticmethod
    def is_slam(tournament_name: str) -> bool:
        norm = _norm(tournament_name or '')
        return any(s in norm for s in SLAM_NAMES)

    @staticmethod
    def get_round_ordinal(round_str: str) -> int:
        ROUND_ORDER = {
            'r128': 1, 'r64': 2, 'r32': 3, 'r16': 4,
            'qf': 5, 'quarter-final': 5, 'quarter final': 5,
            'sf': 6, 'semi-final': 6, 'semi final': 6,
            'f': 7, 'final': 7, 'rr': 3,
        }
        return ROUND_ORDER.get(_norm(round_str or ''), 3)


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    r = TennisRankings()
    r.load(force_refresh=True)

    print("\nRank lookups:")
    for p in ['Carlos Alcaraz', 'Jannik Sinner', 'Novak Djokovic',
              'Iga Swiatek', 'Aryna Sabalenka', 'Coco Gauff']:
        rank = r.get_rank(p)
        tour = r.get_tour(p).upper()
        print(f"  {p:<28} #{int(rank):>3}  ({tour})")

    print("\nSurface lookups:")
    for t in ['Australian Open', 'Roland Garros', 'Wimbledon', 'Indian Wells', 'Madrid Open']:
        print(f"  {t:<30} → {r.get_surface(t)}")