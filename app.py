"""
Mutual Fund Portfolio Dashboard
================================
Build a portfolio from ~500 funds across all categories. See rolling returns
vs Nifty 500, plus a growth chart and look-through breakdowns (cap, sector, stocks).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Page config + styles
# =============================================================================
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="centered",
)

st.markdown("""
<style>
    .block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 960px;}
    #MainMenu, footer, header {visibility: hidden;}

    h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
    .subtitle {opacity: 0.6; font-size: 0.95rem; margin-bottom: 2rem;}

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border-color: rgba(128, 128, 128, 0.25) !important;
    }

    .stButton button {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        background: transparent;
        font-weight: 500;
        transition: all 0.15s;
    }
    .stButton button:hover {
        border-color: rgba(128, 128, 128, 0.7);
        background: rgba(128, 128, 128, 0.08);
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.55;
        margin: 2rem 0 0.75rem 0;
    }

    .card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 0.85rem;
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 1.2rem;
    }
    .card-title {font-size: 1.05rem; font-weight: 600;}
    .card-meta  {font-size: 0.8rem;  opacity: 0.6;}

    .compare-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .compare-col {
        padding: 1rem 1.1rem;
        border-radius: 10px;
    }
    .compare-col.portfolio {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    .compare-col.benchmark {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .compare-header {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.85rem;
    }
    .dot {width: 8px; height: 8px; border-radius: 50%; display: inline-block;}
    .dot.portfolio  {background: #3b82f6;}
    .dot.benchmark  {background: #9ca3af;}
    .dot.large      {background: #3b82f6;}
    .dot.mid        {background: #8b5cf6;}
    .dot.small      {background: #ec4899;}
    .dot.cash       {background: #9ca3af;}

    .compare-stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem;}
    .compare-stat-label {
        font-size: 0.6rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em;
        opacity: 0.55; margin-bottom: 0.2rem;
        white-space: nowrap;
    }
    .compare-stat-value {
        font-size: 1.1rem; font-weight: 700;
        letter-spacing: -0.02em; line-height: 1.1;
    }

    .verdict {
        text-align: center; font-size: 0.85rem; font-weight: 600;
        padding: 0.65rem 1rem; border-radius: 8px; margin-top: 0.4rem;
    }
    .verdict.win  {background: rgba(16, 185, 129, 0.12); color: #10b981;
                   border: 1px solid rgba(16, 185, 129, 0.3);}
    .verdict.loss {background: rgba(239, 68, 68, 0.12); color: #ef4444;
                   border: 1px solid rgba(239, 68, 68, 0.3);}
    .verdict.tie  {background: rgba(128, 128, 128, 0.1); opacity: 0.75;
                   border: 1px solid rgba(128, 128, 128, 0.25);}

    .stack-bar {
        display: flex; height: 44px; border-radius: 10px; overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }
    .stack-seg {
        display: flex; align-items: center; justify-content: center;
        font-size: 0.82rem; font-weight: 600; color: white;
        min-width: 0;
    }
    .stack-legend {
        display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.6rem;
        margin-top: 0.8rem;
    }
    .legend-row {
        display: flex; justify-content: space-between; align-items: center;
        font-size: 0.85rem; padding: 0.25rem 0;
    }
    .legend-left {display: flex; align-items: center; gap: 0.55rem;}
    .legend-name {opacity: 0.85;}
    .legend-value {font-weight: 600; font-variant-numeric: tabular-nums;}

    .bar-row {
        display: grid;
        grid-template-columns: 1fr 60px;
        gap: 0.75rem; align-items: center;
        margin-bottom: 0.55rem;
    }
    .bar-label {
        font-size: 0.88rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .bar-track {
        height: 22px;
        background: rgba(128, 128, 128, 0.1);
        border-radius: 5px; overflow: hidden;
    }
    .bar-fill {height: 100%; border-radius: 5px;}
    .bar-fill.blue {background: linear-gradient(90deg, #3b82f6, #60a5fa);}
    .bar-fill.gray {background: linear-gradient(90deg, #6b7280, #9ca3af);}
    .bar-pct {
        font-size: 0.85rem; font-weight: 600;
        text-align: right; font-variant-numeric: tabular-nums;
    }

    .stock-row {
        display: grid;
        grid-template-columns: 28px 1fr auto 70px;
        gap: 0.75rem; align-items: center;
        padding: 0.7rem 0.9rem;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        margin-bottom: 0.45rem;
    }
    .stock-rank {
        font-size: 0.8rem; font-weight: 600; opacity: 0.4;
        font-variant-numeric: tabular-nums;
    }
    .stock-name {
        font-size: 0.92rem; font-weight: 500;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .cap-badge {
        font-size: 0.65rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em;
        padding: 0.15rem 0.55rem; border-radius: 4px;
    }
    .cap-badge.large {background: rgba(59, 130, 246, 0.15); color: #3b82f6;}
    .cap-badge.mid   {background: rgba(139, 92, 246, 0.15); color: #8b5cf6;}
    .cap-badge.small {background: rgba(236, 72, 153, 0.15); color: #ec4899;}
    .stock-pct {
        font-size: 0.95rem; font-weight: 700;
        text-align: right; font-variant-numeric: tabular-nums;
    }

    .pos {color: #10b981;}
    .neg {color: #ef4444;}
    .neu {color: inherit;}

    div[role="radiogroup"] {gap: 0.4rem;}

    /* Chart return badges */
    .chart-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .chart-stat {
        padding: 0.85rem 1rem;
        border-radius: 10px;
    }
    .chart-stat.portfolio {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    .chart-stat.benchmark {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .chart-stat-header {
        display: flex; align-items: center; gap: 0.4rem;
        font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-bottom: 0.5rem; opacity: 0.85;
    }
    .chart-stat-value {
        font-size: 1.5rem; font-weight: 700;
        letter-spacing: -0.02em; line-height: 1.1;
    }
    .chart-stat-meta {
        font-size: 0.75rem; opacity: 0.6; margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# File configuration
# =============================================================================
DATA_DIR = Path(__file__).parent

# Each NAV parquet file mapped to its display "tag" (shown alongside fund names)
NAV_FILES = {
    "largecap.parquet":   "Large Cap",
    "midcap.parquet":     "Mid Cap",
    "smallcap.parquet":   "Small Cap",
    "flexicap.parquet":   "Flexi Cap",
    "multicap.parquet":   "Multi Cap",
    "debtfunds.parquet":  "Debt",
    "sidefunds2.parquet": "Other",
    "sidefunds3.parquet": "Other",
    "sidefunds4.parquet": "Other",
    "sidefunds5.parquet": "Other",
    "goldetf.parquet":    "Gold ETF",
}

BENCHMARK_FILE = "nifty500.parquet"
ASSETTYPE_FILE = "assets.parquet"
SECTOR_FILE    = "sectors.parquet"
STOCKS_FILE    = "stocks.parquet"
AMFI_FILE      = "AMFI.parquet"


# =============================================================================
# Data loaders (cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_nav_file(path: str) -> pd.DataFrame:
    """Load a pre-parsed NAV parquet file. Validates structure."""
    data = pd.read_parquet(path)
    if not isinstance(data.index, pd.DatetimeIndex):
        st.error(
            f"❌ Bad parquet file: `{Path(path).name}`\n\n"
            f"It doesn't have a Date index. Run `convert_to_parquet.py` locally "
            f"to regenerate it."
        )
        st.stop()
    return data


@st.cache_data(show_spinner=False)
def load_benchmark(path: str) -> pd.Series:
    """Load Nifty 500 close-price series."""
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error(f"❌ Bad parquet file: `{Path(path).name}`. No Date index.")
        st.stop()
    if "Close" in df.columns:
        return df["Close"]
    return df.iloc[:, 0].rename("Close")


@st.cache_data(show_spinner=False)
def load_holdings():
    """Load all three holdings parquets. Returns empty DFs for missing files."""
    missing = []
    empty_at = pd.DataFrame(columns=["Scheme", "AssetType", "Allocation"])
    empty_sec = pd.DataFrame(columns=["Scheme", "Sector", "Allocation"])
    empty_stk = pd.DataFrame(columns=["Scheme", "Stock", "Sector", "Allocation"])

    p = DATA_DIR / ASSETTYPE_FILE
    at = pd.read_parquet(p) if p.exists() else (missing.append(ASSETTYPE_FILE), empty_at)[1]

    p = DATA_DIR / SECTOR_FILE
    sec = pd.read_parquet(p) if p.exists() else (missing.append(SECTOR_FILE), empty_sec)[1]

    p = DATA_DIR / STOCKS_FILE
    stk = pd.read_parquet(p) if p.exists() else (missing.append(STOCKS_FILE), empty_stk)[1]

    return at, sec, stk, missing


def _normalize_stock(name) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\bltd\.?\b|\blimited\b", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data(show_spinner=False)
def load_amfi_classification():
    p = DATA_DIR / AMFI_FILE
    if not p.exists():
        return {}, True
    amfi = pd.read_parquet(p)
    return {_normalize_stock(n): cat for n, cat in zip(amfi["Company Name"], amfi["Category"])}, False


@st.cache_data(show_spinner=False)
def load_all():
    """Load every NAV file and merge into ONE wide dataframe (no category levels).
    Returns (nav, fund_info, benchmark, asset_df, sector_df, stock_df, amfi_map, missing).
    fund_info: dict {fund_name -> {"category": str, "inception": Timestamp}}"""
    fund_info = {}
    frames = []
    for fname, tag in NAV_FILES.items():
        p = DATA_DIR / fname
        if not p.exists():
            st.error(
                f"❌ Missing NAV file: `{fname}`\n\n"
                f"Run `convert_to_parquet.py` locally to generate it from your `.xlsx` "
                f"files, then push the `.parquet` files to your repo."
            )
            st.stop()
        df = load_nav_file(str(p))
        frames.append(df)
        for col in df.columns:
            if col not in fund_info:
                first = df[col].first_valid_index()
                fund_info[col] = {
                    "category": tag,
                    "inception": first if isinstance(first, pd.Timestamp) else None,
                }

    # Concatenate ALL funds into a single wide dataframe (flat columns)
    nav = pd.concat(frames, axis=1, sort=False).sort_index().ffill(limit=5)
    # Drop any duplicate columns (in case the same fund appears in multiple files)
    nav = nav.loc[:, ~nav.columns.duplicated()]

    # Benchmark
    bench_path = DATA_DIR / BENCHMARK_FILE
    if not bench_path.exists():
        st.error(f"❌ Missing benchmark file: `{BENCHMARK_FILE}`")
        st.stop()
    bench = load_benchmark(str(bench_path))

    asset_df, sector_df, stock_df, missing = load_holdings()
    amfi_map, amfi_missing = load_amfi_classification()
    if amfi_missing:
        missing.append(AMFI_FILE)
    return nav, fund_info, bench, asset_df, sector_df, stock_df, amfi_map, missing


# =============================================================================
# Math
# =============================================================================
def build_portfolio(
    nav: pd.DataFrame, selections: list,
    start: pd.Timestamp = None, end: pd.Timestamp = None,
) -> pd.Series:
    """Build a portfolio's normalized growth series (starts at 1.0).
    selections: list of (fund_name, weight%) tuples."""
    funds = [f for f, _ in selections]
    weights = np.array([w for _, w in selections]) / 100.0
    sub = nav[funds]
    if start is not None:
        sub = sub.loc[start:]
    if end is not None:
        sub = sub.loc[:end]
    sub = sub.dropna(how="any")
    if sub.empty:
        return pd.Series(dtype=float)
    normalized = sub.divide(sub.iloc[0])
    return pd.Series(normalized.values @ weights, index=sub.index)


def rolling_cagr(series: pd.Series, window_days: int) -> pd.Series:
    if len(series) < window_days + 1:
        return pd.Series(dtype=float)
    years = window_days / 365.25
    return ((series / series.shift(window_days)) ** (1 / years) - 1).dropna()


def max_drawdown(series: pd.Series) -> float:
    """Return the worst peak-to-trough drawdown as a negative number (e.g. -0.42 = -42%)."""
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown.min()


def classify_cap(stock: str, amfi_map: dict) -> str:
    return amfi_map.get(_normalize_stock(stock), "Small Cap")


def covered_funds(funds: list, df: pd.DataFrame) -> tuple[list, list]:
    """Returns (covered, missing) from selected funds based on holdings data."""
    schemes_in_data = set(df["Scheme"].unique())
    covered = [f for f in funds if f in schemes_in_data]
    missing = [f for f in funds if f not in schemes_in_data]
    return covered, missing


# Map each raw asset-type label to one of 4 buckets: Equity / Gold / Debt / Cash.
# Anything not listed defaults to "Cash" (the safest fold-in for tiny categories
# like REITs, mutual fund units, silver, derivatives options).
_ASSET_BUCKET = {
    # Equity
    "Domestic Equities":                       "Equity",
    "Overseas Equities":                       "Equity",
    "ADRs & GDRs":                             "Equity",
    "Preference Shares":                       "Equity",
    "Derivatives-Futures":                     "Equity",
    # Gold
    "Gold":                                    "Gold",
    "Domestic Mutual Funds Units - Gold":      "Gold",
    # Debt
    "Government Securities":                   "Debt",
    "Treasury Bills":                          "Debt",
    "Corporate Debt":                          "Debt",
    "Commercial Paper":                        "Debt",
    "Certificate of Deposit":                  "Debt",
    "PTC & Securitized Debt":                  "Debt",
    # Cash
    "Cash & Cash Equivalents and Net Assets":  "Cash",
    "Deposits (Placed as Margin)":             "Cash",
}


def bucket_for_asset(asset_type: str) -> str:
    """Bucket a raw asset-type label. Unknown labels (REITs, Silver, MF Units,
    options) fold into Cash to keep the 4-bucket view clean."""
    return _ASSET_BUCKET.get(asset_type, "Cash")


def fund_asset_split(fund_name: str) -> dict:
    """Return {Equity, Gold, Debt, Cash} % split for a single fund (sums to ~100).
    Uses asset-type data when available, falls back to fund category tag otherwise."""
    rows = asset_df[asset_df["Scheme"] == fund_name]
    split = {"Equity": 0.0, "Gold": 0.0, "Debt": 0.0, "Cash": 0.0}

    if not rows.empty:
        for _, r in rows.iterrows():
            bucket = bucket_for_asset(r["AssetType"])
            split[bucket] += float(r["Allocation"])
        # Normalize in case the data sums to slightly off-100
        total = sum(split.values())
        if total > 0:
            for k in split:
                split[k] = split[k] * 100.0 / total
        return split

    # Fallback: classify by category tag
    cat = fund_info.get(fund_name, {}).get("category", "")
    if cat == "Gold ETF":
        split["Gold"] = 100.0
    elif cat == "Debt":
        split["Debt"] = 100.0
    else:
        split["Equity"] = 100.0
    return split


# =============================================================================
# Load data
# =============================================================================
with st.spinner("Loading fund data..."):
    nav, fund_info, benchmark, asset_df, sector_df, stock_df, amfi_map, missing_files = load_all()

# Sort fund list alphabetically (case-insensitive)
ALL_FUNDS = sorted(fund_info.keys(), key=lambda x: x.lower())


def fund_label(f: str) -> str:
    """Display label for a fund in the dropdown — name · category · since YEAR."""
    info = fund_info.get(f, {})
    cat = info.get("category", "")
    inc = info.get("inception")
    if isinstance(inc, pd.Timestamp):
        return f"{f}  ·  {cat}  ·  since {inc.year}"
    return f"{f}  ·  {cat}"


def oldest_fund() -> str:
    """Return any fund with the earliest inception date — used as default."""
    best, best_d = ALL_FUNDS[0], pd.Timestamp.max
    for f in ALL_FUNDS:
        d = fund_info[f].get("inception")
        if isinstance(d, pd.Timestamp) and d < best_d:
            best, best_d = f, d
    return best


# =============================================================================
# Header
# =============================================================================
st.markdown("# Portfolio Dashboard")
st.markdown(
    '<div class="subtitle">Pick any funds, set weights, compare against Nifty 500. '
    'Then explore rolling returns, growth, and holdings.</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# Fund selection — single dropdown, no category
# =============================================================================
if "selections" not in st.session_state:
    # Sensible starter: 3 well-known funds with long history
    default_funds = [
        "Aditya Birla SL Large Cap Fund-Reg(G)",
        "Aditya Birla SL Midcap Fund(G)",
        "Aditya Birla SL Small Cap Fund(G)",
    ]
    seeds = []
    for f, w in zip(default_funds, [40.0, 30.0, 30.0]):
        if f in fund_info:
            seeds.append({"fund": f, "weight": w})
    if not seeds:
        seeds = [{"fund": oldest_fund(), "weight": 100.0}]
    st.session_state.selections = seeds

st.markdown('<div class="section-label">Your portfolio</div>', unsafe_allow_html=True)


to_delete = None
for i, row in enumerate(st.session_state.selections):
    with st.container(border=True):
        c1, c2, c3 = st.columns([6, 1.3, 0.5])
        with c1:
            idx = ALL_FUNDS.index(row["fund"]) if row["fund"] in ALL_FUNDS else 0
            row["fund"] = st.selectbox(
                "Fund", ALL_FUNDS, index=idx, format_func=fund_label,
                key=f"fund_{i}", label_visibility="collapsed",
            )
        with c2:
            row["weight"] = st.number_input(
                "Weight", min_value=0.0, max_value=100.0,
                value=float(row["weight"]), step=5.0,
                key=f"w_{i}", label_visibility="collapsed",
            )
        with c3:
            st.markdown('<div style="padding-top:4px"></div>', unsafe_allow_html=True)
            if st.button("✕", key=f"del_{i}", help="Remove"):
                to_delete = i

if to_delete is not None:
    st.session_state.selections.pop(to_delete)
    # Clear widget state for removed/shifted rows
    for k in list(st.session_state.keys()):
        if k.startswith("w_") or k.startswith("fund_") or k.startswith("del_"):
            del st.session_state[k]
    st.rerun()

col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("＋ Add fund", use_container_width=True):
        # Add a new fund with 0% weight — user sets it themselves
        st.session_state.selections.append({"fund": ALL_FUNDS[0], "weight": 0.0})
        st.rerun()
with col_b:
    total = sum(r["weight"] for r in st.session_state.selections)
    color = "#10b981" if abs(total - 100) < 0.01 else "#ef4444"
    st.markdown(
        f'<div style="text-align:right; padding-top:8px; color:{color}; '
        f'font-weight:600;">Total: {total:.1f}%</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Build selections + early exits
# =============================================================================
selections = [(r["fund"], r["weight"]) for r in st.session_state.selections if r["weight"] > 0]

# Check if total is exactly 100% — if not, show a clean message and stop
if abs(total - 100) > 0.01:
    diff = 100 - total
    direction = "add" if diff > 0 else "remove"
    st.markdown(
        f'<div class="card" style="text-align:center; margin-top:1.5rem;">'
        f'  <div style="font-size:1.4rem; font-weight:700; margin-bottom:0.5rem;">'
        f'    Set weights to exactly 100% to see results'
        f'  </div>'
        f'  <div style="opacity:0.65; font-size:0.95rem;">'
        f'    Your weights sum to <b>{total:.1f}%</b>. '
        f'    {"You need to <b>" + direction + f" {abs(diff):.1f}%</b>." if total > 0 else ""}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not selections:
    st.info("Add at least one fund with a non-zero weight to see results.")
    st.stop()


# =============================================================================
# View selector — 4 tabs
# =============================================================================
st.markdown('<div class="section-label">What do you want to see?</div>', unsafe_allow_html=True)

views_all = ["📈 Performance", "🥧 Asset Mix", "🏭 Sectors", "🏢 Top Stocks"]
view = st.radio("View", views_all, horizontal=True, label_visibility="collapsed")
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# =============================================================================
# Helpers used across views
# =============================================================================
def color_for(v):
    if v > 0.10:  return "pos"
    if v < 0:     return "neg"
    return "neu"


def stat_html(label, value, klass="neu"):
    return (
        f'<div><div class="compare-stat-label">{label}</div>'
        f'<div class="compare-stat-value {klass}">{value}</div></div>'
    )


def column_html(name, dot_class, stats):
    inner = "".join(stat_html(l, v, c) for l, v, c in stats)
    return (
        f'<div class="compare-col {dot_class}">'
        f'  <div class="compare-header"><span class="dot {dot_class}"></span>{name}</div>'
        f'  <div class="compare-stats">{inner}</div>'
        f'</div>'
    )


# =============================================================================
# Rolling-returns metrics (rendered inline inside Performance view on demand)
# =============================================================================
def render_rolling_returns_metrics(start_ts: pd.Timestamp = None, end_ts: pd.Timestamp = None):
    """Render 1Y/3Y/5Y rolling return cards plus Max Drawdown comparison.
    Respects the selected date range (if provided)."""
    portfolio = build_portfolio(nav, selections, start_ts, end_ts)
    if portfolio.empty:
        st.error("No overlapping data for the selected funds in this date range.")
        return

    bench = benchmark.reindex(portfolio.index, method="ffill").bfill()
    backtest_years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25

    period_start = portfolio.index[0].strftime("%b %Y")
    period_end   = portfolio.index[-1].strftime("%b %Y")
    st.markdown(
        f'<div style="opacity:0.55; font-size:0.85rem; margin: 0.5rem 0 1rem 0;">'
        f'Based on {backtest_years:.1f} years of selected history ({period_start} to {period_end})'
        f'</div>',
        unsafe_allow_html=True,
    )

    windows = [
        ("1-Year Returns", 252,     1),
        ("3-Year Returns", 252 * 3, 3),
        ("5-Year Returns", 252 * 5, 5),
    ]

    for title, days, years_needed in windows:
        # Check we have enough history
        if backtest_years < years_needed:
            # Identify the limiting fund
            fund_inceptions = [(f, fund_info[f].get("inception")) for f, _ in selections]
            valid = [(f, d) for f, d in fund_inceptions if isinstance(d, pd.Timestamp)]
            if valid:
                youngest = max(valid, key=lambda x: x[1])
                limiter_msg = (
                    f"<b>{youngest[0]}</b> only has data since "
                    f"<b>{youngest[1].strftime('%b %Y')}</b>. Swap it for an older fund."
                )
            else:
                limiter_msg = "Pick funds with longer history."

            st.markdown(
                f'<div class="card">'
                f'  <div class="card-header">'
                f'    <div class="card-title">{title}</div>'
                f'    <div class="card-meta">Not enough data</div>'
                f'  </div>'
                f'  <div style="opacity:0.75; font-size:0.9rem;">'
                f'    Need at least {years_needed} years of overlapping data; '
                f'    you have {backtest_years:.1f} years. {limiter_msg}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            continue

        rr_p = rolling_cagr(portfolio, days)
        rr_b = rolling_cagr(bench, days)
        if rr_p.empty:
            st.markdown(
                f'<div class="card">'
                f'<div class="card-header">'
                f'<div class="card-title">{title}</div>'
                f'<div class="card-meta">Not enough data</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            continue

        # Compute stats for the rolling window
        pmn, pmd, pmx = rr_p.min(), rr_p.median(), rr_p.max()
        bmn, bmd, bmx = rr_b.min(), rr_b.median(), rr_b.max()

        # Max drawdown over the same window — what's the worst N-year drawdown experienced?
        p_dd = max_drawdown(portfolio)
        b_dd = max_drawdown(bench)

        portfolio_col = column_html("Your Portfolio", "portfolio", [
            ("Min", f"{pmn*100:.1f}%", color_for(pmn)),
            ("Median", f"{pmd*100:.1f}%", color_for(pmd)),
            ("Max", f"{pmx*100:.1f}%", color_for(pmx)),
            ("Max DD", f"{p_dd*100:.1f}%", "neg"),
        ])
        benchmark_col = column_html("Nifty 500", "benchmark", [
            ("Min", f"{bmn*100:.1f}%", color_for(bmn)),
            ("Median", f"{bmd*100:.1f}%", color_for(bmd)),
            ("Max", f"{bmx*100:.1f}%", color_for(bmx)),
            ("Max DD", f"{b_dd*100:.1f}%", "neg"),
        ])

        diff = pmd - bmd
        if abs(diff) < 0.005:
            v_cls, v_txt = "tie", f"In line with Nifty 500 ({diff*100:+.1f}% on median)"
        elif diff > 0:
            v_cls, v_txt = "win", f"↑ Portfolio beat Nifty 500 by {diff*100:.1f}% on median"
        else:
            v_cls, v_txt = "loss", f"↓ Portfolio lagged Nifty 500 by {abs(diff)*100:.1f}% on median"

        st.markdown(
            f'<div class="card">'
            f'  <div class="card-header">'
            f'    <div class="card-title">{title}</div>'
            f'  </div>'
            f'  <div class="compare-grid">{portfolio_col}{benchmark_col}</div>'
            f'  <div class="verdict {v_cls}">{v_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# VIEW: Performance (Growth Chart + optional Rolling Returns toggle)
# =============================================================================
def render_performance():
    # Find earliest valid date across selected funds
    funds_in_portfolio = [f for f, _ in selections]
    inceptions = [fund_info[f].get("inception") for f in funds_in_portfolio]
    inceptions = [d for d in inceptions if isinstance(d, pd.Timestamp)]
    if not inceptions:
        st.error("Selected funds have no valid history.")
        return
    earliest = max(inceptions)
    latest = nav.index.max()

    # Date range pickers
    st.markdown('<div class="section-label">Date range</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input(
            "From", value=earliest.date(),
            min_value=earliest.date(), max_value=latest.date(),
            key="growth_start",
        )
    with c2:
        end = st.date_input(
            "To", value=latest.date(),
            min_value=earliest.date(), max_value=latest.date(),
            key="growth_end",
        )

    if start >= end:
        st.error("Start date must be before end date.")
        return

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    invest = st.number_input(
        "Investment amount (₹)", min_value=1000.0, value=100000.0, step=10000.0,
        key="growth_invest",
    )

    portfolio = build_portfolio(nav, selections, start_ts, end_ts)
    if portfolio.empty:
        st.error("No overlapping data in selected date range.")
        return

    # Align benchmark to portfolio index. Use method="ffill" so dates not in
    # the benchmark (e.g., weekends present in the ffill'd NAV) get the prior
    # trading day's value. bfill handles any leading NaN.
    bench = benchmark.reindex(portfolio.index, method="ffill").bfill()
    bench_normalized = bench / bench.iloc[0]

    chart_df = pd.DataFrame({
        "Your Portfolio": portfolio.values * invest,
        "Nifty 500":      bench_normalized.values * invest,
    }, index=portfolio.index)

    p_final = portfolio.iloc[-1] * invest
    b_final = bench_normalized.iloc[-1] * invest
    p_return = (portfolio.iloc[-1] - 1) * 100
    b_return = (bench_normalized.iloc[-1] - 1) * 100
    years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    p_cagr = (portfolio.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else 0
    b_cagr = (bench_normalized.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else 0

    def stat_card(name, dot_class, final, total_return, cagr):
        ret_cls = "pos" if total_return >= 0 else "neg"
        return (
            f'<div class="chart-stat {dot_class}">'
            f'  <div class="chart-stat-header">'
            f'    <span class="dot {dot_class}"></span>{name}'
            f'  </div>'
            f'  <div class="chart-stat-value {ret_cls}">₹{final:,.0f}</div>'
            f'  <div class="chart-stat-meta">'
            f'    {total_return:+.1f}% total  ·  {cagr:.1f}% p.a.'
            f'  </div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="chart-stats">'
        f'  {stat_card("Your Portfolio", "portfolio", p_final, p_return, p_cagr)}'
        f'  {stat_card("Nifty 500",      "benchmark", b_final, b_return, b_cagr)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Altair chart
    import altair as alt
    chart_long = chart_df.reset_index().melt(
        id_vars="Date", var_name="Series", value_name="Value"
    )
    chart = (
        alt.Chart(chart_long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(grid=False)),
            y=alt.Y("Value:Q", title=None,
                    axis=alt.Axis(format="~s", grid=True, gridOpacity=0.15)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Your Portfolio", "Nifty 500"],
                    range=["#3b82f6", "#9ca3af"],
                ),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Value:Q", title="Value", format=",.0f"),
            ],
        )
        .properties(height=400)
        .configure_view(strokeWidth=0)
        .configure_axis(labelOpacity=0.7, tickOpacity=0.4, domainOpacity=0.2)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown(
        f'<div style="opacity:0.55; font-size:0.8rem; text-align:center; margin-top:0.5rem;">'
        f'Showing how ₹{invest:,.0f} would have grown from '
        f'{portfolio.index[0].strftime("%b %Y")} to {portfolio.index[-1].strftime("%b %Y")} '
        f'({years:.1f} years)</div>',
        unsafe_allow_html=True,
    )

    # Rolling-returns toggle (button)
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if "show_rr" not in st.session_state:
        st.session_state.show_rr = False

    label = "📊 Hide rolling returns" if st.session_state.show_rr else "📊 Show rolling returns (1Y · 3Y · 5Y)"
    if st.button(label, use_container_width=True, key="toggle_rr"):
        st.session_state.show_rr = not st.session_state.show_rr
        st.rerun()

    if st.session_state.show_rr:
        st.markdown(
            '<div class="section-label" style="margin-top:1.5rem;">Rolling returns vs Nifty 500</div>',
            unsafe_allow_html=True,
        )
        render_rolling_returns_metrics(start_ts, end_ts)


# =============================================================================
# VIEW: Asset Mix (Equity / Gold / Debt / Cash split)
# =============================================================================
def render_asset_mix():
    selected_funds = [f for f, _ in selections]

    # Aggregate the 4 buckets across the portfolio, weighted by user weights
    totals = {"Equity": 0.0, "Gold": 0.0, "Debt": 0.0, "Cash": 0.0}
    funds_no_data = []
    for f, w in selections:
        # Was this fund covered by the asset-type file? Track for warning.
        if asset_df[asset_df["Scheme"] == f].empty:
            cat = fund_info.get(f, {}).get("category", "")
            # Only flag as "no data" if we're falling back AND not Gold/Debt category
            # (those have a clean default rule)
            if cat not in ("Gold ETF", "Debt"):
                funds_no_data.append(f)

        split = fund_asset_split(f)  # returns Equity/Gold/Debt/Cash %
        for k, v in split.items():
            totals[k] += v * (w / 100.0)

    # Normalize totals so they sum to 100% (corrects for tiny rounding)
    total_sum = sum(totals.values())
    if total_sum > 0:
        for k in totals:
            totals[k] = totals[k] * 100.0 / total_sum
    else:
        st.error("Could not compute asset mix.")
        return

    if funds_no_data:
        st.info(
            f"ℹ️ Holdings data missing for **{len(funds_no_data)} fund(s)**. "
            "They've been classified as 100% Equity by default (their actual mix may differ)."
        )

    # Colours per bucket
    palette = {
        "Equity": ("#3b82f6", "equity"),
        "Gold":   ("#fbbf24", "gold"),
        "Debt":   ("#8b5cf6", "debt"),
        "Cash":   ("#9ca3af", "cash"),
    }

    segs = [(k, totals[k], palette[k][0], palette[k][1]) for k in ["Equity", "Gold", "Debt", "Cash"]]

    bar_segs = "".join(
        f'<div class="stack-seg" style="background:{c}; width:{v:.4f}%;" '
        f'title="{n}: {v:.1f}%">{(str(int(round(v))) + "%") if v >= 8 else ""}</div>'
        for n, v, c, _ in segs if v > 0.01
    )
    legend = "".join(
        f'<div class="legend-row">'
        f'  <div class="legend-left">'
        f'    <span class="dot" style="background:{palette[n][0]}"></span>'
        f'    <span class="legend-name">{n}</span>'
        f'  </div>'
        f'  <div class="legend-value">{v:.1f}%</div>'
        f'</div>'
        for n, v, _, _ in segs
    )

    st.markdown(
        '<div style="opacity:0.55; font-size:0.85rem; margin-bottom:1rem;">'
        'Where your money actually sits, by asset class. '
        'Multi-asset funds are split proportionally across buckets.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="card">'
        f'  <div class="card-header">'
        f'    <div class="card-title">Asset class split</div>'
        f'    <div class="card-meta">Equity / Gold / Debt / Cash</div>'
        f'  </div>'
        f'  <div class="stack-bar">{bar_segs}</div>'
        f'  <div class="stack-legend">{legend}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# VIEW: Sectors (equity portion only)
# =============================================================================
def render_sectors():
    selected_funds = [f for f, _ in selections]
    covered, missing = covered_funds(selected_funds, sector_df)

    if not covered:
        st.error("None of your selected funds have sector data available.")
        return

    # For each covered fund, multiply (your weight) × (fund's equity portion)
    # × (sector's allocation within that fund).
    parts = []
    total_equity_contrib = 0.0
    for f, w in selections:
        if f not in covered:
            continue
        split = fund_asset_split(f)
        equity_pct = split["Equity"] / 100.0  # fraction
        if equity_pct <= 0:
            continue
        share = (w / 100.0) * equity_pct  # share of total portfolio that's this fund's equity
        total_equity_contrib += share
        rows = sector_df[sector_df["Scheme"] == f]
        # rows["Allocation"] is in % within the fund's equity portion
        parts.append(rows.groupby("Sector")["Allocation"].sum() * share)

    if not parts or total_equity_contrib <= 0:
        st.error("No equity allocation found in your portfolio.")
        return

    # Now sector_totals is in "% of total portfolio that's in this sector"
    sector_totals = pd.concat(parts).groupby(level=0).sum().sort_values(ascending=False)

    # Tell the user how much of their portfolio is being analyzed
    equity_share = total_equity_contrib * 100
    excluded = 100.0 - equity_share
    if missing:
        st.info(
            f"ℹ️ Sector data unavailable for {len(missing)} fund(s) — "
            f"their weight is excluded."
        )

    st.markdown(
        f'<div style="opacity:0.55; font-size:0.85rem; margin-bottom:1rem;">'
        f'Analysing the equity portion of your portfolio (~{equity_share:.0f}%). '
        f'Numbers shown are <b>% of your total portfolio</b>.'
        f'</div>',
        unsafe_allow_html=True,
    )

    show_n = st.slider("How many sectors to show?", 5, 20, 10, 1, key="sector_n")
    top = sector_totals.head(show_n)
    others = sector_totals.iloc[show_n:].sum()
    max_val = top.max() if not top.empty else 1.0

    rows_html = ""
    for sector, value in top.items():
        width_pct = (value / max_val) * 100
        rows_html += (
            f'<div class="bar-row">'
            f'  <div>'
            f'    <div class="bar-label">{sector}</div>'
            f'    <div class="bar-track"><div class="bar-fill blue" style="width:{width_pct:.2f}%"></div></div>'
            f'  </div>'
            f'  <div class="bar-pct">{value:.2f}%</div>'
            f'</div>'
        )
    if others > 0:
        width_pct = (others / max_val) * 100
        rows_html += (
            f'<div class="bar-row" style="opacity:0.6;">'
            f'  <div>'
            f'    <div class="bar-label">Others ({len(sector_totals) - show_n} sectors)</div>'
            f'    <div class="bar-track"><div class="bar-fill gray" style="width:{width_pct:.2f}%"></div></div>'
            f'  </div>'
            f'  <div class="bar-pct">{others:.2f}%</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="card">'
        f'  <div class="card-header">'
        f'    <div class="card-title">Sector exposure (equity only)</div>'
        f'    <div class="card-meta">Top {show_n} sectors</div>'
        f'  </div>'
        f'  {rows_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# VIEW: Top Stocks (equity portion only)
# =============================================================================
def render_top_stocks():
    selected_funds = [f for f, _ in selections]
    covered, missing = covered_funds(selected_funds, stock_df)
    if not covered:
        st.error("None of your selected funds have stock-holdings data available.")
        return

    parts = []
    total_equity_contrib = 0.0
    for f, w in selections:
        if f not in covered:
            continue
        split = fund_asset_split(f)
        equity_pct = split["Equity"] / 100.0
        if equity_pct <= 0:
            continue
        share = (w / 100.0) * equity_pct
        total_equity_contrib += share
        rows = stock_df[stock_df["Scheme"] == f]
        parts.append(rows.groupby("Stock")["Allocation"].sum() * share)

    if not parts or total_equity_contrib <= 0:
        st.error("No equity holdings found in your portfolio.")
        return

    stock_totals = pd.concat(parts).groupby(level=0).sum().sort_values(ascending=False)

    if missing:
        st.info(f"ℹ️ Stock data unavailable for {len(missing)} fund(s) — their weight is excluded.")

    equity_share = total_equity_contrib * 100
    st.markdown(
        f'<div style="opacity:0.55; font-size:0.85rem; margin-bottom:1rem;">'
        f'Analysing the equity portion of your portfolio (~{equity_share:.0f}%). '
        f'Numbers shown are <b>% of your total portfolio</b>. '
        f'{len(stock_totals):,} unique stocks found.'
        f'</div>',
        unsafe_allow_html=True,
    )

    show_n = st.slider("How many stocks to show?", 5, 25, 10, 1, key="stocks_n")
    top = stock_totals.head(show_n)

    rows_html = ""
    for i, (stock, value) in enumerate(top.items(), start=1):
        rows_html += (
            f'<div class="stock-row" style="grid-template-columns: 28px 1fr 80px;">'
            f'  <div class="stock-rank">{i:02d}</div>'
            f'  <div class="stock-name">{stock}</div>'
            f'  <div class="stock-pct">{value:.2f}%</div>'
            f'</div>'
        )

    top_sum = top.sum()
    st.markdown(
        f'<div class="card">'
        f'  <div class="card-header">'
        f'    <div class="card-title">Top {show_n} stock holdings</div>'
        f'    <div class="card-meta">{top_sum:.1f}% of portfolio</div>'
        f'  </div>'
        f'  {rows_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Route to the chosen view
# =============================================================================
if view.endswith("Performance"):
    render_performance()
elif view.endswith("Asset Mix"):
    render_asset_mix()
elif view.endswith("Sectors"):
    render_sectors()
elif view.endswith("Top Stocks"):
    render_top_stocks()
