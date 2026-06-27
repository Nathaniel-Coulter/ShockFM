# cross_rule_cap_aware_greedy_and_subordination_sensitivity
import bisect
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")

SV_DIR = ROOT / "outputs" / "response_surface_v2" / "single_version_live_like"
XR_DIR = SV_DIR / "cross_rule_candidate_input"
XP_BASELINE_DIR = SV_DIR / "cross_rule_portfolio_sim"

OUT_DIR = SV_DIR / "cross_rule_subordination_greedy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "selected_trades": SV_DIR / "sv05_selected_trades.csv",

    "xr01_all_rows": XR_DIR / "xr01_cross_rule_candidate_input_all_rows.csv",
    "xr02_strict_baseline": XR_DIR / "xr02_current_strict_baseline_input.csv",
    "xr03_core": XR_DIR / "xr03_core_candidate_input.csv",
    "xr04_expanded": XR_DIR / "xr04_expanded_research_candidate_input.csv",

    # Optional but useful for baseline comparison.
    "baseline_template_summary": XP_BASELINE_DIR / "xp01_template_summary.csv",
    "baseline_greedy_summary": XP_BASELINE_DIR / "xp07_greedy_final_summary.csv",
}

CANONICAL_SCENARIO = "one_active_raw"

# The prior best fixed cap-aware template:
#   core_cap_grid_hdn-c10_h024-c25_h015v-raw_h015vol-c25
# Nearby sensitivities are included below.
CAP_POLICY_SPECS = {
    "best_fixed_cap_policy": {
        "description": "Prior best deployable-looking fixed cap policy.",
        "groups": {
            "high_down_r022__direction_broad": "cap_10_per_day",
            "high_up_r024__loose5_no_vol": "cap_25_per_day",
            "high_up_r015__velocity_loose10": "one_active_raw",
            "high_up_r015__vol_loose5_no_vol_no_persist": "cap_25_per_day",
        },
    },
    "more_conservative_dd_policy": {
        "description": "Same as best policy, but more conservative on high_up_r024.",
        "groups": {
            "high_down_r022__direction_broad": "cap_10_per_day",
            "high_up_r024__loose5_no_vol": "cap_25_cooldown_30s",
            "high_up_r015__velocity_loose10": "one_active_raw",
            "high_up_r015__vol_loose5_no_vol_no_persist": "cap_25_per_day",
        },
    },
    "higher_pnl_cap_policy": {
        "description": "Looser high_down_r022 cap sensitivity.",
        "groups": {
            "high_down_r022__direction_broad": "cap_25_per_day",
            "high_up_r024__loose5_no_vol": "cap_25_per_day",
            "high_up_r015__velocity_loose10": "one_active_raw",
            "high_up_r015__vol_loose5_no_vol_no_persist": "cap_25_per_day",
        },
    },
    "more_conservative_both_broad_policy": {
        "description": "Conservative on both broad low-margin sleeves.",
        "groups": {
            "high_down_r022__direction_broad": "cap_10_per_day",
            "high_up_r024__loose5_no_vol": "cap_25_cooldown_30s",
            "high_up_r015__velocity_loose10": "one_active_raw",
            "high_up_r015__vol_loose5_no_vol_no_persist": "cap_25_cooldown_30s",
        },
    },
}

SUBORDINATION_MODES = [
    "none",
    "candidate_key_exact",
    "same_entry_ts",
    "same_entry_ts_same_side",
    "entry_within_250ms",
    "entry_within_500ms",
    "dominant_active_window",
    "dominant_active_window_or_500ms",
]

# Greedy marginal construction.
RUN_GREEDY = True
GREEDY_OBJECTIVE = "total_pnl"
GREEDY_MIN_MARGINAL_OBJECTIVE = 0.0
WRITE_GREEDY_TRIAL_EVALS = True

# Trade detail for best outputs.
WRITE_TOP_DETAIL = True
TOP_N_DETAIL_PORTFOLIOS = 20

# If invalid exit timestamp, use only for blocking interval.
FALLBACK_HOLD_SECONDS = 10.0

# Terminal display.
SHOW_TOP_N = 80
SHOW_GREEDY_STEPS_N = 200


# =============================================================================
# CURRENT LIVE SUBORDINATION HYPOTHESES TO TEST
# =============================================================================

SUBORDINATED_RULES = {
    "PARENT__high_up_r022": [
        "PARENT__high_up_r014",
        "PARENT__high_up_r010",
        "PARENT__high_up_r012",
    ],
    "PARENT__mid_up_r022": [
        "PARENT__mid_up_r009",
        "PARENT__mid_up_r004",
    ],
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[ABORT] Missing {name}: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[READ] {name}: {len(df):,} rows, {len(df.columns):,} cols -> {path}")
    return df


def read_csv_optional(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Optional file missing: {name}: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    print(f"[READ] {name}: {len(df):,} rows, {len(df.columns):,} cols -> {path}")
    return df


def text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def num(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def int0(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0


def bool_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0


def pct(n, d) -> float:
    d = num(d, 0.0)
    if d == 0 or np.isnan(d):
        return np.nan
    return float(num(n, 0.0) / d)


def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.|=-]+", "_", s)
    return s[:220]


def scenario_abbrev(s: str) -> str:
    return {
        "one_active_raw": "raw",
        "cap_10_per_day": "c10",
        "cap_25_per_day": "c25",
        "cooldown_30s": "cd30",
        "cap_25_cooldown_30s": "c25cd30",
    }.get(str(s), safe_name(str(s)))


def max_drawdown_from_pnl(pnls: pd.Series) -> float:
    if pnls is None or len(pnls) == 0:
        return 0.0
    x = pd.to_numeric(pnls, errors="coerce").fillna(0.0).to_numpy()
    if len(x) == 0:
        return 0.0
    cum = np.cumsum(x)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def top5_abs_day_concentration(day_pnl: pd.Series) -> float:
    if day_pnl is None or len(day_pnl) == 0:
        return np.nan
    vals = pd.to_numeric(day_pnl, errors="coerce").fillna(0.0).abs().sort_values(ascending=False)
    denom = vals.sum()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(vals.head(5).sum() / denom)


def risk_adjusted_score(total_pnl, max_dd, p95_trades_day, top5_conc, selected_trades) -> float:
    """
    Not a final trading utility. This is a triage score to rank candidate portfolio shapes.
    """
    total_pnl = num(total_pnl, 0.0)
    max_dd = num(max_dd, 0.0)
    p95 = num(p95_trades_day, 0.0)
    conc = num(top5_conc, 0.0)
    n = num(selected_trades, 0.0)

    return float(
        total_pnl
        - 0.50 * abs(min(max_dd, 0.0))
        - 0.03 * max(p95 - 75.0, 0.0)
        - 50.0 * max(conc - 0.50, 0.0)
        - 0.0005 * max(n - 20000.0, 0.0)
    )


# =============================================================================
# DATA NORMALIZATION
# =============================================================================

def normalize_selected_trades(selected: pd.DataFrame) -> pd.DataFrame:
    s = selected.copy()

    required = ["registry_id", "scenario", "candidate_key", "day", "entry_ts_ns", "exit_ts_ns", "pnl_pts"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise SystemExit(f"[ABORT] sv05_selected_trades missing columns: {missing}")

    s["registry_id"] = s["registry_id"].astype(str)
    s["scenario"] = s["scenario"].astype(str)
    s["candidate_key"] = s["candidate_key"].astype(str)
    s["day"] = s["day"].astype(str)

    for c in ["entry_ts_ns", "exit_ts_ns", "pnl_pts", "hold_seconds", "mfe_pts", "mae_pts"]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")

    fallback_ns = int(FALLBACK_HOLD_SECONDS * 1_000_000_000)
    bad_exit = s["exit_ts_ns"].isna() | s["entry_ts_ns"].isna() | (s["exit_ts_ns"] <= s["entry_ts_ns"])
    s.loc[bad_exit & s["entry_ts_ns"].notna(), "exit_ts_ns"] = (
        s.loc[bad_exit & s["entry_ts_ns"].notna(), "entry_ts_ns"] + fallback_ns
    )

    if "exit_reason" not in s.columns:
        s["exit_reason"] = ""

    return s


def selected_lookup(selected: pd.DataFrame) -> dict:
    out = {}
    for (rid, scenario), g in selected.groupby(["registry_id", "scenario"], dropna=False):
        out[(str(rid), str(scenario))] = g.copy()
    return out


def candidate_quality_score(row) -> float:
    total = num(row.get("selected_total_pnl_pts"), 0.0)
    avg = num(row.get("selected_avg_pnl_pts"), 0.0)
    trades = max(num(row.get("selected_trades"), 0.0), 1.0)
    dd = num(row.get("selected_max_drawdown_pts"), 0.0)
    conc = num(row.get("selected_top5_abs_day_concentration"), 0.0)
    active_block = num(row.get("active_block_rate"), 0.0)
    cap = bool_int(row.get("requires_cap_sweep"))

    score = 0.0
    score += total
    score += 125.0 * avg
    score += 8.0 * math.log10(max(trades, 1.0))
    score -= 0.25 * abs(min(dd, 0.0))
    score -= 35.0 * max(conc - 0.50, 0.0)
    score -= 20.0 * max(active_block - 0.85, 0.0)
    score -= 10.0 * cap
    return float(score)


def prepare_candidate_df(df: pd.DataFrame, set_name: str) -> pd.DataFrame:
    c = df.copy()
    c["candidate_set_name"] = set_name
    c["registry_id"] = c["registry_id"].astype(str)

    for col in [
        "selected_total_pnl_pts",
        "selected_avg_pnl_pts",
        "selected_trades",
        "selected_max_drawdown_pts",
        "selected_top5_abs_day_concentration",
        "active_block_rate",
        "current_live_priority",
        "requires_cap_sweep",
        "manual_review_flag",
        "include_in_core_portfolio_candidate_set",
        "include_in_expanded_research_candidate_set",
    ]:
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")

    for col in [
        "matched_rule",
        "side",
        "cross_rule_status",
        "cross_rule_lane",
        "cross_rule_role",
        "version_short_name",
        "representative_group",
        "candidate_input_notes",
    ]:
        if col not in c.columns:
            c[col] = ""

    c["priority_quality_score"] = c.apply(candidate_quality_score, axis=1)
    return c


# =============================================================================
# CAP POLICY ASSIGNMENT
# =============================================================================

def build_policy_map(core_df: pd.DataFrame, policy_name: str) -> tuple[dict, str]:
    if policy_name not in CAP_POLICY_SPECS:
        raise ValueError(f"Unknown cap policy {policy_name}")

    spec = CAP_POLICY_SPECS[policy_name]
    rg_to_rid = dict(zip(core_df["representative_group"].astype(str), core_df["registry_id"].astype(str)))

    policy_map = {}
    pieces = []

    for rg, scenario in spec["groups"].items():
        rid = rg_to_rid.get(rg)
        if rid is None:
            print(f"[WARN] cap policy {policy_name}: representative_group missing: {rg}")
            continue
        policy_map[rid] = scenario
        pieces.append(f"{rg}->{scenario_abbrev(scenario)}")

    label = "|".join(pieces) if pieces else "all_raw"
    return policy_map, label


def apply_source_policy(cands: pd.DataFrame, policy_name: str, policy_map: dict) -> pd.DataFrame:
    out = cands.copy()
    out["source_policy_name"] = policy_name
    out["source_scenario"] = out["registry_id"].astype(str).map(policy_map).fillna(CANONICAL_SCENARIO)
    return out


# =============================================================================
# PRIORITY ORDERING
# =============================================================================

def lane_bucket(row) -> int:
    status = text(row.get("cross_rule_status"))
    lane = text(row.get("cross_rule_lane"))

    if status == "active_strict_reference_strong":
        return 0
    if status == "active_main_representative":
        return 1
    if status == "active_cap_shaped_main":
        return 2
    if lane == "manual_review":
        return 3
    if lane == "cap_diagnostic":
        return 4
    if status == "active_strict_reference_control":
        return 5
    return 9


def order_candidates(cands: pd.DataFrame, order_type: str) -> pd.DataFrame:
    c = cands.copy()
    c["lane_bucket"] = c.apply(lane_bucket, axis=1)
    c["current_live_priority_sort"] = pd.to_numeric(c.get("current_live_priority", 9999), errors="coerce").fillna(9999)

    for col in [
        "priority_quality_score",
        "selected_total_pnl_pts",
        "selected_avg_pnl_pts",
        "selected_trades",
        "selected_max_drawdown_pts",
    ]:
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0.0)

    if order_type == "strict_strong_quality":
        c = c.sort_values(
            ["lane_bucket", "priority_quality_score", "selected_total_pnl_pts", "selected_avg_pnl_pts", "registry_id"],
            ascending=[True, False, False, False, True],
        )

    elif order_type == "quality_no_strict_bias":
        c["quality_bucket"] = np.where(
            c["cross_rule_lane"].isin(["strict_reference", "main_representative", "cap_shaped_main"]),
            0,
            c["lane_bucket"],
        )
        c = c.sort_values(
            ["quality_bucket", "priority_quality_score", "selected_total_pnl_pts", "selected_avg_pnl_pts", "registry_id"],
            ascending=[True, False, False, False, True],
        )

    elif order_type == "total_pnl_first":
        c = c.sort_values(
            ["selected_total_pnl_pts", "selected_avg_pnl_pts", "selected_trades", "registry_id"],
            ascending=[False, False, False, True],
        )

    elif order_type == "avg_pnl_first":
        c = c.sort_values(
            ["selected_avg_pnl_pts", "selected_total_pnl_pts", "selected_trades", "registry_id"],
            ascending=[False, False, False, True],
        )

    elif order_type == "current_live_priority":
        c = c.sort_values(
            ["current_live_priority_sort", "lane_bucket", "selected_total_pnl_pts", "registry_id"],
            ascending=[True, True, False, True],
        )

    elif order_type == "given_order":
        if "priority_rank" not in c.columns:
            raise ValueError("given_order requires priority_rank")
        c = c.sort_values(["priority_rank", "registry_id"], ascending=[True, True])

    else:
        raise ValueError(f"Unknown order_type={order_type}")

    c = c.reset_index(drop=True)
    c["priority_rank"] = np.arange(1, len(c) + 1)
    return c


# =============================================================================
# EVENT BUILDING
# =============================================================================

def build_events(cands: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    parts = []

    meta_cols = [
        "registry_id",
        "matched_rule",
        "side",
        "cross_rule_status",
        "cross_rule_lane",
        "cross_rule_role",
        "version_short_name",
        "representative_group",
        "requires_cap_sweep",
        "manual_review_flag",
        "priority_rank",
        "priority_quality_score",
        "source_policy_name",
        "source_scenario",
        "candidate_set_name",
        "current_live_priority",
    ]

    for _, row in cands.iterrows():
        rid = text(row.get("registry_id"))
        scenario = text(row.get("source_scenario"))
        g = lookup.get((rid, scenario))

        if g is None or g.empty:
            continue

        h = g.copy()
        for col in meta_cols:
            h[col] = row.get(col, "")

        parts.append(h)

    if not parts:
        return pd.DataFrame()

    events = pd.concat(parts, ignore_index=True)

    events["registry_id"] = events["registry_id"].astype(str)
    events["matched_rule"] = events["matched_rule"].astype(str)
    events["side"] = events["side"].astype(str)
    events["candidate_key"] = events["candidate_key"].astype(str)
    events["day"] = events["day"].astype(str)
    events["priority_rank"] = pd.to_numeric(events["priority_rank"], errors="coerce").fillna(999999).astype(int)

    for c in ["entry_ts_ns", "exit_ts_ns", "pnl_pts"]:
        events[c] = pd.to_numeric(events[c], errors="coerce")

    events = events.dropna(subset=["entry_ts_ns"]).copy()

    fallback_ns = int(FALLBACK_HOLD_SECONDS * 1_000_000_000)
    bad_exit = events["exit_ts_ns"].isna() | (events["exit_ts_ns"] <= events["entry_ts_ns"])
    events.loc[bad_exit, "exit_ts_ns"] = events.loc[bad_exit, "entry_ts_ns"] + fallback_ns

    events["pnl_pts"] = pd.to_numeric(events["pnl_pts"], errors="coerce").fillna(0.0)

    events["_event_id"] = np.arange(len(events))
    return events


# =============================================================================
# SUBORDINATION MODES
# =============================================================================

def parse_ms_from_mode(mode: str) -> int:
    if "250ms" in mode:
        return 250
    if "500ms" in mode:
        return 500
    return 0


def build_day_dom_struct(dom_day: pd.DataFrame) -> dict:
    d = dom_day.sort_values("entry_ts_ns").reset_index(drop=True).copy()
    entries = pd.to_numeric(d["entry_ts_ns"], errors="coerce").to_numpy(dtype=np.float64)
    exits = pd.to_numeric(d["exit_ts_ns"], errors="coerce").to_numpy(dtype=np.float64)

    if len(d) == 0:
        return {"df": d, "entries": entries, "exits": exits, "cummax_exit": np.array([]), "cummax_idx": np.array([])}

    cummax_exit = np.empty(len(exits), dtype=np.float64)
    cummax_idx = np.empty(len(exits), dtype=np.int64)

    best_exit = -np.inf
    best_idx = -1
    for i, ex in enumerate(exits):
        if ex > best_exit:
            best_exit = ex
            best_idx = i
        cummax_exit[i] = best_exit
        cummax_idx[i] = best_idx

    return {
        "df": d,
        "entries": entries,
        "exits": exits,
        "cummax_exit": cummax_exit,
        "cummax_idx": cummax_idx,
    }


def nearest_entry_match(struct: dict, entry: float, max_ms: int):
    entries = struct["entries"]
    if entries.size == 0 or np.isnan(entry):
        return None

    max_ns = max_ms * 1_000_000
    pos = bisect.bisect_left(entries, entry)

    candidates = []
    if pos < len(entries):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)

    best_idx = None
    best_dist = None
    for idx in candidates:
        dist = abs(float(entries[idx]) - entry)
        if dist <= max_ns and (best_dist is None or dist < best_dist):
            best_idx = idx
            best_dist = dist

    return best_idx


def exact_entry_match(struct: dict, entry: float):
    entries = struct["entries"]
    if entries.size == 0 or np.isnan(entry):
        return None

    pos = bisect.bisect_left(entries, entry)
    if pos < len(entries) and entries[pos] == entry:
        return pos
    return None


def active_window_match(struct: dict, entry: float):
    entries = struct["entries"]
    if entries.size == 0 or np.isnan(entry):
        return None

    pos = bisect.bisect_right(entries, entry) - 1
    if pos < 0:
        return None

    if struct["cummax_exit"][pos] > entry:
        return int(struct["cummax_idx"][pos])
    return None


def apply_subordination(events: pd.DataFrame, mode: str):
    if mode == "none" or events.empty:
        return events.copy(), pd.DataFrame()

    e = events.copy()
    e["_subordination_blocked"] = False

    blocked_parts = []

    for dominant_rule, subordinate_rules in SUBORDINATED_RULES.items():
        dom_all = e[(e["matched_rule"] == dominant_rule) & (~e["_subordination_blocked"])].copy()
        sub_all = e[(e["matched_rule"].isin(subordinate_rules)) & (~e["_subordination_blocked"])].copy()

        if dom_all.empty or sub_all.empty:
            continue

        for day, sub_day in sub_all.groupby("day", dropna=False):
            dom_day = dom_all[dom_all["day"].eq(day)].copy()
            if dom_day.empty:
                continue

            struct = build_day_dom_struct(dom_day)
            ddf = struct["df"]

            # Precomputed exact sets for candidate_key modes.
            if mode == "candidate_key_exact":
                dom_keys = set(ddf["candidate_key"].astype(str))
            elif mode == "same_entry_ts_same_side":
                dom_by_side_entry = set(zip(ddf["side"].astype(str), ddf["entry_ts_ns"].astype(float)))
            else:
                dom_keys = set()
                dom_by_side_entry = set()

            for idx, srow in sub_day.iterrows():
                if e.at[idx, "_subordination_blocked"]:
                    continue

                entry = num(srow.get("entry_ts_ns"))
                blocker_idx = None
                relation_reason = ""

                if mode == "candidate_key_exact":
                    if text(srow.get("candidate_key")) in dom_keys:
                        # Choose first dominant with same candidate key.
                        matches = ddf[ddf["candidate_key"].astype(str).eq(text(srow.get("candidate_key")))]
                        if not matches.empty:
                            blocker_idx = int(matches.index[0])
                            relation_reason = "candidate_key_exact"

                elif mode == "same_entry_ts":
                    blocker_idx = exact_entry_match(struct, entry)
                    relation_reason = "same_entry_ts"

                elif mode == "same_entry_ts_same_side":
                    key = (text(srow.get("side")), entry)
                    if key in dom_by_side_entry:
                        matches = ddf[
                            ddf["side"].astype(str).eq(text(srow.get("side")))
                            & ddf["entry_ts_ns"].astype(float).eq(entry)
                        ]
                        if not matches.empty:
                            blocker_idx = int(matches.index[0])
                            relation_reason = "same_entry_ts_same_side"

                elif mode in {"entry_within_250ms", "entry_within_500ms"}:
                    ms = parse_ms_from_mode(mode)
                    blocker_idx = nearest_entry_match(struct, entry, ms)
                    relation_reason = mode

                elif mode == "dominant_active_window":
                    blocker_idx = active_window_match(struct, entry)
                    relation_reason = "dominant_active_window"

                elif mode == "dominant_active_window_or_500ms":
                    blocker_idx = active_window_match(struct, entry)
                    relation_reason = "dominant_active_window"
                    if blocker_idx is None:
                        blocker_idx = nearest_entry_match(struct, entry, 500)
                        relation_reason = "entry_within_500ms_fallback"

                else:
                    raise ValueError(f"Unknown subordination mode: {mode}")

                if blocker_idx is None:
                    continue

                brow = ddf.iloc[blocker_idx]

                e.at[idx, "_subordination_blocked"] = True

                out = srow.copy()
                out["portfolio_status"] = "blocked_subordination"
                out["block_reason"] = f"subordination:{mode}"
                out["subordination_mode"] = mode
                out["subordination_relation_reason"] = relation_reason
                out["dominant_rule"] = dominant_rule
                out["subordinate_rule"] = text(srow.get("matched_rule"))
                out["blocked_by_registry_id"] = text(brow.get("registry_id"))
                out["blocked_by_matched_rule"] = text(brow.get("matched_rule"))
                out["blocked_by_candidate_key"] = text(brow.get("candidate_key"))
                out["blocked_by_entry_ts_ns"] = num(brow.get("entry_ts_ns"))
                out["blocked_by_exit_ts_ns"] = num(brow.get("exit_ts_ns"))
                out["blocked_by_source_scenario"] = text(brow.get("source_scenario"))
                blocked_parts.append(out.to_frame().T)

    blocked = pd.concat(blocked_parts, ignore_index=True) if blocked_parts else pd.DataFrame()
    filtered = e[~e["_subordination_blocked"]].drop(columns=["_subordination_blocked"]).copy()

    return filtered, blocked


# =============================================================================
# GLOBAL ONE-ACTIVE SIMULATION
# =============================================================================

def simulate_one_active(events: pd.DataFrame):
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    accepted_rows = []
    blocked_rows = []

    events = events.sort_values(
        ["day", "entry_ts_ns", "priority_rank", "registry_id", "candidate_key"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    for day, g in events.groupby("day", dropna=False, sort=False):
        active_until = -1.0
        active_registry_id = ""
        active_rule = ""
        active_candidate_key = ""

        for _, row in g.iterrows():
            entry = num(row.get("entry_ts_ns"))
            exit_ts = num(row.get("exit_ts_ns"))
            if np.isnan(entry):
                continue

            if np.isnan(exit_ts) or exit_ts <= entry:
                exit_ts = entry + int(FALLBACK_HOLD_SECONDS * 1_000_000_000)

            if entry >= active_until:
                out = row.copy()
                out["portfolio_status"] = "selected"
                out["block_reason"] = ""
                out["blocked_by_registry_id"] = ""
                out["blocked_by_matched_rule"] = ""
                out["blocked_by_candidate_key"] = ""
                accepted_rows.append(out)

                active_until = exit_ts
                active_registry_id = text(row.get("registry_id"))
                active_rule = text(row.get("matched_rule"))
                active_candidate_key = text(row.get("candidate_key"))
            else:
                out = row.copy()
                out["portfolio_status"] = "blocked_active_trade"
                out["block_reason"] = "active_trade"
                out["blocked_by_registry_id"] = active_registry_id
                out["blocked_by_matched_rule"] = active_rule
                out["blocked_by_candidate_key"] = active_candidate_key
                out["active_until_ns"] = active_until
                blocked_rows.append(out)

    accepted = pd.DataFrame(accepted_rows)
    blocked = pd.DataFrame(blocked_rows)
    return accepted, blocked


# =============================================================================
# METRICS AND CONTRIBUTION
# =============================================================================

def selected_metrics(selected: pd.DataFrame) -> dict:
    if selected is None or selected.empty:
        return {
            "selected_trades": 0,
            "selected_total_pnl_pts": 0.0,
            "selected_avg_pnl_pts": np.nan,
            "selected_win_rate": np.nan,
            "selected_max_drawdown_pts": 0.0,
            "selected_active_days": 0,
            "selected_positive_day_rate": np.nan,
            "selected_avg_trades_per_active_day": np.nan,
            "selected_p95_trades_per_active_day": np.nan,
            "selected_max_trades_one_day": 0,
            "selected_top5_abs_day_concentration": np.nan,
        }

    s = selected.copy().sort_values(["day", "entry_ts_ns", "priority_rank"])
    pnl = pd.to_numeric(s["pnl_pts"], errors="coerce").fillna(0.0)

    total = float(pnl.sum())
    n = int(len(s))
    avg = pct(total, n)
    win = float((pnl > 0).mean()) if n else np.nan
    dd = max_drawdown_from_pnl(pnl)

    day_pnl = s.groupby("day")["pnl_pts"].sum()
    day_trades = s.groupby("day")["candidate_key"].count()

    active_days = int(len(day_pnl))
    positive_day_rate = float((day_pnl > 0).mean()) if active_days else np.nan
    avg_trades_day = float(day_trades.mean()) if len(day_trades) else np.nan
    p95_trades_day = float(day_trades.quantile(0.95)) if len(day_trades) else np.nan
    max_trades_day = int(day_trades.max()) if len(day_trades) else 0
    top5_conc = top5_abs_day_concentration(day_pnl)

    return {
        "selected_trades": n,
        "selected_total_pnl_pts": total,
        "selected_avg_pnl_pts": avg,
        "selected_win_rate": win,
        "selected_max_drawdown_pts": dd,
        "selected_active_days": active_days,
        "selected_positive_day_rate": positive_day_rate,
        "selected_avg_trades_per_active_day": avg_trades_day,
        "selected_p95_trades_per_active_day": p95_trades_day,
        "selected_max_trades_one_day": max_trades_day,
        "selected_top5_abs_day_concentration": top5_conc,
    }


def simulate_portfolio(
    portfolio_name: str,
    portfolio_family: str,
    candidate_set_name: str,
    cands: pd.DataFrame,
    lookup: dict,
    subordination_mode: str,
    cap_policy_name: str,
    cap_policy_label: str,
    return_detail: bool = False,
):
    all_events = build_events(cands, lookup)

    post_sub, sub_blocked = apply_subordination(all_events, subordination_mode)
    accepted, active_blocked = simulate_one_active(post_sub)

    metrics = selected_metrics(accepted)

    total_events = int(len(all_events))
    sub_n = int(len(sub_blocked)) if sub_blocked is not None and not sub_blocked.empty else 0
    act_n = int(len(active_blocked)) if active_blocked is not None and not active_blocked.empty else 0

    sub_pnl = (
        float(pd.to_numeric(sub_blocked["pnl_pts"], errors="coerce").fillna(0.0).sum())
        if sub_blocked is not None and not sub_blocked.empty else 0.0
    )
    act_pnl = (
        float(pd.to_numeric(active_blocked["pnl_pts"], errors="coerce").fillna(0.0).sum())
        if active_blocked is not None and not active_blocked.empty else 0.0
    )

    selected = accepted.copy()

    strict_pnl = 0.0
    main_pnl = 0.0
    cap_pnl = 0.0
    manual_pnl = 0.0
    cap_diag_pnl = 0.0

    if not selected.empty:
        strict_pnl = float(selected.loc[selected["cross_rule_lane"].eq("strict_reference"), "pnl_pts"].sum())
        main_pnl = float(selected.loc[selected["cross_rule_lane"].eq("main_representative"), "pnl_pts"].sum())
        cap_pnl = float(selected.loc[selected["cross_rule_lane"].eq("cap_shaped_main"), "pnl_pts"].sum())
        manual_pnl = float(selected.loc[selected["cross_rule_lane"].eq("manual_review"), "pnl_pts"].sum())
        cap_diag_pnl = float(selected.loc[selected["cross_rule_lane"].eq("cap_diagnostic"), "pnl_pts"].sum())

    ras = risk_adjusted_score(
        metrics["selected_total_pnl_pts"],
        metrics["selected_max_drawdown_pts"],
        metrics["selected_p95_trades_per_active_day"],
        metrics["selected_top5_abs_day_concentration"],
        metrics["selected_trades"],
    )

    summary = {
        "portfolio_name": portfolio_name,
        "portfolio_family": portfolio_family,
        "candidate_set_name": candidate_set_name,
        "candidate_rows": int(len(cands)),
        "subordination_mode": subordination_mode,
        "cap_policy_name": cap_policy_name,
        "cap_policy_label": cap_policy_label,
        "source_events_seen": total_events,
        **metrics,
        "blocked_subordination_count": sub_n,
        "blocked_subordination_pnl_pts": sub_pnl,
        "blocked_active_count": act_n,
        "blocked_active_pnl_pts": act_pnl,
        "selected_fraction_of_events": pct(metrics["selected_trades"], total_events),
        "subordination_block_rate": pct(sub_n, total_events),
        "active_block_rate_after_subordination": pct(act_n, act_n + metrics["selected_trades"]),
        "strict_selected_pnl_pts": strict_pnl,
        "main_rep_selected_pnl_pts": main_pnl,
        "cap_shaped_selected_pnl_pts": cap_pnl,
        "manual_selected_pnl_pts": manual_pnl,
        "cap_diagnostic_selected_pnl_pts": cap_diag_pnl,
        "strict_pnl_share": pct(strict_pnl, metrics["selected_total_pnl_pts"]),
        "main_rep_pnl_share": pct(main_pnl, metrics["selected_total_pnl_pts"]),
        "cap_shaped_pnl_share": pct(cap_pnl, metrics["selected_total_pnl_pts"]),
        "manual_pnl_share": pct(manual_pnl, metrics["selected_total_pnl_pts"]),
        "risk_adjusted_score": ras,
    }

    contrib = candidate_contribution(portfolio_name, cands, all_events, sub_blocked, accepted, active_blocked)

    if return_detail:
        if not accepted.empty:
            accepted = accepted.copy()
            accepted["portfolio_name"] = portfolio_name
            accepted["subordination_mode"] = subordination_mode
            accepted["cap_policy_name"] = cap_policy_name

        blocked_parts = []
        if active_blocked is not None and not active_blocked.empty:
            a = active_blocked.copy()
            a["portfolio_name"] = portfolio_name
            a["subordination_mode"] = subordination_mode
            a["cap_policy_name"] = cap_policy_name
            blocked_parts.append(a)
        if sub_blocked is not None and not sub_blocked.empty:
            sb = sub_blocked.copy()
            sb["portfolio_name"] = portfolio_name
            sb["subordination_mode"] = subordination_mode
            sb["cap_policy_name"] = cap_policy_name
            blocked_parts.append(sb)

        blocked_detail = pd.concat(blocked_parts, ignore_index=True) if blocked_parts else pd.DataFrame()
        return summary, contrib, accepted, blocked_detail

    return summary, contrib, pd.DataFrame(), pd.DataFrame()


def aggregate_by_registry(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["registry_id", f"{prefix}_count", f"{prefix}_pnl"])
    return (
        df.groupby("registry_id", dropna=False)
        .agg(
            **{
                f"{prefix}_count": ("candidate_key", "count"),
                f"{prefix}_pnl": ("pnl_pts", "sum"),
            }
        )
        .reset_index()
    )


def candidate_contribution(
    portfolio_name: str,
    cands: pd.DataFrame,
    all_events: pd.DataFrame,
    sub_blocked: pd.DataFrame,
    accepted: pd.DataFrame,
    active_blocked: pd.DataFrame,
) -> pd.DataFrame:
    contrib = cands.copy()

    source = aggregate_by_registry(all_events, "source")
    selected = aggregate_by_registry(accepted, "portfolio_selected")
    sub = aggregate_by_registry(sub_blocked, "blocked_subordination")
    active = aggregate_by_registry(active_blocked, "blocked_active")

    contrib = contrib.merge(source, on="registry_id", how="left")
    contrib = contrib.merge(selected, on="registry_id", how="left")
    contrib = contrib.merge(sub, on="registry_id", how="left")
    contrib = contrib.merge(active, on="registry_id", how="left")

    for col in [
        "source_count",
        "source_pnl",
        "portfolio_selected_count",
        "portfolio_selected_pnl",
        "blocked_subordination_count",
        "blocked_subordination_pnl",
        "blocked_active_count",
        "blocked_active_pnl",
    ]:
        if col not in contrib.columns:
            contrib[col] = 0
        contrib[col] = pd.to_numeric(contrib[col], errors="coerce").fillna(0.0)

    contrib["portfolio_name"] = portfolio_name
    contrib["portfolio_selected_avg_pnl"] = contrib.apply(
        lambda r: pct(r["portfolio_selected_pnl"], r["portfolio_selected_count"]), axis=1
    )
    contrib["portfolio_residual_trade_ratio"] = contrib.apply(
        lambda r: pct(r["portfolio_selected_count"], r["source_count"]), axis=1
    )
    contrib["portfolio_residual_value_ratio"] = contrib.apply(
        lambda r: pct(r["portfolio_selected_pnl"], r["source_pnl"]) if r["source_pnl"] != 0 else np.nan,
        axis=1,
    )

    if active_blocked is not None and not active_blocked.empty and "blocked_by_registry_id" in active_blocked.columns:
        blockers = (
            active_blocked.groupby(["registry_id", "blocked_by_registry_id"], dropna=False)
            .agg(
                largest_active_blocker_count=("candidate_key", "count"),
                largest_active_blocker_blocked_pnl=("pnl_pts", "sum"),
            )
            .reset_index()
            .sort_values(["registry_id", "largest_active_blocker_count"], ascending=[True, False])
            .groupby("registry_id", as_index=False)
            .head(1)
            .rename(columns={"blocked_by_registry_id": "largest_active_blocker_registry_id"})
        )
        contrib = contrib.merge(blockers, on="registry_id", how="left")
    else:
        contrib["largest_active_blocker_registry_id"] = ""
        contrib["largest_active_blocker_count"] = 0
        contrib["largest_active_blocker_blocked_pnl"] = 0.0

    keep_cols = [
        "portfolio_name",
        "priority_rank",
        "registry_id",
        "matched_rule",
        "side",
        "cross_rule_status",
        "cross_rule_lane",
        "cross_rule_role",
        "version_short_name",
        "representative_group",
        "source_policy_name",
        "source_scenario",
        "requires_cap_sweep",
        "manual_review_flag",
        "priority_quality_score",
        "source_count",
        "source_pnl",
        "portfolio_selected_count",
        "portfolio_selected_pnl",
        "portfolio_selected_avg_pnl",
        "blocked_subordination_count",
        "blocked_subordination_pnl",
        "blocked_active_count",
        "blocked_active_pnl",
        "portfolio_residual_trade_ratio",
        "portfolio_residual_value_ratio",
        "largest_active_blocker_registry_id",
        "largest_active_blocker_count",
        "largest_active_blocker_blocked_pnl",
    ]
    return contrib[[c for c in keep_cols if c in contrib.columns]].copy()


# =============================================================================
# GREEDY MARGINAL BUILDER
# =============================================================================

def objective(summary: dict) -> float:
    if GREEDY_OBJECTIVE == "total_pnl":
        return num(summary.get("selected_total_pnl_pts"), 0.0)
    if GREEDY_OBJECTIVE == "risk_adjusted_score":
        return num(summary.get("risk_adjusted_score"), 0.0)
    raise ValueError(f"Unknown GREEDY_OBJECTIVE={GREEDY_OBJECTIVE}")


def simulate_ordered_candidate_list(
    name: str,
    family: str,
    cands: pd.DataFrame,
    lookup: dict,
    subordination_mode: str,
    cap_policy_name: str,
    cap_policy_label: str,
):
    ordered = cands.copy().reset_index(drop=True)
    ordered["priority_rank"] = np.arange(1, len(ordered) + 1)

    summary, contrib, _, _ = simulate_portfolio(
        portfolio_name=name,
        portfolio_family=family,
        candidate_set_name="greedy",
        cands=ordered,
        lookup=lookup,
        subordination_mode=subordination_mode,
        cap_policy_name=cap_policy_name,
        cap_policy_label=cap_policy_label,
        return_detail=False,
    )
    return summary, contrib


def greedy_build(
    greedy_name: str,
    base_cands: pd.DataFrame,
    pool_cands: pd.DataFrame,
    lookup: dict,
    subordination_mode: str,
    cap_policy_name: str,
    cap_policy_label: str,
):
    base = base_cands.copy()
    pool = pool_cands.copy()

    if not base.empty:
        base = base.reset_index(drop=True)
        base["priority_rank"] = np.arange(1, len(base) + 1)

    selected_ids = set(base["registry_id"].astype(str)) if not base.empty else set()
    remaining = pool[~pool["registry_id"].astype(str).isin(selected_ids)].copy()

    if base.empty:
        current_score = 0.0
        current_summary = {
            "selected_total_pnl_pts": 0.0,
            "selected_trades": 0,
            "selected_max_drawdown_pts": 0.0,
            "risk_adjusted_score": 0.0,
        }
    else:
        current_summary, _ = simulate_ordered_candidate_list(
            name=f"{greedy_name}__base",
            family="greedy_base",
            cands=base,
            lookup=lookup,
            subordination_mode=subordination_mode,
            cap_policy_name=cap_policy_name,
            cap_policy_label=cap_policy_label,
        )
        current_score = objective(current_summary)

    steps = []
    evals = []
    step = 0

    selected_order = base.copy()

    while not remaining.empty:
        trial_rows = []
        best = None

        for _, cand in remaining.iterrows():
            trial = pd.concat([selected_order, cand.to_frame().T], ignore_index=True)
            trial["priority_rank"] = np.arange(1, len(trial) + 1)

            trial_summary, _ = simulate_ordered_candidate_list(
                name=f"{greedy_name}__trial_step_{step + 1}",
                family="greedy_trial",
                cands=trial,
                lookup=lookup,
                subordination_mode=subordination_mode,
                cap_policy_name=cap_policy_name,
                cap_policy_label=cap_policy_label,
            )
            trial_score = objective(trial_summary)
            marginal = trial_score - current_score

            erow = {
                "greedy_name": greedy_name,
                "subordination_mode": subordination_mode,
                "cap_policy_name": cap_policy_name,
                "cap_policy_label": cap_policy_label,
                "step": step + 1,
                "trial_registry_id": cand.get("registry_id"),
                "trial_matched_rule": cand.get("matched_rule"),
                "trial_cross_rule_status": cand.get("cross_rule_status"),
                "trial_cross_rule_lane": cand.get("cross_rule_lane"),
                "trial_cross_rule_role": cand.get("cross_rule_role"),
                "trial_version_short_name": cand.get("version_short_name"),
                "trial_representative_group": cand.get("representative_group"),
                "trial_source_scenario": cand.get("source_scenario"),
                "current_objective_before_add": current_score,
                "trial_objective": trial_score,
                "marginal_objective": marginal,
                "trial_total_pnl": trial_summary.get("selected_total_pnl_pts"),
                "trial_selected_trades": trial_summary.get("selected_trades"),
                "trial_avg_pnl": trial_summary.get("selected_avg_pnl_pts"),
                "trial_max_drawdown": trial_summary.get("selected_max_drawdown_pts"),
                "trial_p95_trades_day": trial_summary.get("selected_p95_trades_per_active_day"),
                "trial_top5_concentration": trial_summary.get("selected_top5_abs_day_concentration"),
                "trial_risk_adjusted_score": trial_summary.get("risk_adjusted_score"),
            }
            trial_rows.append(erow)

            if best is None or marginal > best["marginal_objective"]:
                best = dict(erow)
                best["_candidate"] = cand

        eval_df = pd.DataFrame(trial_rows)
        if not eval_df.empty:
            evals.append(eval_df)

        if best is None:
            break

        if best["marginal_objective"] <= GREEDY_MIN_MARGINAL_OBJECTIVE:
            stop = dict(best)
            stop.pop("_candidate", None)
            stop["accepted_into_greedy"] = 0
            stop["stop_reason"] = f"best marginal <= {GREEDY_MIN_MARGINAL_OBJECTIVE}"
            steps.append(stop)
            break

        chosen = best["_candidate"].to_frame().T
        selected_order = pd.concat([selected_order, chosen], ignore_index=True)
        selected_order["priority_rank"] = np.arange(1, len(selected_order) + 1)

        selected_ids.add(text(chosen.iloc[0].get("registry_id")))
        remaining = remaining[~remaining["registry_id"].astype(str).isin(selected_ids)].copy()

        current_summary, _ = simulate_ordered_candidate_list(
            name=f"{greedy_name}__step_{step + 1}",
            family="greedy_step",
            cands=selected_order,
            lookup=lookup,
            subordination_mode=subordination_mode,
            cap_policy_name=cap_policy_name,
            cap_policy_label=cap_policy_label,
        )
        current_score = objective(current_summary)

        srow = dict(best)
        srow.pop("_candidate", None)
        srow["accepted_into_greedy"] = 1
        srow["stop_reason"] = ""
        srow["portfolio_objective_after_add"] = current_score
        srow["portfolio_total_pnl_after_add"] = current_summary.get("selected_total_pnl_pts")
        srow["portfolio_selected_trades_after_add"] = current_summary.get("selected_trades")
        srow["portfolio_avg_pnl_after_add"] = current_summary.get("selected_avg_pnl_pts")
        srow["portfolio_max_drawdown_after_add"] = current_summary.get("selected_max_drawdown_pts")
        srow["portfolio_p95_trades_day_after_add"] = current_summary.get("selected_p95_trades_per_active_day")
        srow["portfolio_top5_concentration_after_add"] = current_summary.get("selected_top5_abs_day_concentration")
        srow["portfolio_risk_adjusted_score_after_add"] = current_summary.get("risk_adjusted_score")
        steps.append(srow)

        step += 1

    final_summary, final_contrib = simulate_ordered_candidate_list(
        name=f"{greedy_name}__final",
        family="greedy_final",
        cands=selected_order,
        lookup=lookup,
        subordination_mode=subordination_mode,
        cap_policy_name=cap_policy_name,
        cap_policy_label=cap_policy_label,
    )

    final_summary["greedy_name"] = greedy_name
    final_summary["greedy_base_rows"] = int(len(base))
    final_summary["greedy_final_rows"] = int(len(selected_order))
    final_summary["greedy_added_rows"] = int(len(selected_order) - len(base))

    final_order = selected_order.copy()
    final_order["greedy_name"] = greedy_name
    final_order["subordination_mode"] = subordination_mode
    final_order["cap_policy_name"] = cap_policy_name

    steps_df = pd.DataFrame(steps)
    evals_df = pd.concat(evals, ignore_index=True) if evals else pd.DataFrame()
    final_summary_df = pd.DataFrame([final_summary])
    final_contrib["greedy_name"] = greedy_name

    return steps_df, evals_df, final_summary_df, final_contrib, final_order


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("[CONFIG]")
    print("SV_DIR:", SV_DIR)
    print("XR_DIR:", XR_DIR)
    print("XP_BASELINE_DIR:", XP_BASELINE_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("CANONICAL_SCENARIO:", CANONICAL_SCENARIO)
    print("SUBORDINATION_MODES:", SUBORDINATION_MODES)
    print("CAP_POLICY_SPECS:", list(CAP_POLICY_SPECS))

    selected_raw = read_csv_required(INPUTS["selected_trades"], "sv05_selected_trades")
    xr02 = read_csv_required(INPUTS["xr02_strict_baseline"], "xr02_current_strict_baseline_input")
    xr03 = read_csv_required(INPUTS["xr03_core"], "xr03_core_candidate_input")
    xr04 = read_csv_required(INPUTS["xr04_expanded"], "xr04_expanded_research_input")
    xr01 = read_csv_required(INPUTS["xr01_all_rows"], "xr01_cross_rule_candidate_input_all_rows")

    baseline_template = read_csv_optional(INPUTS["baseline_template_summary"], "xp01_template_summary")
    baseline_greedy = read_csv_optional(INPUTS["baseline_greedy_summary"], "xp07_greedy_final_summary")

    selected = normalize_selected_trades(selected_raw)
    lookup = selected_lookup(selected)

    strict_df = prepare_candidate_df(xr02, "xr02_current_strict_baseline")
    core_df = prepare_candidate_df(xr03, "xr03_core_candidate_input")
    expanded_df = prepare_candidate_df(xr04, "xr04_expanded_research_input")

    # Core subsets.
    core_strict_strong = core_df[core_df["cross_rule_status"].eq("active_strict_reference_strong")].copy()
    core_non_strict = core_df[core_df["version_type"].ne("strict_baseline")].copy()

    # Strict baseline + all core non-strict allows us to test the old full strict baseline as a forced base.
    strict_plus_core_non_strict = pd.concat([strict_df, core_non_strict], ignore_index=True)

    # Fixed cap-aware templates.
    fixed_summary_parts = []
    fixed_contrib_parts = []
    fixed_priority_parts = []
    fixed_selected_detail_parts = []
    fixed_blocked_detail_parts = []

    fixed_template_records = []

    for policy_name in CAP_POLICY_SPECS:
        policy_map, policy_label = build_policy_map(core_df, policy_name)

        for order_type in ["strict_strong_quality", "quality_no_strict_bias", "avg_pnl_first", "total_pnl_first"]:
            for sub_mode in SUBORDINATION_MODES:
                base = apply_source_policy(core_df, policy_name, policy_map)
                ordered = order_candidates(base, order_type)

                portfolio_name = f"fixed_{policy_name}__{order_type}__sub-{sub_mode}"
                portfolio_name = safe_name(portfolio_name)

                summary, contrib, _, _ = simulate_portfolio(
                    portfolio_name=portfolio_name,
                    portfolio_family="fixed_cap_aware",
                    candidate_set_name="xr03_core_candidate_input",
                    cands=ordered,
                    lookup=lookup,
                    subordination_mode=sub_mode,
                    cap_policy_name=policy_name,
                    cap_policy_label=policy_label,
                    return_detail=False,
                )

                summary["order_type"] = order_type
                fixed_summary_parts.append(pd.DataFrame([summary]))
                fixed_contrib_parts.append(contrib)

                po = ordered.copy()
                po["portfolio_name"] = portfolio_name
                po["portfolio_family"] = "fixed_cap_aware"
                po["order_type"] = order_type
                po["subordination_mode"] = sub_mode
                po["cap_policy_name"] = policy_name
                fixed_priority_parts.append(po)

                fixed_template_records.append({
                    "portfolio_name": portfolio_name,
                    "policy_name": policy_name,
                    "policy_label": policy_label,
                    "order_type": order_type,
                    "subordination_mode": sub_mode,
                    "candidate_df": ordered,
                })

    fixed_summary = pd.concat(fixed_summary_parts, ignore_index=True) if fixed_summary_parts else pd.DataFrame()
    fixed_contrib = pd.concat(fixed_contrib_parts, ignore_index=True) if fixed_contrib_parts else pd.DataFrame()
    fixed_priority = pd.concat(fixed_priority_parts, ignore_index=True) if fixed_priority_parts else pd.DataFrame()

    # Greedy cap-aware marginal builds.
    greedy_steps_parts = []
    greedy_eval_parts = []
    greedy_summary_parts = []
    greedy_contrib_parts = []
    greedy_order_parts = []

    if RUN_GREEDY:
        print("[GREEDY] Running cap-aware greedy builds...")

        for policy_name in CAP_POLICY_SPECS:
            policy_map, policy_label = build_policy_map(core_df, policy_name)

            core_policy = apply_source_policy(core_df, policy_name, policy_map)
            core_policy_quality = order_candidates(core_policy, "strict_strong_quality")
            core_policy_nostrictbias = order_candidates(core_policy, "quality_no_strict_bias")

            strict_strong_policy = core_policy_quality[
                core_policy_quality["cross_rule_status"].eq("active_strict_reference_strong")
            ].copy()
            non_strict_policy = core_policy_quality[
                core_policy_quality["version_type"].ne("strict_baseline")
            ].copy()

            strict_baseline_policy = apply_source_policy(strict_df, policy_name, {})
            strict_baseline_policy = order_candidates(strict_baseline_policy, "current_live_priority")

            strict_plus_policy = pd.concat([strict_baseline_policy, non_strict_policy], ignore_index=True)
            strict_plus_policy = strict_plus_policy.drop_duplicates("registry_id", keep="first")
            strict_plus_base = strict_baseline_policy.copy()
            strict_plus_pool = non_strict_policy.copy()

            for sub_mode in SUBORDINATION_MODES:
                greedy_specs = [
                    (
                        "greedy_empty_core_quality",
                        pd.DataFrame(columns=core_policy_quality.columns),
                        core_policy_quality,
                    ),
                    (
                        "greedy_empty_core_no_strict_bias",
                        pd.DataFrame(columns=core_policy_nostrictbias.columns),
                        core_policy_nostrictbias,
                    ),
                    (
                        "greedy_from_strict_strong_core",
                        strict_strong_policy,
                        non_strict_policy,
                    ),
                    (
                        "greedy_from_current_strict_baseline_plus_core_non_strict",
                        strict_plus_base,
                        strict_plus_pool,
                    ),
                ]

                # Seed from positive contributors in the corresponding fixed best-policy quality template.
                fixed_name = safe_name(f"fixed_{policy_name}__strict_strong_quality__sub-{sub_mode}")
                if not fixed_contrib.empty:
                    fcontrib = fixed_contrib[fixed_contrib["portfolio_name"].eq(fixed_name)].copy()
                    pos_ids = set(
                        fcontrib.loc[
                            pd.to_numeric(fcontrib["portfolio_selected_pnl"], errors="coerce").fillna(0.0) > 0,
                            "registry_id",
                        ].astype(str)
                    )
                    if pos_ids:
                        fixed_positive_base = core_policy_quality[
                            core_policy_quality["registry_id"].astype(str).isin(pos_ids)
                        ].copy()
                        fixed_positive_pool = core_policy_quality[
                            ~core_policy_quality["registry_id"].astype(str).isin(pos_ids)
                        ].copy()
                        greedy_specs.append(
                            (
                                "greedy_from_fixed_positive_contributors",
                                fixed_positive_base,
                                fixed_positive_pool,
                            )
                        )

                for greedy_kind, base, pool in greedy_specs:
                    greedy_name = safe_name(f"{greedy_kind}__{policy_name}__sub-{sub_mode}")
                    print(f"[GREEDY] {greedy_name}")

                    steps, evals, gsummary, gcontrib, gorder = greedy_build(
                        greedy_name=greedy_name,
                        base_cands=base,
                        pool_cands=pool,
                        lookup=lookup,
                        subordination_mode=sub_mode,
                        cap_policy_name=policy_name,
                        cap_policy_label=policy_label,
                    )

                    if not steps.empty:
                        greedy_steps_parts.append(steps)
                    if WRITE_GREEDY_TRIAL_EVALS and not evals.empty:
                        greedy_eval_parts.append(evals)
                    if not gsummary.empty:
                        greedy_summary_parts.append(gsummary)
                    if not gcontrib.empty:
                        greedy_contrib_parts.append(gcontrib)
                    if not gorder.empty:
                        greedy_order_parts.append(gorder)

    greedy_steps = pd.concat(greedy_steps_parts, ignore_index=True) if greedy_steps_parts else pd.DataFrame()
    greedy_evals = pd.concat(greedy_eval_parts, ignore_index=True) if greedy_eval_parts else pd.DataFrame()
    greedy_summary = pd.concat(greedy_summary_parts, ignore_index=True) if greedy_summary_parts else pd.DataFrame()
    greedy_contrib = pd.concat(greedy_contrib_parts, ignore_index=True) if greedy_contrib_parts else pd.DataFrame()
    greedy_order = pd.concat(greedy_order_parts, ignore_index=True) if greedy_order_parts else pd.DataFrame()

    # Combined portfolio summary and contribution.
    all_summary = pd.concat(
        [x for x in [fixed_summary, greedy_summary] if x is not None and not x.empty],
        ignore_index=True,
    ) if (not fixed_summary.empty or not greedy_summary.empty) else pd.DataFrame()

    all_contrib = pd.concat(
        [x for x in [fixed_contrib, greedy_contrib] if x is not None and not x.empty],
        ignore_index=True,
    ) if (not fixed_contrib.empty or not greedy_contrib.empty) else pd.DataFrame()

    # Subordination mode summaries.
    subordination_mode_summary = (
        all_summary.groupby(["portfolio_family", "cap_policy_name", "subordination_mode"], dropna=False)
        .agg(
            portfolios=("portfolio_name", "count"),
            best_total_pnl=("selected_total_pnl_pts", "max"),
            avg_total_pnl=("selected_total_pnl_pts", "mean"),
            best_risk_adjusted_score=("risk_adjusted_score", "max"),
            avg_selected_trades=("selected_trades", "mean"),
            min_drawdown=("selected_max_drawdown_pts", "min"),
            avg_subordination_blocks=("blocked_subordination_count", "mean"),
            max_subordination_blocks=("blocked_subordination_count", "max"),
            avg_subordination_blocked_pnl=("blocked_subordination_pnl_pts", "mean"),
            max_subordination_block_rate=("subordination_block_rate", "max"),
        )
        .reset_index()
        if not all_summary.empty else pd.DataFrame()
    )

    # Block audit detail for best portfolios.
    selected_detail = pd.DataFrame()
    blocked_detail = pd.DataFrame()

    if WRITE_TOP_DETAIL and not all_summary.empty:
        top_names = []
        by_total = all_summary.sort_values("selected_total_pnl_pts", ascending=False).head(TOP_N_DETAIL_PORTFOLIOS)
        by_risk = all_summary.sort_values("risk_adjusted_score", ascending=False).head(TOP_N_DETAIL_PORTFOLIOS)

        for name in list(by_total["portfolio_name"]) + list(by_risk["portfolio_name"]):
            if name not in top_names:
                top_names.append(name)

        # Map fixed portfolio names to their candidate df.
        fixed_map = {r["portfolio_name"]: r for r in fixed_template_records}

        # Greedy order map.
        greedy_order_map = {}
        if not greedy_order.empty:
            for name, g in greedy_order.groupby("greedy_name", dropna=False):
                greedy_order_map[str(name) + "__final"] = g.copy()

        selected_parts = []
        blocked_parts = []

        for pname in top_names[:TOP_N_DETAIL_PORTFOLIOS]:
            row = all_summary[all_summary["portfolio_name"].eq(pname)]
            if row.empty:
                continue

            sub_mode = text(row.iloc[0].get("subordination_mode"))
            policy_name = text(row.iloc[0].get("cap_policy_name"))
            policy_label = text(row.iloc[0].get("cap_policy_label"))

            if pname in fixed_map:
                cands = fixed_map[pname]["candidate_df"].copy()
            else:
                # Greedy final portfolio_name is greedy_name__final.
                gorder = greedy_order_map.get(pname)
                if gorder is None or gorder.empty:
                    continue
                cands = gorder.copy()

            print(f"[DETAIL] {pname}")

            _, _, sel, blk = simulate_portfolio(
                portfolio_name=pname,
                portfolio_family=text(row.iloc[0].get("portfolio_family")),
                candidate_set_name=text(row.iloc[0].get("candidate_set_name")),
                cands=cands,
                lookup=lookup,
                subordination_mode=sub_mode,
                cap_policy_name=policy_name,
                cap_policy_label=policy_label,
                return_detail=True,
            )

            if not sel.empty:
                selected_parts.append(sel)
            if not blk.empty:
                blocked_parts.append(blk)

        selected_detail = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
        blocked_detail = pd.concat(blocked_parts, ignore_index=True) if blocked_parts else pd.DataFrame()

    # Subordination block audit summary.
    if not blocked_detail.empty:
        sub_block_audit = blocked_detail[blocked_detail["portfolio_status"].eq("blocked_subordination")].copy()
    else:
        sub_block_audit = pd.DataFrame()

    sub_block_summary = (
        sub_block_audit.groupby(
            [
                "portfolio_name",
                "subordination_mode",
                "dominant_rule",
                "subordinate_rule",
                "subordination_relation_reason",
            ],
            dropna=False,
        )
        .agg(
            blocked_count=("candidate_key", "count"),
            blocked_pnl=("pnl_pts", "sum"),
            avg_blocked_pnl=("pnl_pts", "mean"),
        )
        .reset_index()
        if not sub_block_audit.empty else pd.DataFrame()
    )

    # Baseline comparison rows.
    baseline_compare_rows = []

    if not baseline_template.empty:
        for template in [
            "strict_current_baseline",
            "core_avg_pnl_first",
            "core_cap_grid_hdn-c10_h024-c25_h015v-raw_h015vol-c25",
        ]:
            b = baseline_template[baseline_template["template_name"].astype(str).eq(template)]
            if not b.empty:
                row = b.iloc[0].to_dict()
                baseline_compare_rows.append({
                    "source": "previous_broad_portfolio_script",
                    "name": template,
                    "selected_trades": row.get("selected_trades"),
                    "selected_total_pnl_pts": row.get("selected_total_pnl_pts"),
                    "selected_avg_pnl_pts": row.get("selected_avg_pnl_pts"),
                    "selected_max_drawdown_pts": row.get("selected_max_drawdown_pts"),
                    "selected_p95_trades_per_active_day": row.get("selected_p95_trades_per_active_day"),
                    "selected_top5_abs_day_concentration": row.get("selected_top5_abs_day_concentration"),
                    "risk_adjusted_score": row.get("risk_adjusted_score"),
                })

    if not baseline_greedy.empty:
        bg = baseline_greedy.sort_values("selected_total_pnl_pts", ascending=False).head(1)
        if not bg.empty:
            row = bg.iloc[0].to_dict()
            baseline_compare_rows.append({
                "source": "previous_broad_portfolio_script",
                "name": row.get("greedy_name", row.get("template_name", "best_previous_greedy")),
                "selected_trades": row.get("selected_trades"),
                "selected_total_pnl_pts": row.get("selected_total_pnl_pts"),
                "selected_avg_pnl_pts": row.get("selected_avg_pnl_pts"),
                "selected_max_drawdown_pts": row.get("selected_max_drawdown_pts"),
                "selected_p95_trades_per_active_day": row.get("selected_p95_trades_per_active_day"),
                "selected_top5_abs_day_concentration": row.get("selected_top5_abs_day_concentration"),
                "risk_adjusted_score": row.get("risk_adjusted_score"),
            })

    if not all_summary.empty:
        best_total = all_summary.sort_values("selected_total_pnl_pts", ascending=False).head(1)
        best_risk = all_summary.sort_values("risk_adjusted_score", ascending=False).head(1)

        for label, src in [("best_total_this_script", best_total), ("best_risk_adjusted_this_script", best_risk)]:
            if not src.empty:
                row = src.iloc[0].to_dict()
                baseline_compare_rows.append({
                    "source": "this_targeted_script",
                    "name": label + "__" + text(row.get("portfolio_name")),
                    "selected_trades": row.get("selected_trades"),
                    "selected_total_pnl_pts": row.get("selected_total_pnl_pts"),
                    "selected_avg_pnl_pts": row.get("selected_avg_pnl_pts"),
                    "selected_max_drawdown_pts": row.get("selected_max_drawdown_pts"),
                    "selected_p95_trades_per_active_day": row.get("selected_p95_trades_per_active_day"),
                    "selected_top5_abs_day_concentration": row.get("selected_top5_abs_day_concentration"),
                    "risk_adjusted_score": row.get("risk_adjusted_score"),
                })

    baseline_compare = pd.DataFrame(baseline_compare_rows)

    # =============================================================================
    # WRITE OUTPUTS
    # =============================================================================

    paths = {
        "manifest": OUT_DIR / "xsg_manifest.json",
        "portfolio_summary": OUT_DIR / "xsg01_portfolio_summary.csv",
        "fixed_template_summary": OUT_DIR / "xsg02_fixed_cap_policy_subordination_summary.csv",
        "greedy_final_summary": OUT_DIR / "xsg03_cap_aware_greedy_final_summary.csv",
        "greedy_steps": OUT_DIR / "xsg04_cap_aware_greedy_steps.csv",
        "greedy_trial_evals": OUT_DIR / "xsg05_cap_aware_greedy_trial_evaluations.csv",
        "candidate_contribution": OUT_DIR / "xsg06_candidate_contribution_by_subordination_mode.csv",
        "subordination_mode_summary": OUT_DIR / "xsg07_subordination_mode_summary.csv",
        "subordination_block_audit": OUT_DIR / "xsg08_subordination_block_audit.csv",
        "subordination_block_summary": OUT_DIR / "xsg09_subordination_block_summary.csv",
        "best_selected_trades": OUT_DIR / "xsg10_best_portfolios_selected_trades.csv",
        "best_blocked_trades": OUT_DIR / "xsg11_best_portfolios_blocked_trades.csv",
        "fixed_priority_orders": OUT_DIR / "xsg12_fixed_priority_orders.csv",
        "greedy_final_orders": OUT_DIR / "xsg13_greedy_final_orders.csv",
        "baseline_compare": OUT_DIR / "xsg14_baseline_compare.csv",
    }

    all_summary.to_csv(paths["portfolio_summary"], index=False)
    fixed_summary.to_csv(paths["fixed_template_summary"], index=False)
    greedy_summary.to_csv(paths["greedy_final_summary"], index=False)
    greedy_steps.to_csv(paths["greedy_steps"], index=False)
    greedy_evals.to_csv(paths["greedy_trial_evals"], index=False)
    all_contrib.to_csv(paths["candidate_contribution"], index=False)
    subordination_mode_summary.to_csv(paths["subordination_mode_summary"], index=False)
    sub_block_audit.to_csv(paths["subordination_block_audit"], index=False)
    sub_block_summary.to_csv(paths["subordination_block_summary"], index=False)
    selected_detail.to_csv(paths["best_selected_trades"], index=False)
    blocked_detail.to_csv(paths["best_blocked_trades"], index=False)
    fixed_priority.to_csv(paths["fixed_priority_orders"], index=False)
    greedy_order.to_csv(paths["greedy_final_orders"], index=False)
    baseline_compare.to_csv(paths["baseline_compare"], index=False)

    manifest = {
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "outputs": {k: str(v) for k, v in paths.items()},
        "canonical_scenario": CANONICAL_SCENARIO,
        "cap_policy_specs": CAP_POLICY_SPECS,
        "subordination_modes": SUBORDINATION_MODES,
        "subordinated_rules": SUBORDINATED_RULES,
        "greedy_objective": GREEDY_OBJECTIVE,
        "notes": [
            "This is a targeted follow-up, not a replacement for the broad cross-rule portfolio script.",
            "It focuses on cap-aware policies discovered by the previous cap grid.",
            "It tests subordination as a pre-filter under multiple explicit definitions.",
            "The dominant_active_window modes use dominant candidate source intervals before global one-active simulation. This is intentionally a sensitivity test and may be more aggressive than live same-event subordination.",
            "Greedy builds are add-only marginal searches. They do not remove base rows once seeded.",
            "Manual/P4 overlay is not the focus here; this script focuses on xr03 core candidates and cap-aware subordination sensitivity.",
        ],
        "row_counts": {
            "portfolio_summary": int(len(all_summary)),
            "fixed_summary": int(len(fixed_summary)),
            "greedy_summary": int(len(greedy_summary)),
            "greedy_steps": int(len(greedy_steps)),
            "candidate_contribution": int(len(all_contrib)),
            "subordination_block_audit": int(len(sub_block_audit)),
            "selected_detail": int(len(selected_detail)),
            "blocked_detail": int(len(blocked_detail)),
        },
    }

    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # =============================================================================
    # TERMINAL READOUT
    # =============================================================================

    pd.set_option("display.width", 360)
    pd.set_option("display.max_columns", 180)
    pd.set_option("display.max_rows", 240)

    def show(title, df, cols=None, n=80, sort_cols=None, ascending=True):
        print("\n" + "=" * 180)
        print(title)
        print("=" * 180)
        if df is None or df.empty:
            print("(empty)")
            return

        out = df.copy()

        if sort_cols:
            sort_cols = [c for c in sort_cols if c in out.columns]
            if sort_cols:
                out = out.sort_values(sort_cols, ascending=ascending)

        if cols:
            cols = [c for c in cols if c in out.columns]
            print(out[cols].head(n).to_string(index=False))
        else:
            print(out.head(n).to_string(index=False))

    print("\n[WRITE OUTPUTS]")
    for name, path in paths.items():
        if path.exists():
            print(f"{name:36s}: {path}")

    show(
        "A) BASELINE COMPARE",
        baseline_compare,
        [
            "source",
            "name",
            "selected_trades",
            "selected_total_pnl_pts",
            "selected_avg_pnl_pts",
            "selected_max_drawdown_pts",
            "selected_p95_trades_per_active_day",
            "selected_top5_abs_day_concentration",
            "risk_adjusted_score",
        ],
        n=40,
    )

    show(
        "B) FIXED CAP-AWARE POLICY × SUBORDINATION SUMMARY — TOP BY TOTAL PNL",
        fixed_summary,
        [
            "portfolio_name",
            "cap_policy_name",
            "subordination_mode",
            "order_type",
            "candidate_rows",
            "source_events_seen",
            "selected_trades",
            "selected_total_pnl_pts",
            "selected_avg_pnl_pts",
            "selected_max_drawdown_pts",
            "selected_p95_trades_per_active_day",
            "selected_top5_abs_day_concentration",
            "blocked_subordination_count",
            "blocked_subordination_pnl_pts",
            "cap_shaped_selected_pnl_pts",
            "cap_shaped_pnl_share",
            "risk_adjusted_score",
        ],
        n=SHOW_TOP_N,
        sort_cols=["selected_total_pnl_pts"],
        ascending=False,
    )

    show(
        "C) FIXED CAP-AWARE POLICY × SUBORDINATION SUMMARY — TOP BY RISK-ADJUSTED SCORE",
        fixed_summary,
        [
            "portfolio_name",
            "cap_policy_name",
            "subordination_mode",
            "order_type",
            "selected_trades",
            "selected_total_pnl_pts",
            "selected_avg_pnl_pts",
            "selected_max_drawdown_pts",
            "selected_p95_trades_per_active_day",
            "selected_top5_abs_day_concentration",
            "blocked_subordination_count",
            "blocked_subordination_pnl_pts",
            "cap_shaped_selected_pnl_pts",
            "risk_adjusted_score",
        ],
        n=SHOW_TOP_N,
        sort_cols=["risk_adjusted_score"],
        ascending=False,
    )

    show(
        "D) SUBORDINATION MODE SUMMARY",
        subordination_mode_summary,
        [
            "portfolio_family",
            "cap_policy_name",
            "subordination_mode",
            "portfolios",
            "best_total_pnl",
            "avg_total_pnl",
            "best_risk_adjusted_score",
            "avg_selected_trades",
            "min_drawdown",
            "avg_subordination_blocks",
            "max_subordination_blocks",
            "avg_subordination_blocked_pnl",
            "max_subordination_block_rate",
        ],
        n=200,
        sort_cols=["cap_policy_name", "best_total_pnl"],
        ascending=[True, False],
    )

    show(
        "E) CAP-AWARE GREEDY FINAL SUMMARY — TOP BY TOTAL PNL",
        greedy_summary,
        [
            "greedy_name",
            "cap_policy_name",
            "subordination_mode",
            "greedy_base_rows",
            "greedy_final_rows",
            "greedy_added_rows",
            "candidate_rows",
            "source_events_seen",
            "selected_trades",
            "selected_total_pnl_pts",
            "selected_avg_pnl_pts",
            "selected_max_drawdown_pts",
            "selected_p95_trades_per_active_day",
            "selected_top5_abs_day_concentration",
            "blocked_subordination_count",
            "blocked_subordination_pnl_pts",
            "cap_shaped_selected_pnl_pts",
            "cap_shaped_pnl_share",
            "risk_adjusted_score",
        ],
        n=SHOW_TOP_N,
        sort_cols=["selected_total_pnl_pts"],
        ascending=False,
    )

    show(
        "F) CAP-AWARE GREEDY FINAL SUMMARY — TOP BY RISK-ADJUSTED SCORE",
        greedy_summary,
        [
            "greedy_name",
            "cap_policy_name",
            "subordination_mode",
            "greedy_base_rows",
            "greedy_final_rows",
            "greedy_added_rows",
            "selected_trades",
            "selected_total_pnl_pts",
            "selected_avg_pnl_pts",
            "selected_max_drawdown_pts",
            "selected_p95_trades_per_active_day",
            "selected_top5_abs_day_concentration",
            "blocked_subordination_count",
            "blocked_subordination_pnl_pts",
            "cap_shaped_selected_pnl_pts",
            "risk_adjusted_score",
        ],
        n=SHOW_TOP_N,
        sort_cols=["risk_adjusted_score"],
        ascending=False,
    )

    show(
        "G) GREEDY ACCEPTED STEPS",
        greedy_steps[greedy_steps.get("accepted_into_greedy", pd.Series(dtype=int)).eq(1)].copy()
        if not greedy_steps.empty else pd.DataFrame(),
        [
            "greedy_name",
            "cap_policy_name",
            "subordination_mode",
            "step",
            "trial_matched_rule",
            "trial_cross_rule_status",
            "trial_cross_rule_lane",
            "trial_cross_rule_role",
            "trial_version_short_name",
            "trial_source_scenario",
            "marginal_objective",
            "portfolio_total_pnl_after_add",
            "portfolio_selected_trades_after_add",
            "portfolio_avg_pnl_after_add",
            "portfolio_max_drawdown_after_add",
            "portfolio_p95_trades_day_after_add",
            "portfolio_risk_adjusted_score_after_add",
        ],
        n=SHOW_GREEDY_STEPS_N,
        sort_cols=["greedy_name", "step"],
        ascending=[True, True],
    )

    show(
        "H) SUBORDINATION BLOCK SUMMARY",
        sub_block_summary,
        [
            "portfolio_name",
            "subordination_mode",
            "dominant_rule",
            "subordinate_rule",
            "subordination_relation_reason",
            "blocked_count",
            "blocked_pnl",
            "avg_blocked_pnl",
        ],
        n=200,
        sort_cols=["portfolio_name", "blocked_count"],
        ascending=[True, False],
    )

    if not all_summary.empty and not all_contrib.empty:
        best_name = all_summary.sort_values("risk_adjusted_score", ascending=False).iloc[0]["portfolio_name"]
        show(
            "I) CANDIDATE CONTRIBUTIONS FOR BEST RISK-ADJUSTED PORTFOLIO",
            all_contrib[all_contrib["portfolio_name"].eq(best_name)].copy(),
            [
                "portfolio_name",
                "priority_rank",
                "matched_rule",
                "cross_rule_status",
                "cross_rule_lane",
                "cross_rule_role",
                "version_short_name",
                "source_scenario",
                "source_count",
                "source_pnl",
                "portfolio_selected_count",
                "portfolio_selected_pnl",
                "portfolio_selected_avg_pnl",
                "blocked_subordination_count",
                "blocked_subordination_pnl",
                "blocked_active_count",
                "blocked_active_pnl",
                "portfolio_residual_value_ratio",
                "largest_active_blocker_registry_id",
                "largest_active_blocker_count",
            ],
            n=120,
            sort_cols=["portfolio_selected_pnl"],
            ascending=False,
        )

    print("\nDone.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
'@ | .\.venv\Scripts\python.exe
