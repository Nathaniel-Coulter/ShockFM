@'
import json
import math
import re
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold")

SV_DIR = ROOT / "outputs" / "response_surface_v2" / "single_version_live_like"
XR_DIR = SV_DIR / "cross_rule_candidate_input"
XSG_DIR = SV_DIR / "cross_rule_subordination_greedy"

OUT_DIR = SV_DIR / "cross_rule_v2_ablation_selection_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "selected_trades": SV_DIR / "sv05_selected_trades.csv",

    "xr02_strict": XR_DIR / "xr02_current_strict_baseline_input.csv",
    "xr03_core": XR_DIR / "xr03_core_candidate_input.csv",
    "xr04_expanded": XR_DIR / "xr04_expanded_research_candidate_input.csv",

    "xsg01_portfolio_summary": XSG_DIR / "xsg01_portfolio_summary.csv",
    "xsg03_greedy_summary": XSG_DIR / "xsg03_cap_aware_greedy_final_summary.csv",
    "xsg04_greedy_steps": XSG_DIR / "xsg04_cap_aware_greedy_steps.csv",
    "xsg06_contribution": XSG_DIR / "xsg06_candidate_contribution_by_subordination_mode.csv",
    "xsg13_greedy_orders": XSG_DIR / "xsg13_greedy_final_orders.csv",
    "xsg14_baseline_compare": XSG_DIR / "xsg14_baseline_compare.csv",
}

CANONICAL_SCENARIO = "one_active_raw"

# Main candidate from the last targeted run.
TARGET_GREEDY_NAME = "greedy_empty_core_no_strict_bias__best_fixed_cap_policy__sub-dominant_active_window"
TARGET_GREEDY_PORTFOLIO_NAME = TARGET_GREEDY_NAME + "__final"

# Candidate maps / caps.
BEST_FIXED_CAP_POLICY = "best_fixed_cap_policy"
CONSERVATIVE_DD_POLICY = "more_conservative_dd_policy"

CAP_POLICY_SPECS = {
    "best_fixed_cap_policy": {
        "description": "Best deployable-looking cap policy from prior cap-grid and targeted greedy.",
        "groups": {
            "high_down_r022__direction_broad": "cap_10_per_day",
            "high_up_r024__loose5_no_vol": "cap_25_per_day",
            "high_up_r015__velocity_loose10": "one_active_raw",
            "high_up_r015__vol_loose5_no_vol_no_persist": "cap_25_per_day",
        },
    },
    "more_conservative_dd_policy": {
        "description": "Drawdown-first nearby policy.",
        "groups": {
            "high_down_r022__direction_broad": "cap_10_per_day",
            "high_up_r024__loose5_no_vol": "cap_25_cooldown_30s",
            "high_up_r015__velocity_loose10": "one_active_raw",
            "high_up_r015__vol_loose5_no_vol_no_persist": "cap_25_per_day",
        },
    },
}

# Only carry forward subordination modes that survived the prior interpretation.
SUBORDINATION_MODES_TO_AUDIT = ["none", "dominant_active_window"]

# Current structural subordination hypotheses.
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

FALLBACK_HOLD_SECONDS = 10.0

# Add-back pool controls.
INCLUDE_EXPANDED_MANUAL_ADDBACKS = True
INCLUDE_CAP_DIAGNOSTIC_ADDBACKS = False

# Terminal print limits.
SHOW_N = 120


# =============================================================================
# HELPERS
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
    if denom == 0:
        return np.nan
    return float(vals.head(5).sum() / denom)


def risk_adjusted_score(total_pnl, max_dd, p95_trades_day, top5_conc, selected_trades) -> float:
    """
    Triage score only, not a final trading utility.
    Designed to punish obvious pressure/drawdown/concentration without hiding total pnl.
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


def candidate_display(row) -> str:
    if text(row.get("version_type")) == "strict_baseline":
        return f"{text(row.get('matched_rule'))} / STRICT_LIVE"

    pieces = [
        text(row.get("matched_rule")),
        text(row.get("strict_reject_reason_primary")),
        text(row.get("admitting_book")),
        text(row.get("approval_ladder")),
    ]
    return " / ".join([p for p in pieces if p])


# =============================================================================
# DATA PREP
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


def build_selected_lookup(selected: pd.DataFrame) -> dict:
    out = {}
    for (rid, scenario), g in selected.groupby(["registry_id", "scenario"], dropna=False):
        out[(str(rid), str(scenario))] = g.copy()
    return out


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
        "version_type",
        "strict_reject_reason_primary",
        "admitting_book",
        "admission_family",
        "approval_ladder",
        "cross_rule_status",
        "cross_rule_lane",
        "cross_rule_role",
        "version_short_name",
        "representative_group",
        "candidate_input_notes",
    ]:
        if col not in c.columns:
            c[col] = ""

    c["candidate_display"] = c.apply(candidate_display, axis=1)

    return c


def build_policy_map(reference_core_df: pd.DataFrame, policy_name: str):
    """
    Build cap-policy registry_id -> source_scenario from the full xr03 core universe,
    not from the temporary candidate subset being simulated.

    This matters because add-back / pruned subsets often do not contain every
    cap-shaped representative, but the policy map still needs to be globally
    defined.
    """
    if policy_name not in CAP_POLICY_SPECS:
        return {}, ""

    spec = CAP_POLICY_SPECS[policy_name]

    if reference_core_df is None or reference_core_df.empty:
        raise SystemExit(f"[ABORT] Cannot build cap policy {policy_name}: reference core df is empty.")

    if "representative_group" not in reference_core_df.columns or "registry_id" not in reference_core_df.columns:
        raise SystemExit(
            f"[ABORT] Cannot build cap policy {policy_name}: reference core df lacks representative_group/registry_id."
        )

    rg_to_rid = dict(
        zip(
            reference_core_df["representative_group"].astype(str),
            reference_core_df["registry_id"].astype(str),
        )
    )

    policy_map = {}
    pieces = []
    missing = []

    for rg, scenario in spec["groups"].items():
        rid = rg_to_rid.get(rg)
        if rid is None:
            missing.append(rg)
            continue
        policy_map[rid] = scenario
        pieces.append(f"{rg}->{scenario}")

    if missing:
        print(f"[WARN] cap policy {policy_name}: representative_group missing from FULL CORE reference: {missing}")

    return policy_map, "|".join(pieces)


def apply_source_policy(cands: pd.DataFrame, policy_name: str, reference_core_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply cap source scenarios to a candidate subset.

    Important: cap-policy maps are always built from full xr03 core, then applied
    to whatever subset is being tested. This prevents false missing-group warnings
    in add-back/pruned audits.
    """
    out = cands.copy()
    out["source_policy_name"] = policy_name

    if "source_scenario" not in out.columns:
        out["source_scenario"] = CANONICAL_SCENARIO

    out["source_scenario"] = out["source_scenario"].fillna(CANONICAL_SCENARIO).replace("", CANONICAL_SCENARIO)

    if policy_name in CAP_POLICY_SPECS:
        ref = reference_core_df
        if ref is None or ref.empty:
            ref = core_df_global

        policy_map, _ = build_policy_map(ref, policy_name)

        mapped = out["registry_id"].astype(str).map(policy_map)
        out["source_scenario"] = mapped.where(mapped.notna(), out["source_scenario"])

    return out


def order_avg_pnl_first(cands: pd.DataFrame) -> pd.DataFrame:
    c = cands.copy()
    for col in ["selected_avg_pnl_pts", "selected_total_pnl_pts", "selected_trades"]:
        c[col] = pd.to_numeric(c.get(col, 0.0), errors="coerce").fillna(0.0)

    c = c.sort_values(
        ["selected_avg_pnl_pts", "selected_total_pnl_pts", "selected_trades", "registry_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    c["priority_rank"] = np.arange(1, len(c) + 1)
    return c


def order_current_live_priority(cands: pd.DataFrame) -> pd.DataFrame:
    c = cands.copy()
    c["current_live_priority"] = pd.to_numeric(c.get("current_live_priority", 9999), errors="coerce").fillna(9999)
    c["selected_total_pnl_pts"] = pd.to_numeric(c.get("selected_total_pnl_pts", 0.0), errors="coerce").fillna(0.0)

    c = c.sort_values(
        ["current_live_priority", "selected_total_pnl_pts", "registry_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    c["priority_rank"] = np.arange(1, len(c) + 1)
    return c


def enrich_order(order_df: pd.DataFrame, core_df: pd.DataFrame, fallback_policy_name: str) -> pd.DataFrame:
    """
    Use xsg13 final order for priority/source_scenario, but use xr03 for metadata safety.
    """
    o = order_df.copy()
    o["registry_id"] = o["registry_id"].astype(str)

    keep_cols = ["registry_id", "priority_rank", "source_scenario", "source_policy_name", "greedy_name"]
    keep_cols = [c for c in keep_cols if c in o.columns]

    o = o[keep_cols].copy()
    o["priority_rank"] = pd.to_numeric(o.get("priority_rank", np.arange(1, len(o) + 1)), errors="coerce")
    o = o.sort_values(["priority_rank", "registry_id"], ascending=[True, True]).drop_duplicates("registry_id", keep="first")

    meta = core_df.copy()
    meta["registry_id"] = meta["registry_id"].astype(str)

    out = o.merge(meta, on="registry_id", how="left", suffixes=("_order", ""))

    for col in ["source_scenario", "source_policy_name"]:
        order_col = col + "_order"
        if order_col in out.columns:
            out[col] = out[order_col].where(out[order_col].notna() & out[order_col].astype(str).ne(""), out.get(col, ""))

    if "source_scenario" not in out.columns:
        out["source_scenario"] = CANONICAL_SCENARIO
    if "source_policy_name" not in out.columns:
        out["source_policy_name"] = fallback_policy_name

    out["source_scenario"] = out["source_scenario"].fillna(CANONICAL_SCENARIO).replace("", CANONICAL_SCENARIO)
    out["source_policy_name"] = out["source_policy_name"].fillna(fallback_policy_name).replace("", fallback_policy_name)

    out["priority_rank"] = pd.to_numeric(out["priority_rank"], errors="coerce").fillna(9999).astype(int)
    out = out.sort_values(["priority_rank", "registry_id"]).reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)

    missing_meta = out["matched_rule"].isna().sum() if "matched_rule" in out.columns else len(out)
    if missing_meta:
        raise SystemExit(f"[ABORT] Could not enrich {missing_meta} greedy order rows from xr03 metadata.")

    return out


# =============================================================================
# CANDIDATE IDENTIFIERS / PRUNING
# =============================================================================

def is_strict(row) -> bool:
    return text(row.get("version_type")) == "strict_baseline" or "STRICT_LIVE" in text(row.get("version_short_name"))


def is_mid_up_r009_strict(row) -> bool:
    return text(row.get("matched_rule")) == "PARENT__mid_up_r009" and is_strict(row)


def is_mid_up_r006_strict(row) -> bool:
    return text(row.get("matched_rule")) == "PARENT__mid_up_r006" and is_strict(row)


def is_high_up_r014_no_vol_repair(row) -> bool:
    return (
        text(row.get("matched_rule")) == "PARENT__high_up_r014"
        and text(row.get("strict_reject_reason_primary")) == "vol_bin_mismatch"
        and text(row.get("admitting_book")) == "LOOSE_5_NO_VOL"
    )


def is_mid_up_r022_compact_repair(row) -> bool:
    return (
        text(row.get("matched_rule")) == "PARENT__mid_up_r022"
        and text(row.get("strict_reject_reason_primary")) == "vol_bin_mismatch"
        and text(row.get("admitting_book")) == "LOOSE_5"
        and "weak" in text(row.get("approval_ladder"))
    )


def is_high_up_r010_strict(row) -> bool:
    return text(row.get("matched_rule")) == "PARENT__high_up_r010" and is_strict(row)


def is_high_up_r011_strict(row) -> bool:
    return text(row.get("matched_rule")) == "PARENT__high_up_r011" and is_strict(row)


def is_mid_up_r016_repair(row) -> bool:
    return (
        text(row.get("matched_rule")) == "PARENT__mid_up_r016"
        and text(row.get("strict_reject_reason_primary")) == "vol_bin_mismatch"
        and text(row.get("admitting_book")) == "LOOSE_10"
    )


def prune_by_predicate(cands: pd.DataFrame, predicate, reason: str) -> pd.DataFrame:
    c = cands.copy()
    mask = c.apply(predicate, axis=1)
    out = c[~mask].copy().reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)
    return out


def remove_registry_ids(cands: pd.DataFrame, remove_ids: set) -> pd.DataFrame:
    out = cands[~cands["registry_id"].astype(str).isin(set(map(str, remove_ids)))].copy().reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)
    return out


def take_prefix(cands: pd.DataFrame, n: int) -> pd.DataFrame:
    out = cands.sort_values("priority_rank").head(n).copy().reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)
    return out


# =============================================================================
# EVENT BUILDING / SUBORDINATION / SIMULATION
# =============================================================================

def build_events(cands: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    parts = []

    meta_cols = [
        "registry_id",
        "matched_rule",
        "side",
        "version_type",
        "strict_reject_reason_primary",
        "admitting_book",
        "approval_ladder",
        "cross_rule_status",
        "cross_rule_lane",
        "cross_rule_role",
        "version_short_name",
        "candidate_display",
        "representative_group",
        "source_policy_name",
        "source_scenario",
        "priority_rank",
    ]

    for _, row in cands.iterrows():
        rid = text(row.get("registry_id"))
        scenario = text(row.get("source_scenario")) or CANONICAL_SCENARIO
        g = lookup.get((rid, scenario))

        if g is None or g.empty:
            continue

        h = g.copy()
        for col in meta_cols:
            h[col] = row.get(col, "")

        parts.append(h)

    if not parts:
        return pd.DataFrame()

    e = pd.concat(parts, ignore_index=True)

    for col in ["registry_id", "matched_rule", "side", "candidate_key", "day"]:
        e[col] = e[col].astype(str)

    for col in ["entry_ts_ns", "exit_ts_ns", "pnl_pts"]:
        e[col] = pd.to_numeric(e[col], errors="coerce")

    e["priority_rank"] = pd.to_numeric(e["priority_rank"], errors="coerce").fillna(999999).astype(int)

    e = e.dropna(subset=["entry_ts_ns"]).copy()

    fallback_ns = int(FALLBACK_HOLD_SECONDS * 1_000_000_000)
    bad_exit = e["exit_ts_ns"].isna() | (e["exit_ts_ns"] <= e["entry_ts_ns"])
    e.loc[bad_exit, "exit_ts_ns"] = e.loc[bad_exit, "entry_ts_ns"] + fallback_ns

    e["pnl_pts"] = pd.to_numeric(e["pnl_pts"], errors="coerce").fillna(0.0)
    e["_event_id"] = np.arange(len(e))

    return e


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

    return {"df": d, "entries": entries, "exits": exits, "cummax_exit": cummax_exit, "cummax_idx": cummax_idx}


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

    if mode != "dominant_active_window":
        raise ValueError(f"This audit only supports none and dominant_active_window. Got {mode}")

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

            for idx, srow in sub_day.iterrows():
                if e.at[idx, "_subordination_blocked"]:
                    continue

                entry = num(srow.get("entry_ts_ns"))
                blocker_idx = active_window_match(struct, entry)

                if blocker_idx is None:
                    continue

                brow = ddf.iloc[blocker_idx]
                e.at[idx, "_subordination_blocked"] = True

                out = srow.copy()
                out["portfolio_status"] = "blocked_subordination"
                out["block_reason"] = "subordination:dominant_active_window"
                out["subordination_mode"] = mode
                out["dominant_rule"] = dominant_rule
                out["subordinate_rule"] = text(srow.get("matched_rule"))
                out["blocked_by_registry_id"] = text(brow.get("registry_id"))
                out["blocked_by_matched_rule"] = text(brow.get("matched_rule"))
                out["blocked_by_candidate_key"] = text(brow.get("candidate_key"))
                out["blocked_by_entry_ts_ns"] = num(brow.get("entry_ts_ns"))
                out["blocked_by_exit_ts_ns"] = num(brow.get("exit_ts_ns"))
                blocked_parts.append(out.to_frame().T)

    blocked = pd.concat(blocked_parts, ignore_index=True) if blocked_parts else pd.DataFrame()
    filtered = e[~e["_subordination_blocked"]].drop(columns=["_subordination_blocked"]).copy()

    return filtered, blocked


def simulate_one_active(events: pd.DataFrame):
    """
    Same one-active-trade logic as before, but avoids per-row Series.copy()
    during the hot loop.

    Logic preserved:
      - same sort order
      - one active trade per day
      - active block if entry_ts_ns < active_until
      - fallback exit timestamp if missing/bad
      - accepted/blocked DataFrames still returned with the same key columns
    """
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    e = events.sort_values(
        ["day", "entry_ts_ns", "priority_rank", "registry_id", "candidate_key"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True).copy()

    n = len(e)
    fallback_ns = int(FALLBACK_HOLD_SECONDS * 1_000_000_000)

    entry_arr = pd.to_numeric(e["entry_ts_ns"], errors="coerce").to_numpy(dtype=np.float64)
    exit_arr = pd.to_numeric(e["exit_ts_ns"], errors="coerce").to_numpy(dtype=np.float64)

    bad_exit = np.isnan(exit_arr) | np.isnan(entry_arr) | (exit_arr <= entry_arr)
    valid_entry_bad_exit = bad_exit & ~np.isnan(entry_arr)
    exit_arr[valid_entry_bad_exit] = entry_arr[valid_entry_bad_exit] + fallback_ns

    e["exit_ts_ns"] = exit_arr

    day_arr = e["day"].astype(str).to_numpy()
    rid_arr = e["registry_id"].astype(str).to_numpy()
    rule_arr = e["matched_rule"].astype(str).to_numpy()
    ck_arr = e["candidate_key"].astype(str).to_numpy()

    selected_mask = np.zeros(n, dtype=bool)
    blocked_mask = np.zeros(n, dtype=bool)

    blocked_by_registry_id = np.full(n, "", dtype=object)
    blocked_by_matched_rule = np.full(n, "", dtype=object)
    blocked_by_candidate_key = np.full(n, "", dtype=object)
    active_until_out = np.full(n, np.nan, dtype=np.float64)

    i = 0
    while i < n:
        day = day_arr[i]
        j = i + 1
        while j < n and day_arr[j] == day:
            j += 1

        active_until = -1.0
        active_registry_id = ""
        active_rule = ""
        active_candidate_key = ""

        for k in range(i, j):
            entry = entry_arr[k]

            # Existing behavior effectively skips unusable entry rows.
            if np.isnan(entry):
                continue

            exit_ts = exit_arr[k]
            if np.isnan(exit_ts) or exit_ts <= entry:
                exit_ts = entry + fallback_ns
                exit_arr[k] = exit_ts

            if entry >= active_until:
                selected_mask[k] = True

                active_until = exit_ts
                active_registry_id = rid_arr[k]
                active_rule = rule_arr[k]
                active_candidate_key = ck_arr[k]
            else:
                blocked_mask[k] = True

                blocked_by_registry_id[k] = active_registry_id
                blocked_by_matched_rule[k] = active_rule
                blocked_by_candidate_key[k] = active_candidate_key
                active_until_out[k] = active_until

        i = j

    e["exit_ts_ns"] = exit_arr

    accepted = e.loc[selected_mask].copy()
    if not accepted.empty:
        accepted["portfolio_status"] = "selected"
        accepted["block_reason"] = ""
        accepted["blocked_by_registry_id"] = ""
        accepted["blocked_by_matched_rule"] = ""
        accepted["blocked_by_candidate_key"] = ""

    blocked = e.loc[blocked_mask].copy()
    if not blocked.empty:
        blocked_idx = blocked.index.to_numpy()

        blocked["portfolio_status"] = "blocked_active_trade"
        blocked["block_reason"] = "active_trade"
        blocked["blocked_by_registry_id"] = blocked_by_registry_id[blocked_idx]
        blocked["blocked_by_matched_rule"] = blocked_by_matched_rule[blocked_idx]
        blocked["blocked_by_candidate_key"] = blocked_by_candidate_key[blocked_idx]
        blocked["active_until_ns"] = active_until_out[blocked_idx]

    return accepted, blocked


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

    return {
        "selected_trades": n,
        "selected_total_pnl_pts": total,
        "selected_avg_pnl_pts": avg,
        "selected_win_rate": win,
        "selected_max_drawdown_pts": dd,
        "selected_active_days": int(len(day_pnl)),
        "selected_positive_day_rate": float((day_pnl > 0).mean()) if len(day_pnl) else np.nan,
        "selected_avg_trades_per_active_day": float(day_trades.mean()) if len(day_trades) else np.nan,
        "selected_p95_trades_per_active_day": float(day_trades.quantile(0.95)) if len(day_trades) else np.nan,
        "selected_max_trades_one_day": int(day_trades.max()) if len(day_trades) else 0,
        "selected_top5_abs_day_concentration": top5_abs_day_concentration(day_pnl),
    }


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


def candidate_contribution(map_name, cands, all_events, sub_blocked, accepted, active_blocked):
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

    contrib["map_name"] = map_name
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

    keep_cols = [
        "map_name",
        "priority_rank",
        "registry_id",
        "matched_rule",
        "side",
        "version_type",
        "strict_reject_reason_primary",
        "admitting_book",
        "approval_ladder",
        "cross_rule_status",
        "cross_rule_lane",
        "cross_rule_role",
        "candidate_display",
        "representative_group",
        "source_policy_name",
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
        "portfolio_residual_trade_ratio",
        "portfolio_residual_value_ratio",
    ]

    return contrib[[c for c in keep_cols if c in contrib.columns]].copy()


def simulate_map(map_name, map_family, cands, lookup, subordination_mode, map_notes="", return_detail=False):
    c = cands.copy().reset_index(drop=True)
    c["priority_rank"] = np.arange(1, len(c) + 1)

    all_events = build_events(c, lookup)
    post_sub, sub_blocked = apply_subordination(all_events, subordination_mode)
    accepted, active_blocked = simulate_one_active(post_sub)

    metrics = selected_metrics(accepted)

    sub_n = int(len(sub_blocked)) if sub_blocked is not None and not sub_blocked.empty else 0
    active_n = int(len(active_blocked)) if active_blocked is not None and not active_blocked.empty else 0
    source_n = int(len(all_events))

    sub_pnl = (
        float(pd.to_numeric(sub_blocked["pnl_pts"], errors="coerce").fillna(0.0).sum())
        if sub_blocked is not None and not sub_blocked.empty else 0.0
    )
    active_pnl = (
        float(pd.to_numeric(active_blocked["pnl_pts"], errors="coerce").fillna(0.0).sum())
        if active_blocked is not None and not active_blocked.empty else 0.0
    )

    lane_pnls = {}
    if accepted is not None and not accepted.empty:
        for lane, g in accepted.groupby("cross_rule_lane", dropna=False):
            lane_pnls[f"lane_pnl__{safe_name(lane)}"] = float(g["pnl_pts"].sum())

    ras = risk_adjusted_score(
        metrics["selected_total_pnl_pts"],
        metrics["selected_max_drawdown_pts"],
        metrics["selected_p95_trades_per_active_day"],
        metrics["selected_top5_abs_day_concentration"],
        metrics["selected_trades"],
    )

    summary = {
        "map_name": map_name,
        "map_family": map_family,
        "map_notes": map_notes,
        "candidate_rows": int(len(c)),
        "subordination_mode": subordination_mode,
        "source_events_seen": source_n,
        **metrics,
        "blocked_subordination_count": sub_n,
        "blocked_subordination_pnl_pts": sub_pnl,
        "blocked_active_count": active_n,
        "blocked_active_pnl_pts": active_pnl,
        "selected_fraction_of_events": pct(metrics["selected_trades"], source_n),
        "subordination_block_rate": pct(sub_n, source_n),
        "active_block_rate_after_subordination": pct(active_n, active_n + metrics["selected_trades"]),
        "risk_adjusted_score": ras,
        **lane_pnls,
    }

    contrib = candidate_contribution(map_name, c, all_events, sub_blocked, accepted, active_blocked)

    if return_detail:
        if accepted is not None and not accepted.empty:
            accepted = accepted.copy()
            accepted["map_name"] = map_name
            accepted["subordination_mode"] = subordination_mode
        blocked_parts = []
        if active_blocked is not None and not active_blocked.empty:
            a = active_blocked.copy()
            a["map_name"] = map_name
            a["subordination_mode"] = subordination_mode
            blocked_parts.append(a)
        if sub_blocked is not None and not sub_blocked.empty:
            sb = sub_blocked.copy()
            sb["map_name"] = map_name
            sb["subordination_mode"] = subordination_mode
            blocked_parts.append(sb)

        blocked = pd.concat(blocked_parts, ignore_index=True) if blocked_parts else pd.DataFrame()
        return summary, contrib, accepted, blocked

    return summary, contrib, pd.DataFrame(), pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("[CONFIG]")
    print("SV_DIR:", SV_DIR)
    print("XR_DIR:", XR_DIR)
    print("XSG_DIR:", XSG_DIR)
    print("OUT_DIR:", OUT_DIR)

    selected_raw = read_csv_required(INPUTS["selected_trades"], "sv05_selected_trades")
    xr02 = read_csv_required(INPUTS["xr02_strict"], "xr02_current_strict_baseline_input")
    xr03 = read_csv_required(INPUTS["xr03_core"], "xr03_core_candidate_input")
    xr04 = read_csv_required(INPUTS["xr04_expanded"], "xr04_expanded_research_input")
    # Only xsg03 and xsg13 are used downstream in this audit layer.
    # The other XSG files remain listed in INPUTS/manifest for provenance, but
    # loading them here only adds I/O cost.
    xsg03 = read_csv_required(INPUTS["xsg03_greedy_summary"], "xsg03_cap_aware_greedy_final_summary")
    xsg13 = read_csv_required(INPUTS["xsg13_greedy_orders"], "xsg13_greedy_final_orders")

    selected = normalize_selected_trades(selected_raw)
    lookup = build_selected_lookup(selected)

    strict_df = prepare_candidate_df(xr02, "xr02_current_strict_baseline")
    core_df = prepare_candidate_df(xr03, "xr03_core_candidate_input")
    expanded_df = prepare_candidate_df(xr04, "xr04_expanded_research_input")

    global core_df_global
    core_df_global = core_df.copy()

    # -------------------------------------------------------------------------
    # Locate and reconstruct the prior best greedy final order.
    # -------------------------------------------------------------------------
    order_mask = xsg13["greedy_name"].astype(str).eq(TARGET_GREEDY_NAME)
    if not order_mask.any():
        print(f"[WARN] Target greedy order not found exactly: {TARGET_GREEDY_NAME}")
        candidates = xsg03[
            xsg03["greedy_name"].astype(str).str.contains("greedy_empty_core_no_strict_bias", regex=False)
            & xsg03["cap_policy_name"].astype(str).eq(BEST_FIXED_CAP_POLICY)
        ].copy()
        if candidates.empty:
            candidates = xsg03.copy()
        candidates = candidates.sort_values(["selected_total_pnl_pts", "risk_adjusted_score"], ascending=[False, False])
        fallback_greedy_name = text(candidates.iloc[0]["greedy_name"])
        print("[WARN] Falling back to:", fallback_greedy_name)
        order_mask = xsg13["greedy_name"].astype(str).eq(fallback_greedy_name)

    greedy_order_raw = xsg13[order_mask].copy()
    if greedy_order_raw.empty:
        raise SystemExit("[ABORT] Could not locate any greedy final order rows.")

    full_greedy = enrich_order(greedy_order_raw, core_df, BEST_FIXED_CAP_POLICY)
    full_greedy["source_policy_name"] = BEST_FIXED_CAP_POLICY

    # Ensure source scenarios reflect best policy if missing.
    policy_map, policy_label = build_policy_map(core_df, BEST_FIXED_CAP_POLICY)
    full_greedy["source_scenario"] = full_greedy["registry_id"].map(policy_map).fillna(full_greedy["source_scenario"])
    full_greedy["source_scenario"] = full_greedy["source_scenario"].fillna(CANONICAL_SCENARIO).replace("", CANONICAL_SCENARIO)

    # Fixed maps.
    best_fixed_core = apply_source_policy(core_df, BEST_FIXED_CAP_POLICY, reference_core_df=core_df)
    best_fixed_avg = order_avg_pnl_first(best_fixed_core)

    conservative_core = apply_source_policy(core_df, CONSERVATIVE_DD_POLICY, reference_core_df=core_df)
    conservative_avg = order_avg_pnl_first(conservative_core)

    strict_baseline = apply_source_policy(strict_df, "strict_raw", reference_core_df=core_df)
    strict_baseline["source_scenario"] = CANONICAL_SCENARIO
    strict_baseline = order_current_live_priority(strict_baseline)

    # Pruned / prefix maps.
    pruned_no_path_artifacts = full_greedy.copy()
    pruned_no_path_artifacts = prune_by_predicate(pruned_no_path_artifacts, is_mid_up_r009_strict, "remove mid_up_r009 strict")
    pruned_no_path_artifacts = prune_by_predicate(pruned_no_path_artifacts, is_mid_up_r006_strict, "remove mid_up_r006 strict")

    pruned_clean = pruned_no_path_artifacts.copy()
    pruned_clean = prune_by_predicate(pruned_clean, is_high_up_r014_no_vol_repair, "remove high_up_r014 no-vol repair")
    pruned_clean = prune_by_predicate(pruned_clean, is_mid_up_r022_compact_repair, "remove mid_up_r022 compact repair")

    primary9 = take_prefix(full_greedy, 9)

    # Core-through-mid004: include all rows up through mid_up_r004 repair in greedy order,
    # but exclude mid_up_r009 strict because its final selected contribution was negative/path-artifact-like.
    through_mid004 = full_greedy.copy()
    mid004_rows = through_mid004[
        (through_mid004["matched_rule"].astype(str).eq("PARENT__mid_up_r004"))
        & (through_mid004["admitting_book"].astype(str).eq("NO_VOL_OR_PERSIST_BIN"))
    ]
    if not mid004_rows.empty:
        cutoff_rank = int(mid004_rows["priority_rank"].min())
    else:
        cutoff_rank = min(13, len(through_mid004))

    core_through_mid004 = through_mid004[through_mid004["priority_rank"] <= cutoff_rank].copy()
    core_through_mid004 = prune_by_predicate(core_through_mid004, is_mid_up_r009_strict, "remove mid_up_r009 strict from through-mid004 map")

    # -------------------------------------------------------------------------
    # Candidate map definitions.
    # -------------------------------------------------------------------------
    maps = []

    def add_map(name, family, cands, sub_mode, notes):
        c = cands.copy().reset_index(drop=True)
        c["priority_rank"] = np.arange(1, len(c) + 1)
        maps.append({
            "map_name": name,
            "map_family": family,
            "cands": c,
            "subordination_mode": sub_mode,
            "notes": notes,
        })

    add_map(
        "V2_STRICT_CURRENT_BASELINE",
        "baseline",
        strict_baseline,
        "dominant_active_window",
        "Current strict-live 15-row reference baseline.",
    )

    add_map(
        "V2_GREEDY_FULL_DAW",
        "greedy_full",
        full_greedy,
        "dominant_active_window",
        "Best prior targeted-script greedy full map: 16 rows, best_fixed_cap_policy, dominant_active_window.",
    )

    add_map(
        "V2_GREEDY_FULL_NONE",
        "greedy_full",
        full_greedy,
        "none",
        "Same 16-row greedy order but no subordination sensitivity.",
    )

    add_map(
        "V2_GREEDY_PRUNED_NO_PATH_ARTIFACTS_DAW",
        "greedy_pruned",
        pruned_no_path_artifacts,
        "dominant_active_window",
        "Remove mid_up_r009 strict and mid_up_r006 strict.",
    )

    add_map(
        "V2_GREEDY_PRUNED_CLEAN_DAW",
        "greedy_pruned",
        pruned_clean,
        "dominant_active_window",
        "Remove path-artifact rows plus high_up_r014 no-vol and mid_up_r022 compact.",
    )

    add_map(
        "V2_GREEDY_PRIMARY9_DAW",
        "greedy_prefix",
        primary9,
        "dominant_active_window",
        "Only first 9 high-marginal greedy rows.",
    )

    add_map(
        "V2_GREEDY_CORE_THROUGH_MID004_DAW",
        "greedy_prefix",
        core_through_mid004,
        "dominant_active_window",
        "Greedy rows through mid_up_r004 repair, excluding mid_up_r009 strict.",
    )

    add_map(
        "V2_FIXED_AVG_PNL_BEST_NONE",
        "fixed_template",
        best_fixed_avg,
        "none",
        "All 19 core rows, best_fixed_cap_policy, avg_pnl_first, no subordination.",
    )

    add_map(
        "V2_FIXED_AVG_PNL_BEST_DAW",
        "fixed_template",
        best_fixed_avg,
        "dominant_active_window",
        "All 19 core rows, best_fixed_cap_policy, avg_pnl_first, dominant_active_window.",
    )

    add_map(
        "V2_FIXED_CONSERVATIVE_DD_NONE",
        "fixed_template",
        conservative_avg,
        "none",
        "All 19 core rows, conservative DD cap policy, avg_pnl_first, no subordination.",
    )

    # -------------------------------------------------------------------------
    # Simulate candidate maps.
    # -------------------------------------------------------------------------
    summary_parts = []
    contribution_parts = []
    selected_detail_parts = []
    blocked_detail_parts = []

    for m in maps:
        print(f"[MAP] {m['map_name']}")
        summary, contrib, sel, blk = simulate_map(
            map_name=m["map_name"],
            map_family=m["map_family"],
            cands=m["cands"],
            lookup=lookup,
            subordination_mode=m["subordination_mode"],
            map_notes=m["notes"],
            return_detail=True,
        )
        summary_parts.append(pd.DataFrame([summary]))
        contribution_parts.append(contrib)
        if not sel.empty:
            selected_detail_parts.append(sel)
        if not blk.empty:
            blocked_detail_parts.append(blk)

    map_summary = pd.concat(summary_parts, ignore_index=True)
    map_contrib = pd.concat(contribution_parts, ignore_index=True)
    selected_detail = pd.concat(selected_detail_parts, ignore_index=True) if selected_detail_parts else pd.DataFrame()
    blocked_detail = pd.concat(blocked_detail_parts, ignore_index=True) if blocked_detail_parts else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Reproduction sanity vs prior xsg summary.
    # -------------------------------------------------------------------------
    sanity_rows = []

    xsg_target = xsg03[xsg03["greedy_name"].astype(str).eq(TARGET_GREEDY_NAME)].copy()
    this_full = map_summary[map_summary["map_name"].eq("V2_GREEDY_FULL_DAW")].copy()

    if not xsg_target.empty and not this_full.empty:
        old = xsg_target.iloc[0]
        new = this_full.iloc[0]
        sanity_rows.append({
            "check_name": "reproduce_targeted_greedy_full_daw",
            "old_selected_trades": old.get("selected_trades"),
            "new_selected_trades": new.get("selected_trades"),
            "delta_selected_trades": num(new.get("selected_trades"), 0.0) - num(old.get("selected_trades"), 0.0),
            "old_selected_total_pnl_pts": old.get("selected_total_pnl_pts"),
            "new_selected_total_pnl_pts": new.get("selected_total_pnl_pts"),
            "delta_selected_total_pnl_pts": num(new.get("selected_total_pnl_pts"), 0.0) - num(old.get("selected_total_pnl_pts"), 0.0),
            "old_max_dd": old.get("selected_max_drawdown_pts"),
            "new_max_dd": new.get("selected_max_drawdown_pts"),
            "delta_max_dd": num(new.get("selected_max_drawdown_pts"), 0.0) - num(old.get("selected_max_drawdown_pts"), 0.0),
        })

    sanity = pd.DataFrame(sanity_rows)

    # -------------------------------------------------------------------------
    # Leave-one-out ablation.
    # -------------------------------------------------------------------------
    loo_rows = []
    loo_contrib_parts = []

    base_map_dict = {
        "V2_GREEDY_FULL_DAW": full_greedy,
        "V2_GREEDY_PRUNED_NO_PATH_ARTIFACTS_DAW": pruned_no_path_artifacts,
        "V2_FIXED_AVG_PNL_BEST_NONE": best_fixed_avg,
    }

    for base_name, base_cands in base_map_dict.items():
        base_row = map_summary[map_summary["map_name"].eq(base_name)]
        if base_row.empty:
            continue
        base_metrics = base_row.iloc[0].to_dict()
        sub_mode = "dominant_active_window" if base_name.endswith("_DAW") else "none"

        for _, cand in base_cands.sort_values("priority_rank").iterrows():
            rid = text(cand.get("registry_id"))
            candidate_name = text(cand.get("candidate_display"))
            test_cands = remove_registry_ids(base_cands, {rid})

            map_name = safe_name(f"LOO__{base_name}__remove__{candidate_name}")
            summary, contrib, _, _ = simulate_map(
                map_name=map_name,
                map_family="leave_one_out",
                cands=test_cands,
                lookup=lookup,
                subordination_mode=sub_mode,
                map_notes=f"Leave-one-out from {base_name}; removed {candidate_name}",
                return_detail=False,
            )

            delta_pnl = summary["selected_total_pnl_pts"] - num(base_metrics.get("selected_total_pnl_pts"), 0.0)
            delta_score = summary["risk_adjusted_score"] - num(base_metrics.get("risk_adjusted_score"), 0.0)
            delta_trades = summary["selected_trades"] - num(base_metrics.get("selected_trades"), 0.0)
            delta_dd = summary["selected_max_drawdown_pts"] - num(base_metrics.get("selected_max_drawdown_pts"), 0.0)

            loo_rows.append({
                "base_map_name": base_name,
                "removed_registry_id": rid,
                "removed_candidate": candidate_name,
                "removed_matched_rule": cand.get("matched_rule"),
                "removed_cross_rule_status": cand.get("cross_rule_status"),
                "removed_cross_rule_lane": cand.get("cross_rule_lane"),
                "removed_source_scenario": cand.get("source_scenario"),
                "base_selected_trades": base_metrics.get("selected_trades"),
                "base_selected_total_pnl_pts": base_metrics.get("selected_total_pnl_pts"),
                "base_risk_adjusted_score": base_metrics.get("risk_adjusted_score"),
                "ablated_selected_trades": summary["selected_trades"],
                "ablated_selected_total_pnl_pts": summary["selected_total_pnl_pts"],
                "ablated_avg_pnl_pts": summary["selected_avg_pnl_pts"],
                "ablated_max_drawdown_pts": summary["selected_max_drawdown_pts"],
                "ablated_p95_trades_day": summary["selected_p95_trades_per_active_day"],
                "ablated_risk_adjusted_score": summary["risk_adjusted_score"],
                "delta_pnl_when_removed": delta_pnl,
                "delta_score_when_removed": delta_score,
                "delta_trades_when_removed": delta_trades,
                "delta_dd_when_removed": delta_dd,
                "keep_value_pnl": -delta_pnl,
                "keep_value_score": -delta_score,
            })

            contrib["parent_ablation_base_map"] = base_name
            contrib["removed_registry_id"] = rid
            loo_contrib_parts.append(contrib)

    leave_one_out = pd.DataFrame(loo_rows)
    loo_contrib = pd.concat(loo_contrib_parts, ignore_index=True) if loo_contrib_parts else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Add-back audit to pruned clean map.
    # -------------------------------------------------------------------------
    pruned_base_name = "V2_GREEDY_PRUNED_CLEAN_DAW"
    pruned_base_row = map_summary[map_summary["map_name"].eq(pruned_base_name)].iloc[0].to_dict()
    pruned_base_ids = set(pruned_clean["registry_id"].astype(str))

    addback_pool_parts = []

    # Core rows not in pruned clean.
    core_add_pool = core_df[~core_df["registry_id"].astype(str).isin(pruned_base_ids)].copy()
    core_add_pool = apply_source_policy(core_add_pool, BEST_FIXED_CAP_POLICY, reference_core_df=core_df)
    core_add_pool["addback_pool"] = "core_not_in_pruned_clean"
    addback_pool_parts.append(core_add_pool)

    if INCLUDE_EXPANDED_MANUAL_ADDBACKS:
        manual = expanded_df[
            expanded_df["cross_rule_lane"].astype(str).eq("manual_review")
            & ~expanded_df["registry_id"].astype(str).isin(pruned_base_ids)
        ].copy()
        manual["source_policy_name"] = "expanded_manual_raw"
        manual["source_scenario"] = CANONICAL_SCENARIO
        manual["addback_pool"] = "expanded_manual_review"
        addback_pool_parts.append(manual)

    if INCLUDE_CAP_DIAGNOSTIC_ADDBACKS:
        cap_diag = expanded_df[
            expanded_df["cross_rule_lane"].astype(str).eq("cap_diagnostic")
            & ~expanded_df["registry_id"].astype(str).isin(pruned_base_ids)
        ].copy()
        cap_diag["source_policy_name"] = "expanded_cap_diagnostic_raw"
        cap_diag["source_scenario"] = CANONICAL_SCENARIO
        cap_diag["addback_pool"] = "expanded_cap_diagnostic"
        addback_pool_parts.append(cap_diag)

    addback_pool = pd.concat(addback_pool_parts, ignore_index=True).drop_duplicates("registry_id", keep="first")
    addback_rows = []

    for _, cand in addback_pool.iterrows():
        rid = text(cand.get("registry_id"))
        candidate_name = text(cand.get("candidate_display"))

        cand_df = cand.to_frame().T.copy()
        cand_df = cand_df.dropna(axis=1, how="all")

        base_df = pruned_clean.copy()
        base_df = base_df.dropna(axis=1, how="all")

        trial = pd.concat([base_df, cand_df], ignore_index=True, sort=False)
        trial["priority_rank"] = np.arange(1, len(trial) + 1)

        map_name = safe_name(f"ADDBACK__{candidate_name}")
        summary, _, _, _ = simulate_map(
            map_name=map_name,
            map_family="addback_to_pruned_clean",
            cands=trial,
            lookup=lookup,
            subordination_mode="dominant_active_window",
            map_notes=f"Add {candidate_name} to V2_GREEDY_PRUNED_CLEAN_DAW at lowest priority.",
            return_detail=False,
        )

        addback_rows.append({
            "base_map_name": pruned_base_name,
            "added_registry_id": rid,
            "added_candidate": candidate_name,
            "added_matched_rule": cand.get("matched_rule"),
            "added_cross_rule_status": cand.get("cross_rule_status"),
            "added_cross_rule_lane": cand.get("cross_rule_lane"),
            "added_cross_rule_role": cand.get("cross_rule_role"),
            "added_source_scenario": cand.get("source_scenario"),
            "addback_pool": cand.get("addback_pool", ""),
            "base_selected_trades": pruned_base_row.get("selected_trades"),
            "base_selected_total_pnl_pts": pruned_base_row.get("selected_total_pnl_pts"),
            "base_risk_adjusted_score": pruned_base_row.get("risk_adjusted_score"),
            "trial_selected_trades": summary["selected_trades"],
            "trial_selected_total_pnl_pts": summary["selected_total_pnl_pts"],
            "trial_avg_pnl_pts": summary["selected_avg_pnl_pts"],
            "trial_max_drawdown_pts": summary["selected_max_drawdown_pts"],
            "trial_p95_trades_day": summary["selected_p95_trades_per_active_day"],
            "trial_risk_adjusted_score": summary["risk_adjusted_score"],
            "delta_pnl_when_added": summary["selected_total_pnl_pts"] - num(pruned_base_row.get("selected_total_pnl_pts"), 0.0),
            "delta_score_when_added": summary["risk_adjusted_score"] - num(pruned_base_row.get("risk_adjusted_score"), 0.0),
            "delta_trades_when_added": summary["selected_trades"] - num(pruned_base_row.get("selected_trades"), 0.0),
            "delta_dd_when_added": summary["selected_max_drawdown_pts"] - num(pruned_base_row.get("selected_max_drawdown_pts"), 0.0),
        })

    addback = pd.DataFrame(addback_rows)

    # -------------------------------------------------------------------------
    # Greedy prefix frontier.
    # -------------------------------------------------------------------------
    prefix_rows = []
    prefix_contrib_parts = []

    for n in range(1, len(full_greedy) + 1):
        cands = take_prefix(full_greedy, n)
        map_name = f"PREFIX_GREEDY_TOP_{n:02d}_DAW"

        summary, contrib, _, _ = simulate_map(
            map_name=map_name,
            map_family="greedy_prefix_frontier",
            cands=cands,
            lookup=lookup,
            subordination_mode="dominant_active_window",
            map_notes=f"First {n} rows from best greedy order.",
            return_detail=False,
        )

        prefix_rows.append(summary)
        prefix_contrib_parts.append(contrib)

    prefix_frontier = pd.DataFrame(prefix_rows)
    prefix_contrib = pd.concat(prefix_contrib_parts, ignore_index=True) if prefix_contrib_parts else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Rule-family ablation from full greedy.
    # -------------------------------------------------------------------------
    family_rows = []

    for rule in sorted(full_greedy["matched_rule"].astype(str).unique()):
        trial = full_greedy[~full_greedy["matched_rule"].astype(str).eq(rule)].copy()
        trial["priority_rank"] = np.arange(1, len(trial) + 1)

        map_name = safe_name(f"FAMILY_ABLATION__remove__{rule}")
        summary, _, _, _ = simulate_map(
            map_name=map_name,
            map_family="rule_family_ablation",
            cands=trial,
            lookup=lookup,
            subordination_mode="dominant_active_window",
            map_notes=f"Remove all candidates from {rule} from V2_GREEDY_FULL_DAW.",
            return_detail=False,
        )

        base_full = map_summary[map_summary["map_name"].eq("V2_GREEDY_FULL_DAW")].iloc[0].to_dict()
        family_rows.append({
            "base_map_name": "V2_GREEDY_FULL_DAW",
            "removed_matched_rule": rule,
            "removed_rows": int((full_greedy["matched_rule"].astype(str) == rule).sum()),
            "base_selected_total_pnl_pts": base_full.get("selected_total_pnl_pts"),
            "family_ablated_selected_total_pnl_pts": summary["selected_total_pnl_pts"],
            "delta_pnl_when_family_removed": summary["selected_total_pnl_pts"] - num(base_full.get("selected_total_pnl_pts"), 0.0),
            "keep_value_family_pnl": num(base_full.get("selected_total_pnl_pts"), 0.0) - summary["selected_total_pnl_pts"],
            "family_ablated_selected_trades": summary["selected_trades"],
            "delta_trades_when_family_removed": summary["selected_trades"] - num(base_full.get("selected_trades"), 0.0),
            "family_ablated_avg_pnl": summary["selected_avg_pnl_pts"],
            "family_ablated_max_dd": summary["selected_max_drawdown_pts"],
            "family_ablated_risk_adjusted_score": summary["risk_adjusted_score"],
            "delta_score_when_family_removed": summary["risk_adjusted_score"] - num(base_full.get("risk_adjusted_score"), 0.0),
        })

    family_ablation = pd.DataFrame(family_rows)

    # -------------------------------------------------------------------------
    # Final candidate recommendation table.
    # -------------------------------------------------------------------------
    full_contrib = map_contrib[map_contrib["map_name"].eq("V2_GREEDY_FULL_DAW")].copy()
    pruned_contrib = map_contrib[map_contrib["map_name"].eq("V2_GREEDY_PRUNED_CLEAN_DAW")].copy()

    rec_universe = pd.concat(
        [
            core_df.assign(recommendation_universe="core"),
            expanded_df[
                expanded_df["cross_rule_lane"].astype(str).isin(["manual_review"])
            ].assign(recommendation_universe="expanded_manual"),
        ],
        ignore_index=True,
    ).drop_duplicates("registry_id", keep="first")

    rec = rec_universe.copy()
    rec["registry_id"] = rec["registry_id"].astype(str)

    rec["in_full_greedy"] = rec["registry_id"].isin(set(full_greedy["registry_id"].astype(str))).astype(int)
    rec["in_pruned_clean"] = rec["registry_id"].isin(set(pruned_clean["registry_id"].astype(str))).astype(int)
    rec["in_primary9"] = rec["registry_id"].isin(set(primary9["registry_id"].astype(str))).astype(int)
    rec["in_core_through_mid004"] = rec["registry_id"].isin(set(core_through_mid004["registry_id"].astype(str))).astype(int)

    full_rank = dict(zip(full_greedy["registry_id"].astype(str), full_greedy["priority_rank"]))
    rec["full_greedy_priority_rank"] = rec["registry_id"].map(full_rank)

    full_metrics_cols = [
        "registry_id",
        "portfolio_selected_count",
        "portfolio_selected_pnl",
        "portfolio_selected_avg_pnl",
        "blocked_active_count",
        "blocked_active_pnl",
        "portfolio_residual_value_ratio",
    ]
    full_metrics = full_contrib[[c for c in full_metrics_cols if c in full_contrib.columns]].copy()
    full_metrics = full_metrics.rename(columns={
        "portfolio_selected_count": "full_selected_count",
        "portfolio_selected_pnl": "full_selected_pnl",
        "portfolio_selected_avg_pnl": "full_selected_avg_pnl",
        "blocked_active_count": "full_blocked_active_count",
        "blocked_active_pnl": "full_blocked_active_pnl",
        "portfolio_residual_value_ratio": "full_residual_value_ratio",
    })

    rec = rec.merge(full_metrics, on="registry_id", how="left")

    loo_full = leave_one_out[leave_one_out["base_map_name"].eq("V2_GREEDY_FULL_DAW")].copy()
    loo_full = loo_full[[
        "removed_registry_id",
        "keep_value_pnl",
        "keep_value_score",
        "delta_pnl_when_removed",
        "delta_score_when_removed",
    ]].rename(columns={
        "removed_registry_id": "registry_id",
        "keep_value_pnl": "loo_full_keep_value_pnl",
        "keep_value_score": "loo_full_keep_value_score",
        "delta_pnl_when_removed": "loo_full_delta_pnl_when_removed",
        "delta_score_when_removed": "loo_full_delta_score_when_removed",
    })

    rec = rec.merge(loo_full, on="registry_id", how="left")

    add_cols = [
        "added_registry_id",
        "delta_pnl_when_added",
        "delta_score_when_added",
        "delta_trades_when_added",
        "trial_selected_total_pnl_pts",
        "trial_risk_adjusted_score",
        "addback_pool",
    ]
    add_metrics = addback[[c for c in add_cols if c in addback.columns]].copy()
    add_metrics = add_metrics.rename(columns={
        "added_registry_id": "registry_id",
        "delta_pnl_when_added": "addback_to_pruned_delta_pnl",
        "delta_score_when_added": "addback_to_pruned_delta_score",
        "delta_trades_when_added": "addback_to_pruned_delta_trades",
    })
    rec = rec.merge(add_metrics, on="registry_id", how="left")

    def classify(row):
        rid = text(row.get("registry_id"))
        disp = text(row.get("candidate_display"))
        selected_pnl = num(row.get("full_selected_pnl"), 0.0)
        selected_avg = num(row.get("full_selected_avg_pnl"), np.nan)
        selected_count = num(row.get("full_selected_count"), 0.0)
        keep_value = num(row.get("loo_full_keep_value_pnl"), np.nan)
        add_pnl = num(row.get("addback_to_pruned_delta_pnl"), np.nan)

        if bool_int(row.get("in_primary9")) == 1:
            return "carry_to_v2_replay_primary"

        if bool_int(row.get("in_pruned_clean")) == 1:
            if selected_pnl >= 40 or (not np.isnan(keep_value) and keep_value >= 25):
                return "carry_to_v2_replay_primary"
            return "carry_to_v2_replay_secondary"

        if bool_int(row.get("in_full_greedy")) == 1:
            if selected_pnl < 0:
                return "holdout_path_artifact_do_not_promote_yet"
            if "mid_up_r006 / STRICT" in disp or "mid_up_r009 / STRICT" in disp:
                return "holdout_needs_replay_confirmation"
            if selected_pnl > 0 and selected_avg > 0:
                return "optional_addback_for_replay_ablation"
            return "sideline_before_replay"

        if not np.isnan(add_pnl):
            if add_pnl >= 25:
                return "fringe_addback_candidate_for_replay"
            if add_pnl > 0:
                return "watchlist_small_positive_addback"
            return "sideline_before_replay"

        return "sideline_before_replay"

    rec["recommendation_label"] = rec.apply(classify, axis=1)

    rec_keep_cols = [
        "recommendation_label",
        "recommendation_universe",
        "in_primary9",
        "in_core_through_mid004",
        "in_pruned_clean",
        "in_full_greedy",
        "full_greedy_priority_rank",
        "registry_id",
        "matched_rule",
        "side",
        "version_type",
        "strict_reject_reason_primary",
        "admitting_book",
        "approval_ladder",
        "cross_rule_status",
        "cross_rule_lane",
        "cross_rule_role",
        "candidate_display",
        "representative_group",
        "full_selected_count",
        "full_selected_pnl",
        "full_selected_avg_pnl",
        "full_residual_value_ratio",
        "loo_full_keep_value_pnl",
        "loo_full_keep_value_score",
        "addback_to_pruned_delta_pnl",
        "addback_to_pruned_delta_score",
        "addback_to_pruned_delta_trades",
        "addback_pool",
    ]
    recommendation = rec[[c for c in rec_keep_cols if c in rec.columns]].copy()

    recommendation["sort_bucket"] = recommendation["recommendation_label"].map({
        "carry_to_v2_replay_primary": 0,
        "carry_to_v2_replay_secondary": 1,
        "optional_addback_for_replay_ablation": 2,
        "fringe_addback_candidate_for_replay": 3,
        "watchlist_small_positive_addback": 4,
        "holdout_needs_replay_confirmation": 5,
        "holdout_path_artifact_do_not_promote_yet": 6,
        "sideline_before_replay": 9,
    }).fillna(99).astype(int)

    recommendation = recommendation.sort_values(
        ["sort_bucket", "full_greedy_priority_rank", "full_selected_pnl", "addback_to_pruned_delta_pnl"],
        ascending=[True, True, False, False],
    ).drop(columns=["sort_bucket"])

    # -------------------------------------------------------------------------
    # Write outputs.
    # -------------------------------------------------------------------------
    paths = {
        "manifest": OUT_DIR / "xva_manifest.json",
        "map_summary": OUT_DIR / "xva01_candidate_map_summary.csv",
        "map_contribution": OUT_DIR / "xva02_candidate_map_contribution.csv",
        "leave_one_out": OUT_DIR / "xva03_leave_one_out_ablation.csv",
        "addback": OUT_DIR / "xva04_addback_to_pruned_clean.csv",
        "prefix_frontier": OUT_DIR / "xva05_greedy_prefix_frontier.csv",
        "rule_family_ablation": OUT_DIR / "xva06_rule_family_ablation.csv",
        "recommendation": OUT_DIR / "xva07_final_candidate_recommendation.csv",
        "selected_detail": OUT_DIR / "xva08_selected_trades_for_named_maps.csv",
        "blocked_detail": OUT_DIR / "xva09_blocked_trades_for_named_maps.csv",
        "sanity": OUT_DIR / "xva10_reproduction_sanity.csv",
        "loo_contrib": OUT_DIR / "xva11_leave_one_out_contribution_detail.csv",
        "prefix_contrib": OUT_DIR / "xva12_prefix_contribution_detail.csv",
    }

    map_summary.to_csv(paths["map_summary"], index=False)
    map_contrib.to_csv(paths["map_contribution"], index=False)
    leave_one_out.to_csv(paths["leave_one_out"], index=False)
    addback.to_csv(paths["addback"], index=False)
    prefix_frontier.to_csv(paths["prefix_frontier"], index=False)
    family_ablation.to_csv(paths["rule_family_ablation"], index=False)
    recommendation.to_csv(paths["recommendation"], index=False)
    selected_detail.to_csv(paths["selected_detail"], index=False)
    blocked_detail.to_csv(paths["blocked_detail"], index=False)
    sanity.to_csv(paths["sanity"], index=False)
    loo_contrib.to_csv(paths["loo_contrib"], index=False)
    prefix_contrib.to_csv(paths["prefix_contrib"], index=False)

    manifest = {
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "outputs": {k: str(v) for k, v in paths.items()},
        "candidate_maps": [m["map_name"] for m in maps],
        "notes": [
            "This is a fast ablation/selection audit using existing selected-trade outputs.",
            "It does not rerun raw tick replay.",
            "It independently resimulates named candidate maps from sv05 selected trade streams.",
            "It tests full greedy, pruned greedy, primary-prefix maps, fixed avg-pnl map, and conservative fixed map.",
            "It performs leave-one-out, add-back to pruned clean, greedy prefix frontier, and rule-family ablation.",
            "The goal is to decide which rows should enter the first v2 replay-layer implementation and which can be sidelined.",
        ],
        "row_counts": {
            "map_summary": int(len(map_summary)),
            "map_contribution": int(len(map_contrib)),
            "leave_one_out": int(len(leave_one_out)),
            "addback": int(len(addback)),
            "prefix_frontier": int(len(prefix_frontier)),
            "rule_family_ablation": int(len(family_ablation)),
            "recommendation": int(len(recommendation)),
        },
    }

    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # -------------------------------------------------------------------------
    # Terminal readout.
    # -------------------------------------------------------------------------
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
            print(f"{name:32s}: {path}")

    show(
        "A) REPRODUCTION SANITY",
        sanity,
        [
            "check_name",
            "old_selected_trades",
            "new_selected_trades",
            "delta_selected_trades",
            "old_selected_total_pnl_pts",
            "new_selected_total_pnl_pts",
            "delta_selected_total_pnl_pts",
            "old_max_dd",
            "new_max_dd",
            "delta_max_dd",
        ],
        n=20,
    )

    show(
        "B) CANDIDATE MAP SUMMARY",
        map_summary,
        [
            "map_name",
            "map_family",
            "candidate_rows",
            "subordination_mode",
            "source_events_seen",
            "selected_trades",
            "selected_total_pnl_pts",
            "selected_avg_pnl_pts",
            "selected_max_drawdown_pts",
            "selected_p95_trades_per_active_day",
            "selected_top5_abs_day_concentration",
            "blocked_subordination_count",
            "blocked_subordination_pnl_pts",
            "blocked_active_count",
            "risk_adjusted_score",
            "map_notes",
        ],
        n=80,
        sort_cols=["selected_total_pnl_pts"],
        ascending=False,
    )

    show(
        "C) LEAVE-ONE-OUT ABLATION — FULL GREEDY",
        leave_one_out[leave_one_out["base_map_name"].eq("V2_GREEDY_FULL_DAW")].copy(),
        [
            "base_map_name",
            "removed_candidate",
            "removed_cross_rule_status",
            "removed_cross_rule_lane",
            "removed_source_scenario",
            "base_selected_total_pnl_pts",
            "ablated_selected_total_pnl_pts",
            "delta_pnl_when_removed",
            "keep_value_pnl",
            "delta_score_when_removed",
            "keep_value_score",
            "ablated_avg_pnl_pts",
            "ablated_max_drawdown_pts",
            "ablated_p95_trades_day",
        ],
        n=120,
        sort_cols=["keep_value_pnl"],
        ascending=False,
    )

    show(
        "D) PRUNED / PREFIX / FIXED MAP COMPARISON",
        map_summary[
            map_summary["map_name"].isin([
                "V2_GREEDY_FULL_DAW",
                "V2_GREEDY_PRUNED_NO_PATH_ARTIFACTS_DAW",
                "V2_GREEDY_PRUNED_CLEAN_DAW",
                "V2_GREEDY_PRIMARY9_DAW",
                "V2_GREEDY_CORE_THROUGH_MID004_DAW",
                "V2_FIXED_AVG_PNL_BEST_NONE",
                "V2_FIXED_CONSERVATIVE_DD_NONE",
            ])
        ].copy(),
        [
            "map_name",
            "candidate_rows",
            "selected_trades",
            "selected_total_pnl_pts",
            "selected_avg_pnl_pts",
            "selected_max_drawdown_pts",
            "selected_p95_trades_per_active_day",
            "selected_top5_abs_day_concentration",
            "risk_adjusted_score",
            "map_notes",
        ],
        n=40,
        sort_cols=["risk_adjusted_score"],
        ascending=False,
    )

    show(
        "E) ADD-BACK TO PRUNED CLEAN — BEST POSITIVE DELTAS",
        addback,
        [
            "added_candidate",
            "added_cross_rule_status",
            "added_cross_rule_lane",
            "added_cross_rule_role",
            "added_source_scenario",
            "addback_pool",
            "base_selected_total_pnl_pts",
            "trial_selected_total_pnl_pts",
            "delta_pnl_when_added",
            "delta_score_when_added",
            "delta_trades_when_added",
            "trial_avg_pnl_pts",
            "trial_max_drawdown_pts",
            "trial_p95_trades_day",
        ],
        n=120,
        sort_cols=["delta_pnl_when_added"],
        ascending=False,
    )

    show(
        "F) GREEDY PREFIX FRONTIER",
        prefix_frontier,
        [
            "map_name",
            "candidate_rows",
            "selected_trades",
            "selected_total_pnl_pts",
            "selected_avg_pnl_pts",
            "selected_max_drawdown_pts",
            "selected_p95_trades_per_active_day",
            "selected_top5_abs_day_concentration",
            "risk_adjusted_score",
        ],
        n=80,
        sort_cols=["candidate_rows"],
        ascending=True,
    )

    show(
        "G) RULE-FAMILY ABLATION FROM FULL GREEDY",
        family_ablation,
        [
            "removed_matched_rule",
            "removed_rows",
            "base_selected_total_pnl_pts",
            "family_ablated_selected_total_pnl_pts",
            "delta_pnl_when_family_removed",
            "keep_value_family_pnl",
            "delta_trades_when_family_removed",
            "family_ablated_avg_pnl",
            "family_ablated_max_dd",
            "family_ablated_risk_adjusted_score",
            "delta_score_when_family_removed",
        ],
        n=120,
        sort_cols=["keep_value_family_pnl"],
        ascending=False,
    )

    show(
        "H) FINAL CANDIDATE RECOMMENDATION",
        recommendation,
        [
            "recommendation_label",
            "recommendation_universe",
            "in_primary9",
            "in_core_through_mid004",
            "in_pruned_clean",
            "in_full_greedy",
            "full_greedy_priority_rank",
            "candidate_display",
            "cross_rule_status",
            "cross_rule_lane",
            "cross_rule_role",
            "representative_group",
            "full_selected_count",
            "full_selected_pnl",
            "full_selected_avg_pnl",
            "full_residual_value_ratio",
            "loo_full_keep_value_pnl",
            "addback_to_pruned_delta_pnl",
            "addback_to_pruned_delta_score",
            "addback_pool",
        ],
        n=160,
    )

    show(
        "I) FULL GREEDY CONTRIBUTION",
        map_contrib[map_contrib["map_name"].eq("V2_GREEDY_FULL_DAW")].copy(),
        [
            "map_name",
            "priority_rank",
            "candidate_display",
            "cross_rule_status",
            "cross_rule_lane",
            "source_scenario",
            "source_count",
            "source_pnl",
            "portfolio_selected_count",
            "portfolio_selected_pnl",
            "portfolio_selected_avg_pnl",
            "portfolio_residual_value_ratio",
            "blocked_active_count",
            "blocked_active_pnl",
        ],
        n=120,
        sort_cols=["portfolio_selected_pnl"],
        ascending=False,
    )

    print("\nDone.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    core_df_global = pd.DataFrame()
    main()
'@ | .\.venv\Scripts\python.exe
