from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype
from tqdm.auto import tqdm

# ============================================================
# REQUIREMENT: Optuna
# ============================================================
try:
    import optuna
except Exception as e:
    raise ImportError(
        "This script requires Optuna. Install with: pip install optuna\n"
        f"Original import error: {e}"
    )

# ============================================================
# SETTINGS (NO CLI ARGS) - EDIT HERE
# ============================================================

# Input files
RECS_PATH = Path("recommendations_output.parquet")
OFFERS_OPM_DATA_PATH = Path("daily_snapshot_offer_performance_metrics_data_20260120_174227.xlsx")

# Merge-asof behavior (same idea as your pipeline)
OFFERS_OPM_DAY_SHIFT_DAYS = 0
OFFERS_OPM_ASOF_DIRECTION = "backward"  # {'backward','forward','nearest'}

# Optimization target setup
TOPK = 3
PROFIT_PROB_COL = "q_api"  # q_prob = clip(col, 0, 1)
BILLABLE_CAP_THRESHOLD = 1.0

# Position weights in revenue proxy
POSITION_WEIGHT_MODE = "dcg"  # {'dcg','equal','custom'}
CUSTOM_POSITION_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.5}  # used only if POSITION_WEIGHT_MODE='custom'

# Dataset split (to avoid overfitting)
SEED = 42
TRAIN_FRACTION = 0.8
MAX_REQUESTS: Optional[int] = None  # None => all requests; set e.g. 5000 for faster iteration

# Optuna study
N_TRIALS = 200
TIMEOUT_SECONDS: Optional[int] = None  # e.g. 3600, or None
N_STARTUP_TRIALS = 30  # TPE warmup
USE_PRUNING = True

# How often report intermediate train avg revenue for pruning
REPORT_EVERY_N_REQUESTS = 50

# Search space bounds
BOUNDS = {
    "gamma": (0.01, 2.50),
    "mu": (-3.0, 3.0),
    "nu": (0.0, 6.0),
    "rho": (-3.0, 3.0),
    "delta": (0.0, 6.0),
    "lambda_": (0.0, 6.0),
    "p_h": (-1.0, 1.0),
}

# Optional: enqueue an initial known-good point (from your last run)
ENQUEUE_INIT_TRIAL = True
INIT_PARAMS = {
    "gamma": 0.19003712932599,
    "mu": 0.0198973389412471,
    "nu": 1.99400362001601,
    "rho": 0.418331944593489,
    "delta": 0.636301389842897,
    "lambda_": 1.9534446234665,
    "p_h": 0.416210088303199,
}

# ============================================================
# CONSTANTS (match business_reranking4 defaults)
# ============================================================

API_MODEL_SCORE_MISSING_SENTINEL = -1.0

CONTRACT_TYPE_CPL = "cpl"
CONTRACT_TYPE_FLAT = "flat"

# contract weights (beta, eta)
CONTRACT_WEIGHTS = {
    "flat": (0.2, 0.7),
    "cpl": (0.6, 0.3),
}

# diversity similarity weights
W_INV = 1.0
W_DEV = 0.5
W_CITY = 0.25

# fairness/VIP feature engineering
FAIRNESS_MODE = "budget_norm_log1p"  # {'none','budget_norm_log1p'}
FAIRNESS_BUDGET_BY_CONTRACT = True
FAIRNESS_CLIP = (-1.0, 1.0)

VIP_MODE = "p90_effective_price"  # {'none','binary_from_lead_limit','p90_effective_price'}
VIP_BY_CONTRACT = True

# ============================================================
# REQUIRED COLS
# ============================================================

RECS_REQUIRED_COLS = [
    "create_date",
    "uuid",
    "request_id",
    "offer_id",
    "property_id",
    "api_model_score",
    "final_score",
    "gap_score",
    "lead_price_score",
]

OFFERS_OPM_REQUIRED_COLS = [
    "id",
    "day",
    "offer_id",
    "group_id",
    "monthly_budget",
    "estimated_monthly_budget",
    "group_budget",
    "lead_limit",
    "leads",
    "leads_estimation",
    "estimated_perc_realization",
    "lead_price",
    "estimated_lead_price",
    "settlement_type",
]

# ============================================================
# DATA STRUCTURES (precomputed arrays per request for speed)
# ============================================================

@dataclass(frozen=True)
class RequestArrays:
    request_id: str

    property_id: np.ndarray          # int64
    baseline_sort_score: np.ndarray  # float64 (api_model_score as-provided)
    pg_sort_score: np.ndarray        # float64 (final_score as-provided)

    # Inputs to business score (pre-sanitized)
    q_eff: np.ndarray                # float64 in [0,1] after normalize_q + clamp
    base_fg: np.ndarray              # float64: 1 + beta*r + eta*g
    m_eff: np.ndarray                # float64 in [-1,1]
    v_eff: np.ndarray                # float64 in {0,1}
    cap_C: np.ndarray                # float64 in [0,1] cap penalty raw C(i)

    inv_id: np.ndarray               # int64 (or -1)
    dev_id: np.ndarray               # int64 (or -1)
    city_id: np.ndarray              # int64 (or -1)

    # Revenue proxy inputs
    price: np.ndarray                # float64 >=0
    q_prob: np.ndarray               # float64 in [0,1]
    is_billable: np.ndarray          # float64 in {0,1}

    @property
    def n(self) -> int:
        return int(self.q_eff.shape[0])


# ============================================================
# HELPERS: validation, IO, preprocessing
# ============================================================

def ensure_required_columns(df: pd.DataFrame, required: list[str], *, df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name}: missing required columns: {missing}")


def normalize_create_date(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize create_date to tz-naive Europe/Warsaw."""
    out = df.copy()
    dt = pd.to_datetime(out["create_date"], errors="raise")
    if isinstance(dt.dtype, DatetimeTZDtype):
        dt = dt.dt.tz_convert("Europe/Warsaw").dt.tz_localize(None)
    out["create_date"] = dt
    return out


def _col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def robust_minmax_log1p(series: pd.Series, *, by: pd.Series | None = None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    x = np.log1p(x.clip(lower=0.0))

    if by is None:
        lo = float(np.nanmin(x.values)) if np.isfinite(x.values).any() else 0.0
        hi = float(np.nanmax(x.values)) if np.isfinite(x.values).any() else 1.0
        denom = hi - lo if hi > lo else 1.0
        out = (x - lo) / denom
        return out.fillna(0.0).clip(0.0, 1.0)

    out = pd.Series(0.0, index=x.index, dtype=float)
    by_key = by.astype(str).fillna("")
    for _, idx in by_key.groupby(by_key).groups.items():
        xs = x.loc[idx]
        finite = np.isfinite(xs.values)
        if not finite.any():
            out.loc[idx] = 0.0
            continue
        lo = float(np.nanmin(xs.values))
        hi = float(np.nanmax(xs.values))
        denom = hi - lo if hi > lo else 1.0
        out.loc[idx] = ((xs - lo) / denom).fillna(0.0).clip(0.0, 1.0)
    return out


def compute_fairness_from_budget(df: pd.DataFrame) -> pd.Series:
    if FAIRNESS_MODE == "none":
        return pd.Series(0.0, index=df.index, dtype=float)

    budget = pd.to_numeric(_col_or_nan(df, "m_budget_raw"), errors="coerce").astype(float)

    by = None
    if FAIRNESS_BUDGET_BY_CONTRACT:
        by = _col_or_nan(df, "contract_type").astype(str).str.lower()

    norm01 = robust_minmax_log1p(budget, by=by)
    m = 2.0 * norm01 - 1.0

    lo, hi = FAIRNESS_CLIP
    return m.clip(lo, hi).fillna(0.0).astype(float)


def compute_effective_lead_price_for_features(df: pd.DataFrame) -> pd.Series:
    lead_price = pd.to_numeric(_col_or_nan(df, "lead_price"), errors="coerce")
    est_lead_price = pd.to_numeric(_col_or_nan(df, "estimated_lead_price"), errors="coerce")

    settlement = pd.to_numeric(_col_or_nan(df, "settlement_type"), errors="coerce")
    contract = _col_or_nan(df, "contract_type").astype(str).str.lower()

    if settlement.notna().any():
        is_cpl = settlement.eq(0)
    else:
        is_cpl = contract.eq(CONTRACT_TYPE_CPL)

    eff = lead_price.where(is_cpl, est_lead_price)

    lead_limit = pd.to_numeric(_col_or_nan(df, "lead_limit_raw"), errors="coerce").replace({0: np.nan})
    est_budget = pd.to_numeric(_col_or_nan(df, "estimated_monthly_budget"), errors="coerce")
    monthly_budget = pd.to_numeric(_col_or_nan(df, "monthly_budget"), errors="coerce")
    leads_est = pd.to_numeric(_col_or_nan(df, "leads_estimation"), errors="coerce").replace({0: np.nan})

    budget_any = est_budget.fillna(monthly_budget)
    fallback1 = budget_any / lead_limit
    fallback2 = budget_any / leads_est

    eff = eff.fillna(fallback1).fillna(fallback2)
    return pd.to_numeric(eff, errors="coerce").astype(float)


def compute_vip_flag(df: pd.DataFrame) -> pd.Series:
    if VIP_MODE == "none":
        return pd.Series(0, index=df.index, dtype=int)

    if VIP_MODE == "binary_from_lead_limit":
        lead_limit = pd.to_numeric(_col_or_nan(df, "lead_limit_raw"), errors="coerce").fillna(0.0)
        return (lead_limit > 0).astype(int)

    price = pd.to_numeric(_col_or_nan(df, "effective_lead_price"), errors="coerce")

    by = None
    if VIP_BY_CONTRACT:
        by = _col_or_nan(df, "contract_type").astype(str).str.lower()

    if by is None:
        thr = float(price.quantile(0.9)) if price.notna().any() else np.inf
        return (price >= thr).fillna(False).astype(int)

    out = pd.Series(0, index=df.index, dtype=int)
    by_key = by.astype(str).fillna("")
    for _, idx in by_key.groupby(by_key).groups.items():
        p = price.loc[idx]
        thr = float(p.quantile(0.9)) if p.notna().any() else np.inf
        out.loc[idx] = (p >= thr).fillna(False).astype(int)
    return out


def _prepare_q(series: pd.Series) -> pd.Series:
    q = pd.to_numeric(series, errors="raise").fillna(0.0)
    if API_MODEL_SCORE_MISSING_SENTINEL is not None:
        q = q.mask(q == API_MODEL_SCORE_MISSING_SENTINEL, 0.0)
    return q.clip(-1.0, 1.0)


def read_offer_performance_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"offer_performance_metrics data file not found: {path.resolve()}")

    df = pd.read_excel(path, sheet_name="data")
    ensure_required_columns(df, OFFERS_OPM_REQUIRED_COLS, df_name="offer_performance_metrics")

    out = df.copy()
    out["day"] = pd.to_datetime(out["day"], errors="raise").dt.normalize()
    out["offer_id"] = pd.to_numeric(out["offer_id"], errors="raise").astype("Int64")
    out["group_id"] = pd.to_numeric(out["group_id"], errors="coerce").astype("Int64")
    out["settlement_type"] = pd.to_numeric(out["settlement_type"], errors="coerce").astype("Int64")

    for c in [
        "monthly_budget",
        "estimated_monthly_budget",
        "group_budget",
        "estimated_perc_realization",
        "lead_price",
        "estimated_lead_price",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    for c in ["lead_limit", "leads", "leads_estimation"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    dup = out.duplicated(["offer_id", "day"], keep=False)
    if dup.any():
        sample = out.loc[dup, ["offer_id", "day"]].head(10)
        raise ValueError(
            "offer_performance_metrics: duplicate rows for (offer_id, day). "
            "This breaks merge_asof assumptions. Sample:\n"
            f"{sample}"
        )

    return out


def enrich_with_offer_performance_metrics(
    recs: pd.DataFrame,
    offers_opm: pd.DataFrame,
    *,
    day_shift_days: int,
    asof_direction: str,
) -> pd.DataFrame:
    if asof_direction not in {"backward", "forward", "nearest"}:
        raise ValueError(
            "asof_direction must be one of {'backward','forward','nearest'}, "
            f"got: {asof_direction!r}"
        )

    out = recs.copy()
    out["offer_id"] = pd.to_numeric(out["offer_id"], errors="raise").astype("Int64")

    out["rec_day"] = out["create_date"].dt.floor("D") + pd.to_timedelta(day_shift_days, unit="D")
    out["_orig_row_order"] = out.index.astype(int)

    left = out.sort_values(["rec_day", "offer_id"], kind="mergesort").reset_index(drop=True)
    right = offers_opm.sort_values(["day", "offer_id"], kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        by="offer_id",
        left_on="rec_day",
        right_on="day",
        direction=asof_direction,
    )

    merged = (
        merged.sort_values("_orig_row_order", kind="mergesort")
        .drop(columns=["_orig_row_order"])
        .reset_index(drop=True)
    )

    settlement_num = pd.to_numeric(merged.get("settlement_type"), errors="coerce")
    merged["contract_type"] = CONTRACT_TYPE_FLAT
    merged.loc[settlement_num == 0, "contract_type"] = CONTRACT_TYPE_CPL
    merged.loc[settlement_num == 1, "contract_type"] = CONTRACT_TYPE_FLAT

    # inv_id: prefer group_id; fallback offer_id
    merged["group_id"] = pd.to_numeric(merged.get("group_id"), errors="coerce").astype("Int64")
    merged["inv_id"] = merged["group_id"].fillna(merged["offer_id"]).astype("Int64")

    # m raw: budget per offer
    budget = pd.to_numeric(merged.get("monthly_budget"), errors="coerce")
    est_budget = pd.to_numeric(merged.get("estimated_monthly_budget"), errors="coerce")
    merged["m"] = budget.fillna(est_budget).fillna(0.0).astype(float)

    # v raw: lead_limit
    merged["v"] = (
        pd.to_numeric(merged.get("lead_limit"), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # cap_ratio (keep >1)
    cap = pd.to_numeric(merged.get("estimated_perc_realization"), errors="coerce")
    leads = pd.to_numeric(merged.get("leads"), errors="coerce")
    lead_limit = pd.to_numeric(merged.get("lead_limit"), errors="coerce").replace({0: np.nan})
    cap_fallback = leads / lead_limit
    cap = cap.fillna(cap_fallback)
    merged["cap_ratio"] = cap.fillna(0.0).clip(lower=0.0).astype(float)

    total_rows = len(merged)
    matched_rows = int(merged["id"].notna().sum()) if "id" in merged.columns else 0
    pct_matched = matched_rows / total_rows if total_rows else 0.0

    print(
        "offer_performance_metrics merge summary:\n"
        f"  rows_total..................: {total_rows}\n"
        f"  rows_matched_snapshot.......: {matched_rows} ({pct_matched:.1%})\n"
        f"  snapshot_day_min/max........: {offers_opm['day'].min()} / {offers_opm['day'].max()}"
    )

    return merged


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    for c in ["api_model_score", "final_score", "gap_score", "lead_price_score"]:
        out[c] = pd.to_numeric(out[c], errors="raise")

    out["q_api"] = _prepare_q(out["api_model_score"])
    out["q_final"] = _prepare_q(out["final_score"])

    out["r"] = out["lead_price_score"].fillna(0.0).clip(0.0, 1.0).astype(float)
    out["g"] = out["gap_score"].fillna(0.0).clip(0.0, 1.0).astype(float)

    if "contract_type" in out.columns:
        out["contract_type"] = out["contract_type"].fillna(CONTRACT_TYPE_FLAT).astype(str)
    else:
        out["contract_type"] = CONTRACT_TYPE_FLAT

    if "cap_ratio" in out.columns:
        out["cap_ratio"] = pd.to_numeric(out["cap_ratio"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    else:
        out["cap_ratio"] = 0.0

    if "inv_id" in out.columns:
        out["inv_id"] = pd.to_numeric(out["inv_id"], errors="coerce").astype("Int64")
    else:
        out["inv_id"] = pd.to_numeric(out["offer_id"], errors="raise").astype("Int64")

    # preserve raw business inputs
    if "m" in out.columns:
        out["m_budget_raw"] = pd.to_numeric(out["m"], errors="coerce").fillna(0.0).astype(float)
    else:
        out["m_budget_raw"] = 0.0

    if "v" in out.columns:
        out["lead_limit_raw"] = pd.to_numeric(out["v"], errors="coerce").fillna(0).astype(int)
    else:
        out["lead_limit_raw"] = 0

    out["effective_lead_price"] = compute_effective_lead_price_for_features(out)
    out["m"] = compute_fairness_from_budget(out)
    out["v"] = compute_vip_flag(out)

    return out


# ============================================================
# Business score math (vectorized; matches business_reranking4 behavior)
# ============================================================

def _normalize_q_piecewise(q: np.ndarray) -> np.ndarray:
    """
    Match business_reranking4._normalize_q + clamp:
      if q < 0: (q+1)/2
      elif q > 1: 1
      else: q
    then clamp to [0,1]
    """
    q = q.astype(np.float64, copy=False)
    q_norm = np.where(q < 0.0, (q + 1.0) / 2.0, q)
    q_norm = np.where(q_norm > 1.0, 1.0, q_norm)
    return np.clip(q_norm, 0.0, 1.0)


def cap_penalty_vec(cap_ratio: np.ndarray) -> np.ndarray:
    """
    Match business_reranking4.cap_penalty:
      0 if <0.7
      linear to 1 between 0.7 and 1.0
      1 if >=1.0
    """
    cr = np.maximum(cap_ratio.astype(np.float64, copy=False), 0.0)
    out = np.zeros_like(cr, dtype=np.float64)
    mid = (cr >= 0.7) & (cr < 1.0)
    out[mid] = (cr[mid] - 0.7) / 0.3
    out[cr >= 1.0] = 1.0
    return out


def diversity_penalty_vec(
    inv_id: np.ndarray,
    dev_id: np.ndarray,
    city_id: np.ndarray,
    selected_indices: List[int],
    *,
    w_inv: float,
    w_dev: float,
    w_city: float,
) -> np.ndarray:
    """
    Vector form of diversity_penalty: returns D(i) for all candidates i
    as max similarity to any selected item.
    """
    n = int(inv_id.shape[0])
    if not selected_indices:
        return np.zeros(n, dtype=np.float64)

    D = np.zeros(n, dtype=np.float64)

    for s in selected_indices:
        inv_s = int(inv_id[s])
        dev_s = int(dev_id[s])
        city_s = int(city_id[s])

        if inv_s != -1:
            D = np.maximum(D, w_inv * (inv_id == inv_s))
        if dev_s != -1:
            D = np.maximum(D, w_dev * (dev_id == dev_s))
        if city_s != -1:
            D = np.maximum(D, w_city * (city_id == city_s))

    return D


def greedy_select_topk_indices(
    req: RequestArrays,
    *,
    k: int,
    gamma: float,
    mu: float,
    nu: float,
    pacing_term: float,
    delta: float,
    lambda_: float,
    w_inv: float,
    w_dev: float,
    w_city: float,
) -> List[int]:
    """
    Greedy reranking exactly like business_reranking4.greedy_rerank:
      - in each iteration pick argmax business_score()
      - ties resolved by first occurrence (np.argmax does that)
    """
    n = req.n
    if n <= 0 or k <= 0:
        return []

    base_component = np.power(req.q_eff, float(gamma)).astype(np.float64, copy=False)
    business_factors = req.base_fg + float(mu) * req.m_eff + float(nu) * req.v_eff
    score_base = base_component * business_factors * float(pacing_term) - float(delta) * req.cap_C

    selected: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)

    steps = min(k, n)
    for _ in range(steps):
        if lambda_ != 0.0 and selected:
            D = diversity_penalty_vec(
                req.inv_id, req.dev_id, req.city_id, selected,
                w_inv=w_inv, w_dev=w_dev, w_city=w_city
            )
            score = score_base - float(lambda_) * D
        else:
            score = score_base

        score = score.copy()
        score[selected_mask] = -np.inf

        best_idx = int(np.argmax(score))
        if not np.isfinite(score[best_idx]) or score[best_idx] == -np.inf:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True

    return selected


# ============================================================
# Revenue/profit proxy
# ============================================================

def get_position_weights(k: int) -> np.ndarray:
    if POSITION_WEIGHT_MODE == "equal":
        return np.array([1.0] * k, dtype=np.float64)
    if POSITION_WEIGHT_MODE == "custom":
        return np.array([float(CUSTOM_POSITION_WEIGHTS.get(i, 0.0)) for i in range(1, k + 1)], dtype=np.float64)
    # default: DCG-like
    return np.array([float(1.0 / np.log2(i + 1)) for i in range(1, k + 1)], dtype=np.float64)


def expected_revenue_for_selected(req: RequestArrays, selected: List[int], pos_w: np.ndarray) -> float:
    rev = 0.0
    max_k = min(len(selected), int(pos_w.shape[0]))
    for r in range(max_k):
        i = selected[r]
        rev += float(req.price[i]) * float(req.q_prob[i]) * float(pos_w[r]) * float(req.is_billable[i])
    return float(rev)


def eval_kurekj_total_revenue(
    requests: List[RequestArrays],
    *,
    k: int,
    pos_w: np.ndarray,
    gamma: float,
    mu: float,
    nu: float,
    rho: float,
    p_h: float,
    delta: float,
    lambda_: float,
) -> float:
    p_h_clip = float(np.clip(float(p_h), -1.0, 1.0))
    pacing_term = 1.0 + float(rho) * p_h_clip

    total = 0.0
    for req in requests:
        sel = greedy_select_topk_indices(
            req,
            k=k,
            gamma=gamma,
            mu=mu,
            nu=nu,
            pacing_term=pacing_term,
            delta=delta,
            lambda_=lambda_,
            w_inv=W_INV,
            w_dev=W_DEV,
            w_city=W_CITY,
        )
        total += expected_revenue_for_selected(req, sel, pos_w)
    return float(total)


def eval_baseline_total_revenue(
    requests: List[RequestArrays],
    *,
    k: int,
    pos_w: np.ndarray,
) -> float:
    total = 0.0
    for req in requests:
        s = req.baseline_sort_score
        pid = req.property_id
        s2 = np.where(np.isfinite(s), s, -np.inf)
        idx = np.lexsort((pid, -s2))  # primary: -score, secondary: property_id
        sel = idx[:k].tolist()
        total += expected_revenue_for_selected(req, sel, pos_w)
    return float(total)


def eval_pg_total_revenue(
    requests: List[RequestArrays],
    *,
    k: int,
    pos_w: np.ndarray,
) -> float:
    total = 0.0
    for req in requests:
        s = req.pg_sort_score
        pid = req.property_id
        s2 = np.where(np.isfinite(s), s, -np.inf)
        idx = np.lexsort((pid, -s2))
        sel = idx[:k].tolist()
        total += expected_revenue_for_selected(req, sel, pos_w)
    return float(total)


# ============================================================
# Build RequestArrays dataset
# ============================================================

def _to_int_sentinel(series: pd.Series, sentinel: int = -1) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce")
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.where(np.isfinite(arr), arr.astype(np.int64), sentinel)
    return out


def build_request_arrays(recs: pd.DataFrame) -> List[RequestArrays]:
    if PROFIT_PROB_COL not in recs.columns:
        raise ValueError(f"profit prob col {PROFIT_PROB_COL!r} not found in recs df columns.")

    # optionally subsample requests for speed
    if MAX_REQUESTS is not None and MAX_REQUESTS > 0:
        unique_rids = recs["request_id"].astype(str).drop_duplicates().tolist()
        if len(unique_rids) > MAX_REQUESTS:
            rng = np.random.default_rng(SEED)
            chosen = rng.choice(np.array(unique_rids, dtype=object), size=MAX_REQUESTS, replace=False)
            chosen_set = set(map(str, chosen.tolist()))
            recs = recs[recs["request_id"].astype(str).isin(chosen_set)].copy()
            print(f"INFO: request subsample enabled -> using {len(chosen_set)} requests.")

    requests: List[RequestArrays] = []
    grouped = recs.groupby("request_id", sort=False)

    for rid, g in tqdm(grouped, total=grouped.ngroups, desc="Building request arrays", unit="request"):
        property_id = pd.to_numeric(g["property_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()

        baseline_sort_score = pd.to_numeric(g["api_model_score"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        pg_sort_score = pd.to_numeric(g["final_score"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        q_api = pd.to_numeric(g["q_api"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        q_eff = _normalize_q_piecewise(q_api)

        contract = g["contract_type"].astype(str).str.lower()
        is_cpl = (contract.values == CONTRACT_TYPE_CPL)

        beta = np.where(is_cpl, CONTRACT_WEIGHTS["cpl"][0], CONTRACT_WEIGHTS["flat"][0]).astype(np.float64)
        eta = np.where(is_cpl, CONTRACT_WEIGHTS["cpl"][1], CONTRACT_WEIGHTS["flat"][1]).astype(np.float64)

        r = pd.to_numeric(g["r"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float64, copy=False)
        gg = pd.to_numeric(g["g"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float64, copy=False)
        base_fg = 1.0 + beta * r + eta * gg

        m_eff = pd.to_numeric(g["m"], errors="coerce").fillna(0.0).clip(-1.0, 1.0).to_numpy(dtype=np.float64, copy=False)
        v_raw = pd.to_numeric(g["v"], errors="coerce").fillna(0).to_numpy(dtype=np.float64, copy=False)
        v_eff = (v_raw != 0).astype(np.float64)

        cap_ratio = pd.to_numeric(g["cap_ratio"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64, copy=False)
        cap_C = cap_penalty_vec(cap_ratio)

        inv_id = _to_int_sentinel(g["inv_id"], sentinel=-1) if "inv_id" in g.columns else _to_int_sentinel(g["offer_id"], sentinel=-1)
        dev_id = _to_int_sentinel(g["dev_id"], sentinel=-1) if "dev_id" in g.columns else np.full_like(inv_id, -1)
        city_id = _to_int_sentinel(g["city_id"], sentinel=-1) if "city_id" in g.columns else np.full_like(inv_id, -1)

        price = pd.to_numeric(g["effective_lead_price"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64, copy=False)
        q_prob = pd.to_numeric(g[PROFIT_PROB_COL], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float64, copy=False)
        is_billable = (cap_ratio < float(BILLABLE_CAP_THRESHOLD)).astype(np.float64)

        requests.append(
            RequestArrays(
                request_id=str(rid),
                property_id=property_id,
                baseline_sort_score=baseline_sort_score,
                pg_sort_score=pg_sort_score,
                q_eff=q_eff,
                base_fg=base_fg,
                m_eff=m_eff,
                v_eff=v_eff,
                cap_C=cap_C,
                inv_id=inv_id,
                dev_id=dev_id,
                city_id=city_id,
                price=price,
                q_prob=q_prob,
                is_billable=is_billable,
            )
        )

    return requests


def split_train_valid(requests: List[RequestArrays]) -> Tuple[List[RequestArrays], List[RequestArrays]]:
    if not (0.0 < TRAIN_FRACTION < 1.0):
        raise ValueError("TRAIN_FRACTION must be in (0,1)")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(requests))
    n_train = int(round(len(requests) * TRAIN_FRACTION))
    train = [requests[i] for i in idx[:n_train]]
    valid = [requests[i] for i in idx[n_train:]]
    return train, valid


# ============================================================
# OUTPUT HELPERS
# ============================================================

def make_output_dir() -> Path:
    script_path = Path(__file__).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = script_path.parent / f"Output_[{script_path.name}]_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def trials_to_dataframe(study: "optuna.Study") -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for tr in study.trials:
        row: Dict[str, object] = {
            "trial": int(tr.number),
            "state": str(tr.state.name),
            "value": float(tr.value) if tr.value is not None else np.nan,
        }
        # params
        for k, v in tr.params.items():
            row[f"param_{k}"] = float(v)
        # user attrs
        for k, v in tr.user_attrs.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[f"attr_{k}"] = v
            else:
                row[f"attr_{k}"] = str(v)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Validate inputs early
    if not RECS_PATH.exists():
        raise FileNotFoundError(f"RECS_PATH not found: {RECS_PATH.resolve()}")
    if not OFFERS_OPM_DATA_PATH.exists():
        raise FileNotFoundError(f"OFFERS_OPM_DATA_PATH not found: {OFFERS_OPM_DATA_PATH.resolve()}")

    if TOPK <= 0:
        raise ValueError("TOPK must be > 0")

    out_dir = make_output_dir()
    print(f"INFO: output_dir = {out_dir.resolve()}")

    # -----------------------------
    # Load + enrich
    # -----------------------------
    recs = pd.read_parquet(RECS_PATH)
    ensure_required_columns(recs, RECS_REQUIRED_COLS, df_name="recs")
    recs = normalize_create_date(recs)

    offers_opm = read_offer_performance_metrics(OFFERS_OPM_DATA_PATH)
    recs = enrich_with_offer_performance_metrics(
        recs,
        offers_opm,
        day_shift_days=OFFERS_OPM_DAY_SHIFT_DAYS,
        asof_direction=OFFERS_OPM_ASOF_DIRECTION,
    )
    recs = build_features(recs)

    # -----------------------------
    # Build request arrays
    # -----------------------------
    pos_w = get_position_weights(TOPK)
    requests = build_request_arrays(recs)
    if not requests:
        raise ValueError("No requests found after preprocessing.")

    train_reqs, valid_reqs = split_train_valid(requests)
    print(f"INFO: requests total={len(requests)} | train={len(train_reqs)} | valid={len(valid_reqs)}")

    # -----------------------------
    # Reference revenues
    # -----------------------------
    base_train = eval_baseline_total_revenue(train_reqs, k=TOPK, pos_w=pos_w)
    base_valid = eval_baseline_total_revenue(valid_reqs, k=TOPK, pos_w=pos_w)

    pg_train = eval_pg_total_revenue(train_reqs, k=TOPK, pos_w=pos_w)
    pg_valid = eval_pg_total_revenue(valid_reqs, k=TOPK, pos_w=pos_w)

    def avg(total: float, n: int) -> float:
        return float(total / n) if n else 0.0

    print("\n=== Revenue proxy reference (TOTAL | AVG per request) ===")
    print(f"BASELINE train: {base_train:.6f} | {avg(base_train, len(train_reqs)):.6f}")
    print(f"BASELINE valid: {base_valid:.6f} | {avg(base_valid, len(valid_reqs)):.6f}")
    print(f"PG       train: {pg_train:.6f} | {avg(pg_train, len(train_reqs)):.6f}")
    print(f"PG       valid: {pg_valid:.6f} | {avg(pg_valid, len(valid_reqs)):.6f}")

    # -----------------------------
    # Optuna objective
    # -----------------------------
    def objective(trial: "optuna.Trial") -> float:
        # NOTE: log=True requires low > 0
        gamma_lo, gamma_hi = BOUNDS["gamma"]
        if gamma_lo <= 0.0:
            raise ValueError(f"BOUNDS['gamma'][0] must be > 0 for log sampling, got {gamma_lo}")

        gamma = trial.suggest_float("gamma", gamma_lo, gamma_hi, log=True)
        mu = trial.suggest_float("mu", BOUNDS["mu"][0], BOUNDS["mu"][1])

        # FIX: nu/delta/lambda_ bounds start at 0.0 -> cannot use log=True in Optuna
        nu = trial.suggest_float("nu", BOUNDS["nu"][0], BOUNDS["nu"][1])            # log=False
        rho = trial.suggest_float("rho", BOUNDS["rho"][0], BOUNDS["rho"][1])
        delta = trial.suggest_float("delta", BOUNDS["delta"][0], BOUNDS["delta"][1])  # log=False
        lambda_ = trial.suggest_float("lambda_", BOUNDS["lambda_"][0], BOUNDS["lambda_"][1])  # log=False
        p_h = trial.suggest_float("p_h", BOUNDS["p_h"][0], BOUNDS["p_h"][1])

        p_h_clip = float(np.clip(float(p_h), -1.0, 1.0))
        pacing_term = 1.0 + float(rho) * p_h_clip

        # ---- Train revenue with optional pruning ----
        train_total = 0.0
        n_train = len(train_reqs)
        for i, req in enumerate(train_reqs, start=1):
            sel = greedy_select_topk_indices(
                req,
                k=TOPK,
                gamma=gamma,
                mu=mu,
                nu=nu,
                pacing_term=pacing_term,
                delta=delta,
                lambda_=lambda_,
                w_inv=W_INV,
                w_dev=W_DEV,
                w_city=W_CITY,
            )
            train_total += expected_revenue_for_selected(req, sel, pos_w)

            if USE_PRUNING and (i % REPORT_EVERY_N_REQUESTS == 0):
                train_avg = train_total / i
                trial.report(train_avg, step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        train_avg = float(train_total / n_train) if n_train else 0.0

        # ---- Valid revenue (optimize validation avg to reduce overfit) ----
        valid_total = eval_kurekj_total_revenue(
            valid_reqs,
            k=TOPK,
            pos_w=pos_w,
            gamma=gamma,
            mu=mu,
            nu=nu,
            rho=rho,
            p_h=p_h,
            delta=delta,
            lambda_=lambda_,
        )
        valid_avg = float(valid_total / len(valid_reqs)) if len(valid_reqs) else 0.0

        trial.set_user_attr("train_total_revenue", float(train_total))
        trial.set_user_attr("train_avg_revenue", float(train_avg))
        trial.set_user_attr("valid_total_revenue", float(valid_total))
        trial.set_user_attr("valid_avg_revenue", float(valid_avg))
        trial.set_user_attr("pacing_term", float(pacing_term))

        return valid_avg

    # -----------------------------
    # Study setup
    # -----------------------------
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(10, N_STARTUP_TRIALS // 2)) if USE_PRUNING else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    if ENQUEUE_INIT_TRIAL:
        # clamp to bounds
        init = {}
        for k, (lo, hi) in BOUNDS.items():
            v = float(INIT_PARAMS.get(k, (lo + hi) / 2.0))
            init[k] = float(np.clip(v, lo, hi))
        study.enqueue_trial(init)

        # also print init performance (optional reference)
        init_total_train = eval_kurekj_total_revenue(
            train_reqs,
            k=TOPK,
            pos_w=pos_w,
            gamma=init["gamma"],
            mu=init["mu"],
            nu=init["nu"],
            rho=init["rho"],
            p_h=init["p_h"],
            delta=init["delta"],
            lambda_=init["lambda_"],
        )
        init_total_valid = eval_kurekj_total_revenue(
            valid_reqs,
            k=TOPK,
            pos_w=pos_w,
            gamma=init["gamma"],
            mu=init["mu"],
            nu=init["nu"],
            rho=init["rho"],
            p_h=init["p_h"],
            delta=init["delta"],
            lambda_=init["lambda_"],
        )
        print("\n=== INIT KUREKJ reference ===")
        print(f"INIT train: {init_total_train:.6f} | {avg(init_total_train, len(train_reqs)):.6f}")
        print(f"INIT valid: {init_total_valid:.6f} | {avg(init_total_valid, len(valid_reqs)):.6f}")

    # -----------------------------
    # Optimize
    # -----------------------------
    print("\n=== OPTUNA START ===")
    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            timeout=TIMEOUT_SECONDS,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    except TypeError:
        # older Optuna may not support show_progress_bar
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            timeout=TIMEOUT_SECONDS,
            gc_after_trial=True,
        )

    # guard: must have at least one completed trial
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed:
        raise RuntimeError("No completed Optuna trials (all pruned/failed). Try disabling pruning or widening bounds.")

    # -----------------------------
    # Results
    # -----------------------------
    best = study.best_trial
    best_params = best.params

    best_valid_avg = float(best.value) if best.value is not None else float("nan")
    best_train_avg = float(best.user_attrs.get("train_avg_revenue", float("nan")))
    best_train_total = float(best.user_attrs.get("train_total_revenue", float("nan")))
    best_valid_total = float(best.user_attrs.get("valid_total_revenue", float("nan")))

    print("\n=== BEST FOUND ===")
    print(f"best_trial_number: {best.number}")
    print(f"BEST valid_avg_revenue_per_request: {best_valid_avg:.8f}")
    print(f"BEST train_avg_revenue_per_request: {best_train_avg:.8f}")
    print("BEST params:")
    for k in ["gamma", "mu", "nu", "rho", "delta", "lambda_", "p_h"]:
        print(f"  {k:8s} = {float(best_params[k]):.12f}")

    # Save trials
    trials_df = trials_to_dataframe(study)
    trials_csv = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    # Save best params + context
    best_json = out_dir / "best_params.json"
    payload = {
        "settings": {
            "RECS_PATH": str(RECS_PATH),
            "OFFERS_OPM_DATA_PATH": str(OFFERS_OPM_DATA_PATH),
            "TOPK": TOPK,
            "PROFIT_PROB_COL": PROFIT_PROB_COL,
            "BILLABLE_CAP_THRESHOLD": BILLABLE_CAP_THRESHOLD,
            "POSITION_WEIGHT_MODE": POSITION_WEIGHT_MODE,
            "CUSTOM_POSITION_WEIGHTS": CUSTOM_POSITION_WEIGHTS,
            "SEED": SEED,
            "TRAIN_FRACTION": TRAIN_FRACTION,
            "MAX_REQUESTS": MAX_REQUESTS,
            "N_TRIALS": N_TRIALS,
            "TIMEOUT_SECONDS": TIMEOUT_SECONDS,
            "N_STARTUP_TRIALS": N_STARTUP_TRIALS,
            "USE_PRUNING": USE_PRUNING,
            "BOUNDS": BOUNDS,
            "DIVERSITY_WEIGHTS": {"w_inv": W_INV, "w_dev": W_DEV, "w_city": W_CITY},
        },
        "data": {
            "requests_total": len(requests),
            "requests_train": len(train_reqs),
            "requests_valid": len(valid_reqs),
            "position_weights": get_position_weights(TOPK).tolist(),
        },
        "reference": {
            "baseline_train_total": base_train,
            "baseline_valid_total": base_valid,
            "pg_train_total": pg_train,
            "pg_valid_total": pg_valid,
            "baseline_train_avg": avg(base_train, len(train_reqs)),
            "baseline_valid_avg": avg(base_valid, len(valid_reqs)),
            "pg_train_avg": avg(pg_train, len(train_reqs)),
            "pg_valid_avg": avg(pg_valid, len(valid_reqs)),
        },
        "best": {
            "trial_number": best.number,
            "valid_avg": best_valid_avg,
            "train_avg": best_train_avg,
            "train_total": best_train_total,
            "valid_total": best_valid_total,
            "params": {k: float(v) for k, v in best_params.items()},
            "user_attrs": {k: best.user_attrs.get(k) for k in best.user_attrs.keys()},
        },
    }

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"  trials: {trials_csv.resolve()}")
    print(f"  best  : {best_json.resolve()}")


if __name__ == "__main__":
    main()
