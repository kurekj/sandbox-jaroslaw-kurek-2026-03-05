# run_optimize_kurekj_params_profit.py  (lightly improved v1: same logic, better tuning)

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype
from tqdm.auto import tqdm

from src.v2.api.services.business_reranking import (  # type: ignore
    Candidate,
    RerankParams,
    greedy_rerank,
)

# =========================================================
# CONFIG
# =========================================================

# --- Inputs ---
RECS_PATH = Path("recommendations_output.parquet")

OFFERS_OPM_DATA_PATH = Path("daily_snapshot_offer_performance_metrics_data_20260120_174227.xlsx")
OFFERS_OPM_DAY_SHIFT_DAYS = 0
OFFERS_OPM_ASOF_DIRECTION = "backward"  # {'backward','forward','nearest'}
REQUIRE_OPM_FOR_PROFIT = True  # dla sensownego profitu lepiej True

# --- TOPK / profit ---
TOPK = 3
PROFIT_PROB_COL = "q_api"  # stabilny prob proxy
BILLABLE_CAP_THRESHOLD = 1.0

POSITION_WEIGHT_MODE = "dcg"  # {'dcg','equal','custom'}
CUSTOM_POSITION_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.5}  # tylko gdy custom

# --- Baseline "start params" (Twoje aktualne) ---
BASE_RERANK_PARAMS_DICT = dict(
    gamma=1.2,
    mu=0.5,
    nu=0.8,
    rho=0.3,
    delta=1.0,
    lambda_=0.5,
)
BASE_P_H = 0.0

# --- Search strategy (lightly improved) ---
RANDOM_SEED = 42

# Ile losowych prób (globalnie)
# (większy budżet + mniej rzadkie próbkowanie w 6D)
N_RANDOM_TRIALS = 3500  # było 1000

# Część triali w fazie "random" robimy jako perturbację wokół aktualnego best
# (tania eksploatacja – zwykle podbija wynik)
RANDOM_AROUND_BEST_PROB = 0.35
RANDOM_AROUND_BEST_REL_STD = 0.35

# Local refine: zamiast jednej sigmy robimy annealing (grubo -> średnio -> drobno)
LOCAL_REFINE_SCHEDULE: List[Tuple[float, int]] = [
    (0.25, 120),  # coarse
    (0.12, 180),  # mid
    (0.06, 300),  # fine
]
N_LOCAL_TRIALS = sum(n for _, n in LOCAL_REFINE_SCHEDULE)  # info (nie używamy bezpośrednio)
LOCAL_REL_STD = 0.15  # zostawione dla kompatybilności (nie używane przy schedule)

# Opcjonalne przyspieszenie: oceniaj na próbce requestów w fazie search
# (Top-N później jest liczony na pełnym train/valid)
TRAIN_FRACTION = 0.8
TRAIN_SAMPLE_REQUESTS: Optional[int] = 12000  # było 5000 (mniej szumu => lepsze best)
VALID_SAMPLE_REQUESTS: Optional[int] = 4000   # tani filtr generalizacji

# Re-eval na full: zwykle trzeba przeliczyć więcej niż 40
TOP_N_REEVAL = 120  # było 40

# Dodatkowy tani prefilter po valid_sample dla kandydatów z top train_sample
TOP_N_VALID_PREFILTER = 300
SAMPLE_VALID_WEIGHT = 0.35  # score_sample = (1-w)*train_sample + w*valid_sample

# Zakresy parametrów (zawężone – zwykle daje lepsze wyniki przy skończonym budżecie)
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "gamma": (0.0, 3.0),
    "mu": (0.0, 3.0),
    "nu": (0.0, 3.0),
    "rho": (0.0, 3.0),
    "delta": (0.0, 3.0),
    "lambda_": (0.0, 3.0),
}
PH_BOUNDS = (0.0, 0.5)

# Jeśli chcesz optymalizować tylko część parametrów:
PARAMS_TO_OPTIMIZE = ["gamma", "mu", "nu", "rho", "delta", "lambda_"]  # lub np. ["gamma","rho"]

# --- Misc ---
API_MODEL_SCORE_MISSING_SENTINEL = -1.0
DEFAULT_CONTRACT_TYPE = "flat"
DEFAULT_CAP_RATIO = 0.0
DEFAULT_M = 0.0
DEFAULT_V = 0
CONTRACT_TYPE_CPL = "cpl"    # settlement_type==0
CONTRACT_TYPE_FLAT = "flat"  # settlement_type==1

OUT_BASENAME = "optimize_kurekj_profit"

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


# =========================================================
# IO / HELPERS
# =========================================================

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def ensure_required_columns(df: pd.DataFrame, required: List[str], *, df_name: str = "df") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name}: missing required columns: {missing}")


def normalize_create_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["create_date"], errors="raise")

    if isinstance(dt.dtype, DatetimeTZDtype):
        dt = dt.dt.tz_convert("Europe/Warsaw").dt.tz_localize(None)

    out["create_date"] = dt
    return out


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


def enrich_with_offer_performance_metrics(recs: pd.DataFrame, offers_opm: pd.DataFrame) -> pd.DataFrame:
    if OFFERS_OPM_ASOF_DIRECTION not in {"backward", "forward", "nearest"}:
        raise ValueError(
            "OFFERS_OPM_ASOF_DIRECTION must be one of {'backward','forward','nearest'}, "
            f"got: {OFFERS_OPM_ASOF_DIRECTION!r}"
        )

    out = recs.copy()
    out["offer_id"] = pd.to_numeric(out["offer_id"], errors="raise").astype("Int64")

    out["rec_day"] = out["create_date"].dt.floor("D") + pd.to_timedelta(OFFERS_OPM_DAY_SHIFT_DAYS, unit="D")
    out["_orig_row_order"] = out.index.astype(int)

    left = out.sort_values(["rec_day", "offer_id"], kind="mergesort").reset_index(drop=True)
    right = offers_opm.sort_values(["day", "offer_id"], kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        by="offer_id",
        left_on="rec_day",
        right_on="day",
        direction=OFFERS_OPM_ASOF_DIRECTION,
    )

    merged = (
        merged.sort_values("_orig_row_order", kind="mergesort")
        .drop(columns=["_orig_row_order"])
        .reset_index(drop=True)
    )

    settlement_num = pd.to_numeric(merged.get("settlement_type"), errors="coerce")
    merged["contract_type"] = DEFAULT_CONTRACT_TYPE
    merged.loc[settlement_num == 0, "contract_type"] = CONTRACT_TYPE_CPL
    merged.loc[settlement_num == 1, "contract_type"] = CONTRACT_TYPE_FLAT

    merged["group_id"] = pd.to_numeric(merged.get("group_id"), errors="coerce").astype("Int64")
    merged["inv_id"] = merged["group_id"].fillna(merged["offer_id"]).astype("Int64")

    budget = pd.to_numeric(merged.get("monthly_budget"), errors="coerce")
    est_budget = pd.to_numeric(merged.get("estimated_monthly_budget"), errors="coerce")
    merged["m"] = budget.fillna(est_budget).fillna(DEFAULT_M).astype(float)

    merged["v"] = pd.to_numeric(merged.get("lead_limit"), errors="coerce").fillna(0).astype(int)

    cap = pd.to_numeric(merged.get("estimated_perc_realization"), errors="coerce")
    leads = pd.to_numeric(merged.get("leads"), errors="coerce")
    lead_limit = pd.to_numeric(merged.get("lead_limit"), errors="coerce")
    cap_fallback = leads / lead_limit
    cap = cap.fillna(cap_fallback)

    merged["cap_ratio"] = cap.fillna(DEFAULT_CAP_RATIO).clip(lower=0.0, upper=1.0).astype(float)

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


def _prepare_q(series: pd.Series) -> pd.Series:
    q = pd.to_numeric(series, errors="raise").fillna(0.0)
    if API_MODEL_SCORE_MISSING_SENTINEL is not None:
        q = q.mask(q == API_MODEL_SCORE_MISSING_SENTINEL, 0.0)
    return q.clip(-1.0, 1.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    for c in ["api_model_score", "final_score", "gap_score", "lead_price_score"]:
        out[c] = pd.to_numeric(out[c], errors="raise")

    out["q_api"] = _prepare_q(out["api_model_score"])
    out["q_final"] = _prepare_q(out["final_score"])

    out["r"] = out["lead_price_score"].fillna(0.0).clip(0.0, 1.0)
    out["g"] = out["gap_score"].fillna(0.0).clip(0.0, 1.0)

    if "m" in out.columns:
        out["m"] = pd.to_numeric(out["m"], errors="coerce").fillna(DEFAULT_M).astype(float)
    else:
        out["m"] = float(DEFAULT_M)

    if "v" in out.columns:
        out["v"] = pd.to_numeric(out["v"], errors="coerce").fillna(DEFAULT_V).astype(int)
    else:
        out["v"] = int(DEFAULT_V)

    if "contract_type" in out.columns:
        out["contract_type"] = out["contract_type"].fillna(DEFAULT_CONTRACT_TYPE).astype(str)
    else:
        out["contract_type"] = DEFAULT_CONTRACT_TYPE

    if "cap_ratio" in out.columns:
        out["cap_ratio"] = (
            pd.to_numeric(out["cap_ratio"], errors="coerce")
            .fillna(DEFAULT_CAP_RATIO)
            .clip(0.0, 1.0)
        )
    else:
        out["cap_ratio"] = float(DEFAULT_CAP_RATIO)

    if "inv_id" in out.columns:
        out["inv_id"] = pd.to_numeric(out["inv_id"], errors="coerce").astype("Int64")
    else:
        out["inv_id"] = pd.to_numeric(out["offer_id"], errors="raise").astype("Int64")

    return out


def _series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def compute_effective_lead_price(df: pd.DataFrame) -> pd.Series:
    lead_price = pd.to_numeric(_series_or_nan(df, "lead_price"), errors="coerce")
    est_lead_price = pd.to_numeric(_series_or_nan(df, "estimated_lead_price"), errors="coerce")

    settlement = pd.to_numeric(_series_or_nan(df, "settlement_type"), errors="coerce")
    contract = _series_or_nan(df, "contract_type").astype(str).str.lower()

    if settlement.notna().any():
        is_cpl = settlement.eq(0)
    else:
        is_cpl = contract.eq(CONTRACT_TYPE_CPL)

    eff = lead_price.where(is_cpl, est_lead_price)

    lead_limit = pd.to_numeric(_series_or_nan(df, "lead_limit"), errors="coerce").replace({0: np.nan})
    est_budget = pd.to_numeric(_series_or_nan(df, "estimated_monthly_budget"), errors="coerce")
    monthly_budget = pd.to_numeric(_series_or_nan(df, "monthly_budget"), errors="coerce")
    leads_est = pd.to_numeric(_series_or_nan(df, "leads_estimation"), errors="coerce").replace({0: np.nan})

    budget_any = est_budget.fillna(monthly_budget)

    fallback1 = budget_any / lead_limit
    fallback2 = budget_any / leads_est

    eff = eff.fillna(fallback1).fillna(fallback2)

    return eff.astype(float)


def get_position_weights(k: int) -> Dict[int, float]:
    if POSITION_WEIGHT_MODE == "equal":
        return {i: 1.0 for i in range(1, k + 1)}
    if POSITION_WEIGHT_MODE == "custom":
        return {i: float(CUSTOM_POSITION_WEIGHTS.get(i, 0.0)) for i in range(1, k + 1)}
    return {i: float(1.0 / np.log2(i + 1)) for i in range(1, k + 1)}  # DCG


# =========================================================
# CANDIDATE CACHE + PROFIT EVAL
# =========================================================

def build_candidate_cache_kurekj(df: pd.DataFrame) -> Dict[object, List[Candidate]]:
    """
    Buduje cache: request_id -> lista Candidate (kurekj: q=q_api)
    Zapisuje do cand.extra:
      - base_value = effective_lead_price * q_prob * is_billable
    """
    work = df.copy()

    work["effective_lead_price"] = compute_effective_lead_price(work)

    # q_prob (profit proxy) – stabilnie w [0,1]
    q_prob = (
        pd.to_numeric(_series_or_nan(work, PROFIT_PROB_COL), errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    work["profit_q_prob"] = q_prob

    cap_ratio = pd.to_numeric(_series_or_nan(work, "cap_ratio"), errors="coerce").fillna(0.0)
    work["profit_is_billable"] = (cap_ratio < BILLABLE_CAP_THRESHOLD).astype(int)

    price = (
        pd.to_numeric(work["effective_lead_price"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    work["base_value"] = price * work["profit_q_prob"] * work["profit_is_billable"]

    missing_price_pct = float(pd.to_numeric(work["effective_lead_price"], errors="coerce").isna().mean()) * 100.0
    print(f"effective_lead_price: missing pct = {missing_price_pct:.2f}%")

    cache: Dict[object, List[Candidate]] = {}

    grouped = work.groupby("request_id", sort=False)
    for rid, g in tqdm(grouped, total=grouped.ngroups, desc="Build candidate cache", unit="request"):
        # Stabilny porządek źródłowy (dla deterministycznych tie-breaków)
        g2 = g.sort_values(["property_id"], kind="mergesort")

        cands: List[Candidate] = []
        for row in g2.itertuples(index=False):
            inv_id = None
            if hasattr(row, "inv_id") and pd.notna(row.inv_id):
                inv_id = int(row.inv_id)

            cands.append(
                Candidate(
                    property_id=int(row.property_id),
                    q=float(row.q_api),  # kurekj używa q_api
                    r=float(row.r),
                    g=float(row.g),
                    m=float(row.m),
                    v=int(row.v),
                    contract_type=str(row.contract_type),
                    cap_ratio=float(row.cap_ratio),
                    inv_id=inv_id,
                    extra={"base_value": float(row.base_value)},
                )
            )

        cache[rid] = cands

    return cache


def eval_profit_kurekj(
    cache: Dict[object, List[Candidate]],
    *,
    params: RerankParams,
    p_h: float,
    k: int,
    pos_w: Dict[int, float],
    request_ids: Optional[Iterable[object]] = None,
) -> float:
    total = 0.0
    if request_ids is None:
        request_ids_iter = cache.keys()
    else:
        request_ids_iter = request_ids

    for rid in request_ids_iter:
        cands = cache.get(rid, [])
        if not cands:
            continue

        ranked = greedy_rerank(list(cands), params, k=k, p_h=p_h)
        for rank, cand in enumerate(ranked, start=1):
            base_val = 0.0
            if cand.extra and "base_value" in cand.extra:
                base_val = float(cand.extra["base_value"]) or 0.0
            total += base_val * float(pos_w.get(rank, 0.0))

    return float(total)


def eval_profit_baseline_qapi(
    cache: Dict[object, List[Candidate]],
    *,
    k: int,
    pos_w: Dict[int, float],
    request_ids: Optional[Iterable[object]] = None,
) -> float:
    """
    Baseline: TOP-K sortowane po q_api desc (a potem property_id asc), bez greedy_rerank.
    """
    total = 0.0
    if request_ids is None:
        request_ids_iter = cache.keys()
    else:
        request_ids_iter = request_ids

    for rid in request_ids_iter:
        cands = cache.get(rid, [])
        if not cands:
            continue
        ranked = sorted(cands, key=lambda c: (-float(c.q), int(c.property_id)))[:k]
        for rank, cand in enumerate(ranked, start=1):
            base_val = 0.0
            if cand.extra and "base_value" in cand.extra:
                base_val = float(cand.extra["base_value"]) or 0.0
            total += base_val * float(pos_w.get(rank, 0.0))

    return float(total)


# =========================================================
# OPTIMIZATION
# =========================================================

def sample_params_uniform(rng: np.random.Generator, base: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Losuje parametry w zakresie PARAM_BOUNDS.
    Parametry nieoptymalizowane zostają jak w `base`.
    """
    out = dict(base)
    for name in PARAMS_TO_OPTIMIZE:
        lo, hi = PARAM_BOUNDS[name]
        out[name] = float(rng.uniform(lo, hi))
    p_h = float(rng.uniform(PH_BOUNDS[0], PH_BOUNDS[1]))
    return out, p_h


def perturb_params_local(
    rng: np.random.Generator,
    best: Dict[str, float],
    best_p_h: float,
    rel_std: float,
) -> Tuple[Dict[str, float], float]:
    """
    Lokalne doszlifowanie: multiplikatywny szum log-normalny wokół best.
    """
    out = dict(best)
    for name in PARAMS_TO_OPTIMIZE:
        lo, hi = PARAM_BOUNDS[name]
        cur = float(out[name])
        if cur <= 0.0:
            proposal = float(rng.uniform(lo, hi))
        else:
            proposal = float(cur * np.exp(rng.normal(0.0, rel_std)))
        out[name] = float(np.clip(proposal, lo, hi))

    ph = float(best_p_h + rng.normal(0.0, rel_std))
    ph = float(np.clip(ph, PH_BOUNDS[0], PH_BOUNDS[1]))
    return out, ph


def choose_excel_engine() -> str:
    try:
        __import__("xlsxwriter")
        return "xlsxwriter"
    except Exception:
        return "openpyxl"


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    recs = read_table(RECS_PATH)
    ensure_required_columns(recs, RECS_REQUIRED_COLS, df_name="recs")
    recs = normalize_create_date(recs)

    if OFFERS_OPM_DATA_PATH.exists():
        offers_opm = read_offer_performance_metrics(OFFERS_OPM_DATA_PATH)
        recs = enrich_with_offer_performance_metrics(recs, offers_opm)
    else:
        msg = f"OFFERS_OPM_DATA_PATH does not exist: {OFFERS_OPM_DATA_PATH.resolve()}"
        if REQUIRE_OPM_FOR_PROFIT:
            raise FileNotFoundError(msg + " (REQUIRE_OPM_FOR_PROFIT=True)")
        print("WARN:", msg, "=> profit będzie prawdopodobnie bez sensu (brak cen/budżetów).")

    recs = build_features(recs)

    # Cache kandydatów per request_id (raz)
    cache = build_candidate_cache_kurekj(recs)

    req_ids = list(cache.keys())
    rng.shuffle(req_ids)

    n = len(req_ids)
    if n == 0:
        raise ValueError("No request_id groups found after preprocessing.")

    split = int(n * TRAIN_FRACTION)
    train_ids = req_ids[:split]
    valid_ids = req_ids[split:] if split < n else []

    # Sample train for fast search
    train_ids_sample = train_ids
    if TRAIN_SAMPLE_REQUESTS is not None and len(train_ids) > TRAIN_SAMPLE_REQUESTS:
        train_ids_sample = rng.choice(train_ids, size=TRAIN_SAMPLE_REQUESTS, replace=False).tolist()

    # Sample valid for cheap prefilter (anti-overfit)
    valid_ids_sample = valid_ids
    if valid_ids and VALID_SAMPLE_REQUESTS is not None and len(valid_ids) > VALID_SAMPLE_REQUESTS:
        valid_ids_sample = rng.choice(valid_ids, size=VALID_SAMPLE_REQUESTS, replace=False).tolist()

    pos_w = get_position_weights(TOPK)

    # Reference profits
    baseline_train = eval_profit_baseline_qapi(cache, k=TOPK, pos_w=pos_w, request_ids=train_ids)
    baseline_valid = eval_profit_baseline_qapi(cache, k=TOPK, pos_w=pos_w, request_ids=valid_ids) if valid_ids else 0.0
    baseline_all = eval_profit_baseline_qapi(cache, k=TOPK, pos_w=pos_w, request_ids=req_ids)

    base_params = RerankParams(**BASE_RERANK_PARAMS_DICT)
    base_train = eval_profit_kurekj(cache, params=base_params, p_h=BASE_P_H, k=TOPK, pos_w=pos_w, request_ids=train_ids)
    base_valid = eval_profit_kurekj(cache, params=base_params, p_h=BASE_P_H, k=TOPK, pos_w=pos_w, request_ids=valid_ids) if valid_ids else 0.0
    base_all = eval_profit_kurekj(cache, params=base_params, p_h=BASE_P_H, k=TOPK, pos_w=pos_w, request_ids=req_ids)

    print("\n=== Reference profit (expected revenue proxy) ===")
    print(f"baseline(q_api sort)  train={baseline_train:.6f}  valid={baseline_valid:.6f}  all={baseline_all:.6f}")
    print(f"current(params)       train={base_train:.6f}  valid={base_valid:.6f}  all={base_all:.6f}")

    trials: List[Dict[str, object]] = []

    # Trial 0: start params (dla porównania na sample)
    train_sample_profit0 = eval_profit_kurekj(
        cache, params=base_params, p_h=BASE_P_H, k=TOPK, pos_w=pos_w, request_ids=train_ids_sample
    )
    trials.append(
        {
            "trial_id": 0,
            "stage": "start",
            **BASE_RERANK_PARAMS_DICT,
            "p_h": BASE_P_H,
            "train_profit_sample": train_sample_profit0,
        }
    )

    best_params_dict = dict(BASE_RERANK_PARAMS_DICT)
    best_p_h = float(BASE_P_H)
    best_profit_sample = float(train_sample_profit0)

    # --- Random search (global + around best mix) ---
    for t in tqdm(range(1, N_RANDOM_TRIALS + 1), desc="Random search", unit="trial"):
        # mieszanka: część triali eksploruje globalnie, część eksploatuje okolice best
        if t > 1 and rng.random() < float(RANDOM_AROUND_BEST_PROB):
            p_dict, p_h = perturb_params_local(
                rng, best_params_dict, best_p_h, rel_std=float(RANDOM_AROUND_BEST_REL_STD)
            )
        else:
            # global random powinien startować od stałej bazy, nie od "best" (ważne przy optymalizacji subset)
            p_dict, p_h = sample_params_uniform(rng, BASE_RERANK_PARAMS_DICT)

        params = RerankParams(**p_dict)

        profit_sample = eval_profit_kurekj(
            cache,
            params=params,
            p_h=p_h,
            k=TOPK,
            pos_w=pos_w,
            request_ids=train_ids_sample,
        )

        row = {"trial_id": t, "stage": "random", **p_dict, "p_h": p_h, "train_profit_sample": profit_sample}
        trials.append(row)

        if profit_sample > best_profit_sample:
            best_profit_sample = float(profit_sample)
            best_params_dict = dict(p_dict)
            best_p_h = float(p_h)

    # --- Local refine around best (annealing schedule) ---
    start_local_id = len(trials)
    trial_id = start_local_id

    for rel_std, n_trials in LOCAL_REFINE_SCHEDULE:
        for _ in tqdm(range(int(n_trials)), desc=f"Local refine std={rel_std}", unit="trial"):
            p_dict, p_h = perturb_params_local(rng, best_params_dict, best_p_h, rel_std=float(rel_std))
            params = RerankParams(**p_dict)

            profit_sample = eval_profit_kurekj(
                cache,
                params=params,
                p_h=p_h,
                k=TOPK,
                pos_w=pos_w,
                request_ids=train_ids_sample,
            )

            row = {
                "trial_id": trial_id,
                "stage": f"local(std={rel_std})",
                **p_dict,
                "p_h": p_h,
                "train_profit_sample": profit_sample,
            }
            trials.append(row)
            trial_id += 1

            if profit_sample > best_profit_sample:
                best_profit_sample = float(profit_sample)
                best_params_dict = dict(p_dict)
                best_p_h = float(p_h)

    trials_df = pd.DataFrame(trials)
    trials_df = trials_df.sort_values("train_profit_sample", ascending=False, kind="mergesort").reset_index(drop=True)

    # --- Re-evaluation top-N on full train/valid/all ---
    # Tani prefilter po valid_sample dla kandydatów z top train_sample
    if valid_ids:
        pre_n = int(max(TOP_N_VALID_PREFILTER, TOP_N_REEVAL))
        cand_df = trials_df.head(pre_n).copy()

        valid_profit_samples: List[float] = []
        for _, rr in tqdm(cand_df.iterrows(), total=len(cand_df), desc="Valid sample prefilter", unit="trial"):
            p_dict = {k: float(rr[k]) for k in PARAM_BOUNDS.keys()}
            p_h = float(rr["p_h"])
            params = RerankParams(**p_dict)

            vps = eval_profit_kurekj(
                cache,
                params=params,
                p_h=p_h,
                k=TOPK,
                pos_w=pos_w,
                request_ids=valid_ids_sample,
            )
            valid_profit_samples.append(float(vps))

        cand_df["valid_profit_sample"] = valid_profit_samples
        w = float(SAMPLE_VALID_WEIGHT)
        cand_df["score_sample"] = (1.0 - w) * cand_df["train_profit_sample"] + w * cand_df["valid_profit_sample"]

        # dorzuć do trials_df (dla excela; dla większości triali będą NaN)
        trials_df = trials_df.merge(
            cand_df[["trial_id", "valid_profit_sample", "score_sample"]],
            on="trial_id",
            how="left",
        )

        top_df = cand_df.sort_values("score_sample", ascending=False, kind="mergesort").head(TOP_N_REEVAL).copy()
    else:
        top_df = trials_df.head(TOP_N_REEVAL).copy()

    full_rows: List[Dict[str, object]] = []

    print(f"\nRe-evaluating TOP {len(top_df)} trials on full train/valid/all ...")

    for _, r in tqdm(top_df.iterrows(), total=len(top_df), desc="Full re-eval", unit="trial"):
        p_dict = {k: float(r[k]) for k in PARAM_BOUNDS.keys()}
        p_h = float(r["p_h"])
        params = RerankParams(**p_dict)

        train_full = eval_profit_kurekj(cache, params=params, p_h=p_h, k=TOPK, pos_w=pos_w, request_ids=train_ids)
        valid_full = eval_profit_kurekj(cache, params=params, p_h=p_h, k=TOPK, pos_w=pos_w, request_ids=valid_ids) if valid_ids else np.nan
        all_full = eval_profit_kurekj(cache, params=params, p_h=p_h, k=TOPK, pos_w=pos_w, request_ids=req_ids)

        full_rows.append(
            {
                "trial_id": int(r["trial_id"]),
                "stage": str(r["stage"]),
                **p_dict,
                "p_h": p_h,
                "train_profit_sample": float(r["train_profit_sample"]),
                "valid_profit_sample": float(r.get("valid_profit_sample", np.nan)) if valid_ids else np.nan,
                "score_sample": float(r.get("score_sample", np.nan)) if valid_ids else np.nan,
                "train_profit_full": float(train_full),
                "valid_profit_full": float(valid_full) if valid_ids else np.nan,
                "all_profit_full": float(all_full),
            }
        )

    full_df = pd.DataFrame(full_rows).sort_values(
        "valid_profit_full" if valid_ids else "train_profit_full",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    best_row = full_df.iloc[0].to_dict()
    best_params_out = {k: float(best_row[k]) for k in PARAM_BOUNDS.keys()}
    best_p_h_out = float(best_row["p_h"])

    best_params_obj = RerankParams(**best_params_out)

    best_train = eval_profit_kurekj(cache, params=best_params_obj, p_h=best_p_h_out, k=TOPK, pos_w=pos_w, request_ids=train_ids)
    best_valid = eval_profit_kurekj(cache, params=best_params_obj, p_h=best_p_h_out, k=TOPK, pos_w=pos_w, request_ids=valid_ids) if valid_ids else 0.0
    best_all = eval_profit_kurekj(cache, params=best_params_obj, p_h=best_p_h_out, k=TOPK, pos_w=pos_w, request_ids=req_ids)

    def pct_lift(new: float, ref: float) -> float:
        if ref == 0.0:
            return float("nan")
        return (new / ref - 1.0) * 100.0

    summary = pd.DataFrame(
        [
            {
                "algo": "baseline(q_api sort)",
                "train_profit": baseline_train,
                "valid_profit": baseline_valid if valid_ids else np.nan,
                "all_profit": baseline_all,
            },
            {
                "algo": "current(params)",
                "train_profit": base_train,
                "valid_profit": base_valid if valid_ids else np.nan,
                "all_profit": base_all,
                "lift_vs_baseline_all_pct": pct_lift(base_all, baseline_all),
            },
            {
                "algo": "best(found)",
                "train_profit": best_train,
                "valid_profit": best_valid if valid_ids else np.nan,
                "all_profit": best_all,
                "lift_vs_baseline_all_pct": pct_lift(best_all, baseline_all),
                "lift_vs_current_all_pct": pct_lift(best_all, base_all),
            },
        ]
    )

    out_dir = Path(__file__).resolve().parent / f"Output_{OUT_BASENAME}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_xlsx = out_dir / f"{OUT_BASENAME}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    out_json = out_dir / "best_params.json"

    best_payload = {
        "best_params": best_params_out,
        "best_p_h": best_p_h_out,
        "objective": "sum_expected_revenue_proxy",
        "topk": TOPK,
        "position_weight_mode": POSITION_WEIGHT_MODE,
        "profit_prob_col": PROFIT_PROB_COL,
        "billable_cap_threshold": BILLABLE_CAP_THRESHOLD,
        "search_config": {
            "n_random_trials": N_RANDOM_TRIALS,
            "random_around_best_prob": RANDOM_AROUND_BEST_PROB,
            "random_around_best_rel_std": RANDOM_AROUND_BEST_REL_STD,
            "local_refine_schedule": LOCAL_REFINE_SCHEDULE,
            "train_sample_requests": TRAIN_SAMPLE_REQUESTS,
            "valid_sample_requests": VALID_SAMPLE_REQUESTS,
            "top_n_valid_prefilter": TOP_N_VALID_PREFILTER,
            "sample_valid_weight": SAMPLE_VALID_WEIGHT,
            "top_n_reeval": TOP_N_REEVAL,
            "param_bounds": PARAM_BOUNDS,
            "ph_bounds": PH_BOUNDS,
        },
        "summary": summary.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(out_xlsx, engine=choose_excel_engine()) as xw:
        summary.to_excel(xw, "summary", index=False)
        trials_df.to_excel(xw, "trials_sorted_by_sample", index=False)
        full_df.to_excel(xw, "top_reeval_full", index=False)

        pd.DataFrame([BASE_RERANK_PARAMS_DICT | {"p_h": BASE_P_H}]).to_excel(xw, "start_params", index=False)
        pd.DataFrame([best_params_out | {"p_h": best_p_h_out}]).to_excel(xw, "best_params", index=False)

    print("\n=== BEST PARAMS (found) ===")
    print(best_params_out, "p_h=", best_p_h_out)
    print("Saved:")
    print(" -", out_xlsx.resolve())
    print(" -", out_json.resolve())


if __name__ == "__main__":
    main()
