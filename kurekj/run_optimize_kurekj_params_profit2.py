# run_optimize_kurekj_params_profit.py (IMPROVED)

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
# CONFIG (IMPROVED)
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

# Opcjonalnie: stabilizacja profitu (domyślnie OFF, żeby zachować kompatybilność)
SOFT_BILLABLE = False  # True => profit_is_billable = 1-cap_ratio zamiast progu binarnego
WINSORIZE_EFFECTIVE_LEAD_PRICE_Q: Optional[float] = None  # np. 0.995; None => OFF

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

# --- Search strategy (IMPROVED) ---
RANDOM_SEED = 42

# Global exploration
RANDOM_SAMPLING = "lhs"  # {'lhs','uniform'}
N_RANDOM_TRIALS = 2000  # było 1000

# Multi-start local refine (lepsza generalizacja)
N_LOCAL_SEEDS = 8
LOCAL_REL_STD_SCHEDULE = (0.25, 0.12, 0.06)  # coarse -> mid -> fine
LOCAL_TRIALS_PER_SEED = 60  # per std per seed

# Próbkowanie requestów dla szybkiego search
TRAIN_FRACTION = 0.8
TRAIN_SAMPLE_REQUESTS: Optional[int] = 5000  # zostawiamy jak było (stabilnie + szybciej)
VALID_SAMPLE_REQUESTS: Optional[int] = 2000  # dodatkowa kontrola overfitu
SAMPLE_OBJECTIVE_VALID_WEIGHT = 0.35  # 0 => tylko train sample, 1 => tylko valid sample

# Ile najlepszych triali przeliczyć na full train/valid/all
TOP_N_REEVAL = 120  # było 40

# Zakresy parametrów (zawężone zgodnie z komentarzem "sensowne defaulty 0..3")
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
PARAMS_TO_OPTIMIZE = ["gamma", "mu", "nu", "rho", "delta", "lambda_"]

# --- Misc ---
API_MODEL_SCORE_MISSING_SENTINEL = -1.0
DEFAULT_CONTRACT_TYPE = "flat"
DEFAULT_CAP_RATIO = 0.0
DEFAULT_M = 0.0
DEFAULT_V = 0
CONTRACT_TYPE_CPL = "cpl"  # settlement_type==0
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
        out["cap_ratio"] = pd.to_numeric(out["cap_ratio"], errors="coerce").fillna(DEFAULT_CAP_RATIO).clip(0.0, 1.0)
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

    # Opcjonalna stabilizacja outlierów w cenie
    if WINSORIZE_EFFECTIVE_LEAD_PRICE_Q is not None:
        q = float(WINSORIZE_EFFECTIVE_LEAD_PRICE_Q)
        if 0.5 < q < 1.0:
            cap = work["effective_lead_price"].quantile(q)
            work["effective_lead_price"] = work["effective_lead_price"].clip(lower=0.0, upper=float(cap))

    # q_prob (profit proxy) – stabilnie w [0,1]
    q_prob = pd.to_numeric(_series_or_nan(work, PROFIT_PROB_COL), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    work["profit_q_prob"] = q_prob

    cap_ratio = pd.to_numeric(_series_or_nan(work, "cap_ratio"), errors="coerce").fillna(0.0)
    if SOFT_BILLABLE:
        work["profit_is_billable"] = (1.0 - cap_ratio.clip(0.0, 1.0)).astype(float)
    else:
        work["profit_is_billable"] = (cap_ratio < BILLABLE_CAP_THRESHOLD).astype(int)

    price = pd.to_numeric(work["effective_lead_price"], errors="coerce").fillna(0.0).clip(lower=0.0)
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


def _lhs_unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """
    Latin Hypercube Sampling w [0,1]^d.
    Lepsze pokrycie przestrzeni niż czysty uniform dla tej samej liczby punktów.
    """
    u = rng.random((n, d))
    out = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + u[:, j]) / n
    return out


def sample_params_lhs(
    rng: np.random.Generator,
    n: int,
    base: Dict[str, float],
) -> List[Tuple[Dict[str, float], float]]:
    names = list(PARAMS_TO_OPTIMIZE)
    d = len(names) + 1  # + p_h
    unit = _lhs_unit(rng, n, d)

    samples: List[Tuple[Dict[str, float], float]] = []
    for i in range(n):
        p = dict(base)
        for j, name in enumerate(names):
            lo, hi = PARAM_BOUNDS[name]
            p[name] = float(lo + unit[i, j] * (hi - lo))
        ph = float(PH_BOUNDS[0] + unit[i, -1] * (PH_BOUNDS[1] - PH_BOUNDS[0]))
        samples.append((p, ph))

    return samples


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

    # Sample valid for anti-overfit objective
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
    base_valid = (
        eval_profit_kurekj(cache, params=base_params, p_h=BASE_P_H, k=TOPK, pos_w=pos_w, request_ids=valid_ids)
        if valid_ids
        else 0.0
    )
    base_all = eval_profit_kurekj(cache, params=base_params, p_h=BASE_P_H, k=TOPK, pos_w=pos_w, request_ids=req_ids)

    print("\n=== Reference profit (expected revenue proxy) ===")
    print(f"baseline(q_api sort)  train={baseline_train:.6f}  valid={baseline_valid:.6f}  all={baseline_all:.6f}")
    print(f"current(params)       train={base_train:.6f}  valid={base_valid:.6f}  all={base_all:.6f}")

    # -----------------------------------------------------
    # Improved objective: blend train_sample + valid_sample
    # -----------------------------------------------------
    def eval_trial_score(params: RerankParams, p_h: float) -> Tuple[float, float, float]:
        """
        Zwraca: (objective_score, train_profit_sample, valid_profit_sample)

        Objective = (1-w)*train_sample + w*valid_sample
        Dzięki temu random/local search mniej overfitują do train sample.
        """
        tr = eval_profit_kurekj(cache, params=params, p_h=p_h, k=TOPK, pos_w=pos_w, request_ids=train_ids_sample)
        if valid_ids:
            va = eval_profit_kurekj(cache, params=params, p_h=p_h, k=TOPK, pos_w=pos_w, request_ids=valid_ids_sample)
            w = float(SAMPLE_OBJECTIVE_VALID_WEIGHT)
            obj = (1.0 - w) * tr + w * va
            return float(obj), float(tr), float(va)

        return float(tr), float(tr), float("nan")

    trials: List[Dict[str, object]] = []

    # Trial 0: start params
    obj0, tr0, va0 = eval_trial_score(base_params, BASE_P_H)
    trials.append(
        {
            "trial_id": 0,
            "stage": "start",
            **BASE_RERANK_PARAMS_DICT,
            "p_h": BASE_P_H,
            "objective_sample": obj0,
            "train_profit_sample": tr0,
            "valid_profit_sample": va0,
        }
    )

    best_params_dict = dict(BASE_RERANK_PARAMS_DICT)
    best_p_h = float(BASE_P_H)
    best_obj = float(obj0)

    # --- Random search (uniform albo LHS) ---
    if RANDOM_SAMPLING == "lhs":
        random_samples = sample_params_lhs(rng, N_RANDOM_TRIALS, BASE_RERANK_PARAMS_DICT)
    elif RANDOM_SAMPLING == "uniform":
        random_samples = [sample_params_uniform(rng, BASE_RERANK_PARAMS_DICT) for _ in range(N_RANDOM_TRIALS)]
    else:
        raise ValueError(f"RANDOM_SAMPLING must be 'lhs' or 'uniform', got: {RANDOM_SAMPLING!r}")

    for t, (p_dict, p_h) in tqdm(
        enumerate(random_samples, start=1),
        total=len(random_samples),
        desc="Random search",
        unit="trial",
    ):
        params = RerankParams(**p_dict)
        obj, tr, va = eval_trial_score(params, p_h)

        trials.append(
            {
                "trial_id": t,
                "stage": "random",
                **p_dict,
                "p_h": p_h,
                "objective_sample": obj,
                "train_profit_sample": tr,
                "valid_profit_sample": va,
            }
        )

        if obj > best_obj:
            best_obj = float(obj)
            best_params_dict = dict(p_dict)
            best_p_h = float(p_h)

    # --- Multi-start local refine: wybierz top seedów i rób annealing ---
    seed_df = (
        pd.DataFrame(trials).sort_values("objective_sample", ascending=False, kind="mergesort").head(N_LOCAL_SEEDS).reset_index(drop=True)
    )

    trial_id = int(max(int(r["trial_id"]) for r in trials)) + 1

    for _, seed in tqdm(
        list(seed_df.iterrows()),
        total=len(seed_df),
        desc="Local refine (multi-start)",
        unit="seed",
    ):
        seed_params = {k: float(seed[k]) for k in PARAM_BOUNDS.keys()}
        seed_p_h = float(seed["p_h"])

        local_best_params = dict(seed_params)
        local_best_p_h = float(seed_p_h)
        local_best_obj = float(seed["objective_sample"])

        for rel_std in LOCAL_REL_STD_SCHEDULE:
            rel_std_f = float(rel_std)
            for _ in range(LOCAL_TRIALS_PER_SEED):
                p_dict, p_h = perturb_params_local(rng, local_best_params, local_best_p_h, rel_std=rel_std_f)
                params = RerankParams(**p_dict)

                obj, tr, va = eval_trial_score(params, p_h)

                trials.append(
                    {
                        "trial_id": trial_id,
                        "stage": f"local(std={rel_std_f})",
                        **p_dict,
                        "p_h": p_h,
                        "objective_sample": obj,
                        "train_profit_sample": tr,
                        "valid_profit_sample": va,
                    }
                )
                trial_id += 1

                # hill-climb per seed
                if obj > local_best_obj:
                    local_best_obj = float(obj)
                    local_best_params = dict(p_dict)
                    local_best_p_h = float(p_h)

                # global best
                if obj > best_obj:
                    best_obj = float(obj)
                    best_params_dict = dict(p_dict)
                    best_p_h = float(p_h)

    trials_df = pd.DataFrame(trials)
    trials_df = trials_df.sort_values("objective_sample", ascending=False, kind="mergesort").reset_index(drop=True)

    # --- Re-evaluation top-N on full train/valid/all ---
    top_df = trials_df.head(TOP_N_REEVAL).copy()
    full_rows: List[Dict[str, object]] = []

    print(f"\nRe-evaluating TOP {len(top_df)} trials on full train/valid/all ...")

    for _, r in tqdm(top_df.iterrows(), total=len(top_df), desc="Full re-eval", unit="trial"):
        p_dict = {k: float(r[k]) for k in PARAM_BOUNDS.keys()}
        p_h = float(r["p_h"])
        params = RerankParams(**p_dict)

        train_full = eval_profit_kurekj(cache, params=params, p_h=p_h, k=TOPK, pos_w=pos_w, request_ids=train_ids)
        valid_full = (
            eval_profit_kurekj(cache, params=params, p_h=p_h, k=TOPK, pos_w=pos_w, request_ids=valid_ids) if valid_ids else np.nan
        )
        all_full = eval_profit_kurekj(cache, params=params, p_h=p_h, k=TOPK, pos_w=pos_w, request_ids=req_ids)

        full_rows.append(
            {
                "trial_id": int(r["trial_id"]),
                "stage": str(r["stage"]),
                **p_dict,
                "p_h": p_h,
                "objective_sample": float(r["objective_sample"]),
                "train_profit_sample": float(r["train_profit_sample"]),
                "valid_profit_sample": float(r["valid_profit_sample"]) if valid_ids else np.nan,
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
    best_valid = (
        eval_profit_kurekj(cache, params=best_params_obj, p_h=best_p_h_out, k=TOPK, pos_w=pos_w, request_ids=valid_ids) if valid_ids else 0.0
    )
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
        "sampling": {
            "random_sampling": RANDOM_SAMPLING,
            "n_random_trials": N_RANDOM_TRIALS,
            "n_local_seeds": N_LOCAL_SEEDS,
            "local_rel_std_schedule": list(LOCAL_REL_STD_SCHEDULE),
            "local_trials_per_seed": LOCAL_TRIALS_PER_SEED,
            "train_sample_requests": TRAIN_SAMPLE_REQUESTS,
            "valid_sample_requests": VALID_SAMPLE_REQUESTS,
            "sample_objective_valid_weight": SAMPLE_OBJECTIVE_VALID_WEIGHT,
            "top_n_reeval": TOP_N_REEVAL,
        },
        "profit_stabilization": {
            "soft_billable": SOFT_BILLABLE,
            "winsorize_effective_lead_price_q": WINSORIZE_EFFECTIVE_LEAD_PRICE_Q,
        },
        "summary": summary.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(out_xlsx, engine=choose_excel_engine()) as xw:
        summary.to_excel(xw, "summary", index=False)
        trials_df.to_excel(xw, "trials_sorted_by_obj_sample", index=False)
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
