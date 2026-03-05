from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.api.types import DatetimeTZDtype
from tqdm.auto import tqdm

from src.v2.api.services.business_reranking import (  # type: ignore
    Candidate,
    RerankParams,
    greedy_rerank,
    business_score,
)

# =========================
# CONFIG
# =========================

RECS_PATH = Path("recommendations_output.parquet")
OUT_XLSX_BASENAME = "compare_top3"

TOPK = 3
P_H = 0.0

API_MODEL_SCORE_MISSING_SENTINEL = -1.0

DEFAULT_CONTRACT_TYPE = "flat"
DEFAULT_CAP_RATIO = 0.0
DEFAULT_M = 0.0
DEFAULT_V = 0

# --- NEW: daily_snapshot.offers_billinggroup snapshot inputs ---
# The goal is to feed Candidate(m, v, contract_type, inv_id) from real business data,
# instead of hard-coded DEFAULT_* values.
OFFERS_BG_DATA_PATH = Path("daily_snapshot_offers_billinggroup_data_20260109_052527.xlsx")
OFFERS_BG_META_PATH = Path(
    "daily_snapshot_offers_billinggroup_meta_20260109_052527.xlsx"
)  # optional (diagnostics only)

# Controls for aligning recommendation create_date to snapshot "day".
# Useful when the snapshot is calculated for a different day than recs.create_date (e.g. "yesterday").
OFFERS_BG_DAY_SHIFT_DAYS = 0  # int

# merge_asof direction. "backward" = take last snapshot day <= rec_day.
OFFERS_BG_ASOF_DIRECTION = "backward"  # {'backward','forward','nearest'}

# Optional mapping offer_id -> group_id if recs doesn't contain group_id.
# Supported formats: .csv / .xlsx / .parquet
OFFER_TO_GROUP_MAP_PATH: Path | None = None

# NOTE: adjust these if src.v2.api.services.business_reranking expects different strings.
CONTRACT_TYPE_CPL = "cpl"  # settlement==0
CONTRACT_TYPE_FLAT = "flat"  # settlement==1 (also DEFAULT_CONTRACT_TYPE)

RERANK_PARAMS_DICT = dict(
    gamma=1.2,
    mu=0.5,
    nu=0.8,
    rho=0.3,
    delta=1.0,
    lambda_=0.5,
)

# =========================

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

OFFERS_BG_REQUIRED_COLS = [
    "id",
    "day",
    "group_id",
    "settlement",
    "billing_cycle",
    "budget_date_range",
    "budget",
    "lead_count",
    "lead_price",
    "realization_count_method",
]


# =========================
# HELPERS
# =========================

def read_table(path: Path) -> pd.DataFrame:
    """Generic reader for optional mapping files and recs input."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def ensure_required_columns(df: pd.DataFrame, required: list[str], *, df_name: str = "df") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name}: missing required columns: {missing}")


def normalize_create_date(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize create_date to tz-naive Europe/Warsaw (for stable day flooring)."""
    out = df.copy()
    dt = pd.to_datetime(out["create_date"], errors="raise")

    if isinstance(dt.dtype, DatetimeTZDtype):
        dt = dt.dt.tz_convert("Europe/Warsaw").dt.tz_localize(None)

    out["create_date"] = dt
    return out


def read_offers_billinggroup(path: Path) -> pd.DataFrame:
    """Read daily_snapshot.offers_billinggroup XLSX dump (sheet: "data").

    Expected columns (based on meta):
    ["id","day","group_id","settlement","billing_cycle","budget_date_range",
     "budget","lead_count","lead_price","realization_count_method"]

    Types:
      - day: datetime64[ns] (date only)
      - group_id: Int64 (nullable)
      - settlement: Int64 (nullable)
      - budget: float
      - lead_count: float (nullable) -> later mapped to Candidate.v (int)
    """
    if not path.exists():
        raise FileNotFoundError(f"offers_billinggroup data file not found: {path.resolve()}")

    df = pd.read_excel(path, sheet_name="data")
    ensure_required_columns(df, OFFERS_BG_REQUIRED_COLS, df_name="offers_billinggroup")

    out = df.copy()

    # day is a snapshot date (no time); we normalize to midnight.
    out["day"] = pd.to_datetime(out["day"], errors="raise").dt.normalize()

    # group_id can be nullable depending on data; keep as pandas nullable integer.
    out["group_id"] = pd.to_numeric(out["group_id"], errors="coerce").astype("Int64")

    # settlement_type: 0=CPL, 1=flat/subscription.
    out["settlement"] = pd.to_numeric(out["settlement"], errors="coerce").astype("Int64")

    # budget: treat as float for Candidate.m.
    out["budget"] = pd.to_numeric(out["budget"], errors="coerce").astype(float)

    # lead_count: numeric, can be NaN.
    out["lead_count"] = pd.to_numeric(out["lead_count"], errors="coerce")

    # Optional: enforce uniqueness for merge_asof key (group_id, day).
    dup = out.duplicated(["group_id", "day"], keep=False)
    if dup.any():
        sample = out.loc[dup, ["group_id", "day"]].head(10)
        raise ValueError(
            "offers_billinggroup: duplicate rows for (group_id, day). "
            "This breaks merge_asof assumptions. Sample:\n"
            f"{sample}"
        )

    return out


def _infer_group_id_column(recs: pd.DataFrame) -> str | None:
    """Try to find a group_id-like column in recs."""
    for c in [
        "group_id",
        "billing_group_id",
        "billinggroup_id",
        "billingGroupId",
        "billing_groupid",
    ]:
        if c in recs.columns:
            return c
    return None


def _read_offer_to_group_map(path: Path) -> pd.DataFrame:
    """Read optional mapping offer_id -> group_id.

    The mapping file must contain:
      - offer_id (or id)
      - group_id (or billing_group_id)
    """
    if not path.exists():
        raise FileNotFoundError(f"OFFER_TO_GROUP_MAP_PATH not found: {path.resolve()}")

    df = read_table(path)

    offer_col = "offer_id" if "offer_id" in df.columns else ("id" if "id" in df.columns else None)
    if offer_col is None:
        raise ValueError(
            "offer_to_group map: missing offer_id column. Expected 'offer_id' (preferred) or 'id'."
        )

    group_col = _infer_group_id_column(df)
    if group_col is None:
        raise ValueError(
            "offer_to_group map: missing group_id column. Expected 'group_id' or 'billing_group_id'."
        )

    out = df[[offer_col, group_col]].copy()
    out = out.rename(columns={offer_col: "offer_id", group_col: "group_id"})

    out["offer_id"] = pd.to_numeric(out["offer_id"], errors="coerce").astype("Int64")
    out["group_id"] = pd.to_numeric(out["group_id"], errors="coerce").astype("Int64")

    out = out.dropna(subset=["offer_id"]).drop_duplicates(subset=["offer_id"], keep="last")

    return out


def enrich_with_offers_billinggroup(
    recs: pd.DataFrame,
    offers_bg: pd.DataFrame,
    *,
    mapping_path: Path | None = None,
) -> pd.DataFrame:
    """Enrich recs with daily_snapshot.offers_billinggroup and build Candidate business features.

    Join strategy:
      - We need group_id on recs side.
        1) If recs already contains group_id (or billing_group_id), we use it.
        2) Otherwise, if mapping_path is provided, we merge mapping (offer_id -> group_id).
        3) If still missing, we keep group_id as NA and the script falls back to DEFAULT_*.

      - Time alignment:
        create_date is a timestamp; offers_bg.day is a date.
        We compute rec_day = floor(create_date to date) + OFFERS_BG_DAY_SHIFT_DAYS,
        and then do merge_asof per group_id, taking the last snapshot day <= rec_day.

    Output columns added/overwritten:
      - group_id, rec_day
      - day/settlement/budget/lead_count... (from offers_bg; may be NaN if no match)
      - m, v, contract_type, cap_ratio, inv_id (ready for Candidate)

    IMPORTANT: we do NOT change greedy_rerank/business_score logic (library code).
    """
    if OFFERS_BG_ASOF_DIRECTION not in {"backward", "forward", "nearest"}:
        raise ValueError(
            "OFFERS_BG_ASOF_DIRECTION must be one of {'backward','forward','nearest'}, "
            f"got: {OFFERS_BG_ASOF_DIRECTION!r}"
        )

    out = recs.copy()

    # --- group_id presence / normalization ---
    src_group_col = _infer_group_id_column(out)
    if src_group_col is None:
        out["group_id"] = pd.NA
    else:
        if src_group_col != "group_id":
            out = out.rename(columns={src_group_col: "group_id"})

    # Optional mapping offer_id -> group_id (useful when recs doesn't carry group_id)
    if mapping_path is not None:
        map_df = _read_offer_to_group_map(mapping_path)
        out = out.merge(map_df, on="offer_id", how="left", suffixes=("", "_map"))
        if "group_id_map" in out.columns:
            out["group_id"] = out["group_id"].fillna(out["group_id_map"])
            out = out.drop(columns=["group_id_map"])

    out["group_id"] = pd.to_numeric(out["group_id"], errors="coerce").astype("Int64")

    # If we still have missing group_id, reranking will work but on DEFAULT_* fallbacks.
    missing_group_rows = int(out["group_id"].isna().sum())
    if missing_group_rows:
        if src_group_col is None and mapping_path is None:
            print(
                "WARN: recs has no group_id/billing_group_id and OFFER_TO_GROUP_MAP_PATH is None. "
                "Business features will fall back to DEFAULT_* for all rows. "
                f"rows_without_group_id={missing_group_rows}/{len(out)}"
            )
        else:
            print(
                "WARN: group_id is missing for some rows even after mapping. "
                "Those rows will use DEFAULT_* fallbacks. "
                f"rows_without_group_id={missing_group_rows}/{len(out)}"
            )

    # --- time alignment (create_date -> rec_day) ---
    out["rec_day"] = out["create_date"].dt.floor("D") + pd.to_timedelta(
        OFFERS_BG_DAY_SHIFT_DAYS, unit="D"
    )

    # Keep original order (merge_asof requires sorting).
    out["_orig_row_order"] = out.index.astype(int)

    # merge_asof requires both frames sorted by the merge key (on) globally.
    # Sorting by [rec_day, group_id] keeps time monotonic (global), and also groups ids within a day.
    left = out.sort_values(["rec_day", "group_id"], kind="mergesort").reset_index(drop=True)
    right = offers_bg.sort_values(["day", "group_id"], kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        by="group_id",
        left_on="rec_day",
        right_on="day",
        direction=OFFERS_BG_ASOF_DIRECTION,
    )

    # Restore original recs order
    merged = (
        merged.sort_values("_orig_row_order", kind="mergesort")
        .drop(columns=["_orig_row_order"])
        .reset_index(drop=True)
    )

    # --- map offers_billinggroup -> Candidate features ---
    # Candidate.m (budget)
    merged["m"] = pd.to_numeric(merged.get("budget"), errors="coerce").fillna(DEFAULT_M).astype(float)

    # Candidate.v (lead_count); NaN -> 0 (then to int)
    merged["v"] = (
        pd.to_numeric(merged.get("lead_count"), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Candidate.contract_type based on settlement_type:
    #   settlement == 0 => CPL (efficiency)
    #   settlement == 1 => flat/subscription
    #   else/NaN       => DEFAULT_CONTRACT_TYPE
    settlement_num = pd.to_numeric(merged.get("settlement"), errors="coerce")
    merged["contract_type"] = DEFAULT_CONTRACT_TYPE
    merged.loc[settlement_num == 0, "contract_type"] = CONTRACT_TYPE_CPL
    merged.loc[settlement_num == 1, "contract_type"] = CONTRACT_TYPE_FLAT

    # Candidate.inv_id: prefer group_id (billing group); fallback to offer_id
    offer_id_int = pd.to_numeric(merged["offer_id"], errors="raise").astype("Int64")
    merged["inv_id"] = merged["group_id"].astype("Int64").fillna(offer_id_int)

    # Candidate.cap_ratio:
    # If estimated_perc_realization exists (not in this snapshot export), use it.
    # TODO: don't infer cap_ratio from budget/lead_price without a proper definition.
    if "estimated_perc_realization" in merged.columns:
        merged["cap_ratio"] = (
            pd.to_numeric(merged["estimated_perc_realization"], errors="coerce")
            .fillna(DEFAULT_CAP_RATIO)
            .clip(0.0, 1.0)
        )
    else:
        merged["cap_ratio"] = DEFAULT_CAP_RATIO

    # --- diagnostics (join quality) ---
    total_rows = len(merged)
    rows_with_group_id = int(merged["group_id"].notna().sum())
    rows_with_budget = int(merged["budget"].notna().sum()) if "budget" in merged.columns else 0
    rows_with_settlement = (
        int(merged["settlement"].notna().sum()) if "settlement" in merged.columns else 0
    )

    # Fallbacks: if any of the key billing fields is missing, we effectively used defaults.
    used_fallback_m = merged["budget"].isna() if "budget" in merged.columns else pd.Series([True] * total_rows)
    used_fallback_contract = (
        merged["settlement"].isna() if "settlement" in merged.columns else pd.Series([True] * total_rows)
    )
    used_fallback_inv = merged["group_id"].isna()
    used_any_fallback = used_fallback_m | used_fallback_contract | used_fallback_inv

    fallback_candidates = int(used_any_fallback.sum())
    fallback_requests_any = int(used_any_fallback.groupby(merged["request_id"], sort=False).any().sum())

    print(
        "offers_billinggroup merge summary:\n"
        f"  rows_total..................: {total_rows}\n"
        f"  rows_with_group_id..........: {rows_with_group_id} ({rows_with_group_id/total_rows:.1%})\n"
        f"  rows_with_budget (m source).: {rows_with_budget} ({rows_with_budget/total_rows:.1%})\n"
        f"  rows_with_settlement........: {rows_with_settlement} ({rows_with_settlement/total_rows:.1%})\n"
        f"  fallback_candidates.........: {fallback_candidates} ({fallback_candidates/total_rows:.1%})\n"
        f"  fallback_requests_any.......: {fallback_requests_any}"
    )

    return merged


def _prepare_q(series: pd.Series) -> pd.Series:
    q = pd.to_numeric(series, errors="raise").fillna(0.0)
    if API_MODEL_SCORE_MISSING_SENTINEL is not None:
        q = q.mask(q == API_MODEL_SCORE_MISSING_SENTINEL, 0.0)
    return q.clip(-1.0, 1.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all model/reranking features.

    Important change vs previous version:
      - If df already has columns: m, v, contract_type, cap_ratio, inv_id
        (e.g. after offers_billinggroup enrichment), we DO NOT overwrite them.
      - Otherwise, we fall back to DEFAULT_* values (preserves previous behavior).
    """
    out = df.copy().reset_index(drop=True)
    out["row_id"] = out.index.astype(int)

    for c in ["api_model_score", "final_score", "gap_score", "lead_price_score"]:
        out[c] = pd.to_numeric(out[c], errors="raise")

    out["q_api"] = _prepare_q(out["api_model_score"])
    out["q_final"] = _prepare_q(out["final_score"])

    out["r"] = out["lead_price_score"].fillna(0.0).clip(0.0, 1.0)
    out["g"] = out["gap_score"].fillna(0.0).clip(0.0, 1.0)

    # Business features (may be pre-filled by enrichment)
    if "m" in out.columns:
        out["m"] = pd.to_numeric(out["m"], errors="coerce").fillna(DEFAULT_M).astype(float)
    else:
        out["m"] = float(DEFAULT_M)

    if "v" in out.columns:
        out["v"] = (
            pd.to_numeric(out["v"], errors="coerce")
            .fillna(DEFAULT_V)
            .astype(int)
        )
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


# =========================
# TOP-K
# =========================

def baseline_topk_by(df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = df.sort_values(
        ["request_id", "q_api", "property_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()

    out["baseline_rank"] = out.groupby("request_id").cumcount() + 1
    return out[out["baseline_rank"] <= k].copy()


def business_topk_by(
    df: pd.DataFrame,
    params: RerankParams,
    k: int,
    p_h: float,
    q_col: str,
    rank_col: str,
    score_col: str,
    desc: str,
) -> pd.DataFrame:
    parts = []
    grouped = df.groupby("request_id", sort=False)

    for _, g in tqdm(grouped, total=grouped.ngroups, desc=desc, unit="request"):
        candidates = []
        for row in g.itertuples(index=False):
            candidates.append(
                Candidate(
                    property_id=int(row.property_id),
                    q=float(getattr(row, q_col)),
                    r=float(row.r),
                    g=float(row.g),
                    m=float(row.m),
                    v=int(row.v),
                    contract_type=row.contract_type,
                    cap_ratio=float(row.cap_ratio),
                    inv_id=int(row.inv_id) if pd.notna(row.inv_id) else None,
                    extra={"row_id": int(row.row_id)},
                )
            )

        ranked = greedy_rerank(candidates, params, k=k, p_h=p_h)

        selected = []
        rows = []
        for rank, cand in enumerate(ranked, start=1):
            score = business_score(cand, params, p_h=p_h, selected=selected)
            selected.append(cand)
            rows.append(
                {
                    "row_id": cand.extra["row_id"],
                    rank_col: rank,
                    score_col: score,
                }
            )

        out = g.merge(pd.DataFrame(rows), on="row_id", how="inner")
        parts.append(out.sort_values(rank_col, kind="mergesort"))

    return pd.concat(parts, ignore_index=True)


# =========================
# SUMMARY WITH SCORES
# =========================

def summary_wide_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    k: int,
    left_rank: str,
    right_rank: str,
    left_prefix: str,
    right_prefix: str,
    left_score_col: str,
    right_score_col: str,
) -> pd.DataFrame:
    meta = (
        left[["request_id", "uuid", "create_date"]]
        .drop_duplicates("request_id")
        .set_index("request_id")
    )

    l_pid = left.pivot(index="request_id", columns=left_rank, values="property_id")
    l_pid.columns = [f"{left_prefix}_pid_{c}" for c in l_pid.columns]

    l_score = left.pivot(index="request_id", columns=left_rank, values=left_score_col)
    l_score.columns = [f"{left_prefix}_score_{c}" for c in l_score.columns]

    r_pid = right.pivot(index="request_id", columns=right_rank, values="property_id")
    r_pid.columns = [f"{right_prefix}_pid_{c}" for c in r_pid.columns]

    r_score = right.pivot(index="request_id", columns=right_rank, values=right_score_col)
    r_score.columns = [f"{right_prefix}_score_{c}" for c in r_score.columns]

    out = meta.join(l_pid).join(l_score).join(r_pid).join(r_score).reset_index()

    for i in range(1, k + 1):
        for col in (
            f"{left_prefix}_pid_{i}",
            f"{left_prefix}_score_{i}",
            f"{right_prefix}_pid_{i}",
            f"{right_prefix}_score_{i}",
        ):
            if col not in out.columns:
                out[col] = pd.NA

    def overlap(row):
        a = {row[f"{left_prefix}_pid_{i}"] for i in range(1, k + 1)}
        b = {row[f"{right_prefix}_pid_{i}"] for i in range(1, k + 1)}
        return len({x for x in a if pd.notna(x)} & {x for x in b if pd.notna(x)})

    out[f"top{k}_overlap_cnt"] = out.apply(overlap, axis=1)

    return out


# =========================
# METRICS SHEET
# =========================

def compute_pair_metrics(
    summary_df: pd.DataFrame,
    k: int,
    left_prefix: str,
    right_prefix: str,
) -> dict:
    n = len(summary_df)

    overlap = summary_df[f"top{k}_overlap_cnt"].fillna(0)

    top1_match = (
        summary_df[f"{left_prefix}_pid_1"]
        == summary_df[f"{right_prefix}_pid_1"]
    ).fillna(False)

    full_match = overlap == k

    return {
        "requests": int(n),
        "avg_overlap_ratio": float((overlap / k).mean()) if n else 0.0,
        "pct_full_topk_match": float(full_match.mean()) if n else 0.0,
        "pct_top1_match": float(top1_match.mean()) if n else 0.0,
    }


def build_metrics_sheet(
    k: int,
    sum_base_pg: pd.DataFrame,
    sum_base_kurekj: pd.DataFrame,
    sum_pg_kurekj: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.append({
        "comparison": "baseline vs propertygroup",
        **compute_pair_metrics(sum_base_pg, k, "base", "pg"),
    })
    rows.append({
        "comparison": "baseline vs kurekj",
        **compute_pair_metrics(sum_base_kurekj, k, "base", "kurekj"),
    })
    rows.append({
        "comparison": "propertygroup vs kurekj",
        **compute_pair_metrics(sum_pg_kurekj, k, "pg", "kurekj"),
    })

    df = pd.DataFrame(rows)

    df["avg_overlap_ratio_pct"] = (df["avg_overlap_ratio"] * 100).round(2)
    df["pct_full_topk_match_pct"] = (df["pct_full_topk_match"] * 100).round(2)
    df["pct_top1_match_pct"] = (df["pct_top1_match"] * 100).round(2)

    ordered = [
        "comparison",
        "requests",
        "avg_overlap_ratio",
        "avg_overlap_ratio_pct",
        "pct_full_topk_match",
        "pct_full_topk_match_pct",
        "pct_top1_match",
        "pct_top1_match_pct",
    ]

    return df[ordered]


# =========================
# EXCEL
# =========================

def choose_excel_engine() -> str:
    try:
        __import__("xlsxwriter")
        return "xlsxwriter"
    except Exception:
        return "openpyxl"


# =========================
# MAIN
# =========================

def main() -> None:
    recs = read_table(RECS_PATH)
    ensure_required_columns(recs, RECS_REQUIRED_COLS, df_name="recs")

    recs = normalize_create_date(recs)

    # Enrich with offers_billinggroup snapshot if available.
    # If not available (or no group_id), the script keeps working on DEFAULT_* values.
    if OFFERS_BG_DATA_PATH.exists():
        offers_bg = read_offers_billinggroup(OFFERS_BG_DATA_PATH)
        recs = enrich_with_offers_billinggroup(
            recs,
            offers_bg,
            mapping_path=OFFER_TO_GROUP_MAP_PATH,
        )
    else:
        print(
            "WARN: OFFERS_BG_DATA_PATH does not exist -> using DEFAULT_* for business features. "
            f"path={OFFERS_BG_DATA_PATH.resolve()}"
        )

    recs = build_features(recs)

    params = RerankParams(**RERANK_PARAMS_DICT)

    baseline_top = baseline_topk_by(recs, TOPK)

    pg_top = business_topk_by(
        recs,
        params,
        TOPK,
        P_H,
        q_col="q_final",
        rank_col="pg_rank",
        score_col="pg_score",
        desc="Propertygroup reranking (final_score)",
    )

    kurekj_top = business_topk_by(
        recs,
        params,
        TOPK,
        P_H,
        q_col="q_api",
        rank_col="kurekj_rank",
        score_col="kurekj_score",
        desc="Kurekj reranking (api_model_score)",
    )

    sum_base_pg = summary_wide_pair(
        baseline_top,
        pg_top,
        TOPK,
        "baseline_rank",
        "pg_rank",
        "base",
        "pg",
        "q_api",
        "pg_score",
    )

    sum_base_kurekj = summary_wide_pair(
        baseline_top,
        kurekj_top,
        TOPK,
        "baseline_rank",
        "kurekj_rank",
        "base",
        "kurekj",
        "q_api",
        "kurekj_score",
    )

    sum_pg_kurekj = summary_wide_pair(
        pg_top,
        kurekj_top,
        TOPK,
        "pg_rank",
        "kurekj_rank",
        "pg",
        "kurekj",
        "pg_score",
        "kurekj_score",
    )

    metrics_df = build_metrics_sheet(
        TOPK,
        sum_base_pg,
        sum_base_kurekj,
        sum_pg_kurekj,
    )

    out_dir = Path(__file__).resolve().parent / f"Output_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{OUT_XLSX_BASENAME}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"

    with pd.ExcelWriter(out_path, engine=choose_excel_engine()) as xw:
        baseline_top.to_excel(xw, "baseline_topk", index=False)
        pg_top.to_excel(xw, "pg_business_topk", index=False)
        kurekj_top.to_excel(xw, "kurekj_business", index=False)

        sum_base_pg.to_excel(xw, "sum_base_vs_pg", index=False)
        sum_base_kurekj.to_excel(xw, "sum_base_vs_kurekj", index=False)
        sum_pg_kurekj.to_excel(xw, "sum_pg_vs_kurekj", index=False)

        metrics_df.to_excel(xw, "metrics", index=False)
        pd.DataFrame([asdict(params)]).to_excel(xw, "params", index=False)

    print(f"OK: saved {out_path.resolve()}")


if __name__ == "__main__":
    main()

# =========================
# CHANGES
# =========================
# - Added OFFERS_BG_* configuration (data/meta paths, day shift, merge_asof direction) and optional OFFER_TO_GROUP_MAP_PATH.
# - Implemented read_offers_billinggroup() with required-column validation and explicit dtype parsing.
# - Implemented enrich_with_offers_billinggroup():
#     * obtains group_id from recs (group_id/billing_group_id) or optional offer_id->group_id map
#     * aligns create_date to snapshot day via rec_day and merge_asof (backward)
#     * maps billinggroup fields to Candidate features: m=budget, v=lead_count, contract_type from settlement, inv_id prefers group_id
#     * keeps cap_ratio on DEFAULT_CAP_RATIO (TODO: use estimated_perc_realization if/when available)
#     * prints merge quality diagnostics (coverage + fallback counts)
# - Updated build_features(): does not overwrite pre-enriched (m,v,contract_type,cap_ratio,inv_id); only cleans/typizes/fills defaults.
