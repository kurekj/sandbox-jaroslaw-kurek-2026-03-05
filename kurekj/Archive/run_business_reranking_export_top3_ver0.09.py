from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from datetime import datetime

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


# =========================
# HELPERS
# =========================

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def ensure_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_create_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["create_date"], errors="raise")

    if isinstance(dt.dtype, DatetimeTZDtype):
        dt = dt.dt.tz_convert("Europe/Warsaw").dt.tz_localize(None)

    out["create_date"] = dt
    return out


def _prepare_q(series: pd.Series) -> pd.Series:
    q = pd.to_numeric(series, errors="raise").fillna(0.0)
    if API_MODEL_SCORE_MISSING_SENTINEL is not None:
        q = q.mask(q == API_MODEL_SCORE_MISSING_SENTINEL, 0.0)
    return q.clip(-1.0, 1.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["row_id"] = out.index.astype(int)

    for c in ["api_model_score", "final_score", "gap_score", "lead_price_score"]:
        out[c] = pd.to_numeric(out[c], errors="raise")

    out["q_api"] = _prepare_q(out["api_model_score"])
    out["q_final"] = _prepare_q(out["final_score"])

    out["r"] = out["lead_price_score"].fillna(0.0).clip(0.0, 1.0)
    out["g"] = out["gap_score"].fillna(0.0).clip(0.0, 1.0)

    out["m"] = DEFAULT_M
    out["v"] = DEFAULT_V
    out["contract_type"] = DEFAULT_CONTRACT_TYPE
    out["cap_ratio"] = DEFAULT_CAP_RATIO
    out["inv_id"] = out["offer_id"]

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

    # LEFT
    l_pid = left.pivot(index="request_id", columns=left_rank, values="property_id")
    l_pid.columns = [f"{left_prefix}_pid_{c}" for c in l_pid.columns]

    l_score = left.pivot(index="request_id", columns=left_rank, values=left_score_col)
    l_score.columns = [f"{left_prefix}_score_{c}" for c in l_score.columns]

    # RIGHT
    r_pid = right.pivot(index="request_id", columns=right_rank, values="property_id")
    r_pid.columns = [f"{right_prefix}_pid_{c}" for c in r_pid.columns]

    r_score = right.pivot(index="request_id", columns=right_rank, values=right_score_col)
    r_score.columns = [f"{right_prefix}_score_{c}" for c in r_score.columns]

    out = meta.join(l_pid).join(l_score).join(r_pid).join(r_score).reset_index()

    # ensure fixed 1..k columns
    for i in range(1, k + 1):
        for col in (
            f"{left_prefix}_pid_{i}",
            f"{left_prefix}_score_{i}",
            f"{right_prefix}_pid_{i}",
            f"{right_prefix}_score_{i}",
        ):
            if col not in out.columns:
                out[col] = pd.NA

    # overlap on pid
    def overlap(row):
        a = {row[f"{left_prefix}_pid_{i}"] for i in range(1, k + 1)}
        b = {row[f"{right_prefix}_pid_{i}"] for i in range(1, k + 1)}
        return len({x for x in a if pd.notna(x)} & {x for x in b if pd.notna(x)})

    out[f"top{k}_overlap_cnt"] = out.apply(overlap, axis=1)

    # column order: pid + score side-by-side
    ordered = ["request_id", "uuid", "create_date"]
    for i in range(1, k + 1):
        ordered += [f"{left_prefix}_pid_{i}", f"{left_prefix}_score_{i}"]
    for i in range(1, k + 1):
        ordered += [f"{right_prefix}_pid_{i}", f"{right_prefix}_score_{i}"]
    ordered += [f"top{k}_overlap_cnt"]

    rest = [c for c in out.columns if c not in ordered]
    out = out[ordered + rest]

    return out


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
    ensure_required_columns(recs, RECS_REQUIRED_COLS)

    recs = normalize_create_date(recs)
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
        baseline_top, pg_top, TOPK,
        "baseline_rank", "pg_rank",
        "base", "pg",
        left_score_col="q_api",
        right_score_col="pg_score",
    )

    sum_base_kurekj = summary_wide_pair(
        baseline_top, kurekj_top, TOPK,
        "baseline_rank", "kurekj_rank",
        "base", "kurekj",
        left_score_col="q_api",
        right_score_col="kurekj_score",
    )

    sum_pg_kurekj = summary_wide_pair(
        pg_top, kurekj_top, TOPK,
        "pg_rank", "kurekj_rank",
        "pg", "kurekj",
        left_score_col="pg_score",
        right_score_col="kurekj_score",
    )

    out_dir = Path(__file__).resolve().parent / f"Output_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{OUT_XLSX_BASENAME}_{datestamp}.xlsx"

    engine = choose_excel_engine()

    with pd.ExcelWriter(out_path, engine=engine) as xw:
        baseline_top.to_excel(excel_writer=xw, sheet_name="baseline_topk", index=False)
        pg_top.to_excel(excel_writer=xw, sheet_name="pg_business_topk", index=False)
        kurekj_top.to_excel(excel_writer=xw, sheet_name="kurekj_business", index=False)

        sum_base_pg.to_excel(excel_writer=xw, sheet_name="sum_base_vs_pg", index=False)
        sum_base_kurekj.to_excel(excel_writer=xw, sheet_name="sum_base_vs_kurekj", index=False)
        sum_pg_kurekj.to_excel(excel_writer=xw, sheet_name="sum_pg_vs_kurekj", index=False)

        pd.DataFrame([asdict(params)]).to_excel(
            excel_writer=xw,
            sheet_name="params",
            index=False,
        )

    print(f"OK: saved {out_path.resolve()} (engine={engine})")


if __name__ == "__main__":
    main()
