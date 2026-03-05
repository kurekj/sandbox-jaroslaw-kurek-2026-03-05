from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # <-- PROGRESS BAR

from src.v2.api.services.business_reranking import (  # type: ignore
    Candidate,
    RerankParams,
    greedy_rerank,
    business_score,
)

# =========================
# CONFIG (EDYTUJ TYLKO TO)
# =========================

RECS_PATH = Path("recommendations_output.parquet")
OUT_XLSX_PATH = Path("compare_top3.xlsx")

TOPK = 3
P_H = 0.0

API_MODEL_SCORE_MISSING_SENTINEL = -1.0  # albo None

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
# KONIEC CONFIG
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

    if pd.api.types.is_datetime64tz_dtype(dt):
        dt = dt.dt.tz_convert("Europe/Warsaw").dt.tz_localize(None)

    out["create_date"] = dt
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.reset_index(drop=True)
    out["row_id"] = out.index.astype(int)

    for c in ["api_model_score", "final_score", "gap_score", "lead_price_score"]:
        out[c] = pd.to_numeric(out[c], errors="raise")

    out["score"] = out["api_model_score"].fillna(0.0)

    if API_MODEL_SCORE_MISSING_SENTINEL is not None:
        out.loc[out["score"] == API_MODEL_SCORE_MISSING_SENTINEL, "score"] = 0.0

    out["score"] = out["score"].clip(-1.0, 1.0)

    out["r"] = out["lead_price_score"].fillna(0.0).clip(0.0, 1.0)
    out["g"] = out["gap_score"].fillna(0.0).clip(0.0, 1.0)

    out["m"] = float(DEFAULT_M)
    out["v"] = int(DEFAULT_V)
    out["contract_type"] = DEFAULT_CONTRACT_TYPE
    out["cap_ratio"] = DEFAULT_CAP_RATIO

    out["inv_id"] = out["offer_id"]

    return out


def baseline_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = df.sort_values(
        ["request_id", "final_score", "property_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()

    out["baseline_rank_final"] = out.groupby("request_id").cumcount() + 1
    return out[out["baseline_rank_final"] <= k].copy()


def business_topk(df: pd.DataFrame, params: RerankParams, k: int, p_h: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    grouped = df.groupby("request_id", sort=False)
    total = grouped.ngroups

    for request_id, g in tqdm(
        grouped,
        total=total,
        desc="Business reranking per request_id",
        unit="request",
    ):
        g2 = g.copy()

        candidates: list[Candidate] = []
        for _, row in g2.iterrows():
            candidates.append(
                Candidate(
                    property_id=int(row["property_id"]),
                    q=float(row["score"]),
                    r=float(row["r"]),
                    g=float(row["g"]),
                    m=float(row["m"]),
                    v=int(row["v"]),
                    contract_type=row["contract_type"],
                    cap_ratio=float(row["cap_ratio"]),
                    inv_id=int(row["inv_id"]) if pd.notna(row["inv_id"]) else None,
                    extra={"row_id": int(row["row_id"])},
                )
            )

        ranked = greedy_rerank(candidates, params, k=k, p_h=p_h)

        selected: list[Candidate] = []
        rows = []

        for rank, cand in enumerate(ranked, start=1):
            score = business_score(cand, params, p_h=p_h, selected=selected)
            selected.append(cand)

            rows.append(
                {
                    "row_id": cand.extra["row_id"],
                    "business_rank": rank,
                    "business_score": score,
                }
            )

        score_df = pd.DataFrame(rows)
        out = g2.merge(score_df, on="row_id", how="inner")
        out["request_size"] = len(g2)

        out = out.sort_values("business_rank", kind="mergesort")
        parts.append(out)

    return pd.concat(parts, ignore_index=True)


def summary_wide(baseline: pd.DataFrame, business: pd.DataFrame, k: int) -> pd.DataFrame:
    meta = (
        baseline[["request_id", "uuid", "create_date"]]
        .drop_duplicates("request_id")
        .set_index("request_id")
    )

    base_w = baseline.pivot(index="request_id", columns="baseline_rank_final", values="property_id")
    base_w.columns = [f"baseline_property_id_{c}" for c in base_w.columns]

    biz_w = business.pivot(index="request_id", columns="business_rank", values="property_id")
    biz_w.columns = [f"business_property_id_{c}" for c in biz_w.columns]

    out = meta.join(base_w).join(biz_w).reset_index()

    def overlap_cnt(row: pd.Series) -> int:
        b1 = {row.get(f"baseline_property_id_{i}") for i in range(1, k + 1)}
        b2 = {row.get(f"business_property_id_{i}") for i in range(1, k + 1)}
        return len({x for x in b1 if pd.notna(x)} & {x for x in b2 if pd.notna(x)})

    out["topk_overlap_cnt"] = out.apply(overlap_cnt, axis=1)
    return out


def main() -> None:
    if not RECS_PATH.exists():
        raise FileNotFoundError(RECS_PATH)

    recs = read_table(RECS_PATH)
    ensure_required_columns(recs, RECS_REQUIRED_COLS)

    recs = normalize_create_date(recs)
    recs = build_features(recs)

    params = RerankParams(**RERANK_PARAMS_DICT)

    base_top = baseline_topk(recs, TOPK)
    biz_top = business_topk(recs, params, TOPK, P_H)
    summ = summary_wide(base_top, biz_top, TOPK)

    OUT_XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX_PATH, engine="openpyxl") as xw:
        base_top.to_excel(xw, sheet_name="baseline_topk_long", index=False)
        biz_top.to_excel(xw, sheet_name="business_topk_long", index=False)
        summ.to_excel(xw, sheet_name="summary_wide", index=False)
        pd.DataFrame([asdict(params)]).to_excel(xw, sheet_name="params", index=False)

    print(f"OK: saved {OUT_XLSX_PATH.resolve()}")
    print(f"baseline rows={len(base_top)}, business rows={len(biz_top)}, summary rows={len(summ)}")


if __name__ == "__main__":
    main()
