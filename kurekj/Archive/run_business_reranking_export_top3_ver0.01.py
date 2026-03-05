from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================

# Skąd czytać rekomendacje (parquet / xlsx / csv). Każdy wiersz = 1 rekomendowana nieruchomość.
RECS_PATH = Path("recommendations_output.parquet")

# Opcjonalnie: snapshot daily_snapshot.offers_billinggroup (parquet/xlsx/csv).
# Jeśli nie masz albo nie chcesz joinować — zostaw None.
BILLINGGROUP_PATH: Optional[Path] = None

# Jak joinować billinggroup do rekomendacji:
# - "offer_id" ma sens, bo offer_id = identyfikator inwestycji :contentReference[oaicite:3]{index=3}
BILLING_JOIN_KEY = "offer_id"

# Gdzie zapisać Excel
OUT_XLSX_PATH = Path("compare_top3.xlsx")

# Ile top wyników na request_id
TOPK = 3

# Pacing p(h) w [-1, 1] (jeśli nie używasz – zostaw 0.0)
P_H = 0.0

# Jeśli u Ciebie api_model_score == -1 to “brak / placeholder”, ustaw sentinel na -1.0.
# Jeśli -1 jest LEGALNYM wynikiem modelu, ustaw None i nie będzie traktowany jako missing.
API_MODEL_SCORE_MISSING_SENTINEL: Optional[float] = -1.0

# Co robić gdy api_model_score jest “missing”:
# - "zero"       => score=0.0 (bez przecieku baseline)
# - "final_score" => score=final_score (UWAGA: to jest przeciek baseline do q)
MISSING_Q_STRATEGY = "zero"

# Domyślne wartości, jeśli nie ma billinggroup
DEFAULT_CONTRACT_TYPE = "flat"   # "flat" albo "cpl"
DEFAULT_CAP_RATIO = 0.0

# Parametry rerankingu (Twoje hyperparametry)
# Zgodnie z Twoim modułem: gamma, mu, nu, rho, delta, lambda_ :contentReference[oaicite:4]{index=4}
RERANK_PARAMS_DICT = dict(
    gamma=1.2,
    mu=0.5,
    nu=0.8,
    rho=0.3,
    delta=1.0,
    lambda_=0.5,
    # opcjonalnie: normalize_q, contract_weights, w_inv, w_dev, w_city itd.
    # normalize_q=True jest domyślne w Twoim module :contentReference[oaicite:5]{index=5}
)

# Skąd importować business_reranking:
# - "repo"  => src.v2.api.services.business_reranking (jak w example_usage) :contentReference[oaicite:6]{index=6}
# - "local" => business_reranking.py obok tego pliku
IMPORT_STYLE = "repo"

# =========================
# KONIEC CONFIG
# =========================

if IMPORT_STYLE == "repo":
    from src.v2.api.services.business_reranking import (  # type: ignore
        Candidate,
        RerankParams,
        greedy_rerank,
        business_score,
    )
elif IMPORT_STYLE == "local":
    from business_reranking import (  # type: ignore
        Candidate,
        RerankParams,
        greedy_rerank,
        business_score,
    )
else:
    raise ValueError("IMPORT_STYLE must be either 'repo' or 'local'.")


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
    raise ValueError(f"Unsupported input format: {suffix} (expected parquet/xlsx/csv)")


def ensure_required_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def normalize_create_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel nie lubi tz-aware datetime, więc:
    - parsujemy create_date
    - jeśli tz-aware -> konwertujemy do Europe/Warsaw i zdejmujemy tz
    - dodajemy create_day (date) do joinu z billinggroup
    """
    out = df.copy()
    out["create_date"] = pd.to_datetime(out["create_date"], errors="raise")

    if pd.api.types.is_datetime64tz_dtype(out["create_date"]):
        out["create_date"] = (
            out["create_date"]
            .dt.tz_convert("Europe/Warsaw")
            .dt.tz_localize(None)
        )

    out["create_day"] = out["create_date"].dt.date
    return out


def merge_billinggroup(recs: pd.DataFrame, billing: pd.DataFrame) -> pd.DataFrame:
    """
    Dołącza billinggroup:
      recs.create_day == billing.day (po dacie) + key (offer_id albo property_id)

    W opisie tabeli jest kolumna day :contentReference[oaicite:7]{index=7}
    """
    if "day" not in billing.columns:
        raise ValueError("billinggroup table must contain 'day' column")

    if BILLING_JOIN_KEY not in recs.columns:
        raise ValueError(f"recs does not have join key column: {BILLING_JOIN_KEY}")
    if BILLING_JOIN_KEY not in billing.columns:
        raise ValueError(f"billinggroup does not have join key column: {BILLING_JOIN_KEY}")

    out = recs.copy()
    b = billing.copy()
    b["day"] = pd.to_datetime(b["day"], errors="raise").dt.date

    # 1) join po tej samej dacie
    out = out.merge(
        b,
        how="left",
        left_on=[BILLING_JOIN_KEY, "create_day"],
        right_on=[BILLING_JOIN_KEY, "day"],
        suffixes=("", "_bg"),
    )

    # 2) fallback: jeśli nie ma snapshotu w danym dniu, weź ostatni dostępny per key
    b_latest = (
        b.sort_values("day")
        .drop_duplicates(subset=[BILLING_JOIN_KEY], keep="last")
        .drop(columns=["day"], errors="ignore")
    )
    b_latest = b_latest.add_suffix("_latest").rename(columns={f"{BILLING_JOIN_KEY}_latest": BILLING_JOIN_KEY})

    out = out.merge(b_latest, how="left", on=BILLING_JOIN_KEY)

    # merge “najważniejszych” pól z *_latest, jeśli puste
    for c in ["settlement_type", "estimated_perc_realization"]:
        if c in out.columns and f"{c}_latest" in out.columns:
            out[c] = out[c].combine_first(out[f"{c}_latest"])

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buduje kolumny pod Candidate:
      score (=q), r, g, m, v, contract_type, cap_ratio, inv_id

    Źródła:
      - api_model_score: model przed regułami biznesowymi :contentReference[oaicite:8]{index=8}
      - final_score, gap_score, lead_price_score: składniki bieżącego rerankingu :contentReference[oaicite:9]{index=9}
      - settlement_type, estimated_perc_realization: billinggroup :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}
    """
    out = df.copy()

    # numeric
    for c in ["api_model_score", "final_score", "gap_score", "lead_price_score"]:
        out[c] = pd.to_numeric(out[c], errors="raise")

    # q / score
    q = out["api_model_score"].copy()
    if API_MODEL_SCORE_MISSING_SENTINEL is not None:
        missing_mask = q == API_MODEL_SCORE_MISSING_SENTINEL
        if MISSING_Q_STRATEGY == "zero":
            q = q.mask(missing_mask, 0.0)
        elif MISSING_Q_STRATEGY == "final_score":
            q = q.mask(missing_mask, out["final_score"])
        else:
            raise ValueError("MISSING_Q_STRATEGY must be 'zero' or 'final_score'.")

    out["score"] = q

    # r, g
    out["r"] = out["lead_price_score"].clip(0.0, 1.0)
    out["g"] = out["gap_score"].clip(0.0, 1.0)

    # fairness / VIP (na razie domyślnie)
    out["m"] = 0.0
    out["v"] = 0

    # diversity po inwestycji: offer_id opisany jako identyfikator inwestycji :contentReference[oaicite:12]{index=12}
    out["inv_id"] = out["offer_id"]

    # contract_type + cap_ratio (jeśli jest billinggroup, to bierzemy z niego)
    if "settlement_type" in out.columns:
        # settlement_type: CPL=0, subskrypcja=1 :contentReference[oaicite:13]{index=13}
        out["contract_type"] = out["settlement_type"].map({0: "cpl", 1: "flat"}).fillna(DEFAULT_CONTRACT_TYPE)
    else:
        out["contract_type"] = DEFAULT_CONTRACT_TYPE

    if "estimated_perc_realization" in out.columns:
        # estimated_perc_realization: prognoza / limit (cap_ratio w Twoim module jest używany w cap_penalty) 게
        out["cap_ratio"] = pd.to_numeric(out["estimated_perc_realization"], errors="raise").fillna(DEFAULT_CAP_RATIO)
        out["cap_ratio"] = out["cap_ratio"].clip(lower=0.0)
    else:
        out["cap_ratio"] = DEFAULT_CAP_RATIO

    return out


def baseline_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Baseline = to co macie obecnie: sort per request_id po final_score DESC.
    final_score jest wynikiem po rerankingu :contentReference[oaicite:14]{index=14}
    """
    out = df.copy()
    out = out.sort_values(["request_id", "final_score", "property_id"], ascending=[True, False, True], kind="mergesort")
    out["baseline_rank_final"] = out.groupby("request_id").cumcount() + 1
    return out[out["baseline_rank_final"] <= k].copy()


def business_topk(df: pd.DataFrame, params: RerankParams, k: int, p_h: float) -> pd.DataFrame:
    """
    Business rerank per request_id: greedy_rerank + business_score.
    To jest zgodne z Twoją implementacją:
      - greedy wybór najlepszych kandydatów :contentReference[oaicite:15]{index=15}
      - score liczony wg business_score (Eq. 1) :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17}
    """
    parts: list[pd.DataFrame] = []

    for request_id, g in df.groupby("request_id", sort=False):
        g2 = g.copy()

        # Budujemy Candidate listę
        candidates = []
        for _, row in g2.iterrows():
            candidates.append(
                Candidate(
                    property_id=int(row["property_id"]),
                    q=float(row["score"]),
                    r=float(row["r"]),
                    g=float(row["g"]),
                    m=float(row["m"]),
                    v=int(row["v"]),
                    contract_type=str(row["contract_type"]),
                    cap_ratio=float(row["cap_ratio"]),
                    inv_id=int(row["inv_id"]) if pd.notna(row["inv_id"]) else None,
                )
            )

        ranked = greedy_rerank(candidates, params, k=k, p_h=p_h)

        # policz business_score w kolejności greedy (selected rośnie)
        selected: list[Candidate] = []
        rows = []
        for rank, cand in enumerate(ranked, start=1):
            s = business_score(cand, params, p_h=p_h, selected=selected)
            selected.append(cand)
            rows.append((cand.property_id, rank, s))

        score_df = pd.DataFrame(rows, columns=["property_id", "business_rank", "business_score"])

        out = g2.merge(score_df, on="property_id", how="inner")
        out["request_size"] = len(g2)

        # zachowaj kolejność greedy
        out = out.sort_values(["business_rank"], ascending=True, kind="mergesort")
        parts.append(out)

    return pd.concat(parts, ignore_index=True) if parts else df.head(0)


def summary_wide(baseline: pd.DataFrame, business: pd.DataFrame, k: int) -> pd.DataFrame:
    keys = ["request_id", "uuid", "create_date"]
    meta = baseline[keys].drop_duplicates(subset=["request_id"], keep="first").set_index("request_id")

    base_w = baseline.pivot(index="request_id", columns="baseline_rank_final", values="property_id")
    base_w.columns = [f"baseline_property_id_{int(c)}" for c in base_w.columns]

    biz_w = business.pivot(index="request_id", columns="business_rank", values="property_id")
    biz_w.columns = [f"business_property_id_{int(c)}" for c in biz_w.columns]

    out = meta.join(base_w, how="left").join(biz_w, how="left").reset_index()

    # overlap metryka
    def overlap_cnt(row: pd.Series) -> int:
        b1 = {int(row[f"baseline_property_id_{i}"]) for i in range(1, k + 1) if pd.notna(row.get(f"baseline_property_id_{i}"))}
        b2 = {int(row[f"business_property_id_{i}"]) for i in range(1, k + 1) if pd.notna(row.get(f"business_property_id_{i}"))}
        return len(b1 & b2)

    out["topk_overlap_cnt"] = out.apply(overlap_cnt, axis=1)
    return out


def main() -> None:
    if not RECS_PATH.exists():
        raise FileNotFoundError(f"RECS_PATH does not exist: {RECS_PATH.resolve()}")

    recs = read_table(RECS_PATH)
    ensure_required_columns(recs, RECS_REQUIRED_COLS, label="recs")
    recs = normalize_create_date(recs)

    if BILLINGGROUP_PATH is not None:
        if not BILLINGGROUP_PATH.exists():
            raise FileNotFoundError(f"BILLINGGROUP_PATH does not exist: {BILLINGGROUP_PATH.resolve()}")
        billing = read_table(BILLINGGROUP_PATH)
        recs = merge_billinggroup(recs, billing)

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
