# business_reranking5.py
"""
Business re‑ranking module for recommendation scores.

This module implements a multi‑parameter business re‑ranking layer that
combines base model scores with a variety of business objectives such as
profitability, plan‑gap, fairness, VIP status and diversity.

    S(u,i) = q(u,i)**γ *
             (1 + β_{c(i)} r(i) + η_{c(i)} g(i) + μ m(i) + ν v(i)) *
             (1 + ρ p(h))
           – δ C(i)
           – λ D(i, L)

where q(u,i) is the base relevance score, r(i) is profitability,
g(i) is the plan gap, m(i) is a fairness boost, v(i) flags a VIP offer,
c(i) denotes the contract type, p(h) adjusts pacing by hour,
C(i) applies exposure caps and D(i,L) measures similarity to already
selected items.

The functions defined here are intentionally decoupled from any specific
database or web framework. They operate on simple data structures to
facilitate unit testing and reuse in both synchronous and asynchronous
contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Candidate:
    """
    Represents a single candidate offer for a user.

    Attributes:
        property_id: Unique identifier of the property.
        q: Base relevance score in the range [-1, 1] or [0, 1].  If values are
            supplied in [-1, 1], they will be mapped to [0, 1] when computing
            the business score.
        r: Profitability of the offer normalised to [0, 1].
        g: Plan‑gap score normalised to [0, 1]; higher values mean the plan
            is further from being met.
        m: Fairness boost, typically in the range [-1, 1]; positive values
            boost smaller developers, negative values down‑weight larger ones.
        v: VIP flag; use 1 for premium leads, otherwise 0.
        contract_type: Either "flat" (subscription) or "cpl" (cost per lead).
        cap_ratio: Current utilisation divided by the cap limit for the offer;
            values in [0, ∞).  This is used to compute the cap penalty.
        inv_id: Optional identifier of the investment.  Used for diversity.
        dev_id: Optional identifier of the developer.  Used for diversity.
        city_id: Optional identifier of the city.  Used for diversity.
        extra: A dictionary to store any additional per‑offer metadata.
    """

    property_id: int
    q: float
    r: float
    g: float
    m: float
    v: int
    contract_type: str
    cap_ratio: float
    inv_id: Optional[int] = None
    dev_id: Optional[int] = None
    city_id: Optional[int] = None
    extra: dict = field(default_factory=dict)


@dataclass
class RerankParams:
    """
    Hyperparameters controlling the business re‑ranking behaviour.

    Attributes:
        gamma: Exponent applied to the base score q.
        mu: Weight applied to the fairness boost m(i).
        nu: Weight applied to the VIP flag v(i).
        rho: Weight applied to the pacing factor p(h).
        delta: Weight applied to the cap penalty C(i).
        lambda_: Weight applied to the diversity penalty D(i, L).
        contract_weights: Mapping of contract types to (β, η) tuples.
        w_inv: Similarity weight when two offers belong to the same investment.
        w_dev: Similarity weight when two offers share the same developer.
        w_city: Similarity weight when two offers are located in the same city.
        normalize_q: If True, values of q in [-1, 1] will be linearly mapped
            to [0, 1] before applying gamma. If False, q is used as is.
    """

    gamma: float = 1.0
    mu: float = 0.0
    nu: float = 0.0
    rho: float = 0.0
    delta: float = 0.0
    lambda_: float = 0.0
    contract_weights: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"flat": (0.2, 0.7), "cpl": (0.6, 0.3)}
    )
    w_inv: float = 1.0
    w_dev: float = 0.5
    w_city: float = 0.25
    normalize_q: bool = True


# -------------------------
# Small safe casting helpers
# -------------------------
def _to_float(x: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float with NaN/None fallback."""
    try:
        if x is None:
            return default
        v = float(x)
        # NaN check: NaN != NaN
        if v != v:
            return default
        return v
    except Exception:
        return default


def _to_int_or_none(x: Any) -> Optional[int]:
    """Best-effort conversion to int; returns None for NaN/None/invalid."""
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return int(v)
    except Exception:
        return None


def _to_int(x: Any, default: int = 0) -> int:
    """Best-effort conversion to int with NaN/None fallback."""
    v = _to_int_or_none(x)
    return default if v is None else v


def _normalize_q(q: float) -> float:
    """
    Convert a base score from [-1, 1] to [0, 1] if necessary.

    The business scoring formula expects q ∈ [0, 1]. In the underlying
    recommendation system similarity scores may lie in [-1, 1] (cosine sim).
    To reconcile ranges, q is mapped to [0, 1] via (q + 1) / 2.

    Args:
        q: Original similarity score in [-1, 1] or [0, 1].

    Returns:
        A value in the range [0, 1].
    """
    if q < 0.0:
        return (q + 1.0) / 2.0
    if q > 1.0:
        return 1.0
    return q


def cap_penalty(cap_ratio: float) -> float:
    """
    Compute the cap penalty C(i) given the utilisation ratio.

    The penalty is zero until 70% of the cap is used. Between 70% and 100%,
    the penalty increases linearly from 0 to 1. Above the cap the penalty
    saturates at 1.

    Args:
        cap_ratio: Utilisation divided by cap limit, in [0, ∞).

    Returns:
        A value in [0, 1] representing the severity of the cap penalty.
    """
    cr = _to_float(cap_ratio, default=0.0)
    if cr < 0.0:
        cr = 0.0

    if cr < 0.7:
        return 0.0
    if cr < 1.0:
        return (cr - 0.7) / 0.3
    return 1.0


def diversity_penalty(
    candidate: Candidate,
    selected: List[Candidate],
    *,
    w_inv: float,
    w_dev: float,
    w_city: float,
) -> float:
    """
    Compute the diversification penalty D(i,L) for a candidate.

    The penalty is defined as the maximum similarity between the candidate
    and any offer already selected, where similarity is a weighted max of
    binary indicators on investment, developer and city identifiers.

    Args:
        candidate: The candidate being evaluated.
        selected: The list of already selected offers.
        w_inv: Weight for identical investments.
        w_dev: Weight for identical developers.
        w_city: Weight for identical cities.

    Returns:
        The maximum similarity score between candidate and items in `selected`.
    """
    max_sim = 0.0
    for sel in selected:
        sim = 0.0
        if candidate.inv_id is not None and sel.inv_id is not None and candidate.inv_id == sel.inv_id:
            sim = max(sim, w_inv)
        if candidate.dev_id is not None and sel.dev_id is not None and candidate.dev_id == sel.dev_id:
            sim = max(sim, w_dev)
        if candidate.city_id is not None and sel.city_id is not None and candidate.city_id == sel.city_id:
            sim = max(sim, w_city)
        if sim > max_sim:
            max_sim = sim
            if max_sim >= w_inv:
                break
    return max_sim


def business_score(candidate: Candidate, params: RerankParams, p_h: float, selected: List[Candidate]) -> float:
    """
    Compute the business score S(u,i) for a candidate.

    IMPORTANT (robustness):
      - r and g are clamped to [0,1]
      - m is clamped to [-1,1]
      - v is binarized to {0,1} (non-zero -> 1)
      - q is clamped to [0,1] (after optional normalization)

    Args:
        candidate: Candidate offer being scored.
        params: Hyperparameters controlling the scoring.
        p_h: Pacing adjustment for the current hour h, in [−1, 1].
        selected: The list of offers that have already been chosen (for diversity).

    Returns:
        The computed business score for the candidate.
    """
    # --- Base score (q) ---
    q_raw = _to_float(candidate.q, default=0.0)
    q_norm = _normalize_q(q_raw) if params.normalize_q else q_raw
    if q_norm < 0.0:
        q_norm = 0.0
    if q_norm > 1.0:
        q_norm = 1.0
    base_component = q_norm ** params.gamma

    # --- Contract weights β and η ---
    beta, eta = params.contract_weights.get(str(candidate.contract_type).lower(), (0.0, 0.0))

    # --- Feature sanitization ---
    r = _to_float(candidate.r, default=0.0)
    if r < 0.0:
        r = 0.0
    if r > 1.0:
        r = 1.0

    g = _to_float(candidate.g, default=0.0)
    if g < 0.0:
        g = 0.0
    if g > 1.0:
        g = 1.0

    m = _to_float(candidate.m, default=0.0)
    if m < -1.0:
        m = -1.0
    if m > 1.0:
        m = 1.0

    v = 1 if _to_int(candidate.v, default=0) != 0 else 0

    # --- Linear combination ---
    business_factors = 1.0 + beta * r + eta * g + params.mu * m + params.nu * v

    # --- Pacing ---
    p_h_clip = max(min(_to_float(p_h, default=0.0), 1.0), -1.0)
    pacing_term = 1.0 + params.rho * p_h_clip

    # --- Cap penalty ---
    cap_pen = params.delta * cap_penalty(candidate.cap_ratio)

    # --- Diversity penalty ---
    div_pen = params.lambda_ * diversity_penalty(
        candidate,
        selected,
        w_inv=params.w_inv,
        w_dev=params.w_dev,
        w_city=params.w_city,
    )

    return base_component * business_factors * pacing_term - cap_pen - div_pen


def greedy_rerank(
    candidates: List[Candidate],
    params: RerankParams,
    *,
    k: Optional[int] = None,
    p_h: float = 0.0,
) -> List[Candidate]:
    """
    Rank a list of candidate offers using a greedy selection algorithm.

    At each iteration the candidate with the highest business score is chosen
    and removed from the pool. The process continues until either `k`
    candidates have been selected or the pool is exhausted.

    Args:
        candidates: The list of candidates to rank.
        params: Hyperparameters controlling the scoring and penalties.
        k: Maximum number of candidates to select. If None, all candidates will be ranked.
        p_h: Hourly pacing adjustment in [−1, 1].

    Returns:
        A new list of candidates in the order they should be presented.
    """
    remaining = candidates.copy()
    selected: List[Candidate] = []
    limit = len(remaining) if k is None else max(min(k, len(remaining)), 0)

    while remaining and len(selected) < limit:
        best_candidate: Optional[Candidate] = None
        best_score: float = float("-inf")

        for cand in remaining:
            score = business_score(cand, params, p_h, selected)
            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def rerank_dataframe(
    df: "pandas.DataFrame",
    params: RerankParams,
    *,
    k: Optional[int] = None,
    p_h: float = 0.0,
    id_col: str = "property_id",
    score_col: str = "score",
    r_col: str = "r",
    g_col: str = "g",
    m_col: str = "m",
    v_col: str = "v",
    contract_col: str = "contract_type",
    cap_ratio_col: str = "cap_ratio",
    inv_id_col: Optional[str] = None,
    dev_id_col: Optional[str] = None,
    city_id_col: Optional[str] = None,
) -> "pandas.DataFrame":
    """
    Apply the greedy business re‑ranking to a pandas DataFrame.

    Returns a new DataFrame sorted by the re‑ranked order and adds
    a `business_score` column.
    """
    import pandas as pd  # Lazy import

    candidates: List[Candidate] = []
    cols = set(df.columns)

    for _, row in df.iterrows():
        inv_id = _to_int_or_none(row[inv_id_col]) if inv_id_col and inv_id_col in cols else None
        dev_id = _to_int_or_none(row[dev_id_col]) if dev_id_col and dev_id_col in cols else None
        city_id = _to_int_or_none(row[city_id_col]) if city_id_col and city_id_col in cols else None

        candidates.append(
            Candidate(
                property_id=_to_int(row[id_col]),
                q=_to_float(row[score_col]),
                r=_to_float(row[r_col]),
                g=_to_float(row[g_col]),
                m=_to_float(row[m_col]),
                v=_to_int(row[v_col]),
                contract_type=str(row[contract_col]).lower(),
                cap_ratio=_to_float(row[cap_ratio_col]),
                inv_id=inv_id,
                dev_id=dev_id,
                city_id=city_id,
            )
        )

    ranked = greedy_rerank(candidates, params, k=k, p_h=p_h)

    score_map: Dict[int, float] = {}
    selected: List[Candidate] = []
    for cand in ranked:
        s = business_score(cand, params, p_h, selected)
        score_map[cand.property_id] = s
        selected.append(cand)

    out = df.copy()
    out["business_score"] = out[id_col].map(score_map)

    out = out.sort_values("business_score", ascending=False)
    if k is not None:
        out = out.head(k)
    return out
