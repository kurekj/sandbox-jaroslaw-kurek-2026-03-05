"""
Business re‑ranking module for recommendation scores.

This module implements a multi‑parameter business re‑ranking layer that
combines base model scores with a variety of business objectives such as
profitability, plan‑gap, fairness, VIP status and diversity.  The
implementation follows the formulation described in the accompanying report:

    S(u,i) = q(u,i)**γ *
             (1 + β_{c(i)} r(i) + η_{c(i)} g(i) + μ m(i) + ν v(i)) *
             (1 + ρ p(h))
           – δ C(i)
           – λ D(i, L)

where q(u,i) is the base relevance score, r(i) is profitability,
g(i) is the plan gap, m(i) is a fairness boost, v(i) flags a VIP offer,
c(i) denotes the contract type, p(h) adjusts pacing by hour,
C(i) applies exposure caps and D(i,L) measures similarity to already
selected items.  See the report for a detailed description of each term.

Typical usage:

>>> candidates = [
...     Candidate(property_id=1, q=0.9, r=0.2, g=0.1, m=-0.2, v=0, contract_type="flat", cap_ratio=0.5),
...     Candidate(property_id=2, q=0.85, r=0.5, g=0.6, m=0.3, v=0, contract_type="flat", cap_ratio=0.4),
... ]
>>> params = RerankParams()
>>> ranked = greedy_rerank(candidates, params, k=2, p_h=0.0)
>>> [c.property_id for c in ranked]
[2, 1]

The functions defined here are intentionally decoupled from any specific
database or web framework.  They operate on simple data structures to
facilitate unit testing and reuse in both synchronous and asynchronous
contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
            values in [0, 1].  This is used to compute the cap penalty.
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
        gamma: Exponent applied to the base score q.  Values >1 sharpen
            differences in high scores; values <1 flatten them.
        mu: Weight applied to the fairness boost m(i).
        nu: Weight applied to the VIP flag v(i).
        rho: Weight applied to the pacing factor p(h).
        delta: Weight applied to the cap penalty C(i).
        lambda_: Weight applied to the diversity penalty D(i,L).
        contract_weights: Mapping of contract types to (β, η) tuples.  β
            controls the importance of profitability r(i) and η controls
            the importance of plan‑gap g(i).  Defaults to values suggested
            in the report.
        w_inv: Similarity weight when two offers belong to the same investment.
        w_dev: Similarity weight when two offers share the same developer.
        w_city: Similarity weight when two offers are located in the same city.
        normalize_q: If True, values of q in [-1, 1] will be linearly mapped
            to [0, 1] before applying gamma.  If False, q is used as is.
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


def _normalize_q(q: float) -> float:
    """
    Convert a base score from [-1, 1] to [0, 1] if necessary.

    The business scoring formula expects q ∈ [0, 1].  In the underlying
    recommendation system the similarity scores may lie in [-1, 1] because
    they are computed as cosine similarities between unit‑norm embeddings.
    To reconcile the ranges, q is mapped to [0, 1] via (q + 1) / 2.

    Args:
        q: Original similarity score in [-1, 1] or [0, 1].

    Returns:
        A value in the range [0, 1].
    """
    if q < 0.0:
        # Map negative values to [0, 0.5)
        return (q + 1.0) / 2.0
    if q > 1.0:
        # Guard against out‑of‑range values
        return 1.0
    return q


def cap_penalty(cap_ratio: float) -> float:
    """
    Compute the cap penalty C(i) given the utilisation ratio.

    The penalty is zero until 70 % of the cap is used.  Between 70 % and
    100 %, the penalty increases linearly from 0 to 1.  Above the cap the
    penalty saturates at 1.

    Args:
        cap_ratio: Utilisation divided by cap limit, in [0, ∞).

    Returns:
        A value in [0, 1] representing the severity of the cap penalty.
    """
    if cap_ratio < 0.7:
        return 0.0
    if cap_ratio < 1.0:
        return (cap_ratio - 0.7) / 0.3
    # At or above the limit the penalty saturates at 1.0
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
    and any offer already selected, where similarity is a weighted sum of
    binary indicators on investment, developer and city identifiers.  The
    resulting value lies in [0, 1].  Higher values indicate that the
    candidate is very similar to something already in the list and should
    therefore be down‑weighted by λ.

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
        # Investment similarity dominates developer and city
        if candidate.inv_id is not None and sel.inv_id is not None and candidate.inv_id == sel.inv_id:
            sim = max(sim, w_inv)
        if candidate.dev_id is not None and sel.dev_id is not None and candidate.dev_id == sel.dev_id:
            sim = max(sim, w_dev)
        if candidate.city_id is not None and sel.city_id is not None and candidate.city_id == sel.city_id:
            sim = max(sim, w_city)
        if sim > max_sim:
            max_sim = sim
            # Early exit: maximum possible similarity reached
            if max_sim >= w_inv:
                break
    return max_sim


def business_score(candidate: Candidate, params: RerankParams, p_h: float, selected: List[Candidate]) -> float:
    """
    Compute the business score S(u,i) for a candidate.

    This function implements Equation (1) from the report.  The base score
    `q` is optionally normalised to [0, 1], then raised to the power γ.
    Contract‑specific weights β and η are looked up based on the candidate's
    contract type.  The profitability r(i), plan gap g(i), fairness boost
    m(i) and VIP flag v(i) are combined linearly.  Pacing is applied via
    the factor 1 + ρ p(h).  Cap and diversity penalties are subtracted.

    Args:
        candidate: Candidate offer being scored.
        params: Hyperparameters controlling the scoring.
        p_h: Pacing adjustment for the current hour h, in [−1, 1].
        selected: The list of offers that have already been chosen.  Used
            solely for computing the diversity penalty.

    Returns:
        The computed business score for the candidate.
    """
    # Normalise and exponentiate the base score
    q = _normalize_q(candidate.q) if params.normalize_q else candidate.q
    base_component = q**params.gamma

    # Look up β and η for the contract type; default to zeros if unknown
    beta_eta = params.contract_weights.get(candidate.contract_type.lower(), (0.0, 0.0))
    beta, eta = beta_eta

    # Linear combination of profitability, plan gap, fairness and VIP
    business_factors = 1.0 + beta * candidate.r + eta * candidate.g + params.mu * candidate.m + params.nu * candidate.v

    # Pacing adjustment; clipping ensures the term stays within [1−ρ, 1+ρ]
    pacing_term = 1.0 + params.rho * max(min(p_h, 1.0), -1.0)

    # Cap penalty
    cap_pen = params.delta * cap_penalty(candidate.cap_ratio)

    # Diversity penalty
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
    and removed from the pool.  The process continues until either `k`
    candidates have been selected or the pool is exhausted.  Diversity is
    enforced by computing the similarity of each unselected candidate to the
    current list and penalising those that are too similar.

    Args:
        candidates: The list of candidates to rank.
        params: Hyperparameters controlling the scoring and penalties.
        k: Maximum number of candidates to select.  If None, all candidates
            will be ranked.
        p_h: Hourly pacing adjustment in [−1, 1].  Positive values boost
            scores when exposure is below target; negative values slow
            down when exposure exceeds target.

    Returns:
        A new list of candidates in the order they should be presented.
    """
    remaining = candidates.copy()
    selected: List[Candidate] = []
    limit = len(remaining) if k is None else max(min(k, len(remaining)), 0)

    while remaining and len(selected) < limit:
        best_candidate: Optional[Candidate] = None
        best_score: float = float("-inf")
        # Evaluate each candidate given the current selection
        for cand in remaining:
            score = business_score(cand, params, p_h, selected)
            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_candidate is None:
            # Should not happen, but guard against empty lists
            break

        # Append the best candidate and remove it from the remaining pool
        selected.append(best_candidate)
        remaining.remove(best_candidate)

        # Cap utilisation update could be implemented here if exposure
        # counters are provided externally.  This function leaves cap_ratio
        # unchanged; the caller may update candidate.cap_ratio values before
        # invoking greedy_rerank again if desired.

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

    This convenience function converts each row of the input DataFrame into a
    `Candidate` object, performs greedy re‑ranking and then returns a new
    DataFrame sorted by the re‑ranked order.  The resulting DataFrame
    includes an additional column ``business_score`` and retains all
    original columns.

    Args:
        df: Input DataFrame where each row corresponds to a candidate offer.
        params: Hyperparameters controlling the re‑ranking.
        k: Maximum number of rows to return.  If None, all rows are returned
            in the re‑ranked order.
        p_h: Hourly pacing adjustment to apply across all candidates.
        id_col: Name of the column containing the property identifier.
        score_col: Name of the column with the base relevance score.
        r_col: Name of the column with profitability values.
        g_col: Name of the column with plan‑gap values.
        m_col: Name of the column with fairness boost values.
        v_col: Name of the column with VIP flag values.
        contract_col: Name of the column with contract type values.
        cap_ratio_col: Name of the column with cap ratio values.
        inv_id_col: Optional column name for investment identifiers.
        dev_id_col: Optional column name for developer identifiers.
        city_id_col: Optional column name for city identifiers.

    Returns:
        A DataFrame sorted by the greedy business re‑ranking.  The top
        ``k`` rows (or all rows) represent the recommended order.
    """
    import pandas as pd  # Imported lazily to avoid imposing a hard dependency at import time

    candidates: List[Candidate] = []
    for _, row in df.iterrows():
        candidates.append(
            Candidate(
                property_id=row[id_col],
                q=row[score_col],
                r=row[r_col],
                g=row[g_col],
                m=row[m_col],
                v=row[v_col],
                contract_type=str(row[contract_col]).lower(),
                cap_ratio=row[cap_ratio_col],
                inv_id=row[inv_id_col] if inv_id_col and inv_id_col in row else None,
                dev_id=row[dev_id_col] if dev_id_col and dev_id_col in row else None,
                city_id=row[city_id_col] if city_id_col and city_id_col in row else None,
            )
        )

    ranked = greedy_rerank(candidates, params, k=k, p_h=p_h)

    # Build a mapping from property_id to computed business score
    score_map: Dict[int, float] = {}
    selected: List[Candidate] = []
    for cand in ranked:
        s = business_score(cand, params, p_h, selected)
        score_map[cand.property_id] = s
        selected.append(cand)

    # Add a business_score column to the DataFrame
    df = df.copy()
    df["business_score"] = df[id_col].map(score_map)

    # Sort rows by business_score in descending order and optionally truncate
    df = df.sort_values("business_score", ascending=False)
    if k is not None:
        df = df.head(k)
    return df