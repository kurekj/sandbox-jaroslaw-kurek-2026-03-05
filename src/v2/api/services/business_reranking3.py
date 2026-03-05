# business_reranking2.py
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

Debug / Excel breakdown (raw + params + multiplied + contributions):

>>> ranked, debug_rows = greedy_rerank_debug(candidates, params, k=2, p_h=0.0)
>>> debug_rows[0].keys()
dict_keys([...])

The functions defined here are intentionally decoupled from any specific
database or web framework.  They operate on simple data structures to
facilitate unit testing and reuse in both synchronous and asynchronous
contexts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


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
        gamma: Exponent applied to the base score q.  Values >1 sharpen
            differences in high scores; values <1 flatten them.
        mu: Weight applied to the fairness boost m(i).
        nu: Weight applied to the VIP flag v(i).
        rho: Weight applied to the pacing factor p(h).
        delta: Weight applied to the cap penalty C(i).
        lambda_: Weight applied to the diversity penalty D(i, L).
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

    The penalty is zero until 70 % of the cap is used.  Between 70 % and
    100 %, the penalty increases linearly from 0 to 1.  Above the cap the
    penalty saturates at 1.

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
    and any offer already selected, where similarity is a weighted max of
    binary indicators on investment, developer and city identifiers.  The
    resulting value lies in [0, 1] if w_inv<=1.  Higher values indicate that the
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


def diversity_penalty_details(
    candidate: Candidate,
    selected: List[Candidate],
    *,
    w_inv: float,
    w_dev: float,
    w_city: float,
) -> Tuple[float, Optional[int], Optional[str]]:
    """
    Debug version of diversity_penalty: returns (D_raw, match_property_id, match_reason).

    match_reason is one of: "inv", "dev", "city" (or None if no match).
    """
    max_sim = 0.0
    match_property_id: Optional[int] = None
    match_reason: Optional[str] = None

    for sel in selected:
        sim = 0.0
        reason: Optional[str] = None

        # Start at 0.0; raise to strongest matching weight
        if candidate.inv_id is not None and sel.inv_id is not None and candidate.inv_id == sel.inv_id:
            if w_inv > sim:
                sim = w_inv
                reason = "inv"

        if candidate.dev_id is not None and sel.dev_id is not None and candidate.dev_id == sel.dev_id:
            if w_dev > sim:
                sim = w_dev
                reason = "dev"

        if candidate.city_id is not None and sel.city_id is not None and candidate.city_id == sel.city_id:
            if w_city > sim:
                sim = w_city
                reason = "city"

        if sim > max_sim:
            max_sim = sim
            match_property_id = sel.property_id
            match_reason = reason

            # Early exit: maximum possible similarity reached
            if max_sim >= w_inv:
                break

    return max_sim, match_property_id, match_reason


def business_score(candidate: Candidate, params: RerankParams, p_h: float, selected: List[Candidate]) -> float:
    """
    Compute the business score S(u,i) for a candidate.

    This function implements Equation (1) from the report.  The base score
    `q` is optionally normalised to [0, 1], then raised to the power γ.
    Contract‑specific weights β and η are looked up based on the candidate's
    contract type.  The profitability r(i), plan gap g(i), fairness boost
    m(i) and VIP flag v(i) are combined linearly.  Pacing is applied via
    the factor 1 + ρ p(h).  Cap and diversity penalties are subtracted.

    IMPORTANT (robustness):
      - r and g are clamped to [0,1] (expected by the formula)
      - m is clamped to [-1,1]
      - v is binarized to {0,1} (non-zero -> 1)
      - q is clamped to [0,1] (after optional normalization)

    Args:
        candidate: Candidate offer being scored.
        params: Hyperparameters controlling the scoring.
        p_h: Pacing adjustment for the current hour h, in [−1, 1].
        selected: The list of offers that have already been chosen.  Used
            solely for computing the diversity penalty.

    Returns:
        The computed business score for the candidate.
    """
    # --- Base score (q) ---
    q_raw = _to_float(candidate.q, default=0.0)
    q_norm = _normalize_q(q_raw) if params.normalize_q else q_raw
    # Guard against any out-of-range values
    if q_norm < 0.0:
        q_norm = 0.0
    if q_norm > 1.0:
        q_norm = 1.0
    base_component = q_norm ** params.gamma

    # --- Contract weights β and η ---
    beta, eta = params.contract_weights.get(str(candidate.contract_type).lower(), (0.0, 0.0))

    # --- Feature sanitization (match the ranges assumed in the report) ---
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

    # --- Linear combination of profitability, plan gap, fairness and VIP ---
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


def business_score_breakdown(
    candidate: Candidate,
    params: RerankParams,
    p_h: float,
    selected: List[Candidate],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute business score + full debug breakdown (raw + params + multiplied + contributions).

    Returns:
        (score_final, breakdown_dict)

    Note:
        This breakdown is guaranteed to match `business_score()` 1:1 (same sanitization and math).
    """
    # --- base score q ---
    q_raw = _to_float(candidate.q, default=0.0)
    q_norm = _normalize_q(q_raw) if params.normalize_q else q_raw
    q_eff = min(max(q_norm, 0.0), 1.0)
    gamma = params.gamma
    base_component = q_eff ** gamma  # q^gamma

    # --- contract weights ---
    contract_type = str(candidate.contract_type).lower()
    beta, eta = params.contract_weights.get(contract_type, (0.0, 0.0))

    # --- raw features ---
    r_raw = _to_float(candidate.r, default=0.0)
    g_raw = _to_float(candidate.g, default=0.0)
    m_raw = _to_float(candidate.m, default=0.0)
    v_raw = _to_int(candidate.v, default=0)

    # --- effective features used in scoring (sanitized to expected ranges) ---
    r_eff = min(max(r_raw, 0.0), 1.0)
    g_eff = min(max(g_raw, 0.0), 1.0)
    m_eff = min(max(m_raw, -1.0), 1.0)
    v_eff = 1 if v_raw != 0 else 0

    # (1 + ...) terms (already multiplied by weights)
    term_profit = beta * r_eff
    term_gap = eta * g_eff
    term_fairness = params.mu * m_eff
    term_vip = params.nu * v_eff
    business_factors = 1.0 + term_profit + term_gap + term_fairness + term_vip

    # --- pacing ---
    p_h_raw = _to_float(p_h, default=0.0)
    p_h_clip = max(min(p_h_raw, 1.0), -1.0)
    pacing_term = 1.0 + params.rho * p_h_clip

    # --- cap penalty ---
    cap_ratio_raw = _to_float(candidate.cap_ratio, default=0.0)
    C_raw = cap_penalty(cap_ratio_raw)
    cap_weighted = params.delta * C_raw  # DELTA * C

    # --- diversity penalty (with details) ---
    D_raw, div_match_property_id, div_match_reason = diversity_penalty_details(
        candidate,
        selected,
        w_inv=params.w_inv,
        w_dev=params.w_dev,
        w_city=params.w_city,
    )
    div_weighted = params.lambda_ * D_raw  # LAMBDA * D

    # --- final score (exactly like business_score) ---
    score_main = base_component * business_factors * pacing_term
    score_final = score_main - cap_weighted - div_weighted

    # --- additive contributions to S (exact decomposition) ---
    base_main_contrib = base_component * pacing_term  # base part from "1" in (1+...)
    profit_contrib = base_component * pacing_term * term_profit
    gap_contrib = base_component * pacing_term * term_gap
    fairness_contrib = base_component * pacing_term * term_fairness
    vip_contrib = base_component * pacing_term * term_vip
    cap_contrib = -cap_weighted
    div_contrib = -div_weighted

    contribs = {
        "base_main_contrib": base_main_contrib,
        "profit_contrib": profit_contrib,
        "gap_contrib": gap_contrib,
        "fairness_contrib": fairness_contrib,
        "vip_contrib": vip_contrib,
        "cap_contrib": cap_contrib,
        "div_contrib": div_contrib,
    }

    abs_total = sum(abs(x) for x in contribs.values())
    if abs_total == 0.0:
        abs_total = 1e-12

    shares_abs_pct = {f"{k}_share_abs_pct": (abs(v) / abs_total) * 100.0 for k, v in contribs.items()}

    breakdown: Dict[str, Any] = {
        # identifiers
        "property_id": candidate.property_id,
        "contract_type": contract_type,
        "inv_id": candidate.inv_id,
        "dev_id": candidate.dev_id,
        "city_id": candidate.city_id,

        # q breakdown
        "q_raw": q_raw,
        "q_norm": q_norm,
        "q_eff_clamped_0_1": q_eff,
        "GAMMA": gamma,
        "base_component_q_pow_gamma": base_component,

        # raw vars (as in latex report)
        "lead_price_score_raw_r": r_raw,
        "gap_score_raw_g": g_raw,
        "fairness_score_raw_m": m_raw,
        "vip_flag_raw_v": v_raw,

        # effective vars used in math
        "lead_price_score_eff_r_clamped_0_1": r_eff,
        "gap_score_eff_g_clamped_0_1": g_eff,
        "fairness_score_eff_m_clamped_-1_1": m_eff,
        "vip_flag_eff_v_binarized": v_eff,

        # weights (params)
        "BETA": beta,
        "ETA": eta,
        "MU": params.mu,
        "NU": params.nu,
        "RHO": params.rho,
        "DELTA": params.delta,
        "LAMBDA": params.lambda_,

        # multiplied in bracket
        "beta_x_r": term_profit,
        "eta_x_g": term_gap,
        "mu_x_m": term_fairness,
        "nu_x_v": term_vip,
        "business_factors_1_plus_sum": business_factors,

        # pacing
        "p_h_raw": p_h_raw,
        "p_h_clip": p_h_clip,
        "pacing_term_1_plus_rho_p": pacing_term,

        # cap penalty raw + weighted
        "cap_ratio_raw": cap_ratio_raw,
        "cap_penalty_raw_C": C_raw,
        "DELTA_x_cap_penalty": cap_weighted,

        # diversity penalty raw + weighted
        "div_penalty_raw_D": D_raw,
        "LAMBDA_x_div_penalty": div_weighted,
        "div_match_property_id": div_match_property_id,
        "div_match_reason": div_match_reason,

        # final scores
        "score_main_base_x_factors_x_pacing": score_main,
        "score_final": score_final,

        # contributions
        **contribs,

        # abs shares (%)
        **shares_abs_pct,

        # your 4 modules (exact final-score impact per module)
        "MOD_cap_term": cap_contrib,              # -DELTA*C
        "MOD_div_term": div_contrib,              # -LAMBDA*D
        "MOD_fairness_term": fairness_contrib,    # + base*pacing*(MU*m)
        "MOD_vip_term": vip_contrib,              # + base*pacing*(NU*v)
    }

    return score_final, breakdown


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


def greedy_rerank_debug(
    candidates: List[Candidate],
    params: RerankParams,
    *,
    k: Optional[int] = None,
    p_h: float = 0.0,
) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """
    Greedy rerank, BUT logs full breakdown for every candidate at every iteration.

    Returns:
        (selected_candidates, debug_rows)

    debug_rows is a flat list of dicts (ready for pandas.DataFrame / Excel):
        - iter_no
        - selected_len_before
        - was_selected_in_iter (0/1)
        - selected_rank (only for winner rows)
        - all fields from business_score_breakdown
    """
    remaining = candidates.copy()
    selected: List[Candidate] = []
    limit = len(remaining) if k is None else max(min(k, len(remaining)), 0)

    debug_rows: List[Dict[str, Any]] = []
    iter_no = 0

    while remaining and len(selected) < limit:
        iter_no += 1
        best_candidate: Optional[Candidate] = None
        best_score: float = float("-inf")

        # Remember indices of rows generated in this iteration (so we can mark the winner)
        iter_rows_idx: List[int] = []

        for cand in remaining:
            score, dbg = business_score_breakdown(cand, params, p_h, selected)

            dbg["iter_no"] = iter_no
            dbg["selected_len_before"] = len(selected)
            dbg["was_selected_in_iter"] = 0
            dbg["selected_rank"] = None

            iter_rows_idx.append(len(debug_rows))
            debug_rows.append(dbg)

            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_candidate is None:
            break

        # Mark winner row in this iteration
        for idx in iter_rows_idx:
            if debug_rows[idx].get("property_id") == best_candidate.property_id:
                debug_rows[idx]["was_selected_in_iter"] = 1
                debug_rows[idx]["selected_rank"] = len(selected) + 1

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected, debug_rows


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

    # Build a mapping from property_id to computed business score
    score_map: Dict[int, float] = {}
    selected: List[Candidate] = []
    for cand in ranked:
        s = business_score(cand, params, p_h, selected)
        score_map[cand.property_id] = s
        selected.append(cand)

    # Add a business_score column to the DataFrame
    out = df.copy()
    out["business_score"] = out[id_col].map(score_map)

    # Sort rows by business_score in descending order and optionally truncate
    out = out.sort_values("business_score", ascending=False)
    if k is not None:
        out = out.head(k)
    return out


def rerank_dataframe_debug(
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
    keep_original_cols: bool = True,
) -> Tuple["pandas.DataFrame", "pandas.DataFrame"]:
    """
    DataFrame wrapper for greedy_rerank_debug.

    Returns:
        (topk_breakdown_df, debug_greedy_all_df)

    Notes:
        - This expects df to represent ONE candidate set (np. 1 request_id).
        - For many request_id: groupby(request_id) and call this per group,
          then concat results.
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

    ranked, debug_rows = greedy_rerank_debug(candidates, params, k=k, p_h=p_h)
    debug_df = pd.DataFrame(debug_rows)

    # Build top-k breakdown for the selected sequence (one row per selected item)
    topk_rows: List[Dict[str, Any]] = []
    selected_so_far: List[Candidate] = []
    for cand in ranked:
        _, b = business_score_breakdown(cand, params, p_h, selected_so_far)
        b["selected_rank"] = len(selected_so_far) + 1
        topk_rows.append(b)
        selected_so_far.append(cand)

    topk_df = pd.DataFrame(topk_rows)

    if keep_original_cols and not topk_df.empty:
        # Merge original df columns for selected property_id rows
        base_sel = df[df[id_col].isin(topk_df["property_id"].tolist())].copy()
        base_sel = base_sel.rename(columns={id_col: "property_id"})
        topk_df = topk_df.merge(base_sel, on="property_id", how="left", suffixes=("", "_orig"))

        # Keep ordering by selected_rank
        if "selected_rank" in topk_df.columns:
            topk_df = topk_df.sort_values("selected_rank", ascending=True)

    return topk_df, debug_df


def _default_output_dir() -> Path:
    """
    Default output directory for debug XLSX.

    Priority:
      1) env BUSINESS_RERANKING2_OUTPUT_DIR (set by runner)
      2) ./Output_debug (cwd)
    """
    env_dir = os.getenv("BUSINESS_RERANKING2_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.cwd() / "Output_debug"


def _default_ts() -> str:
    """
    Timestamp for filenames (optional).

    Priority:
      1) env BUSINESS_RERANKING2_TS (set by runner)
      2) datetime.now()
    """
    env_ts = os.getenv("BUSINESS_RERANKING2_TS")
    if env_ts:
        return env_ts
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_xlsx_path(path: Union[str, Path]) -> Path:
    """
    Resolve XLSX path:
      - if absolute: keep it
      - if relative: put it inside BUSINESS_RERANKING2_OUTPUT_DIR (or Output_debug)
      - ensure parent dir exists
      - ensure .xlsx suffix
    """
    p = Path(path)
    if p.suffix.lower() != ".xlsx":
        p = p.with_suffix(".xlsx")

    if p.is_absolute():
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    out_dir = _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / p


def export_debug_to_excel(
    topk_df: "pandas.DataFrame",
    debug_df: "pandas.DataFrame",
    path: Union[str, Path],
    *,
    topk_sheet: str = "topk_breakdown",
    debug_sheet: str = "debug_greedy_all",
) -> Path:
    """
    Writes 2 DataFrames into an .xlsx:
      - topk_breakdown sheet
      - debug_greedy_all sheet

    If `path` is relative, it is saved into:
      BUSINESS_RERANKING2_OUTPUT_DIR (env) or ./Output_debug (cwd)
    """
    import pandas as pd  # Lazy import

    out_path = _resolve_xlsx_path(path)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        topk_df.to_excel(writer, sheet_name=topk_sheet, index=False)
        debug_df.to_excel(writer, sheet_name=debug_sheet, index=False)

    return out_path.resolve()
