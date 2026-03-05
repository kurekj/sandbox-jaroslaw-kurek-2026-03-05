# business_reranking6.py
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

NEW (XAI / interpretability helpers):
  - business_score_breakdown(): returns a complete, exact breakdown of score
    components for a single candidate at a given greedy step (given `selected`).
  - cap_penalty_details() / diversity_penalty_details(): include "reason" metadata.
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
            the business score (see normalize_q).
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
            to [0, 1] before applying gamma *IF THEY ARE NEGATIVE* (legacy behavior).
            If False, q is used as is.
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


@dataclass
class ScoreBreakdown:
    """
    Exact, paper-friendly breakdown of the business score for ONE candidate at ONE greedy step.

    Interpretation:
      - final_score is computed exactly as business_score().
      - contrib_* show how much each business feature adds in "points"
        GIVEN the current q and pacing (core = base_component * pacing_term).
      - cap/div are penalties in points (already weighted by delta/lambda_).
      - score_no_* are counterfactual scores with a given module removed (selected fixed).
      - diff_no_* = score_no_* - final_score:
           * negative -> the removed module was helping (a boost)
           * positive -> the removed module was hurting (a penalty or negative boost)
    """

    property_id: int

    # --- sanitized inputs ---
    q_raw: float
    q_norm: float
    gamma: float
    base_component: float

    contract_type: str
    beta: float
    eta: float

    r: float
    g: float
    m: float
    v: int

    # --- terms inside business multiplier ---
    term_r: float
    term_g: float
    term_m: float
    term_v: float
    sum_terms: float
    business_factors: float  # 1 + sum_terms

    # --- pacing ---
    p_h_raw: float
    p_h_clip: float
    rho: float
    pace: float          # rho * p_h_clip
    pacing_term: float   # 1 + pace

    # --- exact additive breakdown (points) ---
    core: float          # base_component * pacing_term
    contrib_r: float     # core * term_r
    contrib_g: float     # core * term_g
    contrib_m: float     # core * term_m
    contrib_v: float     # core * term_v

    # --- penalties ---
    cap_ratio: float
    cap_penalty_raw: float
    cap_region: str
    cap_penalty_weighted: float  # delta * cap_penalty_raw

    div_penalty_raw: float
    div_reason: str
    div_match_property_id: Optional[int]
    div_penalty_weighted: float  # lambda_ * div_penalty_raw

    # --- scores ---
    score_no_penalties: float
    final_score: float

    # --- counterfactual scores ---
    score_no_r: float
    score_no_g: float
    score_no_m: float
    score_no_v: float
    score_no_pacing: float
    score_no_cap: float
    score_no_div: float

    # --- counterfactual diffs (score_no_* - final_score) ---
    diff_no_r: float
    diff_no_g: float
    diff_no_m: float
    diff_no_v: float
    diff_no_pacing: float
    diff_no_cap: float
    diff_no_div: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert breakdown to a plain dict (useful for pandas/Excel exports)."""
        from dataclasses import asdict
        return asdict(self)


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
    Convert a base score from [-1, 1] to [0, 1] if necessary (legacy).

    NOTE: This keeps the module's original behavior:
      - If q < 0, map (q + 1) / 2 to [0, 0.5]
      - If q in [0, 1], keep as is
      - If q > 1, clamp to 1

    If your base model outputs cosine similarity in [-1,1] and you want a full mapping,
    consider mapping all values by (q + 1) / 2 before passing them here, or set normalize_q=False
    and handle scaling upstream.
    """
    if q < 0.0:
        return (q + 1.0) / 2.0
    if q > 1.0:
        return 1.0
    return q


# -------------------------
# Penalties (cap + diversity)
# -------------------------
def cap_penalty(cap_ratio: float) -> float:
    """
    Compute the cap penalty C(i) given the utilisation ratio.

    The penalty is zero until 70% of the cap is used. Between 70% and 100%,
    the penalty increases linearly from 0 to 1. Above the cap the penalty
    saturates at 1.
    """
    cr = _to_float(cap_ratio, default=0.0)
    if cr < 0.0:
        cr = 0.0

    if cr < 0.7:
        return 0.0
    if cr < 1.0:
        return (cr - 0.7) / 0.3
    return 1.0


def cap_penalty_details(cap_ratio: float) -> Tuple[float, str]:
    """
    cap_penalty() + interpretable "region" label.

    Returns:
        (penalty_raw, region) where region in {'below','linear','saturated'}.
    """
    cr = _to_float(cap_ratio, default=0.0)
    if cr < 0.0:
        cr = 0.0

    if cr < 0.7:
        return 0.0, "below"
    if cr < 1.0:
        return (cr - 0.7) / 0.3, "linear"
    return 1.0, "saturated"


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
            # early stop if reached max expected similarity
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
) -> Tuple[float, str, Optional[int]]:
    """
    diversity_penalty() + interpretable reason and "which already-selected item caused it".

    Returns:
        (penalty_raw, reason, matched_property_id)
    reason in {'none','inv_id','dev_id','city_id'}.
    """
    max_sim = 0.0
    reason = "none"
    match_pid: Optional[int] = None

    for sel in selected:
        sim_inv = 0.0
        sim_dev = 0.0
        sim_city = 0.0

        if candidate.inv_id is not None and sel.inv_id is not None and candidate.inv_id == sel.inv_id:
            sim_inv = w_inv
        if candidate.dev_id is not None and sel.dev_id is not None and candidate.dev_id == sel.dev_id:
            sim_dev = w_dev
        if candidate.city_id is not None and sel.city_id is not None and candidate.city_id == sel.city_id:
            sim_city = w_city

        sim = max(sim_inv, sim_dev, sim_city)
        if sim > max_sim:
            max_sim = sim
            match_pid = sel.property_id

            # priority: inv > dev > city
            if sim_inv == sim and sim_inv > 0:
                reason = "inv_id"
            elif sim_dev == sim and sim_dev > 0:
                reason = "dev_id"
            elif sim_city == sim and sim_city > 0:
                reason = "city_id"
            else:
                reason = "none"

            if max_sim >= w_inv:
                break

    return max_sim, reason, match_pid


# -------------------------
# Core scoring
# -------------------------
def business_score(candidate: Candidate, params: RerankParams, p_h: float, selected: List[Candidate]) -> float:
    """
    Compute the business score S(u,i) for a candidate.

    IMPORTANT (robustness):
      - r and g are clamped to [0,1]
      - m is clamped to [-1,1]
      - v is binarized to {0,1} (non-zero -> 1)
      - q is clamped to [0,1] (after optional normalization)
      - p_h is clipped to [-1,1]
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


def business_score_breakdown(
    candidate: Candidate,
    params: RerankParams,
    p_h: float,
    selected: List[Candidate],
) -> ScoreBreakdown:
    """
    Same as business_score(), but returns a full, explainable breakdown.

    NOTE:
      This breakdown is LOCAL to the current greedy step:
        - diversity depends on `selected`
        - counterfactuals keep `selected` fixed (they do NOT re-run greedy)
    """
    # --- Base score (q) ---
    q_raw = _to_float(candidate.q, default=0.0)
    q_norm = _normalize_q(q_raw) if params.normalize_q else q_raw
    q_norm = min(max(q_norm, 0.0), 1.0)
    base_component = q_norm ** params.gamma

    # --- Contract weights β and η ---
    contract_type = str(candidate.contract_type).lower()
    beta, eta = params.contract_weights.get(contract_type, (0.0, 0.0))

    # --- Feature sanitization ---
    r = min(max(_to_float(candidate.r, default=0.0), 0.0), 1.0)
    g = min(max(_to_float(candidate.g, default=0.0), 0.0), 1.0)
    m = min(max(_to_float(candidate.m, default=0.0), -1.0), 1.0)
    v = 1 if _to_int(candidate.v, default=0) != 0 else 0

    # --- Business terms ---
    term_r = beta * r
    term_g = eta * g
    term_m = params.mu * m
    term_v = params.nu * v
    sum_terms = term_r + term_g + term_m + term_v
    business_factors = 1.0 + sum_terms

    # --- Pacing ---
    p_h_raw = _to_float(p_h, default=0.0)
    p_h_clip = max(min(p_h_raw, 1.0), -1.0)
    pace = params.rho * p_h_clip
    pacing_term = 1.0 + pace

    # --- Penalties ---
    cap_pen_raw, cap_region = cap_penalty_details(candidate.cap_ratio)
    cap_pen_weighted = params.delta * cap_pen_raw

    div_raw, div_reason, div_match_pid = diversity_penalty_details(
        candidate,
        selected,
        w_inv=params.w_inv,
        w_dev=params.w_dev,
        w_city=params.w_city,
    )
    div_weighted = params.lambda_ * div_raw

    # --- Exact score ---
    score_no_penalties = base_component * business_factors * pacing_term
    final_score = score_no_penalties - cap_pen_weighted - div_weighted

    # --- Exact additive "points" breakdown (simple, intuitive) ---
    core = base_component * pacing_term
    contrib_r = core * term_r
    contrib_g = core * term_g
    contrib_m = core * term_m
    contrib_v = core * term_v

    # --- Counterfactuals (remove a module; keep selected fixed) ---
    score_no_r = base_component * (1.0 + (sum_terms - term_r)) * pacing_term - cap_pen_weighted - div_weighted
    score_no_g = base_component * (1.0 + (sum_terms - term_g)) * pacing_term - cap_pen_weighted - div_weighted
    score_no_m = base_component * (1.0 + (sum_terms - term_m)) * pacing_term - cap_pen_weighted - div_weighted
    score_no_v = base_component * (1.0 + (sum_terms - term_v)) * pacing_term - cap_pen_weighted - div_weighted

    score_no_pacing = base_component * business_factors * 1.0 - cap_pen_weighted - div_weighted
    score_no_cap = base_component * business_factors * pacing_term - 0.0 - div_weighted
    score_no_div = base_component * business_factors * pacing_term - cap_pen_weighted - 0.0

    diff_no_r = score_no_r - final_score
    diff_no_g = score_no_g - final_score
    diff_no_m = score_no_m - final_score
    diff_no_v = score_no_v - final_score
    diff_no_pacing = score_no_pacing - final_score
    diff_no_cap = score_no_cap - final_score
    diff_no_div = score_no_div - final_score

    return ScoreBreakdown(
        property_id=candidate.property_id,
        q_raw=q_raw,
        q_norm=q_norm,
        gamma=params.gamma,
        base_component=base_component,
        contract_type=contract_type,
        beta=beta,
        eta=eta,
        r=r,
        g=g,
        m=m,
        v=v,
        term_r=term_r,
        term_g=term_g,
        term_m=term_m,
        term_v=term_v,
        sum_terms=sum_terms,
        business_factors=business_factors,
        p_h_raw=p_h_raw,
        p_h_clip=p_h_clip,
        rho=params.rho,
        pace=pace,
        pacing_term=pacing_term,
        core=core,
        contrib_r=contrib_r,
        contrib_g=contrib_g,
        contrib_m=contrib_m,
        contrib_v=contrib_v,
        cap_ratio=_to_float(candidate.cap_ratio, default=0.0),
        cap_penalty_raw=cap_pen_raw,
        cap_region=cap_region,
        cap_penalty_weighted=cap_pen_weighted,
        div_penalty_raw=div_raw,
        div_reason=div_reason,
        div_match_property_id=div_match_pid,
        div_penalty_weighted=div_weighted,
        score_no_penalties=score_no_penalties,
        final_score=final_score,
        score_no_r=score_no_r,
        score_no_g=score_no_g,
        score_no_m=score_no_m,
        score_no_v=score_no_v,
        score_no_pacing=score_no_pacing,
        score_no_cap=score_no_cap,
        score_no_div=score_no_div,
        diff_no_r=diff_no_r,
        diff_no_g=diff_no_g,
        diff_no_m=diff_no_m,
        diff_no_v=diff_no_v,
        diff_no_pacing=diff_no_pacing,
        diff_no_cap=diff_no_cap,
        diff_no_div=diff_no_div,
    )


# -------------------------
# Greedy reranking
# -------------------------
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


# -------------------------
# DataFrame helper (optional)
# -------------------------
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
