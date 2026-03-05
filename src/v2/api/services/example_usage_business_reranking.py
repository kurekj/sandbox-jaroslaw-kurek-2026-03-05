"""
Przykładowe użycie modułu business_reranking.

Ten skrypt pokazuje, jak wykorzystać zaimplementowaną warstwę re‑rankingu
do uszeregowania listy ofert oraz DataFrame z kandydatami.  Przed
uruchomieniem upewnij się, że pakiet `pandas` jest zainstalowany.
"""

import pandas as pd

from src.v2.api.services.business_reranking import (
    Candidate,
    RerankParams,
    greedy_rerank,
    rerank_dataframe,
    business_score,
)


def example_list() -> list[tuple[int, float]]:
    """Użycie na liście obiektów Candidate."""
    candidates = [
        Candidate(property_id=1, q=0.92, r=0.4, g=0.3, m=0.1, v=0, contract_type='flat', cap_ratio=0.5),
        Candidate(property_id=2, q=0.85, r=0.6, g=0.4, m=-0.2, v=0, contract_type='cpl', cap_ratio=0.2),
        Candidate(property_id=3, q=0.80, r=0.2, g=0.8, m=0.3, v=1, contract_type='flat', cap_ratio=0.9),
        Candidate(property_id=4, q=0.75, r=0.7, g=0.1, m=0.0, v=0, contract_type='cpl', cap_ratio=0.1),
    ]
    params = RerankParams(
        gamma=1.2,
        mu=0.5,
        nu=0.8,
        rho=0.3,
        delta=1.0,
        lambda_=0.5,
    )
    ranked = greedy_rerank(candidates, params, p_h=0.1)
    result: list[tuple[int, float]] = []
    selected: list[Candidate] = []
    for cand in ranked:
        score = business_score(cand, params, p_h=0.1, selected=selected)
        result.append((cand.property_id, score))
        selected.append(cand)
    return result


def example_dataframe() -> pd.DataFrame:
    """Użycie na pandas DataFrame."""
    data = [
        {'property_id': 101, 'score': 0.88, 'r': 0.5, 'g': 0.4, 'm': 0.2, 'v': 0, 'contract_type': 'flat', 'cap_ratio': 0.4},
        {'property_id': 102, 'score': 0.80, 'r': 0.3, 'g': 0.6, 'm': -0.1, 'v': 0, 'contract_type': 'flat', 'cap_ratio': 0.7},
        {'property_id': 103, 'score': 0.70, 'r': 0.8, 'g': 0.2, 'm': 0.0, 'v': 1, 'contract_type': 'cpl', 'cap_ratio': 0.5},
    ]
    df = pd.DataFrame(data)
    params = RerankParams(gamma=1.0, mu=0.5, nu=1.0, rho=0.0, delta=1.0, lambda_=0.0)
    reranked_df = rerank_dataframe(df, params, p_h=0.0)
    return reranked_df[['property_id', 'business_score']]


if __name__ == '__main__':
    print("Przykład listy kandydatów:")
    for pid, score in example_list():
        print(f"ID: {pid}, business_score={score:.3f}")

    print("\nPrzykład DataFrame:")
    print(example_dataframe())