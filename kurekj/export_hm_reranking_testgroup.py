import sys
import asyncio
import pandas as pd
from datetime import datetime

from src.v2.db_utils import execute_query

# 🔹 Ustaw zakres ręcznie
FROM_TS_STR = "2025-12-02 12:02:00"
TO_TS_STR   = "2026-02-20 23:59:59"   # <-- tutaj podajesz górny timestamp

def _safe_ts_for_filename(ts_str: str) -> str:
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d_%H-%M-%S")

# --- CSV ---
OUTPUT_PATH = (
    f"hm-reranking-testgroup-"
    f"{_safe_ts_for_filename(FROM_TS_STR)}"
    f"_to_"
    f"{_safe_ts_for_filename(TO_TS_STR)}.csv"
)

# --- PARQUET ---
OUTPUT_PARQUET_PATH = (
    f"recommendations_output_"
    f"{_safe_ts_for_filename(TO_TS_STR)}.parquet"
)

QUERY = f"""
WITH ranked AS (
    SELECT
        l.*,
        row_number() OVER (PARTITION BY request_id ORDER BY iteration ASC)  AS rn_first,
        row_number() OVER (PARTITION BY request_id ORDER BY iteration DESC) AS rn_last
    FROM webrec_prod.logs l
    WHERE is_production
      AND create_date >= '{FROM_TS_STR}'::timestamp
      AND create_date <= '{TO_TS_STR}'::timestamp
),

per_request AS (
    SELECT
        last.request_id,
        last.create_date,
        (first.api_input::jsonb ->> 'uuid') AS uuid,
        last.recommendation_details::jsonb AS recommendation_details
    FROM ranked last
    JOIN ranked first
        ON first.request_id = last.request_id
    WHERE last.rn_last = 1
      AND first.rn_first = 1
      AND last.recommendation_details IS NOT NULL
),

per_request_latest AS (
    SELECT *
    FROM (
        SELECT
            p.*,
            row_number() OVER (
                PARTITION BY p.uuid
                ORDER BY p.create_date DESC
            ) AS rn_uuid
        FROM per_request p
    ) x
    WHERE rn_uuid = 1
),

expanded AS (
    SELECT
        p.create_date,
        p.uuid,
        p.request_id,
        (offer ->> 'offer_id')::bigint          AS offer_id,
        (offer ->> 'property_id')::bigint       AS property_id,
        (offer ->> 'api_model_score')::numeric  AS api_model_score,
        (offer ->> 'final_score')::numeric      AS final_score,
        (offer ->> 'gap_score')::numeric        AS gap_score,
        (offer ->> 'lead_price_score')::numeric AS lead_price_score,
        (p.recommendation_details
            -> 'recommendation_funnel'
            ->> 'recommendations_count')::int   AS recommendations_count
    FROM per_request_latest p
    CROSS JOIN LATERAL jsonb_array_elements(
        p.recommendation_details -> 'recommended_offers'
    ) AS offer
),

ranked_offers AS (
    SELECT
        e.*,
        row_number() OVER (
            PARTITION BY e.request_id
            ORDER BY e.final_score DESC
        ) AS rn_offer
    FROM expanded e
    WHERE e.recommendations_count > 1
)

SELECT
    create_date,
    uuid,
    request_id,
    offer_id,
    property_id,
    api_model_score,
    final_score,
    gap_score,
    lead_price_score
FROM ranked_offers
ORDER BY create_date DESC, uuid, final_score DESC;
"""

async def run_query_and_save():
    print("Executing SQL query...")
    rows = await execute_query(QUERY)

    print(f"Fetched {len(rows)} rows. Converting to DataFrame...")
    df = pd.DataFrame(rows)

    if "request_id" in df.columns:
        df["request_id"] = df["request_id"].astype(str)

    if "uuid" in df.columns and not pd.api.types.is_string_dtype(df["uuid"]):
        df["uuid"] = df["uuid"].astype(str)

    print(f"Saving DataFrame to CSV: {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saving DataFrame to Parquet: {OUTPUT_PARQUET_PATH}...")
    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print("DONE ✔️ Query results saved.")
    print(f"CSV file: {OUTPUT_PATH}")
    print(f"Parquet file: {OUTPUT_PARQUET_PATH}")

def main() -> None:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_query_and_save())

if __name__ == "__main__":
    main()