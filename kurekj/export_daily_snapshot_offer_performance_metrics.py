import sys
import asyncio
import pandas as pd
from datetime import datetime

from src.v2.db_utils import execute_query

RUN_ID = f"{datetime.now():%Y%m%d_%H%M%S}"

DATA_XLSX = f"daily_snapshot_offer_performance_metrics_data_{RUN_ID}.xlsx"
META_XLSX = f"daily_snapshot_offer_performance_metrics_meta_{RUN_ID}.xlsx"

TABLE_FQN = "daily_snapshot.offer_performance_metrics"

# 1) Dane tabeli
DATA_QUERY = f"SELECT * FROM {TABLE_FQN};"

# 2) Metadane kolumn (Postgres)
META_QUERY = """
WITH cols AS (
    SELECT
        c.table_schema,
        c.table_name,
        c.ordinal_position,
        c.column_name,
        c.data_type,
        c.udt_name,
        c.is_nullable,
        c.character_maximum_length,
        c.numeric_precision,
        c.numeric_scale,
        c.datetime_precision,
        c.column_default
    FROM information_schema.columns c
    WHERE c.table_schema = %(schema)s
      AND c.table_name   = %(table)s
)
SELECT
    cols.ordinal_position,
    cols.column_name,
    cols.data_type,
    cols.udt_name,
    cols.is_nullable,
    cols.character_maximum_length,
    cols.numeric_precision,
    cols.numeric_scale,
    cols.datetime_precision,
    cols.column_default,
    d.description AS column_comment
FROM cols
LEFT JOIN pg_catalog.pg_class cls
  ON cls.relname = cols.table_name
LEFT JOIN pg_catalog.pg_namespace nsp
  ON nsp.oid = cls.relnamespace AND nsp.nspname = cols.table_schema
LEFT JOIN pg_catalog.pg_attribute a
  ON a.attrelid = cls.oid AND a.attname = cols.column_name
LEFT JOIN pg_catalog.pg_description d
  ON d.objoid = cls.oid AND d.objsubid = a.attnum
ORDER BY cols.ordinal_position;
"""

def _split_schema_table(fqn: str) -> tuple[str, str]:
    parts = fqn.split(".")
    if len(parts) != 2:
        raise ValueError(f"TABLE_FQN must be in form schema.table, got: {fqn}")
    return parts[0], parts[1]

def _to_excel_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel nie lubi części typów (UUID, dict/list, datetime z tz, Decimal).
    Robimy bezpieczne konwersje do tekstu tam gdzie trzeba.
    """
    out = df.copy()

    for col in out.columns:
        s = out[col]

        # obiekty (często dict/list/json) → tekst
        if pd.api.types.is_object_dtype(s):
            if s.map(lambda x: isinstance(x, (dict, list, tuple, set))).any():
                out[col] = s.map(lambda x: None if x is None else str(x))

        # daty z timezone → bez tz (Excel i tak nie wspiera tz)
        if pd.api.types.is_datetime64tz_dtype(s):
            out[col] = s.dt.tz_convert(None)

        # UUID / inne niestandardowe → tekst (gdy pandas trzyma jako object)
        if pd.api.types.is_object_dtype(out[col]) and not pd.api.types.is_string_dtype(out[col]):
            if out[col].dropna().map(lambda x: not isinstance(x, str)).any():
                out[col] = out[col].astype(str)

    return out

def _profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Proste staty per kolumna."""
    total = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))
        rows.append({
            "column_name": col,
            "pandas_dtype": str(s.dtype),
            "rows": total,
            "nulls": nulls,
            "null_pct": (nulls / total * 100.0) if total else 0.0,
            "nunique": nunique,
        })
    return pd.DataFrame(rows).sort_values(["null_pct", "column_name"], ascending=[False, True])

def _freeze_header(writer: pd.ExcelWriter, sheet_names: list[str]) -> None:
    for sheet in sheet_names:
        ws = writer.book[sheet]
        ws.freeze_panes = "A2"

async def run_export():
    schema, table = _split_schema_table(TABLE_FQN)

    # --- DATA ---
    print("Executing data query...")
    data_rows = await execute_query(DATA_QUERY)  # list[dict]
    df_data = pd.DataFrame(data_rows)
    print(f"Fetched {len(df_data)} rows, {len(df_data.columns)} columns.")

    df_data_x = _to_excel_friendly(df_data)

    print(f"Saving DATA to {DATA_XLSX}...")
    with pd.ExcelWriter(DATA_XLSX, engine="openpyxl") as writer:
        df_data_x.to_excel(writer, sheet_name="data", index=False)
        _freeze_header(writer, ["data"])

    # --- META ---
    print("Executing metadata query...")
    meta_rows = await execute_query(
        META_QUERY,
        params={"schema": schema, "table": table},  # jeśli Twoje execute_query wspiera params
    )
    df_meta = pd.DataFrame(meta_rows)

    df_profile = _profile_columns(df_data)

    print(f"Saving META to {META_XLSX}...")
    with pd.ExcelWriter(META_XLSX, engine="openpyxl") as writer:
        df_meta.to_excel(writer, sheet_name="columns_meta", index=False)
        df_profile.to_excel(writer, sheet_name="columns_profile", index=False)
        _freeze_header(writer, ["columns_meta", "columns_profile"])

    print("DONE ✔️ Saved 2 files:")
    print(f" - {DATA_XLSX}")
    print(f" - {META_XLSX}")

def main() -> None:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_export())

if __name__ == "__main__":
    main()
