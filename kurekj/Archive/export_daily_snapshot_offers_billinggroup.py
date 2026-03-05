import sys
import asyncio
import pandas as pd
from datetime import datetime

from src.v2.db_utils import execute_query

#OUTPUT_XLSX = "offers_billinggroup_export.xlsx"
OUTPUT_XLSX = f"daily_snapshot_offers_billinggroup_export_{datetime.now():%Y%m%d_%H%M%S}.xlsx"

TABLE_FQN = "daily_snapshot.offers_billinggroup"

# 1) Dane tabeli
DATA_QUERY = f"SELECT * FROM {TABLE_FQN};"

# 2) Metadane kolumn (Postgres)
# - information_schema.columns: typy, nullowalność, defaulty, itp.
# - pg_catalog.pg_description: komentarze do kolumn (jeśli używacie COMMENT ON COLUMN)
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
            # jeśli w kolumnie są słowniki/listy itp.
            if s.map(lambda x: isinstance(x, (dict, list, tuple, set))).any():
                out[col] = s.map(lambda x: None if x is None else str(x))

        # daty z timezone → bez tz (Excel i tak nie wspiera tz)
        if pd.api.types.is_datetime64tz_dtype(s):
            out[col] = s.dt.tz_convert(None)

        # UUID / inne niestandardowe → tekst (gdy pandas trzyma jako object)
        # (zostawiamy jeśli to zwykły string)
        if pd.api.types.is_object_dtype(out[col]) and not pd.api.types.is_string_dtype(out[col]):
            # heurystyka: jeśli są obiekty nie-None, konwertuj na str
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

async def run_export():
    schema, table = _split_schema_table(TABLE_FQN)

    print("Executing data query...")
    data_rows = await execute_query(DATA_QUERY)  # list[dict]
    df_data = pd.DataFrame(data_rows)

    print(f"Fetched {len(df_data)} rows, {len(df_data.columns)} columns.")

    print("Executing metadata query...")
    meta_rows = await execute_query(
        META_QUERY,
        params={"schema": schema, "table": table},  # jeśli Twoje execute_query wspiera params
    )
    df_meta = pd.DataFrame(meta_rows)

    # Jeżeli Twoje execute_query NIE wspiera params, użyj f-stringów (mniej bezpieczne):
    # meta_rows = await execute_query(META_QUERY_FSTR)
    # gdzie META_QUERY_FSTR z wstrzykniętym schema/table po walidacji

    # Konwersje pod Excel
    df_data_x = _to_excel_friendly(df_data)

    # Profil kolumn (opcjonalnie, ale zwykle mega pomocne)
    df_profile = _profile_columns(df_data)

    print(f"Saving to {OUTPUT_XLSX}...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_data_x.to_excel(writer, sheet_name="data", index=False)
        df_meta.to_excel(writer, sheet_name="columns_meta", index=False)
        df_profile.to_excel(writer, sheet_name="columns_profile", index=False)

        # Ustaw trochę “czytelności” (zamrożenie nagłówka)
        for sheet in ["data", "columns_meta", "columns_profile"]:
            ws = writer.book[sheet]
            ws.freeze_panes = "A2"

    print("DONE. Excel saved with data + metadata + profile.")

def main() -> None:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_export())

if __name__ == "__main__":
    main()
