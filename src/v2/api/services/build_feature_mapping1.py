"""
Script to build a human‑readable feature mapping for one‑hot encoded
categorical features in the real‑estate recommendation system.

This script connects to the same PostgreSQL database used by the
recommendation service and retrieves the dictionary tables that define
the meaning of each facility and natural site identifier.  It then
correlates those identifiers with the one‑hot encoded feature columns
(`facilities_1`, `facilities_2`, … and `natural_sites_1`, …) based on
their order in the arrays returned by ``OFFERS_DICTS_QUERY``.  Using
``OFFERS_DICTS_QUERY`` directly avoids the need to call
``load_offers_data()``, which expects a placeholder parameter and
would raise a ``ProgrammingError`` when executed without one.  The
resulting mapping is written to ``feature_mapping.json`` in the
current working directory and printed to stdout.

Usage: run this script with ``python build_feature_mapping.py``.  It
requires that the environment is configured exactly like the API
service, i.e. the ``config/.env`` file or corresponding environment
variables must provide valid database connection credentials.  If run
outside that environment, connection attempts will fail.

Note: The script makes a few assumptions about the structure of the
dictionary tables.  It expects each table to have an integer ``id``
column and at least one additional column containing a human‑readable
description (e.g. ``name`` or ``value``).  If the column names differ,
the script will attempt to choose the first non‑ID column as the
description.
"""

import asyncio
import json
import sys
import time  # unused import but included for completeness
from typing import Dict, List, Any

# Adjust the event loop policy on Windows to avoid runtime warnings
if sys.platform == "win32":  # pragma: no cover - platform specific
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from src.v2.db_utils import execute_query  # type: ignore
from src.v2.autoencoder.load_data import OFFERS_DICTS_QUERY  # type: ignore


async def _get_dictionary_mapping(table_name: str) -> Dict[int, Any]:
    """Retrieve a mapping from dictionary table IDs to names.

    The function queries ``SELECT * FROM <table_name>`` and then
    inspects the returned rows to find the first column that is not
    ``id`` and not obviously a timestamp.  This column is assumed to
    hold a human‑readable description of the entry.  If no such
    column is found, the IDs themselves are returned as values.

    Parameters
    ----------
    table_name: str
        The fully qualified name of the dictionary table (e.g.
        ``dictionary.rds2__offers_offer__facilities``).

    Returns
    -------
    dict[int, Any]
        A mapping from ID to descriptive name (or the ID itself if
        no name column exists).
    """
    # Fetch all rows from the dictionary table.  If the table does not
    # exist or the connection fails, an exception will propagate.
    rows = await execute_query(f"SELECT * FROM {table_name}")
    if not rows:
        return {}

    # Determine the name column: any column that is not 'id' and not
    # obviously a timestamp.  The heuristics skip columns containing
    # 'created', 'modified', 'updated', etc.
    sample_row = rows[0]
    candidate_cols = []
    for col in sample_row.keys():
        if col.lower() == "id":
            continue
        low = col.lower()
        if any(tok in low for tok in ("created", "modified", "updated", "timestamp")):
            continue
        candidate_cols.append(col)
    # Choose the first candidate or fall back to the ID itself
    name_col = candidate_cols[0] if candidate_cols else None

    mapping: Dict[int, Any] = {}
    for row in rows:
        idx = row.get("id")
        if idx is None:
            continue
        mapping[int(idx)] = row.get(name_col, idx) if name_col else idx
    return mapping


async def build_feature_mapping() -> Dict[str, Any]:
    """Construct and persist a mapping from feature names to descriptions.

    The function performs the following steps:

    1. Execute ``OFFERS_DICTS_QUERY`` to obtain the ordered lists of
       facility and natural site identifiers (keys ``all_facilities`` and
       ``all_natural_sites`` in the returned row).
    2. Query the dictionary tables for facilities and natural sites to
       retrieve human‑readable names for each ID.
    3. Combine the lists and dictionaries into a mapping from
       one‑hot column names (e.g. ``facilities_1``) to descriptive
       names (e.g. ``"Parking"``).  If a description is unavailable,
       the raw ID is used.
    4. Save the mapping as JSON to ``feature_mapping.json``.
    5. Return the mapping for immediate use.
    """
    # Step 1: Retrieve the arrays of facility and natural site IDs using
    # OFFERS_DICTS_QUERY.  The query returns a single row with keys
    # 'all_facilities' and 'all_natural_sites'.
    dict_rows = await execute_query(OFFERS_DICTS_QUERY)
    if not dict_rows:
        raise RuntimeError(
            "OFFERS_DICTS_QUERY returned no rows; cannot determine feature order."
        )
    first_dict_row = dict_rows[0]
    facilities_ids = first_dict_row.get("all_facilities") or []
    natural_ids = first_dict_row.get("all_natural_sites") or []

    # Step 2: Retrieve name mappings for the dictionary tables
    facilities_map = await _get_dictionary_mapping(
        "dictionary.rds2__offers_offer__facilities"
    )
    natural_map = await _get_dictionary_mapping(
        "dictionary.rds2__offers_offer__natural_sites"
    )

    # Step 3: Combine into a unified feature mapping
    feature_mapping: Dict[str, Any] = {}
    for idx, fid in enumerate(facilities_ids, start=1):
        key = f"facilities_{idx}"
        # Cast the identifier to int if possible before lookup.  If casting
        # fails, fall back to the original value.
        try:
            fid_int: int = int(fid)
        except Exception:
            fid_int = fid  # type: ignore[assignment]
        feature_mapping[key] = facilities_map.get(fid_int, fid)
    for idx, nid in enumerate(natural_ids, start=1):
        key = f"natural_sites_{idx}"
        try:
            nid_int: int = int(nid)
        except Exception:
            nid_int = nid  # type: ignore[assignment]
        feature_mapping[key] = natural_map.get(nid_int, nid)

    # Step 4: Persist mapping to JSON
    with open("feature_mapping.json", "w", encoding="utf-8") as f:
        json.dump(feature_mapping, f, ensure_ascii=False, indent=2)

    # Step 5: Return for immediate inspection
    return feature_mapping


if __name__ == "__main__":
    # Run the mapping builder and print the result.  Any exceptions
    # raised here will cause a non-zero exit code and display a
    # stack trace, which can help diagnose connection or query
    # issues.
    mapping = asyncio.run(build_feature_mapping())
    print("Generated feature mapping:")
    for feature, name in mapping.items():
        print(f"{feature}: {name}")