import pandas as pd
from pathlib import Path

PARQUET_PATH = Path("recommendations_output.parquet")
EXCEL_PATH = Path("recommendations_output_latest10_with_stats.xlsx")

DATE_COL = "create_date"
N_LATEST = 10

ID_COLUMNS = ["uuid", "request_id", "offer_id"]
SCORE_COLUMNS = [
    "api_model_score",
    "final_score",
    "gap_score",
    "lead_price_score",
]


def safe_pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((count / total) * 100.0, 2)


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {PARQUET_PATH.resolve()}")

    print(f"Wczytuję parquet: {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")

    if df.empty:
        raise ValueError("Parquet jest pusty (0 rekordów).")

    # --- sortowanie po dacie ---
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values(by=DATE_COL, ascending=False, na_position="last")

    # --- latest 10 ---
    latest = df.head(N_LATEST).copy()

    # --- Excel nie obsługuje timezone-aware datetime ---
    for col in latest.columns:
        if pd.api.types.is_datetime64tz_dtype(latest[col]):
            latest[col] = latest[col].dt.tz_localize(None)

    # =========================
    # 📊 STATYSTYKI – ID
    # =========================
    stats_ids = []

    for col in ID_COLUMNS:
        if col not in df.columns:
            continue

        stats_ids.append(
            {
                "column": col,
                "total": int(df[col].notna().sum()),
                "unique": int(df[col].nunique(dropna=True)),
            }
        )

    df_stats_ids = pd.DataFrame(stats_ids)

    # =========================
    # 📊 STATYSTYKI – SCORES (z %)
    # =========================
    stats_scores = []

    for col in SCORE_COLUMNS:
        if col not in df.columns:
            continue

        values = df[col]
        non_null = int(values.notna().sum())

        cnt_m1 = int((values == -1).sum())
        cnt_0 = int((values == 0).sum())
        cnt_1 = int((values == 1).sum())

        stats_scores.append(
            {
                "column": col,
                "-1_count": cnt_m1,
                "0_count": cnt_0,
                "1_count": cnt_1,
                "non_null": non_null,
                "-1_%": safe_pct(cnt_m1, non_null),
                "0_%": safe_pct(cnt_0, non_null),
                "1_%": safe_pct(cnt_1, non_null),
            }
        )

    df_stats_scores = pd.DataFrame(stats_scores)

    # =========================
    # 📄 LISTA KOLUMN
    # =========================
    df_columns = pd.DataFrame(
        {
            "column_name": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
        }
    )

    # =========================
    # 📤 ZAPIS DO EXCELA
    # =========================
    print(f"Zapisuję do Excela: {EXCEL_PATH} ...")

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        latest.to_excel(writer, sheet_name="latest_10", index=False)
        df_stats_ids.to_excel(writer, sheet_name="stats_ids", index=False)
        df_stats_scores.to_excel(writer, sheet_name="stats_scores", index=False)
        df_columns.to_excel(writer, sheet_name="columns", index=False)

    print("DONE ✔️ Excel zapisany.")


if __name__ == "__main__":
    main()
