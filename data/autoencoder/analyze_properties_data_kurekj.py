import os
import math
import json
import pickle
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_string_dtype,
    is_bool_dtype,
)
from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype  # tz-aware dtype check

# ===== USTAWIENIA OGÓLNE =====
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DEFAULT_PICKLE_NAME = "4.current_properties_after_2022_preprocessed_final.pkl"

# pod Twój zestaw: 131 kolumn – limit 300 nie ogranicza
DEFAULT_CORR_MAX_COLS = 150
# mniej „hałasu” w top-k kategorii w excelach
DEFAULT_TOPK_CATS = 50
HIGH_CORR_THRESHOLD = 0.5

# ===== Cramér’s V – FAST CONFIG (dopasowane do n=71 555, 131 kolumn) =====
# 0 = bez próbkowania (71k jest OK bez kopiowania)
CRAMERS_SAMPLE_ROWS = 0
# twardy limit unikatów – odetnie region_id(610), vendor_id(644), plot_area(574)
CRAMERS_MAX_UNIQUES = 500
# twardy limit wielkości tabeli r*k (np. 322×322>80k -> odetnie ciężkie pary)
CRAMERS_MAX_CELLS = 80_000
# maksymalna liczba kolumn kategorycznych dopuszczonych do Cramér’s V po filtrach
CRAMERS_LIMIT_COLS = 50

# ===== Eta² – FAST CONFIG =====
# 0 = licz na pełnym zbiorze (71k OK)
ETA_SAMPLE_ROWS = 0
# dopuszczalna liczba grup w kategorii do eta² (pozwoli np. 322, 610, 644)
ETA_MAX_GROUPS = 2_000

# ===== Filtry kolumn do Cramér’s V =====
# Pomijamy binarne one-hoty (zwykle mało wnoszą, mnożą pary)
CATS_MIN_UNIQUES_FOR_V = 3
CATS_MAX_UNIQUES_FOR_V = CRAMERS_MAX_UNIQUES
EXCLUDE_FROM_V = {
    # ID / surogaty / jednolite kolumny
    "property_id", "offer_id", "id",
    "all_type_id", "all_facilities", "all_natural_sites", "all_holiday_location",
    "all_quarters", "all_flat_type", "all_kitchen_type", "all_house_type", "all_additional_area_type",
    # obiekty/teksty o dużej kardynalności/strukturze
    "raw_pois", "plot_area",
}
EXCLUDE_PREFIXES_FROM_V = (
    "facilities_", "natural_sites_", "holiday_location_", "quarters_",
    "flat_type_", "kitchen_type_", "house_type_", "additional_area_type_"
)

def write_excel_with_examples(path: str, df: pd.DataFrame) -> None:
    columns_info = columns_catalog_with_examples(df)
    with pd.ExcelWriter(path) as w:
        columns_info.to_excel(w, sheet_name="columns_catalog_with_examples", index=False)

def columns_catalog_with_examples(df: pd.DataFrame, n_examples: int = 5) -> pd.DataFrame:
    rows, n = [], len(df)
    for c in df.columns:
        s = df[c]
        missing = int(s.isna().sum())
        nunique = safe_nunique(s)
        zero_count = int((s == 0).sum()) if is_numeric_dtype(s) and not is_bool_dtype(s) else np.nan
        try:
            vc = safe_value_counts(s, normalize=True, dropna=True)
            top_ratio = float(vc.iloc[0]) if len(vc) > 0 else np.nan
        except Exception:
            top_ratio = np.nan

        # Obsługa kolumn z wartościami typu list
        if isinstance(s.iloc[0], list):
            examples = s.dropna().apply(lambda x: str(x) if isinstance(x, list) else x).unique()[:n_examples]
        # Obsługa kolumn z wartościami typu dict
        elif isinstance(s.iloc[0], dict):
            examples = s.dropna().apply(lambda x: json.dumps(x) if isinstance(x, dict) else x).unique()[:n_examples]
        else:
            # Pobranie przykładowych wartości
            examples = s.dropna().unique()[:n_examples]  # Zbieramy unikalne przykłady wartości

        rows.append({
            "column": c, "dtype": str(s.dtype),
            "non_null": n - missing, "nulls": missing,
            "null_ratio": round(missing / n, 6) if n else np.nan,
            "nunique": nunique, "zero_count": zero_count,
            "memory_mb": round(s.memory_usage(deep=True) / (1024 ** 2), 6),
            "top_value_ratio": round(top_ratio, 6) if pd.notna(top_ratio) else np.nan,
            "examples": ", ".join(map(str, examples))  # Przykładowe wartości
        })
    return pd.DataFrame(rows)


# ===== HASHABLE HELPERS =====
def _is_hashable(x) -> bool:
    try:
        hash(x)
        return True
    except TypeError:
        return False

def _to_hashable(x):
    if isinstance(x, float) and pd.isna(x):
        return x
    if _is_hashable(x):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            return tuple(_to_hashable(e) for e in x)
        except Exception:
            return str(x)
    if isinstance(x, set):
        try:
            return tuple(sorted(_to_hashable(e) for e in x))
        except Exception:
            return str(sorted(list(x)))
    if isinstance(x, dict):
        try:
            return json.dumps(x, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(x)
    return str(x)

def _series_to_hashable(s: pd.Series) -> pd.Series:
    if is_numeric_dtype(s) or is_datetime64_any_dtype(s):
        return s
    return s.map(_to_hashable)

def safe_nunique(s: pd.Series) -> int:
    return int(_series_to_hashable(s).nunique(dropna=True))

def safe_value_counts(s: pd.Series, normalize: bool = False, dropna: bool = True) -> pd.Series:
    return _series_to_hashable(s).value_counts(normalize=normalize, dropna=dropna)

def safe_unique_count_heuristic(s: pd.Series) -> int:
    return safe_nunique(s)

# ===== TZ HELPERS (Excel nie obsługuje tz) =====
def _strip_tz_series(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, DatetimeTZDtype):
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    if is_datetime64_any_dtype(s):
        return s
    if s.dtype == "object":
        def _conv(x):
            if isinstance(x, pd.Timestamp) and x.tz is not None:
                return x.tz_convert("UTC").tz_localize(None)
            return x
        try:
            return s.map(_conv)
        except Exception:
            return s
    return s

def _strip_tz_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = _strip_tz_series(out[col])
    return out

# ===== UTILS =====
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_output_dir(output_root: str, input_basename: str) -> str:
    safe_name = os.path.splitext(os.path.basename(input_basename))[0] or "dataset"
    out_dir = os.path.join(output_root, f"analysis_{safe_name}_{_timestamp()}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_dataframe(path: str, csv_sep: Optional[str] = None, excel_sheet: Optional[str] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, sep=csv_sep if csv_sep is not None else ",")
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=excel_sheet)
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            df = pickle.load(f)
    elif ext in (".feather", ".ft"):
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Nieobsługiwane rozszerzenie: {ext}")
    return pd.DataFrame(df)

# Heurystyczne i ciche parsowanie dat
def infer_and_parse_datetimes(df: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
    df = df.copy()
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return df
    sample = df[obj_cols].sample(min(len(df), sample_size), random_state=RANDOM_STATE)
    candidates = []
    for c in obj_cols:
        s = sample[c].dropna().astype(str).head(100)
        ok = 0
        for v in s:
            try:
                pd.to_datetime(v, errors="raise")
                ok += 1
            except Exception:
                pass
        if len(s) > 0 and ok / len(s) >= 0.7:
            candidates.append(c)
    for c in candidates:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def split_columns_by_type(df: pd.DataFrame) -> Dict[str, List[str]]:
    nums, cats, dates, texts, others = [], [], [], [], []
    for c in df.columns:
        s = df[c]
        if is_datetime64_any_dtype(s):
            dates.append(c); continue
        if is_bool_dtype(s):
            cats.append(c); continue
        if is_numeric_dtype(s):
            nums.append(c); continue
        if isinstance(s.dtype, CategoricalDtype):
            cats.append(c); continue
        if is_string_dtype(s) or s.dtype == "object":
            try:
                uniq = safe_unique_count_heuristic(s)
            except Exception:
                uniq = None
            if uniq is None:
                others.append(c)
            elif uniq <= max(50, int(0.05 * len(s))):
                cats.append(c)
            else:
                texts.append(c)
        else:
            others.append(c)
    return {"numeric": nums, "categorical": cats, "datetime": dates, "text": texts, "other": others}

# ===== ANALIZY POD TABELKI =====
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    mem = df.memory_usage(deep=True).sum()
    return pd.DataFrame({"n_rows":[len(df)], "n_cols":[len(df.columns)], "memory_bytes":[int(mem)], "memory_mb":[round(mem/(1024**2),3)]})

def columns_catalog(df: pd.DataFrame) -> pd.DataFrame:
    rows, n = [], len(df)
    for c in df.columns:
        s = df[c]
        missing = int(s.isna().sum())
        nunique = safe_nunique(s)
        zero_count = int((s == 0).sum()) if is_numeric_dtype(s) and not is_bool_dtype(s) else np.nan
        try:
            vc = safe_value_counts(s, normalize=True, dropna=True)
            top_ratio = float(vc.iloc[0]) if len(vc) > 0 else np.nan
        except Exception:
            top_ratio = np.nan
        rows.append({
            "column": c, "dtype": str(s.dtype),
            "non_null": n - missing, "nulls": missing,
            "null_ratio": round(missing/n, 6) if n else np.nan,
            "nunique": nunique, "zero_count": zero_count,
            "memory_mb": round(s.memory_usage(deep=True)/(1024**2), 6),
            "top_value_ratio": round(top_ratio, 6) if pd.notna(top_ratio) else np.nan
        })
    return pd.DataFrame(rows)

def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame([{"column":c, "missing_count":int(df[c].isna().sum()), "missing_ratio":(df[c].isna().mean() if n else np.nan)} for c in df.columns]).sort_values("missing_ratio", ascending=False)

def numeric_summary(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    num_cols = [c for c in num_cols if not is_bool_dtype(df[c])]
    if not num_cols: return pd.DataFrame()
    desc = df[num_cols].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).T
    desc = desc.rename(columns={"50%":"median"}).reset_index().rename(columns={"index":"column"})
    extra = []
    for c in num_cols:
        s = df[c]
        try:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            iqr_out = int(((s<lower)|(s>upper)).sum())
        except Exception:
            iqr_out = np.nan
        extra.append({
            "column": c, "zero_count": int((s==0).sum()),
            "neg_count": int((s<0).sum()),
            "skewness": float(s.skew(skipna=True)) if s.notna().any() else np.nan,
            "kurtosis": float(s.kurtosis(skipna=True)) if s.notna().any() else np.nan,
            "iqr_outliers": iqr_out
        })
    return desc.merge(pd.DataFrame(extra), on="column", how="left")

def categorical_top_values(df: pd.DataFrame, cat_cols: List[str], topk: int) -> pd.DataFrame:
    rows, total = [], len(df)
    for c in cat_cols:
        vc = safe_value_counts(df[c], normalize=False, dropna=False).head(topk)
        for v, cnt in vc.items():
            rows.append({"column": c, "value": str(v), "count": int(cnt), "ratio": (cnt/total if total else np.nan)})
    return pd.DataFrame(rows)

# ---- DATETIME SUMMARY ----
def datetime_summary(df: pd.DataFrame, dt_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows, long_rows = [], []
    for c in dt_cols:
        s = pd.to_datetime(df[c], errors="coerce")
        if isinstance(s.dtype, DatetimeTZDtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        meta_rows.append({
            "column": c,
            "min": s.min(), "max": s.max(),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True))
        })
        t = s.dropna()
        if not t.empty:
            tmp = pd.DataFrame({
                "year": t.dt.year,
                "month": t.dt.month,
                "dow": t.dt.dayofweek
            })
            agg = tmp.value_counts().reset_index(name="count")
            agg["column"] = c
            long_rows.append(agg[["column", "year", "month", "dow", "count"]])
    meta = pd.DataFrame(meta_rows)
    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(columns=["column","year","month","dow","count"])
    return meta, long

# ===== KORELACJE =====
def compute_correlations(df: pd.DataFrame, num_cols: List[str], corr_max_cols: int) -> Dict[str, pd.DataFrame]:
    keep = [c for c in num_cols if not is_bool_dtype(df[c])]
    if len(keep) == 0:
        return {}
    if len(keep) > corr_max_cols:
        keep = df[keep].var(numeric_only=True).sort_values(ascending=False).head(corr_max_cols).index.tolist()
    sub = df[keep]
    pearson = sub.corr(method="pearson", numeric_only=True)
    spearman = sub.corr(method="spearman", numeric_only=True)
    pairs = []
    cols = pearson.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = pearson.iloc[i, j]
            if pd.notna(r) and abs(r) >= HIGH_CORR_THRESHOLD:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "corr": float(r)})
    out = {
        "pearson": pearson.reset_index().rename(columns={"index": "feature"}),
        "spearman": spearman.reset_index().rename(columns={"index": "feature"}),
        "high_pairs": pd.DataFrame(pairs).sort_values("corr", ascending=False)
    }
    return out

# ===== ETA² – FAST helper =====
def _eta2_fast(cat_codes: np.ndarray, n_groups: int, y: np.ndarray) -> float:
    mask = ~np.isnan(y)
    if mask.sum() == 0 or n_groups <= 1:
        return np.nan
    cc = cat_codes[mask]
    yy = y[mask]
    counts = np.bincount(cc, minlength=n_groups).astype(float)
    sums   = np.bincount(cc, weights=yy, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    m = yy.mean()
    ss_between = np.nansum(counts * (means - m) ** 2)
    ss_total   = np.nansum((yy - m) ** 2)
    if ss_total == 0:
        return np.nan
    return float(ss_between / ss_total)

# ===== Selekcja kolumn do Cramér’s V =====
def _is_excluded_for_v(col: str) -> bool:
    if col in EXCLUDE_FROM_V:
        return True
    return any(col.startswith(p) for p in EXCLUDE_PREFIXES_FROM_V)

def _select_categorical_for_v(df: pd.DataFrame, cat_cols: List[str]) -> List[str]:
    # filtr po nazwach
    candidate_cols = [c for c in cat_cols if not _is_excluded_for_v(c)]
    # filtr po kardynalności
    pairs = []
    for c in candidate_cols:
        try:
            u = safe_nunique(df[c])
        except Exception:
            continue
        if u < CATS_MIN_UNIQUES_FOR_V or u > CATS_MAX_UNIQUES_FOR_V:
            continue
        pairs.append((u, c))
    # sort rosnąco po unikatowych – preferuj mniejsze, szybsze
    pairs.sort(key=lambda x: (x[0], x[1]))
    return [c for _, c in pairs[:CRAMERS_LIMIT_COLS]]

# ===== Cramér’s V + Eta² – FAST =====
def categorical_relationships(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Dict[str, pd.DataFrame]:
    res: Dict[str, pd.DataFrame] = {}

    # 1) Globalny sample wierszy (spójny dla V i eta²)
    if CRAMERS_SAMPLE_ROWS and len(df) > CRAMERS_SAMPLE_ROWS:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(len(df), CRAMERS_SAMPLE_ROWS, replace=False)
        dfx = df.iloc[sample_idx].copy()
    else:
        dfx = df

    # 2) Selekcja kolumn do V (wg reguł)
    selected = _select_categorical_for_v(dfx, cat_cols)

    # 3) Factorize cache
    codes: Dict[str, np.ndarray] = {}
    nunique: Dict[str, int] = {}
    for c in selected:
        s = _series_to_hashable(dfx[c]).fillna("__NA__")
        code, uniques = pd.factorize(s, sort=False)
        codes[c] = code.astype(np.int64, copy=False)
        nunique[c] = int(len(uniques))

    # 4) Cramér’s V (FAST)
    if len(selected) < 2:
        res["cramers_v"] = pd.DataFrame(columns=["cat_a", "cat_b", "cramers_v"])
    else:
        pairs = []
        total_pairs = (len(selected) * (len(selected) - 1)) // 2
        with tqdm(total=total_pairs, desc="Cramér’s V cat↔cat (fast)", unit="para") as t:
            for i in range(len(selected)):
                a = selected[i]
                xa = codes[a]; ra = nunique[a]
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    xb = codes[b]; rb = nunique[b]

                    # limit r*k
                    if ra * rb > CRAMERS_MAX_CELLS:
                        pairs.append({"cat_a": a, "cat_b": b, "cramers_v": np.nan})
                        t.update(1)
                        continue

                    combo = xa * rb + xb
                    counts = np.bincount(combo, minlength=ra * rb).reshape(ra, rb)
                    total = counts.sum()
                    if total == 0:
                        v = np.nan
                    else:
                        row_sums = counts.sum(axis=1, keepdims=True).astype(float)
                        col_sums = counts.sum(axis=0, keepdims=True).astype(float)
                        expected = (row_sums @ col_sums) / float(total)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            chi2 = np.nansum((counts - expected) ** 2 / expected)
                        r1, k1 = ra - 1, rb - 1
                        v = np.sqrt((chi2 / float(total)) / float(min(max(r1, 1), max(k1, 1)))) if r1 > 0 and k1 > 0 else np.nan

                    pairs.append({"cat_a": a, "cat_b": b, "cramers_v": float(v) if pd.notna(v) else np.nan})

                    if t.n % 25 == 0:
                        t.set_postfix_str(f"{a} × {b}")
                    t.update(1)
        res["cramers_v"] = pd.DataFrame(pairs).sort_values("cramers_v", ascending=False)

    # 5) Eta² (FAST, bez groupby) – na tym samym dfx i wybranych kolumnach
    eta_rows = []
    cats_for_eta = selected
    if ETA_SAMPLE_ROWS and len(dfx) > ETA_SAMPLE_ROWS:
        rng = np.random.RandomState(RANDOM_STATE)
        idx_eta = rng.choice(len(dfx), ETA_SAMPLE_ROWS, replace=False)
        dfe = dfx.iloc[idx_eta]
        idx_mask = np.zeros(len(dfx), dtype=bool); idx_mask[idx_eta] = True
    else:
        dfe = dfx
        idx_mask = None

    total_eta = len(cats_for_eta) * len(num_cols)
    if total_eta > 0:
        with tqdm(total=total_eta, desc="Eta² cat→num (fast)", unit="para") as t:
            for c in cats_for_eta:
                cc_full = codes[c]
                ng = nunique[c]
                if ng > ETA_MAX_GROUPS:
                    for n in num_cols:
                        eta_rows.append({"cat": c, "num": n, "eta_squared": np.nan})
                        t.update(1)
                    continue
                cc = cc_full[idx_mask] if idx_mask is not None else cc_full
                for n in num_cols:
                    y = dfe[n].astype("float64").to_numpy(copy=False)
                    val = _eta2_fast(cc, ng, y)
                    eta_rows.append({"cat": c, "num": n, "eta_squared": val})
                    t.update(1)
    res["eta_squared"] = pd.DataFrame(eta_rows).sort_values("eta_squared", ascending=False)

    return res

# ===== EXCEL =====
def _sanitize_sheet_name(name: str) -> str:
    for b in ['\\','/','*','[',']',':','?']:
        name = name.replace(b, '_')
    return name[:31] or "Sheet"

def write_excel(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path) as w:
        for name, d in sheets.items():
            d = d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)
            d = _strip_tz_in_dataframe(d)
            d.to_excel(w, sheet_name=_sanitize_sheet_name(name), index=False)

# ===== MAIN ANALYSIS =====
def analyze_and_export(df: pd.DataFrame, out_dir: str, target_col: Optional[str], corr_max_cols: int, topk_cats: int):
    steps = [
        "00_schema_overview.xlsx",
        "02_missingness.xlsx",
        "03_numeric.xlsx",
        "04_categorical.xlsx",
        "06_datetime.xlsx",
        "07_correlations.xlsx",
        "08_cat_relationships.xlsx",
        "99_samples.xlsx"
    ]
    pbar = tqdm(total=len(steps), desc="Analiza danych", unit="etap")

    write_excel(os.path.join(out_dir, steps[0]), {"dataset_overview": dataset_overview(df), "columns_catalog": columns_catalog(df)}); pbar.update(1)

    split = split_columns_by_type(df)

    write_excel(os.path.join(out_dir, steps[1]), {"missing_by_column": missingness_table(df)}); pbar.update(1)

    num_sum = numeric_summary(df, split["numeric"])
    write_excel(os.path.join(out_dir, steps[2]), {"numeric_summary": num_sum}); pbar.update(1)

    write_excel(os.path.join(out_dir, steps[3]), {
        "categorical_overview": pd.DataFrame([{"column": c, "nunique": safe_nunique(df[c])} for c in split["categorical"]]).sort_values("nunique", ascending=False),
        "top_values_long": categorical_top_values(df, split["categorical"], topk=topk_cats)
    }); pbar.update(1)

    dt_meta, dt_long = datetime_summary(df, split["datetime"])
    write_excel(os.path.join(out_dir, steps[4]), {"datetime_meta": dt_meta, "year_month_dow": dt_long}); pbar.update(1)

    corrs = compute_correlations(df, split["numeric"], corr_max_cols)
    if corrs:
        write_excel(os.path.join(out_dir, steps[5]), corrs)
    pbar.update(1)

    cat_rels = categorical_relationships(df, split["categorical"], split["numeric"])
    write_excel(os.path.join(out_dir, steps[6]), cat_rels); pbar.update(1)

    write_excel(os.path.join(out_dir, steps[7]), {"head": df.head(100), "random_sample": df.sample(min(1000, len(df)), random_state=RANDOM_STATE)}); pbar.update(1)

    pbar.close()

# ===== SAVE PICKLE =====
def save_pickle(df: pd.DataFrame, out_dir: str, name: str) -> str:
    out = os.path.join(out_dir, name)
    with open(out, "wb") as f:
        pickle.dump(df, f)
    return out

# ===== MAIN =====
def main():
    # --- KONFIG DO URUCHOMIENIA ---
    INPUT_PATH = "4.current_properties_after_2022_preprocessed_final.pkl"
    OUTPUT_ROOT = "output_analysis_properties"
    PARSE_DATES = True
    TARGET_COL = None
    CORR_MAX_COLS = DEFAULT_CORR_MAX_COLS
    TOPK_CATS = DEFAULT_TOPK_CATS
    PICKLE_NAME = DEFAULT_PICKLE_NAME

    print(f"[INFO] Wczytywanie: {INPUT_PATH}")
    df = load_dataframe(INPUT_PATH)
    if PARSE_DATES:
        print("[INFO] Parsuję potencjalne kolumny dat…")
        df = infer_and_parse_datetimes(df)

    out_dir = ensure_output_dir(OUTPUT_ROOT, INPUT_PATH)
    print(f"[INFO] Katalog wyjściowy: {out_dir}")

    analyze_and_export(df, out_dir, TARGET_COL, CORR_MAX_COLS, TOPK_CATS)

    write_excel_with_examples(os.path.join(out_dir, "00_columns_overview.xlsx"), df)

    pkl_path = save_pickle(df, out_dir, PICKLE_NAME)
    print(f"[INFO] Zapisano Pickle: {pkl_path}")

    manifest = {
        "input": os.path.abspath(INPUT_PATH),
        "output_dir": os.path.abspath(out_dir),
        "created_at": datetime.now().isoformat(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "pickle_path": os.path.abspath(pkl_path)
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[DONE] Gotowe. Kluczowy plik: 00_schema_overview.xlsx")

if __name__ == "__main__":
    main()
