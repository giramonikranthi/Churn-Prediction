import json
from pathlib import Path
import pandas as pd


CUSTOMER_ID_ALIASES = {
    "customer_id",
    "customerid",
    "customer id",
    "cust_id",
    "custid",
    "id",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column labels and map common customer id aliases."""
    if df.empty:
        return df

    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]

    if "customer_id" not in normalized.columns:
        for col in normalized.columns:
            if str(col).strip().lower() in CUSTOMER_ID_ALIASES:
                normalized = normalized.rename(columns={col: "customer_id"})
                break

    return normalized


def _ensure_customer_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure customer_id exists and is non-empty to keep UI selection stable."""
    if df.empty:
        return df

    result = df.copy()
    if "customer_id" not in result.columns:
        result["customer_id"] = [f"ROW-{idx + 1:06d}" for idx in range(len(result))]
        return result

    # Fill missing/blank ids with deterministic row-based fallback ids.
    raw_series = result["customer_id"]
    existing = raw_series.astype(str).str.strip()
    missing_mask = raw_series.isna() | (existing == "") | (existing.str.lower() == "nan")
    if missing_mask.any():
        fallback_values = []
        for row_pos in range(len(result)):
            fallback_values.append(f"ROW-{row_pos + 1:06d}")
        result.loc[missing_mask, "customer_id"] = pd.Series(fallback_values, index=result.index)[missing_mask]

    return result


def _read_json_file(file_path: Path) -> pd.DataFrame:
    with file_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if isinstance(payload, dict):
        # Support common wrapped JSON structures.
        for key in ["data", "records", "customers", "items", "rows"]:
            nested = payload.get(key)
            if isinstance(nested, list):
                return pd.DataFrame(nested)
        return pd.DataFrame([payload])

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    return pd.DataFrame()


def list_available_files(data_dir: Path, file_patterns: list[str]) -> list[Path]:
    """Return sorted list of matching files in data_dir."""
    files = []
    for pattern in file_patterns:
        files.extend(list(Path(data_dir).glob(pattern)))
    return sorted(files)


def load_single_file(file_path: Path) -> pd.DataFrame:
    """Load a single CSV, XLSX, or JSON file into a DataFrame."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        loaded = pd.read_csv(file_path)
    elif suffix == ".xlsx":
        try:
            loaded = pd.read_excel(file_path)
        except ImportError as exc:
            raise ImportError(
                "Reading .xlsx files requires 'openpyxl'. Add it to requirements.txt."
            ) from exc
    elif suffix == ".json":
        loaded = _read_json_file(file_path)
    else:
        return pd.DataFrame()

    loaded = _normalize_columns(loaded)
    loaded = _ensure_customer_id_column(loaded)
    return loaded


def load_local_customer_data(data_dir: Path, file_patterns: list[str]) -> pd.DataFrame:
    """Load the local customer dataset from CSV, XLSX, or JSON files."""
    dataset_files = list_available_files(data_dir, file_patterns)

    if not dataset_files:
        return pd.DataFrame()

    dataframes = []
    for file_path in dataset_files:
        frame = load_single_file(file_path)
        if not frame.empty:
            dataframes.append(frame)

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True, sort=False)


def get_customer_record(df: pd.DataFrame, customer_id: str) -> pd.DataFrame | None:
    """Return the selected customer row as a dataframe."""
    if df.empty or "customer_id" not in df.columns:
        return None

    match = df.loc[df["customer_id"].astype(str).str.lower() == customer_id.lower()]
    return match.head(1).copy()
