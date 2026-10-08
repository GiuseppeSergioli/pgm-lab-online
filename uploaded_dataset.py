"""Safe, deterministic ingestion of private tabular classification datasets."""

from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO, StringIO
from math import ceil
from pathlib import PurePath
import re
from typing import Any, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_ROWS = 50_000
MAX_UPLOAD_COLUMNS = 500
MAX_EXCEL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_EXCEL_ARCHIVE_MEMBERS = 10_000
MAX_TARGET_CLASSES = 50
MIN_SAMPLES_PER_CLASS = 3
SUPPORTED_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx")
TARGET_NAME_HINTS = (
    "target",
    "label",
    "class",
    "classe",
    "y",
    "outcome",
    "response",
    "diagnosis",
    "species",
)
MISSING_MARKERS = ("?", "NA", "N/A", "null", "NULL", "None", "")


class UploadedDatasetError(ValueError):
    """User-facing validation failure for an uploaded table."""


def safe_uploaded_name(file_name: str) -> str:
    """Return a short basename without trusting client-provided paths."""

    basename = PurePath(str(file_name).replace("\\", "/")).name.strip()
    if not basename:
        return "dataset_caricato"
    return basename[:160]


def display_name_from_file(file_name: str) -> str:
    stem = PurePath(safe_uploaded_name(file_name)).stem
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return (cleaned or "Dataset caricato")[:80]


def sanitize_display_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    if not cleaned:
        raise UploadedDatasetError("Inserisci un nome per il dataset.")
    return cleaned[:80]


def _decode_text(content: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UploadedDatasetError("La codifica testuale del file non è riconosciuta.")


def _detect_delimiter(text: str, suffix: str) -> str:
    if suffix == ".tsv":
        return "\t"
    sample = "\n".join(text.splitlines()[:40])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {delimiter: sample.count(delimiter) for delimiter in ",;\t|"}
        delimiter = max(counts, key=counts.get)
        if counts[delimiter] == 0:
            raise UploadedDatasetError(
                "Non è stato riconosciuto un separatore tra le colonne."
            )
        return delimiter


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise UploadedDatasetError("Il file non contiene una tabella valida.")
    # Preserve named empty columns so the UI can explain whether an empty target
    # or an empty predictor was supplied instead of silently changing the schema.
    cleaned = frame.dropna(axis=0, how="all").copy()
    if cleaned.empty:
        raise UploadedDatasetError("Il file non contiene righe utilizzabili.")
    if cleaned.shape[1] < 2:
        raise UploadedDatasetError(
            "Servono almeno una feature e una colonna target."
        )
    if cleaned.shape[0] > MAX_UPLOAD_ROWS:
        raise UploadedDatasetError(
            f"Il file contiene {cleaned.shape[0]:,} righe; il limite è "
            f"{MAX_UPLOAD_ROWS:,}."
        )
    if cleaned.shape[1] > MAX_UPLOAD_COLUMNS:
        raise UploadedDatasetError(
            f"Il file contiene {cleaned.shape[1]:,} colonne; il limite è "
            f"{MAX_UPLOAD_COLUMNS:,}."
        )

    columns = [str(column).strip() for column in cleaned.columns]
    if any(not column for column in columns):
        raise UploadedDatasetError("Una o più colonne non hanno un nome.")
    folded = [column.casefold() for column in columns]
    duplicates = sorted(
        {columns[index] for index, value in enumerate(folded) if folded.count(value) > 1}
    )
    if duplicates:
        raise UploadedDatasetError(
            "I nomi delle colonne devono essere univoci. Duplicati: "
            + ", ".join(duplicates[:8])
        )
    cleaned.columns = columns
    return cleaned.reset_index(drop=True)


def read_uploaded_table(content: bytes, file_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse a bounded CSV/TSV/TXT/XLSX file without writing it to disk."""

    if not isinstance(content, bytes) or not content:
        raise UploadedDatasetError("Il file caricato è vuoto.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadedDatasetError(
            f"Il file supera il limite di {MAX_UPLOAD_BYTES // 1024**2} MB."
        )
    safe_name = safe_uploaded_name(file_name)
    suffix = PurePath(safe_name).suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UploadedDatasetError(
            "Formato non supportato. Usa CSV, TSV, TXT oppure XLSX."
        )

    metadata: dict[str, Any] = {
        "file_name": safe_name,
        "file_size": len(content),
        "digest": sha256(content).hexdigest(),
        "format": suffix.removeprefix(".").upper(),
    }
    try:
        if suffix == ".xlsx":
            try:
                with ZipFile(BytesIO(content)) as workbook_archive:
                    members = workbook_archive.infolist()
                    if len(members) > MAX_EXCEL_ARCHIVE_MEMBERS:
                        raise UploadedDatasetError(
                            "Il file Excel contiene troppi elementi interni."
                        )
                    uncompressed_size = sum(member.file_size for member in members)
                    if uncompressed_size > MAX_EXCEL_UNCOMPRESSED_BYTES:
                        raise UploadedDatasetError(
                            "Il contenuto Excel decompresso supera il limite di sicurezza."
                        )
            except BadZipFile as error:
                raise UploadedDatasetError("Il file XLSX non è un archivio Excel valido.") from error
            frame = pd.read_excel(BytesIO(content), sheet_name=0)
            metadata.update({"sheet": "prima", "encoding": None, "delimiter": None})
        else:
            text, encoding = _decode_text(content)
            delimiter = _detect_delimiter(text, suffix)
            decimal = "," if delimiter == ";" else "."
            frame = pd.read_csv(
                StringIO(text),
                sep=delimiter,
                decimal=decimal,
                engine="python",
                na_values=list(MISSING_MARKERS),
                keep_default_na=True,
                on_bad_lines="error",
            )
            metadata.update(
                {"encoding": encoding, "delimiter": delimiter, "decimal": decimal}
            )
    except UploadedDatasetError:
        raise
    except Exception as error:
        raise UploadedDatasetError(
            "Non è stato possibile leggere la tabella. Controlla intestazioni, "
            f"separatore e formato. Dettaglio: {error}"
        ) from error

    cleaned = _clean_frame(frame)
    metadata.update({"rows": int(cleaned.shape[0]), "columns": int(cleaned.shape[1])})
    return cleaned, metadata


def suggest_target_column(frame: pd.DataFrame) -> str:
    by_name = {str(column).strip().casefold(): str(column) for column in frame.columns}
    for hint in TARGET_NAME_HINTS:
        if hint in by_name:
            return by_name[hint]
    return str(frame.columns[-1])


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).astype(float)


def inspect_feature_columns(
    frame: pd.DataFrame,
    target_column: str,
) -> dict[str, tuple[str, ...]]:
    """Classify predictors without using the target values."""

    if target_column not in frame.columns:
        raise UploadedDatasetError("La colonna target selezionata non esiste.")
    numeric: list[str] = []
    non_numeric: list[str] = []
    empty_or_constant: list[str] = []
    probable_identifiers: list[str] = []
    for column in frame.columns:
        column = str(column)
        if column == target_column:
            continue
        original = frame[column]
        converted = _coerce_numeric(original)
        observed_count = int(original.notna().sum())
        conversion_ratio = (
            float(converted.notna().sum()) / observed_count
            if observed_count
            else 0.0
        )
        if conversion_ratio < 0.95:
            non_numeric.append(column)
            continue
        if converted.nunique(dropna=True) <= 1:
            empty_or_constant.append(column)
            continue
        numeric.append(column)
        normalized_name = re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_")
        looks_like_id = bool(
            re.search(r"(^id$|_id$|^index$|^row_?id$|uuid|identifier|codice)", normalized_name)
        )
        uniqueness = converted.nunique(dropna=True) / max(1, converted.notna().sum())
        if looks_like_id and uniqueness >= 0.90:
            probable_identifiers.append(column)
    recommended = [column for column in numeric if column not in probable_identifiers]
    return {
        "numeric": tuple(numeric),
        "recommended": tuple(recommended),
        "non_numeric": tuple(non_numeric),
        "empty_or_constant": tuple(empty_or_constant),
        "probable_identifiers": tuple(probable_identifiers),
    }


def _clean_target_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value)


def prepare_uploaded_dataset(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return validated numeric predictors and labels for supervised learning."""

    if target_column not in frame.columns:
        raise UploadedDatasetError("La colonna target selezionata non esiste.")
    selected = tuple(dict.fromkeys(str(column) for column in feature_columns))
    if not selected:
        raise UploadedDatasetError("Seleziona almeno una feature numerica.")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise UploadedDatasetError(
            "Feature non presenti nel file: " + ", ".join(missing[:8])
        )
    if target_column in selected:
        raise UploadedDatasetError("La colonna target non può essere usata come feature.")

    target_raw = frame[target_column]
    valid_target = target_raw.notna() & target_raw.map(
        lambda value: bool(_clean_target_value(value))
    )
    removed_target_rows = int((~valid_target).sum())
    filtered = frame.loc[valid_target].reset_index(drop=True)
    if filtered.empty:
        raise UploadedDatasetError(
            "La colonna target non contiene alcuna etichetta valida."
        )
    y = filtered[target_column].map(_clean_target_value).rename("target")
    if int(y.map(len).max()) > 120:
        raise UploadedDatasetError(
            "Le etichette del target non possono superare 120 caratteri."
        )

    converted: dict[str, pd.Series] = {}
    dropped_features: list[str] = []
    for column in selected:
        values = _coerce_numeric(filtered[column])
        if values.nunique(dropna=True) <= 1:
            dropped_features.append(column)
            continue
        converted[column] = values.reset_index(drop=True)
    if not converted:
        raise UploadedDatasetError(
            "Dopo la pulizia non rimane alcuna feature numerica variabile."
        )
    X = pd.DataFrame(converted)

    class_counts = y.value_counts(dropna=False)
    class_count = int(len(class_counts))
    if class_count < 2:
        raise UploadedDatasetError("Il target deve contenere almeno due classi.")
    if class_count > MAX_TARGET_CLASSES:
        raise UploadedDatasetError(
            f"Il target contiene {class_count} classi; il limite è {MAX_TARGET_CLASSES}."
        )
    rare = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
    if not rare.empty:
        preview = ", ".join(f"{label}: {count}" for label, count in rare.items())
        raise UploadedDatasetError(
            "Ogni classe deve avere almeno "
            f"{MIN_SAMPLES_PER_CLASS} campioni per gli split stratificati. "
            f"Classi insufficienti: {preview}."
        )
    if len(y) < max(10, 2 * class_count):
        raise UploadedDatasetError(
            "Il dataset è troppo piccolo per una valutazione train/test affidabile."
        )

    metadata = {
        "samples": int(len(y)),
        "features": int(X.shape[1]),
        "classes": class_count,
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "removed_missing_target_rows": removed_target_rows,
        "dropped_constant_features": tuple(dropped_features),
        "missing_feature_values": int(X.isna().sum().sum()),
    }
    return X, y.reset_index(drop=True), metadata


def uploaded_dataset_key(
    content_digest: str,
    target_column: str,
    feature_columns: Sequence[str],
) -> str:
    configuration = "\0".join(
        (str(content_digest), str(target_column), *(str(value) for value in feature_columns))
    )
    return "uploaded_" + sha256(configuration.encode("utf-8")).hexdigest()[:16]


def valid_stratified_test_fractions(
    sample_count: int,
    class_count: int,
    candidates: Sequence[float],
) -> tuple[float, ...]:
    """Return fractions that can place every class in both stratified partitions."""

    samples = int(sample_count)
    classes = int(class_count)
    if samples < 2 or classes < 2:
        return ()
    valid: list[float] = []
    for candidate in candidates:
        fraction = float(candidate)
        if not 0.0 < fraction < 1.0:
            continue
        test_samples = ceil(samples * fraction)
        train_samples = samples - test_samples
        if test_samples >= classes and train_samples >= classes:
            valid.append(fraction)
    return tuple(valid)


def spreadsheet_safe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a table while neutralizing spreadsheet-formula text cells."""

    def escape(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value

    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column].dtype) or pd.api.types.is_string_dtype(
            safe[column].dtype
        ):
            safe[column] = safe[column].map(escape)
    return safe.to_csv(index=False).encode("utf-8")
