"""Curated low-dimensional public datasets suitable for explicit c-PGM runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_iris, load_wine


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    openml_id: int
    samples: int
    features: int
    classes: int
    source_url: str
    builtin_fallback: Callable | None = None


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="haberman",
        display_name="Haberman Survival",
        openml_id=43,
        samples=306,
        features=3,
        classes=2,
        source_url="https://archive.ics.uci.edu/dataset/43/haberman+s+survival",
    ),
    DatasetSpec(
        key="balance_scale",
        display_name="Balance Scale",
        openml_id=11,
        samples=625,
        features=4,
        classes=3,
        source_url="https://archive.ics.uci.edu/dataset/12/balance+scale",
    ),
    DatasetSpec(
        key="iris",
        display_name="Iris",
        openml_id=61,
        samples=150,
        features=4,
        classes=3,
        source_url="https://archive.ics.uci.edu/dataset/53/iris",
        builtin_fallback=load_iris,
    ),
    DatasetSpec(
        key="ecoli",
        display_name="Ecoli",
        openml_id=39,
        samples=336,
        features=7,
        classes=8,
        source_url="https://archive.ics.uci.edu/dataset/39/ecoli",
    ),
    DatasetSpec(
        key="glass",
        display_name="Glass Identification",
        openml_id=41,
        samples=214,
        features=9,
        classes=6,
        source_url="https://archive.ics.uci.edu/dataset/42/glass+identification",
    ),
    DatasetSpec(
        key="wine",
        display_name="Wine",
        openml_id=187,
        samples=178,
        features=13,
        classes=3,
        source_url="https://archive.ics.uci.edu/dataset/109/wine",
        builtin_fallback=load_wine,
    ),
)


def get_dataset_spec(key: str) -> DatasetSpec:
    for dataset in DATASETS:
        if dataset.key == key:
            return dataset
    raise KeyError(f"Dataset sconosciuto: {key}")


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Dataset": dataset.display_name,
                "Campioni": dataset.samples,
                "Feature/campione": dataset.features,
                "Classi": dataset.classes,
                "Repository": dataset.source_url,
            }
            for dataset in DATASETS
        ]
    )


def _from_builtin(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    if spec.builtin_fallback is None:
        raise RuntimeError("Nessun fallback locale disponibile.")
    bunch = spec.builtin_fallback(as_frame=True)
    X = bunch.data.copy()
    target = bunch.target
    if hasattr(bunch, "target_names"):
        names = np.asarray(bunch.target_names)
        y = pd.Series(names[np.asarray(target, dtype=int)], name="target")
    else:
        y = pd.Series(target, name="target")
    return X, y


def load_public_dataset(key: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """Download from OpenML, with an offline fallback for Iris and Wine."""

    spec = get_dataset_spec(key)
    # scikit-learn distributes exact local copies of these two public UCI datasets.
    # Prefer them to avoid a needless network round-trip; all other entries are
    # fetched by immutable OpenML data_id.
    if spec.builtin_fallback is not None:
        X, y = _from_builtin(spec)
        source = "scikit-learn locale (copia del dataset pubblico UCI)"
        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.Series(y, name="target").astype(str)
        if X.shape != (spec.samples, spec.features) or y.nunique() != spec.classes:
            raise ValueError("La copia locale non coincide con i metadati del catalogo.")
        return X.reset_index(drop=True), y.reset_index(drop=True), source

    try:
        bunch = fetch_openml(data_id=spec.openml_id, as_frame=True, parser="auto")
        X = bunch.data.copy()
        y_raw = bunch.target
        if isinstance(y_raw, pd.DataFrame):
            if y_raw.shape[1] != 1:
                raise ValueError("Il dataset ha piu di una variabile target.")
            y = y_raw.iloc[:, 0].copy()
        else:
            y = pd.Series(y_raw).copy()
        source = f"OpenML data_id={spec.openml_id}"
    except Exception as openml_error:
        raise RuntimeError(
            "Download OpenML non riuscito. Controllare la connessione e riprovare. "
            f"Dettaglio originale: {openml_error}"
        ) from openml_error

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.Series(y, name="target")
    valid_target = y.notna().to_numpy()
    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(str).reset_index(drop=True)
    if X.shape[0] != spec.samples:
        raise ValueError(
            f"Metadati inattesi: attesi {spec.samples} campioni, ricevuti {X.shape[0]}."
        )
    if X.shape[1] != spec.features:
        raise ValueError(
            f"Metadati inattesi: attese {spec.features} feature, ricevute {X.shape[1]}."
        )
    if y.nunique() != spec.classes:
        raise ValueError(
            f"Metadati inattesi: attese {spec.classes} classi, ricevute {y.nunique()}."
        )
    return X, y, source
