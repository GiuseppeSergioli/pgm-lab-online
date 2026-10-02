"""Curated numerical classification datasets suitable for PGM experiments.

Every OpenML entry is pinned by immutable ``data_id`` and its expected shape is
checked after download.  This prevents a silently changed dataset from entering
an experiment.  The three datasets bundled with scikit-learn are loaded locally
to keep the application useful even when OpenML is temporarily unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    openml_id: int
    samples: int
    features: int
    classes: int
    domain: str
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
        domain="Medicina",
        source_url="https://archive.ics.uci.edu/dataset/43/haberman+s+survival",
    ),
    DatasetSpec(
        key="balance_scale",
        display_name="Balance Scale",
        openml_id=11,
        samples=625,
        features=4,
        classes=3,
        domain="Sintetico",
        source_url="https://archive.ics.uci.edu/dataset/12/balance+scale",
    ),
    DatasetSpec(
        key="iris",
        display_name="Iris",
        openml_id=61,
        samples=150,
        features=4,
        classes=3,
        domain="Botanica",
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
        domain="Bioinformatica",
        source_url="https://archive.ics.uci.edu/dataset/39/ecoli",
    ),
    DatasetSpec(
        key="glass",
        display_name="Glass Identification",
        openml_id=41,
        samples=214,
        features=9,
        classes=6,
        domain="Materiali",
        source_url="https://archive.ics.uci.edu/dataset/42/glass+identification",
    ),
    DatasetSpec(
        key="wine",
        display_name="Wine",
        openml_id=187,
        samples=178,
        features=13,
        classes=3,
        domain="Chimica",
        source_url="https://archive.ics.uci.edu/dataset/109/wine",
        builtin_fallback=load_wine,
    ),
    DatasetSpec(
        key="breast_cancer_diagnostic",
        display_name="Breast Cancer Wisconsin (Diagnostic)",
        openml_id=1510,
        samples=569,
        features=30,
        classes=2,
        domain="Medicina",
        source_url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        builtin_fallback=load_breast_cancer,
    ),
    DatasetSpec(
        key="breast_cancer_original",
        display_name="Breast Cancer Wisconsin (Original)",
        openml_id=15,
        samples=699,
        features=9,
        classes=2,
        domain="Medicina",
        source_url="https://www.openml.org/d/15",
    ),
    DatasetSpec(
        key="diabetes",
        display_name="Pima Indians Diabetes",
        openml_id=37,
        samples=768,
        features=8,
        classes=2,
        domain="Medicina",
        source_url="https://www.openml.org/d/37",
    ),
    DatasetSpec(
        key="heart_statlog",
        display_name="Heart Statlog",
        openml_id=53,
        samples=270,
        features=13,
        classes=2,
        domain="Medicina",
        source_url="https://www.openml.org/d/53",
    ),
    DatasetSpec(
        key="vehicle",
        display_name="Vehicle Silhouettes",
        openml_id=54,
        samples=846,
        features=18,
        classes=4,
        domain="Visione artificiale",
        source_url="https://www.openml.org/d/54",
    ),
    DatasetSpec(
        key="zoo",
        display_name="Zoo",
        openml_id=62,
        samples=101,
        features=16,
        classes=7,
        domain="Zoologia",
        source_url="https://www.openml.org/d/62",
    ),
    DatasetSpec(
        key="ionosphere",
        display_name="Ionosphere",
        openml_id=59,
        samples=351,
        features=34,
        classes=2,
        domain="Radar",
        source_url="https://www.openml.org/d/59",
    ),
    DatasetSpec(
        key="sonar",
        display_name="Sonar Mines vs Rocks",
        openml_id=40,
        samples=208,
        features=60,
        classes=2,
        domain="Segnali sonar",
        source_url="https://www.openml.org/d/40",
    ),
    DatasetSpec(
        key="kc2",
        display_name="KC2 Software Defect",
        openml_id=1063,
        samples=522,
        features=21,
        classes=2,
        domain="Ingegneria software",
        source_url="https://www.openml.org/d/1063",
    ),
    DatasetSpec(
        key="banknote",
        display_name="Banknote Authentication",
        openml_id=1462,
        samples=1372,
        features=4,
        classes=2,
        domain="Autenticazione",
        source_url="https://www.openml.org/d/1462",
    ),
    DatasetSpec(
        key="blood_transfusion",
        display_name="Blood Transfusion Service Center",
        openml_id=1464,
        samples=748,
        features=4,
        classes=2,
        domain="Donazioni di sangue",
        source_url="https://www.openml.org/d/1464",
    ),
    DatasetSpec(
        key="climate_crashes",
        display_name="Climate Model Simulation Crashes",
        openml_id=1467,
        samples=540,
        features=20,
        classes=2,
        domain="Climatologia",
        source_url="https://www.openml.org/d/1467",
    ),
    DatasetSpec(
        key="parkinsons",
        display_name="Parkinsons",
        openml_id=1488,
        samples=195,
        features=22,
        classes=2,
        domain="Medicina",
        source_url="https://www.openml.org/d/1488",
    ),
    DatasetSpec(
        key="seeds",
        display_name="Seeds",
        openml_id=1499,
        samples=210,
        features=7,
        classes=3,
        domain="Agricoltura",
        source_url="https://www.openml.org/d/1499",
    ),
)


def _validate_catalog() -> None:
    keys = [dataset.key for dataset in DATASETS]
    if len(keys) != len(set(keys)):
        raise ValueError("Il catalogo contiene chiavi dataset duplicate.")
    for dataset in DATASETS:
        if min(dataset.samples, dataset.features, dataset.classes) < 1:
            raise ValueError(f"Metadati non validi per il dataset {dataset.key}.")


_validate_catalog()


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
                "Ambito": dataset.domain,
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
    """Load a pinned public dataset and enforce its catalogued metadata."""

    spec = get_dataset_spec(key)
    # scikit-learn distributes exact local copies of these public UCI datasets.
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
