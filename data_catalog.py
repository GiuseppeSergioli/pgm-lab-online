"""Curated numerical classification datasets suitable for PGM experiments.

Every OpenML entry is pinned by immutable ``data_id`` and its expected shape is
checked after download.  This prevents a silently changed dataset from entering
an experiment.  Three datasets bundled with scikit-learn and five reproducible
synthetic benchmarks are loaded locally, so they do not depend on OpenML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import Bunch


SYNTHETIC_SEED = 42
SYNTHETIC_SAMPLES = 600


def _synthetic_bunch(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...] = ("x1", "x2"),
) -> Bunch:
    """Return generated data through the same interface as sklearn loaders."""

    labels = np.unique(y)
    return Bunch(
        data=pd.DataFrame(X, columns=list(feature_names)),
        target=pd.Series(y, name="target"),
        target_names=np.asarray([f"classe {label}" for label in labels]),
    )


def _load_two_moons(*, as_frame: bool = True) -> Bunch:
    del as_frame
    X, y = make_moons(
        n_samples=SYNTHETIC_SAMPLES,
        noise=0.16,
        random_state=SYNTHETIC_SEED,
    )
    return _synthetic_bunch(X, y)


def _load_concentric_circles(*, as_frame: bool = True) -> Bunch:
    del as_frame
    X, y = make_circles(
        n_samples=SYNTHETIC_SAMPLES,
        noise=0.08,
        factor=0.45,
        random_state=SYNTHETIC_SEED,
    )
    return _synthetic_bunch(X, y)


def _load_gaussian_blobs(*, as_frame: bool = True) -> Bunch:
    del as_frame
    X, y = make_blobs(
        n_samples=SYNTHETIC_SAMPLES,
        centers=((-2.2, -1.6), (2.2, -1.4), (0.0, 2.4)),
        cluster_std=0.75,
        random_state=SYNTHETIC_SEED,
    )
    return _synthetic_bunch(X, y)


def _load_xor(*, as_frame: bool = True) -> Bunch:
    del as_frame
    rng = np.random.default_rng(SYNTHETIC_SEED)
    latent = rng.uniform(-1.0, 1.0, size=(SYNTHETIC_SAMPLES, 2))
    y = np.logical_xor(latent[:, 0] >= 0.0, latent[:, 1] >= 0.0).astype(int)
    X = latent + rng.normal(0.0, 0.08, size=latent.shape)
    return _synthetic_bunch(X, y)


def _load_three_spirals(*, as_frame: bool = True) -> Bunch:
    del as_frame
    rng = np.random.default_rng(SYNTHETIC_SEED)
    samples_per_class = SYNTHETIC_SAMPLES // 3
    base_angle = np.linspace(0.35, 3.7 * np.pi, samples_per_class)
    radius = np.linspace(0.08, 1.0, samples_per_class)
    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for class_index in range(3):
        angle = (
            base_angle
            + class_index * 2.0 * np.pi / 3.0
            + rng.normal(0.0, 0.10, size=samples_per_class)
        )
        noisy_radius = radius + rng.normal(0.0, 0.025, size=samples_per_class)
        blocks.append(
            np.column_stack(
                (noisy_radius * np.cos(angle), noisy_radius * np.sin(angle))
            )
        )
        targets.append(np.full(samples_per_class, class_index, dtype=int))
    X = np.vstack(blocks)
    y = np.concatenate(targets)
    return _synthetic_bunch(X, y)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    openml_id: int | None
    samples: int
    features: int
    classes: int
    domain: str
    source_url: str
    builtin_fallback: Callable | None = None
    local_source: str | None = None
    raw_samples: int | None = None
    subsample_seed: int | None = None


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
    DatasetSpec(
        key="two_moons",
        display_name="Two Moons (sintetico)",
        openml_id=None,
        samples=SYNTHETIC_SAMPLES,
        features=2,
        classes=2,
        domain="Benchmark sintetico",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html",
        builtin_fallback=_load_two_moons,
        local_source="scikit-learn make_moons: n=600, noise=0.16, seed=42",
    ),
    DatasetSpec(
        key="banana",
        display_name="Banana OpenML (sottoinsieme fisso)",
        openml_id=1460,
        samples=800,
        features=2,
        classes=2,
        domain="Benchmark sintetico",
        source_url="https://www.openml.org/d/1460",
        raw_samples=5300,
        subsample_seed=42,
    ),
    DatasetSpec(
        key="concentric_circles",
        display_name="Concentric Circles (sintetico)",
        openml_id=None,
        samples=SYNTHETIC_SAMPLES,
        features=2,
        classes=2,
        domain="Benchmark sintetico",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html",
        builtin_fallback=_load_concentric_circles,
        local_source="scikit-learn make_circles: n=600, noise=0.08, factor=0.45, seed=42",
    ),
    DatasetSpec(
        key="gaussian_blobs",
        display_name="Gaussian Blobs, 3 classi (sintetico)",
        openml_id=None,
        samples=SYNTHETIC_SAMPLES,
        features=2,
        classes=3,
        domain="Benchmark sintetico",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html",
        builtin_fallback=_load_gaussian_blobs,
        local_source="scikit-learn make_blobs: n=600, 3 centri, std=0.75, seed=42",
    ),
    DatasetSpec(
        key="xor",
        display_name="XOR rumoroso (sintetico)",
        openml_id=None,
        samples=SYNTHETIC_SAMPLES,
        features=2,
        classes=2,
        domain="Benchmark sintetico",
        source_url="https://github.com/GiuseppeSergioli/pgm-lab-online/blob/main/data_catalog.py",
        builtin_fallback=_load_xor,
        local_source="generatore XOR locale: n=600, rumore gaussiano=0.08, seed=42",
    ),
    DatasetSpec(
        key="three_spirals",
        display_name="Three Spirals (sintetico)",
        openml_id=None,
        samples=SYNTHETIC_SAMPLES,
        features=2,
        classes=3,
        domain="Benchmark sintetico",
        source_url="https://github.com/GiuseppeSergioli/pgm-lab-online/blob/main/data_catalog.py",
        builtin_fallback=_load_three_spirals,
        local_source="generatore Three Spirals locale: n=600, 3 classi, seed=42",
    ),
)


def _validate_catalog() -> None:
    keys = [dataset.key for dataset in DATASETS]
    if len(keys) != len(set(keys)):
        raise ValueError("Il catalogo contiene chiavi dataset duplicate.")
    for dataset in DATASETS:
        if min(dataset.samples, dataset.features, dataset.classes) < 1:
            raise ValueError(f"Metadati non validi per il dataset {dataset.key}.")
        if dataset.openml_id is None and dataset.builtin_fallback is None:
            raise ValueError(f"Nessun loader configurato per il dataset {dataset.key}.")
        if dataset.raw_samples is not None:
            if dataset.raw_samples < dataset.samples:
                raise ValueError(f"Sottoinsieme non valido per {dataset.key}.")
            if dataset.subsample_seed is None:
                raise ValueError(f"Seed del sottoinsieme mancante per {dataset.key}.")


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


def _from_local_loader(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
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


def _fixed_stratified_subset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select the same class-stratified rows on every run."""

    selector = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed)
    selected_indices, _ = next(selector.split(X, y))
    selected_indices = np.sort(selected_indices)
    return (
        X.iloc[selected_indices].reset_index(drop=True),
        y.iloc[selected_indices].reset_index(drop=True),
    )


def load_public_dataset(key: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load a pinned public dataset and enforce its catalogued metadata."""

    spec = get_dataset_spec(key)
    # This branch covers both scikit-learn's local UCI copies and the deterministic
    # synthetic generators. All remaining entries use immutable OpenML data_id.
    if spec.builtin_fallback is not None:
        X, y = _from_local_loader(spec)
        source = spec.local_source or "scikit-learn locale (copia del dataset pubblico UCI)"
        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.Series(y, name="target").astype(str)
        if X.shape != (spec.samples, spec.features) or y.nunique() != spec.classes:
            raise ValueError("La copia locale non coincide con i metadati del catalogo.")
        return X.reset_index(drop=True), y.reset_index(drop=True), source

    try:
        if spec.openml_id is None:
            raise RuntimeError("Identificativo OpenML mancante.")
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
    expected_downloaded_samples = spec.raw_samples or spec.samples
    if X.shape[0] != expected_downloaded_samples:
        raise ValueError(
            "Metadati inattesi: attesi "
            f"{expected_downloaded_samples} campioni, ricevuti {X.shape[0]}."
        )
    if X.shape[1] != spec.features:
        raise ValueError(
            f"Metadati inattesi: attese {spec.features} feature, ricevute {X.shape[1]}."
        )
    if y.nunique() != spec.classes:
        raise ValueError(
            f"Metadati inattesi: attese {spec.classes} classi, ricevute {y.nunique()}."
        )
    if spec.raw_samples is not None:
        if spec.subsample_seed is None:
            raise RuntimeError("Seed del sottoinsieme mancante.")
        X, y = _fixed_stratified_subset(
            X,
            y,
            size=spec.samples,
            seed=spec.subsample_seed,
        )
        source += (
            f"; sottoinsieme stratificato fisso {spec.samples}/{spec.raw_samples}, "
            f"seed={spec.subsample_seed}"
        )
    if X.shape != (spec.samples, spec.features):
        raise ValueError("Il dataset elaborato non coincide con i metadati del catalogo.")
    return X, y, source
