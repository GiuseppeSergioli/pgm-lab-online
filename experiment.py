"""End-to-end experiment pipeline, kept independent from the Streamlit UI."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

from data_catalog import load_public_dataset
from pgm_core import PriorMode, run_all_methods


def run_experiment(
    dataset_key: str,
    copies: int,
    test_fraction: float = 0.20,
    random_seed: int = 42,
    prior_mode: PriorMode = "uniform",
    relative_tolerance: float = 1e-10,
    max_encoded_features: int | None = None,
) -> dict:
    """Load, split, encode and run all three classifiers on one public dataset."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction deve essere strettamente tra 0 e 1.")
    X_frame, y_series, source_used = load_public_dataset(dataset_key)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_frame,
        y_series.to_numpy(),
        test_size=test_fraction,
        random_state=random_seed,
        stratify=y_series.to_numpy(),
    )

    # The paper starts from unit vectors. Every data-dependent transform is fitted
    # only on the training fold.  If a quantum-ready feature budget is requested,
    # standardization and PCA are also learned exclusively on the training split.
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    raw_feature_count = int(X_train_imputed.shape[1])
    if max_encoded_features is None:
        encoded_feature_count = raw_feature_count
    else:
        if not isinstance(max_encoded_features, (int, np.integer)):
            raise ValueError("max_encoded_features deve essere un intero o None.")
        if int(max_encoded_features) < 1:
            raise ValueError("max_encoded_features deve essere positivo.")
        encoded_feature_count = min(raw_feature_count, int(max_encoded_features))

    explained_variance_ratio = 1.0
    feature_transform = "feature originali"
    if encoded_feature_count < raw_feature_count:
        standardizer = StandardScaler()
        X_train_standardized = standardizer.fit_transform(X_train_imputed)
        X_test_standardized = standardizer.transform(X_test_imputed)
        reducer = PCA(n_components=encoded_feature_count, svd_solver="full")
        X_train_for_scaling = reducer.fit_transform(X_train_standardized)
        X_test_for_scaling = reducer.transform(X_test_standardized)
        explained_variance_ratio = float(
            np.nan_to_num(reducer.explained_variance_ratio_, nan=0.0).sum()
        )
        feature_transform = (
            f"PCA train-only {raw_feature_count}->{encoded_feature_count}"
        )
    else:
        X_train_for_scaling = X_train_imputed
        X_test_for_scaling = X_test_imputed

    # [0.001, 1] rules out the zero vector without adding another feature.
    scaler = MinMaxScaler(feature_range=(1e-3, 1.0), clip=True)
    X_train_scaled = scaler.fit_transform(X_train_for_scaling)
    X_test_scaled = scaler.transform(X_test_for_scaling)
    X_train = normalize(X_train_scaled, norm="l2", axis=1).astype(np.float64)
    X_test = normalize(X_test_scaled, norm="l2", axis=1).astype(np.float64)

    results = run_all_methods(
        X_train,
        y_train,
        X_test,
        copies=copies,
        prior_mode=prior_mode,
        relative_tolerance=relative_tolerance,
    )
    return {
        "results": results,
        "y_test": np.asarray(y_test),
        "y_train": np.asarray(y_train),
        "X_train_encoded": X_train,
        "X_test_encoded": X_test,
        "source_used": source_used,
        "n_total": int(X_frame.shape[0]),
        "raw_d": int(X_frame.shape[1]),
        "d": encoded_feature_count,
        "feature_transform": feature_transform,
        "explained_variance_ratio": explained_variance_ratio,
        "class_count": int(y_series.nunique()),
    }
