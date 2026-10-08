"""End-to-end experiment pipeline, independent from the Streamlit UI."""

from __future__ import annotations

from collections.abc import Sequence
from math import comb, log
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

from data_catalog import load_public_dataset
from pgm_core import (
    PriorMode,
    run_all_methods,
    run_all_methods_scalable,
    run_c_pgm,
    run_k_pgm,
    run_r_pgm,
)


DEFAULT_RESCALING_FACTORS = (0.1, 0.2, 0.5, 1.0, 2.0)
ENCODING_SELECTION_MINIMUM_GAIN = 0.005
ENCODING_TUNING_MAX_SAMPLES = 500
ENCODING_CROSS_VALIDATION_MAX_SAMPLES = 360


def stereographic_encoding(X: np.ndarray, factor: float) -> np.ndarray:
    """Return the inverse-stereographic unit vector for every input row."""

    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X deve essere una matrice bidimensionale.")
    if not np.isfinite(matrix).all():
        raise ValueError("X contiene valori NaN o infiniti.")
    if not np.isfinite(factor) or float(factor) <= 0.0:
        raise ValueError("Il fattore di rescaling deve essere positivo e finito.")
    scaled = float(factor) * matrix
    squared_norms = np.einsum("ni,ni->n", scaled, scaled)
    denominators = squared_norms + 1.0
    encoded = np.concatenate(
        (2.0 * scaled, (squared_norms - 1.0)[:, None]), axis=1
    )
    encoded /= denominators[:, None]
    return np.ascontiguousarray(encoded, dtype=np.float64)


def _validated_feature_count(
    raw_feature_count: int,
    max_encoded_features: int | None,
) -> int:
    if max_encoded_features is None:
        return raw_feature_count
    if not isinstance(max_encoded_features, (int, np.integer)):
        raise ValueError("max_encoded_features deve essere un intero o None.")
    if int(max_encoded_features) < 1:
        raise ValueError("max_encoded_features deve essere positivo.")
    return min(raw_feature_count, int(max_encoded_features))


def _fit_base_preprocessor(
    X_fit,
    X_transform,
    *,
    max_encoded_features: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit every data-dependent transformation only on ``X_fit``."""

    imputer = SimpleImputer(strategy="median")
    X_fit_imputed = imputer.fit_transform(X_fit)
    X_transform_imputed = imputer.transform(X_transform)
    raw_feature_count = int(X_fit_imputed.shape[1])
    feature_count = _validated_feature_count(
        raw_feature_count, max_encoded_features
    )

    explained_variance_ratio = 1.0
    feature_transform = "feature originali"
    if feature_count < raw_feature_count:
        standardizer = StandardScaler()
        X_fit_standardized = standardizer.fit_transform(X_fit_imputed)
        X_transform_standardized = standardizer.transform(X_transform_imputed)
        reducer = PCA(n_components=feature_count, svd_solver="full")
        X_fit_for_scaling = reducer.fit_transform(X_fit_standardized)
        X_transform_for_scaling = reducer.transform(X_transform_standardized)
        explained_variance_ratio = float(
            np.nan_to_num(reducer.explained_variance_ratio_, nan=0.0).sum()
        )
        feature_transform = (
            f"PCA train-only {raw_feature_count}->{feature_count}"
        )
    else:
        X_fit_for_scaling = X_fit_imputed
        X_transform_for_scaling = X_transform_imputed

    scaler = MinMaxScaler(feature_range=(1e-3, 1.0), clip=True)
    X_fit_scaled = scaler.fit_transform(X_fit_for_scaling).astype(np.float64)
    X_transform_scaled = scaler.transform(X_transform_for_scaling).astype(
        np.float64
    )
    metadata = {
        "raw_feature_count": raw_feature_count,
        "base_feature_count": feature_count,
        "feature_transform": feature_transform,
        "explained_variance_ratio": explained_variance_ratio,
    }
    return X_fit_scaled, X_transform_scaled, metadata


def _encode(
    X: np.ndarray,
    encoding: str,
    rescaling_factor: float | None,
) -> np.ndarray:
    if encoding == "tensor_l2":
        return normalize(X, norm="l2", axis=1).astype(np.float64)
    if encoding == "stereographic":
        if rescaling_factor is None:
            raise ValueError("Fattore di rescaling stereografico mancante.")
        return stereographic_encoding(X, rescaling_factor)
    raise ValueError(f"Encoding non riconosciuto: {encoding!r}")


def _fastest_exact_validation_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    copies: int,
    prior_mode: PriorMode,
    relative_tolerance: float,
) -> tuple[float, str]:
    """Evaluate one exact PGM representation, choosing the smallest operator."""

    dimension = int(X_train.shape[1])
    tensor_dimension = dimension**copies
    symmetric_dimension = comb(dimension + copies - 1, copies)
    # c-PGM never has a smaller representation than r-PGM. Restricting the
    # numerical backend to k-PGM/r-PGM avoids an unnecessary tensor matrix while
    # leaving the exact classifier unchanged.
    candidates = [
        (symmetric_dimension, "r-PGM", run_r_pgm),
        (len(X_train), "k-PGM", run_k_pgm),
    ]
    _, method_name, runner = min(candidates, key=lambda item: (item[0], item[1]))
    result = runner(
        X_train,
        y_train,
        X_validation,
        copies=copies,
        prior_mode=prior_mode,
        relative_tolerance=relative_tolerance,
    )
    return (
        float(balanced_accuracy_score(y_validation, result.predictions)),
        method_name,
    )


def _tuning_subset(X_train, y_train: np.ndarray, random_seed: int):
    if len(y_train) <= ENCODING_TUNING_MAX_SAMPLES:
        return X_train.reset_index(drop=True), np.asarray(y_train)
    y_array = np.asarray(y_train)
    classes, class_counts = np.unique(y_array, return_counts=True)
    if int(class_counts.min()) < 2:
        # StratifiedShuffleSplit rejects singleton classes. Keep one mandatory
        # sample per class, then fill the bounded tuning set deterministically.
        rng = np.random.default_rng(random_seed + 101)
        mandatory = np.asarray(
            [
                rng.choice(np.flatnonzero(y_array == class_label))
                for class_label in classes
            ],
            dtype=np.int64,
        )
        remaining_pool = np.setdiff1d(
            np.arange(len(y_array), dtype=np.int64),
            mandatory,
            assume_unique=False,
        )
        remaining_count = ENCODING_TUNING_MAX_SAMPLES - len(mandatory)
        selected = np.concatenate(
            (
                mandatory,
                rng.choice(
                    remaining_pool,
                    size=remaining_count,
                    replace=False,
                ),
            )
        )
        selected.sort()
        return (
            X_train.iloc[selected].reset_index(drop=True),
            y_array[selected],
        )
    selector = StratifiedShuffleSplit(
        n_splits=1,
        train_size=ENCODING_TUNING_MAX_SAMPLES,
        random_state=random_seed + 101,
    )
    selected, _ = next(selector.split(np.zeros(len(y_train)), y_train))
    return (
        X_train.iloc[selected].reset_index(drop=True),
        y_array[selected],
    )


def _validation_splits(y_tuning: np.ndarray, random_seed: int):
    class_counts = np.unique(y_tuning, return_counts=True)[1]
    if (
        len(y_tuning) <= ENCODING_CROSS_VALIDATION_MAX_SAMPLES
        and int(class_counts.min()) >= 3
    ):
        splitter = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=random_seed + 211
        )
        return (
            list(splitter.split(np.zeros(len(y_tuning)), y_tuning)),
            "3-fold stratificata sul training",
        )
    # Preserve at least one fitting sample from every class, including classes
    # represented by a singleton after the outer train/test split.
    rng = np.random.default_rng(random_seed + 211)
    fit_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_label in np.unique(y_tuning):
        class_indices = np.flatnonzero(y_tuning == class_label)
        class_indices = rng.permutation(class_indices)
        validation_count = (
            0
            if len(class_indices) == 1
            else min(
                len(class_indices) - 1,
                max(1, int(round(0.20 * len(class_indices)))),
            )
        )
        validation_parts.append(class_indices[:validation_count])
        fit_parts.append(class_indices[validation_count:])
    fit_indices = np.sort(np.concatenate(fit_parts))
    nonempty_validation_parts = [
        part for part in validation_parts if len(part)
    ]
    if not nonempty_validation_parts:
        raise ValueError(
            "Il training set è troppo piccolo per validare automaticamente "
            "l'encoding."
        )
    validation_indices = np.sort(np.concatenate(nonempty_validation_parts))
    return (
        [(fit_indices, validation_indices)],
        "holdout stratificato adattivo sul training (classi rare protette)",
    )


def select_encoding_on_training(
    X_train_raw,
    y_train: np.ndarray,
    *,
    copies: int,
    random_seed: int,
    prior_mode: PriorMode,
    relative_tolerance: float,
    tensor_max_encoded_features: int | None,
    stereographic_max_encoded_features: int | None,
    rescaling_factors: Sequence[float] = DEFAULT_RESCALING_FACTORS,
    minimum_gain: float = ENCODING_SELECTION_MINIMUM_GAIN,
) -> dict[str, Any]:
    """Select the encoding using training data only, never the outer test set."""

    factors = tuple(float(value) for value in rescaling_factors)
    if not factors or any(not np.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("I fattori di rescaling devono essere positivi e finiti.")
    if minimum_gain < 0.0:
        raise ValueError("Il miglioramento minimo non può essere negativo.")

    candidates: list[dict[str, Any]] = [
        {
            "key": "tensor_l2",
            "encoding": "tensor_l2",
            "rescaling_factor": None,
            "max_encoded_features": tensor_max_encoded_features,
        }
    ]
    candidates.extend(
        {
            "key": f"stereographic_t_{factor:g}",
            "encoding": "stereographic",
            "rescaling_factor": factor,
            "max_encoded_features": stereographic_max_encoded_features,
        }
        for factor in factors
    )

    X_tuning, y_tuning = _tuning_subset(
        X_train_raw, np.asarray(y_train), random_seed
    )
    splits, protocol = _validation_splits(y_tuning, random_seed)
    scores: dict[str, list[float]] = {candidate["key"]: [] for candidate in candidates}
    evaluators: dict[str, set[str]] = {
        candidate["key"]: set() for candidate in candidates
    }
    errors: dict[str, list[str]] = {candidate["key"]: [] for candidate in candidates}

    for fit_indices, validation_indices in splits:
        X_fit = X_tuning.iloc[fit_indices]
        X_validation = X_tuning.iloc[validation_indices]
        y_fit = y_tuning[fit_indices]
        y_validation = y_tuning[validation_indices]
        prepared: dict[int | None, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        for candidate in candidates:
            feature_limit = candidate["max_encoded_features"]
            try:
                if feature_limit not in prepared:
                    prepared[feature_limit] = _fit_base_preprocessor(
                        X_fit,
                        X_validation,
                        max_encoded_features=feature_limit,
                    )
                X_fit_base, X_validation_base, _ = prepared[feature_limit]
                X_fit_encoded = _encode(
                    X_fit_base,
                    candidate["encoding"],
                    candidate["rescaling_factor"],
                )
                X_validation_encoded = _encode(
                    X_validation_base,
                    candidate["encoding"],
                    candidate["rescaling_factor"],
                )
                score, evaluator = _fastest_exact_validation_accuracy(
                    X_fit_encoded,
                    y_fit,
                    X_validation_encoded,
                    y_validation,
                    copies=copies,
                    prior_mode=prior_mode,
                    relative_tolerance=relative_tolerance,
                )
                scores[candidate["key"]].append(score)
                evaluators[candidate["key"]].add(evaluator)
            except Exception as error:
                errors[candidate["key"]].append(str(error))

    raw_feature_count = int(X_train_raw.shape[1])
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_scores = scores[candidate["key"]]
        available = len(candidate_scores) == len(splits)
        base_dimension = _validated_feature_count(
            raw_feature_count, candidate["max_encoded_features"]
        )
        encoded_dimension = base_dimension + (
            1 if candidate["encoding"] == "stereographic" else 0
        )
        rows.append(
            {
                **candidate,
                "available": available,
                "validation_accuracy": (
                    float(np.mean(candidate_scores)) if available else None
                ),
                "validation_std": (
                    float(np.std(candidate_scores)) if available else None
                ),
                "base_dimension": base_dimension,
                "encoded_dimension": encoded_dimension,
                "evaluators": ", ".join(sorted(evaluators[candidate["key"]])),
                "error": " | ".join(errors[candidate["key"]]),
            }
        )

    current = rows[0]
    if not current["available"]:
        raise RuntimeError(
            "La validazione dell'encoding tensoriale non è riuscita: "
            + current["error"]
        )
    available_stereographic = [row for row in rows[1:] if row["available"]]
    best_stereographic = (
        min(
            available_stereographic,
            key=lambda row: (
                -float(row["validation_accuracy"]),
                abs(log(float(row["rescaling_factor"]))),
                float(row["rescaling_factor"]),
            ),
        )
        if available_stereographic
        else None
    )

    current_accuracy = float(current["validation_accuracy"])
    stereographic_accuracy = (
        float(best_stereographic["validation_accuracy"])
        if best_stereographic is not None
        else float("-inf")
    )
    selected = (
        best_stereographic
        if stereographic_accuracy > current_accuracy + minimum_gain
        else current
    )
    return {
        "selected_key": selected["key"],
        "encoding": selected["encoding"],
        "rescaling_factor": selected["rescaling_factor"],
        "max_encoded_features": selected["max_encoded_features"],
        "validation_accuracy": float(selected["validation_accuracy"]),
        "current_validation_accuracy": current_accuracy,
        "improvement": float(selected["validation_accuracy"]) - current_accuracy,
        "best_stereographic_validation_accuracy": (
            stereographic_accuracy
            if best_stereographic is not None
            else None
        ),
        "best_stereographic_factor": (
            best_stereographic["rescaling_factor"]
            if best_stereographic is not None
            else None
        ),
        "best_stereographic_gain": (
            stereographic_accuracy - current_accuracy
            if best_stereographic is not None
            else None
        ),
        "minimum_gain": float(minimum_gain),
        "protocol": protocol,
        "tuning_samples": int(len(y_tuning)),
        "folds": int(len(splits)),
        "candidates": rows,
    }


def run_experiment(
    dataset_key: str,
    copies: int,
    test_fraction: float = 0.20,
    random_seed: int = 42,
    prior_mode: PriorMode = "uniform",
    relative_tolerance: float = 1e-10,
    max_encoded_features: int | None = None,
    stereographic_max_encoded_features: int | None = None,
    scalable_full_features: bool = False,
    explicit_dimension_limit: int = 512,
    automatic_encoding_selection: bool = True,
    rescaling_factors: Sequence[float] = DEFAULT_RESCALING_FACTORS,
) -> dict:
    """Load data, select an encoding on training only, and run all three PGMs."""

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
    if stereographic_max_encoded_features is None:
        stereographic_max_encoded_features = max_encoded_features

    if automatic_encoding_selection:
        encoding_selection = select_encoding_on_training(
            X_train_raw,
            np.asarray(y_train),
            copies=copies,
            random_seed=random_seed,
            prior_mode=prior_mode,
            relative_tolerance=relative_tolerance,
            tensor_max_encoded_features=max_encoded_features,
            stereographic_max_encoded_features=(
                stereographic_max_encoded_features
            ),
            rescaling_factors=rescaling_factors,
        )
    else:
        encoding_selection = {
            "selected_key": "tensor_l2",
            "encoding": "tensor_l2",
            "rescaling_factor": None,
            "max_encoded_features": max_encoded_features,
            "validation_accuracy": None,
            "current_validation_accuracy": None,
            "improvement": 0.0,
            "best_stereographic_validation_accuracy": None,
            "best_stereographic_factor": None,
            "best_stereographic_gain": None,
            "minimum_gain": ENCODING_SELECTION_MINIMUM_GAIN,
            "protocol": "selezione automatica disattivata",
            "tuning_samples": 0,
            "folds": 0,
            "candidates": [],
        }

    X_train_base, X_test_base, preprocessing = _fit_base_preprocessor(
        X_train_raw,
        X_test_raw,
        max_encoded_features=encoding_selection["max_encoded_features"],
    )
    X_train = _encode(
        X_train_base,
        encoding_selection["encoding"],
        encoding_selection["rescaling_factor"],
    )
    X_test = _encode(
        X_test_base,
        encoding_selection["encoding"],
        encoding_selection["rescaling_factor"],
    )

    runner = run_all_methods_scalable if scalable_full_features else run_all_methods
    runner_kwargs = {
        "copies": copies,
        "prior_mode": prior_mode,
        "relative_tolerance": relative_tolerance,
    }
    if scalable_full_features:
        runner_kwargs["explicit_dimension_limit"] = explicit_dimension_limit
    results = runner(X_train, y_train, X_test, **runner_kwargs)
    independent_methods = tuple(
        name
        for name, result in results.items()
        if "non materializzata" not in result.execution_mode
    )
    kernel_equivalent_methods = tuple(
        name for name in results if name not in independent_methods
    )
    return {
        "results": results,
        "y_test": np.asarray(y_test),
        "y_train": np.asarray(y_train),
        "X_train_encoded": X_train,
        "X_test_encoded": X_test,
        "X_train_raw": X_train_raw.copy(),
        "X_test_raw": X_test_raw.copy(),
        "source_used": source_used,
        "n_total": int(X_frame.shape[0]),
        "raw_d": int(X_frame.shape[1]),
        "base_d": int(preprocessing["base_feature_count"]),
        "d": int(X_train.shape[1]),
        "feature_transform": preprocessing["feature_transform"],
        "explained_variance_ratio": preprocessing["explained_variance_ratio"],
        "class_count": int(y_series.nunique()),
        "encoding": encoding_selection["encoding"],
        "rescaling_factor": encoding_selection["rescaling_factor"],
        "encoding_selection": encoding_selection,
        "execution_mode": (
            "adattiva_k_o_r" if scalable_full_features else "indipendente"
        ),
        "computational_backend": (
            independent_methods[0] if len(independent_methods) == 1 else "k-PGM"
        ),
        "independent_methods": independent_methods,
        "kernel_equivalent_methods": kernel_equivalent_methods,
    }
