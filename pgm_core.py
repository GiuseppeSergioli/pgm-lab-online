"""Numerically stable implementations of c-PGM, k-PGM and reduced c-PGM.

The three public entry points intentionally use different representations:

* c-PGM: explicit tensor feature vector x ** (tensor c), dimension d**c;
* r-PGM: explicit minimal symmetric feature map, dimension C(d+c-1, c);
* k-PGM: weighted Gram matrix, dimension N_train.

All three use the same sample weights and the same relative eigenspectrum cutoff.
This is essential if their numerical predictions are to be meaningfully compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement
from math import comb, lgamma
from time import perf_counter
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
PriorMode = Literal["uniform", "empirical"]


@dataclass(frozen=True)
class MethodResult:
    """Output and diagnostics for one classifier."""

    method: str
    predictions: NDArray
    scores: FloatArray
    rank: int
    representation_dimension: int
    train_seconds: float
    predict_seconds: float
    spectral_threshold: float
    minimum_eigenvalue: float
    model_state_bytes: int
    maximum_feature_norm_error: float
    execution_mode: str = "esplicita indipendente"


def _as_float_matrix(X: NDArray, name: str) -> FloatArray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} deve essere una matrice bidimensionale.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contiene valori NaN o infiniti.")
    return matrix


def _validate_inputs(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    copies: int,
) -> tuple[FloatArray, NDArray, FloatArray]:
    X_train_f = _as_float_matrix(X_train, "X_train")
    X_test_f = _as_float_matrix(X_test, "X_test")
    y_train_a = np.asarray(y_train)
    if X_train_f.shape[0] != y_train_a.shape[0]:
        raise ValueError("X_train e y_train hanno numerosità diverse.")
    if X_train_f.shape[1] != X_test_f.shape[1]:
        raise ValueError("Train e test devono avere lo stesso numero di feature.")
    if not isinstance(copies, (int, np.integer)) or int(copies) < 1:
        raise ValueError("Il numero di copie c deve essere un intero >= 1.")
    train_norms = np.linalg.norm(X_train_f, axis=1)
    test_norms = np.linalg.norm(X_test_f, axis=1)
    if not np.allclose(train_norms, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("I vettori di training devono avere norma L2 unitaria.")
    if not np.allclose(test_norms, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("I vettori di test devono avere norma L2 unitaria.")
    return X_train_f, y_train_a, X_test_f


def _label_indices(y: NDArray, classes: NDArray) -> NDArray[np.int64]:
    lookup = {label: index for index, label in enumerate(classes.tolist())}
    try:
        return np.asarray([lookup[label] for label in y.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Classe non riconosciuta: {exc.args[0]!r}") from exc


def sample_weights(
    y: NDArray,
    classes: NDArray,
    prior_mode: PriorMode,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Return alpha_i = p(y_i) / n_(y_i), as required by the PGM sigma."""

    y_idx = _label_indices(np.asarray(y), np.asarray(classes))
    counts = np.bincount(y_idx, minlength=len(classes)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Ogni classe deve avere almeno un campione nel training set.")
    if prior_mode == "uniform":
        priors = np.full(len(classes), 1.0 / len(classes), dtype=np.float64)
    elif prior_mode == "empirical":
        priors = counts / counts.sum()
    else:
        raise ValueError(f"Modalita dei prior non supportata: {prior_mode!r}")
    return priors[y_idx] / counts[y_idx], y_idx


def tensor_feature_map(X: NDArray, copies: int) -> FloatArray:
    """Explicit row-wise c-fold Kronecker powers."""

    X_f = _as_float_matrix(X, "X")
    if copies < 1:
        raise ValueError("copies deve essere >= 1.")
    mapped = X_f.copy()
    for _ in range(1, copies):
        mapped = np.einsum("ni,nj->nij", mapped, X_f, optimize=True).reshape(
            X_f.shape[0], -1
        )
    return np.ascontiguousarray(mapped)


@lru_cache(maxsize=64)
def occupation_basis(dimension: int, copies: int) -> tuple[NDArray[np.int64], FloatArray]:
    """Occupation sequences and sqrt-multinomial coefficients for Sym^c(R^d)."""

    if dimension < 1 or copies < 1:
        raise ValueError("dimension e copies devono essere >= 1.")
    rows: list[NDArray[np.int64]] = []
    for combination in combinations_with_replacement(range(dimension), copies):
        rows.append(np.bincount(combination, minlength=dimension).astype(np.int64))
    occupations = np.stack(rows, axis=0)
    log_coefficients = np.asarray(
        [
            0.5
            * (
                lgamma(copies + 1)
                - sum(lgamma(int(exponent) + 1) for exponent in occupation)
            )
            for occupation in occupations
        ],
        dtype=np.float64,
    )
    coefficients = np.exp(log_coefficients)
    occupations.setflags(write=False)
    coefficients.setflags(write=False)
    return occupations, coefficients


def symmetric_feature_map(X: NDArray, copies: int) -> FloatArray:
    """Eq. (24) of the paper, evaluated without constructing x^(tensor c)."""

    X_f = _as_float_matrix(X, "X")
    occupations, coefficients = occupation_basis(X_f.shape[1], copies)
    mapped = np.ones((X_f.shape[0], occupations.shape[0]), dtype=np.float64)
    for feature_index in range(X_f.shape[1]):
        exponents = occupations[:, feature_index]
        if np.any(exponents):
            mapped *= np.power(X_f[:, feature_index, None], exponents[None, :])
    mapped *= coefficients[None, :]
    return np.ascontiguousarray(mapped)


def _positive_eigensystem(
    matrix: FloatArray,
    relative_tolerance: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("La tolleranza spettrale relativa deve essere tra 0 e 1.")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(max(eigenvalues[-1], 0.0))
    threshold = relative_tolerance * largest
    keep = eigenvalues > threshold
    if not np.any(keep):
        raise np.linalg.LinAlgError(
            "Nessun autovalore supera la soglia spettrale; controllare i dati."
        )
    return (
        eigenvalues[keep],
        eigenvectors[:, keep],
        threshold,
        float(eigenvalues[0]),
    )


def stable_predictions(
    scores: FloatArray,
    classes: NDArray,
    relative_tie_tolerance: float = 1e-10,
    absolute_tie_tolerance: float = 1e-12,
) -> NDArray:
    """Argmax with a common deterministic rule for numerically tied scores."""

    maxima = np.max(scores, axis=1, keepdims=True)
    scales = np.maximum(
        np.max(np.abs(scores), axis=1, keepdims=True), np.finfo(np.float64).tiny
    )
    tolerance = absolute_tie_tolerance + relative_tie_tolerance * scales
    tied_with_maximum = scores >= maxima - tolerance
    first = np.argmax(tied_with_maximum, axis=1)
    return np.asarray(classes)[first]


def _score_by_class(
    projected_samples: FloatArray,
    y_indices: NDArray[np.int64],
    class_count: int,
) -> FloatArray:
    squared = np.square(projected_samples)
    scores = np.empty((projected_samples.shape[0], class_count), dtype=np.float64)
    for class_index in range(class_count):
        scores[:, class_index] = squared[:, y_indices == class_index].sum(axis=1)
    return scores


def _run_primal(
    method: str,
    X_train: FloatArray,
    y_train: NDArray,
    X_test: FloatArray,
    classes: NDArray,
    copies: int,
    feature_map: Callable[[NDArray, int], FloatArray],
    prior_mode: PriorMode,
    relative_tolerance: float,
) -> MethodResult:
    train_start = perf_counter()
    train_features = feature_map(X_train, copies)
    alpha, y_indices = sample_weights(y_train, classes, prior_mode)
    weighted_features = train_features * np.sqrt(alpha)[:, None]
    sigma = weighted_features.T @ weighted_features
    eigenvalues, eigenvectors, threshold, minimum_eigenvalue = _positive_eigensystem(
        sigma, relative_tolerance
    )
    # A = B sigma^(-1/2), represented only on the retained support.
    support_coordinates = (weighted_features @ eigenvectors) / np.sqrt(eigenvalues)[
        None, :
    ]
    train_seconds = perf_counter() - train_start

    prediction_start = perf_counter()
    test_features = feature_map(X_test, copies)
    projected_samples = (test_features @ eigenvectors) @ support_coordinates.T
    scores = _score_by_class(projected_samples, y_indices, len(classes))
    predictions = stable_predictions(scores, classes)
    predict_seconds = perf_counter() - prediction_start

    expected_norms = np.ones(train_features.shape[0], dtype=np.float64)
    feature_norm_error = float(
        np.max(np.abs(np.linalg.norm(train_features, axis=1) - expected_norms))
    )
    state_bytes = int(
        eigenvalues.nbytes
        + eigenvectors.nbytes
        + support_coordinates.nbytes
        + y_indices.nbytes
    )
    return MethodResult(
        method=method,
        predictions=predictions,
        scores=scores,
        rank=int(eigenvalues.size),
        representation_dimension=int(train_features.shape[1]),
        train_seconds=float(train_seconds),
        predict_seconds=float(predict_seconds),
        spectral_threshold=float(threshold),
        minimum_eigenvalue=minimum_eigenvalue,
        model_state_bytes=state_bytes,
        maximum_feature_norm_error=feature_norm_error,
    )


def run_c_pgm(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    *,
    copies: int,
    classes: NDArray | None = None,
    prior_mode: PriorMode = "uniform",
    relative_tolerance: float = 1e-10,
) -> MethodResult:
    """Run the explicit tensor-space c-PGM."""

    X_train_f, y_train_a, X_test_f = _validate_inputs(
        X_train, y_train, X_test, copies
    )
    classes_a = np.unique(y_train_a) if classes is None else np.asarray(classes)
    return _run_primal(
        "c-PGM",
        X_train_f,
        y_train_a,
        X_test_f,
        classes_a,
        copies,
        tensor_feature_map,
        prior_mode,
        relative_tolerance,
    )


def run_r_pgm(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    *,
    copies: int,
    classes: NDArray | None = None,
    prior_mode: PriorMode = "uniform",
    relative_tolerance: float = 1e-10,
) -> MethodResult:
    """Run the reduced c-PGM in the symmetric subspace (r-PGM/Rc-PGM)."""

    X_train_f, y_train_a, X_test_f = _validate_inputs(
        X_train, y_train, X_test, copies
    )
    classes_a = np.unique(y_train_a) if classes is None else np.asarray(classes)
    return _run_primal(
        "r-PGM",
        X_train_f,
        y_train_a,
        X_test_f,
        classes_a,
        copies,
        symmetric_feature_map,
        prior_mode,
        relative_tolerance,
    )


def run_k_pgm(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    *,
    copies: int,
    classes: NDArray | None = None,
    prior_mode: PriorMode = "uniform",
    relative_tolerance: float = 1e-10,
) -> MethodResult:
    """Run the dual k-PGM using the homogeneous polynomial kernel <x,z>**c."""

    X_train_f, y_train_a, X_test_f = _validate_inputs(
        X_train, y_train, X_test, copies
    )
    classes_a = np.unique(y_train_a) if classes is None else np.asarray(classes)

    train_start = perf_counter()
    alpha, y_indices = sample_weights(y_train_a, classes_a, prior_mode)
    sqrt_alpha = np.sqrt(alpha)
    gram = np.power(X_train_f @ X_train_f.T, copies)
    weighted_gram = sqrt_alpha[:, None] * gram * sqrt_alpha[None, :]
    eigenvalues, eigenvectors, threshold, minimum_eigenvalue = _positive_eigensystem(
        weighted_gram, relative_tolerance
    )
    train_seconds = perf_counter() - train_start

    prediction_start = perf_counter()
    cross_kernel = np.power(X_test_f @ X_train_f.T, copies)
    weighted_cross_kernel = cross_kernel * sqrt_alpha[None, :]
    projected_samples = (
        (weighted_cross_kernel @ eigenvectors) / np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.T
    scores = _score_by_class(projected_samples, y_indices, len(classes_a))
    predictions = stable_predictions(scores, classes_a)
    predict_seconds = perf_counter() - prediction_start

    state_bytes = int(
        X_train_f.nbytes
        + sqrt_alpha.nbytes
        + eigenvalues.nbytes
        + eigenvectors.nbytes
        + y_indices.nbytes
    )
    return MethodResult(
        method="k-PGM",
        predictions=predictions,
        scores=scores,
        rank=int(eigenvalues.size),
        representation_dimension=int(X_train_f.shape[0]),
        train_seconds=float(train_seconds),
        predict_seconds=float(predict_seconds),
        spectral_threshold=float(threshold),
        minimum_eigenvalue=minimum_eigenvalue,
        model_state_bytes=state_bytes,
        maximum_feature_norm_error=0.0,
        execution_mode="kernel diretto indipendente",
    )


def run_all_methods(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    *,
    copies: int,
    prior_mode: PriorMode = "uniform",
    relative_tolerance: float = 1e-10,
) -> dict[str, MethodResult]:
    """Run all representations independently with common numerical conventions."""

    classes = np.unique(np.asarray(y_train))
    results = [
        run_c_pgm(
            X_train,
            y_train,
            X_test,
            copies=copies,
            classes=classes,
            prior_mode=prior_mode,
            relative_tolerance=relative_tolerance,
        ),
        run_k_pgm(
            X_train,
            y_train,
            X_test,
            copies=copies,
            classes=classes,
            prior_mode=prior_mode,
            relative_tolerance=relative_tolerance,
        ),
        run_r_pgm(
            X_train,
            y_train,
            X_test,
            copies=copies,
            classes=classes,
            prior_mode=prior_mode,
            relative_tolerance=relative_tolerance,
        ),
    ]
    return {result.method: result for result in results}


def _equivalent_result_from_reference(
    reference_result: MethodResult,
    *,
    method: str,
    representation_dimension: int,
) -> MethodResult:
    """Expose an equivalent PGM representation without recomputing it.

    The scores are copied from the exact representation selected as the numerical
    backend. Consequently runtime and model-state memory are deliberately reported
    as NaN/0 instead of being attributed to work performed by another method.
    """

    return MethodResult(
        method=method,
        predictions=reference_result.predictions.copy(),
        scores=reference_result.scores.copy(),
        rank=reference_result.rank,
        representation_dimension=int(representation_dimension),
        train_seconds=float("nan"),
        predict_seconds=float("nan"),
        spectral_threshold=reference_result.spectral_threshold,
        minimum_eigenvalue=reference_result.minimum_eigenvalue,
        model_state_bytes=0,
        maximum_feature_norm_error=float("nan"),
        execution_mode=(
            f"equivalente esatta via {reference_result.method} "
            "(matrice non materializzata)"
        ),
    )


def run_all_methods_scalable(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    *,
    copies: int,
    prior_mode: PriorMode = "uniform",
    relative_tolerance: float = 1e-10,
    explicit_dimension_limit: int = 512,
) -> dict[str, MethodResult]:
    """Run the cheapest exact backend between k-PGM and r-PGM.

    The feature map is never reduced here.  r-PGM is selected when its symmetric
    representation is smaller than the N x N kernel representation and fits the
    explicit safety limit; otherwise k-PGM is selected.  The other two named
    formulations expose copies of the same exact scores because their Gram matrices
    are all ``<x, z>**copies``. This avoids paying for multiple identical dense
    eigendecompositions while preserving the public c/k/r comparison.
    """

    X_train_f, y_train_a, X_test_f = _validate_inputs(
        X_train, y_train, X_test, copies
    )
    if not isinstance(explicit_dimension_limit, (int, np.integer)):
        raise ValueError("explicit_dimension_limit deve essere un intero.")
    if int(explicit_dimension_limit) < 1:
        raise ValueError("explicit_dimension_limit deve essere positivo.")

    classes = np.unique(y_train_a)
    dimension = int(X_train_f.shape[1])
    tensor_dimension = dimension**int(copies)
    symmetric_dimension = comb(dimension + int(copies) - 1, int(copies))

    use_reduced = (
        symmetric_dimension < int(X_train_f.shape[0])
        and symmetric_dimension <= int(explicit_dimension_limit)
    )
    if use_reduced:
        reference_result = run_r_pgm(
            X_train_f,
            y_train_a,
            X_test_f,
            copies=copies,
            classes=classes,
            prior_mode=prior_mode,
            relative_tolerance=relative_tolerance,
        )
        r_result = reference_result
        k_result = _equivalent_result_from_reference(
            reference_result,
            method="k-PGM",
            representation_dimension=int(X_train_f.shape[0]),
        )
    else:
        reference_result = run_k_pgm(
            X_train_f,
            y_train_a,
            X_test_f,
            copies=copies,
            classes=classes,
            prior_mode=prior_mode,
            relative_tolerance=relative_tolerance,
        )
        k_result = reference_result
        r_result = _equivalent_result_from_reference(
            reference_result,
            method="r-PGM",
            representation_dimension=symmetric_dimension,
        )

    c_result = _equivalent_result_from_reference(
        reference_result,
        method="c-PGM",
        representation_dimension=tensor_dimension,
    )

    return {
        "c-PGM": c_result,
        "k-PGM": k_result,
        "r-PGM": r_result,
    }
