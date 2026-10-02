"""Complexity expressions and resource estimates from the supplied paper."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class ComplexityRow:
    method: str
    working_dimension: int
    training_time_big_o: str
    training_time_proxy: int
    training_memory_big_o: str
    training_memory_elements: int
    prediction_time_big_o: str
    prediction_time_proxy: int
    prediction_memory_big_o: str
    prediction_memory_elements: int


@dataclass(frozen=True)
class FeasibilityEstimate:
    feasible: bool
    reasons: tuple[str, ...]
    estimated_peak_bytes: int
    largest_dense_dimension: int


def representation_dimensions(dimension: int, copies: int) -> tuple[int, int]:
    if dimension < 1 or copies < 1:
        raise ValueError("dimension e copies devono essere >= 1")
    return dimension**copies, comb(dimension + copies - 1, copies)


def paper_complexities(
    n_train: int,
    dimension: int,
    copies: int,
    class_count: int,
    gram_rank: int | None = None,
) -> list[ComplexityRow]:
    """Return Table 3/Table 6 terms, with their variables numerically substituted.

    The integer proxies are leading-term counts, not measured FLOPs. Memory follows
    the paper's storage model (including N density matrices for primal methods).
    """

    tensor_dimension, symmetric_dimension = representation_dimensions(dimension, copies)
    rank = n_train if gram_rank is None else int(gram_rank)
    return [
        ComplexityRow(
            method="c-PGM",
            working_dimension=tensor_dimension,
            training_time_big_o="O(max(N d^(2c), l d^(3c)))",
            training_time_proxy=max(
                n_train * tensor_dimension**2,
                class_count * tensor_dimension**3,
            ),
            training_memory_big_o="O(N d^(2c))",
            training_memory_elements=n_train * tensor_dimension**2,
            prediction_time_big_o="O(l d^(2c))",
            prediction_time_proxy=class_count * tensor_dimension**2,
            prediction_memory_big_o="O(l d^(2c))",
            prediction_memory_elements=class_count * tensor_dimension**2,
        ),
        ComplexityRow(
            method="k-PGM",
            working_dimension=n_train,
            training_time_big_o="O(N^3)",
            training_time_proxy=n_train**3,
            training_memory_big_o="O(N^2)",
            training_memory_elements=n_train**2,
            prediction_time_big_o="O(r_G N)",
            prediction_time_proxy=rank * n_train,
            prediction_memory_big_o="O(N(d + r_G))",
            prediction_memory_elements=n_train * (dimension + rank),
        ),
        ComplexityRow(
            method="r-PGM",
            working_dimension=symmetric_dimension,
            training_time_big_o="O(max(N d_sym^2, l d_sym^3))",
            training_time_proxy=max(
                n_train * symmetric_dimension**2,
                class_count * symmetric_dimension**3,
            ),
            training_memory_big_o="O(N d_sym^2)",
            training_memory_elements=n_train * symmetric_dimension**2,
            prediction_time_big_o="O(l d_sym^2)",
            prediction_time_proxy=class_count * symmetric_dimension**2,
            prediction_memory_big_o="O(l d_sym^2)",
            prediction_memory_elements=class_count * symmetric_dimension**2,
        ),
    ]


def implementation_feasibility(
    n_train: int,
    n_test: int,
    dimension: int,
    copies: int,
    memory_budget_bytes: int,
    dense_dimension_limit: int = 2_000,
) -> FeasibilityEstimate:
    """Conservative guard for the concrete optimized NumPy implementation."""

    tensor_dimension, symmetric_dimension = representation_dimensions(dimension, copies)
    # The largest c-PGM allocation is dominated by eigendecomposition workspaces.
    tensor_bytes = 8 * (
        3 * n_train * tensor_dimension
        + n_test * tensor_dimension
        + 8 * tensor_dimension**2
    )
    reduced_bytes = 8 * (
        3 * n_train * symmetric_dimension
        + n_test * symmetric_dimension
        + 8 * symmetric_dimension**2
    )
    kernel_bytes = 8 * (8 * n_train**2 + n_train * dimension)
    peak = max(tensor_bytes, reduced_bytes, kernel_bytes)
    largest = max(tensor_dimension, symmetric_dimension, n_train)
    reasons: list[str] = []
    if peak > memory_budget_bytes:
        reasons.append(
            "la RAM di picco stimata per l'implementazione supera il budget selezionato"
        )
    if largest > dense_dimension_limit:
        reasons.append(
            f"una diagonalizzazione densa avrebbe dimensione {largest:,}, oltre il limite "
            f"prudenziale {dense_dimension_limit:,}"
        )
    return FeasibilityEstimate(
        feasible=not reasons,
        reasons=tuple(reasons),
        estimated_peak_bytes=peak,
        largest_dense_dimension=largest,
    )


def human_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} ZiB"


def scientific_integer(value: int) -> str:
    if value == 0:
        return "0"
    return f"{value:.3e}"
