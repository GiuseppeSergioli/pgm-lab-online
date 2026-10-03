"""Exact Naimark dilation of the trained reduced PGM.

The construction uses the canonical square-root instrument

    V |psi> = sum_j |j> tensor sqrt(F_j) |psi>,

then completes the isometry V to a unitary matrix. Measuring the outcome
register therefore returns outcome j with probability <psi|F_j|psi>.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from math import ceil, comb, log2
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space

from pgm_core import PriorMode, sample_weights, symmetric_feature_map


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PGMMeasurement:
    effects: tuple[FloatArray, ...]
    classes: NDArray
    feature_dimension: int
    support_rank: int
    spectral_threshold: float
    completeness_error: float
    minimum_effect_eigenvalue: float
    maximum_effect_eigenvalue: float


@dataclass(frozen=True)
class CircuitResources:
    feature_dimension: int
    padded_system_dimension: int
    class_count: int
    padded_outcome_dimension: int
    system_qubits: int
    outcome_qubits: int
    total_qubits: int
    unitary_dimension: int
    unitary_bytes: int
    exact_materialization_allowed: bool
    exact_qubit_limit: int


@dataclass(frozen=True)
class NaimarkDilation:
    resources: CircuitResources
    unitary: ComplexArray | None
    isometry: ComplexArray | None
    embedded_effects: tuple[ComplexArray, ...] | None
    input_isometry_error: float | None
    unitary_error: float | None
    probability_operator_error: float | None


@dataclass(frozen=True)
class IsolatedTranspilation:
    status: str
    message: str
    depth: int | None
    size: int | None
    gate_counts: dict[str, int]
    elapsed_seconds: float | None
    diagram_text: str | None
    transpiled_qpy: bytes | None
    return_code: int | None
    equivalence_error: float | None = None
    equivalence_tolerance: float | None = None
    equivalence_certified: bool | None = None
    subspace_fidelity: float | None = None
    optimization_level: int | None = None
    seed_transpiler: int | None = None


ENTANGLING_GATE_NAMES = frozenset(
    {"cx", "cz", "ecr", "rxx", "ryy", "rzz", "iswap", "swap"}
)


def entangling_gate_count(report: IsolatedTranspilation) -> int:
    """Count two-qubit/entangling operations in a transpilation report."""

    return int(
        sum(
            count
            for name, count in report.gate_counts.items()
            if name.lower() in ENTANGLING_GATE_NAMES
        )
    )


def best_certified_transpilation(
    reports: list[IsolatedTranspilation] | tuple[IsolatedTranspilation, ...],
) -> IsolatedTranspilation | None:
    """Choose the most hardware-friendly candidate with certified equivalence.

    Entangling gates dominate current-device error, so they are minimized first;
    circuit depth and total gate count are deterministic tie-breakers.
    """

    certified = [
        report
        for report in reports
        if report.status in {"success", "partial"}
        and report.equivalence_certified is True
        and report.transpiled_qpy is not None
        and report.size is not None
        and report.depth is not None
    ]
    if not certified:
        return None
    return min(
        certified,
        key=lambda report: (
            entangling_gate_count(report),
            report.depth,
            report.size,
            report.optimization_level
            if report.optimization_level is not None
            else 99,
        ),
    )


def subspace_equivalence_metrics(
    reference_matrix: NDArray,
    candidate_matrix: NDArray,
    input_dimension: int,
) -> tuple[float, float]:
    """Return global-phase-insensitive error and fidelity on valid inputs."""

    reference = np.asarray(reference_matrix, dtype=np.complex128)
    candidate = np.asarray(candidate_matrix, dtype=np.complex128)
    if reference.ndim != 2 or reference.shape != candidate.shape:
        raise ValueError("Le matrici da confrontare devono avere la stessa forma.")
    if input_dimension < 1 or input_dimension > reference.shape[1]:
        raise ValueError("Dimensione del sottospazio di ingresso non valida.")
    reference_action = reference[:, :input_dimension]
    candidate_action = candidate[:, :input_dimension]
    overlap = np.vdot(reference_action, candidate_action)
    if abs(overlap) > 0.0:
        candidate_action = candidate_action / (overlap / abs(overlap))
    error = float(np.linalg.norm(reference_action - candidate_action, ord=2))
    fidelity = float(abs(overlap) ** 2 / float(input_dimension**2))
    return error, min(1.0, max(0.0, fidelity))


def symbolic_dilation(resources: CircuitResources) -> NaimarkDilation:
    """Represent the circuit architecture without materializing its dense unitary."""

    return NaimarkDilation(
        resources=resources,
        unitary=None,
        isometry=None,
        embedded_effects=None,
        input_isometry_error=None,
        unitary_error=None,
        probability_operator_error=None,
    )


def _qubit_count(dimension: int) -> int:
    if dimension < 1:
        raise ValueError("La dimensione deve essere positiva.")
    return int(ceil(log2(dimension))) if dimension > 1 else 0


def _power_of_two_ceiling(dimension: int) -> int:
    return 1 << _qubit_count(dimension)


def circuit_resources(
    feature_dimension: int,
    class_count: int,
    *,
    exact_qubit_limit: int = 8,
) -> CircuitResources:
    padded_system = _power_of_two_ceiling(feature_dimension)
    padded_outcomes = _power_of_two_ceiling(class_count)
    system_qubits = _qubit_count(padded_system)
    outcome_qubits = _qubit_count(padded_outcomes)
    total_qubits = system_qubits + outcome_qubits
    unitary_dimension = padded_system * padded_outcomes
    return CircuitResources(
        feature_dimension=int(feature_dimension),
        padded_system_dimension=padded_system,
        class_count=int(class_count),
        padded_outcome_dimension=padded_outcomes,
        system_qubits=system_qubits,
        outcome_qubits=outcome_qubits,
        total_qubits=total_qubits,
        unitary_dimension=unitary_dimension,
        unitary_bytes=16 * unitary_dimension**2,
        exact_materialization_allowed=total_qubits <= exact_qubit_limit,
        exact_qubit_limit=exact_qubit_limit,
    )


def resources_for_dataset(
    feature_count: int,
    copies: int,
    class_count: int,
    *,
    exact_qubit_limit: int = 8,
) -> CircuitResources:
    reduced_dimension = comb(feature_count + copies - 1, copies)
    return circuit_resources(
        reduced_dimension,
        class_count,
        exact_qubit_limit=exact_qubit_limit,
    )


def build_reduced_pgm_measurement(
    X_train: NDArray,
    y_train: NDArray,
    *,
    copies: int,
    prior_mode: PriorMode = "uniform",
    relative_tolerance: float = 1e-10,
) -> PGMMeasurement:
    """Construct the completed PGM effects F_j in the symmetric feature space."""

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X_train e y_train non sono compatibili.")
    if not np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("I vettori di training devono avere norma unitaria.")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("La tolleranza spettrale deve essere tra 0 e 1.")

    classes = np.unique(y)
    features = symmetric_feature_map(X, copies)
    alpha, y_indices = sample_weights(y, classes, prior_mode)
    weighted_features = features * np.sqrt(alpha)[:, None]
    sigma = weighted_features.T @ weighted_features
    sigma = 0.5 * (sigma + sigma.T)

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    largest = float(max(eigenvalues[-1], 0.0))
    threshold = relative_tolerance * largest
    keep = eigenvalues > threshold
    if not np.any(keep):
        raise np.linalg.LinAlgError("La PGM non ha supporto numerico alla soglia scelta.")
    positive_values = eigenvalues[keep]
    support_vectors = eigenvectors[:, keep]
    inverse_square_root = (
        support_vectors / np.sqrt(positive_values)[None, :]
    ) @ support_vectors.T
    support_projector = support_vectors @ support_vectors.T
    kernel_projector = np.eye(features.shape[1]) - support_projector

    effects: list[FloatArray] = []
    for class_index in range(len(classes)):
        class_rows = weighted_features[y_indices == class_index]
        weighted_centroid = class_rows.T @ class_rows
        raw_effect = inverse_square_root @ weighted_centroid @ inverse_square_root
        completed_effect = raw_effect + kernel_projector / len(classes)
        effects.append(0.5 * (completed_effect + completed_effect.T))

    completeness = np.sum(np.stack(effects), axis=0)
    all_effect_eigenvalues = np.concatenate(
        [np.linalg.eigvalsh(effect) for effect in effects]
    )
    return PGMMeasurement(
        effects=tuple(effects),
        classes=classes,
        feature_dimension=int(features.shape[1]),
        support_rank=int(positive_values.size),
        spectral_threshold=float(threshold),
        completeness_error=float(
            np.linalg.norm(completeness - np.eye(features.shape[1]), ord=2)
        ),
        minimum_effect_eigenvalue=float(np.min(all_effect_eigenvalues)),
        maximum_effect_eigenvalue=float(np.max(all_effect_eigenvalues)),
    )


def _positive_square_root(matrix: ComplexArray, tolerance: float = 1e-9) -> ComplexArray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if float(eigenvalues[0]) < -tolerance:
        raise np.linalg.LinAlgError(
            f"Effetto non positivo: autovalore minimo {eigenvalues[0]:.3e}."
        )
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)[None, :]) @ eigenvectors.conj().T


def _embed_effects(measurement: PGMMeasurement) -> tuple[ComplexArray, ...]:
    resources = circuit_resources(
        measurement.feature_dimension, len(measurement.effects)
    )
    padded_system = resources.padded_system_dimension
    class_count = len(measurement.effects)
    embedded: list[ComplexArray] = []
    for effect in measurement.effects:
        extended = np.zeros((padded_system, padded_system), dtype=np.complex128)
        extended[: measurement.feature_dimension, : measurement.feature_dimension] = effect
        if padded_system > measurement.feature_dimension:
            extended[
                measurement.feature_dimension :, measurement.feature_dimension :
            ] = np.eye(padded_system - measurement.feature_dimension) / class_count
        embedded.append(extended)
    while len(embedded) < resources.padded_outcome_dimension:
        embedded.append(
            np.zeros((padded_system, padded_system), dtype=np.complex128)
        )
    return tuple(embedded)


def build_naimark_dilation(
    measurement: PGMMeasurement,
    *,
    exact_qubit_limit: int = 8,
) -> NaimarkDilation:
    """Build one exact unitary completion of the square-root Naimark isometry."""

    resources = circuit_resources(
        measurement.feature_dimension,
        len(measurement.effects),
        exact_qubit_limit=exact_qubit_limit,
    )
    if not resources.exact_materialization_allowed:
        return symbolic_dilation(resources)

    embedded_effects = _embed_effects(measurement)
    square_roots = [_positive_square_root(effect) for effect in embedded_effects]
    isometry = np.vstack(square_roots)

    # Remove the final floating-point drift without changing the represented POVM
    # beyond machine precision: V <- V (V^dagger V)^(-1/2).
    gram_raw = isometry.conj().T @ isometry
    gram = 0.5 * (gram_raw + gram_raw.conj().T)
    gram_values, gram_vectors = np.linalg.eigh(gram)
    if float(gram_values[0]) <= 0.0:
        raise np.linalg.LinAlgError("L'isometria di Naimark ha perso rango numerico.")
    gram_inverse_sqrt = (
        gram_vectors / np.sqrt(gram_values)[None, :]
    ) @ gram_vectors.conj().T
    isometry = isometry @ gram_inverse_sqrt

    complement = null_space(isometry.conj().T)
    unitary = np.concatenate([isometry, complement], axis=1)
    identity_input = np.eye(resources.padded_system_dimension)
    identity_full = np.eye(resources.unitary_dimension)
    input_error = float(
        np.linalg.norm(isometry.conj().T @ isometry - identity_input, ord=2)
    )
    unitary_error = float(
        np.linalg.norm(unitary.conj().T @ unitary - identity_full, ord=2)
    )

    reconstructed = []
    block_size = resources.padded_system_dimension
    for outcome in range(resources.padded_outcome_dimension):
        block = isometry[outcome * block_size : (outcome + 1) * block_size]
        reconstructed.append(block.conj().T @ block)
    probability_error = float(
        max(
            np.linalg.norm(actual - expected, ord=2)
            for actual, expected in zip(reconstructed, embedded_effects)
        )
    )
    return NaimarkDilation(
        resources=resources,
        unitary=np.asarray(unitary, dtype=np.complex128),
        isometry=np.asarray(isometry, dtype=np.complex128),
        embedded_effects=embedded_effects,
        input_isometry_error=input_error,
        unitary_error=unitary_error,
        probability_operator_error=probability_error,
    )


def outcome_probabilities(
    dilation: NaimarkDilation,
    state: NDArray,
) -> FloatArray:
    """Apply the dilation matrix to |0>_outcome tensor |state> and measure outcome."""

    if dilation.unitary is None:
        raise ValueError("La matrice unitaria non è stata materializzata.")
    state_array = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state_array.size != dilation.resources.feature_dimension:
        raise ValueError("Lo stato non ha la dimensione della feature map ridotta.")
    padded = np.zeros(dilation.resources.padded_system_dimension, dtype=np.complex128)
    padded[: state_array.size] = state_array
    input_state = np.zeros(dilation.resources.unitary_dimension, dtype=np.complex128)
    input_state[: dilation.resources.padded_system_dimension] = padded
    output_state = dilation.unitary @ input_state
    blocks = output_state.reshape(
        dilation.resources.padded_outcome_dimension,
        dilation.resources.padded_system_dimension,
    )
    return np.sum(np.abs(blocks) ** 2, axis=1).astype(np.float64)


def build_qiskit_circuit(dilation: NaimarkDilation) -> Any:
    """Return an exact Qiskit circuit, or a clearly labelled symbolic circuit."""

    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Gate
    from qiskit.circuit.library import UnitaryGate

    resources = dilation.resources
    system = QuantumRegister(resources.system_qubits, "sys")
    outcome = QuantumRegister(resources.outcome_qubits, "out")
    classical = ClassicalRegister(resources.outcome_qubits, "m")
    circuit = QuantumCircuit(system, outcome, classical, name="PGM_Naimark")
    circuit.barrier()
    all_qubits = list(system) + list(outcome)
    if dilation.unitary is None:
        gate = Gate("U_PGM_symbolic", resources.total_qubits, [])
    else:
        gate = UnitaryGate(dilation.unitary, label="U_PGM (Naimark)")
    circuit.append(gate, all_qubits)
    circuit.barrier()
    circuit.measure(outcome, classical)
    return circuit


def build_qiskit_isometry_circuit(dilation: NaimarkDilation) -> Any:
    """Build the same PGM on its reachable input subspace using an isometry.

    Only columns acting on ``|0...0>_out tensor |psi>_sys`` are physically used.
    Synthesizing those columns directly avoids paying for an arbitrary unitary
    completion while preserving the PGM probabilities for every valid input.
    """

    if dilation.isometry is None:
        raise ValueError("L'isometria esatta non è stata materializzata.")
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import Isometry

    resources = dilation.resources
    system = QuantumRegister(resources.system_qubits, "sys")
    outcome = QuantumRegister(resources.outcome_qubits, "out")
    classical = ClassicalRegister(resources.outcome_qubits, "m")
    circuit = QuantumCircuit(system, outcome, classical, name="PGM_Naimark_isometry")
    circuit.barrier()
    instruction = Isometry(
        dilation.isometry,
        # The n-m output ancillas are implicit in the rectangular isometry.
        # This argument is only for *additional* helper ancillas.
        num_ancillas_zero=0,
        num_ancillas_dirty=0,
        epsilon=1e-12,
    )
    circuit.append(instruction, list(system) + list(outcome))
    circuit.barrier()
    circuit.measure(outcome, classical)
    return circuit


def circuit_svg(dilation: NaimarkDilation) -> str:
    """Draw the logical circuit as browser-native SVG, without Matplotlib/Pillow."""

    resources = dilation.resources
    row_spacing = 46
    top = 82
    total_rows = max(resources.total_qubits, 1)
    height = max(330, top + row_spacing * total_rows + 82)
    width = 1120
    wire_start = 170
    wire_end = 1030
    gate_x = 385
    gate_width = 350
    first_y = top
    last_y = top + row_spacing * (total_rows - 1)
    gate_y = first_y - 29
    gate_height = last_y - first_y + 58
    gate_colour = "#6f42c1" if dilation.unitary is not None else "#687386"
    status = (
        "unitaria esatta appresa dal training"
        if dilation.unitary is not None
        else "schema simbolico: matrice non materializzata"
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Circuito quantistico PGM mediante dilatazione di Naimark" '
        'style="width:100%;height:auto;max-height:900px;font-family:-apple-system,'
        'BlinkMacSystemFont,Segoe UI,sans-serif;background:#ffffff;border:1px solid '
        '#d9dee7;border-radius:12px">',
        '<text x="24" y="32" font-size="21" font-weight="700" fill="#202531">'
        "Circuito logico della PGM</text>",
        f'<text x="24" y="56" font-size="14" fill="#596579">{status}</text>',
    ]

    for qubit in range(resources.system_qubits):
        y = top + row_spacing * qubit
        parts.extend(
            [
                f'<text x="24" y="{y + 5}" font-size="14" font-weight="600" '
                f'fill="#263247">sys[{qubit}]</text>',
                f'<text x="112" y="{y + 5}" font-size="15" fill="#263247">|φ⟩</text>',
                f'<line x1="{wire_start}" y1="{y}" x2="{wire_end}" y2="{y}" '
                'stroke="#263247" stroke-width="2"/>',
            ]
        )

    for outcome_qubit in range(resources.outcome_qubits):
        row = resources.system_qubits + outcome_qubit
        y = top + row_spacing * row
        parts.extend(
            [
                f'<text x="24" y="{y + 5}" font-size="14" font-weight="600" '
                f'fill="#263247">out[{outcome_qubit}]</text>',
                f'<text x="112" y="{y + 5}" font-size="15" fill="#263247">|0⟩</text>',
                f'<line x1="{wire_start}" y1="{y}" x2="{wire_end}" y2="{y}" '
                'stroke="#263247" stroke-width="2"/>',
                f'<rect x="810" y="{y - 18}" width="44" height="36" rx="5" '
                'fill="#eef2f8" stroke="#263247" stroke-width="1.5"/>',
                f'<text x="832" y="{y + 6}" text-anchor="middle" font-size="15" '
                'font-weight="700" fill="#263247">M</text>',
                f'<line x1="854" y1="{y - 3}" x2="924" y2="{y - 3}" '
                'stroke="#263247" stroke-width="1.4"/>',
                f'<line x1="854" y1="{y + 3}" x2="924" y2="{y + 3}" '
                'stroke="#263247" stroke-width="1.4"/>',
                f'<text x="936" y="{y + 5}" font-size="14" fill="#263247">'
                f'm[{outcome_qubit}]</text>',
            ]
        )

    parts.extend(
        [
            f'<rect x="{gate_x}" y="{gate_y}" width="{gate_width}" '
            f'height="{gate_height}" rx="12" fill="{gate_colour}" '
            'stroke="#3d276f" stroke-width="2"/>',
            f'<text x="{gate_x + gate_width / 2}" y="{gate_y + gate_height / 2 - 18}" '
            'text-anchor="middle" font-size="23" font-weight="700" fill="#ffffff">'
            "U_PGM</text>",
            f'<text x="{gate_x + gate_width / 2}" y="{gate_y + gate_height / 2 + 8}" '
            'text-anchor="middle" font-size="15" fill="#ffffff">'
            "Dilatazione di Naimark</text>",
            f'<text x="{gate_x + gate_width / 2}" y="{gate_y + gate_height / 2 + 31}" '
            'text-anchor="middle" font-size="13" fill="#eee8ff">'
            f'{resources.unitary_dimension} × {resources.unitary_dimension}</text>',
            f'<text x="24" y="{height - 36}" font-size="13" fill="#596579">'
            f'Registro sys: {resources.system_qubits} qubit · Registro out: '
            f'{resources.outcome_qubits} qubit · Totale: {resources.total_qubits} qubit'
            "</text>",
            f'<text x="24" y="{height - 15}" font-size="13" fill="#596579">'
            "Ingresso: stato test codificato |φ_c(x)⟩ · Uscita: bitstring di classe"
            "</text>",
            "</svg>",
        ]
    )
    return "".join(parts)


def qpy_bytes(circuit: Any) -> bytes:
    from qiskit import qpy

    buffer = BytesIO()
    qpy.dump(circuit, buffer)
    return buffer.getvalue()


def generic_unitary_cnot_upper_bound(qubit_count: int) -> int:
    """Known QSD upper bound for a generic n-qubit unitary."""

    if qubit_count < 1:
        raise ValueError("Il numero di qubit deve essere positivo.")
    if qubit_count == 1:
        return 0
    value = (
        (23.0 / 48.0) * (4**qubit_count)
        - (3.0 / 2.0) * (2**qubit_count)
        + 4.0 / 3.0
    )
    return int(round(value))


def isolated_transpile_qpy(
    qpy_payload: bytes | None,
    *,
    isometry_matrix: NDArray | None = None,
    system_qubits: int | None = None,
    outcome_qubits: int | None = None,
    input_state: NDArray | None = None,
    circuit_name: str = "PGM_Naimark_isometry",
    reference_qpy_payload: bytes | None = None,
    input_subspace_dimension: int | None = None,
    timeout_seconds: int = 180,
    max_diagram_gates: int = 5000,
    fold: int = 120,
    optimization_level: int = 0,
    seed_transpiler: int = 42,
    equivalence_tolerance: float = 1e-9,
) -> IsolatedTranspilation:
    """Transpile in a child process so a native crash cannot kill Streamlit."""

    if (qpy_payload is None) == (isometry_matrix is None):
        raise ValueError(
            "Fornire esattamente uno tra circuito QPY e matrice isometrica."
        )
    if isometry_matrix is not None:
        if system_qubits is None or system_qubits < 0:
            raise ValueError("Numero di qubit di sistema non valido.")
        if outcome_qubits is None or outcome_qubits < 1:
            raise ValueError("Numero di qubit di uscita non valido.")
        expected_shape = (
            1 << (int(system_qubits) + int(outcome_qubits)),
            1 << int(system_qubits),
        )
        if np.asarray(isometry_matrix).shape != expected_shape:
            raise ValueError(
                "La matrice isometrica non coincide con le dimensioni dei registri."
            )

    worker = Path(__file__).with_name("transpile_worker.py")
    if not worker.exists():
        return IsolatedTranspilation(
            status="error",
            message="Il worker di transpilation non è presente nel progetto.",
            depth=None,
            size=None,
            gate_counts={},
            elapsed_seconds=None,
            diagram_text=None,
            transpiled_qpy=None,
            return_code=None,
        )

    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "QISKIT_PARALLEL": "FALSE",
            "RAYON_NUM_THREADS": "1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="pgm_transpile_") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.qpy"
        output_path = directory / "output.qpy"
        report_path = directory / "report.json"
        diagram_path = directory / "circuit.txt"
        reference_path = directory / "reference.qpy"
        isometry_path = directory / "isometry.npy"
        state_path = directory / "input_state.npy"
        command = [
            sys.executable,
            str(worker),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--diagram",
            str(diagram_path),
            "--fold",
            str(fold),
            "--max-diagram-gates",
            str(max_diagram_gates),
            "--optimization-level",
            str(int(optimization_level)),
            "--seed-transpiler",
            str(int(seed_transpiler)),
            "--equivalence-tolerance",
            str(float(equivalence_tolerance)),
        ]
        if qpy_payload is not None:
            input_path.write_bytes(qpy_payload)
            command.extend(["--input", str(input_path)])
        else:
            np.save(
                isometry_path,
                np.asarray(isometry_matrix, dtype=np.complex128),
                allow_pickle=False,
            )
            command.extend(
                [
                    "--isometry",
                    str(isometry_path),
                    "--system-qubits",
                    str(int(system_qubits)),
                    "--outcome-qubits",
                    str(int(outcome_qubits)),
                    "--circuit-name",
                    str(circuit_name),
                ]
            )
            if input_state is not None:
                state = np.asarray(input_state, dtype=np.complex128).reshape(-1)
                if state.size > (1 << int(system_qubits)):
                    raise ValueError(
                        "Lo stato di ingresso supera il registro di sistema."
                    )
                np.save(state_path, state, allow_pickle=False)
                command.extend(["--input-state", str(state_path)])
        if reference_qpy_payload is not None:
            if input_subspace_dimension is None or input_subspace_dimension < 1:
                raise ValueError(
                    "La dimensione del sottospazio è necessaria per certificare "
                    "l'equivalenza."
                )
            reference_path.write_bytes(reference_qpy_payload)
            command.extend(
                [
                    "--reference",
                    str(reference_path),
                    "--input-subspace-dimension",
                    str(int(input_subspace_dimension)),
                ]
            )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
                check=False,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return IsolatedTranspilation(
                status="timeout",
                message=(
                    "La sintesi ha superato il limite di "
                    f"{timeout_seconds} secondi ed è stata interrotta in sicurezza."
                ),
                depth=None,
                size=None,
                gate_counts={},
                elapsed_seconds=None,
                diagram_text=None,
                transpiled_qpy=None,
                return_code=None,
            )

        if completed.returncode != 0:
            if completed.returncode < 0:
                detail = f"segnale {-completed.returncode}"
            else:
                detail = f"codice {completed.returncode}"
            stderr_tail = completed.stderr.strip()[-1500:]
            message = (
                "Il processo isolato di sintesi è terminato con " + detail + ". "
                "L'interfaccia principale è rimasta attiva."
            )
            if stderr_tail:
                message += " Dettaglio: " + stderr_tail
            if report_path.exists() and output_path.exists():
                partial_report = json.loads(report_path.read_text(encoding="utf-8"))
                return IsolatedTranspilation(
                    status="partial",
                    message=(
                        message
                        + " La decomposizione e i conteggi sono stati comunque "
                        "recuperati; è fallita soltanto una fase successiva."
                    ),
                    depth=int(partial_report["depth"]),
                    size=int(partial_report["size"]),
                    gate_counts={
                        str(name): int(count)
                        for name, count in partial_report["gate_counts"].items()
                    },
                    elapsed_seconds=float(partial_report["elapsed_seconds"]),
                    diagram_text=(
                        diagram_path.read_text(encoding="utf-8")
                        if diagram_path.exists()
                        else None
                    ),
                    transpiled_qpy=output_path.read_bytes(),
                    return_code=completed.returncode,
                    equivalence_error=partial_report.get("equivalence_error"),
                    equivalence_tolerance=partial_report.get(
                        "equivalence_tolerance"
                    ),
                    equivalence_certified=partial_report.get(
                        "equivalence_certified"
                    ),
                    subspace_fidelity=partial_report.get("subspace_fidelity"),
                    optimization_level=partial_report.get("optimization_level"),
                    seed_transpiler=partial_report.get("seed_transpiler"),
                )
            return IsolatedTranspilation(
                status="crashed",
                message=message,
                depth=None,
                size=None,
                gate_counts={},
                elapsed_seconds=None,
                diagram_text=None,
                transpiled_qpy=None,
                return_code=completed.returncode,
            )

        if not report_path.exists() or not output_path.exists():
            return IsolatedTranspilation(
                status="error",
                message="La sintesi è terminata senza produrre tutti i file attesi.",
                depth=None,
                size=None,
                gate_counts={},
                elapsed_seconds=None,
                diagram_text=None,
                transpiled_qpy=None,
                return_code=completed.returncode,
            )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        diagram = (
            diagram_path.read_text(encoding="utf-8")
            if diagram_path.exists()
            else None
        )
        return IsolatedTranspilation(
            status="success",
            message="Sintesi completata nel processo isolato.",
            depth=int(report["depth"]),
            size=int(report["size"]),
            gate_counts={
                str(name): int(count)
                for name, count in report["gate_counts"].items()
            },
            elapsed_seconds=float(report["elapsed_seconds"]),
            diagram_text=diagram,
            transpiled_qpy=output_path.read_bytes(),
            return_code=completed.returncode,
            equivalence_error=report.get("equivalence_error"),
            equivalence_tolerance=report.get("equivalence_tolerance"),
            equivalence_certified=report.get("equivalence_certified"),
            subspace_fidelity=report.get("subspace_fidelity"),
            optimization_level=report.get("optimization_level"),
            seed_transpiler=report.get("seed_transpiler"),
        )


def dilation_npz_bytes(
    measurement: PGMMeasurement,
    dilation: NaimarkDilation,
    *,
    feature_count: int,
    copies: int,
) -> bytes:
    if dilation.unitary is None:
        raise ValueError("La dilatazione esatta non è disponibile.")
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        unitary=dilation.unitary,
        effects=np.stack(measurement.effects),
        classes=np.asarray([str(value) for value in measurement.classes]),
        raw_feature_count=np.asarray(feature_count),
        copies=np.asarray(copies),
        reduced_feature_dimension=np.asarray(measurement.feature_dimension),
        system_qubits=np.asarray(dilation.resources.system_qubits),
        outcome_qubits=np.asarray(dilation.resources.outcome_qubits),
    )
    return buffer.getvalue()
