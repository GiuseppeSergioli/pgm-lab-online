"""Isolated Qiskit transpilation worker.

This file is intentionally executed in a child process.  A native failure in a
Qiskit synthesis extension therefore cannot terminate the Streamlit server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy, transpile
from qiskit.circuit.library import Isometry, StatePreparation
from qiskit.quantum_info import Operator

from quantum_pgm import subspace_equivalence_metrics


def _unitary_part(circuit):
    unitary = circuit.remove_final_measurements(inplace=False)
    unitary.data = [
        instruction
        for instruction in unitary.data
        if instruction.operation.name != "barrier"
    ]
    return unitary


def _subspace_certificate(reference, candidate, dimension: int) -> tuple[float, float]:
    """Return phase-insensitive operator error and fidelity on used inputs."""

    reference_matrix = np.asarray(Operator(_unitary_part(reference)).data)
    candidate_matrix = np.asarray(Operator(_unitary_part(candidate)).data)
    if reference_matrix.shape != candidate_matrix.shape:
        raise ValueError("I circuiti da confrontare hanno dimensioni differenti.")
    if dimension < 1 or dimension > reference_matrix.shape[1]:
        raise ValueError("Dimensione del sottospazio di ingresso non valida.")
    return subspace_equivalence_metrics(
        reference_matrix, candidate_matrix, dimension
    )


def _load_input_circuit(args):
    if bool(args.input) == bool(args.isometry):
        raise ValueError(
            "Fornire esattamente uno tra --input e --isometry."
        )
    if args.input:
        with Path(args.input).open("rb") as handle:
            circuits = qpy.load(handle)
        if len(circuits) != 1:
            raise ValueError("Il file QPY deve contenere esattamente un circuito.")
        return circuits[0]

    if args.system_qubits is None or args.outcome_qubits is None:
        raise ValueError("Dimensioni dei registri mancanti per l'isometria.")
    isometry = np.load(args.isometry, allow_pickle=False)
    expected_shape = (
        1 << (args.system_qubits + args.outcome_qubits),
        1 << args.system_qubits,
    )
    if isometry.shape != expected_shape:
        raise ValueError("Dimensioni della matrice isometrica non valide.")

    system = QuantumRegister(args.system_qubits, "sys")
    outcome = QuantumRegister(args.outcome_qubits, "out")
    classical = ClassicalRegister(args.outcome_qubits, "m")
    circuit = QuantumCircuit(system, outcome, classical, name=args.circuit_name)
    if args.input_state:
        state = np.asarray(
            np.load(args.input_state, allow_pickle=False), dtype=np.complex128
        ).reshape(-1)
        padded_dimension = 1 << args.system_qubits
        if state.size > padded_dimension:
            raise ValueError("Lo stato non entra nel registro di sistema.")
        padded = np.zeros(padded_dimension, dtype=np.complex128)
        padded[: state.size] = state
        norm = float(np.linalg.norm(padded))
        if not np.isfinite(norm) or not np.isclose(
            norm, 1.0, atol=1e-10, rtol=1e-10
        ):
            raise ValueError("Lo stato di ingresso deve avere norma unitaria.")
        circuit.append(
            StatePreparation(padded, normalize=False, label="Prepare |phi_c(x)>"),
            list(system),
        )
        circuit.barrier()
    circuit.append(
        Isometry(
            isometry,
            num_ancillas_zero=0,
            num_ancillas_dirty=0,
            epsilon=1e-12,
        ),
        list(system) + list(outcome),
    )
    circuit.barrier()
    circuit.measure(outcome, classical)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--isometry")
    parser.add_argument("--input-state")
    parser.add_argument("--system-qubits", type=int)
    parser.add_argument("--outcome-qubits", type=int)
    parser.add_argument("--circuit-name", default="PGM_Naimark_isometry")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--diagram", required=True)
    parser.add_argument("--fold", type=int, default=120)
    parser.add_argument("--max-diagram-gates", type=int, default=5000)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=0)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    parser.add_argument("--reference")
    parser.add_argument("--input-subspace-dimension", type=int)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    circuit = _load_input_circuit(args)

    started = perf_counter()
    transpiled = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    elapsed = perf_counter() - started
    size = int(transpiled.size())
    with Path(args.output).open("wb") as handle:
        qpy.dump(transpiled, handle)

    equivalence_error = None
    equivalence_fidelity = None
    equivalence_certified = None
    if args.reference:
        with Path(args.reference).open("rb") as handle:
            reference_circuits = qpy.load(handle)
        if len(reference_circuits) != 1:
            raise ValueError(
                "Il riferimento QPY deve contenere esattamente un circuito."
            )
        if args.input_subspace_dimension is None:
            raise ValueError("Dimensione del sottospazio di ingresso mancante.")
        equivalence_error, equivalence_fidelity = _subspace_certificate(
            reference_circuits[0],
            transpiled,
            args.input_subspace_dimension,
        )
        equivalence_certified = bool(
            equivalence_error <= args.equivalence_tolerance
        )

    report = {
        "depth": int(transpiled.depth()),
        "size": size,
        "gate_counts": {
            str(name): int(count) for name, count in transpiled.count_ops().items()
        },
        "elapsed_seconds": float(elapsed),
        "diagram_available": False,
        "diagram_requested": size <= args.max_diagram_gates,
        "diagram_gate_limit": int(args.max_diagram_gates),
        "num_qubits": int(transpiled.num_qubits),
        "num_clbits": int(transpiled.num_clbits),
        "optimization_level": int(args.optimization_level),
        "seed_transpiler": int(args.seed_transpiler),
        "equivalence_error": equivalence_error,
        "equivalence_tolerance": (
            float(args.equivalence_tolerance) if args.reference else None
        ),
        "equivalence_certified": equivalence_certified,
        "subspace_fidelity": equivalence_fidelity,
    }
    Path(args.report).write_text(json.dumps(report), encoding="utf-8")

    if report["diagram_requested"]:
        drawing = str(transpiled.draw(output="text", fold=args.fold))
        Path(args.diagram).write_text(drawing, encoding="utf-8")
        report["diagram_available"] = True
        Path(args.report).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
