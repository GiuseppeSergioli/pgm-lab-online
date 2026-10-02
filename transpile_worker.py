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
from qiskit import qpy, transpile
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
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

    with Path(args.input).open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise ValueError("Il file QPY deve contenere esattamente un circuito.")

    started = perf_counter()
    transpiled = transpile(
        circuits[0],
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
