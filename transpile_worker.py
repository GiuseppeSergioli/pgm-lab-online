"""Isolated Qiskit transpilation worker.

This file is intentionally executed in a child process.  A native failure in a
Qiskit synthesis extension therefore cannot terminate the Streamlit server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from qiskit import qpy, transpile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--diagram", required=True)
    parser.add_argument("--fold", type=int, default=120)
    parser.add_argument("--max-diagram-gates", type=int, default=5000)
    args = parser.parse_args()

    with Path(args.input).open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise ValueError("Il file QPY deve contenere esattamente un circuito.")

    started = perf_counter()
    transpiled = transpile(
        circuits[0],
        basis_gates=["rz", "sx", "x", "cx"],
        # Level 0 deliberately avoids optional native optimization passes that have
        # caused SIGSEGV on some macOS/Qiskit combinations. The synthesis remains exact.
        optimization_level=0,
        seed_transpiler=42,
    )
    elapsed = perf_counter() - started
    size = int(transpiled.size())
    with Path(args.output).open("wb") as handle:
        qpy.dump(transpiled, handle)

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
    }
    Path(args.report).write_text(json.dumps(report), encoding="utf-8")

    if report["diagram_requested"]:
        drawing = str(transpiled.draw(output="text", fold=args.fold))
        Path(args.diagram).write_text(drawing, encoding="utf-8")
        report["diagram_available"] = True
        Path(args.report).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
