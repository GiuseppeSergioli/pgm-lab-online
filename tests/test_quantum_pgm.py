from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from evaluation import confusion_frames
from experiment import run_experiment
from pgm_core import run_r_pgm, stable_predictions, symmetric_feature_map
from quantum_pgm import (
    IsolatedTranspilation,
    best_certified_transpilation,
    build_naimark_dilation,
    build_qiskit_circuit,
    build_qiskit_isometry_circuit,
    build_reduced_pgm_measurement,
    circuit_svg,
    entangling_gate_count,
    generic_unitary_cnot_upper_bound,
    isolated_transpile_qpy,
    outcome_probabilities,
    qpy_bytes,
    resources_for_dataset,
    subspace_equivalence_metrics,
)


def unit_rows(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    values = rng.normal(size=(rows, columns))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class NaimarkDilationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2026)
        self.X_train = unit_rows(rng, 18, 3)
        self.X_test = unit_rows(rng, 7, 3)
        self.y_train = np.asarray(["A"] * 8 + ["B"] * 6 + ["C"] * 4)

    def test_completed_effects_form_a_povm(self) -> None:
        measurement = build_reduced_pgm_measurement(
            self.X_train,
            self.y_train,
            copies=2,
            prior_mode="uniform",
            relative_tolerance=1e-11,
        )
        total = np.sum(np.stack(measurement.effects), axis=0)
        np.testing.assert_allclose(total, np.eye(6), atol=2e-10, rtol=2e-10)
        self.assertGreaterEqual(measurement.minimum_effect_eigenvalue, -2e-10)
        self.assertLessEqual(measurement.maximum_effect_eigenvalue, 1.0 + 2e-10)

    def test_dilation_is_unitary_and_reproduces_born_probabilities(self) -> None:
        measurement = build_reduced_pgm_measurement(
            self.X_train,
            self.y_train,
            copies=2,
            prior_mode="uniform",
            relative_tolerance=1e-11,
        )
        dilation = build_naimark_dilation(measurement, exact_qubit_limit=8)
        self.assertIsNotNone(dilation.unitary)
        self.assertLess(dilation.unitary_error, 2e-10)
        self.assertLess(dilation.probability_operator_error, 2e-9)

        reduced_test = symmetric_feature_map(self.X_test, 2)
        for state in reduced_test:
            circuit_probabilities = outcome_probabilities(dilation, state)
            born_probabilities = np.asarray(
                [state @ effect @ state for effect in measurement.effects]
            )
            np.testing.assert_allclose(
                circuit_probabilities[: len(measurement.effects)],
                born_probabilities,
                atol=2e-9,
                rtol=2e-9,
            )
            np.testing.assert_allclose(
                circuit_probabilities[len(measurement.effects) :],
                0.0,
                atol=2e-10,
            )

    def test_isometry_is_exact_action_of_unitary_on_valid_inputs(self) -> None:
        measurement = build_reduced_pgm_measurement(
            self.X_train,
            self.y_train,
            copies=2,
            prior_mode="uniform",
            relative_tolerance=1e-11,
        )
        dilation = build_naimark_dilation(measurement, exact_qubit_limit=8)
        input_dimension = dilation.resources.padded_system_dimension
        np.testing.assert_allclose(
            dilation.unitary[:, :input_dimension],
            dilation.isometry,
            atol=2e-12,
            rtol=2e-12,
        )

    def test_subspace_certificate_ignores_only_global_phase(self) -> None:
        reference = np.eye(4, dtype=np.complex128)
        globally_phased = np.exp(0.37j) * reference
        error, fidelity = subspace_equivalence_metrics(
            reference, globally_phased, input_dimension=2
        )
        self.assertLess(error, 1e-12)
        self.assertAlmostEqual(fidelity, 1.0, places=12)

        changed = globally_phased.copy()
        changed[:, 0] = np.roll(changed[:, 0], 1)
        changed_error, changed_fidelity = subspace_equivalence_metrics(
            reference, changed, input_dimension=2
        )
        self.assertGreater(changed_error, 0.5)
        self.assertLess(changed_fidelity, 1.0)

    def test_completed_povm_keeps_r_pgm_predictions(self) -> None:
        measurement = build_reduced_pgm_measurement(
            self.X_train,
            self.y_train,
            copies=3,
            prior_mode="empirical",
            relative_tolerance=1e-11,
        )
        reduced_test = symmetric_feature_map(self.X_test, 3)
        completed_scores = np.column_stack(
            [
                np.einsum("ni,ij,nj->n", reduced_test, effect, reduced_test)
                for effect in measurement.effects
            ]
        )
        completed_predictions = stable_predictions(
            completed_scores, measurement.classes
        )
        reference = run_r_pgm(
            self.X_train,
            self.y_train,
            self.X_test,
            copies=3,
            prior_mode="empirical",
            relative_tolerance=1e-11,
        )
        np.testing.assert_array_equal(completed_predictions, reference.predictions)

    def test_resource_count_uses_reduced_dimension(self) -> None:
        resources = resources_for_dataset(4, 2, 3)
        self.assertEqual(resources.feature_dimension, 10)
        self.assertEqual(resources.system_qubits, 4)
        self.assertEqual(resources.outcome_qubits, 2)
        self.assertEqual(resources.total_qubits, 6)
        self.assertEqual(resources.unitary_dimension, 64)

    def test_nine_qubit_matrix_can_be_materialized_when_explicitly_allowed(
        self,
    ) -> None:
        resources = resources_for_dataset(
            13,
            2,
            3,
            exact_qubit_limit=9,
        )
        self.assertEqual(resources.total_qubits, 9)
        self.assertEqual(resources.unitary_dimension, 512)
        self.assertEqual(resources.unitary_bytes, 4 * 1024**2)
        self.assertTrue(resources.exact_materialization_allowed)

    def test_generic_unitary_cnot_bounds(self) -> None:
        self.assertEqual(generic_unitary_cnot_upper_bound(1), 0)
        self.assertEqual(generic_unitary_cnot_upper_bound(2), 3)
        self.assertEqual(generic_unitary_cnot_upper_bound(3), 20)
        self.assertEqual(generic_unitary_cnot_upper_bound(4), 100)

    def test_best_certified_transpilation_prioritizes_entangling_gates(self) -> None:
        def report(
            level: int,
            *,
            cx: int,
            depth: int,
            size: int,
            certified: bool = True,
        ) -> IsolatedTranspilation:
            return IsolatedTranspilation(
                status="success",
                message="ok",
                depth=depth,
                size=size,
                gate_counts={"cx": cx, "rz": size - cx},
                elapsed_seconds=0.1,
                diagram_text=None,
                transpiled_qpy=f"level-{level}".encode(),
                return_code=0,
                equivalence_error=1e-12 if certified else 1e-3,
                equivalence_tolerance=1e-9,
                equivalence_certified=certified,
                subspace_fidelity=1.0 if certified else 0.9,
                optimization_level=level,
                seed_transpiler=42,
            )

        level_1 = report(1, cx=12, depth=20, size=30)
        level_2 = report(2, cx=10, depth=25, size=33)
        level_3 = report(3, cx=4, depth=8, size=12, certified=False)
        best = best_certified_transpilation([level_1, level_2, level_3])

        self.assertIs(best, level_2)
        self.assertEqual(entangling_gate_count(best), 10)

    def test_best_certified_transpilation_uses_depth_then_size(self) -> None:
        common = {
            "status": "success",
            "message": "ok",
            "gate_counts": {"cx": 5},
            "elapsed_seconds": 0.1,
            "diagram_text": None,
            "return_code": 0,
            "equivalence_error": 1e-12,
            "equivalence_tolerance": 1e-9,
            "equivalence_certified": True,
            "subspace_fidelity": 1.0,
            "seed_transpiler": 42,
        }
        deeper = IsolatedTranspilation(
            depth=20,
            size=10,
            transpiled_qpy=b"deeper",
            optimization_level=1,
            **common,
        )
        shallower = IsolatedTranspilation(
            depth=15,
            size=30,
            transpiled_qpy=b"shallower",
            optimization_level=2,
            **common,
        )
        self.assertIs(
            best_certified_transpilation([deeper, shallower]), shallower
        )

    def test_best_certified_transpilation_rejects_uncertified_candidates(
        self,
    ) -> None:
        candidate = IsolatedTranspilation(
            status="success",
            message="not equivalent",
            depth=1,
            size=1,
            gate_counts={"cx": 0},
            elapsed_seconds=0.1,
            diagram_text=None,
            transpiled_qpy=b"candidate",
            return_code=0,
            equivalence_error=1e-3,
            equivalence_tolerance=1e-9,
            equivalence_certified=False,
            subspace_fidelity=0.9,
            optimization_level=3,
            seed_transpiler=42,
        )
        self.assertIsNone(best_certified_transpilation([candidate]))

    def test_browser_native_svg_contains_the_registers(self) -> None:
        measurement = build_reduced_pgm_measurement(
            self.X_train,
            self.y_train,
            copies=1,
            prior_mode="uniform",
            relative_tolerance=1e-11,
        )
        dilation = build_naimark_dilation(measurement, exact_qubit_limit=8)
        drawing = circuit_svg(dilation)

        self.assertTrue(drawing.startswith("<svg"))
        self.assertIn("U_PGM", drawing)
        self.assertIn("sys[0]", drawing)
        self.assertIn("out[0]", drawing)
        self.assertNotIn("<script", drawing.lower())

    def test_default_iris_circuit_matches_the_trained_classifier(self) -> None:
        payload = run_experiment(
            "iris",
            copies=2,
            test_fraction=0.20,
            random_seed=42,
            prior_mode="uniform",
            relative_tolerance=1e-10,
        )
        measurement = build_reduced_pgm_measurement(
            payload["X_train_encoded"],
            payload["y_train"],
            copies=2,
            prior_mode="uniform",
            relative_tolerance=1e-10,
        )
        dilation = build_naimark_dilation(measurement, exact_qubit_limit=8)
        reduced_test = symmetric_feature_map(payload["X_test_encoded"], 2)
        probabilities = np.vstack(
            [outcome_probabilities(dilation, state) for state in reduced_test]
        )
        predictions = stable_predictions(
            probabilities[:, : len(measurement.classes)], measurement.classes
        )
        theoretical = np.column_stack(
            [
                np.einsum("ni,ij,nj->n", reduced_test, effect, reduced_test)
                for effect in measurement.effects
            ]
        )

        np.testing.assert_array_equal(
            predictions, payload["results"]["r-PGM"].predictions
        )
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=2e-10)
        np.testing.assert_allclose(
            probabilities[:, : len(measurement.classes)],
            theoretical,
            atol=5e-9,
            rtol=5e-9,
        )
        confusion, _ = confusion_frames(
            payload["y_test"], predictions, measurement.classes
        )
        self.assertEqual(
            int(np.trace(confusion.to_numpy())),
            int(np.sum(predictions == payload["y_test"])),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("qiskit") is not None,
        "Qiskit non installato nell'ambiente di test",
    )
    def test_qiskit_circuit_can_be_built_and_exported(self) -> None:
        measurement = build_reduced_pgm_measurement(
            self.X_train,
            self.y_train,
            copies=1,
            prior_mode="uniform",
            relative_tolerance=1e-11,
        )
        dilation = build_naimark_dilation(measurement, exact_qubit_limit=8)
        circuit = build_qiskit_circuit(dilation)

        self.assertEqual(circuit.num_qubits, dilation.resources.total_qubits)
        self.assertEqual(circuit.num_clbits, dilation.resources.outcome_qubits)
        self.assertGreater(len(qpy_bytes(circuit)), 0)

        report = isolated_transpile_qpy(
            qpy_bytes(circuit), timeout_seconds=60, max_diagram_gates=5_000
        )
        self.assertEqual(report.status, "success")
        self.assertIsNotNone(report.size)
        self.assertIsNotNone(report.diagram_text)
        self.assertIsNotNone(report.transpiled_qpy)

        isometry_circuit = build_qiskit_isometry_circuit(dilation)
        self.assertEqual(
            isometry_circuit.num_qubits, dilation.resources.total_qubits
        )
        optimized = isolated_transpile_qpy(
            None,
            isometry_matrix=dilation.isometry,
            system_qubits=dilation.resources.system_qubits,
            outcome_qubits=dilation.resources.outcome_qubits,
            reference_qpy_payload=qpy_bytes(circuit),
            input_subspace_dimension=dilation.resources.padded_system_dimension,
            timeout_seconds=60,
            max_diagram_gates=5_000,
            optimization_level=1,
            seed_transpiler=42,
        )
        self.assertEqual(optimized.status, "success")
        self.assertTrue(optimized.equivalence_certified)
        self.assertLessEqual(
            optimized.equivalence_error, optimized.equivalence_tolerance
        )
        self.assertGreater(optimized.subspace_fidelity, 1.0 - 1e-10)

    def test_isolated_transpilation_rejects_invalid_isometry_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensioni dei registri"):
            isolated_transpile_qpy(
                None,
                isometry_matrix=np.eye(4, dtype=np.complex128),
                system_qubits=1,
                outcome_qubits=2,
            )

    def test_isolated_transpilation_requires_one_input_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "esattamente uno"):
            isolated_transpile_qpy(None)
        with self.assertRaisesRegex(ValueError, "esattamente uno"):
            isolated_transpile_qpy(
                b"qpy",
                isometry_matrix=np.zeros((4, 2), dtype=np.complex128),
                system_qubits=1,
                outcome_qubits=1,
            )

    def test_isometry_is_sent_to_worker_without_qpy_round_trip(self) -> None:
        isometry = np.eye(4, 2, dtype=np.complex128)

        def fake_run(command, **kwargs):
            self.assertIn("--isometry", command)
            self.assertNotIn("--input", command)
            matrix_path = Path(command[command.index("--isometry") + 1])
            np.testing.assert_array_equal(
                np.load(matrix_path, allow_pickle=False), isometry
            )

            output_path = Path(command[command.index("--output") + 1])
            report_path = Path(command[command.index("--report") + 1])
            output_path.write_bytes(b"standard-gates-only-qpy")
            report_path.write_text(
                json.dumps(
                    {
                        "depth": 1,
                        "size": 1,
                        "gate_counts": {"cx": 1},
                        "elapsed_seconds": 0.01,
                        "equivalence_error": None,
                        "equivalence_tolerance": None,
                        "equivalence_certified": None,
                        "subspace_fidelity": None,
                        "optimization_level": 1,
                        "seed_transpiler": 42,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("quantum_pgm.subprocess.run", side_effect=fake_run):
            report = isolated_transpile_qpy(
                None,
                isometry_matrix=isometry,
                system_qubits=1,
                outcome_qubits=1,
                optimization_level=1,
            )

        self.assertEqual(report.status, "success")
        self.assertEqual(report.transpiled_qpy, b"standard-gates-only-qpy")


if __name__ == "__main__":
    unittest.main()
