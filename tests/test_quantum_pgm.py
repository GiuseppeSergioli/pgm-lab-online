from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from evaluation import confusion_frames
from experiment import run_experiment
from pgm_core import run_r_pgm, stable_predictions, symmetric_feature_map
from quantum_pgm import (
    build_naimark_dilation,
    build_qiskit_circuit,
    build_reduced_pgm_measurement,
    circuit_svg,
    generic_unitary_cnot_upper_bound,
    isolated_transpile_qpy,
    outcome_probabilities,
    qpy_bytes,
    resources_for_dataset,
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

    def test_generic_unitary_cnot_bounds(self) -> None:
        self.assertEqual(generic_unitary_cnot_upper_bound(1), 0)
        self.assertEqual(generic_unitary_cnot_upper_bound(2), 3)
        self.assertEqual(generic_unitary_cnot_upper_bound(3), 20)
        self.assertEqual(generic_unitary_cnot_upper_bound(4), 100)

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


if __name__ == "__main__":
    unittest.main()
