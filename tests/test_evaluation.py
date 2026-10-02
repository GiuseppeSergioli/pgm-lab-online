from __future__ import annotations

import unittest

import numpy as np

from evaluation import (
    classification_details_frame,
    confusion_frames,
    execution_summary,
)


class EvaluationTests(unittest.TestCase):
    def test_confusion_matrix_counts_and_percentages(self) -> None:
        classes = np.asarray(["A", "B", "C"])
        actual = np.asarray(["A", "A", "B", "B", "C"])
        predicted = np.asarray(["A", "B", "B", "B", "A"])

        counts, percentages = confusion_frames(actual, predicted, classes)

        self.assertEqual(int(counts.loc["Reale: A", "Predetta: A"]), 1)
        self.assertEqual(int(counts.loc["Reale: B", "Predetta: B"]), 2)
        self.assertAlmostEqual(
            float(percentages.loc["Reale: A", "Predetta: B"]), 50.0
        )
        np.testing.assert_allclose(percentages.sum(axis=1), 100.0)

    def test_classification_details_marks_errors_and_deviation(self) -> None:
        classes = np.asarray(["A", "B"])
        actual = np.asarray(["A", "B"])
        predicted = np.asarray(["A", "A"])
        theoretical = np.asarray([[0.8, 0.2], [0.4, 0.6]])
        circuit = theoretical + np.asarray([[1e-12, -1e-12], [0.0, 0.0]])

        details = classification_details_frame(
            actual, predicted, classes, theoretical, circuit
        )

        self.assertEqual(details.loc[0, "Esito"], "✓ Corretta")
        self.assertEqual(details.loc[1, "Esito"], "✗ Errata")
        self.assertAlmostEqual(
            float(details.loc[0, "Scostamento max |circuito-teoria|"]), 1e-12
        )
        self.assertIn("Lettura rapida", details.columns)
        self.assertIn("Corretta", str(details.loc[0, "Lettura rapida"]))
        self.assertIn("Errata", str(details.loc[1, "Lettura rapida"]))
        self.assertIn("probabilità", str(details.loc[0, "Lettura rapida"]))
        self.assertIn("decisione netta", str(details.loc[0, "Lettura rapida"]))
        self.assertIn("decisione netta", str(details.loc[1, "Lettura rapida"]))

    def test_classification_details_warns_when_classes_are_close(self) -> None:
        classes = np.asarray(["A", "B"])
        actual = np.asarray(["A"])
        predicted = np.asarray(["A"])
        probabilities = np.asarray([[0.51, 0.49]])

        details = classification_details_frame(
            actual, predicted, classes, probabilities, probabilities
        )

        self.assertIn(
            "classi molto vicine", str(details.loc[0, "Lettura rapida"])
        )

    def test_execution_summary_compares_shots_with_theory(self) -> None:
        classes = np.asarray(["A", "B", "C"])
        counts = {"00": 70, "01": 20, "10": 10}
        theoretical = np.asarray([0.7, 0.2, 0.1, 0.0])

        predicted, distance, table = execution_summary(
            counts, classes, theoretical, outcome_qubits=2
        )

        self.assertEqual(predicted, "A")
        self.assertAlmostEqual(distance, 0.0)
        self.assertEqual(int(table["Conteggi"].sum()), 100)


if __name__ == "__main__":
    unittest.main()
