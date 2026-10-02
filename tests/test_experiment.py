from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import accuracy_score

from experiment import run_experiment


class EndToEndExperimentTests(unittest.TestCase):
    def test_default_iris_run_has_identical_predictions_and_accuracy(self) -> None:
        payload = run_experiment(
            "iris",
            copies=2,
            test_fraction=0.20,
            random_seed=42,
            prior_mode="uniform",
            relative_tolerance=1e-10,
        )
        results = payload["results"]
        y_test = payload["y_test"]
        reference = results["c-PGM"]

        self.assertEqual(len(y_test), 30)
        self.assertEqual(reference.representation_dimension, 16)
        self.assertAlmostEqual(accuracy_score(y_test, reference.predictions), 5 / 6)
        for name in ("k-PGM", "r-PGM"):
            np.testing.assert_array_equal(
                results[name].predictions, reference.predictions
            )
            np.testing.assert_allclose(
                results[name].scores, reference.scores, atol=1e-11, rtol=1e-11
            )


if __name__ == "__main__":
    unittest.main()
