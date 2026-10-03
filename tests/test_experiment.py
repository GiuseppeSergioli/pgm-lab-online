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

    def test_train_only_pca_keeps_all_three_methods_equivalent(self) -> None:
        payload = run_experiment(
            "iris",
            copies=3,
            test_fraction=0.20,
            random_seed=42,
            prior_mode="uniform",
            relative_tolerance=1e-10,
            max_encoded_features=2,
        )
        results = payload["results"]
        reference = results["c-PGM"]

        self.assertEqual(payload["raw_d"], 4)
        self.assertEqual(payload["d"], 2)
        self.assertEqual(payload["feature_transform"], "PCA train-only 4->2")
        self.assertGreater(payload["explained_variance_ratio"], 0.0)
        self.assertLessEqual(payload["explained_variance_ratio"], 1.0 + 1e-12)
        self.assertEqual(reference.representation_dimension, 2**3)
        for name in ("k-PGM", "r-PGM"):
            np.testing.assert_array_equal(
                results[name].predictions, reference.predictions
            )
            np.testing.assert_allclose(
                results[name].scores,
                reference.scores,
                atol=2e-11,
                rtol=2e-11,
            )

    def test_full_feature_scalable_mode_keeps_classification_available(self) -> None:
        payload = run_experiment(
            "iris",
            copies=3,
            test_fraction=0.20,
            random_seed=42,
            prior_mode="uniform",
            relative_tolerance=1e-10,
            max_encoded_features=None,
            scalable_full_features=True,
            explicit_dimension_limit=10,
        )

        self.assertEqual(payload["raw_d"], 4)
        self.assertEqual(payload["d"], 4)
        self.assertEqual(payload["feature_transform"], "feature originali")
        self.assertEqual(payload["execution_mode"], "scalabile_full_features")
        self.assertEqual(payload["independent_methods"], ("k-PGM",))
        self.assertEqual(
            payload["kernel_equivalent_methods"], ("c-PGM", "r-PGM")
        )
        for name in ("c-PGM", "r-PGM"):
            np.testing.assert_array_equal(
                payload["results"][name].predictions,
                payload["results"]["k-PGM"].predictions,
            )


if __name__ == "__main__":
    unittest.main()
