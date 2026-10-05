from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from classifier_benchmark import (
    CLASSIFIER_SPECS,
    benchmark_raw_split,
    build_classifier,
    fit_standard_classifier,
    paired_balanced_accuracy_bootstrap,
)
from experiment import run_experiment


class ClassifierBenchmarkTests(unittest.TestCase):
    def test_old_cached_payload_recovers_the_identical_raw_split(self) -> None:
        payload = run_experiment(
            "iris",
            copies=1,
            automatic_encoding_selection=False,
        )
        old_payload = dict(payload)
        old_payload.pop("X_train_raw")
        old_payload.pop("X_test_raw")

        X_train, y_train, X_test, y_test = benchmark_raw_split(
            old_payload,
            dataset_key="iris",
            test_fraction=0.20,
            random_seed=42,
        )
        self.assertEqual(X_train.shape, (120, 4))
        self.assertEqual(X_test.shape, (30, 4))
        np.testing.assert_array_equal(y_train, payload["y_train"])
        np.testing.assert_array_equal(y_test, payload["y_test"])

    def test_catalog_contains_requested_classifier_families(self) -> None:
        keys = {spec.key for spec in CLASSIFIER_SPECS}
        self.assertEqual(len(keys), len(CLASSIFIER_SPECS))
        self.assertTrue(
            {
                "mlp",
                "random_forest",
                "bernoulli_nb",
                "knn",
                "qda",
                "logistic_regression",
                "extra_tree",
                "extra_trees",
                "rbf_svm",
            }.issubset(keys)
        )
        for key in keys:
            estimator, grid = build_classifier(
                key, n_train=120, random_seed=42
            )
            self.assertIsNotNone(estimator)
            self.assertTrue(grid)

    def test_tuning_and_metrics_use_a_held_out_test_set(self) -> None:
        X, y = make_classification(
            n_samples=180,
            n_features=8,
            n_informative=5,
            weights=[0.70, 0.30],
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        result = fit_standard_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            classifier_key="logistic_regression",
            random_seed=42,
            requested_tuning_folds=3,
        )
        self.assertEqual(result["tuning_folds"], 3)
        self.assertEqual(len(result["predictions"]), len(y_test))
        self.assertIn("C", result["best_parameters"])
        for metric_name in (
            "balanced_accuracy",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "cohen_kappa",
            "matthews_corrcoef",
            "roc_auc",
        ):
            self.assertIn(metric_name, result["metrics"])
            self.assertTrue(np.isfinite(result["metrics"][metric_name]))

    def test_paired_bootstrap_detects_winner_and_tie(self) -> None:
        y_true = np.asarray([0] * 50 + [1] * 50)
        pgm = y_true.copy()
        competitor = y_true.copy()
        competitor[::3] = 1 - competitor[::3]

        comparison = paired_balanced_accuracy_bootstrap(
            y_true,
            pgm,
            competitor,
            resamples=1_000,
            random_seed=42,
        )
        self.assertEqual(comparison["winner"], "pgm")
        self.assertGreater(comparison["confidence_lower"], 0.0)

        tie = paired_balanced_accuracy_bootstrap(
            y_true,
            pgm,
            pgm,
            resamples=500,
            random_seed=42,
        )
        self.assertEqual(tie["winner"], "tie")
        self.assertEqual(tie["difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
