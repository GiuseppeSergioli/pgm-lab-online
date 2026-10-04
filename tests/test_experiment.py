from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score

from data_catalog import load_public_dataset
from experiment import (
    DEFAULT_RESCALING_FACTORS,
    _validation_splits,
    run_experiment,
    select_encoding_on_training,
    stereographic_encoding,
)


class EndToEndExperimentTests(unittest.TestCase):
    def test_default_iris_run_has_identical_predictions_and_accuracy(self) -> None:
        payload = run_experiment(
            "iris",
            copies=2,
            test_fraction=0.20,
            random_seed=42,
            prior_mode="uniform",
            relative_tolerance=1e-10,
            automatic_encoding_selection=False,
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
            automatic_encoding_selection=False,
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
            automatic_encoding_selection=False,
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

    def test_stereographic_encoding_is_unit_norm_and_depends_on_factor(self) -> None:
        X = np.asarray([[0.2, 0.7], [0.9, 0.1], [0.4, 0.4]])
        encoded_small = stereographic_encoding(X, 0.2)
        encoded_large = stereographic_encoding(X, 2.0)

        self.assertEqual(encoded_small.shape, (3, 3))
        np.testing.assert_allclose(
            np.linalg.norm(encoded_small, axis=1), 1.0, atol=1e-12
        )
        np.testing.assert_allclose(
            np.linalg.norm(encoded_large, axis=1), 1.0, atol=1e-12
        )
        self.assertFalse(
            np.allclose(
                encoded_small @ encoded_small.T,
                encoded_large @ encoded_large.T,
            )
        )

    def test_automatic_encoding_selection_keeps_all_methods_equivalent(self) -> None:
        payload = run_experiment(
            "iris",
            copies=2,
            test_fraction=0.20,
            random_seed=42,
            prior_mode="uniform",
            relative_tolerance=1e-10,
        )
        selection = payload["encoding_selection"]
        self.assertIn(payload["encoding"], {"tensor_l2", "stereographic"})
        self.assertEqual(
            len(selection["candidates"]), 1 + len(DEFAULT_RESCALING_FACTORS)
        )
        self.assertEqual(selection["tuning_samples"], len(payload["y_train"]))
        self.assertEqual(selection["folds"], 3)
        if payload["encoding"] == "stereographic":
            self.assertIn(payload["rescaling_factor"], DEFAULT_RESCALING_FACTORS)
            self.assertEqual(payload["d"], payload["base_d"] + 1)
        else:
            self.assertIsNone(payload["rescaling_factor"])
            self.assertEqual(payload["d"], payload["base_d"])

        reference = payload["results"]["c-PGM"]
        for name in ("k-PGM", "r-PGM"):
            np.testing.assert_array_equal(
                payload["results"][name].predictions, reference.predictions
            )
            np.testing.assert_allclose(
                payload["results"][name].scores,
                reference.scores,
                atol=3e-11,
                rtol=3e-11,
            )

    def test_validation_split_preserves_singleton_classes_in_fit(self) -> None:
        labels = np.asarray(["rare", "a", "a", "a", "b", "b", "b", "b"])
        splits, protocol = _validation_splits(labels, random_seed=42)
        fit_indices, validation_indices = splits[0]

        self.assertIn("classi rare protette", protocol)
        self.assertIn(0, fit_indices)
        self.assertNotIn(0, validation_indices)
        self.assertEqual(set(labels[fit_indices]), set(labels))
        self.assertGreater(len(validation_indices), 0)

    def test_selection_requires_more_than_minimum_gain(self) -> None:
        X, y, _ = load_public_dataset("iris")
        X_train = X.iloc[:120].reset_index(drop=True)
        y_train = y.iloc[:120].to_numpy()
        clearly_better_fold = [
            (0.800, "r-PGM"),
            (0.790, "r-PGM"),
            (0.800, "r-PGM"),
            (0.810, "r-PGM"),
            (0.805, "r-PGM"),
            (0.800, "r-PGM"),
        ]
        with patch(
            "experiment._fastest_exact_validation_accuracy",
            side_effect=clearly_better_fold * 3,
        ):
            selected = select_encoding_on_training(
                X_train,
                y_train,
                copies=2,
                random_seed=42,
                prior_mode="uniform",
                relative_tolerance=1e-10,
                tensor_max_encoded_features=4,
                stereographic_max_encoded_features=4,
            )
        self.assertEqual(selected["encoding"], "stereographic")
        self.assertEqual(selected["rescaling_factor"], 0.5)

        marginal_fold = [
            (0.800, "r-PGM"),
            (0.790, "r-PGM"),
            (0.800, "r-PGM"),
            (0.804, "r-PGM"),
            (0.801, "r-PGM"),
            (0.800, "r-PGM"),
        ]
        with patch(
            "experiment._fastest_exact_validation_accuracy",
            side_effect=marginal_fold * 3,
        ):
            selected = select_encoding_on_training(
                X_train,
                y_train,
                copies=2,
                random_seed=42,
                prior_mode="uniform",
                relative_tolerance=1e-10,
                tensor_max_encoded_features=4,
                stereographic_max_encoded_features=4,
            )
        self.assertEqual(selected["encoding"], "tensor_l2")
        self.assertIsNone(selected["rescaling_factor"])


if __name__ == "__main__":
    unittest.main()
