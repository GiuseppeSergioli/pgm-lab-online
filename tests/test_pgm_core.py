from __future__ import annotations

import unittest

import numpy as np

from pgm_core import (
    run_all_methods,
    run_all_methods_scalable,
    symmetric_feature_map,
    tensor_feature_map,
)


def unit_rows(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    values = rng.normal(size=(rows, columns))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class FeatureMapTests(unittest.TestCase):
    def test_explicit_and_symmetric_feature_maps_have_same_gram(self) -> None:
        for copies in (1, 2, 3, 4):
            with self.subTest(copies=copies):
                rng = np.random.default_rng(7)
                X = unit_rows(rng, rows=8, columns=3)
                explicit = tensor_feature_map(X, copies)
                reduced = symmetric_feature_map(X, copies)
                expected = np.power(X @ X.T, copies)

                np.testing.assert_allclose(
                    explicit @ explicit.T, expected, atol=2e-13, rtol=2e-13
                )
                np.testing.assert_allclose(
                    reduced @ reduced.T, expected, atol=2e-13, rtol=2e-13
                )


class ClassifierEquivalenceTests(unittest.TestCase):
    def test_three_classifiers_are_numerically_equivalent(self) -> None:
        for prior_mode in ("uniform", "empirical"):
            for copies in (1, 2, 3):
                with self.subTest(copies=copies, prior_mode=prior_mode):
                    rng = np.random.default_rng(123)
                    X_train = unit_rows(rng, rows=18, columns=3)
                    X_test = unit_rows(rng, rows=9, columns=3)
                    y_train = np.asarray(["A"] * 8 + ["B"] * 6 + ["C"] * 4)

                    results = run_all_methods(
                        X_train,
                        y_train,
                        X_test,
                        copies=copies,
                        prior_mode=prior_mode,
                        relative_tolerance=1e-11,
                    )

                    reference = results["c-PGM"]
                    for name in ("k-PGM", "r-PGM"):
                        np.testing.assert_allclose(
                            results[name].scores,
                            reference.scores,
                            atol=2e-9,
                            rtol=2e-9,
                        )
                        np.testing.assert_array_equal(
                            results[name].predictions, reference.predictions
                        )
                        self.assertEqual(results[name].rank, reference.rank)

    def test_scalable_mode_selects_kernel_when_it_is_smaller(self) -> None:
        rng = np.random.default_rng(321)
        X_train = unit_rows(rng, rows=18, columns=4)
        X_test = unit_rows(rng, rows=7, columns=4)
        y_train = np.asarray(["A"] * 6 + ["B"] * 6 + ["C"] * 6)

        results = run_all_methods_scalable(
            X_train,
            y_train,
            X_test,
            copies=3,
            relative_tolerance=1e-11,
            explicit_dimension_limit=32,
        )

        self.assertIn("k-PGM", results["c-PGM"].execution_mode)
        self.assertEqual(
            results["k-PGM"].execution_mode, "kernel diretto indipendente"
        )
        self.assertIn("k-PGM", results["r-PGM"].execution_mode)
        self.assertTrue(np.isnan(results["c-PGM"].train_seconds))
        self.assertTrue(np.isnan(results["r-PGM"].train_seconds))
        self.assertEqual(results["c-PGM"].representation_dimension, 4**3)
        self.assertEqual(results["r-PGM"].representation_dimension, 20)
        for name in ("c-PGM", "r-PGM"):
            np.testing.assert_array_equal(
                results[name].predictions, results["k-PGM"].predictions
            )
            np.testing.assert_allclose(
                results[name].scores,
                results["k-PGM"].scores,
                atol=2e-9,
                rtol=2e-9,
            )

    def test_scalable_mode_selects_reduced_for_many_samples(self) -> None:
        rng = np.random.default_rng(99)
        X_train = unit_rows(rng, rows=60, columns=2)
        X_test = unit_rows(rng, rows=8, columns=2)
        y_train = np.asarray(["A"] * 30 + ["B"] * 30)

        results = run_all_methods_scalable(
            X_train,
            y_train,
            X_test,
            copies=3,
            relative_tolerance=1e-11,
            explicit_dimension_limit=32,
        )

        self.assertEqual(
            results["r-PGM"].execution_mode, "esplicita indipendente"
        )
        self.assertIn("r-PGM", results["k-PGM"].execution_mode)
        self.assertIn("r-PGM", results["c-PGM"].execution_mode)
        for name in ("c-PGM", "k-PGM"):
            np.testing.assert_array_equal(
                results[name].predictions, results["r-PGM"].predictions
            )
            np.testing.assert_allclose(
                results[name].scores,
                results["r-PGM"].scores,
                atol=2e-9,
                rtol=2e-9,
            )


if __name__ == "__main__":
    unittest.main()
