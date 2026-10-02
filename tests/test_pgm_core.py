from __future__ import annotations

import unittest

import numpy as np

from pgm_core import run_all_methods, symmetric_feature_map, tensor_feature_map


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


if __name__ == "__main__":
    unittest.main()
