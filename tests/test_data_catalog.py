from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from data_catalog import DATASETS, catalog_frame, load_public_dataset


class DatasetCatalogTests(unittest.TestCase):
    def test_builtin_public_datasets_match_catalog(self) -> None:
        expected = {
            "iris": (150, 4, 3),
            "wine": (178, 13, 3),
            "breast_cancer_diagnostic": (569, 30, 2),
        }
        for key, (samples, features, classes) in expected.items():
            with self.subTest(dataset=key):
                X, y, source = load_public_dataset(key)
                self.assertEqual(X.shape, (samples, features))
                self.assertEqual(y.nunique(), classes)
                self.assertIn("scikit-learn", source)

    def test_catalog_is_unique_and_complete(self) -> None:
        keys = [dataset.key for dataset in DATASETS]
        self.assertEqual(len(DATASETS), 26)
        self.assertEqual(len(keys), len(set(keys)))

        frame = catalog_frame()
        self.assertEqual(len(frame), len(DATASETS))
        self.assertEqual(
            list(frame.columns),
            [
                "Dataset",
                "Campioni",
                "Feature/campione",
                "Classi",
                "Ambito",
                "Repository",
            ],
        )

    def test_synthetic_datasets_are_reproducible(self) -> None:
        expected = {
            "two_moons": (600, 2, 2),
            "concentric_circles": (600, 2, 2),
            "gaussian_blobs": (600, 2, 3),
            "xor": (600, 2, 2),
            "three_spirals": (600, 2, 3),
        }
        for key, (samples, features, classes) in expected.items():
            with self.subTest(dataset=key):
                X_first, y_first, source = load_public_dataset(key)
                X_second, y_second, _ = load_public_dataset(key)
                self.assertEqual(X_first.shape, (samples, features))
                self.assertEqual(y_first.nunique(), classes)
                self.assertTrue(X_first.equals(X_second))
                self.assertTrue(y_first.equals(y_second))
                self.assertIn("seed=42", source)

    def test_banana_declares_reproducible_subset(self) -> None:
        banana = next(dataset for dataset in DATASETS if dataset.key == "banana")
        self.assertEqual(banana.openml_id, 1460)
        self.assertEqual(banana.raw_samples, 5300)
        self.assertEqual(banana.samples, 800)
        self.assertEqual(banana.subsample_seed, 42)

    def test_banana_subset_loader_is_stratified_and_reproducible(self) -> None:
        rng = np.random.default_rng(7)
        mock_banana = Bunch(
            data=pd.DataFrame(rng.normal(size=(5300, 2)), columns=["At1", "At2"]),
            target=pd.Series(np.repeat(["-1", "1"], [2900, 2400]), name="Class"),
        )
        with patch("data_catalog.fetch_openml", return_value=mock_banana):
            X_first, y_first, source = load_public_dataset("banana")
            X_second, y_second, _ = load_public_dataset("banana")

        self.assertEqual(X_first.shape, (800, 2))
        self.assertEqual(y_first.nunique(), 2)
        self.assertTrue(X_first.equals(X_second))
        self.assertTrue(y_first.equals(y_second))
        self.assertIn("800/5300", source)
        self.assertIn("seed=42", source)


if __name__ == "__main__":
    unittest.main()
