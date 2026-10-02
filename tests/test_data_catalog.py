from __future__ import annotations

import unittest

from data_catalog import load_public_dataset


class DatasetCatalogTests(unittest.TestCase):
    def test_builtin_public_datasets_match_catalog(self) -> None:
        expected = {
            "iris": (150, 4, 3),
            "wine": (178, 13, 3),
        }
        for key, (samples, features, classes) in expected.items():
            with self.subTest(dataset=key):
                X, y, source = load_public_dataset(key)
                self.assertEqual(X.shape, (samples, features))
                self.assertEqual(y.nunique(), classes)
                self.assertIn("scikit-learn", source)


if __name__ == "__main__":
    unittest.main()
