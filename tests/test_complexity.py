from __future__ import annotations

import unittest
from math import comb

from complexity import (
    automatic_encoded_feature_count,
    implementation_feasibility,
    paper_complexities,
    representation_dimensions,
)
from data_catalog import DATASETS


class ComplexityTests(unittest.TestCase):
    def test_dimensions_match_paper(self) -> None:
        tensor, symmetric = representation_dimensions(4, 3)
        self.assertEqual(tensor, 4**3)
        self.assertEqual(symmetric, comb(4 + 3 - 1, 3))

    def test_complexity_substitution(self) -> None:
        rows = {row.method: row for row in paper_complexities(120, 4, 2, 3, 15)}
        self.assertEqual(rows["c-PGM"].working_dimension, 16)
        self.assertEqual(rows["r-PGM"].working_dimension, 10)
        self.assertEqual(rows["k-PGM"].prediction_time_proxy, 15 * 120)
        self.assertEqual(rows["c-PGM"].training_memory_elements, 120 * 16**2)

    def test_guard_rejects_large_tensor_space(self) -> None:
        estimate = implementation_feasibility(
            n_train=120,
            n_test=30,
            dimension=13,
            copies=4,
            memory_budget_bytes=2 * 1024**3,
        )
        self.assertFalse(estimate.feasible)
        self.assertTrue(estimate.reasons)

    def test_automatic_feature_budget_covers_every_dataset_and_copy(self) -> None:
        for dataset in DATASETS:
            for copies in range(1, 9):
                encoded = automatic_encoded_feature_count(
                    dataset.features,
                    copies,
                    dataset.classes,
                    max_tensor_dimension=512,
                    max_total_qubits=7,
                )
                tensor, symmetric = representation_dimensions(encoded, copies)
                total_qubits = (symmetric - 1).bit_length() + (
                    dataset.classes - 1
                ).bit_length()
                self.assertGreaterEqual(encoded, 2)
                self.assertLessEqual(encoded, dataset.features)
                self.assertLessEqual(tensor, 512)
                self.assertLessEqual(total_qubits, 7)


if __name__ == "__main__":
    unittest.main()
