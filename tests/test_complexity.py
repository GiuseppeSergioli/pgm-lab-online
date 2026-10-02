from __future__ import annotations

import unittest
from math import comb

from complexity import (
    implementation_feasibility,
    paper_complexities,
    representation_dimensions,
)


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


if __name__ == "__main__":
    unittest.main()
