from __future__ import annotations

import unittest

from robust_evaluation import (
    evaluation_seeds,
    paired_seed_summary,
    summarize_metrics,
)


class RobustEvaluationTests(unittest.TestCase):
    def test_seed_sequence_is_reproducible_and_requires_multiple_seeds(self) -> None:
        self.assertEqual(evaluation_seeds(42, 5), (42, 43, 44, 45, 46))
        with self.assertRaises(ValueError):
            evaluation_seeds(42, 1)

    def test_metric_summary_uses_sample_standard_deviation(self) -> None:
        summary = summarize_metrics(
            [
                {"balanced_accuracy": 0.7, "accuracy": 0.8},
                {"balanced_accuracy": 0.9, "accuracy": 0.8},
            ]
        )
        self.assertAlmostEqual(summary["balanced_accuracy"]["mean"], 0.8)
        self.assertAlmostEqual(
            summary["balanced_accuracy"]["std"], 2**0.5 / 10
        )
        self.assertEqual(summary["accuracy"]["std"], 0.0)

    def test_paired_summary_uses_every_seed(self) -> None:
        pgm = [
            {"balanced_accuracy": value}
            for value in (0.80, 0.82, 0.84, 0.86, 0.88)
        ]
        competitor = [
            {"balanced_accuracy": value}
            for value in (0.70, 0.72, 0.74, 0.76, 0.78)
        ]
        summary = paired_seed_summary(pgm, competitor)

        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(summary["difference_mean"], 0.10)
        self.assertEqual(summary["pgm_wins"], 5)
        self.assertEqual(summary["winner"], "pgm")


if __name__ == "__main__":
    unittest.main()
