from __future__ import annotations

import unittest

from robust_evaluation import (
    evaluation_seeds,
    paired_seed_summary,
    select_payload_backend,
    summarize_metrics,
)


class _Result:
    def __init__(self, dimension: int, mode: str) -> None:
        self.representation_dimension = dimension
        self.execution_mode = mode


class RobustEvaluationTests(unittest.TestCase):
    def test_declared_backend_is_used_when_available(self) -> None:
        payload = {
            "results": {"k-PGM": object(), "r-PGM": object()},
            "computational_backend": "k-PGM",
        }
        self.assertEqual(select_payload_backend(payload), "k-PGM")

    def test_missing_backend_is_recovered_from_independent_methods(self) -> None:
        payload = {
            "results": {
                "k-PGM": _Result(
                    100, "equivalente esatta via r-PGM (matrice non materializzata)"
                ),
                "r-PGM": _Result(15, "esplicita indipendente"),
            },
            "independent_methods": ("r-PGM",),
        }
        self.assertEqual(select_payload_backend(payload), "r-PGM")

    def test_legacy_payload_without_metadata_falls_back_to_k_pgm(self) -> None:
        payload = {"results": {"c-PGM": object(), "k-PGM": object()}}
        self.assertEqual(select_payload_backend(payload), "k-PGM")

    def test_proxy_is_never_preferred_over_direct_backend(self) -> None:
        payload = {
            "results": {
                "k-PGM": _Result(80, "kernel diretto indipendente"),
                "r-PGM": _Result(
                    10, "equivalente esatta via k-PGM (matrice non materializzata)"
                ),
            }
        }
        self.assertEqual(select_payload_backend(payload), "k-PGM")

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
        self.assertEqual(summary["pgm_win_rate"], 1.0)
        self.assertEqual(summary["competitor_win_rate"], 0.0)
        self.assertEqual(summary["tie_rate"], 0.0)
        self.assertFalse(summary["exact_tie"])
        self.assertEqual(summary["winner"], "pgm")

    def test_exact_tie_is_distinct_from_inconclusive_result(self) -> None:
        exact = paired_seed_summary(
            [{"balanced_accuracy": 0.8}] * 4,
            [{"balanced_accuracy": 0.8}] * 4,
        )
        inconclusive = paired_seed_summary(
            [
                {"balanced_accuracy": value}
                for value in (0.81, 0.79, 0.82, 0.78)
            ],
            [{"balanced_accuracy": 0.8}] * 4,
        )

        self.assertEqual(exact["winner"], "tie")
        self.assertTrue(exact["exact_tie"])
        self.assertEqual(exact["tie_rate"], 1.0)
        self.assertEqual(inconclusive["winner"], "tie")
        self.assertFalse(inconclusive["exact_tie"])
        self.assertEqual(inconclusive["pgm_win_rate"], 0.5)
        self.assertEqual(inconclusive["competitor_win_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
