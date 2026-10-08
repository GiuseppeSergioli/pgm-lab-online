"""Utilities for reproducible evaluation across several outer random seeds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt

import numpy as np
from scipy.stats import t as student_t


def evaluation_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    """Return a deterministic sequence of distinct seeds.

    Consecutive seeds make a repeated experiment easy to reproduce without ever
    selecting a favourable split after observing the test results.
    """

    if int(count) < 2:
        raise ValueError("La valutazione robusta richiede almeno due seed.")
    if int(base_seed) < 0:
        raise ValueError("Il seed iniziale non può essere negativo.")
    return tuple(int(base_seed) + offset for offset in range(int(count)))


def summarize_metrics(
    records: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float | int]]:
    """Compute mean and sample standard deviation for every metric."""

    if not records:
        raise ValueError("Non ci sono risultati da aggregare.")
    keys = tuple(records[0].keys())
    if any(tuple(record.keys()) != keys for record in records):
        raise ValueError("Tutti i record devono contenere le stesse metriche.")
    summary: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            mean = standard_deviation = float("nan")
        else:
            mean = float(np.mean(finite))
            standard_deviation = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
        summary[key] = {
            "mean": mean,
            "std": standard_deviation,
            "count": int(finite.size),
        }
    return summary


def paired_seed_summary(
    pgm_records: Sequence[Mapping[str, float]],
    competitor_records: Sequence[Mapping[str, float]],
    *,
    metric: str = "balanced_accuracy",
    confidence_level: float = 0.95,
) -> dict[str, float | int | str]:
    """Summarize a paired metric difference over repeated outer splits.

    The confidence interval is a two-sided Student-t interval over the paired
    seed-level differences. It complements, rather than replaces, the reported
    sample standard deviations.
    """

    if len(pgm_records) != len(competitor_records) or len(pgm_records) < 2:
        raise ValueError("Il confronto appaiato richiede almeno due seed comuni.")
    if not 0.5 < float(confidence_level) < 1.0:
        raise ValueError("Il livello di confidenza deve essere tra 0.5 e 1.")
    pgm = np.asarray([record[metric] for record in pgm_records], dtype=np.float64)
    competitor = np.asarray(
        [record[metric] for record in competitor_records], dtype=np.float64
    )
    keep = np.isfinite(pgm) & np.isfinite(competitor)
    pgm = pgm[keep]
    competitor = competitor[keep]
    if pgm.size < 2:
        raise ValueError("Meno di due differenze appaiate sono finite.")
    differences = pgm - competitor
    difference_mean = float(np.mean(differences))
    difference_std = float(np.std(differences, ddof=1))
    alpha = 1.0 - float(confidence_level)
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=pgm.size - 1))
    half_width = critical * difference_std / sqrt(int(pgm.size))
    lower = difference_mean - half_width
    upper = difference_mean + half_width
    if lower > 0.0:
        winner = "pgm"
    elif upper < 0.0:
        winner = "competitor"
    else:
        winner = "tie"
    tolerance = 1e-12
    return {
        "count": int(pgm.size),
        "pgm_mean": float(np.mean(pgm)),
        "pgm_std": float(np.std(pgm, ddof=1)),
        "competitor_mean": float(np.mean(competitor)),
        "competitor_std": float(np.std(competitor, ddof=1)),
        "difference_mean": difference_mean,
        "difference_std": difference_std,
        "confidence_lower": float(lower),
        "confidence_upper": float(upper),
        "confidence_level": float(confidence_level),
        "pgm_wins": int(np.sum(differences > tolerance)),
        "competitor_wins": int(np.sum(differences < -tolerance)),
        "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "winner": winner,
    }
