"""Utilities for reproducible evaluation across several outer random seeds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt

import numpy as np
from scipy.stats import t as student_t


def _result_value(result: object, name: str, default: object = None) -> object:
    """Read a result field from either a dataclass-like object or a mapping."""

    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def select_payload_backend(payload: Mapping[str, object]) -> str:
    """Return the exact PGM backend represented by an experiment payload.

    Version 5.2.0 added the explicit ``computational_backend`` field. Public
    Streamlit deployments can briefly contain a newer ``app.py`` together with
    an older cached payload or an older ``experiment.py`` while a commit is
    propagating. This resolver keeps those payloads usable and never selects a
    non-materialized equivalent when a directly computed k-PGM/r-PGM result is
    available.
    """

    results = payload.get("results")
    if not isinstance(results, Mapping) or not results:
        raise ValueError("Il payload non contiene risultati PGM validi.")

    declared = payload.get("computational_backend")
    if isinstance(declared, str) and declared in results:
        return declared

    preferred_names = ("r-PGM", "k-PGM")
    declared_independent = payload.get("independent_methods", ())
    if isinstance(declared_independent, str):
        declared_independent = (declared_independent,)
    if not isinstance(declared_independent, Sequence):
        declared_independent = ()

    direct_candidates = [
        name
        for name in preferred_names
        if name in results and name in declared_independent
    ]
    if not direct_candidates:
        for name in preferred_names:
            if name not in results:
                continue
            execution_mode = str(
                _result_value(results[name], "execution_mode", "")
            ).lower()
            is_proxy = any(
                marker in execution_mode
                for marker in (
                    "non materializzata",
                    "not materialized",
                    "equivalente esatta via",
                    "exact equivalent via",
                )
            )
            if execution_mode and not is_proxy:
                direct_candidates.append(name)

    if direct_candidates:
        # If a legacy payload materialized both methods, choose the smaller exact
        # representation. r-PGM wins a tie to keep the choice deterministic.
        def backend_cost(name: str) -> tuple[float, int]:
            dimension = _result_value(
                results[name], "representation_dimension", float("inf")
            )
            try:
                numeric_dimension = float(dimension)
            except (TypeError, ValueError):
                numeric_dimension = float("inf")
            return numeric_dimension, preferred_names.index(name)

        return min(direct_candidates, key=backend_cost)

    # Very old payloads may not expose execution metadata. k-PGM was their
    # universally available numerical route; c-PGM is only a last-resort guard.
    for name in ("k-PGM", "r-PGM", "c-PGM"):
        if name in results:
            return name
    raise ValueError("Non è disponibile alcun risultato PGM riconosciuto.")


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
) -> dict[str, float | int | str | bool]:
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
    pgm_wins = int(np.sum(differences > tolerance))
    competitor_wins = int(np.sum(differences < -tolerance))
    exact_ties = int(np.sum(np.abs(differences) <= tolerance))
    comparison_count = int(pgm.size)
    return {
        "count": comparison_count,
        "pgm_mean": float(np.mean(pgm)),
        "pgm_std": float(np.std(pgm, ddof=1)),
        "competitor_mean": float(np.mean(competitor)),
        "competitor_std": float(np.std(competitor, ddof=1)),
        "difference_mean": difference_mean,
        "difference_std": difference_std,
        "confidence_lower": float(lower),
        "confidence_upper": float(upper),
        "confidence_level": float(confidence_level),
        "pgm_wins": pgm_wins,
        "competitor_wins": competitor_wins,
        "ties": exact_ties,
        "pgm_win_rate": pgm_wins / comparison_count,
        "competitor_win_rate": competitor_wins / comparison_count,
        "tie_rate": exact_ties / comparison_count,
        "exact_tie": exact_ties == comparison_count,
        "winner": winner,
    }
