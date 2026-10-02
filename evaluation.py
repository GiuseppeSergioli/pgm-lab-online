"""Tables and diagnostics for classification and quantum-shot results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from hardware import prediction_from_counts


def confusion_frames(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    classes: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = confusion_matrix(y_true, y_predicted, labels=classes)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    row_labels = [f"Reale: {label}" for label in classes]
    column_labels = [f"Predetta: {label}" for label in classes]
    counts = pd.DataFrame(matrix, index=row_labels, columns=column_labels)
    normalized = pd.DataFrame(
        100.0 * percentages,
        index=row_labels,
        columns=column_labels,
    )
    return counts, normalized


def classification_details_frame(
    y_true: np.ndarray,
    predicted: np.ndarray,
    classes: np.ndarray,
    theoretical_probabilities: np.ndarray,
    circuit_probabilities: np.ndarray,
) -> pd.DataFrame:
    sorted_probabilities = np.sort(circuit_probabilities, axis=1)
    margins = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        if circuit_probabilities.shape[1] > 1
        else sorted_probabilities[:, -1]
    )
    correct = predicted == y_true
    confidences = np.max(circuit_probabilities, axis=1)
    deviations = np.max(
        np.abs(circuit_probabilities - theoretical_probabilities), axis=1
    )
    def quick_reading(
        is_correct: bool,
        predicted_label: object,
        confidence: float,
        margin: float,
    ) -> str:
        if margin >= 0.20 - 1e-12:
            interpretation = "decisione netta"
        elif margin >= 0.05 - 1e-12:
            interpretation = "decisione moderata"
        else:
            interpretation = "classi molto vicine: interpretare con cautela"
        return (
            f"{'Corretta' if is_correct else 'Errata'}: la misura favorisce la "
            f"classe {predicted_label} con probabilità {confidence:.1%}; "
            f"il distacco dalla seconda classe è {margin:.1%} ({interpretation})."
        )

    quick_readings = [
        quick_reading(is_correct, predicted_label, confidence, margin)
        for is_correct, predicted_label, confidence, margin in zip(
            correct, predicted, confidences, margins
        )
    ]
    details = pd.DataFrame(
        {
            "Campione test": np.arange(1, len(y_true) + 1),
            "Classe reale": y_true,
            "Classe predetta": predicted,
            "Esito": np.where(correct, "✓ Corretta", "✗ Errata"),
            "Confidenza": confidences,
            "Margine 1ª-2ª": margins,
            "Scostamento max |circuito-teoria|": deviations,
            "Lettura rapida": quick_readings,
        }
    )
    for class_index, class_label in enumerate(classes):
        details[f"P teorica [{class_label}]"] = theoretical_probabilities[
            :, class_index
        ]
        details[f"P circuito [{class_label}]"] = circuit_probabilities[:, class_index]
    return details


def counts_frame(
    counts: dict[str, int],
    classes: np.ndarray,
    *,
    outcome_qubits: int,
) -> pd.DataFrame:
    total = sum(int(value) for value in counts.values())
    rows = []
    for raw_bits, count in sorted(counts.items()):
        bits = str(raw_bits).replace(" ", "")[-outcome_qubits:]
        outcome = int(bits, 2)
        rows.append(
            {
                "Bitstring": bits,
                "Classe": (
                    str(classes[outcome]) if outcome < len(classes) else "Non assegnata"
                ),
                "Conteggi": int(count),
                "Frequenza": int(count) / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def execution_summary(
    counts: dict[str, int],
    classes: np.ndarray,
    theoretical_outcomes: np.ndarray,
    *,
    outcome_qubits: int,
) -> tuple[object, float, pd.DataFrame]:
    predicted, frequencies = prediction_from_counts(
        counts, classes, outcome_qubits=outcome_qubits
    )
    observed = np.zeros_like(theoretical_outcomes, dtype=np.float64)
    for bitstring, frequency in frequencies.items():
        outcome = int(bitstring, 2)
        if outcome < observed.size:
            observed[outcome] = frequency
    total_variation = float(
        0.5 * np.sum(np.abs(observed - theoretical_outcomes))
    )
    return (
        predicted,
        total_variation,
        counts_frame(counts, classes, outcome_qubits=outcome_qubits),
    )
