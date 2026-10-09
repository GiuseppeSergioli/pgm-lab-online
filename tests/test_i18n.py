from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from i18n import (
    DEFAULT_LANGUAGE,
    LocalizedStreamlit,
    localize_dataframe,
    normalize_language,
    translate_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeContext:
    def __init__(self, calls: list[dict] | None = None) -> None:
        self.calls = calls if calls is not None else []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def radio(self, label, options, **kwargs):
        formatter = kwargs["format_func"]
        self.calls.append(
            {
                "method": "radio",
                "label": label,
                "options": list(options),
                "formatted": [formatter(option) for option in options],
            }
        )
        return options[0]

    def tabs(self, labels, *args, **kwargs):
        self.calls.append({"method": "tabs", "labels": list(labels)})
        return [_FakeContext(self.calls) for _ in labels]

    def expander(self, label, *args, **kwargs):
        self.calls.append({"method": "expander", "label": label})
        return _FakeContext(self.calls)

    def dataframe(self, frame, **kwargs):
        self.calls.append({"method": "dataframe", "frame": frame})
        return frame


class InternationalizationTests(unittest.TestCase):
    def test_italian_is_default_and_unknown_language_falls_back_to_it(self) -> None:
        self.assertEqual(DEFAULT_LANGUAGE, "it")
        self.assertEqual(normalize_language(None), "it")
        self.assertEqual(normalize_language("fr"), "it")
        self.assertEqual(normalize_language("en"), "en")

    def test_translation_is_word_safe(self) -> None:
        translated = translate_text(
            "La classificazione usa 150 campioni e 3 classi.", "en"
        )
        self.assertIn("classification", translated)
        self.assertIn("150 samples", translated)
        self.assertIn("3 classes", translated)
        self.assertNotIn("classesficazione", translated)
        self.assertEqual(
            translate_text("Iris - 150 campioni, 4 feature, 3 classi", "en"),
            "Iris - 150 samples, 4 features, 3 classes",
        )

    def test_long_public_intro_is_fully_translated(self) -> None:
        intro = (
            "Confronto riproducibile tra **c-PGM**, **k-PGM** e **r-PGM (Rc-PGM)**. "
            "La classificazione conserva tutte le feature salvo riduzione PCA richiesta "
            "esplicitamente dall'utente e usa il backend esatto meno oneroso tra k-PGM e "
            "r-PGM. Le prestazioni sono aggregate su più split stratificati mediante media "
            "e deviazione standard. Sul solo training set, l'app sceglie l'encoding e il "
            "fattore di rescaling; quando le dimensioni lo consentono costruisce anche il "
            "circuito quantistico della PGM mediante una dilatazione di Naimark."
        )
        translated = translate_text(intro, "en")
        self.assertIn("Reproducible comparison", translated)
        self.assertIn("Naimark dilation", translated)
        self.assertNotIn("Confronto riproducibile", translated)
        self.assertIn("retains every feature", translated)
        self.assertIn("mean and standard deviation", translated)

    def test_dataframe_headers_index_and_text_cells_are_localized(self) -> None:
        frame = pd.DataFrame(
            {
                "Classe reale": ["✓ Corretta"],
                "Lettura rapida": [
                    "Corretta: la misura favorisce la classe A con probabilità 90.0%."
                ],
            },
            index=["Reale: A"],
        )
        localized = localize_dataframe(frame, "en")
        self.assertEqual(list(localized.columns), ["Actual class", "Quick reading"])
        self.assertEqual(list(localized.index), ["Actual: A"])
        self.assertIn("Correct", localized.iloc[0, 1])
        self.assertIn("probability", localized.iloc[0, 1])

    def test_choice_display_is_translated_but_return_value_is_stable(self) -> None:
        backend = _FakeContext()
        localized = LocalizedStreamlit(backend, lambda: "en")
        options = [
            "Tutte le feature originali",
            "Riduzione manuale PCA",
        ]
        selected = localized.radio("Gestione delle feature", options)

        self.assertEqual(selected, options[0])
        call = backend.calls[-1]
        self.assertEqual(call["label"], "Feature handling")
        self.assertEqual(
            call["formatted"],
            ["All original features", "Manual PCA reduction"],
        )
        self.assertEqual(call["options"], options)

    def test_tabs_and_expanders_are_translated(self) -> None:
        backend = _FakeContext()
        localized = LocalizedStreamlit(backend, lambda: "en")
        localized.tabs(["Circuito logico", "Esecuzione quantistica", "Esporta"])
        localized.expander("Diagnostica numerica")

        self.assertEqual(
            backend.calls[0]["labels"],
            ["Logical circuit", "Quantum execution", "Export"],
        )
        self.assertEqual(backend.calls[1]["label"], "Numerical diagnostics")

    def test_app_has_two_flags_and_no_internal_neumark_note(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('APP_VERSION = "5.5.0"', source)
        self.assertIn('"🇮🇹"', source)
        self.assertIn('"🇬🇧"', source)
        self.assertIn('LANGUAGE_SESSION_KEY = "pgm_interface_language"', source)
        self.assertNotIn("Neumark nel paper", source)

    def test_classifier_comparison_section_is_fully_translated(self) -> None:
        intro = (
            "Questa sezione è indipendente dal pulsante principale. Se la abiliti, "
            "confronta la PGM con un classificatore standard sullo stesso split "
            "train/test. Il modello standard viene ottimizzato con una ricerca compatta "
            "sul solo training set; la balanced accuracy decide il confronto ed è "
            "accompagnata da precision, recall, F1-score, Kappa di Cohen, coefficiente "
            "di Matthews e ROC-AUC."
        )
        translated = translate_text(intro, "en")
        self.assertIn("independent of the main run button", translated)
        self.assertIn("training data only", translated)
        self.assertNotIn("Questa sezione", translated)
        self.assertEqual(
            translate_text("3. Confronto con altri classificatori", "en"),
            "3. Comparison with other classifiers",
        )
        self.assertEqual(
            translate_text(
                "Foresta di alberi con pesi bilanciati e aggregazione robusta.",
                "en",
            ),
            "Tree ensemble with balanced weights and robust aggregation.",
        )

    def test_quantitative_full_comparison_labels_are_translated(self) -> None:
        self.assertEqual(
            translate_text("Non conclusivo", "en"), "Inconclusive"
        )
        self.assertEqual(
            translate_text("Pareggio esatto", "en"), "Exact tie"
        )
        self.assertEqual(
            translate_text("Vittoria Random Forest confermata", "en"),
            "Confirmed Random Forest win",
        )
        self.assertEqual(
            translate_text("Esito statistico", "en"),
            "Statistical outcome",
        )

    def test_private_upload_and_guide_are_fully_translated(self) -> None:
        self.assertEqual(
            translate_text("Carica un dataset personale", "en"),
            "Upload your own dataset",
        )
        upload_description = (
            "Trascina qui un file tabellare. L'app riconosce CSV, TSV/TXT ed Excel "
            "XLSX, propone automaticamente il target e usa soltanto le feature "
            "numeriche selezionate. Il file resta nella memoria della tua sessione: "
            "non viene salvato nel repository né condiviso con altri utenti."
        )
        translated_upload = translate_text(upload_description, "en")
        self.assertIn("Drag a tabular file here", translated_upload)
        self.assertNotIn("Trascina qui", translated_upload)
        self.assertEqual(
            translate_text("Guida illustrata all'uso", "en"),
            "Illustrated user guide",
        )
        self.assertEqual(
            translate_text("📘 Scarica la guida PDF completa", "en"),
            "📘 Download the complete PDF guide",
        )

    def test_automatic_encoding_labels_are_translated(self) -> None:
        self.assertEqual(
            translate_text("Encoding selezionato", "en"), "Selected encoding"
        )
        dynamic = (
            "La scelta è stata effettuata esclusivamente sul training set mediante "
            "3-fold stratificata sul training (120 campioni). Il test set non è "
            "stato consultato."
        )
        translated = translate_text(dynamic, "en")
        self.assertIn("training data exclusively", translated)
        self.assertIn("120 samples", translated)
        self.assertNotIn("consultato", translated)

    def test_extended_simulator_catalog_is_fully_translated(self) -> None:
        expected = {
            "🧩 Qiskit BasicSimulator locale — circuito ideale": (
                "🧩 Local Qiskit BasicSimulator — ideal circuit"
            ),
            "🔷 IBM Fake Backend locale — rumore da snapshot QPU": (
                "🔷 Local IBM Fake Backend — QPU snapshot noise"
            ),
            "🟧 Amazon Braket Local — state vector o density matrix": (
                "🟧 Amazon Braket Local — state vector or density matrix"
            ),
            "🟪 AQT Offline — ideale o rumoroso": (
                "🟪 AQT Offline — ideal or noisy"
            ),
            "☁️ AQT Cloud — simulatori autorizzati": (
                "☁️ AQT Cloud — authorized simulators"
            ),
        }
        for italian, english in expected.items():
            self.assertEqual(translate_text(italian, "en"), english)

        message = (
            "Simulatore di riferimento incluso in Qiskit. È ideale e più lento "
            "di Aer, ma non richiede componenti nativi aggiuntivi."
        )
        translated = translate_text(message, "en")
        self.assertIn("Reference simulator included with Qiskit", translated)
        self.assertNotIn("Simulatore di riferimento", translated)

    def test_new_workflow_and_quantum_lab_labels_are_translated(self) -> None:
        expected = {
            "Sezione 3 · opzionale": "Section 3 · optional",
            "Avvia la valutazione PGM multi-seed": (
                "Run the multi-seed PGM evaluation"
            ),
            "Mappa delle attività": "Activity map",
            "Sezione 5 · laboratorio quantistico": (
                "Section 5 · quantum laboratory"
            ),
            "Validazione matematica": "Mathematical validation",
            "Configura simulatore o QPU": "Configure a simulator or QPU",
        }
        for italian, english in expected.items():
            self.assertEqual(translate_text(italian, "en"), english)


if __name__ == "__main__":
    unittest.main()
