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
            "I tre calcoli usano rappresentazioni indipendenti, ma gli stessi dati, "
            "prior, split e soglia spettrale. Dopo il training, l'app costruisce anche "
            "il circuito quantistico della PGM mediante una dilatazione di Naimark."
        )
        translated = translate_text(intro, "en")
        self.assertIn("Reproducible comparison", translated)
        self.assertIn("Naimark dilation", translated)
        self.assertNotIn("Confronto riproducibile", translated)

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
            "Automatica quantum-ready (consigliata)",
            "Tutte le feature originali",
        ]
        selected = localized.radio("Gestione delle feature", options)

        self.assertEqual(selected, options[0])
        call = backend.calls[-1]
        self.assertEqual(call["label"], "Feature handling")
        self.assertEqual(
            call["formatted"],
            ["Automatic quantum-ready (recommended)", "All original features"],
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
        self.assertIn('APP_VERSION = "4.9.0"', source)
        self.assertIn('"🇮🇹"', source)
        self.assertIn('"🇬🇧"', source)
        self.assertIn('LANGUAGE_SESSION_KEY = "pgm_interface_language"', source)
        self.assertNotIn("Neumark nel paper", source)


if __name__ == "__main__":
    unittest.main()
