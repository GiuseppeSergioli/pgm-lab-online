from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ComparisonUiTests(unittest.TestCase):
    def test_comparison_is_between_preflight_and_pgm_results(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        preflight = source.index(
            'st.subheader("2. Controlla le dimensioni prima del calcolo")'
        )
        comparison = source.index(
            'st.subheader("3. Confronto con altri classificatori")'
        )
        results = source.index('st.subheader("4. Risultati PGM")')
        circuit = source.index(
            'st.subheader("5. Circuito quantistico della PGM")'
        )
        self.assertLess(preflight, comparison)
        self.assertLess(comparison, results)
        self.assertLess(results, circuit)

    def test_benchmarks_require_their_own_explicit_buttons(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('key="standard_comparison_enabled"', source)
        self.assertIn("value=False", source)
        self.assertIn("if compare_current_clicked:", source)
        self.assertIn("if full_comparison_clicked:", source)
        self.assertLess(
            source.index("if compare_current_clicked:"),
            source.index("selected_classifier_comparison"),
        )
        self.assertLess(
            source.index("if full_comparison_clicked:"),
            source.index('st.session_state["full_binary_comparison"]'),
        )

    def test_feature_reduction_is_manual_and_multiseed_is_default(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn('"Richiedi manualmente una riduzione PCA"', source)
        self.assertIn('key="manual_feature_reduction"', source)
        self.assertNotIn("automatic_encoded_feature_count", source)
        self.assertIn('"Numero di seed di valutazione"', source)
        self.assertIn('key="evaluation_seed_count"', source)
        self.assertIn("execute_multiseed_pgm", source)


if __name__ == "__main__":
    unittest.main()
