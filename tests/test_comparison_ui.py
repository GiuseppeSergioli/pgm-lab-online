from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ComparisonUiTests(unittest.TestCase):
    def test_comparison_is_between_preflight_and_pgm_results(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        preflight = source.index('"2. Controlla le dimensioni prima del calcolo"')
        comparison = source.index('"3. Confronto con altri classificatori"')
        run_button = source.index('key="run_multiseed_pgm"')
        results = source.index('"4. Risultati PGM"')
        circuit = source.index('"5. Circuito quantistico della PGM"')
        self.assertLess(preflight, comparison)
        self.assertLess(comparison, run_button)
        self.assertLess(run_button, results)
        self.assertLess(results, circuit)

    def test_quantum_tools_are_previewed_before_the_main_run(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

        activity_map = source.index('key="activity_overview"')
        run_button = source.index('key="run_multiseed_pgm"')
        self.assertLess(activity_map, run_button)
        self.assertIn("render_quantum_tool_strip()", source)
        self.assertIn('key="preview_optimizer_disabled"', source)
        self.assertIn('key="placeholder_execution_disabled"', source)

    def test_circuit_limits_expand_only_the_safe_layers(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("EXACT_CIRCUIT_QUBIT_LIMIT = 10", source)
        self.assertIn("ISOLATED_SYNTHESIS_QUBIT_LIMIT = 8", source)
        self.assertIn("largest_manual_feature_count_for_circuit", source)
        self.assertIn("Nessuna feature viene ridotta", source)
        self.assertIn('"automaticamente."', source)

    def test_benchmarks_require_their_own_explicit_buttons(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('key="standard_comparison_enabled"', source)
        self.assertIn("value=False", source)
        self.assertIn("if compare_current_clicked:", source)
        self.assertIn("if full_comparison_clicked:", source)
        self.assertLess(
            source.index("if compare_current_clicked:"),
            source.index('st.session_state["selected_classifier_comparison"]'),
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

    def test_full_comparison_distinguishes_inconclusive_from_exact_tie(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn('"Non conclusivo"', source)
        self.assertIn('"Pareggio esatto"', source)
        self.assertIn('"pgm_win_rate"', source)
        self.assertIn('"competitor_win_rate"', source)
        self.assertIn('"tie_rate"', source)
        self.assertIn('"ci_lower"', source)
        self.assertIn('"ci_upper"', source)


if __name__ == "__main__":
    unittest.main()
