from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResponsiveUiConfigurationTests(unittest.TestCase):
    def test_streamlit_theme_is_explicitly_light_and_high_contrast(self) -> None:
        configuration = tomllib.loads(
            (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(configuration["theme"]["base"], "light")
        self.assertEqual(configuration["theme"]["backgroundColor"], "#FFFFFF")
        self.assertEqual(configuration["theme"]["textColor"], "#202531")
        for selectable_theme in ("light", "dark"):
            self.assertEqual(
                configuration["theme"][selectable_theme]["backgroundColor"],
                "#FFFFFF",
            )
            self.assertEqual(
                configuration["theme"][selectable_theme]["textColor"],
                "#202531",
            )

    def test_app_contains_phone_and_tablet_breakpoints(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        style = re.search(r"<style>(.*?)</style>", source, flags=re.DOTALL)

        self.assertIsNotNone(style)
        assert style is not None
        self.assertEqual(style.group(1).count("{"), style.group(1).count("}"))
        self.assertIn("@media (max-width: 900px)", source)
        self.assertIn("@media (max-width: 640px)", source)
        self.assertIn('data-testid="stHorizontalBlock"', source)
        self.assertIn("color-scheme: only light", source)


if __name__ == "__main__":
    unittest.main()
