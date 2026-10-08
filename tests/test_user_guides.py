from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UserGuideTests(unittest.TestCase):
    def test_bilingual_guides_exist_and_match_the_app_version(self) -> None:
        app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        version_match = re.search(r'^APP_VERSION = "([^"]+)"', app_source, re.M)
        self.assertIsNotNone(version_match)
        app_version = version_match.group(1)

        manifest_path = PROJECT_ROOT / "assets" / "guides" / "guide_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["app_version"], app_version)
        self.assertEqual(set(manifest["files"]), {"it", "en"})

        generator = (PROJECT_ROOT / manifest["generator"]).read_text(
            encoding="utf-8"
        )
        self.assertIn(f'APP_VERSION = "{app_version}"', generator)
        for language, file_name in manifest["files"].items():
            guide_path = manifest_path.parent / file_name
            content = guide_path.read_bytes()
            self.assertTrue(content.startswith(b"%PDF-"), language)
            self.assertGreater(len(content), 30_000, language)
            self.assertGreaterEqual(content.count(b"/Type /Page"), 10, language)

    def test_app_selects_the_guide_from_the_active_language(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("GUIDE_MANIFEST_PATH", source)
        self.assertIn("guide_language = current_language()", source)
        self.assertIn('guide_manifest["files"][guide_language]', source)
        self.assertIn('mime="application/pdf"', source)
        self.assertIn('"📘 Scarica la guida PDF completa"', source)


if __name__ == "__main__":
    unittest.main()
