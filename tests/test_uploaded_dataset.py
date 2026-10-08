from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest

import pandas as pd

from uploaded_dataset import (
    UploadedDatasetError,
    inspect_feature_columns,
    prepare_uploaded_dataset,
    read_uploaded_table,
    suggest_target_column,
    spreadsheet_safe_csv_bytes,
    uploaded_dataset_key,
    valid_stratified_test_fractions,
)


class UploadedDatasetTests(unittest.TestCase):
    def test_xlsx_first_sheet_is_supported(self) -> None:
        buffer = BytesIO()
        pd.DataFrame(
            {
                "x": range(10),
                "target": ["A"] * 5 + ["B"] * 5,
            }
        ).to_excel(buffer, index=False)

        frame, metadata = read_uploaded_table(buffer.getvalue(), "example.xlsx")

        self.assertEqual(frame.shape, (10, 2))
        self.assertEqual(metadata["format"], "XLSX")

    def test_csv_is_parsed_and_target_is_suggested(self) -> None:
        content = b"x1,x2,label\n1,2,A\n3,4,B\n5,6,A\n7,8,B\n9,10,A\n11,12,B\n"
        frame, metadata = read_uploaded_table(content, "example.csv")

        self.assertEqual(frame.shape, (6, 3))
        self.assertEqual(metadata["delimiter"], ",")
        self.assertEqual(suggest_target_column(frame), "label")

    def test_semicolon_csv_supports_decimal_comma(self) -> None:
        content = (
            "x1;x2;classe\n1,5;2,5;A\n3,5;4,5;B\n5,5;6,5;A\n"
            "7,5;8,5;B\n9,5;10,5;A\n11,5;12,5;B\n"
        ).encode("utf-8")
        frame, _ = read_uploaded_table(content, "decimal.csv")

        self.assertAlmostEqual(float(frame.loc[0, "x1"]), 1.5)

    def test_identifier_and_text_columns_are_reported(self) -> None:
        content = (
            b"id,value,city,target\n1,1.0,Rome,A\n2,2.0,Paris,B\n"
            b"3,3.0,Rome,A\n4,4.0,Paris,B\n5,5.0,Rome,A\n6,6.0,Paris,B\n"
        )
        frame, _ = read_uploaded_table(content, "mixed.csv")
        report = inspect_feature_columns(frame, "target")

        self.assertIn("value", report["recommended"])
        self.assertIn("id", report["probable_identifiers"])
        self.assertIn("city", report["non_numeric"])

    def test_preparation_removes_missing_targets_and_preserves_missing_features(self) -> None:
        content = (
            b"x1,x2,target\n1,2,A\n2,,A\n3,4,A\n4,5,A\n5,6,A\n"
            b"6,7,B\n7,8,B\n8,9,B\n9,10,B\n10,11,B\n11,12,\n"
        )
        frame, _ = read_uploaded_table(content, "missing.csv")
        X, y, metadata = prepare_uploaded_dataset(
            frame,
            target_column="target",
            feature_columns=("x1", "x2"),
        )

        self.assertEqual(X.shape, (10, 2))
        self.assertEqual(len(y), 10)
        self.assertEqual(metadata["removed_missing_target_rows"], 1)
        self.assertEqual(metadata["missing_feature_values"], 1)

    def test_rare_target_class_is_rejected(self) -> None:
        content = b"x,target\n1,A\n2,A\n3,A\n4,B\n5,B\n"
        frame, _ = read_uploaded_table(content, "rare.csv")
        with self.assertRaises(UploadedDatasetError):
            prepare_uploaded_dataset(
                frame,
                target_column="target",
                feature_columns=("x",),
            )

    def test_empty_target_is_rejected_with_a_clear_validation_error(self) -> None:
        content = b"x,target\n1,\n2,\n3,\n4,\n5,\n6,\n7,\n8,\n9,\n10,\n"
        frame, _ = read_uploaded_table(content, "empty-target.csv")
        with self.assertRaisesRegex(
            UploadedDatasetError,
            "non contiene alcuna etichetta valida",
        ):
            prepare_uploaded_dataset(
                frame,
                target_column="target",
                feature_columns=("x",),
            )

    def test_dataset_key_changes_with_configuration(self) -> None:
        first = uploaded_dataset_key("abc", "target", ("x1", "x2"))
        second = uploaded_dataset_key("abc", "target", ("x1",))
        self.assertNotEqual(first, second)

    def test_test_fraction_options_protect_multiclass_stratification(self) -> None:
        fractions = valid_stratified_test_fractions(
            15,
            5,
            (0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
        )

        self.assertEqual(fractions, (0.30, 0.35, 0.40))

    def test_csv_export_neutralizes_spreadsheet_formulas(self) -> None:
        exported = spreadsheet_safe_csv_bytes(
            pd.DataFrame(
                {
                    "label": ["=2+2", "+SUM(A1:A2)", "safe"],
                    "score": [0.1, 0.2, 0.3],
                }
            )
        ).decode("utf-8")

        self.assertIn("'=2+2", exported)
        self.assertIn("'+SUM(A1:A2)", exported)
        self.assertIn("safe", exported)

    def test_streamlit_ui_keeps_upload_private_and_uses_it_in_comparisons(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source = (project_root / "app.py").read_text(encoding="utf-8")
        requirements = (project_root / "requirements.txt").read_text(
            encoding="utf-8"
        )

        self.assertIn('"Carica un dataset personale"', source)
        self.assertIn('type=["csv", "tsv", "txt", "xlsx"]', source)
        self.assertIn("PRIVATE_DATASET_SESSION_KEY", source)
        self.assertIn("selected_dataset_override", source)
        self.assertIn("binary_dataset_override", source)
        self.assertIn("dataset_override=dataset_override", source)
        self.assertIn("use_shared_cache=(selected_dataset_override is None)", source)
        self.assertIn("clear_dataset_dependent_session_results", source)
        self.assertIn("openpyxl==3.1.5", requirements)


if __name__ == "__main__":
    unittest.main()
