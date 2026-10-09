"""Streamlit interface for comparing the three equivalent PGM classifiers."""

from __future__ import annotations

import hashlib
from html import escape
import importlib.util
import json
from math import ceil
import os
from pathlib import Path

# Limit native linear-algebra pools before importing NumPy, SciPy or scikit-learn.
# This is intentionally in code as well as in deployment secrets, so a fresh
# Streamlit build remains safe even when no environment variables are configured.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd
import streamlit as _st
from sklearn.metrics import accuracy_score

from classifier_benchmark import (
    CLASSIFIER_SPECS,
    METRIC_LABELS,
    benchmark_raw_split,
    classification_metrics,
    fit_standard_classifier,
    get_classifier_spec,
    paired_balanced_accuracy_bootstrap,
)
from complexity import (
    human_bytes,
    paper_complexities,
    representation_dimensions,
    scientific_integer,
)
from data_catalog import DATASETS, DatasetSpec, catalog_frame
from evaluation import (
    classification_details_frame,
    confusion_frames,
    execution_summary,
)
from experiment import run_experiment
from hardware import (
    aws_available_profiles,
    build_sample_circuit,
    circuit_from_qpy,
    discover_aqt_simulators,
    discover_aws_devices,
    discover_ibm_devices,
    discover_ibm_fake_backends,
    discover_ionq_devices,
    discover_lrz_backends,
    generic_job_counts,
    generic_job_status,
    hardware_python_status,
    simulate_aqt_offline_shots,
    simulate_aer_shots,
    simulate_basic_shots,
    simulate_braket_local_shots,
    simulate_ibm_fake_shots,
    simulate_ideal_shots,
    submit_aqt_job,
    submit_aws_job,
    submit_ibm_job,
    submit_ionq_job,
    submit_lrz_job,
    verify_aws_identity,
)
from i18n import (
    DEFAULT_LANGUAGE,
    LocalizedStreamlit,
    localize_dataframe,
    normalize_language,
    translate_text,
)
from pgm_core import MethodResult, stable_predictions, symmetric_feature_map
from robust_evaluation import (
    evaluation_seeds,
    paired_seed_summary,
    select_payload_backend,
    summarize_metrics,
)
from uploaded_dataset import (
    UploadedDatasetError,
    display_name_from_file,
    inspect_feature_columns,
    prepare_uploaded_dataset,
    read_uploaded_table,
    sanitize_display_name,
    spreadsheet_safe_csv_bytes,
    suggest_target_column,
    uploaded_dataset_key,
    valid_stratified_test_fractions,
)
from quantum_pgm import (
    best_certified_transpilation,
    build_naimark_dilation,
    build_qiskit_circuit,
    build_reduced_pgm_measurement,
    circuit_svg,
    dilation_npz_bytes,
    entangling_gate_count,
    generic_unitary_cnot_upper_bound,
    isolated_transpile_qpy,
    outcome_probabilities,
    qpy_bytes,
    resources_for_dataset,
    symbolic_dilation,
)


_st.set_page_config(
    page_title="PGM Lab · c-PGM · k-PGM · r-PGM",
    page_icon="⚛️",
    layout="wide",
)

_st.markdown(
    """
    <style>
      /*
       * The app deliberately uses one coherent light palette.  Some mobile
       * browsers report a dark system theme to Streamlit while retaining the
       * custom white background; without these fallbacks that produces white
       * text on white.  config.toml sets the canonical theme and the rules below
       * protect the page if a browser has cached a different preference.
       */
      :root {color-scheme: only light;}
      html, body, [data-testid="stAppViewContainer"], .stApp {color-scheme: light;}
      .stApp {
        background: linear-gradient(180deg, #f7f9ff 0%, #ffffff 28rem);
        color: #202531;
      }
      .block-container {
        width: 100%; max-width: 1320px;
        padding-top: calc(4.5rem + env(safe-area-inset-top));
        padding-bottom: 3rem;
      }
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stWidgetLabel"],
      .stApp [data-testid="stExpander"] summary {color: #202531;}
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stMarkdownContainer"] li,
      .stApp [data-testid="stWidgetLabel"] p,
      .stApp [data-testid="stExpander"] summary p {color: #202531;}
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stCaptionContainer"] p,
      .small-note {color: #5f6b7a;}
      .stApp [data-baseweb="select"] > div,
      .stApp [data-baseweb="input"] > div,
      .stApp input, .stApp textarea {
        background-color: #ffffff; color: #202531;
      }
      .stApp [data-testid="stFileUploaderDropzone"] {
        background: #f7f9ff;
        border: 2px dashed #7d8fc9;
        border-radius: 0.9rem;
        padding: 1rem;
      }
      .stApp [data-testid="stFileUploaderDropzone"]:hover {
        background: #eef2ff;
        border-color: #5268ad;
      }
      .stApp [data-baseweb="select"] *,
      .stApp [data-baseweb="input"] *,
      .stApp [data-testid="stRadio"] label,
      .stApp [data-testid="stRadio"] label p,
      .stApp [data-testid="stCheckbox"] label,
      .stApp [data-testid="stCheckbox"] label p,
      .stApp [data-testid="stToggle"] label,
      .stApp [data-testid="stToggle"] label p {color: #202531;}
      [data-testid="stMetricValue"] {font-size: 1.55rem; color: #202531;}
      [data-testid="stMetricLabel"] {color: #4d586b;}
      [data-testid="stMetric"] {
        background: rgba(255,255,255,.88); border: 1px solid #e1e6f0;
        border-radius: 14px; padding: .75rem 1rem; box-shadow: 0 5px 18px rgba(33,43,74,.05);
        min-width: 0;
      }
      [data-baseweb="tab-list"] {gap: .35rem; flex-wrap: wrap;}
      [data-testid="stTabs"] {
        margin-top: .45rem; padding: .65rem .7rem .85rem;
        background: rgba(255,255,255,.82); border: 1px solid #d9d4ee;
        border-radius: 14px; box-shadow: 0 6px 18px rgba(33,43,74,.05);
      }
      [data-baseweb="tab"] {
        background: #edf1fb; border-radius: 10px 10px 0 0; padding: .55rem .9rem;
        color: #263247;
      }
      [data-baseweb="tab"] p {color: #263247;}
      [data-baseweb="tab"][aria-selected="true"] {background: #ded8f8;}
      [data-testid="stAlert"] {border-radius: 12px;}
      div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px; max-width: 100%; min-height: 2.75rem;
        white-space: normal; overflow-wrap: anywhere;
        border: 1px solid #8b82bd; font-weight: 650;
        box-shadow: 0 4px 12px rgba(66, 53, 120, .10);
        transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
      }
      div.stButton > button:not([kind="primary"]),
      div.stDownloadButton > button {
        background: linear-gradient(180deg, #ffffff 0%, #f2effc 100%);
        color: #342a61;
      }
      div.stButton > button:hover:not(:disabled),
      div.stDownloadButton > button:hover:not(:disabled) {
        transform: translateY(-1px); border-color: #6f42c1;
        box-shadow: 0 7px 16px rgba(66, 53, 120, .16);
      }
      div.stButton > button:disabled,
      div.stDownloadButton > button:disabled {
        opacity: .58; box-shadow: none; background: #f2f3f7;
      }
      div.stButton > button[kind="primary"] p {color: #ffffff;}
      .section-banner {
        margin: 1.7rem 0 .85rem; padding: 1rem 1.15rem;
        background: linear-gradient(105deg, #ffffff 0%, #f3f1fc 100%);
        border: 1px solid #d9d4ee; border-left: 5px solid #6f42c1;
        border-radius: 14px; box-shadow: 0 7px 22px rgba(33,43,74,.06);
      }
      .section-banner .section-kicker {
        color: #6f42c1; font-size: .76rem; font-weight: 800;
        letter-spacing: .08em; text-transform: uppercase;
      }
      .section-banner h2 {margin: .12rem 0 .25rem; font-size: 1.55rem;}
      .section-banner p {margin: 0; color: #596579;}
      .quantum-tool-strip {
        margin: .65rem 0 1rem; padding: .8rem 1rem;
        border: 1px solid #cfc7ed; border-radius: 13px;
        background: linear-gradient(90deg, #f7f5ff 0%, #eef8ff 100%);
        color: #302657; font-weight: 650;
      }
      .st-key-pgm_run_panel {
        border-color: #9a8bd6 !important;
        box-shadow: 0 8px 22px rgba(66,53,120,.09);
      }
      .st-key-activity_overview {
        border-color: #cfc7ed !important;
        background: rgba(255,255,255,.72);
      }
      .st-key-language_switcher {
        margin-bottom: -.65rem; padding-top: .2rem; overflow: visible;
      }
      .st-key-language_switcher [data-testid="stImage"] img {
        width: 4.1rem; height: 4.1rem; object-fit: cover;
        border-radius: 50%; border: 2px solid #7968cf;
        box-shadow: 0 4px 14px rgba(48,38,87,.20);
      }
      .st-key-language_switcher div.stButton > button {
        min-width: 3rem; min-height: 2.45rem; padding: .25rem .65rem;
        font-size: 1.25rem; line-height: 1;
      }
      [data-testid="stRadio"] [role="radiogroup"] {
        flex-wrap: wrap; row-gap: .45rem; column-gap: 1rem;
      }
      [data-testid="stRadio"] label {max-width: 100%; white-space: normal;}
      [data-testid="stDataFrame"], [data-testid="stTable"] {max-width: 100%;}
      .stApp svg[role="img"], [data-testid="stImage"] img {
        display: block; max-width: 100%; height: auto;
      }
      .small-note {font-size: 0.92rem;}
      p, li, label, summary, button {overflow-wrap: anywhere;}

      /* Tablet: columns wrap instead of being squeezed beyond readability. */
      @media (max-width: 900px) {
        .block-container {
          max-width: 100%;
          padding: calc(4.25rem + env(safe-area-inset-top)) 1rem 2.5rem;
        }
        [data-testid="stHorizontalBlock"] {
          flex-wrap: wrap; gap: .75rem;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
          flex: 1 1 calc(50% - .75rem) !important;
          width: auto !important; min-width: 16rem !important;
        }
        [data-baseweb="tab-list"] {row-gap: .45rem;}
        [data-baseweb="tab"] {flex: 1 1 auto; justify-content: center;}
      }

      /* Phone: a single readable column, touch-sized controls and compact type. */
      @media (max-width: 640px) {
        .block-container {
          padding-top: calc(4.25rem + env(safe-area-inset-top));
          padding-bottom: 5rem;
          padding-left: max(.75rem, env(safe-area-inset-left));
          padding-right: max(.75rem, env(safe-area-inset-right));
        }
        .stApp h1 {font-size: 2rem; line-height: 1.15;}
        .stApp h2 {font-size: 1.55rem; line-height: 1.22;}
        .stApp h3 {font-size: 1.25rem; line-height: 1.25;}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
          flex: 1 1 100% !important;
          width: 100% !important; min-width: 0 !important;
        }
        [data-testid="stMetric"] {padding: .65rem .8rem;}
        [data-testid="stMetricValue"] {font-size: 1.3rem;}
        [data-baseweb="tab"] {
          flex: 1 1 calc(50% - .35rem); min-width: 8.5rem;
          padding: .5rem .55rem;
        }
        [data-testid="stSelectbox"], [data-testid="stNumberInput"],
        [data-testid="stTextInput"], [data-testid="stTextArea"],
        [data-testid="stSlider"], [data-testid="stSelectSlider"] {
          width: 100%; min-width: 0;
        }
        [data-baseweb="select"] > div {height: auto; min-height: 3rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


LANGUAGE_SESSION_KEY = "pgm_interface_language"
PRIVATE_DATASET_SESSION_KEY = "pgm_private_uploaded_dataset"
PRIVATE_UPLOAD_PARSE_CACHE_KEY = "pgm_private_upload_parse_cache"
PRIVATE_UPLOAD_WIDGET_NONCE_KEY = "pgm_private_upload_widget_nonce"
PRIVATE_DATASET_RESULT_KEYS = (
    "last_pgm_run",
    "selected_classifier_comparison",
    "full_binary_comparison",
    "native_synthesis_report",
    "certified_optimization",
    "local_quantum_result",
    "aer_quantum_result",
    "hardware_preflight",
)
if LANGUAGE_SESSION_KEY not in _st.session_state:
    _st.session_state[LANGUAGE_SESSION_KEY] = DEFAULT_LANGUAGE


def current_language() -> str:
    return normalize_language(_st.session_state.get(LANGUAGE_SESSION_KEY))


def localized_filename(italian: str, english: str) -> str:
    """Use readable download names without changing any exported content."""

    return english if current_language() == "en" else italian


st = LocalizedStreamlit(_st, current_language)


PRIOR_LABELS = {
    "Uniformi tra classi (p_j = 1/l)": "uniform",
    "Empirici (p_j = n_j/N)": "empirical",
}

APP_VERSION = "5.5.0"
EXPERIMENT_CACHE_SCHEMA = "5.3.0-private-dataset-upload"
EXACT_CIRCUIT_QUBIT_LIMIT = 10
ISOLATED_SYNTHESIS_QUBIT_LIMIT = 8
FULL_GATE_DIAGRAM_LIMIT = 5_000
AUTOMATIC_OPTIMIZATION_LEVELS = (1, 2, 3)
SCALABLE_EXPLICIT_DIMENSION_LIMIT = 2_000
GUIDE_DIRECTORY = Path(__file__).resolve().parent / "assets" / "guides"
GUIDE_MANIFEST_PATH = GUIDE_DIRECTORY / "guide_manifest.json"
BRAND_BADGE_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "branding"
    / "giuseppe_sergioli_quantum_badge.png"
)


def render_section_banner(
    section_label: str,
    title: str,
    description: str,
) -> None:
    """Render a compact visual separator without changing application state."""

    language = current_language()
    _st.markdown(
        (
            '<div class="section-banner">'
            f'<div class="section-kicker">{escape(translate_text(section_label, language))}</div>'
            f'<h2>{escape(translate_text(title, language))}</h2>'
            f'<p>{escape(translate_text(description, language))}</p>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_quantum_tool_strip() -> None:
    """Keep the complete quantum workflow visible in both languages."""

    labels = (
        "Circuito logico",
        "Classificazione test",
        "Validazione matematica",
        "Optimizer",
        "Esecuzione quantistica",
        "Esporta",
    )
    localized = " · ".join(
        escape(translate_text(label, current_language())) for label in labels
    )
    _st.markdown(
        f'<div class="quantum-tool-strip">{localized}</div>',
        unsafe_allow_html=True,
    )


def largest_manual_feature_count_for_circuit(
    *,
    copies: int,
    class_count: int,
    qubit_limit: int,
    maximum_base_features: int,
) -> int:
    """Largest PCA dimension that fits both automatic encoding candidates."""

    for feature_count in range(int(maximum_base_features), 0, -1):
        # The stereographic candidate adds one coordinate and is therefore the
        # conservative preview used before train-only encoding selection.
        candidate = resources_for_dataset(
            feature_count + 1,
            copies,
            class_count,
            exact_qubit_limit=qubit_limit,
        )
        if candidate.total_qubits <= qubit_limit:
            return feature_count
    return 0


def _run_experiment_for_app(
    dataset_key: str,
    copies: int,
    test_fraction: float,
    random_seed: int,
    prior_mode: str,
    relative_tolerance: float,
    tensor_max_encoded_features: int | None,
    stereographic_max_encoded_features: int | None,
    scalable_full_features: bool,
    dataset_override: tuple[pd.DataFrame, pd.Series, str] | None,
    cache_schema: str,
) -> dict:
    if cache_schema != EXPERIMENT_CACHE_SCHEMA:
        raise ValueError("Schema della cache dell'esperimento non riconosciuto.")
    payload = run_experiment(
        dataset_key,
        copies=copies,
        test_fraction=test_fraction,
        random_seed=random_seed,
        prior_mode=prior_mode,
        relative_tolerance=relative_tolerance,
        max_encoded_features=tensor_max_encoded_features,
        stereographic_max_encoded_features=stereographic_max_encoded_features,
        scalable_full_features=scalable_full_features,
        explicit_dimension_limit=SCALABLE_EXPLICIT_DIMENSION_LIMIT,
        dataset_override=dataset_override,
    )
    payload["computational_backend"] = select_payload_backend(payload)
    return payload


@st.cache_data(show_spinner=False, max_entries=24)
def _execute_public_experiment(
    dataset_key: str,
    copies: int,
    test_fraction: float,
    random_seed: int,
    prior_mode: str,
    relative_tolerance: float,
    tensor_max_encoded_features: int | None,
    stereographic_max_encoded_features: int | None,
    scalable_full_features: bool,
    cache_schema: str,
) -> dict:
    return _run_experiment_for_app(
        dataset_key,
        copies,
        test_fraction,
        random_seed,
        prior_mode,
        relative_tolerance,
        tensor_max_encoded_features,
        stereographic_max_encoded_features,
        scalable_full_features,
        None,
        cache_schema,
    )


def execute_experiment(
    dataset_key: str,
    copies: int,
    test_fraction: float,
    random_seed: int,
    prior_mode: str,
    relative_tolerance: float,
    tensor_max_encoded_features: int | None,
    stereographic_max_encoded_features: int | None,
    scalable_full_features: bool,
    dataset_override: tuple[pd.DataFrame, pd.Series, str] | None,
    cache_schema: str,
) -> dict:
    """Cache public datasets, while keeping private uploads session-local."""

    arguments = (
        dataset_key,
        copies,
        test_fraction,
        random_seed,
        prior_mode,
        relative_tolerance,
        tensor_max_encoded_features,
        stereographic_max_encoded_features,
        scalable_full_features,
    )
    if dataset_override is None:
        return _execute_public_experiment(*arguments, cache_schema)
    return _run_experiment_for_app(
        *arguments,
        dataset_override,
        cache_schema,
    )


def _fit_standard_classifier_for_app(
    X_train_raw,
    y_train: np.ndarray,
    X_test_raw,
    y_test: np.ndarray,
    classifier_key: str,
    random_seed: int,
    requested_tuning_folds: int,
) -> dict:
    return fit_standard_classifier(
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        classifier_key=classifier_key,
        random_seed=random_seed,
        requested_tuning_folds=requested_tuning_folds,
    )


@st.cache_data(show_spinner=False, max_entries=64)
def _execute_public_standard_classifier(
    X_train_raw,
    y_train: np.ndarray,
    X_test_raw,
    y_test: np.ndarray,
    classifier_key: str,
    random_seed: int,
    requested_tuning_folds: int,
) -> dict:
    return _fit_standard_classifier_for_app(
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        classifier_key,
        random_seed,
        requested_tuning_folds,
    )


def execute_standard_classifier(
    X_train_raw,
    y_train: np.ndarray,
    X_test_raw,
    y_test: np.ndarray,
    classifier_key: str,
    random_seed: int,
    requested_tuning_folds: int,
    *,
    use_shared_cache: bool,
) -> dict:
    """Fit privately uploaded data without placing it in Streamlit's global cache."""

    runner = (
        _execute_public_standard_classifier
        if use_shared_cache
        else _fit_standard_classifier_for_app
    )
    return runner(
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        classifier_key,
        random_seed,
        requested_tuning_folds,
    )


def parse_uploaded_table_for_session(
    content: bytes,
    file_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Parse once inside the current session; never place private data in a global cache."""

    content_digest = hashlib.sha256(content).hexdigest()
    cached = _st.session_state.get(PRIVATE_UPLOAD_PARSE_CACHE_KEY)
    if (
        cached is not None
        and cached.get("digest") == content_digest
        and cached.get("file_name") == file_name
    ):
        return cached["frame"], cached["metadata"]
    frame, metadata = read_uploaded_table(content, file_name)
    _st.session_state[PRIVATE_UPLOAD_PARSE_CACHE_KEY] = {
        "digest": content_digest,
        "file_name": file_name,
        "frame": frame,
        "metadata": metadata,
    }
    return frame, metadata


def clear_dataset_dependent_session_results() -> None:
    """Discard calculations tied to a dataset that is being replaced or removed."""

    for result_key in PRIVATE_DATASET_RESULT_KEYS:
        _st.session_state.pop(result_key, None)


@st.cache_data(show_spinner=False, max_entries=4)
def read_public_guide_asset(path_text: str, app_version: str) -> bytes:
    """Read a version-keyed, public guide file shipped with the application."""

    if app_version != APP_VERSION:
        raise ValueError("Versione della guida non coerente con l'app.")
    return Path(path_text).read_bytes()


def requested_feature_count(
    *,
    raw_feature_count: int,
    manual_reduction: bool,
    manual_feature_count: int,
) -> int:
    """Return the user-requested feature count, never an automatic reduction."""

    if int(raw_feature_count) < 1:
        raise ValueError("Il numero di feature deve essere positivo.")
    if not manual_reduction:
        return int(raw_feature_count)
    if not 1 <= int(manual_feature_count) <= int(raw_feature_count):
        raise ValueError("Il numero manuale di feature non è valido.")
    return int(manual_feature_count)


def pgm_benchmark_metrics(payload: dict) -> dict[str, float]:
    classes = np.unique(payload["y_train"])
    result = payload["results"][select_payload_backend(payload)]
    return classification_metrics(
        payload["y_test"],
        result.predictions,
        classes=classes,
        scores=result.scores,
    )


def compact_comparison_result(
    payload: dict,
    standard_result: dict,
    *,
    bootstrap_resamples: int,
    random_seed: int,
) -> dict:
    computational_backend = select_payload_backend(payload)
    pgm_result = payload["results"][computational_backend]
    bootstrap = paired_balanced_accuracy_bootstrap(
        payload["y_test"],
        pgm_result.predictions,
        standard_result["predictions"],
        resamples=bootstrap_resamples,
        random_seed=random_seed,
    )
    return {
        "pgm_metrics": pgm_benchmark_metrics(payload),
        "standard_metrics": standard_result["metrics"],
        "bootstrap": bootstrap,
        "y_test": payload["y_test"],
        "y_train": payload["y_train"],
        "pgm_predictions": pgm_result.predictions,
        "standard_predictions": standard_result["predictions"],
        "classifier_key": standard_result["classifier_key"],
        "classifier_name": standard_result["classifier_name"],
        "best_parameters": standard_result["best_parameters"],
        "tuning_folds": standard_result["tuning_folds"],
        "validation_balanced_accuracy": standard_result[
            "validation_balanced_accuracy"
        ],
        "tuning_message": standard_result["tuning_message"],
        "fit_seconds": standard_result["fit_seconds"],
        "predict_seconds": standard_result["predict_seconds"],
        "encoding": payload["encoding"],
        "rescaling_factor": payload["rescaling_factor"],
        "base_d": payload["base_d"],
        "d": payload["d"],
        "source_used": payload["source_used"],
        "computational_backend": computational_backend,
    }


def aggregate_seed_comparison(
    pgm_evaluation: dict,
    standard_results: list[dict],
    *,
    bootstrap_resamples: int,
    base_seed: int,
) -> dict:
    """Combine repeated outer splits and retain one split for diagnostics."""

    payloads = pgm_evaluation["payloads"]
    if len(payloads) != len(standard_results) or len(payloads) < 2:
        raise ValueError("Il confronto multi-seed richiede risultati appaiati.")
    pgm_metric_records = [record["metrics"] for record in pgm_evaluation["records"]]
    standard_metric_records = [result["metrics"] for result in standard_results]
    reference = compact_comparison_result(
        payloads[0],
        standard_results[0],
        bootstrap_resamples=bootstrap_resamples,
        random_seed=base_seed,
    )
    reference.update(
        {
            "seed_count": len(payloads),
            "seeds": tuple(pgm_evaluation["seeds"]),
            "pgm_metric_summary": summarize_metrics(pgm_metric_records),
            "standard_metric_summary": summarize_metrics(
                standard_metric_records
            ),
            "paired_seed_summary": paired_seed_summary(
                pgm_metric_records,
                standard_metric_records,
            ),
            "seed_rows": [
                {
                    "Seed": int(seed),
                    "Balanced accuracy PGM": pgm_metrics[
                        "balanced_accuracy"
                    ],
                    "Balanced accuracy confronto": standard_metrics[
                        "balanced_accuracy"
                    ],
                    "Differenza PGM − confronto": (
                        pgm_metrics["balanced_accuracy"]
                        - standard_metrics["balanced_accuracy"]
                    ),
                    "Backend PGM": pgm_record["computational_backend"],
                    "Encoding PGM": pgm_record["encoding"],
                    "Fattore t PGM": pgm_record["rescaling_factor"],
                }
                for seed, pgm_record, pgm_metrics, standard_metrics in zip(
                    pgm_evaluation["seeds"],
                    pgm_evaluation["records"],
                    pgm_metric_records,
                    standard_metric_records,
                )
            ],
        }
    )
    return reference


def seed_outcome_rates(
    seed_summary: dict,
) -> tuple[float, float, float, bool]:
    """Read seed-level percentages with compatibility for 5.2.1 summaries."""

    count = int(seed_summary["count"])
    if count < 1:
        raise ValueError("Il riepilogo non contiene seed validi.")
    pgm_rate = float(
        seed_summary.get("pgm_win_rate", seed_summary["pgm_wins"] / count)
    )
    competitor_rate = float(
        seed_summary.get(
            "competitor_win_rate", seed_summary["competitor_wins"] / count
        )
    )
    tie_rate = float(
        seed_summary.get("tie_rate", seed_summary["ties"] / count)
    )
    exact_tie = bool(
        seed_summary.get("exact_tie", int(seed_summary["ties"]) == count)
    )
    return pgm_rate, competitor_rate, tie_rate, exact_tie


def pgm_seed_record(payload: dict, seed: int) -> dict:
    """Compact, serializable metrics for one leakage-safe outer split."""

    metrics = pgm_benchmark_metrics(payload)
    return {
        "seed": int(seed),
        "metrics": metrics,
        "encoding": payload["encoding"],
        "rescaling_factor": payload["rescaling_factor"],
        "base_d": int(payload["base_d"]),
        "d": int(payload["d"]),
        "computational_backend": select_payload_backend(payload),
    }


def _evaluate_multiseed_pgm(
    dataset_key: str,
    copies: int,
    test_fraction: float,
    base_seed: int,
    seed_count: int,
    prior_mode: str,
    relative_tolerance: float,
    feature_count: int,
    dataset_override: tuple[pd.DataFrame, pd.Series, str] | None,
    cache_schema: str,
) -> dict:
    """Evaluate the same full/manual feature policy over several outer splits."""

    if cache_schema != EXPERIMENT_CACHE_SCHEMA:
        raise ValueError("Schema della cache multi-seed non riconosciuto.")
    seeds = evaluation_seeds(base_seed, seed_count)
    reference_payload = None
    payloads: list[dict] = []
    records: list[dict] = []
    for seed in seeds:
        payload = run_experiment(
            dataset_key,
            copies=copies,
            test_fraction=test_fraction,
            random_seed=seed,
            prior_mode=prior_mode,
            relative_tolerance=relative_tolerance,
            max_encoded_features=feature_count,
            stereographic_max_encoded_features=feature_count,
            scalable_full_features=True,
            explicit_dimension_limit=SCALABLE_EXPLICIT_DIMENSION_LIMIT,
            dataset_override=dataset_override,
        )
        payload["computational_backend"] = select_payload_backend(payload)
        if reference_payload is None:
            reference_payload = payload
        payloads.append(payload)
        records.append(pgm_seed_record(payload, seed))
    if reference_payload is None:
        raise RuntimeError("La valutazione multi-seed non ha prodotto risultati.")
    summary = summarize_metrics([record["metrics"] for record in records])
    return {
        "reference_payload": reference_payload,
        "payloads": payloads,
        "records": records,
        "summary": summary,
        "seeds": seeds,
        "cache_schema": cache_schema,
    }


@st.cache_data(show_spinner=False, max_entries=12)
def _execute_public_multiseed_pgm(
    dataset_key: str,
    copies: int,
    test_fraction: float,
    base_seed: int,
    seed_count: int,
    prior_mode: str,
    relative_tolerance: float,
    feature_count: int,
    cache_schema: str,
) -> dict:
    return _evaluate_multiseed_pgm(
        dataset_key,
        copies,
        test_fraction,
        base_seed,
        seed_count,
        prior_mode,
        relative_tolerance,
        feature_count,
        None,
        cache_schema,
    )


def execute_multiseed_pgm(
    dataset_key: str,
    copies: int,
    test_fraction: float,
    base_seed: int,
    seed_count: int,
    prior_mode: str,
    relative_tolerance: float,
    feature_count: int,
    dataset_override: tuple[pd.DataFrame, pd.Series, str] | None,
    cache_schema: str,
) -> dict:
    """Use shared caching only for catalog datasets, never for private uploads."""

    arguments = (
        dataset_key,
        copies,
        test_fraction,
        base_seed,
        seed_count,
        prior_mode,
        relative_tolerance,
        feature_count,
    )
    if dataset_override is None:
        return _execute_public_multiseed_pgm(*arguments, cache_schema)
    return _evaluate_multiseed_pgm(
        *arguments,
        dataset_override,
        cache_schema,
    )


def _construct_dilation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    copies: int,
    prior_mode: str,
    relative_tolerance: float,
    exact_qubit_limit: int,
):
    measurement = build_reduced_pgm_measurement(
        X_train,
        y_train,
        copies=copies,
        prior_mode=prior_mode,
        relative_tolerance=relative_tolerance,
    )
    dilation = build_naimark_dilation(
        measurement,
        exact_qubit_limit=exact_qubit_limit,
    )
    return measurement, dilation


@st.cache_resource(show_spinner=False, max_entries=6)
def _construct_public_dilation_cached(
    X_train: np.ndarray,
    y_train: np.ndarray,
    copies: int,
    prior_mode: str,
    relative_tolerance: float,
    exact_qubit_limit: int,
):
    return _construct_dilation(
        X_train,
        y_train,
        copies,
        prior_mode,
        relative_tolerance,
        exact_qubit_limit,
    )


def construct_dilation_cached(
    X_train: np.ndarray,
    y_train: np.ndarray,
    copies: int,
    prior_mode: str,
    relative_tolerance: float,
    exact_qubit_limit: int,
    *,
    use_shared_cache: bool,
):
    """Build private circuits outside Streamlit's process-wide resource cache."""

    runner = (
        _construct_public_dilation_cached
        if use_shared_cache
        else _construct_dilation
    )
    return runner(
        X_train,
        y_train,
        copies,
        prior_mode,
        relative_tolerance,
        exact_qubit_limit,
    )


def transpile_in_isolated_process(
    qpy_payload: bytes | None,
    timeout_seconds: int,
    max_diagram_gates: int,
    *,
    isometry_matrix: np.ndarray | None = None,
    system_qubits: int | None = None,
    outcome_qubits: int | None = None,
    input_state: np.ndarray | None = None,
    circuit_name: str = "PGM_Naimark_isometry",
    reference_qpy_payload: bytes | None = None,
    input_subspace_dimension: int | None = None,
    optimization_level: int = 0,
    seed_transpiler: int = 42,
):
    return isolated_transpile_qpy(
        qpy_payload,
        isometry_matrix=isometry_matrix,
        system_qubits=system_qubits,
        outcome_qubits=outcome_qubits,
        input_state=input_state,
        circuit_name=circuit_name,
        reference_qpy_payload=reference_qpy_payload,
        input_subspace_dimension=input_subspace_dimension,
        timeout_seconds=timeout_seconds,
        max_diagram_gates=max_diagram_gates,
        fold=120,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )


def certified_isometry_candidates(
    *,
    isometry_matrix: np.ndarray,
    system_qubits: int,
    outcome_qubits: int,
    reference_qpy_payload: bytes,
    input_subspace_dimension: int,
    timeout_seconds: int,
    seed_transpiler: int,
    input_state: np.ndarray | None = None,
    circuit_name: str = "PGM_Naimark_isometry",
):
    """Try all automatic levels and return every report plus the best one."""

    reports = [
        transpile_in_isolated_process(
            None,
            timeout_seconds,
            FULL_GATE_DIAGRAM_LIMIT,
            isometry_matrix=isometry_matrix,
            system_qubits=system_qubits,
            outcome_qubits=outcome_qubits,
            input_state=input_state,
            circuit_name=circuit_name,
            reference_qpy_payload=reference_qpy_payload,
            input_subspace_dimension=input_subspace_dimension,
            optimization_level=level,
            seed_transpiler=seed_transpiler,
        )
        for level in AUTOMATIC_OPTIMIZATION_LEVELS
    ]
    return reports, best_certified_transpilation(reports)


def optimization_attempts_frame(reports) -> pd.DataFrame:
    """Compact, user-facing comparison of automatic synthesis attempts."""

    return pd.DataFrame(
        [
            {
                "Livello": report.optimization_level,
                "Stato": report.status,
                "Certificato": (
                    "Sì"
                    if report.equivalence_certified is True
                    else (
                        "No"
                        if report.equivalence_certified is False
                        else "Non disponibile"
                    )
                ),
                "Gate entangling": (
                    entangling_gate_count(report)
                    if report.size is not None
                    else None
                ),
                "Profondità": report.depth,
                "Porte totali": report.size,
                "Errore": report.equivalence_error,
            }
            for report in reports
        ]
    )


def _reduction_percent(original: int, optimized: int) -> float:
    if original <= 0:
        return 0.0
    return 100.0 * (original - optimized) / original


def complexity_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metodo": row.method,
                "Dimensione di lavoro": f"{row.working_dimension:,}",
                "Tempo training (Big-O)": row.training_time_big_o,
                "Proxy training": scientific_integer(row.training_time_proxy),
                "Memoria training (Big-O)": row.training_memory_big_o,
                "Stima memoria paper": human_bytes(8 * row.training_memory_elements),
                "Tempo predizione/campione": row.prediction_time_big_o,
                "Proxy predizione": scientific_integer(row.prediction_time_proxy),
                "Memoria predizione": row.prediction_memory_big_o,
                "Stima memoria predizione": human_bytes(
                    8 * row.prediction_memory_elements
                ),
            }
            for row in rows
        ]
    )


def results_frame(results: dict[str, MethodResult], y_test: np.ndarray) -> pd.DataFrame:
    reference = results["c-PGM"].predictions
    records = []
    for name in ("c-PGM", "k-PGM", "r-PGM"):
        result = results[name]
        records.append(
            {
                "Metodo": name,
                "Accuratezza": accuracy_score(y_test, result.predictions),
                "Accordo con c-PGM": float(np.mean(result.predictions == reference)),
                "Rank numerico": result.rank,
                "Dimensione": result.representation_dimension,
                "Calcolo": result.execution_mode,
                "Training (s)": result.train_seconds,
                "Predizione (s)": result.predict_seconds,
                "Stato modello": (
                    human_bytes(result.model_state_bytes)
                    if "non materializzata" not in result.execution_mode
                    else "Non materializzato"
                ),
            }
        )
    return pd.DataFrame(records)


def prediction_frame(results: dict[str, MethodResult], y_test: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "classe_reale": y_test,
            "predizione_c_PGM": results["c-PGM"].predictions,
            "predizione_k_PGM": results["k-PGM"].predictions,
            "predizione_r_PGM": results["r-PGM"].predictions,
        }
    )


def modules_available(*module_names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in module_names)


def safe_error_message(error: Exception, secret: str | None = None) -> str:
    message = str(error)
    if secret:
        message = message.replace(secret, "••••••")
    return message


def configured_secret(*names: str) -> str:
    """Return the first configured value from environment or Streamlit Secrets."""

    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        for name in names:
            value = str(st.secrets.get(name, "")).strip()
            if value:
                return value
    except Exception:
        pass
    return ""


def device_display_label(device) -> str:
    parts = [device.name, device.provider, device.status]
    parts.append(f"{device.qubits} qubit" if device.qubits is not None else "qubit n/d")
    if device.pending_jobs is not None:
        parts.append(f"{device.pending_jobs} job in coda")
    return " · ".join(parts)


def render_execution_result(
    counts,
    *,
    classes,
    theoretical_outcomes,
    outcome_qubits: int,
    actual_class,
    prediction_title: str,
) -> None:
    predicted, distance, table = execution_summary(
        counts,
        classes,
        theoretical_outcomes,
        outcome_qubits=outcome_qubits,
    )
    metrics = st.columns(4)
    metrics[0].metric(prediction_title, str(predicted))
    metrics[1].metric("Classe reale", str(actual_class))
    metrics[2].metric("Esito", "Corretta" if predicted == actual_class else "Errata")
    metrics[3].metric("Distanza dalla teoria", f"{distance:.4f}")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Frequenza": st.column_config.NumberColumn(format="%.4f")
        },
    )
    st.caption(
        "La distanza è la total variation distance fra frequenze osservate e "
        "probabilità teoriche: valori più vicini a zero indicano maggiore accordo."
    )


def render_classifier_comparison(
    comparison: dict,
    *,
    dataset_display_name: str,
) -> None:
    dataset_display_name = translate_text(
        dataset_display_name, current_language()
    )
    competitor_name = translate_text(
        comparison["classifier_name"], current_language()
    )
    seed_summary = comparison.get("paired_seed_summary")
    bootstrap = comparison["bootstrap"]
    decision = seed_summary if seed_summary is not None else bootstrap
    if decision["winner"] == "pgm":
        st.success(
            f"Vincitore: PGM su {dataset_display_name}. L'intervallo appaiato "
            "multi-seed della differenza di balanced accuracy è interamente positivo."
        )
    elif decision["winner"] == "competitor":
        st.success(
            f"Vincitore: {competitor_name} su {dataset_display_name}. L'intervallo "
            "appaiato multi-seed della differenza di balanced accuracy è interamente "
            "negativo."
        )
    else:
        st.info(
            f"Confronto non conclusivo su {dataset_display_name}: l'intervallo "
            "appaiato della differenza include zero, quindi non viene dichiarato "
            "un vincitore."
        )

    summary_columns = st.columns(4)
    if seed_summary is not None:
        (
            pgm_seed_win_rate,
            competitor_seed_win_rate,
            exact_seed_tie_rate,
            _,
        ) = seed_outcome_rates(seed_summary)
        summary_columns[0].metric(
            "Balanced accuracy PGM",
            f"{100.0 * seed_summary['pgm_mean']:.2f}% ± "
            f"{100.0 * seed_summary['pgm_std']:.2f}",
        )
        summary_columns[1].metric(
            f"Balanced accuracy {competitor_name}",
            f"{100.0 * seed_summary['competitor_mean']:.2f}% ± "
            f"{100.0 * seed_summary['competitor_std']:.2f}",
        )
        summary_columns[2].metric(
            "Differenza media PGM − confronto",
            f"{100.0 * seed_summary['difference_mean']:+.2f} ± "
            f"{100.0 * seed_summary['difference_std']:.2f} punti %",
        )
        summary_columns[3].metric(
            "Intervallo appaiato 95%",
            f"[{100.0 * seed_summary['confidence_lower']:+.2f}, "
            f"{100.0 * seed_summary['confidence_upper']:+.2f}] punti %",
        )
        st.caption(
            f"Media ± deviazione standard su {comparison['seed_count']} split "
            "stratificati appaiati. Ogni modello è ottimizzato esclusivamente sul "
            "training del relativo seed; nessun risultato viene scartato."
        )
        seed_outcome_rows = [
            {
                "Esito": translate_text("PGM superiore", current_language()),
                "Percentuale": 100.0 * pgm_seed_win_rate,
                "Gruppo": "seed",
            },
            {
                "Esito": translate_text(
                    f"{competitor_name} superiore", current_language()
                ),
                "Percentuale": 100.0 * competitor_seed_win_rate,
                "Gruppo": "seed",
            },
            {
                "Esito": translate_text(
                    "Esattamente uguali", current_language()
                ),
                "Percentuale": 100.0 * exact_seed_tie_rate,
                "Gruppo": "seed",
            },
        ]
        st.vega_lite_chart(
            pd.DataFrame(seed_outcome_rows),
            {
                "mark": {"type": "bar", "cornerRadius": 4},
                "encoding": {
                    "x": {
                        "field": "Percentuale",
                        "type": "quantitative",
                        "stack": "zero",
                        "title": translate_text(
                            "Percentuale dei seed", current_language()
                        ),
                        "scale": {"domain": [0, 100]},
                    },
                    "y": {"field": "Gruppo", "axis": None},
                    "color": {
                        "field": "Esito",
                        "type": "nominal",
                        "scale": {
                            "domain": [row["Esito"] for row in seed_outcome_rows],
                            "range": ["#16805B", "#C44949", "#7D8792"],
                        },
                        "title": translate_text("Esito", current_language()),
                    },
                    "tooltip": [
                        {"field": "Esito", "type": "nominal"},
                        {
                            "field": "Percentuale",
                            "type": "quantitative",
                            "format": ".1f",
                            "title": translate_text(
                                "Percentuale dei seed", current_language()
                            ),
                        },
                    ],
                },
                "height": 48,
            },
            width="stretch",
        )
        if current_language() == "en":
            st.caption(
                f"Split-by-split comparison: PGM has higher balanced accuracy on "
                f"{100.0 * pgm_seed_win_rate:.1f}% of seeds, "
                f"{competitor_name} on "
                f"{100.0 * competitor_seed_win_rate:.1f}%, and the "
                f"values are exactly equal on "
                f"{100.0 * exact_seed_tie_rate:.1f}%. An overall "
                "inconclusive outcome is not a tie: it means the confidence "
                "interval for the difference includes zero."
            )
        else:
            st.caption(
                f"Confronto split per split: PGM ottiene una balanced accuracy più "
                f"alta nel {100.0 * pgm_seed_win_rate:.1f}% dei seed, "
                f"{competitor_name} nel "
                f"{100.0 * competitor_seed_win_rate:.1f}% e i valori "
                f"sono esattamente uguali nel "
                f"{100.0 * exact_seed_tie_rate:.1f}%. Un esito complessivo "
                "non conclusivo non è un pareggio: indica che l'intervallo di "
                "confidenza della differenza include zero."
            )
    else:
        summary_columns[0].metric(
            "Balanced accuracy PGM",
            f"{100.0 * bootstrap['pgm_balanced_accuracy']:.2f}%",
        )
        summary_columns[1].metric(
            f"Balanced accuracy {competitor_name}",
            f"{100.0 * bootstrap['competitor_balanced_accuracy']:.2f}%",
        )
        summary_columns[2].metric(
            "Differenza PGM − confronto",
            f"{100.0 * bootstrap['difference']:+.2f} punti %",
        )
        summary_columns[3].metric(
            "Intervallo bootstrap 95%",
            f"[{100.0 * bootstrap['confidence_lower']:+.2f}, "
            f"{100.0 * bootstrap['confidence_upper']:+.2f}] punti %",
        )

    metric_rows: list[dict] = []
    chart_rows: list[dict] = []
    for metric_key, metric_label in METRIC_LABELS.items():
        if seed_summary is not None:
            pgm_stats = comparison["pgm_metric_summary"][metric_key]
            standard_stats = comparison["standard_metric_summary"][metric_key]
            pgm_value = float(pgm_stats["mean"])
            standard_value = float(standard_stats["mean"])
            pgm_display = (
                f"{pgm_value:.4f} ± {float(pgm_stats['std']):.4f}"
                if np.isfinite(pgm_value)
                else "—"
            )
            standard_display = (
                f"{standard_value:.4f} ± {float(standard_stats['std']):.4f}"
                if np.isfinite(standard_value)
                else "—"
            )
        else:
            pgm_value = comparison["pgm_metrics"].get(metric_key, float("nan"))
            standard_value = comparison["standard_metrics"].get(
                metric_key, float("nan")
            )
            pgm_display = f"{pgm_value:.4f}" if np.isfinite(pgm_value) else "—"
            standard_display = (
                f"{standard_value:.4f}"
                if np.isfinite(standard_value)
                else "—"
            )
        difference = pgm_value - standard_value
        metric_rows.append(
            {
                "Metrica": metric_label,
                "PGM (media ± dev. std.)": pgm_display,
                f"{competitor_name} (media ± dev. std.)": standard_display,
                "Differenza PGM − confronto": (
                    f"{difference:+.4f}"
                    if np.isfinite(difference)
                    else "—"
                ),
            }
        )
        for classifier_name, value in (
            ("PGM", pgm_value),
            (competitor_name, standard_value),
        ):
            if np.isfinite(value):
                chart_rows.append(
                    {
                        "metric": translate_text(
                            metric_label, current_language()
                        ),
                        "classifier": translate_text(
                            classifier_name, current_language()
                        ),
                        "value": float(value),
                    }
                )

    st.dataframe(
        pd.DataFrame(metric_rows),
        hide_index=True,
        width="stretch",
    )

    if seed_summary is not None:
        with st.expander("Risultati per ciascun seed", expanded=False):
            seed_frame = pd.DataFrame(comparison["seed_rows"])
            st.dataframe(
                seed_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "Balanced accuracy PGM": st.column_config.NumberColumn(
                        format="%.4f"
                    ),
                    "Balanced accuracy confronto": st.column_config.NumberColumn(
                        format="%.4f"
                    ),
                    "Differenza PGM − confronto": st.column_config.NumberColumn(
                        format="%+.4f"
                    ),
                },
            )
    st.vega_lite_chart(
        pd.DataFrame(chart_rows),
        {
            "mark": {"type": "bar", "cornerRadiusEnd": 4},
            "encoding": {
                "x": {
                    "field": "metric",
                    "type": "nominal",
                    "title": None,
                    "axis": {"labelAngle": -30},
                },
                "xOffset": {"field": "classifier"},
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "title": translate_text("Valore", current_language()),
                },
                "color": {
                    "field": "classifier",
                    "type": "nominal",
                    "title": translate_text(
                        "Classificatore", current_language()
                    ),
                    "scale": {
                        "range": ["#6F42C1", "#00A6A6"],
                    },
                },
                "tooltip": [
                    {
                        "field": "metric",
                        "type": "nominal",
                        "title": translate_text("Metrica", current_language()),
                    },
                    {
                        "field": "classifier",
                        "type": "nominal",
                        "title": translate_text(
                            "Classificatore", current_language()
                        ),
                    },
                    {
                        "field": "value",
                        "type": "quantitative",
                        "format": ".4f",
                        "title": translate_text("Valore", current_language()),
                    },
                ],
            },
            "height": 330,
        },
        width="stretch",
    )

    classes = np.unique(comparison["y_train"])
    _, pgm_percentages = confusion_frames(
        comparison["y_test"],
        comparison["pgm_predictions"],
        classes,
    )
    _, standard_percentages = confusion_frames(
        comparison["y_test"],
        comparison["standard_predictions"],
        classes,
    )
    with st.expander(
        "Matrici di confusione e tuning del classificatore sul seed di riferimento",
        expanded=False,
    ):
        confusion_columns = st.columns(2)
        with confusion_columns[0]:
            st.markdown("**PGM — % per classe reale**")
            st.dataframe(
                pgm_percentages.map(lambda value: f"{value:.1f}%"),
                width="stretch",
            )
        with confusion_columns[1]:
            st.markdown(f"**{competitor_name} — % per classe reale**")
            st.dataframe(
                standard_percentages.map(lambda value: f"{value:.1f}%"),
                width="stretch",
            )
        st.write(
            {
                "Fold di tuning": comparison["tuning_folds"],
                "Balanced accuracy di validazione": (
                    comparison["validation_balanced_accuracy"]
                    if comparison["validation_balanced_accuracy"] is not None
                    else "Non disponibile"
                ),
                "Parametri selezionati": (
                    comparison["best_parameters"]
                    if comparison["best_parameters"]
                    else "Configurazione predefinita"
                ),
                "Tempo fit e tuning (s)": comparison["fit_seconds"],
                "Tempo predizione (s)": comparison["predict_seconds"],
                "Encoding PGM (seed di riferimento)": (
                    "Stereografico + encoding in ampiezza"
                    if comparison["encoding"] == "stereographic"
                    else "Encoding in ampiezza normalizzato"
                ),
                "Fattore t PGM (seed di riferimento)": (
                    comparison["rescaling_factor"]
                    if comparison["rescaling_factor"] is not None
                    else "Non applicabile"
                ),
            }
        )
        if comparison["tuning_message"]:
            st.warning(comparison["tuning_message"])


def render_full_binary_comparison(
    rows: list[dict],
    *,
    competitor_name: str,
) -> None:
    language = current_language()
    competitor_name = translate_text(competitor_name, language)
    pgm_win_label = translate_text("Vittoria PGM confermata", language)
    competitor_win_label = translate_text(
        f"Vittoria {competitor_name} confermata", language
    )
    inconclusive_label = translate_text("Non conclusivo", language)
    exact_tie_label = translate_text("Pareggio esatto", language)
    successful = [row for row in rows if not row.get("error")]
    failed = [row for row in rows if row.get("error")]
    if not successful:
        st.error(
            "Il full comparison non ha prodotto risultati utilizzabili. "
            "Consulta i dettagli degli errori."
        )
        if failed:
            st.dataframe(pd.DataFrame(failed), hide_index=True, width="stretch")
        return

    win_count = sum(row["winner"] == "pgm" for row in successful)
    loss_count = sum(row["winner"] == "competitor" for row in successful)
    inconclusive_count = sum(row["winner"] == "tie" for row in successful)
    exact_tie_count = sum(
        row["winner"] == "tie" and bool(row.get("exact_tie", False))
        for row in successful
    )

    def count_and_percentage(count: int) -> str:
        return f"{count} ({100.0 * count / len(successful):.1f}%)"

    aggregate_columns = st.columns(4)
    aggregate_columns[0].metric("Dataset completati", len(successful))
    aggregate_columns[1].metric(
        "Vittorie PGM", count_and_percentage(win_count)
    )
    aggregate_columns[2].metric(
        f"Vittorie {competitor_name}", count_and_percentage(loss_count)
    )
    aggregate_columns[3].metric(
        "Risultati non conclusivi",
        count_and_percentage(inconclusive_count),
    )
    if exact_tie_count:
        if language == "en":
            st.caption(
                f"Among the {inconclusive_count} inconclusive results, "
                f"{exact_tie_count} are exact ties on every seed."
            )
        else:
            st.caption(
                f"Tra i {inconclusive_count} risultati non conclusivi, "
                f"{exact_tie_count} sono pareggi esatti su ogni seed."
            )

    matrix_records = []
    detail_records = []
    chart_records = []
    outcome_state_by_dataset: dict[str, str] = {}
    for row in successful:
        dataset_name = translate_text(row["dataset"], language)
        pgm_seed_rate = float(row.get("pgm_win_rate", 0.0))
        competitor_seed_rate = float(row.get("competitor_win_rate", 0.0))
        exact_seed_tie_rate = float(row.get("tie_rate", 0.0))
        if row["winner"] == "pgm":
            outcome = pgm_win_label
            outcome_state = "pgm"
        elif row["winner"] == "competitor":
            outcome = competitor_win_label
            outcome_state = "competitor"
        elif bool(row.get("exact_tie", False)):
            outcome = exact_tie_label
            outcome_state = "exact_tie"
        else:
            outcome = inconclusive_label
            outcome_state = "inconclusive"
        outcome_state_by_dataset[dataset_name] = outcome_state

        pgm_display = (
            f"{100.0 * row['pgm_balanced_accuracy']:.2f}% ± "
            f"{100.0 * row['pgm_balanced_accuracy_std']:.2f}%"
        )
        competitor_display = (
            f"{100.0 * row['competitor_balanced_accuracy']:.2f}% ± "
            f"{100.0 * row['competitor_balanced_accuracy_std']:.2f}%"
        )
        difference_display = (
            f"{100.0 * row['difference']:+.2f} pp · "
            f"IC [{100.0 * row['confidence_lower']:+.2f}, "
            f"{100.0 * row['confidence_upper']:+.2f}]"
        )
        seed_display = (
            f"PGM {100.0 * pgm_seed_rate:.1f}% · "
            f"{competitor_name} {100.0 * competitor_seed_rate:.1f}% · "
            f"= {100.0 * exact_seed_tie_rate:.1f}%"
        )
        matrix_records.append(
            {
                "Dataset": dataset_name,
                "PGM — balanced accuracy": pgm_display,
                f"{competitor_name} — balanced accuracy": competitor_display,
                "Differenza PGM − confronto (IC 95%)": difference_display,
                "Seed favorevoli PGM / confronto": seed_display,
                "Esito statistico": outcome,
            }
        )
        detail_records.append(
            {
                "Dataset": dataset_name,
                "Balanced accuracy PGM": pgm_display,
                f"Balanced accuracy {competitor_name}": competitor_display,
                "Differenza PGM − confronto": (
                    f"{100.0 * row['difference']:+.2f} ± "
                    f"{100.0 * row['difference_std']:.2f} punti %"
                ),
                "Intervallo appaiato 95%": (
                    f"[{100.0 * row['confidence_lower']:+.2f}, "
                    f"{100.0 * row['confidence_upper']:+.2f}] punti %"
                ),
                "Numero di seed": row["seed_count"],
                "Seed PGM superiore": f"{100.0 * pgm_seed_rate:.1f}%",
                f"Seed {competitor_name} superiore": (
                    f"{100.0 * competitor_seed_rate:.1f}%"
                ),
                "Seed esattamente uguali": (
                    f"{100.0 * exact_seed_tie_rate:.1f}%"
                ),
                "Esito statistico": outcome,
                "Encoding PGM": (
                    "Stereografico + encoding in ampiezza"
                    if row["encoding"] == "stereographic"
                    else "Encoding in ampiezza normalizzato"
                ),
                "Fattore t PGM": (
                    "—"
                    if row["rescaling_factor"] is None
                    else f"{row['rescaling_factor']:g}"
                ),
            }
        )
        chart_records.append(
            {
                "dataset": dataset_name,
                "delta": 100.0 * row["difference"],
                "ci_lower": 100.0 * row["confidence_lower"],
                "ci_upper": 100.0 * row["confidence_upper"],
                "pgm_seed_rate": 100.0 * pgm_seed_rate,
                "competitor_seed_rate": 100.0 * competitor_seed_rate,
                "outcome": outcome,
            }
        )

    st.markdown("#### Confronto quantitativo WIN / LOSS")
    matrix = localize_dataframe(
        pd.DataFrame(matrix_records).set_index("Dataset"), language
    )
    matrix.index.name = translate_text("Dataset", language)

    green_style = (
        "background-color: #d9f2e6; color: #155d3a; font-weight: 700"
    )
    red_style = (
        "background-color: #fde2e2; color: #8a1c1c; font-weight: 700"
    )
    amber_style = (
        "background-color: #fff1cc; color: #795400; font-weight: 700"
    )
    gray_style = (
        "background-color: #edf1f7; color: #42526b; font-weight: 700"
    )

    def matrix_row_colors(row: pd.Series) -> list[str]:
        state = outcome_state_by_dataset.get(str(row.name), "inconclusive")
        styles = [""] * len(row)
        if state == "pgm":
            styles[0], styles[1] = green_style, red_style
            styles[2], styles[4] = green_style, green_style
        elif state == "competitor":
            styles[0], styles[1] = red_style, green_style
            styles[2], styles[4] = red_style, red_style
        elif state == "exact_tie":
            styles = [gray_style] * len(row)
        else:
            styles[2], styles[4] = amber_style, amber_style
        return styles

    st.dataframe(
        matrix.style.apply(matrix_row_colors, axis=1),
        width="stretch",
    )
    st.caption(
        "Una vittoria è confermata soltanto quando l'intervallo Student-t al 95% "
        "delle differenze appaiate tra seed esclude zero. 'Non conclusivo' non "
        "significa che i risultati sono uguali: indica evidenza insufficiente per "
        "dichiarare un vincitore. 'Pareggio esatto' è riservato ai casi in cui le "
        "balanced accuracy coincidono su ogni seed entro la tolleranza numerica. "
        "Le percentuali mostrano in quanti seed ciascun metodo è risultato superiore."
    )

    outcome_domain = [
        pgm_win_label,
        competitor_win_label,
        inconclusive_label,
        exact_tie_label,
    ]
    st.vega_lite_chart(
        pd.DataFrame(chart_records),
        {
            "encoding": {
                "y": {
                    "field": "dataset",
                    "type": "nominal",
                    "sort": "-x",
                    "title": None,
                },
            },
            "layer": [
                {
                    "mark": {"type": "bar", "cornerRadiusEnd": 4},
                    "encoding": {
                        "x": {
                            "field": "delta",
                            "type": "quantitative",
                            "title": translate_text(
                                "Differenza balanced accuracy PGM − confronto (punti %)",
                                language,
                            ),
                        },
                        "color": {
                            "field": "outcome",
                            "type": "nominal",
                            "title": translate_text("Esito statistico", language),
                            "scale": {
                                "domain": outcome_domain,
                                "range": [
                                    "#16805B",
                                    "#C44949",
                                    "#D89C29",
                                    "#7D8792",
                                ],
                            },
                        },
                        "tooltip": [
                            {
                                "field": "dataset",
                                "type": "nominal",
                                "title": translate_text("Dataset", language),
                            },
                            {
                                "field": "delta",
                                "type": "quantitative",
                                "format": "+.2f",
                                "title": translate_text(
                                    "Differenza PGM − confronto", language
                                ),
                            },
                            {
                                "field": "ci_lower",
                                "type": "quantitative",
                                "format": "+.2f",
                                "title": translate_text(
                                    "Limite inferiore IC 95%", language
                                ),
                            },
                            {
                                "field": "ci_upper",
                                "type": "quantitative",
                                "format": "+.2f",
                                "title": translate_text(
                                    "Limite superiore IC 95%", language
                                ),
                            },
                            {
                                "field": "pgm_seed_rate",
                                "type": "quantitative",
                                "format": ".1f",
                                "title": translate_text(
                                    "Seed PGM superiore (%)", language
                                ),
                            },
                            {
                                "field": "competitor_seed_rate",
                                "type": "quantitative",
                                "format": ".1f",
                                "title": translate_text(
                                    "Seed confronto superiore (%)", language
                                ),
                            },
                            {
                                "field": "outcome",
                                "type": "nominal",
                                "title": translate_text(
                                    "Esito statistico", language
                                ),
                            },
                        ],
                    },
                },
                {
                    "mark": {
                        "type": "rule",
                        "color": "#263247",
                        "strokeWidth": 2,
                    },
                    "encoding": {
                        "x": {
                            "field": "ci_lower",
                            "type": "quantitative",
                        },
                        "x2": {"field": "ci_upper"},
                    },
                },
                {
                    "mark": {
                        "type": "point",
                        "filled": True,
                        "size": 55,
                        "color": "#263247",
                    },
                    "encoding": {
                        "x": {
                            "field": "delta",
                            "type": "quantitative",
                        }
                    },
                },
            ],
            "resolve": {"scale": {"color": "shared"}},
            "height": max(320, 30 * len(successful)),
        },
        width="stretch",
    )
    st.caption(
        "Le barre mostrano la differenza media in punti percentuali; il segmento "
        "nero è l'intervallo di confidenza appaiato al 95%. Il giallo indica un "
        "esito statisticamente non conclusivo; direzione e ampiezza della "
        "differenza restano visibili e non vengono chiamate pareggio."
    )
    with st.expander("Risultati completi per dataset", expanded=False):
        st.dataframe(
            pd.DataFrame(detail_records),
            hide_index=True,
            width="stretch",
        )
    if failed:
        with st.expander(
            f"Dataset non completati ({len(failed)})", expanded=False
        ):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Dataset": row["dataset"],
                            "Errore": row["error"],
                        }
                        for row in failed
                    ]
                ),
                hide_index=True,
                width="stretch",
            )


def render_language_selector() -> None:
    """Render the two explicit language flags without affecting scientific state."""

    active_language = current_language()
    with _st.container(
        key="language_switcher",
        horizontal=True,
        wrap=False,
        horizontal_alignment="right",
        gap="small",
    ):
        if BRAND_BADGE_PATH.is_file():
            _st.image(
                str(BRAND_BADGE_PATH),
                width=66,
            )
        italian_clicked = _st.button(
            "🇮🇹",
            key="language_it",
            help="Italiano",
            type="primary" if active_language == "it" else "secondary",
        )
        english_clicked = _st.button(
            "🇬🇧",
            key="language_en",
            help="English",
            type="primary" if active_language == "en" else "secondary",
        )
    requested_language = (
        "it" if italian_clicked else "en" if english_clicked else active_language
    )
    if requested_language != active_language:
        _st.session_state[LANGUAGE_SESSION_KEY] = requested_language
        _st.rerun()


render_language_selector()
st.title("PGM Lab")
st.caption(
    f"Versione {APP_VERSION} · PGM equivalenti, benchmark statistici e "
    "integrazione quantistica protetta"
)
st.write(
    "Confronto riproducibile tra **c-PGM**, **k-PGM** e **r-PGM (Rc-PGM)**. "
    "La classificazione conserva tutte le feature salvo riduzione PCA richiesta "
    "esplicitamente dall'utente e usa il backend esatto meno oneroso tra k-PGM e "
    "r-PGM. Le prestazioni sono aggregate su più split stratificati mediante media "
    "e deviazione standard. Sul solo training set, l'app sceglie l'encoding e il "
    "fattore di rescaling; quando le dimensioni lo consentono costruisce anche il "
    "circuito quantistico della PGM mediante una dilatazione di Naimark."
)

with st.expander("Che cosa significa 'equivalenti'?", expanded=False):
    st.markdown(
        r"""
        - **c-PGM** costruisce esplicitamente il tensore di dimensione $d^c$.
        - **k-PGM** usa il kernel omogeneo $\langle x,z\rangle^c$ e una matrice $N\times N$.
        - **r-PGM** usa la base simmetrica minima di dimensione
          $d_{sym}=\binom{d+c-1}{c}$.

        L'equivalenza teorica riguarda gli score di classe. Durante l'uso l'app sceglie
        la diagonalizzazione meno onerosa tra k-PGM e r-PGM e ricostruisce gli score
        delle altre formulazioni tramite la medesima Gram matrix, senza duplicare il
        costo. La suite numerica verifica separatamente l'equivalenza delle tre mappe.
        Il termine di completamento $P_{ker(\sigma)}/l$ viene omesso dagli score perché
        è uguale per ogni classe e non modifica l'argmax.
        """
    )

render_section_banner(
    "Sezione 1",
    "1. Scegli il dataset",
    "Seleziona dati e copie oppure carica un file personale; ogni scelta resta modificabile.",
)
private_dataset = _st.session_state.get(PRIVATE_DATASET_SESSION_KEY)
with st.expander(
    "Carica un dataset personale",
    expanded=private_dataset is None,
):
    st.write(
        "Trascina qui un file tabellare. L'app riconosce CSV, TSV/TXT ed Excel "
        "XLSX, propone automaticamente il target e usa soltanto le feature "
        "numeriche selezionate. Il file resta nella memoria della tua sessione: "
        "non viene salvato nel repository né condiviso con altri utenti."
    )
    uploaded_file = st.file_uploader(
        "Trascina il file oppure clicca per sceglierlo",
        type=["csv", "tsv", "txt", "xlsx"],
        accept_multiple_files=False,
        key=(
            "private_dataset_uploader_"
            f"{int(_st.session_state.get(PRIVATE_UPLOAD_WIDGET_NONCE_KEY, 0))}"
        ),
        help=(
            "Massimo 25 MB, 50.000 righe e 500 colonne. La prima riga deve "
            "contenere i nomi delle colonne; per Excel viene letta la prima scheda."
        ),
    )
    if uploaded_file is not None:
        try:
            uploaded_content = uploaded_file.getvalue()
            uploaded_frame, upload_parsing = parse_uploaded_table_for_session(
                uploaded_content,
                uploaded_file.name,
            )
            upload_digest = str(upload_parsing["digest"])
            upload_columns = [str(column) for column in uploaded_frame.columns]
            suggested_target = suggest_target_column(uploaded_frame)
            uploaded_target = st.selectbox(
                "Colonna target da predire",
                options=upload_columns,
                index=upload_columns.index(suggested_target),
                key=f"uploaded_target_{upload_digest[:12]}",
                help=(
                    "Il target contiene le etichette delle classi. Puoi correggere "
                    "la scelta proposta prima di aggiungere il dataset."
                ),
            )
            feature_report = inspect_feature_columns(
                uploaded_frame,
                uploaded_target,
            )
            numeric_features = list(feature_report["numeric"])
            recommended_features = list(feature_report["recommended"])
            if not recommended_features:
                recommended_features = numeric_features
            selected_uploaded_features = st.multiselect(
                "Feature numeriche da utilizzare",
                options=numeric_features,
                default=recommended_features,
                key=(
                    f"uploaded_features_{upload_digest[:12]}_"
                    f"{uploaded_target}"
                ),
                help=(
                    "Le colonne testuali, vuote o costanti non sono utilizzabili. "
                    "Gli identificativi univoci sono disponibili ma deselezionati "
                    "per impostazione predefinita."
                ),
            )
            uploaded_display_name = st.text_input(
                "Nome da mostrare nell'app",
                value=display_name_from_file(uploaded_file.name),
                max_chars=80,
                key=f"uploaded_name_{upload_digest[:12]}",
            )

            if feature_report["non_numeric"]:
                st.info(
                    "Colonne non numeriche escluse: "
                    + ", ".join(feature_report["non_numeric"][:12])
                    + ("…" if len(feature_report["non_numeric"]) > 12 else "")
                    + "."
                )
            if feature_report["probable_identifiers"]:
                st.warning(
                    "Possibili identificativi deselezionati automaticamente: "
                    + ", ".join(feature_report["probable_identifiers"])
                    + ". In genere non contengono informazione generalizzabile."
                )
            if feature_report["empty_or_constant"]:
                st.info(
                    "Colonne vuote o costanti ignorate: "
                    + ", ".join(feature_report["empty_or_constant"][:12])
                    + "."
                )

            uploaded_X, uploaded_y, upload_validation = prepare_uploaded_dataset(
                uploaded_frame,
                target_column=uploaded_target,
                feature_columns=selected_uploaded_features,
            )
            if upload_validation["dropped_constant_features"]:
                st.warning(
                    "Feature diventate vuote o costanti dopo la rimozione delle "
                    "righe senza target: "
                    + ", ".join(upload_validation["dropped_constant_features"])
                    + "."
                )
            upload_metrics = st.columns(4)
            upload_metrics[0].metric(
                "Campioni validi", f"{upload_validation['samples']:,}"
            )
            upload_metrics[1].metric(
                "Feature utilizzabili", f"{upload_validation['features']:,}"
            )
            upload_metrics[2].metric(
                "Classi", f"{upload_validation['classes']:,}"
            )
            upload_metrics[3].metric(
                "Valori feature mancanti",
                f"{upload_validation['missing_feature_values']:,}",
            )
            st.caption(
                "I valori mancanti delle feature saranno imputati usando soltanto "
                "il training set di ciascun seed. Le righe senza target vengono "
                "rimosse prima dello split."
            )
            with st.expander("Anteprima e distribuzione delle classi", expanded=False):
                st.dataframe(uploaded_frame.head(8), hide_index=True, width="stretch")
                st.write({"Campioni per classe": upload_validation["class_counts"]})

            if st.button(
                "Aggiungi al catalogo e seleziona",
                key=f"confirm_upload_{upload_digest[:12]}",
                type="primary",
            ):
                clean_display_name = sanitize_display_name(uploaded_display_name)
                custom_key = uploaded_dataset_key(
                    upload_digest,
                    uploaded_target,
                    selected_uploaded_features,
                )
                custom_source = (
                    f"Upload privato in memoria: {upload_parsing['file_name']} · "
                    f"SHA-256 {upload_digest[:12]} · target {uploaded_target}"
                )
                custom_spec = DatasetSpec(
                    key=custom_key,
                    display_name=clean_display_name,
                    openml_id=None,
                    samples=int(upload_validation["samples"]),
                    features=int(upload_validation["features"]),
                    classes=int(upload_validation["classes"]),
                    domain="Upload privato",
                    source_url="",
                    local_source=custom_source,
                )
                clear_dataset_dependent_session_results()
                _st.session_state[PRIVATE_DATASET_SESSION_KEY] = {
                    "key": custom_key,
                    "spec": custom_spec,
                    "X": uploaded_X,
                    "y": uploaded_y,
                    "source": custom_source,
                    "file_name": upload_parsing["file_name"],
                    "digest": upload_digest,
                    "target": uploaded_target,
                    "metadata": upload_validation,
                }
                _st.session_state["dataset_choice"] = custom_key
                _st.rerun()
        except UploadedDatasetError as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                "Il file non può essere aggiunto in modo sicuro. Dettaglio: "
                + safe_error_message(error)
            )

    private_dataset = _st.session_state.get(PRIVATE_DATASET_SESSION_KEY)
    if private_dataset is not None:
        private_spec = private_dataset["spec"]
        st.success(
            f"Dataset personale disponibile: {private_spec.display_name} · "
            f"{private_spec.samples:,} campioni · {private_spec.features:,} feature · "
            f"{private_spec.classes:,} classi."
        )
        if st.button(
            "Rimuovi il dataset personale dalla sessione",
            key="remove_private_dataset",
        ):
            removed_key = private_dataset["key"]
            del _st.session_state[PRIVATE_DATASET_SESSION_KEY]
            _st.session_state.pop(PRIVATE_UPLOAD_PARSE_CACHE_KEY, None)
            clear_dataset_dependent_session_results()
            _st.session_state[PRIVATE_UPLOAD_WIDGET_NONCE_KEY] = (
                int(_st.session_state.get(PRIVATE_UPLOAD_WIDGET_NONCE_KEY, 0)) + 1
            )
            if _st.session_state.get("dataset_choice") == removed_key:
                _st.session_state["dataset_choice"] = "iris"
            _st.rerun()

catalog_to_display = catalog_frame()
if private_dataset is not None:
    private_spec = private_dataset["spec"]
    catalog_to_display = pd.concat(
        [
            catalog_to_display,
            pd.DataFrame(
                [
                    {
                        "Dataset": f"📤 {private_spec.display_name}",
                        "Campioni": private_spec.samples,
                        "Feature/campione": private_spec.features,
                        "Classi": private_spec.classes,
                        "Ambito": "Sessione privata",
                        "Repository": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
st.dataframe(
    catalog_to_display,
    hide_index=True,
    width="stretch",
    column_config={
        "Repository": st.column_config.LinkColumn(
            "Repository pubblico", display_text="Apri fonte"
        )
    },
)

available_specs = {dataset.key: dataset for dataset in DATASETS}
if private_dataset is not None:
    available_specs[private_dataset["key"]] = private_dataset["spec"]

left, middle, right = st.columns([1.5, 1, 1])
with left:
    selected_key = st.selectbox(
        "Dataset",
        options=list(available_specs),
        index=2,
        key="dataset_choice",
        format_func=lambda key: (
            ("📤 " if private_dataset is not None and key == private_dataset["key"] else "")
            + f"{available_specs[key].display_name} - "
            f"{available_specs[key].samples} campioni, "
            f"{available_specs[key].features} feature, "
            f"{available_specs[key].classes} classi"
        ),
    )
with middle:
    copies = st.slider(
        "Numero di copie c",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
        key="copies_choice",
    )
with right:
    memory_budget_gib = st.select_slider(
        "Budget RAM per il calcolo",
        options=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        value=8.0,
        key="memory_budget_choice",
        format_func=lambda value: f"{value:g} GiB",
        help=(
            "È un limite di sicurezza dell'app, non aumenta la RAM fisicamente "
            "disponibile sul server."
        ),
    )

spec = available_specs[selected_key]
selected_dataset_override = (
    (
        private_dataset["X"],
        private_dataset["y"],
        private_dataset["source"],
    )
    if private_dataset is not None and selected_key == private_dataset["key"]
    else None
)
manual_feature_reduction = st.toggle(
    "Richiedi manualmente una riduzione PCA",
    value=False,
    key="manual_feature_reduction",
    help=(
        "Disattivata per impostazione predefinita: il PGM usa tutte le feature. "
        "Se la attivi, scegli tu quante componenti mantenere; la PCA viene appresa "
        "esclusivamente sul training set di ciascun seed."
    ),
)
if manual_feature_reduction:
    manual_feature_count = st.slider(
        "Numero di feature dopo la PCA manuale",
        min_value=1,
        max_value=max(1, spec.features - 1),
        value=max(1, min(16, spec.features - 1)),
        step=1,
        key=f"manual_feature_count_{selected_key}",
    )
    feature_mode = "Riduzione manuale PCA"
else:
    manual_feature_count = spec.features
    feature_mode = "Tutte le feature originali"

with st.expander("Impostazioni avanzate", expanded=False):
    test_fraction_options = valid_stratified_test_fractions(
        spec.samples,
        spec.classes,
        (0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    )
    if not test_fraction_options:
        test_fraction_options = (0.40,)
    default_test_fraction = (
        0.20 if 0.20 in test_fraction_options else test_fraction_options[0]
    )
    advanced_1, advanced_2, advanced_3, advanced_4 = st.columns(4)
    with advanced_1:
        test_fraction = st.select_slider(
            "Quota test set",
            options=test_fraction_options,
            value=default_test_fraction,
            key=f"test_fraction_choice_{selected_key}",
            format_func=lambda value: f"{value:.0%}",
            help=(
                "Sono proposte soltanto quote che consentono allo split "
                "stratificato di rappresentare ogni classe sia nel training sia "
                "nel test set."
            ),
        )
    with advanced_2:
        random_seed = st.number_input(
            "Seed iniziale",
            min_value=0,
            max_value=1_000_000,
            value=42,
            step=1,
            key="split_seed_choice",
        )
    with advanced_3:
        prior_label = st.selectbox(
            "Prior di classe",
            list(PRIOR_LABELS),
            key="class_prior_choice",
            help=(
                "I prior uniformi seguono l'Eq. (4). Per classi sbilanciate il k-PGM "
                "usa il Gram pesato, così resta esattamente equivalente ai due metodi "
                "primali."
            ),
        )
    with advanced_4:
        seed_count = st.select_slider(
            "Numero di seed di valutazione",
            options=[3, 5, 10, 15, 20],
            value=10,
            key="evaluation_seed_count",
            help=(
                "Ogni seed genera un nuovo split stratificato. Sono riportate "
                "media e deviazione standard; nessun seed viene scartato."
            ),
        )
    tolerance_exponent = st.select_slider(
        "Soglia spettrale relativa",
        options=[8, 9, 10, 11, 12],
        value=10,
        key="spectral_tolerance_choice",
        format_func=lambda exponent: f"10^-{exponent}",
        help="Gli autovalori <= soglia x lambda_max sono esclusi in tutti e tre i metodi.",
    )

selected_feature_count = requested_feature_count(
    raw_feature_count=spec.features,
    manual_reduction=manual_feature_reduction,
    manual_feature_count=manual_feature_count,
)
tensor_base_feature_count = selected_feature_count
stereographic_base_feature_count = selected_feature_count

# The stereographic map adds one coordinate.  Resource previews use the larger
# candidate dimension, so whichever encoding wins can be executed safely.
stereographic_encoded_dimension = stereographic_base_feature_count + 1
preview_encoded_dimension = max(
    tensor_base_feature_count, stereographic_encoded_dimension
)
n_test_estimate = ceil(spec.samples * test_fraction)
n_train_estimate = spec.samples - n_test_estimate
tensor_dimension, symmetric_dimension = representation_dimensions(
    preview_encoded_dimension, copies
)
adaptive_backend = (
    "r-PGM"
    if symmetric_dimension < n_train_estimate
    and symmetric_dimension <= SCALABLE_EXPLICIT_DIMENSION_LIMIT
    else "k-PGM"
)
adaptive_dimension = (
    symmetric_dimension if adaptive_backend == "r-PGM" else n_train_estimate
)
adaptive_peak_bytes = 8 * (
    6 * adaptive_dimension**2
    + (n_train_estimate + n_test_estimate) * adaptive_dimension
    + n_train_estimate * preview_encoded_dimension
)
adaptive_feasible = (
    adaptive_peak_bytes <= int(memory_budget_gib * 1024**3)
    and adaptive_dimension <= 2_000
)

render_section_banner(
    "Sezione 2",
    "2. Controlla le dimensioni prima del calcolo",
    "Verifica subito fattibilità classica, dimensione del circuito e limiti di sicurezza.",
)
metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("N training stimato", f"{n_train_estimate:,}")
metric_2.metric("Feature originali", f"{spec.features:,}")
metric_3.metric(
    "Dimensioni candidate (ampiezza / stereografico)",
    f"{tensor_base_feature_count:,} / {stereographic_encoded_dimension:,}",
)
metric_4.metric("d^c (c-PGM)", f"{tensor_dimension:,}")
metric_5.metric("d_sym (r-PGM)", f"{symmetric_dimension:,}")

if manual_feature_reduction:
    st.info(
        "Riduzione richiesta dall'utente: la PCA conserva "
        f"{selected_feature_count} delle {spec.features} feature. Viene adattata "
        "separatamente sul solo training set di ciascun seed; il test set non "
        "partecipa mai alla selezione."
    )
else:
    st.success(
        "Nessuna feature selection automatica: il PGM usa tutte le "
        f"{spec.features} feature originali. L'encoding stereografico aggiunge "
        "soltanto la propria coordinata geometrica."
    )

preview_circuit_resources = resources_for_dataset(
    preview_encoded_dimension,
    copies,
    spec.classes,
    exact_qubit_limit=EXACT_CIRCUIT_QUBIT_LIMIT,
)
suggested_circuit_features = largest_manual_feature_count_for_circuit(
    copies=copies,
    class_count=spec.classes,
    qubit_limit=EXACT_CIRCUIT_QUBIT_LIMIT,
    maximum_base_features=spec.features,
)
if preview_circuit_resources.exact_materialization_allowed:
    st.info(
        "Circuito quantistico previsto: "
        f"{preview_circuit_resources.system_qubits} qubit di sistema + "
        f"{preview_circuit_resources.outcome_qubits} qubit di esito = "
        f"{preview_circuit_resources.total_qubits} qubit. La dilatazione unitaria "
        "esatta verrà costruita dopo il training."
    )
    if preview_circuit_resources.total_qubits > ISOLATED_SYNTHESIS_QUBIT_LIMIT:
        st.warning(
            "Il circuito logico esatto, la classificazione test e la validazione "
            "matematica saranno disponibili. La decomposizione completa, "
            "l'ottimizzazione e l'esecuzione gate-by-gate resteranno disabilitate: "
            f"questa configurazione usa {preview_circuit_resources.total_qubits} "
            f"qubit, oltre il limite protetto di {ISOLATED_SYNTHESIS_QUBIT_LIMIT}."
        )
else:
    reduction_hint = (
        " Per rientrare nel limite con questi valori di copie e classi, puoi "
        "richiedere manualmente una PCA con al massimo "
        f"{suggested_circuit_features} componenti."
        if suggested_circuit_features > 0
        else (
            " Con questo numero di copie e classi non basta una PCA a una sola "
            "componente: occorre ridurre le copie oppure scegliere un dataset con "
            "meno classi."
        )
    )
    st.warning(
        "Circuito quantistico previsto: "
        f"{preview_circuit_resources.total_qubits} qubit e matrice densa "
        f"{preview_circuit_resources.unitary_dimension:,} x "
        f"{preview_circuit_resources.unitary_dimension:,} "
        f"({human_bytes(preview_circuit_resources.unitary_bytes)}). La matrice esatta "
        f"supera il limite prudenziale di {EXACT_CIRCUIT_QUBIT_LIMIT} qubit. "
        "La classificazione resta disponibile senza ridurre le feature. Se desideri "
        "anche il circuito esatto, la riduzione resta sempre una scelta esplicita "
        "dell'utente."
        + reduction_hint
    )

with st.container(border=True, key="activity_overview"):
    st.markdown("### Mappa delle attività")
    st.caption(
        "Le funzioni sono indipendenti e partono soltanto dal relativo comando. "
        "Questa mappa anticipa ciò che diventerà disponibile dopo la valutazione."
    )
    activity_columns = st.columns(4)
    with activity_columns[0]:
        st.markdown("**① Valutazione PGM**")
        st.write("Multi-seed, media e deviazione standard.")
        st.caption("Pronta" if adaptive_feasible else "Bloccata dal budget")
    with activity_columns[1]:
        st.markdown("**② Confronto opzionale**")
        st.write("Classificatore scelto o full comparison.")
        st.caption("Avvio indipendente")
    with activity_columns[2]:
        st.markdown("**③ Risultati e test**")
        st.write("Metriche, confusion matrix e singoli campioni.")
        st.caption("Dopo la valutazione PGM")
    with activity_columns[3]:
        st.markdown("**④ Laboratorio quantistico**")
        st.write("Circuito, verifica, optimizer, simulatori/QPU ed export.")
        if not preview_circuit_resources.exact_materialization_allowed:
            st.caption("Schema simbolico; circuito esatto oltre il limite")
        elif preview_circuit_resources.total_qubits > ISOLATED_SYNTHESIS_QUBIT_LIMIT:
            st.caption("Circuito esatto; sintesi ed esecuzione oltre il limite")
        else:
            st.caption("Disponibile dopo la valutazione PGM")
    render_quantum_tool_strip()
    preview_action_columns = st.columns(3)
    preview_action_columns[0].button(
        "Visualizza il circuito logico",
        disabled=True,
        width="stretch",
        key="preview_logical_circuit_disabled",
    )
    preview_action_columns[1].button(
        "Ottimizza e certifica equivalenza",
        disabled=True,
        width="stretch",
        key="preview_optimizer_disabled",
    )
    preview_action_columns[2].button(
        "Configura simulatore o QPU",
        disabled=True,
        width="stretch",
        key="preview_execution_disabled",
    )
    st.caption(
        "Questi comandi si attivano dopo la valutazione PGM quando la dimensione "
        "del circuito rispetta i limiti indicati. Il campionamento PGM ideale può "
        "restare disponibile anche quando la sintesi gate-by-gate è disabilitata."
    )

preview_complexities = paper_complexities(
    n_train=n_train_estimate,
    dimension=preview_encoded_dimension,
    copies=copies,
    class_count=spec.classes,
    gram_rank=None,
)
st.dataframe(
    complexity_frame(preview_complexities),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Le formule e la memoria asintotica seguono le Tabelle 1, 2 e 5 del paper. "
    "I proxy sono conteggi dei termini dominanti, non FLOP misurati. Prima del run, "
    "per k-PGM si usa il limite superiore r_G=N; dopo il run compare il rank effettivo. "
    "Le dimensioni mostrate usano, in modo prudenziale, il candidato di encoding "
    "più grande. Dopo il run saranno ricalcolate sulla configurazione selezionata."
)

scalable_full_features = True
classical_run_allowed = adaptive_feasible

if adaptive_feasible:
    st.success(
        f"Backend esatto previsto: {adaptive_backend}, dimensione numerica "
        f"{adaptive_dimension:,}. Nessuna feature viene eliminata automaticamente. "
        f"Picco prudenziale stimato: {human_bytes(adaptive_peak_bytes)}."
    )
else:
    st.error(
        f"Il backend esatto meno oneroso ({adaptive_backend}, dimensione "
        f"{adaptive_dimension:,}) supera il limite prudenziale selezionato. "
        "Aumenta il budget soltanto se il computer dispone realmente della RAM, "
        "oppure richiedi manualmente una riduzione PCA."
    )

configuration_key = (
    selected_key,
    copies,
    feature_mode,
    tensor_base_feature_count,
    stereographic_base_feature_count,
    float(test_fraction),
    int(random_seed),
    int(seed_count),
    PRIOR_LABELS[prior_label],
    int(tolerance_exponent),
    scalable_full_features,
)
run_button_label = (
    "Esegui la valutazione multi-seed e costruisci il circuito"
    if preview_circuit_resources.exact_materialization_allowed
    else "Esegui la valutazione multi-seed (circuito non materializzato)"
)
render_section_banner(
    "Sezione 3 · opzionale",
    "3. Confronto con altri classificatori",
    "Avvia soltanto i benchmark che desideri; questa sezione non esegue automaticamente la PGM principale.",
)
st.write(
    "Questa sezione è indipendente dal pulsante principale. Se la abiliti, confronta "
    "la PGM con un classificatore standard sugli stessi split stratificati e sugli "
    "stessi seed. Il modello standard viene ottimizzato con una ricerca compatta sul "
    "solo training set; per ogni metrica vengono riportate media e deviazione standard."
)
st.caption(
    "Nel confronto, PGM indica la configurazione scelta automaticamente sul "
    "training set (encoding e, se applicabile, fattore t). c-PGM, k-PGM e r-PGM "
    "hanno la stessa decisione teorica; il calcolo usa il backend esatto meno oneroso "
    "tra k-PGM e r-PGM, senza riduzione automatica delle feature."
)
comparison_enabled = st.toggle(
    "Abilita il confronto opzionale",
    value=False,
    key="standard_comparison_enabled",
    help=(
        "Nessun benchmark viene eseguito finché non abiliti questa sezione e premi "
        "uno dei due pulsanti."
    ),
)

if comparison_enabled:
    classifier_key = st.selectbox(
        "Classificatore standard",
        options=[candidate.key for candidate in CLASSIFIER_SPECS],
        index=1,
        key="standard_classifier_choice",
        format_func=lambda key: (
            f"{translate_text(get_classifier_spec(key).display_name, current_language())} · "
            f"{translate_text(get_classifier_spec(key).family, current_language())}"
        ),
    )
    classifier_spec = get_classifier_spec(classifier_key)
    st.caption(classifier_spec.description)
    st.info(
        f"Il confronto corrente usa {int(seed_count)} seed e fino a 3 fold di tuning "
        "interni per ciascun training set. Il full comparison usa gli stessi seed e "
        "2 fold per contenere i tempi; i download OpenML sono memorizzati in cache."
    )
    comparison_actions = st.columns(2)
    compare_current_clicked = comparison_actions[0].button(
        "Confronta sul dataset selezionato",
        key="run_selected_dataset_comparison",
        type="primary",
        disabled=not classical_run_allowed,
        width="stretch",
    )
    full_comparison_clicked = comparison_actions[1].button(
        "Full comparison sui dataset binari",
        key="run_full_binary_comparison",
        width="stretch",
    )

    selected_comparison_key = (
        configuration_key,
        classifier_key,
        3,
    )
    if compare_current_clicked:
        try:
            with st.spinner(
                f"Confronto appaiato su {int(seed_count)} seed in corso..."
            ):
                comparison_pgm_evaluation = execute_multiseed_pgm(
                    selected_key,
                    copies,
                    float(test_fraction),
                    int(random_seed),
                    int(seed_count),
                    PRIOR_LABELS[prior_label],
                    10.0 ** (-int(tolerance_exponent)),
                    selected_feature_count,
                    selected_dataset_override,
                    EXPERIMENT_CACHE_SCHEMA,
                )
                standard_results: list[dict] = []
                for seed, comparison_pgm_payload in zip(
                    comparison_pgm_evaluation["seeds"],
                    comparison_pgm_evaluation["payloads"],
                ):
                    (
                        comparison_X_train_raw,
                        comparison_y_train,
                        comparison_X_test_raw,
                        comparison_y_test,
                    ) = benchmark_raw_split(
                        comparison_pgm_payload,
                        dataset_key=selected_key,
                        test_fraction=float(test_fraction),
                        random_seed=int(seed),
                    )
                    standard_results.append(
                        execute_standard_classifier(
                            comparison_X_train_raw,
                            comparison_y_train,
                            comparison_X_test_raw,
                            comparison_y_test,
                            classifier_key,
                            int(seed),
                            3,
                            use_shared_cache=(selected_dataset_override is None),
                        )
                    )
                selected_comparison = aggregate_seed_comparison(
                    comparison_pgm_evaluation,
                    standard_results,
                    bootstrap_resamples=2_000,
                    base_seed=int(random_seed),
                )
            st.session_state["selected_classifier_comparison"] = {
                "key": selected_comparison_key,
                "comparison": selected_comparison,
            }
        except Exception as error:
            st.exception(error)

    saved_comparison = st.session_state.get(
        "selected_classifier_comparison"
    )
    if (
        saved_comparison
        and saved_comparison["key"] == selected_comparison_key
    ):
        render_classifier_comparison(
            saved_comparison["comparison"],
            dataset_display_name=spec.display_name,
        )
    elif saved_comparison:
        st.info(
            "Le impostazioni del confronto sono cambiate: premi il pulsante per "
            "calcolare il nuovo caso."
        )

    binary_specs = [dataset for dataset in DATASETS if dataset.classes == 2]
    if private_dataset is not None and private_dataset["spec"].classes == 2:
        binary_specs.append(private_dataset["spec"])
    full_comparison_key = (
        APP_VERSION,
        tuple(dataset.key for dataset in binary_specs),
        classifier_key,
        copies,
        feature_mode,
        int(selected_feature_count),
        float(test_fraction),
        int(random_seed),
        int(seed_count),
        PRIOR_LABELS[prior_label],
        int(tolerance_exponent),
    )
    if full_comparison_clicked:
        full_rows: list[dict] = []
        progress = st.progress(
            0.0,
            text=translate_text(
                "Preparazione del full comparison...",
                current_language(),
            ),
        )
        for dataset_index, binary_spec in enumerate(binary_specs, start=1):
            progress.progress(
                (dataset_index - 1) / len(binary_specs),
                text=translate_text(
                    f"Dataset {dataset_index}/{len(binary_specs)}: "
                    f"{translate_text(binary_spec.display_name, current_language())}",
                    current_language(),
                ),
            )
            try:
                binary_dataset_override = (
                    (
                        private_dataset["X"],
                        private_dataset["y"],
                        private_dataset["source"],
                    )
                    if private_dataset is not None
                    and binary_spec.key == private_dataset["key"]
                    else None
                )
                full_feature_count = requested_feature_count(
                    raw_feature_count=binary_spec.features,
                    manual_reduction=manual_feature_reduction,
                    manual_feature_count=min(
                        int(manual_feature_count), binary_spec.features
                    ),
                )
                full_test_estimate = ceil(
                    binary_spec.samples * float(test_fraction)
                )
                full_train_estimate = binary_spec.samples - full_test_estimate
                _, full_symmetric_dimension = representation_dimensions(
                    full_feature_count + 1,
                    copies,
                )
                full_backend_dimension = (
                    full_symmetric_dimension
                    if full_symmetric_dimension < full_train_estimate
                    and full_symmetric_dimension
                    <= SCALABLE_EXPLICIT_DIMENSION_LIMIT
                    else full_train_estimate
                )
                full_peak_bytes = 8 * (
                    6 * full_backend_dimension**2
                    + (full_train_estimate + full_test_estimate)
                    * full_backend_dimension
                    + full_train_estimate * (full_feature_count + 1)
                )
                if (
                    full_backend_dimension > SCALABLE_EXPLICIT_DIMENSION_LIMIT
                    or full_peak_bytes > int(memory_budget_gib * 1024**3)
                ):
                    raise ValueError(
                        "Confronto non avviato: la rappresentazione esatta supera "
                        "il budget computazionale selezionato."
                    )
                full_pgm_evaluation = execute_multiseed_pgm(
                    binary_spec.key,
                    copies,
                    float(test_fraction),
                    int(random_seed),
                    int(seed_count),
                    PRIOR_LABELS[prior_label],
                    10.0 ** (-int(tolerance_exponent)),
                    full_feature_count,
                    binary_dataset_override,
                    EXPERIMENT_CACHE_SCHEMA,
                )
                full_standard_results: list[dict] = []
                for seed, full_pgm_payload in zip(
                    full_pgm_evaluation["seeds"],
                    full_pgm_evaluation["payloads"],
                ):
                    (
                        full_X_train_raw,
                        full_y_train,
                        full_X_test_raw,
                        full_y_test,
                    ) = benchmark_raw_split(
                        full_pgm_payload,
                        dataset_key=binary_spec.key,
                        test_fraction=float(test_fraction),
                        random_seed=int(seed),
                    )
                    full_standard_results.append(
                        execute_standard_classifier(
                            full_X_train_raw,
                            full_y_train,
                            full_X_test_raw,
                            full_y_test,
                            classifier_key,
                            int(seed),
                            2,
                            use_shared_cache=(binary_dataset_override is None),
                        )
                    )
                full_case = aggregate_seed_comparison(
                    full_pgm_evaluation,
                    full_standard_results,
                    bootstrap_resamples=750,
                    base_seed=int(random_seed),
                )
                full_seed_summary = full_case["paired_seed_summary"]
                (
                    full_pgm_win_rate,
                    full_competitor_win_rate,
                    full_tie_rate,
                    full_exact_tie,
                ) = seed_outcome_rates(full_seed_summary)
                full_rows.append(
                    {
                        "dataset": binary_spec.display_name,
                        "dataset_key": binary_spec.key,
                        "pgm_balanced_accuracy": full_seed_summary["pgm_mean"],
                        "pgm_balanced_accuracy_std": full_seed_summary["pgm_std"],
                        "competitor_balanced_accuracy": full_seed_summary[
                            "competitor_mean"
                        ],
                        "competitor_balanced_accuracy_std": full_seed_summary[
                            "competitor_std"
                        ],
                        "difference": full_seed_summary["difference_mean"],
                        "difference_std": full_seed_summary["difference_std"],
                        "confidence_lower": full_seed_summary["confidence_lower"],
                        "confidence_upper": full_seed_summary["confidence_upper"],
                        "winner": full_seed_summary["winner"],
                        "seed_count": full_seed_summary["count"],
                        "pgm_win_rate": full_pgm_win_rate,
                        "competitor_win_rate": full_competitor_win_rate,
                        "tie_rate": full_tie_rate,
                        "exact_tie": full_exact_tie,
                        "encoding": full_case["encoding"],
                        "rescaling_factor": full_case[
                            "rescaling_factor"
                        ],
                        "error": "",
                    }
                )
            except Exception as error:
                full_rows.append(
                    {
                        "dataset": binary_spec.display_name,
                        "dataset_key": binary_spec.key,
                        "error": safe_error_message(error),
                    }
                )
        progress.progress(
            1.0,
            text=translate_text(
                "Full comparison completato.",
                current_language(),
            ),
        )
        st.session_state["full_binary_comparison"] = {
            "key": full_comparison_key,
            "rows": full_rows,
            "classifier_name": classifier_spec.display_name,
        }

    saved_full_comparison = st.session_state.get(
        "full_binary_comparison"
    )
    if (
        saved_full_comparison
        and saved_full_comparison["key"] == full_comparison_key
    ):
        render_full_binary_comparison(
            saved_full_comparison["rows"],
            competitor_name=saved_full_comparison["classifier_name"],
        )
    elif saved_full_comparison:
        st.info(
            "Le impostazioni del full comparison sono cambiate: premi il pulsante "
            "per ricalcolarlo."
        )

with st.container(border=True, key="pgm_run_panel"):
    st.markdown("### Avvia la valutazione PGM multi-seed")
    st.write(
        f"Il comando esegue {int(seed_count)} split stratificati e produce i "
        "risultati della sezione 4. Il circuito usa soltanto lo split di "
        f"riferimento con seed {int(random_seed)} e non altera le statistiche."
    )
    if preview_circuit_resources.exact_materialization_allowed:
        st.caption(
            "Con questa configurazione verranno resi disponibili anche il circuito "
            "logico esatto e gli strumenti quantistici compatibili con i limiti "
            "indicati sopra."
        )
    else:
        st.caption(
            "La valutazione classica resta completa; per questa configurazione il "
            "circuito sarà mostrato come schema simbolico dimensionato."
        )
    run_clicked = st.button(
        run_button_label,
        type="primary",
        disabled=not classical_run_allowed,
        width="stretch",
        key="run_multiseed_pgm",
    )

if run_clicked:
    try:
        with st.spinner(
            f"Valutazione PGM su {int(seed_count)} seed in corso..."
        ):
            multiseed_evaluation = execute_multiseed_pgm(
                selected_key,
                copies,
                float(test_fraction),
                int(random_seed),
                int(seed_count),
                PRIOR_LABELS[prior_label],
                10.0 ** (-int(tolerance_exponent)),
                selected_feature_count,
                selected_dataset_override,
                EXPERIMENT_CACHE_SCHEMA,
            )
            payload = multiseed_evaluation["reference_payload"]
        st.session_state["last_pgm_run"] = {
            "configuration_key": configuration_key,
            "payload": payload,
            "multiseed_evaluation": multiseed_evaluation,
        }
    except Exception as error:
        st.exception(error)

render_section_banner(
    "Sezione 4",
    "4. Risultati PGM",
    "Statistiche multi-seed, equivalenza dei metodi, diagnosi e risultati campione per campione.",
)
saved_run = st.session_state.get("last_pgm_run")
current_run_available = bool(
    isinstance(saved_run, dict)
    and saved_run.get("configuration_key") == configuration_key
)
if not saved_run:
    st.info(
        "I risultati compariranno qui dopo aver premuto il pulsante di valutazione "
        "multi-seed immediatamente sopra."
    )
if current_run_available:
    payload = saved_run["payload"]
    multiseed_evaluation = saved_run["multiseed_evaluation"]
    results: dict[str, MethodResult] = payload["results"]
    y_test = payload["y_test"]
    table = results_frame(results, y_test)

    robust_summary = multiseed_evaluation["summary"]
    robust_columns = st.columns(4)
    for column, metric_key, label, scale, suffix in zip(
        robust_columns,
        ("balanced_accuracy", "accuracy", "f1_macro", "cohen_kappa"),
        ("Balanced accuracy", "Accuratezza", "F1-score macro", "Kappa di Cohen"),
        (100.0, 100.0, 100.0, 1.0),
        ("%", "%", "%", ""),
    ):
        statistics = robust_summary[metric_key]
        column.metric(
            f"{label} · media ± dev. std.",
            f"{scale * float(statistics['mean']):.2f}{suffix} ± "
            f"{scale * float(statistics['std']):.2f}",
        )
    st.caption(
        f"Valutazione esterna su {len(multiseed_evaluation['seeds'])} split "
        f"stratificati (seed {multiseed_evaluation['seeds'][0]}–"
        f"{multiseed_evaluation['seeds'][-1]}). Sono inclusi tutti i risultati; "
        "le statistiche sono comuni a c-PGM, k-PGM e r-PGM perché le predizioni "
        "sono esattamente equivalenti. "
        f"Lo split con seed {int(random_seed)} è usato sotto soltanto per matrici "
        "di confusione, dettaglio dei campioni e circuito."
    )
    with st.expander("Dettaglio della valutazione multi-seed", expanded=False):
        robust_rows = []
        for record in multiseed_evaluation["records"]:
            robust_rows.append(
                {
                    "Seed": record["seed"],
                    "Balanced accuracy": record["metrics"]["balanced_accuracy"],
                    "Accuratezza": record["metrics"]["accuracy"],
                    "F1-score macro": record["metrics"]["f1_macro"],
                    "Kappa di Cohen": record["metrics"]["cohen_kappa"],
                    "Backend PGM": record["computational_backend"],
                    "Encoding": record["encoding"],
                    "Fattore t": record["rescaling_factor"],
                }
            )
        st.dataframe(
            pd.DataFrame(robust_rows),
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.4f")
                for name in (
                    "Balanced accuracy",
                    "Accuratezza",
                    "F1-score macro",
                    "Kappa di Cohen",
                )
            },
        )

    selection = payload["encoding_selection"]
    selected_is_stereographic = payload["encoding"] == "stereographic"
    selected_encoding_label = (
        "Stereografico + encoding in ampiezza"
        if selected_is_stereographic
        else "Encoding in ampiezza normalizzato"
    )
    if selected_is_stereographic:
        st.success(
            "Selezione automatica completata: è stato scelto l'encoding "
            "stereografico con fattore di rescaling "
            f"t = {payload['rescaling_factor']:g}. Questa configurazione viene "
            "usata nel training finale, nei tre classificatori, nel circuito e "
            "nelle eventuali esecuzioni su simulatore o QPU."
        )
    else:
        st.success(
            "Selezione automatica completata: è stato scelto l'encoding in "
            "ampiezza con normalizzazione L2. Nessun candidato stereografico ha "
            "superato la baseline oltre la soglia minima richiesta."
        )

    selection_metrics = st.columns(4)
    selection_metrics[0].metric(
        "Encoding selezionato", selected_encoding_label
    )
    selection_metrics[1].metric(
        "Fattore di rescaling t",
        (
            f"{payload['rescaling_factor']:g}"
            if selected_is_stereographic
            else "Non applicabile"
        ),
    )
    selection_metrics[2].metric(
        "Balanced accuracy di validazione",
        f"{100.0 * selection['validation_accuracy']:.2f}%",
    )
    selection_metrics[3].metric(
        "Vantaggio sulla baseline",
        f"{100.0 * selection['improvement']:+.2f} punti %",
    )
    st.caption(
        "La scelta è stata effettuata esclusivamente sul training set mediante "
        f"{selection['protocol']} ({selection['tuning_samples']} campioni). Il test "
        "set non è stato consultato. Per evitare una scelta dovuta al rumore, lo "
        "stereografico deve migliorare la baseline di oltre "
        f"{100.0 * selection['minimum_gain']:.2f} punti percentuali."
    )

    candidate_rows = []
    for candidate in selection["candidates"]:
        is_selected = candidate["key"] == selection["selected_key"]
        if is_selected:
            candidate_status = "✓ Selezionato"
        elif candidate["available"]:
            candidate_status = "Disponibile"
        else:
            candidate_status = "Non disponibile"
        candidate_rows.append(
            {
                "Encoding": (
                    "Encoding in ampiezza normalizzato"
                    if candidate["encoding"] == "tensor_l2"
                    else "Stereografico + encoding in ampiezza"
                ),
                "Fattore t": (
                    "—"
                    if candidate["rescaling_factor"] is None
                    else f"{candidate['rescaling_factor']:g}"
                ),
                "Feature dopo PCA": candidate["base_dimension"],
                "Dimensione encoding": candidate["encoded_dimension"],
                "Balanced accuracy validazione": (
                    f"{100.0 * candidate['validation_accuracy']:.2f}%"
                    if candidate["available"]
                    else "—"
                ),
                "Deviazione standard": (
                    f"{100.0 * candidate['validation_std']:.2f} punti %"
                    if candidate["available"]
                    else "—"
                ),
                "Valutatore PGM": candidate["evaluators"] or "—",
                "Stato": candidate_status,
            }
        )
    with st.expander(
        "Confronto degli encoding sul training set", expanded=False
    ):
        st.dataframe(
            pd.DataFrame(candidate_rows),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Il valutatore indicato è la rappresentazione PGM esatta meno costosa "
            "per quella fold. c-PGM, k-PGM e r-PGM hanno gli stessi score teorici; "
            "il training finale evita diagonalizzazioni duplicate e conserva gli "
            "score equivalenti delle tre formulazioni."
        )

    accuracy_columns = st.columns(3)
    for column, method in zip(accuracy_columns, ("c-PGM", "k-PGM", "r-PGM")):
        accuracy = accuracy_score(y_test, results[method].predictions)
        column.metric(
            f"Accuratezza {method} · seed di riferimento",
            f"{100.0 * accuracy:.4f}%",
        )

    all_predictions_equal = (
        np.array_equal(results["c-PGM"].predictions, results["k-PGM"].predictions)
        and np.array_equal(results["c-PGM"].predictions, results["r-PGM"].predictions)
    )
    pairwise_differences = {
        "c-PGM / k-PGM": float(
            np.max(np.abs(results["c-PGM"].scores - results["k-PGM"].scores))
        ),
        "c-PGM / r-PGM": float(
            np.max(np.abs(results["c-PGM"].scores - results["r-PGM"].scores))
        ),
        "k-PGM / r-PGM": float(
            np.max(np.abs(results["k-PGM"].scores - results["r-PGM"].scores))
        ),
    }
    maximum_score_difference = max(pairwise_differences.values())
    kernel_equivalent_methods = tuple(
        payload.get("kernel_equivalent_methods", ())
    )
    independent_methods = tuple(payload.get("independent_methods", ()))
    if kernel_equivalent_methods:
        if all_predictions_equal:
            st.success(
                "Classificazione completata con la politica di feature richiesta: le tre "
                f"formulazioni restituiscono le stesse predizioni sui {len(y_test)} "
                "campioni di test."
            )
        else:
            st.warning(
                "Una formulazione calcolata esplicitamente non coincide con il "
                "backend esatto scelto su tutti i campioni. Il caso può essere "
                "numericamente quasi degenere: consulta la diagnostica."
            )
        st.info(
            f"Backend adattivo attivo: {select_payload_backend(payload)}. "
            + ", ".join(independent_methods)
            + " è stato calcolato direttamente; "
            + ", ".join(kernel_equivalent_methods)
            + " sono stati valutati mediante l'identità esatta delle Gram matrix "
            f"con {select_payload_backend(payload)}, senza costruire le rispettive "
            "matrici. Gli zeri negli "
            "scarti che coinvolgono questi metodi derivano quindi dall'equivalenza "
            "matematica, non da tre diagonalizzazioni duplicate."
        )
    elif all_predictions_equal:
        st.success(
            "Verifica superata: le predizioni dei tre metodi sono identiche su tutti "
            f"i {len(y_test)} campioni di test. Scarto massimo tra score: "
            f"{maximum_score_difference:.3e}."
        )
    else:
        st.warning(
            "Le predizioni non coincidono tutte. Controlla la soglia spettrale e gli "
            "scarti tra score: il caso può essere numericamente quasi degenere."
        )

    display_table = table.copy()
    display_table["Accuratezza"] = display_table["Accuratezza"].map(
        lambda value: f"{100.0 * value:.4f}%"
    )
    display_table["Accordo con c-PGM"] = display_table["Accordo con c-PGM"].map(
        lambda value: f"{100.0 * value:.2f}%"
    )
    display_table["Training (s)"] = display_table["Training (s)"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.6f}"
    )
    display_table["Predizione (s)"] = display_table["Predizione (s)"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.6f}"
    )
    st.dataframe(display_table, hide_index=True, width="stretch")
    encoding_preprocessing = (
        f"mappa stereografica con t={payload['rescaling_factor']:g}, seguita "
        "dalla preparazione in ampiezza"
        if selected_is_stereographic
        else "normalizzazione L2 e preparazione in ampiezza"
    )
    st.caption(
        f"Dataset caricato da: {payload['source_used']}. Preprocessing finale: "
        f"imputazione mediana, {payload['feature_transform']}, min-max [0.001, 1] e "
        f"{encoding_preprocessing}. Ogni trasformazione dipendente dai dati e la "
        "scelta dell'encoding sono apprese solo sul training set."
    )
    encoding_columns = st.columns(4)
    encoding_columns[0].metric("Feature originali", payload["raw_d"])
    encoding_columns[1].metric("Feature dopo PCA", payload["base_d"])
    encoding_columns[2].metric("Dimensione encoding", payload["d"])
    encoding_columns[3].metric(
        "Varianza PCA conservata",
        (
            f"{100.0 * payload['explained_variance_ratio']:.2f}%"
            if payload["base_d"] < payload["raw_d"]
            else "100% (nessuna PCA)"
        ),
    )

    actual_circuit_resources = resources_for_dataset(
        payload["d"],
        copies,
        payload["class_count"],
        exact_qubit_limit=EXACT_CIRCUIT_QUBIT_LIMIT,
    )
    if not actual_circuit_resources.exact_materialization_allowed:
        classical_predictions = results[select_payload_backend(payload)].predictions
        classical_classes = np.unique(payload["y_train"])
        classical_correct = classical_predictions == y_test
        classical_counts, classical_percentages = confusion_frames(
            y_test, classical_predictions, classical_classes
        )
        st.markdown("#### Classificazione test (calcolo classico adattivo)")
        st.write(
            "Questa valutazione non richiede la costruzione del circuito. Ogni riga "
            "confronta la classe reale con la predizione comune a c-PGM, k-PGM e "
            "r-PGM."
        )
        classical_metrics = st.columns(3)
        classical_metrics[0].metric("Campioni test", f"{len(y_test):,}")
        classical_metrics[1].metric(
            "Classificazioni corrette", f"{int(np.sum(classical_correct)):,}"
        )
        classical_metrics[2].metric(
            "Errori", f"{int(np.sum(~classical_correct)):,}"
        )
        confusion_left, confusion_right = st.columns(2)
        with confusion_left:
            st.markdown("**Matrice di confusione — campioni**")
            st.dataframe(classical_counts, width="stretch")
        with confusion_right:
            st.markdown("**Matrice di confusione — % per classe reale**")
            st.dataframe(
                classical_percentages.map(lambda value: f"{value:.1f}%"),
                width="stretch",
            )
        classical_rows = pd.DataFrame(
            {
                "Campione test": np.arange(1, len(y_test) + 1),
                "Classe reale": y_test,
                "Classe predetta": classical_predictions,
                "Esito": np.where(classical_correct, "✓ Corretta", "✗ Errata"),
            }
        )
        with st.expander("Risultato di ogni campione", expanded=False):
            st.dataframe(classical_rows, hide_index=True, width="stretch")

    with st.expander("Diagnostica numerica", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Confronto": name, "Scarto massimo assoluto score": difference}
                    for name, difference in pairwise_differences.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.write(
            {
                method: {
                    "rank": result.rank,
                    "soglia_spettrale": result.spectral_threshold,
                    "autovalore_minimo_grezzo": result.minimum_eigenvalue,
                    "errore_massimo_norma_feature": result.maximum_feature_norm_error,
                }
                for method, result in results.items()
            }
        )

    actual_rank = results["k-PGM"].rank
    actual_complexities = paper_complexities(
        n_train=len(payload["y_train"]),
        dimension=payload["d"],
        copies=copies,
        class_count=payload["class_count"],
        gram_rank=actual_rank,
    )
    st.markdown("#### Complessità con i valori effettivi del run")
    st.dataframe(
        complexity_frame(actual_complexities),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "I tempi osservati dipendono da BLAS, CPU, cache e carico della macchina; le "
        "classi di complessità descrivono invece la crescita asintotica del paper. "
        "L'implementazione usa decomposizioni a rank ridotto e non materializza le N "
        "matrici densità, quindi la colonna 'Stato modello' non coincide con il bound "
        "di memoria della costruzione didattica del paper."
    )

    render_section_banner(
        "Sezione 5 · laboratorio quantistico",
        "5. Circuito quantistico della PGM",
        "Circuito logico, classificazione, validazione, optimizer, simulatori, QPU ed export in un unico spazio protetto.",
    )
    render_quantum_tool_strip()
    resources = actual_circuit_resources
    circuit_metrics = st.columns(5)
    circuit_metrics[0].metric("Dimensione ridotta", f"{resources.feature_dimension:,}")
    circuit_metrics[1].metric("Qubit sistema", resources.system_qubits)
    circuit_metrics[2].metric("Qubit esito", resources.outcome_qubits)
    circuit_metrics[3].metric("Qubit totali", resources.total_qubits)
    circuit_metrics[4].metric(
        "Matrice U_PGM",
        f"{resources.unitary_dimension:,} x {resources.unitary_dimension:,}",
    )

    st.markdown(
        r"""
        L'app usa la PGM ridotta, equivalente alle altre due formulazioni. Completa
        gli effetti sul nucleo di $\sigma$ e costruisce l'isometria canonica

        $$
        U_{\mathrm{PGM}}\bigl(|0\rangle_{\mathrm{out}}\otimes|\psi\rangle_{\mathrm{sys}}\bigr)
        =\sum_j |j\rangle_{\mathrm{out}}\otimes\sqrt{F_j}\,|\psi\rangle_{\mathrm{sys}},
        \qquad
        \Pr(j\mid\psi)=\langle\psi|F_j|\psi\rangle .
        $$

        Il registro **sys** riceve lo stato test già codificato nella base simmetrica;
        il registro **out** parte da zero e la sua misura restituisce la classe.
        """
    )

    classes = np.unique(payload["y_train"])
    outcome_table = pd.DataFrame(
        [
            {
                "Indice esito": outcome,
                "Bitstring misurata": format(
                    outcome, f"0{resources.outcome_qubits}b"
                ),
                "Interpretazione": (
                    f"Classe: {classes[outcome]}"
                    if outcome < len(classes)
                    else "Esito non usato (probabilità teorica zero)"
                ),
            }
            for outcome in range(resources.padded_outcome_dimension)
        ]
    )

    try:
        if resources.exact_materialization_allowed:
            with st.spinner(
                "Costruzione della POVM e della dilatazione unitaria esatta..."
            ):
                measurement, dilation = construct_dilation_cached(
                    payload["X_train_encoded"],
                    payload["y_train"],
                    copies,
                    PRIOR_LABELS[prior_label],
                    10.0 ** (-int(tolerance_exponent)),
                    EXACT_CIRCUIT_QUBIT_LIMIT,
                    use_shared_cache=(selected_dataset_override is None),
                )
                circuit = build_qiskit_circuit(dilation)
                circuit_drawing = translate_text(
                    circuit_svg(dilation), current_language()
                )
                test_states = symmetric_feature_map(
                    payload["X_test_encoded"], copies
                )
                circuit_probabilities = np.vstack(
                    [outcome_probabilities(dilation, state) for state in test_states]
                )
                theoretical_probabilities = np.column_stack(
                    [
                        np.einsum(
                            "ni,ij,nj->n",
                            test_states,
                            effect,
                            test_states,
                            optimize=True,
                        )
                        for effect in measurement.effects
                    ]
                )
                class_probabilities = circuit_probabilities[:, : len(classes)]
                circuit_predictions = stable_predictions(
                    class_probabilities, classes
                )
                classification_details = classification_details_frame(
                    payload["y_test"],
                    circuit_predictions,
                    classes,
                    theoretical_probabilities,
                    class_probabilities,
                )
                confusion_counts, confusion_percentages = confusion_frames(
                    payload["y_test"], circuit_predictions, classes
                )

            st.success(
                "Circuito esatto costruito: la matrice U_PGM contiene i parametri "
                "appresi nel training, non è un blocco puramente illustrativo."
            )
            logical_qpy = qpy_bytes(circuit)
            synthesis_id = (
                f"{APP_VERSION}|{selected_key}|{copies}|{int(random_seed)}|"
                f"{PRIOR_LABELS[prior_label]}|{int(tolerance_exponent)}|"
                f"{feature_mode}|{float(test_fraction):.3f}|{payload['encoding']}|"
                f"{payload['rescaling_factor']}|d={payload['d']}"
            )
            circuit_tab, outcomes_tab, checks_tab, hardware_tab, export_tab = st.tabs(
                [
                    "Circuito logico",
                    "Classificazione test",
                    "Validazione matematica",
                    "Esecuzione quantistica",
                    "Esporta",
                ]
            )
            with circuit_tab:
                st.markdown(circuit_drawing, unsafe_allow_html=True)
                st.caption(
                    "Circuito logico esatto: ingresso codificato su sys, ancilla "
                    "|0...0> su out, dilatazione di Naimark e misura dell'esito."
                )
                st.info(
                    "U_PGM è mostrato come un'unica porta unitaria per mantenere il "
                    "diagramma logico leggibile. Sotto puoi sintetizzarlo nella base "
                    "generica rz/sx/x/cx e vedere il circuito completo su più righe."
                )
                cnot_upper_bound = generic_unitary_cnot_upper_bound(
                    resources.total_qubits
                )
                estimate_columns = st.columns(3)
                estimate_columns[0].metric(
                    "Limite superiore CNOT (unitaria generica)",
                    f"{cnot_upper_bound:,}",
                )
                estimate_columns[1].metric(
                    "Soglia disegno completo", f"{FULL_GATE_DIAGRAM_LIMIT:,} porte"
                )
                estimate_columns[2].metric(
                    "Sintesi isolata fino a", f"{ISOLATED_SYNTHESIS_QUBIT_LIMIT} qubit"
                )
                st.caption(
                    "Il limite CNOT è una stima prudenziale pre-sintesi per un'unitaria "
                    "arbitraria. I conteggi esatti compaiono dopo la transpilation."
                )

                if resources.total_qubits <= ISOLATED_SYNTHESIS_QUBIT_LIMIT:
                    if st.button(
                        "Sintetizza e mostra la decomposizione completa",
                        key=f"native_{selected_key}_{copies}_{int(random_seed)}",
                    ):
                        with st.spinner(
                            "Sintesi in un processo protetto: l'interfaccia non può "
                            "più essere chiusa da un crash nativo..."
                        ):
                            synthesis_report = transpile_in_isolated_process(
                                logical_qpy,
                                300 if resources.total_qubits == 6 else 180,
                                FULL_GATE_DIAGRAM_LIMIT,
                            )
                        st.session_state["native_synthesis_report"] = {
                            "id": synthesis_id,
                            "report": synthesis_report,
                        }

                    saved_synthesis = st.session_state.get("native_synthesis_report")
                    if saved_synthesis and saved_synthesis["id"] == synthesis_id:
                        synthesis_report = saved_synthesis["report"]
                        if synthesis_report.status in {"success", "partial"}:
                            if synthesis_report.status == "success":
                                st.success(
                                    "Decomposizione completata senza coinvolgere il "
                                    "processo Streamlit."
                                )
                            else:
                                st.warning(synthesis_report.message)
                            synthesis_metrics = st.columns(3)
                            synthesis_metrics[0].metric(
                                "Porte totali", f"{synthesis_report.size:,}"
                            )
                            synthesis_metrics[1].metric(
                                "Profondità", f"{synthesis_report.depth:,}"
                            )
                            synthesis_metrics[2].metric(
                                "Tempo sintesi",
                                f"{synthesis_report.elapsed_seconds:.2f} s",
                            )
                            gate_table = pd.DataFrame(
                                [
                                    {"Gate": name, "Numero": count}
                                    for name, count in sorted(
                                        synthesis_report.gate_counts.items(),
                                        key=lambda item: (-item[1], item[0]),
                                    )
                                ]
                            )
                            st.markdown("**Conteggio esatto delle porte**")
                            st.dataframe(
                                gate_table, hide_index=True, width="stretch"
                            )
                            if synthesis_report.diagram_text is not None:
                                st.markdown(
                                    "**Circuito completo** — scorri verticalmente e "
                                    "orizzontalmente; nessuna porta è omessa."
                                )
                                st.text_area(
                                    "Diagramma circuitale completo",
                                    value=synthesis_report.diagram_text,
                                    height=650,
                                    disabled=True,
                                    key=f"diagram_{synthesis_id}",
                                )
                                st.download_button(
                                    "Scarica circuito completo (TXT)",
                                    data=synthesis_report.diagram_text.encode("utf-8"),
                                    file_name=localized_filename(
                                        f"circuito_completo_{selected_key}_c{copies}.txt",
                                        f"full_circuit_{selected_key}_c{copies}.txt",
                                    ),
                                    mime="text/plain",
                                )
                            else:
                                st.warning(
                                    "La sintesi è riuscita e i conteggi sopra sono "
                                    "esatti, ma il circuito supera "
                                    f"{FULL_GATE_DIAGRAM_LIMIT:,} porte: il disegno "
                                    "integrale sarebbe poco utilizzabile."
                                )
                            if synthesis_report.transpiled_qpy is not None:
                                st.download_button(
                                    "Scarica circuito decomposto (QPY)",
                                    data=synthesis_report.transpiled_qpy,
                                    file_name=localized_filename(
                                        f"circuito_decomposto_{selected_key}_c{copies}.qpy",
                                        f"decomposed_circuit_{selected_key}_c{copies}.qpy",
                                    ),
                                    mime="application/octet-stream",
                                )
                        else:
                            st.warning(synthesis_report.message)
                            st.info(
                                "Il crash è rimasto confinato nel processo di sintesi: "
                                "Streamlit e tutti i risultati del training restano attivi."
                            )

                    st.divider()
                    st.markdown("#### Ottimizzazione certificata")
                    st.write(
                        "L'ottimizzazione sintetizza direttamente l'isometria di "
                        "Naimark, confronta tre strategie e conserva quella con meno "
                        "porte entangling. Il risultato viene accettato soltanto se "
                        "l'azione coincide con U_PGM su ogni ingresso valido, non "
                        "soltanto sui campioni del test set."
                    )
                    st.caption(
                        "Fuori dal sottospazio con il registro out inizializzato a zero "
                        "le due estensioni unitarie possono differire: quella parte non "
                        "è mai utilizzata dalla PGM."
                    )
                    if st.button(
                        "Ottimizza e certifica equivalenza",
                        type="primary",
                        key=f"optimize_{synthesis_id}",
                    ):
                        with st.spinner(
                            "Sintesi originale, ricerca del circuito più corto e "
                            "certificazione numerica in processi isolati..."
                        ):
                            baseline_report = None
                            if (
                                saved_synthesis
                                and saved_synthesis["id"] == synthesis_id
                                and saved_synthesis["report"].status
                                in {"success", "partial"}
                            ):
                                baseline_report = saved_synthesis["report"]
                            if baseline_report is None:
                                baseline_report = transpile_in_isolated_process(
                                    logical_qpy,
                                    300 if resources.total_qubits == 6 else 180,
                                    FULL_GATE_DIAGRAM_LIMIT,
                                )

                            candidate_reports, best_report = (
                                certified_isometry_candidates(
                                    isometry_matrix=dilation.isometry,
                                    system_qubits=resources.system_qubits,
                                    outcome_qubits=resources.outcome_qubits,
                                    reference_qpy_payload=logical_qpy,
                                    input_subspace_dimension=(
                                        resources.padded_system_dimension
                                    ),
                                    timeout_seconds=(
                                        300
                                        if resources.total_qubits == 6
                                        else 180
                                    ),
                                    seed_transpiler=42,
                                )
                            )
                        st.session_state["certified_optimization"] = {
                            "id": synthesis_id,
                            "baseline": baseline_report,
                            "best": best_report,
                            "attempts": candidate_reports,
                        }

                    saved_optimization = st.session_state.get(
                        "certified_optimization"
                    )
                    if (
                        saved_optimization
                        and saved_optimization["id"] == synthesis_id
                    ):
                        baseline_report = saved_optimization["baseline"]
                        best_report = saved_optimization["best"]
                        attempts = saved_optimization["attempts"]
                        if (
                            baseline_report.status not in {"success", "partial"}
                            or baseline_report.size is None
                            or baseline_report.depth is None
                        ):
                            st.error(
                                "Non è stato possibile sintetizzare il circuito "
                                "originale di riferimento. Nessuna ottimizzazione è "
                                "stata dichiarata equivalente."
                            )
                        elif best_report is None:
                            st.error(
                                "Nessuno dei tentativi ha superato la certificazione. "
                                "Il circuito originale resta invariato e utilizzabile."
                            )
                            with st.expander("Dettagli dei tentativi", expanded=False):
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "Livello": report.optimization_level,
                                                "Stato": report.status,
                                                "Errore": report.equivalence_error,
                                                "Messaggio": report.message,
                                            }
                                            for report in attempts
                                        ]
                                    ),
                                    hide_index=True,
                                    width="stretch",
                                )
                        else:
                            original_cx = entangling_gate_count(baseline_report)
                            optimized_cx = entangling_gate_count(best_report)
                            comparison = pd.DataFrame(
                                [
                                    {
                                        "Indicatore": "Porte totali",
                                        "Originale": baseline_report.size,
                                        "Ottimizzato": best_report.size,
                                        "Riduzione": (
                                            f"{_reduction_percent(baseline_report.size, best_report.size):.1f}%"
                                        ),
                                    },
                                    {
                                        "Indicatore": "CX / porte entangling",
                                        "Originale": original_cx,
                                        "Ottimizzato": optimized_cx,
                                        "Riduzione": (
                                            f"{_reduction_percent(original_cx, optimized_cx):.1f}%"
                                        ),
                                    },
                                    {
                                        "Indicatore": "Profondità",
                                        "Originale": baseline_report.depth,
                                        "Ottimizzato": best_report.depth,
                                        "Riduzione": (
                                            f"{_reduction_percent(baseline_report.depth, best_report.depth):.1f}%"
                                        ),
                                    },
                                ]
                            )
                            st.success(
                                "Equivalenza certificata sull'intero sottospazio PGM. "
                                "Classi, probabilità teoriche e accuratezza restano "
                                "invariate entro la tolleranza indicata."
                            )
                            certificate_columns = st.columns(3)
                            certificate_columns[0].metric(
                                "Errore massimo",
                                f"{best_report.equivalence_error:.3e}",
                            )
                            certificate_columns[1].metric(
                                "Tolleranza",
                                f"{best_report.equivalence_tolerance:.1e}",
                            )
                            certificate_columns[2].metric(
                                "Fedeltà sottospazio",
                                f"{best_report.subspace_fidelity:.12f}",
                            )
                            st.dataframe(
                                comparison, hide_index=True, width="stretch"
                            )
                            st.caption(
                                f"Migliore sintesi: livello "
                                f"{best_report.optimization_level}, seed "
                                f"{best_report.seed_transpiler}. Criterio: minimo numero "
                                "di porte entangling, poi profondità e porte totali."
                            )
                            with st.expander(
                                "Circuito ottimizzato e conteggio porte",
                                expanded=False,
                            ):
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {"Gate": name, "Numero": count}
                                            for name, count in sorted(
                                                best_report.gate_counts.items(),
                                                key=lambda item: (-item[1], item[0]),
                                            )
                                        ]
                                    ),
                                    hide_index=True,
                                    width="stretch",
                                )
                                if best_report.diagram_text is not None:
                                    st.text_area(
                                        "Diagramma completo ottimizzato",
                                        best_report.diagram_text,
                                        height=650,
                                        disabled=True,
                                        key=f"optimized_diagram_{synthesis_id}",
                                    )
                                else:
                                    st.info(
                                        "Il circuito supera la soglia grafica; i "
                                        "conteggi e la certificazione restano completi."
                                    )
                            download_columns = st.columns(2)
                            if best_report.transpiled_qpy is not None:
                                download_columns[0].download_button(
                                    "Scarica circuito ottimizzato (QPY)",
                                    data=best_report.transpiled_qpy,
                                    file_name=localized_filename(
                                        f"circuito_ottimizzato_{selected_key}_c{copies}.qpy",
                                        f"optimized_circuit_{selected_key}_c{copies}.qpy",
                                    ),
                                    mime="application/octet-stream",
                                )
                            if best_report.diagram_text is not None:
                                download_columns[1].download_button(
                                    "Scarica diagramma ottimizzato (TXT)",
                                    data=best_report.diagram_text.encode("utf-8"),
                                    file_name=localized_filename(
                                        f"circuito_ottimizzato_{selected_key}_c{copies}.txt",
                                        f"optimized_circuit_{selected_key}_c{copies}.txt",
                                    ),
                                    mime="text/plain",
                                )
                else:
                    st.warning(
                        "Il circuito logico esatto, la classificazione test e la "
                        "validazione restano disponibili. La sintesi gate-by-gate e "
                        "l'optimizer sono invece disabilitati oltre "
                        f"{ISOLATED_SYNTHESIS_QUBIT_LIMIT} qubit: per questa unitaria "
                        f"generica il limite prudenziale è circa {cnot_upper_bound:,} "
                        "CNOT, oltre alle rotazioni a un qubit. Questa separazione "
                        "protegge l'interfaccia da tempi e memoria non prevedibili."
                    )
                    protected_columns = st.columns(2)
                    protected_columns[0].button(
                        "Sintetizza e mostra la decomposizione completa",
                        disabled=True,
                        width="stretch",
                        key=f"native_disabled_{synthesis_id}",
                    )
                    protected_columns[1].button(
                        "Ottimizza e certifica equivalenza",
                        disabled=True,
                        width="stretch",
                        key=f"optimizer_disabled_{synthesis_id}",
                    )
                    st.caption(
                        "Per attivare questi due comandi puoi richiedere una PCA "
                        "manuale, ridurre il numero di copie oppure scegliere un "
                        "dataset con meno classi. La PGM e le sue feature non vengono "
                        "modificate automaticamente."
                    )

            with outcomes_tab:
                correct_mask = circuit_predictions == payload["y_test"]
                correct_count = int(np.sum(correct_mask))
                error_count = int(len(correct_mask) - correct_count)
                classification_metrics = st.columns(4)
                classification_metrics[0].metric(
                    "Campioni test", f"{len(correct_mask):,}"
                )
                classification_metrics[1].metric(
                    "Classificazioni corrette", f"{correct_count:,}"
                )
                classification_metrics[2].metric(
                    "Errori", f"{error_count:,}"
                )
                classification_metrics[3].metric(
                    "Accuratezza", f"{100.0 * np.mean(correct_mask):.2f}%"
                )

                st.markdown("##### Corrispondenza qubit misurati → classe")
                mapping_text = "   ·   ".join(
                    f"`{format(index, f'0{resources.outcome_qubits}b')}` → classe **{label}**"
                    for index, label in enumerate(classes)
                )
                st.info(mapping_text)
                st.caption(
                    "La bitstring è il valore misurato nel registro di uscita `out`: "
                    "per esempio `000` identifica la prima classe elencata. Gli eventuali "
                    "stati binari eccedenti non sono assegnati e hanno probabilità teorica zero."
                )

                st.markdown("##### Matrice di confusione")
                st.caption(
                    "Le righe sono le classi reali e le colonne le classi predette. "
                    "La diagonale contiene le classificazioni corrette."
                )
                confusion_left, confusion_right = st.columns(2)
                with confusion_left:
                    st.markdown("**Numero di campioni**")
                    st.dataframe(confusion_counts, width="stretch")
                with confusion_right:
                    st.markdown("**Percentuale per classe reale**")
                    displayed_percentages = confusion_percentages.map(
                        lambda value: f"{value:.1f}%"
                    )
                    st.dataframe(displayed_percentages, width="stretch")

                st.markdown("##### Risultato di ogni campione")
                st.write(
                    "Ogni riga confronta l'etichetta reale con quella scelta dal circuito. "
                    "La **confidenza** è la probabilità della classe predetta; il "
                    "**margine 1ª-2ª** è il distacco dalla seconda classe più probabile. "
                    "La colonna **Lettura rapida** riassume il risultato in una frase."
                )
                st.caption(
                    "Lo scostamento circuito-teoria è un controllo numerico: dovrebbe "
                    "restare vicino alla precisione macchina e non misura la qualità "
                    "statistica della classificazione."
                )
                row_filter = st.radio(
                    "Mostra",
                    ["Solo errori", "Tutti i campioni", "Solo corretti"],
                    horizontal=True,
                    key=f"classification_filter_{synthesis_id}",
                )
                if row_filter == "Solo errori":
                    displayed_details = classification_details[
                        classification_details["Esito"] == "✗ Errata"
                    ]
                elif row_filter == "Solo corretti":
                    displayed_details = classification_details[
                        classification_details["Esito"] == "✓ Corretta"
                    ]
                else:
                    displayed_details = classification_details

                compact_columns = [
                    "Campione test",
                    "Classe reale",
                    "Classe predetta",
                    "Esito",
                    "Confidenza",
                    "Margine 1ª-2ª",
                    "Scostamento max |circuito-teoria|",
                    "Lettura rapida",
                ]
                if displayed_details.empty:
                    st.success("Nessun campione in questa categoria.")
                else:
                    st.dataframe(
                        displayed_details[compact_columns],
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Confidenza": st.column_config.NumberColumn(
                                format="%.6f"
                            ),
                            "Margine 1ª-2ª": st.column_config.NumberColumn(
                                format="%.6f"
                            ),
                            "Scostamento max |circuito-teoria|": (
                                st.column_config.NumberColumn(format="%.3e")
                            ),
                            "Lettura rapida": st.column_config.TextColumn(
                                width="large"
                            ),
                        },
                    )

                with st.expander(
                    "Dettaglio completo delle probabilità teoriche e circuitali",
                    expanded=False,
                ):
                    st.dataframe(
                        classification_details,
                        hide_index=True,
                        width="stretch",
                    )
                    st.markdown("**Mappa bitstring → classe**")
                    st.dataframe(
                        outcome_table, hide_index=True, width="stretch"
                    )

                st.download_button(
                    "Scarica classificazioni dettagliate (CSV)",
                    data=spreadsheet_safe_csv_bytes(
                        localize_dataframe(
                            classification_details, current_language()
                        )
                    ),
                    file_name=localized_filename(
                        f"classificazioni_dettagliate_{selected_key}_c{copies}.csv",
                        f"detailed_classifications_{selected_key}_c{copies}.csv",
                    ),
                    mime="text/csv",
                )

            with checks_tab:
                circuit_agreement = float(
                    np.mean(circuit_predictions == results["r-PGM"].predictions)
                )
                probability_sum_error = float(
                    np.max(np.abs(circuit_probabilities.sum(axis=1) - 1.0))
                )
                probability_deviations = np.abs(
                    class_probabilities - theoretical_probabilities
                )
                maximum_theory_deviation = float(np.max(probability_deviations))
                mean_theory_deviation = float(np.mean(probability_deviations))

                st.markdown("##### Circuito numerico rispetto alla teoria")
                st.write(
                    "Per ogni campione e classe confronto la probabilità ottenuta "
                    "applicando la matrice unitaria del circuito con il valore teorico "
                    r"$\langle\phi_c(x)|F_j|\phi_c(x)\rangle$. Uno scostamento vicino "
                    "alla precisione macchina indica che la dilatazione implementa "
                    "correttamente la POVM."
                )
                validation_metrics = st.columns(4)
                validation_metrics[0].metric(
                    "Scostamento massimo", f"{maximum_theory_deviation:.3e}"
                )
                validation_metrics[1].metric(
                    "Scostamento medio", f"{mean_theory_deviation:.3e}"
                )
                validation_metrics[2].metric(
                    "Errore somma probabilità", f"{probability_sum_error:.3e}"
                )
                validation_metrics[3].metric(
                    "Accordo circuito/r-PGM", f"{100.0 * circuit_agreement:.2f}%"
                )

                check_rows = [
                    {
                        "Controllo": "Somma degli effetti F_j = I",
                        "Errore": measurement.completeness_error,
                        "Significato": "La POVM è completa",
                    },
                    {
                        "Controllo": "Isometria V†V = I",
                        "Errore": dilation.input_isometry_error,
                        "Significato": "V conserva la norma",
                    },
                    {
                        "Controllo": "Unitarietà U†U = I",
                        "Errore": dilation.unitary_error,
                        "Significato": "U_PGM è una trasformazione quantistica valida",
                    },
                    {
                        "Controllo": "Probabilità Born ricostruite",
                        "Errore": dilation.probability_operator_error,
                        "Significato": "Il circuito riproduce gli effetti F_j",
                    },
                    {
                        "Controllo": "Somma probabilità sul test set = 1",
                        "Errore": probability_sum_error,
                        "Significato": "Ogni distribuzione è normalizzata",
                    },
                ]
                st.dataframe(
                    pd.DataFrame(check_rows), hide_index=True, width="stretch"
                )
                if circuit_agreement == 1.0:
                    st.success(
                        "Verifica superata: le predizioni del circuito coincidono con "
                        "r-PGM sul 100% del test set."
                    )
                else:
                    st.warning(
                        "Accordo circuito/r-PGM: "
                        f"{100.0 * circuit_agreement:.4f}%. Controllare i casi di "
                        "pareggio numerico."
                    )
                st.caption(
                    "La dilatazione unitaria non è unica: completamenti unitari diversi "
                    "producono le stesse probabilità sui dati in ingresso con out=0."
                )

            with hardware_tab:
                st.markdown("##### Esegui la PGM su simulatore o hardware reale")
                st.write(
                    "Il circuito eseguibile include la preparazione dello stato del "
                    "campione selezionato. Per le esecuzioni circuitali, l'app prova "
                    "automaticamente i livelli di ottimizzazione 1, 2 e 3 e usa "
                    "soltanto il miglior circuito che supera la certificazione. Puoi "
                    "configurare shot, seed, modello di rumore e provider. Le credenziali "
                    "non entrano negli export o nella cache dell'app; possono essere "
                    "lette da variabili d'ambiente o dal file locale Streamlit Secrets."
                )
                python_status = hardware_python_status()
                st.caption(
                    f"Python {python_status['version']} · AWS Braket richiede Python "
                    ">=3.11 · LRZ MQSS richiede Python >=3.9 e <3.14 · Aer, IBM, "
                    "IonQ e AQT sono consigliati con Python >=3.10."
                )

                sample_index = st.selectbox(
                    "Campione del test set da eseguire",
                    options=list(range(len(payload["y_test"]))),
                    format_func=lambda index: (
                        f"Test #{index + 1} · reale={payload['y_test'][index]} · "
                        f"predizione ideale={circuit_predictions[index]}"
                    ),
                    key=f"hardware_sample_{synthesis_id}",
                )
                execution_settings = st.columns(2)
                with execution_settings[0]:
                    shots = int(
                        st.number_input(
                            "Numero di shot",
                            min_value=10,
                            max_value=100_000,
                            value=1_000,
                            step=100,
                            key=f"hardware_shots_{synthesis_id}",
                        )
                    )
                with execution_settings[1]:
                    execution_seed = int(
                        st.number_input(
                            "Seed simulatore/transpiler",
                            min_value=0,
                            max_value=2_147_483_647,
                            value=int(random_seed) + int(sample_index),
                            step=1,
                            key=f"execution_seed_{synthesis_id}",
                        )
                    )
                st.caption(
                    "Esecuzioni circuitali: ottimizzazione automatica certificata, "
                    "minimizzando gate entangling, poi profondità e porte totali."
                )

                execution_kind = st.radio(
                    "Tipo di risorsa",
                    ["🧪 Simulatori", "⚛️ Computer quantistici reali"],
                    horizontal=True,
                    key=f"execution_kind_{synthesis_id}",
                )

                selected_provider = None
                selected_device = None
                aws_profile = None
                aws_region = None
                aws_access_key_id = ""
                aws_secret_access_key = ""
                aws_session_token = ""
                lrz_token = ""
                ionq_token = ""
                aqt_token = ""
                ibm_token = ""
                ibm_instance = ""
                lrz_queue_offline = True
                ionq_noise_model = "ideal"
                is_paid_external_resource = False

                if execution_kind == "🧪 Simulatori":
                    simulator_choice = st.selectbox(
                        "Simulatore",
                        [
                            "💻 PGM ideale locale — campionamento diretto",
                            "🧩 Qiskit BasicSimulator locale — circuito ideale",
                            "🧰 Qiskit Aer locale — circuito ideale o rumoroso",
                            "🔷 IBM Fake Backend locale — rumore da snapshot QPU",
                            "🟧 Amazon Braket Local — state vector o density matrix",
                            "🟪 AQT Offline — ideale o rumoroso",
                            "☁️ AQT Cloud — simulatori autorizzati",
                            "☁️ IonQ Cloud — ideale o modello di rumore",
                            "☁️ Amazon Braket — simulatore gestito AWS",
                        ],
                        key=f"simulator_choice_{synthesis_id}",
                    )

                    if simulator_choice.startswith("💻"):
                        st.caption(
                            "Modalità più veloce: campiona direttamente dalla distribuzione "
                            "teorica della PGM, senza simulare ogni porta del circuito."
                        )
                        local_id = (
                            f"{synthesis_id}|ideal|{sample_index}|{shots}|{execution_seed}"
                        )
                        if st.button(
                            "Esegui campionamento ideale",
                            type="primary",
                            key=f"local_simulate_{local_id}",
                        ):
                            local_counts = simulate_ideal_shots(
                                dilation,
                                test_states[sample_index],
                                shots=shots,
                                seed=execution_seed,
                            )
                            st.session_state["local_quantum_result"] = {
                                "id": local_id,
                                "sample_index": sample_index,
                                "counts": local_counts,
                                "engine": "ideal",
                            }
                        local_result = st.session_state.get("local_quantum_result")
                        if local_result and local_result["id"] == local_id:
                            result_sample = int(local_result["sample_index"])
                            render_execution_result(
                                local_result["counts"],
                                classes=classes,
                                theoretical_outcomes=circuit_probabilities[result_sample],
                                outcome_qubits=resources.outcome_qubits,
                                actual_class=payload["y_test"][result_sample],
                                prediction_title="Predizione a shot",
                            )

                    elif simulator_choice.startswith("🧰"):
                        st.caption(
                            "Qiskit Aer esegue il circuito completo. Puoi scegliere il "
                            "metodo numerico e aggiungere un semplice rumore depolarizzante."
                        )
                        if not modules_available("qiskit_aer"):
                            st.warning(
                                "Qiskit Aer non è installato. Usa "
                                "`Installa_Provider_Quantistici.command` e scegli "
                                "Qiskit Aer, oppure installa "
                                "`requirements-simulators.txt`."
                            )
                        else:
                            aer_methods = {
                                "Automatico": "automatic",
                                "Statevector": "statevector",
                                "Matrice densità": "density_matrix",
                                "Matrix Product State": "matrix_product_state",
                            }
                            aer_method_label = st.selectbox(
                                "Metodo Aer",
                                list(aer_methods),
                                key=f"aer_method_{synthesis_id}",
                            )
                            aer_method = aer_methods[aer_method_label]
                            aer_noise_mode = st.radio(
                                "Modello di rumore",
                                ["Ideale", "Depolarizzante configurabile"],
                                horizontal=True,
                                key=f"aer_noise_mode_{synthesis_id}",
                            )
                            one_qubit_error = 0.0
                            two_qubit_error = 0.0
                            readout_error = 0.0
                            if aer_noise_mode != "Ideale":
                                noise_columns = st.columns(3)
                                one_qubit_error = float(
                                    noise_columns[0].number_input(
                                        "Errore gate 1-qubit",
                                        min_value=0.0,
                                        max_value=0.20,
                                        value=0.001,
                                        step=0.001,
                                        format="%.4f",
                                        key=f"aer_p1_{synthesis_id}",
                                    )
                                )
                                two_qubit_error = float(
                                    noise_columns[1].number_input(
                                        "Errore gate 2-qubit",
                                        min_value=0.0,
                                        max_value=0.40,
                                        value=0.010,
                                        step=0.005,
                                        format="%.4f",
                                        key=f"aer_p2_{synthesis_id}",
                                    )
                                )
                                readout_error = float(
                                    noise_columns[2].number_input(
                                        "Errore di lettura",
                                        min_value=0.0,
                                        max_value=0.40,
                                        value=0.010,
                                        step=0.005,
                                        format="%.4f",
                                        key=f"aer_readout_{synthesis_id}",
                                    )
                                )
                            aer_id = (
                                f"{synthesis_id}|aer|{sample_index}|{shots}|"
                                f"{execution_seed}|auto123|{aer_method}|"
                                f"{one_qubit_error}|{two_qubit_error}|{readout_error}"
                            )
                            aer_too_large = (
                                resources.total_qubits
                                > ISOLATED_SYNTHESIS_QUBIT_LIMIT
                            )
                            if aer_too_large:
                                st.error(
                                    "Ottimizzazione Aer disabilitata: il circuito "
                                    f"supera {ISOLATED_SYNTHESIS_QUBIT_LIMIT} qubit."
                                )
                            if st.button(
                                "Ottimizza automaticamente ed esegui con Aer",
                                type="primary",
                                disabled=aer_too_large,
                                key=f"aer_simulate_{aer_id}",
                            ):
                                try:
                                    reference_sample_circuit = build_sample_circuit(
                                        dilation,
                                        test_states[sample_index],
                                        name=(
                                            f"PGM_test_{sample_index + 1}_reference"
                                        ),
                                        optimized_isometry=False,
                                    )
                                    with st.spinner(
                                        "Provo i livelli 1, 2 e 3, certifico "
                                        "l'equivalenza e simulo il migliore..."
                                    ):
                                        aer_attempts, aer_best = (
                                            certified_isometry_candidates(
                                                isometry_matrix=dilation.isometry,
                                                system_qubits=(
                                                    resources.system_qubits
                                                ),
                                                outcome_qubits=(
                                                    resources.outcome_qubits
                                                ),
                                                input_state=(
                                                    test_states[sample_index]
                                                ),
                                                circuit_name=(
                                                    f"PGM_test_{sample_index + 1}"
                                                ),
                                                reference_qpy_payload=qpy_bytes(
                                                    reference_sample_circuit
                                                ),
                                                input_subspace_dimension=(
                                                    resources.padded_system_dimension
                                                ),
                                                timeout_seconds=300,
                                                seed_transpiler=execution_seed,
                                            )
                                        )
                                        aer_counts = None
                                        if aer_best is not None:
                                            executable_circuit = circuit_from_qpy(
                                                aer_best.transpiled_qpy
                                            )
                                            aer_counts = simulate_aer_shots(
                                                executable_circuit,
                                                shots=shots,
                                                seed=execution_seed,
                                                method=aer_method,
                                                # The selected circuit is already
                                                # optimized and certified.
                                                optimization_level=0,
                                                one_qubit_error=one_qubit_error,
                                                two_qubit_error=two_qubit_error,
                                                readout_error=readout_error,
                                            )
                                    st.session_state["aer_quantum_result"] = {
                                        "id": aer_id,
                                        "sample_index": sample_index,
                                        "counts": aer_counts,
                                        "best": aer_best,
                                        "attempts": aer_attempts,
                                    }
                                except Exception as error:
                                    st.error(f"Simulazione Aer non riuscita: {error}")
                            aer_result = st.session_state.get("aer_quantum_result")
                            if aer_result and aer_result["id"] == aer_id:
                                aer_best = aer_result["best"]
                                with st.expander(
                                    "Confronto automatico dei livelli 1, 2 e 3",
                                    expanded=False,
                                ):
                                    st.dataframe(
                                        optimization_attempts_frame(
                                            aer_result["attempts"]
                                        ),
                                        hide_index=True,
                                        width="stretch",
                                    )
                                if aer_best is None:
                                    st.error(
                                        "Nessun candidato Aer ha superato la "
                                        "certificazione: la simulazione non è stata "
                                        "eseguita."
                                    )
                                else:
                                    st.success(
                                        "Aer ha eseguito il migliore circuito "
                                        f"certificato: livello "
                                        f"{aer_best.optimization_level}, "
                                        f"{entangling_gate_count(aer_best):,} gate "
                                        f"entangling, profondità "
                                        f"{aer_best.depth:,}."
                                    )
                                    result_sample = int(aer_result["sample_index"])
                                    render_execution_result(
                                        aer_result["counts"],
                                        classes=classes,
                                        theoretical_outcomes=(
                                            circuit_probabilities[result_sample]
                                        ),
                                        outcome_qubits=resources.outcome_qubits,
                                        actual_class=payload["y_test"][result_sample],
                                        prediction_title="Predizione Aer",
                                    )

                    elif simulator_choice.startswith(("🧩", "🔷", "🟧", "🟪")):
                        local_engine = ""
                        local_backend_name = ""
                        local_engine_ready = True
                        local_qubit_limit = ISOLATED_SYNTHESIS_QUBIT_LIMIT
                        local_shot_limit = 100_000

                        if simulator_choice.startswith("🧩"):
                            local_engine = "basic"
                            local_backend_name = "basic_simulator"
                            st.caption(
                                "Simulatore di riferimento incluso in Qiskit. È ideale "
                                "e più lento di Aer, ma non richiede componenti nativi "
                                "aggiuntivi."
                            )
                            if not modules_available("qiskit"):
                                local_engine_ready = False
                                st.warning("Qiskit non è installato.")

                        elif simulator_choice.startswith("🔷"):
                            local_engine = "ibm_fake"
                            st.caption(
                                "Esecuzione locale con topologia, gate e snapshot di "
                                "rumore di una QPU IBM. Non richiede token e non invia job."
                            )
                            if not modules_available(
                                "qiskit_aer", "qiskit_ibm_runtime"
                            ):
                                local_engine_ready = False
                                st.warning(
                                    "IBM Fake Backend richiede Qiskit Aer e il provider "
                                    "IBM già elencati in requirements.txt."
                                )
                            else:
                                try:
                                    fake_devices = discover_ibm_fake_backends(
                                        min_num_qubits=resources.total_qubits,
                                        max_num_qubits=32,
                                    )
                                except Exception as error:
                                    fake_devices = ()
                                    local_engine_ready = False
                                    st.error(
                                        "Lettura degli snapshot IBM non riuscita: "
                                        f"{error}"
                                    )
                                if fake_devices:
                                    fake_device = st.selectbox(
                                        "Snapshot IBM",
                                        fake_devices,
                                        format_func=device_display_label,
                                        key=f"ibm_fake_device_{synthesis_id}",
                                    )
                                    local_backend_name = fake_device.name
                                    local_qubit_limit = int(
                                        fake_device.qubits
                                        or ISOLATED_SYNTHESIS_QUBIT_LIMIT
                                    )
                                    local_shot_limit = int(
                                        fake_device.max_shots or 100_000
                                    )
                                else:
                                    local_engine_ready = False
                                    st.warning(
                                        "Nessuno snapshot IBM locale compatibile con il "
                                        "numero di qubit del circuito."
                                    )

                        elif simulator_choice.startswith("🟧"):
                            local_engine = "braket_local"
                            st.caption(
                                "Amazon Braket eseguito interamente sul server dell'app: "
                                "non richiede account AWS, S3 o credenziali."
                            )
                            if not python_status["aws_compatible"]:
                                local_engine_ready = False
                                st.warning(
                                    "Amazon Braket Local richiede Python 3.11 o successivo."
                                )
                            elif not modules_available(
                                "braket", "qiskit_braket_provider"
                            ):
                                local_engine_ready = False
                                st.warning(
                                    "Amazon Braket Local non è installato. Usa "
                                    "`Installa_Provider_Quantistici.command` oppure "
                                    "installa `requirements-aws.txt`."
                                )
                            else:
                                braket_local_modes = {
                                    "State vector (braket_sv)": "braket_sv",
                                    "Density matrix (braket_dm)": "braket_dm",
                                }
                                braket_local_label = st.selectbox(
                                    "Metodo Braket Local",
                                    list(braket_local_modes),
                                    key=f"braket_local_method_{synthesis_id}",
                                )
                                local_backend_name = braket_local_modes[
                                    braket_local_label
                                ]
                                # AWS documents 25 qubits for braket_sv and 12
                                # for braket_dm.  The global isolated-synthesis
                                # guard may impose an even smaller limit.
                                local_qubit_limit = (
                                    12
                                    if local_backend_name == "braket_dm"
                                    else 25
                                )

                        else:
                            local_engine = "aqt_offline"
                            st.caption(
                                "Simulatori AQT inclusi nel provider: uno ideale e uno "
                                "con rumore. L'esecuzione è locale e non richiede token."
                            )
                            if not modules_available("qiskit_aqt_provider"):
                                local_engine_ready = False
                                st.warning(
                                    "Provider AQT non installato. Usa "
                                    "`Installa_Provider_Quantistici.command` oppure "
                                    "installa `requirements-aqt.txt`."
                                )
                            else:
                                aqt_offline_modes = {
                                    "Ideale": "offline_simulator_no_noise",
                                    "Rumoroso": "offline_simulator_noise",
                                }
                                aqt_offline_label = st.selectbox(
                                    "Metodo AQT Offline",
                                    list(aqt_offline_modes),
                                    key=f"aqt_offline_method_{synthesis_id}",
                                )
                                local_backend_name = aqt_offline_modes[
                                    aqt_offline_label
                                ]
                                local_qubit_limit = 20
                                local_shot_limit = 2_000

                        local_simulation_id = (
                            f"{synthesis_id}|{local_engine}|{local_backend_name}|"
                            f"{sample_index}|{shots}|{execution_seed}|auto123"
                        )
                        local_too_large = (
                            resources.total_qubits
                            > min(
                                ISOLATED_SYNTHESIS_QUBIT_LIMIT,
                                local_qubit_limit,
                            )
                        )
                        local_too_many_shots = shots > local_shot_limit
                        if local_too_large:
                            st.error(
                                "Simulazione disabilitata: il circuito supera il limite "
                                f"sicuro di {min(ISOLATED_SYNTHESIS_QUBIT_LIMIT, local_qubit_limit)} "
                                "qubit per questo motore."
                            )
                        if local_too_many_shots:
                            st.error(
                                f"Il simulatore selezionato accetta al massimo "
                                f"{local_shot_limit:,} shot."
                            )
                        if st.button(
                            "Ottimizza, certifica ed esegui sul simulatore selezionato",
                            type="primary",
                            disabled=(
                                not local_engine_ready
                                or not local_backend_name
                                or local_too_large
                                or local_too_many_shots
                            ),
                            key=f"extended_local_simulate_{local_simulation_id}",
                        ):
                            try:
                                reference_sample_circuit = build_sample_circuit(
                                    dilation,
                                    test_states[sample_index],
                                    name=(
                                        f"PGM_test_{sample_index + 1}_reference"
                                    ),
                                    optimized_isometry=False,
                                )
                                with st.spinner(
                                    "Ottimizzo in un processo isolato, certifico "
                                    "l'equivalenza ed eseguo il circuito..."
                                ):
                                    local_attempts, local_best = (
                                        certified_isometry_candidates(
                                            isometry_matrix=dilation.isometry,
                                            system_qubits=resources.system_qubits,
                                            outcome_qubits=resources.outcome_qubits,
                                            input_state=test_states[sample_index],
                                            circuit_name=(
                                                f"PGM_test_{sample_index + 1}"
                                            ),
                                            reference_qpy_payload=qpy_bytes(
                                                reference_sample_circuit
                                            ),
                                            input_subspace_dimension=(
                                                resources.padded_system_dimension
                                            ),
                                            timeout_seconds=300,
                                            seed_transpiler=execution_seed,
                                        )
                                    )
                                    local_counts = None
                                    if local_best is not None:
                                        if (
                                            local_engine == "aqt_offline"
                                            and local_best.size is not None
                                            and local_best.size > 2_000
                                        ):
                                            raise ValueError(
                                                "Il circuito supera il limite AQT di "
                                                "2.000 porte."
                                            )
                                        executable_circuit = circuit_from_qpy(
                                            local_best.transpiled_qpy
                                        )
                                        if local_engine == "basic":
                                            local_counts = simulate_basic_shots(
                                                executable_circuit,
                                                shots=shots,
                                                seed=execution_seed,
                                                optimization_level=0,
                                            )
                                        elif local_engine == "ibm_fake":
                                            local_counts = simulate_ibm_fake_shots(
                                                executable_circuit,
                                                backend_name=local_backend_name,
                                                shots=shots,
                                                seed=execution_seed,
                                                optimization_level=1,
                                            )
                                        elif local_engine == "braket_local":
                                            local_counts = simulate_braket_local_shots(
                                                executable_circuit,
                                                backend_name=local_backend_name,
                                                shots=shots,
                                                seed=execution_seed,
                                                optimization_level=1,
                                            )
                                        else:
                                            local_counts = simulate_aqt_offline_shots(
                                                executable_circuit,
                                                backend_name=local_backend_name,
                                                shots=shots,
                                                optimization_level=1,
                                            )
                                st.session_state["extended_local_quantum_result"] = {
                                    "id": local_simulation_id,
                                    "sample_index": sample_index,
                                    "counts": local_counts,
                                    "best": local_best,
                                    "attempts": local_attempts,
                                    "engine": local_engine,
                                    "backend": local_backend_name,
                                }
                            except Exception as error:
                                st.error(
                                    "Simulazione locale non riuscita: "
                                    + safe_error_message(error)
                                )

                        extended_result = st.session_state.get(
                            "extended_local_quantum_result"
                        )
                        if (
                            extended_result
                            and extended_result["id"] == local_simulation_id
                        ):
                            with st.expander(
                                "Confronto automatico dei livelli 1, 2 e 3",
                                expanded=False,
                            ):
                                st.dataframe(
                                    optimization_attempts_frame(
                                        extended_result["attempts"]
                                    ),
                                    hide_index=True,
                                    width="stretch",
                                )
                            local_best = extended_result["best"]
                            if local_best is None:
                                st.error(
                                    "Nessun circuito candidato ha superato la "
                                    "certificazione: la simulazione non è stata eseguita."
                                )
                            else:
                                st.success(
                                    f"{extended_result['backend']} ha eseguito il "
                                    "migliore circuito certificato: livello "
                                    f"{local_best.optimization_level}, "
                                    f"{entangling_gate_count(local_best):,} gate "
                                    f"entangling, profondità {local_best.depth:,}."
                                )
                                result_sample = int(
                                    extended_result["sample_index"]
                                )
                                render_execution_result(
                                    extended_result["counts"],
                                    classes=classes,
                                    theoretical_outcomes=(
                                        circuit_probabilities[result_sample]
                                    ),
                                    outcome_qubits=resources.outcome_qubits,
                                    actual_class=payload["y_test"][result_sample],
                                    prediction_title="Predizione simulatore",
                                )

                    elif simulator_choice.startswith("☁️ AQT"):
                        selected_provider = "aqt_simulator"
                        is_paid_external_resource = True
                    elif simulator_choice.startswith("☁️ IonQ"):
                        selected_provider = "ionq_simulator"
                        is_paid_external_resource = True
                    else:
                        selected_provider = "aws_simulator"
                        is_paid_external_resource = True
                else:
                    provider_labels = {
                        "🇩🇪 LRZ Quantum — Munich Quantum Software Stack": "lrz",
                        "🔵 IBM Quantum — QPU superconduttive": "ibm_qpu",
                        "🟣 IonQ Quantum Cloud — QPU a ioni intrappolati": "ionq_qpu",
                        "🇺🇸 Amazon Braket — QPU gate-based": "aws_qpu",
                    }
                    selected_provider_label = st.selectbox(
                        "Provider hardware",
                        list(provider_labels),
                        key=f"real_provider_{synthesis_id}",
                    )
                    selected_provider = provider_labels[selected_provider_label]
                    is_paid_external_resource = selected_provider != "lrz"

                if selected_provider in {"aws_simulator", "aws_qpu"}:
                    st.markdown("###### Connessione sicura ad Amazon Braket")
                    st.info(
                        "AWS non usa un singolo token Braket. Online puoi usare "
                        "credenziali temporanee della tua sessione AWS; in locale puoi "
                        "anche selezionare un profilo SSO già configurato. Le "
                        "credenziali non vengono salvate dall'app né inserite nei file."
                    )
                    if not python_status["aws_compatible"]:
                        st.error("Amazon Braket richiede Python 3.11 o successivo.")
                    elif not modules_available(
                        "boto3", "braket", "qiskit_braket_provider"
                    ):
                        st.warning(
                            "Componenti AWS non installati. Usa "
                            "`Installa_Provider_Quantistici.command` oppure segui "
                            "`INSTALLAZIONE_PROVIDER_MAC.md`."
                        )
                    else:
                        profiles = aws_available_profiles()
                        configured_aws_access_key = configured_secret(
                            "AWS_ACCESS_KEY_ID"
                        )
                        configured_aws_secret_key = configured_secret(
                            "AWS_SECRET_ACCESS_KEY"
                        )
                        configured_aws_session_token = configured_secret(
                            "AWS_SESSION_TOKEN"
                        )
                        has_deployment_credentials = bool(
                            configured_aws_access_key and configured_aws_secret_key
                        )
                        auth_modes = ["Credenziali temporanee AWS"]
                        if has_deployment_credentials:
                            auth_modes.insert(0, "Credenziali protette del deployment")
                        if profiles:
                            auth_modes.append("Profilo AWS locale / SSO")
                        aws_auth_mode = st.radio(
                            "Autenticazione AWS",
                            auth_modes,
                            horizontal=True,
                            key=f"aws_auth_mode_{synthesis_id}_{selected_provider}",
                        )

                        if aws_auth_mode == "Profilo AWS locale / SSO":
                            aws_profile = st.selectbox(
                                "Profilo AWS locale",
                                profiles,
                                key=f"aws_profile_{synthesis_id}_{selected_provider}",
                            )
                        elif aws_auth_mode == "Credenziali protette del deployment":
                            aws_access_key_id = configured_aws_access_key
                            aws_secret_access_key = configured_aws_secret_key
                            aws_session_token = configured_aws_session_token
                            st.success(
                                "Credenziali AWS protette disponibili sul server."
                            )
                        else:
                            aws_columns = st.columns(2)
                            aws_access_key_id = aws_columns[0].text_input(
                                "AWS Access Key ID",
                                type="password",
                                key=f"aws_access_key_{synthesis_id}_{selected_provider}",
                            ).strip()
                            aws_secret_access_key = aws_columns[1].text_input(
                                "AWS Secret Access Key",
                                type="password",
                                key=f"aws_secret_key_{synthesis_id}_{selected_provider}",
                            ).strip()
                            aws_session_token = st.text_input(
                                "AWS Session Token (facoltativo; necessario per "
                                "credenziali temporanee STS)",
                                type="password",
                                key=f"aws_session_token_{synthesis_id}_{selected_provider}",
                            ).strip()
                            st.caption(
                                "Preferisci credenziali STS temporanee e con permessi "
                                "limitati ad Amazon Braket. Non usare credenziali root."
                            )

                        region_labels = {
                            "🇺🇸 us-east-1": "us-east-1",
                            "🇺🇸 us-west-1": "us-west-1",
                            "🇺🇸 us-west-2": "us-west-2",
                            "🇸🇪 eu-north-1": "eu-north-1",
                            "🇬🇧 eu-west-2": "eu-west-2",
                        }
                        selected_region_label = st.selectbox(
                            "Regione AWS",
                            list(region_labels),
                            key=f"aws_region_{synthesis_id}_{selected_provider}",
                        )
                        aws_region = region_labels[selected_region_label]
                        credential_fingerprint = hashlib.sha256(
                            "\0".join(
                                (
                                    aws_profile or "",
                                    aws_access_key_id,
                                    aws_secret_access_key,
                                    aws_session_token,
                                )
                            ).encode("utf-8")
                        ).hexdigest()[:12]
                        connection_id = (
                            f"{selected_provider}|{aws_auth_mode}|{aws_region}|"
                            f"{credential_fingerprint}"
                        )
                        credentials_ready = bool(
                            aws_profile
                            or (aws_access_key_id and aws_secret_access_key)
                        )
                        if st.button(
                            "Verifica identità e aggiorna dispositivi",
                            disabled=not credentials_ready,
                            key=f"aws_connect_{connection_id}",
                        ):
                            try:
                                with st.spinner(
                                    "Connessione AWS e lettura dei dispositivi..."
                                ):
                                    identity = verify_aws_identity(
                                        aws_profile,
                                        aws_region,
                                        access_key_id=aws_access_key_id or None,
                                        secret_access_key=(
                                            aws_secret_access_key or None
                                        ),
                                        session_token=aws_session_token or None,
                                    )
                                    devices = discover_aws_devices(
                                        aws_profile,
                                        aws_region,
                                        simulators=(
                                            selected_provider == "aws_simulator"
                                        ),
                                        access_key_id=aws_access_key_id or None,
                                        secret_access_key=(
                                            aws_secret_access_key or None
                                        ),
                                        session_token=aws_session_token or None,
                                    )
                                st.session_state["aws_connection"] = {
                                    "id": connection_id,
                                    "identity": identity,
                                    "devices": devices,
                                }
                            except Exception as error:
                                error_message = safe_error_message(
                                    error, aws_secret_access_key
                                )
                                error_message = error_message.replace(
                                    aws_access_key_id, "••••••"
                                ) if aws_access_key_id else error_message
                                error_message = error_message.replace(
                                    aws_session_token, "••••••"
                                ) if aws_session_token else error_message
                                st.error(
                                    f"Connessione AWS non riuscita: {error_message}"
                                )
                        aws_connection = st.session_state.get("aws_connection")
                        if aws_connection and aws_connection["id"] == connection_id:
                            identity = aws_connection["identity"]
                            st.success(
                                "Connessione verificata · account "
                                f"{identity['account']} · {identity['region']}"
                            )
                            devices = aws_connection["devices"]
                            if not devices:
                                st.warning(
                                    "Nessun dispositivo compatibile trovato per "
                                    "queste credenziali e questa selezione."
                                )
                            else:
                                selected_device = st.selectbox(
                                    "Dispositivo Amazon Braket",
                                    devices,
                                    format_func=device_display_label,
                                    key=f"aws_device_{connection_id}",
                                )

                elif selected_provider == "aqt_simulator":
                    st.markdown("###### Connessione ai simulatori AQT Cloud")
                    st.info(
                        "AQT usa un access token del portale AQT. L'app mostra "
                        "soltanto i simulatori autorizzati per l'account; i simulatori "
                        "offline senza token sono disponibili nella voce separata AQT "
                        "Offline."
                    )
                    if not python_status["extended_providers_compatible"]:
                        st.error("AQT richiede Python 3.10 o successivo.")
                    elif not modules_available("qiskit_aqt_provider"):
                        st.warning(
                            "Provider AQT non installato. Usa "
                            "`Installa_Provider_Quantistici.command` oppure installa "
                            "`requirements-aqt.txt`."
                        )
                    else:
                        default_aqt_token = configured_secret("AQT_TOKEN")
                        aqt_token = st.text_input(
                            "AQT access token",
                            value=default_aqt_token,
                            type="password",
                            key=f"aqt_token_{synthesis_id}",
                        )
                        aqt_fingerprint = hashlib.sha256(
                            aqt_token.encode("utf-8")
                        ).hexdigest()[:12]
                        aqt_connection_id = (
                            f"{synthesis_id}|aqt_simulator|{aqt_fingerprint}"
                        )
                        if st.button(
                            "Connetti e aggiorna simulatori AQT",
                            disabled=not bool(aqt_token),
                            key=f"aqt_connect_{aqt_connection_id}",
                        ):
                            try:
                                with st.spinner(
                                    "Lettura dei simulatori AQT autorizzati..."
                                ):
                                    aqt_devices = discover_aqt_simulators(
                                        aqt_token,
                                        offline=False,
                                    )
                                st.session_state["aqt_devices"] = {
                                    "id": aqt_connection_id,
                                    "devices": aqt_devices,
                                }
                            except Exception as error:
                                st.error(
                                    "Connessione AQT non riuscita: "
                                    + safe_error_message(error, aqt_token)
                                )
                        saved_aqt = st.session_state.get("aqt_devices")
                        if saved_aqt and saved_aqt["id"] == aqt_connection_id:
                            aqt_devices = saved_aqt["devices"]
                            if not aqt_devices:
                                st.warning(
                                    "Nessun simulatore AQT Cloud è autorizzato per "
                                    "questo account. AQT Offline resta disponibile."
                                )
                            else:
                                selected_device = st.selectbox(
                                    "Simulatore AQT Cloud",
                                    aqt_devices,
                                    format_func=device_display_label,
                                    key=f"aqt_device_{aqt_connection_id}",
                                )

                elif selected_provider == "lrz":
                    st.markdown("###### Connessione a LRZ Quantum tramite MQSS")
                    st.info(
                        "L'app interroga prima le risorse online e ignora in modo sicuro "
                        "i metadati mancanti dei target offline. In questo modo un backend "
                        "non disponibile, come MUNIQC-Atoms20, non impedisce di mostrare "
                        "una risorsa operativa come EQE1."
                    )
                    if not python_status["lrz_compatible"]:
                        st.error("LRZ MQSS richiede Python >=3.9 e <3.14.")
                    elif not modules_available("mqss"):
                        st.warning(
                            "Adapter LRZ non installato. Usa "
                            "`Installa_Provider_Quantistici.command` oppure esegui "
                            "`python -m pip install -r requirements-lrz.txt`."
                        )
                    else:
                        default_lrz_token = configured_secret("LRZ_MQSS_TOKEN")
                        lrz_token = st.text_input(
                            "Token Munich Quantum Portal",
                            value=default_lrz_token,
                            type="password",
                            key=f"lrz_token_{synthesis_id}",
                            help=(
                                "Per il riempimento automatico usa la variabile "
                                "LRZ_MQSS_TOKEN o .streamlit/secrets.toml."
                            ),
                        )
                        lrz_options = st.columns(2)
                        lrz_online_only = lrz_options[0].checkbox(
                            "Mostra soltanto risorse online",
                            value=True,
                            key=f"lrz_online_only_{synthesis_id}",
                        )
                        lrz_queue_offline = lrz_options[1].checkbox(
                            "Accoda se la risorsa diventa offline",
                            value=True,
                            key=f"lrz_queue_{synthesis_id}",
                        )
                        lrz_connection_id = (
                            f"{synthesis_id}|online={int(lrz_online_only)}"
                        )
                        if st.button(
                            "Connetti e aggiorna risorse LRZ",
                            disabled=not bool(lrz_token),
                            key=f"lrz_connect_{lrz_connection_id}",
                        ):
                            try:
                                with st.spinner("Lettura delle risorse autorizzate..."):
                                    lrz_devices = discover_lrz_backends(
                                        lrz_token,
                                        online_only=lrz_online_only,
                                    )
                                st.session_state["lrz_devices"] = {
                                    "id": lrz_connection_id,
                                    "devices": lrz_devices,
                                }
                            except Exception as error:
                                st.error(
                                    "Connessione LRZ non riuscita: "
                                    + safe_error_message(error, lrz_token)
                                )
                        saved_lrz = st.session_state.get("lrz_devices")
                        if saved_lrz and saved_lrz["id"] == lrz_connection_id:
                            lrz_devices = saved_lrz["devices"]
                            if not lrz_devices:
                                st.warning(
                                    "Il token è valido ma non restituisce risorse "
                                    "compatibili con il filtro selezionato."
                                )
                            else:
                                st.success(
                                    f"Trovate {len(lrz_devices)} risorse autorizzate."
                                )
                                selected_device = st.selectbox(
                                    "Risorsa LRZ/MQSS",
                                    lrz_devices,
                                    format_func=device_display_label,
                                    key=f"lrz_device_{lrz_connection_id}",
                                )
                                if selected_device.notes:
                                    st.caption(selected_device.notes)
                        st.caption(
                            "LRZ Qaptiva usa un flusso VPN/SSH/HPC separato e non è "
                            "presentato come falso backend MQSS."
                        )

                elif selected_provider in {"ionq_simulator", "ionq_qpu"}:
                    st.markdown("###### Connessione a IonQ Quantum Cloud")
                    st.info(
                        "IonQ usa una API key. Il provider espone sia il simulatore "
                        "cloud, con modelli di rumore opzionali, sia le QPU disponibili "
                        "per il tuo account."
                    )
                    if not python_status["extended_providers_compatible"]:
                        st.error("IonQ è consigliato con Python 3.10 o successivo.")
                    elif not modules_available("qiskit_ionq"):
                        st.warning(
                            "Provider IonQ non installato. Usa "
                            "`Installa_Provider_Quantistici.command` oppure installa "
                            "`requirements-ionq.txt`."
                        )
                    else:
                        default_ionq_token = configured_secret(
                            "QISKIT_IONQ_API_TOKEN",
                            "IONQ_API_KEY",
                            "IONQ_API_TOKEN",
                        )
                        ionq_token = st.text_input(
                            "IonQ API key",
                            value=default_ionq_token,
                            type="password",
                            key=f"ionq_token_{synthesis_id}_{selected_provider}",
                        )
                        if selected_provider == "ionq_simulator":
                            ionq_noise_labels = {
                                "Ideale": "ideal",
                                "Forte 1": "forte-1",
                                "Forte Enterprise 1": "forte-enterprise-1",
                            }
                            ionq_noise_label = st.selectbox(
                                "Modello di rumore IonQ",
                                list(ionq_noise_labels),
                                key=f"ionq_noise_{synthesis_id}",
                            )
                            ionq_noise_model = ionq_noise_labels[ionq_noise_label]
                            if ionq_noise_model == "ideal":
                                st.caption(
                                    "Il simulatore ideale IonQ restituisce probabilità. "
                                    "L'app le converte nel numero di shot virtuali scelto "
                                    "sopra, senza alterare la distribuzione restituita."
                                )
                        ionq_connection_id = (
                            f"{synthesis_id}|{selected_provider}"
                        )
                        if st.button(
                            "Connetti e aggiorna risorse IonQ",
                            disabled=not bool(ionq_token),
                            key=f"ionq_connect_{ionq_connection_id}",
                        ):
                            try:
                                with st.spinner("Lettura dei backend IonQ..."):
                                    ionq_devices = discover_ionq_devices(
                                        ionq_token,
                                        simulators=(
                                            selected_provider == "ionq_simulator"
                                        ),
                                    )
                                st.session_state["ionq_devices"] = {
                                    "id": ionq_connection_id,
                                    "devices": ionq_devices,
                                }
                            except Exception as error:
                                st.error(
                                    "Connessione IonQ non riuscita: "
                                    + safe_error_message(error, ionq_token)
                                )
                        saved_ionq = st.session_state.get("ionq_devices")
                        if saved_ionq and saved_ionq["id"] == ionq_connection_id:
                            ionq_devices = saved_ionq["devices"]
                            if not ionq_devices:
                                st.warning(
                                    "Nessun backend IonQ compatibile è stato restituito."
                                )
                            else:
                                selected_device = st.selectbox(
                                    "Backend IonQ",
                                    ionq_devices,
                                    format_func=device_display_label,
                                    key=f"ionq_device_{ionq_connection_id}",
                                )

                elif selected_provider == "ibm_qpu":
                    st.markdown("###### Connessione a IBM Quantum")
                    st.info(
                        "Puoi incollare token e CRN dell'istanza oppure lasciare il token "
                        "vuoto se hai già salvato localmente un account IBM Quantum. "
                        "L'elenco mostra soltanto QPU operative con qubit sufficienti."
                    )
                    if not python_status["extended_providers_compatible"]:
                        st.error("IBM Quantum è consigliato con Python 3.10 o successivo.")
                    elif not modules_available("qiskit_ibm_runtime"):
                        st.warning(
                            "Provider IBM non installato. Usa "
                            "`Installa_Provider_Quantistici.command` oppure installa "
                            "`requirements-ibm.txt`."
                        )
                    else:
                        default_ibm_token = configured_secret(
                            "QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN"
                        )
                        default_ibm_instance = configured_secret(
                            "QISKIT_IBM_INSTANCE", "IBM_QUANTUM_INSTANCE"
                        )
                        ibm_columns = st.columns(2)
                        ibm_token = ibm_columns[0].text_input(
                            "IBM Quantum API key (facoltativa se già salvata)",
                            value=default_ibm_token,
                            type="password",
                            key=f"ibm_token_{synthesis_id}",
                        )
                        ibm_instance = ibm_columns[1].text_input(
                            "CRN istanza IBM (consigliato)",
                            value=default_ibm_instance,
                            key=f"ibm_instance_{synthesis_id}",
                        )
                        ibm_connection_id = (
                            f"{synthesis_id}|ibm|{ibm_instance or 'default'}"
                        )
                        if st.button(
                            "Connetti e aggiorna QPU IBM",
                            key=f"ibm_connect_{ibm_connection_id}",
                        ):
                            try:
                                with st.spinner("Lettura delle QPU IBM operative..."):
                                    ibm_devices = discover_ibm_devices(
                                        ibm_token or None,
                                        ibm_instance or None,
                                        simulators=False,
                                        min_num_qubits=resources.total_qubits,
                                    )
                                st.session_state["ibm_devices"] = {
                                    "id": ibm_connection_id,
                                    "devices": ibm_devices,
                                }
                            except Exception as error:
                                st.error(
                                    "Connessione IBM non riuscita: "
                                    + safe_error_message(error, ibm_token)
                                )
                        saved_ibm = st.session_state.get("ibm_devices")
                        if saved_ibm and saved_ibm["id"] == ibm_connection_id:
                            ibm_devices = saved_ibm["devices"]
                            if not ibm_devices:
                                st.warning(
                                    "Nessuna QPU IBM operativa con abbastanza qubit è "
                                    "disponibile per questo account."
                                )
                            else:
                                selected_device = st.selectbox(
                                    "QPU IBM Quantum",
                                    ibm_devices,
                                    format_func=device_display_label,
                                    key=f"ibm_device_{ibm_connection_id}",
                                )

                if selected_device is not None:
                    if selected_device.notes and selected_provider != "lrz":
                        st.caption(selected_device.notes)
                    st.markdown("###### Preflight del circuito eseguibile")
                    device_has_capacity = (
                        selected_device.qubits is None
                        or resources.total_qubits <= selected_device.qubits
                    )
                    if not device_has_capacity:
                        st.error(
                            f"Il circuito richiede {resources.total_qubits} qubit, ma "
                            f"la risorsa dichiara {selected_device.qubits} qubit."
                        )
                    if (
                        selected_device.max_shots is not None
                        and shots > selected_device.max_shots
                    ):
                        st.warning(
                            f"La risorsa dichiara un massimo di "
                            f"{selected_device.max_shots:,} shot; riduci il valore."
                        )
                    if selected_provider == "lrz" and shots > 200:
                        st.warning(
                            "Alcune risorse LRZ, in particolare AQT, documentano limiti "
                            "di shot più bassi. Il limite effettivo dipende dal backend."
                        )
                    st.write(
                        "Il preflight aggiunge la preparazione dello stato test, prova "
                        "automaticamente i livelli 1, 2 e 3 in processi isolati e "
                        "sceglie il migliore tra quelli equivalenti a U_PGM. Il "
                        "provider eseguirà poi la necessaria conversione nella base "
                        "nativa del dispositivo."
                    )
                    preflight_id = (
                        f"{synthesis_id}|{selected_provider}|"
                        f"{selected_device.identifier}|{sample_index}|"
                        f"auto123|seed={execution_seed}"
                    )
                    if resources.total_qubits > ISOLATED_SYNTHESIS_QUBIT_LIMIT:
                        st.error(
                            "Esecuzione esterna disabilitata: questo circuito supera "
                            f"{ISOLATED_SYNTHESIS_QUBIT_LIMIT} qubit e la sintesi sicura "
                            "non è praticabile sul computer locale."
                        )
                    else:
                        if st.button(
                            "Ottimizza, certifica e prepara il circuito",
                            disabled=not device_has_capacity,
                            key=f"preflight_{preflight_id}",
                        ):
                            try:
                                reference_sample_circuit = build_sample_circuit(
                                    dilation,
                                    test_states[sample_index],
                                    name=f"PGM_test_{sample_index + 1}_reference",
                                    optimized_isometry=False,
                                )
                                with st.spinner(
                                    "Provo i livelli 1, 2 e 3 sul circuito completo "
                                    "e certifico ogni candidato..."
                                ):
                                    preflight_attempts, preflight_report = (
                                        certified_isometry_candidates(
                                            isometry_matrix=dilation.isometry,
                                            system_qubits=(
                                                resources.system_qubits
                                            ),
                                            outcome_qubits=(
                                                resources.outcome_qubits
                                            ),
                                            input_state=(
                                                test_states[sample_index]
                                            ),
                                            circuit_name=(
                                                f"PGM_test_{sample_index + 1}"
                                            ),
                                            reference_qpy_payload=qpy_bytes(
                                                reference_sample_circuit
                                            ),
                                            input_subspace_dimension=(
                                                resources.padded_system_dimension
                                            ),
                                            timeout_seconds=300,
                                            seed_transpiler=execution_seed,
                                        )
                                    )
                                st.session_state["hardware_preflight"] = {
                                    "id": preflight_id,
                                    "sample_index": sample_index,
                                    "report": preflight_report,
                                    "attempts": preflight_attempts,
                                }
                            except Exception as error:
                                st.error(f"Preflight non riuscito: {error}")

                        saved_preflight = st.session_state.get("hardware_preflight")
                        if (
                            saved_preflight
                            and saved_preflight["id"] == preflight_id
                        ):
                            preflight_report = saved_preflight["report"]
                            preflight_attempts = saved_preflight["attempts"]
                            with st.expander(
                                "Confronto automatico dei livelli 1, 2 e 3",
                                expanded=False,
                            ):
                                st.dataframe(
                                    optimization_attempts_frame(
                                        preflight_attempts
                                    ),
                                    hide_index=True,
                                    width="stretch",
                                )
                            if preflight_report is None:
                                st.error(
                                    "Nessuno dei tre candidati ha superato la "
                                    "certificazione. Il job non può essere inviato."
                                )
                                with st.expander(
                                    "Messaggi diagnostici", expanded=False
                                ):
                                    for attempt in preflight_attempts:
                                        st.write(
                                            f"Livello {attempt.optimization_level}: "
                                            f"{attempt.message}"
                                        )
                            elif preflight_report.status not in {
                                "success",
                                "partial",
                            }:
                                st.error(preflight_report.message)
                            else:
                                if preflight_report.status == "partial":
                                    st.warning(preflight_report.message)
                                preflight_columns = st.columns(4)
                                preflight_columns[0].metric(
                                    "Livello scelto",
                                    preflight_report.optimization_level,
                                )
                                preflight_columns[1].metric(
                                    "Gate entangling",
                                    f"{entangling_gate_count(preflight_report):,}",
                                )
                                preflight_columns[2].metric(
                                    "Profondità", f"{preflight_report.depth:,}"
                                )
                                preflight_columns[3].metric(
                                    "Porte totali", f"{preflight_report.size:,}"
                                )
                                st.caption(
                                    f"Circuito su {resources.total_qubits} qubit. "
                                    "Criterio di scelta: gate entangling, profondità, "
                                    "poi porte totali."
                                )
                                preflight_gate_table = pd.DataFrame(
                                    [
                                        {"Gate": name, "Numero": count}
                                        for name, count in sorted(
                                            preflight_report.gate_counts.items(),
                                            key=lambda item: (-item[1], item[0]),
                                        )
                                    ]
                                )
                                st.dataframe(
                                    preflight_gate_table,
                                    hide_index=True,
                                    width="stretch",
                                )
                                st.caption(
                                    "I conteggi sono nella base generica rz/sx/x/cx; "
                                    "la compilazione nativa del provider può modificarli."
                                )
                                if preflight_report.equivalence_certified is True:
                                    st.success(
                                        "Migliore preflight certificato equivalente a "
                                        f"U_PGM: livello "
                                        f"{preflight_report.optimization_level} · "
                                        f"errore massimo "
                                        f"{preflight_report.equivalence_error:.3e}."
                                    )
                                else:
                                    st.error(
                                        "Il preflight non ha superato la certificazione "
                                        "di equivalenza: l'invio è disabilitato."
                                    )
                                if preflight_report.diagram_text is not None:
                                    with st.expander(
                                        "Mostra il circuito hardware completo",
                                        expanded=False,
                                    ):
                                        st.text_area(
                                            "Circuito con state preparation",
                                            preflight_report.diagram_text,
                                            height=650,
                                            disabled=True,
                                            key=f"hardware_diagram_{preflight_id}",
                                        )
                                elif preflight_report.size is not None:
                                    st.warning(
                                        "Circuito troppo grande per il disegno integrale; "
                                        "il conteggio delle porte resta esatto."
                                    )

                                if (
                                    selected_provider in {"lrz", "aqt_simulator"}
                                    and preflight_report.size is not None
                                    and preflight_report.size > 2_000
                                ):
                                    st.warning(
                                        "Il circuito supera 2.000 porte, limite noto per "
                                        "alcune risorse AQT/LRZ; altri backend possono "
                                        "avere limiti differenti."
                                    )

                                confirmation_messages = {
                                    "lrz": (
                                        "Confermo di voler inviare il job alla risorsa "
                                        "LRZ selezionata usando la mia allocazione."
                                    ),
                                    "aws_simulator": (
                                        "Confermo di voler inviare un task AWS che può "
                                        "usare quota o generare costi."
                                    ),
                                    "aws_qpu": (
                                        "Confermo di voler inviare un task AWS che può "
                                        "generare costi sul mio account."
                                    ),
                                    "ionq_simulator": (
                                        "Confermo di voler inviare un job IonQ che può "
                                        "usare quota o generare costi."
                                    ),
                                    "aqt_simulator": (
                                        "Confermo di voler inviare il job al simulatore "
                                        "AQT Cloud selezionato e di accettarne quota o "
                                        "costi."
                                    ),
                                    "ionq_qpu": (
                                        "Confermo di voler inviare il job alla QPU IonQ "
                                        "selezionata e di accettarne quota o costi."
                                    ),
                                    "ibm_qpu": (
                                        "Confermo di voler inviare il job alla QPU IBM "
                                        "selezionata usando la mia istanza."
                                    ),
                                }
                                external_confirmation = st.checkbox(
                                    confirmation_messages[selected_provider],
                                    key=f"confirm_external_{preflight_id}",
                                )
                                if is_paid_external_resource:
                                    st.warning(
                                        "Prima dell'invio verifica quota, piano e prezzi "
                                        "nel portale del provider: l'app non usa costi "
                                        "hard-coded."
                                    )
                                if st.button(
                                    "Invia job quantistico",
                                    type="primary",
                                    disabled=(
                                        not external_confirmation
                                        or preflight_report.transpiled_qpy is None
                                        or preflight_report.equivalence_certified
                                        is not True
                                    ),
                                    key=f"submit_external_{preflight_id}",
                                ):
                                    secret_for_error = (
                                        lrz_token
                                        or ionq_token
                                        or aqt_token
                                        or ibm_token
                                        or aws_secret_access_key
                                        or None
                                    )
                                    try:
                                        executable_circuit = circuit_from_qpy(
                                            preflight_report.transpiled_qpy
                                        )
                                        native_optimization_level = int(
                                            preflight_report.optimization_level or 1
                                        )
                                        with st.spinner(
                                            "Invio del job; il risultato verrà letto "
                                            "solo su richiesta..."
                                        ):
                                            if selected_provider == "lrz":
                                                if not lrz_token:
                                                    raise ValueError(
                                                        "Reinserire il token LRZ."
                                                    )
                                                job, job_id = submit_lrz_job(
                                                    executable_circuit,
                                                    token=lrz_token,
                                                    backend_name=selected_device.name,
                                                    shots=shots,
                                                    queued=lrz_queue_offline,
                                                )
                                            elif selected_provider == "aqt_simulator":
                                                if not aqt_token:
                                                    raise ValueError(
                                                        "Reinserire il token AQT."
                                                    )
                                                job, job_id = submit_aqt_job(
                                                    executable_circuit,
                                                    token=aqt_token,
                                                    backend_name=selected_device.name,
                                                    shots=shots,
                                                    optimization_level=(
                                                        native_optimization_level
                                                    ),
                                                    workspace=(
                                                        selected_device.region
                                                        if selected_device.region
                                                        != "AQT Cloud"
                                                        else None
                                                    ),
                                                )
                                            elif selected_provider in {
                                                "ionq_simulator",
                                                "ionq_qpu",
                                            }:
                                                if not ionq_token:
                                                    raise ValueError(
                                                        "Reinserire la API key IonQ."
                                                    )
                                                job, job_id = submit_ionq_job(
                                                    executable_circuit,
                                                    token=ionq_token,
                                                    backend_name=selected_device.name,
                                                    shots=shots,
                                                    optimization_level=(
                                                        native_optimization_level
                                                    ),
                                                    noise_model=(
                                                        ionq_noise_model
                                                        if selected_provider
                                                        == "ionq_simulator"
                                                        else None
                                                    ),
                                                )
                                            elif selected_provider == "ibm_qpu":
                                                job, job_id = submit_ibm_job(
                                                    executable_circuit,
                                                    token=ibm_token or None,
                                                    instance=ibm_instance or None,
                                                    backend_name=selected_device.name,
                                                    shots=shots,
                                                    optimization_level=(
                                                        native_optimization_level
                                                    ),
                                                )
                                            else:
                                                if not aws_region or not (
                                                    aws_profile
                                                    or (
                                                        aws_access_key_id
                                                        and aws_secret_access_key
                                                    )
                                                ):
                                                    raise ValueError(
                                                        "Credenziali o regione AWS "
                                                        "mancanti."
                                                    )
                                                job, job_id = submit_aws_job(
                                                    executable_circuit,
                                                    profile=aws_profile,
                                                    region=aws_region,
                                                    backend_name=selected_device.name,
                                                    backend_identifier=(
                                                        selected_device.identifier
                                                    ),
                                                    shots=shots,
                                                    access_key_id=(
                                                        aws_access_key_id or None
                                                    ),
                                                    secret_access_key=(
                                                        aws_secret_access_key or None
                                                    ),
                                                    session_token=(
                                                        aws_session_token or None
                                                    ),
                                                )
                                        st.session_state["external_quantum_job"] = {
                                            "preflight_id": preflight_id,
                                            "provider": selected_device.provider,
                                            "device": selected_device.name,
                                            "job_id": job_id,
                                            "job": job,
                                            "sample_index": sample_index,
                                            "shots": shots,
                                            "status": "SUBMITTED",
                                            "counts": None,
                                        }
                                        st.success(f"Job inviato. ID: {job_id}")
                                    except Exception as error:
                                        st.error(
                                            "Invio non riuscito: "
                                            + safe_error_message(
                                                error, secret_for_error
                                            )
                                        )

                    external_job = st.session_state.get("external_quantum_job")
                    if (
                        external_job
                        and external_job["preflight_id"] == preflight_id
                    ):
                        st.markdown("###### Job remoto")
                        st.write(
                            {
                                "provider": external_job["provider"],
                                "dispositivo": external_job["device"],
                                "job_id": external_job["job_id"],
                                "stato": external_job["status"],
                                "shots": external_job["shots"],
                            }
                        )
                        job_actions = st.columns(3)
                        if job_actions[0].button(
                            "Aggiorna stato", key=f"refresh_{preflight_id}"
                        ):
                            try:
                                external_job["status"] = generic_job_status(
                                    external_job["job"]
                                )
                            except Exception as error:
                                st.error(f"Stato non disponibile: {error}")
                        completed_status = str(external_job["status"]).upper()
                        can_read_result = any(
                            word in completed_status
                            for word in ("DONE", "COMPLETED", "SUCCESS")
                        )
                        if job_actions[1].button(
                            "Scarica risultato",
                            disabled=not can_read_result,
                            key=f"result_{preflight_id}",
                        ):
                            try:
                                external_job["counts"] = generic_job_counts(
                                    external_job["job"],
                                    shots=int(external_job["shots"]),
                                )
                            except Exception as error:
                                st.error(f"Risultato non disponibile: {error}")
                        cancel_method = getattr(external_job["job"], "cancel", None)
                        if job_actions[2].button(
                            "Annulla job",
                            disabled=not callable(cancel_method) or can_read_result,
                            key=f"cancel_{preflight_id}",
                        ):
                            try:
                                cancel_method()
                                external_job["status"] = "CANCEL_REQUESTED"
                            except Exception as error:
                                st.error(f"Annullamento non riuscito: {error}")

                        if external_job["counts"]:
                            result_sample = int(external_job["sample_index"])
                            render_execution_result(
                                external_job["counts"],
                                classes=classes,
                                theoretical_outcomes=(
                                    circuit_probabilities[result_sample]
                                ),
                                outcome_qubits=resources.outcome_qubits,
                                actual_class=payload["y_test"][result_sample],
                                prediction_title="Predizione provider",
                            )


            with export_tab:
                export_columns = st.columns(3)
                export_columns[0].download_button(
                    "Scarica diagramma SVG",
                    data=circuit_drawing.encode("utf-8"),
                    file_name=localized_filename(
                        f"circuito_PGM_{selected_key}_c{copies}.svg",
                        f"PGM_circuit_{selected_key}_c{copies}.svg",
                    ),
                    mime="image/svg+xml",
                    width="stretch",
                )
                export_columns[1].download_button(
                    "Scarica circuito QPY",
                    data=qpy_bytes(circuit),
                    file_name=localized_filename(
                        f"circuito_PGM_{selected_key}_c{copies}.qpy",
                        f"PGM_circuit_{selected_key}_c{copies}.qpy",
                    ),
                    mime="application/octet-stream",
                    width="stretch",
                )
                export_columns[2].download_button(
                    "Scarica matrici NPZ",
                    data=dilation_npz_bytes(
                        measurement,
                        dilation,
                        raw_feature_count=payload["raw_d"],
                        base_feature_count=payload["base_d"],
                        encoded_feature_count=payload["d"],
                        encoding=payload["encoding"],
                        rescaling_factor=payload["rescaling_factor"],
                        copies=copies,
                    ),
                    file_name=localized_filename(
                        f"dilatazione_PGM_{selected_key}_c{copies}.npz",
                        f"PGM_dilation_{selected_key}_c{copies}.npz",
                    ),
                    mime="application/octet-stream",
                    width="stretch",
                )
                st.caption(
                    "QPY conserva il circuito Qiskit; NPZ contiene U_PGM, gli effetti "
                    "F_j, l'ordine delle classi e i metadati della codifica."
                )
        else:
            dilation = symbolic_dilation(resources)
            circuit = build_qiskit_circuit(dilation)
            circuit_drawing = translate_text(
                circuit_svg(dilation), current_language()
            )
            actual_feature_suggestion = largest_manual_feature_count_for_circuit(
                copies=copies,
                class_count=int(payload["class_count"]),
                qubit_limit=EXACT_CIRCUIT_QUBIT_LIMIT,
                maximum_base_features=int(payload["raw_d"]),
            )
            st.warning(
                "Per questa configurazione la dilatazione avrebbe una matrice densa "
                f"{resources.unitary_dimension:,} x {resources.unitary_dimension:,} "
                f"({human_bytes(resources.unitary_bytes)} in complex128). Per proteggere "
                "la memoria, l'app mostra l'architettura dimensionata ma non materializza "
                "U_PGM. I risultati classici, la matrice di confusione e il dettaglio "
                "dei campioni restano disponibili sopra."
            )
            if actual_feature_suggestion > 0:
                st.info(
                    "Per attivare il circuito esatto con le copie e le classi "
                    "correnti, abilita volontariamente la PCA nella sezione 1 e "
                    "imposta al massimo "
                    f"{actual_feature_suggestion} componenti. In alternativa riduci "
                    "il numero di copie. Nessuna feature viene ridotta "
                    "automaticamente."
                )
            else:
                st.info(
                    "Con il numero corrente di copie e classi, neppure una PCA a una "
                    "componente rientra nel limite sicuro. Riduci il numero di copie "
                    "oppure usa un dataset con meno classi; la classificazione PGM "
                    "classica resta comunque completa."
                )
            st.markdown(circuit_drawing, unsafe_allow_html=True)
            st.caption(
                "Schema logico dimensionato; U_PGM_symbolic indica la matrice non "
                "materializzata per questa configurazione."
            )
            st.markdown("**Corrispondenza tra esiti e classi**")
            st.dataframe(outcome_table, hide_index=True, width="stretch")
            with st.container(
                border=True,
                key=(
                    f"quantum_unavailable_{selected_key}_{copies}_"
                    f"{int(random_seed)}"
                ),
            ):
                st.markdown("#### Strumenti quantistici non attivi per questa configurazione")
                st.write(
                    "Le funzioni restano visibili, ma richiedono la matrice esatta "
                    "U_PGM. Il blocco evita che un calcolo denso eccessivo interrompa "
                    "l'intera applicazione."
                )
                unavailable_columns = st.columns(3)
                unavailable_columns[0].button(
                    "Sintetizza il circuito",
                    disabled=True,
                    width="stretch",
                    key=f"symbolic_synthesis_disabled_{selected_key}_{copies}",
                )
                unavailable_columns[1].button(
                    "Ottimizza e certifica",
                    disabled=True,
                    width="stretch",
                    key=f"symbolic_optimizer_disabled_{selected_key}_{copies}",
                )
                unavailable_columns[2].button(
                    "Configura simulatore o QPU",
                    disabled=True,
                    width="stretch",
                    key=f"symbolic_execution_disabled_{selected_key}_{copies}",
                )
    except ImportError as error:
        st.error(
            "Manca una dipendenza necessaria per disegnare il circuito. Ferma l'app "
            "con Control+C, esegui `python -m pip install -r requirements.txt` nella "
            "cartella del progetto e riavviala. Dettaglio: "
            f"{error}"
        )
    except Exception as error:
        st.exception(error)

    predictions = prediction_frame(results, y_test)
    st.download_button(
        "Scarica le predizioni (CSV)",
        data=spreadsheet_safe_csv_bytes(
            localize_dataframe(predictions, current_language())
        ),
        file_name=localized_filename(
            f"predizioni_{selected_key}_c{copies}_seed{int(random_seed)}.csv",
            f"predictions_{selected_key}_c{copies}_seed{int(random_seed)}.csv",
        ),
        mime="text/csv",
    )
elif saved_run:
    st.info("Le impostazioni sono cambiate: premi il pulsante per calcolare il nuovo caso.")

if not current_run_available:
    render_section_banner(
        "Sezione 5 · laboratorio quantistico",
        "5. Circuito quantistico della PGM",
        "Circuito logico, classificazione, validazione, optimizer, simulatori, QPU ed export in un unico spazio protetto.",
    )
    render_quantum_tool_strip()
    with st.container(border=True, key="quantum_laboratory_placeholder"):
        st.info(
            "Esegui prima la valutazione PGM nella sezione immediatamente precedente. "
            "Il laboratorio si attiverà senza avviare automaticamente sintesi, "
            "optimizer, simulatori o QPU."
        )
        placeholder_columns = st.columns(3)
        placeholder_columns[0].button(
            "Visualizza il circuito logico",
            disabled=True,
            width="stretch",
            key="placeholder_logical_circuit_disabled",
        )
        placeholder_columns[1].button(
            "Ottimizza e certifica equivalenza",
            disabled=True,
            width="stretch",
            key="placeholder_optimizer_disabled",
        )
        placeholder_columns[2].button(
            "Configura simulatore o QPU",
            disabled=True,
            width="stretch",
            key="placeholder_execution_disabled",
        )

st.divider()
st.caption(
    "Nota: r-PGM è il nome breve usato qui per il reduced c-PGM (Rc-PGM) del paper. "
    "Per c=1 la mappa ridotta coincide con la mappa originale."
)

st.divider()
render_section_banner(
    "Documentazione",
    "Guida illustrata all'uso",
    "Manuale bilingue aggiornato con flusso operativo, interpretazione dei risultati, circuito e provider.",
)
st.caption(
    "Scarica la guida completa nella lingua attiva: contiene il percorso passo passo, "
    "illustrazioni dell'interfaccia, interpretazione statistica, circuito quantistico, "
    "provider, privacy e risoluzione dei problemi."
)
try:
    guide_manifest = json.loads(GUIDE_MANIFEST_PATH.read_text(encoding="utf-8"))
    guide_version = str(guide_manifest["app_version"])
    if guide_version != APP_VERSION:
        raise RuntimeError(
            "La guida deve essere aggiornata alla versione corrente dell'app."
        )
    guide_language = current_language()
    guide_file_name = str(guide_manifest["files"][guide_language])
    guide_path = (GUIDE_DIRECTORY / guide_file_name).resolve()
    if guide_path.parent != GUIDE_DIRECTORY.resolve() or not guide_path.is_file():
        raise FileNotFoundError("File della guida non disponibile.")
    guide_columns = st.columns([1, 2, 1])
    with guide_columns[1]:
        st.download_button(
            "📘 Scarica la guida PDF completa",
            data=read_public_guide_asset(str(guide_path), guide_version),
            file_name=guide_file_name,
            mime="application/pdf",
            width="stretch",
        )
except Exception as error:
    st.warning(
        "La guida PDF è temporaneamente in aggiornamento. Dettaglio: "
        + safe_error_message(error)
    )
