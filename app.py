"""Streamlit interface for comparing the three equivalent PGM classifiers."""

from __future__ import annotations

import hashlib
import importlib.util
from math import ceil
import os

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
import streamlit as st
from sklearn.metrics import accuracy_score

from complexity import (
    human_bytes,
    implementation_feasibility,
    paper_complexities,
    representation_dimensions,
    scientific_integer,
)
from data_catalog import DATASETS, catalog_frame, get_dataset_spec
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
    discover_aws_devices,
    discover_ibm_devices,
    discover_ionq_devices,
    discover_lrz_backends,
    generic_job_counts,
    generic_job_status,
    hardware_python_status,
    simulate_aer_shots,
    simulate_ideal_shots,
    submit_aws_job,
    submit_ibm_job,
    submit_ionq_job,
    submit_lrz_job,
    verify_aws_identity,
)
from pgm_core import MethodResult, stable_predictions, symmetric_feature_map
from quantum_pgm import (
    build_naimark_dilation,
    build_qiskit_circuit,
    build_reduced_pgm_measurement,
    circuit_svg,
    dilation_npz_bytes,
    generic_unitary_cnot_upper_bound,
    isolated_transpile_qpy,
    outcome_probabilities,
    qpy_bytes,
    resources_for_dataset,
    symbolic_dilation,
)


st.set_page_config(
    page_title="PGM Lab - c-PGM, k-PGM e r-PGM",
    page_icon="⚛️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg, #f7f9ff 0%, #ffffff 28rem);}
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      [data-testid="stMetric"] {
        background: rgba(255,255,255,.88); border: 1px solid #e1e6f0;
        border-radius: 14px; padding: .75rem 1rem; box-shadow: 0 5px 18px rgba(33,43,74,.05);
      }
      [data-baseweb="tab-list"] {gap: .35rem; flex-wrap: wrap;}
      [data-baseweb="tab"] {
        background: #edf1fb; border-radius: 10px 10px 0 0; padding: .55rem .9rem;
      }
      [data-baseweb="tab"][aria-selected="true"] {background: #ded8f8;}
      [data-testid="stAlert"] {border-radius: 12px;}
      div.stButton > button, div.stDownloadButton > button {border-radius: 10px;}
      .small-note {color: #5f6b7a; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


PRIOR_LABELS = {
    "Uniformi tra classi (p_j = 1/l)": "uniform",
    "Empirici (p_j = n_j/N)": "empirical",
}

APP_VERSION = "4.3.0"
EXACT_CIRCUIT_QUBIT_LIMIT = 8
ISOLATED_SYNTHESIS_QUBIT_LIMIT = 6
FULL_GATE_DIAGRAM_LIMIT = 5_000


@st.cache_data(show_spinner=False, max_entries=12)
def execute_experiment(
    dataset_key: str,
    copies: int,
    test_fraction: float,
    random_seed: int,
    prior_mode: str,
    relative_tolerance: float,
) -> dict:
    return run_experiment(
        dataset_key,
        copies=copies,
        test_fraction=test_fraction,
        random_seed=random_seed,
        prior_mode=prior_mode,
        relative_tolerance=relative_tolerance,
    )


@st.cache_resource(show_spinner=False, max_entries=6)
def construct_dilation_cached(
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


def transpile_in_isolated_process(
    qpy_payload: bytes,
    timeout_seconds: int,
    max_diagram_gates: int,
):
    return isolated_transpile_qpy(
        qpy_payload,
        timeout_seconds=timeout_seconds,
        max_diagram_gates=max_diagram_gates,
        fold=120,
    )


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
                "Training (s)": result.train_seconds,
                "Predizione (s)": result.predict_seconds,
                "Stato modello": human_bytes(result.model_state_bytes),
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


st.title("PGM Lab")
st.caption(
    f"Versione {APP_VERSION} · classificatori equivalenti, circuito completo e "
    "integrazione quantistica protetta"
)
st.write(
    "Confronto riproducibile tra **c-PGM**, **k-PGM** e **r-PGM (Rc-PGM)**. "
    "I tre calcoli usano rappresentazioni indipendenti, ma gli stessi dati, prior, "
    "split e soglia spettrale. Dopo il training, l'app costruisce anche il circuito "
    "quantistico della PGM mediante una dilatazione di Naimark (Neumark nel paper)."
)

with st.expander("Che cosa significa 'equivalenti'?", expanded=False):
    st.markdown(
        r"""
        - **c-PGM** costruisce esplicitamente il tensore di dimensione $d^c$.
        - **k-PGM** usa il kernel omogeneo $\langle x,z\rangle^c$ e una matrice $N\times N$.
        - **r-PGM** usa la base simmetrica minima di dimensione
          $d_{sym}=\binom{d+c-1}{c}$.

        L'equivalenza teorica riguarda gli score di classe. L'app verifica sia lo
        scarto massimo tra gli score sia l'identità delle predizioni, usando la stessa
        regola deterministica in caso di pareggio numerico. Il termine di completamento
        $P_{ker(\sigma)}/l$ viene omesso dagli score perché è uguale per ogni classe e
        non modifica l'argmax, come osservato nell'appendice del paper.
        """
    )

st.subheader("1. Scegli il dataset")
st.dataframe(
    catalog_frame(),
    hide_index=True,
    width="stretch",
    column_config={
        "Repository": st.column_config.LinkColumn(
            "Repository pubblico", display_text="Apri fonte"
        )
    },
)

left, middle, right = st.columns([1.5, 1, 1])
with left:
    selected_key = st.selectbox(
        "Dataset",
        options=[dataset.key for dataset in DATASETS],
        index=2,
        format_func=lambda key: (
            f"{get_dataset_spec(key).display_name} - "
            f"{get_dataset_spec(key).samples} campioni, "
            f"{get_dataset_spec(key).features} feature, "
            f"{get_dataset_spec(key).classes} classi"
        ),
    )
with middle:
    copies = st.slider("Numero di copie c", min_value=1, max_value=8, value=2, step=1)
with right:
    memory_budget_gib = st.select_slider(
        "Budget RAM per il calcolo",
        options=[0.5, 1.0, 2.0, 4.0, 8.0],
        value=2.0,
        format_func=lambda value: f"{value:g} GiB",
    )

with st.expander("Impostazioni avanzate", expanded=False):
    advanced_1, advanced_2, advanced_3 = st.columns(3)
    with advanced_1:
        test_fraction = st.slider(
            "Quota test set", min_value=0.15, max_value=0.40, value=0.20, step=0.05
        )
    with advanced_2:
        random_seed = st.number_input(
            "Seed dello split", min_value=0, max_value=1_000_000, value=42, step=1
        )
    with advanced_3:
        prior_label = st.selectbox(
            "Prior di classe",
            list(PRIOR_LABELS),
            help=(
                "I prior uniformi seguono l'Eq. (4). Per classi sbilanciate il k-PGM "
                "usa il Gram pesato, così resta esattamente equivalente ai due metodi "
                "primali."
            ),
        )
    tolerance_exponent = st.select_slider(
        "Soglia spettrale relativa",
        options=[8, 9, 10, 11, 12],
        value=10,
        format_func=lambda exponent: f"10^-{exponent}",
        help="Gli autovalori <= soglia x lambda_max sono esclusi in tutti e tre i metodi.",
    )

spec = get_dataset_spec(selected_key)
n_test_estimate = ceil(spec.samples * test_fraction)
n_train_estimate = spec.samples - n_test_estimate
tensor_dimension, symmetric_dimension = representation_dimensions(spec.features, copies)
feasibility = implementation_feasibility(
    n_train=n_train_estimate,
    n_test=n_test_estimate,
    dimension=spec.features,
    copies=copies,
    memory_budget_bytes=int(memory_budget_gib * 1024**3),
)

st.subheader("2. Controlla le dimensioni prima del calcolo")
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("N training stimato", f"{n_train_estimate:,}")
metric_2.metric("d (feature codificate)", f"{spec.features:,}")
metric_3.metric("d^c (c-PGM)", f"{tensor_dimension:,}")
metric_4.metric("d_sym (r-PGM)", f"{symmetric_dimension:,}")

preview_circuit_resources = resources_for_dataset(
    spec.features,
    copies,
    spec.classes,
    exact_qubit_limit=EXACT_CIRCUIT_QUBIT_LIMIT,
)
if preview_circuit_resources.exact_materialization_allowed:
    st.info(
        "Circuito quantistico previsto: "
        f"{preview_circuit_resources.system_qubits} qubit di sistema + "
        f"{preview_circuit_resources.outcome_qubits} qubit di esito = "
        f"{preview_circuit_resources.total_qubits} qubit. La dilatazione unitaria "
        "esatta verrà costruita dopo il training."
    )
else:
    st.warning(
        "Circuito quantistico previsto: "
        f"{preview_circuit_resources.total_qubits} qubit e matrice densa "
        f"{preview_circuit_resources.unitary_dimension:,} x "
        f"{preview_circuit_resources.unitary_dimension:,} "
        f"({human_bytes(preview_circuit_resources.unitary_bytes)}). L'app mostrerà "
        "lo schema logico, ma non materializzerà la matrice esatta oltre il limite "
        f"prudenziale di {EXACT_CIRCUIT_QUBIT_LIMIT} qubit."
    )

preview_complexities = paper_complexities(
    n_train=n_train_estimate,
    dimension=spec.features,
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
    "per k-PGM si usa il limite superiore r_G=N; dopo il run compare il rank effettivo."
)

if feasibility.feasible:
    st.success(
        "Configurazione eseguibile con il limite prudenziale dell'app. "
        f"Picco NumPy stimato: {human_bytes(feasibility.estimated_peak_bytes)}."
    )
else:
    st.error(
        "Il confronto numerico è bloccato per evitare un crash: "
        + "; ".join(feasibility.reasons)
        + ". Riduci c, scegli un dataset con meno feature o aumenta il budget RAM. "
        "La tabella di complessità rimane comunque valida."
    )

configuration_key = (
    selected_key,
    copies,
    float(test_fraction),
    int(random_seed),
    PRIOR_LABELS[prior_label],
    int(tolerance_exponent),
)
run_clicked = st.button(
    "Esegui i classificatori e costruisci il circuito",
    type="primary",
    disabled=not feasibility.feasible,
    width="stretch",
)

if run_clicked:
    try:
        with st.spinner("Download/cache del dataset e calcolo dei tre PGM in corso..."):
            payload = execute_experiment(
                selected_key,
                copies,
                float(test_fraction),
                int(random_seed),
                PRIOR_LABELS[prior_label],
                10.0 ** (-int(tolerance_exponent)),
            )
        st.session_state["last_pgm_run"] = {
            "configuration_key": configuration_key,
            "payload": payload,
        }
    except Exception as error:
        st.exception(error)

saved_run = st.session_state.get("last_pgm_run")
if saved_run and saved_run["configuration_key"] == configuration_key:
    payload = saved_run["payload"]
    results: dict[str, MethodResult] = payload["results"]
    y_test = payload["y_test"]
    table = results_frame(results, y_test)

    st.subheader("3. Risultati")
    accuracy_columns = st.columns(3)
    for column, method in zip(accuracy_columns, ("c-PGM", "k-PGM", "r-PGM")):
        accuracy = accuracy_score(y_test, results[method].predictions)
        column.metric(f"Accuratezza {method}", f"{100.0 * accuracy:.4f}%")

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
    if all_predictions_equal:
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
        lambda value: f"{value:.6f}"
    )
    display_table["Predizione (s)"] = display_table["Predizione (s)"].map(
        lambda value: f"{value:.6f}"
    )
    st.dataframe(display_table, hide_index=True, width="stretch")
    st.caption(
        f"Dataset caricato da: {payload['source_used']}. Preprocessing: imputazione "
        "mediana, min-max [0.001, 1] appreso solo sul training set, normalizzazione L2."
    )

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

    st.subheader("4. Circuito quantistico della PGM")
    resources = resources_for_dataset(
        payload["d"],
        copies,
        payload["class_count"],
        exact_qubit_limit=EXACT_CIRCUIT_QUBIT_LIMIT,
    )
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

        (La stessa costruzione è chiamata anche *Neumark dilation* nel paper.)

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
                )
                circuit = build_qiskit_circuit(dilation)
                circuit_drawing = circuit_svg(dilation)
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
                f"{selected_key}|{copies}|{int(random_seed)}|"
                f"{PRIOR_LABELS[prior_label]}|{int(tolerance_exponent)}"
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
                                    file_name=(
                                        f"circuito_completo_{selected_key}_c{copies}.txt"
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
                                    file_name=(
                                        f"circuito_decomposto_{selected_key}_c{copies}.qpy"
                                    ),
                                    mime="application/octet-stream",
                                )
                        else:
                            st.warning(synthesis_report.message)
                            st.info(
                                "Il crash è rimasto confinato nel processo di sintesi: "
                                "Streamlit e tutti i risultati del training restano attivi."
                            )
                else:
                    st.warning(
                        "La sintesi completa non viene avviata automaticamente oltre "
                        f"{ISOLATED_SYNTHESIS_QUBIT_LIMIT} qubit: il limite prudenziale "
                        f"è circa {cnot_upper_bound:,} CNOT, oltre alle rotazioni a un "
                        "qubit. La transpilation per uno specifico backend fornirà i "
                        "conteggi effettivi solo se il circuito supera il preflight."
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
                    data=classification_details.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        f"classificazioni_dettagliate_{selected_key}_c{copies}.csv"
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
                    "campione selezionato. Oltre agli shot puoi configurare seed, "
                    "ottimizzazione, modello di rumore e provider. Le credenziali "
                    "non entrano negli export o nella cache dell'app; possono essere "
                    "lette da variabili d'ambiente o dal file locale Streamlit Secrets."
                )
                python_status = hardware_python_status()
                st.caption(
                    f"Python {python_status['version']} · AWS Braket richiede Python "
                    ">=3.11 · LRZ MQSS richiede Python >=3.9 e <3.14 · Aer, IBM e "
                    "IonQ sono consigliati con Python >=3.10."
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
                execution_settings = st.columns(3)
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
                    optimization_level = int(
                        st.select_slider(
                            "Ottimizzazione transpiler",
                            options=[0, 1, 2, 3],
                            value=1,
                            help=(
                                "0 conserva la struttura; 3 cerca una riduzione più "
                                "aggressiva delle porte e può richiedere più tempo."
                            ),
                            key=f"optimization_level_{synthesis_id}",
                        )
                    )
                with execution_settings[2]:
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
                            "🧰 Qiskit Aer locale — circuito ideale o rumoroso",
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
                                f"{execution_seed}|{optimization_level}|{aer_method}|"
                                f"{one_qubit_error}|{two_qubit_error}|{readout_error}"
                            )
                            if st.button(
                                "Esegui circuito con Qiskit Aer",
                                type="primary",
                                key=f"aer_simulate_{aer_id}",
                            ):
                                try:
                                    sample_circuit = build_sample_circuit(
                                        dilation,
                                        test_states[sample_index],
                                        name=f"PGM_test_{sample_index + 1}",
                                    )
                                    with st.spinner("Simulazione del circuito completo..."):
                                        aer_counts = simulate_aer_shots(
                                            sample_circuit,
                                            shots=shots,
                                            seed=execution_seed,
                                            method=aer_method,
                                            optimization_level=optimization_level,
                                            one_qubit_error=one_qubit_error,
                                            two_qubit_error=two_qubit_error,
                                            readout_error=readout_error,
                                        )
                                    st.session_state["aer_quantum_result"] = {
                                        "id": aer_id,
                                        "sample_index": sample_index,
                                        "counts": aer_counts,
                                    }
                                except Exception as error:
                                    st.error(f"Simulazione Aer non riuscita: {error}")
                            aer_result = st.session_state.get("aer_quantum_result")
                            if aer_result and aer_result["id"] == aer_id:
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
                        "Il preflight aggiunge la preparazione dello stato test e "
                        "sintetizza tutto in un processo isolato. Il provider eseguirà "
                        "poi una seconda transpilation nella base nativa del dispositivo."
                    )
                    preflight_id = (
                        f"{synthesis_id}|{selected_provider}|"
                        f"{selected_device.identifier}|{sample_index}|opt={optimization_level}"
                    )
                    if resources.total_qubits > ISOLATED_SYNTHESIS_QUBIT_LIMIT:
                        st.error(
                            "Esecuzione esterna disabilitata: questo circuito supera "
                            f"{ISOLATED_SYNTHESIS_QUBIT_LIMIT} qubit e la sintesi sicura "
                            "non è praticabile sul computer locale."
                        )
                    else:
                        if st.button(
                            "Prepara e verifica il circuito hardware",
                            disabled=not device_has_capacity,
                            key=f"preflight_{preflight_id}",
                        ):
                            try:
                                sample_circuit = build_sample_circuit(
                                    dilation,
                                    test_states[sample_index],
                                    name=f"PGM_test_{sample_index + 1}",
                                )
                                with st.spinner(
                                    "Decomposizione completa, inclusa la preparazione "
                                    "dello stato..."
                                ):
                                    preflight_report = transpile_in_isolated_process(
                                        qpy_bytes(sample_circuit),
                                        300,
                                        FULL_GATE_DIAGRAM_LIMIT,
                                    )
                                st.session_state["hardware_preflight"] = {
                                    "id": preflight_id,
                                    "sample_index": sample_index,
                                    "report": preflight_report,
                                }
                            except Exception as error:
                                st.error(f"Preflight non riuscito: {error}")

                        saved_preflight = st.session_state.get("hardware_preflight")
                        if (
                            saved_preflight
                            and saved_preflight["id"] == preflight_id
                        ):
                            preflight_report = saved_preflight["report"]
                            if preflight_report.status not in {"success", "partial"}:
                                st.error(preflight_report.message)
                            else:
                                if preflight_report.status == "partial":
                                    st.warning(preflight_report.message)
                                preflight_columns = st.columns(3)
                                preflight_columns[0].metric(
                                    "Porte preflight", f"{preflight_report.size:,}"
                                )
                                preflight_columns[1].metric(
                                    "Profondità", f"{preflight_report.depth:,}"
                                )
                                preflight_columns[2].metric(
                                    "Qubit", resources.total_qubits
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
                                    selected_provider == "lrz"
                                    and preflight_report.size is not None
                                    and preflight_report.size > 2_000
                                ):
                                    st.warning(
                                        "Il circuito supera 2.000 porte, limite noto per "
                                        "alcune risorse LRZ; altri backend possono avere "
                                        "limiti differenti."
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
                                    ),
                                    key=f"submit_external_{preflight_id}",
                                ):
                                    secret_for_error = (
                                        lrz_token
                                        or ionq_token
                                        or ibm_token
                                        or aws_secret_access_key
                                        or None
                                    )
                                    try:
                                        executable_circuit = circuit_from_qpy(
                                            preflight_report.transpiled_qpy
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
                                                        optimization_level
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
                                                        optimization_level
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
                    file_name=f"circuito_PGM_{selected_key}_c{copies}.svg",
                    mime="image/svg+xml",
                    width="stretch",
                )
                export_columns[1].download_button(
                    "Scarica circuito QPY",
                    data=qpy_bytes(circuit),
                    file_name=f"circuito_PGM_{selected_key}_c{copies}.qpy",
                    mime="application/octet-stream",
                    width="stretch",
                )
                export_columns[2].download_button(
                    "Scarica matrici NPZ",
                    data=dilation_npz_bytes(
                        measurement,
                        dilation,
                        feature_count=payload["d"],
                        copies=copies,
                    ),
                    file_name=f"dilatazione_PGM_{selected_key}_c{copies}.npz",
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
            circuit_drawing = circuit_svg(dilation)
            st.warning(
                "Per questa configurazione la dilatazione avrebbe una matrice densa "
                f"{resources.unitary_dimension:,} x {resources.unitary_dimension:,} "
                f"({human_bytes(resources.unitary_bytes)} in complex128). Per proteggere "
                "la memoria, l'app mostra l'architettura dimensionata ma non materializza "
                "U_PGM. Riduci il numero di copie o scegli un dataset con meno feature "
                "per ottenere il circuito esatto esportabile."
            )
            st.markdown(circuit_drawing, unsafe_allow_html=True)
            st.caption(
                "Schema logico dimensionato; U_PGM_symbolic indica la matrice non "
                "materializzata per questa configurazione."
            )
            st.markdown("**Corrispondenza tra esiti e classi**")
            st.dataframe(outcome_table, hide_index=True, width="stretch")
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
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"predizioni_{selected_key}_c{copies}_seed{int(random_seed)}.csv"
        ),
        mime="text/csv",
    )
elif saved_run:
    st.info("Le impostazioni sono cambiate: premi il pulsante per calcolare il nuovo caso.")

st.divider()
st.caption(
    "Nota: r-PGM è il nome breve usato qui per il reduced c-PGM (Rc-PGM) del paper. "
    "Per c=1 la mappa ridotta coincide con la mappa originale."
)
