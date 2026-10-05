"""Leakage-safe benchmarks between PGM and standard classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    display_name: str
    family: str
    description: str


CLASSIFIER_SPECS: tuple[ClassifierSpec, ...] = (
    ClassifierSpec(
        "mlp",
        "Rete neurale MLP",
        "Reti neurali",
        "Multi-Layer Perceptron feed-forward con regolarizzazione.",
    ),
    ClassifierSpec(
        "random_forest",
        "Random Forest",
        "Ensemble",
        "Foresta di alberi con pesi bilanciati e aggregazione robusta.",
    ),
    ClassifierSpec(
        "bernoulli_nb",
        "Naive Bayes Bernoulli",
        "Bayesiani",
        "Modello bayesiano su feature binarizzate dopo scaling train-only.",
    ),
    ClassifierSpec(
        "knn",
        "k-Nearest Neighbors",
        "Vicinato",
        "Classificazione per vicinato con distanze standardizzate.",
    ),
    ClassifierSpec(
        "qda",
        "Analisi discriminante quadratica (QDA)",
        "Discriminanti",
        "Frontiere quadratiche con regolarizzazione della covarianza.",
    ),
    ClassifierSpec(
        "logistic_regression",
        "Regressione logistica",
        "Lineari",
        "Modello lineare probabilistico con bilanciamento delle classi.",
    ),
    ClassifierSpec(
        "extra_tree",
        "Extra Tree",
        "Alberi",
        "Singolo albero estremamente randomizzato.",
    ),
    ClassifierSpec(
        "extra_trees",
        "Extra Trees (ensemble)",
        "Ensemble",
        "Ensemble di alberi estremamente randomizzati.",
    ),
    ClassifierSpec(
        "rbf_svm",
        "SVM con kernel RBF",
        "Kernel",
        "Support Vector Machine non lineare con kernel gaussiano.",
    ),
    ClassifierSpec(
        "linear_svm",
        "SVM lineare",
        "Lineari",
        "Support Vector Machine lineare con classi bilanciate.",
    ),
    ClassifierSpec(
        "hist_gradient_boosting",
        "HistGradientBoosting",
        "Boosting",
        "Boosting istogrammico rapido con regolarizzazione.",
    ),
    ClassifierSpec(
        "gradient_boosting",
        "Gradient Boosting",
        "Boosting",
        "Boosting classico di alberi decisionali.",
    ),
    ClassifierSpec(
        "lda",
        "Analisi discriminante lineare (LDA)",
        "Discriminanti",
        "Discriminante lineare con shrinkage della covarianza.",
    ),
    ClassifierSpec(
        "gaussian_nb",
        "Naive Bayes gaussiano",
        "Bayesiani",
        "Baseline probabilistica gaussiana veloce.",
    ),
    ClassifierSpec(
        "adaboost",
        "AdaBoost",
        "Boosting",
        "Ensemble adattivo di classificatori deboli.",
    ),
    ClassifierSpec(
        "decision_tree",
        "Albero decisionale",
        "Alberi",
        "Albero CART con bilanciamento delle classi.",
    ),
    ClassifierSpec(
        "ridge",
        "Ridge Classifier",
        "Lineari",
        "Classificatore lineare regolarizzato, rapido e stabile.",
    ),
)

CLASSIFIER_BY_KEY = {spec.key: spec for spec in CLASSIFIER_SPECS}

METRIC_LABELS: dict[str, str] = {
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuratezza",
    "precision_macro": "Precision macro",
    "recall_macro": "Recall macro",
    "f1_macro": "F1-score macro",
    "cohen_kappa": "Kappa di Cohen",
    "matthews_corrcoef": "Coefficiente di Matthews",
    "roc_auc": "ROC-AUC macro",
}


def benchmark_raw_split(
    payload: dict[str, Any],
    *,
    dataset_key: str,
    test_fraction: float,
    random_seed: int,
) -> tuple[Any, np.ndarray, Any, np.ndarray]:
    """Return the exact raw split, including compatibility with old cache entries.

    Version 5.1 added raw frames to the experiment payload. Streamlit can retain a
    payload created by an earlier version during a hot reload, so this function
    reconstructs the deterministic split when those two fields are absent and
    verifies its labels before it is used by a benchmark.
    """

    if "X_train_raw" in payload and "X_test_raw" in payload:
        return (
            payload["X_train_raw"],
            np.asarray(payload["y_train"]),
            payload["X_test_raw"],
            np.asarray(payload["y_test"]),
        )

    from data_catalog import load_public_dataset

    X_frame, y_series, _ = load_public_dataset(dataset_key)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_frame,
        y_series.to_numpy(),
        test_size=float(test_fraction),
        random_state=int(random_seed),
        stratify=y_series.to_numpy(),
    )
    expected_train = np.asarray(payload["y_train"])
    expected_test = np.asarray(payload["y_test"])
    if not (
        np.array_equal(np.asarray(y_train), expected_train)
        and np.array_equal(np.asarray(y_test), expected_test)
    ):
        raise RuntimeError(
            "Impossibile ricostruire con certezza lo stesso split train/test "
            "della PGM. Ricalcolare l'esperimento."
        )
    return X_train_raw, expected_train, X_test_raw, expected_test


def get_classifier_spec(key: str) -> ClassifierSpec:
    try:
        return CLASSIFIER_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"Classificatore sconosciuto: {key!r}") from error


def _scaled_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _tree_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def build_classifier(
    key: str,
    *,
    n_train: int,
    random_seed: int,
) -> tuple[Pipeline, dict[str, list[Any]]]:
    """Return a deterministic estimator and a compact train-only tuning grid."""

    get_classifier_spec(key)
    if key == "mlp":
        return (
            _scaled_pipeline(
                MLPClassifier(
                    hidden_layer_sizes=(48,),
                    activation="relu",
                    solver="lbfgs",
                    alpha=1e-3,
                    max_iter=350,
                    random_state=random_seed,
                )
            ),
            {
                "model__hidden_layer_sizes": [(32,), (64,), (48, 24)],
                "model__alpha": [1e-4, 1e-3],
            },
        )
    if key == "random_forest":
        return (
            _tree_pipeline(
                RandomForestClassifier(
                    n_estimators=180,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    random_state=random_seed,
                    n_jobs=1,
                )
            ),
            {
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt"],
            },
        )
    if key == "bernoulli_nb":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "scaler",
                        MinMaxScaler(feature_range=(0.0, 1.0), clip=True),
                    ),
                    ("model", BernoulliNB(binarize=0.5)),
                ]
            ),
            {"model__alpha": [0.1, 1.0, 10.0]},
        )
    if key == "knn":
        maximum_neighbor = max(1, min(19, int(n_train) // 3))
        neighbor_values = sorted(
            {
                max(1, min(maximum_neighbor, candidate))
                for candidate in (3, 7, 15)
            }
        )
        return (
            _scaled_pipeline(
                KNeighborsClassifier(weights="distance", n_jobs=1)
            ),
            {"model__n_neighbors": neighbor_values},
        )
    if key == "qda":
        return (
            _scaled_pipeline(QuadraticDiscriminantAnalysis(reg_param=0.1)),
            {"model__reg_param": [0.0, 0.1, 0.3]},
        )
    if key == "logistic_regression":
        return (
            _scaled_pipeline(
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=random_seed,
                )
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        )
    if key == "extra_tree":
        return (
            _tree_pipeline(
                ExtraTreeClassifier(
                    class_weight="balanced",
                    max_features="sqrt",
                    random_state=random_seed,
                )
            ),
            {"model__min_samples_leaf": [1, 2, 4]},
        )
    if key == "extra_trees":
        return (
            _tree_pipeline(
                ExtraTreesClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    max_features="sqrt",
                    random_state=random_seed,
                    n_jobs=1,
                )
            ),
            {"model__min_samples_leaf": [1, 2, 4]},
        )
    if key == "rbf_svm":
        return (
            _scaled_pipeline(
                SVC(
                    C=2.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=False,
                    random_state=random_seed,
                )
            ),
            {"model__C": [0.5, 2.0, 8.0]},
        )
    if key == "linear_svm":
        return (
            _scaled_pipeline(
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=5_000,
                    random_state=random_seed,
                )
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        )
    if key == "hist_gradient_boosting":
        return (
            _tree_pipeline(
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=150,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    random_state=random_seed,
                )
            ),
            {
                "model__max_leaf_nodes": [15, 31],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    if key == "gradient_boosting":
        return (
            _tree_pipeline(
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.08,
                    random_state=random_seed,
                )
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    if key == "lda":
        return (
            _scaled_pipeline(
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            ),
            {"model__shrinkage": ["auto", 0.1, 0.5]},
        )
    if key == "gaussian_nb":
        return (
            _scaled_pipeline(GaussianNB()),
            {"model__var_smoothing": [1e-11, 1e-9, 1e-7]},
        )
    if key == "adaboost":
        return (
            _tree_pipeline(
                AdaBoostClassifier(
                    n_estimators=150,
                    learning_rate=0.8,
                    random_state=random_seed,
                )
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.5, 1.0],
            },
        )
    if key == "decision_tree":
        return (
            _tree_pipeline(
                DecisionTreeClassifier(
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=random_seed,
                )
            ),
            {"model__max_depth": [None, 5, 10]},
        )
    if key == "ridge":
        return (
            _scaled_pipeline(
                RidgeClassifier(class_weight="balanced", alpha=1.0)
            ),
            {"model__alpha": [0.1, 1.0, 10.0]},
        )
    raise AssertionError("Catalogo dei classificatori incoerente.")


def _aligned_scores(
    estimator: Pipeline,
    X_test,
    classes: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    estimator_classes = np.asarray(estimator.classes_)
    class_positions = {
        label: index for index, label in enumerate(estimator_classes.tolist())
    }
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(X_test), dtype=np.float64)
        aligned = np.column_stack(
            [raw[:, class_positions[label]] for label in classes.tolist()]
        )
        return aligned, "predict_proba"
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X_test), dtype=np.float64)
        if raw.ndim == 1:
            if len(classes) != 2:
                return None, None
            positive_label = estimator_classes[1]
            sign = 1.0 if classes[1] == positive_label else -1.0
            return sign * raw, "decision_function"
        aligned = np.column_stack(
            [raw[:, class_positions[label]] for label in classes.tolist()]
        )
        return aligned, "decision_function"
    return None, None


def macro_roc_auc(
    y_true: np.ndarray,
    scores: np.ndarray | None,
    classes: np.ndarray,
) -> float:
    if scores is None:
        return float("nan")
    y_array = np.asarray(y_true)
    score_array = np.asarray(scores, dtype=np.float64)
    aucs: list[float] = []
    for class_index, class_label in enumerate(classes.tolist()):
        binary_target = (y_array == class_label).astype(np.int8)
        if np.unique(binary_target).size < 2:
            continue
        if score_array.ndim == 1:
            class_score = score_array if class_index == 1 else -score_array
        else:
            class_score = score_array[:, class_index]
        aucs.append(float(roc_auc_score(binary_target, class_score)))
    return float(np.mean(aucs)) if aucs else float("nan")


def classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    classes: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    y_array = np.asarray(y_true)
    predicted = np.asarray(predictions)
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_array, predicted)
        ),
        "accuracy": float(accuracy_score(y_array, predicted)),
        "precision_macro": float(
            precision_score(
                y_array,
                predicted,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_array,
                predicted,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_array,
                predicted,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(cohen_kappa_score(y_array, predicted)),
        "matthews_corrcoef": float(
            matthews_corrcoef(y_array, predicted)
        ),
        "roc_auc": macro_roc_auc(y_array, scores, classes),
    }


def fit_standard_classifier(
    X_train,
    y_train: np.ndarray,
    X_test,
    y_test: np.ndarray,
    *,
    classifier_key: str,
    random_seed: int,
    requested_tuning_folds: int = 3,
) -> dict[str, Any]:
    """Tune on training folds only, refit on all training data, then test once."""

    y_train_array = np.asarray(y_train)
    y_test_array = np.asarray(y_test)
    classes, class_counts = np.unique(y_train_array, return_counts=True)
    estimator, parameter_grid = build_classifier(
        classifier_key,
        n_train=len(y_train_array),
        random_seed=random_seed,
    )
    effective_folds = min(
        max(0, int(requested_tuning_folds)),
        int(class_counts.min()),
    )
    best_parameters: dict[str, Any] = {}
    validation_score: float | None = None
    tuning_message = ""
    fit_start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FitFailedWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        if effective_folds >= 2 and parameter_grid:
            splitter = StratifiedKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=random_seed + 503,
            )
            search = GridSearchCV(
                estimator,
                parameter_grid,
                scoring="balanced_accuracy",
                cv=splitter,
                n_jobs=1,
                refit=True,
                error_score=np.nan,
                return_train_score=False,
            )
            try:
                search.fit(X_train, y_train_array)
                if not np.isfinite(search.best_score_):
                    raise ValueError(
                        "Nessuna configurazione ha prodotto una validazione finita."
                    )
                fitted = search.best_estimator_
                best_parameters = {
                    key.replace("model__", ""): value
                    for key, value in search.best_params_.items()
                }
                validation_score = float(search.best_score_)
            except Exception as error:
                tuning_message = (
                    "Tuning non disponibile; usata la configurazione robusta "
                    f"predefinita. Dettaglio: {error}"
                )
                estimator.fit(X_train, y_train_array)
                fitted = estimator
                effective_folds = 0
        else:
            estimator.fit(X_train, y_train_array)
            fitted = estimator
            tuning_message = (
                "Tuning disattivato perché una classe ha troppo pochi campioni "
                "nel training; usata la configurazione robusta predefinita."
            )
    fit_seconds = perf_counter() - fit_start

    prediction_start = perf_counter()
    predictions = np.asarray(fitted.predict(X_test))
    scores, score_kind = _aligned_scores(fitted, X_test, classes)
    predict_seconds = perf_counter() - prediction_start
    metrics = classification_metrics(
        y_test_array,
        predictions,
        classes=classes,
        scores=scores,
    )
    return {
        "classifier_key": classifier_key,
        "classifier_name": get_classifier_spec(classifier_key).display_name,
        "predictions": predictions,
        "scores": scores,
        "score_kind": score_kind,
        "classes": classes,
        "metrics": metrics,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "tuning_folds": int(effective_folds),
        "validation_balanced_accuracy": validation_score,
        "best_parameters": best_parameters,
        "tuning_message": tuning_message,
    }


def paired_balanced_accuracy_bootstrap(
    y_true: np.ndarray,
    pgm_predictions: np.ndarray,
    competitor_predictions: np.ndarray,
    *,
    resamples: int = 2_000,
    random_seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Paired stratified bootstrap CI for the balanced-accuracy difference."""

    if resamples < 100:
        raise ValueError("Il bootstrap richiede almeno 100 ricampionamenti.")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("Il livello di confidenza deve essere tra 0.5 e 1.")
    y_array = np.asarray(y_true)
    pgm_array = np.asarray(pgm_predictions)
    competitor_array = np.asarray(competitor_predictions)
    if not (
        y_array.shape == pgm_array.shape == competitor_array.shape
    ):
        raise ValueError("Le predizioni da confrontare devono avere la stessa forma.")

    classes = np.unique(y_array)
    pgm_per_class: list[np.ndarray] = []
    competitor_per_class: list[np.ndarray] = []
    for class_label in classes.tolist():
        positions = np.flatnonzero(y_array == class_label)
        pgm_per_class.append((pgm_array[positions] == class_label).astype(float))
        competitor_per_class.append(
            (competitor_array[positions] == class_label).astype(float)
        )

    observed_pgm = float(np.mean([values.mean() for values in pgm_per_class]))
    observed_competitor = float(
        np.mean([values.mean() for values in competitor_per_class])
    )
    rng = np.random.default_rng(random_seed + 907)
    differences = np.zeros(int(resamples), dtype=np.float64)
    for pgm_correct, competitor_correct in zip(
        pgm_per_class, competitor_per_class
    ):
        sampled_positions = rng.integers(
            0,
            len(pgm_correct),
            size=(int(resamples), len(pgm_correct)),
        )
        differences += (
            pgm_correct[sampled_positions].mean(axis=1)
            - competitor_correct[sampled_positions].mean(axis=1)
        ) / len(classes)

    alpha = 1.0 - float(confidence_level)
    lower, upper = np.quantile(
        differences,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    if lower > 0.0:
        winner = "pgm"
    elif upper < 0.0:
        winner = "competitor"
    else:
        winner = "tie"
    return {
        "pgm_balanced_accuracy": observed_pgm,
        "competitor_balanced_accuracy": observed_competitor,
        "difference": observed_pgm - observed_competitor,
        "confidence_lower": float(lower),
        "confidence_upper": float(upper),
        "confidence_level": float(confidence_level),
        "pgm_superiority_probability": float(np.mean(differences > 0.0)),
        "winner": winner,
        "resamples": int(resamples),
    }
