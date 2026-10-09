"""Presentation-only internationalization for the Streamlit interface.

The scientific code keeps stable Italian/internal values so changing language can
never alter a branch, cache key, prediction, circuit or provider request.  This
module translates only values at the rendering boundary and maps widget choices
back to their original values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import re
from typing import Any

import pandas as pd


DEFAULT_LANGUAGE = "it"
SUPPORTED_LANGUAGES = ("it", "en")


# Short labels and complete standalone messages.  Keys are the canonical Italian
# UI text already used by the application; values are public-facing English.
EXACT_TRANSLATIONS: dict[str, str] = {
    "Uniformi tra classi (p_j = 1/l)": "Uniform across classes (p_j = 1/l)",
    "Empirici (p_j = n_j/N)": "Empirical (p_j = n_j/N)",
    "Livello": "Level",
    "Stato": "Status",
    "Certificato": "Certified",
    "Non disponibile": "Not available",
    "Gate entangling": "Entangling gates",
    "Profondità": "Depth",
    "Porte totali": "Total gates",
    "Errore": "Error",
    "Messaggio": "Message",
    "Metodo": "Method",
    "Dimensione di lavoro": "Working dimension",
    "Tempo training (Big-O)": "Training time (Big-O)",
    "Proxy training": "Training proxy",
    "Memoria training (Big-O)": "Training memory (Big-O)",
    "Stima memoria paper": "Paper memory estimate",
    "Tempo predizione/campione": "Prediction time/sample",
    "Proxy predizione": "Prediction proxy",
    "Memoria predizione": "Prediction memory",
    "Stima memoria predizione": "Prediction memory estimate",
    "Accuratezza": "Accuracy",
    "Accordo con c-PGM": "Agreement with c-PGM",
    "Rank numerico": "Numerical rank",
    "Dimensione": "Dimension",
    "Calcolo": "Computation",
    "Training (s)": "Training (s)",
    "Predizione (s)": "Prediction (s)",
    "Stato modello": "Model state",
    "feature originali": "original features",
    "Non materializzato": "Not materialized",
    "esplicita indipendente": "independent explicit",
    "kernel diretto indipendente": "independent direct kernel",
    "equivalente esatta via kernel (matrice non materializzata)": (
        "exactly equivalent via kernel (matrix not materialized)"
    ),
    "Classe reale": "Actual class",
    "Classe predetta": "Predicted class",
    "Classe": "Class",
    "classe_reale": "actual_class",
    "predizione_c_PGM": "c_PGM_prediction",
    "predizione_k_PGM": "k_PGM_prediction",
    "predizione_r_PGM": "r_PGM_prediction",
    "Corretta": "Correct",
    "Errata": "Incorrect",
    "✓ Corretta": "✓ Correct",
    "✗ Errata": "✗ Incorrect",
    "Esito": "Outcome",
    "Distanza dalla teoria": "Distance from theory",
    "Frequenza": "Frequency",
    "Conteggi": "Counts",
    "Bitstring": "Bitstring",
    "Non assegnata": "Unassigned",
    "Versione": "Version",
    "Che cosa significa 'equivalenti'?": "What does ‘equivalent’ mean?",
    "Sezione 1": "Section 1",
    "Sezione 2": "Section 2",
    "Sezione 3 · opzionale": "Section 3 · optional",
    "Sezione 4": "Section 4",
    "Sezione 5 · laboratorio quantistico": "Section 5 · quantum laboratory",
    "Documentazione": "Documentation",
    "Seleziona dati e copie oppure carica un file personale; ogni scelta resta modificabile.": (
        "Select data and copies or upload your own file; every choice remains editable."
    ),
    "Verifica subito fattibilità classica, dimensione del circuito e limiti di sicurezza.": (
        "Immediately check classical feasibility, circuit size, and safety limits."
    ),
    "Avvia soltanto i benchmark che desideri; questa sezione non esegue automaticamente la PGM principale.": (
        "Run only the benchmarks you want; this section never runs the main PGM automatically."
    ),
    "Statistiche multi-seed, equivalenza dei metodi, diagnosi e risultati campione per campione.": (
        "Multi-seed statistics, method equivalence, diagnostics, and sample-by-sample results."
    ),
    "Circuito logico, classificazione, validazione, optimizer, simulatori, QPU ed export in un unico spazio protetto.": (
        "Logical circuit, classification, validation, optimizer, simulators, QPUs, and export in one protected workspace."
    ),
    "Manuale bilingue aggiornato con flusso operativo, interpretazione dei risultati, circuito e provider.": (
        "Updated bilingual manual covering the workflow, result interpretation, circuit, and providers."
    ),
    "1. Scegli il dataset": "1. Choose a dataset",
    "Carica un dataset personale": "Upload your own dataset",
    "Trascina il file oppure clicca per sceglierlo": (
        "Drag the file here or click to browse"
    ),
    "Colonna target da predire": "Target column to predict",
    "Feature numeriche da utilizzare": "Numeric features to use",
    "Nome da mostrare nell'app": "Name displayed in the app",
    "Campioni validi": "Valid samples",
    "Feature utilizzabili": "Usable features",
    "Valori feature mancanti": "Missing feature values",
    "Anteprima e distribuzione delle classi": (
        "Preview and class distribution"
    ),
    "Campioni per classe": "Samples per class",
    "Aggiungi al catalogo e seleziona": "Add to catalogue and select",
    "Rimuovi il dataset personale dalla sessione": (
        "Remove the private dataset from this session"
    ),
    "Guida illustrata all'uso": "Illustrated user guide",
    "📘 Scarica la guida PDF completa": "📘 Download the complete PDF guide",
    "Sessione privata": "Private session",
    "Upload privato": "Private upload",
    "Repository pubblico": "Public repository",
    "Apri fonte": "Open source",
    "Dataset": "Dataset",
    "Campioni": "Samples",
    "Feature/campione": "Features/sample",
    "Classi": "Classes",
    "Ambito": "Domain",
    "Repository": "Repository",
    "Numero di copie c": "Number of copies c",
    "Budget RAM per il calcolo": "Computation RAM budget",
    "Gestione delle feature": "Feature handling",
    "Automatica quantum-ready (consigliata)": "Automatic quantum-ready (recommended)",
    "Tutte le feature originali": "All original features",
    "Richiedi manualmente una riduzione PCA": "Manually request PCA reduction",
    "Numero di feature dopo la PCA manuale": "Features after manual PCA",
    "Riduzione manuale PCA": "Manual PCA reduction",
    "Impostazioni avanzate": "Advanced settings",
    "Quota test set": "Test-set fraction",
    "Seed dello split": "Split seed",
    "Seed iniziale": "Initial seed",
    "Numero di seed di valutazione": "Number of evaluation seeds",
    "Prior di classe": "Class prior",
    "Soglia spettrale relativa": "Relative spectral threshold",
    "2. Controlla le dimensioni prima del calcolo": "2. Check dimensions before running",
    "Mappa delle attività": "Activity map",
    "### Mappa delle attività": "### Activity map",
    "① Valutazione PGM": "① PGM evaluation",
    "② Confronto opzionale": "② Optional comparison",
    "③ Risultati e test": "③ Results and testing",
    "④ Laboratorio quantistico": "④ Quantum laboratory",
    "**① Valutazione PGM**": "**① PGM evaluation**",
    "**② Confronto opzionale**": "**② Optional comparison**",
    "**③ Risultati e test**": "**③ Results and testing**",
    "**④ Laboratorio quantistico**": "**④ Quantum laboratory**",
    "Multi-seed, media e deviazione standard.": (
        "Multi-seed evaluation, mean, and standard deviation."
    ),
    "Classificatore scelto o full comparison.": (
        "Selected classifier or full comparison."
    ),
    "Metriche, confusion matrix e singoli campioni.": (
        "Metrics, confusion matrix, and individual samples."
    ),
    "Circuito, verifica, optimizer, simulatori/QPU ed export.": (
        "Circuit, validation, optimizer, simulators/QPUs, and export."
    ),
    "Pronta": "Ready",
    "Bloccata dal budget": "Blocked by the budget",
    "Avvio indipendente": "Independent launch",
    "Dopo la valutazione PGM": "After PGM evaluation",
    "Schema simbolico; circuito esatto oltre il limite": (
        "Symbolic diagram; exact circuit above the limit"
    ),
    "Circuito esatto; sintesi ed esecuzione oltre il limite": (
        "Exact circuit; synthesis and execution above the limit"
    ),
    "Disponibile dopo la valutazione PGM": (
        "Available after PGM evaluation"
    ),
    "Circuito logico": "Logical circuit",
    "Classificazione test": "Test classification",
    "Validazione matematica": "Mathematical validation",
    "Optimizer": "Optimizer",
    "Esecuzione quantistica": "Quantum execution",
    "Esporta": "Export",
    "Visualizza il circuito logico": "View the logical circuit",
    "Configura simulatore o QPU": "Configure a simulator or QPU",
    "Le funzioni sono indipendenti e partono soltanto dal relativo comando. Questa mappa anticipa ciò che diventerà disponibile dopo la valutazione.": (
        "Functions are independent and start only from their own command. This map previews what becomes available after evaluation."
    ),
    "Questi comandi si attivano dopo la valutazione PGM quando la dimensione del circuito rispetta i limiti indicati. Il campionamento PGM ideale può restare disponibile anche quando la sintesi gate-by-gate è disabilitata.": (
        "These commands activate after PGM evaluation when the circuit size meets the stated limits. Ideal PGM sampling may remain available even when gate-by-gate synthesis is disabled."
    ),
    "N training stimato": "Estimated training N",
    "Feature originali": "Original features",
    "Feature codificate": "Encoded features",
    "Feature utilizzate": "Features used",
    "Dimensioni candidate (ampiezza / stereografico)": (
        "Candidate dimensions (amplitude / stereographic)"
    ),
    "Encoding selezionato": "Selected encoding",
    "Encoding in ampiezza normalizzato": "Normalized amplitude encoding",
    "Stereografico + encoding in ampiezza": (
        "Stereographic + amplitude encoding"
    ),
    "Fattore di rescaling t": "Rescaling factor t",
    "Non applicabile": "Not applicable",
    "Accuratezza di validazione": "Validation accuracy",
    "Balanced accuracy di validazione": "Validation balanced accuracy",
    "Balanced accuracy validazione": "Validation balanced accuracy",
    "Vantaggio sulla baseline": "Gain over baseline",
    "Confronto degli encoding sul training set": (
        "Training-set encoding comparison"
    ),
    "Encoding": "Encoding",
    "Fattore t": "Factor t",
    "Feature dopo PCA": "Features after PCA",
    "Dimensione encoding": "Encoding dimension",
    "Deviazione standard": "Standard deviation",
    "Valutatore PGM": "PGM evaluator",
    "✓ Selezionato": "✓ Selected",
    "Disponibile": "Available",
    "Esegui i classificatori e costruisci il circuito": (
        "Run the classifiers and build the circuit"
    ),
    "Esegui i classificatori (circuito non materializzato)": (
        "Run the classifiers (circuit not materialized)"
    ),
    "Esegui la valutazione multi-seed e costruisci il circuito": (
        "Run multi-seed evaluation and build the circuit"
    ),
    "Esegui la valutazione multi-seed (circuito non materializzato)": (
        "Run multi-seed evaluation (circuit not materialized)"
    ),
    "3. Risultati": "3. Results",
    "3. Confronto con altri classificatori": (
        "3. Comparison with other classifiers"
    ),
    "Avvia la valutazione PGM multi-seed": (
        "Run the multi-seed PGM evaluation"
    ),
    "### Avvia la valutazione PGM multi-seed": (
        "### Run the multi-seed PGM evaluation"
    ),
    "Con questa configurazione verranno resi disponibili anche il circuito logico esatto e gli strumenti quantistici compatibili con i limiti indicati sopra.": (
        "This configuration will also enable the exact logical circuit and the quantum tools compatible with the limits stated above."
    ),
    "La valutazione classica resta completa; per questa configurazione il circuito sarà mostrato come schema simbolico dimensionato.": (
        "Classical evaluation remains complete; for this configuration the circuit will be shown as a dimensioned symbolic diagram."
    ),
    "I risultati compariranno qui dopo aver premuto il pulsante di valutazione multi-seed immediatamente sopra.": (
        "Results will appear here after you press the multi-seed evaluation button immediately above."
    ),
    "Abilita il confronto opzionale": "Enable optional comparison",
    "Classificatore standard": "Standard classifier",
    "Confronta sul dataset selezionato": (
        "Compare on the selected dataset"
    ),
    "Full comparison sui dataset binari": (
        "Full comparison on binary datasets"
    ),
    "4. Risultati PGM": "4. PGM results",
    "Varianza PCA conservata": "PCA variance retained",
    "100% (nessuna PCA)": "100% (no PCA)",
    "Campioni test": "Test samples",
    "Classificazioni corrette": "Correct classifications",
    "Errori": "Errors",
    "Campione test": "Test sample",
    "Risultato di ogni campione": "Result for each sample",
    "Diagnostica numerica": "Numerical diagnostics",
    "Confronto": "Comparison",
    "Scarto massimo assoluto score": "Maximum absolute score difference",
    "soglia_spettrale": "spectral_threshold",
    "autovalore_minimo_grezzo": "raw_minimum_eigenvalue",
    "errore_massimo_norma_feature": "maximum_feature_norm_error",
    "4. Circuito quantistico della PGM": "4. PGM quantum circuit",
    "5. Circuito quantistico della PGM": "5. PGM quantum circuit",
    "Sintetizza il circuito": "Synthesize the circuit",
    "Ottimizza e certifica": "Optimize and certify",
    "Strumenti quantistici non attivi per questa configurazione": (
        "Quantum tools unavailable for this configuration"
    ),
    "#### Strumenti quantistici non attivi per questa configurazione": (
        "#### Quantum tools unavailable for this configuration"
    ),
    "Le funzioni restano visibili, ma richiedono la matrice esatta U_PGM. Il blocco evita che un calcolo denso eccessivo interrompa l'intera applicazione.": (
        "The functions remain visible, but they require the exact U_PGM matrix. This guard prevents an oversized dense computation from interrupting the entire application."
    ),
    "Esegui prima la valutazione PGM nella sezione immediatamente precedente. Il laboratorio si attiverà senza avviare automaticamente sintesi, optimizer, simulatori o QPU.": (
        "First run the PGM evaluation in the immediately preceding section. The laboratory will activate without automatically starting synthesis, the optimizer, simulators, or QPUs."
    ),
    "Balanced accuracy": "Balanced accuracy",
    "Precision macro": "Macro precision",
    "Recall macro": "Macro recall",
    "F1-score macro": "Macro F1-score",
    "Kappa di Cohen": "Cohen's kappa",
    "Coefficiente di Matthews": "Matthews correlation coefficient",
    "ROC-AUC macro": "Macro ROC-AUC",
    "Valore": "Value",
    "Classificatore": "Classifier",
    "Balanced accuracy PGM": "PGM balanced accuracy",
    "Differenza PGM − confronto": "PGM − comparator difference",
    "Differenza media PGM − confronto": "Mean PGM − comparator difference",
    "Intervallo bootstrap 95%": "95% bootstrap interval",
    "Intervallo appaiato 95%": "95% paired interval",
    "Numero di seed": "Number of seeds",
    "Risultati per ciascun seed": "Results for each seed",
    "Backend PGM": "PGM backend",
    "Balanced accuracy confronto": "Comparator balanced accuracy",
    "Matrici di confusione e tuning del classificatore sul seed di riferimento": (
        "Confusion matrices and classifier tuning on the reference seed"
    ),
    "Dettaglio della valutazione multi-seed": "Multi-seed evaluation details",
    "#### Classificazione test (calcolo classico adattivo)": (
        "#### Test classification (adaptive classical computation)"
    ),
    "Matrici di confusione e tuning del classificatore": (
        "Confusion matrices and classifier tuning"
    ),
    "Fold di tuning": "Tuning folds",
    "Balanced accuracy di validazione": "Validation balanced accuracy",
    "Parametri selezionati": "Selected parameters",
    "Tempo fit e tuning (s)": "Fit and tuning time (s)",
    "Encoding PGM": "PGM encoding",
    "Fattore t PGM": "PGM factor t",
    "Configurazione predefinita": "Default configuration",
    "Dataset completati": "Completed datasets",
    "Vittorie PGM": "PGM wins",
    "Pareggi / non conclusivi": "Ties / inconclusive",
    "Pareggio": "Tie",
    "Vincitore": "Winner",
    "#### Matrice WIN / TIE / LOSS": "#### WIN / TIE / LOSS matrix",
    "Risultati non conclusivi": "Inconclusive results",
    "Non conclusivo": "Inconclusive",
    "Pareggio esatto": "Exact tie",
    "Vittoria PGM confermata": "Confirmed PGM win",
    "#### Confronto quantitativo WIN / LOSS": (
        "#### Quantitative WIN / LOSS comparison"
    ),
    "PGM — balanced accuracy": "PGM — balanced accuracy",
    "Differenza PGM − confronto (IC 95%)": (
        "PGM − comparator difference (95% CI)"
    ),
    "Seed favorevoli PGM / confronto": "Seeds favoring PGM / comparator",
    "Esito statistico": "Statistical outcome",
    "Seed PGM superiore": "Seeds with higher PGM score",
    "Seed esattamente uguali": "Seeds with exactly equal scores",
    "Percentuale dei seed": "Percentage of seeds",
    "PGM superiore": "PGM higher",
    "Esattamente uguali": "Exactly equal",
    "Limite inferiore IC 95%": "95% CI lower bound",
    "Limite superiore IC 95%": "95% CI upper bound",
    "Seed PGM superiore (%)": "Seeds with higher PGM score (%)",
    "Seed confronto superiore (%)": "Seeds with higher comparator score (%)",
    "Risultati completi per dataset": "Complete results by dataset",
    "Preparazione del full comparison...": "Preparing full comparison...",
    "Full comparison completato.": "Full comparison completed.",
    "Reti neurali": "Neural networks",
    "Ensemble": "Ensemble",
    "Bayesiani": "Bayesian",
    "Vicinato": "Nearest-neighbor",
    "Discriminanti": "Discriminant analysis",
    "Lineari": "Linear",
    "Alberi": "Trees",
    "Kernel": "Kernel",
    "Boosting": "Boosting",
    "Rete neurale MLP": "MLP neural network",
    "Random Forest": "Random Forest",
    "Naive Bayes Bernoulli": "Bernoulli Naive Bayes",
    "k-Nearest Neighbors": "k-Nearest Neighbors",
    "Analisi discriminante quadratica (QDA)": (
        "Quadratic Discriminant Analysis (QDA)"
    ),
    "Regressione logistica": "Logistic Regression",
    "Extra Tree": "Extra Tree",
    "Extra Trees (ensemble)": "Extra Trees (ensemble)",
    "SVM con kernel RBF": "RBF-kernel SVM",
    "SVM lineare": "Linear SVM",
    "HistGradientBoosting": "HistGradientBoosting",
    "Gradient Boosting": "Gradient Boosting",
    "Analisi discriminante lineare (LDA)": (
        "Linear Discriminant Analysis (LDA)"
    ),
    "Naive Bayes gaussiano": "Gaussian Naive Bayes",
    "AdaBoost": "AdaBoost",
    "Albero decisionale": "Decision Tree",
    "Ridge Classifier": "Ridge Classifier",
    "Metrica": "Metric",
    "Tempo predizione (s)": "Prediction time (s)",
    "Differenza balanced accuracy PGM − confronto (punti %)": (
        "PGM − comparator balanced-accuracy difference (pp)"
    ),
    "**PGM — % per classe reale**": "**PGM — % by actual class**",
    "Dimensione ridotta": "Reduced dimension",
    "Qubit sistema": "System qubits",
    "Qubit esito": "Outcome qubits",
    "Qubit totali": "Total qubits",
    "Matrice U_PGM": "U_PGM matrix",
    "Indice esito": "Outcome index",
    "Bitstring misurata": "Measured bitstring",
    "Interpretazione": "Interpretation",
    "Circuito logico": "Logical circuit",
    "Classificazione test": "Test classification",
    "Validazione matematica": "Mathematical validation",
    "Esecuzione quantistica": "Quantum execution",
    "Esporta": "Export",
    "Limite superiore CNOT (unitaria generica)": "CNOT upper bound (generic unitary)",
    "Soglia disegno completo": "Full-diagram threshold",
    "Sintesi isolata fino a": "Isolated synthesis up to",
    "Sintetizza e mostra la decomposizione completa": (
        "Synthesize and show the full decomposition"
    ),
    "Tempo sintesi": "Synthesis time",
    "Numero": "Count",
    "Diagramma circuitale completo": "Full circuit diagram",
    "Scarica circuito completo (TXT)": "Download full circuit (TXT)",
    "Scarica circuito decomposto (QPY)": "Download decomposed circuit (QPY)",
    "Dettagli dei tentativi": "Attempt details",
    "Indicatore": "Metric",
    "Originale": "Original",
    "Ottimizzato": "Optimized",
    "Riduzione": "Reduction",
    "Errore massimo": "Maximum error",
    "Tolleranza": "Tolerance",
    "Fedeltà sottospazio": "Subspace fidelity",
    "Circuito ottimizzato e conteggio porte": "Optimized circuit and gate counts",
    "Diagramma completo ottimizzato": "Full optimized diagram",
    "Scarica circuito ottimizzato (QPY)": "Download optimized circuit (QPY)",
    "Scarica diagramma ottimizzato (TXT)": "Download optimized diagram (TXT)",
    "Mostra": "Show",
    "Solo errori": "Errors only",
    "Tutti i campioni": "All samples",
    "Solo corretti": "Correct only",
    "Confidenza": "Confidence",
    "Margine 1ª-2ª": "1st–2nd margin",
    "Scostamento max |circuito-teoria|": "Max |circuit-theory| deviation",
    "Lettura rapida": "Quick reading",
    "Nessun campione in questa categoria.": "No samples in this category.",
    "Dettaglio completo delle probabilità teoriche e circuitali": (
        "Full theoretical and circuit probability details"
    ),
    "Scarica classificazioni dettagliate (CSV)": (
        "Download detailed classifications (CSV)"
    ),
    "Scostamento massimo": "Maximum deviation",
    "Scostamento medio": "Mean deviation",
    "Errore somma probabilità": "Probability-sum error",
    "Accordo circuito/r-PGM": "Circuit/r-PGM agreement",
    "Controllo": "Check",
    "Significato": "Meaning",
    "Isometria V†V = I": "Isometry V†V = I",
    "Unitarietà U†U = I": "Unitarity U†U = I",
    "La POVM è completa": "The POVM is complete",
    "V conserva la norma": "V preserves the norm",
    "U_PGM è una trasformazione quantistica valida": (
        "U_PGM is a valid quantum transformation"
    ),
    "Il circuito riproduce gli effetti F_j": "The circuit reproduces the F_j effects",
    "Ogni distribuzione è normalizzata": "Every distribution is normalized",
    "Campione del test set da eseguire": "Test-set sample to run",
    "Numero di shot": "Number of shots",
    "Seed simulatore/transpiler": "Simulator/transpiler seed",
    "Tipo di risorsa": "Resource type",
    "🧪 Simulatori": "🧪 Simulators",
    "⚛️ Computer quantistici reali": "⚛️ Real quantum computers",
    "Simulatore": "Simulator",
    "💻 PGM ideale locale — campionamento diretto": (
        "💻 Local ideal PGM — direct sampling"
    ),
    "🧩 Qiskit BasicSimulator locale — circuito ideale": (
        "🧩 Local Qiskit BasicSimulator — ideal circuit"
    ),
    "🧰 Qiskit Aer locale — circuito ideale o rumoroso": (
        "🧰 Local Qiskit Aer — ideal or noisy circuit"
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
    "☁️ IonQ Cloud — ideale o modello di rumore": (
        "☁️ IonQ Cloud — ideal or noise model"
    ),
    "☁️ Amazon Braket — simulatore gestito AWS": (
        "☁️ Amazon Braket — AWS-managed simulator"
    ),
    "Esegui campionamento ideale": "Run ideal sampling",
    "Predizione a shot": "Shot-based prediction",
    "Automatico": "Automatic",
    "Matrice densità": "Density matrix",
    "Modello di rumore": "Noise model",
    "Ideale": "Ideal",
    "Depolarizzante configurabile": "Configurable depolarizing noise",
    "Errore gate 1-qubit": "1-qubit gate error",
    "Errore gate 2-qubit": "2-qubit gate error",
    "Errore di lettura": "Readout error",
    "Metodo Aer": "Aer method",
    "Snapshot IBM": "IBM snapshot",
    "Metodo Braket Local": "Braket Local method",
    "Metodo AQT Offline": "AQT Offline method",
    "Rumoroso": "Noisy",
    "Ottimizza, certifica ed esegui sul simulatore selezionato": (
        "Optimize, certify, and run on the selected simulator"
    ),
    "Predizione simulatore": "Simulator prediction",
    "Ottimizza automaticamente ed esegui con Aer": (
        "Optimize automatically and run with Aer"
    ),
    "Confronto automatico dei livelli 1, 2 e 3": (
        "Automatic comparison of levels 1, 2 and 3"
    ),
    "Predizione Aer": "Aer prediction",
    "Provider hardware": "Hardware provider",
    "🔵 IBM Quantum — QPU superconduttive": "🔵 IBM Quantum — superconducting QPUs",
    "🟣 IonQ Quantum Cloud — QPU a ioni intrappolati": (
        "🟣 IonQ Quantum Cloud — trapped-ion QPUs"
    ),
    "Autenticazione AWS": "AWS authentication",
    "Credenziali temporanee AWS": "Temporary AWS credentials",
    "Credenziali protette del deployment": "Protected deployment credentials",
    "Profilo AWS locale / SSO": "Local AWS profile / SSO",
    "Profilo AWS locale": "Local AWS profile",
    "Regione AWS": "AWS region",
    "Verifica identità e aggiorna dispositivi": "Verify identity and refresh devices",
    "Dispositivo Amazon Braket": "Amazon Braket device",
    "Token Munich Quantum Portal": "Munich Quantum Portal token",
    "Mostra soltanto risorse online": "Show online resources only",
    "Accoda se la risorsa diventa offline": "Queue if the resource goes offline",
    "Connetti e aggiorna risorse LRZ": "Connect and refresh LRZ resources",
    "Risorsa LRZ/MQSS": "LRZ/MQSS resource",
    "Modello di rumore IonQ": "IonQ noise model",
    "Connetti e aggiorna risorse IonQ": "Connect and refresh IonQ resources",
    "Backend IonQ": "IonQ backend",
    "AQT access token": "AQT access token",
    "Connetti e aggiorna simulatori AQT": (
        "Connect and refresh AQT simulators"
    ),
    "Simulatore AQT Cloud": "AQT Cloud simulator",
    "IBM Quantum API key (facoltativa se già salvata)": (
        "IBM Quantum API key (optional if already saved)"
    ),
    "CRN istanza IBM (consigliato)": "IBM instance CRN (recommended)",
    "Connetti e aggiorna QPU IBM": "Connect and refresh IBM QPUs",
    "QPU IBM Quantum": "IBM Quantum QPU",
    "Livello scelto": "Selected level",
    "Ottimizza, certifica e prepara il circuito": (
        "Optimize, certify and prepare the circuit"
    ),
    "Messaggi diagnostici": "Diagnostic messages",
    "Mostra il circuito hardware completo": "Show the full hardware circuit",
    "Circuito con state preparation": "Circuit with state preparation",
    "Invia job quantistico": "Submit quantum job",
    "Aggiorna stato": "Refresh status",
    "Scarica risultato": "Download result",
    "Annulla job": "Cancel job",
    "Predizione provider": "Provider prediction",
    "CX / porte entangling": "CX / entangling gates",
    "Gate": "Gate",
    "Scarica diagramma SVG": "Download SVG diagram",
    "Scarica circuito QPY": "Download QPY circuit",
    "Scarica matrici NPZ": "Download NPZ matrices",
    "Scarica le predizioni (CSV)": "Download predictions (CSV)",
    "Confermo di voler inviare il job alla risorsa LRZ selezionata usando la mia allocazione.": (
        "I confirm that I want to submit the job to the selected LRZ resource "
        "using my allocation."
    ),
    "Confermo di voler inviare un task AWS che può usare quota o generare costi.": (
        "I confirm that I want to submit an AWS task that may consume quota or "
        "incur charges."
    ),
    "Confermo di voler inviare un task AWS che può generare costi sul mio account.": (
        "I confirm that I want to submit an AWS task that may incur charges on "
        "my account."
    ),
    "Confermo di voler inviare un job IonQ che può usare quota o generare costi.": (
        "I confirm that I want to submit an IonQ job that may consume quota or "
        "incur charges."
    ),
    "Confermo di voler inviare il job al simulatore AQT Cloud selezionato e di accettarne quota o costi.": (
        "I confirm that I want to submit the job to the selected AQT Cloud "
        "simulator and accept the associated quota usage or charges."
    ),
    "Confermo di voler inviare il job alla QPU IonQ selezionata e di accettarne quota o costi.": (
        "I confirm that I want to submit the job to the selected IonQ QPU and "
        "accept the associated quota usage or charges."
    ),
    "Confermo di voler inviare il job alla QPU IBM selezionata usando la mia istanza.": (
        "I confirm that I want to submit the job to the selected IBM QPU using "
        "my instance."
    ),
    "Medicina": "Medicine",
    "Sintetico": "Synthetic",
    "Botanica": "Botany",
    "Bioinformatica": "Bioinformatics",
    "Materiali": "Materials",
    "Chimica": "Chemistry",
    "Visione artificiale": "Computer vision",
    "Zoologia": "Zoology",
    "Segnali sonar": "Sonar signals",
    "Ingegneria software": "Software engineering",
    "Autenticazione": "Authentication",
    "Donazioni di sangue": "Blood donation",
    "Climatologia": "Climatology",
    "Agricoltura": "Agriculture",
    "Benchmark sintetico": "Synthetic benchmark",
    "Two Moons (sintetico)": "Two Moons (synthetic)",
    "Banana OpenML (sottoinsieme fisso)": "Banana OpenML (fixed subset)",
    "Concentric Circles (sintetico)": "Concentric Circles (synthetic)",
    "Gaussian Blobs, 3 classi (sintetico)": "Gaussian Blobs, 3 classes (synthetic)",
    "XOR rumoroso (sintetico)": "Noisy XOR (synthetic)",
    "Three Spirals (sintetico)": "Three Spirals (synthetic)",
    "Sì": "Yes",
    "No": "No",
    "QPU / RISORSA MQSS": "QPU / MQSS RESOURCE",
    "scikit-learn locale (copia del dataset pubblico UCI)": (
        "local scikit-learn copy of the public UCI dataset"
    ),
    "dispositivo": "device",
    "stato": "status",
    "Il fattore di rescaling deve essere positivo e finito.": (
        "The rescaling factor must be positive and finite."
    ),
    "Fattore di rescaling stereografico mancante.": (
        "The stereographic rescaling factor is missing."
    ),
    "I fattori di rescaling devono essere positivi e finiti.": (
        "Rescaling factors must be positive and finite."
    ),
    "Il miglioramento minimo non può essere negativo.": (
        "The minimum gain cannot be negative."
    ),
    "Le feature aggiunte dall'encoding non possono essere negative.": (
        "The number of features added by the encoding cannot be negative."
    ),
    "Schema della cache dell'esperimento non riconosciuto.": (
        "Unrecognized experiment-cache schema."
    ),
    (
        "Impossibile ricostruire con certezza lo stesso split train/test della PGM. "
        "Ricalcolare l'esperimento."
    ): (
        "The exact PGM train/test split could not be reconstructed safely. "
        "Run the experiment again."
    ),
}


# Long Markdown blocks are matched after whitespace normalization.  Keeping them
# here preserves equations and formatting while providing natural English prose.
NORMALIZED_TRANSLATIONS: dict[str, str] = {
    (
        "Questa sezione è indipendente dal pulsante principale. Se la abiliti, "
        "confronta la PGM con un classificatore standard sullo stesso split "
        "train/test. Il modello standard viene ottimizzato con una ricerca compatta "
        "sul solo training set; la balanced accuracy decide il confronto ed è "
        "accompagnata da precision, recall, F1-score, Kappa di Cohen, coefficiente "
        "di Matthews e ROC-AUC."
    ): (
        "This section is independent of the main run button. When enabled, it "
        "compares the PGM with a standard classifier on the same train/test split. "
        "The standard model is tuned with a compact search using training data only; "
        "balanced accuracy is the primary comparison metric, accompanied by "
        "precision, recall, F1-score, Cohen's kappa, Matthews correlation coefficient, "
        "and ROC-AUC."
    ),
    (
        "Confronto riproducibile tra **c-PGM**, **k-PGM** e **r-PGM (Rc-PGM)**. "
        "La classificazione conserva tutte le feature salvo riduzione PCA richiesta "
        "esplicitamente dall'utente e usa il backend esatto meno oneroso tra k-PGM e "
        "r-PGM. Le prestazioni sono aggregate su più split stratificati mediante media "
        "e deviazione standard. Sul solo training set, l'app sceglie l'encoding e il "
        "fattore di rescaling; quando le dimensioni lo consentono costruisce anche il "
        "circuito quantistico della PGM mediante una dilatazione di Naimark."
    ): (
        "Reproducible comparison of **c-PGM**, **k-PGM**, and **r-PGM (Rc-PGM)**. "
        "Classification retains every feature unless the user explicitly requests "
        "PCA reduction, and it uses the least expensive exact backend between k-PGM "
        "and r-PGM. Performance is aggregated over several stratified splits using "
        "mean and standard deviation. Using training data only, the app selects the "
        "encoding and rescaling factor; when dimensions permit, it also builds the "
        "PGM quantum circuit through a Naimark dilation."
    ),
    (
        "- **c-PGM** costruisce esplicitamente il tensore di dimensione $d^c$. "
        "- **k-PGM** usa il kernel omogeneo $\\langle x,z\\rangle^c$ e una matrice "
        "$N\\times N$. - **r-PGM** usa la base simmetrica minima di dimensione "
        "$d_{sym}=\\binom{d+c-1}{c}$. L'equivalenza teorica riguarda gli score di "
        "classe. Durante l'uso l'app sceglie la diagonalizzazione meno onerosa tra "
        "k-PGM e r-PGM e ricostruisce gli score delle altre formulazioni tramite la "
        "medesima Gram matrix, senza duplicare il costo. La suite numerica verifica "
        "separatamente l'equivalenza delle tre mappe. Il termine di completamento "
        "$P_{ker(\\sigma)}/l$ viene omesso dagli score perché è uguale per ogni classe "
        "e non modifica l'argmax."
    ): (
        "- **c-PGM** explicitly builds the tensor representation of dimension $d^c$.\n"
        "- **k-PGM** uses the homogeneous kernel $\\langle x,z\\rangle^c$ and an "
        "$N\\times N$ matrix.\n"
        "- **r-PGM** uses the minimal symmetric basis of dimension "
        "$d_{sym}=\\binom{d+c-1}{c}$.\n\n"
        "The theoretical equivalence concerns the class scores. At runtime, the app "
        "selects the less expensive eigendecomposition between k-PGM and r-PGM and "
        "reconstructs the other formulations' scores through the same Gram matrix, "
        "without duplicating the cost. The numerical suite independently verifies "
        "the equivalence of the three maps. The completion term "
        "$P_{ker(\\sigma)}/l$ is omitted because it is identical for every class and "
        "does not change the argmax."
    ),
    (
        "L'app usa la PGM ridotta, equivalente alle altre due formulazioni. Completa "
        "gli effetti sul nucleo di $\\sigma$ e costruisce l'isometria canonica $$ "
        "U_{\\mathrm{PGM}}\\bigl(|0\\rangle_{\\mathrm{out}}\\otimes|\\psi\\rangle_{\\mathrm{sys}}\\bigr) "
        "=\\sum_j |j\\rangle_{\\mathrm{out}}\\otimes\\sqrt{F_j}\\,|\\psi\\rangle_{\\mathrm{sys}}, "
        "\\qquad \\Pr(j\\mid\\psi)=\\langle\\psi|F_j|\\psi\\rangle . $$ Il registro "
        "**sys** riceve lo stato test già codificato nella base simmetrica; il registro "
        "**out** parte da zero e la sua misura restituisce la classe."
    ): (
        "The app uses the reduced PGM, which is equivalent to the other two "
        "formulations. It completes the effects on the kernel of $\\sigma$ and builds "
        "the canonical isometry\n\n$$\n"
        "U_{\\mathrm{PGM}}\\bigl(|0\\rangle_{\\mathrm{out}}\\otimes|\\psi\\rangle_{\\mathrm{sys}}\\bigr) "
        "=\\sum_j |j\\rangle_{\\mathrm{out}}\\otimes\\sqrt{F_j}\\,|\\psi\\rangle_{\\mathrm{sys}}, "
        "\\qquad \\Pr(j\\mid\\psi)=\\langle\\psi|F_j|\\psi\\rangle .\n$$\n\n"
        "The **sys** register receives the test state already encoded in the symmetric "
        "basis. The **out** register starts at zero, and measuring it returns the class."
    ),
}


# Ordered substitutions cover dynamic f-strings whose numbers, provider messages,
# class labels, or backend names are only known at runtime.  Longer fragments are
# applied first to prevent partial replacements.
PHRASE_TRANSLATIONS: tuple[tuple[str, str], ...] = (
    ("Scarica la guida completa nella lingua attiva: contiene il percorso passo passo, illustrazioni dell'interfaccia, interpretazione statistica, circuito quantistico, provider, privacy e risoluzione dei problemi.", "Download the complete guide in the active language: it includes the step-by-step workflow, interface illustrations, statistical interpretation, quantum circuit, providers, privacy, and troubleshooting."),
    ("La guida PDF è temporaneamente in aggiornamento. Dettaglio:", "The PDF guide is temporarily being updated. Details:"),
    ("La guida deve essere aggiornata alla versione corrente dell'app.", "The guide must be updated to the current app version."),
    ("Versione della guida non coerente con l'app.", "The guide version does not match the app."),
    ("File della guida non disponibile.", "Guide file unavailable."),
    ("Trascina qui un file tabellare. L'app riconosce CSV, TSV/TXT ed Excel XLSX, propone automaticamente il target e usa soltanto le feature numeriche selezionate. Il file resta nella memoria della tua sessione: non viene salvato nel repository né condiviso con altri utenti.", "Drag a tabular file here. The app accepts CSV, TSV/TXT, and Excel XLSX, automatically suggests the target, and uses only the selected numeric features. The file remains in your session memory: it is neither saved to the repository nor shared with other users."),
    ("Massimo 25 MB, 50.000 righe e 500 colonne. La prima riga deve contenere i nomi delle colonne; per Excel viene letta la prima scheda.", "Maximum 25 MB, 50,000 rows, and 500 columns. The first row must contain column names; for Excel, the first sheet is read."),
    ("Il target contiene le etichette delle classi. Puoi correggere la scelta proposta prima di aggiungere il dataset.", "The target contains the class labels. You can correct the suggested choice before adding the dataset."),
    ("Le colonne testuali, vuote o costanti non sono utilizzabili. Gli identificativi univoci sono disponibili ma deselezionati per impostazione predefinita.", "Text, empty, or constant columns cannot be used. Unique identifiers remain available but are deselected by default."),
    ("I valori mancanti delle feature saranno imputati usando soltanto il training set di ciascun seed. Le righe senza target vengono rimosse prima dello split.", "Missing feature values are imputed using only the training set for each seed. Rows without a target are removed before splitting."),
    ("Colonne non numeriche escluse:", "Excluded non-numeric columns:"),
    ("Possibili identificativi deselezionati automaticamente:", "Possible identifiers automatically deselected:"),
    ("In genere non contengono informazione generalizzabile.", "They generally do not contain generalizable information."),
    ("Colonne vuote o costanti ignorate:", "Ignored empty or constant columns:"),
    ("Dataset personale disponibile:", "Private dataset available:"),
    ("Il file non può essere aggiunto in modo sicuro. Dettaglio:", "The file cannot be added safely. Details:"),
    ("Upload privato in memoria:", "Private in-memory upload:"),
    ("Il file caricato è vuoto.", "The uploaded file is empty."),
    ("Il file supera il limite di", "The file exceeds the limit of"),
    ("Formato non supportato. Usa CSV, TSV, TXT oppure XLSX.", "Unsupported format. Use CSV, TSV, TXT, or XLSX."),
    ("La codifica testuale del file non è riconosciuta.", "The file's text encoding is not recognized."),
    ("Non è stato riconosciuto un separatore tra le colonne.", "No column delimiter could be detected."),
    ("Il file non contiene una tabella valida.", "The file does not contain a valid table."),
    ("Il file non contiene righe utilizzabili.", "The file contains no usable rows."),
    ("Servono almeno una feature e una colonna target.", "At least one feature and one target column are required."),
    ("Il file contiene", "The file contains"),
    ("righe; il limite è", "rows; the limit is"),
    ("colonne; il limite è", "columns; the limit is"),
    ("Una o più colonne non hanno un nome.", "One or more columns have no name."),
    ("I nomi delle colonne devono essere univoci. Duplicati:", "Column names must be unique. Duplicates:"),
    ("Il file Excel contiene troppi elementi interni.", "The Excel file contains too many internal items."),
    ("Il contenuto Excel decompresso supera il limite di sicurezza.", "The uncompressed Excel content exceeds the safety limit."),
    ("Il file XLSX non è un archivio Excel valido.", "The XLSX file is not a valid Excel archive."),
    ("Non è stato possibile leggere la tabella. Controlla intestazioni, separatore e formato. Dettaglio:", "The table could not be read. Check headers, delimiter, and format. Details:"),
    ("La colonna target selezionata non esiste.", "The selected target column does not exist."),
    ("La colonna target non contiene alcuna etichetta valida.", "The target column contains no valid labels."),
    ("Inserisci un nome per il dataset.", "Enter a name for the dataset."),
    ("Seleziona almeno una feature numerica.", "Select at least one numeric feature."),
    ("Feature non presenti nel file:", "Features not found in the file:"),
    ("La colonna target non può essere usata come feature.", "The target column cannot be used as a feature."),
    ("Dopo la pulizia non rimane alcuna feature numerica variabile.", "No varying numeric feature remains after cleaning."),
    ("Feature diventate vuote o costanti dopo la rimozione delle righe senza target:", "Features that became empty or constant after removing rows without a target:"),
    ("Il target deve contenere almeno due classi.", "The target must contain at least two classes."),
    ("Il target contiene", "The target contains"),
    ("classi; il limite è", "classes; the limit is"),
    ("Ogni classe deve avere almeno", "Each class must have at least"),
    ("campioni per gli split stratificati. Classi insufficienti:", "samples for stratified splits. Classes with insufficient samples:"),
    ("Il dataset è troppo piccolo per una valutazione train/test affidabile.", "The dataset is too small for a reliable train/test evaluation."),
    ("Confronto non avviato: la rappresentazione esatta supera il budget computazionale selezionato.", "Comparison not started: the exact representation exceeds the selected computation budget."),
    ("Le etichette del target non possono superare 120 caratteri.", "Target labels cannot exceed 120 characters."),
    ("Disattivata per impostazione predefinita: il PGM usa tutte le feature. Se la attivi, scegli tu quante componenti mantenere; la PCA viene appresa esclusivamente sul training set di ciascun seed.", "Disabled by default: the PGM uses every feature. If enabled, you choose how many components to retain; PCA is fitted exclusively on each seed's training set."),
    ("Ogni seed genera un nuovo split stratificato. Sono riportate media e deviazione standard; nessun seed viene scartato.", "Each seed creates a new stratified split. Mean and standard deviation are reported; no seed is discarded."),
    ("Sono proposte soltanto quote che consentono allo split stratificato di rappresentare ogni classe sia nel training sia nel test set.", "Only fractions that allow the stratified split to represent every class in both the training and test sets are offered."),
    ("Nessuna feature selection automatica: il PGM usa tutte le", "No automatic feature selection: the PGM uses all"),
    ("feature originali. L'encoding stereografico aggiunge soltanto la propria coordinata geometrica.", "original features. Stereographic encoding only adds its geometric coordinate."),
    ("Riduzione richiesta dall'utente: la PCA conserva", "User-requested reduction: PCA retains"),
    ("feature. Viene adattata separatamente sul solo training set di ciascun seed; il test set non partecipa mai alla selezione.", "features. It is fitted separately on each seed's training set only; the test set never participates in selection."),
    ("La classificazione resta disponibile senza ridurre le feature. Se desideri anche il circuito esatto, puoi richiedere esplicitamente una PCA manuale e scegliere il numero di componenti.", "Classification remains available without reducing features. If you also need the exact circuit, you can explicitly request manual PCA and choose the component count."),
    ("Per ottenere il circuito esatto esportabile puoi richiedere manualmente una PCA con meno componenti oppure ridurre il numero di copie.", "To obtain the exact exportable circuit, you can manually request PCA with fewer components or reduce the number of copies."),
    ("Backend esatto previsto:", "Expected exact backend:"),
    ("Nessuna feature viene eliminata automaticamente.", "No feature is removed automatically."),
    ("Picco prudenziale stimato:", "Conservative peak estimate:"),
    ("Il backend esatto meno oneroso", "The least expensive exact backend"),
    ("supera il limite prudenziale selezionato. Aumenta il budget soltanto se il computer dispone realmente della RAM, oppure richiedi manualmente una riduzione PCA.", "exceeds the selected conservative limit. Increase the budget only if the computer actually has that RAM, or manually request PCA reduction."),
    ("Valutazione PGM su", "PGM evaluation across"),
    ("seed in corso...", "seeds in progress..."),
    ("Questa sezione è indipendente dal pulsante principale. Se la abiliti, confronta la PGM con un classificatore standard sugli stessi split stratificati e sugli stessi seed. Il modello standard viene ottimizzato con una ricerca compatta sul solo training set; per ogni metrica vengono riportate media e deviazione standard.", "This section is independent of the main run button. When enabled, it compares the PGM with a standard classifier on the same stratified splits and seeds. The standard model is tuned with a compact search on training data only; mean and standard deviation are reported for every metric."),
    ("Nel confronto, PGM indica la configurazione scelta automaticamente sul training set (encoding e, se applicabile, fattore t). c-PGM, k-PGM e r-PGM hanno la stessa decisione teorica; il calcolo usa il backend esatto meno oneroso tra k-PGM e r-PGM, senza riduzione automatica delle feature.", "In this comparison, PGM denotes the configuration selected automatically on training data (encoding and, when applicable, factor t). c-PGM, k-PGM, and r-PGM have the same theoretical decision; computation uses the least expensive exact backend between k-PGM and r-PGM, without automatic feature reduction."),
    ("Confronto appaiato su", "Paired comparison across"),
    ("Il confronto corrente usa", "The current comparison uses"),
    ("seed e fino a 3 fold di tuning interni per ciascun training set. Il full comparison usa gli stessi seed e 2 fold per contenere i tempi; i download OpenML sono memorizzati in cache.", "seeds and up to 3 internal tuning folds for each training set. The full comparison uses the same seeds and 2 folds to limit runtime; OpenML downloads are cached."),
    ("L'intervallo appaiato multi-seed della differenza di balanced accuracy è interamente positivo.", "The paired multi-seed interval for the balanced-accuracy difference is entirely positive."),
    ("L'intervallo appaiato multi-seed della differenza di balanced accuracy è interamente negativo.", "The paired multi-seed interval for the balanced-accuracy difference is entirely negative."),
    ("l'intervallo appaiato della differenza include zero, quindi non viene dichiarato un vincitore.", "the paired interval for the difference includes zero, so no winner is declared."),
    ("Media ± deviazione standard su", "Mean ± standard deviation over"),
    ("split stratificati appaiati. Ogni modello è ottimizzato esclusivamente sul training del relativo seed; nessun risultato viene scartato.", "paired stratified splits. Each model is tuned exclusively on the corresponding seed's training set; no result is discarded."),
    ("media ± dev. std.", "mean ± std. dev."),
    ("seed di riferimento", "reference seed"),
    ("Valutazione esterna su", "Outer evaluation over"),
    ("split stratificati", "stratified splits"),
    ("Sono inclusi tutti i risultati; lo split con seed", "All results are included; the split with seed"),
    ("Sono inclusi tutti i risultati; le statistiche sono comuni a c-PGM, k-PGM e r-PGM perché le predizioni sono esattamente equivalenti.", "All results are included; the statistics are shared by c-PGM, k-PGM, and r-PGM because their predictions are exactly equivalent."),
    ("Lo split con seed", "The split with seed"),
    ("è usato sotto soltanto per matrici di confusione, dettaglio dei campioni e circuito.", "is used below only for confusion matrices, sample details, and the circuit."),
    ("Backend adattivo attivo:", "Adaptive backend active:"),
    ("Una formulazione calcolata esplicitamente non coincide con il backend esatto scelto su tutti i campioni.", "An explicitly computed formulation does not match the selected exact backend on every sample."),
    ("equivalente esatta via", "exactly equivalent via"),
    ("matrice non materializzata", "matrix not materialized"),
    ("è stato calcolato direttamente;", "was computed directly;"),
    ("sono stati valutati mediante l'identità esatta delle Gram matrix con", "were evaluated through the exact Gram-matrix identity with"),
    ("senza costruire le rispettive matrici. Gli zeri negli scarti che coinvolgono questi metodi derivano quindi dall'equivalenza matematica, non da tre diagonalizzazioni duplicate.", "without constructing their matrices. Zero discrepancies involving these methods therefore come from mathematical equivalence, not from three duplicate eigendecompositions."),
    ("WIN o LOSS sono assegnati soltanto quando l'intervallo Student-t al 95% delle differenze appaiate tra seed esclude zero; negli altri casi il risultato è TIE. Le celle riportano medie calcolate su tutti i seed.", "WIN or LOSS is assigned only when the 95% Student-t interval of paired seed differences excludes zero; otherwise the result is TIE. Cells report means over all seeds."),
    ("Una vittoria è confermata soltanto quando l'intervallo Student-t al 95% delle differenze appaiate tra seed esclude zero. 'Non conclusivo' non significa che i risultati sono uguali: indica evidenza insufficiente per dichiarare un vincitore. 'Pareggio esatto' è riservato ai casi in cui le balanced accuracy coincidono su ogni seed entro la tolleranza numerica. Le percentuali mostrano in quanti seed ciascun metodo è risultato superiore.", "A win is confirmed only when the 95% Student-t interval of paired seed differences excludes zero. ‘Inconclusive’ does not mean the results are equal: it means there is insufficient evidence to declare a winner. ‘Exact tie’ is reserved for cases where balanced accuracy matches on every seed within numerical tolerance. Percentages show on how many seeds each method scored higher."),
    ("Le barre mostrano la differenza media in punti percentuali; il segmento nero è l'intervallo di confidenza appaiato al 95%. Il giallo indica un esito statisticamente non conclusivo; direzione e ampiezza della differenza restano visibili e non vengono chiamate pareggio.", "Bars show the mean difference in percentage points; the black segment is the paired 95% confidence interval. Yellow indicates a statistically inconclusive outcome; the direction and size of the difference remain visible and are not called a tie."),
    ("Un esito complessivo non conclusivo non è un pareggio: indica che l'intervallo di confidenza della differenza include zero.", "An overall inconclusive outcome is not a tie: it means the confidence interval for the difference includes zero."),
    ("Confronto split per split:", "Split-by-split comparison:"),
    ("PGM ottiene una balanced accuracy più alta nel", "PGM has higher balanced accuracy on"),
    ("e i valori sono esattamente uguali nel", "and the values are exactly equal on"),
    ("Tra i", "Among the"),
    ("risultati non conclusivi,", "inconclusive results,"),
    ("sono pareggi esatti su ogni seed.", "are exact ties on every seed."),
    ("Vittoria ", "Confirmed "),
    (" confermata", " win"),
    (" superiore", " higher"),
    ("Nel confronto, PGM indica la configurazione scelta automaticamente sul training set (encoding e, se applicabile, fattore t). c-PGM, k-PGM e r-PGM hanno la stessa decisione teorica; per le metriche viene usata k-PGM.", "In this comparison, PGM denotes the configuration selected automatically on the training set (encoding and, when applicable, factor t). c-PGM, k-PGM, and r-PGM have the same theoretical decision; k-PGM is used to compute the metrics."),
    ("· PGM equivalenti, benchmark statistici e integrazione quantistica protetta", "· equivalent PGMs, statistical benchmarks, and protected quantum integration"),
    ("Multi-Layer Perceptron feed-forward con regolarizzazione.", "Regularized feed-forward Multi-Layer Perceptron."),
    ("Foresta di alberi con pesi bilanciati e aggregazione robusta.", "Tree ensemble with balanced weights and robust aggregation."),
    ("Modello bayesiano su feature binarizzate dopo scaling train-only.", "Bayesian model on features binarized after train-only scaling."),
    ("Classificazione per vicinato con distanze standardizzate.", "Nearest-neighbor classification with standardized distances."),
    ("Frontiere quadratiche con regolarizzazione della covarianza.", "Quadratic decision boundaries with covariance regularization."),
    ("Modello lineare probabilistico con bilanciamento delle classi.", "Probabilistic linear model with class balancing."),
    ("Singolo albero estremamente randomizzato.", "Single extremely randomized tree."),
    ("Ensemble di alberi estremamente randomizzati.", "Ensemble of extremely randomized trees."),
    ("Support Vector Machine non lineare con kernel gaussiano.", "Nonlinear Support Vector Machine with a Gaussian kernel."),
    ("Support Vector Machine lineare con classi bilanciate.", "Linear Support Vector Machine with balanced classes."),
    ("Boosting istogrammico rapido con regolarizzazione.", "Fast regularized histogram-based boosting."),
    ("Boosting classico di alberi decisionali.", "Classic decision-tree boosting."),
    ("Discriminante lineare con shrinkage della covarianza.", "Linear discriminant analysis with covariance shrinkage."),
    ("Baseline probabilistica gaussiana veloce.", "Fast Gaussian probabilistic baseline."),
    ("Ensemble adattivo di classificatori deboli.", "Adaptive ensemble of weak classifiers."),
    ("Albero CART con bilanciamento delle classi.", "CART tree with class balancing."),
    ("Classificatore lineare regolarizzato, rapido e stabile.", "Fast, stable regularized linear classifier."),
    ("Nessun benchmark viene eseguito finché non abiliti questa sezione e premi uno dei due pulsanti.", "No benchmark runs until you enable this section and press one of its two buttons."),
    ("Il confronto sul dataset corrente usa fino a 3 fold di tuning. Il full comparison usa 2 fold per contenere i tempi e include tutti i dataset binari del catalogo; i download OpenML sono memorizzati in cache.", "The current-dataset comparison uses up to 3 tuning folds. The full comparison uses 2 folds to limit runtime and includes every binary dataset in the catalogue; OpenML downloads are cached."),
    ("Training della PGM, tuning del classificatore e bootstrap appaiato in corso...", "Training the PGM, tuning the classifier, and running the paired bootstrap..."),
    ("Le impostazioni del confronto sono cambiate: premi il pulsante per calcolare il nuovo caso.", "Comparison settings have changed; press the button to compute the new case."),
    ("Le impostazioni del full comparison sono cambiate: premi il pulsante per ricalcolarlo.", "Full-comparison settings have changed; press the button to recompute it."),
    ("Vincitore: PGM su", "Winner: PGM on"),
    ("Vincitore:", "Winner:"),
    ("L'intervallo bootstrap appaiato della differenza di balanced accuracy è interamente positivo.", "The paired bootstrap interval for the balanced-accuracy difference is entirely positive."),
    ("L'intervallo bootstrap appaiato della differenza di balanced accuracy è interamente negativo.", "The paired bootstrap interval for the balanced-accuracy difference is entirely negative."),
    ("Confronto non conclusivo su", "Inconclusive comparison on"),
    ("l'intervallo bootstrap della differenza include zero, quindi non viene dichiarato un vincitore.", "the bootstrap interval for the difference includes zero, so no winner is declared."),
    ("Il vincitore è dichiarato soltanto quando l'intervallo bootstrap stratificato e appaiato al 95% non include zero. Tutte le metriche sono calcolate sul medesimo test set, mai usato per il tuning.", "A winner is declared only when the paired stratified 95% bootstrap interval excludes zero. Every metric is computed on the same test set, which is never used for tuning."),
    ("WIN o LOSS sono assegnati soltanto quando l'intervallo bootstrap appaiato al 95% della differenza di balanced accuracy esclude zero; negli altri casi il risultato è TIE.", "WIN or LOSS is assigned only when the paired 95% bootstrap interval for the balanced-accuracy difference excludes zero; otherwise the result is TIE."),
    ("Il full comparison non ha prodotto risultati utilizzabili. Consulta i dettagli degli errori.", "The full comparison produced no usable results. Review the error details."),
    ("Dataset non completati", "Datasets not completed"),
    ("Vittorie", "Wins"),
    ("Classificatore sconosciuto:", "Unknown classifier:"),
    ("Catalogo dei classificatori incoerente.", "Inconsistent classifier catalogue."),
    ("Nessuna configurazione ha prodotto una validazione finita.", "No configuration produced a finite validation score."),
    ("Tuning non disponibile; usata la configurazione robusta predefinita. Dettaglio:", "Tuning was unavailable; the robust default configuration was used. Details:"),
    ("Tuning disattivato perché una classe ha troppo pochi campioni nel training; usata la configurazione robusta predefinita.", "Tuning was disabled because one class has too few training samples; the robust default configuration was used."),
    ("Il bootstrap richiede almeno 100 ricampionamenti.", "Bootstrap requires at least 100 resamples."),
    ("Il livello di confidenza deve essere tra 0.5 e 1.", "The confidence level must be between 0.5 and 1."),
    ("Le predizioni da confrontare devono avere la stessa forma.", "The predictions being compared must have the same shape."),
    ("· classificatori equivalenti, circuito completo e integrazione quantistica protetta", "· equivalent classifiers, full circuit, and protected quantum integration"),
    ("La distanza è la total variation distance fra frequenze osservate e probabilità teoriche: valori più vicini a zero indicano maggiore accordo.", "The distance is the total variation distance between observed frequencies and theoretical probabilities: values closer to zero indicate better agreement."),
    ("È un limite di sicurezza dell'app, non aumenta la RAM fisicamente disponibile sul server.", "This is an app safety limit; it does not increase the RAM physically available on the server."),
    ("La modalità automatica usa tutte le feature quando possibile; altrimenti applica una PCA appresa solo sul training set, così restano eseguibili c-PGM, k-PGM, r-PGM e il circuito.", "Automatic mode uses all features whenever possible; otherwise it applies PCA fitted only on the training set so c-PGM, k-PGM, r-PGM, and the circuit remain executable."),
    ("Riduzione automatica quantum-ready: la baseline usa", "Automatic quantum-ready reduction: the baseline uses"),
    ("feature di base; l'encoding stereografico usa", "base features; stereographic encoding uses"),
    ("feature di base più una coordinata stereografica.", "base features plus one stereographic coordinate."),
    ("Ogni PCA e ogni confronto vengono appresi esclusivamente sul training set.", "Every PCA transform and every comparison are fitted exclusively on the training set."),
    ("La configurazione scelta sarà mostrata dopo il calcolo.", "The selected configuration will be shown after computation."),
    ("Nessuna PCA necessaria: entrambi i candidati usano tutte le feature originali; l'encoding stereografico aggiunge una coordinata.", "No PCA is needed: both candidates use all original features; stereographic encoding adds one coordinate."),
    ("I prior uniformi seguono l'Eq. (4). Per classi sbilanciate il k-PGM usa il Gram pesato, così resta esattamente equivalente ai due metodi primali.", "Uniform priors follow Eq. (4). For imbalanced classes, k-PGM uses the weighted Gram matrix and remains exactly equivalent to the two primal methods."),
    ("Gli autovalori <= soglia x lambda_max sono esclusi in tutti e tre i metodi.", "Eigenvalues <= threshold × lambda_max are discarded in all three methods."),
    ("Riduzione automatica necessaria:", "Automatic reduction required:"),
    ("feature mediante PCA train-only. La percentuale di varianza conservata verrà mostrata dopo il calcolo.", "features using train-only PCA. The retained variance will be shown after computation."),
    ("Nessuna riduzione necessaria: questa configurazione usa tutte le feature originali.", "No reduction is needed: this configuration uses all original features."),
    ("Circuito quantistico previsto:", "Expected quantum circuit:"),
    ("qubit di sistema +", "system qubits +"),
    ("qubit di esito =", "outcome qubits ="),
    ("qubit. La dilatazione unitaria esatta verrà costruita dopo il training.", "qubits. The exact unitary dilation will be built after training."),
    ("qubit e matrice densa", "qubits and dense matrix"),
    ("La matrice esatta supera il limite prudenziale di", "The exact matrix exceeds the conservative limit of"),
    ("La classificazione classica resta disponibile; seleziona la modalità automatica quantum-ready soltanto se vuoi anche il circuito esatto, la validazione circuitale e l'esecuzione quantistica.", "Classical classification remains available. Select automatic quantum-ready mode only if you also need the exact circuit, circuit validation, and quantum execution."),
    ("Le formule e la memoria asintotica seguono le Tabelle 1, 2 e 5 del paper.", "The formulas and asymptotic memory follow Tables 1, 2, and 5 of the paper."),
    ("I proxy sono conteggi dei termini dominanti, non FLOP misurati.", "The proxies count dominant terms; they are not measured FLOPs."),
    ("Prima del run, per k-PGM si usa il limite superiore r_G=N; dopo il run compare il rank effettivo.", "Before execution, k-PGM uses the upper bound r_G=N; after execution, the effective rank is shown."),
    ("Le dimensioni mostrate usano, in modo prudenziale, il candidato di encoding più grande.", "The displayed dimensions conservatively use the larger encoding candidate."),
    ("Dopo il run saranno ricalcolate sulla configurazione selezionata.", "After execution, they are recalculated for the selected configuration."),
    ("Configurazione eseguibile con il limite prudenziale dell'app.", "Configuration executable within the app's conservative limit."),
    ("Picco NumPy stimato:", "Estimated NumPy peak:"),
    ("Le rappresentazioni primali complete sono troppo grandi da materializzare, ma la classificazione è disponibile in modalità scalabile con tutte le feature originali.", "The full primal representations are too large to materialize, but classification remains available in scalable mode with all original features."),
    ("Il k-PGM viene calcolato direttamente; c-PGM e r-PGM vengono eseguiti anche in forma esplicita quando la loro dimensione è <=", "k-PGM is computed directly; c-PGM and r-PGM are also run explicitly when their dimension is <="),
    (", altrimenti sono valutati tramite lo stesso kernel esatto ⟨x,z⟩^c.", "; otherwise they are evaluated through the same exact kernel ⟨x,z⟩^c."),
    ("L'interfaccia distinguerà chiaramente i metodi materializzati da quelli equivalenti via kernel.", "The interface clearly distinguishes materialized methods from kernel-equivalent ones."),
    ("Anche il calcolo kernel supera il limite prudenziale selezionato:", "The kernel computation also exceeds the selected conservative limit:"),
    ("Usa la modalità automatica quantum-ready oppure aumenta il budget solo se il computer dispone realmente di quella RAM.", "Use automatic quantum-ready mode, or increase the budget only if the computer actually has that amount of RAM."),
    ("La tabella di complessità rimane comunque valida.", "The complexity table remains valid."),
    ("Download/cache del dataset e calcolo dei tre PGM in corso...", "Downloading/caching the dataset and computing the three PGMs..."),
    ("Selezione automatica completata: è stato scelto l'encoding stereografico con fattore di rescaling", "Automatic selection completed: stereographic encoding was selected with rescaling factor"),
    ("Questa configurazione viene usata nel training finale, nei tre classificatori, nel circuito e nelle eventuali esecuzioni su simulatore o QPU.", "This configuration is used for final training, all three classifiers, the circuit, and any simulator or QPU execution."),
    ("Selezione automatica completata: è stato scelto l'encoding in ampiezza con normalizzazione L2.", "Automatic selection completed: amplitude encoding with L2 normalization was selected."),
    ("Nessun candidato stereografico ha superato la baseline oltre la soglia minima richiesta.", "No stereographic candidate exceeded the baseline by more than the required minimum threshold."),
    ("La scelta è stata effettuata esclusivamente sul training set mediante", "Selection used training data exclusively, with"),
    ("3-fold stratificata sul training", "3-fold stratified cross-validation on training data"),
    ("holdout stratificato 80/20 sul training", "a stratified 80/20 holdout on training data"),
    ("holdout stratificato adattivo sul training (classi rare protette)", "an adaptive stratified holdout on training data (rare classes protected)"),
    ("Il test set non è stato consultato.", "The test set was not inspected."),
    ("Per evitare una scelta dovuta al rumore, lo stereografico deve migliorare la baseline di oltre", "To avoid a noise-driven choice, stereographic encoding must improve on the baseline by more than"),
    ("punti percentuali.", "percentage points."),
    ("punti %", "pp"),
    ("Il valutatore indicato è la rappresentazione PGM esatta meno costosa per quella fold.", "The listed evaluator is the least expensive exact PGM representation for that fold."),
    ("c-PGM, k-PGM e r-PGM hanno gli stessi score teorici; il training finale continua comunque a verificarli tutti e tre.", "c-PGM, k-PGM, and r-PGM have the same theoretical scores; final training still verifies all three."),
    ("Classificazione completata con tutte le feature originali: le tre formulazioni restituiscono le stesse predizioni sui", "Classification completed with all original features: the three formulations return identical predictions for all"),
    ("campioni di test.", "test samples."),
    ("Una formulazione calcolata esplicitamente non coincide con il risultato kernel su tutti i campioni.", "An explicitly evaluated formulation does not match the kernel result on every sample."),
    ("Il caso può essere numericamente quasi degenere: consulta la diagnostica.", "The case may be numerically near-degenerate; review the diagnostics."),
    ("Calcolo scalabile attivo.", "Scalable computation is active."),
    ("sono stati calcolati direttamente e indipendentemente;", "were computed directly and independently;"),
    ("sono stati valutati mediante l'identità esatta delle Gram matrix con k-PGM, senza costruire le rispettive matrici primali.", "were evaluated using the exact Gram-matrix identity with k-PGM, without building their primal matrices."),
    ("Gli zeri negli scarti che coinvolgono questi metodi derivano quindi dall'equivalenza usata, non da tre esecuzioni indipendenti.", "Zero discrepancies involving these methods therefore follow from the equivalence used, not from three independent executions."),
    ("Verifica superata: le predizioni dei tre metodi sono identiche su tutti i", "Check passed: all three methods give identical predictions for all"),
    ("Scarto massimo tra score:", "Maximum score difference:"),
    ("Le predizioni non coincidono tutte.", "The predictions do not all match."),
    ("Controlla la soglia spettrale e gli scarti tra score: il caso può essere numericamente quasi degenere.", "Check the spectral threshold and score discrepancies; the case may be numerically near-degenerate."),
    ("Dataset caricato da:", "Dataset loaded from:"),
    ("Preprocessing finale: imputazione mediana,", "Final preprocessing: median imputation,"),
    (", min-max [0.001, 1] e ", ", min-max [0.001, 1], and "),
    ("mappa stereografica con", "stereographic map with"),
    ("seguita dalla preparazione in ampiezza", "followed by amplitude preparation"),
    ("normalizzazione L2 e preparazione in ampiezza", "L2 normalization and amplitude preparation"),
    ("Ogni trasformazione dipendente dai dati e la scelta dell'encoding sono apprese solo sul training set.", "Every data-dependent transformation and the encoding selection are fitted only on the training set."),
    ("Preprocessing: imputazione mediana,", "Preprocessing: median imputation,"),
    (", min-max [0.001, 1] e normalizzazione L2; ogni trasformazione dipendente dai dati è appresa solo sul training set.", ", min-max [0.001, 1], and L2 normalization; every data-dependent transformation is fitted only on the training set."),
    ("#### Classificazione test (calcolo classico scalabile)", "#### Test classification (scalable classical computation)"),
    ("Questa valutazione non richiede la costruzione del circuito.", "This evaluation does not require constructing the circuit."),
    ("Ogni riga confronta la classe reale con la predizione comune a c-PGM, k-PGM e r-PGM.", "Each row compares the actual class with the prediction shared by c-PGM, k-PGM, and r-PGM."),
    ("**Matrice di confusione — campioni**", "**Confusion matrix — samples**"),
    ("**Matrice di confusione — % per classe reale**", "**Confusion matrix — % by actual class**"),
    ("#### Complessità con i valori effettivi del run", "#### Complexity with actual run values"),
    ("I tempi osservati dipendono da BLAS, CPU, cache e carico della macchina;", "Observed times depend on BLAS, CPU, cache, and machine load;"),
    ("le classi di complessità descrivono invece la crescita asintotica del paper.", "complexity classes instead describe the asymptotic growth analyzed in the paper."),
    ("L'implementazione usa decomposizioni a rank ridotto e non materializza le N matrici densità,", "The implementation uses reduced-rank decompositions and does not materialize the N density matrices,"),
    ("quindi la colonna 'Stato modello' non coincide con il bound di memoria della costruzione didattica del paper.", "so the ‘Model state’ column does not equal the memory bound of the paper's pedagogical construction."),
    ("Classe:", "Class:"),
    ("Esito non usato (probabilità teorica zero)", "Unused outcome (zero theoretical probability)"),
    ("Costruzione della POVM e della dilatazione unitaria esatta...", "Building the POVM and exact unitary dilation..."),
    ("Circuito esatto costruito: la matrice U_PGM contiene i parametri appresi nel training, non è un blocco puramente illustrativo.", "Exact circuit built: the U_PGM matrix contains the parameters learned during training; it is not merely an illustrative block."),
    ("Circuito logico esatto: ingresso codificato su sys, ancilla |0...0> su out, dilatazione di Naimark e misura dell'esito.", "Exact logical circuit: encoded input on sys, |0...0> ancilla on out, Naimark dilation, and outcome measurement."),
    ("U_PGM è mostrato come un'unica porta unitaria per mantenere il diagramma logico leggibile.", "U_PGM is shown as a single unitary gate to keep the logical diagram readable."),
    ("Sotto puoi sintetizzarlo nella base generica rz/sx/x/cx e vedere il circuito completo su più righe.", "Below, you can synthesize it in the generic rz/sx/x/cx basis and view the full circuit over multiple lines."),
    ("Il limite CNOT è una stima prudenziale pre-sintesi per un'unitaria arbitraria.", "The CNOT bound is a conservative pre-synthesis estimate for an arbitrary unitary."),
    ("I conteggi esatti compaiono dopo la transpilation.", "Exact counts appear after transpilation."),
    ("Sintesi in un processo protetto: l'interfaccia non può più essere chiusa da un crash nativo...", "Synthesis in an isolated process: a native crash can no longer terminate the interface..."),
    ("Decomposizione completata senza coinvolgere il processo Streamlit.", "Decomposition completed outside the Streamlit process."),
    ("**Conteggio esatto delle porte**", "**Exact gate counts**"),
    ("**Circuito completo** — scorri verticalmente e orizzontalmente; nessuna porta è omessa.", "**Full circuit** — scroll vertically and horizontally; no gate is omitted."),
    ("La sintesi è riuscita e i conteggi sopra sono esatti, ma il circuito supera", "Synthesis succeeded and the counts above are exact, but the circuit exceeds"),
    ("porte: il disegno integrale sarebbe poco utilizzabile.", "gates, so a full drawing would not be practical."),
    ("Il crash è rimasto confinato nel processo di sintesi: Streamlit e tutti i risultati del training restano attivi.", "The crash was confined to the synthesis process: Streamlit and all training results remain active."),
    ("#### Ottimizzazione certificata", "#### Certified optimization"),
    ("L'ottimizzazione sintetizza direttamente l'isometria di Naimark, confronta tre strategie e conserva quella con meno porte entangling.", "Optimization directly synthesizes the Naimark isometry, compares three strategies, and retains the one with the fewest entangling gates."),
    ("Il risultato viene accettato soltanto se l'azione coincide con U_PGM su ogni ingresso valido, non soltanto sui campioni del test set.", "A result is accepted only if its action matches U_PGM for every valid input, not only for test-set samples."),
    ("Fuori dal sottospazio con il registro out inizializzato a zero le due estensioni unitarie possono differire: quella parte non è mai utilizzata dalla PGM.", "Outside the subspace where the out register is initialized to zero, the two unitary extensions may differ; that part is never used by the PGM."),
    ("Ottimizza e certifica equivalenza", "Optimize and certify equivalence"),
    ("Sintesi originale, ricerca del circuito più corto e certificazione numerica in processi isolati...", "Original synthesis, shortest-circuit search, and numerical certification in isolated processes..."),
    ("Non è stato possibile sintetizzare il circuito originale di riferimento.", "The original reference circuit could not be synthesized."),
    ("Nessuna ottimizzazione è stata dichiarata equivalente.", "No optimization was declared equivalent."),
    ("Nessuno dei tentativi ha superato la certificazione.", "None of the attempts passed certification."),
    ("Il circuito originale resta invariato e utilizzabile.", "The original circuit remains unchanged and usable."),
    ("Equivalenza certificata sull'intero sottospazio PGM.", "Equivalence certified over the full PGM subspace."),
    ("Classi, probabilità teoriche e accuratezza restano invariate entro la tolleranza indicata.", "Classes, theoretical probabilities, and accuracy remain unchanged within the stated tolerance."),
    ("Migliore sintesi: livello", "Best synthesis: level"),
    ("Criterio: minimo numero di porte entangling, poi profondità e porte totali.", "Criterion: fewest entangling gates, then depth, then total gates."),
    ("Il circuito supera la soglia grafica; i conteggi e la certificazione restano completi.", "The circuit exceeds the drawing threshold; counts and certification remain complete."),
    ("La sintesi completa non viene avviata automaticamente oltre", "Full synthesis is not started automatically above"),
    ("qubit: il limite prudenziale è circa", "qubits: the conservative bound is approximately"),
    ("CNOT, oltre alle rotazioni a un qubit.", "CNOTs, in addition to single-qubit rotations."),
    ("La transpilation per uno specifico backend fornirà i conteggi effettivi solo se il circuito supera il preflight.", "Transpilation for a specific backend will provide actual counts only if the circuit passes preflight."),
    ("##### Corrispondenza qubit misurati → classe", "##### Measured qubits → class mapping"),
    ("→ classe", "→ class"),
    ("La bitstring è il valore misurato nel registro di uscita `out`:", "The bitstring is the value measured in the `out` register:"),
    ("per esempio `000` identifica la prima classe elencata.", "for example, `000` identifies the first listed class."),
    ("Gli eventuali stati binari eccedenti non sono assegnati e hanno probabilità teorica zero.", "Any additional binary states are unassigned and have zero theoretical probability."),
    ("##### Matrice di confusione", "##### Confusion matrix"),
    ("Le righe sono le classi reali e le colonne le classi predette.", "Rows are actual classes and columns are predicted classes."),
    ("La diagonale contiene le classificazioni corrette.", "The diagonal contains correct classifications."),
    ("**Numero di campioni**", "**Number of samples**"),
    ("**Percentuale per classe reale**", "**Percentage by actual class**"),
    ("##### Risultato di ogni campione", "##### Result for each sample"),
    ("Ogni riga confronta l'etichetta reale con quella scelta dal circuito.", "Each row compares the actual label with the label selected by the circuit."),
    ("La **confidenza** è la probabilità della classe predetta;", "**Confidence** is the probability of the predicted class;"),
    ("il **margine 1ª-2ª** è il distacco dalla seconda classe più probabile.", "the **1st–2nd margin** is the gap from the second most likely class."),
    ("La colonna **Lettura rapida** riassume il risultato in una frase.", "The **Quick reading** column summarizes the result in one sentence."),
    ("Lo scostamento circuito-teoria è un controllo numerico:", "The circuit-theory deviation is a numerical check:"),
    ("dovrebbe restare vicino alla precisione macchina e non misura la qualità statistica della classificazione.", "it should remain near machine precision and does not measure the statistical quality of classification."),
    ("**Mappa bitstring → classe**", "**Bitstring → class map**"),
    ("##### Circuito numerico rispetto alla teoria", "##### Numerical circuit versus theory"),
    ("Per ogni campione e classe confronto la probabilità ottenuta applicando la matrice unitaria del circuito con il valore teorico", "For every sample and class, the probability obtained by applying the circuit unitary is compared with the theoretical value"),
    ("Uno scostamento vicino alla precisione macchina indica che la dilatazione implementa correttamente la POVM.", "A deviation near machine precision indicates that the dilation correctly implements the POVM."),
    ("Somma degli effetti F_j = I", "Sum of effects F_j = I"),
    ("Probabilità Born ricostruite", "Reconstructed Born probabilities"),
    ("Somma probabilità sul test set = 1", "Test-set probability sum = 1"),
    ("Verifica superata: le predizioni del circuito coincidono con r-PGM sul 100% del test set.", "Check passed: circuit predictions match r-PGM on 100% of the test set."),
    ("Accordo circuito/r-PGM:", "Circuit/r-PGM agreement:"),
    ("Controllare i casi di pareggio numerico.", "Check numerical tie cases."),
    ("La dilatazione unitaria non è unica: completamenti unitari diversi producono le stesse probabilità sui dati in ingresso con out=0.", "The unitary dilation is not unique: different unitary completions produce the same probabilities for inputs with out=0."),
    ("##### Esegui la PGM su simulatore o hardware reale", "##### Run the PGM on a simulator or real hardware"),
    ("Il circuito eseguibile include la preparazione dello stato del campione selezionato.", "The executable circuit includes state preparation for the selected sample."),
    ("Per le esecuzioni circuitali, l'app prova automaticamente i livelli di ottimizzazione 1, 2 e 3 e usa soltanto il miglior circuito che supera la certificazione.", "For circuit executions, the app automatically tries optimization levels 1, 2, and 3 and uses only the best circuit that passes certification."),
    ("Puoi configurare shot, seed, modello di rumore e provider.", "You can configure shots, seed, noise model, and provider."),
    ("Le credenziali non entrano negli export o nella cache dell'app; possono essere lette da variabili d'ambiente o dal file locale Streamlit Secrets.", "Credentials are never included in exports or the app cache; they can be read from environment variables or the local Streamlit Secrets file."),
    ("richiede Python", "requires Python"),
    ("sono consigliati con Python", "are recommended with Python"),
    ("reale=", "actual="),
    ("predizione ideale=", "ideal prediction="),
    ("Esecuzioni circuitali: ottimizzazione automatica certificata, minimizzando gate entangling, poi profondità e porte totali.", "Circuit executions use certified automatic optimization, minimizing entangling gates first, then depth and total gates."),
    ("Modalità più veloce: campiona direttamente dalla distribuzione teorica della PGM, senza simulare ogni porta del circuito.", "Fastest mode: sample directly from the theoretical PGM distribution without simulating every circuit gate."),
    ("Simulatore di riferimento incluso in Qiskit. È ideale e più lento di Aer, ma non richiede componenti nativi aggiuntivi.", "Reference simulator included with Qiskit. It is ideal and slower than Aer, but requires no additional native components."),
    ("Qiskit non è installato.", "Qiskit is not installed."),
    ("Esecuzione locale con topologia, gate e snapshot di rumore di una QPU IBM. Non richiede token e non invia job.", "Local execution using an IBM QPU topology, gate set, and noise snapshot. It requires no token and submits no job."),
    ("IBM Fake Backend richiede Qiskit Aer e il provider IBM già elencati in requirements.txt.", "IBM Fake Backend requires Qiskit Aer and the IBM provider already listed in requirements.txt."),
    ("Lettura degli snapshot IBM non riuscita:", "Reading IBM snapshots failed:"),
    ("Nessuno snapshot IBM locale compatibile con il numero di qubit del circuito.", "No local IBM snapshot is compatible with the circuit qubit count."),
    ("Amazon Braket eseguito interamente sul server dell'app: non richiede account AWS, S3 o credenziali.", "Amazon Braket runs entirely on the app server: it requires no AWS account, S3 bucket, or credentials."),
    ("Amazon Braket Local richiede Python 3.11 o successivo.", "Amazon Braket Local requires Python 3.11 or later."),
    ("Amazon Braket Local non è installato.", "Amazon Braket Local is not installed."),
    ("Simulatori AQT inclusi nel provider: uno ideale e uno con rumore. L'esecuzione è locale e non richiede token.", "AQT simulators bundled with the provider: one ideal and one noisy. Execution is local and requires no token."),
    ("Provider AQT non installato.", "The AQT provider is not installed."),
    ("Simulazione disabilitata: il circuito supera il limite sicuro di", "Simulation disabled: the circuit exceeds the safe limit of"),
    ("qubit per questo motore.", "qubits for this engine."),
    ("Il simulatore selezionato accetta al massimo", "The selected simulator accepts at most"),
    ("Ottimizzo in un processo isolato, certifico l'equivalenza ed eseguo il circuito...", "Optimizing in an isolated process, certifying equivalence, and running the circuit..."),
    ("Il circuito supera il limite AQT di 2.000 porte.", "The circuit exceeds AQT's 2,000-gate limit."),
    ("Simulazione locale non riuscita:", "Local simulation failed:"),
    ("Nessun circuito candidato ha superato la certificazione: la simulazione non è stata eseguita.", "No candidate circuit passed certification; simulation was not run."),
    ("ha eseguito il migliore circuito certificato: livello", "ran the best certified circuit: level"),
    ("Qiskit Aer esegue il circuito completo.", "Qiskit Aer runs the full circuit."),
    ("Puoi scegliere il metodo numerico e aggiungere un semplice rumore depolarizzante.", "You can choose the numerical method and add a simple depolarizing noise model."),
    ("Qiskit Aer non è installato.", "Qiskit Aer is not installed."),
    ("Usa `Installa_Provider_Quantistici.command` e scegli Qiskit Aer, oppure installa `requirements-simulators.txt`.", "Use `Installa_Provider_Quantistici.command` and select Qiskit Aer, or install `requirements-simulators.txt`."),
    ("Ottimizzazione Aer disabilitata: il circuito supera", "Aer optimization disabled: the circuit exceeds"),
    ("Provo i livelli 1, 2 e 3, certifico l'equivalenza e simulo il migliore...", "Trying levels 1, 2, and 3, certifying equivalence, and simulating the best candidate..."),
    ("Simulazione Aer non riuscita:", "Aer simulation failed:"),
    ("Nessun candidato Aer ha superato la certificazione: la simulazione non è stata eseguita.", "No Aer candidate passed certification; simulation was not run."),
    ("Aer ha eseguito il migliore circuito certificato:", "Aer ran the best certified circuit:"),
    ("gate entangling, profondità", "entangling gates, depth"),
    ("###### Connessione sicura ad Amazon Braket", "###### Secure connection to Amazon Braket"),
    ("AWS non usa un singolo token Braket.", "AWS does not use a single Braket token."),
    ("Online puoi usare credenziali temporanee della tua sessione AWS; in locale puoi anche selezionare un profilo SSO già configurato.", "Online, you can use temporary AWS session credentials; locally, you can also select an existing SSO profile."),
    ("Le credenziali non vengono salvate dall'app né inserite nei file.", "Credentials are neither stored by the app nor written to files."),
    ("o successivo.", "or later."),
    ("Componenti AWS non installati.", "AWS components are not installed."),
    ("Usa `Installa_Provider_Quantistici.command` oppure segui `INSTALLAZIONE_PROVIDER_MAC.md`.", "Use `Installa_Provider_Quantistici.command` or follow `INSTALLAZIONE_PROVIDER_MAC.md`."),
    ("Credenziali AWS protette disponibili sul server.", "Protected AWS credentials are available on the server."),
    ("AWS Session Token (facoltativo; necessario per credenziali temporanee STS)", "AWS Session Token (optional; required for temporary STS credentials)"),
    ("Preferisci credenziali STS temporanee e con permessi limitati ad Amazon Braket.", "Prefer temporary STS credentials with permissions limited to Amazon Braket."),
    ("Non usare credenziali root.", "Do not use root credentials."),
    ("Connessione AWS e lettura dei dispositivi...", "Connecting to AWS and reading devices..."),
    ("Connessione AWS non riuscita:", "AWS connection failed:"),
    ("Connessione verificata · account", "Connection verified · account"),
    ("Nessun dispositivo compatibile trovato per queste credenziali e questa selezione.", "No compatible device was found for these credentials and this selection."),
    ("###### Connessione ai simulatori AQT Cloud", "###### Connect to AQT Cloud simulators"),
    ("AQT usa un access token del portale AQT.", "AQT uses an access token from the AQT portal."),
    ("L'app mostra soltanto i simulatori autorizzati per l'account; i simulatori offline senza token sono disponibili nella voce separata AQT Offline.", "The app lists only simulators authorized for the account; token-free offline simulators are available under the separate AQT Offline option."),
    ("AQT richiede Python 3.10 o successivo.", "AQT requires Python 3.10 or later."),
    ("Lettura dei simulatori AQT autorizzati...", "Reading authorized AQT simulators..."),
    ("Connessione AQT non riuscita:", "AQT connection failed:"),
    ("Nessun simulatore AQT Cloud è autorizzato per questo account. AQT Offline resta disponibile.", "No AQT Cloud simulator is authorized for this account. AQT Offline remains available."),
    ("###### Connessione a LRZ Quantum tramite MQSS", "###### Connect to LRZ Quantum through MQSS"),
    ("L'app interroga prima le risorse online e ignora in modo sicuro i metadati mancanti dei target offline.", "The app queries online resources first and safely ignores missing target metadata for offline resources."),
    ("In questo modo un backend non disponibile, come MUNIQC-Atoms20, non impedisce di mostrare una risorsa operativa come EQE1.", "This prevents an unavailable backend such as MUNIQC-Atoms20 from hiding an operational resource such as EQE1."),
    ("Adapter LRZ non installato.", "The LRZ adapter is not installed."),
    ("Per il riempimento automatico usa la variabile LRZ_MQSS_TOKEN o .streamlit/secrets.toml.", "For automatic filling, use the LRZ_MQSS_TOKEN variable or .streamlit/secrets.toml."),
    ("Lettura delle risorse autorizzate...", "Reading authorized resources..."),
    ("Connessione LRZ non riuscita:", "LRZ connection failed:"),
    ("Il token è valido ma non restituisce risorse compatibili con il filtro selezionato.", "The token is valid but returns no resources matching the selected filter."),
    ("Trovate", "Found"),
    ("risorse autorizzate.", "authorized resources."),
    ("LRZ Qaptiva usa un flusso VPN/SSH/HPC separato e non è presentato come falso backend MQSS.", "LRZ Qaptiva uses a separate VPN/SSH/HPC workflow and is not presented as an MQSS backend."),
    ("###### Connessione a IonQ Quantum Cloud", "###### Connect to IonQ Quantum Cloud"),
    ("IonQ usa una API key.", "IonQ uses an API key."),
    ("Il provider espone sia il simulatore cloud, con modelli di rumore opzionali, sia le QPU disponibili per il tuo account.", "The provider exposes both the cloud simulator, with optional noise models, and the QPUs available to your account."),
    ("è consigliato con Python", "is recommended with Python"),
    ("Provider IonQ non installato.", "The IonQ provider is not installed."),
    ("Il simulatore ideale IonQ restituisce probabilità.", "The ideal IonQ simulator returns probabilities."),
    ("L'app le converte nel numero di shot virtuali scelto sopra, senza alterare la distribuzione restituita.", "The app converts them to the number of virtual shots selected above without altering the returned distribution."),
    ("Lettura dei backend IonQ...", "Reading IonQ backends..."),
    ("Connessione IonQ non riuscita:", "IonQ connection failed:"),
    ("Nessun backend IonQ compatibile è stato restituito.", "No compatible IonQ backend was returned."),
    ("###### Connessione a IBM Quantum", "###### Connect to IBM Quantum"),
    ("Puoi incollare token e CRN dell'istanza oppure lasciare il token vuoto se hai già salvato localmente un account IBM Quantum.", "You can enter the token and instance CRN, or leave the token empty if an IBM Quantum account is already saved locally."),
    ("L'elenco mostra soltanto QPU operative con qubit sufficienti.", "The list shows only operational QPUs with enough qubits."),
    ("Provider IBM non installato.", "The IBM provider is not installed."),
    ("Lettura delle QPU IBM operative...", "Reading operational IBM QPUs..."),
    ("Connessione IBM non riuscita:", "IBM connection failed:"),
    ("Nessuna QPU IBM operativa con abbastanza qubit è disponibile per questo account.", "No operational IBM QPU with enough qubits is available for this account."),
    ("###### Preflight del circuito eseguibile", "###### Executable-circuit preflight"),
    ("Il circuito richiede", "The circuit requires"),
    ("qubit, ma la risorsa dichiara", "qubits, but the resource reports"),
    ("La risorsa dichiara un massimo di", "The resource reports a maximum of"),
    ("shot; riduci il valore.", "shots; reduce the value."),
    ("Alcune risorse LRZ, in particolare AQT, documentano limiti di shot più bassi.", "Some LRZ resources, especially AQT, document lower shot limits."),
    ("Il limite effettivo dipende dal backend.", "The actual limit depends on the backend."),
    ("Il preflight aggiunge la preparazione dello stato test, prova automaticamente i livelli 1, 2 e 3 in processi isolati e sceglie il migliore tra quelli equivalenti a U_PGM.", "Preflight adds test-state preparation, automatically tries levels 1, 2, and 3 in isolated processes, and selects the best candidate equivalent to U_PGM."),
    ("Il provider eseguirà poi la necessaria conversione nella base nativa del dispositivo.", "The provider will then perform the required conversion to the device's native basis."),
    ("Esecuzione esterna disabilitata: questo circuito supera", "External execution disabled: this circuit exceeds"),
    ("qubit e la sintesi sicura non è praticabile sul computer locale.", "qubits and safe synthesis is not practical on the local computer."),
    ("Provo i livelli 1, 2 e 3 sul circuito completo e certifico ogni candidato...", "Trying levels 1, 2, and 3 on the full circuit and certifying each candidate..."),
    ("Preflight non riuscito:", "Preflight failed:"),
    ("Nessuno dei tre candidati ha superato la certificazione.", "None of the three candidates passed certification."),
    ("Il job non può essere inviato.", "The job cannot be submitted."),
    ("Criterio di scelta: gate entangling, profondità, poi porte totali.", "Selection criterion: entangling gates, depth, then total gates."),
    ("Circuito su", "Circuit on"),
    ("I conteggi sono nella base generica rz/sx/x/cx; la compilazione nativa del provider può modificarli.", "Counts are in the generic rz/sx/x/cx basis; the provider's native compilation may change them."),
    ("Migliore preflight certificato equivalente a U_PGM:", "Best preflight certified equivalent to U_PGM:"),
    ("errore massimo", "maximum error"),
    ("Il preflight non ha superato la certificazione di equivalenza: l'invio è disabilitato.", "Preflight did not pass equivalence certification; submission is disabled."),
    ("Circuito troppo grande per il disegno integrale; il conteggio delle porte resta esatto.", "The circuit is too large for a full drawing; gate counts remain exact."),
    ("Il circuito supera 2.000 porte, limite noto per alcune risorse AQT/LRZ; altri backend possono avere limiti differenti.", "The circuit exceeds 2,000 gates, a known limit for some AQT/LRZ resources; other backends may have different limits."),
    ("Confermo di voler inviare", "I confirm that I want to submit"),
    ("il job alla risorsa LRZ selezionata", "the job to the selected LRZ resource"),
    ("il job alla QPU IonQ selezionata", "the job to the selected IonQ QPU"),
    ("il job alla QPU IBM selezionata", "the job to the selected IBM QPU"),
    ("usando la mia allocazione.", "using my allocation."),
    ("un task AWS che può usare quota o generare costi.", "an AWS task that may consume quota or incur charges."),
    ("un task AWS che può generare costi sul mio account.", "an AWS task that may incur charges on my account."),
    ("un job IonQ che può usare quota o generare costi.", "an IonQ job that may consume quota or incur charges."),
    ("selezionata e di accettarne quota o costi.", "selected and accept the associated quota usage or costs."),
    ("selezionata usando la mia istanza.", "selected using my instance."),
    ("Prima dell'invio verifica quota, piano e prezzi nel portale del provider: l'app non usa costi hard-coded.", "Before submission, check quota, plan, and pricing in the provider portal; the app does not use hard-coded costs."),
    ("Invio del job; il risultato verrà letto solo su richiesta...", "Submitting the job; the result will be retrieved only on request..."),
    ("Reinserire il token LRZ.", "Re-enter the LRZ token."),
    ("Reinserire la API key IonQ.", "Re-enter the IonQ API key."),
    ("Reinserire il token AQT.", "Re-enter the AQT token."),
    ("Credenziali o regione AWS mancanti.", "AWS credentials or region are missing."),
    ("Job inviato. ID:", "Job submitted. ID:"),
    ("Invio non riuscito:", "Submission failed:"),
    ("###### Job remoto", "###### Remote job"),
    ("Stato non disponibile:", "Status unavailable:"),
    ("Risultato non disponibile:", "Result unavailable:"),
    ("Annullamento non riuscito:", "Cancellation failed:"),
    ("QPY conserva il circuito Qiskit; NPZ contiene U_PGM, gli effetti F_j, l'ordine delle classi e i metadati della codifica.", "QPY preserves the Qiskit circuit; NPZ contains U_PGM, the F_j effects, class order, and encoding metadata."),
    ("Per questa configurazione la dilatazione avrebbe una matrice densa", "For this configuration, the dilation would require a dense matrix"),
    ("Per proteggere la memoria, l'app mostra l'architettura dimensionata ma non materializza U_PGM.", "To protect memory, the app shows the dimensioned architecture but does not materialize U_PGM."),
    ("I risultati classici, la matrice di confusione e il dettaglio dei campioni restano disponibili sopra.", "Classical results, the confusion matrix, and sample details remain available above."),
    ("Seleziona **Automatica quantum-ready** per ottenere il circuito esatto esportabile e l'esecuzione quantistica; in alternativa riduci il numero di copie.", "Select **Automatic quantum-ready** to obtain the exportable exact circuit and quantum execution; alternatively, reduce the number of copies."),
    ("Schema logico dimensionato; U_PGM_symbolic indica la matrice non materializzata per questa configurazione.", "Dimensioned logical diagram; U_PGM_symbolic denotes the matrix not materialized for this configuration."),
    ("**Corrispondenza tra esiti e classi**", "**Outcome-to-class mapping**"),
    ("Manca una dipendenza necessaria per disegnare il circuito.", "A dependency required to draw the circuit is missing."),
    ("Ferma l'app con Control+C, esegui", "Stop the app with Control+C, run"),
    ("nella cartella del progetto e riavviala.", "in the project folder, and restart it."),
    ("Dettaglio:", "Details:"),
    ("Le impostazioni sono cambiate: premi il pulsante per calcolare il nuovo caso.", "Settings have changed; press the button to compute the new case."),
    ("Nota: r-PGM è il nome breve usato qui per il reduced c-PGM (Rc-PGM) del paper.", "Note: r-PGM is the short name used here for the paper's reduced c-PGM (Rc-PGM)."),
    ("Per c=1 la mappa ridotta coincide con la mappa originale.", "For c=1, the reduced map equals the original map."),
    ("Reale:", "Actual:"),
    ("Predetta:", "Predicted:"),
    ("Test #", "Test #"),
    ("decisione netta", "clear decision"),
    ("decisione moderata", "moderately clear decision"),
    ("classi molto vicine: interpretare con cautela", "very close classes: interpret with caution"),
    ("Corretta: la misura favorisce la classe", "Correct: the measurement favors class"),
    ("Errata: la misura favorisce la classe", "Incorrect: the measurement favors class"),
    ("con probabilità", "with probability"),
    ("il distacco dalla seconda classe è", "the margin over the second-ranked class is"),
    ("P teorica", "Theoretical P"),
    ("P circuito", "Circuit P"),
    ("unitaria esatta appresa dal training", "exact unitary learned during training"),
    ("schema simbolico: matrice non materializzata", "symbolic diagram: matrix not materialized"),
    ("Circuito quantistico PGM mediante dilatazione di Naimark", "PGM quantum circuit using a Naimark dilation"),
    ("Circuito logico della PGM", "PGM logical circuit"),
    ("Dilatazione di Naimark", "Naimark dilation"),
    ("Registro sys:", "sys register:"),
    ("Registro out:", "out register:"),
    ("Totale:", "Total:"),
    ("Ingresso: stato test codificato", "Input: encoded test state"),
    ("Uscita: bitstring di classe", "Output: class bitstring"),
    ("SIMULATORE", "SIMULATOR"),
    ("QPU SUPERCONDUTTIVA", "SUPERCONDUCTING QPU"),
    ("QPU IONI INTRAPPOLATI", "TRAPPED-ION QPU"),
    ("Attesa media dichiarata:", "Reported average wait:"),
    ("Prestazioni dichiarate come degradate", "Reported degraded performance"),
    ("Metadati target non disponibili:", "Target metadata unavailable:"),
    ("ID non disponibile", "ID unavailable"),
    ("job in coda", "queued jobs"),
    ("qubit n/d", "qubits n/a"),
    ("La dimensione deve essere positiva.", "The dimension must be positive."),
    ("I limiti automatici devono essere positivi.", "Automatic limits must be positive."),
    ("Nessuna dimensione di codifica soddisfa contemporaneamente i limiti classici e quantistici selezionati.", "No encoding dimension satisfies all selected classical and quantum limits."),
    ("la RAM di picco stimata per l'implementazione supera il budget selezionato", "the implementation's estimated peak RAM exceeds the selected budget"),
    ("una diagonalizzazione densa avrebbe dimensione", "a dense eigendecomposition would have dimension"),
    (", oltre il limite prudenziale", ", above the conservative limit"),
    ("Il catalogo contiene chiavi dataset duplicate.", "The catalogue contains duplicate dataset keys."),
    ("Metadati non validi per il dataset", "Invalid metadata for dataset"),
    ("Nessun loader configurato per il dataset", "No loader is configured for dataset"),
    ("Sottoinsieme non valido per", "Invalid subset for"),
    ("Seed del sottoinsieme mancante per", "Missing subset seed for"),
    ("Dataset sconosciuto:", "Unknown dataset:"),
    ("3 centri", "3 centers"),
    ("generatore XOR locale", "local XOR generator"),
    ("generatore Three Spirals locale", "local Three Spirals generator"),
    ("rumore gaussiano", "Gaussian noise"),
    ("sottoinsieme stratificato fisso", "fixed stratified subset"),
    ("Nessun fallback locale disponibile.", "No local fallback is available."),
    ("La copia locale non coincide con i metadati del catalogo.", "The local copy does not match the catalogue metadata."),
    ("Il dataset ha piu di una variabile target.", "The dataset has more than one target variable."),
    ("Download OpenML non riuscito. Controllare la connessione e riprovare.", "OpenML download failed. Check the connection and try again."),
    ("Dettaglio originale:", "Original details:"),
    ("Metadati inattesi: attesi", "Unexpected metadata: expected"),
    ("Metadati inattesi: attese", "Unexpected metadata: expected"),
    ("ricevuti", "received"),
    ("ricevute", "received"),
    ("Seed del sottoinsieme mancante.", "The subset seed is missing."),
    ("Il dataset elaborato non coincide con i metadati del catalogo.", "The processed dataset does not match the catalogue metadata."),
    ("Per l'esecuzione hardware serve la dilatazione esatta.", "Hardware execution requires the exact dilation."),
    ("Lo stato test non ha la dimensione ridotta attesa.", "The test state does not have the expected reduced dimension."),
    ("Lo stato test deve avere norma unitaria.", "The test state must have unit norm."),
    ("Il file QPY deve contenere esattamente un circuito.", "The QPY file must contain exactly one circuit."),
    ("Il numero di shot deve essere positivo.", "The number of shots must be positive."),
    ("errore a un qubit", "one-qubit error"),
    ("errore a due qubit", "two-qubit error"),
    ("errore di lettura", "readout error"),
    ("La probabilità di", "The probability of"),
    ("deve essere in [0, 1).", "must be in [0, 1)."),
    ("Il risultato non contiene conteggi utilizzabili.", "The result contains no usable counts."),
    ("Tutti gli shot sono finiti in esiti binari non assegnati.", "All shots ended in unassigned binary outcomes."),
    ("Il provider ha restituito un risultato vuoto.", "The provider returned an empty result."),
    ("Simulatore Amazon Braket locale non riconosciuto.", "Unknown Amazon Braket local simulator."),
    ("Backend AQT non trovato:", "AQT backend not found:"),
    ("Snapshot IBM non trovato:", "IBM snapshot not found:"),
    ("Questa versione del provider Braket non accetta una sessione AWS isolata.", "This version of the Braket provider does not accept an isolated AWS session."),
    ("Aggiornare qiskit-braket-provider.", "Update qiskit-braket-provider."),
    ("Nessun profilo o credenziale AWS disponibile.", "No AWS profile or credentials are available."),
    ("Backend Amazon Braket non trovato:", "Amazon Braket backend not found:"),
    ("Il job contiene più di un circuito.", "The job contains more than one circuit."),
    ("Il provider non ha restituito conteggi riconoscibili.", "The provider returned no recognizable counts."),
    ("Le matrici da confrontare devono avere la stessa forma.", "The matrices being compared must have the same shape."),
    ("Dimensione del sottospazio di ingresso non valida.", "Invalid input-subspace dimension."),
    ("X_train e y_train non sono compatibili.", "X_train and y_train are incompatible."),
    ("I vettori di training devono avere norma unitaria.", "Training vectors must have unit norm."),
    ("I vettori di training devono avere norma L2 unitaria.", "Training vectors must have unit L2 norm."),
    ("La tolleranza spettrale deve essere tra 0 e 1.", "The spectral tolerance must be between 0 and 1."),
    ("La PGM non ha supporto numerico alla soglia scelta.", "The PGM has no numerical support at the selected threshold."),
    ("Effetto non positivo: autovalore minimo", "Non-positive effect: minimum eigenvalue"),
    ("L'isometria di Naimark ha perso rango numerico.", "The Naimark isometry lost numerical rank."),
    ("La matrice unitaria non è stata materializzata.", "The unitary matrix was not materialized."),
    ("Lo stato non ha la dimensione della feature map ridotta.", "The state does not have the reduced feature-map dimension."),
    ("L'isometria esatta non è stata materializzata.", "The exact isometry was not materialized."),
    ("Il numero di qubit deve essere positivo.", "The number of qubits must be positive."),
    ("Fornire esattamente uno tra circuito QPY e matrice isometrica.", "Provide exactly one of a QPY circuit or an isometry matrix."),
    ("Numero di qubit di sistema non valido.", "Invalid number of system qubits."),
    ("Numero di qubit di uscita non valido.", "Invalid number of outcome qubits."),
    ("La matrice isometrica non coincide con le dimensioni dei registri.", "The isometry matrix does not match the register dimensions."),
    ("Il worker di transpilation non è presente nel progetto.", "The transpilation worker is missing from the project."),
    ("Lo stato di ingresso supera il registro di sistema.", "The input state exceeds the system register."),
    ("La dimensione del sottospazio è necessaria per certificare l'equivalenza.", "The subspace dimension is required to certify equivalence."),
    ("La sintesi ha superato il limite di", "Synthesis exceeded the time limit of"),
    ("secondi ed è stata interrotta in sicurezza.", "seconds and was stopped safely."),
    ("Il processo isolato di sintesi è terminato con", "The isolated synthesis process exited with"),
    ("L'interfaccia principale è rimasta attiva.", "The main interface remained active."),
    ("segnale", "signal"),
    ("codice", "code"),
    ("La decomposizione e i conteggi sono stati comunque recuperati; è fallita soltanto una fase successiva.", "The decomposition and counts were recovered; only a later stage failed."),
    ("La sintesi è terminata senza produrre tutti i file attesi.", "Synthesis ended without producing all expected files."),
    ("Sintesi completata nel processo isolato.", "Synthesis completed in the isolated process."),
    ("La dilatazione esatta non è disponibile.", "The exact dilation is not available."),
    ("test_fraction deve essere strettamente tra 0 e 1.", "test_fraction must be strictly between 0 and 1."),
    ("Encoding non riconosciuto:", "Unknown encoding:"),
    ("La validazione dell'encoding tensoriale non è riuscita:", "Validation of the normalized amplitude encoding failed:"),
    ("selezione automatica disattivata", "automatic selection disabled"),
    ("Il training set è troppo piccolo per validare automaticamente l'encoding.", "The training set is too small for automatic encoding validation."),
    ("max_encoded_features deve essere un intero o None.", "max_encoded_features must be an integer or None."),
    ("max_encoded_features deve essere positivo.", "max_encoded_features must be positive."),
    ("deve essere una matrice bidimensionale.", "must be a two-dimensional matrix."),
    ("contiene valori NaN o infiniti.", "contains NaN or infinite values."),
    ("Train e test devono avere lo stesso numero di feature.", "Training and test data must have the same number of features."),
    ("Il numero di copie c deve essere un intero >= 1.", "The number of copies c must be an integer >= 1."),
    ("I vettori di test devono avere norma L2 unitaria.", "Test vectors must have unit L2 norm."),
    ("Classe non riconosciuta:", "Unknown class:"),
    ("Modalita dei prior non supportata:", "Unsupported prior mode:"),
    ("copies deve essere >= 1.", "copies must be >= 1."),
    ("La tolleranza spettrale relativa deve essere tra 0 e 1.", "The relative spectral tolerance must be between 0 and 1."),
    ("Nessun autovalore supera la soglia spettrale; controllare i dati.", "No eigenvalue exceeds the spectral threshold; check the data."),
    ("explicit_dimension_limit deve essere un intero.", "explicit_dimension_limit must be an integer."),
    ("explicit_dimension_limit deve essere positivo.", "explicit_dimension_limit must be positive."),
    (
        "Il circuito logico esatto, la classificazione test e la validazione matematica saranno disponibili.",
        "The exact logical circuit, test classification, and mathematical validation will be available.",
    ),
    (
        "La decomposizione completa, l'ottimizzazione e l'esecuzione gate-by-gate resteranno disabilitate:",
        "Full decomposition, optimization, and gate-by-gate execution will remain disabled:",
    ),
    ("questa configurazione usa", "this configuration uses"),
    ("qubit, oltre il limite protetto di", "qubits, above the protected limit of"),
    (
        "Per rientrare nel limite con questi valori di copie e classi, puoi richiedere manualmente una PCA con al massimo",
        "To fit the limit with these copy and class values, you may manually request PCA with at most",
    ),
    ("componenti.", "components."),
    (
        "Con questo numero di copie e classi non basta una PCA a una sola componente: occorre ridurre le copie oppure scegliere un dataset con meno classi.",
        "With this number of copies and classes, even one PCA component is insufficient: reduce the number of copies or choose a dataset with fewer classes.",
    ),
    (
        "La classificazione resta disponibile senza ridurre le feature. Se desideri anche il circuito esatto, la riduzione resta sempre una scelta esplicita dell'utente.",
        "Classification remains available without reducing features. If you also need the exact circuit, reduction always remains an explicit user choice.",
    ),
    (
        "Il comando esegue",
        "This command runs",
    ),
    (
        "split stratificati e produce i risultati della sezione 4.",
        "stratified splits and produces the Section 4 results.",
    ),
    (
        "Il circuito usa soltanto lo split di riferimento con seed",
        "The circuit uses only the reference split with seed",
    ),
    ("e non altera le statistiche.", "and does not alter the statistics."),
    (
        "Il circuito logico esatto, la classificazione test e la validazione restano disponibili.",
        "The exact logical circuit, test classification, and validation remain available.",
    ),
    (
        "La sintesi gate-by-gate e l'optimizer sono invece disabilitati oltre",
        "Gate-by-gate synthesis and the optimizer are disabled above",
    ),
    (" qubit:", " qubits:"),
    (
        "per questa unitaria generica il limite prudenziale è circa",
        "for this generic unitary, the conservative upper bound is approximately",
    ),
    (
        "CNOT, oltre alle rotazioni a un qubit.",
        "CNOT gates, in addition to single-qubit rotations.",
    ),
    (
        "Questa separazione protegge l'interfaccia da tempi e memoria non prevedibili.",
        "This separation protects the interface from unpredictable runtime and memory use.",
    ),
    (
        "Per attivare questi due comandi puoi richiedere una PCA manuale, ridurre il numero di copie oppure scegliere un dataset con meno classi.",
        "To enable these two commands, you may request manual PCA, reduce the number of copies, or choose a dataset with fewer classes.",
    ),
    (
        "La PGM e le sue feature non vengono modificate automaticamente.",
        "The PGM and its features are never modified automatically.",
    ),
    (
        "Per attivare il circuito esatto con le copie e le classi correnti, abilita volontariamente la PCA nella sezione 1 e imposta al massimo",
        "To enable the exact circuit with the current copies and classes, explicitly enable PCA in Section 1 and set at most",
    ),
    (
        "In alternativa riduci il numero di copie.",
        "Alternatively, reduce the number of copies.",
    ),
    (
        "Nessuna feature viene ridotta automaticamente.",
        "No feature is reduced automatically.",
    ),
    (
        "Con il numero corrente di copie e classi, neppure una PCA a una componente rientra nel limite sicuro.",
        "With the current number of copies and classes, even one PCA component does not fit the safe limit.",
    ),
    (
        "Riduci il numero di copie oppure usa un dataset con meno classi; la classificazione PGM classica resta comunque completa.",
        "Reduce the number of copies or use a dataset with fewer classes; classical PGM classification remains complete.",
    ),
)


_ORDERED_PHRASES = tuple(
    sorted(PHRASE_TRANSLATIONS, key=lambda item: len(item[0]), reverse=True)
)


# These short terms appear inside dynamic labels.  Word boundaries are required:
# raw substring replacement of ``classi`` would corrupt ``classificazione``.
WORD_TRANSLATIONS: tuple[tuple[str, str], ...] = (
    ("classificazione", "classification"),
    ("porte", "gates"),
    ("campioni", "samples"),
    ("classi", "classes"),
    ("classe", "class"),
    ("risorse", "resources"),
    ("risorsa", "resource"),
)


def normalize_language(language: str | None) -> str:
    return language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def translate_text(value: Any, language: str | None) -> Any:
    """Translate a user-visible string while leaving all non-strings untouched."""

    if normalize_language(language) != "en" or not isinstance(value, str):
        return value
    if value in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[value]
    normalized = _normalized_text(value)
    if normalized in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[normalized]
    if normalized in NORMALIZED_TRANSLATIONS:
        return NORMALIZED_TRANSLATIONS[normalized]
    translated = value
    for source, target in _ORDERED_PHRASES:
        translated = translated.replace(source, target)
    for source, target in WORD_TRANSLATIONS:
        translated = re.sub(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            target,
            translated,
        )
    translated = re.sub(r"(?<=\d) feature\b", " features", translated)
    return translated


def localize_dataframe(frame: pd.DataFrame, language: str | None) -> pd.DataFrame:
    """Return a presentation copy with translated headers, index and text cells."""

    if normalize_language(language) != "en":
        return frame
    localized = frame.copy()
    localized = localized.rename(
        columns=lambda value: translate_text(value, "en"),
        index=lambda value: translate_text(value, "en"),
    )
    for column in localized.columns:
        if pd.api.types.is_object_dtype(localized[column].dtype) or isinstance(
            localized[column].dtype, pd.StringDtype
        ):
            localized[column] = localized[column].map(
                lambda value: translate_text(value, "en")
            )
    return localized


def localize_value(value: Any, language: str | None) -> Any:
    if normalize_language(language) != "en":
        return value
    if isinstance(value, str):
        return translate_text(value, "en")
    if isinstance(value, pd.DataFrame):
        return localize_dataframe(value, "en")
    if isinstance(value, Mapping):
        return {
            localize_value(key, "en"): localize_value(item, "en")
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(localize_value(item, "en") for item in value)
    if isinstance(value, list):
        return [localize_value(item, "en") for item in value]
    return value


class _ColumnConfigProxy:
    def __init__(self, owner: Any, language_getter: Callable[[], str]):
        self._owner = owner
        self._language_getter = language_getter

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._owner, name)
        if not callable(attribute):
            return attribute

        def localized_factory(*args: Any, **kwargs: Any) -> Any:
            language = self._language_getter()
            localized_args = list(args)
            if localized_args and isinstance(localized_args[0], str):
                localized_args[0] = translate_text(localized_args[0], language)
            for key in ("label", "help", "display_text"):
                if key in kwargs and isinstance(kwargs[key], str):
                    kwargs[key] = translate_text(kwargs[key], language)
            return attribute(*localized_args, **kwargs)

        return localized_factory


class LocalizedStreamlit:
    """Small transparent proxy that localizes Streamlit presentation methods."""

    _TEXT_METHODS = {
        "title",
        "header",
        "subheader",
        "caption",
        "markdown",
        "text",
        "info",
        "success",
        "warning",
        "error",
        "toast",
    }
    _WIDGET_METHODS = {
        "button",
        "download_button",
        "link_button",
        "checkbox",
        "toggle",
        "text_input",
        "text_area",
        "number_input",
        "slider",
        "file_uploader",
        "camera_input",
        "form_submit_button",
    }
    _CHOICE_METHODS = {
        "selectbox",
        "radio",
        "select_slider",
        "multiselect",
        "segmented_control",
        "pills",
    }

    def __init__(self, owner: Any, language_getter: Callable[[], str]):
        self._owner = owner
        self._language_getter = language_getter

    def __enter__(self) -> "LocalizedStreamlit":
        self._owner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return self._owner.__exit__(exc_type, exc, traceback)

    def _language(self) -> str:
        return normalize_language(self._language_getter())

    def _wrap(self, owner: Any) -> "LocalizedStreamlit":
        return LocalizedStreamlit(owner, self._language_getter)

    def __getattr__(self, name: str) -> Any:
        if name == "column_config":
            return _ColumnConfigProxy(
                getattr(self._owner, name), self._language_getter
            )
        attribute = getattr(self._owner, name)
        if not callable(attribute):
            if name == "sidebar":
                return self._wrap(attribute)
            return attribute

        if name in self._TEXT_METHODS:
            def localized_text(*args: Any, **kwargs: Any) -> Any:
                localized_args = list(args)
                if localized_args:
                    localized_args[0] = translate_text(
                        localized_args[0], self._language()
                    )
                return attribute(*localized_args, **kwargs)

            return localized_text

        if name == "write":
            def localized_write(*args: Any, **kwargs: Any) -> Any:
                return attribute(
                    *(localize_value(value, self._language()) for value in args),
                    **kwargs,
                )

            return localized_write

        if name in {"dataframe", "table", "data_editor"}:
            def localized_table(*args: Any, **kwargs: Any) -> Any:
                localized_args = list(args)
                if localized_args:
                    localized_args[0] = localize_value(
                        localized_args[0], self._language()
                    )
                if "column_config" in kwargs:
                    kwargs["column_config"] = localize_value(
                        kwargs["column_config"], self._language()
                    )
                return attribute(*localized_args, **kwargs)

            return localized_table

        if name == "metric":
            def localized_metric(*args: Any, **kwargs: Any) -> Any:
                localized_args = list(args)
                for index in range(min(3, len(localized_args))):
                    localized_args[index] = translate_text(
                        localized_args[index], self._language()
                    )
                for key in ("label", "value", "delta", "help"):
                    if key in kwargs:
                        kwargs[key] = translate_text(kwargs[key], self._language())
                return attribute(*localized_args, **kwargs)

            return localized_metric

        if name in self._WIDGET_METHODS:
            def localized_widget(*args: Any, **kwargs: Any) -> Any:
                localized_args = list(args)
                if localized_args:
                    localized_args[0] = translate_text(
                        localized_args[0], self._language()
                    )
                for key in ("label", "help", "placeholder"):
                    if key in kwargs:
                        kwargs[key] = translate_text(kwargs[key], self._language())
                return attribute(*localized_args, **kwargs)

            return localized_widget

        if name in self._CHOICE_METHODS:
            def localized_choice(*args: Any, **kwargs: Any) -> Any:
                localized_args = list(args)
                if localized_args:
                    localized_args[0] = translate_text(
                        localized_args[0], self._language()
                    )
                elif "label" in kwargs:
                    kwargs["label"] = translate_text(
                        kwargs["label"], self._language()
                    )
                if "help" in kwargs:
                    kwargs["help"] = translate_text(
                        kwargs["help"], self._language()
                    )
                original_formatter = kwargs.get("format_func", str)

                def localized_formatter(option: Any) -> Any:
                    return translate_text(
                        original_formatter(option), self._language()
                    )

                kwargs["format_func"] = localized_formatter
                return attribute(*localized_args, **kwargs)

            return localized_choice

        if name == "columns":
            def localized_columns(*args: Any, **kwargs: Any) -> list[Any]:
                return [self._wrap(column) for column in attribute(*args, **kwargs)]

            return localized_columns

        if name == "tabs":
            def localized_tabs(labels: Sequence[str], *args: Any, **kwargs: Any) -> list[Any]:
                localized_labels = [
                    translate_text(label, self._language()) for label in labels
                ]
                return [
                    self._wrap(tab)
                    for tab in attribute(localized_labels, *args, **kwargs)
                ]

            return localized_tabs

        if name in {"container", "empty", "popover", "form"}:
            def localized_container(*args: Any, **kwargs: Any) -> Any:
                if name in {"popover", "form"} and args:
                    args = (
                        translate_text(args[0], self._language()),
                        *args[1:],
                    )
                return self._wrap(attribute(*args, **kwargs))

            return localized_container

        if name == "expander":
            def localized_expander(label: str, *args: Any, **kwargs: Any) -> Any:
                return self._wrap(
                    attribute(
                        translate_text(label, self._language()), *args, **kwargs
                    )
                )

            return localized_expander

        if name in {"spinner", "status"}:
            def localized_status(text: str, *args: Any, **kwargs: Any) -> Any:
                result = attribute(
                    translate_text(text, self._language()), *args, **kwargs
                )
                return self._wrap(result) if name == "status" else result

            return localized_status

        if name == "exception":
            def localized_exception(error: BaseException, *args: Any, **kwargs: Any) -> Any:
                if self._language() == "en":
                    error = RuntimeError(translate_text(str(error), "en"))
                return attribute(error, *args, **kwargs)

            return localized_exception

        return attribute
