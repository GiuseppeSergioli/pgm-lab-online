#!/usr/bin/env python3
"""Generate the two illustrated, versioned PGM Lab user guides."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


APP_VERSION = "5.3.0"
GUIDE_RELEASE_DATE = "2026-10-08"
APP_URL = "https://pgm-lab-sergioli.streamlit.app/"
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 17 * mm
MARGIN_TOP = 19 * mm
MARGIN_BOTTOM = 17 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X

NAVY = HexColor("#18233D")
BLUE = HexColor("#4767C7")
PURPLE = HexColor("#7657B5")
CYAN = HexColor("#1A8CA6")
GREEN = HexColor("#278A63")
AMBER = HexColor("#D48A1F")
RED = HexColor("#C45252")
INK = HexColor("#202531")
MUTED = HexColor("#5F6B7A")
PALE_BLUE = HexColor("#EEF2FF")
PALE_GREEN = HexColor("#EAF7F1")
PALE_AMBER = HexColor("#FFF6E5")
PALE_RED = HexColor("#FDEEEE")
PALE_GRAY = HexColor("#F5F7FA")
WHITE = colors.white
GRID = HexColor("#D9DFEA")


def _register_fonts() -> tuple[str, str]:
    candidates = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ),
    )
    for regular_path, bold_path in candidates:
        if Path(regular_path).exists() and Path(bold_path).exists():
            pdfmetrics.registerFont(TTFont("GuideSans", regular_path))
            pdfmetrics.registerFont(TTFont("GuideSans-Bold", bold_path))
            return "GuideSans", "GuideSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()


@dataclass(frozen=True)
class GuideText:
    language_code: str
    language_name: str
    file_name: str
    title: str
    subtitle: str
    updated: str
    sections: dict[str, object]


TEXTS: dict[str, GuideText] = {
    "it": GuideText(
        language_code="it",
        language_name="Italiano",
        file_name="PGM_Lab_Guida_Illustrata_IT.pdf",
        title="PGM Lab",
        subtitle="Guida illustrata all'uso dell'app",
        updated="Aggiornata alla versione 5.3.0 - 8 ottobre 2026",
        sections={
            "cover_note": (
                "Dalla scelta del dataset alla PGM, dal confronto statistico al "
                "circuito quantistico e all'esecuzione su simulatori o QPU."
            ),
            "contents": "In questa guida",
            "contents_items": [
                "Orientamento e avvio rapido",
                "Dataset pubblici e upload personale",
                "Impostazioni scientifiche e calcolo",
                "Risultati e confronto con altri classificatori",
                "Circuito quantistico, ottimizzazione ed esecuzione",
                "Privacy, download e risoluzione dei problemi",
            ],
            "start_title": "1. Orientamento e avvio rapido",
            "start_intro": (
                "PGM Lab confronta c-PGM, k-PGM e r-PGM, tre realizzazioni "
                "matematicamente equivalenti del Pretty Good Measurement classifier. "
                "L'interfaccia seleziona il backend esatto meno oneroso senza cambiare "
                "la regola di decisione."
            ),
            "ui_map": "Mappa della pagina",
            "quick": "Percorso consigliato in 6 passi",
            "quick_steps": [
                "Scegli un dataset pubblico oppure carica il tuo file.",
                "Imposta il numero di copie c e, se serve, il budget RAM.",
                "Controlla dimensioni, backend previsto e fattibilità.",
                "Avvia la valutazione multi-seed PGM.",
                "Leggi metriche, deviazioni standard, confusion matrix e campioni.",
                "Solo se necessario, confronta modelli o costruisci/esegui il circuito.",
            ],
            "language": (
                "Le bandiere in alto a destra cambiano l'intera interfaccia. Dati, "
                "impostazioni e risultati non vengono ricalcolati durante il cambio lingua."
            ),
            "upload_title": "2. Dataset pubblici e upload personale",
            "upload_intro": (
                "Nel punto 1 apri Carica un dataset personale e trascina il file nella "
                "zona tratteggiata. Il dataset validato compare nella tabella e nel menu "
                "insieme ai dataset pubblici, contrassegnato dall'icona di upload."
            ),
            "upload_steps": [
                "Trascina CSV, TSV, TXT o XLSX (prima scheda Excel).",
                "Controlla la colonna target proposta e correggila se necessario.",
                "Conferma le feature numeriche; gli ID probabili sono deselezionati.",
                "Assegna un nome leggibile e verifica l'anteprima e le classi.",
                "Premi Aggiungi al catalogo e seleziona.",
            ],
            "requirements": "Requisiti del file",
            "requirements_rows": [
                ["Formato", "CSV, TSV/TXT, XLSX"],
                ["Dimensione", "max 25 MB; 50.000 righe; 500 colonne"],
                ["Struttura", "prima riga con nomi di colonna univoci"],
                ["Target", "2-50 classi; almeno 3 campioni per classe"],
                ["Feature", "numeriche; almeno una variabile"],
                ["Campioni", "almeno 10 dopo la pulizia"],
            ],
            "validation": (
                "Le righe senza target vengono rimosse. I valori mancanti delle feature "
                "sono imputati con la mediana appresa soltanto sul training set di ogni "
                "seed. Colonne testuali, vuote o costanti vengono segnalate chiaramente."
            ),
            "privacy": (
                "Il file personale resta nella memoria della sessione, non entra nel "
                "repository e non usa la cache condivisa dei dataset pubblici. Rimuovi "
                "dataset personale cancella anche i risultati dipendenti dal file."
            ),
            "settings_title": "3. Impostazioni scientifiche e controllo dimensionale",
            "settings_intro": (
                "Per impostazione predefinita il PGM usa tutte le feature selezionate. "
                "La PCA non è mai automatica: viene applicata soltanto se l'utente attiva "
                "Richiedi manualmente una riduzione PCA."
            ),
            "settings_rows": [
                ["Numero di copie c", "potenza del kernel e ordine della mappa simmetrica"],
                ["Quota test", "solo valori compatibili con uno split stratificato"],
                ["Seed", "split riproducibili; default: 10 seed consecutivi"],
                ["Prior", "uniformi oppure empirici rispetto alle frequenze"],
                ["Soglia spettrale", "controlla il rank numerico in modo coerente"],
                ["Budget RAM", "limite di sicurezza; non aumenta la RAM del server"],
            ],
            "encoding": (
                "Su ciascun training set l'app confronta l'encoding in ampiezza e quello "
                "stereografico con fattori t = 0.1, 0.2, 0.5, 1, 2. Il test set non "
                "partecipa alla scelta. L'encoding vincente è dichiarato nei risultati."
            ),
            "dimension_note": (
                "Se il circuito è troppo grande, la classificazione classica può restare "
                "disponibile. Il circuito esatto è materializzato fino a 9 qubit; la "
                "sintesi completa automatica è limitata prudenzialmente a 7 qubit."
            ),
            "results_title": "4. Eseguire la PGM e interpretare i risultati",
            "results_intro": (
                "Premi Esegui la valutazione multi-seed. Il riepilogo principale usa "
                "media e deviazione standard su tutti gli split; un singolo seed di "
                "riferimento serve soltanto per diagnostica, campioni e circuito."
            ),
            "metrics_rows": [
                ["Balanced accuracy", "media del recall delle classi; utile con classi sbilanciate"],
                ["Accuracy", "quota totale di predizioni corrette"],
                ["Precision macro", "precision calcolata per classe e poi mediata"],
                ["Recall macro", "sensibilità media tra le classi"],
                ["F1 macro", "media armonica macro di precision e recall"],
                ["Kappa / MCC", "accordo corretto per il caso / correlazione globale"],
                ["ROC-AUC", "capacità di ordinamento; può non essere disponibile"],
            ],
            "read_samples": (
                "In Risultato di ogni campione filtra corretti o errati. Confidenza è la "
                "probabilità della classe scelta; Margine è il distacco dalla seconda. "
                "Scostamento circuito-teoria deve restare vicino alla tolleranza numerica."
            ),
            "comparison_title": "5. Confronto con altri classificatori",
            "comparison_intro": (
                "Questa sezione è opzionale e non parte con il pulsante PGM. Abilitala, "
                "scegli il concorrente e avvia il confronto sul dataset selezionato. I due "
                "modelli usano gli stessi seed e gli stessi split esterni."
            ),
            "comparison_points": [
                "Il tuning del concorrente avviene soltanto sul training set.",
                "La differenza di balanced accuracy è appaiata seed per seed.",
                "Verde: PGM confermata; rosso: concorrente confermato.",
                "Giallo: direzione visibile ma intervallo al 95% include zero.",
                "Grigio: uguaglianza numerica esatta su tutti i seed.",
                "Full comparison ripete l'analisi sui dataset binari disponibili.",
            ],
            "circuit_title": "6. Circuito quantistico e ottimizzazione certificata",
            "circuit_intro": (
                "Dopo il training, l'app costruisce la POVM ridotta e la dilatazione di "
                "Naimark U_PGM. Il registro sys contiene lo stato test codificato; la "
                "misura del registro out restituisce l'indice della classe."
            ),
            "circuit_points": [
                "La tabella bitstring-classe indica, per esempio, 000 -> classe 1.",
                "Le probabilità circuitali sono confrontate con quelle teoriche.",
                "La decomposizione completa è isolata dal processo Streamlit.",
                "I livelli di ottimizzazione 1, 2 e 3 vengono confrontati.",
                "Un circuito è accettato solo se l'equivalenza numerica è certificata.",
                "Se il disegno è enorme, restano gate count, profondità e download QPY.",
            ],
            "quantum_title": "7. Simulatori, cloud e computer quantistici reali",
            "provider_rows": [
                ["Locale", "campionatore PGM ideale; Qiskit Aer"],
                ["LRZ / MQSS", "risorse autorizzate restituite dinamicamente"],
                ["IBM Quantum", "simulatori o QPU disponibili per l'account"],
                ["IonQ Cloud", "simulatori e QPU IonQ"],
                ["Amazon Braket", "simulatori gestiti e QPU supportate"],
            ],
            "quantum_steps": [
                "Scegli simulatore o hardware e configura shot e opzioni.",
                "Inserisci le credenziali solo nel campo protetto o nei secrets.",
                "Scopri i dispositivi: l'elenco dipende dall'account e dallo stato online.",
                "Esegui il preflight e verifica qubit, gate e costo potenziale.",
                "Conferma esplicitamente l'invio; poi controlla stato e risultato.",
            ],
            "cost_warning": (
                "Attenzione: un job su hardware o simulatore cloud può avere costi e code. "
                "PGM Lab non incorpora token, non garantisce la disponibilità dei device e "
                "non invia nulla prima della conferma finale."
            ),
            "export_title": "8. Download, privacy e buone pratiche",
            "export_rows": [
                ["CSV", "predizioni e classificazioni dettagliate"],
                ["SVG / TXT", "diagramma logico o decomposizione leggibile"],
                ["QPY", "circuito Qiskit serializzato"],
                ["NPZ", "U_PGM, effetti, classi e metadati di encoding"],
            ],
            "security_points": [
                "Non inserire mai token nel codice, nei CSV o nel repository.",
                "Revoca immediatamente una credenziale pubblicata per errore.",
                "Verifica il dataset e la licenza prima di caricare dati sensibili.",
                "I risultati scientifici dipendono da split, seed, prior e preprocessing.",
                "Conserva versione dell'app e parametri insieme agli export.",
            ],
            "trouble_title": "9. Risoluzione rapida dei problemi",
            "trouble_rows": [
                ["Il file non viene letto", "controlla formato, intestazioni, separatore e limiti"],
                ["Nessuna feature", "seleziona colonne numeriche variabili; escludi testo e ID"],
                ["Split non valido", "usa una quota proposta o aggiungi campioni alle classi rare"],
                ["Calcolo bloccato", "riduci c, alza il budget solo se la RAM esiste, o usa PCA manuale"],
                ["Circuito non creato", "la classificazione resta valida; riduci dimensione o copie"],
                ["Provider assente", "verifica dipendenze, credenziali, regione e autorizzazioni"],
                ["Connection error", "controlla i log Streamlit e riavvia il servizio"],
            ],
            "final_note": (
                "Per una prova controllata inizia con Iris, c=2 e 3 seed; passa poi a 10 "
                "seed. Con un file personale, verifica prima la distribuzione delle classi "
                "e una singola esecuzione, poi abilita full comparison o hardware remoto."
            ),
            "footer": "PGM Lab - Guida italiana",
        },
    ),
    "en": GuideText(
        language_code="en",
        language_name="English",
        file_name="PGM_Lab_Illustrated_User_Guide_EN.pdf",
        title="PGM Lab",
        subtitle="Illustrated user guide",
        updated="Updated for version 5.3.0 - 8 October 2026",
        sections={
            "cover_note": (
                "From dataset selection to the PGM, from statistical comparison to "
                "quantum-circuit execution on simulators or QPUs."
            ),
            "contents": "Inside this guide",
            "contents_items": [
                "Orientation and quick start",
                "Public datasets and private upload",
                "Scientific settings and computation",
                "Results and comparison with other classifiers",
                "Quantum circuit, optimization, and execution",
                "Privacy, downloads, and troubleshooting",
            ],
            "start_title": "1. Orientation and quick start",
            "start_intro": (
                "PGM Lab compares c-PGM, k-PGM, and r-PGM, three mathematically "
                "equivalent realizations of the Pretty Good Measurement classifier. "
                "The interface selects the least expensive exact backend without "
                "changing the decision rule."
            ),
            "ui_map": "Page map",
            "quick": "Recommended path in 6 steps",
            "quick_steps": [
                "Choose a public dataset or upload your own file.",
                "Set the copy number c and, if needed, the RAM budget.",
                "Check dimensions, expected backend, and feasibility.",
                "Run the multi-seed PGM evaluation.",
                "Read metrics, standard deviations, confusion matrix, and samples.",
                "Only when needed, compare models or build and execute the circuit.",
            ],
            "language": (
                "The flags in the upper-right corner switch the entire interface. Data, "
                "settings, and results are not recomputed when the language changes."
            ),
            "upload_title": "2. Public datasets and private upload",
            "upload_intro": (
                "In section 1, open Upload your own dataset and drag the file into the "
                "dashed area. Once validated, it appears in the table and selector next "
                "to public datasets, marked with an upload icon."
            ),
            "upload_steps": [
                "Drag a CSV, TSV, TXT, or XLSX file (first Excel sheet).",
                "Check the suggested target column and correct it if needed.",
                "Confirm numeric features; likely identifiers are deselected.",
                "Choose a readable name and inspect the preview and class counts.",
                "Press Add to catalogue and select.",
            ],
            "requirements": "File requirements",
            "requirements_rows": [
                ["Format", "CSV, TSV/TXT, XLSX"],
                ["Size", "max 25 MB; 50,000 rows; 500 columns"],
                ["Structure", "first row with unique column names"],
                ["Target", "2-50 classes; at least 3 samples per class"],
                ["Features", "numeric; at least one varying feature"],
                ["Samples", "at least 10 after cleaning"],
            ],
            "validation": (
                "Rows without a target are removed. Missing feature values are imputed "
                "with the median learned only from each seed's training set. Text, empty, "
                "and constant columns are reported clearly."
            ),
            "privacy": (
                "A private file stays in session memory, never enters the repository, and "
                "does not use the shared cache for public datasets. Remove private dataset "
                "also clears results that depend on that file."
            ),
            "settings_title": "3. Scientific settings and dimension checks",
            "settings_intro": (
                "By default, the PGM uses every selected feature. PCA is never automatic: "
                "it is applied only when the user enables Manually request PCA reduction."
            ),
            "settings_rows": [
                ["Copy number c", "kernel power and symmetric-map order"],
                ["Test fraction", "only values compatible with a stratified split"],
                ["Seeds", "reproducible splits; default: 10 consecutive seeds"],
                ["Class prior", "uniform or empirical class frequencies"],
                ["Spectral threshold", "controls numerical rank consistently"],
                ["RAM budget", "safety limit; it does not increase server RAM"],
            ],
            "encoding": (
                "On each training set, the app compares amplitude encoding with "
                "stereographic encoding at t = 0.1, 0.2, 0.5, 1, 2. The test set never "
                "participates in this choice. The winning encoding is reported."
            ),
            "dimension_note": (
                "When the circuit is too large, classical classification may remain "
                "available. The exact circuit is materialized up to 9 qubits; automatic "
                "full synthesis is prudently limited to 7 qubits."
            ),
            "results_title": "4. Running the PGM and interpreting results",
            "results_intro": (
                "Press Run the multi-seed evaluation. The main summary uses mean and "
                "standard deviation across all splits; one reference seed is used only "
                "for diagnostics, sample details, and circuit construction."
            ),
            "metrics_rows": [
                ["Balanced accuracy", "mean class recall; useful for imbalanced classes"],
                ["Accuracy", "overall fraction of correct predictions"],
                ["Macro precision", "precision computed per class and then averaged"],
                ["Macro recall", "mean sensitivity across classes"],
                ["Macro F1", "macro harmonic mean of precision and recall"],
                ["Kappa / MCC", "chance-corrected agreement / global correlation"],
                ["ROC-AUC", "ranking quality; may be unavailable for some models"],
            ],
            "read_samples": (
                "In Result for each sample, filter correct or incorrect cases. Confidence "
                "is the probability of the selected class; Margin is the gap to the runner-"
                "up. Circuit-theory deviation should remain near numerical tolerance."
            ),
            "comparison_title": "5. Comparison with other classifiers",
            "comparison_intro": (
                "This section is optional and does not run with the PGM button. Enable it, "
                "choose a competitor, and start the selected-dataset comparison. Both "
                "models use the same seeds and outer splits."
            ),
            "comparison_points": [
                "Competitor tuning uses training data only.",
                "Balanced-accuracy differences are paired seed by seed.",
                "Green: confirmed PGM win; red: confirmed competitor win.",
                "Yellow: visible direction, but the 95% interval includes zero.",
                "Gray: exact numerical equality on every seed.",
                "Full comparison repeats the analysis on available binary datasets.",
            ],
            "circuit_title": "6. Quantum circuit and certified optimization",
            "circuit_intro": (
                "After training, the app constructs the reduced POVM and the Naimark "
                "dilation U_PGM. The sys register contains the encoded test state; "
                "measuring the out register returns the class index."
            ),
            "circuit_points": [
                "The bitstring-to-class table shows mappings such as 000 -> class 1.",
                "Circuit probabilities are compared with theoretical probabilities.",
                "Full decomposition runs outside the Streamlit process.",
                "Optimization levels 1, 2, and 3 are compared.",
                "A circuit is accepted only after numerical equivalence certification.",
                "For huge diagrams, gate count, depth, and QPY download remain available.",
            ],
            "quantum_title": "7. Simulators, cloud services, and real QPUs",
            "provider_rows": [
                ["Local", "ideal PGM sampler; Qiskit Aer"],
                ["LRZ / MQSS", "authorized resources discovered dynamically"],
                ["IBM Quantum", "simulators or QPUs available to the account"],
                ["IonQ Cloud", "IonQ simulators and QPUs"],
                ["Amazon Braket", "managed simulators and supported QPUs"],
            ],
            "quantum_steps": [
                "Choose a simulator or QPU and configure shots and options.",
                "Enter credentials only in the protected field or secrets.",
                "Discover devices; the list depends on account rights and online status.",
                "Run preflight and check qubits, gates, and potential cost.",
                "Explicitly confirm submission, then inspect status and result.",
            ],
            "cost_warning": (
                "Caution: cloud simulator or hardware jobs may incur costs and queues. "
                "PGM Lab embeds no tokens, cannot guarantee device availability, and "
                "submits nothing before final confirmation."
            ),
            "export_title": "8. Downloads, privacy, and good practice",
            "export_rows": [
                ["CSV", "predictions and detailed classifications"],
                ["SVG / TXT", "logical diagram or readable decomposition"],
                ["QPY", "serialized Qiskit circuit"],
                ["NPZ", "U_PGM, effects, classes, and encoding metadata"],
            ],
            "security_points": [
                "Never place tokens in code, CSV files, or the repository.",
                "Immediately revoke a credential published by mistake.",
                "Check dataset permissions before uploading sensitive data.",
                "Scientific results depend on splits, seeds, priors, and preprocessing.",
                "Store the app version and parameters alongside exports.",
            ],
            "trouble_title": "9. Quick troubleshooting",
            "trouble_rows": [
                ["File is not read", "check format, headers, delimiter, and size limits"],
                ["No features", "select varying numeric columns; exclude text and IDs"],
                ["Invalid split", "use an offered fraction or add samples to rare classes"],
                ["Computation blocked", "reduce c, raise budget only with real RAM, or request PCA"],
                ["Circuit not built", "classification stays valid; reduce dimensions or copies"],
                ["Provider absent", "check packages, credentials, region, and authorization"],
                ["Connection error", "inspect Streamlit logs and restart the service"],
            ],
            "final_note": (
                "For a controlled first run, use Iris, c=2, and 3 seeds, then move to 10 "
                "seeds. With a private file, first inspect class distribution and one run; "
                "only then enable full comparison or remote hardware."
            ),
            "footer": "PGM Lab - English user guide",
        },
    ),
}


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=31,
            leading=34,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=base["Heading2"],
            fontName=FONT,
            fontSize=16,
            leading=21,
            textColor=HexColor("#DDE6FF"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12.5,
            leading=15,
            textColor=BLUE,
            spaceBefore=6,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.2,
            leading=13.1,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.8,
            leading=10.2,
            textColor=MUTED,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.8,
            leading=12.2,
            leftIndent=13,
            firstLineIndent=-8,
            textColor=INK,
            spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.7,
            leading=12,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=8.1,
            leading=10,
            textColor=WHITE,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.7,
            leading=9.7,
            textColor=INK,
        ),
        "table_bold": ParagraphStyle(
            "table_bold",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.7,
            leading=9.7,
            textColor=INK,
        ),
    }


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def bullet_list(items: list[str], st: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [P(f"<b>{index}.</b> {item}", st["bullet"]) for index, item in enumerate(items, 1)]


def dot_list(items: list[str], st: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [P(f"<font color='#4767C7'>&bull;</font> {item}", st["bullet"]) for item in items]


def callout(
    text: str,
    st: dict[str, ParagraphStyle],
    *,
    background: colors.Color = PALE_BLUE,
    border: colors.Color = BLUE,
) -> Table:
    table = Table([[P(text, st["callout"])]], colWidths=[CONTENT_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def data_table(
    rows: list[list[str]],
    headers: tuple[str, str],
    st: dict[str, ParagraphStyle],
    widths: tuple[float, float] = (0.31, 0.69),
) -> Table:
    data = [
        [P(headers[0], st["table_header"]), P(headers[1], st["table_header"])]
    ]
    data.extend(
        [P(row[0], st["table_bold"]), P(row[1], st["table"])] for row in rows
    )
    table = Table(
        data,
        colWidths=[CONTENT_WIDTH * widths[0], CONTENT_WIDTH * widths[1]],
        repeatRows=1,
    )
    commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def _label(drawing: Drawing, x: float, y: float, text: str, size: int = 8, color=INK, bold=False):
    drawing.add(
        String(
            x,
            y,
            text,
            fontName=FONT_BOLD if bold else FONT,
            fontSize=size,
            fillColor=color,
        )
    )


def ui_map_drawing(language: str) -> Drawing:
    d = Drawing(CONTENT_WIDTH, 166)
    d.add(Rect(0, 0, CONTENT_WIDTH, 166, rx=8, ry=8, fillColor=WHITE, strokeColor=GRID))
    d.add(Rect(0, 143, CONTENT_WIDTH, 23, rx=8, ry=8, fillColor=NAVY, strokeColor=NAVY))
    _label(d, 13, 151, "PGM Lab", 10, WHITE, True)
    _label(d, CONTENT_WIDTH - 72, 151, "IT   EN", 8, HexColor("#DDE6FF"), True)
    labels_it = [
        ("1  Dataset", "Catalogo, upload, copie e RAM"),
        ("2  Dimensioni", "Backend, memoria e circuito previsto"),
        ("3  Confronto", "Benchmark opzionale e full comparison"),
        ("4  Risultati", "Metriche, campioni e complessita"),
        ("5  Quantum", "Circuito, ottimizzazione e provider"),
    ]
    labels_en = [
        ("1  Dataset", "Catalogue, upload, copies, and RAM"),
        ("2  Dimensions", "Backend, memory, and circuit preview"),
        ("3  Comparison", "Optional benchmark and full comparison"),
        ("4  Results", "Metrics, samples, and complexity"),
        ("5  Quantum", "Circuit, optimization, and providers"),
    ]
    labels = labels_it if language == "it" else labels_en
    row_colors = [BLUE, CYAN, PURPLE, GREEN, AMBER]
    for index, ((title, subtitle), row_color) in enumerate(zip(labels, row_colors)):
        y = 116 - 27 * index
        d.add(Rect(12, y, 25, 18, rx=4, ry=4, fillColor=row_color, strokeColor=row_color))
        _label(d, 19, y + 6, str(index + 1), 8, WHITE, True)
        _label(d, 48, y + 10, title, 8, NAVY, True)
        _label(d, 176, y + 10, subtitle, 7.5, MUTED)
        if index < 4:
            d.add(Line(24.5, y - 8, 24.5, y, strokeColor=GRID, strokeWidth=1.5))
    return d


def upload_drawing(language: str) -> Drawing:
    d = Drawing(CONTENT_WIDTH, 128)
    labels = (
        ["1. TRASCINA", "2. VERIFICA", "3. USA"]
        if language == "it"
        else ["1. DRAG", "2. VALIDATE", "3. USE"]
    )
    subtitles = (
        ["CSV / TSV / TXT / XLSX", "target, feature, classi", "catalogo, PGM, confronti"]
        if language == "it"
        else ["CSV / TSV / TXT / XLSX", "target, features, classes", "catalogue, PGM, comparisons"]
    )
    colors_list = [BLUE, CYAN, GREEN]
    box_width = 142
    gap = (CONTENT_WIDTH - 3 * box_width) / 2
    for index in range(3):
        x = index * (box_width + gap)
        d.add(Rect(x, 17, box_width, 92, rx=10, ry=10, fillColor=WHITE, strokeColor=colors_list[index], strokeWidth=1.5))
        d.add(Circle(x + 25, 83, 13, fillColor=colors_list[index], strokeColor=colors_list[index]))
        _label(d, x + 21, 79, str(index + 1), 9, WHITE, True)
        _label(d, x + 48, 81, labels[index], 8.5, NAVY, True)
        _label(d, x + 13, 48, subtitles[index], 7.2, MUTED)
        if index == 0:
            d.add(Rect(x + 46, 22, 48, 17, rx=3, ry=3, fillColor=PALE_BLUE, strokeColor=BLUE, strokeDashArray=[3, 2]))
        elif index == 1:
            for row in range(3):
                d.add(Rect(x + 35, 22 + row * 7, 70, 4, fillColor=PALE_GREEN if row < 2 else PALE_AMBER, strokeColor=None))
        else:
            d.add(Rect(x + 42, 22, 56, 18, rx=4, ry=4, fillColor=GREEN, strokeColor=GREEN))
            _label(d, x + 54, 28, "PGM", 7.5, WHITE, True)
        if index < 2:
            start = x + box_width + 5
            d.add(Line(start, 63, start + gap - 10, 63, strokeColor=MUTED, strokeWidth=1.2))
            d.add(Line(start + gap - 16, 68, start + gap - 10, 63, strokeColor=MUTED, strokeWidth=1.2))
            d.add(Line(start + gap - 16, 58, start + gap - 10, 63, strokeColor=MUTED, strokeWidth=1.2))
    return d


def leakage_safe_drawing(language: str) -> Drawing:
    d = Drawing(CONTENT_WIDTH, 136)
    title = "FLUSSO SENZA DATA LEAKAGE" if language == "it" else "LEAKAGE-SAFE WORKFLOW"
    _label(d, 0, 124, title, 8, NAVY, True)
    boxes = (
        [
            ("Dataset", "grezzo"),
            ("Split", "stratificato"),
            ("Training", "imputer + encoding"),
            ("Test", "solo trasformazione"),
            ("Metriche", "multi-seed"),
        ]
        if language == "it"
        else [
            ("Dataset", "raw"),
            ("Split", "stratified"),
            ("Training", "imputer + encoding"),
            ("Test", "transform only"),
            ("Metrics", "multi-seed"),
        ]
    )
    widths = [78, 80, 112, 92, 82]
    gap = 12
    x = 0
    palette = [NAVY, BLUE, PURPLE, CYAN, GREEN]
    for index, ((line1, line2), width, fill) in enumerate(zip(boxes, widths, palette)):
        d.add(Rect(x, 47, width, 50, rx=7, ry=7, fillColor=fill, strokeColor=fill))
        _label(d, x + 10, 76, line1, 8, WHITE, True)
        _label(d, x + 10, 61, line2, 7, HexColor("#E8EEFF"))
        if index < len(boxes) - 1:
            d.add(Line(x + width + 2, 72, x + width + gap - 2, 72, strokeColor=MUTED))
        x += width + gap
    caption = (
        "Il test non decide imputazione, PCA, scaling, encoding o fattore t."
        if language == "it"
        else "The test set never selects imputation, PCA, scaling, encoding, or t."
    )
    d.add(Rect(0, 9, CONTENT_WIDTH, 24, rx=5, ry=5, fillColor=PALE_GREEN, strokeColor=GREEN))
    _label(d, 12, 17, caption, 7.5, GREEN, True)
    return d


def confusion_drawing(language: str) -> Drawing:
    d = Drawing(CONTENT_WIDTH, 154)
    title = "ESEMPIO DI LETTURA" if language == "it" else "READING EXAMPLE"
    _label(d, 0, 142, title, 8, NAVY, True)
    origin_x, origin_y, cell = 32, 22, 32
    values = [[18, 2, 0], [1, 15, 2], [0, 3, 19]]
    for row in range(3):
        for col in range(3):
            diagonal = row == col
            fill = PALE_GREEN if diagonal else PALE_RED
            d.add(Rect(origin_x + col * cell, origin_y + (2 - row) * cell, cell, cell, fillColor=fill, strokeColor=WHITE))
            _label(d, origin_x + col * cell + 11, origin_y + (2 - row) * cell + 12, str(values[row][col]), 9, GREEN if diagonal else RED, True)
    _label(d, 31, 123, "predetta" if language == "it" else "predicted", 7, MUTED)
    _label(d, 1, 72, "reale" if language == "it" else "actual", 7, MUTED)
    # Companion probability card.
    d.add(Rect(190, 22, CONTENT_WIDTH - 190, 96, rx=8, ry=8, fillColor=WHITE, strokeColor=GRID))
    sample = "Campione 7" if language == "it" else "Sample 7"
    verdict = "Corretta - decisione netta" if language == "it" else "Correct - clear decision"
    _label(d, 205, 96, sample, 9, NAVY, True)
    _label(d, 205, 80, verdict, 8, GREEN, True)
    labels = ["A  0.82", "B  0.13", "C  0.05"]
    lengths = [180, 60, 25]
    for index, (label, length) in enumerate(zip(labels, lengths)):
        y = 58 - index * 17
        _label(d, 205, y, label, 7.2, INK)
        d.add(Rect(265, y - 1, length, 7, rx=2, ry=2, fillColor=[GREEN, AMBER, RED][index], strokeColor=None))
    return d


def comparison_drawing(language: str) -> Drawing:
    d = Drawing(CONTENT_WIDTH, 146)
    _label(d, 0, 134, "DIFFERENZA DI BALANCED ACCURACY" if language == "it" else "BALANCED-ACCURACY DIFFERENCE", 8, NAVY, True)
    axis_x0, axis_x1, axis_y = 52, CONTENT_WIDTH - 34, 73
    d.add(Line(axis_x0, axis_y, axis_x1, axis_y, strokeColor=GRID, strokeWidth=2))
    zero_x = (axis_x0 + axis_x1) / 2
    d.add(Line(zero_x, 32, zero_x, 113, strokeColor=NAVY, strokeWidth=1.2))
    _label(d, axis_x0, 18, "concorrente" if language == "it" else "competitor", 7, RED, True)
    _label(d, axis_x1 - 28, 18, "PGM", 7, GREEN, True)
    examples = [
        ("A", zero_x + 82, 44, 108, GREEN),
        ("B", zero_x - 64, 43, 92, RED),
        ("C", zero_x + 20, 90, 65, AMBER),
    ]
    for name, point_x, half_width, y, color in examples:
        _label(d, 17, y - 3, name, 8, NAVY, True)
        d.add(Line(point_x - half_width, y, point_x + half_width, y, strokeColor=color, strokeWidth=4))
        d.add(Circle(point_x, y, 5, fillColor=color, strokeColor=WHITE, strokeWidth=1))
    note = (
        "Il segmento e l'intervallo al 95%; il punto e la differenza media."
        if language == "it"
        else "The segment is the 95% interval; the point is the mean difference."
    )
    _label(d, 84, 3, note, 7.2, MUTED)
    return d


def circuit_drawing(language: str) -> Drawing:
    d = Drawing(CONTENT_WIDTH, 154)
    _label(d, 0, 142, "DILATAZIONE DI NAIMARK" if language == "it" else "NAIMARK DILATION", 8, NAVY, True)
    wire_y = [104, 79, 54]
    wire_labels = ["out[0] |0>", "sys[0] |psi>", "sys[1] |psi>"]
    for y, label in zip(wire_y, wire_labels):
        _label(d, 0, y - 3, label, 7.2, MUTED)
        d.add(Line(78, y, CONTENT_WIDTH - 12, y, strokeColor=MUTED, strokeWidth=1.1))
    d.add(Rect(132, 40, 172, 79, rx=8, ry=8, fillColor=PALE_BLUE, strokeColor=PURPLE, strokeWidth=1.5))
    _label(d, 188, 87, "U_PGM", 13, PURPLE, True)
    _label(d, 159, 66, "POVM + isometria" if language == "it" else "POVM + isometry", 7.5, MUTED)
    d.add(Rect(365, 92, 42, 25, rx=5, ry=5, fillColor=PALE_GREEN, strokeColor=GREEN))
    _label(d, 379, 101, "M", 10, GREEN, True)
    d.add(Line(407, 104, 452, 104, strokeColor=GREEN, strokeWidth=2))
    _label(d, 414, 114, "000", 7, GREEN, True)
    mapping = "000 -> classe 1" if language == "it" else "000 -> class 1"
    _label(d, 351, 18, mapping, 8, NAVY, True)
    d.add(Rect(0, 2, 318, 25, rx=5, ry=5, fillColor=PALE_GREEN, strokeColor=GREEN))
    certificate = (
        "Ottimizzato solo se ||U_ref - U_opt|| <= tolleranza"
        if language == "it"
        else "Optimized only if ||U_ref - U_opt|| <= tolerance"
    )
    _label(d, 10, 10, certificate, 7.2, GREEN, True)
    return d


def provider_drawing(language: str) -> Drawing:
    d = Drawing(CONTENT_WIDTH, 135)
    center_x = CONTENT_WIDTH / 2 - 53
    d.add(Rect(center_x, 48, 106, 42, rx=8, ry=8, fillColor=NAVY, strokeColor=NAVY))
    _label(d, center_x + 23, 66, "PGM Lab", 11, WHITE, True)
    providers = [
        ("Aer", 8, 93, BLUE),
        ("LRZ", 8, 18, PURPLE),
        ("IBM", CONTENT_WIDTH - 78, 93, CYAN),
        ("IonQ", CONTENT_WIDTH - 78, 18, GREEN),
        ("AWS", center_x + 17, 3, AMBER),
    ]
    for label, x, y, color in providers:
        d.add(Rect(x, y, 70, 29, rx=6, ry=6, fillColor=WHITE, strokeColor=color, strokeWidth=1.4))
        _label(d, x + 22, y + 10, label, 8.5, color, True)
        target_x = center_x if x < center_x else center_x + 106
        target_y = 69
        source_x = x + 70 if x < center_x else x
        source_y = y + 14
        if label == "AWS":
            source_x, source_y = x + 35, y + 29
            target_x, target_y = center_x + 53, 48
        d.add(Line(source_x, source_y, target_x, target_y, strokeColor=GRID, strokeWidth=1.1))
    note = (
        "Preflight -> conferma -> invio"
        if language == "it"
        else "Preflight -> confirmation -> submission"
    )
    _label(d, center_x - 5, 108, note, 7.2, MUTED, True)
    return d


def privacy_drawing(language: str) -> Drawing:
    d = Drawing(CONTENT_WIDTH, 118)
    d.add(Rect(0, 12, 238, 90, rx=9, ry=9, fillColor=PALE_GREEN, strokeColor=GREEN, strokeWidth=1.5))
    d.add(Rect(CONTENT_WIDTH - 238, 12, 238, 90, rx=9, ry=9, fillColor=PALE_BLUE, strokeColor=BLUE, strokeWidth=1.5))
    left_title = "SESSIONE PRIVATA" if language == "it" else "PRIVATE SESSION"
    right_title = "RISORSE PUBBLICHE" if language == "it" else "PUBLIC RESOURCES"
    _label(d, 18, 80, left_title, 9, GREEN, True)
    _label(d, CONTENT_WIDTH - 220, 80, right_title, 9, BLUE, True)
    left_lines = (
        ["file caricato", "risultati e circuito", "nessuna cache condivisa"]
        if language == "it"
        else ["uploaded file", "results and circuit", "no shared cache"]
    )
    right_lines = (
        ["catalogo OpenML/UCI", "cache di download", "guide PDF"]
        if language == "it"
        else ["OpenML/UCI catalogue", "download cache", "PDF guides"]
    )
    for index, line in enumerate(left_lines):
        _label(d, 23, 59 - index * 15, "- " + line, 7.5, INK)
    for index, line in enumerate(right_lines):
        _label(d, CONTENT_WIDTH - 215, 59 - index * 15, "- " + line, 7.5, INK)
    d.add(Line(252, 20, 252, 94, strokeColor=RED, strokeWidth=2, strokeDashArray=[4, 3]))
    _label(d, 242, 4, "NO", 7, RED, True)
    return d


class GuideDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, guide: GuideText, **kwargs):
        super().__init__(filename, **kwargs)
        self.guide = guide
        frame = Frame(
            MARGIN_X,
            MARGIN_BOTTOM,
            CONTENT_WIDTH,
            PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            id="main",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="guide", frames=[frame], onPage=self._page))

    def _page(self, canvas, doc):
        canvas.saveState()
        if doc.page == 1:
            canvas.setFillColor(NAVY)
            canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
            canvas.setFillColor(BLUE)
            canvas.circle(PAGE_WIDTH - 38 * mm, PAGE_HEIGHT - 36 * mm, 35 * mm, fill=1, stroke=0)
            canvas.setFillColor(PURPLE)
            canvas.circle(PAGE_WIDTH - 8 * mm, PAGE_HEIGHT - 62 * mm, 24 * mm, fill=1, stroke=0)
        else:
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.5)
            canvas.line(MARGIN_X, PAGE_HEIGHT - 13 * mm, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 13 * mm)
            canvas.setFont(FONT_BOLD, 7.5)
            canvas.setFillColor(NAVY)
            canvas.drawString(MARGIN_X, PAGE_HEIGHT - 9.5 * mm, "PGM Lab")
            canvas.setFont(FONT, 7.2)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 9.5 * mm, f"v{APP_VERSION}")
            canvas.line(MARGIN_X, 11 * mm, PAGE_WIDTH - MARGIN_X, 11 * mm)
            canvas.drawString(MARGIN_X, 7.2 * mm, str(self.guide.sections["footer"]))
            canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 7.2 * mm, str(doc.page))
        canvas.restoreState()


def cover_story(guide: GuideText, st: dict[str, ParagraphStyle]) -> list:
    s = guide.sections
    cover_small = ParagraphStyle(
        "cover_small",
        parent=st["small"],
        textColor=HexColor("#DDE6FF"),
        leading=11.5,
    )
    contents = [
        P(f"<b>{index}.</b> {item}", cover_small)
        for index, item in enumerate(s["contents_items"], 1)
    ]
    contents_table = Table([[contents[:3], contents[3:]]], colWidths=[CONTENT_WIDTH / 2 - 8, CONTENT_WIDTH / 2 - 8])
    contents_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return [
        Spacer(1, 28 * mm),
        P(guide.title, st["cover_title"]),
        P(guide.subtitle, st["cover_subtitle"]),
        Spacer(1, 4 * mm),
        Table(
            [[P(guide.updated, cover_small)]],
            colWidths=[92 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), HexColor("#2B3A62")),
                    ("BOX", (0, 0), (-1, -1), 0.6, HexColor("#6E83C9")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
                ]
            ),
        ),
        Spacer(1, 13 * mm),
        P(str(s["cover_note"]), ParagraphStyle("cover_note", parent=st["body"], fontSize=12, leading=17, textColor=WHITE, spaceAfter=10)),
        Spacer(1, 8 * mm),
        P(str(s["contents"]), ParagraphStyle("cover_contents", parent=st["h2"], textColor=HexColor("#DDE6FF"))),
        contents_table,
        Spacer(1, 14 * mm),
        P(f"<link href='{APP_URL}' color='#DDE6FF'>{APP_URL}</link>", ParagraphStyle("cover_link", parent=st["small"], textColor=HexColor("#DDE6FF"))),
        PageBreak(),
    ]


def build_story(guide: GuideText) -> list:
    st = styles()
    s = guide.sections
    header_pair = ("Voce", "Significato") if guide.language_code == "it" else ("Item", "Meaning")
    story: list = cover_story(guide, st)

    story.extend(
        [
            P(str(s["start_title"]), st["h1"]),
            P(str(s["start_intro"]), st["body"]),
            P(str(s["ui_map"]), st["h2"]),
            ui_map_drawing(guide.language_code),
            Spacer(1, 5),
            P(str(s["quick"]), st["h2"]),
            *bullet_list(list(s["quick_steps"]), st),
            Spacer(1, 4),
            callout(str(s["language"]), st, background=PALE_BLUE, border=BLUE),
            PageBreak(),
            P(str(s["upload_title"]), st["h1"]),
            P(str(s["upload_intro"]), st["body"]),
            upload_drawing(guide.language_code),
            *bullet_list(list(s["upload_steps"]), st),
            P(str(s["requirements"]), st["h2"]),
            data_table(list(s["requirements_rows"]), header_pair, st),
            Spacer(1, 7),
            callout(str(s["validation"]), st, background=PALE_AMBER, border=AMBER),
            Spacer(1, 6),
            callout(str(s["privacy"]), st, background=PALE_GREEN, border=GREEN),
            PageBreak(),
            P(str(s["settings_title"]), st["h1"]),
            P(str(s["settings_intro"]), st["body"]),
            leakage_safe_drawing(guide.language_code),
            data_table(list(s["settings_rows"]), header_pair, st),
            Spacer(1, 7),
            callout(str(s["encoding"]), st, background=PALE_BLUE, border=PURPLE),
            Spacer(1, 6),
            callout(str(s["dimension_note"]), st, background=PALE_AMBER, border=AMBER),
            PageBreak(),
            P(str(s["results_title"]), st["h1"]),
            P(str(s["results_intro"]), st["body"]),
            confusion_drawing(guide.language_code),
            data_table(list(s["metrics_rows"]), header_pair, st),
            Spacer(1, 7),
            callout(str(s["read_samples"]), st, background=PALE_GREEN, border=GREEN),
            PageBreak(),
            P(str(s["comparison_title"]), st["h1"]),
            P(str(s["comparison_intro"]), st["body"]),
            comparison_drawing(guide.language_code),
            *dot_list(list(s["comparison_points"]), st),
            Spacer(1, 8),
            callout(
                (
                    "Il giallo non significa pareggio: mostra un vantaggio medio la cui "
                    "incertezza non permette ancora una conclusione al 95%."
                    if guide.language_code == "it"
                    else "Yellow does not mean a tie: it shows a mean advantage whose "
                    "uncertainty does not yet support a 95% conclusion."
                ),
                st,
                background=PALE_AMBER,
                border=AMBER,
            ),
            PageBreak(),
            P(str(s["circuit_title"]), st["h1"]),
            P(str(s["circuit_intro"]), st["body"]),
            circuit_drawing(guide.language_code),
            *dot_list(list(s["circuit_points"]), st),
            Spacer(1, 8),
            callout(
                (
                    "Ottimizzato non significa approssimato: l'app conserva soltanto "
                    "circuiti che superano il certificato di equivalenza."
                    if guide.language_code == "it"
                    else "Optimized does not mean approximated: the app retains only "
                    "circuits that pass equivalence certification."
                ),
                st,
                background=PALE_GREEN,
                border=GREEN,
            ),
            PageBreak(),
            P(str(s["quantum_title"]), st["h1"]),
            provider_drawing(guide.language_code),
            data_table(
                list(s["provider_rows"]),
                ("Modalita", "Disponibilita") if guide.language_code == "it" else ("Mode", "Availability"),
                st,
            ),
            Spacer(1, 7),
            *bullet_list(list(s["quantum_steps"]), st),
            Spacer(1, 6),
            callout(str(s["cost_warning"]), st, background=PALE_RED, border=RED),
            PageBreak(),
            P(str(s["export_title"]), st["h1"]),
            privacy_drawing(guide.language_code),
            data_table(
                list(s["export_rows"]),
                ("Formato", "Contenuto") if guide.language_code == "it" else ("Format", "Contents"),
                st,
            ),
            Spacer(1, 7),
            *dot_list(list(s["security_points"]), st),
            PageBreak(),
            P(str(s["trouble_title"]), st["h1"]),
            data_table(
                list(s["trouble_rows"]),
                ("Problema", "Azione consigliata") if guide.language_code == "it" else ("Problem", "Recommended action"),
                st,
                widths=(0.30, 0.70),
            ),
            Spacer(1, 12),
            callout(str(s["final_note"]), st, background=PALE_BLUE, border=BLUE),
            Spacer(1, 18),
            P(
                (
                    "Questa guida e versionata insieme all'app. Se APP_VERSION cambia, "
                    "i test richiedono l'aggiornamento delle guide prima del rilascio."
                    if guide.language_code == "it"
                    else "This guide is versioned with the app. If APP_VERSION changes, "
                    "the tests require guide regeneration before release."
                ),
                st["small"],
            ),
        ]
    )
    return story


def generate(output_directory: Path) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for guide in TEXTS.values():
        output_path = output_directory / guide.file_name
        doc = GuideDocTemplate(
            str(output_path),
            guide,
            pagesize=A4,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=f"{guide.title} - {guide.subtitle}",
            author="PGM Lab",
            subject=f"PGM Lab {APP_VERSION} user guide ({guide.language_code})",
            keywords="PGM Lab, c-PGM, k-PGM, r-PGM, quantum classifier, user guide",
        )
        doc.build(build_story(guide))
        generated.append(output_path)
    manifest = {
        "app_version": APP_VERSION,
        "generated_on": GUIDE_RELEASE_DATE,
        "generator": "scripts/generate_user_guides.py",
        "files": {
            guide.language_code: guide.file_name for guide in TEXTS.values()
        },
    }
    (output_directory / "guide_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return generated


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/pdf")
    for path in generate(output):
        print(path)


if __name__ == "__main__":
    main()
