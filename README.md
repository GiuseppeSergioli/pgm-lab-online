# PGM Lab: c-PGM, k-PGM e r-PGM

**Versione 4.1.0**

Applicazione Streamlit per confrontare le tre formulazioni equivalenti descritte in
*Computational Complexity Analysis of Quantum-Inspired Pretty Good Measurement
Classifiers*:

- **c-PGM** nel tensor space esplicito, di dimensione `d**c`;
- **k-PGM** nella rappresentazione duale, tramite il kernel
  `k(x, z) = <x, z>**c`;
- **r-PGM** (Rc-PGM nel paper) nella base del sottospazio simmetrico, di dimensione
  `comb(d + c - 1, c)`.

L'app mostra accuratezza, accordo campione-per-campione, scarto tra gli score,
rank numerico, tempi osservati e complessità teoriche di tempo e memoria. Dopo il
training costruisce inoltre una dilatazione di Naimark (chiamata *Neumark dilation*
nel paper) della PGM, ne disegna il circuito Qiskit e verifica le sue probabilità
sul test set.

## Novità della versione 4.1.0

- In **Classificazione test** la corrispondenza tra bitstring misurata e classe è
  nuovamente visibile in primo piano, per esempio `000 → classe 1`.
- In **Risultato di ogni campione** è presente una colonna *Lettura rapida* che
  traduce esito, confidenza e margine in una breve frase.
- La scoperta LRZ usa il filtro per le risorse online e non si interrompe più se
  un singolo backend espone metadati `Target` temporaneamente non disponibili.
- La scheda **Esecuzione quantistica** permette ora di configurare numero di shot,
  seed, livello di ottimizzazione, rumore, provider e dispositivo.
- Sono disponibili Qiskit Aer locale, IonQ Cloud, IBM Quantum e Amazon Braket,
  oltre a LRZ/MQSS e al campionatore PGM ideale locale.
- Le credenziali possono essere caricate automaticamente da variabili d'ambiente
  o da `.streamlit/secrets.toml`, senza inserirle nel codice o negli export.

## Avvio

Il confronto locale richiede Python 3.10 o successivo. Per usare sia Amazon Braket
sia LRZ/MQSS è raccomandato Python 3.11, 3.12 o 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Su macOS, dopo la prima installazione, si può anche fare doppio clic su
`Avvia_PGM_Lab.command`. Il launcher cerca prima `.venv`, poi l'ambiente Conda
`pgm-lab`, imposta i limiti di thread che evitano crash delle librerie numeriche e
avvia Streamlit con `faulthandler`. Se macOS blocca il primo avvio, usare tasto
destro → **Apri** una sola volta.

I connettori hardware sono opzionali. Su macOS il metodo consigliato è fare
doppio clic su `Installa_Provider_Quantistici.command` e scegliere LRZ, Amazon
AWS, Qiskit Aer, IBM Quantum, IonQ oppure l'installazione completa. L'installazione
separata impedisce che un problema di un provider blocchi gli altri. In alternativa:

```bash
python -m pip install -r requirements-simulators.txt
python -m pip install -r requirements-lrz.txt
python -m pip install -r requirements-aws.txt
python -m pip install -r requirements-ionq.txt
python -m pip install -r requirements-ibm.txt
```

Amazon Braket include dipendenze numeriche compilate. Se `pip` mostra
`Failed building wheel for llvmlite`, non tentare di compilare LLVM: seguire
`INSTALLAZIONE_PROVIDER_MAC.md` o usare lo script guidato, che installa prima i
binari Conda compatibili. `requirements-hardware.txt` resta disponibile per
l'installazione cumulativa su sistemi gia compatibili.

Il vincolo `joblib>=1.4,<1.6` risolve inoltre la sovrapposizione tra
`cloudpickle 2.2.1`, richiesto dal simulatore Braket 1.40.x, e `joblib 1.6+`,
senza violare il requisito `joblib>=1.4` di scikit-learn.

Haberman, Balance Scale, Ecoli e Glass richiedono una connessione al primo
caricamento da OpenML e usano poi la cache di scikit-learn. Iris e Wine sono
caricati dalle copie pubbliche UCI incluse direttamente in scikit-learn.

## Scelte numeriche esplicite

Il paper assume che i dati siano gia codificati come vettori unitari. L'app applica,
senza leakage dal test set:

1. split stratificato;
2. imputazione con la mediana appresa sul training set;
3. trasformazione min-max nell'intervallo `[0.001, 1]`, appresa sul training set;
4. normalizzazione L2 riga per riga.

L'estremo positivo `0.001` evita il vettore nullo senza aggiungere una feature: la
dimensione `d` mostrata nell'interfaccia resta quindi il numero di feature del
dataset.

Per i prior di classe sono disponibili due opzioni:

- **uniformi**, `p_j = 1/l`, impostazione predefinita coerente con l'Eq. (4);
- **empirici**, `p_j = n_j/N`.

In entrambi i casi il codice assegna a ciascun campione il peso
`alpha_i = p_(y_i) / n_(y_i)`. Il k-PGM usa il Gram pesato
`diag(sqrt(alpha)) K diag(sqrt(alpha))`: in questo modo l'equivalenza resta esatta
anche con classi sbilanciate e prior uniformi.

I tre metodi condividono la stessa soglia relativa sugli autovalori. Le predizioni
usano inoltre la stessa regola deterministica per i pareggi numerici.

## Complessita riportate

Con `D = d**c`, `S = comb(d+c-1, c)`, `N` campioni di training, `l` classi e
`r_G` rank numerico del Gram:

| Metodo | Training tempo | Training memoria | Predizione tempo | Predizione memoria |
|---|---:|---:|---:|---:|
| c-PGM | `O(max(N D^2, l D^3))` | `O(N D^2)` | `O(l D^2)` | `O(l D^2)` |
| k-PGM | `O(N^3)` | `O(N^2)` | `O(r_G N)` | `O(N(d+r_G))` |
| r-PGM | `O(max(N S^2, l S^3))` | `O(N S^2)` | `O(l S^2)` | `O(l S^2)` |

Le stime di memoria dell'interfaccia seguono il modello del paper e assumono
`float64`. I tempi cronometrici, invece, riguardano questa implementazione NumPy,
che sfrutta il supporto spettrale e non memorizza una matrice densità per campione.
Per questo motivo tempo osservato, stato del modello e bound didattici non devono
essere interpretati come la stessa quantita.

Il costo per costruire il vettore kernel di un nuovo campione è `O(Nd)`; il paper
riporta `O(r_G N)` per la successiva applicazione spettrale. L'interfaccia conserva
la formula del paper e documenta separatamente questa operazione.

Per il solo confronto classico, il termine comune `P_ker(sigma)/l` può essere omesso:
aggiunge lo stesso valore a tutti gli score e non cambia la classe scelta. Per il
circuito quantistico, invece, il codice lo reinserisce esplicitamente, ottenendo una
POVM completa con `sum_j F_j = I`.

## Circuito quantistico della PGM

L'app usa la rappresentazione r-PGM, perché è equivalente a c-PGM e k-PGM ma vive
nel sottospazio simmetrico minimo di dimensione
`S = comb(d+c-1, c)`. Il vettore codificato è inserito nei primi `S` stati di un
registro di `ceil(log2(S))` qubit; le ampiezze di padding sono nulle.

Partendo dagli effetti completi `F_j`, il codice costruisce l'isometria canonica

```text
V|psi> = sum_j |j>_out tensor sqrt(F_j)|psi>_sys
```

e la completa numericamente a una matrice unitaria `U_PGM`. Misurando il registro
`out` si ottiene quindi
`Pr(j|psi) = <psi|F_j|psi>`. Se il numero di classi non è una potenza di due, gli
esiti binari aggiuntivi hanno probabilità teorica zero. La dilatazione unitaria non
è unica: il completamento scelto può cambiare, mentre le probabilità sui dati con
il registro `out` inizializzato a zero restano le stesse.

Nell'interfaccia sono disponibili:

- il diagramma logico con il blocco unitario esatto `U_PGM`;
- confusion matrix assoluta e percentuale;
- classificazioni corrette/errate campione per campione, con frase interpretativa;
- mappa sempre visibile fra bitstring del registro di uscita e classe;
- probabilità teoriche, probabilità circuitali, confidenza e margine;
- i residui numerici di completezza, isometria, unitarietà e regola di Born;
- il confronto tra le predizioni del circuito e quelle di r-PGM;
- il download del diagramma SVG, del circuito QPY e delle matrici NPZ;
- fino a 6 qubit, una sintesi isolata nella base generica `rz`, `sx`, `x`, `cx`;
- il circuito completo su più righe fino a 5.000 porte e, oltre tale soglia, il
  conteggio esatto delle porte oppure un limite superiore prudenziale.

La sintesi Qiskit viene eseguita in un processo figlio con timeout e thread numerici
limitati. Un eventuale errore nativo del transpiler non può quindi terminare il
server Streamlit: l'app mostra l'errore e conserva training e risultati.

La matrice di un'unitaria generica occupa memoria esponenziale. Per questo la
materializzazione esatta è limitata a 8 qubit totali. Oltre tale soglia l'app mostra
lo schema dimensionato e la memoria richiesta, ma non alloca la matrice. Riducendo
`c` o scegliendo meno feature si torna al circuito esatto.

Il circuito logico mostrato implementa la **misura PGM** e assume che lo stato test
sia già preparato nel registro `sys`. Prima di un'esecuzione su simulatore remoto o
hardware, l'app genera un secondo circuito che include esplicitamente la state
preparation del campione selezionato e ripete la sintesi completa nel preflight.

Il diagramma è prodotto come SVG direttamente dal browser. Questa scelta evita le
dipendenze grafiche native Matplotlib/Pillow, che su alcune versioni macOS possono
causare un arresto anomalo del processo Streamlit, senza modificare il circuito
Qiskit o la matrice `U_PGM`.

## Simulatori e hardware reale

La scheda **Esecuzione quantistica** separa esplicitamente:

- campionatore PGM ideale locale, senza account né costo;
- Qiskit Aer locale, ideale o con rumore depolarizzante e di lettura;
- simulatore IonQ Cloud, ideale o con modello di rumore selezionabile;
- simulatori gestiti Amazon Braket;
- risorse LRZ restituite dinamicamente dal Munich Quantum Software Stack;
- QPU IBM Quantum, IonQ e Amazon Braket.

Per tutte le modalità sono configurabili gli shot. Quando pertinenti, sono
disponibili anche seed, livello di ottimizzazione del transpiler, metodo numerico,
modello di rumore, regione, profilo e dispositivo. I backend cloud vengono scoperti
dinamicamente in base alle autorizzazioni dell'account: nomi, disponibilità e code
non sono incorporati nel codice.

Si esegue inizialmente un campione test alla volta. Prima dell'invio sono mostrati
qubit, profondità, gate count e circuito completo quando leggibile. L'invio remoto
rimane disabilitato fino al completamento del preflight e a una conferma esplicita.
Il job è asincrono: l'app conserva job ID e oggetto del job nella sola sessione
corrente e offre pulsanti separati per stato, risultato e annullamento.

### LRZ Quantum

L'integrazione usa il pacchetto ufficiale `mqss-qiskit`, non il provider MQP
deprecato. Il token viene creato nel
[Munich Quantum Portal](https://portal.quantum.lrz.de/) e può essere incollato in
un campo password oppure caricato localmente da `LRZ_MQSS_TOKEN` o da Streamlit
Secrets. Non viene inserito nei circuiti, nei CSV, nei QPY, negli NPZ o nei log
dell'app. I backend non sono hard-coded: la ricerca usa
`MQSSQiskitAdapter.backends(online=True)` per mostrare, per impostazione
predefinita, solo le risorse online autorizzate.

La lettura dei metadati è isolata backend per backend. Se una risorsa, per esempio
MUNIQC-Atoms20, è elencata ma il relativo `Target` Qiskit non è disponibile, la
scoperta continua e può comunque mostrare una risorsa operativa come EQE1. Il
numero di qubit viene indicato come non disponibile soltanto per la risorsa che ha
restituito l'errore.

Eviden Qaptiva non viene presentato come backend MQSS: LRZ lo espone tramite un
flusso VPN/SSH/HPC separato. La relativa integrazione richiederebbe una modalità HPC
dedicata e credenziali infrastrutturali diverse.

Documentazione:

- [MQSS Qiskit Adapter](https://munich-quantum-software-stack.github.io/MQSS-Interfaces/qiskit/user_guide/getting_started/)
- [LRZ Quantum Computing](https://www.lrz.de/en/technologies/quantum-computing)

### Qiskit Aer

Qiskit Aer esegue localmente il circuito completo, inclusa la preparazione dello
stato del campione. Sono selezionabili i metodi `automatic`, `statevector`,
`density_matrix` e `matrix_product_state`. Il modello didattico di rumore permette
di impostare separatamente errore depolarizzante a uno e due qubit ed errore di
lettura. Questa modalità non usa token né servizi esterni.

### IonQ Quantum Cloud

Il provider `qiskit-ionq` espone il simulatore cloud e le QPU autorizzate per la API
key. Il simulatore può essere ideale o usare un modello di rumore IonQ, per esempio
`forte-1` o `forte-enterprise-1`. Poiché il simulatore ideale può restituire
probabilità anziché conteggi interi, l'app le converte in conteggi coerenti con il
numero di shot richiesto per mantenere uniforme la lettura dei risultati.

La chiave può essere fornita tramite `QISKIT_IONQ_API_TOKEN`, `IONQ_API_KEY`,
`IONQ_API_TOKEN` o Streamlit Secrets.

Documentazione:

- [Qiskit IonQ Provider](https://qiskit-community.github.io/qiskit-ionq/)
- [IonQ simulation with noise models](https://docs.ionq.com/features/simulation-with-noise-models)

### IBM Quantum

L'integrazione usa `qiskit-ibm-runtime`, scopre soltanto QPU operative con qubit
sufficienti e ordina l'elenco privilegiando le code più brevi quando il dato è
disponibile. L'invio avviene con `SamplerV2`; il risultato viene letto dal registro
classico del circuito e convertito nella stessa tabella usata dagli altri provider.

Token e CRN dell'istanza possono essere caricati da `QISKIT_IBM_TOKEN` e
`QISKIT_IBM_INSTANCE`, oppure da un account IBM già salvato localmente.

Documentazione:

- [IBM Quantum Runtime service](https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/qiskit-runtime-service)
- [IBM Quantum SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2)

### Amazon Braket

Amazon Braket non usa un singolo token. Per evitare di raccogliere Access Key e
Secret Key, l'app legge esclusivamente profili AWS locali. La configurazione
raccomandata usa IAM Identity Center/SSO:

```bash
aws configure sso --profile pgm-braket
aws sso login --profile pgm-braket
```

L'interfaccia permette quindi di scegliere profilo e regione, verifica l'identità
con STS e scopre dinamicamente simulatori o QPU online. Prezzi e nomi non sono
hard-coded. Prima di inviare una task a una risorsa AWS è obbligatoria una conferma
esplicita dei possibili costi.

Documentazione:

- [AWS CLI e IAM Identity Center](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
- [Amazon Braket e profili boto3](https://docs.aws.amazon.com/braket/latest/developerguide/braket-using-boto3-profiles.html)
- [Dispositivi Amazon Braket](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html)
- [Prezzi e limiti di spesa](https://docs.aws.amazon.com/braket/latest/developerguide/braket-pricing.html)

## Credenziali e token

Nessun token è incorporato nel pacchetto. Per ottenere il riempimento automatico
dei campi password su macOS, fare doppio clic su
`Configura_Credenziali.command`. Lo script richiede i valori senza mostrarli a
schermo e crea localmente `.streamlit/secrets.toml` con permessi restrittivi. Se il
file esiste già, un campo lasciato vuoto conserva il valore precedente; inserendo
`-` il valore di quel provider viene rimosso.

In alternativa, copiare `.streamlit/secrets.toml.example` come
`.streamlit/secrets.toml` e inserire credenziali nuove, oppure esportare le relative
variabili d'ambiente. Il file reale `secrets.toml` è escluso da `.gitignore` e non
deve essere inserito in archivi da condividere.

Se un token è stato incollato in chat, email, ticket o documenti condivisi, va
revocato e rigenerato prima dell'uso. Il pacchetto distribuito contiene soltanto
segnaposto.

## Protezione da configurazioni ingestibili

Il c-PGM esplicito cresce molto rapidamente. Prima di abilitare il calcolo, l'app
stima le principali allocazioni NumPy e controlla sia il budget RAM scelto sia la
dimensione della diagonalizzazione densa. Se il run è bloccato, le complessità
teoriche sono comunque mostrate; si può ridurre `c` o scegliere un dataset con meno
feature.

## Test

```bash
python -m unittest discover -s tests -v
```

I test verificano:

- identità delle Gram matrix delle mappe tensoriale, simmetrica e kernel;
- uguaglianza degli score e delle predizioni dei tre classificatori per prior
  uniformi ed empirici;
- sostituzione numerica delle formule di complessità;
- funzionamento del limite preventivo sulle risorse;
- positività e completezza della POVM quantistica;
- unitarietà della dilatazione di Naimark;
- identità tra probabilità circuitali e regola di Born;
- identità tra le predizioni del circuito e quelle di r-PGM;
- confusion matrix e dettaglio corretto/errato;
- frasi interpretative campione per campione e mappa bitstring-classe;
- simulazione locale a shot e distanza dalla teoria;
- tolleranza LRZ a un backend con `Target` non disponibile;
- lettura dei risultati IBM Sampler V2 e conversione delle probabilità cloud;
- limiti di gate, SVG nativo e worker di transpilation isolato quando Qiskit è
  disponibile.
