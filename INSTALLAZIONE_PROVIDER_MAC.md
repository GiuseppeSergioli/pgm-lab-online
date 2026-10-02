# Installazione dei provider quantistici su macOS

I provider quantistici sono opzionali: anche se la loro installazione non riesce,
il confronto PGM, il campionatore ideale locale e il disegno dei circuiti
continuano a funzionare.

## Metodo consigliato

Fai doppio clic su `Installa_Provider_Quantistici.command` e scegli:

1. LRZ Quantum;
2. Amazon AWS Braket;
3. Qiskit Aer, per la simulazione locale del circuito;
4. IBM Quantum;
5. IonQ Quantum Cloud;
6. tutti i provider.

Lo script usa l'ambiente Conda `pgm-lab`. Per Amazon AWS installa `numba` e
`llvmlite` come pacchetti binari Conda prima dei componenti Braket, evitando la
compilazione locale di LLVM che può fallire su macOS.

Se macOS blocca un file `.command` al primo avvio, fai tasto destro sul file,
scegli **Apri** e conferma una sola volta.

## Configurazione sicura delle credenziali

Fai doppio clic su `Configura_Credenziali.command`. I token sono richiesti con input
nascosto e salvati esclusivamente nel file locale `.streamlit/secrets.toml`, con
permessi restrittivi. Il file non è incluso nel pacchetto e non deve essere
condiviso. Nei successivi aggiornamenti, lascia vuoto un campo per conservare il
valore esistente oppure inserisci `-` per rimuoverlo.

Sono supportate queste chiavi:

```toml
LRZ_MQSS_TOKEN = "..."
IONQ_API_KEY = "..."
QISKIT_IBM_TOKEN = "..."
QISKIT_IBM_INSTANCE = "..."
```

Amazon Braket usa invece un profilo AWS locale, preferibilmente configurato con
IAM Identity Center/SSO; non incollare Access Key e Secret Key nell'app.

## Installazione manuale

Nel Terminale:

```bash
conda activate pgm-lab
cd /percorso/del/package
python -m pip install --prefer-binary -r requirements-simulators.txt
python -m pip install --prefer-binary -r requirements-lrz.txt
python -m pip install --prefer-binary -r requirements-ionq.txt
python -m pip install --prefer-binary -r requirements-ibm.txt
```

Per Amazon Braket:

```bash
conda activate pgm-lab
cd /percorso/del/package
conda install -n pgm-lab -c conda-forge -y numba llvmlite
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary -r requirements-aws.txt
python -m pip check
```

Se una precedente installazione mostra il conflitto
`joblib 1.6.0 requires cloudpickle>=3.0`, correggilo lasciando invariata la
versione di `cloudpickle` richiesta da Braket:

```bash
python -m pip install "joblib>=1.4,<1.6"
python -m pip check
```

## Avvio

Al termine, avvia l'app con doppio clic su `Avvia_PGM_Lab.command` oppure con:

```bash
python -m streamlit run app.py
```
