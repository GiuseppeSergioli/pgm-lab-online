#!/bin/bash

# Salva localmente le credenziali in Streamlit Secrets senza mostrarle a schermo.
project_dir="$(cd "$(dirname "$0")" && pwd)"
secrets_dir="$project_dir/.streamlit"
secrets_file="$secrets_dir/secrets.toml"

mkdir -p "$secrets_dir" || exit 1
chmod 700 "$secrets_dir"

echo
echo "PGM Lab - configurazione credenziali locali"
echo "I valori restano in .streamlit/secrets.toml e non vengono esportati dall'app."
echo "Lascia vuoto un campo per conservare il valore esistente; inserisci - per rimuoverlo."
echo
read -r -s -p "Token LRZ MQSS: " LRZ_VALUE
echo
read -r -s -p "IonQ API key: " IONQ_VALUE
echo
read -r -s -p "AQT access token: " AQT_VALUE
echo
read -r -s -p "IBM Quantum API key: " IBM_VALUE
echo
read -r -p "IBM Quantum instance CRN: " IBM_INSTANCE_VALUE

export LRZ_VALUE IONQ_VALUE AQT_VALUE IBM_VALUE IBM_INSTANCE_VALUE SECRETS_FILE="$secrets_file"
python3 - <<'PY'
from __future__ import annotations

import json
import os
import re
from pathlib import Path

path = Path(os.environ["SECRETS_FILE"])
keys = (
    "LRZ_MQSS_TOKEN",
    "IONQ_API_KEY",
    "AQT_TOKEN",
    "QISKIT_IBM_TOKEN",
    "QISKIT_IBM_INSTANCE",
)
submitted = {
    "LRZ_MQSS_TOKEN": os.environ.get("LRZ_VALUE", ""),
    "IONQ_API_KEY": os.environ.get("IONQ_VALUE", ""),
    "AQT_TOKEN": os.environ.get("AQT_VALUE", ""),
    "QISKIT_IBM_TOKEN": os.environ.get("IBM_VALUE", ""),
    "QISKIT_IBM_INSTANCE": os.environ.get("IBM_INSTANCE_VALUE", ""),
}

existing: dict[str, str] = {}
if path.exists():
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.+?)\s*$", raw_line)
        if not match or match.group(1) not in keys:
            continue
        try:
            value = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            existing[match.group(1)] = value

for key, value in submitted.items():
    if value == "-":
        existing.pop(key, None)
    elif value:
        existing[key] = value

lines = [
    "# Credenziali locali PGM Lab. Non condividere questo file.",
    "# Rigeneralo con Configura_Credenziali.command quando necessario.",
]
for key in keys:
    value = existing.get(key, "")
    if value:
        lines.append(f"{key} = {json.dumps(value)}")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
status=$?
unset LRZ_VALUE IONQ_VALUE AQT_VALUE IBM_VALUE IBM_INSTANCE_VALUE SECRETS_FILE

if [ "$status" -eq 0 ]; then
  chmod 600 "$secrets_file"
  echo
  echo "Credenziali aggiornate in: $secrets_file"
else
  echo
  echo "Configurazione non riuscita."
fi
echo
read -r -p "Premi Invio per chiudere..."
exit "$status"
