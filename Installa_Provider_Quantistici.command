#!/bin/bash

# Installazione guidata dei connettori quantistici opzionali su macOS.
project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir" || exit 1

find_conda() {
  for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$HOME/opt/anaconda3/bin/conda" \
    "/opt/anaconda3/bin/conda" \
    "/opt/homebrew/bin/conda" \
    "/usr/local/bin/conda"
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  command -v conda 2>/dev/null
}

conda_executable="$(find_conda)"
if [ -z "$conda_executable" ]; then
  echo
  echo "Non trovo Conda. Segui INSTALLAZIONE_PROVIDER_MAC.md."
  echo
  read -r -p "Premi Invio per chiudere..."
  exit 1
fi

run_pip_requirements() {
  requirements_file="$1"
  "$conda_executable" run --no-capture-output -n pgm-lab \
    python -m pip install --prefer-binary -r "$requirements_file"
}

install_lrz() {
  run_pip_requirements requirements-lrz.txt
}

install_aws() {
  echo
  echo "Installazione dei binari Numba/llvmlite con Conda..."
  "$conda_executable" install -n pgm-lab -c conda-forge -y numba llvmlite || return 1
  "$conda_executable" run --no-capture-output -n pgm-lab \
    python -m pip install --upgrade pip setuptools wheel || return 1
  run_pip_requirements requirements-aws.txt || return 1
  "$conda_executable" run --no-capture-output -n pgm-lab python -m pip check
}

install_aer() {
  run_pip_requirements requirements-simulators.txt
}

install_ibm() {
  run_pip_requirements requirements-ibm.txt
}

install_ionq() {
  run_pip_requirements requirements-ionq.txt
}

install_all() {
  install_aws || return 1
  install_lrz || return 1
  install_aer || return 1
  install_ibm || return 1
  install_ionq || return 1
  "$conda_executable" run --no-capture-output -n pgm-lab python -m pip check
}

echo
echo "PGM Lab - installazione provider quantistici"
echo "1) LRZ Quantum"
echo "2) Amazon AWS Braket"
echo "3) Qiskit Aer (simulatori locali)"
echo "4) IBM Quantum"
echo "5) IonQ Quantum Cloud"
echo "6) Tutti i provider"
echo
read -r -p "Scegli da 1 a 6: " choice

case "$choice" in
  1) install_lrz ;;
  2) install_aws ;;
  3) install_aer ;;
  4) install_ibm ;;
  5) install_ionq ;;
  6) install_all ;;
  *)
    echo "Scelta non valida."
    read -r -p "Premi Invio per chiudere..."
    exit 1
    ;;
esac

status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "Installazione completata. Ora puoi aprire Avvia_PGM_Lab.command."
else
  echo "Installazione non completata. Copia l'ultima parte dell'errore e inviamela."
fi
echo
read -r -p "Premi Invio per chiudere..."
exit "$status"
