#!/bin/bash

# Avvio stabile di PGM Lab su macOS. La finestra del Terminale deve restare aperta.
project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir" || exit 1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export QISKIT_PARALLEL=FALSE
export RAYON_NUM_THREADS=1

if [ -x "$project_dir/.venv/bin/python" ]; then
  exec "$project_dir/.venv/bin/python" -X faulthandler -m streamlit run app.py
fi

for conda_executable in \
  "$HOME/miniconda3/bin/conda" \
  "$HOME/anaconda3/bin/conda" \
  "$HOME/opt/anaconda3/bin/conda" \
  "/opt/anaconda3/bin/conda" \
  "/opt/homebrew/bin/conda" \
  "/usr/local/bin/conda"
do
  if [ -x "$conda_executable" ]; then
    exec "$conda_executable" run --no-capture-output -n pgm-lab \
      python -X faulthandler -m streamlit run app.py
  fi
done

if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n pgm-lab \
    python -X faulthandler -m streamlit run app.py
fi

if command -v python3 >/dev/null 2>&1 && \
   python3 -c "import streamlit" >/dev/null 2>&1
then
  exec python3 -X faulthandler -m streamlit run app.py
fi

echo
echo "Non trovo l'ambiente pgm-lab o una Python con Streamlit installato."
echo "Apri il Terminale, attiva pgm-lab e avvia: python -m streamlit run app.py"
echo
read -r -p "Premi Invio per chiudere..."
