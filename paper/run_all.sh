#!/usr/bin/env bash
# Rebuild every paper figure and table from the server logs.   Run from the repo root:
#   conda activate gemma && bash paper/run_all.sh
# Optional:  LOGS=logs  ANALYSIS=analysis_out/tables  RAW=1 (also re-run the raw-data analysis figures, CPU, ~30 min)
set -euo pipefail
LOGS=${LOGS:-logs}; ANALYSIS=${ANALYSIS:-analysis_out/tables}
pip install -q fonttools >/dev/null 2>&1 || true
python paper/collect.py --logs "$LOGS" --analysis "$ANALYSIS"
python paper/figures.py
python paper/tables.py
# architecture diagram (needs xelatex or pdflatex; Libertinus OTFs ship in paper/fonts)
( cd paper/arch && (xelatex -interaction=nonstopmode gemma_arch.tex >/dev/null || pdflatex -interaction=nonstopmode gemma_arch.tex >/dev/null) \
  && echo "[figure] paper/arch/gemma_arch.pdf" ) || echo "[warn] could not compile paper/arch/gemma_arch.tex"
( cd paper/preview && xelatex -interaction=nonstopmode preview.tex >/dev/null && xelatex -interaction=nonstopmode preview.tex >/dev/null \
  && echo "[preview] paper/preview/preview.pdf" ) || echo "[warn] preview not built (needs xelatex)"
if [ "${RAW:-0}" = "1" ]; then
  # raw-data figures (exploration law curves, cover example, rank law) in the paper style
  GEMMA_PAPER_STYLE=1 bash analysis/run_all.sh
fi
