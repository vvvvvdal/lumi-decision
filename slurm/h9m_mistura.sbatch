#!/usr/bin/env bash
#SBATCH --job-name=lumi-h9m
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# H9m: mistura o adaptador de partida (A) com o ajustado (B) no peso W, sem treinar, e avalia dev e calib de
# felhen e telepatia (as fontes da análise da H9).
# Uso:  sbatch slurm/h9m_mistura.sbatch runs/h4_log runs/h9_k100 0.5 runs/h9m50_k100
# Reentrante: a avaliação pula casos já feitos com a mesma configuração.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
A="$1"; B="$2"; W="$3"; OUT="$4"
lumi_exec python scripts/misturar_adaptadores.py --a "$A" --b "$B" --w "$W" --out "$OUT" || exit 1
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
lumi_gpu_smoke || exit 1
for src in felhen telepatia; do
  for split in dev calib; do
    lumi_exec lumi-eval run --backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d \
      --dtype bfloat16 --native --adapter "$OUT/adapter" --source "$src" --split "$split" --orders canonical,reversed \
      --out "${OUT}_${src}_${split}" || { echo ">> avaliação $src $split falhou"; exit 1; }
  done
done
echo ">> H9m concluído: ${OUT}_{felhen,telepatia}_{dev,calib}"
