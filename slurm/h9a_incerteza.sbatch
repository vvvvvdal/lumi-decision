#!/usr/bin/env bash
#SBATCH --job-name=lumi-h9a
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# H9a: avalia o modelo de partida no TRAIN das famílias reservadas do Felhen (o pool do treino curto), para a
# seleção por dúvida (`--adapt-selecao duvida --adapt-preds ...`). Telepatia não precisa: o pool dela é o dev.
# Uso:  sbatch slurm/h9a_incerteza.sbatch runs/h4_log [N por família, padrão 400]
# Saída: runs/<nome>_felhen_trainres
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
RUN="$1"; N="${2:-400}"
NOME=$(basename "$RUN")
lumi_gpu_smoke || exit 1
lumi_exec lumi-eval run --backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d \
  --dtype bfloat16 --native --adapter "$RUN/adapter" --source felhen --split train --held-out only \
  --orders canonical,reversed --no-generate --per-family "$N" --out "runs/${NOME}_felhen_trainres" || exit 1
echo ">> H9a: runs/${NOME}_felhen_trainres"
