#!/usr/bin/env bash
#SBATCH --job-name=lumi-h13
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# H13: avalia um adaptador no dev do Felhen e da Telepatia com o prompt que NUNCA entra no treino (variante 4),
# com as mesmas ordens do E3 (canonical,reversed). Saída runs/<NOME>_<fonte>_devvar, para o comparar_h4.py --split devvar.
# Uso:  sbatch slurm/h13_variante.sbatch NOME          (ex.: h4_log, h4_aug_s1; lê runs/NOME/adapter)
#       teste curto: H13_N=20 sbatch slurm/h13_variante.sbatch h4_aug_curto
# Reentrante: a avaliação pula casos já feitos com a mesma configuração.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
NOME="${1:?uso: sbatch slurm/h13_variante.sbatch NOME}"
N="${H13_N:-0}"
SEL=""
if [ "$N" -gt 0 ]; then SEL="--per-family $N"; fi
[ -f "runs/$NOME/adapter/adapter_model.safetensors" ] || { echo ">> sem adaptador em runs/$NOME/adapter"; exit 2; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
lumi_gpu_smoke || exit 1
for src in felhen telepatia; do
  lumi_exec lumi-eval run --backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d \
    --dtype bfloat16 --native --adapter "runs/$NOME/adapter" --source "$src" --split dev --orders canonical,reversed \
    --variante 4 $SEL --out "runs/${NOME}_${src}_devvar" || { echo ">> H13 $src falhou"; exit 1; }
done
echo ">> H13 concluído: runs/${NOME}_{felhen,telepatia}_devvar"
