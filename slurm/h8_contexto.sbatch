#!/usr/bin/env bash
#SBATCH --job-name=lumi-h8
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# H8: exemplos com gabarito da mesma família no prompt (só famílias RESERVADAS), com o adaptador do h4_log.
# Avalia o dev das reservadas de felhen e telepatia para cada K (0 = controle no mesmo fluxo).
# Uso:  sbatch slurm/h8_contexto.sbatch [ADAPTADOR] [KS]
#       teste curto: H8_N=20 sbatch slurm/h8_contexto.sbatch runs/h4_log 3
#       teto do prompt: H8_MAXTOK (padrão 12000 tokens; o Felhen tem ~800 tokens por exemplo)
# Reentrante: a avaliação pula casos já feitos com a mesma configuração.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
RUN="${1:-runs/h4_log}"
KS="${2:-0,3,8}"
N="${H8_N:-0}"
SEL=""; SUF=""
if [ "$N" -gt 0 ]; then SEL="--per-family $N"; SUF="_curto${N}"; fi
NOME=$(basename "$RUN")
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
lumi_gpu_smoke || exit 1
for K in ${KS//,/ }; do
  for src in felhen telepatia; do
    lumi_exec lumi-eval run --backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d \
      --dtype bfloat16 --native --adapter "$RUN/adapter" --source "$src" --split dev --held-out only \
      --orders canonical,reversed --no-generate --exemplos-contexto "$K" --max-prompt-tokens "${H8_MAXTOK:-12000}" $SEL \
      --out "runs/h8_${NOME}_k${K}_${src}_dev${SUF}" || { echo ">> H8 K=$K $src falhou"; exit 1; }
  done
done
echo ">> H8 concluído: runs/h8_${NOME}_k{${KS}}_{felhen,telepatia}_dev${SUF}"
