#!/usr/bin/env bash
#SBATCH --job-name=lumi-h11p
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# H11 piloto: o Lumi raciocina antes de decidir? Gera até N tokens no <think> com o adaptador e lê as letras
# depois do </think>; compara com a leitura de sempre (runs/<nome>_<fonte>_dev) em scripts/h11_piloto.py.
# Uso:  sbatch slurm/h11_piloto.sbatch [RUN=runs/h4_log] [N=256] [POR_FAMILIA=8]
# Teste curto: H11_LIMITE=4 sbatch slurm/h11_piloto.sbatch   (pastas runs/h11p_teste_*; o relatório mostra os
# raciocínios de exemplo)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
RUN="${1:-runs/h4_log}"; N="${2:-256}"; PF="${3:-8}"
LIM="${H11_LIMITE:-0}"
NOME=$(basename "$RUN")
PIL=h11p; EXTRA=""
if [ "$LIM" -gt 0 ]; then PIL=h11p_teste; EXTRA="--limit $LIM"; fi
lumi_gpu_smoke || exit 1
for src in felhen telepatia; do
  lumi_exec lumi-eval run --backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d \
    --dtype bfloat16 --native --adapter "$RUN/adapter" --source "$src" --split dev --orders canonical \
    --no-generate --per-family "$PF" --raciocinio "$N" $EXTRA --out "runs/${PIL}_${src}_dev" \
    || { echo ">> H11: $src falhou"; exit 1; }
done
lumi_exec python scripts/h11_piloto.py --base "$NOME" --piloto "$PIL" --out "analises/${PIL}.md"
echo ">> H11 piloto: analises/${PIL}.md"
