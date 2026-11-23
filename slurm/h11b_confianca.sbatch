#!/usr/bin/env bash
#SBATCH --job-name=lumi-h11b
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# H11b: confiança ESCRITA depois do raciocínio. Reaproveita os raciocínios do piloto da H11 (runs/h11p2_*),
# pergunta ao Lumi (mesmo adaptador) a probabilidade de 0 a 100 de a resposta estar certa e compara com as
# outras fontes de confiança (scripts/h11b_confianca.py). Critério: analises/plano_h11b_confianca.md.
# Uso:         sbatch slurm/h11b_confianca.sbatch [RUN=runs/h4_log]
# Teste curto: H11B_LIMITE=4 sbatch slurm/h11b_confianca.sbatch   (pastas runs/h11b_teste_*)
# Tempo curto (20 min) para o Slurm encaixar o job nos buracos da fila; roda em ~5 min. Se estourar, é só
# reenviar: a geração retoma de onde parou.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
RUN="${1:-runs/h4_log}"
LIM="${H11B_LIMITE:-0}"
NOME=h11b; EXTRA=""
if [ "$LIM" -gt 0 ]; then NOME=h11b_teste; EXTRA="--limite $LIM --overwrite"; fi
lumi_gpu_smoke || exit 1
lumi_exec python scripts/h11b_confianca.py --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d \
  --dtype bfloat16 --native --adapter "$RUN/adapter" --piloto h11p2 --base "$(basename "$RUN")" --nome "$NOME" \
  --out "analises/${NOME}.md" $EXTRA || { echo ">> H11b falhou"; exit 1; }
echo ">> H11b: analises/${NOME}.md"
