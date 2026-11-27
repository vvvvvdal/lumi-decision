#!/usr/bin/env bash
#SBATCH --job-name=lumi-anl
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Análise sem GPU da triagem H14 (analises/plano_h14_botoes.md), semente 0, contra o h9r_k100.
# Uso: sbatch slurm/analise_h14.sbatch   (depois: cat analises/h14_*.md)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
py() { echo ">> $*"; lumi_exec python "$@" || echo ">> FALHOU: $*"; }
BRACOS="h14_k300 h14_ep6 h14_lr2 h14_lr05 h14_k300_ep6 h14_bal h14_k300_bal"
for b in $BRACOS; do
  py scripts/h9_recalibrar.py --base h9r_k100 --novo $b --sementes 0 --excluir-tambem h9r_k100 --out analises/h14_receita_$b.md
  py scripts/onde_erra.py --nome $b --sementes 0 --fontes felhen --out analises/h14_onde_erra_$b.md
done
py scripts/onde_erra.py --nome h9r_k100 --sementes 0 --fontes felhen --out analises/h14_onde_erra_h9r_k100.md
py scripts/analise_h9.py --base h9r_k100 --novos "$(echo $BRACOS | tr ' ' ',')" --excluir-tambem h9r_k100 --out analises/h14_acc.md
echo ">> pronto"
