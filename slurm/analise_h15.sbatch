#!/usr/bin/env bash
#SBATCH --job-name=lumi-anl
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Análise sem GPU da H15 (analises/planos_em_andamento.md, seção H15): h15_rlcd (exata) e h15_rlcda (amostrada) contra o
# h9r_k100, 3 sementes. Uso: sbatch slurm/analise_h15.sbatch   (depois: cat analises/h15_*.md)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
py() { echo ">> $*"; lumi_exec python "$@" || echo ">> FALHOU: $*"; }
for b in h9r_k100 h15_rlcd h15_rlcda; do
  py scripts/onde_erra.py --nome $b --out analises/h15_onde_erra_$b.md
done
for b in h15_rlcd h15_rlcda; do
  py scripts/h9_recalibrar.py --base h9r_k100 --novo $b --out analises/h15_receita_$b.md
  py scripts/analise_sabe.py --base h9r_k100 --novo $b --sementes 0,1,2 --out analises/h15_sabe_$b.md
  for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s")
    py scripts/analise_h9.py --base h9r_k100$n --novos $b$n --out analises/h15_acc_${b}_s$s.md
  done
done
echo ">> pronto"
