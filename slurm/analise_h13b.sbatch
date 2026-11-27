#!/usr/bin/env bash
#SBATCH --job-name=lumi-anl
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Análises sem GPU da H13b (resultados/Experimentos.md, seção H13b (plano)): receita com o prompt variado (h9ra_k100) contra a
# receita de hoje (h9r_k100), no prompt de sempre (dev) e no nunca visto (devvar).
# Uso: sbatch slurm/analise_h13b.sbatch   (depois: cat analises/h13b_*.md)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
py() { echo ">> $*"; lumi_exec python "$@" || echo ">> FALHOU: $*"; }
N=h9r_k100,h9r_k100_s1,h9r_k100_s2,h9ra_k100,h9ra_k100_s1,h9ra_k100_s2
py scripts/h13_letras.py --nomes $N --out analises/h13b_letras.md
for sp in dev devvar; do
  py scripts/h9_recalibrar.py --base h9r_k100 --novo h9ra_k100 --split $sp --out analises/h13b_receita_$sp.md
  for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s")
    py scripts/analise_h9.py --base h9r_k100$n --novos h9ra_k100$n --split $sp --out analises/h13b_${sp}_s$s.md
  done
done
py scripts/analise_sabe.py --base h9r_k100 --novo h9ra_k100 --sementes 0,1,2 --out analises/h13b_sabe.md
echo ">> pronto"
