#!/usr/bin/env bash
#SBATCH --job-name=lumi-anl
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Análises sem GPU da H13 (analises/plano_h13_dados.md) e da H12c (analises/plano_h12c_logbrier_receita.md).
# Uso: sbatch slurm/analise_h13_h12c.sbatch   (depois: cat analises/h13_*.md analises/h12c_*.md)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
py() { echo ">> $*"; lumi_exec python "$@" || echo ">> FALHOU: $*"; }
# H13
py scripts/h13_letras.py --out analises/h13_letras.md
py scripts/comparar_h4.py --base log --novos log --split dev --split-novo devvar --out analises/h13_sensibilidade.md
py scripts/comparar_h4.py --base log --novos aug --out analises/h13_aug_dev.md
py scripts/comparar_h4.py --base log --novos aug --split devvar --out analises/h13_aug_devvar.md
# H12c
py scripts/h9_recalibrar.py --base h9r_k100 --novo h9rb_k100 --out analises/h12c_receita.md
for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s")
  py scripts/analise_h9.py --base h9r_k100$n --novos h9rb_k100$n --out analises/h12c_s$s.md
done
py scripts/analise_sabe.py --base h9r_k100 --novo h9rb_k100 --sementes 0,1,2 --out analises/h12c_sabe.md
echo ">> pronto"
