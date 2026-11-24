#!/usr/bin/env bash
#SBATCH --job-name=lumi-sabe
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# "Sabe que não sabe" (scripts/analise_sabe.py): só CPU, com as predições que já existem.
# Plano e leitura registrados: analises/plano_sabe_que_nao_sabe.md.
# Por sbatch e não srun: o srun interativo caiu esperando na fila ("Connection reset by peer").
# Uso: sbatch slurm/analise_sabe.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
lumi_exec python scripts/analise_sabe.py --sementes 0,1,2 --out analises/sabe_que_nao_sabe.md \
  || { echo ">> sabe que não sabe falhou"; exit 1; }
echo ">> runs res2 que existem (para a parte 2):"
ls -d runs/res2dev_* runs/res2dif_* 2>/dev/null || echo "(nenhum)"
echo ">> pronto: analises/sabe_que_nao_sabe.md"
