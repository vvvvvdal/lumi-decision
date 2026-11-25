#!/usr/bin/env bash
#SBATCH --job-name=lumi-anl
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Etapa 0 do ciclo H14–H16 (sem GPU): onde o Felhen erra depois da receita, inventário das opções (escalas e
# tamanho do pool das reservadas) e se o braço "diverso" da fase 2 rodou.
# Uso: sbatch slurm/etapa0_onde_erra.sbatch   (depois: cat analises/onde_erra.md analises/escalas_inventario.md)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
lumi_exec python scripts/onde_erra.py --nome h9r_k100 --out analises/onde_erra.md || echo ">> FALHOU: onde_erra"
lumi_exec python scripts/escalas.py --out analises/escalas_inventario.md || echo ">> FALHOU: escalas"
echo ">> fase 2, braço diverso:"; ls -d runs/f3p_* runs/f21_* 2>/dev/null || echo "(nenhum)"; ls analises/f3p* 2>/dev/null
echo ">> passos do h4_log:"; python3 -c "import json;print(json.load(open('runs/h4_log/treino_meta.json'))['passos_total'])"
echo ">> pronto"
