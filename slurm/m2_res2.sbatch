#!/usr/bin/env bash
#SBATCH --job-name=lumi-m2-res2
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# M2 (analises/reservadas_v2.md): checkpoints existentes no dev das reservadas v2, todas as rotações, com a
# leitura registrada de cada um (o backend escolhe pelo adaptador).
#   sbatch slurm/m2_res2.sbatch congelado e3_lora_r16 ...   ("congelado" = sem adaptador)
# Saída: runs/res2dev_<nome>_r4 (restricoes, regra, ordem: 4 rotações) e runs/res2dev_<nome>_r3 (custo_plano).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
lumi_gpu_smoke || exit 1
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --source res2 --split dev ${EVAL_EXTRA:-}"  # EVAL_EXTRA=--abster (H5)
for nome in "$@"; do
  AD=""
  if [ "$nome" != congelado ]; then
    [ -d "runs/$nome/adapter" ] || { echo ">> sem adaptador: runs/$nome/adapter"; exit 1; }
    AD="--adapter runs/$nome/adapter"
  fi
  lumi_exec lumi-eval run $BASE $AD --families restricoes_oferta,regra_beneficio,ordem_eventos \
    --orders canonical,rot1,rot2,rot3 --out "runs/res2dev_${nome}_r4" || { echo ">> $nome r4 falhou"; exit 1; }
  lumi_exec lumi-eval run $BASE $AD --families custo_plano \
    --orders canonical,rot1,rot2 --out "runs/res2dev_${nome}_r3" || { echo ">> $nome r3 falhou"; exit 1; }
done
echo ">> M2 concluída: $*"
