#!/usr/bin/env bash
#SBATCH --job-name=lumi-h17
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# H17: um modelo de configs/modelos_h17.toml nas decisões de agente, pela leitura das letras, nas 2 ordens:
# escolher a função (bfcl), escolher a ferramenta em PT (traces) e o ENEM (metade explorar, para o "passar adiante").
# Uso:      sbatch slurm/h17_harness.sbatch MODELO            -> runs/h17_MODELO_<fonte>
# Sanidade: H17_N=5 sbatch slurm/h17_harness.sbatch MODELO    -> runs/h17s_MODELO_<fonte>_dev (5 casos por família, com geração)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
M="${1:?modelo de configs/modelos_h17.toml}"
read -r ID REV NAT TPL ADP <<< "$(python3 -c "
import tomllib;v=tomllib.load(open('configs/modelos_h17.toml','rb'))['$M']
print(v['id'], v['revisao'], int(v['nativo']), v['template'], v.get('adaptador','-'))")"
FLAGS=""; [ "$NAT" = 1 ] && FLAGS="--native"; [ "$ADP" != "-" ] && FLAGS="$FLAGS --adapter $ADP"
N="${H17_N:-0}"
for par in bfcl:dev traces:dev vestbench_enem:explorar; do
  src=${par%%:*}; split=${par#*:}
  if [ "$N" -gt 0 ]; then OUT="runs/h17s_${M}_${src}_dev"; EXTRA="--per-family $N --overwrite"; else OUT="runs/h17_${M}_${src}"; EXTRA="--no-generate"; fi
  lumi_exec lumi-eval run --backend hf --model "$ID" --revision "$REV" --dtype bfloat16 $FLAGS --template "$TPL" \
    --source "$src" --split "$split" --orders canonical,reversed $EXTRA --out "$OUT" || echo ">> FALHOU: $M $src"
done
echo ">> pronto: $M"
