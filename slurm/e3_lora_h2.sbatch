#!/usr/bin/env bash
#SBATCH --job-name=lumi-e3
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x-%j.out
# E3 (H2): treina LoRA no readout das letras (só famílias vistas; teto por família; ordens sorteadas),
# avalia dev e calib das três fontes com as MESMAS ordens do E1 (canonical,reversed), calibra e compara
# com o Lumi congelado do E1 (critério da H2: média com peso igual das famílias reservadas).
# Reentrante: o treino retoma do checkpoint; a avaliação pula casos já feitos com a mesma configuração.
# Perto do limite de tempo o Slurm manda SIGUSR1: o treino salva o checkpoint e sai; ressubmeta igual.
# Uso:  sbatch slurm/e3_lora_h2.sbatch [NOME] [args extras do lumi-treino, ex.: --lr 2e-4 --r 32]
#       E3 curto (valida o fluxo): E3_N=20 sbatch slurm/e3_lora_h2.sbatch e3_curto --max-steps 40 --save-every 10
#       -> avalia 20 casos por família e compara com o E1 curto20 (mesmos casos e ordens).
#       Teste de interrupção: scancel --signal=USR1 --batch <job>; depois ressubmeta o MESMO comando.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"   # funciona submetendo de qualquer pasta
source slurm/config.sh
export HF_HUB_OFFLINE=1
NOME="${1:-e3_lora_r16}"; shift || true
OUT="runs/$NOME"
N="${E3_N:-0}"
SEL=""; SUF=completo
EVAL_EXTRA="${EVAL_EXTRA:-}"   # ex.: EVAL_EXTRA=--no-generate (E4: a pointer head não gera texto)
if [ "$N" -gt 0 ]; then SEL="--per-family $N"; SUF="curto${N}"; fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
lumi_gpu_smoke || exit 1

if [ ! -f "$OUT/CONCLUIDO" ]; then
  mkdir -p "$OUT"; rm -f "$OUT/PARAR"   # sobra de uma parada que não chegou a ser atendida
  lumi_exec_bg python -m lumi_decision.lora --out "$OUT" "$@"
  PID=$LUMI_BG_PID
  # O Apptainer não repassa o SIGUSR1 ao Python (E3 36835); o treino checa o arquivo PARAR a cada passo.
  trap 'echo ">> SIGUSR1: pedindo parada ao treino ($OUT/PARAR)"; touch "$OUT/PARAR"; kill -USR1 $PID 2>/dev/null' USR1
  wait $PID; rc=$?
  while kill -0 $PID 2>/dev/null; do wait $PID; rc=$?; done   # wait volta cedo quando o trap dispara
  trap - USR1
  if [ ! -f "$OUT/CONCLUIDO" ]; then
    echo ">> treino não concluiu (código $rc); checkpoint em $OUT/checkpoint — ressubmeta o mesmo comando"
    exit 3
  fi
fi

E1=runs/e1_lumi_27b30f4_nativa
for src in felhen telepatia traces; do
  for split in dev calib; do
    lumi_exec lumi-eval run --backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d \
      --dtype bfloat16 --native --adapter "$OUT/adapter" --source "$src" --split "$split" --orders canonical,reversed \
      $SEL $EVAL_EXTRA --out "${OUT}_${src}_${split}" || { echo ">> avaliação $src $split falhou"; exit 1; }
  done
  lumi_exec lumi-eval calibrar --calib "${OUT}_${src}_calib" --alvo "${OUT}_${src}_dev" --risco 0.10 \
    || echo ">> calibração de $src falhou"
done
if [[ "$EVAL_EXTRA" == *--abster* ]]; then
  # H5: a opção "Não sei" muda as opções apresentadas, então o critério da H2 (pareado com o E1) não se aplica.
  echo ">> H5 (--abster): pulei o lumi-eval h2; use scripts/analise_abster.py"
else
  lumi_exec lumi-eval h2 --base "${E1}_felhen_dev_${SUF},${E1}_telepatia_dev_${SUF},${E1}_traces_dev_${SUF}" \
    --novo "${OUT}_felhen_dev,${OUT}_telepatia_dev,${OUT}_traces_dev" --out "$OUT/h2.json"
fi
