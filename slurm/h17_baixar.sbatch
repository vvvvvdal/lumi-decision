#!/usr/bin/env bash
#SBATCH --job-name=lumi-baixar
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# H17: baixa (com internet, sem GPU) o BFCL, os modelos de configs/modelos_h17.toml nas revisões fixadas, exporta
# os chat templates e baixa o LLMRouterBench (1,3 GB) para conferir o formato antes de escrever o loader.
# Uso: sbatch slurm/h17_baixar.sbatch   (depois: tail -40 logs/lumi-baixar-*.out)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
lumi_exec lumi-dados baixar bfcl || echo ">> FALHOU: BFCL"
for m in $(python3 -c "import tomllib;print(' '.join(k for k,v in tomllib.load(open('configs/modelos_h17.toml','rb')).items() if not v['id'].startswith('ceia-br')))"); do
  read -r ID REV <<< "$(python3 -c "import tomllib;v=tomllib.load(open('configs/modelos_h17.toml','rb'))['$m'];print(v['id'],v['revisao'])")"
  echo ">> $m: $ID@$REV"
  lumi_exec python -c "from huggingface_hub import snapshot_download as s; print(s('$ID', revision='$REV'))" || echo ">> FALHOU: $m"
done
lumi_exec python scripts/exportar_template.py || echo ">> FALHOU: templates"
mkdir -p data/llmrouterbench
lumi_exec python -c "from huggingface_hub import hf_hub_download as h; print(h('NPULH/LLMRouterBench', 'bench-release.tar.gz', repo_type='dataset', revision='b95870f9e60c589e82f7972c91ffa8de9ca89f85', local_dir='data/llmrouterbench'))" \
  && (cd data/llmrouterbench && tar xzf bench-release.tar.gz && find . -maxdepth 3 | head -80 > ../../analises/llmrouterbench_conteudo.txt && f=$(find . -name '*.json*' | head -1) && echo "== $f" >> ../../analises/llmrouterbench_conteudo.txt && head -c 3000 "$f" >> ../../analises/llmrouterbench_conteudo.txt) \
  || echo ">> FALHOU: LLMRouterBench"
echo ">> pronto"
