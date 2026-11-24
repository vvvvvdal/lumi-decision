# Plano H12: o braço Clef (entropia cruzada com label smoothing + Brier)

Registrado antes de rodar (24/11/2026).

## Por que

O Clef (Cloudflare, 2026, <https://blog.cloudflare.com/clef-decision-models/>) treina os modelos de decisão com
**entropia cruzada com label smoothing somada a uma perda Brier**. Na H4, testamos a perda log e a Brier cada
uma sozinha: elas empatam (3 sementes). Mas nunca testamos a soma, nem o label smoothing. Este braço fecha a
pergunta "fizemos o que o Clef faz?".

O post não dá fórmula, pesos nem o algoritmo de RL. O que ele descreve, comparado com o nosso projeto:

| No Clef | No nosso projeto |
|---|---|
| Lê a escolha direto das opções, sem gerar texto | Sim: a leitura das letras |
| Modelo base congelado + LoRA (posto 256) | Sim, com posto 16 |
| Cabeça de roteamento treinada junto | Testada pelo Palhares (ponteiro, H3), e foi pior |
| Entropia cruzada com label smoothing + Brier | **Este braço** |
| RLCD como objetivo secundário: crédito parcial para opções vizinhas numa escala, prêmio por saída precisa e penalidade de referência | Em parte: RL sem gabarito (H6) e "não sei" com coleira KL (H5). O crédito parcial só vale para opções em escala, e as nossas não são |
| Dados sintéticos permutando ordem de campos, prompts e esquemas | Em parte: a ordem das opções é sorteada a cada época |

## O que esperamos, e por quê

- **A soma não deve mudar o ponto de chegada.** O log e o Brier são regras próprias: as duas são mínimas quando
  a confiança é igual à chance real de acerto.
- **O label smoothing puxa a confiança para o meio.** No ótimo, os números crus saem um pouco menos confiantes.
  Num caso de teste com alvo (0,6; 0,3; 0,1), o ótimo vira (0,58; 0,30; 0,12). A temperatura da receita final
  deve corrigir isso.
- **Nossa aposta é empate com o `h4_log`.**

## Desenho

- **Perda** `log_brier` (`lora.option_loss`):
  - entropia cruzada com o alvo suavizado `q_s = (1 − ε)·q + ε/K`;
  - mais o Brier com o alvo original;
  - pesos 1 e 1. O post não dá pesos; essa escolha é nossa.
- **Label smoothing:** ε = 0,1 (`--suavizar 0.1`).
- **Todo o resto igual ao `h4_log`:** dados, `lr`, épocas, LoRA r = 16. O `slurm/e3_lora_h2.sbatch` treina,
  avalia `calib` e `dev` nas duas fontes e recalibra.
- **Sementes:** 0, 1 e 2. Os runs se chamam `h4_clef`, `h4_clef_s1` e `h4_clef_s2`, para o
  `scripts/comparar_h4.py` ler sem mudança.

## Critério

Diferença `clef − log` com IC 95% por bootstrap de grupos (`comparar_h4.py`, métricas cruas), em cada fonte
(Felhen, Telepatia) e grupo (vistas, reservadas):
- **Empate:** os ICs de ECE, Brier, NLL e acurácia passam pelo zero. A receita do Clef não acrescenta nada à
  perda log no nosso uso, e a receita final não muda.
- **Clef melhor:** o ECE ou o Brier das reservadas cai, com o IC fora do zero, sem perder acurácia. Aí vale testar
  a receita final com a perda do Clef.
- **Clef pior:** registrar o custo. Se for só o label smoothing deixando os números crus menos confiantes, a
  diferença deve sumir depois da temperatura (relatórios do `calibrar` que o sbatch roda).

## Como rodar

Teste curto:

```bash
cd /raid/$USER/lumi-decision && git pull -q && E3_N=20 sbatch slurm/e3_lora_h2.sbatch h4_clef_curto --loss log_brier --suavizar 0.1 --max-steps 40 --save-every 10
```

3 sementes:

```bash
cd /raid/$USER/lumi-decision && sbatch slurm/e3_lora_h2.sbatch h4_clef --loss log_brier --suavizar 0.1 --seed 0 && sbatch slurm/e3_lora_h2.sbatch h4_clef_s1 --loss log_brier --suavizar 0.1 --seed 1 && sbatch slurm/e3_lora_h2.sbatch h4_clef_s2 --loss log_brier --suavizar 0.1 --seed 2
```

Comparação (sem GPU):

```bash
cd /raid/$USER/lumi-decision && srun -p h100n2 --cpus-per-task=2 --mem=8G bash -c 'source slurm/config.sh && lumi_exec python scripts/comparar_h4.py --base log --novos clef --sementes 0,1,2 --out analises/h12_clef.md'
```

## Limites

- **Pesos e ε escolhidos por nós,** porque o Clef não publica os dele. Não há busca.
- **Posto 16, não 256,** e um modelo de 4B, não de 27B.
- **Só `dev`.**

## Nota (24/11/2026): resultado

Rodou com 3 sementes. Critério: **Clef pior.** A calibração crua piora em 9 de 12 células, e a temperatura
conserta só em parte: o Clef sai pouco confiante (T = 0,83 no Felhen, contra 1,06 do `log`), e nas reservadas do
Felhen a temperatura até piora (ECE 0,101 contra 0,062). O único ganho é a acurácia da Telepatia (+1 a +2 pontos).
A receita final não muda. Detalhes em `resultados/Resultados H12 - braço Clef.md`.
