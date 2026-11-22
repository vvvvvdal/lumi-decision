# H4: a forma da recompensa muda a calibração do Lumi? (resultado parcial)

**Autor:** Felipe Vidal, equipe de Benchmarks e Agentes, PDC SLMs (CEIA/UFG)
**Data dos experimentos:** 18/11/2026
**Base:** `Palharess/lumi-decision` (receita A1). Código da H4 no fork `vvvvvdal/lumi-decision`, branch
`h4-recompensas`, commit `1af8b29`.
**Status:** `log` e `brier` com 3 sementes (0, 1, 2) e IC 95% por bootstrap (19/11); `acerto` com a semente 0.
Tudo medido no `dev`; o `test` oficial não foi usado.

---

## 1. Pergunta

No RLCD (*Reinforcement Learning for Calibrated Decisions*), o modelo devolve uma decisão entre opções com
uma probabilidade que precisa ser **honesta**: quando diz 70%, acerta 70%. A teoria diz que:

- recompensar só o **acerto** ensina o modelo a pôr 100% numa opção, ou seja, **superconfiança**;
- recompensar com uma **regra de pontuação própria** (log score, Brier) ensina probabilidades honestas.

A H4 testa isso no Lumi-4B: **mantendo toda a receita A1 igual, trocar só a perda muda a calibração e a
abstenção, principalmente nas famílias reservadas?**

## 2. Desenho

Tudo igual ao E3 do Palhares: mesmo readout das letras, LoRA r=16, dados (Felhen + Telepatia + Traces,
20.350 exemplos), 1 época, lr 1e-4 e ordem das opções sorteada. Só muda a perda por exemplo
(`lora.option_loss`):

| Braço | Perda | Tipo |
|---|---|---|
| `log` (controle = A1) | $-\sum_k q_k \log p_k$ | regra própria, ilimitada |
| `brier` | $\sum_k (p_k - q_k)^2$ | regra própria, limitada |
| `acerto` | $-\sum_k q_k\, p_k$ | acerto esperado (estilo RLVR), **não** própria |

$p$ é a distribuição do Lumi entre as opções e $q$ o alvo (one-hot do gabarito, ou `gold_dist` na
Telepatia).

**Avaliação** (código do Palhares, `metrics.py` / `lumi-eval calibrar`):

- temperatura $T$ e limiar de abstenção ajustados no `calib` (famílias vistas, alvo de 10% de risco) e
  aplicados fixos no `dev`;
- duas ordens das opções (original e invertida);
- ECE com 15 faixas sobre a probabilidade máxima.

**Jobs** (cluster DGX, `h100n2`): E1 completo 37565; `h4_log` 37566; `h4_brier` 37567; `h4_acerto` 37568.

## 3. Validação: o ambiente reproduz o Palhares

| Felhen vistas | Palhares | Este trabalho |
|---|---|---|
| Lumi congelado: $T$, ECE antes → depois | 1,43; 0,062 → 0,035 | 1,43; 0,062 → 0,035 |
| Lumi congelado: cobertura / risco | 16% / 4,5% | 16,5% / 4,5% |
| A1 (`log`): $T$; cobertura / risco | 1,06; 90% / 8,6% | 1,06; 89,4% / 8,4% |

O E1 e o braço `log` reproduzem os números registrados no `h1_h3_para_calibracao.md`. Por isso os outros
braços podem ser comparados com o controle.

## 4. Resultados (dev, semente 0)

Colunas:
- **ECE:** calibração antes da temperatura;
- **ECE+T:** calibração depois da temperatura ajustada no `calib`;
- **cobert. / risco:** fração de casos aceitos e erro entre eles, com o limiar de 10%;
- **aleat.:** erro ao aceitar ao acaso.

### Felhen

| Modelo | Grupo | Acc | NLL | ECE | ECE+T | $T$ | Cobert. | Risco | Aleat. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sem treino | vistas | 0,723 | 0,852 | 0,062 | 0,035 | 1,43 | 0,165 | 0,045 | 0,277 |
| sem treino | reservadas | 0,501 | 1,590 | 0,169 | 0,118 | 1,43 | 0,078 | 0,238 | 0,499 |
| log | vistas | 0,869 | 0,364 | 0,016 | 0,016 | 1,06 | 0,894 | 0,084 | 0,131 |
| log | reservadas | 0,515 | 1,282 | 0,078 | 0,068 | 1,06 | 0,383 | 0,289 | 0,485 |
| brier | vistas | 0,872 | 0,391 | 0,020 | 0,015 | 0,98 | 0,906 | 0,089 | 0,128 |
| brier | reservadas | 0,521 | 1,268 | 0,057 | 0,063 | 0,98 | 0,429 | 0,304 | 0,479 |
| **acerto** | vistas | 0,835 | **2,552** | **0,159** | 0,105 | **5,74** | 0,346 | 0,079 | 0,165 |
| **acerto** | reservadas | 0,501 | **6,481** | **0,483** | 0,185 | **5,74** | **0,000** | — | 0,499 |

### Telepatia

| Modelo | Grupo | Acc | NLL | ECE | ECE+T | $T$ | Cobert. | Risco | Aleat. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sem treino | vistas | 0,520 | 0,990 | 0,137 | 0,054 | 1,79 | 0,130 | 0,110 | 0,480 |
| sem treino | reservadas | 0,557 | 1,111 | 0,155 | 0,124 | 1,79 | 0,174 | 0,308 | 0,443 |
| log | vistas | 0,725 | 0,697 | 0,153 | 0,033 | 0,40 | 0,371 | 0,099 | 0,275 |
| log | reservadas | 0,606 | 0,898 | 0,105 | 0,141 | 0,40 | 0,393 | 0,346 | 0,394 |
| brier | vistas | 0,735 | 0,697 | 0,160 | 0,034 | 0,40 | 0,413 | 0,115 | 0,265 |
| brier | reservadas | 0,607 | 0,900 | 0,095 | 0,136 | 0,40 | 0,403 | 0,343 | 0,393 |
| **acerto** | vistas | 0,700 | **3,882** | **0,289** | 0,067 | **8,26** | 0,061 | 0,157 | 0,300 |
| **acerto** | reservadas | 0,547 | **4,622** | **0,425** | 0,131 | **8,26** | **0,000** | — | 0,453 |

Na Telepatia, as reservadas são a família `invoice_processing`.

### Com 3 sementes e intervalo de confiança (19/11)

`scripts/comparar_h4.py` (job 37646): diferença **novo − log**, juntando as sementes disponíveis. O IC 95%
reamostra grupos de casos. "Empate" = o IC inclui zero. ECE, Brier e NLL: menor é melhor.

**Brier − log (sementes 0, 1, 2)**

| Fonte | Grupo | ΔECE | ΔBrier | ΔNLL | ΔAcc |
|---|---|---|---|---|---|
| Felhen | vistas | +0,003 [−0,007; +0,011] empate | +0,005 [+0,002; +0,008] pior | +0,029 [+0,019; +0,039] pior | −0,000 empate |
| Felhen | reservadas | −0,010 [−0,021; +0,014] empate | +0,001 empate | +0,021 [+0,003; +0,038] pior | −0,010 [−0,022; +0,004] empate |
| Telepatia | vistas | +0,003 empate | +0,002 empate | +0,005 [+0,000; +0,010] pior | +0,001 empate |
| Telepatia | reservadas | **+0,025 [+0,017; +0,032] pior** | −0,002 empate | +0,004 [+0,001; +0,007] pior | **+0,022 [+0,015; +0,027] melhor** |

**Acerto − log (semente 0)**

| Fonte | Grupo | ΔECE | ΔNLL | ΔAcc |
|---|---|---|---|---|
| Felhen | vistas | +0,144 [+0,118; +0,157] pior | +2,19 pior | −0,034 [−0,048; −0,020] pior |
| Felhen | reservadas | +0,405 [+0,363; +0,426] pior | +5,20 pior | −0,014 empate |
| Telepatia | vistas | +0,137 [+0,061; +0,222] pior | +3,19 pior | −0,025 empate |
| Telepatia | reservadas | +0,320 [+0,289; +0,346] pior | +3,72 pior | −0,059 [−0,072; −0,046] pior |

## 5. Leitura

**1. A recompensa de acerto destrói a calibração. A previsão se confirmou.**
- A acurácia fica perto da do controle (Felhen vistas: 83,5% contra 86,9%).
- A NLL multiplica por 5–7, e a ECE vai a 0,16–0,29 nas vistas e a 0,43–0,48 nas reservadas.
- A temperatura que o `calib` pede é 5,74 e 8,26: a confiança precisa ser dividida por ~6 para ficar
  honesta.
- Mesmo depois da temperatura, a ECE das reservadas fica a pior entre os braços, e a abstenção para de
  funcionar: com o limiar do `calib`, a cobertura nas reservadas vai a zero.

É o comportamento previsto para uma recompensa linear em $p$: o ótimo põe toda a massa numa opção, e nada
no sinal pune a certeza exagerada.

**2. O Brier não melhora a calibração em relação ao log. A pista da semente 0 era sorte.**
- Com a semente 0, o Brier parecia ter ECE menor nas reservadas. Com 3 sementes e IC, isso não se sustenta:
  no Felhen é empate, e na Telepatia (`invoice`) o Brier fica **pior** em ECE (+0,025).
- A NLL fica um pouco pior com o Brier em todos os grupos. Faz sentido: a perda log otimiza exatamente a
  NLL.
- A única vantagem do Brier é a acurácia em `invoice` (+2,2 pontos, IC acima de zero), acompanhada de ECE
  pior. Ele acerta um pouco mais, mas é menos honesto sobre isso.
- Conclusão: entre duas regras próprias, a escolha quase não muda nada. O que destrói a calibração é usar
  uma regra **não** própria (acerto). Este é o exemplo de por que rodar sementes antes de concluir.

**3. A abstenção continua sem transferir em todos os braços.** O limiar ajustado para 10% de risco nas
vistas dá 29–35% de risco nas reservadas no log e no Brier. Trocar a regra própria não resolve o problema
aberto do lumi-decision. A Telepatia continua pedindo $T$ = 0,40, que afia as vistas mas piora a ECE de
`invoice` (0,105 → 0,141 no log), como no E3 original.

## 6. Limites

- **O acerto tem uma semente só.** A diferença é tão grande que sorte não a explicaria, mas fica registrado.
- **O IC reamostra casos**, juntando as sementes. Ele não mede separadamente a variação entre treinos.
- **As tabelas por fonte da seção 4 são da semente 0.** A temperatura e a abstenção das sementes 1 e 2 não
  foram resumidas.
- **Só `dev`.** O `test` oficial não foi tocado.
- **Mesmo `lr` (1e-4) para todas as perdas.** O gradiente do Brier é menor que o do log, então um `lr`
  próprio poderia mudar o resultado.
- **Os braços log/brier/acerto são a versão exata (sem amostragem) da recompensa.** Ainda não há RL com ação
  amostrada.

## 7. Próximos passos

1. ~~IC por bootstrap~~ e ~~réplicas de semente~~: feitos em 19/11.
2. Opcional: `lr` próprio para o Brier, antes de descartá-lo de vez.
3. **Braço (e), RL de abstenção:** "abster" como ação da política, com recompensa $+1$ / $-\lambda$ / $0$,
   treinada por REINFORCE ou GRPO. É o passo que ataca diretamente a abstenção que não transfere e o
   "falta informação" que quase nunca é escolhido.
4. Opcional: comparar com o **Clef-flash** (Cloudflare) nos mesmos casos.

## 8. Como reproduzir

No cluster, na raiz do fork, branch `h4-recompensas`:

```bash
LUMI_IMPL_FLAG=--native sbatch slurm/e1_h1_dev.sbatch lumi
for L in log brier acerto; do sbatch slurm/e3_lora_h2.sbatch h4_$L --loss $L; done
```

Os resultados ficam em `runs/h4_<perda>_<fonte>_dev/` (`relatorio.md` e `calibracao.json`).
