# Experimentos do RLCD no Lumi (H4 a H13b)

Um experimento por seção: o resultado e, logo abaixo, o plano registrado antes de rodar (clique para abrir). O resumo de tudo fica em `Resultados finais - RLCD no Lumi.md`; os planos que ainda estão rodando ficam em `analises/planos_em_andamento.md`. A data de cada plano está no histórico do git.

**Seções:** H4 · H5 · H6 · H7 · H8 · H9 · H9a · H9b · H9r e H9m · H10 · H11 e H11b · H12 · H12b · H12c · H13 · H13b · Sabe que não sabe · Onde o Felhen erra

---

## H4: a forma da recompensa muda a calibração do Lumi? (resultado parcial)

**Autor:** Felipe Vidal, equipe de Benchmarks e Agentes, PDC SLMs (CEIA/UFG)
**Data dos experimentos:** 18/11/2026
**Base:** `Palharess/lumi-decision` (receita A1). Código da H4 no fork `vvvvvdal/lumi-decision`, branch
`h4-recompensas`, commit `1af8b29`.
**Status:** `log` e `brier` com 3 sementes (0, 1, 2) e IC 95% por bootstrap (19/11); `acerto` com a semente 0.
Tudo medido no `dev`; o `test` oficial não foi usado.

---

### 1. Pergunta

No RLCD (*Reinforcement Learning for Calibrated Decisions*), o modelo devolve uma decisão entre opções com
uma probabilidade que precisa ser **honesta**: quando diz 70%, acerta 70%. A teoria diz que:

- recompensar só o **acerto** ensina o modelo a pôr 100% numa opção, ou seja, **superconfiança**;
- recompensar com uma **regra de pontuação própria** (log score, Brier) ensina probabilidades honestas.

A H4 testa isso no Lumi-4B: **mantendo toda a receita A1 igual, trocar só a perda muda a calibração e a
abstenção, principalmente nas famílias reservadas?**

### 2. Desenho

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

### 3. Validação: o ambiente reproduz o Palhares

| Felhen vistas | Palhares | Este trabalho |
|---|---|---|
| Lumi congelado: $T$, ECE antes → depois | 1,43; 0,062 → 0,035 | 1,43; 0,062 → 0,035 |
| Lumi congelado: cobertura / risco | 16% / 4,5% | 16,5% / 4,5% |
| A1 (`log`): $T$; cobertura / risco | 1,06; 90% / 8,6% | 1,06; 89,4% / 8,4% |

O E1 e o braço `log` reproduzem os números registrados no `h1_h3_para_calibracao.md`. Por isso os outros
braços podem ser comparados com o controle.

### 4. Resultados (dev, semente 0)

Colunas:
- **ECE:** calibração antes da temperatura;
- **ECE+T:** calibração depois da temperatura ajustada no `calib`;
- **cobert. / risco:** fração de casos aceitos e erro entre eles, com o limiar de 10%;
- **aleat.:** erro ao aceitar ao acaso.

#### Felhen

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

#### Telepatia

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

#### Com 3 sementes e intervalo de confiança (19/11)

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

### 5. Leitura

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

### 6. Limites

- **O acerto tem uma semente só.** A diferença é tão grande que sorte não a explicaria, mas fica registrado.
- **O IC reamostra casos**, juntando as sementes. Ele não mede separadamente a variação entre treinos.
- **As tabelas por fonte da seção 4 são da semente 0.** A temperatura e a abstenção das sementes 1 e 2 não
  foram resumidas.
- **Só `dev`.** O `test` oficial não foi tocado.
- **Mesmo `lr` (1e-4) para todas as perdas.** O gradiente do Brier é menor que o do log, então um `lr`
  próprio poderia mudar o resultado.
- **Os braços log/brier/acerto são a versão exata (sem amostragem) da recompensa.** Ainda não há RL com ação
  amostrada.

### 7. Próximos passos

1. ~~IC por bootstrap~~ e ~~réplicas de semente~~: feitos em 19/11.
2. Opcional: `lr` próprio para o Brier, antes de descartá-lo de vez.
3. **Braço (e), RL de abstenção:** "abster" como ação da política, com recompensa $+1$ / $-\lambda$ / $0$,
   treinada por REINFORCE ou GRPO. É o passo que ataca diretamente a abstenção que não transfere e o
   "falta informação" que quase nunca é escolhido.
4. Opcional: comparar com o **Clef-flash** (Cloudflare) nos mesmos casos.

### 8. Como reproduzir

No cluster, na raiz do fork, branch `h4-recompensas`:

```bash
LUMI_IMPL_FLAG=--native sbatch slurm/e1_h1_dev.sbatch lumi
for L in log brier acerto; do sbatch slurm/e3_lora_h2.sbatch h4_$L --loss $L; done
```

Os resultados ficam em `runs/h4_<perda>_<fonte>_dev/` (`relatorio.md` e `calibracao.json`).

---

## H5: abstenção aprendida por RL no Lumi (resultado, semente 0)

**Data:** 19/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h5-abster` (commits `a60c0d4`, `18943ba`)
**Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H5 (plano)`
**Status:** semente 0 só, no `dev`; o `test` não foi usado. Pela regra do plano, réplicas só se a hipótese
principal parecesse positiva, e ela não pareceu.

### O que foi testado

Braço (e): uma opção extra "Não sei / passar para um humano" e uma perda com três partes:
- **crença:** log score nas opções normais;
- **decisão:** responder ou abster, aprendida por REINFORCE com recompensa +1 / −λ / 0;
- **coleira:** KL para o Lumi sem adaptador.

Mesma receita do E3/`h4_log` (dados, LoRA, passos). Rodamos λ = 3 (abster se P(acerto) < 0,75) e λ = 9
(< 0,9).

**Jobs:** teste curto 37648; λ = 3 e λ = 9 completos + `res2`; análise 37657 (`scripts/analise_abster.py`).

### Resultado 1: a crença continuou honesta ✅

ΔECE nas opções normais contra o `h4_log`: empate ou melhor em todos os grupos (Telepatia reservadas:
−0,051 com λ = 3, −0,027 com λ = 9). Separar crença e decisão funcionou: o RL não deixou o modelo
superconfiante.

### Resultado 2: a hipótese principal não se sustentou ❌

Risco entre os respondidos, **na mesma cobertura**, contra o `h4_log` + limiar (λ = 3):

| Fonte | Grupo | Cobertura | Risco (e) | Risco base | Δ [IC 95%] |
|---|---|---:|---:|---:|---|
| Felhen | vistas | 0,865 | 0,083 | 0,078 | +0,006 [−0,003; +0,015] empate |
| Felhen | reservadas | 0,672 | 0,480 | 0,411 | **+0,069 [+0,027; +0,113] pior** |
| Telepatia | vistas | 0,089 | 0,067 | 0,013 | +0,053 [+0,012; +0,119] pior |
| Telepatia | reservadas | 0,075 | 0,124 | 0,124 | +0,000 empate |

O "não sei" aprendido escolhe quais casos responder **igual ou pior** que a confiança do `h4_log`. Na matéria
nova do Felhen, pior.

Na Telepatia o modelo abstém em ~91% dos casos. Faz sentido pela regra: lá a acurácia nas vistas (~73%) fica
abaixo do limiar de 0,75 do λ = 3, e o alvo é suave (`gold_dist`), o que baixa a recompensa de responder.
Com λ = 9, quase não responde: cobertura de 0–4% na Telepatia e 18% nas reservadas do Felhen.

### Resultado 3: o "falta informação" apareceu pela primeira vez, com custo ⚠️

`regra_beneficio` (reservadas v2): em quantos casos o modelo absteve **ou** pediu o documento, por resposta
certa.

| Resposta certa | h4_log | h5 λ = 3 | h5 λ = 9 |
|---|---:|---:|---:|
| conceder (documentos completos) | 0/332 | **7/332 (2%)** | 299/332 |
| encaminhar | 0/316 | 61/316 (19%) | 316/316 |
| negar | 7/312 | 106/312 (34%) | 190/312 |
| **pedir_documento** | 3/272 | **109/272 (40%)** | 272/272 |

- **λ = 9 abstém em quase tudo**, então seus 100% não indicam nada.
- **λ = 3 discrimina:** 40% quando falta documento contra 2% quando tudo está presente. No gerador da família
  (`res2.py`), cerca de 60% dos casos de "negar" também têm documento faltando, o que é coerente com os 34% em
  "negar".
- Os modelos anteriores do projeto chegavam no máximo a 19/272 nessa classe (M2). Este é o primeiro sinal de
  que o modelo reage à falta de informação.
- O custo: ele também abstém em casos decidíveis, e por isso não vence a base na mesma cobertura.

#### Réplica do sinal com 3 sementes (20/11, job 37673, `scripts/analise_falta_info.py`)

λ = 3, sementes 0, 1 e 2 (jobs de treino 37658 e 37660). IC 95% reamostrando grupos de casos (308 grupos),
juntando sementes e rotações.

| Resposta certa | Abstém ou pede o documento | s0 | s1 | s2 |
|---|---:|---:|---:|---:|
| conceder | 0,016 | 7/332 | 7/332 | 2/332 |
| encaminhar | 0,114 | 61/316 | 41/316 | 6/316 |
| negar | 0,378 | 106/312 | 126/312 | 122/312 |
| pedir_documento | 0,309 | 109/272 | 105/272 | 38/272 |

| Diferença | Δ | IC 95% | Leitura |
|---|---:|---|---|
| pedir_documento − conceder | +0,293 | [+0,217; +0,375] | reage mais quando falta documento |
| pedir_documento − encaminhar | +0,195 | [+0,109; +0,284] | reage mais quando falta documento |
| pedir_documento − negar | −0,069 | [−0,176; +0,043] | empate |

- **O sinal se confirma:** nas 3 sementes, o braço (e) reage muito mais quando falta documento (31%) do que
  quando dá para conceder (1,6%), com o IC bem acima de zero.
- **Ele não separa "pedir documento" de "negar":** as duas classes ficam empatadas. É coerente com o gerador,
  em que boa parte dos casos de "negar" também tem documento faltando. O modelo parece reagir a "falta
  documento", não à decisão certa a tomar.
- **Variação entre sementes:** a semente 2 reage bem menos em `pedir_documento` (38/272 contra ~107/272).
  O IC acima reamostra casos e não mede essa variação entre treinos; o efeito existe nas 3, mas o tamanho
  varia.

#### O teste limpo: é falta de documento ou outra coisa? (20/11, job 37674)

`scripts/analise_falta_info_fatores.py` usa os bastidores do gerador (`meta["dados"]`: documentos exigidos ×
anexados, valor × teto). O teste limpo é **dentro de "negar"**: a resposta certa é a mesma, e só muda se
falta documento.

| Caso | h4_log (s0) | h5 λ = 3 (s0, s1, s2) |
|---|---:|---:|
| negar COM documento faltando | 0,036 | 0,362 |
| negar SEM documento faltando | 0,000 | 0,405 |
| pedir_documento (sempre falta) | 0,011 | 0,309 |
| conceder (nada falta, valor ok) | 0,000 | 0,016 |
| encaminhar (nada falta, valor acima) | 0,000 | 0,114 |

| Contraste (h5 λ = 3) | Δ | IC 95% | Leitura |
|---|---:|---|---|
| negar: com − sem documento faltando | −0,043 | [−0,188; +0,114] | **empate** |
| valor acima − ok, sem documento faltando | +0,098 | [+0,053; +0,143] | maior |

**Isto corrige a leitura anterior.** O braço (e) **não** detecta a falta de documento: nos casos de "negar",
ele abstém igual com ou sem documento faltando. O padrão real é outro: ele abstém **quando o caso tem algum
problema** (regra violada, documento faltando ou valor acima do teto) e quase nunca quando está tudo limpo
(conceder, 1,6%). É um sinal de "caso complicado", não de "falta informação". A comparação
`pedir_documento − conceder` parecia isolar a falta de documento, mas misturava vários fatores.

### Conclusão

> Treinar a abstenção como ação, com a crença separada, **mantém o Lumi honesto**. O modelo passa a abster
> em casos "com problema" e quase nunca nos limpos, nas 3 sementes. Mas o teste limpo mostra que ele **não
> detecta a falta de documento em si**, e ele **não supera a regra simples** de abster pela confiança: na
> mesma cobertura, erra igual ou mais. A H5 principal não se confirma (semente 0). A secundária parecia se
> confirmar (31% contra 1,6%), mas o controle por fatores mostrou que era confundida.

### Limites

- Uma semente. O sinal do "falta informação" (40% contra 2%) é grande, mas não foi replicado.
- λ fixo por fonte. A Telepatia, com acurácia menor e alvo suave, precisaria de outro λ (ou de recompensa
  com alvo duro) para não colapsar em abster.
- A análise do `res2` não usa IC.

### Próximos passos possíveis

1. **Replicar só o sinal do "falta informação"** (λ = 3, sementes 1 e 2) e medir com IC a diferença entre
   `pedir_documento` e `conceder`. É o achado mais promissor.
2. λ por fonte, ou recompensa com o gabarito duro na Telepatia.
3. Combinar: confiança do `h4_log` para escolher os casos + o "não sei" aprendido só como sinal de falta de
   informação.
4. Levar ao Palhares: o achado conversa direto com o ponto cego da M2.

<details>
<summary><b>Plano registrado antes de rodar</b>: Desenho do braço (e): RL com "não sei" como ação (H5)</summary>

**Projeto:** RLCD no Lumi-4B, equipe de Benchmarks e Agentes (PDC SLMs, CEIA/UFG). Fork
`vvvvvdal/lumi-decision`.
**Antes:** a H4 (`Experimentos.md, seção H4`) mostrou que a recompensa de acerto destrói a calibração e que
log ≈ Brier.

#### 1. O problema

Em nenhum braço da H4 o "não sei" funciona na matéria nova:
- com o limiar ajustado para 10% de risco nas famílias vistas, o risco nas reservadas fica em 29–35%;
- na família `regra_beneficio` (M2 do Palhares), quase nenhum modelo escolhe "pedir o documento que falta"
  (0–19 de 272).

Hoje, abster é uma **regra colada por fora**: se a confiança fica abaixo de um limiar, o modelo abstém. Ela
depende de a probabilidade ser boa, e nas reservadas não é.

#### 2. A ideia

Fazer de "abster" uma **ação que o modelo aprende**, com aprendizado por reforço de verdade:
1. o modelo vê o caso e as opções A, B, C… e uma opção extra, **"X. Não sei / passar para um humano"**;
2. ele **sorteia** entre responder e abster;
3. recebe a recompensa: **+1** se respondeu certo, **−λ** se respondeu errado, **0** se absteve;
4. aprende com a recompensa (REINFORCE com a média do grupo como base, no estilo GRPO), sem precisar de
   gabarito para "abster".

#### 3. A pegadinha, e como o desenho escapa dela

A recompensa +1/−λ/0 é **linear**, como a de acerto. Se ela treinasse o modelo inteiro, ele voltaria a ficar
arrogante (H4). Por isso a perda tem partes separadas, que não brigam:

| Parte | O que treina | Como |
|---|---|---|
| **Crença** | as probabilidades de A, B, C (renormalizadas sem o X) | log score, a perda de sempre (honesta) |
| **Decisão** | só a chance de abster, $P(\text{abster}) = \sigma(z_X - \log\sum_k e^{z_k})$ | RL com +1 / −λ / 0; o gradiente mexe só no logit de X |
| **Coleira** | tudo | KL para o Lumi sem adaptador, peso pequeno |

Quando o modelo responde, a resposta é a opção mais provável segundo a crença. Na avaliação vale a mesma
regra: abstém se $P(\text{abster}) > 0{,}5$; senão, responde a opção mais provável entre as normais.

Pela teoria, com custo λ o ideal é abster quando a chance de acerto é menor que $\lambda/(1+\lambda)$ (0,75
com λ = 3; 0,9 com λ = 9). A pergunta da H5 é se o modelo aprende isso **melhor que o limiar**, usando pistas
que a probabilidade não mostra, como perceber que falta informação no texto.

#### 4. Hipótese H5 (registrada antes de rodar)

> Nas famílias reservadas, **na mesma cobertura**, o braço (e) tem risco entre os aceitos menor que o
> `h4_log` com limiar. O ECE nas opções normais não piora mais que 0,02.

- **Secundária:** em `regra_beneficio`, quando o gabarito é `pedir_documento`, o braço (e) escolhe abster ou
  pedir o documento em mais casos que o `h4_log`.
- **Conta só se:** o IC 95% inteiro ficar do lado bom, nas 3 sementes. Tudo no `dev`; o `test` fica intocado.

**"Na mesma cobertura"** quer dizer: se o braço (e) responde 60% dos casos, o `h4_log` também responde só os
60% em que tem mais confiança. Aí comparamos quem erra menos entre os respondidos. Sem isso, um modelo que
abstém mais sempre "ganharia".

#### 5. Como rodar (depois de implementado)

1. Teste curto: o fluxo roda e a taxa de abstenção não colapsa (abster sempre ou nunca).
2. λ = 3, semente 0, completo, com avaliação nas 3 fontes + reservadas v2 (`res2`).
3. λ = 9 (o alvo de 10% de risco do Palhares).
4. Se a H5 parecer positiva: sementes 1 e 2.

#### 6. Riscos

| Risco | O que fazer |
|---|---|
| Colapso: abster sempre ou nunca | O log de treino mostra a taxa de abstenção; reduzir α ou λ |
| O RL estraga a crença (ECE piora) | Medir ECE só nas opções normais; aumentar o peso da coleira |
| O "abster" aprendido só imita o limiar | Resultado possível; a comparação na mesma cobertura mostra isso honestamente |
| Custo de GPU | +1 forward por lote (a coleira); ~1,3× o E3 |

#### 7. Perguntas para o Palhares

1. Além da `regra_beneficio` (`pedir_documento`), há outras famílias com uma resposta do tipo "falta
   informação"?
2. Faz sentido deixar a opção "X. Não sei" **fixa no fim**, fora da inversão e do sorteio das ordens?
3. Mantenho `lr`, LoRA e número de passos iguais aos do E3, para a comparação com o `h4_log` ser justa?

</details>

---

## H6: RL só com o resultado da escolha (feedback de bandit), semente 0

**Data:** 20/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `9100746`)
**Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H6 (plano)`
**Análise:** job 37679, `scripts/comparar_h4.py`. Tudo no `dev`, semente 0, IC 95% por bootstrap.

### O que foi testado

O Lumi sorteia 8 respostas por caso e só fica sabendo se cada uma acertou. Ele nunca vê o gabarito.

| Braço | Recompensa |
|---|---|
| `bandit_acerto` | 1 se acertou (estilo RLVR) |
| `bandit_log` | 1/π(a) se acertou, cortado em 10: em média, o gradiente do log score |

Os dois são comparados com o `h4_log` (log score **com** gabarito completo). O `acerto` com gabarito completo
entra como referência.

### ECE (menor é melhor)

| Fonte | Grupo | log (gabarito) | **bandit_log** | **bandit_acerto** | acerto (gabarito) |
|---|---|---:|---:|---:|---:|
| Felhen | vistas | 0,016 | 0,063 | 0,182 | 0,159 |
| Felhen | reservadas | 0,078 | **0,180** | **0,457** | 0,483 |
| Telepatia | vistas | 0,153 | 0,164 | 0,307 | 0,289 |
| Telepatia | reservadas | 0,105 | **0,143** | **0,372** | 0,425 |

### Acurácia (maior é melhor)

| Fonte | Grupo | log (gabarito) | bandit_log | bandit_acerto |
|---|---|---:|---:|---:|
| Felhen | vistas | 0,869 | 0,864 (empate) | 0,815 (pior, −5,5) |
| Felhen | reservadas | 0,515 | 0,512 (empate) | 0,515 (empate) |
| Telepatia | vistas | 0,725 | 0,727 (empate) | 0,675 (pior, −5,0) |
| Telepatia | reservadas | 0,606 | **0,633 (melhor, +2,7)** | 0,602 (empate) |

### Leitura

**H6a confirmada:** sem gabarito, recompensar só o acerto deixa o Lumi tão superconfiante quanto com
gabarito (ECE 0,46 e 0,37 nas reservadas, perto dos 0,48 e 0,43 do `acerto` com gabarito). Ele também
perde acurácia nas vistas. A regra da H4 vale com e sem gabarito: **o que estraga a calibração é a
recompensa, não a falta de informação**.

**H6b não se confirma pelo critério registrado (≤ 0,02 acima do `h4_log`), mas o resultado é bom:**
- o `bandit_log` mantém **toda a acurácia** do treino com gabarito, e ainda ganha +2,7 pontos na Telepatia
  reservada;
- na calibração, ele recupera **73% (Felhen) e 86% (Telepatia)** da distância entre o `acerto` e o `log` nas
  reservadas. Fica bem mais honesto que o `bandit_acerto`, mas não chega no `log` com gabarito (ECE 0,180
  contra 0,078 no Felhen reservado).

**Custo de não ver o gabarito:** quase nenhum em acurácia; um pouco de calibração.

**Por que a calibração não chega lá (hipóteses):**
- o corte do peso em 10 subestima os acertos raros, em que o modelo dava menos de 10% de chance à resposta
  certa;
- o sinal tem mais variância que o supervisionado, e o número de passos foi o mesmo.

### Teste do corte do peso (20/11, commit `330f4fd`, `--bandit-clip`)

Mesmo `bandit_log`, variando só o corte do peso 1/π. ECE e acurácia contra o `log` com gabarito (semente 0):

| Corte | ECE Felhen res. | ECE Telepatia res. | ECE Felhen vistas | Acc Felhen res. | Acc Telepatia res. |
|---|---:|---:|---:|---:|---:|
| log (gabarito) | 0,078 | 0,105 | 0,016 | 0,515 | 0,606 |
| 10 | 0,180 | 0,143 | 0,063 | 0,512 | 0,633 |
| 30 | 0,174 | 0,128 | 0,064 | 0,517 | 0,636 |
| 1000 (quase sem corte) | 0,161 | 0,141 | 0,061 | 0,524 | 0,629 |

- Subir o corte melhora **pouco** a calibração no Felhen (0,180 → 0,161) e não tem efeito consistente na
  Telepatia. A diferença para o `log` continua com o IC acima de zero em todos os cortes.
- A acurácia não muda: fica igual ao `log` e um pouco acima na Telepatia reservada.
- **Conclusão:** o corte **não** é a causa principal da diferença de calibração. Os candidatos que sobram
  são a variância do sinal de bandit e o número de passos (o mesmo do supervisionado). Diferenças de 0,01 a
  0,02 entre cortes estão dentro do que uma semente pode variar.

### Teste de mais sorteios por caso (20/11, commit `357b611`, `--bandit-amostras 32`)

Mesmo `bandit_log` com corte 1000. Muda só o número de respostas sorteadas por caso: G = 8 → G = 32.
Job 37685, semente 0. Δ = novo − `log` com gabarito, com IC 95%:

| Fonte | Grupo | log (gabarito) | G = 8 | **G = 32** | Δ G=32 − log [IC 95%] |
|---|---|---:|---:|---:|---|
| Felhen | vistas | 0,016 | 0,061 | **0,017** | +0,002 [−0,012; +0,013] empate |
| Felhen | reservadas | 0,078 | 0,161 | **0,099** | +0,021 [−0,010; +0,042] empate |
| Telepatia | vistas | 0,153 | 0,159 | 0,176 | +0,024 [+0,006; +0,041] pior |
| Telepatia | reservadas | 0,105 | 0,141 | **0,085** | −0,020 [−0,034; −0,009] **melhor** |

Acurácia com G = 32: empate com o `log` em tudo, menos na Telepatia reservada (0,595 contra 0,606, −1,1 ponto).

- **A variância era o gargalo.** Com 4× mais sorteios, o `bandit_log` fica **empatado** com o log score
  com gabarito no Felhen (vistas e reservadas) e **melhor** na Telepatia reservada. Só na Telepatia vista
  fica um pouco pior (+0,024).
- Critério da H6b (≤ 0,02 acima do `log`): no limite. Felhen reservadas +0,021, com IC que inclui zero;
  Telepatia reservadas abaixo do `log`. Por uma semente, a H6b **passa na prática**.
- Custo: sortear 32 em vez de 8 não precisa de forward extra; o treino levou o mesmo tempo.
- **Leitura:** dá para treinar o Lumi **sem nunca mostrar o gabarito**, só com "acertou/errou", e chegar
  perto da calibração do treino supervisionado. Para isso, a recompensa precisa ser própria (1/π) e o sinal
  precisa de amostras suficientes.

### G = 32 com 3 sementes (20/11, job 37691)

`bandit_log_c1000_g32` com as sementes 0, 1 e 2, contra o `log` com gabarito nas mesmas 3 sementes. IC 95%
juntando as sementes:

| Fonte | Grupo | ECE log | ECE G=32 | Δ [IC 95%] | Δ acurácia |
|---|---|---:|---:|---|---|
| Felhen | vistas | 0,011 | 0,021 | +0,011 [−0,003; +0,019] empate | empate |
| Felhen | reservadas | 0,065 | 0,114 | **+0,049 [+0,029; +0,061] pior** | −1,3 pior |
| Telepatia | vistas | 0,160 | 0,176 | +0,016 [+0,006; +0,025] pior | empate |
| Telepatia | reservadas | 0,091 | 0,091 | −0,000 [−0,008; +0,007] **empate** | empate |

- **A semente 0 foi otimista.** Com as 3 sementes, o Felhen reservado volta a ficar pior que o `log`
  (+0,049). Isso passa do critério de 0,02, então **a H6b não se confirma** pelo critério registrado.
- Mesmo assim, o G = 32 empata com o `log` em duas das quatro células (Felhen vistas e Telepatia
  reservadas), e a acurácia fica igual em três delas.
- Na Telepatia reservada, o Brier e a NLL ficam até um pouco melhores que os do `log`.
- O G = 8 tem só a semente 0, então a comparação G = 8 × G = 32 continua sendo de uma semente. Nela, a
  diferença no Felhen reservado caiu de +0,082 para +0,021.

**Conclusão honesta da H6:** sem gabarito, o RL com recompensa própria (1/π) e mais sorteios chega
**perto** do supervisionado: empata em parte das medidas e fica um pouco pior na matéria nova do Felhen. Já
recompensar só o acerto (`bandit_acerto`) estraga a calibração como na H4.

### H6c: mais passos (plano registrado em 20/11, antes de rodar)

- **Runs:** `h4_log_e2` e `h4_bandit_log_c1000_g32_e2`, ambos com `--epochs 2`, semente 0.
- **Comparação justa:** bandit e2 contra log e2 (os dois com o mesmo número de passos). Também log e2 contra
  log e1, para separar o efeito de "treinar mais".
- **Ajuda:** a diferença bandit − log no Felhen reservado cai abaixo da obtida com 1 época na mesma semente
  (+0,021) e fica ≤ 0,02 também nas outras células.
- **Não ajuda:** a diferença fica parecida. Então o RL sem gabarito tem um custo real de calibração.
- Se ajudar, replicar nas sementes 1 e 2 antes de concluir.

#### Resultado H6c (20/11, semente 0)

Comparação justa, com os dois treinados 2 épocas. ECE (menor é melhor):

| Fonte | Grupo | log e2 | bandit e2 | Δ [IC 95%] | Δ com 1 época (s0) |
|---|---|---:|---:|---|---:|
| Felhen | vistas | 0,034 | 0,047 | +0,014 [+0,006; +0,022] pior | +0,002 |
| Felhen | reservadas | 0,090 | 0,135 | **+0,044 [+0,014; +0,068] pior** | +0,021 |
| Telepatia | vistas | 0,160 | 0,164 | +0,004 empate | +0,024 |
| Telepatia | reservadas | 0,051 | 0,063 | +0,012 empate | −0,020 |

Acurácia: empate em todas as células.

- **Mais passos não ajudam.** A diferença no Felhen reservado não caiu, foi de +0,021 para +0,044. Pelo
  critério registrado, vale o "não ajuda": **o RL sem gabarito tem um custo real de calibração no Felhen**,
  de cerca de 0,04–0,05 de ECE. Isso bate com as 3 sementes do G = 32 (+0,049).
- Detalhe: no Felhen reservado, o **Brier empata** (+0,008), mas a **NLL piora** (+0,13). Os erros a mais
  são poucos casos com confiança muito alta na resposta errada. O sinal de bandit quase nunca corrige uma
  resposta certa a que o modelo dava chance muito baixa, porque ela é raramente sorteada.
- Efeito lateral: treinar 2 épocas com o `log` melhora bastante a calibração da Telepatia reservada (ECE
  0,105 → 0,051), mas perde 2,5 pontos de acurácia ali.

**Fecha a H6:** com recompensa própria, o RL sem gabarito mantém a acurácia e evita a arrogância do
`bandit_acerto`. Ainda assim, paga um custo pequeno e real de calibração na matéria nova do Felhen, e nem
mais sorteios nem mais passos o removem.

### H6d: exploração eps = 0,2 (20/11, commit `7a985a3`, job 37696, semente 0)

O sorteio vem de (1 − eps)·pi + eps/K, com correção de importância. Mesmo G = 32, corte 1000, 1 época.

| Fonte | Grupo | ECE log | ECE G=32 | **ECE G=32 + explorar** | NLL G=32 → explorar |
|---|---|---:|---:|---:|---|
| Felhen | vistas | 0,016 | 0,017 | 0,015 | 0,400 → 0,379 |
| Felhen | reservadas | 0,078 | 0,099 | **0,096** | 1,303 → 1,301 |
| Telepatia | vistas | 0,153 | 0,176 | 0,174 | 0,718 → 0,707 |
| Telepatia | reservadas | 0,105 | 0,085 | 0,097 | 0,893 → 0,900 |

Acurácia: empate com o `log` em todas as células.

- **A exploração não muda nada que importe.** No Felhen reservado, a ECE vai de 0,099 para 0,096 e a NLL
  de 1,303 para 1,301. Passa no critério por milímetros, mas diferenças desse tamanho estão muito abaixo do
  que uma semente varia. Nas vistas, a NLL melhora um pouco. Na Telepatia reservada, a ECE piora um pouco.
- **Leitura:** a falta de exploração não é a causa do custo de calibração. Não vale gastar sementes extras
  nisso.

### Fechamento da H6

Foram testados corte, mais sorteios, mais passos e exploração. Só o **G = 32** mudou algo de forma clara
(na semente 0), e com 3 sementes ainda sobra um custo de cerca de 0,05 de ECE no Felhen reservado. A
conclusão estável é:
- o RL **sem gabarito**, com recompensa própria, mantém a acurácia e fica perto da calibração supervisionada;
- recompensar só o acerto deixa o Lumi arrogante.

### Limites

- O G = 8, os testes de corte e a exploração têm só a semente 0.
- Mesmo `lr` e mesmo número de passos do supervisionado.

### Próximos passos possíveis

1. ~~Variar o corte~~ (feito: efeito pequeno). ~~G = 32~~ (feito: ajuda, mas não fecha a diferença no Felhen
   reservado com 3 sementes).
2. Mais passos de treino ou G = 64, para ver se a diferença restante do Felhen cai.
3. Combinar com a H5: abstenção aprendida com feedback de bandit.

<details>
<summary><b>Plano registrado antes de rodar</b>: H6: RL só com o resultado da escolha (feedback de bandit)</summary>

Registrado antes de rodar, em 20/11/2026. Branch `h6-bandit`, a partir da `h5-abster`.

#### Pergunta

Na H4, o Lumi via o gabarito completo de cada caso (aprendizado supervisionado). Um agente de decisão em
produção não vê isso: ele escolhe uma opção e só fica sabendo **se aquela escolha deu certo**. A H6 pergunta:

> Treinando o Lumi só com o resultado da opção que ele sorteou, ele ainda aprende a decidir e fica calibrado?
> E a forma da recompensa importa como na H4?

#### Braços (mesma receita do E3/`h4_log`: dados, LoRA r=16, lr 1e-4, 1 época; só muda a perda)

| Run | Perda | O que o modelo vê |
|---|---|---|
| `h4_log` (já existe) | log score, gabarito completo | tudo (referência supervisionada) |
| `h4_acerto` (já existe) | acerto esperado, gabarito completo | tudo |
| `h4_bandit_acerto` | REINFORCE, recompensa $r(a) = 1[a = y]$ | só se a opção sorteada acertou |
| `h4_bandit_log` | REINFORCE, recompensa $r(a)/\pi(a)$, cortada em 10 | só se a opção sorteada acertou |

Os dois braços novos sorteiam G = 8 opções por caso, com a média do grupo como base (estilo GRPO). Com alvo
suave (Telepatia), $r(a) = q_a$.

O `bandit_log` é a regra própria possível sem gabarito: em valor esperado, seu gradiente é igual ao do log
score com o gabarito completo (`tests/test_h6_bandit.py` confere isso).

#### Hipóteses

- **H6a:** o `h4_bandit_acerto` fica superconfiante, como o `h4_acerto`: ECE nas reservadas maior que a do
  `h4_log`, com IC acima de zero.
- **H6b:** o `h4_bandit_log` fica calibrado: ECE nas reservadas no máximo 0,02 acima da do `h4_log`.
- **Custo da informação:** medir quanto de acurácia o `h4_bandit_log` perde em relação ao `h4_log`. É
  medição, não hipótese.

Tudo no `dev`, comparado com `scripts/comparar_h4.py` (IC 95% por bootstrap). Semente 0 primeiro; 1 e 2 só se
as hipóteses parecerem se confirmar.

#### Riscos

- **Variância alta do peso $1/\pi$** quando o modelo dá pouca probabilidade à opção certa. O corte em 10
  limita isso, mas introduz um viés pequeno.
- **Sinal mais fraco:** com o mesmo número de passos, o bandit pode aprender menos. Se a perda não cair no
  teste curto, avaliar mais passos antes de concluir.

#### H6d: exploração (registrado em 20/11, antes de rodar)

**Diagnóstico da H6c:** no Felhen reservado, o `bandit_log` empata no Brier e perde na NLL. Os erros a mais
são poucos casos com confiança alta na resposta errada. Quando pi(certa) ≈ 0, a certa quase nunca é
sorteada, e o modelo não recebe sinal para corrigir.

**Mudança:** `--bandit-explorar eps`.
- O sorteio passa a vir de mu = (1 − eps)·pi + eps/K.
- A correção pi/mu mantém o mesmo gradiente esperado. O peso fica limitado a K/eps.
- Com eps = 0, o resultado é idêntico à H6 (teste `test_exploracao_zero_e_identica_a_h6`).

**Run:** `h4_bandit_log_c1000_g32_x02` (eps = 0,2, G = 32, corte 1000, 1 época, semente 0), contra o `h4_log`.

**Leitura:**
- **Ajuda:** o ECE do Felhen reservado fica abaixo de 0,099 (G = 32 sem exploração, s0) e a NLL do Felhen
  reservado cai em relação ao G = 32.
- **Não ajuda:** valores iguais ou piores. Nesse caso, o custo de calibração não vem da falta de
  exploração.
- Só conta como confirmado com as sementes 1 e 2.

</details>

---

## H7: quantos exemplos de uma família nova bastam para recalibrar?

**Data:** 20/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `a1c859a`)
**Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H7 (plano)`
**Análise:** job 37699, `scripts/h7_poucos_rotulos.py`. Usa o `h4_log` nas sementes 0, 1 e 2, com 100
sorteios por k. Só CPU, sem treino.

### O que foi testado

Hoje, a temperatura e o limiar do "não sei" são ajustados nas famílias **vistas** e usados nas novas. Aqui,
cada família reservada ganha k exemplos com gabarito (k = 10, 25, 50), e a calibração é reajustada só para
ela. Alvo: no máximo 10% de erro entre as perguntas que o Lumi responde.

### Resultado (reservadas do `dev`; média e faixa de 95% entre sorteios)

| Fonte | Método | k | ECE | Cobertura | Risco entre respondidas |
|---|---|---:|---:|---:|---:|
| Felhen | global (hoje) | — | 0,062 | 37% | 0,29 |
| Felhen | T + limiar da família | 25 | 0,063 | 15% | 0,18 [0,00; 0,36] |
| Felhen | T + limiar da família | 50 | 0,062 | 16% | 0,14 [0,00; 0,31] |
| Felhen | oráculo (família toda) | — | 0,043 | **12%** | 0,10 |
| Telepatia | global (hoje) | — | 0,189 | 44% | 0,37 |
| Telepatia | T da família | 25 | **0,111** | 14% | 0,19 |
| Telepatia | T + limiar da família | 50 | 0,108 | 6% | 0,13 [0,02; 0,25] |
| Telepatia | oráculo (família toda) | — | 0,110 | **2%** | 0,09 |

(A tabela completa, com k = 10 e NLL, está em `analises/h7_poucos_rotulos.md` no cluster.)

### Leitura

**1. Na Telepatia, a temperatura estava errada, e 25 exemplos consertam.** O ECE cai de 0,189 para 0,111,
igual ao oráculo (0,110). A NLL cai de 1,20 para 0,93. Com 10 exemplos já chega quase lá (0,117).

**2. No Felhen, a temperatura já estava boa.** O ECE global (0,062) está perto do oráculo (0,043), e
reajustar com k exemplos não muda nada.

**3. O "não sei" errado tem outra causa: o Lumi sabe pouco da matéria nova.** Mesmo com o ajuste perfeito
(oráculo), ficar abaixo de 10% de erro exige responder só **12%** dos casos no Felhen e **2%** na
Telepatia. O limiar global erra 29–37% porque responde demais (37–44% dos casos), não porque as
probabilidades mentem.

**4. Com 50 exemplos da família, o risco fica perto do alvo** (0,13–0,14 na média), mas a faixa entre
sorteios é larga, e o Lumi passa a responder só 6–16% dos casos.

**Pelo critério registrado:** **parcial**. O ECE recupera tudo na Telepatia já com k = 25. O risco só chega
a ≤ 0,15 com k = 50. E o oráculo mostra que o limite não é de escala (temperatura), e sim de **competência**:
o modelo não separa bem os casos que acerta dos que erra na matéria nova (acurácia de cerca de 0,52 no
Felhen e 0,60 na Telepatia).

### O que isso fecha na história do projeto

- A **honestidade** do Lumi (calibração) se resolve com a recompensa certa (H4, H6) mais uns 25 exemplos
  quando o tipo de caso é novo (H7).
- O que **não** se resolve calibrando é saber a matéria. Na família nova, o Lumi honesto diz "não sei" quase
  sempre, e isso é o comportamento correto.
- O próximo ganho real viria de **acurácia nas famílias novas** (mais dados, modelo maior, raciocínio), não
  de mais calibração.

### Limites

- As famílias reservadas não têm `calib` (Felhen: o split de `train` delas fica sem uso; Telepatia: tudo vai
  para o `dev`). Por isso, os k exemplos saíram do próprio `dev` (k grupos), e a medida usou só os grupos
  restantes. O script faz isso sozinho quando a família não está no `calib`.
- O oráculo é otimista: ajusta e mede no mesmo conjunto.
- Só o `h4_log`. Os outros braços não foram testados aqui.

<details>
<summary><b>Plano registrado antes de rodar</b>: H7: quantos rótulos de uma família nova bastam para recalibrar? (registrado em 20/11, antes de rodar)</summary>

**Problema:** a temperatura T e o limiar tau, ajustados nas famílias vistas do `calib`, não transferem
para as reservadas. Lá, o risco entre os aceitos fica em 29–35% com alvo de 10% (E3/H4).

**Ideia (da literatura):** recalibrar por tarefa. Ver o Thermometer (arXiv 2403.08819) e conformal
prediction para múltipla escolha (arXiv 2305.18404).

**Desenho:** `scripts/h7_poucos_rotulos.py`, só CPU, com as predições do `h4_log` nas 3 sementes.
- Para cada família reservada, sortear k grupos rotulados (k = 10, 25, 50) do `calib` da mesma família.
  Ajustar T, ou T e tau, só para ela. Medir no `dev` da família.
- São 100 sorteios por k e semente.
- Comparações: `global` (o que se faz hoje) e `oraculo` (ajuste no próprio `dev` da família, limite
  otimista).

**Leitura registrada:**
- **Funciona:** com k = 25, `T_fam` reduz o ECE das reservadas para menos da metade do caminho entre
  `global` e `oraculo`, e `T_fam+tau_fam` deixa o risco entre os aceitos ≤ 0,15 na média.
- **Parcial:** o ECE melhora, mas o risco continua > 0,15. Nesse caso o limiar precisa de mais rótulos que
  a temperatura.
- **Não funciona:** k = 50 não se aproxima do oráculo. Nesse caso o problema não é de escala (temperatura)
  e sim de ordenação: o modelo não separa bem os casos certos dos errados na matéria nova.

</details>

---

## H8: mostrar exemplos da família nova no prompt ensina o Lumi?

**Data:** 20/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commits `7fcef9e` e `ce7ba71`)
**Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H8 (plano)`
**Rodadas:** job 37711 (avaliação, cerca de 40 min) e job 37717 (análise). Adaptador `h4_log`, semente 0,
reservadas do `dev`.

### O que foi testado

Antes da pergunta, o prompt ganha K casos já respondidos da mesma família (K = 0, 3, 8), como "exemplos
resolvidos antes da prova". Não há treino.

### Resultado

**Acurácia: não muda.**

| Fonte | K = 0 | K = 3 | K = 8 |
|---|---:|---:|---:|
| Felhen | 0,511 | 0,532 (empate) | 0,526 (empate) |
| Telepatia | 0,606 | 0,605 (empate) | 0,621 (empate) |

**Calibração: melhora, sem nenhum ajuste de temperatura.**

| Fonte | ECE K = 0 | ECE K = 3 | ECE K = 8 |
|---|---:|---:|---:|
| Felhen | 0,088 | 0,055 (empate, IC até +0,008) | 0,050 (empate, IC até +0,008) |
| Telepatia | 0,104 | **0,038 (melhor)** | **0,029 (melhor)** |

Na Telepatia, o Brier e a NLL também melhoram com o IC inteiro abaixo de zero.

**Cobertura com risco de 10%** (T e limiar por família com 25 rótulos):
- Felhen: 16% em todos os K.
- Telepatia: 7,5% → 14% (K = 3) → 12% (K = 8).
- O oráculo oscila muito (Telepatia 2% → 9% → 2%). Com tão poucos casos respondidos, essa medida é instável.

### Leitura

- **Pelo critério registrado: não ajuda.** A acurácia empata nas duas fontes. Mostrar exemplos **não ensina**
  a matéria nova ao Lumi-4B.
- **Efeito inesperado:** os exemplos **calibram**. Na Telepatia, o ECE cai de 0,104 para 0,029 sem
  nenhuma temperatura, mais que os 25 rótulos da H7 conseguiram (0,111). O Lumi vê como são as respostas
  daquela família e ajusta a confiança, mas não passa a acertar mais.
- No Felhen, os textos são longos (cerca de 1100 tokens por caso). Com K = 8, cabem em média 6,2 exemplos no
  teto de 12000 tokens.

### Onde isso deixa o projeto

| Ferramenta | Calibra na família nova? | Ensina (acurácia)? |
|---|---|---|
| Recompensa própria (H4/H6) | sim, nas vistas | não muda |
| 25 rótulos para a temperatura (H7) | sim (Telepatia) | não |
| Exemplos no prompt (H8) | **sim (Telepatia, sem ajuste)** | **não** |

O que falta é competência na matéria nova, e nada que só muda a **leitura** resolve isso. Pelo plano, o
próximo passo é **treinar**: uma LoRA curta com os exemplos da família, partindo do `h4_log`
(`--init-adapter` já existe).

### Limites

- Semente 0, um adaptador (`h4_log`).
- Felhen com K = 8 cortado pelo teto (6,2 exemplos em média).
- A cobertura no risco de 10% depende de poucos casos muito confiantes e oscila.

<details>
<summary><b>Plano registrado antes de rodar</b>: H8: exemplos da família nova no prompt (registrado em 20/11, antes de rodar)</summary>

**Motivo (H7):** nas famílias reservadas, recalibrar com cerca de 25 rótulos acerta a régua, mas o gargalo é
competência. Com risco ≤ 10%, o oráculo só responde 12% (Felhen) e 2% (Telepatia) dos casos.

**Pergunta:** mostrar K casos com gabarito da mesma família no prompt (aprendizado em contexto, sem treino)
aumenta a acurácia e a cobertura segura nas reservadas?

**Desenho:**
- Adaptador `runs/h4_log` (semente 0). Comando: `lumi-eval run --held-out only --exemplos-contexto K`.
- K = 0 (controle no mesmo fluxo), 3 e 8. Ordens canonical e reversed, sem geração de texto.
- **Origem dos exemplos:**
  - Felhen: o `train` das famílias reservadas, nunca usado em treino.
  - Telepatia: as reservadas só existem no `dev`, então os exemplos vêm de **outros grupos** do `dev` (nunca
    o mesmo estado do caso avaliado).
  - Um exemplo por grupo, em ordem fixa por caso.
- Teto de 6000 tokens no prompt. Se passar, os últimos exemplos saem, e o K usado fica registrado.
- Análise: `scripts/analise_h8.py`.
  - K contra K = 0: acurácia, ECE, Brier e NLL, com IC por bootstrap de grupos.
  - Cobertura no risco de 10%, com T e limiar por família ajustados em 25 grupos (como na H7), e o oráculo.

**Leitura registrada:**
- **Ajuda:** com K = 8, a acurácia nas reservadas sobe com o IC acima de zero em pelo menos uma fonte, e a
  cobertura do oráculo no risco de 10% sobe em relação a K = 0.
- **Não ajuda:** acurácia em empate nas duas fontes. Nesse caso, ensinar a família nova exige treino (LoRA
  curta com os exemplos, via `--init-adapter`).
- Se o K usado médio ficar bem abaixo do pedido (prompts longos), registrar isso como limite.

</details>

---

## H9: treinar um pouco com exemplos da família nova ensina o Lumi?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `a93f842`)
**Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H9 (plano)`
**Rodadas:** jobs 37727 (k = 25) e 37728 (k = 100); análise no job 37738 (`scripts/analise_h9.py`). Semente 0.
Depois, k = 100 nas sementes 1 e 2 (análises nos jobs 37750 e 37751).

### O que foi testado

Partindo do Lumi já treinado (`h4_log`), mais 3 passadas de treino **só com k grupos rotulados** das três
famílias reservadas:
- Felhen: área jurídica (`juristcu_area`) e veracidade (`factck_veracidade`);
- Telepatia: faturas (`invoice_processing`).

A perda é a log (honesta). Os grupos usados no treino saem da avaliação nos dois lados. A Telepatia perde
25 ou 100 grupos do `dev`; o Felhen não perde nenhum, porque os exemplos vieram do `train`.

### Resultado

**Acurácia nas famílias novas: sobe muito.**

| Fonte | `h4_log` | k = 25 | k = 100 |
|---|---:|---:|---:|
| Felhen | 0,515 | **0,567** (+5,2 [+0,2; +10,2]) | **0,650** (+13,6 [+8,6; +18,8]) |
| Telepatia | 0,606 / 0,616 | **0,712** (+10,6 [+8,3; +12,8]) | **0,781** (+16,6 [+13,3; +19,7]) |

**As famílias vistas não pioram.**

| Fonte | `h4_log` | k = 25 | k = 100 |
|---|---:|---:|---:|
| Felhen | 0,869 | 0,870 (empate) | 0,869 (empate) |
| Telepatia | 0,725 | 0,707 (empate) | 0,706 (empate) |

O Brier e a NLL das vistas pioram um pouquinho (+0,004 a +0,014).

**Calibração nas famílias novas (ECE):**

| Fonte | `h4_log` | k = 25 | k = 100 |
|---|---:|---:|---:|
| Felhen | 0,078 | 0,052 (empate) | 0,122 (empate, IC até +0,083) |
| Telepatia | 0,110 | **0,063** (melhor) | **0,071** (melhor) |

O Brier e a NLL das reservadas melhoram em todos os casos.

**Quantos casos o Lumi consegue responder errando no máximo 10%** (oráculo, T e limiar por família):

| Fonte | `h4_log` | k = 25 | k = 100 |
|---|---:|---:|---:|
| Felhen | 14% | 24% | 25% |
| Telepatia | 2,5% | **41%** | **71%** |

### Leitura

**Pelo critério registrado: ensina.** Com k = 100, a acurácia das reservadas sobe com o IC acima de zero
**nas duas fontes**, e as vistas ficam em empate. Com k = 25 também sobe nas duas.

- É o **primeiro resultado do projeto que aumenta a acurácia na matéria nova**. Ler o Lumi de outro jeito
  (H7, H8) só deixava ele mais honesto. Treinar com poucos exemplos ensina.
- O efeito aparece onde importa. Na Telepatia, o Lumi passa de responder com segurança **2,5%** dos casos
  para **71%**. Ele continua honesto (ECE melhor) e agora sabe mais.
- **Atenção no Felhen com k = 100:** o ECE sobe de 0,078 para 0,122. O IC inclui zero, mas o valor passa do
  limite de 0,02 registrado. A acurácia sobe tanto que o Brier e a NLL melhoram. Mesmo assim, vale
  recalibrar a temperatura (H7) depois do treino curto.

### Confirmação com 3 sementes (k = 100)

Sementes 1 e 2: cada uma parte do `h4_log` da mesma semente e sorteia outros 100 grupos (jobs 37750 e
37751, `analises/h9_adapt_s1.md` e `h9_adapt_s2.md` no cluster).

**Acurácia nas reservadas:**

| Fonte | Semente 0 | Semente 1 | Semente 2 | Média |
|---|---:|---:|---:|---:|
| Felhen | 0,515 → 0,650 (+13,6) | 0,536 → 0,657 (+12,1 [+7,6; +16,5]) | 0,536 → 0,627 (+9,1 [+4,9; +13,1]) | 0,529 → **0,645** |
| Telepatia | 0,616 → 0,781 (+16,6) | 0,535 → 0,784 (+24,9 [+21,2; +29,0]) | 0,600 → 0,794 (+19,5 [+15,9; +23,4]) | 0,584 → **0,786** |

O IC fica acima de zero nas 6 combinações.

**Cobertura do oráculo com risco de 10%:**

| Fonte | Semente 0 | Semente 1 | Semente 2 | Média |
|---|---:|---:|---:|---:|
| Felhen | 14% → 25% | 13% → 34% | 11% → 30% | 12% → **30%** |
| Telepatia | 2,5% → 71% | 1,8% → 66% | 1,3% → 69% | 2% → **69%** |

**ECE nas reservadas:**
- Telepatia: melhora nas 3 sementes, de 0,114 para 0,074 na média, com o IC abaixo de zero.
- Felhen: piora nas 3 sementes, de 0,072 para 0,112 na média (+0,03 a +0,05). Em cada semente o IC encosta
  no zero. Como a direção se repete, o mais provável é que seja real: o Lumi fica **confiante demais** no
  Felhen novo.

**Famílias vistas (esquecimento):**
- Felhen: a acurácia empata nas 3 sementes (−0,1 a −0,3 ponto). O Brier e a NLL pioram um pouco, de +0,003 a
  +0,015.
- Telepatia: a acurácia cai de 1,5 a 3,0 pontos. Na semente 2, o IC fica todo abaixo de zero
  (−3,0 [−5,7; −0,4]). São só 84 grupos, mas a direção se repete. O ECE melhora nas 3 sementes.

**Leitura das 3 sementes:** o "ensina" se confirma. Ficam dois custos pequenos e consistentes:
1. o Felhen novo sai confiante demais, então é preciso recalibrar a temperatura depois do treino;
2. há um esquecimento leve nas vistas da Telepatia, que passa do limite de 1 ponto registrado no plano.

### A história do projeto até aqui

1. **Recompensa certa → Lumi honesto** (H4; H6 mostra que funciona até sem gabarito).
2. **Na matéria nova, ele é honesto mas sabe pouco** (H7): para errar no máximo 10%, responde quase nada.
3. **Ler diferente não ensina** (H7, H8); só calibra.
4. **Treinar um pouco com 25–100 exemplos da matéria nova ensina** (H9), sem esquecer a antiga, e a
   honestidade se mantém na Telepatia.

### Limites

- k = 100 confirmado em 3 sementes. k = 25 só tem a semente 0.
- Uma LoRA para as 3 famílias juntas. Os números por família estão em `analises/h9_adapt.md` no cluster.
- O oráculo é otimista: ajusta e mede na família inteira.
- Na Telepatia, com k = 100, a avaliação fica com 200 grupos, em vez de 275 com k = 25 e 300 originais.

<details>
<summary><b>Plano registrado antes de rodar</b>: H9: treino curto com k grupos da família nova (registrado em 20/11, antes de rodar)</summary>

**Motivo:** H7 e H8 calibram a matéria nova, mas não aumentam a acurácia. O gargalo é competência.

**Pergunta:** continuar a LoRA do `h4_log` por poucos passos, só com k grupos rotulados das famílias
reservadas, ensina o Lumi sem ele esquecer as famílias vistas?

**Desenho:**
- Famílias: `felhen:juristcu_area`, `felhen:factck_veracidade` e `telepatia:invoice_processing`, numa LoRA só.
- k = 25 e k = 100 grupos por família. Perda `log`, mesmo `lr`, 3 épocas, semente 0.
- Ponto de partida: `--init-adapter runs/h4_log/adapter`.
- **Origem dos exemplos:**
  - Felhen: o split `train` das reservadas, nunca usado.
  - Telepatia: o `dev`, porque é o único split que existe. Esses grupos saem da avaliação nos dois braços
    (`adapt_grupos.json`).
- Comparação com o `h4_log` nos mesmos casos restantes do `dev`: `scripts/analise_h9.py`.

**Leitura registrada:**
- **Ensina:** com k = 100, a acurácia nas reservadas sobe com o IC acima de zero em pelo menos uma fonte, e as
  vistas não caem mais que 1 ponto (IC).
- **Ensina pouco:** a acurácia sobe só numa família, ou só com k = 100.
- **Não ensina:** empate. Nesse caso, a competência na matéria nova exige mais dados ou um modelo maior que o
  Lumi-4B.
- Calibração: o ECE das reservadas não deve piorar mais que 0,02.

</details>

---

## H9a: escolher os 25 exemplos onde o Lumi está em dúvida ensina mais que sortear?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `895ef22`)
**Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H9a (plano)`
**Rodadas (semente 0):**
- job 37767: avaliação do pool do Felhen (`runs/h4_log_felhen_trainres`, 400 casos por família);
- job 37768: `h9r_k25`, 25 grupos sorteados + replay 1:1;
- job 37769: `h9ar_k25`, 25 grupos escolhidos por dúvida + replay 1:1;
- análise: job 37778, com os grupos de treino dos **dois** braços fora da avaliação.

### Os dois braços contra o `h4_log`

| Recorte | Métrica | `h4_log` | Sorteio (`h9r_k25`) | Dúvida (`h9ar_k25`) |
|---|---|---:|---:|---:|
| Felhen reservadas | acc | 0,515 | 0,580 (**+6,5**) | 0,598 (**+8,3**) |
| Telepatia reservadas | acc | 0,613 | 0,738 (**+12,6**) | 0,719 (**+10,7**) |
| Felhen vistas | acc | 0,869 | 0,863 (−0,7) | 0,867 (−0,2) |
| Telepatia vistas | acc | 0,725 | 0,732 (+0,7) | 0,717 (−0,8) |
| Felhen vistas | ECE | 0,016 | **0,056** | **0,054** |
| Cobertura do oráculo, Felhen | | 14% | 25% | 22% |
| Cobertura do oráculo, Telepatia | | 3% | 47% | 46% |

Em negrito, diferenças com o IC 95% todo de um lado.

### Comparação direta: dúvida contra sorteio

| Recorte | Métrica | Δ (dúvida − sorteio) | IC 95% | Leitura |
|---|---|---:|---|---|
| Felhen reservadas | acc | +1,9 | [−0,7; +4,5] | empate |
| Felhen reservadas | NLL | −0,148 | [−0,216; −0,084] | dúvida melhor |
| Felhen reservadas | Brier | −0,033 | [−0,057; −0,007] | dúvida melhor |
| Telepatia reservadas | acc | −1,9 | [−4,2; +0,3] | empate |
| Telepatia reservadas | NLL | +0,044 | [+0,007; +0,079] | dúvida pior |
| Telepatia vistas | acc | −1,5 | [−3,6; +0,7] | empate |

### Receita completa com 25 rótulos (T e limiar da família), alvo de 10% de erro

| Fonte | Sorteio: responde / erra | Dúvida: responde / erra | Replay k = 100 (média de 3 sementes) |
|---|---|---|---|
| Felhen | 29% / 0,16 | 29% / 0,16 | 37% / 0,15 |
| Telepatia | 50% / 0,11 | 47% / 0,11 | 72% / 0,10 |

Nas famílias vistas, a temperatura global volta a resolver a superconfiança do replay no Felhen: o ECE cai de
0,056 para 0,019 no sorteio e de 0,054 para 0,018 na dúvida, com T = 1,52 nos dois.

### Leitura

**Pelo critério registrado, é empate: escolher por dúvida não compensa.**
- A acurácia nas reservadas não muda de forma clara em nenhuma fonte. Fica +1,9 no Felhen e −1,9 na Telepatia,
  com os dois ICs passando pelo zero.
- O único sinal a favor da dúvida está nas probabilidades do Felhen: a NLL cai 0,15, com o IC abaixo de zero.
  Na Telepatia, a NLL vai no sentido contrário. Como é uma semente só e as fontes discordam, não dá para dizer
  que a dúvida ajuda.
- Na receita completa, os dois braços respondem praticamente o mesmo tanto.
- Uma explicação possível: a faixa de dúvida mistura casos úteis com casos ambíguos ou de rótulo ruim, e os dois
  efeitos se cancelam. Além disso, ela exige avaliar o pool antes de treinar.
- Como o plano mandava, **não rodei as sementes 1 e 2**.

**Pergunta secundária: quanto sobra com k = 25 e replay?**
- O sorteio com replay e k = 25 ganha +6,5 pontos no Felhen e +12,6 na Telepatia. Isso é parecido com a H9 sem
  replay e k = 25 (+5,2 e +10,6), e o replay continua evitando o esquecimento (Telepatia vistas +0,7).
- Com k = 100 e replay, o ganho na mesma semente foi +13,7 e +17,7. Ou seja, 25 exemplos dão cerca de metade do
  ganho no Felhen e cerca de 70% na Telepatia.
- Na receita completa, a diferença é grande na Telepatia: com 100 exemplos, o Lumi responde 72% das faturas
  novas com 10% de erro; com 25, responde 50%.

### Limites

- Uma semente só.
- Na Telepatia, o pool da dúvida é o próprio `dev`, então a seleção tira da avaliação os casos difíceis. Como
  esses grupos saem dos dois braços, a comparação continua pareada, mas o `dev` restante fica mais fácil.
- O pool do Felhen foi limitado a 400 casos por família, e a faixa foi fixada em 3 × k. Não testei outras faixas.
- Na Telepatia, a temperatura das vistas (T = 0,52) piora as reservadas (ECE 0,045 → 0,116). Com 25 rótulos, a
  temperatura da família corrige isso.

### Conclusão

A receita final não muda: **sortear os exemplos, usar replay e recalibrar**. Escolher por dúvida não traz ganho
claro e tem um custo extra. Quantos exemplos usar continua sendo o que mais pesa: 100 exemplos ensinam bem mais
que 25.

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H9a: escolher os exemplos onde o Lumi está em dúvida (k = 25, com replay)</summary>

Registrado antes de rodar (21/11/2026).

#### Pergunta

Com só 25 grupos por família nova, escolher os grupos em que o `h4_log` está **em dúvida** ensina mais que
sortear? E quanto do ganho de k = 100 sobra com k = 25 e replay?

#### Desenho

- **Dois braços**, os dois com replay 1:1 e 3 épocas, a partir do `h4_log`:
  - `h9r_k25`: os 25 grupos são sorteados, como na H9;
  - `h9ar_k25`: a seleção é por dúvida (`--adapt-selecao duvida`). O script ordena os grupos do pool pela
    confiança média do `h4_log` (a probabilidade máxima nas duas ordens), pega a faixa dos 75 de menor
    confiança (`--adapt-faixa 3`) e sorteia 25 dentro dela. A faixa evita pegar só os casos mais duvidosos,
    que tendem a ser ambíguos ou ter rótulo ruim. A ideia vem do filtro de dificuldade do post "harness
    adapters" da Hugging Face.
- **De onde vêm as confianças do pool:**
  - Felhen: o `train` das reservadas, avaliado pelo `slurm/h9a_incerteza.sbatch` com 400 casos por família
    (`runs/h4_log_felhen_trainres`). O pool da seleção por dúvida fica limitado a esses casos;
  - Telepatia: o pool é o `dev`, e as predições `runs/h4_log_telepatia_dev` já existem.
- **Avaliação justa:** os dois braços treinam em grupos diferentes, então a análise tira os grupos de treino
  dos DOIS (`--excluir-tambem`). Na Telepatia, a seleção por dúvida tira do `dev` os casos difíceis, mas como
  eles saem dos dois braços, a comparação continua pareada.
- **Semente 0 primeiro.** Se a diferença for promissora, rodar as sementes 1 e 2 (cada uma com o `h4_log` da
  mesma semente e a sua própria avaliação do pool do Felhen).

#### Leitura

- **Dúvida ajuda:** o `h9ar_k25` tem acurácia nas reservadas acima do `h9r_k25` em pelo menos uma fonte (IC
  acima de zero), sem perder na outra.
- **Empate:** escolher por dúvida não compensa o custo (uma avaliação extra do pool).
- **Dúvida atrapalha:** o `h9ar_k25` fica abaixo. Isso indica que os casos duvidosos são ambíguos ou ruidosos.
- **Pergunta secundária:** o ganho do `h9r_k25` contra o `h4_log`, comparado com o da H9 com k = 25 (+5,2 no
  Felhen e +10,6 na Telepatia, na semente 0, sem replay) e com o do replay com k = 100.
- Calibração: as duas seleções passam pela recalibração da H9b (`scripts/h9_recalibrar.py`).

</details>

---

## H9b: recalibrar depois do treino curto devolve a honestidade?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `5807ba6`)
**Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H9b (plano)`
**Análise:** job 37752, `scripts/h9_recalibrar.py`. Usa o `h4_log` e o `h9_k100`, sementes 0, 1 e 2, com 100
sorteios por k. Só CPU.

### O que foi testado

A H9 ensinou, mas deixou o Felhen novo superconfiante. Aqui aplicamos a receita da H7 aos dois braços: k grupos
rotulados da família ajustam a temperatura T e o limiar do "não sei". Os grupos do treino curto saem da conta.
Os grupos de calibração são sorteados do resto e também ficam fora da medida.

### Resultado (reservadas do `dev`, média das 3 sementes e dos sorteios)

**Calibração (ECE):**

| Fonte | Braço | Sem recalibrar | T global (vistas) | T da família, k = 25 |
|---|---|---:|---:|---:|
| Felhen | `h4_log` | 0,073 | 0,062 | 0,064 |
| Felhen | `h9_k100` | 0,113 | **0,059** | **0,069** |
| Telepatia | `h4_log` | 0,114 | 0,190 | 0,114 |
| Telepatia | `h9_k100` | 0,074 | **0,044** | **0,045** |

**Receita completa (T e limiar da família com 25 rótulos), alvo de 10% de erro:**

| Fonte | Braço | Acurácia | Responde | Erro entre as respondidas |
|---|---|---:|---:|---:|
| Felhen | `h4_log` | 0,529 | 16% | 0,19 |
| Felhen | `h9_k100` | 0,645 | **36%** | 0,15 |
| Telepatia | `h4_log` | 0,583 | 7% | 0,18 |
| Telepatia | `h9_k100` | 0,787 | **68%** | **0,10** |

Com k = 50, o Felhen da H9 responde 32% e erra 0,12.

### Leitura

**Pelo critério registrado: conserta.**
- O ECE do Felhen na H9 com 25 rótulos (0,069) fica a 0,005 do `h4_log` (0,064), dentro da margem de 0,02.
- A Telepatia melhora (0,045 contra 0,114).
- A superconfiança do treino curto era só de escala, e uma temperatura resolve.

**Surpresa:** depois da H9, a temperatura das famílias **vistas** já serve nas novas (Felhen 0,059, Telepatia
0,044). No `h4_log`, essa mesma temperatura estragava a Telepatia (0,190). Os 25 rótulos extras servem
principalmente para o **limiar** do "não sei".

**O número que resume o projeto:** na Telepatia, com 100 exemplos para treinar e 25 para calibrar, o Lumi
responde **68%** das faturas novas e erra **10%**, exatamente o alvo. Antes ele respondia 7% e errava 18%.
- No Felhen, ele responde 36% e erra 15%. Com k = 50, erra 12%.
- O Felhen continua difícil (acurácia de 0,65), mas o Lumi responde o dobro de casos e erra menos.

### A receita completa

1. **Recompensa honesta** (log, H4): o Lumi fica calibrado nas famílias que conhece.
2. **Treino curto com cerca de 100 exemplos da família nova** (H9): ele passa a saber a matéria.
3. **Recalibração com cerca de 25 rótulos** (H7/H9b): ele volta a ser honesto e sabe quando dizer "não sei".

### Limites

- A faixa entre sorteios é larga no Felhen: com k = 25, a cobertura vai de 6% a 66% e o erro de 0,05 a 0,27.
  Com 25 rótulos, o limiar ainda é instável. k = 50 estreita a faixa.
- Ainda há um esquecimento leve nas vistas da Telepatia (H9), que é o próximo teste (replay).

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H9b: recalibrar depois do treino curto</summary>

Registrado antes de rodar (21/11/2026).

#### Pergunta

A H9 (k = 100) aumenta a acurácia nas reservadas, mas deixa o Felhen superconfiante: o ECE sobe de 0,072
para 0,112 na média das 3 sementes. A receita da H7 conserta isso? A receita é ajustar a temperatura T e
o limiar tau com k grupos rotulados da família.

#### Desenho

- Script `scripts/h9_recalibrar.py`, só CPU, com as predições que já existem.
- Braços: `h4_log` e `h9_k100`, sementes 0, 1 e 2.
- Os grupos do treino curto saem do `dev` dos dois braços.
- Os k grupos de calibração (k = 10, 25, 50; 100 sorteios) vêm do `dev` restante e ficam fora da medida.
- Os dois braços sorteiam os mesmos grupos (mesmo gerador).
- Métodos (os da H7):
  - sem recalibrar;
  - global (vistas);
  - `T_fam`;
  - `T_fam+tau_fam`;
  - oráculo.

#### Leitura (k = 25)

- **Conserta:** com `T_fam`, o ECE do Felhen reservado da H9 fica a no máximo 0,02 do ECE do `h4_log`
  com `T_fam`. A Telepatia não piora mais que 0,02.
- **Receita completa vale a pena:** com `T_fam+tau_fam`, a H9 responde mais casos que o `h4_log` com risco
  médio ≤ 0,15. Esta é a cobertura realista, com 25 rótulos e não o oráculo.
- **Não conserta:** o ECE do Felhen na H9 com `T_fam` continua mais de 0,02 acima do `h4_log` com
  `T_fam`. Nesse caso, a superconfiança não é só de escala.

</details>

---

## H9r e H9m: como evitar que o treino curto faça o Lumi esquecer?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commits `42fa52c` replay, `c283255` mistura)
**Planos registrados antes de rodar:** `resultados/Experimentos.md, seção H9r e H9m (plano)` e `resultados/Experimentos.md, seção H9r e H9m (plano)`
**Rodadas:**
- replay: jobs 37753–37755, sementes 0, 1 e 2;
- mistura: jobs 37756–37757, semente 0;
- análise: job 37763 e seguintes (`scripts/analise_h9.py`).

### O problema

Na H9, a acurácia nas **vistas da Telepatia** caía de 1,5 a 3,0 pontos (−2,1 na média). Duas correções:
- **Replay (H9r):** repete o treino curto, mas junta um caso já visto para cada caso novo, em rodízio entre
  as famílias vistas.
- **Mistura (H9m):** não treina de novo. O adaptador fica entre o `h4_log` e a H9 com peso w:
  ΔW = (1 − w)·ΔW_h4 + w·ΔW_h9. A conta é exata e foi conferida no cluster (erro de cerca de 1e-7).

### Replay (3 sementes), diferença para o `h4_log` da mesma semente

| Recorte | Métrica | H9 (s0 / s1 / s2) | Média H9 | H9r (s0 / s1 / s2) | Média H9r |
|---|---|---|---:|---|---:|
| Telepatia vistas | acc | −1,9 / −1,5 / **−3,0** | −2,1 | −0,5 / −0,5 / −1,5 | **−0,8** |
| Felhen reservadas | acc | +13,6 / +12,1 / +9,1 | +11,6 | +13,7 / +12,3 / +10,4 | **+12,1** |
| Telepatia reservadas | acc | +16,6 / +24,9 / +19,5 | +20,3 | +17,7 / +24,4 / +20,6 | **+20,9** |
| Felhen vistas | acc | 0,0 / −0,3 / −0,1 | −0,1 | −0,1 / **−1,5** / −0,7 | −0,8 |
| Felhen vistas | ECE | +0,012 / +0,017 / +0,005 | +0,011 | **+0,057 / +0,062 / +0,036** | **+0,052** |
| Felhen reservadas | ECE | +0,044 / +0,030 / +0,046 | +0,040 | **+0,085** / +0,026 / +0,043 | +0,051 |

Em negrito, valores com o IC todo de um lado.

Cobertura do oráculo com risco de 10% (média das 3 sementes):
- Felhen: H9 29%, H9r 31%;
- Telepatia: H9 69%, H9r 72%.

### Mistura (semente 0), diferença para o `h4_log`

| Recorte | Métrica | w = 0,5 | w = 0,75 | w = 1 (H9) | Replay |
|---|---|---:|---:|---:|---:|
| Telepatia vistas | acc | **−0,2** | −2,0 | −1,9 | −0,5 |
| Felhen reservadas | acc | +8,7 | +12,2 | +13,6 | +13,7 |
| Telepatia reservadas | acc | +14,2 | +15,7 | +16,6 | +17,7 |
| Felhen reservadas | ECE | **−0,001** | +0,015 | +0,044 | +0,085 |
| Felhen vistas | ECE | +0,004 | +0,010 | +0,012 | +0,057 |
| Cobertura oráculo Telepatia | | 57% | 66% | 71% | 74% |

### Leitura

**Replay, pelo critério registrado: resolve.**
- Nas vistas da Telepatia, a perda cai para −0,8 ponto (limite de 1), e nenhuma semente fica com o IC todo
  abaixo de zero.
- O ganho nas reservadas fica igual ou um pouco maior que o da H9 nas duas fontes.
- **Custo novo:** o Felhen **visto** fica superconfiante (ECE +0,05, com o IC acima de zero nas 3 sementes),
  e na semente 1 perde 1,5 ponto. A explicação provável: o replay pega poucos casos de cada família vista e
  repete esses casos por 3 épocas. O modelo decora esses casos e fica confiante demais na matéria antiga.
  Como na H9b, isso parece ser só de escala, e a temperatura deve resolver. Falta confirmar.

**Mistura, pelo critério registrado: troca.**
- w = 0,5 elimina o esquecimento (−0,2) e é a **mais honesta** de todas: o ECE do Felhen novo fica igual ao
  do `h4_log`, sem precisar recalibrar.
- Só que ela perde 4,9 pontos do ganho no Felhen, acima do limite de 3.
- w = 0,75 mantém quase todo o ganho, mas esquece igual à H9.
- Como a regra não foi cumprida, não rodei as sementes 1 e 2.

**Qual usar:**
- Com orçamento para um treino a mais, o **replay** é o melhor em acurácia e não esquece, mas precisa
  recalibrar.
- Sem treino novo, a **mistura com w = 0,5** custa só avaliação e não precisa de calibração extra, mas ensina
  menos no Felhen.

### Limites

- Mistura só na semente 0, com w escolhido no próprio `dev` (otimista).
- As vistas da Telepatia têm 84 grupos, então as diferenças de 1–2 pontos ficam perto do ruído.
- O replay usou proporção 1:1 e rodízio igual por família. Não testei proporções diferentes.

### Recalibrar o replay (job 37766, commit `2fc1c2d`)

`scripts/h9_recalibrar.py` com o `h4_log` como base, média das 3 sementes. A mistura só tem a semente 0.

**Famílias vistas, com a temperatura global** (a que o pipeline já ajusta nas vistas do `calib`):

| Fonte | Braço | Acurácia | ECE cru | ECE com T | T |
|---|---|---:|---:|---:|---:|
| Felhen | `h4_log` | 0,871 | 0,015 | 0,014 | 1,06 |
| Felhen | H9 | 0,870 | 0,026 | 0,019 | 1,25 |
| Felhen | **replay** | 0,863 | **0,066** | **0,017** | 1,62 |
| Telepatia | `h4_log` | 0,732 | 0,160 | 0,040 | 0,41 |
| Telepatia | replay | 0,724 | 0,106 | 0,034 | 0,56 |

A superconfiança do replay no Felhen visto **era só de escala**. A temperatura (T = 1,62) leva o ECE de
0,066 para 0,017, praticamente igual ao `h4_log`. Sobram 0,8 ponto a menos de acurácia e a NLL um pouco maior
(0,406 contra 0,362).

**Famílias novas, receita completa com 25 rótulos (T e limiar da família), alvo de 10% de erro:**

| Fonte | `h4_log` | H9 | **Replay** | Mistura w = 0,5 (s0) |
|---|---|---|---|---|
| Felhen: responde / erra | 16% / 0,18 | 36% / 0,15 | **37% / 0,15** | 32% / 0,16 |
| Telepatia: responde / erra | 7% / 0,18 | 68% / 0,10 | **72% / 0,10** | 56% / 0,10 |
| ECE Felhen com T da família | 0,064 | 0,068 | **0,059** | 0,071 |

### Conclusão

**Configuração final: replay + recalibração.**
- Nas famílias **vistas**, o Lumi não esquece (Telepatia −0,8 ponto) e, com a temperatura que o pipeline já
  ajusta, continua honesto (ECE 0,017 no Felhen e 0,034 na Telepatia).
- Nas famílias **novas**, ele sabe mais (+12 pontos no Felhen, +21 na Telepatia) e é o braço que mais responde
  dentro do alvo: 72% das faturas novas com 10% de erro.

A mistura w = 0,5 é a alternativa sem treino novo, mas responde menos (56% na Telepatia).

### Receita final do projeto

1. **Recompensa honesta** (perda log, H4).
2. **Treino curto com cerca de 100 exemplos da família nova + o mesmo número de casos já vistos** (H9 + replay).
3. **Temperatura das vistas + limiar da família com cerca de 25 rótulos** (H7/H9b).

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H9r: replay das famílias vistas no treino curto</summary>

Registrado antes de rodar (21/11/2026).

#### Pergunta

A H9 (k = 100) ensina as reservadas, mas a acurácia nas **vistas da Telepatia** cai de 1,5 a 3,0 pontos (na
semente 2, o IC fica abaixo de zero). Misturar casos já vistos no treino curto evita esse esquecimento sem
perder o ganho nas novas?

#### Desenho

- Igual à H9 (`--adapt-k 100`, 3 épocas, a partir do `h4_log` da mesma semente), mais `--adapt-replay 1.0`:
  entra um caso visto para cada caso novo.
- Os vistos vêm do conjunto de treino original (`build_training_set`, split `train`, famílias vistas) em
  rodízio entre as famílias. Cada família vista entra por igual, inclusive as da Telepatia.
- Os grupos das reservadas sorteados são os mesmos da H9 (mesmo `data_seed`), então a diferença é só o replay.
- O `adapt_grupos.json` registra só as reservadas, porque os vistos são do `train` e não tocam o `dev`.
- Sementes 0, 1 e 2: `h9r_k100`, `h9r_k100_s1`, `h9r_k100_s2`.

#### Análise

`scripts/analise_h9.py`, com cada semente contra o seu `h4_log` e contra a sua `h9_k100`.

#### Leitura (média das 3 sementes)

- **Replay resolve:** nas vistas da Telepatia, a acurácia fica a no máximo 1 ponto do `h4_log`, e nenhuma
  semente tem o IC todo abaixo de zero. Nas reservadas, o ganho fica a no máximo 3 pontos do ganho da H9 nas
  duas fontes.
- **Troca:** o esquecimento some, mas o ganho nas reservadas cai mais que 3 pontos.
- **Não resolve:** o esquecimento nas vistas da Telepatia continua igual ao da H9.

</details>

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H9m: misturar o `h4_log` com a H9 em vez de fazer replay</summary>

Registrado antes de rodar (21/11/2026).

#### Pergunta

Ficar **entre** o adaptador antigo (`h4_log`) e o ajustado (`h9_k100`) evita o esquecimento nas vistas da
Telepatia e mantém a maior parte do ganho nas reservadas, sem treinar de novo? A ideia vem do post "harness
adapters" da Hugging Face (2026) e do WiSE-FT (Wortsman et al., 2022).

#### Desenho

- `scripts/misturar_adaptadores.py` faz uma mistura exata, ΔW = (1 − w)·ΔW_h4 + w·ΔW_h9:
  - concatena os adaptadores no posto, que passa de r para 2r;
  - ajusta o alpha para manter a escala;
  - confere ΔW no próprio cluster.
- O `slurm/h9m_mistura.sbatch` mistura e avalia `dev` e `calib` de Felhen e Telepatia.
- **Etapa 1 (semente 0):** w = 0,5 e 0,75 (`h9m50_k100`, `h9m75_k100`). Os pontos w = 0 (`h4_log`) e
  w = 1 (`h9_k100`) já existem.
- **Etapa 2:** o melhor w, pelo critério abaixo, roda nas sementes 1 e 2. Escolher w no `dev` da semente 0 é
  otimista, então essas sementes servem de confirmação.
- Análise: `scripts/analise_h9.py` contra o `h4_log` da mesma semente (os grupos do treino curto ficam fora)
  e `scripts/h9_recalibrar.py`.

#### Leitura

- **Mistura resolve:** nas vistas da Telepatia, a acurácia fica a no máximo 1 ponto do `h4_log`. Nas
  reservadas, o ganho fica a no máximo 3 pontos do ganho da H9 nas duas fontes.
- **Escolha de w:** o maior w que cumpre a regra das vistas.
- Comparar com o replay (H9r): o replay custa um treino novo, e a mistura custa só avaliação.

</details>

---

## H10: o Lumi aprende a matéria nova só com "acertou / errou"? (só o plano; trocado pelo piloto da H11)


<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H10: o Lumi aprende a matéria nova só com "acertou / errou"?</summary>

Registrado antes de rodar (21/11/2026).

#### Pergunta

A receita da H9r (treino curto com k = 100 + replay) ensina a matéria nova, mas é supervisionada: o Lumi vê o
gabarito de cada caso. Se ele vir **só o resultado da opção que sorteou** (feedback de bandit, como na H6),
ainda aprende a matéria nova? E a recompensa de acerto (estilo RLVR) volta a deixar ele superconfiante nela?

#### Desenho

Os três braços são iguais ao `h9r_k100` da semente 0:
- partem do `runs/h4_log/adapter`;
- usam k = 100 por família reservada e replay 1:1;
- treinam por 3 épocas, com a semente 0.

Com a mesma semente, o sorteio dos grupos de treino é o mesmo nos três braços. Muda só a perda, com a melhor
configuração do bandit na H6 (G = 32, corte 1000):

| Run | Perda | O que o Lumi vê |
|---|---|---|
| `h9r_k100` (já existe) | log, gabarito completo | tudo |
| `h10_bandit_k100` | `bandit_log`, `--bandit-clip 1000 --bandit-amostras 32` | só se a opção sorteada acertou |
| `h10_acerto_k100` | `bandit_acerto`, `--bandit-amostras 32` | só se a opção sorteada acertou (RLVR) |

Não há código novo: as flags do bandit já combinam com `--adapt-*` e `--init-adapter`.

#### Leitura

- **H10a, RL ensina a matéria nova:** o `h10_bandit_k100` sobe a acurácia das reservadas contra o `h4_log`,
  com IC acima de zero em pelo menos uma fonte. Além disso, fica com pelo menos metade do ganho do `h9r_k100`
  na mesma semente (Felhen +13,7, Telepatia +17,7).
- **H10b, RLVR estraga a honestidade também na matéria nova:** o ECE das reservadas do `h10_acerto_k100` fica
  acima do ECE do `h10_bandit_k100`, com IC acima de zero.
- **Sem esquecer:** nas vistas da Telepatia, a acurácia fica no máximo 1 ponto abaixo do `h4_log`.
- **Receita completa:** com `scripts/h9_recalibrar.py` e 25 rótulos, medir quanto o `h10_bandit_k100` responde
  com 10% de erro e comparar com o `h9r_k100`.
- Se a H10a se confirmar, rodar as sementes 1 e 2, a partir do `h4_log_s1` e do `h4_log_s2`.

#### Análise

- `scripts/analise_h9.py --base h4_log --novos h9r_k100,h10_bandit_k100,h10_acerto_k100`
- `scripts/analise_h9.py --base h9r_k100 --novos h10_bandit_k100,h10_acerto_k100` (o custo de não ver o
  gabarito)
- `scripts/h9_recalibrar.py --base h4_log --novo h10_bandit_k100 --sementes 0 --ks 25`

#### Nota (21/11/2026): não rodada

Este desenho continua no formato de um passo só, em que o RL só estima com ruído o gradiente que a perda log já
dá exato. Por isso, ele no máximo empata com o `h9r_k100`. Trocamos pelo piloto da H11, em que o Lumi raciocina
antes de decidir (`resultados/Experimentos.md, seção H11 e H11b (plano)`).

</details>

---

## H11 e H11b (pilotos): o Lumi consegue raciocinar antes de decidir?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (código em `f09e7d4`, resultado registrado em `84de61c`)
**Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H11 e H11b (plano)` (critério e três notas, todas antes do
piloto inteiro)
**Run:**
- `sbatch slurm/h11_piloto.sbatch runs/h4_log 512`;
- análise em `scripts/h11_piloto.py`, que gera `analises/h11p2.md` (no cluster);
- adaptador `h4_log`, `dev`, ordem canônica, 8 casos por família, Felhen e Telepatia.

### Por que este piloto

Até a H6, a decisão do Lumi era lida das letras numa passada só. Nesse formato, a perda log já dá o gradiente
exato, então o RL no máximo empata com o treino supervisionado.

O RL passa a ser necessário quando o modelo **sorteia um texto antes de decidir**, como um raciocínio. Esse
texto não é derivável, então só dá para aprender com recompensa. Essa é a ideia do RLCR (Damani et al., 2025):
1. o modelo raciocina;
2. a decisão é lida depois do raciocínio;
3. a recompensa é uma regra própria (Brier) nessa decisão.

Antes de montar esse laço (GRPO), o piloto responde quatro perguntas:
- o Lumi raciocina?
- a decisão sobrevive ao raciocínio?
- há sinal para o GRPO?
- cabe no tempo do cluster?

### O que foi testado

- **Guloso:** o Lumi raciocina até 512 tokens, sem sorteio. A letra é lida depois de `</think>\n\n`. A
  comparação é com a leitura de sempre (`<think></think>` vazio), nos mesmos 88 casos.
- **Sorteio:** 4 raciocínios sorteados por caso (temperatura 1,0, top_p 0,95), como um grupo do GRPO. Cada um
  é lido depois de `</think>\n\n`, e a recompensa é −Brier dessa leitura.

### Resultado: guloso ("sem" = leitura de sempre, "com" = depois do raciocínio)

| Fonte | Recorte | n | Acurácia sem → com | ECE sem → com | NLL sem → com | Decisão mudou | Tokens (mediana) | Não fechou | Tempo |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| Felhen | vistas | 40 | 0,900 → 0,875 | 0,115 → 0,137 | 0,36 → 0,98 | 8% | 152 | 10% | 3,4 s |
| Felhen | reservadas | 16 | 0,625 → 0,438 | 0,099 → 0,529 | 0,96 → 3,70 | 56% | 221 | 38% | 6,3 s |
| Telepatia | vistas | 24 | 0,500 → 0,375 | 0,119 → 0,546 | 0,83 → 4,24 | 33% | 216 | 29% | 4,0 s |
| Telepatia | reservadas | 8 | 0,750 → 0,625 | 0,193 → 0,375 | 0,88 → 3,43 | 25% | 210 | 0% | 3,9 s |

No geral, 81% dos raciocínios fecharam o `</think>`, 1% degenerou, e o tempo mediano foi de 3,9 s.

### Resultado: sorteio (grupos de 4)

| Fonte | Recorte | Grupos | Acurácia | ECE | NLL | Fecha | Grupos mistos | Grupos com sinal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Felhen | vistas | 40 | 0,775 | 0,193 | 1,33 | 78% | 32% | 45% |
| Felhen | reservadas | 16 | 0,453 | 0,424 | 3,10 | 44% | 69% | 88% |
| Telepatia | vistas | 24 | 0,458 | 0,435 | 2,99 | 61% | 58% | 88% |
| Telepatia | reservadas | 8 | 0,781 | 0,190 | 1,14 | 53% | 25% | 62% |

- **Fecha:** a amostra fechou o `</think>` antes de 512 tokens.
- **Misto:** no grupo, há decisão certa e errada.
- **Com sinal:** as 4 recompensas do grupo não são todas iguais (desvio > 0,01). Sem isso, o GRPO não aprende
  nada daquele caso.

No geral, 65% das amostras fecharam, nenhuma degenerou, e cada grupo levou 10,1 s (os 4 num lote só).

### O critério registrado

| Critério | Limite | Resultado | |
|---|---|---|---|
| Guloso: fecha o `</think>` | ≥ 80% | 81% | passa (raspando) |
| Guloso: degenera | ≤ 10% | 1% | passa |
| A decisão não quebra (vistas) | queda ≤ 5 pontos | Felhen −2,5; Telepatia −12,5; juntas −6,3 | **não passa** |
| Tempo por raciocínio | ≤ 6 s | 3,9 s | passa |
| Sorteio: fecha o `</think>` | ≥ 80% | 65% | **não passa** |
| Sorteio: degenera | ≤ 10% | 0% | passa |
| Sinal para o GRPO | ≥ 30% dos grupos | 66% | passa |
| Orçamento do GRPO (600 grupos) | < 4 h | 1,7 h | passa |

**Pelo critério: ainda não vale montar o GRPO com o Lumi oficial do jeito que está.** Duas coisas falham:
- a acurácia cai nas vistas da Telepatia;
- um terço dos raciocínios sorteados não termina em 512 tokens.

A queda da Telepatia são 3 casos em 24 (de 12 acertos para 9). A amostra é pequena, mas o critério registrado é
esse.

O plano B registrado (partir do `t6-lumi-4b`) **não se aplica**: ele era para o caso "não raciocina", e o Lumi
raciocina (só 1% degenerou).

### Leitura

**1. O Lumi raciocina, e rápido.** 81% dos raciocínios fecham, quase nada degenera, e cada um leva 3,9 s.

**2. O raciocínio deixa o Lumi superconfiante. Esse é o achado principal.**
- Na leitura de sempre, a H4 deixou o Lumi honesto (ECE de 0,10 a 0,19).
- Depois de raciocinar, ele dá quase 100% para uma letra. Quando erra, erra com certeza.
- O NLL mostra isso. Ele mede o quanto o modelo se surpreende com a resposta certa: 0 é "tinha certeza e
  acertou", e cada erro confiante pesa muito.
  - Nas vistas da Telepatia, ele vai de 0,83 para 4,24.
  - Na média geométrica, a probabilidade dada à resposta certa cai de 44% para 1,4%.
- Até no Felhen visto, onde a acurácia quase não muda (−2,5 pontos), o NLL quase triplica (0,36 → 0,98).

É exatamente o problema que o RLCR ataca: recompensar com Brier a decisão lida depois do raciocínio. O piloto
mostra que esse problema existe no Lumi, o que motiva o RLCD com raciocínio.

**3. Ele raciocina mais longo nos recortes mais difíceis.**
- No guloso, os raciocínios que não fecham são 10% nas vistas do Felhen. Nas reservadas do Felhen são 38%, e
  nas vistas da Telepatia, 29%.
- No sorteio, 78% fecham nas vistas do Felhen, contra 44% a 61% no resto.
- Quando o raciocínio não fecha, nós fechamos o `</think>` e lemos a letra de um pensamento incompleto.
- Dentro de cada recorte, os casos que não fecham não são mais difíceis para a leitura de sempre: ela acerta
  0,71 deles, contra 0,73 dos que fecham.
- A separação mais abaixo mostra o que o corte causa.

**4. Há sinal para o GRPO onde importa.**
- 66% dos grupos têm sinal, e 45% são mistos.
- O sinal é mais forte onde o Lumi erra mais: 88% nas reservadas do Felhen e nas vistas da Telepatia.
- Nas vistas do Felhen, só 45% têm sinal. Na maioria dos outros grupos, os 4 sorteios já acertam com
  certeza, e todos recebem a mesma recompensa.
- Atenção: como as letras saem quase sempre com 100% depois de um raciocínio que termina, esse sinal é quase
  todo de acerto contra erro, e não de confiança (veja a separação abaixo).

**5. Onde ler a letra.**
- Colada no `</think>`, a leitura continua ruim (acurácia de 0,17 a 0,33 em três recortes).
- `\n` e `\n\n` quase empatam.
- `\n` tem NLL um pouco menor nos 4 recortes (por exemplo, 3,98 contra 4,24 nas vistas da Telepatia), e é o
  que o Lumi escreve sozinho.

A leitura principal continuou `\n\n`, como registrado. Para o próximo ciclo, `\n` é a escolha natural.

**6. Cabe no tempo.** Com os 4 sorteios num lote, 600 grupos levam cerca de 1,7 h de geração. O tempo dos
passos de treino não entra nessa conta.

### Separação: os que fecharam e os que não fecharam (feita depois do piloto)

**Data:** 21/11/2026, job 37786 (só CPU), commit `2996b2e`. A regra de leitura foi registrada no plano antes de
rodar.

**Guloso, todos os casos juntos** ("sem" = leitura de sempre nos mesmos casos):

| Raciocínio | n | Acurácia sem → com | NLL sem → com | Confiança média (com) | Erros com confiança ≥ 0,9 |
|---|---:|---|---|---:|---:|
| Fechou | 71 | 0,732 → 0,704 | 0,59 → 2,74 | 1,00 | 21 de 21 |
| Não fechou | 17 | 0,706 → 0,353 | 0,84 → 1,94 | 0,73 | 4 de 11 |

**Sorteio, todas as amostras juntas:**

| Raciocínio | Amostras | Acurácia | NLL | Confiança média | Erros com confiança ≥ 0,9 |
|---|---:|---:|---:|---:|---:|
| Fechou | 229 | 0,716 | 2,35 | 1,00 | 64 de 65 |
| Não fechou | 123 | 0,472 | 1,60 | 0,75 | 16 de 65 |

Os números por recorte estão em `analises/h11p2.md`, no cluster. Um exemplo, nas vistas da Telepatia:
- os que fecharam vão de 0,353 para 0,412 de acurácia, mas o NLL vai de 0,90 para 5,24, e os 10 erros têm
  confiança ≥ 0,9;
- os que não fecharam caem de 0,857 para 0,286.

**Pela regra registrada: o comprimento não é a causa principal.**
- A condição 1 passa: nos que fecharam, a acurácia cai só 2,8 pontos (o limite era 5).
- A condição 2 falha: dos 25 erros confiantes do guloso, só 4 (16%) estão nos que não fecharam.

Então a superconfiança é do próprio raciocínio, o GRPO com Brier é o remédio, e o comprimento é um problema a
mais.

**Leitura: são dois problemas, com duas causas.**

1. **A queda de acurácia vem dos raciocínios cortados.**
   - Nos mesmos casos, os que não fecharam vão de 0,706 para 0,353 (−35 pontos). Os que fecharam perdem só 2,8.
   - Os cortados não eram casos mais difíceis: a leitura de sempre acerta 0,71 deles.
   - Ler a letra de um pensamento pela metade estraga a decisão. É isso que derruba a acurácia das vistas da
     Telepatia no critério.
2. **A superconfiança vem dos raciocínios que terminam.**
   - Quando o raciocínio fecha, a confiança média é 1,00. Os 21 erros do guloso têm confiança de pelo menos
     0,9, e no sorteio são 64 de 65.
   - Os cortados são menos confiantes (0,73), porque o pensamento não chegou a uma conclusão.

**Por que isso acontece (interpretação):** o raciocínio termina dizendo a resposta ("So answer N"). Depois do
`</think>`, a leitura das letras só copia essa conclusão, com certeza total. A probabilidade das letras deixa de
medir dúvida.

**Consequência para o GRPO (ponto de atenção no desenho):**
- A recompensa planejada é o Brier das letras lidas depois do raciocínio. Se essas letras saem quase sempre com
  100%, o Brier vira quase só "acertou (0) ou errou (−2)".
- Essa é a recompensa de acerto que, na H4, estragou a honestidade.
- O GRPO até pode aprender a terminar o raciocínio com dúvida (os cortados mostram que a leitura pode sair
  menos confiante), mas nada garante.
- No RLCR, o modelo **escreve uma confiança** (um número) depois da resposta, e a recompensa soma o acerto e o
  Brier dessa confiança. É uma confiança que o modelo consegue graduar, o que parece o caminho mais seguro.

### H11b: o Lumi escreve uma confiança que serve? (feito depois do piloto)

**Resposta curta: ele escreve o número, mas quase sempre "100". O número não separa os acertos dos erros o
bastante, e o critério falha em 1 de 3 pontos.**

- **Plano e critério,** registrados antes de rodar: `resultados/Experimentos.md, seção H11 e H11b (plano)`.
- **Run:** job 37901, relatório `analises/h11b.md` (no cluster).
- **Desenho:** os mesmos 88 casos e raciocínios do piloto. Depois da letra, um turno novo pergunta "Qual é a
  probabilidade, de 0 a 100, de que a sua resposta esteja correta?".

| critério | limite | resultado | |
|---|---|---|---|
| Escreve: respostas com número de 0 a 100 | ≥ 90% | 100% | passa |
| Gradua: o valor mais comum | ≤ 80% das respostas | 77% (valor 100) | passa |
| Separa: AUROC da confiança escrita | ≥ 0,65 | 0,639 | **não passa** |

**Quatro fontes de confiança nos mesmos casos (guloso):**

| fonte de confiança | acurácia da decisão | AUROC | confiança média: acerta / erra | ECE | Brier |
|---|---:|---:|---|---:|---:|
| confiança escrita | 0,636 | 0,639 | 0,97 / 0,91 | 0,337 | 0,334 |
| letras depois do raciocínio | 0,636 | 0,592 | 0,97 / 0,91 | 0,324 | 0,317 |
| consistência dos 4 sorteios | 0,636 | 0,769 | 0,78 / 0,41 | 0,125 | 0,195 |
| letras sem raciocínio (H4; outra decisão) | 0,727 | 0,798 | 0,75 / 0,54 | 0,066 | 0,160 |

O AUROC é a chance de um acerto ter confiança maior que um erro: 0,5 é sorteio, 1,0 é perfeito.

**Leitura:** olhamos se o número muda pelo motivo que a gente acha.
- **O "gradua" passou só no papel.** 85 dos 88 números ficam entre 90 e 100, e só 4 valores diferentes
  aparecem. Na prática, ele escreve "100" ou "quase 100".
- **Ele diz cerca de 95% e acerta 64%** (ECE 0,34). A confiança escrita repete a das letras, com as mesmas médias
  (0,97 quando acerta, 0,91 quando erra).
- **O AUROC de 0,639 vem mais da diferença entre grupos do que de julgar cada caso.**
  - Dentro de cada recorte (fonte e vistas ou reservadas), ele fica entre 0,46 e 0,59.
  - Pelas médias, os poucos números baixos ficam nas reservadas do Felhen, onde ele também acerta menos (44%).
  - Ou seja, ele percebe um pouco que aquela matéria é mais difícil, mas não qual caso vai errar.
- **A melhor confiança depois do raciocínio é a consistência dos 4 sorteios** (AUROC 0,769, ECE 0,125). Quando
  os raciocínios chegam a letras diferentes, ele costuma errar. O custo são 4 raciocínios por caso.
- **As letras sem raciocínio (H4) continuam as mais honestas** (AUROC 0,798, ECE 0,066) e acertam mais (0,727
  contra 0,636). Mas elas julgam outra decisão, a de sempre, sem raciocínio.
- **No sorteio:**
  - a confiança escrita muda dentro do grupo de 4 em 45% dos grupos;
  - a recompensa do RLCR tem sinal em 55% dos grupos, contra 66% com o Brier das letras.

**O que isso muda:** a leitura registrada para este resultado é "escreve, mas não separa". O formato funciona:
100% de números, 0,18 s por resposta. Mas a confiança escrita terá de ser **aprendida**. Caminhos:
- o GRPO estilo RLCR com o Brier no número escrito desde o começo;
- um SFT curto do formato antes. Uma ideia ainda não testada é usar a consistência dos 4 sorteios como alvo do
  número escrito, porque é ela que separa melhor.

**Limites da H11b:**
- **88 casos e 32 erros no guloso.** O AUROC tem erro de cerca de ±0,06, então 0,639 e 0,65 não se distinguem
  de verdade. O que pesa mais é o número quase sempre "100" e a separação perto de 0,5 dentro de cada recorte.
- **A pergunta vem depois, num turno novo.** No RLCR, o número sai na mesma geração do raciocínio.
- **O adaptador `h4_log` nunca viu um segundo turno.**

### O que isso muda no próximo ciclo

**Atualização (24/11/2026): a trilha do raciocínio foi encerrada depois da H11b.** O motivo está em
`Resultados finais - RLCD no Lumi.md`, seção "Decisão: parar a trilha do raciocínio". O texto abaixo fica como
estava, para o caso de alguém retomar.

O caminho continua sendo o **GRPO com raciocínio e recompensa Brier** (RLCR). O sinal (66% dos grupos) e o
tempo (1,7 h de geração) estão bons. Há dois ajustes antes de montar:

1. **Comprimento:** é ele que derruba a acurácia. Opções:
   - mais tokens;
   - recompensa baixa para o raciocínio que não fecha, para o GRPO aprender a fechar;
   - pedir no prompt um raciocínio curto.
2. **De onde vem a confiança:** depois de um raciocínio que termina, a leitura das letras sai com 100%. Para o
   Brier ensinar honestidade, a confiança precisa vir de algo que o modelo consiga graduar, como a confiança
   escrita do RLCR. A H11b mostrou que, perguntada depois, a confiança escrita também sai quase sempre 100:
   ela terá de ser aprendida.

### Limites

- **Poucos casos:** 88 no total, 8 por família. As reservadas da Telepatia são 8 casos de uma família só. O ECE
  aqui é só indicativo.
- **Um adaptador só:** o `h4_log` foi treinado com o `<think>` vazio. Não sabemos se o Lumi oficial sem
  adaptador fica igualmente superconfiante. Parte do efeito pode ser o adaptador fora do formato.
- **Uma semente de sorteio,** fixa por caso.
- **Raciocínios cortados:** os que não fecham são lidos com o `</think>` fechado por nós. A separação mostra
  que a queda de acurácia vem quase toda daí.
- **A separação foi feita depois de ver os dados,** com só 17 raciocínios cortados no guloso. Ela orienta o
  desenho, mas não muda o critério.
- **Só `dev`.** O `test` oficial não foi usado.

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H11 (piloto): o Lumi consegue raciocinar antes de decidir?</summary>

Registrado antes de rodar (21/11/2026).

#### Por que

No `lumi-decision`, a decisão é lida das letras numa passada só. Essa distribuição é derivável, então a perda
log já é o gradiente exato, e o RL só estima esse mesmo gradiente com ruído
(`tests/test_h6_bandit.py::test_bandit_log_tem_o_gradiente_do_log_score_em_media`). Por isso, nesse formato, o
RL no máximo empata com o supervisionado (H5, H6).

O RL passa a ser **necessário** quando algo entre a pergunta e a decisão é sorteado e não derivável. O caso
clássico é o raciocínio em texto (RLCR, Damani et al., 2025):
- o Lumi raciocina;
- a decisão é lida depois do raciocínio;
- a recompensa é uma regra própria (Brier) nessa decisão.

O laço de GRPO fica para o próximo ciclo. Este piloto só responde se vale a pena montá-lo.

#### Desenho

- **Código:** `lumi-eval run --raciocinio N`.
  - O prompt abre o `<think>` (`render(..., pensar=True)`; se o template ignorar, `prompt_pensando` abre).
  - O Lumi gera até N tokens, de forma gulosa, e para no `</think>`.
  - A letra é lida depois do `</think>`, na mesma fronteira de sempre (`com_raciocinio`).
- **Run:** `slurm/h11_piloto.sbatch`, com o adaptador `runs/h4_log`, `dev`, ordem canônica, 8 casos por
  família, Felhen e Telepatia (vistas e reservadas), N = 256.
- **Comparação:** a leitura de sempre do `h4_log` nos mesmos casos (`runs/h4_log_<fonte>_dev`), feita por
  `scripts/h11_piloto.py`.
- **Teste curto antes:** `H11_LIMITE=4`, que gera as pastas `runs/h11p_teste_*`.

#### Critério (vale montar o GRPO com o Lumi oficial?)

- **Raciocina:** pelo menos 80% dos raciocínios fecham o `</think>` antes do limite, e no máximo 10% degeneram
  (vazio, ou mais de 50% de 4-gramas repetidos).
- **A decisão não quebra:** nas vistas, a acurácia com raciocínio fica no máximo 5 pontos abaixo da leitura de
  sempre. O adaptador foi treinado com o `<think>` vazio, então alguma perda é esperada.
- **Cabe no orçamento:** tempo mediano de até 6 s por raciocínio. Assim, cerca de 2400 rollouts
  (300 casos × G = 4 × 2 épocas) cabem em menos de 4 h de geração.
- **Só informativo:** se o raciocínio muda a acurácia e o ECE, principalmente nas reservadas. Com 8 casos por
  família, o ECE é só indicativo.
- **Se não raciocinar:** o plano B do próximo ciclo é partir de um Lumi com raciocínio
  (`gustavobarrosss/t6-lumi-4b`).

#### Nota (21/11/2026): 1º teste curto (job 37782) e a correção da leitura

- **O Lumi raciocina e acerta no texto.** Nos 4 casos vistos do Felhen, o raciocínio termina com a letra
  certa ("So answer N", "That's option E", "So answer F", "Thus answer: T"). 75% fecharam o `</think>`, nada
  degenerou, e o tempo mediano foi de 3,9 s.
- **A leitura colada no `</think>` quebrou.** A acurácia vista caiu para 0, e o NLL do Felhen (3,23) é
  praticamente o de um chute uniforme entre 25 opções (ln 25 = 3,22). Com raciocínio, o Lumi escreve um
  separador depois do `</think>` antes da resposta, então ler a letra colada é ler fora do lugar.
- **Correção (antes do 2º teste):**
  - a leitura principal passa a ser depois de `</think>\n\n`;
  - também lemos as variantes colada e `\n`, para comparar;
  - gravamos os 12 tokens que o Lumi escreve sozinho depois do `</think>`;
  - extraímos a letra que o próprio texto diz (`letra_do_raciocinio`).
- Pastas novas: `runs/h11p2_*`. O critério acima não muda e vale para a leitura principal. A acurácia pela
  letra do texto entra como informação.

#### Nota (21/11/2026): 2º teste curto e o braço com sorteio (registrado antes do piloto inteiro)

**O 2º teste curto (`analises/h11p2_teste.md`, 4 casos vistos por fonte):**
- com a leitura depois de `</think>\n\n`, a acurácia vai de 1,0 para 1,0 no Felhen e de 0,25 para 0,5 na
  Telepatia;
- 88% dos raciocínios fecham, a mediana é 3,2 s, e `\n` e `\n\n` empatam;
- depois do raciocínio, o Lumi fica quase certo de tudo: NLL 0,000 no Felhen, e 2,14 na Telepatia, onde erra
  com confiança. Esse é o problema que o RLCR ataca.

**Por que sortear:** o GRPO não usa a geração gulosa. Ele sorteia G raciocínios por pergunta e aprende com a
diferença de recompensa entre eles. Os cards do Lumi-Think e do Nemotron recomendam temperatura 1,0 e top_p
0,95 com raciocínio.

**Desenho:**
- `--raciocinio-amostras 4 --temperatura 1.0 --top-p 0.95`;
- 4 raciocínios sorteados por caso, num lote só, com semente fixa por caso;
- cada um lido depois de `</think>\n\n`;
- recompensa = −Brier dessa leitura.

**Critério do sorteio (vale montar o GRPO?):**
- **Raciocina sorteando:** pelo menos 80% das amostras fecham o `</think>`, e no máximo 10% degeneram.
- **Há sinal para o GRPO:** em pelo menos 30% dos grupos, o desvio da recompensa é maior que 0,01. Abaixo
  disso, a maioria dos casos não ensina nada ao GRPO.
- **Cabe no orçamento:** com o tempo mediano por grupo, 2400 raciocínios (600 grupos de 4) ficam abaixo de
  4 h de geração.
- **Só informativo:**
  - a fração de grupos mistos (decisão certa e errada no mesmo grupo);
  - a acurácia, o ECE e o NLL das amostras.

#### Nota (21/11/2026): 3º teste curto (sorteio) e o orçamento do piloto inteiro

**O 3º teste curto (`analises/h11p2_teste.md`, 4 casos vistos por fonte, G = 4, N = 256):**
- o lote de 4 funcionou: 5,6 s por grupo, e o GRPO projetado fica em cerca de 0,9 h;
- há sinal de sobra: 88% dos grupos têm desvio da recompensa maior que 0,01, e 75% são mistos;
- nenhuma amostra degenerou;
- **mas só 56% das amostras fecharam o `</think>` em 256 tokens**, contra 88% no guloso. Sorteando, o Lumi
  raciocina mais longo.

**Mudança antes do piloto inteiro:** N passa de 256 para **512** (`sbatch slurm/h11_piloto.sbatch runs/h4_log
512`). O critério não muda. A mudança só dá orçamento para o raciocínio sorteado terminar.

**Observação:** o Lumi escreveu `"\nB"` depois do `</think>`, então o separador natural parece ser `\n`, não
`\n\n`. A leitura principal continua `\n\n`, como registrado, e a tabela de variantes do piloto mostra se isso
muda alguma coisa. O GRPO do próximo ciclo escolhe a leitura com esses dados.

#### Resultado do piloto inteiro (21/11/2026, `analises/h11p2.md` no cluster)

88 casos (8 por família), N = 512, G = 4. Leitura principal `\n\n`, como registrado.

| Critério | Limite | Resultado | |
|---|---|---|---|
| Guloso: fecha o `</think>` | ≥ 80% | 81% | passa |
| Guloso: degenera | ≤ 10% | 1% | passa |
| A decisão não quebra (vistas) | queda ≤ 5 pontos | Felhen −2,5; Telepatia −12,5; juntas −6,3 | **não passa** |
| Tempo por raciocínio | ≤ 6 s | 3,9 s | passa |
| Sorteio: fecha o `</think>` | ≥ 80% | 65% | **não passa** |
| Sorteio: degenera | ≤ 10% | 0% | passa |
| Sinal para o GRPO | ≥ 30% dos grupos | 66% | passa |
| Orçamento do GRPO | < 4 h | 1,7 h | passa |

**Pelo critério: ainda não vale montar o GRPO com o Lumi oficial do jeito que está.** O plano B (t6) não se
aplica, porque ele era para "não raciocina", e o Lumi raciocina.

Informativo:
- depois do raciocínio, o Lumi fica superconfiante: o NLL sobe de 0,36–0,96 para 0,98–4,24, e o ECE chega
  a 0,55;
- os raciocínios que não fecham se concentram onde ele sabe menos (Felhen reservadas e Telepatia vistas);
- `\n` fica um pouco melhor que `\n\n` no NLL nos 4 recortes.

Detalhes em `Experimentos.md, seção H11 e H11b`, na pasta `Decision Models`.

#### Nota (21/11/2026): separar os que fecharam dos que não fecharam (decidida depois do piloto)

**Por que:** os dois pontos que falharam (a queda nas vistas e o sorteio que não fecha) podem ter a mesma
causa: o comprimento do raciocínio.

**O que:** `scripts/h11_piloto.py` separa, no guloso e no sorteio, os raciocínios que fecharam o `</think>`
dos que bateram no limite. Para cada grupo, mostra:
- a acurácia sem → com, nos mesmos casos;
- o NLL;
- a confiança média;
- os erros com confiança ≥ 0,9.

Só CPU, sem rodar a GPU de novo.

**Leitura (registrada antes de ver, na linha "todas", porque cada recorte tem poucos casos):**
- **O comprimento é a causa principal** se as duas condições valem no guloso:
  - nos que fecharam, a acurácia com raciocínio fica no máximo 5 pontos abaixo da leitura de sempre;
  - mais da metade dos erros confiantes (≥ 0,9) está nos que não fecharam.

  Nesse caso, o próximo ciclo começa pelo comprimento.
- **Senão, a superconfiança é do próprio raciocínio.** O GRPO com Brier é o remédio, e o comprimento é um
  problema a mais.
- É uma análise feita depois de ver os dados, com poucos casos. Ela orienta o desenho do próximo ciclo, mas
  não muda o critério do piloto.

**Resultado da separação (job 37786, linha "todas" do guloso):**

| Raciocínio | n | Acurácia sem → com | Confiança média | Erros com confiança ≥ 0,9 |
|---|---:|---|---:|---:|
| Fechou | 71 | 0,732 → 0,704 | 1,00 | 21 de 21 |
| Não fechou | 17 | 0,706 → 0,353 | 0,73 | 4 de 11 |

- A condição 1 passa (−2,8 pontos).
- A condição 2 falha: só 4 dos 25 erros confiantes (16%) estão nos que não fecharam.

**Pela regra: a superconfiança é do próprio raciocínio.**
- A queda de acurácia vem dos cortados.
- No sorteio, os que fecharam têm 64 de 65 erros confiantes.

**Ponto de atenção para o GRPO:** com as letras a 100% depois do raciocínio, o Brier das letras vira quase só
acerto ou erro. O próximo ciclo deve tirar a confiança de algo que o modelo consiga graduar, como a confiança
escrita do RLCR.

</details>

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H11b (piloto): o Lumi escreve uma confiança graduada depois de raciocinar?</summary>

Registrado antes de rodar (22/11/2026).

#### Por que

No piloto da H11, quando o raciocínio termina, a leitura das letras sai com confiança de 1,00. Os 21 erros do
guloso tinham confiança de pelo menos 0,9, e no sorteio foram 64 de 65 (`resultados/Experimentos.md, seção H11 e H11b (plano)`). O
raciocínio termina dizendo a resposta, e as letras só copiam essa conclusão.

Por isso, se a recompensa do GRPO for o Brier das letras, ela vira quase só "acertou ou errou". Essa é a
recompensa que estragou a honestidade na H4.

No RLCR (Damani et al., 2025), o modelo **escreve um número de confiança** depois da resposta, e a recompensa
soma o acerto e o Brier desse número. Este piloto pergunta se o Lumi, do jeito que está, já escreve um número
que muda com a dúvida e é mais baixo quando ele erra.

#### Desenho

- **Sem raciocínio novo:** o piloto reaproveita os raciocínios do piloto da H11 (`runs/h11p2_<fonte>_dev`, 88
  casos, adaptador `h4_log`).
- **Remontagem:** o prompt em que a letra foi lida é remontado e conferido pelo `prompt_sha1` gravado na H11.
- **A pergunta:** depois da letra escolhida, entra um **turno novo do usuário**:
  "Qual é a probabilidade, de 0 a 100, de que a sua resposta esteja correta? Responda apenas com o número."
  O Lumi responde no modo de sempre (`<think></think>`), de forma gulosa, em até 16 tokens.
- **O formato do turno** sai do próprio template (`prompts.turno_confianca`).
- **Os sorteados também:** o mesmo vale para os 4 raciocínios sorteados de cada caso.
- **Código e saída:**
  - `scripts/h11b_confianca.py`, `slurm/h11b_confianca.sbatch`;
  - pastas `runs/h11b_<fonte>_dev`, relatório `analises/h11b.md`.

**Comparação, nos mesmos casos:**
- a confiança escrita;
- as letras depois do raciocínio;
- a consistência dos 4 sorteios (fração que escolheu a mesma letra);
- as letras sem raciocínio da H4, só como referência, porque julgam outra decisão.

**Métricas:**
- AUROC, a chance de um acerto ter confiança maior que um erro;
- a confiança média quando acerta e quando erra;
- ECE e Brier binários.

#### Critério (no guloso, com os 88 casos)

**A confiança escrita serve de ponto de partida para o GRPO estilo RLCR?**
- **Escreve:** pelo menos 90% das respostas trazem um número de 0 a 100.
- **Gradua:** o valor mais comum aparece em no máximo 80% das respostas com número.
- **Separa:** AUROC da confiança escrita de pelo menos 0,65.
- **Só informativo:**
  - a comparação com as letras e com a consistência;
  - o ECE e o Brier;
  - no sorteio, se a confiança escrita muda dentro do grupo, e a fração de grupos com sinal para a recompensa do
    RLCR (acerto − (c − acerto)², desvio > 0,01), comparada com o Brier das letras.

#### Leitura

- **Passa tudo:** o próximo ciclo usa o formato do RLCR (resposta + confiança escrita, recompensa = acerto +
  Brier da confiança), partindo do Lumi como está.
- **Escreve, mas não gradua ou não separa:** o formato funciona, mas a confiança terá de ser aprendida. O
  caminho é o GRPO com o Brier na confiança escrita desde o começo, ou um SFT curto do formato antes.
- **Não escreve:** o problema é de formato. O ajuste é na pergunta ou no SFT do formato.

#### Limites

- **A confiança é perguntada depois, num turno novo.** No RLCR, ela sai na mesma geração do raciocínio, e o
  modelo pode pensar na dúvida antes. Este piloto mede se o Lumi julga o próprio raciocínio quando perguntado.
- **Poucos casos:** 88, com cerca de 25 erros no guloso. O AUROC tem erro de cerca de ±0,06.
- **O adaptador `h4_log` nunca viu um segundo turno.**

#### Teste curto antes

`H11B_LIMITE=4` (pastas `runs/h11b_teste_*`). Conferir:
- o fim do prompt, que deve ter a letra, o fim do turno e a pergunta;
- as respostas cruas;
- se o `prompt_sha1` bate (senão, o job para com erro);
- o tempo por resposta.

#### Nota (23/11/2026): teste curto (job 37794, `analises/h11b_teste.md`)

8 casos (4 vistos por fonte), com os 4 sorteados de cada um.

**A mecânica funciona:**
- o prompt termina como planejado: o raciocínio, `</think>\n\n`, a letra, `<|im_end|>`, a pergunta nova e
  `<|im_start|>assistant\n<think></think>`;
- o `prompt_sha1` bateu em todos os casos (o job não parou);
- 100% das respostas trazem um número;
- 0,27 s por resposta.

**Primeiros sinais** (8 casos, só indicativos; o critério vale no piloto inteiro):
- a confiança escrita fica sempre em 90 ou mais, e 75% das respostas são "100";
- no guloso, ela não separa acerto de erro (AUROC 0,30 com 3 erros);
- nos 32 sorteados, separa um pouco (0,71);
- a consistência dos 4 sorteios separou os 8 casos (AUROC 1,0).

Nada muda no desenho. O piloto inteiro roda com `sbatch slurm/h11b_confianca.sbatch` (88 casos, pastas
`runs/h11b_*`).

#### Nota (24/11/2026): piloto inteiro (job 37901, `analises/h11b.md`)

**O critério falha em 1 de 3 pontos:**
- **Escreve:** 100% das respostas trazem um número (passa);
- **Gradua:** o valor mais comum, "100", aparece em 77% (passa, perto do limite de 80%);
- **Separa:** AUROC 0,639, abaixo de 0,65 (não passa).

**Na leitura registrada, é o caso "escreve, mas não separa":** o formato funciona, e a confiança escrita terá de
ser aprendida, com o GRPO usando o Brier no número desde o começo ou com um SFT curto do formato antes.

**O que pesa, além do critério:**
- **o "gradua" passa só no papel:** 85 dos 88 números ficam entre 90 e 100, com 4 valores diferentes;
- **a separação vem mais da diferença entre grupos:** dentro de cada recorte, o AUROC fica entre 0,46 e 0,59;
- **a consistência dos 4 sorteios é a melhor fonte depois do raciocínio** (AUROC 0,769, ECE 0,125).

Resultado completo em `resultados/Experimentos.md, seção H11 e H11b`.

</details>

---

## H12: a perda do Clef (entropia cruzada com label smoothing + Brier) no Lumi

### Resposta curta

**Não vale trocar.** A perda do Clef acerta um pouco mais na Telepatia (+1 a +2 pontos), mas deixa a confiança
menos honesta no Felhen, e a temperatura não conserta tudo. A receita final continua com a perda log.

Agora a pergunta "fizemos o que o Clef faz?" tem resposta medida: a perda dele foi testada, com 3 sementes, e não
ganhou da nossa.

### O que foi testado

- **Perda `log_brier`:**
  - entropia cruzada com o alvo suavizado `q_s = (1 − ε)·q + ε/K`, com ε = 0,1;
  - mais o Brier com o alvo original;
  - pesos 1 e 1. O post do Clef não dá pesos nem ε, então a escolha é nossa.
- **Todo o resto igual ao `h4_log`:** dados, `lr`, épocas, LoRA r = 16.
- **3 sementes** (`h4_clef`, `h4_clef_s1`, `h4_clef_s2`), comparadas com as 3 do `h4_log` no `dev`.
- **Plano e critério registrados antes de rodar:** `resultados/Experimentos.md, seção H12 (plano)`.

### Números crus (sem temperatura)

Diferença `clef − log`, 3 sementes juntas, IC 95% por bootstrap de grupos (`scripts/comparar_h4.py`).

| Recorte | Acurácia | ECE | Brier | NLL |
|---|---|---|---|---|
| Felhen vistas | 0,871 → 0,872 (empate) | 0,011 → 0,047 (**pior**) | **pior** (+0,004) | 0,360 → 0,401 (**pior**) |
| Felhen reservadas | 0,529 → 0,522 (empate) | 0,065 → 0,054 (empate) | **pior** (+0,005) | 1,239 → 1,268 (**pior**) |
| Telepatia vistas | 0,732 → 0,742 (**melhor**) | 0,160 → 0,185 (**pior**) | **pior** (+0,005) | 0,698 → 0,722 (**pior**) |
| Telepatia reservadas | 0,582 → 0,600 (**melhor**) | 0,091 → 0,106 (**pior**) | empate | 0,919 → 0,919 (empate) |

Em negrito, diferenças com o IC 95% todo de um lado.

### Depois da temperatura

O `calibrar` do sbatch ajusta uma temperatura nas vistas do `calib` de cada fonte e aplica no `dev`
(`scripts/resumo_calibracao.py`, média das 3 sementes, sem IC).

| Recorte | Braço | Temperatura | ECE antes → depois | NLL antes → depois |
|---|---|---:|---|---|
| Felhen vistas | log | 1,06 | 0,015 → 0,014 | 0,360 → 0,362 |
| | clef | **0,83** | 0,048 → **0,028** | 0,401 → **0,379** |
| Felhen reservadas | log | 1,06 | 0,073 → 0,062 | 1,239 → 1,230 |
| | clef | 0,83 | 0,060 → **0,101** | 1,268 → **1,290** |
| Telepatia vistas | log | 0,41 | 0,160 → 0,040 | 0,698 → 0,591 |
| | clef | 0,40 | 0,185 → 0,043 | 0,722 → 0,589 |
| Telepatia reservadas | log | 0,41 | 0,111 → 0,189 | 0,919 → 1,201 |
| | clef | 0,40 | 0,107 → 0,183 | 0,919 → 1,164 |

O ECE "antes" desta tabela é a média do ECE de cada semente. O da tabela anterior junta as 3 sementes num só
cálculo. Por isso os dois diferem um pouco.

### Leitura

- **Critério registrado:** **Clef pior.**
  - Não é empate: a calibração crua piora em 9 das 12 células de ECE, Brier e NLL.
  - Também não é "Clef melhor": o ECE e o Brier das reservadas não caem.
- **Pelo motivo que a gente achava? Em parte, sim.**
  - O label smoothing ensina o modelo a nunca dar 100% para uma letra. Por isso o Clef sai **pouco confiante**:
    a temperatura dele no Felhen é 0,83, ou seja, ela precisa afiar as probabilidades. A do `log` é 1,06, quase
    nada.
  - Nas vistas do Felhen, a temperatura tira mais da metade da diferença (ECE 0,028 contra 0,014), mas não toda.
  - Nas reservadas do Felhen, a mesma temperatura **piora** o Clef (ECE 0,060 → 0,101). Ela é ajustada nas
    vistas, onde o modelo acerta muito, e afiar a confiança na matéria nova, onde ele erra metade, deixa o
    Clef superconfiante.
  - Na Telepatia, os dois braços precisam da mesma temperatura forte (cerca de 0,40). Depois dela, os dois
    empatam. A temperatura piora as reservadas da Telepatia nos dois braços, e isso já era conhecido: a matéria
    nova precisa da sua própria temperatura (H7).
- **A acurácia na Telepatia (+1 a +2 pontos) é o único ganho.**
  - A temperatura não muda qual letra é escolhida, então esse ganho fica.
  - Mas ele não aparece no Felhen, onde a acurácia das reservadas fica até um pouco abaixo (−0,7, empate).
  - Não sabemos se o ganho vem do smoothing, do Brier ou da soma dos dois. Uma explicação possível é o smoothing
    funcionar como regularização numa fonte com poucos grupos vistos (84). Isso não foi testado.

### O que isso muda

- **A receita final não muda:** a perda log continua.
  - A perda do Clef só renderia o ganho da Telepatia em troca de uma confiança pior no Felhen, mesmo depois da
    temperatura.
  - Na receita final, o treino curto na matéria nova (H9) já dá +20 pontos na Telepatia. O +1,8 do Clef é
    pequeno perto disso.
- **A tabela "Clef × nosso projeto" fica fechada na parte da perda.** O que falta do Clef é estrutural: posto 256,
  cabeça de roteamento (já testada pelo Palhares na H3, e foi pior) e dados sintéticos com prompts e formatos
  variados.

### Limites

- **ε = 0,1 e pesos 1:1 escolhidos por nós,** sem busca. Um ε menor deve ficar mais perto do `log`.
- **Não separamos as duas peças** (smoothing e Brier). Na H4, o Brier sozinho empatou com o log, então o
  suspeito principal do custo na calibração é o smoothing, mas isso não foi medido à parte.
- **Posto 16, não 256,** e um modelo de 4B.
- **Só `dev`.** A tabela pós-temperatura não tem IC.

### Como reproduzir

```bash
cd /raid/$USER/lumi-decision && sbatch slurm/e3_lora_h2.sbatch h4_clef --loss log_brier --suavizar 0.1 --seed 0 && sbatch slurm/e3_lora_h2.sbatch h4_clef_s1 --loss log_brier --suavizar 0.1 --seed 1 && sbatch slurm/e3_lora_h2.sbatch h4_clef_s2 --loss log_brier --suavizar 0.1 --seed 2
```

```bash
cd /raid/$USER/lumi-decision && srun -p h100n2 --cpus-per-task=2 --mem=8G bash -c 'source slurm/config.sh && lumi_exec python scripts/comparar_h4.py --base log --novos clef --sementes 0,1,2 --out analises/h12_clef.md'
```

```bash
cd /raid/$USER/lumi-decision && python3 scripts/resumo_calibracao.py --bracos log,clef --sementes 0,1,2
```

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H12: o braço Clef (entropia cruzada com label smoothing + Brier)</summary>

Registrado antes de rodar (24/11/2026).

#### Por que

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

#### O que esperamos, e por quê

- **A soma não deve mudar o ponto de chegada.** O log e o Brier são regras próprias: as duas são mínimas quando
  a confiança é igual à chance real de acerto.
- **O label smoothing puxa a confiança para o meio.** No ótimo, os números crus saem um pouco menos confiantes.
  Num caso de teste com alvo (0,6; 0,3; 0,1), o ótimo vira (0,58; 0,30; 0,12). A temperatura da receita final
  deve corrigir isso.
- **Nossa aposta é empate com o `h4_log`.**

#### Desenho

- **Perda** `log_brier` (`lora.option_loss`):
  - entropia cruzada com o alvo suavizado `q_s = (1 − ε)·q + ε/K`;
  - mais o Brier com o alvo original;
  - pesos 1 e 1. O post não dá pesos; essa escolha é nossa.
- **Label smoothing:** ε = 0,1 (`--suavizar 0.1`).
- **Todo o resto igual ao `h4_log`:** dados, `lr`, épocas, LoRA r = 16. O `slurm/e3_lora_h2.sbatch` treina,
  avalia `calib` e `dev` nas duas fontes e recalibra.
- **Sementes:** 0, 1 e 2. Os runs se chamam `h4_clef`, `h4_clef_s1` e `h4_clef_s2`, para o
  `scripts/comparar_h4.py` ler sem mudança.

#### Critério

Diferença `clef − log` com IC 95% por bootstrap de grupos (`comparar_h4.py`, métricas cruas), em cada fonte
(Felhen, Telepatia) e grupo (vistas, reservadas):
- **Empate:** os ICs de ECE, Brier, NLL e acurácia passam pelo zero. A receita do Clef não acrescenta nada à
  perda log no nosso uso, e a receita final não muda.
- **Clef melhor:** o ECE ou o Brier das reservadas cai, com o IC fora do zero, sem perder acurácia. Aí vale testar
  a receita final com a perda do Clef.
- **Clef pior:** registrar o custo. Se for só o label smoothing deixando os números crus menos confiantes, a
  diferença deve sumir depois da temperatura (relatórios do `calibrar` que o sbatch roda).

#### Como rodar

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

#### Limites

- **Pesos e ε escolhidos por nós,** porque o Clef não publica os dele. Não há busca.
- **Posto 16, não 256,** e um modelo de 4B, não de 27B.
- **Só `dev`.**

#### Nota (24/11/2026): resultado

Rodou com 3 sementes. Critério: **Clef pior.** A calibração crua piora em 9 de 12 células, e a temperatura
conserta só em parte: o Clef sai pouco confiante (T = 0,83 no Felhen, contra 1,06 do `log`), e nas reservadas do
Felhen a temperatura até piora (ECE 0,101 contra 0,062). O único ganho é a acurácia da Telepatia (+1 a +2 pontos).
A receita final não muda. Detalhes em `resultados/Experimentos.md, seção H12`.

</details>

---

## H12b: na perda do Clef, o que faz o quê? O smoothing ou o Brier?

### Resposta curta

**O custo na calibração vem do label smoothing. O ganho de acurácia na Telepatia vem do Brier.**

A perda `log_brier` (entropia cruzada + Brier, **sem** smoothing):
- fica com o ganho da Telepatia: +2,2 pontos nas reservadas, mais que a perda inteira do Clef (+1,8);
- tem só um custo pequeno na calibração.

Pelo critério registrado, ela é a versão da perda do Clef que vale manter. O próximo passo é testá-la dentro
da receita final (H9).

### O que foi testado

Dois braços novos, cada um com uma peça da perda do Clef. Todo o resto é igual ao `h4_log`. São 3 sementes, no
`dev`, e o plano foi registrado antes de rodar em `resultados/Experimentos.md, seção H12b (plano)`.

| Braço | O que tem |
|---|---|
| `log` | Entropia cruzada pura |
| `suave` | Só o smoothing (ε = 0,1) |
| `logbrier` | Só o Brier |
| `clef` (H12) | As duas peças |

### Números crus (sem temperatura)

Diferença contra a `log`, IC 95% por bootstrap de grupos (`scripts/comparar_h4.py`). Em negrito, as
diferenças com o IC todo de um lado.

| Recorte | Medida | `suave` | `logbrier` | `clef` |
|---|---|---|---|---|
| Felhen vistas | ECE | **+0,049** | −0,004 (empate) | **+0,036** |
| Felhen vistas | NLL | **+0,046** | **+0,005** | **+0,041** |
| Felhen reservadas | NLL | **+0,021** | **−0,013** (melhor) | **+0,029** |
| Telepatia vistas | acurácia | +0,2 (empate) | +0,4 (empate) | **+1,0** |
| Telepatia reservadas | acurácia | **+0,9** | **+2,2** [+1,7; +2,6] | **+1,8** |
| Telepatia reservadas | ECE | **+0,008** | **+0,014** | **+0,014** |
| Telepatia reservadas | NLL | +0,004 (empate) | **−0,006** (melhor) | 0,000 (empate) |

### Depois da temperatura

`scripts/resumo_calibracao.py`, média das 3 sementes.

| Braço | T no Felhen | ECE Felhen vistas (depois) | ECE Felhen reservadas (depois) | ECE Telepatia reservadas (depois) |
|---|---:|---:|---:|---:|
| `log` | 1,06 | 0,014 | 0,062 | 0,189 |
| `suave` | **0,81** | 0,028 | **0,106** | 0,196 |
| `logbrier` | 1,05 | 0,013 | **0,053** | **0,176** |
| `clef` | **0,83** | 0,028 | **0,101** | 0,183 |

### Leitura

- **Custo da calibração: vem do smoothing.**
  - O `suave` repete o padrão do `clef`:
    - as vistas do Felhen pioram (ECE +0,049);
    - a temperatura fica abaixo de 1 (0,81), sinal de modelo pouco confiante;
    - depois da temperatura, as reservadas do Felhen pioram (0,106 contra 0,062).
  - O `logbrier` não repete esse padrão. Ele tem T = 1,05, como a `log`, e, depois da temperatura, fica até um
    pouco melhor que ela.
  - O que esperávamos antes de rodar aconteceu.
- **Ganho de acurácia da Telepatia: vem do Brier.**
  - O `logbrier` repete o ganho e passa dele: +2,2 contra +1,8 do `clef`.
  - O `suave` sozinho dá só +0,9.
  - O smoothing até tira um pouco do ganho quando entra junto.
- **O custo que sobra no `logbrier`:**
  - ECE +0,014 nas reservadas da Telepatia, cru;
  - NLL +0,005 nas vistas do Felhen, cerca de 1/9 do custo do `clef`.
  - Depois da temperatura, os dois somem: o `logbrier` fica igual à `log` ou melhor.
- **Pelo motivo certo? Não sabemos ainda.**
  - Uma explicação possível: o Brier é limitado (no máximo 2), e a perda log não é. Com o Brier somado, os
    exemplos muito errados puxam menos o treino, e isso ajudaria numa fonte com poucos grupos vistos.
  - Não testamos isso.
  - O ganho só aparece nas reservadas da Telepatia, que são uma família só. No Felhen, a acurácia empata (+0,6).

### O que isso muda

- **A receita final ainda não muda.**
  - O ganho foi medido no `h4_log`, antes do treino curto na matéria nova.
  - Na receita, a H9 já dá +20 pontos na Telepatia, e os +2,2 podem sumir.
- **Próximo passo:** rodar a receita com o `logbrier` no lugar da `log`:
  - um treino curto (H9r) a partir do `h4_logbrier`;
  - e a recalibração com 25 rótulos.

  Se ele responder mais casos com o mesmo erro, ele entra na receita.

### Limites

- Só `dev`, com 3 sementes.
- Um ε só (0,1) e peso 1 no Brier, sem busca.
- A matéria nova da Telepatia é uma família só.

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H12b: separar o label smoothing do Brier na perda do Clef</summary>

Registrado antes de rodar (24/11/2026).

#### Por que

Na H12, a perda do Clef (entropia cruzada com label smoothing ε = 0,1 + Brier) saiu pior que a `log` na
calibração e melhor na acurácia da Telepatia (+1 a +2 pontos). A perda tem duas peças, e não sabemos qual causa
cada efeito. O suspeito do custo é o smoothing: a temperatura do Clef foi 0,83, ou seja, ele saiu pouco
confiante, e na H4 o Brier sozinho empatou com a `log`. Mas isso não foi medido.

Se o ganho de acurácia vier do Brier e o custo vier do smoothing, dá para manter uma versão da perda do Clef sem
o smoothing.

#### Desenho

Dois braços novos, cada um com uma peça só. Todo o resto é igual ao `h4_log` e ao `h4_clef`.

| Braço | Perda | O que tem |
|---|---|---|
| `h4_log` (já existe) | `log` | Entropia cruzada pura |
| `h4_suave` | `log_suave`, `--suavizar 0.1` | Só o smoothing (entropia cruzada com alvo suavizado, sem Brier) |
| `h4_logbrier` | `log_brier`, `--suavizar 0` | Só o Brier (entropia cruzada pura + Brier) |
| `h4_clef` (já existe) | `log_brier`, `--suavizar 0.1` | As duas peças |

3 sementes cada (0, 1 e 2). O `log_suave` foi criado agora em `lora.option_loss`: o ótimo dele é exatamente o
alvo suavizado `q_s = 0,9·q + 0,1/K` (testado em `tests/test_h4_perdas.py`).

#### O que esperamos

- **Calibração:** o `suave` fica parecido com o `clef` (pior que a `log`, com temperatura abaixo de 1), e o
  `logbrier` empata com a `log`.
- **Acurácia da Telepatia:** não temos aposta. Uma das peças deve carregar o ganho, ou só a soma das duas.

#### Critério

Diferença contra a `log` com IC 95% por bootstrap de grupos (`comparar_h4.py`), e a conferência pós-temperatura
(`resumo_calibracao.py`).

- **Custo da calibração:**
  - vem do smoothing se o `suave` repetir o padrão do `clef` (ECE ou NLL pior nas vistas do Felhen, com IC fora
    do zero) e o `logbrier` não repetir;
  - vem do Brier se acontecer o contrário;
  - vem da soma se só o `clef` repetir.
- **Ganho de acurácia da Telepatia:** fica com o braço que repetir o +1 a +2 pontos com IC fora do zero, nas
  vistas ou nas reservadas. Se nenhum braço sozinho repetir, o ganho vem da soma, ou é frágil.
- **Decisão sobre manter a perda do Clef:**
  - se o `logbrier` tiver o ganho de acurácia sem o custo de calibração, ele é a versão a manter, e o próximo
    passo é testá-lo dentro da receita final (H9);
  - se o ganho vier só do smoothing, ele vem junto com o custo. Aí a `log` continua, e a opção seria testar um
    ε menor (não registrado aqui).

#### Como rodar

Sem teste curto: o `log_suave` passa pelo mesmo caminho do `log_brier`, que já rodou na H12, e os testes
locais cobrem a conta. 6 treinos:

```bash
cd /raid/$USER/lumi-decision && git pull -q && for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s"); sbatch slurm/e3_lora_h2.sbatch h4_suave$n --loss log_suave --suavizar 0.1 --seed $s; sbatch slurm/e3_lora_h2.sbatch h4_logbrier$n --loss log_brier --suavizar 0 --seed $s; done; squeue -u $USER
```

Comparação (sem GPU):

```bash
cd /raid/$USER/lumi-decision && srun -p h100n2 --cpus-per-task=2 --mem=8G bash -c 'source slurm/config.sh && lumi_exec python scripts/comparar_h4.py --base log --novos suave,logbrier,clef --sementes 0,1,2 --out analises/h12b_separar.md' && cat analises/h12b_separar.md && python3 scripts/resumo_calibracao.py --bracos log,suave,logbrier,clef --sementes 0,1,2
```

#### Limites

- **Só ε = 0,1 e pesos 1:1,** sem busca.
- **Só `dev`.** A tabela pós-temperatura não tem IC.

</details>

---

## H12c: a perda `log_brier` dentro da receita final

### Resposta curta

**Empate. A receita continua com a perda log.** Antes do treino curto, a `log_brier` acertava +2,2 pontos na
Telepatia nova (H12b). Depois da receita inteira, esse ganho some: os dois braços respondem a mesma coisa, com o
mesmo erro.

### O que foi testado

- **Braço `h9rb_k100`:** a receita final inteira com a `log_brier` (`--suavizar 0`):
  - parte do `h4_logbrier`;
  - treino curto igual ao `h9r_k100`: 100 exemplos por família nova + replay, 3 épocas;
  - a mesma recalibração, com 25 rótulos.
- **Comparação:** contra o `h9r_k100` da mesma semente, com os mesmos grupos no treino curto. São 3 sementes.
- **Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H12c (plano)`.

### Resultado

**Receita (25 rótulos, alvo de 10% de erro, média das 3 sementes):**

| Fonte | Receita de hoje (`log`): responde / erra | Com `log_brier`: responde / erra |
|---|---|---|
| Telepatia | 72% / 0,101 | 72% / 0,100 |
| Felhen | 35% / 0,149 | 35% / 0,153 |

**Acurácia, `log_brier` − `log`, por semente** (IC 95% por bootstrap de grupos):

| Recorte | Semente 0 | Semente 1 | Semente 2 |
|---|---|---|---|
| Telepatia reservadas | +0,5 (empate) | +0,1 (empate) | +0,4 (empate) |
| Felhen reservadas | −1,4 (empate) | −0,9 (empate) | −0,1 (empate) |
| Telepatia vistas | +1,2 (empate) | +0,6 (empate) | **+1,4** |
| Felhen vistas | +0,1 (empate) | +0,1 (empate) | −0,3 (empate) |

**Outras medidas:**
- **Calibração das vistas depois da temperatura:** ECE 0,017 nos dois braços no Felhen; na Telepatia, 0,034 na
  `log` contra 0,040 na `log_brier`.
- **"Sabe que não sabe":** o AUROC dentro da família é 0,75 contra 0,76 no Felhen e 0,83 nos dois na Telepatia.

### Leitura

- **Critério registrado: empate.** A cobertura e a acurácia ficam dentro do ruído. A `log` continua, porque é a
  mais simples.
- **Pelo motivo certo?**
  - O ganho da H12b era em matéria que o modelo **nunca tinha visto**.
  - Depois que o treino curto ensina essa matéria (+20 pontos na Telepatia), o jeito de pontuar o treino deixa de
    fazer diferença.
  - Era o risco registrado na H12b, e foi o que aconteceu.
- **O que se aprende:** uma melhora pequena medida antes do treino curto não garante melhora na receita. Toda peça
  nova precisa ser testada dentro da receita.

### Limites

- Só `dev`, com 3 sementes.
- Na Telepatia, a matéria nova é uma família só.

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H12c: a perda `log_brier` dentro da receita final</summary>

Registrado antes de rodar (24/11/2026).

#### Pergunta

Na H12b, a `log_brier` (entropia cruzada + Brier, sem smoothing) acertou +2,2 pontos nas reservadas da
Telepatia sem o custo de calibração do smoothing. Mas isso foi antes do treino curto, e na receita final a H9
já dá +20 pontos na Telepatia. **O ganho continua depois da receita?**

#### Desenho

- **Braço `h9rb_k100`:** a receita final inteira com a `log_brier` (`--suavizar 0`) nas duas etapas:
  1. parte do `h4_logbrier` da mesma semente, e não do `h4_log`;
  2. treino curto igual ao `h9r_k100`: `--adapt-k 100`, `--adapt-replay 1.0`, 3 épocas e as mesmas 3 famílias
     reservadas;
  3. a mesma recalibração, com 25 rótulos da família.
- **Comparação:** contra o `h9r_k100` da mesma semente. Os grupos sorteados para o treino curto são os mesmos,
  porque vêm da mesma semente.
- **Sementes:** 0, 1 e 2 (`h9rb_k100`, `h9rb_k100_s1`, `h9rb_k100_s2`).

#### Critério

- **Receita** (`scripts/h9_recalibrar.py --base h9r_k100 --novo h9rb_k100`, k = 25, alvo de 10%):
  - quanto cada braço responde e quanto erra entre os respondidos, nas reservadas.
- **Acurácia e esquecimento** (`scripts/analise_h9.py`, cada semente contra o seu `h9r_k100`):
  - a diferença nas reservadas e nas vistas, com IC.
- **"Sabe que não sabe"** (`scripts/analise_sabe.py --base h9r_k100 --novo h9rb_k100`):
  - o AUROC dentro da família não pode cair.

**Leitura:**
- **Entra na receita** se, nas duas fontes:
  - responde mais que o `h9r_k100` com o erro dos respondidos a no máximo 1 ponto de distância;
  - não esquece: nas vistas, a acurácia fica a no máximo 1 ponto do `h9r_k100`;
  - a temperatura das vistas continua consertando a calibração.
- **Empate:** a cobertura e a acurácia ficam dentro do ruído entre sementes. A `log` continua, porque é a mais
  simples.
- **Piora:** responde menos ou esquece mais. A `log` continua.

#### Como rodar

```bash
cd /raid/$USER/lumi-decision && git pull -q && F=felhen:juristcu_area,felhen:factck_veracidade,telepatia:invoice_processing && for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s"); sbatch --time=04:00:00 slurm/e3_lora_h2.sbatch h9rb_k100$n --init-adapter runs/h4_logbrier$n/adapter --loss log_brier --suavizar 0 --adapt-familias $F --adapt-k 100 --adapt-replay 1.0 --epochs 3 --seed $s; done; squeue -u $USER
```

#### Limites

- Só `dev`, com 3 sementes.
- Na Telepatia, a matéria nova é uma família só.

</details>

---

## H13: variar o texto do prompt no treino (a parte de dados do Clef)

### Resposta curta

**O `h4_log` é frágil a um prompt que nunca viu, e variar o prompt no treino conserta boa parte disso.**
Na matéria nova, o resultado depende da fonte.

- **Com um prompt nunca visto:**
  - o `h4_log` perde 3 a 5 pontos nas famílias vistas;
  - em muitos casos, ele deixa de querer responder com uma letra.
- **O `h4_aug` (treinado com 4 textos de prompt diferentes) é mais robusto:**
  - recupera +2 pontos no prompt novo;
  - quase sempre responde com uma letra.
- **No prompt de sempre, a matéria nova muda de jeitos opostos:**
  - Telepatia: **+3,9 pontos**;
  - Felhen: **−1,8 ponto**.

### O que foi testado

- **Variantes do prompt** (`src/lumi_decision/prompts.py`):
  - **0:** o prompt de sempre;
  - **1 a 3:** usadas no treino. Mudam títulos, frases, a ordem dos blocos e o formato das opções. Na Telepatia,
    as chaves do JSON também são embaralhadas;
  - **4:** só na avaliação, nunca vista no treino.
- **Braço `h4_aug`:** igual ao `h4_log` em tudo, menos o texto. A cada época, cada exemplo recebe uma variante de
  0 a 3. São 3 sementes.
- **Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H13 (plano)`.

### Resultado

Diferenças com IC 95% por bootstrap de grupos, 3 sementes juntas. Em negrito, as que têm o IC todo de um lado.

**1. Quanto o `h4_log` sofre com o prompt nunca visto** (prompt 4 − prompt de sempre):

| Recorte | Acurácia | NLL |
|---|---|---|
| Felhen vistas | **−2,9** | **+0,098** |
| Felhen reservadas | **−3,0** | **+0,061** |
| Telepatia vistas | **−4,6** | **+0,052** |
| Telepatia reservadas | **+2,0** | +0,009 (empate) |

**Ele ainda responde com uma letra?**
- Medida: a massa de probabilidade nas letras, onde 1 quer dizer que ele só quer escrever uma letra
  (`scripts/h13_letras.py`).
- No prompt de sempre, fica em 0,99 em todos os braços.

| Prompt nunca visto | Massa nas letras | Casos com massa abaixo de 0,5 | Gera a letra que a leitura escolheu |
|---|---|---|---|
| `h4_log`, Felhen | 0,42 a 0,58 | 35% a 75% | 34% a 71% |
| `h4_aug`, Felhen | 0,60 a 0,69 | 15% a 34% | 72% a 90% |
| `h4_log`, Telepatia | 0,50 a 0,64 | 16% a 45% | 70% a 95% |
| `h4_aug`, Telepatia | 0,70 a 0,77 | 0% a 1% | 97% a 98% |

(Faixa entre as 3 sementes.)

**2. `aug − log` no prompt de sempre:**

| Recorte | Acurácia | ECE | NLL |
|---|---|---|---|
| Felhen vistas | +0,1 (empate) | empate | empate |
| Felhen reservadas | **−1,8** | +0,009 (empate) | **+0,027** |
| Telepatia vistas | +1,0 (empate) | empate | empate |
| Telepatia reservadas | **+3,9** [3,0; 4,8] | **+0,019** | **−0,015** |

**3. `aug − log` no prompt nunca visto:**

| Recorte | Acurácia | ECE | NLL |
|---|---|---|---|
| Felhen vistas | **+2,1** | **−0,009** | **−0,050** |
| Felhen reservadas | −1,0 (empate) | empate | empate |
| Telepatia vistas | **+2,3** | **+0,020** | **−0,016** |
| Telepatia reservadas | **+2,5** | +0,018 (empate) | **−0,031** |

### Leitura

- **Robustez: o `h4_aug` ganha.**
  - No prompt nunca visto, as vistas sobem +2 pontos nas duas fontes.
  - Ele continua respondendo com uma letra, principalmente na Telepatia.
  - O `h4_log` perdia o formato: no Felhen, até 75% dos casos tinham menos da metade da probabilidade nas letras.
  - Pelo critério registrado, o ganho do item 3 é **robustez**.
- **Generalização: depende da fonte.**
  - **Telepatia:** a matéria nova sobe +3,9 no prompt de sempre, mas o ECE piora (+0,019). Pelo critério
    registrado, piorar a calibração conta como "piora", mesmo com o NLL e o Brier melhorando.
  - **Felhen:** a matéria nova cai −1,8. É uma "piora".
- **Pelo motivo certo? Há um sinal de que não, na Telepatia.**
  - O próprio `h4_log` acerta **+2,0** na Telepatia nova só por trocar o texto do prompt, sem treino nenhum.
  - Então a família nova da Telepatia é sensível ao jeito de apresentar o caso.
  - Parte do +3,9 pode ser o `aug` ter visto mais formatos, e não ter aprendido mais a matéria. Isso não foi
    separado.
- **A lição da H12c vale aqui:** uma melhora medida antes do treino curto pode sumir dentro da receita.

### O que isso muda

- **A receita final ainda não muda.**
- O que o `aug` traz de mais claro é **robustez a prompts diferentes**. Isso importa no uso real, em que cada
  pessoa escreve o pedido de um jeito.
- O próximo passo seria testar a receita com o `aug`:
  - o treino curto parte do `h4_aug`, também com o prompt variado;
  - avaliar no prompt de sempre e no nunca visto.

  A receita entra se mantiver a cobertura e ficar mais robusta.

### Limites

- **Variantes escritas por nós:** em português, e só 4 no treino.
- **O formato da resposta não muda:** sempre uma letra.
- **Só `dev`, com 3 sementes.** Na Telepatia, a matéria nova é uma família só.

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H13: variar o texto do prompt no treino (a parte de dados do Clef)</summary>

Registrado antes de rodar (24/11/2026).

#### Por que

O Clef treina com dados sintéticos, em que trocam a ordem dos campos, os prompts e os formatos. A gente só sorteia a
ordem das opções. Hoje o Lumi treina sempre com **o mesmo texto de prompt**, e não sabemos:
- quanto ele piora com um prompt que nunca viu;
- se variar o texto no treino ajuda, tanto na robustez quanto na matéria nova.

#### Desenho

- **Variantes do prompt** (`src/lumi_decision/prompts.py`, `_VARIANTES`):
  - **0:** o prompt de sempre, sem nenhuma mudança;
  - **1 a 3:** outros títulos e outra frase de pedido. Mudam também a ordem dos blocos (na 2, a pergunta vem antes)
    e o formato das linhas de opção (`A.`, `A)`, `A -`);
  - **4:** nunca entra no treino e serve só para medir a robustez. As opções vêm antes da pergunta, no formato `(A)`;
  - em todas, a resposta é a mesma: uma letra depois de `<think></think>`;
  - **Telepatia:** nas variantes 1 a 4, a ordem das chaves do JSON do caso também é sorteada.
- **Braço `h4_aug`** (`--variar-prompt true`):
  - a cada época, cada exemplo recebe uma variante de 0 a 3, sorteada por caso e por época;
  - todo o resto é igual ao `h4_log`: perda log, os mesmos exemplos, o mesmo número de passos, LoRA r = 16;
  - a única coisa que muda é o texto, então a comparação é limpa.
- **Sementes:** 0, 1 e 2 (`h4_aug`, `h4_aug_s1`, `h4_aug_s2`).
- **Prompt novo:** o `slurm/h13_variante.sbatch` avalia o `dev` com a variante 4 em `runs/<NOME>_<fonte>_devvar`,
  para o `h4_log` e para o `h4_aug`, nas 3 sementes.

#### Critério

Diferenças com IC 95% por bootstrap de grupos (`scripts/comparar_h4.py`):

1. **Quanto o `h4_log` sofre com um prompt novo** (`--base log --novos log --split dev --split-novo devvar`):
   - acurácia e calibração no prompt novo contra o prompt de sempre;
   - se nada muda, o Lumi já é robusto, e a H13 só pode ajudar na matéria nova.
2. **No prompt de sempre** (`--base log --novos aug`):
   - **ajuda:** a acurácia das reservadas sobe com o IC fora do zero, sem piorar a calibração;
   - **empate:** os ICs passam pelo zero;
   - **piora:** a acurácia cai, ou a calibração piora.
3. **No prompt novo** (`--base log --novos aug --split devvar`): mesma leitura.

**Leitura:**
- se só o item 3 melhorar, o ganho é **robustez**, não saber mais a matéria;
- se as reservadas melhorarem no item 2, o ganho é **generalização**;
- conferir a massa das letras (`cand_mass`) e se a geração começa pela letra (`gen_first_is_readout_argmax`) no
  prompt novo. O modelo tem que continuar respondendo com uma letra.

#### Como rodar

Teste curto (a perda tem que cair, e o prompt novo tem que rodar):

```bash
cd /raid/$USER/lumi-decision && git pull -q && E3_N=20 sbatch slurm/e3_lora_h2.sbatch h4_aug_curto --variar-prompt true --max-steps 40 --save-every 10
```

3 sementes:

```bash
cd /raid/$USER/lumi-decision && sbatch slurm/e3_lora_h2.sbatch h4_aug --variar-prompt true --seed 0 && sbatch slurm/e3_lora_h2.sbatch h4_aug_s1 --variar-prompt true --seed 1 && sbatch slurm/e3_lora_h2.sbatch h4_aug_s2 --variar-prompt true --seed 2
```

Prompt novo (o `h4_log` já pode rodar; o `h4_aug` só depois do treino):

```bash
cd /raid/$USER/lumi-decision && for n in h4_log h4_log_s1 h4_log_s2; do sbatch slurm/h13_variante.sbatch $n; done
```

#### Limites

- **Variantes escritas por nós,** em português, e só 4 para o treino. O Clef não publica as dele.
- **O formato da resposta não muda** (sempre uma letra), por escolha: a leitura das letras é a base do projeto.
- **Só `dev`.**

</details>

---

## H13b: a receita final com o prompt variado

### Resposta curta

**A receita não muda: o prompt variado tem um custo no prompt de sempre.** Mas a H13b mostrou uma coisa
importante: **a receita de hoje é frágil a um prompt que nunca viu.**

- **No prompt nunca visto:** a receita de hoje cai de 72% para **50%** de faturas novas respondidas.
- **Com o prompt variado:**
  - o prompt nunca visto melhora (57%);
  - mas o prompt de sempre piora (66% contra 72%).

### O que foi testado

- **Braço `h9ra_k100`:** a receita com o prompt variado nas duas etapas. Parte do `h4_aug`, e o treino curto
  também usa `--variar-prompt true`. São 3 sementes.
- **Comparação:** contra a receita de hoje (`h9r_k100`), com os mesmos exemplos no treino curto.
- **Dois prompts:**
  - o de sempre (`dev`);
  - a variante 4, que nenhum treino viu (`devvar`).
- **Recalibração:** 25 rótulos da família, no próprio prompt avaliado.
- **Plano registrado antes de rodar:** `resultados/Experimentos.md, seção H13b (plano)`.

### Resultado

**Receita (25 rótulos, alvo de 10% de erro, média das 3 sementes): responde / erra**

| Fonte | Prompt | Receita de hoje | Com prompt variado |
|---|---|---|---|
| Telepatia | de sempre | 72% / 0,101 | **66%** / 0,102 |
| Telepatia | nunca visto | **50%** / 0,104 | 57% / 0,103 |
| Felhen | de sempre | 35% / 0,149 | 33% / 0,156 |
| Felhen | nunca visto | 31% / 0,156 | 31% / 0,165 |

**Ele ainda responde com uma letra no prompt nunca visto?**

| | Massa nas letras | Casos com massa abaixo de 0,5 |
|---|---|---|
| Receita de hoje, Felhen | 0,38 a 0,53 | 45% a 73% |
| Prompt variado, Felhen | 0,48 a 0,65 | 24% a 55% |
| Receita de hoje, Telepatia | 0,51 a 0,57 | 31% a 49% |
| Prompt variado, Telepatia | 0,59 a 0,72 | 0,5% a 20% |

(Faixa entre as 3 sementes. No prompt de sempre, fica em 0,99 nos dois braços.)

**Acurácia, prompt variado − receita de hoje (IC 95% por bootstrap de grupos):**
- **Prompt de sempre** (semente 0):
  - Felhen reservadas: **−3,2**;
  - Telepatia reservadas: **−1,5**;
  - vistas: empate.
- **Prompt nunca visto:**
  - Telepatia reservadas: **+2,6**, +1,3 (empate) e **+2,5**;
  - Felhen vistas: **+1,4**, **+6,0** e **+4,3**;
  - Felhen reservadas: **−2,1**, +1,0 e +1,7 (empates).

### Leitura

1. **A receita de hoje é frágil?** **Sim.**
   - No prompt nunca visto, a Telepatia cai de 72% para 50% de respostas.
   - Em até 73% dos casos do Felhen, o Lumi já nem quer responder com uma letra.
   - O treino curto não resolveu a fragilidade que a H13 achou no `h4_log`.
2. **O prompt variado piora o prompt de sempre?** **Sim.**
   - A Telepatia responde 6 pontos a menos, e o limite registrado era 2.
   - A acurácia das reservadas cai.
   - Pelo critério, isso já basta: **a receita fica como está**.
3. **E no prompt nunca visto?** **O prompt variado ganha.**
   - A Telepatia responde +7 pontos e acerta mais.
   - As vistas do Felhen sobem.
   - O Lumi volta a responder com uma letra.

**Pelo motivo certo?**
- É uma troca.
- Treinar com 4 textos diferentes ensina o Lumi a não depender do texto exato: ganha no prompt novo.
- Ao mesmo tempo, ele fica um pouco menos afiado no texto de sempre: perde no prompt de sempre.
- Não é um ganho de graça. Por isso o critério escolheu a receita mais simples.

### O que isso muda

- **A receita final não muda,** mas ganha um **limite novo**: ela foi medida só com o prompt de sempre. Com um
  texto diferente, responde bem menos.
- **Ideias para depois (não registradas):**
  - misturar o prompt de sempre na maior parte do treino e as variantes numa parte menor;
  - ou treinar com muito mais variantes, para que nenhuma seja "a de sempre".

### Limites

- Uma variante nunca vista só (a 4), escrita por nós.
- Só `dev`, com 3 sementes. A comparação semente por semente no prompt de sempre só foi impressa para a
  semente 0. A receita tem as 3.
- Na Telepatia, a matéria nova é uma família só.

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano H13b: a receita final com o prompt variado</summary>

Registrado antes de rodar (25/11/2026).

#### Pergunta

Na H13, treinar com o texto do prompt variado (`h4_aug`) deixou o Lumi mais robusto a um prompt que ele nunca
viu, mas isso foi antes do treino curto. Na H12c, um ganho medido nesse ponto sumiu dentro da receita.
**A robustez continua depois da receita?** E a receita com o prompt variado responde tanto quanto a de hoje?

#### Desenho

- **Braço `h9ra_k100`:** a receita final com o prompt variado nas duas etapas:
  1. parte do `h4_aug` da mesma semente;
  2. treino curto igual ao `h9r_k100` (100 exemplos por família nova + replay, 3 épocas, perda log), também com
     `--variar-prompt true`;
  3. a mesma recalibração, com 25 rótulos.
- **Comparação:** contra o `h9r_k100` da mesma semente, com os mesmos grupos no treino curto. São 3 sementes.
- **Dois prompts:**
  - `dev`: o prompt de sempre;
  - `devvar`: a variante 4, nunca vista em treino nenhum (`slurm/h13_variante.sbatch`), avaliada nos dois braços.
- **Recalibração no prompt nunca visto:**
  - os 25 rótulos da família e a medida saem do `devvar`;
  - a temperatura das vistas continua vindo do `calib` do prompt de sempre;
  - o `h9_recalibrar.py` e o `analise_h9.py` ganharam `--split`.

#### Critério

Análises em `slurm/analise_h13b.sbatch`.

1. **A receita de hoje é frágil?** Comparar o `h9r_k100` no `devvar` com ele mesmo no `dev`:
   - se a cobertura e a acurácia quase não mudam, e a massa nas letras fica perto de 1, o treino curto já
     resolveu a fragilidade, e o `aug` não é necessário.
2. **No prompt de sempre (`dev`), o `aug` não pode piorar a receita:**
   - a cobertura das reservadas fica a no máximo 2 pontos do `h9r_k100`;
   - o erro dos respondidos fica a no máximo 1 ponto;
   - as vistas ficam a no máximo 1 ponto de acurácia.
3. **No prompt nunca visto (`devvar`), o `aug` tem que ganhar:**
   - responde mais na receita, ou acerta mais, com IC fora do zero em pelo menos uma fonte;
   - e a massa nas letras é maior.

**Leitura:**
- **Entra na receita** se o item 2 passar e o item 3 ganhar.
- **Empate:** se o item 3 empatar, a receita fica como está, porque é mais simples.
- **Piora:** se o item 2 falhar, a receita também fica como está.

#### Como rodar

Os treinos, as avaliações no prompt novo e a análise ficam encadeados:

```bash
cd /raid/$USER/lumi-decision && git pull -q && F=felhen:juristcu_area,felhen:factck_veracidade,telepatia:invoice_processing && J="" && for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s"); T=$(sbatch --parsable --time=04:00:00 slurm/e3_lora_h2.sbatch h9ra_k100$n --init-adapter runs/h4_aug$n/adapter --variar-prompt true --adapt-familias $F --adapt-k 100 --adapt-replay 1.0 --epochs 3 --seed $s) && V=$(sbatch --parsable --dependency=afterok:$T slurm/h13_variante.sbatch h9ra_k100$n) && R=$(sbatch --parsable slurm/h13_variante.sbatch h9r_k100$n) && J="$J:$V:$R"; done; sbatch --dependency=afterok$J slurm/analise_h13b.sbatch; squeue -u $USER
```

#### Limites

- Só `dev`, com 3 sementes.
- Uma variante nunca vista só (a 4), escrita por nós.
- Na Telepatia, a matéria nova é uma família só.

</details>

---

## Sabe que não sabe: o Lumi sabe o que não sabe?

### Resposta curta

**Sim, depois da receita final.** Na matéria nova, quando ele diz "não sei", erraria 45% a 47% daqueles casos.
Quando responde, erra 10% a 16%.

A confiança separa acerto de erro **dentro da mesma família** (AUROC 0,75 a 0,83). Então o "não sei" não está só
separando matéria difícil de matéria fácil.

Antes da receita, o mesmo teste dá bem menos: AUROC de 0,59 na Telepatia e de 0,69 no Felhen.

### O que foi medido

- **Script:** `scripts/analise_sabe.py`, sem GPU, com as predições que já existiam. O plano foi registrado
  antes de rodar em `resultados/Experimentos.md, seção Sabe que não sabe (plano)`.
- **Modelos:** o `h4_log` e o `h9r_k100` (a receita final), com 3 sementes.
- **Matéria nova:** a mesma recalibração da receita final:
  - a temperatura e o limiar saem de 25 rótulos da família;
  - risco-alvo de 10%;
  - os mesmos sorteios do `h9_recalibrar.py`, com 100 repetições;
  - os grupos do treino curto e os 25 rótulos ficam fora da medida.
- **"Não sei"** = confiança abaixo do limiar. O erro dos "não sei" é o que ele erraria se fosse forçado a
  responder.
- **AUROC:** a chance de um acerto ter confiança maior que um erro. 0,5 é o acaso, e 1 é a separação perfeita.
- **AUROC dentro da família:** a mesma conta, sem misturar matérias diferentes. É o teste limpo.
- **IC 95%:** por bootstrap de grupos e de repetições.

### Resultado

**Matéria nova (famílias reservadas):**

| Fonte | Modelo | Responde | Erro quando responde | Erro quando diz "não sei" | AUROC dentro da família |
|---|---|---:|---|---|---|
| Telepatia | `h4_log` | 7% | 19% [3; 32] | 43% [39; 48] | 0,59 [0,56; 0,62] |
| Telepatia | **receita final** | **72%** | **10%** [6; 15] | **47%** [39; 53] | **0,83** [0,80; 0,85] |
| Felhen | `h4_log` | 16% | 20% [5; 33] | 53% [47; 59] | 0,69 [0,64; 0,73] |
| Felhen | **receita final** | **35%** | **16%** [7; 26] | **45%** [38; 53] | **0,75** [0,71; 0,79] |

**Matéria vista, como referência:**
- no Felhen, a confiança separa bem (AUROC dentro da família de 0,85 no `h4_log` e de 0,82 na receita);
- na Telepatia, separa menos (0,72 nos dois).

### Leitura

- **Os dois critérios registrados passam na receita final, nas duas fontes:**
  - o erro dos "não sei" é bem maior que o dos respondidos, e os ICs não se encostam;
  - o AUROC dentro da família passa de 0,7. No Felhen passa por pouco: o IC começa em 0,71.
- **Pelo motivo certo? Sim, no teste que dava para fazer.**
  - O AUROC dentro da família é igual ao geral (0,75 contra 0,75 no Felhen).
  - Se o "não sei" só refletisse a matéria, o AUROC dentro da família cairia para perto de 0,5. Foi o que
    aconteceu com a confiança escrita na H11b, e aqui não acontece.
  - Na Telepatia, a matéria nova é uma família só, então os dois AUROCs são a mesma conta.
- **O treino curto não só faz ele responder mais. Ele aprende a ordenar os casos.**
  - Na Telepatia, o AUROC vai de 0,59 para 0,83.
  - Antes, o "não sei" errava quase o mesmo que o modelo inteiro (43% contra 42%). Com só 7% de respostas, o
    "não sei" é quase tudo, então essa comparação dizia pouco. O AUROC mostra que a separação era fraca, mas
    não nula.
- **O preço:** entre os "não sei", ele acertaria cerca de metade dos casos. Para manter o erro em 10%, a regra
  simples joga fora muita resposta certa.
- **O que este teste não diz:** se o "não sei" aparece porque **falta informação** no caso ou porque **falta
  conhecimento** ao modelo. Isso é a parte 2, o teste limpo da `regra_beneficio`.

### Conferência

O plano pedia que a cobertura e o erro dos respondidos repetissem os números da receita final.

- **Telepatia:** repete os publicados (72% / 0,101) ✅.
- **`h4_log` nas duas fontes:** repete (16% / 0,19 e 7% / 0,18) ✅.
- **Felhen da receita:** deu **35%** / 0,149, mas o publicado é **37%** / 0,15 ⚠️.
  - O próprio `h9_recalibrar.py`, rodado hoje nos mesmos arquivos, dá o mesmo 35,1% / 0,149, e os arquivos são
    de 21/11.
  - Então esta análise repete a recalibração da receita. A diferença está no número publicado, e a origem dele
    ainda está sendo conferida.

### Limites

- Só `dev`, com 3 sementes.
- Só 3 famílias reservadas: 2 do Felhen e 1 da Telepatia.
- O "não sei" aqui é a regra simples (confiança abaixo do limiar), não a opção "não sei" da H5.

<details>
<summary><b>Plano registrado antes de rodar</b>: Plano: o Lumi sabe o que não sabe?</summary>

Registrado antes de rodar (24/11/2026).

#### A pergunta

Na receita final, o Lumi responde 72% das faturas novas com 10% de erro e diz "não sei" no resto. Esse "não sei"
acontece **porque ele sabe que erraria**? Ou só porque a matéria nova inteira é difícil?

#### O que medir

`scripts/analise_sabe.py`, sem GPU, com as predições que já existem:

- **Modelos:** `h4_log` e `h9r_k100` (a receita final), com 3 sementes.
- **Recorte:** famílias reservadas, com a mesma recalibração da receita final:
  - a temperatura e o limiar saem de 25 rótulos da família;
  - os grupos do treino curto ficam fora;
  - os sorteios são os mesmos do `h9_recalibrar.py`.
- **"Não sei"** = confiança abaixo do limiar.
- **Medidas,** com IC 95% por bootstrap de grupos e de sorteios:
  - quanto ele responde e quanto erra entre os respondidos;
  - **quanto erraria entre os "não sei"**, se fosse forçado a responder;
  - a diferença entre esses dois erros;
  - **AUROC** da confiança: a chance de um acerto ter confiança maior que um erro;
  - **AUROC dentro da família:** o mesmo, sem misturar matérias.
- **Referência:** as famílias vistas, com a temperatura e o limiar do pipeline.

#### Leitura (registrada antes)

- **"Sabe que não sabe":**
  - o erro entre os "não sei" é bem maior que o erro entre os respondidos, com os ICs separados;
  - **e** o AUROC dentro da família passa de 0,7.
- **Pelo motivo certo?**
  - Se o AUROC geral for alto e o de dentro da família ficar perto de 0,5, o "não sei" só separa matéria difícil de
    matéria fácil. Foi o que aconteceu com a confiança escrita na H11b.
  - O AUROC dentro da família é o teste limpo.
- **Conferência:** a cobertura e o erro dos respondidos do `h9r_k100` precisam repetir os números publicados:
  - Telepatia: 72% e 10%;
  - Felhen: 37% e 15%.

  Se não repetirem, a recalibração está diferente, e o resultado não vale até isso ser explicado.

#### Fora deste plano

O teste limpo da `regra_beneficio` (`scripts/analise_falta_info_fatores.py`), com o "não sei" pela confiança.
Antes, é preciso ver quais `runs/res2dev_*` existem.

#### Como rodar

```bash
cd /raid/$USER/lumi-decision && git pull -q && sbatch slurm/analise_sabe.sbatch
```

</details>

---

## Onde o Felhen erra, depois da receita final

### Resposta curta

**Na `factck_veracidade`, o Lumi não aprendeu nada: responde "falso" em 95% dos casos.** A acurácia dele ali
(67%) é exatamente a de chutar sempre a classe mais comum. Na `juristcu_area`, ele aprendeu de verdade.

### O que foi medido

- **Script:** `scripts/onde_erra.py`, na receita final (`h9r_k100`), com 3 sementes.
- **Recorte:** as famílias reservadas, sem os grupos usados no treino curto.
- **Medidas:**
  - o acerto em cada classe;
  - a confusão entre o gabarito e a resposta;
  - o acerto médio por classe. Chutar sempre a mesma classe dá 1 / (nº de classes).

### Resultado

**`factck_veracidade`** (3 classes: falso, impreciso e verdadeiro):

| Classe | No gabarito | Nas respostas do Lumi | Acerto da classe |
|---|---:|---:|---:|
| falso | 66,7% | **95,4%** | 0,978 |
| impreciso | 23,5% | 2,5% | 0,056 |
| verdadeiro | 9,8% | 2,1% | 0,033 |

- Acurácia: 0,668. Chutar sempre "falso" dá 0,667.
- Acerto médio por classe: 0,356. Chutar sempre a mesma classe dá 0,333.
- Dos 60 casos "verdadeiro", 54 viram "falso". Esse é o erro grave: ir de uma ponta da escala à outra.

**`juristcu_area`** (10 classes):
- Acurácia: 0,644. Chutar sempre a classe mais comum dá 0,143.
- **Aprendeu de verdade.**
- As classes fracas são `gestao_administrativa` (0,36), `desestatizacao` (0,49) e `financas_publicas` (0,50).

**`invoice_processing`** (Telepatia):
- Acurácia: 0,792.
- Só o `reject`, que é raro (0,7% dos casos), quase nunca sai certo (0,05).

### Leitura

- **Pelo motivo certo? Na `factck`, não.**
  - Os 67% pareciam aprendizado, mas são o chute da classe mais comum.
  - A média do Felhen (65%) junta essa família com a `juristcu`, que aprendeu.
- **Por que pode estar acontecendo:**
  - nos 100 exemplos sorteados do treino curto, cerca de 2 de cada 3 são "falso";
  - a perda log, honesta, aprende essa proporção e não consegue separar as classes.
  - Isso não foi testado.
- **O que muda:**
  - a H14 ganha dois braços com o sorteio equilibrado por classe na `factck`;
  - a H14 e a H15 passam a olhar também o acerto médio por classe.

### Limites

- A `factck` tem 102 grupos no `dev`, com 6 a 10 casos "verdadeiro" por semente.
- O resultado é do `dev`.
