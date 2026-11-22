# H6: RL só com o resultado da escolha (feedback de bandit), semente 0

**Data:** 20/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `9100746`)
**Plano registrado antes de rodar:** `analises/plano_h6_bandit.md`
**Análise:** job 37679, `scripts/comparar_h4.py`. Tudo no `dev`, semente 0, IC 95% por bootstrap.

## O que foi testado

O Lumi sorteia 8 respostas por caso e só fica sabendo se cada uma acertou. Ele nunca vê o gabarito.

| Braço | Recompensa |
|---|---|
| `bandit_acerto` | 1 se acertou (estilo RLVR) |
| `bandit_log` | 1/π(a) se acertou, cortado em 10: em média, o gradiente do log score |

Os dois são comparados com o `h4_log` (log score **com** gabarito completo). O `acerto` com gabarito completo
entra como referência.

## ECE (menor é melhor)

| Fonte | Grupo | log (gabarito) | **bandit_log** | **bandit_acerto** | acerto (gabarito) |
|---|---|---:|---:|---:|---:|
| Felhen | vistas | 0,016 | 0,063 | 0,182 | 0,159 |
| Felhen | reservadas | 0,078 | **0,180** | **0,457** | 0,483 |
| Telepatia | vistas | 0,153 | 0,164 | 0,307 | 0,289 |
| Telepatia | reservadas | 0,105 | **0,143** | **0,372** | 0,425 |

## Acurácia (maior é melhor)

| Fonte | Grupo | log (gabarito) | bandit_log | bandit_acerto |
|---|---|---:|---:|---:|
| Felhen | vistas | 0,869 | 0,864 (empate) | 0,815 (pior, −5,5) |
| Felhen | reservadas | 0,515 | 0,512 (empate) | 0,515 (empate) |
| Telepatia | vistas | 0,725 | 0,727 (empate) | 0,675 (pior, −5,0) |
| Telepatia | reservadas | 0,606 | **0,633 (melhor, +2,7)** | 0,602 (empate) |

## Leitura

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

## Teste do corte do peso (20/11, commit `330f4fd`, `--bandit-clip`)

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

## Teste de mais sorteios por caso (20/11, commit `357b611`, `--bandit-amostras 32`)

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

## G = 32 com 3 sementes (20/11, job 37691)

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

## H6c: mais passos (plano registrado em 20/11, antes de rodar)

- **Runs:** `h4_log_e2` e `h4_bandit_log_c1000_g32_e2`, ambos com `--epochs 2`, semente 0.
- **Comparação justa:** bandit e2 contra log e2 (os dois com o mesmo número de passos). Também log e2 contra
  log e1, para separar o efeito de "treinar mais".
- **Ajuda:** a diferença bandit − log no Felhen reservado cai abaixo da obtida com 1 época na mesma semente
  (+0,021) e fica ≤ 0,02 também nas outras células.
- **Não ajuda:** a diferença fica parecida. Então o RL sem gabarito tem um custo real de calibração.
- Se ajudar, replicar nas sementes 1 e 2 antes de concluir.

### Resultado H6c (20/11, semente 0)

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

## H6d: exploração eps = 0,2 (20/11, commit `7a985a3`, job 37696, semente 0)

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

## Fechamento da H6

Foram testados corte, mais sorteios, mais passos e exploração. Só o **G = 32** mudou algo de forma clara
(na semente 0), e com 3 sementes ainda sobra um custo de cerca de 0,05 de ECE no Felhen reservado. A
conclusão estável é:
- o RL **sem gabarito**, com recompensa própria, mantém a acurácia e fica perto da calibração supervisionada;
- recompensar só o acerto deixa o Lumi arrogante.

## Limites

- O G = 8, os testes de corte e a exploração têm só a semente 0.
- Mesmo `lr` e mesmo número de passos do supervisionado.

## Próximos passos possíveis

1. ~~Variar o corte~~ (feito: efeito pequeno). ~~G = 32~~ (feito: ajuda, mas não fecha a diferença no Felhen
   reservado com 3 sementes).
2. Mais passos de treino ou G = 64, para ver se a diferença restante do Felhen cai.
3. Combinar com a H5: abstenção aprendida com feedback de bandit.
