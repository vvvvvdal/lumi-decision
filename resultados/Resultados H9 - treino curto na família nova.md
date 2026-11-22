# H9: treinar um pouco com exemplos da família nova ensina o Lumi?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `a93f842`)
**Plano registrado antes de rodar:** `analises/plano_h9_adapt.md`
**Rodadas:** jobs 37727 (k = 25) e 37728 (k = 100); análise no job 37738 (`scripts/analise_h9.py`). Semente 0.
Depois, k = 100 nas sementes 1 e 2 (análises nos jobs 37750 e 37751).

## O que foi testado

Partindo do Lumi já treinado (`h4_log`), mais 3 passadas de treino **só com k grupos rotulados** das três
famílias reservadas:
- Felhen: área jurídica (`juristcu_area`) e veracidade (`factck_veracidade`);
- Telepatia: faturas (`invoice_processing`).

A perda é a log (honesta). Os grupos usados no treino saem da avaliação nos dois lados. A Telepatia perde
25 ou 100 grupos do `dev`; o Felhen não perde nenhum, porque os exemplos vieram do `train`.

## Resultado

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

## Leitura

**Pelo critério registrado: ensina.** Com k = 100, a acurácia das reservadas sobe com o IC acima de zero
**nas duas fontes**, e as vistas ficam em empate. Com k = 25 também sobe nas duas.

- É o **primeiro resultado do projeto que aumenta a acurácia na matéria nova**. Ler o Lumi de outro jeito
  (H7, H8) só deixava ele mais honesto. Treinar com poucos exemplos ensina.
- O efeito aparece onde importa. Na Telepatia, o Lumi passa de responder com segurança **2,5%** dos casos
  para **71%**. Ele continua honesto (ECE melhor) e agora sabe mais.
- **Atenção no Felhen com k = 100:** o ECE sobe de 0,078 para 0,122. O IC inclui zero, mas o valor passa do
  limite de 0,02 registrado. A acurácia sobe tanto que o Brier e a NLL melhoram. Mesmo assim, vale
  recalibrar a temperatura (H7) depois do treino curto.

## Confirmação com 3 sementes (k = 100)

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

## A história do projeto até aqui

1. **Recompensa certa → Lumi honesto** (H4; H6 mostra que funciona até sem gabarito).
2. **Na matéria nova, ele é honesto mas sabe pouco** (H7): para errar no máximo 10%, responde quase nada.
3. **Ler diferente não ensina** (H7, H8); só calibra.
4. **Treinar um pouco com 25–100 exemplos da matéria nova ensina** (H9), sem esquecer a antiga, e a
   honestidade se mantém na Telepatia.

## Limites

- k = 100 confirmado em 3 sementes. k = 25 só tem a semente 0.
- Uma LoRA para as 3 famílias juntas. Os números por família estão em `analises/h9_adapt.md` no cluster.
- O oráculo é otimista: ajusta e mede na família inteira.
- Na Telepatia, com k = 100, a avaliação fica com 200 grupos, em vez de 275 com k = 25 e 300 originais.
