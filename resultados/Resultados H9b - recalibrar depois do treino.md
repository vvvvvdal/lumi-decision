# H9b: recalibrar depois do treino curto devolve a honestidade?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `5807ba6`)
**Plano registrado antes de rodar:** `analises/plano_h9b_recal.md`
**Análise:** job 37752, `scripts/h9_recalibrar.py`. Usa o `h4_log` e o `h9_k100`, sementes 0, 1 e 2, com 100
sorteios por k. Só CPU.

## O que foi testado

A H9 ensinou, mas deixou o Felhen novo superconfiante. Aqui aplicamos a receita da H7 aos dois braços: k grupos
rotulados da família ajustam a temperatura T e o limiar do "não sei". Os grupos do treino curto saem da conta.
Os grupos de calibração são sorteados do resto e também ficam fora da medida.

## Resultado (reservadas do `dev`, média das 3 sementes e dos sorteios)

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

## Leitura

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

## A receita completa

1. **Recompensa honesta** (log, H4): o Lumi fica calibrado nas famílias que conhece.
2. **Treino curto com cerca de 100 exemplos da família nova** (H9): ele passa a saber a matéria.
3. **Recalibração com cerca de 25 rótulos** (H7/H9b): ele volta a ser honesto e sabe quando dizer "não sei".

## Limites

- A faixa entre sorteios é larga no Felhen: com k = 25, a cobertura vai de 6% a 66% e o erro de 0,05 a 0,27.
  Com 25 rótulos, o limiar ainda é instável. k = 50 estreita a faixa.
- Ainda há um esquecimento leve nas vistas da Telepatia (H9), que é o próximo teste (replay).
