# H9r e H9m: como evitar que o treino curto faça o Lumi esquecer?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commits `42fa52c` replay, `c283255` mistura)
**Planos registrados antes de rodar:** `analises/plano_h9r_replay.md` e `analises/plano_h9m_mistura.md`
**Rodadas:**
- replay: jobs 37753–37755, sementes 0, 1 e 2;
- mistura: jobs 37756–37757, semente 0;
- análise: job 37763 e seguintes (`scripts/analise_h9.py`).

## O problema

Na H9, a acurácia nas **vistas da Telepatia** caía de 1,5 a 3,0 pontos (−2,1 na média). Duas correções:
- **Replay (H9r):** repete o treino curto, mas junta um caso já visto para cada caso novo, em rodízio entre
  as famílias vistas.
- **Mistura (H9m):** não treina de novo. O adaptador fica entre o `h4_log` e a H9 com peso w:
  ΔW = (1 − w)·ΔW_h4 + w·ΔW_h9. A conta é exata e foi conferida no cluster (erro de cerca de 1e-7).

## Replay (3 sementes), diferença para o `h4_log` da mesma semente

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

## Mistura (semente 0), diferença para o `h4_log`

| Recorte | Métrica | w = 0,5 | w = 0,75 | w = 1 (H9) | Replay |
|---|---|---:|---:|---:|---:|
| Telepatia vistas | acc | **−0,2** | −2,0 | −1,9 | −0,5 |
| Felhen reservadas | acc | +8,7 | +12,2 | +13,6 | +13,7 |
| Telepatia reservadas | acc | +14,2 | +15,7 | +16,6 | +17,7 |
| Felhen reservadas | ECE | **−0,001** | +0,015 | +0,044 | +0,085 |
| Felhen vistas | ECE | +0,004 | +0,010 | +0,012 | +0,057 |
| Cobertura oráculo Telepatia | | 57% | 66% | 71% | 74% |

## Leitura

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

## Limites

- Mistura só na semente 0, com w escolhido no próprio `dev` (otimista).
- As vistas da Telepatia têm 84 grupos, então as diferenças de 1–2 pontos ficam perto do ruído.
- O replay usou proporção 1:1 e rodízio igual por família. Não testei proporções diferentes.

## Recalibrar o replay (job 37766, commit `2fc1c2d`)

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

## Conclusão

**Configuração final: replay + recalibração.**
- Nas famílias **vistas**, o Lumi não esquece (Telepatia −0,8 ponto) e, com a temperatura que o pipeline já
  ajusta, continua honesto (ECE 0,017 no Felhen e 0,034 na Telepatia).
- Nas famílias **novas**, ele sabe mais (+12 pontos no Felhen, +21 na Telepatia) e é o braço que mais responde
  dentro do alvo: 72% das faturas novas com 10% de erro.

A mistura w = 0,5 é a alternativa sem treino novo, mas responde menos (56% na Telepatia).

## Receita final do projeto

1. **Recompensa honesta** (perda log, H4).
2. **Treino curto com cerca de 100 exemplos da família nova + o mesmo número de casos já vistos** (H9 + replay).
3. **Temperatura das vistas + limiar da família com cerca de 25 rótulos** (H7/H9b).
