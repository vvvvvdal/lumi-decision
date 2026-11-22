# H8: mostrar exemplos da família nova no prompt ensina o Lumi?

**Data:** 20/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commits `7fcef9e` e `ce7ba71`)
**Plano registrado antes de rodar:** `analises/plano_h8_contexto.md`
**Rodadas:** job 37711 (avaliação, cerca de 40 min) e job 37717 (análise). Adaptador `h4_log`, semente 0,
reservadas do `dev`.

## O que foi testado

Antes da pergunta, o prompt ganha K casos já respondidos da mesma família (K = 0, 3, 8), como "exemplos
resolvidos antes da prova". Não há treino.

## Resultado

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

## Leitura

- **Pelo critério registrado: não ajuda.** A acurácia empata nas duas fontes. Mostrar exemplos **não ensina**
  a matéria nova ao Lumi-4B.
- **Efeito inesperado:** os exemplos **calibram**. Na Telepatia, o ECE cai de 0,104 para 0,029 sem
  nenhuma temperatura, mais que os 25 rótulos da H7 conseguiram (0,111). O Lumi vê como são as respostas
  daquela família e ajusta a confiança, mas não passa a acertar mais.
- No Felhen, os textos são longos (cerca de 1100 tokens por caso). Com K = 8, cabem em média 6,2 exemplos no
  teto de 12000 tokens.

## Onde isso deixa o projeto

| Ferramenta | Calibra na família nova? | Ensina (acurácia)? |
|---|---|---|
| Recompensa própria (H4/H6) | sim, nas vistas | não muda |
| 25 rótulos para a temperatura (H7) | sim (Telepatia) | não |
| Exemplos no prompt (H8) | **sim (Telepatia, sem ajuste)** | **não** |

O que falta é competência na matéria nova, e nada que só muda a **leitura** resolve isso. Pelo plano, o
próximo passo é **treinar**: uma LoRA curta com os exemplos da família, partindo do `h4_log`
(`--init-adapter` já existe).

## Limites

- Semente 0, um adaptador (`h4_log`).
- Felhen com K = 8 cortado pelo teto (6,2 exemplos em média).
- A cobertura no risco de 10% depende de poucos casos muito confiantes e oscila.
