# Plano H10: o Lumi aprende a matéria nova só com "acertou / errou"?

Registrado antes de rodar (21/11/2026).

## Pergunta

A receita da H9r (treino curto com k = 100 + replay) ensina a matéria nova, mas é supervisionada: o Lumi vê o
gabarito de cada caso. Se ele vir **só o resultado da opção que sorteou** (feedback de bandit, como na H6),
ainda aprende a matéria nova? E a recompensa de acerto (estilo RLVR) volta a deixar ele superconfiante nela?

## Desenho

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

## Leitura

- **H10a, RL ensina a matéria nova:** o `h10_bandit_k100` sobe a acurácia das reservadas contra o `h4_log`,
  com IC acima de zero em pelo menos uma fonte. Além disso, fica com pelo menos metade do ganho do `h9r_k100`
  na mesma semente (Felhen +13,7, Telepatia +17,7).
- **H10b, RLVR estraga a honestidade também na matéria nova:** o ECE das reservadas do `h10_acerto_k100` fica
  acima do ECE do `h10_bandit_k100`, com IC acima de zero.
- **Sem esquecer:** nas vistas da Telepatia, a acurácia fica no máximo 1 ponto abaixo do `h4_log`.
- **Receita completa:** com `scripts/h9_recalibrar.py` e 25 rótulos, medir quanto o `h10_bandit_k100` responde
  com 10% de erro e comparar com o `h9r_k100`.
- Se a H10a se confirmar, rodar as sementes 1 e 2, a partir do `h4_log_s1` e do `h4_log_s2`.

## Análise

- `scripts/analise_h9.py --base h4_log --novos h9r_k100,h10_bandit_k100,h10_acerto_k100`
- `scripts/analise_h9.py --base h9r_k100 --novos h10_bandit_k100,h10_acerto_k100` (o custo de não ver o
  gabarito)
- `scripts/h9_recalibrar.py --base h4_log --novo h10_bandit_k100 --sementes 0 --ks 25`
