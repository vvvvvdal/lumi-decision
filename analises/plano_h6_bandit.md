# H6: RL só com o resultado da escolha (feedback de bandit)

Registrado antes de rodar, em 20/11/2026. Branch `h6-bandit`, a partir da `h5-abster`.

## Pergunta

Na H4, o Lumi via o gabarito completo de cada caso (aprendizado supervisionado). Um agente de decisão em
produção não vê isso: ele escolhe uma opção e só fica sabendo **se aquela escolha deu certo**. A H6 pergunta:

> Treinando o Lumi só com o resultado da opção que ele sorteou, ele ainda aprende a decidir e fica calibrado?
> E a forma da recompensa importa como na H4?

## Braços (mesma receita do E3/`h4_log`: dados, LoRA r=16, lr 1e-4, 1 época; só muda a perda)

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

## Hipóteses

- **H6a:** o `h4_bandit_acerto` fica superconfiante, como o `h4_acerto`: ECE nas reservadas maior que a do
  `h4_log`, com IC acima de zero.
- **H6b:** o `h4_bandit_log` fica calibrado: ECE nas reservadas no máximo 0,02 acima da do `h4_log`.
- **Custo da informação:** medir quanto de acurácia o `h4_bandit_log` perde em relação ao `h4_log`. É
  medição, não hipótese.

Tudo no `dev`, comparado com `scripts/comparar_h4.py` (IC 95% por bootstrap). Semente 0 primeiro; 1 e 2 só se
as hipóteses parecerem se confirmar.

## Riscos

- **Variância alta do peso $1/\pi$** quando o modelo dá pouca probabilidade à opção certa. O corte em 10
  limita isso, mas introduz um viés pequeno.
- **Sinal mais fraco:** com o mesmo número de passos, o bandit pode aprender menos. Se a perda não cair no
  teste curto, avaliar mais passos antes de concluir.
