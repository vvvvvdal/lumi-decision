# H9a: escolher os 25 exemplos onde o Lumi está em dúvida ensina mais que sortear?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `895ef22`)
**Plano registrado antes de rodar:** `analises/plano_h9a_duvida.md`
**Rodadas (semente 0):**
- job 37767: avaliação do pool do Felhen (`runs/h4_log_felhen_trainres`, 400 casos por família);
- job 37768: `h9r_k25`, 25 grupos sorteados + replay 1:1;
- job 37769: `h9ar_k25`, 25 grupos escolhidos por dúvida + replay 1:1;
- análise: job 37778, com os grupos de treino dos **dois** braços fora da avaliação.

## Os dois braços contra o `h4_log`

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

## Comparação direta: dúvida contra sorteio

| Recorte | Métrica | Δ (dúvida − sorteio) | IC 95% | Leitura |
|---|---|---:|---|---|
| Felhen reservadas | acc | +1,9 | [−0,7; +4,5] | empate |
| Felhen reservadas | NLL | −0,148 | [−0,216; −0,084] | dúvida melhor |
| Felhen reservadas | Brier | −0,033 | [−0,057; −0,007] | dúvida melhor |
| Telepatia reservadas | acc | −1,9 | [−4,2; +0,3] | empate |
| Telepatia reservadas | NLL | +0,044 | [+0,007; +0,079] | dúvida pior |
| Telepatia vistas | acc | −1,5 | [−3,6; +0,7] | empate |

## Receita completa com 25 rótulos (T e limiar da família), alvo de 10% de erro

| Fonte | Sorteio: responde / erra | Dúvida: responde / erra | Replay k = 100 (média de 3 sementes) |
|---|---|---|---|
| Felhen | 29% / 0,16 | 29% / 0,16 | 37% / 0,15 |
| Telepatia | 50% / 0,11 | 47% / 0,11 | 72% / 0,10 |

Nas famílias vistas, a temperatura global volta a resolver a superconfiança do replay no Felhen: o ECE cai de
0,056 para 0,019 no sorteio e de 0,054 para 0,018 na dúvida, com T = 1,52 nos dois.

## Leitura

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

## Limites

- Uma semente só.
- Na Telepatia, o pool da dúvida é o próprio `dev`, então a seleção tira da avaliação os casos difíceis. Como
  esses grupos saem dos dois braços, a comparação continua pareada, mas o `dev` restante fica mais fácil.
- O pool do Felhen foi limitado a 400 casos por família, e a faixa foi fixada em 3 × k. Não testei outras faixas.
- Na Telepatia, a temperatura das vistas (T = 0,52) piora as reservadas (ECE 0,045 → 0,116). Com 25 rótulos, a
  temperatura da família corrige isso.

## Conclusão

A receita final não muda: **sortear os exemplos, usar replay e recalibrar**. Escolher por dúvida não traz ganho
claro e tem um custo extra. Quantos exemplos usar continua sendo o que mais pesa: 100 exemplos ensinam bem mais
que 25.
