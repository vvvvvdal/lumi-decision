# Plano H9r: replay das famílias vistas no treino curto

Registrado antes de rodar (21/11/2026).

## Pergunta

A H9 (k = 100) ensina as reservadas, mas a acurácia nas **vistas da Telepatia** cai de 1,5 a 3,0 pontos (na
semente 2, o IC fica abaixo de zero). Misturar casos já vistos no treino curto evita esse esquecimento sem
perder o ganho nas novas?

## Desenho

- Igual à H9 (`--adapt-k 100`, 3 épocas, a partir do `h4_log` da mesma semente), mais `--adapt-replay 1.0`:
  entra um caso visto para cada caso novo.
- Os vistos vêm do conjunto de treino original (`build_training_set`, split `train`, famílias vistas) em
  rodízio entre as famílias. Cada família vista entra por igual, inclusive as da Telepatia.
- Os grupos das reservadas sorteados são os mesmos da H9 (mesmo `data_seed`), então a diferença é só o replay.
- O `adapt_grupos.json` registra só as reservadas, porque os vistos são do `train` e não tocam o `dev`.
- Sementes 0, 1 e 2: `h9r_k100`, `h9r_k100_s1`, `h9r_k100_s2`.

## Análise

`scripts/analise_h9.py`, com cada semente contra o seu `h4_log` e contra a sua `h9_k100`.

## Leitura (média das 3 sementes)

- **Replay resolve:** nas vistas da Telepatia, a acurácia fica a no máximo 1 ponto do `h4_log`, e nenhuma
  semente tem o IC todo abaixo de zero. Nas reservadas, o ganho fica a no máximo 3 pontos do ganho da H9 nas
  duas fontes.
- **Troca:** o esquecimento some, mas o ganho nas reservadas cai mais que 3 pontos.
- **Não resolve:** o esquecimento nas vistas da Telepatia continua igual ao da H9.
