# Plano H14: os botões do treino curto, para subir o Felhen

Registrado antes de rodar (25/11/2026).

## Pergunta

Na receita final, o Felhen novo acerta 65% e responde só 35% dos casos (a Telepatia responde 72%). Os botões do
treino curto (k, épocas, `lr`) nunca foram ajustados. Mexer neles faz o Felhen responder mais sem errar mais?

## O que a etapa 0 mostrou (26/11, antes de rodar a triagem)

`analises/onde_erra.md`, na receita (`h9r_k100`, 3 sementes):
- **`factck_veracidade`:** o Lumi responde "falso" em **95%** dos casos.
  - A acurácia (0,668) é igual à de chutar sempre a classe mais comum (0,667).
  - Acerta 6% dos "impreciso" e 3% dos "verdadeiro".
  - **Na `factck`, ele não aprendeu nada além da classe mais comum.**
- **`juristcu_area`:** aprendeu de verdade (0,64, contra 0,14 de chutar a classe mais comum). As classes mais
  fracas são `gestao_administrativa` (0,36), `desestatizacao` e `financas_publicas` (cerca de 0,5).
- **Pool do treino curto (`train` das reservadas):** 927 grupos na `factck` e 3.240 na `juristcu`. Então k = 300
  cabe nas duas.

Por isso entram dois braços com **sorteio equilibrado por classe** na `factck`
(`--adapt-balancear felhen:factck_veracidade`): um rodízio entre as classes, que acaba sendo a classe rara.

## Desenho

- **O que muda:** só a 2ª etapa da receita, o treino curto a partir do `h4_log`, com replay e a mesma
  recalibração (25 rótulos).
- **Código novo:** `--adapt-familias` aceita `fonte:família=k`, para um k só daquela família.
  - O Felhen tira os exemplos do `train` das reservadas, que nunca foi usado, então subir o k não encolhe a
    avaliação.
  - A Telepatia fica em k = 100, porque os exemplos dela saem do `dev`.
- **Triagem, semente 0:**

| Braço | k do Felhen | Épocas | `lr` |
|---|---|---|---|
| `h9r_k100` (já existe) | 100 | 3 | 1e-4 |
| `h14_k300` | 300, ou o pool inteiro se for menor | 3 | 1e-4 |
| `h14_ep6` | 100 | 6 | 1e-4 |
| `h14_lr2` | 100 | 3 | 2e-4 |
| `h14_lr05` | 100 | 3 | 5e-5 |
| `h14_k300_ep6` | 300 | 6 | 1e-4 |
| `h14_bal` | 100, equilibrado por classe na `factck` | 3 | 1e-4 |
| `h14_k300_bal` | 300, equilibrado por classe na `factck` | 3 | 1e-4 |

Análise encadeada: `slurm/analise_h14.sbatch`. Os grupos de treino do `h9r_k100` também saem da medida
(`--excluir-tambem h9r_k100`), porque os braços equilibrados sorteiam outros grupos.

## Critério

Análise: `h9_recalibrar.py --base h9r_k100 --novo <braço> --sementes 0` e `analise_h9.py --base h9r_k100`.

- **Um braço passa na triagem** se:
  - o Felhen responder pelo menos 5 pontos a mais;
  - o erro dos respondidos ficar a no máximo 1 ponto;
  - não perder mais de 1 ponto nas vistas nem na cobertura da Telepatia.
- **O melhor braço que passar** roda nas sementes 1 e 2, e vale a média das 3.
- **Se nenhum passar,** a 2ª etapa está no limite. O próximo botão é a 1ª etapa (posto 64 e `lr` no `h4_log`),
  que é mais cara.
- **Critério novo, para a `factck` (`onde_erra.py`):**
  - o acerto médio por classe tem que passar de 0,40. Hoje é cerca de 0,36, e chutar sempre a mesma classe dá
    0,33;
  - sem a acurácia cair mais que 3 pontos.

  Um braço que só melhore a `factck` assim também conta como sinal, mesmo sem os 5 pontos de cobertura.
- **Pelo motivo certo?**
  - Conferir no `onde_erra.py` se o ganho vem das duas famílias, ou só de uma.
  - Conferir também se o ganho vem de acertar as classes raras, ou só de chutar mais a classe comum.

## Limites

- A triagem tem uma semente só. Por isso a régua é de 5 pontos.
- Na `factck` (102 grupos no `dev`), a variação entre sementes é grande.
