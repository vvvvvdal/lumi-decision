# H7: quantos exemplos de uma família nova bastam para recalibrar?

**Data:** 20/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (commit `a1c859a`)
**Plano registrado antes de rodar:** `analises/plano_h7_poucos_rotulos.md`
**Análise:** job 37699, `scripts/h7_poucos_rotulos.py`. Usa o `h4_log` nas sementes 0, 1 e 2, com 100
sorteios por k. Só CPU, sem treino.

## O que foi testado

Hoje, a temperatura e o limiar do "não sei" são ajustados nas famílias **vistas** e usados nas novas. Aqui,
cada família reservada ganha k exemplos com gabarito (k = 10, 25, 50), e a calibração é reajustada só para
ela. Alvo: no máximo 10% de erro entre as perguntas que o Lumi responde.

## Resultado (reservadas do `dev`; média e faixa de 95% entre sorteios)

| Fonte | Método | k | ECE | Cobertura | Risco entre respondidas |
|---|---|---:|---:|---:|---:|
| Felhen | global (hoje) | — | 0,062 | 37% | 0,29 |
| Felhen | T + limiar da família | 25 | 0,063 | 15% | 0,18 [0,00; 0,36] |
| Felhen | T + limiar da família | 50 | 0,062 | 16% | 0,14 [0,00; 0,31] |
| Felhen | oráculo (família toda) | — | 0,043 | **12%** | 0,10 |
| Telepatia | global (hoje) | — | 0,189 | 44% | 0,37 |
| Telepatia | T da família | 25 | **0,111** | 14% | 0,19 |
| Telepatia | T + limiar da família | 50 | 0,108 | 6% | 0,13 [0,02; 0,25] |
| Telepatia | oráculo (família toda) | — | 0,110 | **2%** | 0,09 |

(A tabela completa, com k = 10 e NLL, está em `analises/h7_poucos_rotulos.md` no cluster.)

## Leitura

**1. Na Telepatia, a temperatura estava errada, e 25 exemplos consertam.** O ECE cai de 0,189 para 0,111,
igual ao oráculo (0,110). A NLL cai de 1,20 para 0,93. Com 10 exemplos já chega quase lá (0,117).

**2. No Felhen, a temperatura já estava boa.** O ECE global (0,062) está perto do oráculo (0,043), e
reajustar com k exemplos não muda nada.

**3. O "não sei" errado tem outra causa: o Lumi sabe pouco da matéria nova.** Mesmo com o ajuste perfeito
(oráculo), ficar abaixo de 10% de erro exige responder só **12%** dos casos no Felhen e **2%** na
Telepatia. O limiar global erra 29–37% porque responde demais (37–44% dos casos), não porque as
probabilidades mentem.

**4. Com 50 exemplos da família, o risco fica perto do alvo** (0,13–0,14 na média), mas a faixa entre
sorteios é larga, e o Lumi passa a responder só 6–16% dos casos.

**Pelo critério registrado:** **parcial**. O ECE recupera tudo na Telepatia já com k = 25. O risco só chega
a ≤ 0,15 com k = 50. E o oráculo mostra que o limite não é de escala (temperatura), e sim de **competência**:
o modelo não separa bem os casos que acerta dos que erra na matéria nova (acurácia de cerca de 0,52 no
Felhen e 0,60 na Telepatia).

## O que isso fecha na história do projeto

- A **honestidade** do Lumi (calibração) se resolve com a recompensa certa (H4, H6) mais uns 25 exemplos
  quando o tipo de caso é novo (H7).
- O que **não** se resolve calibrando é saber a matéria. Na família nova, o Lumi honesto diz "não sei" quase
  sempre, e isso é o comportamento correto.
- O próximo ganho real viria de **acurácia nas famílias novas** (mais dados, modelo maior, raciocínio), não
  de mais calibração.

## Limites

- As famílias reservadas não têm `calib` (Felhen: o split de `train` delas fica sem uso; Telepatia: tudo vai
  para o `dev`). Por isso, os k exemplos saíram do próprio `dev` (k grupos), e a medida usou só os grupos
  restantes. O script faz isso sozinho quando a família não está no `calib`.
- O oráculo é otimista: ajusta e mede no mesmo conjunto.
- Só o `h4_log`. Os outros braços não foram testados aqui.
