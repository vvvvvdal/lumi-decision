# H12b: na perda do Clef, o que faz o quê? O smoothing ou o Brier?

## Resposta curta

**O custo na calibração vem do label smoothing. O ganho de acurácia na Telepatia vem do Brier.**

A perda `log_brier` (entropia cruzada + Brier, **sem** smoothing):
- fica com o ganho da Telepatia: +2,2 pontos nas reservadas, mais que a perda inteira do Clef (+1,8);
- tem só um custo pequeno na calibração.

Pelo critério registrado, ela é a versão da perda do Clef que vale manter. O próximo passo é testá-la dentro
da receita final (H9).

## O que foi testado

Dois braços novos, cada um com uma peça da perda do Clef. Todo o resto é igual ao `h4_log`. São 3 sementes, no
`dev`, e o plano foi registrado antes de rodar em `analises/plano_h12b_separar.md`.

| Braço | O que tem |
|---|---|
| `log` | Entropia cruzada pura |
| `suave` | Só o smoothing (ε = 0,1) |
| `logbrier` | Só o Brier |
| `clef` (H12) | As duas peças |

## Números crus (sem temperatura)

Diferença contra a `log`, IC 95% por bootstrap de grupos (`scripts/comparar_h4.py`). Em negrito, as
diferenças com o IC todo de um lado.

| Recorte | Medida | `suave` | `logbrier` | `clef` |
|---|---|---|---|---|
| Felhen vistas | ECE | **+0,049** | −0,004 (empate) | **+0,036** |
| Felhen vistas | NLL | **+0,046** | **+0,005** | **+0,041** |
| Felhen reservadas | NLL | **+0,021** | **−0,013** (melhor) | **+0,029** |
| Telepatia vistas | acurácia | +0,2 (empate) | +0,4 (empate) | **+1,0** |
| Telepatia reservadas | acurácia | **+0,9** | **+2,2** [+1,7; +2,6] | **+1,8** |
| Telepatia reservadas | ECE | **+0,008** | **+0,014** | **+0,014** |
| Telepatia reservadas | NLL | +0,004 (empate) | **−0,006** (melhor) | 0,000 (empate) |

## Depois da temperatura

`scripts/resumo_calibracao.py`, média das 3 sementes.

| Braço | T no Felhen | ECE Felhen vistas (depois) | ECE Felhen reservadas (depois) | ECE Telepatia reservadas (depois) |
|---|---:|---:|---:|---:|
| `log` | 1,06 | 0,014 | 0,062 | 0,189 |
| `suave` | **0,81** | 0,028 | **0,106** | 0,196 |
| `logbrier` | 1,05 | 0,013 | **0,053** | **0,176** |
| `clef` | **0,83** | 0,028 | **0,101** | 0,183 |

## Leitura

- **Custo da calibração: vem do smoothing.**
  - O `suave` repete o padrão do `clef`:
    - as vistas do Felhen pioram (ECE +0,049);
    - a temperatura fica abaixo de 1 (0,81), sinal de modelo pouco confiante;
    - depois da temperatura, as reservadas do Felhen pioram (0,106 contra 0,062).
  - O `logbrier` não repete esse padrão. Ele tem T = 1,05, como a `log`, e, depois da temperatura, fica até um
    pouco melhor que ela.
  - O que esperávamos antes de rodar aconteceu.
- **Ganho de acurácia da Telepatia: vem do Brier.**
  - O `logbrier` repete o ganho e passa dele: +2,2 contra +1,8 do `clef`.
  - O `suave` sozinho dá só +0,9.
  - O smoothing até tira um pouco do ganho quando entra junto.
- **O custo que sobra no `logbrier`:**
  - ECE +0,014 nas reservadas da Telepatia, cru;
  - NLL +0,005 nas vistas do Felhen, cerca de 1/9 do custo do `clef`.
  - Depois da temperatura, os dois somem: o `logbrier` fica igual à `log` ou melhor.
- **Pelo motivo certo? Não sabemos ainda.**
  - Uma explicação possível: o Brier é limitado (no máximo 2), e a perda log não é. Com o Brier somado, os
    exemplos muito errados puxam menos o treino, e isso ajudaria numa fonte com poucos grupos vistos.
  - Não testamos isso.
  - O ganho só aparece nas reservadas da Telepatia, que são uma família só. No Felhen, a acurácia empata (+0,6).

## O que isso muda

- **A receita final ainda não muda.**
  - O ganho foi medido no `h4_log`, antes do treino curto na matéria nova.
  - Na receita, a H9 já dá +20 pontos na Telepatia, e os +2,2 podem sumir.
- **Próximo passo:** rodar a receita com o `logbrier` no lugar da `log`:
  - um treino curto (H9r) a partir do `h4_logbrier`;
  - e a recalibração com 25 rótulos.

  Se ele responder mais casos com o mesmo erro, ele entra na receita.

## Limites

- Só `dev`, com 3 sementes.
- Um ε só (0,1) e peso 1 no Brier, sem busca.
- A matéria nova da Telepatia é uma família só.
