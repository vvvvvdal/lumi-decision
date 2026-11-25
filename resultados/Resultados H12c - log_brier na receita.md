# H12c: a perda `log_brier` dentro da receita final

## Resposta curta

**Empate. A receita continua com a perda log.** Antes do treino curto, a `log_brier` acertava +2,2 pontos na
Telepatia nova (H12b). Depois da receita inteira, esse ganho some: os dois braços respondem a mesma coisa, com o
mesmo erro.

## O que foi testado

- **Braço `h9rb_k100`:** a receita final inteira com a `log_brier` (`--suavizar 0`):
  - parte do `h4_logbrier`;
  - treino curto igual ao `h9r_k100`: 100 exemplos por família nova + replay, 3 épocas;
  - a mesma recalibração, com 25 rótulos.
- **Comparação:** contra o `h9r_k100` da mesma semente, com os mesmos grupos no treino curto. São 3 sementes.
- **Plano registrado antes de rodar:** `analises/plano_h12c_logbrier_receita.md`.

## Resultado

**Receita (25 rótulos, alvo de 10% de erro, média das 3 sementes):**

| Fonte | Receita de hoje (`log`): responde / erra | Com `log_brier`: responde / erra |
|---|---|---|
| Telepatia | 72% / 0,101 | 72% / 0,100 |
| Felhen | 35% / 0,149 | 35% / 0,153 |

**Acurácia, `log_brier` − `log`, por semente** (IC 95% por bootstrap de grupos):

| Recorte | Semente 0 | Semente 1 | Semente 2 |
|---|---|---|---|
| Telepatia reservadas | +0,5 (empate) | +0,1 (empate) | +0,4 (empate) |
| Felhen reservadas | −1,4 (empate) | −0,9 (empate) | −0,1 (empate) |
| Telepatia vistas | +1,2 (empate) | +0,6 (empate) | **+1,4** |
| Felhen vistas | +0,1 (empate) | +0,1 (empate) | −0,3 (empate) |

**Outras medidas:**
- **Calibração das vistas depois da temperatura:** ECE 0,017 nos dois braços no Felhen; na Telepatia, 0,034 na
  `log` contra 0,040 na `log_brier`.
- **"Sabe que não sabe":** o AUROC dentro da família é 0,75 contra 0,76 no Felhen e 0,83 nos dois na Telepatia.

## Leitura

- **Critério registrado: empate.** A cobertura e a acurácia ficam dentro do ruído. A `log` continua, porque é a
  mais simples.
- **Pelo motivo certo?**
  - O ganho da H12b era em matéria que o modelo **nunca tinha visto**.
  - Depois que o treino curto ensina essa matéria (+20 pontos na Telepatia), o jeito de pontuar o treino deixa de
    fazer diferença.
  - Era o risco registrado na H12b, e foi o que aconteceu.
- **O que se aprende:** uma melhora pequena medida antes do treino curto não garante melhora na receita. Toda peça
  nova precisa ser testada dentro da receita.

## Limites

- Só `dev`, com 3 sementes.
- Na Telepatia, a matéria nova é uma família só.
