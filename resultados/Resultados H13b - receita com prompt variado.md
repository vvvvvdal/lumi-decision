# H13b: a receita final com o prompt variado

## Resposta curta

**A receita não muda: o prompt variado tem um custo no prompt de sempre.** Mas a H13b mostrou uma coisa
importante: **a receita de hoje é frágil a um prompt que nunca viu.**

- **No prompt nunca visto:** a receita de hoje cai de 72% para **50%** de faturas novas respondidas.
- **Com o prompt variado:**
  - o prompt nunca visto melhora (57%);
  - mas o prompt de sempre piora (66% contra 72%).

## O que foi testado

- **Braço `h9ra_k100`:** a receita com o prompt variado nas duas etapas. Parte do `h4_aug`, e o treino curto
  também usa `--variar-prompt true`. São 3 sementes.
- **Comparação:** contra a receita de hoje (`h9r_k100`), com os mesmos exemplos no treino curto.
- **Dois prompts:**
  - o de sempre (`dev`);
  - a variante 4, que nenhum treino viu (`devvar`).
- **Recalibração:** 25 rótulos da família, no próprio prompt avaliado.
- **Plano registrado antes de rodar:** `analises/plano_h13b_receita_aug.md`.

## Resultado

**Receita (25 rótulos, alvo de 10% de erro, média das 3 sementes): responde / erra**

| Fonte | Prompt | Receita de hoje | Com prompt variado |
|---|---|---|---|
| Telepatia | de sempre | 72% / 0,101 | **66%** / 0,102 |
| Telepatia | nunca visto | **50%** / 0,104 | 57% / 0,103 |
| Felhen | de sempre | 35% / 0,149 | 33% / 0,156 |
| Felhen | nunca visto | 31% / 0,156 | 31% / 0,165 |

**Ele ainda responde com uma letra no prompt nunca visto?**

| | Massa nas letras | Casos com massa abaixo de 0,5 |
|---|---|---|
| Receita de hoje, Felhen | 0,38 a 0,53 | 45% a 73% |
| Prompt variado, Felhen | 0,48 a 0,65 | 24% a 55% |
| Receita de hoje, Telepatia | 0,51 a 0,57 | 31% a 49% |
| Prompt variado, Telepatia | 0,59 a 0,72 | 0,5% a 20% |

(Faixa entre as 3 sementes. No prompt de sempre, fica em 0,99 nos dois braços.)

**Acurácia, prompt variado − receita de hoje (IC 95% por bootstrap de grupos):**
- **Prompt de sempre** (semente 0):
  - Felhen reservadas: **−3,2**;
  - Telepatia reservadas: **−1,5**;
  - vistas: empate.
- **Prompt nunca visto:**
  - Telepatia reservadas: **+2,6**, +1,3 (empate) e **+2,5**;
  - Felhen vistas: **+1,4**, **+6,0** e **+4,3**;
  - Felhen reservadas: **−2,1**, +1,0 e +1,7 (empates).

## Leitura

1. **A receita de hoje é frágil?** **Sim.**
   - No prompt nunca visto, a Telepatia cai de 72% para 50% de respostas.
   - Em até 73% dos casos do Felhen, o Lumi já nem quer responder com uma letra.
   - O treino curto não resolveu a fragilidade que a H13 achou no `h4_log`.
2. **O prompt variado piora o prompt de sempre?** **Sim.**
   - A Telepatia responde 6 pontos a menos, e o limite registrado era 2.
   - A acurácia das reservadas cai.
   - Pelo critério, isso já basta: **a receita fica como está**.
3. **E no prompt nunca visto?** **O prompt variado ganha.**
   - A Telepatia responde +7 pontos e acerta mais.
   - As vistas do Felhen sobem.
   - O Lumi volta a responder com uma letra.

**Pelo motivo certo?**
- É uma troca.
- Treinar com 4 textos diferentes ensina o Lumi a não depender do texto exato: ganha no prompt novo.
- Ao mesmo tempo, ele fica um pouco menos afiado no texto de sempre: perde no prompt de sempre.
- Não é um ganho de graça. Por isso o critério escolheu a receita mais simples.

## O que isso muda

- **A receita final não muda,** mas ganha um **limite novo**: ela foi medida só com o prompt de sempre. Com um
  texto diferente, responde bem menos.
- **Ideias para depois (não registradas):**
  - misturar o prompt de sempre na maior parte do treino e as variantes numa parte menor;
  - ou treinar com muito mais variantes, para que nenhuma seja "a de sempre".

## Limites

- Uma variante nunca vista só (a 4), escrita por nós.
- Só `dev`, com 3 sementes. A comparação semente por semente no prompt de sempre só foi impressa para a
  semente 0. A receita tem as 3.
- Na Telepatia, a matéria nova é uma família só.
