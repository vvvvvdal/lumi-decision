# Onde o Felhen erra, depois da receita final

## Resposta curta

**Na `factck_veracidade`, o Lumi não aprendeu nada: responde "falso" em 95% dos casos.** A acurácia dele ali
(67%) é exatamente a de chutar sempre a classe mais comum. Na `juristcu_area`, ele aprendeu de verdade.

## O que foi medido

- **Script:** `scripts/onde_erra.py`, na receita final (`h9r_k100`), com 3 sementes.
- **Recorte:** as famílias reservadas, sem os grupos usados no treino curto.
- **Medidas:**
  - o acerto em cada classe;
  - a confusão entre o gabarito e a resposta;
  - o acerto médio por classe. Chutar sempre a mesma classe dá 1 / (nº de classes).

## Resultado

**`factck_veracidade`** (3 classes: falso, impreciso e verdadeiro):

| Classe | No gabarito | Nas respostas do Lumi | Acerto da classe |
|---|---:|---:|---:|
| falso | 66,7% | **95,4%** | 0,978 |
| impreciso | 23,5% | 2,5% | 0,056 |
| verdadeiro | 9,8% | 2,1% | 0,033 |

- Acurácia: 0,668. Chutar sempre "falso" dá 0,667.
- Acerto médio por classe: 0,356. Chutar sempre a mesma classe dá 0,333.
- Dos 60 casos "verdadeiro", 54 viram "falso". Esse é o erro grave: ir de uma ponta da escala à outra.

**`juristcu_area`** (10 classes):
- Acurácia: 0,644. Chutar sempre a classe mais comum dá 0,143.
- **Aprendeu de verdade.**
- As classes fracas são `gestao_administrativa` (0,36), `desestatizacao` (0,49) e `financas_publicas` (0,50).

**`invoice_processing`** (Telepatia):
- Acurácia: 0,792.
- Só o `reject`, que é raro (0,7% dos casos), quase nunca sai certo (0,05).

## Leitura

- **Pelo motivo certo? Na `factck`, não.**
  - Os 67% pareciam aprendizado, mas são o chute da classe mais comum.
  - A média do Felhen (65%) junta essa família com a `juristcu`, que aprendeu.
- **Por que pode estar acontecendo:**
  - nos 100 exemplos sorteados do treino curto, cerca de 2 de cada 3 são "falso";
  - a perda log, honesta, aprende essa proporção e não consegue separar as classes.
  - Isso não foi testado.
- **O que muda:**
  - a H14 ganha dois braços com o sorteio equilibrado por classe na `factck`;
  - a H14 e a H15 passam a olhar também o acerto médio por classe.

## Limites

- A `factck` tem 102 grupos no `dev`, com 6 a 10 casos "verdadeiro" por semente.
- O resultado é do `dev`.
