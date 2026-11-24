# O Lumi sabe o que não sabe?

## Resposta curta

**Sim, depois da receita final.** Na matéria nova, quando ele diz "não sei", erraria 45% a 47% daqueles casos.
Quando responde, erra 10% a 16%.

A confiança separa acerto de erro **dentro da mesma família** (AUROC 0,75 a 0,83). Então o "não sei" não está só
separando matéria difícil de matéria fácil.

Antes da receita, o mesmo teste dá bem menos: AUROC de 0,59 na Telepatia e de 0,69 no Felhen.

## O que foi medido

- **Script:** `scripts/analise_sabe.py`, sem GPU, com as predições que já existiam. O plano foi registrado
  antes de rodar em `analises/plano_sabe_que_nao_sabe.md`.
- **Modelos:** o `h4_log` e o `h9r_k100` (a receita final), com 3 sementes.
- **Matéria nova:** a mesma recalibração da receita final:
  - a temperatura e o limiar saem de 25 rótulos da família;
  - risco-alvo de 10%;
  - os mesmos sorteios do `h9_recalibrar.py`, com 100 repetições;
  - os grupos do treino curto e os 25 rótulos ficam fora da medida.
- **"Não sei"** = confiança abaixo do limiar. O erro dos "não sei" é o que ele erraria se fosse forçado a
  responder.
- **AUROC:** a chance de um acerto ter confiança maior que um erro. 0,5 é o acaso, e 1 é a separação perfeita.
- **AUROC dentro da família:** a mesma conta, sem misturar matérias diferentes. É o teste limpo.
- **IC 95%:** por bootstrap de grupos e de repetições.

## Resultado

**Matéria nova (famílias reservadas):**

| Fonte | Modelo | Responde | Erro quando responde | Erro quando diz "não sei" | AUROC dentro da família |
|---|---|---:|---|---|---|
| Telepatia | `h4_log` | 7% | 19% [3; 32] | 43% [39; 48] | 0,59 [0,56; 0,62] |
| Telepatia | **receita final** | **72%** | **10%** [6; 15] | **47%** [39; 53] | **0,83** [0,80; 0,85] |
| Felhen | `h4_log` | 16% | 20% [5; 33] | 53% [47; 59] | 0,69 [0,64; 0,73] |
| Felhen | **receita final** | **35%** | **16%** [7; 26] | **45%** [38; 53] | **0,75** [0,71; 0,79] |

**Matéria vista, como referência:**
- no Felhen, a confiança separa bem (AUROC dentro da família de 0,85 no `h4_log` e de 0,82 na receita);
- na Telepatia, separa menos (0,72 nos dois).

## Leitura

- **Os dois critérios registrados passam na receita final, nas duas fontes:**
  - o erro dos "não sei" é bem maior que o dos respondidos, e os ICs não se encostam;
  - o AUROC dentro da família passa de 0,7. No Felhen passa por pouco: o IC começa em 0,71.
- **Pelo motivo certo? Sim, no teste que dava para fazer.**
  - O AUROC dentro da família é igual ao geral (0,75 contra 0,75 no Felhen).
  - Se o "não sei" só refletisse a matéria, o AUROC dentro da família cairia para perto de 0,5. Foi o que
    aconteceu com a confiança escrita na H11b, e aqui não acontece.
  - Na Telepatia, a matéria nova é uma família só, então os dois AUROCs são a mesma conta.
- **O treino curto não só faz ele responder mais. Ele aprende a ordenar os casos.**
  - Na Telepatia, o AUROC vai de 0,59 para 0,83.
  - Antes, o "não sei" errava quase o mesmo que o modelo inteiro (43% contra 42%). Com só 7% de respostas, o
    "não sei" é quase tudo, então essa comparação dizia pouco. O AUROC mostra que a separação era fraca, mas
    não nula.
- **O preço:** entre os "não sei", ele acertaria cerca de metade dos casos. Para manter o erro em 10%, a regra
  simples joga fora muita resposta certa.
- **O que este teste não diz:** se o "não sei" aparece porque **falta informação** no caso ou porque **falta
  conhecimento** ao modelo. Isso é a parte 2, o teste limpo da `regra_beneficio`.

## Conferência

O plano pedia que a cobertura e o erro dos respondidos repetissem os números da receita final.

- **Telepatia:** repete os publicados (72% / 0,101) ✅.
- **`h4_log` nas duas fontes:** repete (16% / 0,19 e 7% / 0,18) ✅.
- **Felhen da receita:** deu **35%** / 0,149, mas o publicado é **37%** / 0,15 ⚠️.
  - O próprio `h9_recalibrar.py`, rodado hoje nos mesmos arquivos, dá o mesmo 35,1% / 0,149, e os arquivos são
    de 21/11.
  - Então esta análise repete a recalibração da receita. A diferença está no número publicado, e a origem dele
    ainda está sendo conferida.

## Limites

- Só `dev`, com 3 sementes.
- Só 3 famílias reservadas: 2 do Felhen e 1 da Telepatia.
- O "não sei" aqui é a regra simples (confiança abaixo do limiar), não a opção "não sei" da H5.
