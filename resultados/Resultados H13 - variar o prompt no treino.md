# H13: variar o texto do prompt no treino (a parte de dados do Clef)

## Resposta curta

**O `h4_log` é frágil a um prompt que nunca viu, e variar o prompt no treino conserta boa parte disso.**
Na matéria nova, o resultado depende da fonte.

- **Com um prompt nunca visto:**
  - o `h4_log` perde 3 a 5 pontos nas famílias vistas;
  - em muitos casos, ele deixa de querer responder com uma letra.
- **O `h4_aug` (treinado com 4 textos de prompt diferentes) é mais robusto:**
  - recupera +2 pontos no prompt novo;
  - quase sempre responde com uma letra.
- **No prompt de sempre, a matéria nova muda de jeitos opostos:**
  - Telepatia: **+3,9 pontos**;
  - Felhen: **−1,8 ponto**.

## O que foi testado

- **Variantes do prompt** (`src/lumi_decision/prompts.py`):
  - **0:** o prompt de sempre;
  - **1 a 3:** usadas no treino. Mudam títulos, frases, a ordem dos blocos e o formato das opções. Na Telepatia,
    as chaves do JSON também são embaralhadas;
  - **4:** só na avaliação, nunca vista no treino.
- **Braço `h4_aug`:** igual ao `h4_log` em tudo, menos o texto. A cada época, cada exemplo recebe uma variante de
  0 a 3. São 3 sementes.
- **Plano registrado antes de rodar:** `analises/plano_h13_dados.md`.

## Resultado

Diferenças com IC 95% por bootstrap de grupos, 3 sementes juntas. Em negrito, as que têm o IC todo de um lado.

**1. Quanto o `h4_log` sofre com o prompt nunca visto** (prompt 4 − prompt de sempre):

| Recorte | Acurácia | NLL |
|---|---|---|
| Felhen vistas | **−2,9** | **+0,098** |
| Felhen reservadas | **−3,0** | **+0,061** |
| Telepatia vistas | **−4,6** | **+0,052** |
| Telepatia reservadas | **+2,0** | +0,009 (empate) |

**Ele ainda responde com uma letra?**
- Medida: a massa de probabilidade nas letras, onde 1 quer dizer que ele só quer escrever uma letra
  (`scripts/h13_letras.py`).
- No prompt de sempre, fica em 0,99 em todos os braços.

| Prompt nunca visto | Massa nas letras | Casos com massa abaixo de 0,5 | Gera a letra que a leitura escolheu |
|---|---|---|---|
| `h4_log`, Felhen | 0,42 a 0,58 | 35% a 75% | 34% a 71% |
| `h4_aug`, Felhen | 0,60 a 0,69 | 15% a 34% | 72% a 90% |
| `h4_log`, Telepatia | 0,50 a 0,64 | 16% a 45% | 70% a 95% |
| `h4_aug`, Telepatia | 0,70 a 0,77 | 0% a 1% | 97% a 98% |

(Faixa entre as 3 sementes.)

**2. `aug − log` no prompt de sempre:**

| Recorte | Acurácia | ECE | NLL |
|---|---|---|---|
| Felhen vistas | +0,1 (empate) | empate | empate |
| Felhen reservadas | **−1,8** | +0,009 (empate) | **+0,027** |
| Telepatia vistas | +1,0 (empate) | empate | empate |
| Telepatia reservadas | **+3,9** [3,0; 4,8] | **+0,019** | **−0,015** |

**3. `aug − log` no prompt nunca visto:**

| Recorte | Acurácia | ECE | NLL |
|---|---|---|---|
| Felhen vistas | **+2,1** | **−0,009** | **−0,050** |
| Felhen reservadas | −1,0 (empate) | empate | empate |
| Telepatia vistas | **+2,3** | **+0,020** | **−0,016** |
| Telepatia reservadas | **+2,5** | +0,018 (empate) | **−0,031** |

## Leitura

- **Robustez: o `h4_aug` ganha.**
  - No prompt nunca visto, as vistas sobem +2 pontos nas duas fontes.
  - Ele continua respondendo com uma letra, principalmente na Telepatia.
  - O `h4_log` perdia o formato: no Felhen, até 75% dos casos tinham menos da metade da probabilidade nas letras.
  - Pelo critério registrado, o ganho do item 3 é **robustez**.
- **Generalização: depende da fonte.**
  - **Telepatia:** a matéria nova sobe +3,9 no prompt de sempre, mas o ECE piora (+0,019). Pelo critério
    registrado, piorar a calibração conta como "piora", mesmo com o NLL e o Brier melhorando.
  - **Felhen:** a matéria nova cai −1,8. É uma "piora".
- **Pelo motivo certo? Há um sinal de que não, na Telepatia.**
  - O próprio `h4_log` acerta **+2,0** na Telepatia nova só por trocar o texto do prompt, sem treino nenhum.
  - Então a família nova da Telepatia é sensível ao jeito de apresentar o caso.
  - Parte do +3,9 pode ser o `aug` ter visto mais formatos, e não ter aprendido mais a matéria. Isso não foi
    separado.
- **A lição da H12c vale aqui:** uma melhora medida antes do treino curto pode sumir dentro da receita.

## O que isso muda

- **A receita final ainda não muda.**
- O que o `aug` traz de mais claro é **robustez a prompts diferentes**. Isso importa no uso real, em que cada
  pessoa escreve o pedido de um jeito.
- O próximo passo seria testar a receita com o `aug`:
  - o treino curto parte do `h4_aug`, também com o prompt variado;
  - avaliar no prompt de sempre e no nunca visto.

  A receita entra se mantiver a cobertura e ficar mais robusta.

## Limites

- **Variantes escritas por nós:** em português, e só 4 no treino.
- **O formato da resposta não muda:** sempre uma letra.
- **Só `dev`, com 3 sementes.** Na Telepatia, a matéria nova é uma família só.
