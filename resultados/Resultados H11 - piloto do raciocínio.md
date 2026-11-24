# H11 (piloto): o Lumi consegue raciocinar antes de decidir?

**Data:** 21/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h6-bandit` (código em `f09e7d4`, resultado registrado em `84de61c`)
**Plano registrado antes de rodar:** `analises/plano_h11_piloto.md` (critério e três notas, todas antes do
piloto inteiro)
**Run:**
- `sbatch slurm/h11_piloto.sbatch runs/h4_log 512`;
- análise em `scripts/h11_piloto.py`, que gera `analises/h11p2.md` (no cluster);
- adaptador `h4_log`, `dev`, ordem canônica, 8 casos por família, Felhen e Telepatia.

## Por que este piloto

Até a H6, a decisão do Lumi era lida das letras numa passada só. Nesse formato, a perda log já dá o gradiente
exato, então o RL no máximo empata com o treino supervisionado.

O RL passa a ser necessário quando o modelo **sorteia um texto antes de decidir**, como um raciocínio. Esse
texto não é derivável, então só dá para aprender com recompensa. Essa é a ideia do RLCR (Damani et al., 2025):
1. o modelo raciocina;
2. a decisão é lida depois do raciocínio;
3. a recompensa é uma regra própria (Brier) nessa decisão.

Antes de montar esse laço (GRPO), o piloto responde quatro perguntas:
- o Lumi raciocina?
- a decisão sobrevive ao raciocínio?
- há sinal para o GRPO?
- cabe no tempo do cluster?

## O que foi testado

- **Guloso:** o Lumi raciocina até 512 tokens, sem sorteio. A letra é lida depois de `</think>\n\n`. A
  comparação é com a leitura de sempre (`<think></think>` vazio), nos mesmos 88 casos.
- **Sorteio:** 4 raciocínios sorteados por caso (temperatura 1,0, top_p 0,95), como um grupo do GRPO. Cada um
  é lido depois de `</think>\n\n`, e a recompensa é −Brier dessa leitura.

## Resultado: guloso ("sem" = leitura de sempre, "com" = depois do raciocínio)

| Fonte | Recorte | n | Acurácia sem → com | ECE sem → com | NLL sem → com | Decisão mudou | Tokens (mediana) | Não fechou | Tempo |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| Felhen | vistas | 40 | 0,900 → 0,875 | 0,115 → 0,137 | 0,36 → 0,98 | 8% | 152 | 10% | 3,4 s |
| Felhen | reservadas | 16 | 0,625 → 0,438 | 0,099 → 0,529 | 0,96 → 3,70 | 56% | 221 | 38% | 6,3 s |
| Telepatia | vistas | 24 | 0,500 → 0,375 | 0,119 → 0,546 | 0,83 → 4,24 | 33% | 216 | 29% | 4,0 s |
| Telepatia | reservadas | 8 | 0,750 → 0,625 | 0,193 → 0,375 | 0,88 → 3,43 | 25% | 210 | 0% | 3,9 s |

No geral, 81% dos raciocínios fecharam o `</think>`, 1% degenerou, e o tempo mediano foi de 3,9 s.

## Resultado: sorteio (grupos de 4)

| Fonte | Recorte | Grupos | Acurácia | ECE | NLL | Fecha | Grupos mistos | Grupos com sinal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Felhen | vistas | 40 | 0,775 | 0,193 | 1,33 | 78% | 32% | 45% |
| Felhen | reservadas | 16 | 0,453 | 0,424 | 3,10 | 44% | 69% | 88% |
| Telepatia | vistas | 24 | 0,458 | 0,435 | 2,99 | 61% | 58% | 88% |
| Telepatia | reservadas | 8 | 0,781 | 0,190 | 1,14 | 53% | 25% | 62% |

- **Fecha:** a amostra fechou o `</think>` antes de 512 tokens.
- **Misto:** no grupo, há decisão certa e errada.
- **Com sinal:** as 4 recompensas do grupo não são todas iguais (desvio > 0,01). Sem isso, o GRPO não aprende
  nada daquele caso.

No geral, 65% das amostras fecharam, nenhuma degenerou, e cada grupo levou 10,1 s (os 4 num lote só).

## O critério registrado

| Critério | Limite | Resultado | |
|---|---|---|---|
| Guloso: fecha o `</think>` | ≥ 80% | 81% | passa (raspando) |
| Guloso: degenera | ≤ 10% | 1% | passa |
| A decisão não quebra (vistas) | queda ≤ 5 pontos | Felhen −2,5; Telepatia −12,5; juntas −6,3 | **não passa** |
| Tempo por raciocínio | ≤ 6 s | 3,9 s | passa |
| Sorteio: fecha o `</think>` | ≥ 80% | 65% | **não passa** |
| Sorteio: degenera | ≤ 10% | 0% | passa |
| Sinal para o GRPO | ≥ 30% dos grupos | 66% | passa |
| Orçamento do GRPO (600 grupos) | < 4 h | 1,7 h | passa |

**Pelo critério: ainda não vale montar o GRPO com o Lumi oficial do jeito que está.** Duas coisas falham:
- a acurácia cai nas vistas da Telepatia;
- um terço dos raciocínios sorteados não termina em 512 tokens.

A queda da Telepatia são 3 casos em 24 (de 12 acertos para 9). A amostra é pequena, mas o critério registrado é
esse.

O plano B registrado (partir do `t6-lumi-4b`) **não se aplica**: ele era para o caso "não raciocina", e o Lumi
raciocina (só 1% degenerou).

## Leitura

**1. O Lumi raciocina, e rápido.** 81% dos raciocínios fecham, quase nada degenera, e cada um leva 3,9 s.

**2. O raciocínio deixa o Lumi superconfiante. Esse é o achado principal.**
- Na leitura de sempre, a H4 deixou o Lumi honesto (ECE de 0,10 a 0,19).
- Depois de raciocinar, ele dá quase 100% para uma letra. Quando erra, erra com certeza.
- O NLL mostra isso. Ele mede o quanto o modelo se surpreende com a resposta certa: 0 é "tinha certeza e
  acertou", e cada erro confiante pesa muito.
  - Nas vistas da Telepatia, ele vai de 0,83 para 4,24.
  - Na média geométrica, a probabilidade dada à resposta certa cai de 44% para 1,4%.
- Até no Felhen visto, onde a acurácia quase não muda (−2,5 pontos), o NLL quase triplica (0,36 → 0,98).

É exatamente o problema que o RLCR ataca: recompensar com Brier a decisão lida depois do raciocínio. O piloto
mostra que esse problema existe no Lumi, o que motiva o RLCD com raciocínio.

**3. Ele raciocina mais longo nos recortes mais difíceis.**
- No guloso, os raciocínios que não fecham são 10% nas vistas do Felhen. Nas reservadas do Felhen são 38%, e
  nas vistas da Telepatia, 29%.
- No sorteio, 78% fecham nas vistas do Felhen, contra 44% a 61% no resto.
- Quando o raciocínio não fecha, nós fechamos o `</think>` e lemos a letra de um pensamento incompleto.
- Dentro de cada recorte, os casos que não fecham não são mais difíceis para a leitura de sempre: ela acerta
  0,71 deles, contra 0,73 dos que fecham.
- A separação mais abaixo mostra o que o corte causa.

**4. Há sinal para o GRPO onde importa.**
- 66% dos grupos têm sinal, e 45% são mistos.
- O sinal é mais forte onde o Lumi erra mais: 88% nas reservadas do Felhen e nas vistas da Telepatia.
- Nas vistas do Felhen, só 45% têm sinal. Na maioria dos outros grupos, os 4 sorteios já acertam com
  certeza, e todos recebem a mesma recompensa.
- Atenção: como as letras saem quase sempre com 100% depois de um raciocínio que termina, esse sinal é quase
  todo de acerto contra erro, e não de confiança (veja a separação abaixo).

**5. Onde ler a letra.**
- Colada no `</think>`, a leitura continua ruim (acurácia de 0,17 a 0,33 em três recortes).
- `\n` e `\n\n` quase empatam.
- `\n` tem NLL um pouco menor nos 4 recortes (por exemplo, 3,98 contra 4,24 nas vistas da Telepatia), e é o
  que o Lumi escreve sozinho.

A leitura principal continuou `\n\n`, como registrado. Para o próximo ciclo, `\n` é a escolha natural.

**6. Cabe no tempo.** Com os 4 sorteios num lote, 600 grupos levam cerca de 1,7 h de geração. O tempo dos
passos de treino não entra nessa conta.

## Separação: os que fecharam e os que não fecharam (feita depois do piloto)

**Data:** 21/11/2026, job 37786 (só CPU), commit `2996b2e`. A regra de leitura foi registrada no plano antes de
rodar.

**Guloso, todos os casos juntos** ("sem" = leitura de sempre nos mesmos casos):

| Raciocínio | n | Acurácia sem → com | NLL sem → com | Confiança média (com) | Erros com confiança ≥ 0,9 |
|---|---:|---|---|---:|---:|
| Fechou | 71 | 0,732 → 0,704 | 0,59 → 2,74 | 1,00 | 21 de 21 |
| Não fechou | 17 | 0,706 → 0,353 | 0,84 → 1,94 | 0,73 | 4 de 11 |

**Sorteio, todas as amostras juntas:**

| Raciocínio | Amostras | Acurácia | NLL | Confiança média | Erros com confiança ≥ 0,9 |
|---|---:|---:|---:|---:|---:|
| Fechou | 229 | 0,716 | 2,35 | 1,00 | 64 de 65 |
| Não fechou | 123 | 0,472 | 1,60 | 0,75 | 16 de 65 |

Os números por recorte estão em `analises/h11p2.md`, no cluster. Um exemplo, nas vistas da Telepatia:
- os que fecharam vão de 0,353 para 0,412 de acurácia, mas o NLL vai de 0,90 para 5,24, e os 10 erros têm
  confiança ≥ 0,9;
- os que não fecharam caem de 0,857 para 0,286.

**Pela regra registrada: o comprimento não é a causa principal.**
- A condição 1 passa: nos que fecharam, a acurácia cai só 2,8 pontos (o limite era 5).
- A condição 2 falha: dos 25 erros confiantes do guloso, só 4 (16%) estão nos que não fecharam.

Então a superconfiança é do próprio raciocínio, o GRPO com Brier é o remédio, e o comprimento é um problema a
mais.

**Leitura: são dois problemas, com duas causas.**

1. **A queda de acurácia vem dos raciocínios cortados.**
   - Nos mesmos casos, os que não fecharam vão de 0,706 para 0,353 (−35 pontos). Os que fecharam perdem só 2,8.
   - Os cortados não eram casos mais difíceis: a leitura de sempre acerta 0,71 deles.
   - Ler a letra de um pensamento pela metade estraga a decisão. É isso que derruba a acurácia das vistas da
     Telepatia no critério.
2. **A superconfiança vem dos raciocínios que terminam.**
   - Quando o raciocínio fecha, a confiança média é 1,00. Os 21 erros do guloso têm confiança de pelo menos
     0,9, e no sorteio são 64 de 65.
   - Os cortados são menos confiantes (0,73), porque o pensamento não chegou a uma conclusão.

**Por que isso acontece (interpretação):** o raciocínio termina dizendo a resposta ("So answer N"). Depois do
`</think>`, a leitura das letras só copia essa conclusão, com certeza total. A probabilidade das letras deixa de
medir dúvida.

**Consequência para o GRPO (ponto de atenção no desenho):**
- A recompensa planejada é o Brier das letras lidas depois do raciocínio. Se essas letras saem quase sempre com
  100%, o Brier vira quase só "acertou (0) ou errou (−2)".
- Essa é a recompensa de acerto que, na H4, estragou a honestidade.
- O GRPO até pode aprender a terminar o raciocínio com dúvida (os cortados mostram que a leitura pode sair
  menos confiante), mas nada garante.
- No RLCR, o modelo **escreve uma confiança** (um número) depois da resposta, e a recompensa soma o acerto e o
  Brier dessa confiança. É uma confiança que o modelo consegue graduar, o que parece o caminho mais seguro.

## H11b: o Lumi escreve uma confiança que serve? (feito depois do piloto)

**Resposta curta: ele escreve o número, mas quase sempre "100". O número não separa os acertos dos erros o
bastante, e o critério falha em 1 de 3 pontos.**

- **Plano e critério,** registrados antes de rodar: `analises/plano_h11b_confianca.md`.
- **Run:** job 37901, relatório `analises/h11b.md` (no cluster).
- **Desenho:** os mesmos 88 casos e raciocínios do piloto. Depois da letra, um turno novo pergunta "Qual é a
  probabilidade, de 0 a 100, de que a sua resposta esteja correta?".

| critério | limite | resultado | |
|---|---|---|---|
| Escreve: respostas com número de 0 a 100 | ≥ 90% | 100% | passa |
| Gradua: o valor mais comum | ≤ 80% das respostas | 77% (valor 100) | passa |
| Separa: AUROC da confiança escrita | ≥ 0,65 | 0,639 | **não passa** |

**Quatro fontes de confiança nos mesmos casos (guloso):**

| fonte de confiança | acurácia da decisão | AUROC | confiança média: acerta / erra | ECE | Brier |
|---|---:|---:|---|---:|---:|
| confiança escrita | 0,636 | 0,639 | 0,97 / 0,91 | 0,337 | 0,334 |
| letras depois do raciocínio | 0,636 | 0,592 | 0,97 / 0,91 | 0,324 | 0,317 |
| consistência dos 4 sorteios | 0,636 | 0,769 | 0,78 / 0,41 | 0,125 | 0,195 |
| letras sem raciocínio (H4; outra decisão) | 0,727 | 0,798 | 0,75 / 0,54 | 0,066 | 0,160 |

O AUROC é a chance de um acerto ter confiança maior que um erro: 0,5 é sorteio, 1,0 é perfeito.

**Leitura:** olhamos se o número muda pelo motivo que a gente acha.
- **O "gradua" passou só no papel.** 85 dos 88 números ficam entre 90 e 100, e só 4 valores diferentes
  aparecem. Na prática, ele escreve "100" ou "quase 100".
- **Ele diz cerca de 95% e acerta 64%** (ECE 0,34). A confiança escrita repete a das letras, com as mesmas médias
  (0,97 quando acerta, 0,91 quando erra).
- **O AUROC de 0,639 vem mais da diferença entre grupos do que de julgar cada caso.**
  - Dentro de cada recorte (fonte e vistas ou reservadas), ele fica entre 0,46 e 0,59.
  - Pelas médias, os poucos números baixos ficam nas reservadas do Felhen, onde ele também acerta menos (44%).
  - Ou seja, ele percebe um pouco que aquela matéria é mais difícil, mas não qual caso vai errar.
- **A melhor confiança depois do raciocínio é a consistência dos 4 sorteios** (AUROC 0,769, ECE 0,125). Quando
  os raciocínios chegam a letras diferentes, ele costuma errar. O custo são 4 raciocínios por caso.
- **As letras sem raciocínio (H4) continuam as mais honestas** (AUROC 0,798, ECE 0,066) e acertam mais (0,727
  contra 0,636). Mas elas julgam outra decisão, a de sempre, sem raciocínio.
- **No sorteio:**
  - a confiança escrita muda dentro do grupo de 4 em 45% dos grupos;
  - a recompensa do RLCR tem sinal em 55% dos grupos, contra 66% com o Brier das letras.

**O que isso muda:** a leitura registrada para este resultado é "escreve, mas não separa". O formato funciona:
100% de números, 0,18 s por resposta. Mas a confiança escrita terá de ser **aprendida**. Caminhos:
- o GRPO estilo RLCR com o Brier no número escrito desde o começo;
- um SFT curto do formato antes. Uma ideia ainda não testada é usar a consistência dos 4 sorteios como alvo do
  número escrito, porque é ela que separa melhor.

**Limites da H11b:**
- **88 casos e 32 erros no guloso.** O AUROC tem erro de cerca de ±0,06, então 0,639 e 0,65 não se distinguem
  de verdade. O que pesa mais é o número quase sempre "100" e a separação perto de 0,5 dentro de cada recorte.
- **A pergunta vem depois, num turno novo.** No RLCR, o número sai na mesma geração do raciocínio.
- **O adaptador `h4_log` nunca viu um segundo turno.**

## O que isso muda no próximo ciclo

**Atualização (24/11/2026): a trilha do raciocínio foi encerrada depois da H11b.** O motivo está em
`Resultados finais - RLCD no Lumi.md`, seção "Decisão: parar a trilha do raciocínio". O texto abaixo fica como
estava, para o caso de alguém retomar.

O caminho continua sendo o **GRPO com raciocínio e recompensa Brier** (RLCR). O sinal (66% dos grupos) e o
tempo (1,7 h de geração) estão bons. Há dois ajustes antes de montar:

1. **Comprimento:** é ele que derruba a acurácia. Opções:
   - mais tokens;
   - recompensa baixa para o raciocínio que não fecha, para o GRPO aprender a fechar;
   - pedir no prompt um raciocínio curto.
2. **De onde vem a confiança:** depois de um raciocínio que termina, a leitura das letras sai com 100%. Para o
   Brier ensinar honestidade, a confiança precisa vir de algo que o modelo consiga graduar, como a confiança
   escrita do RLCR. A H11b mostrou que, perguntada depois, a confiança escrita também sai quase sempre 100:
   ela terá de ser aprendida.

## Limites

- **Poucos casos:** 88 no total, 8 por família. As reservadas da Telepatia são 8 casos de uma família só. O ECE
  aqui é só indicativo.
- **Um adaptador só:** o `h4_log` foi treinado com o `<think>` vazio. Não sabemos se o Lumi oficial sem
  adaptador fica igualmente superconfiante. Parte do efeito pode ser o adaptador fora do formato.
- **Uma semente de sorteio,** fixa por caso.
- **Raciocínios cortados:** os que não fecham são lidos com o `</think>` fechado por nós. A separação mostra
  que a queda de acurácia vem quase toda daí.
- **A separação foi feita depois de ver os dados,** com só 17 raciocínios cortados no guloso. Ela orienta o
  desenho, mas não muda o critério.
- **Só `dev`.** O `test` oficial não foi usado.
