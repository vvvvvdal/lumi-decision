# RLCD no Lumi: resumo final dos experimentos (H4 a H9a, H12 a H13, "sabe que não sabe" e pilotos H11 e H11b)

## A pergunta

O Lumi-4B escolhe uma opção e diz quanta confiança tem. Queremos que essa confiança seja **honesta**: quando
ele diz 70%, deve acertar 70% das vezes. Com isso, ele pode dizer "não sei" quando a chance de errar é alta
(os dois jeitos de fazer isso estão logo abaixo).

Os experimentos tentaram responder três perguntas:
1. que recompensa deixa o Lumi honesto;
2. se ele continua honesto em matéria nova (as famílias reservadas, que ele não viu no treino);
3. como fazer ele **saber mais** na matéria nova sem esquecer a antiga.

## Os dois tipos de "não sei"

Exemplo: chega um ticket, e o Lumi precisa escolher o time A, B ou C. Há dois jeitos de ele "não responder", e
eles não são a mesma coisa.

| | "Não sei" pela confiança (a regra simples) | "Não sei" como opção no prompt (H5) |
|---|---|---|
| Como funciona | O prompt só tem A, B e C, e o Lumi sempre escolhe uma. Se a confiança ficar abaixo de um limite, o **sistema** passa o caso para uma pessoa | O prompt ganha uma opção a mais, "Não sei / passar para um humano" (`prompts.with_abstain`). O **próprio Lumi** escolhe essa opção, como escolheria uma letra |
| Como é aprendido | O Lumi não aprende nada novo. O limite é escolhido com casos rotulados, para errar no máximo 10% das respostas | Com RL: acertar vale +1, errar vale −λ e "não sei" vale 0. Com λ = 3, só vale responder com mais de 75% de chance |
| Onde aparece | H4, H7, H9b e a receita final (de 7% para 72% das faturas novas respondidas) | Só na H5 |
| Resultado | É o que a receita final usa | Continua honesto, mas não ganha da regra simples: respondendo a mesma quantidade de casos, erra igual ou mais |

**Na prática:**
- **"Quando ele tinha o 'não sei' como opção, respondia muito 'não sei'"** é a H5. Na Telepatia, ele escolhia a
  opção em cerca de 91% dos casos com λ = 3, e em quase todos com λ = 9.
- **"Depois do treino com 100 exemplos, ele responde mais"** é a H9 com a **regra simples**. Ele passou a saber a
  matéria, a confiança subiu nos casos que ele sabe, e menos casos ficaram abaixo do limite.
- **A H5 não foi repetida depois da H9.** Essa comparação está em aberto.

## Ele sabe o que não sabe?

**Resposta curta: sim, depois de aprender a matéria** (medido caso a caso em `Resultados - sabe que não sabe.md`).

- **No "não sei" pela confiança (o que a gente usa):** ele manda para uma pessoa os casos mais difíceis.
  - Depois da receita final, nas faturas novas, ele erra 10% dos casos que responde e erraria 47% dos que
    passa adiante. A confiança separa acerto de erro dentro da mesma família (AUROC 0,83 na Telepatia e 0,75 no
    Felhen).
  - Mas não é perfeito: em mais ou menos metade dos casos que ele passa adiante, ele teria acertado.
  - Antes do treino curto, a separação era fraca (AUROC 0,59 na Telepatia).
- **No "não sei" como opção (H5):** ele diz "não sei" quando o caso parece complicado, mas não percebe quando
  falta uma informação de verdade.
- **O que ainda não sabemos:** se o "não sei" pela confiança aparece porque falta informação no caso ou só
  porque ele ainda não aprendeu aquele tipo de caso.

Os detalhes vêm abaixo.

A pergunta: o "não sei" acontece porque o Lumi **reconhece** que não sabe, ou é só um jeito de esconder os
erros? Se ele acerta bem nos casos que responde e manda para uma pessoa justamente os que erraria, ele reconhece
o que sabe.

Há duas razões diferentes para um caso merecer um "não sei":
- **falta informação no caso:** nem uma pessoa acertaria;
- **falta conhecimento no modelo:** uma pessoa acertaria, mas o Lumi ainda não aprendeu.

**1. Regra simples: o "não sei" pega os casos que ele erraria?**

Medido caso a caso (`scripts/analise_sabe.py`), com a recalibração da receita final (25 rótulos, alvo de 10%),
3 sementes e IC 95% por bootstrap de grupos. "AUROC dentro da família" é a chance de um acerto ter confiança
maior que um erro, sem misturar matérias: 0,5 é o acaso.

| Matéria nova | Responde | Erro nos que responde | Erro nos "não sei" | AUROC dentro da família |
|---|---:|---|---|---|
| Faturas (Telepatia), `h4_log` | 7% | 19% [3; 32] | 43% [39; 48] | 0,59 [0,56; 0,62] |
| Faturas (Telepatia), receita final | 72% | 10% [6; 15] | 47% [39; 53] | 0,83 [0,80; 0,85] |
| Felhen, `h4_log` | 16% | 20% [5; 33] | 53% [47; 59] | 0,69 [0,64; 0,73] |
| Felhen, receita final | 35% | 16% [7; 26] | 45% [38; 53] | 0,75 [0,71; 0,79] |

**Leitura:**
- **Depois da receita, separa.** Os "não sei" erram de 2,8 a 4,7 vezes mais que as respostas, com os ICs separados, e
  o AUROC dentro da família passa de 0,7 nas duas fontes.
- **Pelo motivo certo:** o AUROC dentro da família é igual ao geral, então o "não sei" não está só separando
  matéria difícil de fácil (o problema da confiança escrita na H11b).
- **Antes do treino curto, a separação era fraca** (AUROC 0,59 na Telepatia).
- **Mas não é perfeito.** Cerca de metade dos casos que ele manda para uma pessoa ele teria acertado.
- **Conferência:** o Felhen deu 35% de respostas, e o número publicado antes era 37%. O próprio
  `h9_recalibrar.py` dá 35% hoje nos mesmos arquivos. A origem do 37% está sendo conferida.

**2. "Não sei" como opção (H5): ele reage à falta de informação?**

A família `regra_beneficio` permite um teste limpo: o gerador sabe em quais casos falta um documento. Dentro dos
casos cuja resposta certa é "negar", a única diferença é faltar documento ou não (3 sementes, λ = 3):

| Caso | Diz "não sei" ou pede o documento |
|---|---:|
| Negar, **com** documento faltando | 36% |
| Negar, **sem** documento faltando | 40% |
| Conceder (tudo certo) | 1,6% |

Se ele reconhecesse a falta de informação, a primeira linha seria bem maior que a segunda. **As duas empatam.**
Ele diz "não sei" quando o caso tem algum problema (regra violada, documento faltando ou valor acima do teto) e
quase nunca quando está tudo certo. É um sinal de "caso complicado", não de "me falta informação". Detalhes em
`Resultados H5 - abstenção por RL.md`.

## O caminho, em uma tabela

| Etapa | Pergunta | Resposta | Número principal |
|---|---|---|---|
| H4 | Recompensa de acerto, Brier ou log? | Acerto destrói a honestidade. Brier e log empatam. | Com recompensa de acerto, o ECE das reservadas vai a 0,43–0,48 (3 sementes para log e Brier) |
| H5 | Treinar o "não sei" como opção no prompt, com RL? | Mantém o Lumi honesto, mas não supera o "não sei" pela confiança (a regra simples) | Na mesma cobertura, erra igual ou mais |
| H6 | RL sem gabarito (só "acertou ou errou")? | O que estraga a honestidade é a recompensa, não a falta do gabarito | O `bandit_log` mantém a acurácia e recupera 73–86% da calibração |
| H7 | Quantos rótulos da matéria nova para recalibrar? | 25 consertam a escala. O limite real é saber pouco da matéria. | Mesmo com ajuste perfeito, responde só 12% (Felhen) e 2% (Telepatia) com 10% de erro |
| H8 | Mostrar exemplos no prompt ensina? | Não ensina, mas calibra | ECE da Telepatia de 0,104 para 0,029, com a mesma acurácia |
| H9 | Treinar um pouco com 100 exemplos da matéria nova? | **Ensina** (3 sementes) | Acurácia: Felhen +11,6, Telepatia +20,3 |
| H9b | Recalibrar depois desse treino? | Conserta a superconfiança | Telepatia: responde 68% com 10% de erro |
| H9r | Replay (misturar casos antigos no treino)? | Evita esquecer a matéria antiga (3 sementes) | Telepatia vista: −0,8 ponto, contra −2,1 sem replay |
| H9m | Misturar os adaptadores em vez de treinar de novo? | É uma troca: não esquece, mas ensina menos | Felhen +8,7, contra +13,6 |
| H9a | Escolher os exemplos onde o Lumi está em dúvida? | Empate com sortear | Diferença de ±1,9 ponto, com os dois ICs passando pelo zero |
| H11 (piloto) | O Lumi raciocina antes de decidir? Dá para montar o GRPO? | Raciocina, mas fica superconfiante, e o raciocínio sorteado é longo demais. O critério falha em 2 de 8 pontos: a queda vem dos raciocínios cortados, e a superconfiança, dos que terminam | NLL das vistas da Telepatia de 0,83 para 4,24; 65% dos raciocínios sorteados fecham em 512 tokens |
| H11b (piloto) | O Lumi escreve uma confiança que serve para o RLCR? | Escreve, mas quase sempre "100". Não separa acerto de erro o bastante (o critério falha em 1 de 3 pontos) | AUROC 0,639 (limite 0,65); a consistência de 4 sorteios chega a 0,769 |
| H12 | A perda do Clef (entropia cruzada com label smoothing + Brier) é melhor que a log? | Não. Acerta um pouco mais na Telepatia, mas a confiança fica menos honesta no Felhen, mesmo depois da temperatura | Telepatia reservadas: acc +1,8 ponto; Felhen reservadas depois da temperatura: ECE 0,101 contra 0,062 (3 sementes) |
| H12b | Na perda do Clef, o que faz o quê? | O smoothing traz o custo na calibração. O Brier traz o ganho de acurácia | `logbrier`: Telepatia reservadas +2,2 pontos, com T = 1,05 (a do Clef é 0,83) |
| Sabe que não sabe | O "não sei" da receita pega os casos que ele erraria? | Sim, dentro da mesma família | Telepatia: erra 10% quando responde, 47% nos "não sei"; AUROC 0,83 (antes 0,59) |
| H12c | A `log_brier` dentro da receita final? | Empate: o ganho de antes do treino curto some | Telepatia: responde 72% com 10% de erro nos dois braços |
| H13 | Variar o texto do prompt no treino? | Fica mais robusto a prompts nunca vistos. Na matéria nova, ajuda na Telepatia e piora no Felhen | Prompt nunca visto: vistas +2 pontos; matéria nova no prompt de sempre: Telepatia +3,9, Felhen −1,8 |
| H13b | A receita com o prompt variado? | Não entra: ganha no prompt nunca visto e perde no de sempre. E a receita de hoje é frágil a um prompt novo | Telepatia responde: 72% → 66% no prompt de sempre; no nunca visto, 50% (hoje) contra 57% |

## Receita final

1. **Recompensa honesta:** usar a perda log, uma regra de pontuação própria (H4). A perda do Clef não ganhou
   dela (H12). A versão sem smoothing (`log_brier`) acerta mais na Telepatia antes do treino curto (H12b), mas
   empata dentro da receita (H12c).
2. **Treino curto na matéria nova:** cerca de 100 exemplos **sorteados** da família nova, mais o mesmo número
   de casos já vistos (replay), por 3 épocas a partir do adaptador da H4 (H9 + H9r).
3. **Recalibrar:**
   - nas famílias vistas, usar a temperatura que o pipeline já ajusta;
   - nas novas, usar cerca de 25 rótulos da família para ajustar a temperatura e o limite do "não sei" pela
     confiança (H7 e H9b).

## O antes e o depois

Alvo: errar no máximo 10% das vezes em que responde. Matéria nova, média de 3 sementes.

| Fonte | Lumi depois da H4: responde / erra | Receita final: responde / erra |
|---|---|---|
| Telepatia (faturas novas) | 7% / 18% | **72% / 10%** |
| Felhen (direito e checagem de fatos novos) | 16% / 19% | **37% / 15%** |

Na matéria antiga, o Lumi não esquece (Telepatia −0,8 ponto) e continua honesto (ECE 0,017 no Felhen e 0,034
na Telepatia).

## O que aprendemos

- **A recompensa decide a honestidade.** Qualquer regra própria (log ou Brier) serve. Recompensar só o acerto
  ensina a chutar com certeza, com ou sem gabarito (H4 e H6).
- **Honestidade não é competência.** Ajustar a escala (temperatura, exemplos no prompt) torna o "não sei"
  honesto, mas não faz o Lumi acertar mais na matéria nova (H7 e H8).
- **Pouco treino já ensina, e o replay evita esquecer.** Saber mais muda muito o resultado: na Telepatia, o
  Lumi passa de responder 7% para 72% (H9, H9r e H9b).
- **Quantos exemplos usar importa mais que quais.** Escolher por dúvida empatou com sortear. Usar 25 exemplos
  em vez de 100 tira cerca de metade do ganho no Felhen (H9a).

## O que é RLCD aqui e o que é extra

- **RLCD (o foco da task):**
  - H4: que recompensa usar;
  - H5: o "não sei" como ação de RL;
  - H6: RL só com "acertou/errou";
  - H11 (piloto): preparar o RL com raciocínio, onde o RL é necessário.

  Essas etapas mostram, no Lumi, a ideia central do RLCD: a recompensa precisa ser uma regra própria, ou a
  probabilidade deixa de ser honesta.
- **Extra (adaptar à matéria nova):** H7 a H9a. São treino supervisionado e recalibração, sem RL. É daqui que
  vem o ganho prático (72% das faturas novas), mas não é RL.

## Por que o RL não ganhou do supervisionado

- No `lumi-decision`, a decisão é lida das letras **numa passada só**. A recompensa depende só dessa
  distribuição, que é derivável.
- Então a perda log já dá o gradiente **exato**, e o RL só estima esse mesmo gradiente com ruído. O teste
  `test_bandit_log_tem_o_gradiente_do_log_score_em_media` confere isso.
- Nesse formato, o RL no máximo empata, que foi o que vimos na H5 e na H6.

O RL passa a ser necessário quando algo entre a pergunta e a decisão é **sorteado e não derivável**, como o
raciocínio em texto (RLCR, Damani et al., 2025). Isso levou ao **piloto da H11**.

## Piloto H11: raciocinar antes de decidir

Detalhes em `Resultados H11 - piloto do raciocínio.md`.

- **O Lumi raciocina:** 81% dos raciocínios fecham o `</think>`, 1% degenera, 3,9 s cada.
- **Depois de raciocinar, ele fica superconfiante:**
  - o NLL sobe de 0,36–0,96 para 0,98–4,24, e o ECE chega a 0,55;
  - quando erra, erra com certeza.

  É o problema que o RLCR ataca com Brier depois do raciocínio.
- **Há sinal e tempo para o GRPO:** 66% dos grupos de 4 sorteios têm recompensas diferentes, e 600 grupos
  levam 1,7 h de geração.
- **Mas o critério registrado falha em 2 pontos:**
  - a acurácia das vistas cai 6,3 pontos (Telepatia −12,5, com 3 casos em 24);
  - só 65% dos raciocínios sorteados terminam em 512 tokens.
- **A separação entre os raciocínios que fecharam e os que não fecharam mostra duas causas** (feita depois do
  piloto, com a regra registrada antes de rodar):
  - **a queda de acurácia vem dos raciocínios cortados:** nos mesmos casos, de 0,706 para 0,353, contra −2,8
    pontos nos que fecharam;
  - **a superconfiança vem dos que terminam:** confiança média de 1,00, e todos os 21 erros com confiança de
    pelo menos 0,9. O raciocínio termina dizendo a resposta, e a leitura das letras só copia.
- **Consequência:** com as letras a 100%, o Brier das letras vira quase só "acertou ou errou", a recompensa que
  estragou a honestidade na H4. Antes do GRPO, é preciso resolver o comprimento e tirar a confiança de algo que
  o modelo consiga graduar, como a confiança escrita do RLCR.

## Piloto H11b: escrever a confiança

**Resposta curta: ele escreve o número, mas quase sempre "100". O número não separa os acertos dos erros o
bastante, então a confiança escrita terá de ser aprendida.** Detalhes em
`Resultados H11 - piloto do raciocínio.md`.

- **Como foi:** nos mesmos 88 casos do piloto H11, depois da letra, um turno novo pergunta "Qual é a
  probabilidade, de 0 a 100, de que a sua resposta esteja correta?".
- **O critério falha em 1 de 3 pontos:**
  - escreve um número em 100% das respostas;
  - o valor mais comum, "100", aparece em 77% (limite: 80%);
  - mas o AUROC fica em 0,639 (limite: 0,65).
- **Na prática, ele escreve "100" ou "quase 100".** 85 dos 88 números ficam entre 90 e 100. Ele diz cerca de
  95% e acerta 64%.
- **A separação que aparece vem da matéria, não do caso.** Dentro de cada grupo de casos, o AUROC fica entre
  0,46 e 0,59, quase um sorteio. Os poucos números baixos ficam na matéria mais difícil, as reservadas do
  Felhen.
- **O que separa melhor depois do raciocínio é a consistência:** quando 4 raciocínios sorteados chegam a letras
  diferentes, ele costuma errar (AUROC 0,769, ECE 0,125).
- **Consequência para o GRPO:** o RLCR terá de ensinar o número desde o começo, com o Brier na confiança
  escrita, ou com um SFT curto do formato antes.

## Decisão: parar a trilha do raciocínio

**Paramos aqui. O Lumi segue com o raciocínio desligado, que é a receita final.**

- **Por que testamos o raciocínio:**
  - com o raciocínio desligado, a escolha da letra é uma conta direta, e o treino supervisionado (perda log) já
    dá a direção exata de ajuste;
  - o RL só estima essa mesma direção com ruído, e por isso empatou na H5 e na H6;
  - o RL só tem vantagem quando há algo sorteado no meio, como um raciocínio em texto (RLCR).
- **Por que paramos:**
  - com raciocínio, ele acerta menos: 0,636 contra 0,727 nos mesmos 88 casos;
  - fica superconfiante: a leitura das letras sai com 100%, e a confiança escrita, quase sempre "100";
  - sem raciocínio, a confiança é a mais honesta (AUROC 0,798, ECE 0,066), e a receita final já funciona.
- **O que fica sem resposta:**
  - o adaptador `h4_log` foi treinado sem raciocínio, e parte da queda pode vir dele estar fora do formato, e não
    do raciocínio em si;
  - o Lumi oficial pensando, sem adaptador, nunca foi medido.
- **O que fica pronto se alguém retomar:** `scripts/h11_piloto.py`, `scripts/h11b_confianca.py` e os planos em
  `analises/`.

## Fizemos o que o Clef faz?

**Na parte da perda, sim, e ela não ganhou da nossa.** O Clef (Cloudflare, 2026) treina com entropia cruzada com
label smoothing somada ao Brier. Testamos isso com 3 sementes na H12 (`Resultados H12 - braço Clef.md`):

- acerta 1 a 2 pontos a mais na Telepatia;
- deixa a confiança menos honesta, porque o label smoothing ensina a nunca dar 100%;
- a temperatura conserta isso só em parte: nas reservadas do Felhen, ela até piora (ECE 0,101 contra 0,062).

A H12b separou as duas peças (`Resultados H12b - smoothing ou Brier.md`): o custo vem do smoothing, e o ganho
da Telepatia vem do Brier. A entropia cruzada + Brier sem smoothing (`log_brier`) acerta +2,2 pontos nas
reservadas da Telepatia sem o custo do smoothing. Dentro da receita final, esse ganho some (H12c).

Na parte de dados, a H13 treinou com o texto do prompt variado (`Resultados H13 - variar o prompt no treino.md`).
O Lumi fica mais robusto a um prompt que nunca viu, mas na matéria nova ganha na Telepatia e perde no Felhen.

| No Clef | No nosso projeto |
|---|---|
| Lê a escolha direto das opções, sem gerar texto | Sim: a leitura das letras |
| Modelo base congelado + LoRA (posto 256) | Sim, com posto 16 |
| Cabeça de roteamento treinada junto | Testada pelo Palhares (ponteiro, H3), e foi pior |
| Entropia cruzada com label smoothing + Brier | Testada na H12: não ganhou da perda log. Sem o smoothing, ganha antes do treino curto (H12b), mas empata na receita (H12c) |
| RLCD como objetivo secundário: crédito parcial para opções vizinhas numa escala, prêmio por saída precisa e penalidade de referência | Em parte: RL sem gabarito (H6) e "não sei" com coleira KL (H5). O crédito parcial só vale para opções em escala, e as nossas não são |
| Dados sintéticos permutando ordem de campos, prompts e esquemas | Testado na H13: prompts, formato das opções e ordem dos campos variados. Fica mais robusto; na matéria nova, depende da fonte |

## Limites

- **Só `dev`.** O `test` oficial não foi tocado.
- **Um prompt só.** Os números da receita valem para o prompt de sempre. Com um texto nunca visto, a Telepatia
  responde 50% em vez de 72% (H13b).
- **Sementes:**
  - têm 3 sementes: H4 (log e Brier), H12, H12b e H12c (perdas do Clef), H13, H13b, H6 (`bandit_log` com G = 32, job 37691), H9 com k = 100 e H9r;
  - H5: a comparação com a regra simples (abster pela confiança) usa só a semente 0. O λ = 3 foi repetido com
    3 sementes para ver em que casos ele abstém;
  - os outros braços da H6, H8, H9m e H9a têm só a semente 0.
- **Matéria nova em poucas famílias:** só 3 reservadas (2 do Felhen e 1 da Telepatia). As vistas da Telepatia
  têm só 84 grupos.
- **Oráculo otimista:** os números de "oráculo" ajustam e medem no mesmo conjunto. A receita final usa o ajuste
  honesto com 25 rótulos.
- **Mesmo `lr` em tudo,** e nenhuma busca de hiperparâmetros.
- **Felhen ainda é difícil:** acurácia de cerca de 0,65 na matéria nova, mesmo depois do treino curto.

## Em aberto (próximo ciclo)

- Confirmar a receita final no `test` oficial.
- Testar com mais famílias reservadas e ver se o ganho se repete.
- Ver se o RL de abstenção (H5) faz diferença quando o modelo já sabe a matéria, ou seja, depois da H9.
- Robustez ao texto do prompt: a receita responde 72% no prompt de sempre e só 50% num nunca visto (H13b). Treinar só com variantes não resolve sem custo; testar misturar mais o prompt de sempre, ou usar muito mais variantes.
- Conferir de onde veio o 37% publicado para o Felhen na receita final (hoje a mesma conta dá 35%).
- **GEPA + destilação de contexto (matéria nova com poucos rótulos), ideia guardada em 25/11:**
  - otimizar só a descrição da tarefa da família nova com 25 rótulos (GEPA, arXiv 2507.19457), com um modelo de
    reflexão aberto no cluster. O Clef não serve, porque não gera texto;
  - depois, destilar no Lumi: o Lumi com a descrição boa responde casos sem rótulo, e o Lumi com o prompt simples
    aprende a dar as mesmas probabilidades;
  - comparar com o treino curto de 100 rótulos (H9);
  - cuidados:
    - o paper não testou classificação com modelo pequeno;
    - a H8 mostrou que pôr informação no prompt calibrou, mas não ensinou;
    - o GEPA não resolve o usuário escrever do jeito dele. Isso é a H13.
- **Rodar o teste limpo da `regra_beneficio`** (`scripts/analise_falta_info_fatores.py`) tratando "confiança
  abaixo do limite" como "não sei". Isso mostra se esse "não sei" pega os casos em que falta informação ou só os
  que o modelo não aprendeu. O script precisa de uma opção nova para isso.
- **Só se a trilha do raciocínio for retomada** (ver "Decisão: parar a trilha do raciocínio"):
  - medir primeiro o Lumi oficial pensando, sem adaptador;
  - depois, resolver o comprimento do raciocínio e ensinar a confiança escrita antes do GRPO (RLCR).
