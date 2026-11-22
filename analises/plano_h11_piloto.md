# Plano H11 (piloto): o Lumi consegue raciocinar antes de decidir?

Registrado antes de rodar (21/11/2026).

## Por que

No `lumi-decision`, a decisão é lida das letras numa passada só. Essa distribuição é derivável, então a perda
log já é o gradiente exato, e o RL só estima esse mesmo gradiente com ruído
(`tests/test_h6_bandit.py::test_bandit_log_tem_o_gradiente_do_log_score_em_media`). Por isso, nesse formato, o
RL no máximo empata com o supervisionado (H5, H6).

O RL passa a ser **necessário** quando algo entre a pergunta e a decisão é sorteado e não derivável. O caso
clássico é o raciocínio em texto (RLCR, Damani et al., 2025):
- o Lumi raciocina;
- a decisão é lida depois do raciocínio;
- a recompensa é uma regra própria (Brier) nessa decisão.

O laço de GRPO fica para o próximo ciclo. Este piloto só responde se vale a pena montá-lo.

## Desenho

- **Código:** `lumi-eval run --raciocinio N`.
  - O prompt abre o `<think>` (`render(..., pensar=True)`; se o template ignorar, `prompt_pensando` abre).
  - O Lumi gera até N tokens, de forma gulosa, e para no `</think>`.
  - A letra é lida depois do `</think>`, na mesma fronteira de sempre (`com_raciocinio`).
- **Run:** `slurm/h11_piloto.sbatch`, com o adaptador `runs/h4_log`, `dev`, ordem canônica, 8 casos por
  família, Felhen e Telepatia (vistas e reservadas), N = 256.
- **Comparação:** a leitura de sempre do `h4_log` nos mesmos casos (`runs/h4_log_<fonte>_dev`), feita por
  `scripts/h11_piloto.py`.
- **Teste curto antes:** `H11_LIMITE=4`, que gera as pastas `runs/h11p_teste_*`.

## Critério (vale montar o GRPO com o Lumi oficial?)

- **Raciocina:** pelo menos 80% dos raciocínios fecham o `</think>` antes do limite, e no máximo 10% degeneram
  (vazio, ou mais de 50% de 4-gramas repetidos).
- **A decisão não quebra:** nas vistas, a acurácia com raciocínio fica no máximo 5 pontos abaixo da leitura de
  sempre. O adaptador foi treinado com o `<think>` vazio, então alguma perda é esperada.
- **Cabe no orçamento:** tempo mediano de até 6 s por raciocínio. Assim, cerca de 2400 rollouts
  (300 casos × G = 4 × 2 épocas) cabem em menos de 4 h de geração.
- **Só informativo:** se o raciocínio muda a acurácia e o ECE, principalmente nas reservadas. Com 8 casos por
  família, o ECE é só indicativo.
- **Se não raciocinar:** o plano B do próximo ciclo é partir de um Lumi com raciocínio
  (`gustavobarrosss/t6-lumi-4b`).

## Nota (21/11/2026): 1º teste curto (job 37782) e a correção da leitura

- **O Lumi raciocina e acerta no texto.** Nos 4 casos vistos do Felhen, o raciocínio termina com a letra
  certa ("So answer N", "That's option E", "So answer F", "Thus answer: T"). 75% fecharam o `</think>`, nada
  degenerou, e o tempo mediano foi de 3,9 s.
- **A leitura colada no `</think>` quebrou.** A acurácia vista caiu para 0, e o NLL do Felhen (3,23) é
  praticamente o de um chute uniforme entre 25 opções (ln 25 = 3,22). Com raciocínio, o Lumi escreve um
  separador depois do `</think>` antes da resposta, então ler a letra colada é ler fora do lugar.
- **Correção (antes do 2º teste):**
  - a leitura principal passa a ser depois de `</think>\n\n`;
  - também lemos as variantes colada e `\n`, para comparar;
  - gravamos os 12 tokens que o Lumi escreve sozinho depois do `</think>`;
  - extraímos a letra que o próprio texto diz (`letra_do_raciocinio`).
- Pastas novas: `runs/h11p2_*`. O critério acima não muda e vale para a leitura principal. A acurácia pela
  letra do texto entra como informação.

## Nota (21/11/2026): 2º teste curto e o braço com sorteio (registrado antes do piloto inteiro)

**O 2º teste curto (`analises/h11p2_teste.md`, 4 casos vistos por fonte):**
- com a leitura depois de `</think>\n\n`, a acurácia vai de 1,0 para 1,0 no Felhen e de 0,25 para 0,5 na
  Telepatia;
- 88% dos raciocínios fecham, a mediana é 3,2 s, e `\n` e `\n\n` empatam;
- depois do raciocínio, o Lumi fica quase certo de tudo: NLL 0,000 no Felhen, e 2,14 na Telepatia, onde erra
  com confiança. Esse é o problema que o RLCR ataca.

**Por que sortear:** o GRPO não usa a geração gulosa. Ele sorteia G raciocínios por pergunta e aprende com a
diferença de recompensa entre eles. Os cards do Lumi-Think e do Nemotron recomendam temperatura 1,0 e top_p
0,95 com raciocínio.

**Desenho:**
- `--raciocinio-amostras 4 --temperatura 1.0 --top-p 0.95`;
- 4 raciocínios sorteados por caso, num lote só, com semente fixa por caso;
- cada um lido depois de `</think>\n\n`;
- recompensa = −Brier dessa leitura.

**Critério do sorteio (vale montar o GRPO?):**
- **Raciocina sorteando:** pelo menos 80% das amostras fecham o `</think>`, e no máximo 10% degeneram.
- **Há sinal para o GRPO:** em pelo menos 30% dos grupos, o desvio da recompensa é maior que 0,01. Abaixo
  disso, a maioria dos casos não ensina nada ao GRPO.
- **Cabe no orçamento:** com o tempo mediano por grupo, 2400 raciocínios (600 grupos de 4) ficam abaixo de
  4 h de geração.
- **Só informativo:**
  - a fração de grupos mistos (decisão certa e errada no mesmo grupo);
  - a acurácia, o ECE e o NLL das amostras.

## Nota (21/11/2026): 3º teste curto (sorteio) e o orçamento do piloto inteiro

**O 3º teste curto (`analises/h11p2_teste.md`, 4 casos vistos por fonte, G = 4, N = 256):**
- o lote de 4 funcionou: 5,6 s por grupo, e o GRPO projetado fica em cerca de 0,9 h;
- há sinal de sobra: 88% dos grupos têm desvio da recompensa maior que 0,01, e 75% são mistos;
- nenhuma amostra degenerou;
- **mas só 56% das amostras fecharam o `</think>` em 256 tokens**, contra 88% no guloso. Sorteando, o Lumi
  raciocina mais longo.

**Mudança antes do piloto inteiro:** N passa de 256 para **512** (`sbatch slurm/h11_piloto.sbatch runs/h4_log
512`). O critério não muda. A mudança só dá orçamento para o raciocínio sorteado terminar.

**Observação:** o Lumi escreveu `"\nB"` depois do `</think>`, então o separador natural parece ser `\n`, não
`\n\n`. A leitura principal continua `\n\n`, como registrado, e a tabela de variantes do piloto mostra se isso
muda alguma coisa. O GRPO do próximo ciclo escolhe a leitura com esses dados.

## Resultado do piloto inteiro (21/11/2026, `analises/h11p2.md` no cluster)

88 casos (8 por família), N = 512, G = 4. Leitura principal `\n\n`, como registrado.

| Critério | Limite | Resultado | |
|---|---|---|---|
| Guloso: fecha o `</think>` | ≥ 80% | 81% | passa |
| Guloso: degenera | ≤ 10% | 1% | passa |
| A decisão não quebra (vistas) | queda ≤ 5 pontos | Felhen −2,5; Telepatia −12,5; juntas −6,3 | **não passa** |
| Tempo por raciocínio | ≤ 6 s | 3,9 s | passa |
| Sorteio: fecha o `</think>` | ≥ 80% | 65% | **não passa** |
| Sorteio: degenera | ≤ 10% | 0% | passa |
| Sinal para o GRPO | ≥ 30% dos grupos | 66% | passa |
| Orçamento do GRPO | < 4 h | 1,7 h | passa |

**Pelo critério: ainda não vale montar o GRPO com o Lumi oficial do jeito que está.** O plano B (t6) não se
aplica, porque ele era para "não raciocina", e o Lumi raciocina.

Informativo:
- depois do raciocínio, o Lumi fica superconfiante: o NLL sobe de 0,36–0,96 para 0,98–4,24, e o ECE chega
  a 0,55;
- os raciocínios que não fecham se concentram onde ele sabe menos (Felhen reservadas e Telepatia vistas);
- `\n` fica um pouco melhor que `\n\n` no NLL nos 4 recortes.

Detalhes em `Resultados H11 - piloto do raciocínio.md`, na pasta `Decision Models`.

## Nota (21/11/2026): separar os que fecharam dos que não fecharam (decidida depois do piloto)

**Por que:** os dois pontos que falharam (a queda nas vistas e o sorteio que não fecha) podem ter a mesma
causa: o comprimento do raciocínio.

**O que:** `scripts/h11_piloto.py` separa, no guloso e no sorteio, os raciocínios que fecharam o `</think>`
dos que bateram no limite. Para cada grupo, mostra:
- a acurácia sem → com, nos mesmos casos;
- o NLL;
- a confiança média;
- os erros com confiança ≥ 0,9.

Só CPU, sem rodar a GPU de novo.

**Leitura (registrada antes de ver, na linha "todas", porque cada recorte tem poucos casos):**
- **O comprimento é a causa principal** se as duas condições valem no guloso:
  - nos que fecharam, a acurácia com raciocínio fica no máximo 5 pontos abaixo da leitura de sempre;
  - mais da metade dos erros confiantes (≥ 0,9) está nos que não fecharam.

  Nesse caso, o próximo ciclo começa pelo comprimento.
- **Senão, a superconfiança é do próprio raciocínio.** O GRPO com Brier é o remédio, e o comprimento é um
  problema a mais.
- É uma análise feita depois de ver os dados, com poucos casos. Ela orienta o desenho do próximo ciclo, mas
  não muda o critério do piloto.

**Resultado da separação (job 37786, linha "todas" do guloso):**

| Raciocínio | n | Acurácia sem → com | Confiança média | Erros com confiança ≥ 0,9 |
|---|---:|---|---:|---:|
| Fechou | 71 | 0,732 → 0,704 | 1,00 | 21 de 21 |
| Não fechou | 17 | 0,706 → 0,353 | 0,73 | 4 de 11 |

- A condição 1 passa (−2,8 pontos).
- A condição 2 falha: só 4 dos 25 erros confiantes (16%) estão nos que não fecharam.

**Pela regra: a superconfiança é do próprio raciocínio.**
- A queda de acurácia vem dos cortados.
- No sorteio, os que fecharam têm 64 de 65 erros confiantes.

**Ponto de atenção para o GRPO:** com as letras a 100% depois do raciocínio, o Brier das letras vira quase só
acerto ou erro. O próximo ciclo deve tirar a confiança de algo que o modelo consiga graduar, como a confiança
escrita do RLCR.
