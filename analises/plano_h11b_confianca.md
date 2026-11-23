# Plano H11b (piloto): o Lumi escreve uma confiança graduada depois de raciocinar?

Registrado antes de rodar (22/11/2026).

## Por que

No piloto da H11, quando o raciocínio termina, a leitura das letras sai com confiança de 1,00. Os 21 erros do
guloso tinham confiança de pelo menos 0,9, e no sorteio foram 64 de 65 (`analises/plano_h11_piloto.md`). O
raciocínio termina dizendo a resposta, e as letras só copiam essa conclusão.

Por isso, se a recompensa do GRPO for o Brier das letras, ela vira quase só "acertou ou errou". Essa é a
recompensa que estragou a honestidade na H4.

No RLCR (Damani et al., 2025), o modelo **escreve um número de confiança** depois da resposta, e a recompensa
soma o acerto e o Brier desse número. Este piloto pergunta se o Lumi, do jeito que está, já escreve um número
que muda com a dúvida e é mais baixo quando ele erra.

## Desenho

- **Sem raciocínio novo:** o piloto reaproveita os raciocínios do piloto da H11 (`runs/h11p2_<fonte>_dev`, 88
  casos, adaptador `h4_log`).
- **Remontagem:** o prompt em que a letra foi lida é remontado e conferido pelo `prompt_sha1` gravado na H11.
- **A pergunta:** depois da letra escolhida, entra um **turno novo do usuário**:
  "Qual é a probabilidade, de 0 a 100, de que a sua resposta esteja correta? Responda apenas com o número."
  O Lumi responde no modo de sempre (`<think></think>`), de forma gulosa, em até 16 tokens.
- **O formato do turno** sai do próprio template (`prompts.turno_confianca`).
- **Os sorteados também:** o mesmo vale para os 4 raciocínios sorteados de cada caso.
- **Código e saída:**
  - `scripts/h11b_confianca.py`, `slurm/h11b_confianca.sbatch`;
  - pastas `runs/h11b_<fonte>_dev`, relatório `analises/h11b.md`.

**Comparação, nos mesmos casos:**
- a confiança escrita;
- as letras depois do raciocínio;
- a consistência dos 4 sorteios (fração que escolheu a mesma letra);
- as letras sem raciocínio da H4, só como referência, porque julgam outra decisão.

**Métricas:**
- AUROC, a chance de um acerto ter confiança maior que um erro;
- a confiança média quando acerta e quando erra;
- ECE e Brier binários.

## Critério (no guloso, com os 88 casos)

**A confiança escrita serve de ponto de partida para o GRPO estilo RLCR?**
- **Escreve:** pelo menos 90% das respostas trazem um número de 0 a 100.
- **Gradua:** o valor mais comum aparece em no máximo 80% das respostas com número.
- **Separa:** AUROC da confiança escrita de pelo menos 0,65.
- **Só informativo:**
  - a comparação com as letras e com a consistência;
  - o ECE e o Brier;
  - no sorteio, se a confiança escrita muda dentro do grupo, e a fração de grupos com sinal para a recompensa do
    RLCR (acerto − (c − acerto)², desvio > 0,01), comparada com o Brier das letras.

## Leitura

- **Passa tudo:** o próximo ciclo usa o formato do RLCR (resposta + confiança escrita, recompensa = acerto +
  Brier da confiança), partindo do Lumi como está.
- **Escreve, mas não gradua ou não separa:** o formato funciona, mas a confiança terá de ser aprendida. O
  caminho é o GRPO com o Brier na confiança escrita desde o começo, ou um SFT curto do formato antes.
- **Não escreve:** o problema é de formato. O ajuste é na pergunta ou no SFT do formato.

## Limites

- **A confiança é perguntada depois, num turno novo.** No RLCR, ela sai na mesma geração do raciocínio, e o
  modelo pode pensar na dúvida antes. Este piloto mede se o Lumi julga o próprio raciocínio quando perguntado.
- **Poucos casos:** 88, com cerca de 25 erros no guloso. O AUROC tem erro de cerca de ±0,06.
- **O adaptador `h4_log` nunca viu um segundo turno.**

## Teste curto antes

`H11B_LIMITE=4` (pastas `runs/h11b_teste_*`). Conferir:
- o fim do prompt, que deve ter a letra, o fim do turno e a pergunta;
- as respostas cruas;
- se o `prompt_sha1` bate (senão, o job para com erro);
- o tempo por resposta.

## Nota (23/11/2026): teste curto (job 37794, `analises/h11b_teste.md`)

8 casos (4 vistos por fonte), com os 4 sorteados de cada um.

**A mecânica funciona:**
- o prompt termina como planejado: o raciocínio, `</think>\n\n`, a letra, `<|im_end|>`, a pergunta nova e
  `<|im_start|>assistant\n<think></think>`;
- o `prompt_sha1` bateu em todos os casos (o job não parou);
- 100% das respostas trazem um número;
- 0,27 s por resposta.

**Primeiros sinais** (8 casos, só indicativos; o critério vale no piloto inteiro):
- a confiança escrita fica sempre em 90 ou mais, e 75% das respostas são "100";
- no guloso, ela não separa acerto de erro (AUROC 0,30 com 3 erros);
- nos 32 sorteados, separa um pouco (0,71);
- a consistência dos 4 sorteios separou os 8 casos (AUROC 1,0).

Nada muda no desenho. O piloto inteiro roda com `sbatch slurm/h11b_confianca.sbatch` (88 casos, pastas
`runs/h11b_*`).
