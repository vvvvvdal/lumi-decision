# lumi-decision

Experimentos para adaptar o **Lumi-4B** (CEIA/UFG, `ceia-br/lumi-4b`) como *decision model* tipado em PT-BR:
dado um contexto, uma pergunta e um conjunto de opções, o modelo devolve uma distribuição sobre as opções
numa única passada, sem gerar texto (linha Jev / System One).

## Hipóteses

| | Hipótese | Comparação |
|---|---|---|
| **H1** | O Lumi congelado já decide com qualidade suficiente, e sua distribuição permite reduzir o erro entre as decisões aceitas quando ele pode se abster. | Readout congelado (A0) contra acaso e maioria; risco × cobertura contra aceitar a mesma fração ao acaso. Geração curta como referência operacional. |
| **H2** | LoRA melhora a decisão também em famílias inéditas. | A1 (LoRA, mesmo readout) contra A0, separando famílias vistas e reservadas. TF-IDF supervisionado como referência nas vistas. |
| **H3** | Uma pointer head melhora qualidade ou estabilidade em relação ao LM head com LoRA. | B (pointer head + LoRA) contra A1 com mesmos dados, passos e alvos. Só é empate se o IC 95% inteiro ficar dentro de ±2 pontos; IC que cruza a margem é inconclusivo. |

## Continuação: RLCD (H4 em diante, branch `h6-bandit`)

Depois de H1–H3, a mesma base foi usada para estudar RLCD, isto é, treinar o Lumi para que a confiança dele
seja calibrada. O resumo está em [`resultados/Resultados finais - RLCD no Lumi.md`](resultados/Resultados%20finais%20-%20RLCD%20no%20Lumi.md),
cada experimento em [`resultados/Experimentos.md`](resultados/Experimentos.md), o que ainda está rodando em
[`analises/planos_em_andamento.md`](analises/planos_em_andamento.md) e as fontes em
[`resultados/Referências - RLCD no Lumi.md`](resultados/Refer%C3%AAncias%20-%20RLCD%20no%20Lumi.md).

## Mecanismo de readout

O prompt termina em `<|im_start|>assistant\n<think></think>` (thinking desligado), exatamente onde a geração
do Lumi escreve a letra. As opções recebem letras `A`..`Z`; cada uma precisa ser um único token nessa posição
(checado caso a caso em `prompts.candidate_tokens`). A decisão é o softmax das log-probs dessas letras.
No piloto (`referencias/piloto40`), nessa posição, readout e geração gulosa escolheram a mesma opção em 80 de 80
casos. Inverter as opções mudou 10 de 40 decisões, e o acerto foi 29/40 na ordem canônica e 32/40 na invertida.

## Dados e splits (`configs/splits.toml`)

| Fonte | Papel | Reservado para generalização |
|---|---|---|
| Felhen PT-BR Typed Decisions Bench | 7 tarefas brasileiras, avaliação principal | tarefas `juristcu_area`, `factck_veracidade` |
| Typed Decisions PT (telepatia) | workflows com distribuição de referência; as 5 perguntas de um estado ficam no mesmo split | workflow `invoice_processing` |
| Tool Calling Traces PT-BR | escolher ferramenta ou responder direto (menu de 5 com distratoras). **Só diagnóstico**: em 6 dos 9 casos reservados do dev a ferramenta correta é a única reservada do menu (`data.traces_shortcut_stats`); conclusão sobre ferramentas inéditas vem do benchmark do time | tipos `viagens`, `saude` |

- Revisões fixadas em `src/lumi_decision/data.py`. `lumi-dados baixar` grava `data/manifesto.json` com SHA-256.
- Splits: `train`, `dev`, `calib` (temperatura e limiares), `test` (oficial, não usado para nenhuma escolha).
- Famílias reservadas nunca entram em treino (`data.training_cases`) nem em calibração. O dev delas mede transferência.
- Grupos são o **texto normalizado** (o mesmo comentário aparece nas duas tarefas OLID). Um texto nunca fica em dois
  splits ativos. Linhas de treino que repetem texto de dev/teste oficial ou de família reservada viram `descartado`
  (445 nas OLID, 8 no Traces); o teste oficial fica intacto.
- Limiares de abstenção e temperatura são escolhidos em `calib` e aplicados fixos em `dev`/`test`
  (`lumi-eval calibrar`); a curva risco × cobertura no próprio dev é só diagnóstico. O Traces tem `calib` (10% do treino visto).
- `lumi-dados resumo` mostra as contagens e falha se um grupo cruzar splits.

## Onde roda o quê

| Trabalho | Onde |
|---|---|
| Dados, prompts, avaliador, métricas, testes, iteração de template com GGUF | PC local (`--backend llamacpp`, Vulkan) |
| Números oficiais (checkpoint BF16), E0, E1, treinos (E2–E4), Nemotron base, Saracura/Eikos | Cluster H100 via Slurm (`--backend hf`) |

## Uso

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,baselines]"
lumi-dados baixar
pytest -q

# Local (GGUF via llama.cpp)
lumi-eval run --backend llamacpp --model lumi-4b-Q6_K_XL.gguf --server-bin llama-server.exe \
  --source felhen --split dev --orders canonical,reversed --out runs/h1_felhen_gguf
lumi-eval baselines --source felhen --split dev --out runs/baselines_felhen_dev   # maioria, TF-IDF, lexical (sobre o menu)
lumi-eval calibrar --calib runs/X_calib --alvo runs/X_dev --risco 0.10   # T e limiar no calib, fixos no dev
lumi-eval comparar runs/A runs/B          # IC 95% pareado, bootstrap por grupo

# Cluster DGX (partição h100n2), sempre da raiz do repo em /raid/$USER/lumi-decision
mkdir -p logs
sbatch slurm/00_preparar_ambiente.sbatch  # imagem Apptainer, venv, kernels, dados, checkpoints, testes
sbatch slurm/e0_sanidade.sbatch          # falha se nenhuma implementação passar
LUMI_IMPL_FLAG=--native sbatch slurm/e1_h1_dev.sbatch lumi 0   # E1 (também: nemotron 0; N>0 = curto)
sbatch slurm/e2_lora.sbatch              # E2: sanidade do LoRA + memória; falha se reprovar
sbatch slurm/e3_lora_h2.sbatch e3_lora_r16   # E3: treina, avalia dev/calib, calibra e aplica o critério da H2
sbatch slurm/e3_lora_h2.sbatch e3b_eikos_s0 --sources felhen,telepatia,traces,eikos   # E3b
sbatch slurm/e3_lora_h2.sbatch e3b_ctrl_s0 --epochs 2 --max-steps 891                   # controle do E3b
sbatch --dependency=afterok:<E3b>:<ctrl> slurm/e3b_comparar.sbatch                  # critérios do E3b
squeue -u $USER
```

### Regras do cluster aplicadas nos scripts

- Tudo (repo, venv, caches do HF/pip/torch/triton, imagens `.sif`, checkpoints) fica em `/raid/$USER`, nunca na home (`slurm/config.sh`).
- Nada usa GPU fora do Slurm: preparação do ambiente, E0 e E1 são `sbatch`.
- Execução dentro de container Apptainer (`apptainer exec --nv`), imagem NGC PyTorch por padrão
  (`LUMI_NGC_IMAGE` em `slurm/config.sh`, `pytorch:26.04-py3`, com wheels prontas de `mamba-ssm` para ela).
- Jobs de avaliação são reentrantes: pulam casos já gravados, mas só se a configuração (modelo, revisão,
  implementação, prompt, template, splits, casos selecionados, ordens e adaptador) for a mesma; senão recusam a pasta.
- O treino (E3) salva checkpoint a cada 100 passos e ao receber o SIGUSR1 que o Slurm manda 10 min antes do limite
  (`--signal=B:USR1@600`; o Apptainer não repassa o sinal ao Python, então o sbatch cria `runs/<nome>/PARAR`, que o treino checa a cada passo); ressubmeter o mesmo comando retoma. O checkpoint é uma pasta
  trocada de uma vez (pesos, otimizador, agendador, RNG, passo, config, hash dos dados) e a retomada recusa outra
  config ou outros dados. Interromper e retomar dá o mesmo adaptador, bit a bit (teste em CPU).

Cada execução grava `predicoes.jsonl` (uma linha por caso × ordem), `meta.json` (backend, modelo, versão do
prompt, commit) e `relatorio.{json,md}` (acurácia, acurácia balanceada por família, NLL, Brier, ECE,
risco × cobertura, sensibilidade à ordem, latência).

## Experimentos

| | Objetivo | Estado |
|---|---|---|
| Piloto | 40 casos de dev, GGUF Q4 local | feito; reproduzido pelo repo em `referencias/piloto40` |
| E0 | Sanidade do BF16 oficial. Obrigatório: pesos sem faltas, template, estado do readout = entrada do LM head, conteúdo do padding não vaza, cache × sem cache, parada em EOS | feito (job 36805): implementação nativa aprovada; o remote code quebra no `generate` com transformers 5.x |
| E1 | H1: dev completo das três fontes, duas ordens, Lumi e Nemotron base (rev. `dfaf35d`) | feito (jobs 36807/36808); análise em `scripts/analise_e1.py` |
| E2 | Sanidade do LoRA: alvos, kernel fundido do Mamba acionado no treino (forward com `use_cache=False`), gradientes finitos e não nulos (lora_A só depois do 1º passo, pois B começa em 0), sobreajuste em 64 exemplos, readout de treino = HFBackend, recarga; memória por micro-lote | feito (job 36812): aprovado; kernel fundido em 21/21 camadas no treino (0/21 com cache); 8192 tokens/micro-lote = 35 GB (16384 = 72,8 GB, sem ganho de vazão) |
| E3 | H2: LoRA (r=16) no readout das letras, Felhen + Typed Decisions PT + Traces, só famílias vistas, teto de 4 mil por família (20.350 exemplos), ordem sorteada por exemplo, alvo suave quando há `gold_dist`; perda = entropia cruzada normalizada entre as opções; atualizações montadas do plano real de micro-lotes, perda média sobre o lote efetivo | feito (job 36852, 508 passos, 37 min). Parada e retomada validadas (36849 → 36851). H2: +2,7 pp [−0,3; +5,9] nas reservadas, **inconclusivo** (factck +2,9; juristcu +0,2; invoice +4,8). Vistas: Felhen +14,7 [+12,8; +16,6], Telepatia +20,2 [+15,7; +25,0]. Decisão que muda com a ordem: Felhen vistas 18% → 3%, reservadas 32% → 12% |
| E3b | Ablação de dados: receita do E3 + Eikos filtrado (16.191 linhas, 891 atualizações), contra um controle só com os dados do E3 no mesmo nº de atualizações e cronograma de lr (2 épocas cortadas em 891). Protocolo abaixo | feito (jobs 36857/36858/36859). Principal E3b − controle: −0,8 [−3,1; +1,3], **inconclusivo** (factck −0,5; juristcu +2,3; invoice −4,3). Secundária E3b − congelado: +1,3 [−1,8; +4,4], inconclusivo. A queda em invoice vem toda de `duplicate` (E3b diz "sim" em 557 de 600; só 56 são duplicatas); nas outras 4 perguntas de invoice o E3b supera o controle. Vistas: sem diferença |
| Diag. | Intenções em PT-BR com MASSIVE (oficial é pt-PT; há localização pt-BR comunitária feita com LLM) | a fazer |
| E4 | H3: pointer head + LoRA (B) contra o E3 (A1), mesmos dados, alvos, ordens e 508 atualizações. Protocolo abaixo | feito (sanidade 36884 com RNG da CPU e da CUDA intactos; treino 36885, 28 min; comparação 36886). Qualidade: **piora**, −4,4 [−7,2; −1,6] (factck −17,6; juristcu +3,8; invoice +0,7). Estabilidade: **piora**, +4,7 [+0,6; +8,7] (factck 11% → 45% de decisões que mudam com a ordem; invoice 28% → 9%). Vistas: Felhen +0,4, Telepatia +2,6 [+0,5; +4,6] |

**Abordagem escolhida (13/11/2026): A1** (LoRA r=16 lido pelas letras, receita do E3, adaptador em
`runs/e3_lora_r16/adapter`), com temperatura e limiar do `calib`. Justificativa em
`analises/resultados_e0_e3.md`, seção "Decisão".

### Métrica principal e critério da H2 (fixados antes do treino)

- Métrica principal: média das acurácias por ordem (original e invertida, as mesmas do E1). A ordem original
  sozinha é enviesada: na Felhen reservada o Lumi congelado acerta 54,7% na original e 45,5% na invertida.
- H2: média, com peso igual, de acc(LoRA) − acc(Lumi congelado) nas três famílias reservadas
  (`factck_veracidade`, `juristcu_area`, `invoice_processing`), sobre as duas ordens; IC 95% pareado por
  (caso, ordem), reamostrando grupos de texto/estado dentro de cada família (`lumi-eval h2`). Ganho se o IC
  inteiro ficar acima de zero. Traces é só diagnóstico. Os dois lados precisam ter exatamente os mesmos pares,
  sem repetição, com o mesmo gabarito, a mesma ordem apresentada e o mesmo prompt, e as três famílias presentes.

### Protocolo do E3b (registrado em 13/11/2026, antes de rodar)

**Pergunta:** acrescentar um corpus externo e variado de decisões (Eikos) melhora a transferência para
famílias inéditas, além do que dá treinar o mesmo tempo só com os dados do E3?

**Dados do Eikos** (`lumi-dados auditar-eikos`, saída em `analises/eikos_auditoria.json`):
- revisão `cebb4a26` (`main` depois do PR #2, `770d6bdf`, que tirou o FinEntity), `core/train`, 19.239 linhas,
  sha256 fixado em `data.EIKOS_SHA256`;
- exclusões por semelhança de tarefa com as reservadas (filtro `eikos-filtro-v1`, regex sobre tópico,
  instrução e estado): faturas, pedidos de compra, contas a pagar e conferência fatura × pedido (2.886);
  checagem de fatos e veracidade (157); TCU e acórdãos (5);
- sobreposição de texto com dev/teste/calib/reservadas das nossas fontes (texto igual ou 13 palavras
  seguidas em comum): 0;
- ficam 16.191 linhas (14.033 em inglês, 2.158 em PT-BR), 27 famílias, todas abaixo do teto de 4 mil.

**Braços** (semente 0; mesma receita do E3 em todo o resto):

| Braço | Dados | Atualizações | Tokens de conteúdo |
|---|---|---|---|
| E3b (`e3b_eikos_s0`) | E3 + Eikos, 1 época (36.541 exemplos) | 891 | 16,1 M |
| Controle (`e3b_ctrl_s0`) | só E3, 2 épocas cortadas em 891 (`--epochs 2 --max-steps 891`; 35.825 exemplos) | 891 | 14,2 M |

O cronograma de lr (aquecimento + cosseno) depende só do total de atualizações, então os dois braços têm o
mesmo cronograma. O controle repete cerca de 75% dos exemplos do E3 numa segunda época, com outra ordem das
opções.

**Critério** (mesma métrica e mesmo bootstrap da H2):
- **Principal:** E3b − controle, média com peso igual das três reservadas, duas ordens. O Eikos ajuda se o
  IC 95% inteiro ficar acima de zero; atrapalha se ficar inteiro abaixo; senão, inconclusivo.
- **Secundária:** E3b − Lumi congelado (a H2 de novo, com a receita ampliada).
- **Diagnósticos, sem decisão:** E3b − E3 original; controle − congelado; famílias vistas; viés de ordem;
  abstenção; `invoice_processing` por pergunta, com acurácia balanceada (`duplicate` em especial).
- **Limites da conclusão:** uma semente; o Eikos muda junto a variedade de tarefas e o idioma (87% inglês).
  Um resultado positivo justifica replicar com mais sementes e um braço só com o Eikos em PT-BR
  (fonte `eikos_pt`), nunca escolher configuração olhando o dev.

### Protocolo do E4 (registrado em 13/11/2026, antes de rodar o treino)

**Pergunta (H3):** trocar a leitura pelas letras (LM head) por uma pointer head treinada melhora a qualidade
ou a estabilidade da decisão, com tudo o mais igual? O resultado compara duas **receitas** treinadas: a da
pointer head inclui a inicialização e a lr próprias da cabeça.

**Referência (A1) = E3 original** (`runs/e3_lora_r16`, 508 atualizações). O Eikos fica fora.

**O que fica igual ao E3, conferido localmente:**
- mesmos 20.350 exemplos e mesmos ids de tokens;
- mesmos alvos e mesmas ordens sorteadas (`shuffle0`);
- mesmo plano de micro-lotes e as mesmas 508 atualizações;
- mesmo prompt (com as letras na lista de opções) e mesma perda (entropia cruzada entre opções);
- mesmo LoRA: r=16, mesmos módulos, lr 1e-4, mesmo cronograma, semente 0. A cabeça nasce na CPU, com só o
  gerador da CPU semeado e depois restaurado (`ponteiro.new_head`). O gerador da CUDA não é tocado, então o
  LoRA nasce com os mesmos pesos do E3 e o dropout na GPU segue a mesma sequência. A primeira versão usava
  `torch.manual_seed`, que também resemeia a CUDA (revisão 7); a sanidade agora confere CPU e CUDA.

**A cabeça** (`ponteiro.py`), no molde do `PointerHead` do kev (`kev/model.py`):
- `s_i = (W_q h_q) · (W_k h_i) / √256`, com `W_q`, `W_k` lineares 3136 → 256 com viés (cerca de 1,6 M
  parâmetros), em FP32;
- `h_q` é o estado final na última posição do prompt (o ponto em que o A1 lê as letras);
- `h_i` é o estado final no token que contém o último caractere da opção i. O token pode avançar sobre a
  quebra de linha seguinte, mas só sobre espaço ou quebra de linha. Em 1.800 prompts (13.185 opções) das três
  fontes, nas ordens canônica, invertida e sorteada, não houve falha, e 97,7% desses tokens cobrem a quebra de
  linha. O kev usa um token delimitador próprio (`</opt>`); aqui o prompt fica idêntico ao do E3;
- a distribuição é o softmax dos `s_i`.
- **lr da cabeça = lr do LoRA (1e-4).** É o padrão do kev (`--head_lr 0` significa "igual à lr"; o card do Kev-4B
  lista uma lr só, 5e-5). Nas rodadas de ajuste completo com a cabeça começando do zero, o kev usou 1e-4 na
  cabeça. A proposta anterior
  (1e-3) não tinha apoio na referência.

**O que o kev isola, e o que não dá para isolar aqui:** o kev isola **perguntas** que compartilham um
contexto; as opções de uma mesma pergunta interagem (o README do kev diz que trocar a ordem das opções pode
mudar a resposta). A opção `option_isolation` do kev só existe para backbones só de atenção e é recusada em
backbones híbridos. O Lumi é híbrido (21 camadas Mamba), então o E4 usa uma passada única, como o kev faz nos
modelos híbridos, e a sensibilidade à ordem é medida.

**Sanidade antes do treino** (`slurm/e4_sanidade.sbatch`, todas obrigatórias):
- posições das opções;
- gradientes finitos e não nulos (cabeça e lora_B no passo 0; tudo depois);
- kernel fundido do Mamba em 21/21 camadas no treino;
- sobreajuste em 64 exemplos;
- leitura no treino = leitura do backend;
- recarga conjunta de adaptador e cabeça;
- RNG intacto.

**Avaliação:** dev e calib das três fontes, ordens canônica e invertida, uma passada por ordem nos dois
modelos, mesmas linhas pareadas (ids, gabarito, ordem apresentada e hash do prompt conferidos). Sem geração no
B (`EVAL_EXTRA=--no-generate`).

**Critério** (`lumi-eval h3`, mesmo bootstrap da H2: grupos de texto/estado reamostrados dentro de cada família):
- **Guarda:** o `h3` recusa a comparação se alguma predição do B não tiver sido lida pela pointer head, ou se
  alguma do A1 não tiver sido lida pelas letras. Sem `ponteiro.safetensors` o backend voltaria às letras em
  silêncio.
- **Principal, qualidade:** B − A1 em acurácia, média com peso igual das três reservadas, duas ordens. Direção
  e equivalência são afirmações separadas e as duas são registradas:
  - **direção:** melhora se o IC 95% inteiro ficar acima de zero; piora se ficar inteiro abaixo;
  - **equivalência:** os dois são equivalentes se o IC inteiro ficar dentro de ±2 pontos;
  - **rótulo final:** melhora ou piora quando há direção; empate quando só há equivalência; inconclusivo nos
    outros casos. Um IC de [+0,3; +1,7] é ao mesmo tempo ganho detectável e equivalência prática.
- **Secundário, estabilidade:** para cada caso, 1 se a escolha muda entre as duas ordens. B − A1 com a mesma
  média por família e os mesmos grupos reamostrados nos dois modelos. Conta como melhora só se o IC da
  diferença ficar inteiro abaixo de zero **e** o limite inferior do IC de qualidade ficar acima de −2 pontos.
- **Secundárias:** o mesmo nas famílias vistas; NLL e ECE.
- **Diagnósticos, sem decisão:** A1 com duas passadas (média das probabilidades das duas ordens, o dobro do
  custo); abstenção com limiar do calib; `invoice_processing` por pergunta, com acurácia balanceada, falsos
  positivos e AUROC em `duplicate`; curva de perda.
- **Alcance:** uma semente; a conclusão vale para estes modelos. Resultado positivo justifica replicar com
  mais sementes nos dois braços.

**Execução:** `sbatch slurm/e4_sanidade.sbatch`; se aprovar,
`EVAL_EXTRA=--no-generate sbatch slurm/e3_lora_h2.sbatch e4_ponteiro_s0 --readout ponteiro` e
`sbatch --dependency=afterok:<job> slurm/e4_comparar.sbatch e4_ponteiro_s0`.

## Achados técnicos já verificados

- O Jinja padrão deixa um `\n` antes de `<|im_start|>` neste template; o transformers (trim/lstrip_blocks)
  não. Seguimos o transformers. No piloto, essa diferença mudou 4 de 80 decisões, todas em quase-empates.
- No código remoto do Lumi, o bloco não repassa `attention_mask` aos mixers: usar padding à direita.
- No treino com kernel fundido, `out_proj` do Mamba recebe o peso direto, então LoRA nesse módulo não treina.
- `transformers>=5` tem `nemotron_h` nativo; o E0 compara remote code e implementação nativa.
- Na geração com GGUF, todas as respostas terminaram no token 11 (`<|im_end|>`), mas o config do HF diz EOS 2.
  O E0 verifica isso no BF16.
- O chat template do Nemotron base (rev. `dfaf35d`) é idêntico ao do Lumi.
- Com a imagem NGC 26.09 (torch 2.14, headers C++20) o `mamba-ssm` não compila (job 36783). A 26.04 tem wheels prontas.
- A QOS `onejob` do cluster permite 2 jobs simultâneos por usuário.
- Implementação nativa (`transformers.models.nemotron_h`): LM head em BF16 (nosso readout projeta em FP32,
  diferença ~0,1 em log-prob); máscara com zeros muda o caminho numérico (~0,09), mas o conteúdo do padding
  não vaza (0,0). No treino, o caminho fundido também passa `out_proj.weight` direto ao kernel.
- No Apptainer, o Triton precisa de `TRITON_LIBCUDA_PATH=/.singularity.d/libs` (feito no `lumi_exec`).
- O config do Lumi tem `use_cache: true`; no transformers 5.22 o `NemotronHModel.forward` herda isso e cria cache,
  e com cache o Mamba não usa o kernel fundido de treino. O forward de treino passa `use_cache=False`.
