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
