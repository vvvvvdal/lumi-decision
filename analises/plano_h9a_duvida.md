# Plano H9a: escolher os exemplos onde o Lumi está em dúvida (k = 25, com replay)

Registrado antes de rodar (21/11/2026).

## Pergunta

Com só 25 grupos por família nova, escolher os grupos em que o `h4_log` está **em dúvida** ensina mais que
sortear? E quanto do ganho de k = 100 sobra com k = 25 e replay?

## Desenho

- **Dois braços**, os dois com replay 1:1 e 3 épocas, a partir do `h4_log`:
  - `h9r_k25`: os 25 grupos são sorteados, como na H9;
  - `h9ar_k25`: a seleção é por dúvida (`--adapt-selecao duvida`). O script ordena os grupos do pool pela
    confiança média do `h4_log` (a probabilidade máxima nas duas ordens), pega a faixa dos 75 de menor
    confiança (`--adapt-faixa 3`) e sorteia 25 dentro dela. A faixa evita pegar só os casos mais duvidosos,
    que tendem a ser ambíguos ou ter rótulo ruim. A ideia vem do filtro de dificuldade do post "harness
    adapters" da Hugging Face.
- **De onde vêm as confianças do pool:**
  - Felhen: o `train` das reservadas, avaliado pelo `slurm/h9a_incerteza.sbatch` com 400 casos por família
    (`runs/h4_log_felhen_trainres`). O pool da seleção por dúvida fica limitado a esses casos;
  - Telepatia: o pool é o `dev`, e as predições `runs/h4_log_telepatia_dev` já existem.
- **Avaliação justa:** os dois braços treinam em grupos diferentes, então a análise tira os grupos de treino
  dos DOIS (`--excluir-tambem`). Na Telepatia, a seleção por dúvida tira do `dev` os casos difíceis, mas como
  eles saem dos dois braços, a comparação continua pareada.
- **Semente 0 primeiro.** Se a diferença for promissora, rodar as sementes 1 e 2 (cada uma com o `h4_log` da
  mesma semente e a sua própria avaliação do pool do Felhen).

## Leitura

- **Dúvida ajuda:** o `h9ar_k25` tem acurácia nas reservadas acima do `h9r_k25` em pelo menos uma fonte (IC
  acima de zero), sem perder na outra.
- **Empate:** escolher por dúvida não compensa o custo (uma avaliação extra do pool).
- **Dúvida atrapalha:** o `h9ar_k25` fica abaixo. Isso indica que os casos duvidosos são ambíguos ou ruidosos.
- **Pergunta secundária:** o ganho do `h9r_k25` contra o `h4_log`, comparado com o da H9 com k = 25 (+5,2 no
  Felhen e +10,6 na Telepatia, na semente 0, sem replay) e com o do replay com k = 100.
- Calibração: as duas seleções passam pela recalibração da H9b (`scripts/h9_recalibrar.py`).
