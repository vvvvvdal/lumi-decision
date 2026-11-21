# Plano H9m: misturar o `h4_log` com a H9 em vez de fazer replay

Registrado antes de rodar (21/11/2026).

## Pergunta

Ficar **entre** o adaptador antigo (`h4_log`) e o ajustado (`h9_k100`) evita o esquecimento nas vistas da
Telepatia e mantém a maior parte do ganho nas reservadas, sem treinar de novo? A ideia vem do post "harness
adapters" da Hugging Face (2026) e do WiSE-FT (Wortsman et al., 2022).

## Desenho

- `scripts/misturar_adaptadores.py` faz uma mistura exata, ΔW = (1 − w)·ΔW_h4 + w·ΔW_h9:
  - concatena os adaptadores no posto, que passa de r para 2r;
  - ajusta o alpha para manter a escala;
  - confere ΔW no próprio cluster.
- O `slurm/h9m_mistura.sbatch` mistura e avalia `dev` e `calib` de Felhen e Telepatia.
- **Etapa 1 (semente 0):** w = 0,5 e 0,75 (`h9m50_k100`, `h9m75_k100`). Os pontos w = 0 (`h4_log`) e
  w = 1 (`h9_k100`) já existem.
- **Etapa 2:** o melhor w, pelo critério abaixo, roda nas sementes 1 e 2. Escolher w no `dev` da semente 0 é
  otimista, então essas sementes servem de confirmação.
- Análise: `scripts/analise_h9.py` contra o `h4_log` da mesma semente (os grupos do treino curto ficam fora)
  e `scripts/h9_recalibrar.py`.

## Leitura

- **Mistura resolve:** nas vistas da Telepatia, a acurácia fica a no máximo 1 ponto do `h4_log`. Nas
  reservadas, o ganho fica a no máximo 3 pontos do ganho da H9 nas duas fontes.
- **Escolha de w:** o maior w que cumpre a regra das vistas.
- Comparar com o replay (H9r): o replay custa um treino novo, e a mistura custa só avaliação.
