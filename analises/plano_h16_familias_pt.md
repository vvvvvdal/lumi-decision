# Plano H16: mais famílias em português, para a matéria nunca vista

Registrado antes de rodar (25/11/2026).

## Pergunta

Sem nenhum exemplo de uma família nova, o Lumi acerta pouco. Treinar com **mais tipos de decisão em português**
melhora o desempenho nas famílias que ele nunca viu?

## O que já se sabe

- **O E3b treinou com o Eikos** (27 famílias, 87% em inglês), em `analises/e3br_replica.md`:
  - ajudou nas famílias sintéticas novas (res2, +3,4 nas 2 sementes);
  - ajudou nas 7 reservadas juntas (+1,7 a +2,8);
  - **não ajudou** nas 3 reservadas do Felhen e da Telepatia.
- O plano da fase 2 apontou o idioma como motivo provável.
- O braço "diverso" da fase 2 (intenção e emoção em português) foi desenhado. Se ele rodou, isso se confere na
  etapa 0, e o resultado é lido antes deste.

## Desenho

- **Braço `h4_pt`:** igual ao `h4_log`, com fontes a mais:
  - `--sources felhen,telepatia,traces,eikos_pt,f3`;
  - `eikos_pt`: a parte em português do Eikos;
  - `f3`: `massive_intencao` e `goemo_emocao`. A `pss_mais_simples` é reservada e fica fora.
- **O mesmo número de atualizações** do `h4_log` (`--max-steps`, lido de `runs/h4_log/treino_meta.json`). A
  única diferença é a variedade dos dados. O braço novo vê menos exemplos de cada fonte antiga.
- **Sementes:** 0 e 1 (`h4_pt`, `h4_pt_s1`), contra o `h4_log` e o `h4_log_s1`.

## Critério

- **Principal** (`comparar_h4.py --base log --novos pt --sementes 0,1`):
  - "ajuda" se a acurácia das reservadas subir com o IC fora do zero, nas duas fontes ou na média, sem perder
    mais de 1 ponto nas vistas;
  - "empate" se o IC passar pelo zero;
  - "piora" se as reservadas ou as vistas caírem.
- **Secundário:** as 4 famílias do res2 (`slurm/m2_res2.sbatch`), para comparar com o E3b.
- **Se ajudar:** rodar a receita a partir do `h4_pt` (H9r), porque a H12c mostrou que um ganho antes do treino
  curto pode sumir na receita.
- **Se ajudar e sobreviver:** gerar famílias sintéticas novas em português no `res2.py`, separadas das 4 de
  avaliação.

## Como rodar

```bash
cd /raid/$USER/lumi-decision && git pull -q && P=$(python3 -c "import json;print(json.load(open('runs/h4_log/treino_meta.json'))['passos_total'])") && echo "passos: $P" && for s in 0 1; do n=$([ $s = 0 ] && echo "" || echo "_s$s"); sbatch slurm/e3_lora_h2.sbatch h4_pt$n --sources felhen,telepatia,traces,eikos_pt,f3 --max-steps $P --seed $s; done; squeue -u $USER
```

## Limites

- Só 2 sementes, como no E3b.
- Na Telepatia, as reservadas são uma família só.
