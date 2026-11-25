# Plano H13b: a receita final com o prompt variado

Registrado antes de rodar (25/11/2026).

## Pergunta

Na H13, treinar com o texto do prompt variado (`h4_aug`) deixou o Lumi mais robusto a um prompt que ele nunca
viu, mas isso foi antes do treino curto. Na H12c, um ganho medido nesse ponto sumiu dentro da receita.
**A robustez continua depois da receita?** E a receita com o prompt variado responde tanto quanto a de hoje?

## Desenho

- **Braço `h9ra_k100`:** a receita final com o prompt variado nas duas etapas:
  1. parte do `h4_aug` da mesma semente;
  2. treino curto igual ao `h9r_k100` (100 exemplos por família nova + replay, 3 épocas, perda log), também com
     `--variar-prompt true`;
  3. a mesma recalibração, com 25 rótulos.
- **Comparação:** contra o `h9r_k100` da mesma semente, com os mesmos grupos no treino curto. São 3 sementes.
- **Dois prompts:**
  - `dev`: o prompt de sempre;
  - `devvar`: a variante 4, nunca vista em treino nenhum (`slurm/h13_variante.sbatch`), avaliada nos dois braços.
- **Recalibração no prompt nunca visto:**
  - os 25 rótulos da família e a medida saem do `devvar`;
  - a temperatura das vistas continua vindo do `calib` do prompt de sempre;
  - o `h9_recalibrar.py` e o `analise_h9.py` ganharam `--split`.

## Critério

Análises em `slurm/analise_h13b.sbatch`.

1. **A receita de hoje é frágil?** Comparar o `h9r_k100` no `devvar` com ele mesmo no `dev`:
   - se a cobertura e a acurácia quase não mudam, e a massa nas letras fica perto de 1, o treino curto já
     resolveu a fragilidade, e o `aug` não é necessário.
2. **No prompt de sempre (`dev`), o `aug` não pode piorar a receita:**
   - a cobertura das reservadas fica a no máximo 2 pontos do `h9r_k100`;
   - o erro dos respondidos fica a no máximo 1 ponto;
   - as vistas ficam a no máximo 1 ponto de acurácia.
3. **No prompt nunca visto (`devvar`), o `aug` tem que ganhar:**
   - responde mais na receita, ou acerta mais, com IC fora do zero em pelo menos uma fonte;
   - e a massa nas letras é maior.

**Leitura:**
- **Entra na receita** se o item 2 passar e o item 3 ganhar.
- **Empate:** se o item 3 empatar, a receita fica como está, porque é mais simples.
- **Piora:** se o item 2 falhar, a receita também fica como está.

## Como rodar

Os treinos, as avaliações no prompt novo e a análise ficam encadeados:

```bash
cd /raid/$USER/lumi-decision && git pull -q && F=felhen:juristcu_area,felhen:factck_veracidade,telepatia:invoice_processing && J="" && for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s"); T=$(sbatch --parsable --time=04:00:00 slurm/e3_lora_h2.sbatch h9ra_k100$n --init-adapter runs/h4_aug$n/adapter --variar-prompt true --adapt-familias $F --adapt-k 100 --adapt-replay 1.0 --epochs 3 --seed $s) && V=$(sbatch --parsable --dependency=afterok:$T slurm/h13_variante.sbatch h9ra_k100$n) && R=$(sbatch --parsable slurm/h13_variante.sbatch h9r_k100$n) && J="$J:$V:$R"; done; sbatch --dependency=afterok$J slurm/analise_h13b.sbatch; squeue -u $USER
```

## Limites

- Só `dev`, com 3 sementes.
- Uma variante nunca vista só (a 4), escrita por nós.
- Na Telepatia, a matéria nova é uma família só.
