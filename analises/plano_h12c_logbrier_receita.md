# Plano H12c: a perda `log_brier` dentro da receita final

Registrado antes de rodar (24/11/2026).

## Pergunta

Na H12b, a `log_brier` (entropia cruzada + Brier, sem smoothing) acertou +2,2 pontos nas reservadas da
Telepatia sem o custo de calibração do smoothing. Mas isso foi antes do treino curto, e na receita final a H9
já dá +20 pontos na Telepatia. **O ganho continua depois da receita?**

## Desenho

- **Braço `h9rb_k100`:** a receita final inteira com a `log_brier` (`--suavizar 0`) nas duas etapas:
  1. parte do `h4_logbrier` da mesma semente, e não do `h4_log`;
  2. treino curto igual ao `h9r_k100`: `--adapt-k 100`, `--adapt-replay 1.0`, 3 épocas e as mesmas 3 famílias
     reservadas;
  3. a mesma recalibração, com 25 rótulos da família.
- **Comparação:** contra o `h9r_k100` da mesma semente. Os grupos sorteados para o treino curto são os mesmos,
  porque vêm da mesma semente.
- **Sementes:** 0, 1 e 2 (`h9rb_k100`, `h9rb_k100_s1`, `h9rb_k100_s2`).

## Critério

- **Receita** (`scripts/h9_recalibrar.py --base h9r_k100 --novo h9rb_k100`, k = 25, alvo de 10%):
  - quanto cada braço responde e quanto erra entre os respondidos, nas reservadas.
- **Acurácia e esquecimento** (`scripts/analise_h9.py`, cada semente contra o seu `h9r_k100`):
  - a diferença nas reservadas e nas vistas, com IC.
- **"Sabe que não sabe"** (`scripts/analise_sabe.py --base h9r_k100 --novo h9rb_k100`):
  - o AUROC dentro da família não pode cair.

**Leitura:**
- **Entra na receita** se, nas duas fontes:
  - responde mais que o `h9r_k100` com o erro dos respondidos a no máximo 1 ponto de distância;
  - não esquece: nas vistas, a acurácia fica a no máximo 1 ponto do `h9r_k100`;
  - a temperatura das vistas continua consertando a calibração.
- **Empate:** a cobertura e a acurácia ficam dentro do ruído entre sementes. A `log` continua, porque é a mais
  simples.
- **Piora:** responde menos ou esquece mais. A `log` continua.

## Como rodar

```bash
cd /raid/$USER/lumi-decision && git pull -q && F=felhen:juristcu_area,felhen:factck_veracidade,telepatia:invoice_processing && for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s"); sbatch --time=04:00:00 slurm/e3_lora_h2.sbatch h9rb_k100$n --init-adapter runs/h4_logbrier$n/adapter --loss log_brier --suavizar 0 --adapt-familias $F --adapt-k 100 --adapt-replay 1.0 --epochs 3 --seed $s; done; squeue -u $USER
```

## Limites

- Só `dev`, com 3 sementes.
- Na Telepatia, a matéria nova é uma família só.
