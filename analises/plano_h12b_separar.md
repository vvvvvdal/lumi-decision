# Plano H12b: separar o label smoothing do Brier na perda do Clef

Registrado antes de rodar (24/11/2026).

## Por que

Na H12, a perda do Clef (entropia cruzada com label smoothing ε = 0,1 + Brier) saiu pior que a `log` na
calibração e melhor na acurácia da Telepatia (+1 a +2 pontos). A perda tem duas peças, e não sabemos qual causa
cada efeito. O suspeito do custo é o smoothing: a temperatura do Clef foi 0,83, ou seja, ele saiu pouco
confiante, e na H4 o Brier sozinho empatou com a `log`. Mas isso não foi medido.

Se o ganho de acurácia vier do Brier e o custo vier do smoothing, dá para manter uma versão da perda do Clef sem
o smoothing.

## Desenho

Dois braços novos, cada um com uma peça só. Todo o resto é igual ao `h4_log` e ao `h4_clef`.

| Braço | Perda | O que tem |
|---|---|---|
| `h4_log` (já existe) | `log` | Entropia cruzada pura |
| `h4_suave` | `log_suave`, `--suavizar 0.1` | Só o smoothing (entropia cruzada com alvo suavizado, sem Brier) |
| `h4_logbrier` | `log_brier`, `--suavizar 0` | Só o Brier (entropia cruzada pura + Brier) |
| `h4_clef` (já existe) | `log_brier`, `--suavizar 0.1` | As duas peças |

3 sementes cada (0, 1 e 2). O `log_suave` foi criado agora em `lora.option_loss`: o ótimo dele é exatamente o
alvo suavizado `q_s = 0,9·q + 0,1/K` (testado em `tests/test_h4_perdas.py`).

## O que esperamos

- **Calibração:** o `suave` fica parecido com o `clef` (pior que a `log`, com temperatura abaixo de 1), e o
  `logbrier` empata com a `log`.
- **Acurácia da Telepatia:** não temos aposta. Uma das peças deve carregar o ganho, ou só a soma das duas.

## Critério

Diferença contra a `log` com IC 95% por bootstrap de grupos (`comparar_h4.py`), e a conferência pós-temperatura
(`resumo_calibracao.py`).

- **Custo da calibração:**
  - vem do smoothing se o `suave` repetir o padrão do `clef` (ECE ou NLL pior nas vistas do Felhen, com IC fora
    do zero) e o `logbrier` não repetir;
  - vem do Brier se acontecer o contrário;
  - vem da soma se só o `clef` repetir.
- **Ganho de acurácia da Telepatia:** fica com o braço que repetir o +1 a +2 pontos com IC fora do zero, nas
  vistas ou nas reservadas. Se nenhum braço sozinho repetir, o ganho vem da soma, ou é frágil.
- **Decisão sobre manter a perda do Clef:**
  - se o `logbrier` tiver o ganho de acurácia sem o custo de calibração, ele é a versão a manter, e o próximo
    passo é testá-lo dentro da receita final (H9);
  - se o ganho vier só do smoothing, ele vem junto com o custo. Aí a `log` continua, e a opção seria testar um
    ε menor (não registrado aqui).

## Como rodar

Sem teste curto: o `log_suave` passa pelo mesmo caminho do `log_brier`, que já rodou na H12, e os testes
locais cobrem a conta. 6 treinos:

```bash
cd /raid/$USER/lumi-decision && git pull -q && for s in 0 1 2; do n=$([ $s = 0 ] && echo "" || echo "_s$s"); sbatch slurm/e3_lora_h2.sbatch h4_suave$n --loss log_suave --suavizar 0.1 --seed $s; sbatch slurm/e3_lora_h2.sbatch h4_logbrier$n --loss log_brier --suavizar 0 --seed $s; done; squeue -u $USER
```

Comparação (sem GPU):

```bash
cd /raid/$USER/lumi-decision && srun -p h100n2 --cpus-per-task=2 --mem=8G bash -c 'source slurm/config.sh && lumi_exec python scripts/comparar_h4.py --base log --novos suave,logbrier,clef --sementes 0,1,2 --out analises/h12b_separar.md' && cat analises/h12b_separar.md && python3 scripts/resumo_calibracao.py --bracos log,suave,logbrier,clef --sementes 0,1,2
```

## Limites

- **Só ε = 0,1 e pesos 1:1,** sem busca.
- **Só `dev`.** A tabela pós-temperatura não tem IC.
