# H12: a perda do Clef (entropia cruzada com label smoothing + Brier) no Lumi

## Resposta curta

**Não vale trocar.** A perda do Clef acerta um pouco mais na Telepatia (+1 a +2 pontos), mas deixa a confiança
menos honesta no Felhen, e a temperatura não conserta tudo. A receita final continua com a perda log.

Agora a pergunta "fizemos o que o Clef faz?" tem resposta medida: a perda dele foi testada, com 3 sementes, e não
ganhou da nossa.

## O que foi testado

- **Perda `log_brier`:**
  - entropia cruzada com o alvo suavizado `q_s = (1 − ε)·q + ε/K`, com ε = 0,1;
  - mais o Brier com o alvo original;
  - pesos 1 e 1. O post do Clef não dá pesos nem ε, então a escolha é nossa.
- **Todo o resto igual ao `h4_log`:** dados, `lr`, épocas, LoRA r = 16.
- **3 sementes** (`h4_clef`, `h4_clef_s1`, `h4_clef_s2`), comparadas com as 3 do `h4_log` no `dev`.
- **Plano e critério registrados antes de rodar:** `analises/plano_h12_clef.md`.

## Números crus (sem temperatura)

Diferença `clef − log`, 3 sementes juntas, IC 95% por bootstrap de grupos (`scripts/comparar_h4.py`).

| Recorte | Acurácia | ECE | Brier | NLL |
|---|---|---|---|---|
| Felhen vistas | 0,871 → 0,872 (empate) | 0,011 → 0,047 (**pior**) | **pior** (+0,004) | 0,360 → 0,401 (**pior**) |
| Felhen reservadas | 0,529 → 0,522 (empate) | 0,065 → 0,054 (empate) | **pior** (+0,005) | 1,239 → 1,268 (**pior**) |
| Telepatia vistas | 0,732 → 0,742 (**melhor**) | 0,160 → 0,185 (**pior**) | **pior** (+0,005) | 0,698 → 0,722 (**pior**) |
| Telepatia reservadas | 0,582 → 0,600 (**melhor**) | 0,091 → 0,106 (**pior**) | empate | 0,919 → 0,919 (empate) |

Em negrito, diferenças com o IC 95% todo de um lado.

## Depois da temperatura

O `calibrar` do sbatch ajusta uma temperatura nas vistas do `calib` de cada fonte e aplica no `dev`
(`scripts/resumo_calibracao.py`, média das 3 sementes, sem IC).

| Recorte | Braço | Temperatura | ECE antes → depois | NLL antes → depois |
|---|---|---:|---|---|
| Felhen vistas | log | 1,06 | 0,015 → 0,014 | 0,360 → 0,362 |
| | clef | **0,83** | 0,048 → **0,028** | 0,401 → **0,379** |
| Felhen reservadas | log | 1,06 | 0,073 → 0,062 | 1,239 → 1,230 |
| | clef | 0,83 | 0,060 → **0,101** | 1,268 → **1,290** |
| Telepatia vistas | log | 0,41 | 0,160 → 0,040 | 0,698 → 0,591 |
| | clef | 0,40 | 0,185 → 0,043 | 0,722 → 0,589 |
| Telepatia reservadas | log | 0,41 | 0,111 → 0,189 | 0,919 → 1,201 |
| | clef | 0,40 | 0,107 → 0,183 | 0,919 → 1,164 |

O ECE "antes" desta tabela é a média do ECE de cada semente. O da tabela anterior junta as 3 sementes num só
cálculo. Por isso os dois diferem um pouco.

## Leitura

- **Critério registrado:** **Clef pior.**
  - Não é empate: a calibração crua piora em 9 das 12 células de ECE, Brier e NLL.
  - Também não é "Clef melhor": o ECE e o Brier das reservadas não caem.
- **Pelo motivo que a gente achava? Em parte, sim.**
  - O label smoothing ensina o modelo a nunca dar 100% para uma letra. Por isso o Clef sai **pouco confiante**:
    a temperatura dele no Felhen é 0,83, ou seja, ela precisa afiar as probabilidades. A do `log` é 1,06, quase
    nada.
  - Nas vistas do Felhen, a temperatura tira mais da metade da diferença (ECE 0,028 contra 0,014), mas não toda.
  - Nas reservadas do Felhen, a mesma temperatura **piora** o Clef (ECE 0,060 → 0,101). Ela é ajustada nas
    vistas, onde o modelo acerta muito, e afiar a confiança na matéria nova, onde ele erra metade, deixa o
    Clef superconfiante.
  - Na Telepatia, os dois braços precisam da mesma temperatura forte (cerca de 0,40). Depois dela, os dois
    empatam. A temperatura piora as reservadas da Telepatia nos dois braços, e isso já era conhecido: a matéria
    nova precisa da sua própria temperatura (H7).
- **A acurácia na Telepatia (+1 a +2 pontos) é o único ganho.**
  - A temperatura não muda qual letra é escolhida, então esse ganho fica.
  - Mas ele não aparece no Felhen, onde a acurácia das reservadas fica até um pouco abaixo (−0,7, empate).
  - Não sabemos se o ganho vem do smoothing, do Brier ou da soma dos dois. Uma explicação possível é o smoothing
    funcionar como regularização numa fonte com poucos grupos vistos (84). Isso não foi testado.

## O que isso muda

- **A receita final não muda:** a perda log continua.
  - A perda do Clef só renderia o ganho da Telepatia em troca de uma confiança pior no Felhen, mesmo depois da
    temperatura.
  - Na receita final, o treino curto na matéria nova (H9) já dá +20 pontos na Telepatia. O +1,8 do Clef é
    pequeno perto disso.
- **A tabela "Clef × nosso projeto" fica fechada na parte da perda.** O que falta do Clef é estrutural: posto 256,
  cabeça de roteamento (já testada pelo Palhares na H3, e foi pior) e dados sintéticos com prompts e formatos
  variados.

## Limites

- **ε = 0,1 e pesos 1:1 escolhidos por nós,** sem busca. Um ε menor deve ficar mais perto do `log`.
- **Não separamos as duas peças** (smoothing e Brier). Na H4, o Brier sozinho empatou com o log, então o
  suspeito principal do custo na calibração é o smoothing, mas isso não foi medido à parte.
- **Posto 16, não 256,** e um modelo de 4B.
- **Só `dev`.** A tabela pós-temperatura não tem IC.

## Como reproduzir

```bash
cd /raid/$USER/lumi-decision && sbatch slurm/e3_lora_h2.sbatch h4_clef --loss log_brier --suavizar 0.1 --seed 0 && sbatch slurm/e3_lora_h2.sbatch h4_clef_s1 --loss log_brier --suavizar 0.1 --seed 1 && sbatch slurm/e3_lora_h2.sbatch h4_clef_s2 --loss log_brier --suavizar 0.1 --seed 2
```

```bash
cd /raid/$USER/lumi-decision && srun -p h100n2 --cpus-per-task=2 --mem=8G bash -c 'source slurm/config.sh && lumi_exec python scripts/comparar_h4.py --base log --novos clef --sementes 0,1,2 --out analises/h12_clef.md'
```

```bash
cd /raid/$USER/lumi-decision && python3 scripts/resumo_calibracao.py --bracos log,clef --sementes 0,1,2
```
