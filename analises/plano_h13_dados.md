# Plano H13: variar o texto do prompt no treino (a parte de dados do Clef)

Registrado antes de rodar (24/11/2026).

## Por que

O Clef treina com dados sintéticos, em que trocam a ordem dos campos, os prompts e os formatos. A gente só sorteia a
ordem das opções. Hoje o Lumi treina sempre com **o mesmo texto de prompt**, e não sabemos:
- quanto ele piora com um prompt que nunca viu;
- se variar o texto no treino ajuda, tanto na robustez quanto na matéria nova.

## Desenho

- **Variantes do prompt** (`src/lumi_decision/prompts.py`, `_VARIANTES`):
  - **0:** o prompt de sempre, sem nenhuma mudança;
  - **1 a 3:** outros títulos e outra frase de pedido. Mudam também a ordem dos blocos (na 2, a pergunta vem antes)
    e o formato das linhas de opção (`A.`, `A)`, `A -`);
  - **4:** nunca entra no treino e serve só para medir a robustez. As opções vêm antes da pergunta, no formato `(A)`;
  - em todas, a resposta é a mesma: uma letra depois de `<think></think>`;
  - **Telepatia:** nas variantes 1 a 4, a ordem das chaves do JSON do caso também é sorteada.
- **Braço `h4_aug`** (`--variar-prompt true`):
  - a cada época, cada exemplo recebe uma variante de 0 a 3, sorteada por caso e por época;
  - todo o resto é igual ao `h4_log`: perda log, os mesmos exemplos, o mesmo número de passos, LoRA r = 16;
  - a única coisa que muda é o texto, então a comparação é limpa.
- **Sementes:** 0, 1 e 2 (`h4_aug`, `h4_aug_s1`, `h4_aug_s2`).
- **Prompt novo:** o `slurm/h13_variante.sbatch` avalia o `dev` com a variante 4 em `runs/<NOME>_<fonte>_devvar`,
  para o `h4_log` e para o `h4_aug`, nas 3 sementes.

## Critério

Diferenças com IC 95% por bootstrap de grupos (`scripts/comparar_h4.py`):

1. **Quanto o `h4_log` sofre com um prompt novo** (`--base log --novos log --split dev --split-novo devvar`):
   - acurácia e calibração no prompt novo contra o prompt de sempre;
   - se nada muda, o Lumi já é robusto, e a H13 só pode ajudar na matéria nova.
2. **No prompt de sempre** (`--base log --novos aug`):
   - **ajuda:** a acurácia das reservadas sobe com o IC fora do zero, sem piorar a calibração;
   - **empate:** os ICs passam pelo zero;
   - **piora:** a acurácia cai, ou a calibração piora.
3. **No prompt novo** (`--base log --novos aug --split devvar`): mesma leitura.

**Leitura:**
- se só o item 3 melhorar, o ganho é **robustez**, não saber mais a matéria;
- se as reservadas melhorarem no item 2, o ganho é **generalização**;
- conferir a massa das letras (`cand_mass`) e se a geração começa pela letra (`gen_first_is_readout_argmax`) no
  prompt novo. O modelo tem que continuar respondendo com uma letra.

## Como rodar

Teste curto (a perda tem que cair, e o prompt novo tem que rodar):

```bash
cd /raid/$USER/lumi-decision && git pull -q && E3_N=20 sbatch slurm/e3_lora_h2.sbatch h4_aug_curto --variar-prompt true --max-steps 40 --save-every 10
```

3 sementes:

```bash
cd /raid/$USER/lumi-decision && sbatch slurm/e3_lora_h2.sbatch h4_aug --variar-prompt true --seed 0 && sbatch slurm/e3_lora_h2.sbatch h4_aug_s1 --variar-prompt true --seed 1 && sbatch slurm/e3_lora_h2.sbatch h4_aug_s2 --variar-prompt true --seed 2
```

Prompt novo (o `h4_log` já pode rodar; o `h4_aug` só depois do treino):

```bash
cd /raid/$USER/lumi-decision && for n in h4_log h4_log_s1 h4_log_s2; do sbatch slurm/h13_variante.sbatch $n; done
```

## Limites

- **Variantes escritas por nós,** em português, e só 4 para o treino. O Clef não publica as dele.
- **O formato da resposta não muda** (sempre uma letra), por escolha: a leitura das letras é a base do projeto.
- **Só `dev`.**
