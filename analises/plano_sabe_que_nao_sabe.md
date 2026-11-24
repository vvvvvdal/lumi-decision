# Plano: o Lumi sabe o que não sabe?

Registrado antes de rodar (24/11/2026).

## A pergunta

Na receita final, o Lumi responde 72% das faturas novas com 10% de erro e diz "não sei" no resto. Esse "não sei"
acontece **porque ele sabe que erraria**? Ou só porque a matéria nova inteira é difícil?

## O que medir

`scripts/analise_sabe.py`, sem GPU, com as predições que já existem:

- **Modelos:** `h4_log` e `h9r_k100` (a receita final), com 3 sementes.
- **Recorte:** famílias reservadas, com a mesma recalibração da receita final:
  - a temperatura e o limiar saem de 25 rótulos da família;
  - os grupos do treino curto ficam fora;
  - os sorteios são os mesmos do `h9_recalibrar.py`.
- **"Não sei"** = confiança abaixo do limiar.
- **Medidas,** com IC 95% por bootstrap de grupos e de sorteios:
  - quanto ele responde e quanto erra entre os respondidos;
  - **quanto erraria entre os "não sei"**, se fosse forçado a responder;
  - a diferença entre esses dois erros;
  - **AUROC** da confiança: a chance de um acerto ter confiança maior que um erro;
  - **AUROC dentro da família:** o mesmo, sem misturar matérias.
- **Referência:** as famílias vistas, com a temperatura e o limiar do pipeline.

## Leitura (registrada antes)

- **"Sabe que não sabe":**
  - o erro entre os "não sei" é bem maior que o erro entre os respondidos, com os ICs separados;
  - **e** o AUROC dentro da família passa de 0,7.
- **Pelo motivo certo?**
  - Se o AUROC geral for alto e o de dentro da família ficar perto de 0,5, o "não sei" só separa matéria difícil de
    matéria fácil. Foi o que aconteceu com a confiança escrita na H11b.
  - O AUROC dentro da família é o teste limpo.
- **Conferência:** a cobertura e o erro dos respondidos do `h9r_k100` precisam repetir os números publicados:
  - Telepatia: 72% e 10%;
  - Felhen: 37% e 15%.

  Se não repetirem, a recalibração está diferente, e o resultado não vale até isso ser explicado.

## Fora deste plano

O teste limpo da `regra_beneficio` (`scripts/analise_falta_info_fatores.py`), com o "não sei" pela confiança.
Antes, é preciso ver quais `runs/res2dev_*` existem.

## Como rodar

```bash
cd /raid/$USER/lumi-decision && git pull -q && srun -p h100n2 --cpus-per-task=2 --mem=16G bash -c 'source slurm/config.sh && lumi_exec python scripts/analise_sabe.py --sementes 0,1,2 --out analises/sabe_que_nao_sabe.md' && ls -d runs/res2dev_* runs/res2dif_* 2>/dev/null
```
