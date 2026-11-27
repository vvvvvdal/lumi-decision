# RLCD no Lumi: resumo

O Lumi-4B escolhe uma opção e dá uma probabilidade para cada uma. A ideia do RLCD é treinar para que essa
probabilidade seja honesta (quando diz 70%, acerta 70%), porque aí dá para responder "não sei" nos casos
duvidosos. Os detalhes de cada experimento estão em `Experimentos.md`.

## A receita que funciona hoje

1. SFT com LoRA (posto 16) usando a perda log, que é uma regra de pontuação própria (H4).
2. Um treino curto na família nova: cerca de 100 exemplos sorteados, mais o mesmo tanto de casos já vistos
   (replay), 3 épocas a partir do adaptador da H4 (H9 e H9r).
3. Recalibrar: temperatura das vistas para as famílias vistas; para a família nova, temperatura e limite do
   "não sei" ajustados com 25 rótulos dela, mirando 10% de erro (H7 e H9b).

Com isso, na matéria nova (média de 3 sementes, alvo de 10% de erro):

| Fonte | Antes (só H4): responde / erra | Receita: responde / erra |
|---|---|---|
| Telepatia (faturas) | 7% / 18% | 72% / 10% |
| Felhen (TCU e checagem de fatos) | 16% / 19% | 35% / 15% |

Na matéria antiga ele quase não esquece (−0,8 ponto na Telepatia) e a calibração continua boa (ECE 0,017 no
Felhen e 0,034 na Telepatia).

## O que cada experimento mostrou

| | Pergunta | Resultado |
|---|---|---|
| H4 | Recompensa de acerto, Brier ou log? | Acerto quebra a calibração (ECE 0,43–0,48 nas reservadas). Log e Brier empatam |
| H5 | "Não sei" como opção, aprendido com RL | Continua calibrado, mas não ganha do limite simples na confiança |
| H6 | RL só com "acertou/errou" | O que estraga é a recompensa, não a falta de gabarito. `bandit_log` recupera 73–86% da calibração |
| H7 | Quantos rótulos para recalibrar a família nova | 25 bastam para a escala; o limite é saber pouco da matéria |
| H8 | Exemplos no prompt | Calibra (ECE 0,104 → 0,029 na Telepatia), mas não ensina |
| H9 | Treino curto com 100 exemplos | Ensina: +11,6 no Felhen e +20,3 na Telepatia |
| H9b | Recalibrar depois do treino curto | Conserta a superconfiança |
| H9r / H9m | Replay ou misturar adaptadores contra o esquecimento | Replay resolve (−0,8 contra −2,1). Misturar ensina menos |
| H9a | Escolher os exemplos em que ele está em dúvida | Empata com sortear |
| H11 / H11b | Raciocinar antes de decidir; escrever a confiança | Fica superconfiante e escreve "100" quase sempre. Trilha encerrada |
| H12 | Perda do Clef (label smoothing + Brier) | Pior na calibração |
| H12b | Qual peça faz o quê | O smoothing estraga a calibração; o Brier dá +2,2 na Telepatia nova |
| H12c | `log_brier` dentro da receita | Empata; o ganho some depois do treino curto |
| H13 | Treinar com o texto do prompt variado | Mais robusto a prompt novo; na matéria nova, +3,9 na Telepatia e −1,8 no Felhen |
| H13b | Receita com prompt variado | Não entra: perde 6 pontos de cobertura no prompt de sempre |

## Ele sabe o que não sabe?

Depende de qual "não sei". Com o limite na confiança, depois do treino curto, sim: nas faturas novas ele erra
10% do que responde e erraria 47% do que passa adiante, e a confiança separa acerto de erro dentro da mesma
família (AUROC 0,83 na Telepatia e 0,75 no Felhen; antes do treino curto era 0,59). Ainda assim, metade do que
ele passa adiante ele teria acertado.

O que ele não faz é perceber quando falta informação. Na `regra_beneficio`, com o "não sei" como opção (H5), ele
responde "não sei" em 36% dos casos com documento faltando e em 40% dos casos sem documento faltando, ou seja,
reage a "caso complicado", não a "falta informação". Com o limite na confiança isso ainda não foi testado.

## Por que o RL não ganhou

A decisão é lida dos logits das letras numa passada só, então a perda log já dá o gradiente exato. O RL estima
esse mesmo gradiente por amostragem e só acrescenta ruído; por isso empatou na H5 e na H6. RL só faz diferença
quando há algo amostrado e não derivável no caminho, como um raciocínio em texto. Testei isso nos pilotos
H11 e H11b: com raciocínio ele acerta menos (0,636 contra 0,727) e fica confiante demais, então parei essa
trilha. O Lumi oficial pensando, sem adaptador, nunca foi medido.

## Comparação com o Clef

O Clef (Cloudflare) também lê a decisão direto das opções e usa base congelada + LoRA. A perda deles não ganhou
da minha (H12 a H12c). A parte de dados (prompts variados) deixou o Lumi mais robusto, mas com custo (H13 e
H13b). O RLCD deles (crédito parcial na escala e penalidade de referência) não tem fórmula publicada; a minha
versão está sendo testada na H15.

## Limites

- Tudo no `dev`; o `test` oficial não foi usado.
- Os números valem para o prompt de sempre. Com um prompt nunca visto, a Telepatia cai de 72% para 50% (H13b).
- Só 3 famílias reservadas (2 do Felhen e 1 da Telepatia), e as vistas da Telepatia têm só 84 grupos.
- Vários braços têm só a semente 0 (H5 na comparação com o limite, parte da H6, H8, H9m e H9a).
- Na `factck_veracidade` o Lumi só chuta "falso" (95% das respostas); o acerto de 67% é o da classe mais comum.
  A `juristcu_area` aprendeu de verdade.

## Em aberto

- Os experimentos rodando agora (H14 a H16) estão em `analises/planos_em_andamento.md`.
- Confirmar a receita no `test` oficial e com mais famílias reservadas.
- Robustez ao texto do prompt: talvez misturar mais o prompt de sempre no treino, ou usar muito mais variantes.
- Testar se o "não sei" pela confiança percebe falta de informação (a `regra_beneficio`, com
  `scripts/analise_falta_info_fatores.py`).
- Repetir o RL de abstenção (H5) depois do treino curto.
- Ideia guardada: GEPA (arXiv 2507.19457) para reescrever a descrição da tarefa nova com 25 rótulos e depois
  destilar no Lumi. O paper não testou classificação com modelo pequeno, e a H8 sugere que informação no prompt
  calibra mas não ensina.
