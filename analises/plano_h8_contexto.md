# H8: exemplos da família nova no prompt (registrado em 20/11, antes de rodar)

**Motivo (H7):** nas famílias reservadas, recalibrar com cerca de 25 rótulos acerta a régua, mas o gargalo é
competência. Com risco ≤ 10%, o oráculo só responde 12% (Felhen) e 2% (Telepatia) dos casos.

**Pergunta:** mostrar K casos com gabarito da mesma família no prompt (aprendizado em contexto, sem treino)
aumenta a acurácia e a cobertura segura nas reservadas?

**Desenho:**
- Adaptador `runs/h4_log` (semente 0). Comando: `lumi-eval run --held-out only --exemplos-contexto K`.
- K = 0 (controle no mesmo fluxo), 3 e 8. Ordens canonical e reversed, sem geração de texto.
- **Origem dos exemplos:**
  - Felhen: o `train` das famílias reservadas, nunca usado em treino.
  - Telepatia: as reservadas só existem no `dev`, então os exemplos vêm de **outros grupos** do `dev` (nunca
    o mesmo estado do caso avaliado).
  - Um exemplo por grupo, em ordem fixa por caso.
- Teto de 6000 tokens no prompt. Se passar, os últimos exemplos saem, e o K usado fica registrado.
- Análise: `scripts/analise_h8.py`.
  - K contra K = 0: acurácia, ECE, Brier e NLL, com IC por bootstrap de grupos.
  - Cobertura no risco de 10%, com T e limiar por família ajustados em 25 grupos (como na H7), e o oráculo.

**Leitura registrada:**
- **Ajuda:** com K = 8, a acurácia nas reservadas sobe com o IC acima de zero em pelo menos uma fonte, e a
  cobertura do oráculo no risco de 10% sobe em relação a K = 0.
- **Não ajuda:** acurácia em empate nas duas fontes. Nesse caso, ensinar a família nova exige treino (LoRA
  curta com os exemplos, via `--init-adapter`).
- Se o K usado médio ficar bem abaixo do pedido (prompts longos), registrar isso como limite.
