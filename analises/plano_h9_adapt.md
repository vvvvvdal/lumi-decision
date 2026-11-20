# H9: treino curto com k grupos da família nova (registrado em 20/11, antes de rodar)

**Motivo:** H7 e H8 calibram a matéria nova, mas não aumentam a acurácia. O gargalo é competência.

**Pergunta:** continuar a LoRA do `h4_log` por poucos passos, só com k grupos rotulados das famílias
reservadas, ensina o Lumi sem ele esquecer as famílias vistas?

**Desenho:**
- Famílias: `felhen:juristcu_area`, `felhen:factck_veracidade` e `telepatia:invoice_processing`, numa LoRA só.
- k = 25 e k = 100 grupos por família. Perda `log`, mesmo `lr`, 3 épocas, semente 0.
- Ponto de partida: `--init-adapter runs/h4_log/adapter`.
- **Origem dos exemplos:**
  - Felhen: o split `train` das reservadas, nunca usado.
  - Telepatia: o `dev`, porque é o único split que existe. Esses grupos saem da avaliação nos dois braços
    (`adapt_grupos.json`).
- Comparação com o `h4_log` nos mesmos casos restantes do `dev`: `scripts/analise_h9.py`.

**Leitura registrada:**
- **Ensina:** com k = 100, a acurácia nas reservadas sobe com o IC acima de zero em pelo menos uma fonte, e as
  vistas não caem mais que 1 ponto (IC).
- **Ensina pouco:** a acurácia sobe só numa família, ou só com k = 100.
- **Não ensina:** empate. Nesse caso, a competência na matéria nova exige mais dados ou um modelo maior que o
  Lumi-4B.
- Calibração: o ECE das reservadas não deve piorar mais que 0,02.
