# Referências: RLCD no Lumi

As cinco fontes principais do trabalho. Todos os links foram conferidos em 22/11/2026.

1. **RLCD**: Sébastien Dubois (2026), *Reinforcement Learning for Calibrated Decisions (RLCD)*. É um resumo da
   proposta da TypeSafe AI, que não publicou paper. Explica a ideia: treinar decisões com uma recompensa que
   premia a confiança certa.
   <https://www.dsebastien.net/reinforcement-learning-for-calibrated-decisions-rlcd/>
2. **Clef**: Cloudflare (2026), *Introducing Clef: our open-source decision models, and new RL fine-tuning
   platform*. São modelos de decisão parecidos com o nosso uso do Lumi, treinados com RL.
   <https://blog.cloudflare.com/clef-decision-models/>
3. **Recompensa de acerto e superconfiança**: Yaldiz et al. (2026), *Balancing Classification and Calibration
   Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning*. Mostra o mesmo que a nossa
   H4: o RL que premia só o acerto deixa o modelo confiante demais.
   <https://arxiv.org/abs/2601.13284>
4. **RLCR**: Damani et al. (2025), *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty*. O
   modelo raciocina, responde e escreve a confiança, e a recompensa é o acerto mais o Brier dessa confiança. É a
   base do nosso próximo passo.
   <https://arxiv.org/abs/2507.16806>
5. **GRPO**: Shao et al. (2024), *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
   Models*. É o artigo que apresentou o GRPO, o algoritmo de RL que o RLCR usa: compara várias respostas da
   mesma pergunta e reforça as melhores do grupo.
   <https://arxiv.org/abs/2402.03300>

## A perda em outros decision models (pesquisa de 24/11/2026, para a H12)

**Resposta curta:** o padrão é entropia cruzada pura sobre os rótulos, com a calibração resolvida depois
(temperatura ou limiar). Só o Clef publica a soma de smoothing e Brier, e ele não mostra números de calibração
que a sustentem. Os números abaixo vêm das fontes e não foram todos conferidos na tabela original.

- **Label smoothing em SFT:** Huang et al. (ICML 2025). O smoothing (β = 0,1) só corrige excesso de confiança, e
  o ganho cai em modelos pequenos com vocabulário grande. Bate com a H12: o Lumi com a `log` não estava confiante
  demais, e o smoothing o deixou pouco confiante.
  <https://arxiv.org/abs/2508.00264>
- **Calibração de guard models** (Llama Guard, ShieldGemma e outros): leem P("sim") renormalizada, como a nossa
  leitura das letras. São confiantes demais (ECE de 10% a 50%), e a temperatura ajuda só em parte.
  <https://arxiv.org/abs/2410.10414>
- **Granite Guardian:** SFT com o rótulo Sim/Não, sem smoothing. A probabilidade sai do softmax das variantes de
  sim e não. <https://arxiv.org/abs/2412.07724>
- **RouteLLM:** o roteador com LLM trata os rótulos como tokens e faz softmax sobre eles. Não discute calibração;
  usa um limiar. <https://arxiv.org/abs/2406.18665>
- **Refine, then calibrate:** Berta et al. (2025). Treinar para separar bem e calibrar depois com temperatura,
  porque as duas coisas não chegam ao melhor ponto no mesmo momento do treino. É o que a receita final já faz.
  <https://arxiv.org/abs/2501.19195>
- **Por que o RLCR usa Brier:** a recompensa é limitada. O log, sem limite, pode premiar errar com confiança
  dentro do RL. Na nossa perda supervisionada sobre as letras, log e Brier são as duas próprias, e esse argumento
  não se aplica.

**Base do código:** o `lumi-decision` do Palhares (H1 a H3). O trabalho está no fork `vvvvvdal/lumi-decision`,
branch `h6-bandit`.
