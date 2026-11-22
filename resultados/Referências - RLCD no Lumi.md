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

**Base do código:** o `lumi-decision` do Palhares (H1 a H3). O trabalho está no fork `vvvvvdal/lumi-decision`,
branch `h6-bandit`.
