# Referências

Links conferidos em 22/11/2026 e 24/11/2026.

- Dubois (2026), *Reinforcement Learning for Calibrated Decisions (RLCD)*. Resumo da ideia da TypeSafe AI, que não
  publicou paper nem fórmula. <https://www.dsebastien.net/reinforcement-learning-for-calibrated-decisions-rlcd/>
- Cloudflare (2026), *Introducing Clef*. Decision models com base congelada, LoRA e leitura direta das opções;
  treino com CE + label smoothing + Brier e depois um RLCD descrito só por alto.
  <https://blog.cloudflare.com/clef-decision-models/>
- Yaldiz et al. (2026), *Calibration Aware Reinforcement Learning*. RL que premia só o acerto deixa o modelo
  confiante demais, o mesmo que vi na H4. <https://arxiv.org/abs/2601.13284>
- Damani et al. (2025), *Beyond Binary Rewards* (RLCR). O modelo raciocina, responde e escreve a confiança;
  a recompensa é acerto + Brier da confiança. Usa Brier porque a recompensa precisa ser limitada no RL.
  <https://arxiv.org/abs/2507.16806>
- Shao et al. (2024), *DeepSeekMath*. Introduz o GRPO. <https://arxiv.org/abs/2402.03300>

Sobre a perda em outros decision models (pesquisado para a H12): o comum é CE pura e calibração depois, com
temperatura ou limiar.

- Huang et al. (2025): label smoothing só corrige excesso de confiança e ajuda pouco em modelo pequeno.
  <https://arxiv.org/abs/2508.00264>
- Calibração de guard models (Llama Guard, ShieldGemma etc.): leem P("sim") como a minha leitura das letras e
  saem confiantes demais. <https://arxiv.org/abs/2410.10414>
- Granite Guardian: SFT no rótulo sim/não, sem smoothing. <https://arxiv.org/abs/2412.07724>
- RouteLLM: softmax sobre os rótulos e um limiar. <https://arxiv.org/abs/2406.18665>
- Berta et al. (2025), *Refine, then calibrate*: treinar para separar e calibrar depois, como a minha receita.
  <https://arxiv.org/abs/2501.19195>

O código parte do `lumi-decision` do Palhares (H1 a H3); este trabalho está no fork `vvvvvdal/lumi-decision`,
branch `h6-bandit`.
