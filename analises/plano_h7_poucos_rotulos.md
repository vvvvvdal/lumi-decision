# H7: quantos rótulos de uma família nova bastam para recalibrar? (registrado em 20/11, antes de rodar)

**Problema:** a temperatura T e o limiar tau, ajustados nas famílias vistas do `calib`, não transferem
para as reservadas. Lá, o risco entre os aceitos fica em 29–35% com alvo de 10% (E3/H4).

**Ideia (da literatura):** recalibrar por tarefa. Ver o Thermometer (arXiv 2403.08819) e conformal
prediction para múltipla escolha (arXiv 2305.18404).

**Desenho:** `scripts/h7_poucos_rotulos.py`, só CPU, com as predições do `h4_log` nas 3 sementes.
- Para cada família reservada, sortear k grupos rotulados (k = 10, 25, 50) do `calib` da mesma família.
  Ajustar T, ou T e tau, só para ela. Medir no `dev` da família.
- São 100 sorteios por k e semente.
- Comparações: `global` (o que se faz hoje) e `oraculo` (ajuste no próprio `dev` da família, limite
  otimista).

**Leitura registrada:**
- **Funciona:** com k = 25, `T_fam` reduz o ECE das reservadas para menos da metade do caminho entre
  `global` e `oraculo`, e `T_fam+tau_fam` deixa o risco entre os aceitos ≤ 0,15 na média.
- **Parcial:** o ECE melhora, mas o risco continua > 0,15. Nesse caso o limiar precisa de mais rótulos que
  a temperatura.
- **Não funciona:** k = 50 não se aproxima do oráculo. Nesse caso o problema não é de escala (temperatura)
  e sim de ordenação: o modelo não separa bem os casos certos dos errados na matéria nova.
