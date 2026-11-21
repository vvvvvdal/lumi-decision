# Plano H9b: recalibrar depois do treino curto

Registrado antes de rodar (21/11/2026).

## Pergunta

A H9 (k = 100) aumenta a acurácia nas reservadas, mas deixa o Felhen superconfiante: o ECE sobe de 0,072
para 0,112 na média das 3 sementes. A receita da H7 conserta isso? A receita é ajustar a temperatura T e
o limiar tau com k grupos rotulados da família.

## Desenho

- Script `scripts/h9_recalibrar.py`, só CPU, com as predições que já existem.
- Braços: `h4_log` e `h9_k100`, sementes 0, 1 e 2.
- Os grupos do treino curto saem do `dev` dos dois braços.
- Os k grupos de calibração (k = 10, 25, 50; 100 sorteios) vêm do `dev` restante e ficam fora da medida.
- Os dois braços sorteiam os mesmos grupos (mesmo gerador).
- Métodos (os da H7):
  - sem recalibrar;
  - global (vistas);
  - `T_fam`;
  - `T_fam+tau_fam`;
  - oráculo.

## Leitura (k = 25)

- **Conserta:** com `T_fam`, o ECE do Felhen reservado da H9 fica a no máximo 0,02 do ECE do `h4_log`
  com `T_fam`. A Telepatia não piora mais que 0,02.
- **Receita completa vale a pena:** com `T_fam+tau_fam`, a H9 responde mais casos que o `h4_log` com risco
  médio ≤ 0,15. Esta é a cobertura realista, com 25 rótulos e não o oráculo.
- **Não conserta:** o ECE do Felhen na H9 com `T_fam` continua mais de 0,02 acima do `h4_log` com
  `T_fam`. Nesse caso, a superconfiança não é só de escala.
