# Desenho do braço (e): RL com "não sei" como ação (H5)

**Projeto:** RLCD no Lumi-4B, equipe de Benchmarks e Agentes (PDC SLMs, CEIA/UFG). Fork
`vvvvvdal/lumi-decision`.
**Antes:** a H4 (`Resultados H4 - parcial.md`) mostrou que a recompensa de acerto destrói a calibração e que
log ≈ Brier.

## 1. O problema

Em nenhum braço da H4 o "não sei" funciona na matéria nova:
- com o limiar ajustado para 10% de risco nas famílias vistas, o risco nas reservadas fica em 29–35%;
- na família `regra_beneficio` (M2 do Palhares), quase nenhum modelo escolhe "pedir o documento que falta"
  (0–19 de 272).

Hoje, abster é uma **regra colada por fora**: se a confiança fica abaixo de um limiar, o modelo abstém. Ela
depende de a probabilidade ser boa, e nas reservadas não é.

## 2. A ideia

Fazer de "abster" uma **ação que o modelo aprende**, com aprendizado por reforço de verdade:
1. o modelo vê o caso e as opções A, B, C… e uma opção extra, **"X. Não sei / passar para um humano"**;
2. ele **sorteia** entre responder e abster;
3. recebe a recompensa: **+1** se respondeu certo, **−λ** se respondeu errado, **0** se absteve;
4. aprende com a recompensa (REINFORCE com a média do grupo como base, no estilo GRPO), sem precisar de
   gabarito para "abster".

## 3. A pegadinha, e como o desenho escapa dela

A recompensa +1/−λ/0 é **linear**, como a de acerto. Se ela treinasse o modelo inteiro, ele voltaria a ficar
arrogante (H4). Por isso a perda tem partes separadas, que não brigam:

| Parte | O que treina | Como |
|---|---|---|
| **Crença** | as probabilidades de A, B, C (renormalizadas sem o X) | log score, a perda de sempre (honesta) |
| **Decisão** | só a chance de abster, $P(\text{abster}) = \sigma(z_X - \log\sum_k e^{z_k})$ | RL com +1 / −λ / 0; o gradiente mexe só no logit de X |
| **Coleira** | tudo | KL para o Lumi sem adaptador, peso pequeno |

Quando o modelo responde, a resposta é a opção mais provável segundo a crença. Na avaliação vale a mesma
regra: abstém se $P(\text{abster}) > 0{,}5$; senão, responde a opção mais provável entre as normais.

Pela teoria, com custo λ o ideal é abster quando a chance de acerto é menor que $\lambda/(1+\lambda)$ (0,75
com λ = 3; 0,9 com λ = 9). A pergunta da H5 é se o modelo aprende isso **melhor que o limiar**, usando pistas
que a probabilidade não mostra, como perceber que falta informação no texto.

## 4. Hipótese H5 (registrada antes de rodar)

> Nas famílias reservadas, **na mesma cobertura**, o braço (e) tem risco entre os aceitos menor que o
> `h4_log` com limiar. O ECE nas opções normais não piora mais que 0,02.

- **Secundária:** em `regra_beneficio`, quando o gabarito é `pedir_documento`, o braço (e) escolhe abster ou
  pedir o documento em mais casos que o `h4_log`.
- **Conta só se:** o IC 95% inteiro ficar do lado bom, nas 3 sementes. Tudo no `dev`; o `test` fica intocado.

**"Na mesma cobertura"** quer dizer: se o braço (e) responde 60% dos casos, o `h4_log` também responde só os
60% em que tem mais confiança. Aí comparamos quem erra menos entre os respondidos. Sem isso, um modelo que
abstém mais sempre "ganharia".

## 5. Como rodar (depois de implementado)

1. Teste curto: o fluxo roda e a taxa de abstenção não colapsa (abster sempre ou nunca).
2. λ = 3, semente 0, completo, com avaliação nas 3 fontes + reservadas v2 (`res2`).
3. λ = 9 (o alvo de 10% de risco do Palhares).
4. Se a H5 parecer positiva: sementes 1 e 2.

## 6. Riscos

| Risco | O que fazer |
|---|---|
| Colapso: abster sempre ou nunca | O log de treino mostra a taxa de abstenção; reduzir α ou λ |
| O RL estraga a crença (ECE piora) | Medir ECE só nas opções normais; aumentar o peso da coleira |
| O "abster" aprendido só imita o limiar | Resultado possível; a comparação na mesma cobertura mostra isso honestamente |
| Custo de GPU | +1 forward por lote (a coleira); ~1,3× o E3 |

## 7. Perguntas para o Palhares

1. Além da `regra_beneficio` (`pedir_documento`), há outras famílias com uma resposta do tipo "falta
   informação"?
2. Faz sentido deixar a opção "X. Não sei" **fixa no fim**, fora da inversão e do sorteio das ordens?
3. Mantenho `lr`, LoRA e número de passos iguais aos do E3, para a comparação com o `h4_log` ser justa?

