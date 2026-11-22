# RLCD no Lumi: resumo final dos experimentos (H4 a H9a e piloto H11)

**Autor:** Felipe Vidal, equipe de Benchmarks e Agentes, PDC SLMs (CEIA/UFG)
**Período:** 18 a 21/11/2026. Experimentos encerrados em 21/11.
**Código:** fork `vvvvvdal/lumi-decision`, branch `h6-bandit` (último commit `28be789`), a partir do
`Palharess/lumi-decision`.
**Avaliação:** tudo no `dev`; o `test` oficial não foi usado. Os detalhes estão em cada `Resultados H*.md`.

## A pergunta

O Lumi-4B escolhe uma opção e diz quanta confiança tem. Queremos que essa confiança seja **honesta**: quando
ele diz 70%, deve acertar 70% das vezes. Com isso, ele pode dizer "não sei" quando a chance de errar é alta.

Os experimentos tentaram responder três perguntas:
1. que recompensa deixa o Lumi honesto;
2. se ele continua honesto em matéria nova (as famílias reservadas, que ele não viu no treino);
3. como fazer ele **saber mais** na matéria nova sem esquecer a antiga.

## O caminho, em uma tabela

| Etapa | Pergunta | Resposta | Número principal |
|---|---|---|---|
| H4 | Recompensa de acerto, Brier ou log? | Acerto destrói a honestidade. Brier e log empatam. | Com recompensa de acerto, o ECE das reservadas vai a 0,43–0,48 (3 sementes para log e Brier) |
| H5 | Treinar o "não sei" como ação de RL? | Mantém o Lumi honesto, mas não supera a regra simples de abster pela confiança | Na mesma cobertura, erra igual ou mais |
| H6 | RL sem gabarito (só "acertou ou errou")? | O que estraga a honestidade é a recompensa, não a falta do gabarito | O `bandit_log` mantém a acurácia e recupera 73–86% da calibração |
| H7 | Quantos rótulos da matéria nova para recalibrar? | 25 consertam a escala. O limite real é saber pouco da matéria. | Mesmo com ajuste perfeito, responde só 12% (Felhen) e 2% (Telepatia) com 10% de erro |
| H8 | Mostrar exemplos no prompt ensina? | Não ensina, mas calibra | ECE da Telepatia de 0,104 para 0,029, com a mesma acurácia |
| H9 | Treinar um pouco com 100 exemplos da matéria nova? | **Ensina** (3 sementes) | Acurácia: Felhen +11,6, Telepatia +20,3 |
| H9b | Recalibrar depois desse treino? | Conserta a superconfiança | Telepatia: responde 68% com 10% de erro |
| H9r | Replay (misturar casos antigos no treino)? | Evita esquecer a matéria antiga (3 sementes) | Telepatia vista: −0,8 ponto, contra −2,1 sem replay |
| H9m | Misturar os adaptadores em vez de treinar de novo? | É uma troca: não esquece, mas ensina menos | Felhen +8,7, contra +13,6 |
| H9a | Escolher os exemplos onde o Lumi está em dúvida? | Empate com sortear | Diferença de ±1,9 ponto, com os dois ICs passando pelo zero |
| H11 (piloto) | O Lumi raciocina antes de decidir? Dá para montar o GRPO? | Raciocina, mas fica superconfiante, e o raciocínio sorteado é longo demais. O critério falha em 2 de 8 pontos: a queda vem dos raciocínios cortados, e a superconfiança, dos que terminam | NLL das vistas da Telepatia de 0,83 para 4,24; 65% dos raciocínios sorteados fecham em 512 tokens |

## Receita final

1. **Recompensa honesta:** usar a perda log, uma regra de pontuação própria (H4).
2. **Treino curto na matéria nova:** cerca de 100 exemplos **sorteados** da família nova, mais o mesmo número
   de casos já vistos (replay), por 3 épocas a partir do adaptador da H4 (H9 + H9r).
3. **Recalibrar:**
   - nas famílias vistas, usar a temperatura que o pipeline já ajusta;
   - nas novas, usar cerca de 25 rótulos da família para ajustar a temperatura e o limiar do "não sei" (H7 e
     H9b).

## O antes e o depois

Alvo: errar no máximo 10% das vezes em que responde. Matéria nova, média de 3 sementes.

| Fonte | Lumi depois da H4: responde / erra | Receita final: responde / erra |
|---|---|---|
| Telepatia (faturas novas) | 7% / 18% | **72% / 10%** |
| Felhen (direito e checagem de fatos novos) | 16% / 19% | **37% / 15%** |

Na matéria antiga, o Lumi não esquece (Telepatia −0,8 ponto) e continua honesto (ECE 0,017 no Felhen e 0,034
na Telepatia).

## O que aprendemos

- **A recompensa decide a honestidade.** Qualquer regra própria (log ou Brier) serve. Recompensar só o acerto
  ensina a chutar com certeza, com ou sem gabarito (H4 e H6).
- **Honestidade não é competência.** Ajustar a escala (temperatura, exemplos no prompt) torna o "não sei"
  honesto, mas não faz o Lumi acertar mais na matéria nova (H7 e H8).
- **Pouco treino já ensina, e o replay evita esquecer.** Saber mais muda muito o resultado: na Telepatia, o
  Lumi passa de responder 7% para 72% (H9, H9r e H9b).
- **Quantos exemplos usar importa mais que quais.** Escolher por dúvida empatou com sortear. Usar 25 exemplos
  em vez de 100 tira cerca de metade do ganho no Felhen (H9a).

## O que é RLCD aqui e o que é extra

- **RLCD (o foco da task):**
  - H4: que recompensa usar;
  - H5: o "não sei" como ação de RL;
  - H6: RL só com "acertou/errou";
  - H11 (piloto): preparar o RL com raciocínio, onde o RL é necessário.

  Essas etapas mostram, no Lumi, a ideia central do RLCD: a recompensa precisa ser uma regra própria, ou a
  probabilidade deixa de ser honesta.
- **Extra (adaptar à matéria nova):** H7 a H9a. São treino supervisionado e recalibração, sem RL. É daqui que
  vem o ganho prático (72% das faturas novas), mas não é RL.

## Por que o RL não ganhou do supervisionado

- No `lumi-decision`, a decisão é lida das letras **numa passada só**. A recompensa depende só dessa
  distribuição, que é derivável.
- Então a perda log já dá o gradiente **exato**, e o RL só estima esse mesmo gradiente com ruído. O teste
  `test_bandit_log_tem_o_gradiente_do_log_score_em_media` confere isso.
- Nesse formato, o RL no máximo empata, que foi o que vimos na H5 e na H6.

O RL passa a ser necessário quando algo entre a pergunta e a decisão é **sorteado e não derivável**, como o
raciocínio em texto (RLCR, Damani et al., 2025). Isso levou ao **piloto da H11**.

## Piloto H11: raciocinar antes de decidir

Detalhes em `Resultados H11 - piloto do raciocínio.md`.

- **O Lumi raciocina:** 81% dos raciocínios fecham o `</think>`, 1% degenera, 3,9 s cada.
- **Depois de raciocinar, ele fica superconfiante:**
  - o NLL sobe de 0,36–0,96 para 0,98–4,24, e o ECE chega a 0,55;
  - quando erra, erra com certeza.

  É o problema que o RLCR ataca com Brier depois do raciocínio.
- **Há sinal e tempo para o GRPO:** 66% dos grupos de 4 sorteios têm recompensas diferentes, e 600 grupos
  levam 1,7 h de geração.
- **Mas o critério registrado falha em 2 pontos:**
  - a acurácia das vistas cai 6,3 pontos (Telepatia −12,5, com 3 casos em 24);
  - só 65% dos raciocínios sorteados terminam em 512 tokens.
- **A separação entre os raciocínios que fecharam e os que não fecharam mostra duas causas** (feita depois do
  piloto, com a regra registrada antes de rodar):
  - **a queda de acurácia vem dos raciocínios cortados:** nos mesmos casos, de 0,706 para 0,353, contra −2,8
    pontos nos que fecharam;
  - **a superconfiança vem dos que terminam:** confiança média de 1,00, e todos os 21 erros com confiança de
    pelo menos 0,9. O raciocínio termina dizendo a resposta, e a leitura das letras só copia.
- **Consequência:** com as letras a 100%, o Brier das letras vira quase só "acertou ou errou", a recompensa que
  estragou a honestidade na H4. Antes do GRPO, é preciso resolver o comprimento e tirar a confiança de algo que
  o modelo consiga graduar, como a confiança escrita do RLCR.

## Limites

- **Só `dev`.** O `test` oficial não foi tocado.
- **Sementes:** H4 (log e Brier), H6 (`bandit_log` com G = 32, job 37691), H9 com k = 100 e H9r têm 3 sementes.
  H5, os outros braços da H6, H8, H9m e H9a têm só a semente 0.
- **Matéria nova em poucas famílias:** só 3 reservadas (2 do Felhen e 1 da Telepatia). As vistas da Telepatia
  têm só 84 grupos.
- **Oráculo otimista:** os números de "oráculo" ajustam e medem no mesmo conjunto. A receita final usa o ajuste
  honesto com 25 rótulos.
- **Mesmo `lr` em tudo,** e nenhuma busca de hiperparâmetros.
- **Felhen ainda é difícil:** acurácia de cerca de 0,65 na matéria nova, mesmo depois do treino curto.

## Em aberto (próximo ciclo)

- Confirmar a receita final no `test` oficial.
- Testar com mais famílias reservadas e ver se o ganho se repete.
- Ver se o RL de abstenção (H5) faz diferença quando o modelo já sabe a matéria, ou seja, depois da H9.
- **GRPO com raciocínio e recompensa Brier (RLCR).** Antes, dois ajustes da H11:
  - **o comprimento do raciocínio.** Opções:
    - mais tokens;
    - recompensa baixa para o raciocínio que não fecha;
    - pedir no prompt um raciocínio curto.
  - **a confiança escrita no texto,** como no RLCR, em vez da leitura das letras, que sai com 100% depois do
    raciocínio.

  Também falta medir o Lumi oficial sem adaptador com raciocínio.
