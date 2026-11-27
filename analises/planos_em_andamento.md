# Planos em andamento

Planos escritos antes de rodar. Quando um termina, ele vai para `resultados/Experimentos.md` junto com o
resultado. As versões completas estão no histórico do git (commit 878fdaf).

---

## H14: ajustar o treino curto para o Felhen

Na receita, o Felhen novo responde só 35% dos casos. Na `factck_veracidade` o modelo responde "falso" em 95% das
vezes e acerta só o que a classe mais comum dá (0,668 contra 0,667); a `juristcu_area` aprendeu de verdade. O
treino curto nunca teve hiperparâmetros ajustados.

Triagem com a semente 0, partindo do `h4_log`, com replay e a mesma recalibração (25 rótulos). A Telepatia fica
em k = 100 porque os exemplos dela saem do `dev`; o Felhen sai do `train` das reservadas (927 grupos na `factck`,
3.240 na `juristcu`).

| Braço | k do Felhen | Épocas | lr |
|---|---|---|---|
| `h9r_k100` (base) | 100 | 3 | 1e-4 |
| `h14_k300` | 300 | 3 | 1e-4 |
| `h14_ep6` | 100 | 6 | 1e-4 |
| `h14_lr2` | 100 | 3 | 2e-4 |
| `h14_lr05` | 100 | 3 | 5e-5 |
| `h14_k300_ep6` | 300 | 6 | 1e-4 |
| `h14_bal` | 100, equilibrado por classe na `factck` | 3 | 1e-4 |
| `h14_k300_bal` | 300, equilibrado por classe na `factck` | 3 | 1e-4 |

Critério: um braço passa se o Felhen responder pelo menos 5 pontos a mais, com o erro a no máximo 1 ponto e sem
perder mais de 1 ponto nas vistas nem na cobertura da Telepatia. Na `factck`, também conta como sinal o acerto
médio por classe passar de 0,40 (hoje 0,36; chutar dá 0,33) sem a acurácia cair mais de 3 pontos. O melhor que
passar roda nas sementes 1 e 2. Se nenhum passar, o próximo passo é mexer na primeira etapa (posto 64, lr).
Análise: `slurm/analise_h14.sbatch`.

---

## H15: o RLCD do Clef no Lumi

O Clef descreve o RLCD como um objetivo secundário com crédito parcial para a opção vizinha numa escala, prêmio
para o registro inteiro certo e uma penalidade de referência, mas não publica fórmula nem pesos. A minha versão:

$$L = -\log p(y) \;-\; \lambda \sum_a p(a)\,R(a,y) \;+\; \beta\,\mathrm{KL}(p \,\|\, p_{\text{ref}})$$

com R = 1 no gabarito, 0,5 na vizinha da escala e 0 no resto, $p_{\text{ref}}$ = o modelo no começo do treino,
λ = 1 e β = 0,02. As escalas estão em `configs/escalas.toml` (falso < impreciso < verdadeiro, e as escalas 0–3 e
0–4 da Telepatia). Duas versões: `rlcd` (esperança exata) e `rlcd_amostrado` (REINFORCE com 8 amostras, o "RL de
verdade"); em média os gradientes são iguais (`tests/test_h15_rlcd.py`). O prêmio do registro inteiro fica para
depois.

Roda dentro da receita (o `h9r_k100` com a perda nova), 3 sementes de cada versão.

Critério: ajuda se, na `factck`, cair a fração de erros graves (falso ↔ verdadeiro) sem a acurácia cair mais de
1 ponto, sem piorar a calibração depois da temperatura e com a cobertura a no máximo 1 ponto do `h9r_k100`. Se as
duas versões empatarem entre si, fica confirmado que o RL sobre as letras é o mesmo objetivo com ruído. O ganho
tem que aparecer onde há escala; se aparecer só fora dela, veio do KL e não do crédito parcial.
Análise: `slurm/analise_h15.sbatch`.

---

## H16: mais famílias em português

Sem exemplo nenhum de uma família nova, o Lumi acerta pouco. O E3b já treinou com o Eikos (27 famílias, 87% em
inglês) e ajudou nas famílias sintéticas (+3,4) e nas 7 reservadas juntas (+1,7 a +2,8), mas não nas 3 reservadas
do Felhen e da Telepatia; o idioma é o suspeito.

`h4_pt` é o `h4_log` com `eikos_pt` e `f3` (intenção e emoção) a mais, com os mesmos 508 passos, sementes 0 e 1.

Critério (`comparar_h4.py --base log --novos pt --sementes 0,1`): ajuda se a acurácia das reservadas subir com o
IC fora do zero sem perder mais de 1 ponto nas vistas. Se ajudar, rodar a receita a partir dele, e depois pensar
em gerar famílias sintéticas novas em português no `res2.py`.
