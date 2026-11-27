# Experimentos do RLCD no Lumi

Cada seção tem a pergunta, o que foi feito, o resultado e o critério que registramos antes de rodar. Os planos completos e as versões longas estão no histórico do git (commit 878fdaf).

Seções: H4 · H5 · H6 · H7 · H8 · H9 · H9a · H9b · H9r e H9m · H10 · H11 e H11b · H12 · H12b · H12c · H13 · H13b · Sabe que não sabe · Onde o Felhen erra

---

## H4: a forma da recompensa muda a calibração?

Recompensar só o acerto destrói a calibração do Lumi, como a teoria previa. Entre as duas regras próprias (log e Brier) quase não há diferença: o problema é usar uma recompensa que não é própria.

Mantivemos a receita A1 inteira (LoRA r=16, 20.350 exemplos de Felhen + Telepatia + Traces, 1 época, lr 1e-4) e trocamos só a perda: `log` ($-\sum_k q_k \log p_k$), `brier` ($\sum_k (p_k - q_k)^2$) e `acerto` ($-\sum_k q_k\, p_k$). Temperatura e limiar de abstenção foram ajustados no `calib` e aplicados no `dev`. Antes conferimos que o ambiente reproduz o E1 e o A1 originais. `log` e `brier` rodaram com as sementes 0, 1 e 2; `acerto` só com a semente 0 (18 e 19/11).

| Braço | Grupo (Felhen) | Acc | NLL | ECE | $T$ | Cobertura |
|---|---|---:|---:|---:|---:|---:|
| `log` | vistas | 0,869 | 0,364 | 0,016 | 1,06 | 0,894 |
| `log` | reservadas | 0,515 | 1,282 | 0,078 | 1,06 | 0,383 |
| `brier` | vistas | 0,872 | 0,391 | 0,020 | 0,98 | 0,906 |
| `brier` | reservadas | 0,521 | 1,268 | 0,057 | 0,98 | 0,429 |
| `acerto` | vistas | 0,835 | 2,552 | 0,159 | 5,74 | 0,346 |
| `acerto` | reservadas | 0,501 | 6,481 | 0,483 | 5,74 | 0,000 |

Com `acerto` a acurácia quase não cai, mas a NLL multiplica por 5–7 e o `calib` pede temperatura 5,74 no Felhen e 8,26 na Telepatia. Com o limiar do `calib`, a cobertura nas reservadas vai a zero. É o esperado: com uma recompensa linear em $p$, o ótimo é pôr toda a massa numa opção, e nada pune a certeza exagerada.

Na semente 0 o Brier parecia ter ECE menor nas reservadas, mas com 3 sementes e IC deu empate no Felhen e ficou pior em `invoice` (+0,025), mesmo acertando +2,2 pontos ali. A NLL fica um pouco pior com Brier em todos os grupos, o que faz sentido, já que a perda log otimiza a própria NLL. Em nenhum braço a abstenção transfere: o limiar de 10% de risco nas vistas vira 29–35% de risco nas reservadas.

Sem plano separado; a previsão, escrita antes, era que `acerto` deixasse o Lumi superconfiante e que `log` e `brier` mantivessem probabilidades honestas, comparando cada braço com o `log` por IC 95% de bootstrap (empate quando o IC inclui zero).

Limites: `acerto` tem uma semente só, e usamos o mesmo `lr` para todas as perdas, embora o gradiente do Brier seja menor.

---

## H5: abstenção aprendida por RL

Treinar o "não sei" como ação de RL manteve a crença honesta, mas não ganhou da regra simples de abster pela confiança: na mesma cobertura, erra igual ou mais. O sinal de "falta informação" que parecia aparecer era confundido. O modelo abstém em casos com algum problema, não quando falta documento.

Adicionamos uma opção "Não sei / passar para um humano" e uma perda em três partes: crença (log score nas opções normais), decisão (REINFORCE com +1 / −λ / 0, mexendo só no logit do "não sei") e uma coleira de KL para o Lumi sem adaptador. Rodamos λ = 3 e λ = 9 na semente 0 e depois λ = 3 nas sementes 1 e 2 para checar o sinal secundário (19 e 20/11).

| Fonte | Grupo | Cobertura | Risco (e) | Risco `h4_log` | Δ [IC 95%] |
|---|---|---:|---:|---:|---|
| Felhen | vistas | 0,865 | 0,083 | 0,078 | +0,006 [−0,003; +0,015] empate |
| Felhen | reservadas | 0,672 | 0,480 | 0,411 | +0,069 [+0,027; +0,113] pior |
| Telepatia | vistas | 0,089 | 0,067 | 0,013 | +0,053 [+0,012; +0,119] pior |
| Telepatia | reservadas | 0,075 | 0,124 | 0,124 | +0,000 empate |

A ECE nas opções normais empatou ou melhorou contra o `h4_log`, então separar crença e decisão funcionou. Na Telepatia o modelo abstém em ~91% dos casos com λ = 3, porque a acurácia nas vistas (~73%) fica abaixo do limiar de 0,75 que esse λ implica; com λ = 9 ele quase não responde. Em `regra_beneficio`, com 3 sementes, ele abstém ou pede documento em 31% dos casos de `pedir_documento` contra 1,6% em `conceder`. Só que dentro de "negar", onde a resposta certa é a mesma, ele abstém igual com ou sem documento faltando (−0,043 [−0,188; +0,114]). O padrão real é "caso complicado", não "falta informação" (`scripts/analise_falta_info_fatores.py`).

Critério escrito antes de rodar: nas reservadas, na mesma cobertura, o braço (e) teria risco menor que o `h4_log` com limiar, sem piorar a ECE das opções normais mais que 0,02; a secundária era abster ou pedir documento mais vezes em `pedir_documento`. Só contaria com o IC 95% inteiro do lado bom nas 3 sementes.

Limites: a hipótese principal só tem a semente 0 (o plano mandava replicar só se ela parecesse positiva), e o λ foi o mesmo nas duas fontes.

---

## H6: RL só com "acertou / errou" (bandit)

Sem ver o gabarito, recompensar o acerto deixa o Lumi tão superconfiante quanto com gabarito. Com recompensa própria (1/π) ele mantém a acurácia e chega perto da calibração supervisionada, mas sobra um custo pequeno e real no Felhen reservado.

O Lumi sorteia G respostas por caso e só sabe se cada uma acertou. Comparamos `bandit_acerto` (recompensa 1 se acertou) e `bandit_log` (1/π(a) se acertou, com corte) com o `h4_log`. Depois variamos o corte (10, 30, 1000), os sorteios (G = 8 e 32), as épocas (2) e a exploração (eps = 0,2), tudo em 20/11.

| Braço | Sementes | ECE Felhen res. | ECE Telepatia res. |
|---|---|---:|---:|
| `log` com gabarito | 0 | 0,078 | 0,105 |
| `acerto` com gabarito | 0 | 0,483 | 0,425 |
| `bandit_acerto` | 0 | 0,457 | 0,372 |
| `bandit_log`, G = 8, corte 10 | 0 | 0,180 | 0,143 |
| `bandit_log`, G = 32, corte 1000 | 0 | 0,099 | 0,085 |
| `log` com gabarito | 0, 1, 2 | 0,065 | 0,091 |
| `bandit_log`, G = 32, corte 1000 | 0, 1, 2 | 0,114 | 0,091 |

O que estraga a calibração é a recompensa, não a falta de informação. O `bandit_log` empata com o `log` em acurácia, enquanto o `bandit_acerto` perde 5,5 e 5,0 pontos nas vistas. Subir o corte mudou pouco, e G = 32 foi o que mais ajudou, mas a semente 0 foi otimista: com 3 sementes o Felhen reservado fica +0,049 pior que o `log`. Treinar 2 épocas não ajudou (a diferença foi de +0,021 para +0,044) e a exploração também não (0,099 → 0,096). No Felhen reservado o Brier empata e a NLL piora, então o custo vem de poucos casos muito confiantes na resposta errada: se o modelo dá chance quase zero à resposta certa, ela quase nunca é sorteada e não recebe sinal.

Critério escrito antes de rodar: H6a, `bandit_acerto` com ECE nas reservadas acima do `h4_log` (IC acima de zero); H6b, `bandit_log` com ECE nas reservadas no máximo 0,02 acima do `h4_log`. A H6a se confirmou e a H6b não.

Limites: G = 8, os cortes e a exploração têm só a semente 0, e o bandit usou o mesmo `lr` e número de passos do supervisionado.

---

## H7: quantos exemplos de uma família nova bastam para recalibrar?

Na Telepatia a temperatura estava errada, e 25 exemplos da família nova consertam a ECE. Mas o "não sei" errado não é problema de calibração. Mesmo com ajuste perfeito, para errar no máximo 10% o Lumi só pode responder 12% dos casos no Felhen e 2% na Telepatia. Falta saber a matéria.

Sem treino, só CPU: com as predições do `h4_log` (sementes 0, 1 e 2), cada família reservada ganhou k exemplos com gabarito (k = 10, 25, 50; 100 sorteios por k) para reajustar a temperatura e o limiar só para ela. Comparamos com o ajuste global de hoje e com um oráculo que ajusta na família toda (`scripts/h7_poucos_rotulos.py`, 20/11).

| Fonte | Método | k | ECE | Cobertura | Risco |
|---|---|---:|---:|---:|---:|
| Felhen | global (hoje) | — | 0,062 | 37% | 0,29 |
| Felhen | T + limiar da família | 50 | 0,062 | 16% | 0,14 |
| Felhen | oráculo | — | 0,043 | 12% | 0,10 |
| Telepatia | global (hoje) | — | 0,189 | 44% | 0,37 |
| Telepatia | T da família | 25 | 0,111 | 14% | 0,19 |
| Telepatia | T + limiar da família | 50 | 0,108 | 6% | 0,13 |
| Telepatia | oráculo | — | 0,110 | 2% | 0,09 |

Na Telepatia, k = 25 leva a ECE de 0,189 para 0,111, igual ao oráculo, e com 10 exemplos já fica em 0,117. No Felhen a temperatura global já estava boa. O limiar global erra 29–37% porque responde demais (37–44% dos casos), não porque as probabilidades mentem. Com 50 exemplos o risco médio chega perto do alvo, mas varia muito entre sorteios e o Lumi passa a responder só 6–16% dos casos.

Critério escrito antes de rodar: funcionaria se, com k = 25, a temperatura da família cobrisse mais da metade do caminho entre global e oráculo e o limiar da família deixasse o risco médio ≤ 0,15. Deu parcial, porque o risco só chega a ≤ 0,15 com k = 50.

Limites: as reservadas não têm `calib`, então os k exemplos saíram do próprio `dev`, e o oráculo é otimista porque ajusta e mede no mesmo conjunto.

---

## H8: exemplos da família nova no prompt ensinam o Lumi?

Não ensinam: a acurácia empata. Mas calibram. Na Telepatia a ECE cai de 0,104 para 0,029 sem nenhum ajuste de temperatura.

Sem treino, com o adaptador `h4_log` (semente 0), colocamos antes da pergunta K = 0, 3 ou 8 casos já respondidos da mesma família (`--exemplos-contexto K`) e medimos nas reservadas do `dev` (`scripts/analise_h8.py`, 20/11).

| Fonte | Acc K = 0 | Acc K = 3 | Acc K = 8 | ECE K = 0 | ECE K = 3 | ECE K = 8 |
|---|---:|---:|---:|---:|---:|---:|
| Felhen | 0,511 | 0,532 | 0,526 | 0,088 | 0,055 | 0,050 |
| Telepatia | 0,606 | 0,605 | 0,621 | 0,104 | 0,038 | 0,029 |

Todas as diferenças de acurácia são empates. No Felhen a ECE também empata (IC até +0,008); na Telepatia ela melhora com IC abaixo de zero, junto com Brier e NLL. É mais do que os 25 rótulos da H7 conseguiram (0,111). Parece que o Lumi vê como são as respostas daquela família e ajusta a confiança, mas não passa a acertar mais. A cobertura com 10% de risco oscila muito porque depende de poucos casos muito confiantes.

Critério escrito antes de rodar: ajudaria se, com K = 8, a acurácia nas reservadas subisse com IC acima de zero em pelo menos uma fonte; empate nas duas significaria que ensinar a família nova exige treino. Deu empate, e o passo seguinte foi a LoRA curta da H9.

Limites: uma semente, e no Felhen os textos longos fizeram caber em média só 6,2 dos 8 exemplos.

---

## H9: treinar um pouco com a família nova ensina o Lumi?

Ensina. É o primeiro resultado do projeto que aumenta a acurácia na matéria nova, e com k = 100 isso se repete nas duas fontes e nas 3 sementes. Ficam dois custos pequenos: o Felhen novo sai confiante demais e há um esquecimento leve nas vistas da Telepatia.

Partindo do `h4_log`, continuamos a LoRA por 3 épocas só com k = 25 ou 100 grupos rotulados de `juristcu_area`, `factck_veracidade` e `invoice_processing`, com a perda log. Os grupos de treino saem da avaliação. k = 100 rodou nas sementes 0, 1 e 2; k = 25 só na 0 (21/11, `scripts/analise_h9.py`).

| Fonte (k = 100, média de 3 sementes) | Acc reservadas | Cobertura do oráculo com 10% de risco | ECE reservadas |
|---|---:|---:|---:|
| Felhen | 0,529 → 0,645 | 12% → 30% | 0,072 → 0,112 |
| Telepatia | 0,584 → 0,786 | 2% → 69% | 0,114 → 0,074 |

O IC da acurácia fica acima de zero nas 6 combinações de fonte e semente, e com k = 25 também sobe (+5,2 no Felhen e +10,6 na Telepatia). Na Telepatia o Lumi passa a responder com segurança muito mais casos e ainda fica mais honesto. No Felhen a ECE piora nas 3 sementes (+0,03 a +0,05); cada IC encosta no zero, mas a direção se repete, então provavelmente é real. Nas vistas o Felhen empata, e a Telepatia cai de 1,5 a 3,0 pontos, com IC todo abaixo de zero na semente 2.

Critério escrito antes de rodar: ensinaria se, com k = 100, a acurácia nas reservadas subisse com IC acima de zero em pelo menos uma fonte e as vistas não caíssem mais que 1 ponto; a ECE das reservadas não deveria piorar mais que 0,02. A acurácia passou, mas o esquecimento na Telepatia e a ECE do Felhen estouraram os limites, o que levou à H9b e à H9r.

Limites: uma LoRA só para as 3 famílias, e na Telepatia com k = 100 a avaliação fica com 200 grupos em vez de 300.

---

## H9a: escolher os exemplos em que o Lumi está em dúvida ensina mais que sortear?

Não. Com 25 exemplos, escolher por dúvida empata com sortear na acurácia das duas fontes. A receita segue sendo sortear, usar replay e recalibrar, e o que mais pesa é quantos exemplos usar.

Dois braços com k = 25, replay 1:1 e 3 épocas a partir do `h4_log`, semente 0: `h9r_k25` sorteia os grupos e `h9ar_k25` sorteia dentro da faixa dos 75 grupos de menor confiança do `h4_log`. Os grupos de treino dos dois braços ficam fora da avaliação (21/11).

| Recorte | Métrica | Δ (dúvida − sorteio) | IC 95% |
|---|---|---:|---|
| Felhen reservadas | acc | +1,9 | [−0,7; +4,5] empate |
| Felhen reservadas | NLL | −0,148 | [−0,216; −0,084] dúvida melhor |
| Telepatia reservadas | acc | −1,9 | [−4,2; +0,3] empate |
| Telepatia reservadas | NLL | +0,044 | [+0,007; +0,079] dúvida pior |

O único sinal a favor da dúvida é a NLL do Felhen, e na Telepatia ela vai para o lado contrário, então com uma semente não dá para dizer que ajuda. Na receita completa com 25 rótulos os dois respondem praticamente o mesmo tanto (Felhen 29% com erro 0,16 nos dois). Uma explicação possível é que a faixa de dúvida mistura casos úteis com casos ambíguos ou de rótulo ruim. Comparando com k = 100 e replay na mesma semente, 25 exemplos dão cerca de metade do ganho no Felhen (+6,5 contra +13,7) e cerca de 70% na Telepatia (+12,6 contra +17,7).

Critério escrito antes de rodar: a dúvida ajudaria se o `h9ar_k25` tivesse acurácia nas reservadas acima do `h9r_k25` em pelo menos uma fonte (IC acima de zero) sem perder na outra; empate significaria que não compensa a avaliação extra do pool. Deu empate, e por isso não rodamos as sementes 1 e 2.

Limites: na Telepatia o pool é o próprio `dev`, então a seleção tira casos difíceis da avaliação (dos dois braços, o que mantém a comparação pareada).

---

## H9b: recalibrar depois do treino curto devolve a honestidade?

Devolve. A superconfiança do Felhen depois da H9 era só de escala, e uma temperatura resolve. Com 100 exemplos para treinar e 25 para calibrar, o Lumi responde 68% das faturas novas errando 10%, exatamente o alvo; antes respondia 7% e errava 18%.

Só CPU (`scripts/h9_recalibrar.py`): aplicamos a receita da H7, temperatura e limiar da família com k grupos rotulados, ao `h4_log` e ao `h9_k100`, nas sementes 0, 1 e 2 com 100 sorteios. Os grupos de treino e de calibração ficam fora da medida (21/11).

| Fonte | Braço | ECE sem recalibrar | ECE com T da família (k = 25) | Responde | Erro |
|---|---|---:|---:|---:|---:|
| Felhen | `h4_log` | 0,073 | 0,064 | 16% | 0,19 |
| Felhen | `h9_k100` | 0,113 | 0,069 | 36% | 0,15 |
| Telepatia | `h4_log` | 0,114 | 0,114 | 7% | 0,18 |
| Telepatia | `h9_k100` | 0,074 | 0,045 | 68% | 0,10 |

A ECE do Felhen na H9 recalibrada fica a 0,005 da do `h4_log`. Uma surpresa: depois da H9, a temperatura das famílias vistas já serve nas novas (Felhen 0,059, Telepatia 0,044), enquanto no `h4_log` essa mesma temperatura estragava a Telepatia (0,190). Ou seja, os 25 rótulos servem principalmente para o limiar do "não sei". No Felhen, com k = 50, o Lumi responde 32% e erra 0,12.

Critério escrito antes de rodar: consertaria se, com a temperatura da família e k = 25, a ECE do Felhen reservado da H9 ficasse a no máximo 0,02 da do `h4_log` e a Telepatia não piorasse mais que 0,02; a receita completa valeria a pena se a H9 respondesse mais casos que o `h4_log` com risco médio ≤ 0,15.

Limites: no Felhen, com k = 25, a faixa entre sorteios é larga (cobertura de 6% a 66%, erro de 0,05 a 0,27); k = 50 estreita.

---

## H9r e H9m: como evitar que o treino curto faça o Lumi esquecer?

O replay, que junta um caso já visto para cada caso novo no treino curto, resolve o esquecimento nas vistas da Telepatia sem perder o ganho nas novas. O custo dele (Felhen visto superconfiante) era só de escala e a temperatura global resolve. Misturar os adaptadores com w = 0,5 também evita esquecer, mas ensina menos no Felhen.

O replay (`--adapt-replay 1.0`) repetiu a H9 com k = 100 nas sementes 0, 1 e 2. A mistura não treina de novo: o adaptador fica em ΔW = (1 − w)·ΔW_h4 + w·ΔW_h9, com w = 0,5 e 0,75, só na semente 0 (`scripts/misturar_adaptadores.py`, 21/11).

| Recorte (Δ contra `h4_log`) | H9, média 3 sementes | Replay, média 3 sementes | H9, semente 0 | Mistura w = 0,5, semente 0 |
|---|---:|---:|---:|---:|
| Telepatia vistas, acc | −2,1 | −0,8 | −1,9 | −0,2 |
| Felhen reservadas, acc | +11,6 | +12,1 | +13,6 | +8,7 |
| Telepatia reservadas, acc | +20,3 | +20,9 | +16,6 | +14,2 |
| Felhen vistas, ECE | +0,011 | +0,052 | +0,012 | +0,004 |

Com replay, as vistas da Telepatia perdem só 0,8 ponto e nenhuma semente fica com IC todo abaixo de zero. Em troca, o Felhen visto fica superconfiante, provavelmente porque o replay pega poucos casos de cada família vista e os repete por 3 épocas. Com a temperatura global (T = 1,62) a ECE dele cai de 0,066 para 0,017, praticamente igual ao `h4_log`. Na receita completa com 25 rótulos, o replay responde 37% no Felhen (erro 0,15) e 72% na Telepatia (erro 0,10), contra 32% e 56% da mistura. A mistura w = 0,5 é a mais honesta sem recalibrar, mas perde 4,9 pontos do ganho no Felhen, e w = 0,75 esquece igual à H9.

Critério escrito antes de rodar: resolveria se a acurácia nas vistas da Telepatia ficasse a no máximo 1 ponto do `h4_log`, sem semente com IC todo abaixo de zero, e o ganho nas reservadas ficasse a no máximo 3 pontos do da H9 nas duas fontes; para a mistura, valeria o maior w que cumprisse isso. O replay passou; a mistura deu troca, e as sementes 1 e 2 dela não rodaram.

A receita que fica: perda log (H4), treino curto com cerca de 100 exemplos da família nova mais o mesmo número de casos já vistos (H9 + replay), e temperatura das vistas com limiar da família ajustado em cerca de 25 rótulos (H7/H9b).

Limites: a mistura só tem a semente 0, com w escolhido no próprio `dev`, e as vistas da Telepatia têm só 84 grupos, então diferenças de 1–2 pontos ficam perto do ruído.

---

## H10: o Lumi aprende a matéria nova só com "acertou / errou"? (não rodada)

Não rodamos. O desenho repetia a H6 em cima da receita da H9r, mas num passo só o RL apenas estima com ruído o gradiente que a perda log já dá exato, então no máximo empataria com o `h9r_k100`. Trocamos pelo piloto da H11, em que o Lumi raciocina antes de decidir.

O plano era treinar dois braços iguais ao `h9r_k100` da semente 0 (k = 100, replay 1:1, 3 épocas a partir do `h4_log`), mudando só a perda: `h10_bandit_k100` com `bandit_log` (G = 32, corte 1000) e `h10_acerto_k100` com `bandit_acerto`. Não precisava de código novo.

Critério escrito antes de rodar: H10a, o `h10_bandit_k100` subiria a acurácia das reservadas com IC acima de zero em pelo menos uma fonte e ficaria com pelo menos metade do ganho do `h9r_k100` (Felhen +13,7, Telepatia +17,7); H10b, a ECE das reservadas do `h10_acerto_k100` ficaria acima da do `h10_bandit_k100`, com IC acima de zero. As vistas da Telepatia não poderiam cair mais que 1 ponto.

---

## H11 e H11b: o Lumi consegue raciocinar antes de decidir?

O Lumi raciocina (81% dos raciocínios fecham o `</think>`, 3,9 s cada), mas pelo critério ainda não valia montar o GRPO: a acurácia cai nas vistas da Telepatia e só 65% dos raciocínios sorteados terminam em 512 tokens. O achado principal é que, depois de raciocinar, ele fica superconfiante. Em 24/11/2026 a trilha do raciocínio foi encerrada.

Testamos o adaptador `h4_log` no `dev`, com 8 casos por família (88 no total), de dois jeitos: raciocínio guloso de até 512 tokens e 4 raciocínios sorteados por caso (temperatura 1,0, top_p 0,95), como um grupo do GRPO. A letra é lida depois de `</think>\n\n` e comparada com a leitura de sempre, sem raciocínio (`scripts/h11_piloto.py`).

| Recorte (guloso) | Acurácia sem → com | NLL sem → com | Não fechou |
|---|---|---|---:|
| Felhen vistas | 0,900 → 0,875 | 0,36 → 0,98 | 10% |
| Felhen reservadas | 0,625 → 0,438 | 0,96 → 3,70 | 38% |
| Telepatia vistas | 0,500 → 0,375 | 0,83 → 4,24 | 29% |
| Telepatia reservadas | 0,750 → 0,625 | 0,88 → 3,43 | 0% |

São dois problemas com causas diferentes. A queda de acurácia vem dos raciocínios cortados: nos que não fecharam, ela vai de 0,706 para 0,353, e nos que fecharam cai só 2,8 pontos. A superconfiança vem dos que terminam: o raciocínio acaba dizendo a resposta ("So answer N"), a leitura das letras só copia essa conclusão (confiança média 1,00), e os 21 erros do guloso nesses casos têm confiança de pelo menos 0,9. Com isso, o Brier das letras vira quase só "acertou ou errou", a mesma recompensa que estragou a honestidade na H4. Sinal para o GRPO havia (66% dos grupos), e 600 grupos levariam cerca de 1,7 h de geração.

Critério escrito antes de rodar: pelo menos 80% dos raciocínios fechando, no máximo 10% degenerando, queda de no máximo 5 pontos nas vistas, até 6 s por raciocínio e sinal em pelo menos 30% dos grupos. Falharam a queda nas vistas (Telepatia −12,5; juntas −6,3) e o fechamento no sorteio (65%).

Limites: são só 88 casos com uma semente de sorteio, e o `h4_log` foi treinado com o `<think>` vazio, então parte do efeito pode ser do adaptador.

### H11b: e se ele escrever a confiança?

Ele escreve o número, mas quase sempre "100", e o número não separa bem acertos de erros (AUROC 0,639). A confiança escrita teria de ser aprendida.

Usamos os mesmos 88 casos e raciocínios. Depois da letra, um turno novo pergunta a probabilidade, de 0 a 100, de a resposta estar correta (relatório `analises/h11b.md`).

| Fonte de confiança | AUROC | ECE |
|---|---:|---:|
| confiança escrita | 0,639 | 0,337 |
| letras depois do raciocínio | 0,592 | 0,324 |
| consistência dos 4 sorteios | 0,769 | 0,125 |
| letras sem raciocínio (H4, outra decisão) | 0,798 | 0,066 |

85 dos 88 números ficam entre 90 e 100: ele diz cerca de 95% e acerta 64%. Dentro de cada recorte o AUROC fica entre 0,46 e 0,59, ou seja, ele percebe um pouco que uma matéria é mais difícil, mas não qual caso vai errar. Depois do raciocínio, a melhor confiança é a consistência dos 4 sorteios, ao custo de 4 raciocínios por caso.

Critério escrito antes de rodar: pelo menos 90% de respostas com número, o valor mais comum em no máximo 80% delas e AUROC de pelo menos 0,65. Passou nos dois primeiros (100% e 77%) e falhou no último.

Limites: com 32 erros, o AUROC tem erro de cerca de ±0,06, e a pergunta vem num turno novo, que o `h4_log` nunca viu.

---

## H12: a perda do Clef no Lumi

Não vale trocar. A perda do Clef (entropia cruzada com label smoothing + Brier) acerta um pouco mais na Telepatia (+1 a +2 pontos), mas deixa a confiança menos honesta no Felhen, e a temperatura não conserta tudo. A receita final continua com a perda log.

Treinamos a perda `log_brier` com ε = 0,1 e pesos 1 e 1 (o post do Clef não dá esses valores, então a escolha foi nossa), com todo o resto igual ao `h4_log`. Foram 3 sementes, comparadas no `dev` com `scripts/comparar_h4.py` e `scripts/resumo_calibracao.py`.

| Recorte | Acurácia `log` → `clef` | NLL `log` → `clef` | ECE após temperatura, `log` / `clef` |
|---|---|---|---|
| Felhen vistas | 0,871 → 0,872 | 0,360 → 0,401 | 0,014 / 0,028 |
| Felhen reservadas | 0,529 → 0,522 | 1,239 → 1,268 | 0,062 / 0,101 |
| Telepatia vistas | 0,732 → 0,742 | 0,698 → 0,722 | 0,040 / 0,043 |
| Telepatia reservadas | 0,582 → 0,600 | 0,919 → 0,919 | 0,189 / 0,183 |

A calibração crua piora em 9 das 12 células de ECE, Brier e NLL. O label smoothing ensina o modelo a nunca dar 100% a uma letra, então o Clef sai pouco confiante: a temperatura dele no Felhen é 0,83, contra 1,06 da `log`. Nas reservadas do Felhen, essa temperatura, ajustada nas vistas, deixa o Clef superconfiante e piora o ECE. O único ganho é a acurácia da Telepatia, pequeno perto dos +20 pontos que o treino curto da H9 já dá.

Critério escrito antes de rodar: empate se todos os ICs passassem pelo zero; Clef melhor se o ECE ou o Brier das reservadas caísse sem perder acurácia; Clef pior nos outros casos, conferindo se a diferença some com a temperatura. Deu Clef pior.

Limites: ε e pesos sem busca, posto 16 em vez de 256, só `dev`.

---

## H12b: na perda do Clef, o que faz o quê?

O custo na calibração vem do label smoothing, e o ganho de acurácia na Telepatia vem do Brier. A `log_brier` sem smoothing fica com o ganho (+2,2 pontos nas reservadas da Telepatia, mais que os +1,8 da perda inteira) e quase sem custo, então é ela a versão da perda do Clef que valia levar para a receita final.

Separamos as peças em dois braços novos, `suave` (só o smoothing, ε = 0,1) e `logbrier` (entropia cruzada + Brier), com o resto igual ao `h4_log`, 3 sementes, no `dev`.

| Braço | T no Felhen | ECE Felhen reservadas (após T) | Acurácia Telepatia reservadas, contra `log` |
|---|---:|---:|---:|
| `log` | 1,06 | 0,062 | |
| `suave` | 0,81 | 0,106 | +0,9 |
| `logbrier` | 1,05 | 0,053 | +2,2 |
| `clef` | 0,83 | 0,101 | +1,8 |

O `suave` repete o padrão do Clef: ECE pior nas vistas do Felhen (+0,049), temperatura abaixo de 1 e reservadas do Felhen piores depois dela. O `logbrier` não repete; depois da temperatura, fica igual à `log` ou melhor. Ainda não sabemos por que o Brier ajuda. Uma hipótese, não testada, é que o Brier é limitado (no máximo 2) e a perda log não, então exemplos muito errados puxam menos o treino. O ganho só aparece nas reservadas da Telepatia, que são uma família só; no Felhen a acurácia empata.

Critério escrito antes de rodar: o custo vem do smoothing se o `suave` repetir o padrão do `clef` nas vistas do Felhen e o `logbrier` não; o ganho fica com o braço que repetir +1 a +2 pontos com IC fora do zero. Se o `logbrier` tivesse o ganho sem o custo, o próximo passo seria testá-lo na receita.

Limites: só ε = 0,1 e peso 1 no Brier, sem busca, e só `dev`.

---

## H12c: a perda `log_brier` dentro da receita final

Empate, e a receita continua com a perda log. O ganho de +2,2 pontos da `log_brier` na Telepatia nova (H12b) some depois da receita inteira: os dois braços respondem a mesma fração de casos, com o mesmo erro.

O braço `h9rb_k100` roda a receita final completa com a `log_brier`: parte do `h4_logbrier`, faz o mesmo treino curto do `h9r_k100` (100 exemplos por família nova + replay, 3 épocas) e a mesma recalibração com 25 rótulos. Comparamos semente a semente, com 3 sementes.

| Fonte | `log`: responde / erra | `log_brier`: responde / erra |
|---|---|---|
| Telepatia | 72% / 0,101 | 72% / 0,100 |
| Felhen | 35% / 0,149 | 35% / 0,153 |

A acurácia empata em quase todos os recortes e sementes; só a Telepatia vista na semente 2 sai acima (+1,4). O AUROC dentro da família também fica igual (0,75 contra 0,76 no Felhen, 0,83 nos dois na Telepatia). O ganho da H12b era em matéria que o modelo nunca tinha visto, e depois que o treino curto ensina essa matéria o jeito de pontuar o treino deixa de fazer diferença. Daqui em diante, toda peça nova precisa ser testada dentro da receita.

Critério escrito antes de rodar: entraria na receita se respondesse mais com o erro a no máximo 1 ponto, sem esquecer as vistas (no máximo 1 ponto de acurácia) e com a temperatura ainda consertando a calibração. Empate dentro do ruído mantém a `log`, que é mais simples.

Limites: só `dev`, e na Telepatia a matéria nova é uma família só.

---

## H13: variar o texto do prompt no treino

O `h4_log` é frágil a um prompt que nunca viu, e variar o texto do prompt no treino conserta boa parte disso. Com o prompt novo, o `h4_log` perde 3 a 5 pontos nas famílias vistas e muitas vezes deixa de querer responder com uma letra; o `h4_aug` recupera cerca de 2 pontos e quase sempre responde com letra. Na matéria nova, no prompt de sempre, o efeito depende da fonte: +3,9 pontos na Telepatia e −1,8 no Felhen.

Escrevemos variantes em `src/lumi_decision/prompts.py`: a 0 é o prompt de sempre, as variantes 1 a 3 mudam títulos, frases, ordem dos blocos e formato das opções (na Telepatia, também a ordem das chaves do JSON), e a 4 fica só para a avaliação. O `h4_aug` é igual ao `h4_log`, mas a cada época cada exemplo recebe uma variante de 0 a 3. Foram 3 sementes.

| Recorte (acurácia, pontos) | `log`: nunca visto − de sempre | `aug − log`, de sempre | `aug − log`, nunca visto |
|---|---:|---:|---:|
| Felhen vistas | −2,9 | +0,1 | +2,1 |
| Felhen reservadas | −3,0 | −1,8 | −1,0 |
| Telepatia vistas | −4,6 | +1,0 | +2,3 |
| Telepatia reservadas | +2,0 | +3,9 | +2,5 |

No prompt novo, o `h4_log` chega a ter até 75% dos casos do Felhen com menos da metade da probabilidade nas letras; com o `h4_aug` isso cai para 15% a 34% no Felhen e 0% a 1% na Telepatia. Pelo critério, esse ganho é de robustez. Na matéria nova, a Telepatia sobe mas o ECE piora (+0,019), o que conta como piora, e o Felhen cai. O +3,9 pode nem ser aprendizado: o próprio `h4_log` acerta +2,0 na Telepatia nova só trocando o texto, então parte do ganho pode vir de o `aug` ter visto mais formatos. Isso não foi separado.

Critério escrito antes de rodar: medir quanto o `h4_log` perde no prompt novo e depois comparar `aug − log` nos dois prompts. Ganho só no prompt novo conta como robustez; reservadas melhores no prompt de sempre, sem piorar a calibração, contam como generalização.

Limites: as variantes foram escritas por nós, em português, só 4 no treino, e tudo é no `dev`.

---

## H13b: a receita final com o prompt variado

A receita não muda, porque o prompt variado custa no prompt de sempre. Mas apareceu um limite novo: a receita de hoje é frágil a um prompt que nunca viu. Nele, a Telepatia cai de 72% para 50% de faturas novas respondidas.

O braço `h9ra_k100` usa o prompt variado nas duas etapas (parte do `h4_aug` e o treino curto também usa `--variar-prompt true`), com 3 sementes, contra o `h9r_k100`. Avaliamos no prompt de sempre (`dev`) e na variante 4 (`devvar`), recalibrando com 25 rótulos no próprio prompt avaliado.

| Fonte | Prompt | Receita de hoje: responde / erra | Prompt variado: responde / erra |
|---|---|---|---|
| Telepatia | de sempre | 72% / 0,101 | 66% / 0,102 |
| Telepatia | nunca visto | 50% / 0,104 | 57% / 0,103 |
| Felhen | de sempre | 35% / 0,149 | 33% / 0,156 |
| Felhen | nunca visto | 31% / 0,156 | 31% / 0,165 |

O treino curto não resolveu a fragilidade da H13: no prompt novo, em até 73% dos casos do Felhen a receita de hoje nem quer responder com uma letra. O prompt variado melhora o prompt nunca visto (a Telepatia vai de 50% para 57%, e as vistas do Felhen acertam mais), mas no prompt de sempre a Telepatia responde 6 pontos a menos e a acurácia das reservadas cai (−3,2 no Felhen e −1,5 na Telepatia, na semente 0). É uma troca: deixar de depender do texto exato custa um pouco de precisão no texto de sempre.

Critério escrito antes de rodar: o prompt variado entraria se, no prompt de sempre, ficasse a no máximo 2 pontos de cobertura, 1 de erro e 1 de acurácia nas vistas, e se ganhasse no prompt nunca visto. Falhou na primeira parte, pois o limite era 2 pontos e a Telepatia perdeu 6.

Limites: uma variante nunca vista só, escrita por nós, e a comparação semente a semente no prompt de sempre só saiu para a semente 0.

---

## Sabe que não sabe: o Lumi sabe o que não sabe?

Sim, depois da receita final. Na matéria nova, quando ele diz "não sei", erraria 45% a 47% daqueles casos; quando responde, erra 10% a 16%. A confiança separa acerto de erro dentro da mesma família (AUROC 0,75 a 0,83), então o "não sei" não reflete só a dificuldade da matéria. Antes da receita, o AUROC era bem menor: 0,59 na Telepatia e 0,69 no Felhen.

Usamos `scripts/analise_sabe.py`, sem GPU, sobre as predições do `h4_log` e da receita final (`h9r_k100`), com 3 sementes e a mesma recalibração da receita (25 rótulos, risco-alvo de 10%, 100 repetições). "Não sei" é confiança abaixo do limiar, e o erro dele é o que o modelo erraria se fosse forçado a responder.

| Fonte | Modelo | Responde | Erro quando responde | Erro quando diz "não sei" | AUROC dentro da família |
|---|---|---:|---|---|---|
| Telepatia | `h4_log` | 7% | 19% [3; 32] | 43% [39; 48] | 0,59 [0,56; 0,62] |
| Telepatia | receita final | 72% | 10% [6; 15] | 47% [39; 53] | 0,83 [0,80; 0,85] |
| Felhen | `h4_log` | 16% | 20% [5; 33] | 53% [47; 59] | 0,69 [0,64; 0,73] |
| Felhen | receita final | 35% | 16% [7; 26] | 45% [38; 53] | 0,75 [0,71; 0,79] |

No Felhen, o AUROC dentro da família é igual ao geral (0,75 contra 0,75). Se o "não sei" só refletisse a matéria, ele cairia para perto de 0,5, como aconteceu com a confiança escrita na H11b. O treino curto não só faz o Lumi responder mais, ele aprende a ordenar os casos: na Telepatia, o AUROC vai de 0,59 para 0,83. O preço é que, entre os "não sei", ele acertaria cerca de metade dos casos, e o teste não diz se o "não sei" vem de falta de informação no caso ou de falta de conhecimento do modelo.

Na conferência, a Telepatia e o `h4_log` repetiram os números publicados, mas o Felhen da receita deu 35% / 0,149 contra 37% / 0,15 publicados. O próprio `h9_recalibrar.py`, rodado nos mesmos arquivos, dá 35,1% / 0,149, então a diferença está no número publicado, cuja origem ainda está sendo conferida.

Critério escrito antes de rodar: erro dos "não sei" bem maior que o dos respondidos, com ICs separados, e AUROC dentro da família acima de 0,7. Passou nas duas fontes; no Felhen, por pouco (o IC começa em 0,71).

Limites: só 3 famílias reservadas, e o "não sei" aqui é a regra simples do limiar, não a opção "não sei" da H5.

---

## Onde o Felhen erra, depois da receita final

Na `factck_veracidade`, o Lumi não aprendeu nada: responde "falso" em 95% dos casos, e a acurácia (0,668) é a de chutar sempre a classe mais comum (0,667). Na `juristcu_area`, ele aprendeu de verdade.

Rodamos `scripts/onde_erra.py` na receita final (`h9r_k100`), com 3 sementes, nas famílias reservadas sem os grupos do treino curto, olhando o acerto por classe e a confusão entre gabarito e resposta.

| `factck_veracidade` | No gabarito | Nas respostas do Lumi | Acerto da classe |
|---|---:|---:|---:|
| falso | 66,7% | 95,4% | 0,978 |
| impreciso | 23,5% | 2,5% | 0,056 |
| verdadeiro | 9,8% | 2,1% | 0,033 |

O acerto médio por classe é 0,356, quase o 0,333 de chutar sempre a mesma classe. Dos 60 casos "verdadeiro", 54 viram "falso", indo de uma ponta à outra da escala, que é o erro mais grave. Já a `juristcu_area` (10 classes) acerta 0,644 contra 0,143 da classe mais comum, e a média do Felhen (65%) juntava as duas famílias e escondia isso. Uma explicação ainda não testada: no treino curto, cerca de 2 de cada 3 exemplos são "falso", e a perda log aprende essa proporção sem separar as classes. Na Telepatia (`invoice_processing`, acurácia 0,792), só o `reject`, que é raro, quase nunca sai certo.

Análise exploratória, sem critério escrito antes. Ela levou a H14 a ganhar dois braços com sorteio equilibrado por classe na `factck`, e a H14 e a H15 passam a olhar também o acerto médio por classe.

Limites: a `factck` tem 102 grupos no `dev`, com só 6 a 10 casos "verdadeiro" por semente.
