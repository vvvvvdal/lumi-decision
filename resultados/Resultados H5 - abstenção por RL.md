# H5: abstenção aprendida por RL no Lumi (resultado, semente 0)

**Data:** 19/11/2026
**Branch:** `vvvvvdal/lumi-decision`, `h5-abster` (commits `a60c0d4`, `18943ba`)
**Plano registrado antes de rodar:** `analises/plano_h5_abster.md`
**Status:** semente 0 só, no `dev`; o `test` não foi usado. Pela regra do plano, réplicas só se a hipótese
principal parecesse positiva, e ela não pareceu.

## O que foi testado

Braço (e): uma opção extra "Não sei / passar para um humano" e uma perda com três partes:
- **crença:** log score nas opções normais;
- **decisão:** responder ou abster, aprendida por REINFORCE com recompensa +1 / −λ / 0;
- **coleira:** KL para o Lumi sem adaptador.

Mesma receita do E3/`h4_log` (dados, LoRA, passos). Rodamos λ = 3 (abster se P(acerto) < 0,75) e λ = 9
(< 0,9).

**Jobs:** teste curto 37648; λ = 3 e λ = 9 completos + `res2`; análise 37657 (`scripts/analise_abster.py`).

## Resultado 1: a crença continuou honesta ✅

ΔECE nas opções normais contra o `h4_log`: empate ou melhor em todos os grupos (Telepatia reservadas:
−0,051 com λ = 3, −0,027 com λ = 9). Separar crença e decisão funcionou: o RL não deixou o modelo
superconfiante.

## Resultado 2: a hipótese principal não se sustentou ❌

Risco entre os respondidos, **na mesma cobertura**, contra o `h4_log` + limiar (λ = 3):

| Fonte | Grupo | Cobertura | Risco (e) | Risco base | Δ [IC 95%] |
|---|---|---:|---:|---:|---|
| Felhen | vistas | 0,865 | 0,083 | 0,078 | +0,006 [−0,003; +0,015] empate |
| Felhen | reservadas | 0,672 | 0,480 | 0,411 | **+0,069 [+0,027; +0,113] pior** |
| Telepatia | vistas | 0,089 | 0,067 | 0,013 | +0,053 [+0,012; +0,119] pior |
| Telepatia | reservadas | 0,075 | 0,124 | 0,124 | +0,000 empate |

O "não sei" aprendido escolhe quais casos responder **igual ou pior** que a confiança do `h4_log`. Na matéria
nova do Felhen, pior.

Na Telepatia o modelo abstém em ~91% dos casos. Faz sentido pela regra: lá a acurácia nas vistas (~73%) fica
abaixo do limiar de 0,75 do λ = 3, e o alvo é suave (`gold_dist`), o que baixa a recompensa de responder.
Com λ = 9, quase não responde: cobertura de 0–4% na Telepatia e 18% nas reservadas do Felhen.

## Resultado 3: o "falta informação" apareceu pela primeira vez, com custo ⚠️

`regra_beneficio` (reservadas v2): em quantos casos o modelo absteve **ou** pediu o documento, por resposta
certa.

| Resposta certa | h4_log | h5 λ = 3 | h5 λ = 9 |
|---|---:|---:|---:|
| conceder (documentos completos) | 0/332 | **7/332 (2%)** | 299/332 |
| encaminhar | 0/316 | 61/316 (19%) | 316/316 |
| negar | 7/312 | 106/312 (34%) | 190/312 |
| **pedir_documento** | 3/272 | **109/272 (40%)** | 272/272 |

- **λ = 9 abstém em quase tudo**, então seus 100% não indicam nada.
- **λ = 3 discrimina:** 40% quando falta documento contra 2% quando tudo está presente. No gerador da família
  (`res2.py`), cerca de 60% dos casos de "negar" também têm documento faltando, o que é coerente com os 34% em
  "negar".
- Os modelos anteriores do projeto chegavam no máximo a 19/272 nessa classe (M2). Este é o primeiro sinal de
  que o modelo reage à falta de informação.
- O custo: ele também abstém em casos decidíveis, e por isso não vence a base na mesma cobertura.

### Réplica do sinal com 3 sementes (20/11, job 37673, `scripts/analise_falta_info.py`)

λ = 3, sementes 0, 1 e 2 (jobs de treino 37658 e 37660). IC 95% reamostrando grupos de casos (308 grupos),
juntando sementes e rotações.

| Resposta certa | Abstém ou pede o documento | s0 | s1 | s2 |
|---|---:|---:|---:|---:|
| conceder | 0,016 | 7/332 | 7/332 | 2/332 |
| encaminhar | 0,114 | 61/316 | 41/316 | 6/316 |
| negar | 0,378 | 106/312 | 126/312 | 122/312 |
| pedir_documento | 0,309 | 109/272 | 105/272 | 38/272 |

| Diferença | Δ | IC 95% | Leitura |
|---|---:|---|---|
| pedir_documento − conceder | +0,293 | [+0,217; +0,375] | reage mais quando falta documento |
| pedir_documento − encaminhar | +0,195 | [+0,109; +0,284] | reage mais quando falta documento |
| pedir_documento − negar | −0,069 | [−0,176; +0,043] | empate |

- **O sinal se confirma:** nas 3 sementes, o braço (e) reage muito mais quando falta documento (31%) do que
  quando dá para conceder (1,6%), com o IC bem acima de zero.
- **Ele não separa "pedir documento" de "negar":** as duas classes ficam empatadas. É coerente com o gerador,
  em que boa parte dos casos de "negar" também tem documento faltando. O modelo parece reagir a "falta
  documento", não à decisão certa a tomar.
- **Variação entre sementes:** a semente 2 reage bem menos em `pedir_documento` (38/272 contra ~107/272).
  O IC acima reamostra casos e não mede essa variação entre treinos; o efeito existe nas 3, mas o tamanho
  varia.

### O teste limpo: é falta de documento ou outra coisa? (20/11, job 37674)

`scripts/analise_falta_info_fatores.py` usa os bastidores do gerador (`meta["dados"]`: documentos exigidos ×
anexados, valor × teto). O teste limpo é **dentro de "negar"**: a resposta certa é a mesma, e só muda se
falta documento.

| Caso | h4_log (s0) | h5 λ = 3 (s0, s1, s2) |
|---|---:|---:|
| negar COM documento faltando | 0,036 | 0,362 |
| negar SEM documento faltando | 0,000 | 0,405 |
| pedir_documento (sempre falta) | 0,011 | 0,309 |
| conceder (nada falta, valor ok) | 0,000 | 0,016 |
| encaminhar (nada falta, valor acima) | 0,000 | 0,114 |

| Contraste (h5 λ = 3) | Δ | IC 95% | Leitura |
|---|---:|---|---|
| negar: com − sem documento faltando | −0,043 | [−0,188; +0,114] | **empate** |
| valor acima − ok, sem documento faltando | +0,098 | [+0,053; +0,143] | maior |

**Isto corrige a leitura anterior.** O braço (e) **não** detecta a falta de documento: nos casos de "negar",
ele abstém igual com ou sem documento faltando. O padrão real é outro: ele abstém **quando o caso tem algum
problema** (regra violada, documento faltando ou valor acima do teto) e quase nunca quando está tudo limpo
(conceder, 1,6%). É um sinal de "caso complicado", não de "falta informação". A comparação
`pedir_documento − conceder` parecia isolar a falta de documento, mas misturava vários fatores.

## Conclusão

> Treinar a abstenção como ação, com a crença separada, **mantém o Lumi honesto**. O modelo passa a abster
> em casos "com problema" e quase nunca nos limpos, nas 3 sementes. Mas o teste limpo mostra que ele **não
> detecta a falta de documento em si**, e ele **não supera a regra simples** de abster pela confiança: na
> mesma cobertura, erra igual ou mais. A H5 principal não se confirma (semente 0). A secundária parecia se
> confirmar (31% contra 1,6%), mas o controle por fatores mostrou que era confundida.

## Limites

- Uma semente. O sinal do "falta informação" (40% contra 2%) é grande, mas não foi replicado.
- λ fixo por fonte. A Telepatia, com acurácia menor e alvo suave, precisaria de outro λ (ou de recompensa
  com alvo duro) para não colapsar em abster.
- A análise do `res2` não usa IC.

## Próximos passos possíveis

1. **Replicar só o sinal do "falta informação"** (λ = 3, sementes 1 e 2) e medir com IC a diferença entre
   `pedir_documento` e `conceder`. É o achado mais promissor.
2. λ por fonte, ou recompensa com o gabarito duro na Telepatia.
3. Combinar: confiança do `h4_log` para escolher os casos + o "não sei" aprendido só como sinal de falta de
   informação.
4. Levar ao Palhares: o achado conversa direto com o ponto cego da M2.
