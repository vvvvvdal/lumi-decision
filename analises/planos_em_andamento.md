# Planos em andamento

Planos registrados antes de rodar e ainda sem resultado. Quando um experimento termina, a seção dele vai para `resultados/Experimentos.md`, junto com o resultado.

---

## Plano H14: os botões do treino curto, para subir o Felhen

Registrado antes de rodar (25/11/2026).

### Pergunta

Na receita final, o Felhen novo acerta 65% e responde só 35% dos casos (a Telepatia responde 72%). Os botões do
treino curto (k, épocas, `lr`) nunca foram ajustados. Mexer neles faz o Felhen responder mais sem errar mais?

### O que a etapa 0 mostrou (26/11, antes de rodar a triagem)

`analises/onde_erra.md`, na receita (`h9r_k100`, 3 sementes):
- **`factck_veracidade`:** o Lumi responde "falso" em **95%** dos casos.
  - A acurácia (0,668) é igual à de chutar sempre a classe mais comum (0,667).
  - Acerta 6% dos "impreciso" e 3% dos "verdadeiro".
  - **Na `factck`, ele não aprendeu nada além da classe mais comum.**
- **`juristcu_area`:** aprendeu de verdade (0,64, contra 0,14 de chutar a classe mais comum). As classes mais
  fracas são `gestao_administrativa` (0,36), `desestatizacao` e `financas_publicas` (cerca de 0,5).
- **Pool do treino curto (`train` das reservadas):** 927 grupos na `factck` e 3.240 na `juristcu`. Então k = 300
  cabe nas duas.

Por isso entram dois braços com **sorteio equilibrado por classe** na `factck`
(`--adapt-balancear felhen:factck_veracidade`): um rodízio entre as classes, que acaba sendo a classe rara.

### Desenho

- **O que muda:** só a 2ª etapa da receita, o treino curto a partir do `h4_log`, com replay e a mesma
  recalibração (25 rótulos).
- **Código novo:** `--adapt-familias` aceita `fonte:família=k`, para um k só daquela família.
  - O Felhen tira os exemplos do `train` das reservadas, que nunca foi usado, então subir o k não encolhe a
    avaliação.
  - A Telepatia fica em k = 100, porque os exemplos dela saem do `dev`.
- **Triagem, semente 0:**

| Braço | k do Felhen | Épocas | `lr` |
|---|---|---|---|
| `h9r_k100` (já existe) | 100 | 3 | 1e-4 |
| `h14_k300` | 300, ou o pool inteiro se for menor | 3 | 1e-4 |
| `h14_ep6` | 100 | 6 | 1e-4 |
| `h14_lr2` | 100 | 3 | 2e-4 |
| `h14_lr05` | 100 | 3 | 5e-5 |
| `h14_k300_ep6` | 300 | 6 | 1e-4 |
| `h14_bal` | 100, equilibrado por classe na `factck` | 3 | 1e-4 |
| `h14_k300_bal` | 300, equilibrado por classe na `factck` | 3 | 1e-4 |

Análise encadeada: `slurm/analise_h14.sbatch`. Os grupos de treino do `h9r_k100` também saem da medida
(`--excluir-tambem h9r_k100`), porque os braços equilibrados sorteiam outros grupos.

### Critério

Análise: `h9_recalibrar.py --base h9r_k100 --novo <braço> --sementes 0` e `analise_h9.py --base h9r_k100`.

- **Um braço passa na triagem** se:
  - o Felhen responder pelo menos 5 pontos a mais;
  - o erro dos respondidos ficar a no máximo 1 ponto;
  - não perder mais de 1 ponto nas vistas nem na cobertura da Telepatia.
- **O melhor braço que passar** roda nas sementes 1 e 2, e vale a média das 3.
- **Se nenhum passar,** a 2ª etapa está no limite. O próximo botão é a 1ª etapa (posto 64 e `lr` no `h4_log`),
  que é mais cara.
- **Critério novo, para a `factck` (`onde_erra.py`):**
  - o acerto médio por classe tem que passar de 0,40. Hoje é cerca de 0,36, e chutar sempre a mesma classe dá
    0,33;
  - sem a acurácia cair mais que 3 pontos.

  Um braço que só melhore a `factck` assim também conta como sinal, mesmo sem os 5 pontos de cobertura.
- **Pelo motivo certo?**
  - Conferir no `onde_erra.py` se o ganho vem das duas famílias, ou só de uma.
  - Conferir também se o ganho vem de acertar as classes raras, ou só de chutar mais a classe comum.

### Limites

- A triagem tem uma semente só. Por isso a régua é de 5 pontos.
- Na `factck` (102 grupos no `dev`), a variação entre sementes é grande.

---

## Plano H15: o RLCD do Clef no Lumi

Registrado antes de rodar (25/11/2026).

### Pergunta

O RLCD do Clef melhora o Lumi dentro da receita final? A pergunta vale principalmente para a família em escala
que hoje é a mais fraca do Felhen, a `factck_veracidade` (falso < impreciso < verdadeiro).

### O que as fontes dizem (lido em 25/11)

- **TypeSafe, que criou o nome:** não publica algoritmo, recompensa nem dados.
- **Cloudflare (Clef):** é o único que descreve as peças.
  - O RLCD é um objetivo secundário, depois da CE com label smoothing + Brier, com três termos:
    1. crédito parcial para a opção vizinha numa escala;
    2. prêmio para o registro inteiro certo;
    3. penalidade de referência.
  - Ele age numa cabeça que dá um logit por opção, não em texto.
  - Não publica a fórmula, o algoritmo, os pesos nem o efeito isolado do RL.
- **O que segue abaixo é a nossa versão, a partir dessa descrição.**

### Desenho

- **Perda `rlcd`** (`lora.rlcd_loss`):
  - perda = log score (a base honesta, no lugar da CE com smoothing, que piorou na H12b)
    − λ · Σₐ p(a) · R(a)
    + β · KL(p ‖ p_ref);
  - **R(a), o crédito parcial:** 1 no gabarito, 0,5 na opção vizinha da escala e 0 no resto. As escalas ficam em
    `configs/escalas.toml`, preenchido a partir de `analises/escalas_inventario.md`. Fora das escalas, R é só o
    acerto;
  - **p_ref, a referência:** o modelo no começo do treino. Na receita, é o `h4_log`;
  - **pesos:** λ = 1 e β = 0,02 (o β da H5). Sem busca.
- **Duas versões:**
  - `rlcd`: a recompensa esperada exata. É derivável, porque a leitura é numa passada só;
  - `rlcd_amostrado`: o mesmo objetivo por REINFORCE, sorteando 8 opções. Esta é a versão "RL de verdade".
  - Em média, os gradientes das duas são iguais (`tests/test_h15_rlcd.py`).
- **Fora desta hipótese:** o prêmio do registro inteiro (as 5 perguntas da Telepatia). Fica para uma H15b, só
  se a H15 der sinal.
- **Onde testar:** dentro da receita. É o `h9r_k100` com `--loss rlcd` (`h15_rlcd`) ou `--loss rlcd_amostrado`
  (`h15_rlcda`), nas 3 sementes, contra o `h9r_k100`.
  - A `factck` só é treinada no treino curto.
  - A H12c mostrou que uma peça nova tem que ser testada dentro da receita.

### Critério

- **Ajuda** se, nas 3 sementes juntas:
  - na `factck`, a fração de erros graves (falso ↔ verdadeiro) cair (`scripts/onde_erra.py`);
  - a acurácia da `factck` não cair mais que 1 ponto;
  - a calibração depois da temperatura não piorar (ECE e NLL, `analise_h9.py`);
  - a cobertura da receita ficar a no máximo 1 ponto (`h9_recalibrar.py`).
- **Empate:** nada muda além do ruído. A `log` continua, porque é a mais simples.
- **Exato contra amostrado:** se os dois empatarem entre si, isso confirma que o RL sobre letras é o mesmo
  objetivo com ruído, como na H6.
- **Pelo motivo certo?** O ganho tem que aparecer onde há escala (a `factck`), e não nas famílias sem escala.
  Se aparecer só fora da escala, quem fez o efeito foi o KL ou a recompensa de acerto, e não o crédito parcial.

### Como rodar

1. **Escalas preenchidas em 26/11** (`configs/escalas.toml`, antes de rodar):
   - `falso < impreciso < verdadeiro` (a `factck`);
   - `0 < 1 < 2 < 3`: risco, frustração, divergência da fatura e urgência na Telepatia, incluindo a `invoice`
     reservada;
   - `0 < … < 4`: a gravidade na `security_incidents`.

   Ficaram de fora os conjuntos sem ordem clara.
   - **Atenção (etapa 0):** na receita, a `factck` responde "falso" em 95% dos casos. O crédito parcial pode
     ajudar a sair desse chute, ou não fazer nada, se o modelo não distingue as classes. O critério do erro grave
     continua valendo, e o acerto médio por classe também entra.
2. **Teste curto:** a perda cai, sem `nan`, e o `treino_meta.json` mostra `rlcd_exemplos_com_escala` > 0.
3. **3 sementes de cada versão** (comandos na mensagem do ciclo).

### Limites

- A fórmula é nossa: o Clef não publica a dele.
- Um λ e um β só.
- A `factck` tem 102 grupos no `dev`.

---

## Plano H16: mais famílias em português, para a matéria nunca vista

Registrado antes de rodar (25/11/2026).

### Pergunta

Sem nenhum exemplo de uma família nova, o Lumi acerta pouco. Treinar com **mais tipos de decisão em português**
melhora o desempenho nas famílias que ele nunca viu?

### O que já se sabe

- **O E3b treinou com o Eikos** (27 famílias, 87% em inglês), em `analises/e3br_replica.md`:
  - ajudou nas famílias sintéticas novas (res2, +3,4 nas 2 sementes);
  - ajudou nas 7 reservadas juntas (+1,7 a +2,8);
  - **não ajudou** nas 3 reservadas do Felhen e da Telepatia.
- O plano da fase 2 apontou o idioma como motivo provável.
- O braço "diverso" da fase 2 (intenção e emoção em português) foi desenhado. Se ele rodou, isso se confere na
  etapa 0, e o resultado é lido antes deste.

### Desenho

- **Braço `h4_pt`:** igual ao `h4_log`, com fontes a mais:
  - `--sources felhen,telepatia,traces,eikos_pt,f3`;
  - `eikos_pt`: a parte em português do Eikos;
  - `f3`: `massive_intencao` e `goemo_emocao`. A `pss_mais_simples` é reservada e fica fora.
- **O mesmo número de atualizações** do `h4_log` (`--max-steps`, lido de `runs/h4_log/treino_meta.json`). A
  única diferença é a variedade dos dados. O braço novo vê menos exemplos de cada fonte antiga.
- **Sementes:** 0 e 1 (`h4_pt`, `h4_pt_s1`), contra o `h4_log` e o `h4_log_s1`.

### Critério

- **Principal** (`comparar_h4.py --base log --novos pt --sementes 0,1`):
  - "ajuda" se a acurácia das reservadas subir com o IC fora do zero, nas duas fontes ou na média, sem perder
    mais de 1 ponto nas vistas;
  - "empate" se o IC passar pelo zero;
  - "piora" se as reservadas ou as vistas caírem.
- **Secundário:** as 4 famílias do res2 (`slurm/m2_res2.sbatch`), para comparar com o E3b.
- **Se ajudar:** rodar a receita a partir do `h4_pt` (H9r), porque a H12c mostrou que um ganho antes do treino
  curto pode sumir na receita.
- **Se ajudar e sobreviver:** gerar famílias sintéticas novas em português no `res2.py`, separadas das 4 de
  avaliação.

### Como rodar

```bash
cd /raid/$USER/lumi-decision && git pull -q && P=$(python3 -c "import json;print(json.load(open('runs/h4_log/treino_meta.json'))['passos_total'])") && echo "passos: $P" && for s in 0 1; do n=$([ $s = 0 ] && echo "" || echo "_s$s"); sbatch slurm/e3_lora_h2.sbatch h4_pt$n --sources felhen,telepatia,traces,eikos_pt,f3 --max-steps $P --seed $s; done; squeue -u $USER
```

### Limites

- Só 2 sementes, como no E3b.
- Na Telepatia, as reservadas são uma família só.
