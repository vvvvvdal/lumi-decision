# Plano H15: o RLCD do Clef no Lumi

Registrado antes de rodar (25/11/2026).

## Pergunta

O RLCD do Clef melhora o Lumi dentro da receita final? A pergunta vale principalmente para a família em escala
que hoje é a mais fraca do Felhen, a `factck_veracidade` (falso < impreciso < verdadeiro).

## O que as fontes dizem (lido em 25/11)

- **TypeSafe, que criou o nome:** não publica algoritmo, recompensa nem dados.
- **Cloudflare (Clef):** é o único que descreve as peças.
  - O RLCD é um objetivo secundário, depois da CE com label smoothing + Brier, com três termos:
    1. crédito parcial para a opção vizinha numa escala;
    2. prêmio para o registro inteiro certo;
    3. penalidade de referência.
  - Ele age numa cabeça que dá um logit por opção, não em texto.
  - Não publica a fórmula, o algoritmo, os pesos nem o efeito isolado do RL.
- **O que segue abaixo é a nossa versão, a partir dessa descrição.**

## Desenho

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

## Critério

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

## Como rodar

1. **Preencher `configs/escalas.toml`** com as escalas do inventário (etapa 0).
2. **Teste curto:** a perda cai, sem `nan`, e o `treino_meta.json` mostra `rlcd_exemplos_com_escala` > 0.
3. **3 sementes de cada versão** (comandos na mensagem do ciclo).

## Limites

- A fórmula é nossa: o Clef não publica a dele.
- Um λ e um β só.
- A `factck` tem 102 grupos no `dev`.
