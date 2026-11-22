"""Montagem do prompt e das opções apresentadas.

A decisão é lida exatamente na posição em que a geração do Lumi escreve a letra:
logo após `<|im_start|>assistant\\n<think></think>`, com letras sem espaço antes.
Qualquer mudança no texto abaixo muda PROMPT_VERSION.
"""

from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .schema import Case

PROMPT_VERSION = "v1"
LETTERS = string.ascii_uppercase
MAX_OPTIONS = len(LETTERS)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def load_template(path: Path):
    """Mesmo ambiente Jinja que o transformers usa em `apply_chat_template`."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    return env.from_string(path.read_text(encoding="utf-8"))


ABSTER = ("__abster__", "Não sei / passar para um humano.")  # H5: opção de abstenção (sempre a última)


def with_abstain(case: Case) -> Case:
    """Cópia do caso com a opção ABSTER no fim (H5). Não existe gabarito "abster"."""
    if any(o[0] == ABSTER[0] for o in case.options):
        return case
    d = case.to_dict()
    d["options"] = d["options"] + [list(ABSTER)]
    return Case.from_dict(d)


def order_options(case: Case, order: str) -> list[tuple[str, str]]:
    """`canonical` (ordem do dataset), `reversed`, `shuffle<seed>` (determinística por caso), `rot<k>` (rotação
    de k posições, módulo n) ou `perm<índices>` (permutação explícita, ex.: `perm201`; exige n = nº de dígitos).
    A opção ABSTER (H5), se existir, fica fora da ordenação e sempre por último; `n` conta só as outras."""
    opts = [o for o in case.options if o[0] != ABSTER[0]]
    if len(opts) != len(case.options):
        return _order(case, opts, order) + [tuple(ABSTER)]
    return _order(case, opts, order)


def _order(case: Case, opts: list, order: str) -> list[tuple[str, str]]:
    if order == "canonical":
        return opts
    if order == "reversed":
        return opts[::-1]
    if order.startswith("shuffle"):
        seed = int(order.removeprefix("shuffle") or 0)
        random.Random(f"{seed}:{case.id}").shuffle(opts)
        return opts
    if order.startswith("rot"):
        k = int(order.removeprefix("rot")) % len(opts)
        return opts[k:] + opts[:k]
    if order.startswith("perm"):
        idx = [int(ch) for ch in order.removeprefix("perm")]
        if sorted(idx) != list(range(len(opts))):
            raise ValueError(f"{case.id}: {order} não é permutação de {len(opts)} opções")
        return [opts[i] for i in idx]
    raise ValueError(f"ordem desconhecida: {order}")


def user_message(case: Case, options: list[tuple[str, str]]) -> str:
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"{case.id}: {len(options)} opções (suportado: 2..{MAX_OPTIONS})")
    choices = "\n".join(f"{LETTERS[i]}. {desc}" for i, (_, desc) in enumerate(options))
    return (f"Contexto:\n{case.state}\n\nPergunta: {case.question}\n\n"
            f"Opções:\n{choices}\n\nResponda apenas com a letra da opção escolhida.")


def option_char_ends(prompt: str, case: Case, options: list[tuple[str, str]]) -> list[int]:
    """Índice (no prompt renderizado) do último caractere não branco de cada linha de opção "X. descrição".
    Recalcula a mensagem com `user_message` (o prompt NÃO muda) e exige que ela apareça uma única vez."""
    msg = user_message(case, options)
    if prompt.count(msg) != 1:
        raise ValueError(f"{case.id}: mensagem do usuário não aparece exatamente uma vez no prompt")
    head = f"Contexto:\n{case.state}\n\nPergunta: {case.question}\n\nOpções:\n"
    if not msg.startswith(head):
        raise ValueError(f"{case.id}: formato da mensagem mudou")
    pos, ends = prompt.index(msg) + len(head), []
    for i, (_, desc) in enumerate(options):
        line = f"{LETTERS[i]}. {desc}"
        ends.append(pos + len(line.rstrip()) - 1)
        pos += len(line) + 1
    return ends


def option_token_positions(offsets, prompt: str, char_ends: list[int]) -> list[int]:
    """Token que contém o último caractere de cada opção (E4: chave da pointer head). O token pode avançar
    sobre a quebra de linha seguinte, mas só sobre espaço/quebra de linha; posições estritamente crescentes."""
    import bisect

    spans = sorted((s, e, t) for t, (s, e) in enumerate(offsets) if e > s)  # tokens especiais podem ter (0, 0)
    starts = [s for s, _, _ in spans]
    out = []
    for c in char_ends:
        s, e, t = spans[max(0, bisect.bisect_right(starts, c) - 1)]
        if not s <= c < e:
            raise ValueError(f"caractere {c} sem token (offsets {(s, e)})")
        if prompt[c + 1:e].strip():
            raise ValueError(f"token da opção avança sobre texto não branco: {prompt[s:e]!r}")
        if out and t <= out[-1]:
            raise ValueError("duas opções no mesmo token")
        out.append(t)
    if out and out[-1] >= len(offsets) - 1:
        raise ValueError("opção na última posição do prompt")
    return out


def exemplo_turnos(ex: Case) -> list[dict]:
    """H8: um caso com gabarito vira um par user/assistant (opções na ordem canônica; resposta = letra certa)."""
    opts = order_options(ex, "canonical")
    letra = LETTERS[[o[0] for o in opts].index(ex.gold)]
    return [{"role": "user", "content": user_message(ex, opts)}, {"role": "assistant", "content": letra}]


def render(template, case: Case, options: list[tuple[str, str]], exemplos=(), pensar: bool = False) -> str:
    """Prompt do caso. H8: `exemplos` (casos com gabarito da mesma família) entram antes, como turnos
    user/assistant já respondidos. Sem exemplos, o prompt é exatamente o de sempre.
    H11: `pensar=True` pede ao template o modo com raciocínio (`<think>` aberto)."""
    msgs = [t for ex in exemplos for t in exemplo_turnos(ex)]
    msgs.append({"role": "user", "content": user_message(case, options)})
    return template.render(messages=msgs, add_generation_prompt=True, enable_thinking=pensar)


FIM_THINK = "</think>"


def prompt_pensando(prompt_on: str, prompt_off: str) -> str:
    """H11: prompt em que o Lumi começa a raciocinar. Se o template ignorar `enable_thinking=True` (os dois
    prompts iguais), abre o `<think>` tirando o `</think>` final do prompt de sempre."""
    if prompt_on != prompt_off:
        return prompt_on
    if prompt_off.endswith(FIM_THINK):
        return prompt_off[: -len(FIM_THINK)]
    raise ValueError("o prompt não termina em </think>; não sei onde abrir o raciocínio")


def com_raciocinio(prompt_gen: str, prompt_off: str, texto: str) -> str:
    """H11: prompt de leitura depois do raciocínio. Corta o texto gerado no primeiro `</think>`, fecha o
    raciocínio e termina igual ao prompt de sempre (o que vem depois do `</think>` nele), para a letra ser
    lida na mesma fronteira."""
    if FIM_THINK not in prompt_off:
        raise ValueError("o prompt de sempre não tem </think>")
    sufixo = prompt_off.rsplit(FIM_THINK, 1)[1]
    return prompt_gen + texto.split(FIM_THINK, 1)[0] + FIM_THINK + sufixo


# H11: depois do `</think>` o Lumi escreve um separador antes da resposta. O 1º teste curto (job 37782) leu
# a letra colada no `</think>` e saiu quase uniforme. Agora lemos as três variantes; a principal é "nl2".
SUFIXOS_RAC = {"cru": "", "nl": "\n", "nl2": "\n\n"}
SUFIXO_RAC_PRINCIPAL = "nl2"

_RESPOSTA_RE = re.compile(r"(?:answer|resposta|op[çc][ãa]o|option)\W{0,4}(?:is\W{0,3}|é\W{0,3})?"
                          r"(?:option\W{0,3}|op[çc][ãa]o\W{0,3}|letra\W{0,3})?\**\(?([A-Z])\b", re.IGNORECASE)


def letra_do_raciocinio(texto: str, n: int) -> str | None:
    """H11: a última letra que o raciocínio diz ser a resposta ("answer N", "That's option E", "Resposta: B").
    Só letras entre as `n` opções; None se o texto não disser."""
    achadas = [m[1].upper() for m in _RESPOSTA_RE.finditer(texto.split(FIM_THINK, 1)[0])]
    validas = [x for x in achadas if x.isupper() and LETTERS.index(x) < n]
    return validas[-1] if validas else None


# H11b: a confiança ESCRITA (estilo RLCR), perguntada num turno novo do usuário depois da resposta. Depois de um
# raciocínio que termina, a leitura das letras sai com ~100% (copia a conclusão do texto); o número escrito é uma
# confiança que o modelo pode graduar.
PERGUNTA_CONFIANCA = ("Qual é a probabilidade, de 0 a 100, de que a sua resposta esteja correta? "
                      "Responda apenas com o número.")
_SENTINELA = "@@RESPOSTA_DO_ASSISTENTE@@"


def turno_confianca(template, pergunta: str = PERGUNTA_CONFIANCA) -> str:
    """H11b: o texto que vem DEPOIS da resposta do assistente: o fim do turno dele, a pergunta do usuário e o
    começo do turno novo (no modo de sempre, `<think></think>`). Sai do próprio template, com uma resposta-
    sentinela no histórico, para não escrever tags à mão."""
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": _SENTINELA},
            {"role": "user", "content": pergunta}]
    txt = template.render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
    if txt.count(_SENTINELA) != 1 or pergunta not in txt.split(_SENTINELA, 1)[1]:
        raise ValueError("o template não mostra a resposta do assistente uma vez antes da pergunta nova")
    return txt.split(_SENTINELA, 1)[1]


def prompt_confianca(prompt_lido: str, letra: str, depois: str) -> str:
    """H11b: o prompt em que a letra foi lida (raciocínio + separador), a letra escolhida e o turno da pergunta."""
    return prompt_lido + letra + depois


_NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%?)")


def ler_confianca(texto: str) -> float | None:
    """H11b: o primeiro número da resposta, como probabilidade de 0 a 1 ("85", "85%", "Confiança: 90%"). Um
    decimal até 1 sem % ("0,7", "0.7") já é probabilidade. Sem número ou fora de 0–100: None."""
    m = _NUM_RE.search(texto)
    if not m:
        return None
    v = float(m[1].replace(",", "."))
    if v <= 1 and not m[2] and any(s in m[1] for s in ",."):
        return v
    return v / 100 if 0 <= v <= 100 else None


def candidate_tokens(tokenize, prompt: str, n: int) -> tuple[list[int], list[int]]:
    """Tokeniza o prompt e devolve (ids do prompt, id de cada letra candidata).

    Exige que cada letra seja exatamente um token novo sem alterar a tokenização do prompt;
    caso contrário o readout não corresponderia à geração.
    """
    base = tokenize(prompt)
    cands = []
    for letter in LETTERS[:n]:
        ids = tokenize(prompt + letter)
        if len(ids) != len(base) + 1 or ids[:-1] != base:
            raise ValueError(f"letra {letter} não é um único token na fronteira do prompt")
        cands.append(ids[-1])
    if len(set(cands)) != n:
        raise ValueError("letras candidatas com ids repetidos")
    return base, cands
