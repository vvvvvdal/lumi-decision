"""H17: decisões de agente como decisões tipadas (só avaliação).

- `bfcl_cases`: escolher a função certa no BFCL (Berkeley Function Calling Leaderboard, v3), categorias `multiple`,
  `live_multiple`, `irrelevance` e `live_irrelevance`. Opções = as funções oferecidas (nome + descrição) e
  "nenhuma"; o gabarito é a função da resposta esperada, ou "nenhuma" nas categorias de irrelevância.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .data import DATA_DIR, DISCARDED, sha256_file
from .prompts import LETTERS
from .schema import Case

BFCL = {"repo": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"}
# categoria -> (arquivo das perguntas, arquivo do gabarito ou None)
BFCL_ARQUIVOS = {
    "multiple": ("BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json"),
    "live_multiple": ("BFCL_v3_live_multiple.json", "possible_answer/BFCL_v3_live_multiple.json"),
    "irrelevance": ("BFCL_v3_irrelevance.json", None),
    "live_irrelevance": ("BFCL_v3_live_irrelevance.json", None),
}
BFCL_SHA256 = {
    "BFCL_v3_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "BFCL_v3_live_multiple.json": "af2520ac5bb0b05aa9b8abe14a79b87198e5a3137a707d4f5633c792330f0cee",
    "BFCL_v3_irrelevance.json": "975f51c51f688649fd190078efd87081241e0a326f9114a2ea3c1ca2440d8690",
    "BFCL_v3_live_irrelevance.json": "0d259e1c3ab6ba06c2a51911e2f49ecaa2eac3723a4a14f183a7687955e61338",
    "possible_answer/BFCL_v3_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    "possible_answer/BFCL_v3_live_multiple.json": "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084",
}
NENHUMA = ("nenhuma", "Nenhuma: nenhuma das funções serve para este pedido.")
PERGUNTA_BFCL = ("Qual função o assistente deve chamar para atender este pedido? Se nenhuma das funções servir, "
                 "escolha a opção \"nenhuma\".")
MAX_DESC = 240


def _linhas(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _descricao(f: dict) -> str:
    d = re.sub(r"\s+", " ", str(f.get("description", ""))).strip()
    if len(d) > MAX_DESC:
        d = d[: MAX_DESC - 1].rstrip() + "…"
    return f"{f['name']}: {d}" if d else f["name"]


def _estado(turnos) -> str:
    rot = {"user": "Usuário", "assistant": "Assistente", "system": "Sistema"}
    msgs = [m for t in turnos for m in t]
    return "\n".join(f"{rot.get(m.get('role'), m.get('role'))}: {m.get('content', '')}" for m in msgs)


def bfcl_cases(cfg: dict | None = None, base: Path | None = None, conferir: bool = True) -> list[Case]:
    """Casos de avaliação do BFCL. `base` só para testes (padrão: `data/bfcl`). Casos com funções repetidas, com o
    gabarito fora do menu, sem nenhuma função ou com mais opções do que letras viram `descartado` com o motivo."""
    base = base or DATA_DIR / "bfcl"
    cases = []
    for fam, (arq, arq_gab) in BFCL_ARQUIVOS.items():
        for a in filter(None, (arq, arq_gab)):
            if conferir and sha256_file(base / a) != BFCL_SHA256[a]:
                raise ValueError(f"{base / a} não é o arquivo da revisão fixada do BFCL (sha256)")
        gab = {}
        if arq_gab:
            for r in _linhas(base / arq_gab):
                gab[r["id"]] = next(iter(r["ground_truth"][0]))  # nome da (única) função esperada
        for r in _linhas(base / arq):
            funcs = r["function"]
            nomes = [f["name"] for f in funcs]
            gold = gab.get(r["id"], NENHUMA[0]) if arq_gab else NENHUMA[0]
            opcoes = [(f["name"], _descricao(f)) for f in funcs] + [NENHUMA]
            motivo = None
            if not funcs:
                motivo = "sem funções no menu"
            elif len(set(nomes)) != len(nomes) or NENHUMA[0] in nomes:
                motivo = "nomes de função repetidos"
            elif len(opcoes) > len(LETTERS):
                motivo = f"mais de {len(LETTERS)} opções"
            elif gold not in {o[0] for o in opcoes}:
                motivo = "gabarito fora do menu"
            meta = {"fonte": BFCL["revision"], "n_funcoes": len(funcs)}
            if motivo:
                meta["descarte"] = motivo
            cases.append(Case(
                id=f"bfcl:{r['id']}", group_id=f"bfcl:{r['id']}", source="bfcl", family=fam,
                split=DISCARDED if motivo else "dev", held_out=True, qtype="choice", state=_estado(r["question"]),
                question=PERGUNTA_BFCL, options=opcoes, gold=gold if not motivo or gold in nomes else NENHUMA[0],
                meta=meta,
            ))
    return cases
