"""H5: o braço (e) reage à FALTA DE DOCUMENTO ou a outra coisa? (regra_beneficio, reservadas v2)

O gerador da `regra_beneficio` (`res2.py`) guarda em `meta["dados"]` os documentos exigidos, os anexados, o
valor e o teto. Daí saem dois fatores por caso:
- `falta_doc`: algum documento exigido não foi anexado;
- `acima`: o valor pedido passa do teto.

O teste limpo é DENTRO de "negar": a resposta certa é a mesma, e só muda se falta documento. Se o modelo
abstém mais em "negar" com documento faltando do que em "negar" sem, ele reage à falta de informação.
Também separa o efeito de `acima` (o "encaminhar" tem valor acima e nenhum documento faltando).

Reação = abster (probs["__abster__"] > 0,5) ou escolher `pedir_documento`. IC 95% reamostrando grupos de
casos (`group_id`), juntando sementes e rotações.

Uso (na raiz do repo, via `srun ... lumi_exec python ...`):
    python scripts/analise_falta_info_fatores.py --nomes h4_log,h5_abster_l3 --out analises/h5_fatores.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lumi_decision.data import load_cases, load_config

ABST = "__abster__"


def fatores() -> dict[str, tuple[bool, bool]]:
    out = {}
    for c in load_cases("res2", load_config()):
        if c.family != "regra_beneficio":
            continue
        d = c.meta["dados"]
        out[c.id] = (not set(d["exigidos"]) <= set(d["anexos"]), d["valor"] > d["teto"])
    return out


def reage(r: dict) -> int:
    norm = {k: v for k, v in r["probs"].items() if k != ABST}
    return int(r["probs"].get(ABST, 0.0) > 0.5 or max(norm, key=norm.get) == "pedir_documento")


def pastas(runs: Path, nome: str, sementes):
    for s in sementes:
        p = runs / f"res2dev_{nome if s == 0 else f'{nome}_s{s}'}_r4" / "predicoes.jsonl"
        if p.exists():
            yield s, p


CELULAS = [("negar", True, None), ("negar", False, None), ("pedir_documento", True, None),
           ("conceder", False, False), ("encaminhar", False, True)]
NOMES_CEL = {("negar", True, None): "negar COM documento faltando", ("negar", False, None): "negar SEM documento faltando",
             ("pedir_documento", True, None): "pedir_documento (sempre falta)",
             ("conceder", False, False): "conceder (nada falta, valor ok)",
             ("encaminhar", False, True): "encaminhar (nada falta, valor acima)"}
CONTRASTES = [(0, 1, "negar: com − sem documento faltando (o teste limpo)"),
              (4, 3, "valor acima − ok, sem documento faltando (encaminhar − conceder)")]


def analisar(rows, fat, n_boot, seed):
    por_grupo = defaultdict(lambda: np.zeros((len(CELULAS), 2)))
    for r in rows:
        if r.get("family") != "regra_beneficio" or r["case_id"] not in fat:
            continue
        falta, acima = fat[r["case_id"]]
        for i, (gold, f, ac) in enumerate(CELULAS):
            if r["gold"] == gold and falta == f and (ac is None or acima == ac):
                por_grupo[r["group_id"]][i] += (reage(r), 1)
    arr = np.stack(list(por_grupo.values()))  # [grupos, células, 2]

    def taxa(x):
        t = x.sum(axis=0)
        return t[:, 0] / np.maximum(t[:, 1], 1)

    ponto = taxa(arr)
    rng = np.random.default_rng(seed)
    boots = np.array([taxa(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n_boot)])
    n = arr.sum(axis=0)
    cont = []
    for a, b, nome in CONTRASTES:
        lo, hi = np.percentile(boots[:, a] - boots[:, b], [2.5, 97.5])
        cont.append((nome, ponto[a] - ponto[b], lo, hi))
    return ponto, n, cont


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--nomes", default="h4_log,h5_abster_l3")
    p.add_argument("--sementes", default="0,1,2")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    fat = fatores()
    partes = []
    for nome in a.nomes.split(","):
        usadas, rows = [], []
        for s, f in pastas(Path(a.runs), nome, map(int, a.sementes.split(","))):
            usadas.append(s)
            rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not usadas:
            partes.append(f"**{nome}**: nenhuma pasta res2dev encontrada")
            continue
        ponto, n, cont = analisar(rows, fat, a.n_boot, a.seed)
        t = [f"**{nome}** (sementes {','.join(map(str, usadas))})", "",
             "| caso | reage (abstém ou pede documento) | linhas |", "|---|---:|---:|"]
        t += [f"| {NOMES_CEL[c]} | {ponto[i]:.3f} | {int(n[i, 1])} |" for i, c in enumerate(CELULAS)]
        t += ["", "| contraste | Δ | IC 95% | leitura |", "|---|---:|---|---|"]
        for nome_c, d, lo, hi in cont:
            t.append(f"| {nome_c} | {d:+.3f} | [{lo:+.3f}; {hi:+.3f}] | "
                     f"{'maior' if lo > 0 else ('menor' if hi < 0 else 'empate')} |")
        partes.append("\n".join(t))
    texto = "\n\n".join(partes)
    print(texto)
    if a.out:
        Path(a.out).write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
