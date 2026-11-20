"""H5: o braço (e) reage à falta de informação? (regra_beneficio, reservadas v2)

Para cada resposta certa da família `regra_beneficio`, mede a fração de linhas em que o modelo abstém
(probs["__abster__"] > 0,5) ou escolhe `pedir_documento`. A pergunta é se essa fração é MAIOR quando falta
documento (`pedir_documento`) do que quando todos estão presentes (`conceder`). O IC 95% reamostra grupos de
casos (`group_id`), juntando as sementes e as rotações das opções.

Pastas: `runs/res2dev_<nome>_r4` (semente 0) e `runs/res2dev_<nome>_s<s>_r4`.

Uso (na raiz do repo, via `srun ... lumi_exec python ...`):
    python scripts/analise_falta_info.py --nome h5_abster_l3 --out analises/h5_falta_info.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ABST = "__abster__"
CLASSES = ("conceder", "encaminhar", "negar", "pedir_documento")


def pasta(runs: Path, nome: str, s: int) -> Path:
    return runs / f"res2dev_{nome if s == 0 else f'{nome}_s{s}'}_r4"


def reage(r: dict) -> bool:
    norm = {k: v for k, v in r["probs"].items() if k != ABST}
    return r["probs"].get(ABST, 0.0) > 0.5 or max(norm, key=norm.get) == "pedir_documento"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--nome", required=True)
    p.add_argument("--sementes", default="0,1,2")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    runs = Path(a.runs)
    usadas, grupos = [], defaultdict(lambda: {c: [0, 0] for c in CLASSES})
    por_semente = {}
    for s in map(int, a.sementes.split(",")):
        f = pasta(runs, a.nome, s) / "predicoes.jsonl"
        if not f.exists():
            continue
        usadas.append(s)
        cont = {c: [0, 0] for c in CLASSES}
        for linha in f.read_text(encoding="utf-8").splitlines():
            if not linha.strip():
                continue
            r = json.loads(linha)
            if r.get("family") != "regra_beneficio" or r["gold"] not in CLASSES:
                continue
            x = int(reage(r))
            grupos[r["group_id"]][r["gold"]][0] += x
            grupos[r["group_id"]][r["gold"]][1] += 1
            cont[r["gold"]][0] += x
            cont[r["gold"]][1] += 1
        por_semente[s] = cont
    if not usadas:
        raise SystemExit(f"nenhuma pasta res2dev_{a.nome}*_r4 encontrada")
    g = list(grupos.values())
    arr = np.array([[[gr[c][0], gr[c][1]] for c in CLASSES] for gr in g], dtype=float)  # [grupos, classes, 2]

    def taxas(x):
        tot = x.sum(axis=0)
        return tot[:, 0] / np.maximum(tot[:, 1], 1)

    ponto = taxas(arr)
    rng = np.random.default_rng(a.seed)
    boots = np.array([taxas(arr[rng.integers(0, len(g), len(g))]) for _ in range(a.n_boot)])
    ip = CLASSES.index("pedir_documento")
    linhas = [f"**{a.nome}**, sementes {','.join(map(str, usadas))}, {len(g)} grupos de casos",
              "", "| resposta certa | abstém ou pede o documento | por semente |", "|---|---:|---|"]
    for i, c in enumerate(CLASSES):
        ps = " · ".join(f"s{s}: {por_semente[s][c][0]}/{por_semente[s][c][1]}" for s in usadas)
        linhas.append(f"| {c} | {ponto[i]:.3f} | {ps} |")
    linhas += ["", "| diferença | Δ | IC 95% | leitura |", "|---|---:|---|---|"]
    for i, c in enumerate(CLASSES):
        if i == ip:
            continue
        d = boots[:, ip] - boots[:, i]
        lo, hi = np.percentile(d, [2.5, 97.5])
        leitura = "reage mais quando falta documento" if lo > 0 else ("empate" if hi >= 0 else "reage menos")
        linhas.append(f"| pedir_documento − {c} | {ponto[ip] - ponto[i]:+.3f} | [{lo:+.3f}; {hi:+.3f}] | {leitura} |")
    texto = "\n".join(linhas)
    print(texto)
    if a.out:
        Path(a.out).write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
