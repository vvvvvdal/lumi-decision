"""Inventário dos conjuntos de opções (para achar as perguntas em ESCALA, usadas pelo crédito parcial da H15).

Lista, por fonte e família, cada conjunto distinto de opções (ids e descrições, na ordem canônica) e quantos casos
o usam. A ordem das escalas é decidida à mão, em `configs/escalas.toml`: uma escala vale para todo caso cujo
conjunto de ids de opção seja exatamente o dela.

Uso (no cluster, sem GPU): python scripts/escalas.py --out analises/escalas_inventario.md
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from lumi_decision.data import load_cases, load_config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    cfg = load_config()
    linhas = []
    for fonte in a.fontes.split(","):
        cont: dict[str, Counter] = defaultdict(Counter)
        desc: dict[tuple, str] = {}
        grupos: dict[tuple, set] = defaultdict(set)
        casos = load_cases(fonte, cfg)
        for c in casos:
            if c.held_out:
                grupos[(c.family, c.split)].add(c.group_id)
        if grupos:  # H14: quantos grupos cada família reservada tem por split (o pool do treino curto)
            linhas.append(f"## {fonte}: grupos das reservadas por split\n")
            linhas.append("| família | split | grupos |")
            linhas.append("|---|---|---:|")
            linhas += [f"| {f} | {sp} | {len(g)} |" for (f, sp), g in sorted(grupos.items())]
            linhas.append("")
        for c in casos:
            ids = tuple(o[0] for o in c.options)
            cont[c.family][ids] += 1
            desc.setdefault((c.family, ids), " | ".join(f"{i}: {d[:40]}" for i, d in c.options))
        linhas.append(f"## {fonte}\n")
        for fam in sorted(cont):
            linhas.append(f"### {fam}\n")
            linhas.append("| casos | ids | opções |")
            linhas.append("|---:|---|---|")
            for ids, n in cont[fam].most_common(30):
                linhas.append(f"| {n} | `{','.join(ids)}` | {desc[(fam, ids)]} |")
            linhas.append("")
    txt = "\n".join(linhas)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
