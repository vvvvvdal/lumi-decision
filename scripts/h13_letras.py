"""H13: o Lumi continua respondendo com uma letra no prompt novo? Massa das letras (cand_mass) por run.

Uso: python scripts/h13_letras.py [--runs runs] [--out analises/h13_letras.md]
"""

import argparse
import json
from pathlib import Path

import numpy as np

RUNS = [f"{b}{s}" for b in ("h4_log", "h4_aug") for s in ("", "_s1", "_s2")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    linhas = ["| run | fonte | prompt | n | massa das letras (média) | casos com massa < 0,5 | gera a mesma letra |",
              "|---|---|---|---:|---:|---:|---:|"]
    for nome in RUNS:
        for fonte in a.fontes.split(","):
            for split, rot in (("dev", "de sempre"), ("devvar", "novo (variante 4)")):
                f = Path(a.runs) / f"{nome}_{fonte}_{split}" / "predicoes.jsonl"
                if not f.exists():
                    continue
                rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
                m = np.array([r["cand_mass"] for r in rows if r.get("cand_mass") is not None])
                g = [r["gen_first_is_readout_argmax"] for r in rows if r.get("gen_first_is_readout_argmax") is not None]
                ger = f"{np.mean(g):.3f}" if g else "—"
                linhas.append(f"| {nome} | {fonte} | {rot} | {len(m)} | {m.mean():.3f} | {np.mean(m < 0.5):.1%} | {ger} |")
    txt = "\n".join(linhas)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
