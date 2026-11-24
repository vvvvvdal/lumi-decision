"""Resume os `calibracao.json` (do `lumi-eval calibrar`) de braços da H4: temperatura, ECE e NLL antes/depois.

Para cada braço, fonte e grupo (vistas, reservadas), mostra a média entre as sementes. Serve para conferir se a
diferença crua entre dois braços some depois da temperatura (critério da H12). Sem IC: é uma conferência.

Pastas lidas: semente 0 = `runs/h4_<braço>_<fonte>_dev`; semente s > 0 = `runs/h4_<braço>_s<s>_<fonte>_dev`.

Uso (na raiz do repo, sem GPU):
    python scripts/resumo_calibracao.py --bracos log,clef --sementes 0,1,2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pasta(runs: Path, braco: str, semente: int, fonte: str) -> Path:
    nome = f"h4_{braco}" if semente == 0 else f"h4_{braco}_s{semente}"
    return runs / f"{nome}_{fonte}_dev"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--bracos", default="log,clef")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--sementes", default="0,1,2")
    a = p.parse_args(argv)
    runs = Path(a.runs)
    sementes = [int(s) for s in a.sementes.split(",")]
    print("| braço | fonte | grupo | sementes | T média | ECE antes | ECE depois | NLL antes | NLL depois |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|")
    for fonte in a.fontes.split(","):
        for grupo in ("vistas", "reservadas"):
            for braco in a.bracos.split(","):
                achou, ts, vals = [], [], {k: [] for k in ("ece_antes", "ece_depois", "nll_antes", "nll_depois")}
                for s in sementes:
                    f = pasta(runs, braco, s, fonte) / "calibracao.json"
                    if not f.exists():
                        continue
                    d = json.loads(f.read_text(encoding="utf-8"))
                    g = d.get(f"{fonte}|{grupo}")
                    if not g:
                        continue
                    achou.append(str(s))
                    ts.append(d["temperatura"])
                    for k in vals:
                        vals[k].append(g[k])
                if not achou:
                    print(f"| {braco} | {fonte} | {grupo} | nenhuma | | | | | |")
                    continue
                m = {k: sum(v) / len(v) for k, v in vals.items()}
                print(f"| {braco} | {fonte} | {grupo} | {','.join(achou)} | {sum(ts) / len(ts):.3f} | "
                      f"{m['ece_antes']:.3f} | {m['ece_depois']:.3f} | {m['nll_antes']:.3f} | {m['nll_depois']:.3f} |")


if __name__ == "__main__":
    main()
