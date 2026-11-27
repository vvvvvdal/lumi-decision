"""H17: compara os modelos de `configs/modelos_h17.toml` nas decisões de agente (pela leitura das letras).

Lê `runs/h17_<modelo>_<fonte>/predicoes.jsonl` (slurm/h17_harness.sbatch). Por modelo, fonte e família:
acurácia, ECE, AUROC da confiança, cobertura com 10% de erro (limiar ajustado no próprio conjunto: otimista, igual
para todos), viés de ordem (fração de casos em que a resposta muda ao inverter as opções) e massa nas letras.
Diferença de acurácia contra o `lumi` com IC 95% por bootstrap pareado nos mesmos casos.

Uso (sem GPU): python scripts/analise_h17.py --out analises/h17_harness.md
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

from lumi_decision import metrics

RAIZ = Path(__file__).resolve().parents[1]
FONTES = ("bfcl", "traces", "vestbench_enem")


def carregar(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def medir(rows: list[dict]) -> dict:
    conf = np.array([max(r["probs"].values()) for r in rows])
    hit = np.array([r["pred"] == r["gold"] for r in rows])
    por_caso = defaultdict(dict)
    for r in rows:
        por_caso[r["case_id"]][r["order"]] = r["pred"]
    pares = [v for v in por_caso.values() if len(v) >= 2]
    vies = float(np.mean([len(set(v.values())) > 1 for v in pares])) if pares else float("nan")
    tau = metrics.threshold_for_risk(rows, 0.10, min_accept=10)
    cob = float(np.mean(conf >= tau)) if tau is not None else 0.0
    massa = [r["cand_mass"] for r in rows if r.get("cand_mass") is not None]
    return {"n": len(rows), "acc": float(hit.mean()), "ece": metrics.ece(rows), "auroc": metrics.auroc(conf, hit),
            "cobertura_10": cob, "vies_ordem": vies, "massa": float(np.mean(massa)) if massa else float("nan")}


def acerto_por_caso(rows: list[dict]) -> dict[str, float]:
    d = defaultdict(list)
    for r in rows:
        d[r["case_id"]].append(r["pred"] == r["gold"])
    return {k: float(np.mean(v)) for k, v in d.items()}


def dif_pareada(a: dict, b: dict, n_boot: int, rng) -> tuple[float, float, float]:
    """Acurácia de b − a nos casos comuns, com IC 95% por bootstrap dos casos."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return float("nan"), float("nan"), float("nan")
    d = np.array([b[i] - a[i] for i in ids])
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.1%}" if pct else f"{x:.3f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--prefixo", default="h17")
    p.add_argument("--base", default="lumi")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    modelos = list(tomllib.loads((RAIZ / "configs" / "modelos_h17.toml").read_text(encoding="utf-8")))
    rng = np.random.default_rng(0)
    t1 = ["| fonte | família | modelo | n | acurácia | ECE | AUROC | cobertura (10% de erro) | viés de ordem | massa nas letras |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    t2 = [f"| fonte | família | modelo | acurácia − {a.base} | IC 95% |", "|---|---|---|---:|---|"]
    for fonte in FONTES:
        dados = {}
        for m in modelos:
            f = Path(a.runs) / f"{a.prefixo}_{m}_{fonte}" / "predicoes.jsonl"
            if f.exists():
                dados[m] = carregar(f)
        if not dados:
            continue
        fams = sorted({r["family"] for rows in dados.values() for r in rows})
        for fam in ["(todas)"] + (fams if len(fams) > 1 else []):
            sel = {m: [r for r in rows if fam == "(todas)" or r["family"] == fam] for m, rows in dados.items()}
            for m, rows in sel.items():
                if not rows:
                    continue
                x = medir(rows)
                t1.append(f"| {fonte} | {fam} | {m} | {x['n']} | {fmt(x['acc'], True)} | {fmt(x['ece'])} | "
                          f"{fmt(x['auroc'])} | {fmt(x['cobertura_10'], True)} | {fmt(x['vies_ordem'], True)} | "
                          f"{fmt(x['massa'])} |")
            if a.base in sel and sel[a.base]:
                base = acerto_por_caso(sel[a.base])
                for m, rows in sel.items():
                    if m == a.base or not rows:
                        continue
                    d, lo, hi = dif_pareada(base, acerto_por_caso(rows), a.n_boot, rng)
                    t2.append(f"| {fonte} | {fam} | {m} | {d * 100:+.1f} | [{lo * 100:+.1f}; {hi * 100:+.1f}] |")
    txt = ("\n".join(t1) + "\n\nA cobertura usa um limiar ajustado no próprio conjunto (otimista, igual para todos).\n\n"
           + "\n".join(t2) + "\n")
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
