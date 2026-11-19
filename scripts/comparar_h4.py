"""H4: compara a calibração entre braços de perda (log, brier, acerto) com IC 95% por bootstrap.

Para cada fonte (felhen, telepatia) e grupo (vistas, reservadas), mede no `dev` a diferença
novo − base de ECE, Brier, NLL e acurácia, juntando as sementes disponíveis. O IC reamostra GRUPOS de casos
(`group_id`, como `metrics.paired_bootstrap`), levando junto todas as ordens e sementes de cada grupo, e
recalcula as métricas na amostra (ECE não é soma de itens, então não dá para reamostrar diferenças por item).
As métricas são as cruas, sem temperatura.

Pastas lidas: semente 0 = `runs/h4_<perda>_<fonte>_dev`; semente s > 0 = `runs/h4_<perda>_s<s>_<fonte>_dev`.
Só entram as sementes que existem para os DOIS braços comparados.

Uso (na raiz do repo, sem GPU):
    python scripts/comparar_h4.py                       # log contra brier e acerto
    python scripts/comparar_h4.py --base log --novos brier --sementes 0,1,2 --out analises/h4_ic.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-12
METRICAS = ("ece", "brier", "nll", "acc")


def run_dir(runs: Path, perda: str, semente: int, fonte: str) -> Path:
    nome = f"h4_{perda}" if semente == 0 else f"h4_{perda}_s{semente}"
    return runs / f"{nome}_{fonte}_dev"


def carregar(pasta: Path) -> list[dict]:
    return [json.loads(x) for x in (pasta / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def por_item(rows: list[dict]) -> dict[str, np.ndarray]:
    """Vetores por linha: confiança máxima, acerto, NLL e Brier (as mesmas contas de `metrics`)."""
    conf = np.array([max(r["probs"].values()) for r in rows])
    hit = np.array([r["pred"] == r["gold"] for r in rows], dtype=float)
    nll = np.array([-math.log(max(r["probs"].get(r["gold"], 0.0), EPS)) for r in rows])
    bri = np.array([sum((p - (k == r["gold"])) ** 2 for k, p in r["probs"].items()) for r in rows])
    return {"conf": conf, "hit": hit, "nll": nll, "brier": bri}


def ece_vet(conf: np.ndarray, hit: np.ndarray, bins: int = 15) -> float:
    """ECE de `metrics.ece` (probabilidade máxima, bins de largura igual), vetorizado."""
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)  # (lo, hi]; conf = 0 cai no 1º bin
    n = np.bincount(idx, minlength=bins)
    s_conf = np.bincount(idx, weights=conf, minlength=bins)
    s_hit = np.bincount(idx, weights=hit, minlength=bins)
    m = n > 0
    return float(np.sum(np.abs(s_hit[m] - s_conf[m])) / len(conf))


def metricas(v: dict[str, np.ndarray], sel: np.ndarray) -> dict[str, float]:
    return {"ece": ece_vet(v["conf"][sel], v["hit"][sel]), "brier": float(v["brier"][sel].mean()),
            "nll": float(v["nll"][sel].mean()), "acc": float(v["hit"][sel].mean())}


def comparar(rows_a: list[dict], rows_b: list[dict], n_boot: int, seed: int) -> dict:
    """rows_a/rows_b já juntam as sementes (cada linha com a chave `_semente`). Pareia por (semente, caso, ordem)."""
    ka = {(r["_semente"], r["case_id"], r["order"]): r for r in rows_a}
    kb = {(r["_semente"], r["case_id"], r["order"]): r for r in rows_b}
    if len(ka) != len(rows_a) or len(kb) != len(rows_b):
        raise ValueError("par (semente, caso, ordem) repetido")
    chaves = sorted(set(ka) & set(kb))
    if len(chaves) != len(ka) or len(chaves) != len(kb):
        raise ValueError(f"braços não correspondem: {len(ka)} x {len(kb)} linhas, {len(chaves)} em comum")
    A = [ka[k] for k in chaves]
    B = [kb[k] for k in chaves]
    va, vb = por_item(A), por_item(B)
    grupos = defaultdict(list)
    for i, r in enumerate(A):
        grupos[r["group_id"]].append(i)
    gl = [np.array(ix) for ix in grupos.values()]
    tudo = np.arange(len(A))
    ma, mb = metricas(va, tudo), metricas(vb, tudo)
    rng = np.random.default_rng(seed)
    boots = {m: [] for m in METRICAS}
    for _ in range(n_boot):
        sel = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
        xa, xb = metricas(va, sel), metricas(vb, sel)
        for m in METRICAS:
            boots[m].append(xb[m] - xa[m])
    out = {"n_linhas": len(A), "n_grupos": len(gl)}
    for m in METRICAS:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        out[m] = {"base": ma[m], "novo": mb[m], "delta": mb[m] - ma[m], "ic95": [float(lo), float(hi)]}
    return out


def leitura(m: str, d: dict) -> str:
    lo, hi = d["ic95"]
    if lo <= 0 <= hi:
        return "empate"
    melhor = d["delta"] > 0 if m == "acc" else d["delta"] < 0
    return "novo melhor" if melhor else "novo pior"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--base", default="log")
    p.add_argument("--novos", default="brier,acerto")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--sementes", default="0,1,2")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="grava a tabela em Markdown (e o JSON ao lado, .json)")
    a = p.parse_args(argv)
    runs = Path(a.runs)
    sementes = [int(s) for s in a.sementes.split(",")]
    linhas = ["| novo − base | fonte | grupo | sementes | n grupos | métrica | base | novo | Δ | IC 95% | leitura |",
              "|---|---|---|---|---:|---|---:|---:|---:|---|---|"]
    res = {}
    for novo in a.novos.split(","):
        for fonte in a.fontes.split(","):
            usadas = [s for s in sementes if (run_dir(runs, a.base, s, fonte) / "predicoes.jsonl").exists()
                      and (run_dir(runs, novo, s, fonte) / "predicoes.jsonl").exists()]
            if not usadas:
                print(f">> {novo} × {a.base} em {fonte}: nenhuma semente com os dois braços; pulei")
                continue
            ra, rb = [], []
            for s in usadas:
                for dst, perda in ((ra, a.base), (rb, novo)):
                    for r in carregar(run_dir(runs, perda, s, fonte)):
                        r["_semente"] = s
                        dst.append(r)
            for held, nome in ((False, "vistas"), (True, "reservadas")):
                c = comparar([r for r in ra if r["held_out"] == held], [r for r in rb if r["held_out"] == held],
                             a.n_boot, a.seed)
                res[f"{novo}-{a.base}|{fonte}|{nome}"] = {"sementes": usadas, **c}
                for m in METRICAS:
                    d = c[m]
                    linhas.append(f"| {novo} − {a.base} | {fonte} | {nome} | {','.join(map(str, usadas))} | "
                                  f"{c['n_grupos']} | {m} | {d['base']:.3f} | {d['novo']:.3f} | {d['delta']:+.3f} | "
                                  f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}] | {leitura(m, d)} |")
    tabela = "\n".join(linhas)
    print(tabela)
    print("\nLeitura: 'empate' quando o IC 95% inclui zero. ECE, Brier e NLL: menor é melhor; acc: maior é melhor.")
    if a.out:
        Path(a.out).write_text(tabela + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
