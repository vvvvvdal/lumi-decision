"""H7: quantos exemplos rotulados de uma família NOVA bastam para recalibrar o Lumi?

Hoje (E3/H4) a temperatura T e o limiar de abstenção tau são ajustados nas famílias VISTAS do `calib` e usados,
fixos, nas reservadas. Lá o risco entre os aceitos sobe para 29–35% com alvo de 10%. Aqui, para cada família
reservada, sorteamos k GRUPOS de casos rotulados dessa família (k = 10, 25, 50), reajustamos a calibração só
para ela e medimos no `dev` dessa família. Nenhum treino: só as predições que já existem.

Métodos (todos medidos nas reservadas do `dev`, por fonte, juntando as famílias):
- `global`: T e tau das vistas do `calib` (o que o `lumi-eval calibrar` faz hoje);
- `T_fam`: T ajustada nos k grupos da família; tau global (o limiar vale na escala já calibrada);
- `T_fam+tau_fam`: T e tau ajustados nos k grupos da família;
- `oraculo`: T e tau ajustados no próprio `dev` da família inteira (limite superior, otimista).

Os k grupos rotulados vêm das linhas reservadas do `calib` da mesma família (`--rotulos calib`, padrão), sem
sobreposição com o `dev`. Se a família não estiver no `calib`, ou com `--rotulos dev`, os k grupos saem do
próprio `dev` e a medida usa só os grupos restantes. Cada configuração é repetida `--repeticoes` vezes
(sorteios diferentes dos k grupos); a tabela mostra a média e a faixa de 2,5% a 97,5% entre os sorteios.

Pastas: semente 0 = `runs/h4_<run>_<fonte>_{calib,dev}`; s > 0 = `runs/h4_<run>_s<s>_<fonte>_{calib,dev}`.

Uso (na raiz do repo, sem GPU):
    python scripts/h7_poucos_rotulos.py --run log --sementes 0,1,2 --out analises/h7_poucos_rotulos.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lumi_decision import metrics

METODOS = ("global", "T_fam", "T_fam+tau_fam", "oraculo")


def pasta(runs: Path, run: str, semente: int, fonte: str, split: str) -> Path:
    nome = f"h4_{run}" if semente == 0 else f"h4_{run}_s{semente}"
    return runs / f"{nome}_{fonte}_{split}"


def carregar(p: Path) -> list[dict]:
    return [json.loads(x) for x in (p / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def por_grupo(rows: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for r in rows:
        g[r["group_id"]].append(r)
    return dict(g)


def ajustar(rows: list[dict], risco: float, min_aceitos: int) -> tuple[float, float | None]:
    T = metrics.fit_temperature(rows)
    return T, metrics.threshold_for_risk(metrics.with_temperature(rows, T), risco, min_accept=min_aceitos)


def medir(rows: list[dict]) -> dict:
    """rows já com a temperatura aplicada e a chave `_tau` (limiar da família de cada linha)."""
    conf = np.array([max(r["probs"].values()) for r in rows])
    err = np.array([r["pred"] != r["gold"] for r in rows], dtype=float)
    tau = np.array([np.inf if r["_tau"] is None else r["_tau"] for r in rows])
    acc = conf >= tau
    return {"ece": metrics.ece(rows), "nll": metrics.nll(rows), "cobertura": float(acc.mean()),
            "risco_aceitos": float(err[acc].mean()) if acc.any() else float("nan")}


def aplicar(rows: list[dict], T: float, tau: float | None) -> list[dict]:
    return [{**r, "_tau": tau} for r in metrics.with_temperature(rows, T)]


def uma_fonte(calib: list[dict], dev: list[dict], ks: list[int], reps: int, risco: float, rotulos: str,
              min_aceitos: int, rng: np.random.Generator) -> dict:
    """Resultados de uma semente × fonte: {metodo: {k: [medidas por repetição]}} (global e oráculo sem k)."""
    vistas = [r for r in calib if not r["held_out"]]
    T_g, tau_g = ajustar(vistas, risco, 20)
    res_dev = [r for r in dev if r["held_out"]]
    fams = sorted({r["family"] for r in res_dev})
    dev_f = {f: [r for r in res_dev if r["family"] == f] for f in fams}
    cal_f = {f: [r for r in calib if r["held_out"] and r["family"] == f] for f in fams}
    out = {"global": medir(aplicar(res_dev, T_g, tau_g)), "T_global": T_g, "tau_global": tau_g,
           "familias": {f: {"n_dev": len(dev_f[f]), "n_calib": len(cal_f[f])} for f in fams}}
    ora = []
    for f in fams:
        T, tau = ajustar(dev_f[f], risco, min_aceitos)
        out["familias"][f]["T_oraculo"] = T
        ora += aplicar(dev_f[f], T, tau)
    out["oraculo"] = medir(ora)
    for k in ks:
        reps_T, reps_Ttau = [], []
        for _ in range(reps):
            linhas_T, linhas_Ttau = [], []
            for f in fams:
                do_calib = rotulos == "calib" and len(por_grupo(cal_f[f])) >= k
                pool = por_grupo(cal_f[f] if do_calib else dev_f[f])
                ids = sorted(pool)
                if len(ids) < k + (0 if do_calib else 5):
                    raise SystemExit(f"família {f}: só {len(ids)} grupos para k = {k}")
                esc = set(rng.choice(ids, size=k, replace=False).tolist())
                rot = [r for g in esc for r in pool[g]]
                alvo = dev_f[f] if do_calib else [r for r in dev_f[f] if r["group_id"] not in esc]
                T, tau = ajustar(rot, risco, min_aceitos)
                linhas_T += aplicar(alvo, T, tau_g)
                linhas_Ttau += aplicar(alvo, T, tau)
            reps_T.append(medir(linhas_T))
            reps_Ttau.append(medir(linhas_Ttau))
        out[f"T_fam|{k}"] = reps_T
        out[f"T_fam+tau_fam|{k}"] = reps_Ttau
    return out


def resumo(vals: list[float]) -> str:
    v = np.array([x for x in vals if not np.isnan(x)])
    if not len(v):
        return "—"
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"{v.mean():.3f} [{lo:.3f}; {hi:.3f}]"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--run", default="log")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--sementes", default="0,1,2")
    p.add_argument("--ks", default="10,25,50")
    p.add_argument("--repeticoes", type=int, default=100)
    p.add_argument("--risco", type=float, default=0.10)
    p.add_argument("--min-aceitos", type=int, default=5, help="mínimo de aceitos para o limiar por família")
    p.add_argument("--rotulos", choices=("calib", "dev"), default="calib")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    runs, ks = Path(a.runs), [int(k) for k in a.ks.split(",")]
    rng = np.random.default_rng(a.seed)
    linhas = ["| fonte | método | k | ECE | NLL | cobertura | risco aceitos |", "|---|---|---:|---|---|---|---|"]
    bruto = {}
    for fonte in a.fontes.split(","):
        por_sem = {}
        for s in (int(x) for x in a.sementes.split(",")):
            pc, pd = pasta(runs, a.run, s, fonte, "calib"), pasta(runs, a.run, s, fonte, "dev")
            if not ((pc / "predicoes.jsonl").exists() and (pd / "predicoes.jsonl").exists()):
                print(f">> {fonte} semente {s}: faltam {pc} ou {pd}; pulei")
                continue
            por_sem[s] = uma_fonte(carregar(pc), carregar(pd), ks, a.repeticoes, a.risco, a.rotulos,
                                   a.min_aceitos, rng)
        if not por_sem:
            continue
        bruto[fonte] = por_sem
        sem = list(por_sem.values())

        def linha(metodo, k, lista):
            cel = [resumo([m[c] for m in lista]) for c in ("ece", "nll", "cobertura", "risco_aceitos")]
            linhas.append(f"| {fonte} | {metodo} | {k} | " + " | ".join(cel) + " |")

        linha("global (hoje)", "—", [x["global"] for x in sem])
        for k in ks:
            linha("T_fam", k, [m for x in sem for m in x[f"T_fam|{k}"]])
        for k in ks:
            linha("T_fam+tau_fam", k, [m for x in sem for m in x[f"T_fam+tau_fam|{k}"]])
        linha("oráculo (família toda)", "—", [x["oraculo"] for x in sem])
    tabela = "\n".join(linhas)
    nota = (f"\nReservadas do dev, risco-alvo {a.risco:.0%}. Média [2,5%; 97,5%] entre sementes e sorteios dos k grupos "
            "(global e oráculo: só entre sementes). Cobertura = fração respondida; risco = erro entre as respondidas.")
    print(tabela + nota)
    if a.out:
        Path(a.out).write_text(tabela + "\n" + nota + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(bruto, indent=1, ensure_ascii=False, default=float),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
