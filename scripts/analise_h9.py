"""H9: treinar um pouco com k grupos das famílias reservadas ensina o Lumi (acurácia) sem esquecer as vistas?

Compara o braço novo (LoRA do `h4_log` continuada nos k grupos, `--adapt-k`) com a base (`h4_log`) nos MESMOS
casos do `dev`: os grupos usados no treino (`runs/<novo>/adapt_grupos.json`) saem dos dois lados. Relata, com IC
95% por bootstrap de grupos (`comparar_h4.comparar`), acurácia, ECE, Brier e NLL:
- nas reservadas de cada fonte e em cada família reservada;
- nas vistas (esquecimento).
E a cobertura no risco de 10% do oráculo por fonte (T e limiar por família, funções da H7).

Uso (na raiz do repo, sem GPU):
    python scripts/analise_h9.py --base h4_log --novos h9_k25,h9_k100 --out analises/h9_adapt.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def _mod(nome):
    spec = importlib.util.spec_from_file_location(nome, AQUI / f"{nome}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c4, h7 = _mod("comparar_h4"), _mod("h7_poucos_rotulos")


def grupos_de_treino(runs: Path, novo: str) -> set[str]:
    reg = json.loads((runs / novo / "adapt_grupos.json").read_text(encoding="utf-8"))
    return {g for r in reg for g in r["grupos"]}


def oraculo(rows: list[dict], risco: float) -> dict:
    lin = []
    for f in sorted({r["family"] for r in rows}):
        rf = [r for r in rows if r["family"] == f]
        T, tau = h7.ajustar(rf, risco, 5)
        lin += h7.aplicar(rf, T, tau)
    return h7.medir(lin)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--base", default="h4_log")
    p.add_argument("--novos", default="h9_k25,h9_k100")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--risco", type=float, default=0.10)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--excluir-tambem", default="",
                   help="runs cujos grupos de treino também saem da avaliação (comparar braços com grupos diferentes)")
    p.add_argument("--out", default="")
    p.add_argument("--split", default="dev", help="pasta avaliada (H13b: devvar = prompt nunca visto); o calib continua o de sempre")
    a = p.parse_args(argv)
    runs = Path(a.runs)
    t1 = ["| novo | fonte | recorte | n grupos | métrica | base | novo | Δ | IC 95% | leitura |",
          "|---|---|---|---:|---|---:|---:|---:|---|---|"]
    t2 = ["| braço | fonte | acc reservadas | cobertura oráculo (risco 10%) | risco oráculo |", "|---|---|---:|---:|---:|"]
    res = {}
    for novo in a.novos.split(","):
        if not (runs / novo / "adapt_grupos.json").exists():
            print(f">> falta {runs / novo / 'adapt_grupos.json'}; pulei {novo}")
            continue
        fora = grupos_de_treino(runs, novo)
        for outro in filter(None, a.excluir_tambem.split(",")):
            fora |= grupos_de_treino(runs, outro)
        for fonte in a.fontes.split(","):
            pb, pn = runs / f"{a.base}_{fonte}_{a.split}", runs / f"{novo}_{fonte}_{a.split}"
            if not ((pb / "predicoes.jsonl").exists() and (pn / "predicoes.jsonl").exists()):
                print(f">> falta {pb} ou {pn}; pulei")
                continue
            rb = [dict(r, _semente=0) for r in c4.carregar(pb) if r["group_id"] not in fora]
            rn = [dict(r, _semente=0) for r in c4.carregar(pn) if r["group_id"] not in fora]
            recortes = [("vistas", lambda r: not r["held_out"]), ("reservadas", lambda r: r["held_out"])]
            recortes += [(f"res: {f}", (lambda f: lambda r: r["held_out"] and r["family"] == f)(f))
                         for f in sorted({r["family"] for r in rb if r["held_out"]})]
            for nome, filtro in recortes:
                xb, xn = [r for r in rb if filtro(r)], [r for r in rn if filtro(r)]
                if not xb:
                    continue
                c = c4.comparar(xb, xn, a.n_boot, 0)
                res[f"{novo}|{fonte}|{nome}"] = c
                for m in c4.METRICAS:
                    d = c[m]
                    t1.append(f"| {novo} | {fonte} | {nome} | {c['n_grupos']} | {m} | {d['base']:.3f} | {d['novo']:.3f} | "
                              f"{d['delta']:+.3f} | [{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}] | {c4.leitura(m, d)} |")
            for braco, rows in ((a.base, rb), (novo, rn)):
                res_rows = [r for r in rows if r["held_out"]]
                o = oraculo(res_rows, a.risco)
                acc = sum(r["pred"] == r["gold"] for r in res_rows) / len(res_rows)
                t2.append(f"| {braco} (sem os grupos de {novo}) | {fonte} | {acc:.3f} | {o['cobertura']:.3f} | "
                          f"{o['risco_aceitos']:.3f} |")
    txt = "\n".join(t1) + "\n\n" + "\n".join(t2) + (
        "\n\nDev, sem os grupos usados no treino curto (nos dois braços). Δ = novo − base. "
        "Oráculo: T e limiar ajustados em cada família inteira (otimista).")
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
