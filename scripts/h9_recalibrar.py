"""H9b: recalibrar (H7) depois do treino curto (H9) devolve a honestidade nas famílias reservadas?

A H9 aumenta a acurácia nas reservadas, mas o Felhen sai superconfiante (ECE +0,04 nas 3 sementes). Aqui
aplicamos a receita da H7 (k grupos rotulados da família para ajustar T e o limiar tau) aos dois braços:
- base: `h4_log` (é a própria H7);
- novo: `h9_k100` (LoRA do `h4_log` continuada nos k grupos de treino).

Para ser justo, os grupos usados no treino curto (`runs/<novo>/adapt_grupos.json`) saem do `dev` dos DOIS
braços: o modelo já viu a resposta deles, então não servem para calibrar nem para medir. Os k grupos de
calibração são sorteados entre os grupos restantes do `dev` (as reservadas não têm `calib`). A medida usa o
resto. O mesmo gerador, com a mesma semente, nos dois braços faz os dois sortearem os mesmos grupos
(comparação pareada). Métodos e medidas são os de `h7_poucos_rotulos.uma_fonte`.

Pastas: semente 0 = `runs/<prefixo>_<fonte>_{calib,dev}`; s > 0 = `runs/<prefixo>_s<s>_<fonte>_{calib,dev}`.

Uso (na raiz do repo, sem GPU):
    python scripts/h9_recalibrar.py --base h4_log --novo h9_k100 --sementes 0,1,2 --out analises/h9b_recal.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent


def _mod(nome):
    spec = importlib.util.spec_from_file_location(nome, AQUI / f"{nome}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


h7 = _mod("h7_poucos_rotulos")


def nome(prefixo: str, semente: int) -> str:
    return prefixo if semente == 0 else f"{prefixo}_s{semente}"


def grupos_de_treino(runs: Path, novo: str) -> set[str]:
    reg = json.loads((runs / novo / "adapt_grupos.json").read_text(encoding="utf-8"))
    return {g for r in reg for g in r["grupos"]}


def vistas(calib: list[dict], dev: list[dict]) -> dict:
    """Famílias vistas: ECE/NLL crus e com a T global (ajustada nas vistas do calib, como o `lumi-eval calibrar`)."""
    m = h7.metrics
    vc, vd = [r for r in calib if not r["held_out"]], [r for r in dev if not r["held_out"]]
    T = m.fit_temperature(vc)
    vt = m.with_temperature(vd, T)
    return {"acc": acc(vd), "T": T, "ece_cru": m.ece(vd), "ece_T": m.ece(vt), "nll_cru": m.nll(vd), "nll_T": m.nll(vt)}


def acc(rows: list[dict]) -> float:
    return sum(r["pred"] == r["gold"] for r in rows) / len(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--base", default="h4_log")
    p.add_argument("--novo", default="h9_k100")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--sementes", default="0,1,2")
    p.add_argument("--ks", default="10,25,50")
    p.add_argument("--repeticoes", type=int, default=100)
    p.add_argument("--risco", type=float, default=0.10)
    p.add_argument("--min-aceitos", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--excluir-tambem", default="", help="prefixos de runs cujos grupos de treino também saem")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    runs, ks = Path(a.runs), [int(k) for k in a.ks.split(",")]
    linhas = ["| fonte | braço | acc res. | método | k | ECE | NLL | cobertura | risco aceitos |",
              "|---|---|---:|---|---:|---|---|---|---|"]
    bruto = {}
    t_vis = ["| fonte | braço | recorte | acc | ECE cru | ECE com T global | NLL cru | NLL com T global | T |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for fonte in a.fontes.split(","):
        por_braco = {"base": [], "novo": []}
        accs = {"base": [], "novo": []}
        vist = {"base": [], "novo": []}
        for s in (int(x) for x in a.sementes.split(",")):
            nb, nn = nome(a.base, s), nome(a.novo, s)
            if not (runs / nn / "adapt_grupos.json").exists():
                print(f">> falta {runs / nn / 'adapt_grupos.json'}; pulei semente {s}")
                continue
            fora = grupos_de_treino(runs, nn)
            for outro in filter(None, a.excluir_tambem.split(",")):
                fora |= grupos_de_treino(runs, nome(outro, s))
            dados = {}
            for braco, nm in (("base", nb), ("novo", nn)):
                pc, pd = runs / f"{nm}_{fonte}_calib", runs / f"{nm}_{fonte}_dev"
                if not ((pc / "predicoes.jsonl").exists() and (pd / "predicoes.jsonl").exists()):
                    print(f">> {fonte} semente {s}: faltam {pc} ou {pd}; pulei")
                    break
                dev = [r for r in h7.carregar(pd) if r["group_id"] not in fora]
                dados[braco] = (h7.carregar(pc), dev)
            if len(dados) < 2:
                continue
            for braco, (calib, dev) in dados.items():
                # mesma semente nos dois braços -> mesmos grupos sorteados (os dev filtrados têm os mesmos group_id)
                rng = np.random.default_rng([a.seed, s])
                r = h7.uma_fonte(calib, dev, ks, a.repeticoes, a.risco, "calib", a.min_aceitos, rng)
                por_braco[braco].append(r)
                accs[braco].append(acc([x for x in dev if x["held_out"]]))
                vist[braco].append(vistas(calib, dev))
                bruto.setdefault(fonte, {}).setdefault(braco, {})[s] = r
        for braco, rotulo in (("base", a.base), ("novo", a.novo)):
            sem = por_braco[braco]
            if not sem:
                continue
            ac = f"{np.mean(accs[braco]):.3f}"

            def linha(metodo, k, lista):
                cel = [h7.resumo([m[c] for m in lista]) for c in ("ece", "nll", "cobertura", "risco_aceitos")]
                linhas.append(f"| {fonte} | {rotulo} | {ac} | {metodo} | {k} | " + " | ".join(cel) + " |")

            linha("sem recalibrar (T = 1)", "—", [x["sem_T"] for x in sem])
            linha("global (vistas)", "—", [x["global"] for x in sem])
            for k in ks:
                linha("T_fam", k, [m for x in sem for m in x[f"T_fam|{k}"]])
            for k in ks:
                linha("T_fam+tau_fam", k, [m for x in sem for m in x[f"T_fam+tau_fam|{k}"]])
            linha("oráculo (família toda)", "—", [x["oraculo"] for x in sem])
            v = vist[braco]
            t_vis.append(f"| {fonte} | {rotulo} | vistas | {np.mean([x['acc'] for x in v]):.3f} | "
                         f"{np.mean([x['ece_cru'] for x in v]):.3f} | {np.mean([x['ece_T'] for x in v]):.3f} | "
                         f"{np.mean([x['nll_cru'] for x in v]):.3f} | {np.mean([x['nll_T'] for x in v]):.3f} | "
                         f"{np.mean([x['T'] for x in v]):.2f} |")
    tabela = "\n".join(linhas) + "\n\nFamílias VISTAS do dev (T ajustada nas vistas do calib, média entre sementes):\n" + "\n".join(t_vis)
    nota = (f"\nReservadas do dev SEM os grupos do treino curto (nos dois braços), risco-alvo {a.risco:.0%}. "
            "Média [2,5%; 97,5%] entre sementes e sorteios dos k grupos (demais linhas: só entre sementes). "
            "Os k grupos de calibração saem do mesmo dev e ficam fora da medida.")
    print(tabela + nota)
    if a.out:
        Path(a.out).write_text(tabela + "\n" + nota + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(bruto, indent=1, ensure_ascii=False, default=float),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
