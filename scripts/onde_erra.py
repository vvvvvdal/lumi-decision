"""Onde o Lumi erra nas famílias reservadas, depois da receita: acerto por classe, confusão e, nas escalas,
erro vizinho contra erro grave (pular uma ou mais posições da escala).

Lê `runs/<nome>[_s<s>]_<fonte>_dev/predicoes.jsonl`, só as reservadas, sem os grupos do treino curto
(`runs/<nome>/adapt_grupos.json`), as 2 ordens e as sementes juntas. As escalas vêm de `configs/escalas.toml`
(uma escala vale quando o conjunto de ids das opções é o dela).

Uso (sem GPU): python scripts/onde_erra.py --nome h9r_k100 --out analises/onde_erra.md
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def carregar_escalas(path: Path = RAIZ / "configs" / "escalas.toml") -> list[list[str]]:
    if not path.exists():
        return []
    return [list(e["ordem"]) for e in tomllib.loads(path.read_text(encoding="utf-8")).get("escala", [])]


def escala_de(ids, escalas: list[list[str]]) -> list[str] | None:
    s = set(ids)
    return next((e for e in escalas if set(e) == s), None)


def nome_semente(nome: str, s: int) -> str:
    return nome if s == 0 else f"{nome}_s{s}"


def linhas(runs: Path, nome: str, fonte: str, sementes: list[int]) -> list[dict]:
    out = []
    for s in sementes:
        nm = nome_semente(nome, s)
        f = runs / f"{nm}_{fonte}_dev" / "predicoes.jsonl"
        if not f.exists():
            print(f">> falta {f}; pulei")
            continue
        fora = set()
        reg = runs / nm / "adapt_grupos.json"
        if reg.exists():
            fora = {g for r in json.loads(reg.read_text(encoding="utf-8")) for g in r["grupos"]}
        for x in f.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                if r["held_out"] and r["group_id"] not in fora:
                    out.append(r)
    return out


def resumo_familia(rows: list[dict], escalas: list[list[str]]) -> list[str]:
    n = len(rows)
    acc = sum(r["pred"] == r["gold"] for r in rows) / n
    gold, pred = Counter(r["gold"] for r in rows), Counter(r["pred"] for r in rows)
    conf = defaultdict(Counter)
    for r in rows:
        conf[r["gold"]][r["pred"]] += 1
    classes = sorted(set(gold) | set(pred), key=lambda c: -gold[c])
    recs = [conf[c][c] / gold[c] for c in gold]
    t = [f"Linhas: {n} · acurácia {acc:.3f} · classe mais comum {gold.most_common(1)[0][1] / n:.3f} · "
         f"acerto médio por classe {sum(recs) / len(recs):.3f} (chutar sempre a mesma classe dá {1 / len(recs):.3f})", "",
         "| classe | no gabarito | nas respostas | acerto da classe |", "|---|---:|---:|---:|"]
    for c in classes:
        rec = conf[c][c] / gold[c] if gold[c] else float("nan")
        t.append(f"| `{c}` | {gold[c] / n:.1%} | {pred[c] / n:.1%} | {rec:.3f} |")
    if len(classes) <= 6:
        t += ["", "Confusão (linha = gabarito, coluna = resposta):", "",
              "| | " + " | ".join(f"`{c}`" for c in classes) + " |", "|---|" + "---:|" * len(classes)]
        t += [f"| `{g}` | " + " | ".join(str(conf[g][p]) for p in classes) + " |" for g in classes]
    erros = [r for r in rows if r["pred"] != r["gold"]]
    com_escala = [(r, escala_de(r["probs"].keys(), escalas)) for r in erros]
    com_escala = [(r, e) for r, e in com_escala if e]
    if com_escala:
        viz = sum(abs(e.index(r["pred"]) - e.index(r["gold"])) == 1 for r, e in com_escala)
        t += ["", f"Escala: {len(com_escala)} erros em perguntas com escala; vizinhos {viz} "
                  f"({viz / len(com_escala):.1%}), graves {len(com_escala) - viz} ({1 - viz / len(com_escala):.1%})."]
    return t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--nome", default="h9r_k100")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--sementes", default="0,1,2")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    escalas = carregar_escalas()
    sem = [int(x) for x in a.sementes.split(",")]
    txt = [f"# Onde erra: `{a.nome}`, reservadas, sementes {a.sementes}\n"]
    for fonte in a.fontes.split(","):
        rows = linhas(Path(a.runs), a.nome, fonte, sem)
        for fam in sorted({r["family"] for r in rows}):
            txt += [f"## {fonte} · {fam}\n"] + resumo_familia([r for r in rows if r["family"] == fam], escalas) + [""]
    out = "\n".join(txt)
    print(out)
    if a.out:
        Path(a.out).write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
