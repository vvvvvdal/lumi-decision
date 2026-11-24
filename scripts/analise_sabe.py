"""Sabe que não sabe: o "não sei" pela confiança separa o que o Lumi erraria do que ele acerta?

Usa a MESMA recalibração da receita final (`h9_recalibrar.py`, método `T_fam+tau_fam` com k = 25):
- para cada família reservada, a temperatura T e o limiar tau são ajustados em 25 grupos rotulados da família;
- a medida usa o resto do `dev`;
- os grupos do treino curto saem dos dois braços;
- a semente do sorteio é a mesma do `h9_recalibrar.py`, então a cobertura e o erro dos respondidos repetem os
  números publicados.

Em cada linha medida, "responde" quando a confiança com T ≥ tau, e "não sei" quando fica abaixo.

Mede, para o `h4_log` (base) e o `h9r_k100` (novo):
- cobertura e erro entre os respondidos;
- erro entre os "não sei": o que ele erraria se fosse forçado a responder;
- a diferença entre os dois erros;
- AUROC da confiança: a chance de um acerto ter confiança maior que um erro;
- AUROC dentro de cada família, como média ponderada pelo nº de linhas.

O último é o teste "pelo motivo certo". Se o AUROC geral é alto e o de dentro da família fica perto de 0,5, o
"não sei" só separa uma matéria difícil de uma fácil, e não os casos difíceis dentro da mesma matéria.

Famílias vistas, como referência: T e tau ajustados nas vistas do `calib`, como o `lumi-eval calibrar` faz.

IC 95%: o bootstrap reamostra os grupos (`group_id`, juntando sementes e ordens) e, a cada volta, sorteia uma das
repetições dos 25 rótulos. Assim, o IC inclui a variação de quais casos entram e a de quais 25 rótulos foram usados.

Uso (na raiz do repo, sem GPU):
    python scripts/analise_sabe.py --sementes 0,1,2 --out analises/sabe_que_nao_sabe.md
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


h9 = _mod("h9_recalibrar")
h7 = h9.h7
metrics = h7.metrics

MEDIDAS = ("acc", "cobertura", "erro_respondidos", "erro_nao_sei", "diferenca", "auroc", "auroc_dentro")


def vetores(linhas: list[dict]) -> dict[str, np.ndarray]:
    """Linhas já com a T aplicada e a chave `_tau` -> vetores (confiança, acerto, responde, família, grupo)."""
    conf = np.array([max(r["probs"].values()) for r in linhas])
    tau = np.array([np.inf if r["_tau"] is None else r["_tau"] for r in linhas])
    return {"conf": conf, "hit": np.array([r["pred"] == r["gold"] for r in linhas]), "resp": conf >= tau,
            "fam": np.array([r["family"] for r in linhas]), "grupo": np.array([r["group_id"] for r in linhas])}


def medir(v: dict[str, np.ndarray], sel: np.ndarray | None = None) -> dict[str, float]:
    if sel is not None:
        v = {k: x[sel] for k, x in v.items()}
    hit, resp = v["hit"], v["resp"]
    nan = float("nan")
    e_r = float(1 - hit[resp].mean()) if resp.any() else nan
    e_n = float(1 - hit[~resp].mean()) if (~resp).any() else nan
    dentro, peso = 0.0, 0
    for f in np.unique(v["fam"]):
        m = v["fam"] == f
        a = metrics.auroc(v["conf"][m], hit[m])
        if not np.isnan(a):
            dentro += a * m.sum()
            peso += int(m.sum())
    return {"acc": float(hit.mean()), "cobertura": float(resp.mean()), "erro_respondidos": e_r, "erro_nao_sei": e_n,
            "diferenca": e_n - e_r, "auroc": metrics.auroc(v["conf"], hit), "auroc_dentro": dentro / peso if peso else nan}


def bootstrap(reps: list[dict[str, np.ndarray]], n_boot: int, rng: np.random.Generator) -> dict[str, list[float]]:
    """IC 95%: reamostra grupos e, a cada volta, uma das repetições (sorteios dos rótulos)."""
    todos = sorted(set().union(*(set(v["grupo"].tolist()) for v in reps)))
    idx_por_rep = []
    for v in reps:
        ordem = np.argsort(v["grupo"], kind="stable")
        g = v["grupo"][ordem]
        ini = np.searchsorted(g, todos, side="left")
        fim = np.searchsorted(g, todos, side="right")
        idx_por_rep.append((ordem, ini, fim))
    amostras = {m: [] for m in MEDIDAS}
    for _ in range(n_boot):
        j = int(rng.integers(len(reps)))
        ordem, ini, fim = idx_por_rep[j]
        esc = rng.integers(0, len(todos), len(todos))
        sel = np.concatenate([ordem[ini[i]:fim[i]] for i in esc])
        r = medir(reps[j], sel)
        for m in MEDIDAS:
            amostras[m].append(r[m])
    out = {}
    for m, xs in amostras.items():
        xs = np.array([x for x in xs if not np.isnan(x)])
        out[m] = [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if len(xs) else [float("nan")] * 2
    return out


def juntar(listas: list[list[dict]]) -> list[dict]:
    return [r for linhas in listas for r in linhas]


def reservadas(runs: Path, base: str, novo: str, fonte: str, sementes: list[int], a) -> dict:
    """{braço: [linhas da repetição 1 (todas as sementes juntas), ...]} e o resumo do h7 (para conferir)."""
    por_braco = {"base": [], "novo": []}
    conferir = {"base": [], "novo": []}
    for s in sementes:
        nb, nn = h9.nome(base, s), h9.nome(novo, s)
        if not (runs / nn / "adapt_grupos.json").exists():
            print(f">> falta {runs / nn / 'adapt_grupos.json'}; pulei semente {s}")
            continue
        fora = h9.grupos_de_treino(runs, nn)
        for outro in filter(None, a.excluir_tambem.split(",")):
            fora |= h9.grupos_de_treino(runs, h9.nome(outro, s))
        dados = {}
        for braco, nm in (("base", nb), ("novo", nn)):
            pc, pd = runs / f"{nm}_{fonte}_calib", runs / f"{nm}_{fonte}_dev"
            if not ((pc / "predicoes.jsonl").exists() and (pd / "predicoes.jsonl").exists()):
                print(f">> {fonte} semente {s}: faltam {pc} ou {pd}; pulei")
                break
            dados[braco] = (h7.carregar(pc), [r for r in h7.carregar(pd) if r["group_id"] not in fora])
        if len(dados) < 2:
            continue
        for braco, (calib, dev) in dados.items():
            rng = np.random.default_rng([a.seed, s])  # o mesmo sorteio do h9_recalibrar.py
            col = {}
            res = h7.uma_fonte(calib, dev, a.ks_sorteio, a.repeticoes, a.risco, "calib", a.min_aceitos, rng, coletar=col)
            por_braco[braco].append(col[a.k])
            conferir[braco] += res[f"T_fam+tau_fam|{a.k}"]
    out = {}
    for braco, por_sem in por_braco.items():
        if por_sem:
            # repetição i de todas as sementes juntas (a i-ésima repetição de cada semente)
            out[braco] = [juntar([sem[i] for sem in por_sem]) for i in range(len(por_sem[0]))]
    return out, conferir


def vistas(runs: Path, nome_run: str, fonte: str, sementes: list[int], a) -> list[dict]:
    linhas = []
    for s in sementes:
        nm = h9.nome(nome_run, s)
        pc, pd = runs / f"{nm}_{fonte}_calib", runs / f"{nm}_{fonte}_dev"
        if not ((pc / "predicoes.jsonl").exists() and (pd / "predicoes.jsonl").exists()):
            continue
        vc = [r for r in h7.carregar(pc) if not r["held_out"]]
        vd = [r for r in h7.carregar(pd) if not r["held_out"]]
        T, tau = h7.ajustar(vc, a.risco, 20)  # como o `lumi-eval calibrar`
        linhas += h7.aplicar(vd, T, tau)
    return linhas


def fmt(x: float, ic: list[float] | None = None, pct: bool = False) -> str:
    if np.isnan(x):
        return "—"
    f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.3f}")
    return f(x) if ic is None else f"{f(x)} [{f(ic[0])}; {f(ic[1])}]"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--base", default="h4_log")
    p.add_argument("--novo", default="h9r_k100")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--sementes", default="0,1,2")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--repeticoes", type=int, default=100)
    p.add_argument("--risco", type=float, default=0.10)
    p.add_argument("--min-aceitos", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--excluir-tambem", default="")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    # o h9_recalibrar.py sorteia k = 10, 25 e 50 nessa ordem com o mesmo gerador: repetir para cair nos mesmos grupos
    a.ks_sorteio = sorted({10, 25, 50, a.k})
    runs, sementes = Path(a.runs), [int(x) for x in a.sementes.split(",")]
    rng = np.random.default_rng(a.seed)
    cab = ("| fonte | braço | recorte | acc | responde | erro dos respondidos | erro dos \"não sei\" | diferença | "
           "AUROC | AUROC dentro da família |")
    linhas = [cab, "|---|---|---|---|---|---|---|---|---|---|"]
    conf_linhas = ["| fonte | braço | responde (h7) | erro dos respondidos (h7) |", "|---|---|---:|---:|"]
    bruto = {}
    for fonte in a.fontes.split(","):
        res, conferir = reservadas(runs, a.base, a.novo, fonte, sementes, a)
        for braco, nome_run in (("base", a.base), ("novo", a.novo)):
            casos = []
            if braco in res:
                casos.append(("reservadas", [vetores(lin) for lin in res[braco]]))
                cf = conferir[braco]
                conf_linhas.append(f"| {fonte} | {nome_run} | {np.mean([m['cobertura'] for m in cf]):.0%} | "
                                   f"{np.nanmean([m['risco_aceitos'] for m in cf]):.3f} |")
            vis = vistas(runs, nome_run, fonte, sementes, a)
            if vis:
                casos.append(("vistas", [vetores(vis)]))
            for recorte, reps in casos:
                pontos = [medir(v) for v in reps]
                ponto = {m: float(np.nanmean([x[m] for x in pontos])) for m in MEDIDAS}
                ic = bootstrap(reps, a.n_boot, rng)
                bruto[f"{fonte}|{nome_run}|{recorte}"] = {"ponto": ponto, "ic95": ic, "n_linhas": int(len(reps[0]["hit"]))}
                linhas.append(f"| {fonte} | {nome_run} | {recorte} | {fmt(ponto['acc'], pct=True)} | "
                              f"{fmt(ponto['cobertura'], ic['cobertura'], pct=True)} | "
                              f"{fmt(ponto['erro_respondidos'], ic['erro_respondidos'], pct=True)} | "
                              f"{fmt(ponto['erro_nao_sei'], ic['erro_nao_sei'], pct=True)} | "
                              f"{fmt(ponto['diferenca'], ic['diferenca'], pct=True)} | "
                              f"{fmt(ponto['auroc'], ic['auroc'])} | {fmt(ponto['auroc_dentro'], ic['auroc_dentro'])} |")
    nota = (f"\nReservadas: receita final (T e limiar da família com {a.k} rótulos, risco-alvo {a.risco:.0%}), média das "
            f"{a.repeticoes} repetições dos sorteios; IC 95% por bootstrap de grupos e de repetições. Vistas: T e limiar "
            "das vistas do calib. \"Não sei\" = confiança abaixo do limiar. Diferença = erro dos \"não sei\" − erro dos "
            "respondidos.\n\nConferência (tem que bater com os números publicados da receita final):\n"
            + "\n".join(conf_linhas))
    tabela = "\n".join(linhas)
    print(tabela + "\n" + nota)
    if a.out:
        Path(a.out).write_text(tabela + "\n" + nota + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(bruto, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
