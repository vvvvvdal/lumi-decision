"""H11b piloto: o Lumi escreve uma confiança graduada depois de raciocinar?

Reaproveita os raciocínios do piloto da H11 (`runs/<piloto>_<fonte>_dev`), sem gerar raciocínio de novo:
- remonta o prompt em que a letra foi lida (conferido pelo `prompt_sha1` gravado na H11);
- acrescenta a letra escolhida e pergunta, num turno novo do usuário, a probabilidade de 0 a 100 de a resposta
  estar certa (`prompts.PERGUNTA_CONFIANCA`), para o raciocínio guloso e para cada um dos sorteados;
- compara quatro fontes de confiança nos mesmos casos: a escrita, as letras depois do raciocínio, a consistência
  dos sorteios e as letras sem raciocínio (a leitura honesta da H4, `runs/<base>_<fonte>_dev`).

Critério em `analises/plano_h11b_confianca.md`.

Uso (na raiz do repo):
    python scripts/h11b_confianca.py --model ceia-br/lumi-4b --revision <rev> --native --adapter runs/h4_log/adapter
    python scripts/h11b_confianca.py --so-analise        (só CPU)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from lumi_decision.prompts import (LETTERS, PERGUNTA_CONFIANCA, SUFIXOS_RAC, com_raciocinio,  # noqa: E402
                                   ler_confianca, load_template, order_options, prompt_confianca,
                                   prompt_pensando, render, turno_confianca)

_spec = importlib.util.spec_from_file_location("h11_piloto", RAIZ / "scripts" / "h11_piloto.py")
h11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h11)

# Critério registrado (analises/plano_h11b_confianca.md), no guloso.
MIN_VALIDAS, MAX_MODA, MIN_AUROC = 0.90, 0.80, 0.65


# ------------------------------------------------------------------ geração (GPU)

def prompt_lido(template, case, row: dict, raciocinio: str) -> str:
    """O prompt em que a H11 leu as letras: o raciocínio fechado + o separador escolhido (`rac_sufixo`)."""
    options = order_options(case, row["order"])
    if [o[0] for o in options] != row["presented"]:
        raise ValueError(f"{case.id}: a ordem das opções difere da H11")
    prompt = render(template, case, options)
    p_gen = prompt_pensando(render(template, case, options, pensar=True), prompt)
    return com_raciocinio(p_gen, prompt, raciocinio) + SUFIXOS_RAC[row["rac_sufixo"]]


def sha1_ids(ids) -> str:
    return hashlib.sha1(json.dumps(list(ids)).encode()).hexdigest()


def perguntar(backend, prompt: str, max_tokens: int) -> dict:
    _, texto, info = backend.generate(backend.tokenize(prompt), max_tokens)
    return {"texto": texto, "conf": ler_confianca(texto), "ms": info.get("latency_ms"),
            "parada": info.get("stop_type")}


def gerar_caso(backend, template, depois: str, case, row: dict, max_tokens: int = 16) -> dict:
    """A confiança escrita para o raciocínio guloso e para cada sorteado de um caso da H11."""
    lido = prompt_lido(template, case, row, row["raciocinio"])
    if sha1_ids(backend.tokenize(lido)) != row["prompt_sha1"]:
        raise ValueError(f"{case.id}: o prompt remontado não é o mesmo que a H11 leu (prompt_sha1)")
    p = prompt_confianca(lido, LETTERS[row["presented"].index(row["pred"])], depois)
    out = {"case_id": row["case_id"], "order": row["order"], "pred": row["pred"], "gold": row["gold"],
           **perguntar(backend, p, max_tokens), "fim_do_prompt": p[-400:], "amostras": []}
    for a in row.get("rac_amostras") or []:
        pa = max(a["probs"], key=a["probs"].get)
        pr = prompt_confianca(prompt_lido(template, case, row, a["raciocinio"]),
                              LETTERS[row["presented"].index(pa)], depois)
        out["amostras"].append({"pred": pa, **perguntar(backend, pr, max_tokens)})
    return out


def ler_jsonl(path: Path) -> list[dict]:
    """Lê as linhas gravadas; descarta uma última linha truncada (job interrompido)."""
    linhas = path.read_text(encoding="utf-8").splitlines()
    boas = []
    for i, l in enumerate(linhas):
        if not l.strip():
            continue
        try:
            boas.append(json.loads(l))
        except json.JSONDecodeError:
            if i != len(linhas) - 1:
                raise
            path.write_text("".join(x + "\n" for x in linhas[:-1]), encoding="utf-8")
    return boas


def gerar_fonte(backend, template, casos: dict, linhas: list[dict], path: Path, max_tokens: int = 16,
                overwrite: bool = False) -> int:
    """Grava uma linha por caso em `path`. Retomável: pula os casos já gravados. Devolve quantos gerou."""
    depois = turno_confianca(template)
    feitos = set()
    if path.exists() and not overwrite:
        feitos = {(r["case_id"], r["order"]) for r in ler_jsonl(path)}
    n, t0 = 0, time.time()
    with path.open("w" if overwrite else "a", encoding="utf-8") as f:
        for r in linhas:
            if (r["case_id"], r["order"]) in feitos:
                continue
            f.write(json.dumps(gerar_caso(backend, template, depois, casos[r["case_id"]], r, max_tokens),
                               ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"{path.parent.name}: {n} casos  {time.time() - t0:.0f}s", flush=True)
    return n


def gerar(a) -> None:
    from lumi_decision.backends import make_backend
    from lumi_decision.data import load_cases, load_config
    from lumi_decision.evaluate import git_commit

    runs = Path(a.runs)
    template = load_template(Path(a.template))
    backend = make_backend("hf", model=a.model, revision=a.revision, dtype=a.dtype, trust_remote_code=not a.native,
                           adapter=a.adapter, use_pointer=False)
    try:
        for fonte in a.fontes.split(","):
            linhas = [r for r in h11.carregar(runs / f"{a.piloto}_{fonte}_dev") if "raciocinio" in r]
            if a.limite:
                linhas = linhas[: a.limite]
            casos = {c.id: c for c in load_cases(fonte, load_config())}
            out = runs / f"{a.nome}_{fonte}_dev"
            out.mkdir(parents=True, exist_ok=True)
            (out / "meta.json").write_text(json.dumps(
                {"args": vars(a), "pergunta": PERGUNTA_CONFIANCA, "git": git_commit(),
                 "inicio": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": backend.describe()},
                indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            n = gerar_fonte(backend, template, casos, linhas, out / "confianca.jsonl", a.max_tokens, a.overwrite)
            print(f">> {fonte}: {n} casos novos", flush=True)
    finally:
        if hasattr(backend, "close"):
            backend.close()


# ------------------------------------------------------------------ análise (CPU)

from lumi_decision.metrics import auroc  # noqa: E402  (era definida aqui; foi para metrics)


def ece_binaria(conf, acerto, bins: int = 10) -> float:
    c, y = np.asarray(conf, float), np.asarray(acerto, float)
    b = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(c[b == k].mean() - y[b == k].mean()) for k in range(bins) if (b == k).any()))


def medir(conf, acerto) -> dict:
    """AUROC, confiança média quando acerta e quando erra, ECE e Brier de uma fonte de confiança binária."""
    conf, acerto = list(conf), list(acerto)
    if not conf:
        return {"n": 0}
    ca = [c for c, y in zip(conf, acerto) if y]
    ce = [c for c, y in zip(conf, acerto) if not y]
    return {"n": len(conf), "acc": float(np.mean(acerto)), "auroc": auroc(conf, acerto),
            "conf_acerta": float(np.mean(ca)) if ca else float("nan"),
            "conf_erra": float(np.mean(ce)) if ce else float("nan"),
            "ece": ece_binaria(conf, acerto), "brier": float(np.mean([(c - y) ** 2 for c, y in zip(conf, acerto)]))}


def escrita(confs: list) -> dict:
    """Como ele escreve: fração com número válido, em 100, ≥ 0,9, valores distintos, o mais comum, por décimos."""
    val = [round(c, 4) for c in confs if c is not None]
    if not val:
        return {"n": len(confs), "validas": 0.0, "moda": float("nan"), "moda_frac": 1.0}
    v, k = Counter(val).most_common(1)[0]
    return {"n": len(confs), "validas": len(val) / len(confs), "em_100": float(np.mean([c >= 1 for c in val])),
            "acima_90": float(np.mean([c >= 0.9 for c in val])), "distintos": len(set(val)), "moda": v,
            "moda_frac": k / len(val), "decimos": [sum(min(int(c * 10), 9) == d for c in val) for d in range(10)]}


def recompensa_rlcr(c: float, acerto: bool) -> float:
    """RLCR: acerto (0/1) menos o Brier da confiança escrita."""
    return float(acerto) - (c - float(acerto)) ** 2


def _f(x, fmt=".3f") -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def analisar(runs: Path, nome: str, piloto: str, base: str, fontes: list[str]) -> tuple[str, dict]:
    fontes_conf = {k: ([], []) for k in ("escrita", "letras", "consistencia", "sem_rac")}
    confs_g, por_recorte, exemplos, fim_prompt, ms = [], {}, [], None, []
    am_conf, am_letras, confs_a, grupos = ([], []), ([], []), [], []
    for fonte in fontes:
        pc, pp, pb = (runs / f"{nome}_{fonte}_dev" / "confianca.jsonl", runs / f"{piloto}_{fonte}_dev",
                      runs / f"{base}_{fonte}_dev")
        if not (pc.exists() and (pp / "predicoes.jsonl").exists()):
            print(f">> {fonte}: faltam {pc} ou {pp}; pulei")
            continue
        pil = {(r["case_id"], r["order"]): r for r in h11.carregar(pp) if "raciocinio" in r}
        b = ({r["case_id"]: r for r in h11.carregar(pb) if r["order"] == "canonical"}
             if (pb / "predicoes.jsonl").exists() else {})
        for c in ler_jsonl(pc):
            r = pil[(c["case_id"], c["order"])]
            ok = r["pred"] == r["gold"]
            confs_g.append(c["conf"])
            ms.append(c.get("ms") or 0.0)
            por_recorte.setdefault(f"{fonte}|{'reservadas' if r['held_out'] else 'vistas'}", []).append((c["conf"], ok))
            if c["conf"] is not None:
                fontes_conf["escrita"][0].append(c["conf"]), fontes_conf["escrita"][1].append(ok)
            fontes_conf["letras"][0].append(max(r["probs"].values())), fontes_conf["letras"][1].append(ok)
            am = r.get("rac_amostras") or []
            if am:
                cons = float(np.mean([max(a["probs"], key=a["probs"].get) == r["pred"] for a in am]))
                fontes_conf["consistencia"][0].append(cons), fontes_conf["consistencia"][1].append(ok)
            if r["case_id"] in b:
                x = b[r["case_id"]]
                fontes_conf["sem_rac"][0].append(max(x["probs"].values()))
                fontes_conf["sem_rac"][1].append(x["pred"] == x["gold"])
            rec_escrita, rec_letras, cg = [], [], []
            for a, ca in zip(am, c.get("amostras", [])):
                ya = ca["pred"] == r["gold"]
                confs_a.append(ca["conf"])
                cg.append(ca["conf"])
                if ca["conf"] is not None:
                    am_conf[0].append(ca["conf"]), am_conf[1].append(ya)
                    rec_escrita.append(recompensa_rlcr(ca["conf"], ya))
                am_letras[0].append(max(a["probs"].values())), am_letras[1].append(ya)
                rec_letras.append(h11.recompensa_brier(a["probs"], r["gold"]))
            if am:
                grupos.append({"muda": len({round(x, 4) for x in cg if x is not None}) > 1,
                               "sinal_rlcr": len(rec_escrita) > 1 and float(np.std(rec_escrita)) > 0.01,
                               "sinal_letras": float(np.std(rec_letras)) > 0.01})
            if len(exemplos) < 10:
                exemplos.append(f"- {fonte} · {r['family']} · {'reservada' if r['held_out'] else 'vista'} · gabarito "
                                f"{r['gold']} · escolheu {r['pred']} ({'certo' if ok else 'errado'}) · escreveu "
                                f"`{json.dumps(c['texto'], ensure_ascii=False)}` → {_f(c['conf'], '.2f')}")
            fim_prompt = fim_prompt or c.get("fim_do_prompt")
    if not confs_g:
        raise SystemExit("nenhuma confiança escrita encontrada")

    esc, med = escrita(confs_g), {k: medir(*v) for k, v in fontes_conf.items()}
    mg = med["escrita"]
    crit = [("Escreve: respostas com número de 0 a 100", "≥ 90%", f"{esc['validas']:.0%}",
             esc["validas"] >= MIN_VALIDAS),
            ("Gradua: o valor mais comum", "≤ 80% das respostas",
             f"{esc['moda_frac']:.0%} (valor {_f(esc['moda'] * 100, '.0f')})", esc["moda_frac"] <= MAX_MODA),
            ("Separa: AUROC da confiança escrita", "≥ 0,65", _f(mg.get("auroc")),
             mg.get("n", 0) > 0 and not math.isnan(mg["auroc"]) and mg["auroc"] >= MIN_AUROC)]
    nomes = {"escrita": "confiança escrita", "letras": "letras depois do raciocínio",
             "consistencia": "consistência dos 4 sorteios", "sem_rac": "letras sem raciocínio (H4; outra decisão)"}
    tab = ["| fonte de confiança | n | acurácia da decisão | AUROC | confiança média: acerta / erra | ECE | Brier |",
           "|---|---:|---:|---:|---|---:|---:|"]
    for k, m in med.items():
        if m["n"]:
            tab.append(f"| {nomes[k]} | {m['n']} | {m['acc']:.3f} | {_f(m['auroc'])} | {_f(m['conf_acerta'], '.2f')} / "
                       f"{_f(m['conf_erra'], '.2f')} | {m['ece']:.3f} | {m['brier']:.3f} |")
    rec = ["| fonte | recorte | n | acurácia | válidas | AUROC | confiança média: acerta / erra |",
           "|---|---|---:|---:|---:|---:|---|"]
    bruto_rec = {}
    for k, xs in por_recorte.items():
        val = [(c, y) for c, y in xs if c is not None]
        m = medir([c for c, _ in val], [y for _, y in val])
        bruto_rec[k] = {**m, "validas": len(val) / len(xs)}
        fonte, nome_r = k.split("|")
        rec.append(f"| {fonte} | {nome_r} | {len(xs)} | {np.mean([y for _, y in xs]):.3f} | {len(val) / len(xs):.0%} | "
                   f"{_f(m.get('auroc'))} | {_f(m.get('conf_acerta'), '.2f')} / {_f(m.get('conf_erra'), '.2f')} |")
    bruto = {"criterio": [{"nome": n, "limite": l, "resultado": r, "passa": bool(p)} for n, l, r, p in crit],
             "escrita": esc, "fontes": med, "por_recorte": bruto_rec, "ms_mediano": float(np.median(ms))}
    md = ["# H11b: confiança escrita depois do raciocínio", "",
          f"Pergunta, num turno novo depois da letra: \"{PERGUNTA_CONFIANCA}\"", "",
          f"Raciocínios do piloto da H11 (`{piloto}`), mesmo adaptador. Tempo mediano por resposta: "
          f"{np.median(ms) / 1000:.2f} s.", "",
          "## Critério (guloso)", "", "| critério | limite | resultado | |", "|---|---|---|---|"]
    md += [f"| {n} | {l} | {r} | {'passa' if p else '**não passa**'} |" for n, l, r, p in crit]
    md += ["", "## Quatro fontes de confiança (guloso, mesmos casos)", ""] + tab
    md += ["", "AUROC = chance de um acerto ter confiança maior que um erro (0,5 = não separa). As letras sem "
           "raciocínio julgam a decisão de sempre, que é outra; entram só como referência.", "",
           "## Como ele escreve (guloso)", "",
           f"Com número válido: {esc['validas']:.0%}. Em 100: {_f(esc.get('em_100'), '.0%')}. "
           f"Em 90 ou mais: {_f(esc.get('acima_90'), '.0%')}. Valores diferentes: {esc.get('distintos', 0)}. "
           f"O mais comum: {_f(esc['moda'] * 100, '.0f')} ({esc['moda_frac']:.0%}).", ""]
    if esc.get("decimos"):
        md += ["Por décimos (0–9, 10–19, …, 90–100): " + " · ".join(map(str, esc["decimos"])), ""]
    md += ["## Por recorte (confiança escrita, guloso)", ""] + rec
    if grupos:
        sa, sl = medir(*am_conf), medir(*am_letras)
        ea = escrita(confs_a)
        md += ["", "## Sorteio (cada um dos 4 raciocínios sorteados)", "",
               "| fonte de confiança | amostras | válidas | AUROC | confiança média: acerta / erra | ECE | Brier |",
               "|---|---:|---:|---:|---|---:|---:|"]
        for nm, m, v in (("confiança escrita", sa, f"{ea['validas']:.0%}"), ("letras depois do raciocínio", sl, "—")):
            if m["n"]:
                md.append(f"| {nm} | {m['n']} | {v} | {_f(m['auroc'])} | {_f(m['conf_acerta'], '.2f')} / "
                          f"{_f(m['conf_erra'], '.2f')} | {m['ece']:.3f} | {m['brier']:.3f} |")
        g = {k: float(np.mean([x[k] for x in grupos])) for k in ("muda", "sinal_rlcr", "sinal_letras")}
        md += ["", f"Grupos de 4: a confiança escrita muda dentro do grupo em {g['muda']:.0%}. Grupos com sinal "
               f"(desvio > 0,01): recompensa do RLCR (acerto − Brier da confiança escrita) {g['sinal_rlcr']:.0%}, "
               f"contra {g['sinal_letras']:.0%} com o Brier das letras."]
        bruto["sorteio"] = {"escrita": sa, "letras": sl, "como_escreve": ea, **g, "grupos": len(grupos)}
    md += ["", "## Respostas cruas (exemplos)", ""] + exemplos
    if fim_prompt:
        md += ["", "## Fim de um prompt (para conferir o formato)", "", "```",
               json.dumps(fim_prompt, ensure_ascii=False), "```"]
    return "\n".join(md) + "\n", bruto


def main(argv=None):
    from lumi_decision.data import ARTIFACTS_DIR

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--nome", default="h11b", help="pastas runs/<nome>_<fonte>_dev")
    p.add_argument("--piloto", default="h11p2", help="raciocínios da H11: runs/<piloto>_<fonte>_dev")
    p.add_argument("--base", default="h4_log", help="leitura sem raciocínio: runs/<base>_<fonte>_dev")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--out", default="")
    p.add_argument("--so-analise", action="store_true")
    p.add_argument("--limite", type=int, default=0, help="só os N primeiros casos de cada fonte (teste curto)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--model", default="ceia-br/lumi-4b")
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--native", action="store_true")
    p.add_argument("--adapter", default=None)
    p.add_argument("--template", default=str(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))
    a = p.parse_args(argv)
    if not a.so_analise:
        gerar(a)
    texto, bruto = analisar(Path(a.runs), a.nome, a.piloto, a.base, a.fontes.split(","))
    print(texto)
    if a.out:
        Path(a.out).write_text(texto, encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(bruto, indent=1, ensure_ascii=False, default=float),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
