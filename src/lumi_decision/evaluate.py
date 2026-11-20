"""Execução e relatório.

    lumi-eval run --backend llamacpp --model X.gguf --server-bin llama-server.exe \\
        --source felhen --split dev --orders canonical,reversed --out runs/h1_gguf
    lumi-eval report runs/h1_gguf
    lumi-eval baselines --source felhen --split dev --out runs/baselines_felhen_dev
    lumi-eval comparar runs/A runs/B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import REPO_ROOT, metrics
from .data import ARTIFACTS_DIR, SPLITS_CONFIG, check_integrity, load_cases, load_config
from .prompts import (LETTERS, PROMPT_VERSION, candidate_tokens, load_template, option_char_ends,
                      option_token_positions, order_options, render, with_abstain)

GEN_RE = re.compile(r"\s*(?:\*\*)?([A-Z])(?:\*\*)?[.)]?\s*")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        return (out.stdout.strip() or "sem-commit") + ("+alteracoes" if dirty.stdout.strip() else "")
    except OSError:
        return None


def select_cases(args) -> list:
    cases = load_cases(args.source, load_config())
    check_integrity(cases)
    cases = [c for c in cases if c.split == args.split] if args.split != "all" else cases
    if args.families:
        fams = set(args.families.split(","))
        cases = [c for c in cases if c.family in fams]
    if args.held_out == "exclude":
        cases = [c for c in cases if not c.held_out]
    elif args.held_out == "only":
        cases = [c for c in cases if c.held_out]
    if args.per_family:
        by = defaultdict(list)
        for c in cases:
            by[c.family].append(c)
        cases = []
        for fam in sorted(by):
            items = sorted(by[fam], key=lambda c: hashlib.sha256(f"{args.seed}:{c.id}".encode()).hexdigest())
            cases.extend(items[: args.per_family])
    if args.limit:
        cases = cases[: args.limit]
    if getattr(args, "abster", False):  # H5: acrescenta a opção "Não sei" (sempre a última)
        cases = [with_abstain(c) for c in cases]
    return cases


def escolher_exemplos(case, pool: list, k: int, seed: int = 0) -> list:
    """H8: k casos com gabarito da MESMA família, nunca do mesmo grupo do caso (mesmo texto/estado).
    Ordem determinística por caso (hash), para a avaliação ser reentrante e reprodutível."""
    cands = [c for c in pool if c.family == case.family and c.group_id != case.group_id]
    cands.sort(key=lambda c: hashlib.sha256(f"{seed}:{case.id}:{c.id}".encode()).hexdigest())
    escolhidos, grupos = [], set()
    for c in cands:  # um exemplo por grupo: perguntas diferentes do mesmo estado não contam como exemplos novos
        if c.group_id not in grupos:
            escolhidos.append(c)
            grupos.add(c.group_id)
        if len(escolhidos) == k:
            break
    return escolhidos


def predict_one(backend, template, case, order, generate=True, max_new_tokens=16, exemplos=(),
                max_prompt_tokens=0) -> dict:
    options = order_options(case, order)
    exemplos = list(exemplos)
    while True:  # H8: tira exemplos do fim até o prompt caber no teto de tokens
        prompt = render(template, case, options, exemplos)
        ids, cands = candidate_tokens(backend.tokenize, prompt, len(options))
        if not exemplos or not max_prompt_tokens or len(ids) <= max_prompt_tokens:
            break
        exemplos.pop()
    pointer = getattr(backend, "pointer", None) is not None
    if pointer:  # E4: mesmo prompt; a pointer head lê o fim de cada opção e a última posição
        ids2, offs = backend.offsets(prompt)
        if list(ids2) != list(ids):
            raise RuntimeError(f"{case.id}: tokenização com offsets difere")
        pos = option_token_positions(offs, prompt, option_char_ends(prompt, case, options))
        res, rinfo = backend.pointer_readout_batch([ids], [pos], batch_cands=[cands])
        lps = res[0]
    else:
        lps, rinfo = backend.readout(ids, cands)
    m = max(lps)
    w = [math.exp(x - m) for x in lps]
    z = sum(w)
    probs = {options[i][0]: w[i] / z for i in range(len(options))}
    row = {
        "case_id": case.id, "group_id": case.group_id, "source": case.source, "family": case.family,
        "split": case.split, "held_out": case.held_out, "qtype": case.qtype, "order": order,
        "presented": [o[0] for o in options], "gold": case.gold, "probs": probs,
        "pred": max(probs, key=probs.get), "cand_mass": None if pointer else float(sum(math.exp(x) for x in lps)),
        "readout": ("residual" if getattr(backend.pointer, "residual", False) else "ponteiro") if pointer else "letras",
        "prompt_tokens": len(ids), "prompt_sha1": hashlib.sha1(json.dumps(ids).encode()).hexdigest(),
        "readout_ms": rinfo.get("latency_ms"),
    }
    if case.gold_dist:
        row["gold_dist"] = case.gold_dist
    if exemplos:
        row["exemplos_ids"] = [e.id for e in exemplos]
    if generate:
        toks, text, ginfo = backend.generate(ids, max_new_tokens)
        mm = GEN_RE.fullmatch(text)
        letter = mm[1] if mm and LETTERS.index(mm[1]) < len(options) else None
        row.update({
            "gen_text": text, "gen_token_ids": toks, "gen_stop": ginfo.get("stop_type"),
            "gen_pred": options[LETTERS.index(letter)][0] if letter else None,
            "gen_first_is_readout_argmax": None if pointer else (bool(toks) and toks[0] == cands[int(np.argmax(lps))]),
            "gen_ms": ginfo.get("latency_ms"),
        })
    return row


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_fingerprint(args, cases) -> dict:
    """Tudo que muda o significado do conjunto de predições: configuração do modelo e do prompt,
    e também QUAIS casos e ordens entram (o relatório lê todas as linhas acumuladas na pasta)."""
    model_id = args.model
    if args.backend == "llamacpp" and args.model and Path(args.model).is_file():
        model_id = f"{Path(args.model).name}:{Path(args.model).stat().st_size}"
    return {
        "backend": args.backend, "model": model_id, "revision": args.revision, "dtype": args.dtype,
        "native": bool(getattr(args, "native", False)), "ctx": args.ctx if args.backend == "llamacpp" else None,
        "prompt_version": PROMPT_VERSION, "template_sha256": sha256_of(Path(args.template)),
        "splits_sha256": sha256_of(SPLITS_CONFIG), "source": args.source, "split": args.split,
        "generate": not args.no_generate, "orders": args.orders.split(","),
        "n_casos": len(cases), "casos_sha256": hashlib.sha256("\n".join(c.id for c in cases).encode()).hexdigest(),
        **({"adapter_sha256": sha256_of(Path(args.adapter) / "adapter_model.safetensors")}
           if getattr(args, "adapter", None) else {}),
        **({"ponteiro_sha256": sha256_of(Path(args.adapter) / "ponteiro.safetensors")}
           if getattr(args, "adapter", None) and (Path(args.adapter) / "ponteiro.safetensors").exists()
           and not getattr(args, "sem_ponteiro", False) else {}),
        **({"sem_ponteiro": True} if getattr(args, "sem_ponteiro", False) else {}),
        **({"exemplos_contexto": args.exemplos_contexto, "max_prompt_tokens": args.max_prompt_tokens}
           if getattr(args, "exemplos_contexto", 0) else {}),
    }


def read_existing(pred_path: Path) -> set:
    """Lê predições já gravadas; descarta uma última linha truncada (job interrompido)."""
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    good = []
    for i, line in enumerate(lines):
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            pred_path.write_text("".join(l + "\n" for l in lines[:-1]), encoding="utf-8")
    return {(r["case_id"], r["order"]) for r in good}


def cmd_run(args):
    from .backends import make_backend

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path, meta_path = out / "predicoes.jsonl", out / "meta.json"
    cases = select_cases(args)
    fp = run_fingerprint(args, cases)
    done, history = set(), []
    if pred_path.exists() and not args.overwrite:
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if old.get("impressao") != fp:
            raise SystemExit(f"{out} tem predições de outra configuração; use outra pasta ou --overwrite.\n"
                             f"antes: {old.get('impressao')}\nagora: {fp}")
        done = read_existing(pred_path)
        history = old.get("execucoes", [])
    orders = args.orders.split(",")
    template = load_template(Path(args.template))
    exemplos = {}
    if args.exemplos_contexto > 0:  # H8: exemplos com gabarito da mesma família, só para casos RESERVADOS
        todos = load_cases(args.source, load_config())
        # pool: casos da família reservada fora do split avaliado (Felhen: o train das reservadas, nunca usado);
        # se a família não tiver nenhum (Telepatia: reservadas só no dev), usa o próprio split, sem o grupo do caso
        for c in cases:
            if not c.held_out:
                continue
            outros = [x for x in todos if x.family == c.family and x.split not in (args.split, "test")]
            pool = outros or [x for x in todos if x.family == c.family and x.split == args.split]
            exemplos[c.id] = escolher_exemplos(c, pool, args.exemplos_contexto, args.seed)
    kw = {k: v for k, v in {"model": args.model, "server_bin": args.server_bin, "url": args.url,
                            "revision": args.revision, "dtype": args.dtype}.items() if v is not None}
    if args.backend == "llamacpp":
        kw.pop("revision", None), kw.pop("dtype", None)
        kw["log_path"] = str(out / "llama_server.log")
        kw["ctx"] = args.ctx
    else:
        kw["trust_remote_code"] = not args.native
        if getattr(args, "adapter", None):
            kw["adapter"] = args.adapter
        if getattr(args, "sem_ponteiro", False):
            kw["use_pointer"] = False
    backend = make_backend(args.backend, **kw)
    meta = {
        "impressao": fp, "args": {k: v for k, v in vars(args).items() if k != "func"},
        "n_casos": len(cases), "familias": dict(Counter(c.family for c in cases)),
        "execucoes": history + [{"inicio": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_commit(),
                                 "maquina": platform.node(), "plataforma": platform.platform(),
                                 "backend": backend.describe(), "ja_feitos": len(done)}],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    t0 = time.time()
    try:
        with pred_path.open("w" if args.overwrite else "a", encoding="utf-8") as f:
            for i, case in enumerate(cases):
                for order in orders:
                    if (case.id, order) in done:
                        continue
                    row = predict_one(backend, template, case, order, generate=not args.no_generate,
                                      exemplos=exemplos.get(case.id, ()), max_prompt_tokens=args.max_prompt_tokens)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                if (i + 1) % 25 == 0 or i + 1 == len(cases):
                    print(f"{i + 1}/{len(cases)} casos  {time.time() - t0:.0f}s", flush=True)
    finally:
        if hasattr(backend, "close"):
            backend.close()
    cmd_report(argparse.Namespace(run=str(out)))


def load_rows(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _latency(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))} if v else None


def build_report(rows) -> dict:
    rep = {"geral": {}, "por_familia": {}, "permutacao": {}, "latencia_ms": {}}
    by_src = defaultdict(list)
    for r in rows:
        by_src[(r["source"], r["held_out"])].append(r)
    for (src, held), rs in sorted(by_src.items()):
        key = f"{src}|{'reservadas' if held else 'vistas'}"
        fams = defaultdict(list)
        for r in rs:
            fams[r["family"]].append(r)
        per_fam = {f: metrics.summarize(v) for f, v in sorted(fams.items())}
        rep["por_familia"][key] = per_fam
        rep["geral"][key] = {
            **metrics.summarize(rs),
            "acuracia_balanceada_media_familias": float(np.mean([m["acuracia_balanceada"] for m in per_fam.values()])),
        }
        rep["permutacao"][key] = metrics.permutation_stats(rs)
        rep["latencia_ms"][key] = {"readout": _latency(rs, "readout_ms"), "geracao": _latency(rs, "gen_ms")}
    return rep


def cmd_report(args):
    run = Path(args.run)
    rep = build_report(load_rows(run))
    (run / "relatorio.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["| grupo | n | acc | acc bal. média fam. | NLL | ECE | AURC | risco@cob80 | muda c/ ordem | geração = readout |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, g in rep["geral"].items():
        p = rep["permutacao"].get(k) or {}
        lines.append(f"| {k} | {g['n']} | {g['acuracia']:.3f} | {g['acuracia_balanceada_media_familias']:.3f} | "
                     f"{g['nll']:.3f} | {g['ece']:.3f} | {g['aurc']:.3f} | {g['risco@cob80']:.3f} | "
                     f"{p.get('decisao_muda_entre_ordens', float('nan')):.3f} | "
                     f"{g.get('geracao_concorda_readout', float('nan')):.3f} |")
    lines += ["", "| grupo | família | n | acc | acc bal. | acaso |", "|---|---|---:|---:|---:|---:|"]
    for k, fams in rep["por_familia"].items():
        for f, m in fams.items():
            lines.append(f"| {k} | {f} | {m['n']} | {m['acuracia']:.3f} | {m['acuracia_balanceada']:.3f} | {m['acaso']:.3f} |")
    (run / "relatorio.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def question_key(c) -> tuple:
    """Unidade de uma referência supervisionada: mesma família, mesma pergunta, mesmo conjunto de opções.
    (Na Telepatia cada estado tem 5 perguntas diferentes; misturá-las tiraria a classe do menu.)
    No Traces a família é o tipo da ferramenta correta e o menu é montado em torno dela: usar qualquer
    um dos dois vazaria o gabarito, então a referência é global (frequência no treino, restrita ao menu)."""
    if c.source == "traces":
        return ("traces",)
    return (c.family, c.meta.get("question_key") or c.question, tuple(sorted(o[0] for o in c.options)))


def baseline_probs(name: str, cases, evalset) -> tuple[list[dict], dict]:
    """Devolve (linhas, notas). Toda distribuição é sobre o menu apresentado no caso."""
    from .data import _cosine, _trigrams

    train = defaultdict(list)
    for c in cases:
        if c.split == "train" and not c.held_out:
            train[question_key(c)].append(c)
    notes = Counter()
    rows = []
    by_key = defaultdict(list)
    for c in evalset:
        by_key[question_key(c)].append(c)
    for key, items in sorted(by_key.items()):
        tr = train.get(key, [])
        if name == "lexical":  # não supervisionado: similaridade de trigramas contexto × descrição da opção
            probas = []
            for c in items:
                q = _trigrams(f"{c.question} {c.state}")
                s = {o[0]: _cosine(q, _trigrams(o[1])) for o in c.options}
                m = max(s.values())
                w = {k: math.exp((v - m) / 0.05) for k, v in s.items()}
                probas.append(w)
        elif not tr:
            notes["sem_treino_para_a_pergunta(reservada)"] += len(items)
            continue
        elif name == "maioria":
            cnt = Counter(c.gold for c in tr)
            probas = []
            for c in items:
                menu = {o[0]: float(cnt.get(o[0], 0)) for o in c.options}
                top = max(menu, key=lambda k: (menu[k], -list(menu).index(k)))
                if menu[top] == 0:
                    notes["maioria_sem_classe_no_menu"] += 1
                probas.append({k: float(k == top) for k in menu})
        elif name == "tfidf":
            if items[0].source == "traces":
                notes["tfidf_nao_aplicavel_em_traces(menu_variavel)"] += len(items)
                continue
            if len({c.gold for c in tr}) < 2:
                notes["tfidf_uma_classe_no_treino"] += len(items)
                continue
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                                max_features=200000, sublinear_tf=True),
                                LogisticRegression(max_iter=1000))
            clf.fit([c.state for c in tr], [c.gold for c in tr])
            P = clf.predict_proba([c.state for c in items])
            classes = list(clf.classes_)
            probas = [{o[0]: (float(P[i][classes.index(o[0])]) if o[0] in classes else 0.0) for o in c.options}
                      for i, c in enumerate(items)]
        else:
            raise ValueError(name)
        for c, p in zip(items, probas):
            s = sum(p.values())
            p = {k: v / s for k, v in p.items()} if s > 0 else {o[0]: 1 / len(c.options) for o in c.options}
            rows.append({"case_id": c.id, "group_id": c.group_id, "source": c.source, "family": c.family,
                         "split": c.split, "held_out": c.held_out, "qtype": c.qtype, "order": "canonical",
                         "gold": c.gold, "probs": p, "pred": max(p, key=p.get),
                         **({"gold_dist": c.gold_dist} if c.gold_dist else {})})
    return rows, dict(notes)


def cmd_baselines(args):
    """Referências sobre o menu de cada caso: maioria e TF-IDF (supervisionadas, só perguntas vistas no
    treino) e similaridade lexical (não supervisionada, vale também para famílias reservadas)."""
    cases = load_cases(args.source, load_config())
    check_integrity(cases)
    evalset = [c for c in cases if c.split == args.split]
    out = Path(args.out)
    for name in ("maioria", "tfidf", "lexical"):
        rows, notes = baseline_probs(name, cases, evalset)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "predicoes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"baseline": name, "notas": notes, "git": git_commit(),
                                                "args": {k: v for k, v in vars(args).items() if k != "func"}},
                                               indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n## {name}  {notes}")
        if rows:
            cmd_report(argparse.Namespace(run=str(d)))


def cmd_calibrar(args):
    """Ajusta temperatura e limiar de abstenção na execução de CALIBRAÇÃO e aplica, fixos, na execução alvo
    (dev ou test). Grava `calibracao.json` na pasta alvo."""
    calib = [r for r in load_rows(Path(args.calib)) if not r["held_out"]]
    target = load_rows(Path(args.alvo))
    if not calib:
        raise SystemExit("execução de calibração sem casos vistos")
    T = metrics.fit_temperature(calib)
    calib_T = metrics.with_temperature(calib, T)
    res = {"temperatura": T, "n_calib": len(calib), "risco_alvo": args.risco,
           "calib": {"ece_antes": metrics.ece(calib), "ece_depois": metrics.ece(calib_T)}}
    tau = metrics.threshold_for_risk(calib_T, args.risco)
    res["limiar"] = tau
    groups = defaultdict(list)
    for r in target:
        groups[f"{r['source']}|{'reservadas' if r['held_out'] else 'vistas'}"].append(r)
    for g, rs in sorted(groups.items()):
        rT = metrics.with_temperature(rs, T)
        res[g] = {"ece_antes": metrics.ece(rs), "ece_depois": metrics.ece(rT),
                  "nll_antes": metrics.nll(rs), "nll_depois": metrics.nll(rT),
                  "abstencao_com_limiar_fixo": metrics.selective_at(rT, tau)}
    (Path(args.alvo) / "calibracao.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_comparar(args):
    a, b = load_rows(Path(args.a)), load_rows(Path(args.b))
    groups = defaultdict(lambda: ([], []))
    for r in a:
        groups[(r["source"], r["held_out"])][0].append(r)
    for r in b:
        groups[(r["source"], r["held_out"])][1].append(r)
    res = {f"{s}|{'reservadas' if h else 'vistas'}": metrics.paired_bootstrap(ra, rb)
           for (s, h), (ra, rb) in sorted(groups.items()) if ra and rb}
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_h2(args):
    """Critério da H2, fixado antes do treino: média com peso igual, nas famílias RESERVADAS de Felhen e
    Telepatia, de acc(novo) - acc(base) sobre todas as ordens; IC pareado reamostrando grupos de texto/estado
    dentro de cada família. Exige os mesmos pares (caso, ordem) dos dois lados. Traces é só diagnóstico."""
    base = [r for d in args.base.split(",") for r in load_rows(Path(d))]
    novo = [r for d in args.novo.split(",") for r in load_rows(Path(d))]
    cfg = load_config()
    exigidas = list(cfg["felhen"]["held_out_tasks"]) + list(cfg["telepatia"]["held_out_workflows"])
    res = {"criterio": "media_familias_reservadas_felhen_telepatia", "familias_exigidas": exigidas,
           "h2": metrics.family_mean_bootstrap([r for r in base if r["held_out"] and r["source"] != "traces"],
                                               [r for r in novo if r["held_out"] and r["source"] != "traces"],
                                               require_families=exigidas)}
    for src in sorted({r["source"] for r in base}):
        for h, nome in ((False, "vistas"), (True, "reservadas")):
            a = [r for r in base if r["source"] == src and r["held_out"] == h]
            b = [r for r in novo if r["source"] == src and r["held_out"] == h]
            if a and b:
                res[f"{src}|{nome}"] = {
                    "base": metrics.order_mean_accuracy(a), "novo": metrics.order_mean_accuracy(b),
                    "delta": metrics.paired_bootstrap(a, b),
                    "familias": metrics.family_mean_bootstrap(a, b)["por_familia"],
                    "muda_com_ordem": {"base": metrics.permutation_stats(a).get("decisao_muda_entre_ordens"),
                                       "novo": metrics.permutation_stats(b).get("decisao_muda_entre_ordens")}}
    lo, hi = res["h2"]["ic95"]
    res["veredito_h2"] = "ganho" if lo > 0 else ("perda" if hi < 0 else "inconclusivo")
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


def cmd_h3(args):
    """Critério do E4 (H3), registrado no README antes de rodar. Base = A1 (E3), novo = B (pointer head).
    Qualidade (principal): mesma métrica e mesmo bootstrap da H2 nas reservadas. Estabilidade (secundária):
    diferença na taxa de casos cuja escolha muda entre as ordens, mesma média e mesmo bootstrap por grupo."""
    base = [r for d in args.base.split(",") for r in load_rows(Path(d))]
    novo = [r for d in args.novo.split(",") for r in load_rows(Path(d))]
    # Sem `ponteiro.safetensors` o backend volta às letras: o B precisa ter lido TODO caso pela cabeça, e o A1
    # pelas letras (as predições do E3 são anteriores ao campo `readout`, que então vale "letras").
    metrics.require_readout(base, "letras", "base (A1)")
    metrics.require_readout(novo, getattr(args, "readout_novo", "ponteiro"), "novo (B)")
    cfg = load_config()
    exigidas = list(cfg["felhen"]["held_out_tasks"]) + list(cfg["telepatia"]["held_out_workflows"])
    res_a = [r for r in base if r["held_out"] and r["source"] != "traces"]
    res_b = [r for r in novo if r["held_out"] and r["source"] != "traces"]
    q = metrics.family_mean_bootstrap(res_a, res_b, require_families=exigidas)
    f = metrics.family_mean_flip_bootstrap(res_a, res_b, require_families=exigidas)
    res = {"criterio": "h3_e4", "familias_exigidas": exigidas, "qualidade": q, "estabilidade": f,
           "veredito": metrics.h3_verdict(q["ic95"], f["ic95"])}
    for src in sorted({r["source"] for r in base}):
        for h, nome in ((False, "vistas"), (True, "reservadas")):
            a = [r for r in base if r["source"] == src and r["held_out"] == h]
            b = [r for r in novo if r["source"] == src and r["held_out"] == h]
            if a and b:
                pa, pb = metrics.permutation_stats(a), metrics.permutation_stats(b)
                res[f"{src}|{nome}"] = {
                    "base": metrics.order_mean_accuracy(a), "novo": metrics.order_mean_accuracy(b),
                    "delta": metrics.paired_bootstrap(a, b),
                    "familias": metrics.family_mean_bootstrap(a, b)["por_familia"],
                    "muda_com_ordem": {"base": pa.get("decisao_muda_entre_ordens"),
                                       "novo": pb.get("decisao_muda_entre_ordens")},
                    # diagnóstico de custo: A1 com DUAS passadas (média das probabilidades das ordens)
                    "base_duas_passadas": pa.get("acuracia_media_das_ordens"),
                    "nll": {"base": metrics.nll(a), "novo": metrics.nll(b)},
                    "ece": {"base": metrics.ece(a), "novo": metrics.ece(b)}}
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_selection(sp):
        sp.add_argument("--source", required=True, help="felhen | telepatia | traces | caminho de arquivo de casos")
        sp.add_argument("--split", default="dev", help="train | dev | calib | test | all")
        sp.add_argument("--families", default=None)
        sp.add_argument("--held-out", choices=("include", "exclude", "only"), default="include")
        sp.add_argument("--per-family", type=int, default=0, help="amostra determinística por família")
        sp.add_argument("--limit", type=int, default=0)
        sp.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run")
    add_selection(r)
    r.add_argument("--backend", choices=("llamacpp", "hf"), required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--server-bin", default=None)
    r.add_argument("--url", default=None)
    r.add_argument("--revision", default=None)
    r.add_argument("--dtype", default=None)
    r.add_argument("--native", action="store_true", help="HF: implementação nativa do transformers")
    r.add_argument("--adapter", default=None, help="HF: pasta de um adaptador LoRA (PEFT), aplicado sem fundir")
    r.add_argument("--ctx", type=int, default=8192)
    r.add_argument("--template", default=str(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))
    r.add_argument("--orders", default="canonical,reversed")
    r.add_argument("--no-generate", action="store_true")
    r.add_argument("--sem-ponteiro", action="store_true", help="HF: ignora a pointer head do adaptador e lê pelas letras")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--abster", action="store_true", help="H5: acrescenta a opção 'Não sei' no fim de cada caso")
    r.add_argument("--exemplos-contexto", type=int, default=0,
                   help="H8: nº de casos com gabarito da mesma família no prompt (só casos reservados)")
    r.add_argument("--max-prompt-tokens", type=int, default=6000, help="H8: teto do prompt; corta exemplos se passar")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report")
    rp.add_argument("run")
    rp.set_defaults(func=cmd_report)

    b = sub.add_parser("baselines")
    b.add_argument("--source", required=True)
    b.add_argument("--split", default="dev")
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_baselines)

    k = sub.add_parser("calibrar", help="ajusta T e limiar no calib e aplica fixos no alvo")
    k.add_argument("--calib", required=True)
    k.add_argument("--alvo", required=True)
    k.add_argument("--risco", type=float, default=0.10, help="risco-alvo entre as decisões aceitas")
    k.set_defaults(func=cmd_calibrar)

    c = sub.add_parser("comparar")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_comparar)

    h = sub.add_parser("h2", help="critério da H2: novo x base nas famílias reservadas (peso igual)")
    h.add_argument("--base", required=True, help="pastas de run separadas por vírgula (ex.: E1 do Lumi)")
    h.add_argument("--novo", required=True, help="pastas de run separadas por vírgula (ex.: E3 com LoRA)")
    h.add_argument("--out", default=None)
    h.set_defaults(func=cmd_h2)

    h3 = sub.add_parser("h3", help="critério do E4 (H3): pointer head x A1 (qualidade e estabilidade)")
    h3.add_argument("--base", required=True, help="pastas do A1 (E3) separadas por vírgula")
    h3.add_argument("--novo", required=True, help="pastas do B (E4) separadas por vírgula")
    h3.add_argument("--readout-novo", default="ponteiro", help="leitura exigida no braço novo (ponteiro | residual)")
    h3.add_argument("--out", default=None)
    h3.set_defaults(func=cmd_h3)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
