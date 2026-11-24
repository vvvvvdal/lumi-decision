"""Métricas sobre linhas de predição (uma linha por caso × ordem de opções).

Campos usados de cada linha: case_id, group_id, family, order, gold, probs{opção: p},
pred, e opcionalmente gold_dist, gen_pred.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

EPS = 1e-12


def accuracy(rows) -> float:
    return float(np.mean([r["pred"] == r["gold"] for r in rows])) if rows else float("nan")


def balanced_accuracy(rows) -> float:
    by_gold = defaultdict(list)
    for r in rows:
        by_gold[r["gold"]].append(r["pred"] == r["gold"])
    return float(np.mean([np.mean(v) for v in by_gold.values()])) if by_gold else float("nan")


def nll(rows) -> float:
    return float(np.mean([-math.log(max(r["probs"].get(r["gold"], 0.0), EPS)) for r in rows]))


def brier(rows) -> float:
    """Brier multiclasse: soma sobre as opções de (p - y)^2, média sobre itens (intervalo 0..2)."""
    return float(np.mean([sum((p - (k == r["gold"])) ** 2 for k, p in r["probs"].items()) for r in rows]))


def ece(rows, bins: int = 15) -> float:
    """ECE sobre a probabilidade máxima, bins de largura igual."""
    conf = np.array([max(r["probs"].values()) for r in rows])
    hit = np.array([r["pred"] == r["gold"] for r in rows], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def soft_metrics(rows) -> dict:
    """Aderência à distribuição de referência (só onde há gold_dist)."""
    rows = [r for r in rows if r.get("gold_dist")]
    if not rows:
        return {}
    ce = [-sum(q * math.log(max(r["probs"].get(k, 0.0), EPS)) for k, q in r["gold_dist"].items()) for r in rows]
    tv = [0.5 * sum(abs(r["probs"].get(k, 0.0) - r["gold_dist"].get(k, 0.0))
                    for k in set(r["probs"]) | set(r["gold_dist"])) for r in rows]
    return {"n_soft": len(rows), "cross_entropy_vs_ref": float(np.mean(ce)), "tv_vs_ref": float(np.mean(tv))}


def risk_coverage(rows, risk_targets=(0.05, 0.10, 0.20)) -> dict:
    """Curva risco × cobertura aceitando os itens de maior confiança primeiro.

    `risco_aleatorio` é o erro esperado ao aceitar a mesma fração ao acaso (= erro global).
    """
    conf = np.array([max(r["probs"].values()) for r in rows])
    err = np.array([r["pred"] != r["gold"] for r in rows], dtype=float)
    idx = np.argsort(-conf, kind="stable")
    cum_risk = np.cumsum(err[idx]) / np.arange(1, len(rows) + 1)
    cov = np.arange(1, len(rows) + 1) / len(rows)
    out = {"risco_global": float(err.mean()), "aurc": float(cum_risk.mean())}
    for c in (0.5, 0.8):
        out[f"risco@cob{int(c * 100)}"] = float(cum_risk[max(0, math.ceil(c * len(rows)) - 1)])
    for t in risk_targets:
        ok = np.nonzero(cum_risk <= t)[0]
        out[f"cobertura@risco{int(t * 100)}"] = float(cov[ok.max()]) if ok.size else 0.0
    return out


def summarize(rows) -> dict:
    if not rows:
        return {}
    out = {
        "n": len(rows),
        "acuracia": accuracy(rows),
        "acuracia_balanceada": balanced_accuracy(rows),
        "nll": nll(rows),
        "brier": brier(rows),
        "ece": ece(rows),
        "acaso": float(np.mean([1 / len(r["probs"]) for r in rows])),
        **risk_coverage(rows),
        **soft_metrics(rows),
    }
    gen = [r for r in rows if "gen_pred" in r]
    if gen:
        out["geracao_acuracia"] = float(np.mean([r["gen_pred"] == r["gold"] for r in gen]))
        out["geracao_invalida"] = float(np.mean([r["gen_pred"] is None for r in gen]))
        out["geracao_concorda_readout"] = float(np.mean([r["gen_pred"] == r["pred"] for r in gen]))
    return out


def permutation_stats(rows) -> dict:
    """Sensibilidade à ordem: compara as ordens de cada caso (exige ≥2 ordens por caso)."""
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["case_id"]][r["order"]] = r
    orders = sorted({r["order"] for r in rows})
    cases = [v for v in by_case.values() if len(v) == len(orders)]
    if len(orders) < 2 or not cases:
        return {}
    ref = "canonical" if "canonical" in orders else orders[0]
    flips = [len({v[o]["pred"] for o in orders}) > 1 for v in cases]
    ref_ok = [v for v in cases if v[ref]["pred"] == v[ref]["gold"]]
    keep = [all(v[o]["pred"] == v[o]["gold"] for o in orders) for v in ref_ok]
    avg_hits = []
    for v in cases:
        keys = v[ref]["probs"].keys()
        mean = {k: np.mean([v[o]["probs"][k] for o in orders]) for k in keys}
        avg_hits.append(max(mean, key=mean.get) == v[ref]["gold"])
    return {
        "ordens": orders,
        "n_casos": len(cases),
        "decisao_muda_entre_ordens": float(np.mean(flips)),
        "retencao_dos_acertos": float(np.mean(keep)) if keep else float("nan"),
        "acuracia_por_ordem": {o: accuracy([v[o] for v in cases]) for o in orders},
        "acuracia_media_das_ordens": float(np.mean(avg_hits)),
        "acerta_em_todas_as_ordens": float(np.mean([all(v[o]["pred"] == v[o]["gold"] for o in orders) for v in cases])),
    }


def _paired(rows_a, rows_b, strict: bool):
    """Indexa por (caso, ordem). Sempre recusa chaves repetidas. strict: exige o mesmo conjunto de pares e,
    em cada par, o mesmo gabarito, a mesma ordem apresentada e o mesmo prompt (quando os campos existem)."""
    a, b = {}, {}
    for name, rows, d in (("A", rows_a, a), ("B", rows_b, b)):
        for r in rows:
            k = (r["case_id"], r["order"])
            if k in d:
                raise ValueError(f"par (caso, ordem) repetido em {name}: {k}")
            d[k] = r
    if strict:
        if set(a) != set(b):
            only_a, only_b = set(a) - set(b), set(b) - set(a)
            raise ValueError(f"pares (caso, ordem) não correspondem: {len(only_a)} só em A, {len(only_b)} só em B; "
                             f"ex.: {sorted(only_a)[:2]} {sorted(only_b)[:2]}")
        for k in a:
            for f in ("gold", "presented", "prompt_sha1", "family", "group_id"):
                if f in a[k] and f in b[k] and a[k][f] != b[k][f]:
                    raise ValueError(f"par {k} com '{f}' diferente: {a[k][f]!r} x {b[k][f]!r}")
    return a, b, sorted(set(a) & set(b))


def order_mean_accuracy(rows) -> dict:
    """Acurácia de cada ordem e a média delas (métrica principal). Exige todas as ordens em todos os casos."""
    orders = sorted({r["order"] for r in rows})
    by_case = defaultdict(set)
    for r in rows:
        by_case[r["case_id"]].add(r["order"])
    incompletos = sum(len(v) != len(orders) for v in by_case.values())
    if incompletos:
        raise ValueError(f"{incompletos} casos sem todas as ordens {orders}")
    per = {o: accuracy([r for r in rows if r["order"] == o]) for o in orders}
    return {"por_ordem": per, "media": float(np.mean(list(per.values())))}


def paired_bootstrap(rows_a, rows_b, n_boot: int = 2000, seed: int = 0, strict: bool = True) -> dict:
    """IC 95% de acc(B) - acc(A) nos mesmos (caso, ordem), reamostrando grupos inteiros.

    strict: os dois lados precisam ter exatamente os mesmos pares (caso, ordem); senão erro.
    """
    a, b, keys = _paired(rows_a, rows_b, strict)
    groups = defaultdict(list)
    for k in keys:
        groups[a[k]["group_id"]].append((a[k]["pred"] == a[k]["gold"], b[k]["pred"] == b[k]["gold"]))
    g = list(groups.values())
    sums = np.array([[sum(x[1] - x[0] for x in grp), len(grp)] for grp in g], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        s = sums[rng.integers(0, len(g), len(g))]
        deltas.append(s[:, 0].sum() / s[:, 1].sum())
    point = sums[:, 0].sum() / sums[:, 1].sum()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n_itens": len(keys), "n_grupos": len(g), "delta": float(point), "ic95": [float(lo), float(hi)]}


def family_mean_bootstrap(rows_a, rows_b, n_boot: int = 2000, seed: int = 0, require_families=None) -> dict:
    """Critério da H2: média, com PESO IGUAL por família, de acc(B) - acc(A) sobre todas as ordens.

    Pareado por (caso, ordem), com correspondência exata exigida. O IC reamostra grupos de texto/estado
    dentro de cada família (estratificado): famílias grandes não dominam e perguntas do mesmo estado
    andam juntas.
    """
    a, b, keys = _paired(rows_a, rows_b, strict=True)
    fam = defaultdict(lambda: defaultdict(list))
    for k in keys:
        fam[a[k]["family"]][a[k]["group_id"]].append(
            float(b[k]["pred"] == b[k]["gold"]) - float(a[k]["pred"] == a[k]["gold"]))
    if require_families is not None and set(fam) != set(require_families):
        raise ValueError(f"famílias {sorted(fam)} != exigidas {sorted(require_families)}")
    arrays = {f: np.array([[sum(v), len(v)] for v in g.values()]) for f, g in fam.items()}
    per = {f: float(x[:, 0].sum() / x[:, 1].sum()) for f, x in arrays.items()}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ds = []
        for x in arrays.values():
            s = x[rng.integers(0, len(x), len(x))]
            ds.append(s[:, 0].sum() / s[:, 1].sum())
        boots.append(np.mean(ds))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta_media_familias": float(np.mean(list(per.values()))), "ic95": [float(lo), float(hi)],
            "por_familia": per, "grupos_por_familia": {f: len(x) for f, x in arrays.items()}, "n_itens": len(keys)}


def _flips(rows) -> dict:
    """{case_id: (família, grupo, 1 se a escolha muda entre as ordens)}; exige as mesmas ordens em todo caso."""
    orders = sorted({r["order"] for r in rows})
    by = defaultdict(dict)
    for r in rows:
        if r["order"] in by[r["case_id"]]:
            raise ValueError(f"par (caso, ordem) repetido: {(r['case_id'], r['order'])}")
        by[r["case_id"]][r["order"]] = r
    if len(orders) < 2 or any(len(v) != len(orders) for v in by.values()):
        raise ValueError(f"todo caso precisa de todas as ordens {orders} (mínimo 2)")
    return {c: (v[orders[0]]["family"], v[orders[0]]["group_id"], float(len({x["pred"] for x in v.values()}) > 1))
            for c, v in by.items()}


def family_mean_flip_bootstrap(rows_a, rows_b, n_boot: int = 2000, seed: int = 0, require_families=None) -> dict:
    """Estabilidade (E4/H3): média com peso igual por família de flip(B) - flip(A), em que flip = a escolha muda
    entre as ordens do caso. Pareado por caso; IC reamostrando os MESMOS grupos nos dois modelos, por família."""
    fa, fb = _flips(rows_a), _flips(rows_b)
    if set(fa) != set(fb):
        raise ValueError(f"casos não correspondem: {len(set(fa) - set(fb))} só em A, {len(set(fb) - set(fa))} só em B")
    fam = defaultdict(lambda: defaultdict(list))
    for c, (f, g, x) in fa.items():
        if fb[c][:2] != (f, g):
            raise ValueError(f"caso {c} com família/grupo diferente")
        fam[f][g].append(fb[c][2] - x)
    if require_families is not None and set(fam) != set(require_families):
        raise ValueError(f"famílias {sorted(fam)} != exigidas {sorted(require_families)}")
    arrays = {f: np.array([[sum(v), len(v)] for v in g.values()]) for f, g in fam.items()}
    per = {f: float(x[:, 0].sum() / x[:, 1].sum()) for f, x in arrays.items()}
    rate = {}
    for name, d in (("base", fa), ("novo", fb)):
        acc = defaultdict(list)
        for f, _, x in d.values():
            acc[f].append(x)
        rate[name] = {f: float(np.mean(v)) for f, v in acc.items()}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ds = []
        for x in arrays.values():
            s = x[rng.integers(0, len(x), len(x))]
            ds.append(s[:, 0].sum() / s[:, 1].sum())
        boots.append(np.mean(ds))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta_media_familias": float(np.mean(list(per.values()))), "ic95": [float(lo), float(hi)],
            "por_familia": per, "taxa_base": rate["base"], "taxa_novo": rate["novo"],
            "grupos_por_familia": {f: len(x) for f, x in arrays.items()}, "n_casos": len(fa)}


def require_readout(rows, expected: str, side: str) -> None:
    """Recusa um lado da comparação se alguma predição foi lida de outro jeito (campo ausente = letras)."""
    got = Counter(r.get("readout", "letras") for r in rows)
    if not rows or set(got) != {expected}:
        raise ValueError(f"{side}: esperado readout '{expected}' em todas as predições, veio {dict(got)}")


def h3_verdict(quality_ci, flip_ci, margin: float = 0.02) -> dict:
    """Regra registrada do E4. Qualidade (principal): direção e equivalência prática são afirmações SEPARADAS
    (um IC [+0,3; +1,7] pp é ganho detectável E fica dentro de ±2 pp). Estabilidade (secundária): melhora só
    se o IC da diferença de mudanças ficar inteiro abaixo de zero E o limite inferior da qualidade > −margem."""
    lo, hi = quality_ci
    direcao = "B melhor" if lo > 0 else ("B pior" if hi < 0 else "sem direção detectada")
    equivalente = -margin < lo and hi < margin
    if lo > 0 or hi < 0:
        qualidade = "melhora" if lo > 0 else "piora"
    else:
        qualidade = "empate" if equivalente else "inconclusivo"
    estab = "melhora" if flip_ci[1] < 0 and lo > -margin else ("piora" if flip_ci[0] > 0 else "sem melhora demonstrada")
    return {"qualidade": qualidade, "direcao": direcao,
            "equivalencia": "equivalentes (IC inteiro dentro de ±2 pp)" if equivalente else "equivalência não demonstrada",
            "estabilidade": estab, "margem": margin}


# ---------------------------------------------------------------- calibração (ajuste em calib, uso fixo em dev/test)

def auroc(conf, acerto) -> float:
    """P(a confiança de um acerto > a de um erro); empate vale meio (Mann–Whitney, pelos postos médios).
    nan sem acertos ou sem erros."""
    c = np.asarray(conf, dtype=float)
    y = np.asarray(acerto, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, cnt = np.unique(c, return_inverse=True, return_counts=True)
    postos = (np.cumsum(cnt) - cnt + (cnt + 1) / 2)[inv]  # posto médio (1 = menor) de cada valor
    return float((postos[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def with_temperature(rows, T: float) -> list[dict]:
    """Reescala as probabilidades: p_k ∝ p_k^(1/T). Não muda o argmax."""
    out = []
    for r in rows:
        logp = {k: math.log(max(p, EPS)) / T for k, p in r["probs"].items()}
        m = max(logp.values())
        w = {k: math.exp(v - m) for k, v in logp.items()}
        z = sum(w.values())
        out.append({**r, "probs": {k: v / z for k, v in w.items()}})
    return out


def fit_temperature(rows) -> float:
    """Temperatura que minimiza a NLL (busca em grade log-espaçada e refinamento local)."""
    grid = np.exp(np.linspace(math.log(0.2), math.log(10), 61))
    best = min(grid, key=lambda t: nll(with_temperature(rows, t)))
    fine = np.linspace(best / 1.1, best * 1.1, 21)
    return float(min(fine, key=lambda t: nll(with_temperature(rows, t))))


def threshold_for_risk(rows, target: float, min_accept: int = 20) -> float | None:
    """Menor limiar de confiança cujo risco entre os aceitos (no conjunto dado) fica <= alvo."""
    conf = np.array([max(r["probs"].values()) for r in rows])
    err = np.array([r["pred"] != r["gold"] for r in rows], dtype=float)
    idx = np.argsort(-conf, kind="stable")
    cum_risk = np.cumsum(err[idx]) / np.arange(1, len(rows) + 1)
    sc = conf[idx]
    # Só posições onde a confiança muda são cortes realizáveis: com um limiar fixo, empates entram juntos.
    cuts = [i for i in range(min_accept - 1, len(rows)) if (i == len(rows) - 1 or sc[i] > sc[i + 1])]
    ok = [i for i in cuts if cum_risk[i] <= target]
    return float(sc[max(ok)]) if ok else None


def selective_at(rows, tau: float | None) -> dict:
    """Aplica um limiar FIXO: cobertura, risco entre os aceitos e o risco de aceitar a mesma fração ao acaso."""
    err = np.array([r["pred"] != r["gold"] for r in rows], dtype=float)
    if tau is None or not len(rows):
        return {"limiar": tau, "cobertura": 0.0, "risco_aceitos": None, "risco_aleatorio": float(err.mean()) if len(rows) else None}
    acc = np.array([max(r["probs"].values()) >= tau for r in rows])
    return {"limiar": tau, "cobertura": float(acc.mean()),
            "risco_aceitos": float(err[acc].mean()) if acc.any() else None,
            "risco_aleatorio": float(err.mean()), "n": len(rows)}
