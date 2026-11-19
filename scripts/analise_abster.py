"""H5: o braço (e) (abstenção aprendida por RL) contra a base (h4_log + limiar) NA MESMA COBERTURA.

Braço (e): em cada linha, abstém se probs["__abster__"] > 0,5; senão responde a opção mais provável entre as
normais (a mesma regra do treino). Base: responde os casos de maior confiança, na mesma fração que o braço (e)
respondeu naquele grupo. Compara o risco entre os respondidos, com IC 95% reamostrando grupos de casos
(`group_id`, juntando sementes e ordens, como `scripts/comparar_h4.py`). Também mede o ganho médio com o custo
lambda (+1 / -lambda / 0) e o ECE nas opções normais.

Pastas: semente 0 = `runs/<nome>_<fonte>_dev`; semente s > 0 = `runs/<nome>_s<s>_<fonte>_dev`.

Uso (na raiz do repo, via `srun ... lumi_exec python ...`):
    python scripts/analise_abster.py --novo h5_abster_l3 --custo 3 --out analises/h5_l3.md
    python scripts/analise_abster.py --novo h5_abster_l3 --custo 3 --res2   # + regra_beneficio (reservadas v2)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ABST = "__abster__"


def pasta(runs: Path, nome: str, semente: int, fonte: str) -> Path:
    return runs / f"{nome if semente == 0 else f'{nome}_s{semente}'}_{fonte}_dev"


def carregar(p: Path) -> list[dict]:
    return [json.loads(x) for x in (p / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def decisao(r: dict) -> tuple[bool, bool, float]:
    """(respondeu, acertou, confiança na resposta), com as opções normais renormalizadas."""
    pa = r["probs"].get(ABST, 0.0)
    norm = {k: v for k, v in r["probs"].items() if k != ABST}
    z = sum(norm.values()) or 1.0
    k = max(norm, key=norm.get)
    return pa <= 0.5, k == r["gold"], norm[k] / z


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 15) -> float:
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    n = np.bincount(idx, minlength=bins)
    m = n > 0
    return float(np.sum(np.abs(np.bincount(idx, hit, bins)[m] - np.bincount(idx, conf, bins)[m])) / len(conf))


def vetores(rows):
    d = [decisao(r) for r in rows]
    return (np.array([x[0] for x in d], dtype=bool), np.array([x[1] for x in d], dtype=float),
            np.array([x[2] for x in d]))


def medir(e, b, sel, lam):
    """e = (respondeu, acertou, conf) do braço (e); b = (acertou, conf) da base; sel = índices da amostra."""
    resp, hit_e, conf_e = e[0][sel], e[1][sel], e[2][sel]
    hit_b, conf_b = b[0][sel], b[1][sel]
    n = len(sel)
    cob = resp.mean()
    k = int(round(cob * n))
    risco_e = 1 - hit_e[resp].mean() if resp.any() else np.nan
    ordem = np.argsort(-conf_b, kind="stable")[:k]
    risco_b = 1 - hit_b[ordem].mean() if k else np.nan
    ganho_e = np.where(resp, np.where(hit_e > 0, 1.0, -lam), 0.0).mean()
    aceita_b = np.zeros(n, dtype=bool)
    aceita_b[ordem] = True
    ganho_b = np.where(aceita_b, np.where(hit_b > 0, 1.0, -lam), 0.0).mean()
    return {"cobertura": cob, "risco_novo": risco_e, "risco_base": risco_b, "delta_risco": risco_e - risco_b,
            "ganho_novo": ganho_e, "ganho_base": ganho_b, "ece_novo": ece(conf_e, hit_e), "ece_base": ece(conf_b, hit_b)}


def comparar(rows_e, rows_b, lam, n_boot, seed):
    ke = {(r["_s"], r["case_id"], r["order"]): r for r in rows_e}
    kb = {(r["_s"], r["case_id"], r["order"]): r for r in rows_b}
    chaves = sorted(set(ke) & set(kb))
    if len(chaves) != len(ke) or len(chaves) != len(kb):
        raise ValueError(f"braços não correspondem: {len(ke)} x {len(kb)}, {len(chaves)} em comum")
    E = vetores([ke[c] for c in chaves])
    B = vetores([kb[c] for c in chaves])[1:]
    grupos = defaultdict(list)
    for i, c in enumerate(chaves):
        grupos[ke[c]["group_id"]].append(i)
    gl = [np.array(v) for v in grupos.values()]
    tudo = np.arange(len(chaves))
    ponto = medir(E, B, tudo, lam)
    rng = np.random.default_rng(seed)
    boots = defaultdict(list)
    for _ in range(n_boot):
        sel = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
        m = medir(E, B, sel, lam)
        for key in ("delta_risco", "ganho_novo", "ganho_base", "ece_novo", "ece_base"):
            boots[key].append(m[key])
    ic = {key: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for key, v in boots.items()}
    ic["delta_ganho"] = [float(x) for x in np.nanpercentile(np.array(boots["ganho_novo"]) - np.array(boots["ganho_base"]), [2.5, 97.5])]
    ic["delta_ece"] = [float(x) for x in np.nanpercentile(np.array(boots["ece_novo"]) - np.array(boots["ece_base"]), [2.5, 97.5])]
    return {"n_linhas": len(chaves), "n_grupos": len(gl), **{k: float(v) for k, v in ponto.items()}, "ic95": ic}


def res2_regra(runs: Path, novo: str, base: str) -> str:
    """Secundária: na regra_beneficio, quando o gabarito é pedir_documento, quantas vezes cada um abstém ou pede
    o documento (lê runs/res2dev_<nome>_r4)."""
    linhas = []
    for nome in (base, novo):
        p = runs / f"res2dev_{nome}_r4"
        if not (p / "predicoes.jsonl").exists():
            linhas.append(f"| {nome} | (sem {p}) | |")
            continue
        rows = [r for r in carregar(p) if r.get("family") == "regra_beneficio" and r["gold"] == "pedir_documento"]
        ok = 0
        for r in rows:
            resp, _, _ = decisao(r)
            norm = {k: v for k, v in r["probs"].items() if k != ABST}
            ok += (not resp) or max(norm, key=norm.get) == "pedir_documento"
        linhas.append(f"| {nome} | {ok}/{len(rows)} | {ok / max(1, len(rows)):.3f} |")
    return "| modelo | abstém ou pede o documento | fração |\n|---|---:|---:|\n" + "\n".join(linhas)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--novo", required=True, help="nome do run do braço (e), ex.: h5_abster_l3")
    p.add_argument("--base", default="h4_log")
    p.add_argument("--custo", type=float, required=True, help="o lambda usado no treino do braço (e)")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--sementes", default="0,1,2")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--res2", action="store_true")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    runs = Path(a.runs)
    linhas = ["| fonte | grupo | sementes | cobertura | risco (e) | risco base, mesma cob. | Δ risco [IC 95%] | "
              "ganho (e) − base [IC 95%] | ΔECE normais [IC 95%] |", "|---|---|---|---:|---:|---:|---|---|---|"]
    res = {}
    for fonte in a.fontes.split(","):
        sem = [s for s in map(int, a.sementes.split(",")) if (pasta(runs, a.novo, s, fonte) / "predicoes.jsonl").exists()
               and (pasta(runs, a.base, s, fonte) / "predicoes.jsonl").exists()]
        if not sem:
            print(f">> {fonte}: nenhuma semente com os dois braços")
            continue
        re_, rb = [], []
        for s in sem:
            for dst, nome in ((re_, a.novo), (rb, a.base)):
                for r in carregar(pasta(runs, nome, s, fonte)):
                    r["_s"] = s
                    dst.append(r)
        for held, g in ((False, "vistas"), (True, "reservadas")):
            c = comparar([r for r in re_ if r["held_out"] == held], [r for r in rb if r["held_out"] == held],
                         a.custo, a.n_boot, a.seed)
            res[f"{fonte}|{g}"] = {"sementes": sem, **c}
            ic = c["ic95"]
            linhas.append(
                f"| {fonte} | {g} | {','.join(map(str, sem))} | {c['cobertura']:.3f} | {c['risco_novo']:.3f} | "
                f"{c['risco_base']:.3f} | {c['delta_risco']:+.3f} [{ic['delta_risco'][0]:+.3f}; {ic['delta_risco'][1]:+.3f}] | "
                f"{c['ganho_novo'] - c['ganho_base']:+.3f} [{ic['delta_ganho'][0]:+.3f}; {ic['delta_ganho'][1]:+.3f}] | "
                f"{c['ece_novo'] - c['ece_base']:+.3f} [{ic['delta_ece'][0]:+.3f}; {ic['delta_ece'][1]:+.3f}] |")
    texto = "\n".join(linhas)
    texto += ("\n\nΔ risco < 0 com o IC inteiro abaixo de zero = o braço (e) erra menos que a base respondendo a mesma "
              "fração de casos. ΔECE: só nas opções normais.")
    if a.res2:
        texto += "\n\n**regra_beneficio, gabarito pedir_documento (reservadas v2):**\n\n" + res2_regra(runs, a.novo, a.base)
    print(texto)
    if a.out:
        Path(a.out).write_text(texto + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
