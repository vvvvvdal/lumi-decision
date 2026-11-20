"""H8: exemplos da família no prompt melhoram o Lumi nas famílias reservadas?

Compara cada K com K = 0 (mesmo fluxo, mesmo adaptador) nas reservadas do `dev`:
- acurácia, ECE, Brier e NLL com IC 95% por bootstrap de grupos (`comparar_h4.comparar`);
- cobertura com risco de 10%: T e limiar ajustados por família em k grupos rotulados do próprio `dev`, medida
  nos grupos restantes (as funções da H7), e o oráculo (ajuste na família inteira).

Pastas: `runs/h8_<nome>_k<K>_<fonte>_dev[<suf>]`.

Uso (na raiz do repo, sem GPU):
    python scripts/analise_h8.py --nome h4_log --ks 0,3,8 --out analises/h8_contexto.md
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


c4, h7 = _mod("comparar_h4"), _mod("h7_poucos_rotulos")


def cobertura(rows: list[dict], k: int, reps: int, risco: float, rng) -> tuple[dict, dict]:
    """(T+limiar por família com k grupos rotulados do próprio conjunto, oráculo por família)."""
    fams = sorted({r["family"] for r in rows})
    por_f = {f: [r for r in rows if r["family"] == f] for f in fams}
    ora = []
    for f in fams:
        T, tau = h7.ajustar(por_f[f], risco, 5)
        ora += h7.aplicar(por_f[f], T, tau)
    medidas = []
    for _ in range(reps):
        linhas = []
        for f in fams:
            pool = h7.por_grupo(por_f[f])
            esc = set(rng.choice(sorted(pool), size=min(k, len(pool) - 5), replace=False).tolist())
            T, tau = h7.ajustar([r for g in esc for r in pool[g]], risco, 5)
            linhas += h7.aplicar([r for r in por_f[f] if r["group_id"] not in esc], T, tau)
        medidas.append(h7.medir(linhas))
    media = {m: float(np.nanmean([x[m] for x in medidas])) for m in medidas[0]}
    return media, h7.medir(ora)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--nome", default="h4_log")
    p.add_argument("--ks", default="0,3,8")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--suf", default="", help="ex.: _curto20")
    p.add_argument("--k-rotulos", type=int, default=25)
    p.add_argument("--repeticoes", type=int, default=100)
    p.add_argument("--risco", type=float, default=0.10)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    runs, ks = Path(a.runs), [int(k) for k in a.ks.split(",")]
    rng = np.random.default_rng(0)
    t1 = ["| fonte | K | métrica | K=0 | K | Δ | IC 95% | leitura |", "|---|---:|---|---:|---:|---:|---|---|"]
    t2 = [f"| fonte | K | acc | cobertura (T+limiar, {a.k_rotulos} rótulos) | risco | cobertura oráculo | risco oráculo |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    res = {}
    for fonte in a.fontes.split(","):
        rows = {}
        for k in ks:
            d = runs / f"h8_{a.nome}_k{k}_{fonte}_dev{a.suf}"
            if (d / "predicoes.jsonl").exists():
                rows[k] = [dict(r, _semente=0) for r in c4.carregar(d) if r["held_out"]]
            else:
                print(f">> falta {d}; pulei")
        for k, rs in rows.items():
            cob, ora = cobertura(rs, a.k_rotulos, a.repeticoes, a.risco, rng)
            acc = float(np.mean([r["pred"] == r["gold"] for r in rs]))
            ntok = int(np.median([r["prompt_tokens"] for r in rs]))
            kus = float(np.mean([len(r.get("exemplos_ids", [])) for r in rs]))
            res[f"{fonte}|k{k}"] = {"acc": acc, "cobertura": cob, "oraculo": ora, "tokens_mediana": ntok,
                                    "k_usado_medio": kus}
            t2.append(f"| {fonte} | {k} (usado {kus:.1f}; {ntok} tokens) | {acc:.3f} | {cob['cobertura']:.3f} | "
                      f"{cob['risco_aceitos']:.3f} | {ora['cobertura']:.3f} | {ora['risco_aceitos']:.3f} |")
            if k == 0 or 0 not in rows:
                continue
            c = c4.comparar(rows[0], rs, a.n_boot, 0)
            res[f"{fonte}|k{k}-k0"] = c
            for m in c4.METRICAS:
                d = c[m]
                t1.append(f"| {fonte} | {k} | {m} | {d['base']:.3f} | {d['novo']:.3f} | {d['delta']:+.3f} | "
                          f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}] | {c4.leitura(m, d)} |")
    txt = "\n".join(t1) + "\n\n" + "\n".join(t2) + (
        f"\n\nReservadas do dev. Δ = K − (K=0). Cobertura = fração respondida com risco-alvo {a.risco:.0%}; "
        "oráculo = T e limiar ajustados na família inteira (otimista).")
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
