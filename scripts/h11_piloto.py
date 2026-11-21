"""H11 piloto: o Lumi raciocina antes de decidir? Compara a leitura das letras DEPOIS do raciocínio
(`lumi-eval run --raciocinio N`, pastas `runs/<piloto>_<fonte>_dev`) com a leitura de sempre do mesmo
adaptador (`runs/<base>_<fonte>_dev`, ordem canônica, mesmos `case_id`).

Relata, por fonte e recorte (vistas / reservadas):
- acurácia, ECE e NLL sem e com raciocínio, e a fração de decisões que mudou;
- tamanho do raciocínio, fração que bateu no limite sem fechar o `</think>`, fração degenerada (repetição)
  e o tempo mediano;
- o custo projetado de um GRPO (`--rollouts` raciocínios) com esse tempo.
E grava alguns raciocínios de exemplo. Critério em `analises/plano_h11_piloto.md`.

Uso (na raiz do repo, sem GPU):
    python scripts/h11_piloto.py --base h4_log --piloto h11p --out analises/h11_piloto.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumi_decision import metrics  # noqa: E402


def carregar(pasta: Path) -> list[dict]:
    with (pasta / "predicoes.jsonl").open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fracao_repetida(texto: str, n: int = 4) -> float:
    """Fração de n-gramas de palavras que repetem um n-grama anterior (0 = nada repetido)."""
    w = texto.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def degenerado(r: dict, limiar: float = 0.5) -> bool:
    return not r["raciocinio"].strip() or fracao_repetida(r["raciocinio"]) > limiar


def recorte(base: list[dict], pil: list[dict]) -> dict:
    """Mesmos casos nos dois lados (pares por case_id, ordem canônica)."""
    b = {r["case_id"]: r for r in base if r["order"] == "canonical"}
    pares = [(b[r["case_id"]], r) for r in pil if r["case_id"] in b]
    if not pares:
        return {"n": 0}
    sem, com = [x for x, _ in pares], [y for _, y in pares]
    return {
        "n": len(pares),
        "acc_sem": metrics.accuracy(sem), "acc_com": metrics.accuracy(com),
        "ece_sem": metrics.ece(sem), "ece_com": metrics.ece(com),
        "nll_sem": metrics.nll(sem), "nll_com": metrics.nll(com),
        "mudou": float(np.mean([x["pred"] != y["pred"] for x, y in pares])),
        "tokens_med": float(np.median([y["rac_tokens"] for y in com])),
        "limite": float(np.mean([y["rac_parada"] == "limit" for y in com])),
        "degenerado": float(np.mean([degenerado(y) for y in com])),
        "ms_med": float(np.median([y["rac_ms"] for y in com])),
        # a letra que o próprio texto do raciocínio diz ("answer N"); sem letra conta como erro
        "acc_texto": float(np.mean([y.get("rac_letra_texto") == y["gold"] for y in com])),
        "texto_disse": float(np.mean([y.get("rac_letra_texto") is not None for y in com])),
        "sufixos": por_sufixo(com),
    }


def por_sufixo(com: list[dict]) -> dict:
    """Acurácia e NLL da leitura das letras com cada separador depois do </think> (cru, \\n, \\n\\n)."""
    nomes = sorted({k for y in com for k in y.get("rac_probs_sufixo", {})})
    out = {}
    for nome in nomes:
        rows = [{**y, "probs": y["rac_probs_sufixo"][nome],
                 "pred": max(y["rac_probs_sufixo"][nome], key=y["rac_probs_sufixo"][nome].get)}
                for y in com if nome in y.get("rac_probs_sufixo", {})]
        out[nome] = {"n": len(rows), "acc": metrics.accuracy(rows), "nll": metrics.nll(rows)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--base", default="h4_log")
    p.add_argument("--piloto", default="h11p")
    p.add_argument("--fontes", default="felhen,telepatia")
    p.add_argument("--rollouts", type=int, default=2400, help="GRPO projetado: 300 casos x G=4 x 2 épocas")
    p.add_argument("--exemplos", type=int, default=5)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    runs = Path(a.runs)
    linhas = ["| fonte | recorte | n | acc sem → com | ECE sem → com | NLL sem → com | decisão mudou | tokens (mediana) "
              "| bateu no limite | degenerado | s por raciocínio |",
              "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|"]
    leit = ["| fonte | recorte | n | acc sem raciocínio | acc pela letra do texto (texto disse) "
            "| colada no </think>: acc / NLL | + \\n: acc / NLL | + \\n\\n: acc / NLL |",
            "|---|---|---:|---:|---|---|---|---|"]
    todos, exemplos, bruto = [], [], {}
    for fonte in a.fontes.split(","):
        pp, pb = runs / f"{a.piloto}_{fonte}_dev", runs / f"{a.base}_{fonte}_dev"
        if not ((pp / "predicoes.jsonl").exists() and (pb / "predicoes.jsonl").exists()):
            print(f">> {fonte}: faltam {pp} ou {pb}; pulei")
            continue
        pil = [r for r in carregar(pp) if "raciocinio" in r]
        base = carregar(pb)
        todos += pil
        for nome, filtro in (("vistas", False), ("reservadas", True)):
            m = recorte(base, [r for r in pil if r["held_out"] == filtro])
            bruto[f"{fonte}|{nome}"] = m
            if not m["n"]:
                continue
            linhas.append(
                f"| {fonte} | {nome} | {m['n']} | {m['acc_sem']:.3f} → {m['acc_com']:.3f} | "
                f"{m['ece_sem']:.3f} → {m['ece_com']:.3f} | {m['nll_sem']:.3f} → {m['nll_com']:.3f} | "
                f"{m['mudou']:.0%} | {m['tokens_med']:.0f} | {m['limite']:.0%} | {m['degenerado']:.0%} | "
                f"{m['ms_med'] / 1000:.1f} |")
            suf = [f"{m['sufixos'][k]['acc']:.3f} / {m['sufixos'][k]['nll']:.3f}" if k in m["sufixos"] else "—"
                   for k in ("cru", "nl", "nl2")]
            leit.append(f"| {fonte} | {nome} | {m['n']} | {m['acc_sem']:.3f} | {m['acc_texto']:.3f} "
                        f"({m['texto_disse']:.0%}) | " + " | ".join(suf) + " |")
        for r in pil[: a.exemplos]:
            texto = r["raciocinio"].strip()
            exemplos.append(f"**{fonte} · {r['family']} · {'reservada' if r['held_out'] else 'vista'} · "
                            f"gabarito {r['gold']} · leu {r['pred']} · texto disse {r.get('rac_letra_texto')} · "
                            f"{r['rac_tokens']} tokens, {r['rac_parada']} · depois do </think>: "
                            f"`{json.dumps(r.get('rac_depois_think'), ensure_ascii=False)}`**\n\n"
                            f"> {(texto[:700] + ' […]') if len(texto) > 700 else texto or '(vazio)'}".replace("\n", "\n> "))
    if not todos:
        raise SystemExit("nenhuma predição com raciocínio encontrada")
    ms = float(np.median([r["rac_ms"] for r in todos]))
    resumo = {
        "n": len(todos),
        "fechou_think": float(np.mean([r["rac_parada"] == "fim" for r in todos])),
        "degenerado": float(np.mean([degenerado(r) for r in todos])),
        "s_mediano": ms / 1000,
        "horas_grpo": a.rollouts * ms / 3.6e6,
    }
    bruto["geral"] = resumo
    texto = ("\n".join(linhas)
             + f"\n\nGeral: {resumo['n']} raciocínios; fecharam o </think> antes do limite: {resumo['fechou_think']:.0%}; "
             f"degenerados: {resumo['degenerado']:.0%}; tempo mediano {resumo['s_mediano']:.1f} s; "
             f"GRPO com {a.rollouts} raciocínios ≈ {resumo['horas_grpo']:.1f} h só de geração (lote 1).\n"
             "\nSem = leitura de sempre (`<think></think>` vazio); com = leitura depois do raciocínio. Mesmos casos, "
             "ordem canônica. Poucos casos por recorte: o ECE aqui é só indicativo.\n"
             "\n## Onde ler a decisão depois do raciocínio\n\n" + "\n".join(leit) + "\n"
             "\nA tabela de cima usa a leitura principal (`+ \\n\\n`). Letra do texto = a última letra que o "
             "raciocínio diz ser a resposta (\"answer N\"); quando o texto não diz, conta como erro.\n"
             "\n## Exemplos\n\n" + "\n\n".join(exemplos) + "\n")
    print(texto)
    if a.out:
        Path(a.out).write_text(texto, encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(bruto, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
