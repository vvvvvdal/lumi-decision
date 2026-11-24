"""Sabe que não sabe (`scripts/analise_sabe.py`) com predições sintéticas."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("analise_sabe", RAIZ / "scripts" / "analise_sabe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _linhas(rng, fonte, fam, held, n_grupos, split, sinal):
    """Casos de 3 opções; a chance de acertar sobe com a confiança (sinal > 0) dentro da família."""
    out = []
    for g in range(n_grupos):
        gid = f"{fonte}:{fam}:{split}:{g}"
        for order in ("canonical", "reversed"):
            c = float(rng.uniform(0.34, 0.99))
            acerta = rng.random() < 0.5 + sinal * ((c - 0.34) / 0.65 - 0.5)
            gold = "a"
            pred = "a" if acerta else "b"
            resto = (1 - c) / 2
            probs = {"a": resto, "b": resto, "c": resto}
            probs[pred] = c
            out.append({"case_id": gid, "group_id": gid, "family": fam, "held_out": held, "order": order,
                        "source": fonte, "gold": gold, "pred": pred, "probs": probs})
    return out


def _escrever(pasta: Path, linhas):
    pasta.mkdir(parents=True)
    (pasta / "predicoes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in linhas), encoding="utf-8")


@pytest.fixture()
def runs(tmp_path):
    rng = np.random.default_rng(1)
    r = tmp_path / "runs"
    for nm, sinal in (("h4_log", 0.2), ("h9r_k100", 1.0)):
        calib = _linhas(rng, "felhen", "vista", False, 80, "calib", sinal)
        dev = (_linhas(rng, "felhen", "vista", False, 60, "dev", sinal)
               + _linhas(rng, "felhen", "nova", True, 75, "dev", sinal))
        _escrever(r / f"{nm}_felhen_calib", calib)
        _escrever(r / f"{nm}_felhen_dev", dev)
    (r / "h9r_k100").mkdir()
    (r / "h9r_k100" / "adapt_grupos.json").write_text(
        json.dumps([{"familia": "felhen:nova", "grupos": [f"felhen:nova:dev:{g}" for g in range(5)]}]), encoding="utf-8")
    return r


def test_bate_com_o_h7_e_ve_o_sinal(runs, tmp_path):
    m = _mod()
    out = tmp_path / "sabe.md"
    m.main(["--runs", str(runs), "--fontes", "felhen", "--sementes", "0", "--repeticoes", "8", "--n-boot", "50",
            "--out", str(out)])
    res = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(res) == {"felhen|h4_log|reservadas", "felhen|h4_log|vistas",
                        "felhen|h9r_k100|reservadas", "felhen|h9r_k100|vistas"}
    # mesma conta do h7 (uma semente): cobertura e erro dos respondidos iguais à média do T_fam+tau_fam
    h7 = m.h7
    a = type("A", (), {"excluir_tambem": "", "seed": 0, "ks_sorteio": [10, 25, 50], "repeticoes": 8, "risco": 0.10,
                       "min_aceitos": 5, "k": 25})()
    _, conferir = m.reservadas(runs, "h4_log", "h9r_k100", "felhen", [0], a)
    for braco, nome in (("base", "h4_log"), ("novo", "h9r_k100")):
        ponto = res[f"felhen|{nome}|reservadas"]["ponto"]
        assert ponto["cobertura"] == pytest.approx(np.mean([x["cobertura"] for x in conferir[braco]]), nan_ok=True)
        assert ponto["erro_respondidos"] == pytest.approx(np.nanmean([x["risco_aceitos"] for x in conferir[braco]]), nan_ok=True)
    # os grupos do treino curto ficam fora da medida e a medida não usa os 25 rótulos
    assert res["felhen|h9r_k100|reservadas"]["n_linhas"] == (75 - 5 - 25) * 2
    # com sinal forte, o "não sei" erra mais que os respondidos, e o AUROC dentro da família passa de 0,5
    forte = res["felhen|h9r_k100|reservadas"]["ponto"]
    assert forte["auroc_dentro"] > 0.65 and forte["diferenca"] > 0
    for v in res.values():
        lo, hi = v["ic95"]["auroc"]
        assert lo <= v["ponto"]["auroc"] + 0.05 and v["ponto"]["auroc"] - 0.05 <= hi
    assert "Conferência" in out.read_text(encoding="utf-8")


def test_medir_conta_a_mao():
    m = _mod()
    v = {"conf": np.array([0.9, 0.8, 0.4, 0.3]), "hit": np.array([True, False, False, False]),
         "resp": np.array([True, True, False, False]), "fam": np.array(["f"] * 4), "grupo": np.array(list("abcd"))}
    r = m.medir(v)
    assert r["cobertura"] == 0.5 and r["erro_respondidos"] == 0.5 and r["erro_nao_sei"] == 1.0
    assert r["diferenca"] == 0.5 and r["auroc"] == 1.0 and r["auroc_dentro"] == 1.0
    v["resp"] = np.array([True] * 4)
    assert math.isnan(m.medir(v)["erro_nao_sei"])
