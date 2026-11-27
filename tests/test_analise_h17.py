"""H17: análise comparativa (`scripts/analise_h17.py`) com predições sintéticas."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_p = Path(__file__).resolve().parents[1] / "scripts" / "analise_h17.py"
_s = importlib.util.spec_from_file_location("analise_h17", _p)
a17 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(a17)


def _rows(acertos, mudam=0):
    out = []
    for i, ok in enumerate(acertos):
        for order in ("canonical", "reversed"):
            pred = "a" if ok else "b"
            if order == "reversed" and i < mudam:
                pred = "c"
            out.append({"case_id": f"c{i}", "family": "f", "order": order, "gold": "a", "pred": pred,
                        "probs": {"a": 0.7 if ok else 0.2, "b": 0.3 if ok else 0.8, "c": 0.0}, "cand_mass": 0.9})
    return out


def test_medir_e_viés_de_ordem():
    x = a17.medir(_rows([True, True, False, False], mudam=1))
    assert x["n"] == 8 and x["vies_ordem"] == pytest.approx(0.25) and x["massa"] == pytest.approx(0.9)
    assert x["acc"] == pytest.approx(3 / 8)


def test_diferenca_pareada():
    a = {"c0": 1.0, "c1": 0.0, "c2": 0.0}
    b = {"c0": 1.0, "c1": 1.0, "c2": 1.0, "c9": 0.0}  # c9 não está em a: fica fora
    d, lo, hi = a17.dif_pareada(a, b, 200, np.random.default_rng(0))
    assert d == pytest.approx(2 / 3) and lo <= d <= hi


def test_main_le_os_runs(tmp_path):
    for m, acc in (("lumi", [True, False, False, True]), ("qwen3_4b", [True, True, True, True])):
        d = tmp_path / f"h17_{m}_bfcl"
        d.mkdir()
        (d / "predicoes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows(acc)), encoding="utf-8")
    out = tmp_path / "r.md"
    a17.main(["--runs", str(tmp_path), "--n-boot", "50", "--out", str(out)])
    txt = out.read_text(encoding="utf-8")
    assert "| bfcl | (todas) | qwen3_4b | +50.0 |" in txt
