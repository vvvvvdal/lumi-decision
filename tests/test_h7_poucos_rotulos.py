"""H7: recalibração por família com poucos rótulos (`scripts/h7_poucos_rotulos.py`), em dados sintéticos."""

import importlib.util
import math
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "h7_poucos_rotulos.py"
_spec = importlib.util.spec_from_file_location("h7", _p)
h7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h7)


def _casos(rng, n_grupos, familia, held, split, escala):
    """Cada grupo = 1 caso em 2 ordens. A chance real de acerto é p; o modelo reporta p 'esticada' por `escala`
    (escala > 1 = superconfiante, como nas famílias reservadas)."""
    rows = []
    for g in range(n_grupos):
        p = rng.uniform(0.5, 0.97)  # "a" é sempre a mais provável (pred = argmax)
        acerta = rng.random() < p
        z = escala * math.log(p / (1 - p))
        conf = 1 / (1 + math.exp(-z))
        probs = {"a": conf, "b": 1 - conf}
        for ordem in ("canonical", "reversed"):
            rows.append({"case_id": f"{familia}-{split}-{g}", "group_id": f"{familia}-{split}-{g}", "order": ordem,
                         "family": familia, "held_out": held, "probs": probs, "pred": "a",
                         "gold": "a" if acerta else "b"})
    return rows


def _dados(seed=0):
    rng = np.random.default_rng(seed)
    calib = _casos(rng, 300, "vista", False, "calib", 1.0) + _casos(rng, 120, "nova", True, "calib", 3.0)
    dev = _casos(rng, 300, "vista", False, "dev", 1.0) + _casos(rng, 200, "nova", True, "dev", 3.0)
    return calib, dev


def test_recalibrar_com_poucos_rotulos_melhora_a_familia_nova():
    calib, dev = _dados()
    r = h7.uma_fonte(calib, dev, ks=[25], reps=10, risco=0.10, rotulos="calib", min_aceitos=5,
                     rng=np.random.default_rng(0))
    ece_T = np.mean([m["ece"] for m in r["T_fam|25"]])
    assert r["global"]["ece"] > 0.10           # a família nova chega superconfiante
    # 25 grupos rotulados recuperam quase tudo que o oráculo recupera (o piso de ~0,08 é ruído de amostra do ECE)
    assert ece_T < r["global"]["ece"] - 0.06
    assert ece_T < r["oraculo"]["ece"] + 0.04
    assert r["familias"]["nova"]["T_oraculo"] > 2  # a temperatura achada desfaz o 'esticamento'


def test_rotulos_do_dev_nao_se_misturam_com_a_medida():
    calib, dev = _dados(1)
    calib = [x for x in calib if not x["held_out"]]  # família nova ausente do calib -> usa o dev
    r = h7.uma_fonte(calib, dev, ks=[10], reps=3, risco=0.10, rotulos="calib", min_aceitos=5,
                     rng=np.random.default_rng(0))
    assert all(not math.isnan(m["ece"]) for m in r["T_fam|10"])
