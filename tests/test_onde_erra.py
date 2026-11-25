"""Onde erra (`scripts/onde_erra.py`): erro vizinho contra grave numa escala."""

import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "onde_erra.py"
_s = importlib.util.spec_from_file_location("onde_erra", _p)
oe = importlib.util.module_from_spec(_s)
_s.loader.exec_module(oe)

ESC = [["f", "i", "v"]]


def _r(gold, pred):
    return {"gold": gold, "pred": pred, "probs": {"f": 0.3, "i": 0.3, "v": 0.4}}


def test_vizinho_e_grave():
    rows = [_r("f", "f"), _r("f", "i"), _r("f", "v"), _r("v", "f"), _r("i", "v")]
    txt = "\n".join(oe.resumo_familia(rows, ESC))
    assert "4 erros em perguntas com escala; vizinhos 2 (50.0%), graves 2 (50.0%)" in txt
    assert "acurácia 0.200" in txt


def test_escala_pelo_conjunto_de_ids():
    assert oe.escala_de(["v", "f", "i"], ESC) == ["f", "i", "v"]
    assert oe.escala_de(["a", "b"], ESC) is None
    assert oe.carregar_escalas(Path("/nao/existe.toml")) == []
