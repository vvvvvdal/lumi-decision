"""H9m: mistura exata de dois LoRAs (`scripts/misturar_adaptadores.py`). Precisa de torch (roda no cluster)."""

import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_p = Path(__file__).resolve().parents[1] / "scripts" / "misturar_adaptadores.py"
_spec = importlib.util.spec_from_file_location("mist", _p)
mist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mist)


def _sd(g, r=4, i=6, o=5):
    m = "base_model.model.layers.0.proj"
    return {f"{m}.lora_A.weight": torch.randn(r, i, generator=g), f"{m}.lora_B.weight": torch.randn(o, r, generator=g)}


@pytest.mark.parametrize("rslora", [False, True])
def test_mistura_e_exata(rslora):
    g = torch.Generator().manual_seed(0)
    cfg = {"r": 4, "lora_alpha": 8, "use_rslora": rslora, "target_modules": ["proj"], "base_model_name_or_path": "x"}
    sa, sb = _sd(g), _sd(g)
    for w in (0.0, 0.5, 0.75, 1.0):
        cm, sm = mist.config_misturada(cfg, cfg), mist.misturar_estados(sa, sb, w)
        assert cm["r"] == 8
        assert mist._escala(cm) == pytest.approx(mist._escala(cfg))
        assert mist.verificar(sa, cfg, sb, cfg, sm, cm, w) < 1e-5


def test_configs_diferentes_dao_erro():
    c = {"r": 4, "lora_alpha": 8, "target_modules": ["proj"]}
    with pytest.raises(ValueError):
        mist.config_misturada(c, dict(c, r=8))
    assert math.isclose(mist._escala(c), 2.0)
