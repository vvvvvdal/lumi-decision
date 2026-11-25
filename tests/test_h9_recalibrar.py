"""H9b: recalibração (H7) sobre o braço da H9 (`scripts/h9_recalibrar.py`), em pastas sintéticas."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "h9_recalibrar.py"
_spec = importlib.util.spec_from_file_location("h9b", _p)
h9b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h9b)


def _casos(rng, n_grupos, familia, held, split, escala, p_lo):
    rows = []
    for g in range(n_grupos):
        p = rng.uniform(p_lo, 0.97)
        acerta = rng.random() < p
        conf = 1 / (1 + math.exp(-escala * math.log(p / (1 - p))))
        for ordem in ("canonical", "reversed"):
            rows.append({"case_id": f"{familia}-{split}-{g}", "group_id": f"{familia}-{split}-{g}", "order": ordem,
                         "family": familia, "held_out": held, "probs": {"a": conf, "b": 1 - conf}, "pred": "a",
                         "gold": "a" if acerta else "b"})
    return rows


def _salvar(pasta: Path, rows):
    pasta.mkdir(parents=True)
    (pasta / "predicoes.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _runs(tmp: Path, treino: list[str]):
    rng = np.random.default_rng(0)
    for nm, escala, p_lo in (("h4_log", 1.0, 0.5), ("h9_k100", 3.0, 0.7)):  # H9: acerta mais, mas superconfiante
        _salvar(tmp / f"{nm}_felhen_calib", _casos(rng, 200, "vista", False, "calib", 1.0, 0.5))
        _salvar(tmp / f"{nm}_felhen_dev", _casos(rng, 200, "vista", False, "dev", 1.0, 0.5)
                + _casos(rng, 150, "nova", True, "dev", escala, p_lo))
    (tmp / "h9_k100").mkdir()
    (tmp / "h9_k100" / "adapt_grupos.json").write_text(json.dumps(
        [{"fonte": "felhen", "familia": "nova", "split": "dev", "grupos": treino}]), encoding="utf-8")


def test_recalibrar_h9_remove_treino_e_desfaz_superconfianca(tmp_path):
    treino = [f"nova-dev-{g}" for g in range(20)]
    _runs(tmp_path, treino)
    out = tmp_path / "r.md"
    h9b.main(["--runs", str(tmp_path), "--fontes", "felhen", "--sementes", "0", "--ks", "25",
              "--repeticoes", "5", "--out", str(out)])
    assert "h9_k100" in out.read_text(encoding="utf-8")
    bruto = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    novo = bruto["felhen"]["novo"]["0"]
    assert novo["familias"]["nova"]["n_dev"] == 2 * 130          # os 20 grupos do treino saíram
    ece_cru = novo["sem_T"]["ece"]
    ece_T = np.mean([m["ece"] for m in novo["T_fam|25"]])
    assert ece_cru > 0.10 and ece_T < ece_cru - 0.04               # a recalibração desfaz a superconfiança
    assert "Famílias VISTAS" in out.read_text(encoding="utf-8")


def test_split_le_outra_pasta_e_mantem_o_calib(tmp_path):
    """H13b: --split devvar mede as pastas _devvar com o mesmo calib; com a mesma cópia, o resultado é igual."""
    import shutil
    _runs(tmp_path, [f"nova-dev-{g}" for g in range(20)])
    for nm in ("h4_log", "h9_k100"):
        shutil.copytree(tmp_path / f"{nm}_felhen_dev", tmp_path / f"{nm}_felhen_devvar")
    args = ["--runs", str(tmp_path), "--fontes", "felhen", "--sementes", "0", "--ks", "25", "--repeticoes", "5"]
    h9b.main(args + ["--out", str(tmp_path / "a.md")])
    h9b.main(args + ["--split", "devvar", "--out", str(tmp_path / "b.md")])
    assert (tmp_path / "a.md").read_text(encoding="utf-8") == (tmp_path / "b.md").read_text(encoding="utf-8")
    shutil.rmtree(tmp_path / "h9_k100_felhen_devvar")
    h9b.main(args + ["--split", "devvar", "--out", str(tmp_path / "c.md")])  # falta a pasta: pula sem quebrar
