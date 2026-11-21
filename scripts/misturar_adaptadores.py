"""H9m: mistura dois adaptadores LoRA com peso w: ΔW = (1 − w)·ΔW_a + w·ΔW_b, sem treinar.

Ideia do post "harness adapters" (Hugging Face, 2026) e do WiSE-FT (Wortsman et al., 2022): ficar entre o modelo
antigo (`h4_log`, sem esquecimento) e o ajustado (`h9_k100`, que sabe as famílias novas). Interpolar A e B
separadamente NÃO dá a média de ΔW = B·A; por isso concatenamos no posto (como o `combination_type="cat"` do
PEFT): A = [A_a; A_b], B = [(1 − w)·B_a, w·B_b]. O posto dobra (2r), e o lora_alpha é ajustado para a escala
de cada parte continuar a mesma (alpha/r, ou alpha/√r com rsLoRA). O resultado é EXATO, não aproximado.

Grava `<out>/adapter/` (adapter_config.json + adapter_model.safetensors) e copia `adapt_grupos.json` do run b
(os grupos do treino curto continuam fora da avaliação). Depois confere ΔW em alguns módulos.

Uso (no cluster, precisa de torch e safetensors):
    python scripts/misturar_adaptadores.py --a runs/h4_log --b runs/h9_k100 --w 0.5 --out runs/h9m50_k100
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def _cfg(d: Path) -> dict:
    return json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))


def _escala(cfg: dict) -> float:
    return cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])


def misturar_estados(sa: dict, sb: dict, w: float) -> dict:
    import torch

    if set(sa) != set(sb):
        raise ValueError("os adaptadores têm módulos diferentes")
    out = {}
    for k in sa:
        if k.endswith("lora_A.weight"):
            out[k] = torch.cat([sa[k], sb[k]], 0)
        elif k.endswith("lora_B.weight"):
            out[k] = torch.cat([(1 - w) * sa[k].float(), w * sb[k].float()], 1).to(sa[k].dtype)
        else:
            raise ValueError(f"chave inesperada (só lora_A/lora_B são aceitas): {k}")
    return out


def config_misturada(ca: dict, cb: dict) -> dict:
    for c in ("r", "lora_alpha", "use_rslora", "base_model_name_or_path"):
        if ca.get(c) != cb.get(c):
            raise ValueError(f"configs diferentes em {c}: {ca.get(c)} != {cb.get(c)}")
    if sorted(ca.get("target_modules") or []) != sorted(cb.get("target_modules") or []):
        raise ValueError("target_modules diferentes")
    if ca.get("rank_pattern") or ca.get("alpha_pattern") or cb.get("rank_pattern") or cb.get("alpha_pattern"):
        raise ValueError("rank_pattern/alpha_pattern não suportados")
    out = dict(ca)
    out["r"] = 2 * ca["r"]
    out["lora_alpha"] = ca["lora_alpha"] * (math.sqrt(2) if ca.get("use_rslora") else 2)
    return out


def delta(sd: dict, cfg: dict, chave_a: str):
    chave_b = chave_a.replace("lora_A.weight", "lora_B.weight")
    return _escala(cfg) * (sd[chave_b].float() @ sd[chave_a].float())


def verificar(sa, ca, sb, cb, sm, cm, w: float, n: int = 6) -> float:
    """Maior erro relativo de ΔW_mistura contra (1 − w)·ΔW_a + w·ΔW_b em n módulos."""
    chaves = sorted(k for k in sa if k.endswith("lora_A.weight"))
    pior = 0.0
    for k in chaves[:: max(1, len(chaves) // n)][:n]:
        ref = (1 - w) * delta(sa, ca, k) + w * delta(sb, cb, k)
        err = float((delta(sm, cm, k) - ref).norm() / ref.norm().clamp_min(1e-12))
        pior = max(pior, err)
    return pior


def main(argv=None):
    from safetensors.torch import load_file, save_file

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", required=True, help="run de partida (ex.: runs/h4_log); usa <a>/adapter")
    p.add_argument("--b", required=True, help="run ajustado (ex.: runs/h9_k100); usa <b>/adapter")
    p.add_argument("--w", type=float, required=True, help="peso de b (0 = só a, 1 = só b)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if not 0.0 <= a.w <= 1.0:
        raise SystemExit("--w precisa estar em [0, 1]")
    da, db, out = Path(a.a) / "adapter", Path(a.b) / "adapter", Path(a.out)
    ca, cb = _cfg(da), _cfg(db)
    sa, sb = load_file(str(da / "adapter_model.safetensors")), load_file(str(db / "adapter_model.safetensors"))
    cm, sm = config_misturada(ca, cb), misturar_estados(sa, sb, a.w)
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    save_file(sm, str(out / "adapter" / "adapter_model.safetensors"))
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(cm, indent=2), encoding="utf-8")
    if (Path(a.b) / "adapt_grupos.json").exists():
        shutil.copy(Path(a.b) / "adapt_grupos.json", out / "adapt_grupos.json")
    (out / "mistura.json").write_text(json.dumps({"a": a.a, "b": a.b, "w": a.w}, indent=1), encoding="utf-8")
    err = verificar(sa, ca, sb, cb, sm, cm, a.w)
    print(f">> mistura w={a.w}: {len(sm) // 2} módulos, posto {ca['r']} -> {cm['r']}, erro relativo de ΔW {err:.2e}")
    if err > 1e-2:
        raise SystemExit(">> ΔW da mistura não confere; abortei")


if __name__ == "__main__":
    main()
