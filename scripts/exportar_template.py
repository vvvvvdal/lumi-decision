"""H17: salva o chat template de cada modelo de `configs/modelos_h17.toml` no caminho `template` dele.

O Lumi já tem o dele em `artifacts/lumi/`. Para os outros, lê o template do tokenizer na revisão fixada. Se o
tokenizer tiver vários templates (dict), usa o "default". Avisa se o template não citar `enable_thinking`: a
leitura das letras passa enable_thinking=False, e um template que não conhece a variável simplesmente a ignora.

Uso (no cluster, com internet ou com o cache já baixado): python scripts/exportar_template.py
"""

from __future__ import annotations

import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main():
    from transformers import AutoTokenizer

    cfg = tomllib.loads((RAIZ / "configs" / "modelos_h17.toml").read_text(encoding="utf-8"))
    for nome, m in cfg.items():
        destino = RAIZ / m["template"]
        if destino.exists():
            print(f"{nome}: já existe {destino}")
            continue
        tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revisao"], trust_remote_code=not m["nativo"])
        tpl = tok.chat_template
        if isinstance(tpl, dict):
            tpl = tpl.get("default") or next(iter(tpl.values()))
        if not tpl:
            raise SystemExit(f"{nome}: o tokenizer não tem chat_template")
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(tpl, encoding="utf-8")
        aviso = "" if "enable_thinking" in tpl else " (não cita enable_thinking)"
        print(f"{nome}: {destino}{aviso}")


if __name__ == "__main__":
    main()
