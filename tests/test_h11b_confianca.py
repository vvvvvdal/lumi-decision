"""H11b: confiança escrita depois do raciocínio (`prompts.turno_confianca`, `ler_confianca` e
`scripts/h11b_confianca.py`). Tudo em CPU, com os raciocínios de uma H11 falsa."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from lumi_decision.evaluate import predict_one
from lumi_decision.prompts import PERGUNTA_CONFIANCA, ler_confianca, load_template, turno_confianca
from test_h11_raciocinio import CHATML, Falso, caso, tpl

_p = Path(__file__).resolve().parents[1] / "scripts" / "h11b_confianca.py"
_spec = importlib.util.spec_from_file_location("h11b", _p)
h11b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h11b)


def test_ler_confianca():
    assert ler_confianca("85") == pytest.approx(0.85)
    assert ler_confianca("85%") == pytest.approx(0.85)
    assert ler_confianca("Confiança: 90%") == pytest.approx(0.90)
    assert ler_confianca("0,7") == pytest.approx(0.7) and ler_confianca("0.7") == pytest.approx(0.7)
    assert ler_confianca("100") == 1.0 and ler_confianca("0") == 0.0
    assert ler_confianca("não sei") is None and ler_confianca("150") is None


def test_turno_confianca_sai_do_template(tmp_path):
    d = turno_confianca(tpl(tmp_path))
    assert d.startswith("<|im_end|>") and PERGUNTA_CONFIANCA in d and d.endswith("assistant\n<think></think>")
    (tmp_path / "u.jinja").write_text("{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}{% endif %}"
                                      "{% endfor %}", encoding="utf-8")
    with pytest.raises(ValueError):                       # o template esconde a resposta do assistente
        turno_confianca(load_template(tmp_path / "u.jinja"))


def test_auroc():
    assert h11b.auroc([0.9, 0.8, 0.2], [True, True, False]) == 1.0
    assert h11b.auroc([0.2, 0.9], [True, False]) == 0.0
    assert h11b.auroc([1.0, 1.0, 1.0], [True, False, True]) == 0.5          # empates valem meio
    assert math.isnan(h11b.auroc([0.5, 0.6], [True, True]))


class ConfFalso:
    """Mesmo tokenizer de caracteres do `Falso` da H11; responde a confiança pelo raciocínio no prompt."""

    def __init__(self):
        self.prompts = []

    def tokenize(self, s):
        return [ord(c) for c in s]

    def generate(self, ids, n):
        p = "".join(map(chr, ids))
        self.prompts.append(p)
        return [1], ("Acho que 30" if "Alfa, answer A" in p else "85%"), {"latency_ms": 200.0, "stop_type": "eos"}


def _h11_falsa(tmp_path):
    """Dois casos da H11 (raciocínio guloso + 3 sorteados) e a leitura de sempre, como no cluster."""
    t = tpl(tmp_path)
    casos = {c.id: c for c in (caso(0, gold="b"), caso(1, gold="a"))}
    linhas = [predict_one(Falso("acho que é Beta\n</think>"), t, c, "canonical", generate=False, raciocinio=64,
                          amostras=3) for c in casos.values()]
    runs = tmp_path / "runs"
    for nome, rows in (("h11p2_t_dev", linhas),
                       ("h4_log_t_dev", [{"case_id": c.id, "order": "canonical", "gold": c.gold, "pred": "b",
                                          "probs": {"a": 0.4, "b": 0.6}} for c in casos.values()])):
        (runs / nome).mkdir(parents=True)
        (runs / nome / "predicoes.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return t, casos, linhas, runs


def test_gera_uma_confianca_por_caso_e_por_sorteio(tmp_path):
    t, casos, linhas, runs = _h11_falsa(tmp_path)
    be = ConfFalso()
    out = runs / "h11b_t_dev" / "confianca.jsonl"
    out.parent.mkdir()
    assert h11b.gerar_fonte(be, t, casos, linhas, out) == 2
    rows = h11b.ler_jsonl(out)
    assert [r["conf"] for r in rows] == [pytest.approx(0.85)] * 2
    assert [a["conf"] for a in rows[0]["amostras"]] == [pytest.approx(0.3), pytest.approx(0.85), pytest.approx(0.85)]
    assert [a["pred"] for a in rows[0]["amostras"]] == ["a", "b", "b"]
    # a letra escolhida entra depois do raciocínio e do separador lido na H11, seguida do turno da pergunta
    assert "acho que é Beta\n</think>\n\nB<|im_end|>" in be.prompts[0]
    assert PERGUNTA_CONFIANCA in be.prompts[0] and be.prompts[0].endswith("assistant\n<think></think>")
    assert h11b.gerar_fonte(be, t, casos, linhas, out) == 0                 # retomável
    ruim = [{**linhas[0], "prompt_sha1": "0" * 40}]
    with pytest.raises(ValueError):                                          # prompt remontado ≠ o da H11
        h11b.gerar_fonte(be, t, casos, ruim, out, overwrite=True)


def test_relatorio(tmp_path):
    t, casos, linhas, runs = _h11_falsa(tmp_path)
    out = runs / "h11b_t_dev" / "confianca.jsonl"
    out.parent.mkdir()
    h11b.gerar_fonte(ConfFalso(), t, casos, linhas, out)
    md, bruto = h11b.analisar(runs, "h11b", "h11p2", "h4_log", ["t"])
    esc = bruto["escrita"]
    assert esc["validas"] == 1.0 and esc["moda"] == pytest.approx(0.85) and esc["moda_frac"] == 1.0
    crit = {c["nome"].split(":")[0]: c["passa"] for c in bruto["criterio"]}
    assert crit == {"Escreve": True, "Gradua": False, "Separa": False}      # sempre 85: não gradua nem separa
    f = bruto["fontes"]
    assert f["escrita"]["n"] == 2 and f["escrita"]["auroc"] == 0.5          # c0 acerta, c1 erra, ambos 85
    assert f["consistencia"]["n"] == 2 and f["sem_rac"]["n"] == 2
    s = bruto["sorteio"]
    assert s["escrita"]["n"] == 6 and s["muda"] == 1.0 and s["sinal_rlcr"] == 1.0
    assert "## Critério (guloso)" in md and "**não passa**" in md and "Fim de um prompt" in md
