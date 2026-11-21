"""H11: raciocínio antes da leitura das letras (`prompts.com_raciocinio`, `evaluate.predict_one(raciocinio=N)`)
e o relatório do piloto (`scripts/h11_piloto.py`). Tudo em CPU, com backend falso."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from lumi_decision.evaluate import predict_one, run_fingerprint
from lumi_decision.prompts import com_raciocinio, letra_do_raciocinio, load_template, prompt_pensando, render
from lumi_decision.schema import Case

CHATML = ("{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
          "{%- if add_generation_prompt %}<|im_start|>assistant\n<think></think>{%- endif %}")
CHATML_PENSA = ("{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
                "{%- if add_generation_prompt %}<|im_start|>assistant\n"
                "{% if enable_thinking %}<think>\n{% else %}<think></think>{% endif %}{%- endif %}")

_p = Path(__file__).resolve().parents[1] / "scripts" / "h11_piloto.py"
_spec = importlib.util.spec_from_file_location("h11p", _p)
h11p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h11p)


def tpl(tmp_path, texto=CHATML):
    p = tmp_path / "t.jinja"
    p.write_text(texto, encoding="utf-8")
    return load_template(p)


def caso(i=0, held=False, gold="b"):
    return Case(id=f"c{i}", group_id=f"g{i}", source="t", family="f", split="dev", held_out=held, qtype="choice",
                state=f"estado {i}", question="Qual?", options=[("a", "Alfa"), ("b", "Beta")], gold=gold)


class Falso:
    """Tokenizer de caracteres (cada letra é um token novo na fronteira) e geração fixa."""
    pointer = None

    def __init__(self, texto, parada="stop"):
        self.texto, self.parada, self.chamadas = texto, parada, []

    def tokenize(self, s):
        return [ord(c) for c in s]

    def generate(self, ids, n, stop_ids=None):
        self.chamadas.append("".join(map(chr, ids)))
        if len(self.chamadas) > 1:                       # a continuação depois do </think>
            return [1, 2], "\n\nB", {"latency_ms": 5.0, "stop_type": "eos"}
        self.prompt_gen, self.n, self.stop_ids = self.chamadas[0], n, stop_ids
        return [1] * 7, self.texto, {"latency_ms": 1500.0, "stop_type": self.parada}

    def readout(self, ids, cands):
        self.prompt_lido = "".join(map(chr, ids))
        return [-2.0, -0.1], {"latency_ms": 1.0}


def test_sem_pensar_o_prompt_nao_muda(tmp_path):
    t, c = tpl(tmp_path, CHATML_PENSA), caso()
    assert render(t, c, c.options).endswith("assistant\n<think></think>")
    assert render(t, c, c.options, pensar=True).endswith("assistant\n<think>\n")


def test_prompt_pensando_abre_o_think_se_o_template_ignorar():
    off = "x<|im_start|>assistant\n<think></think>"
    assert prompt_pensando(off, off) == "x<|im_start|>assistant\n<think>"
    assert prompt_pensando("outro<think>\n", off) == "outro<think>\n"
    with pytest.raises(ValueError):
        prompt_pensando("sem", "sem")


def test_com_raciocinio_corta_no_fim_e_termina_igual_ao_de_sempre():
    off = "P<think></think>"
    assert com_raciocinio("P<think>\n", off, "pensei. logo B\n</think>\nB lixo") == "P<think>\npensei. logo B\n</think>"
    assert com_raciocinio("P<think>", off, "sem fechar") == "P<think>sem fechar</think>"
    assert com_raciocinio("P<think>", "P<think></think>\n", "r").endswith("r</think>\n")


def test_predict_one_le_as_letras_depois_do_raciocinio(tmp_path):
    t, c = tpl(tmp_path), caso()
    be = Falso("acho que é Beta\n</think>")
    row = predict_one(be, t, c, "canonical", generate=False, raciocinio=64)
    assert be.prompt_gen.endswith("assistant\n<think>") and be.n == 64
    assert be.stop_ids is None                       # </think> não é um token só no tokenizer falso
    assert be.prompt_lido.endswith("<think>acho que é Beta\n</think>\n\n")   # leitura principal: + \n\n
    assert be.chamadas[1].endswith("acho que é Beta\n</think>") and row["rac_depois_think"] == "\n\nB"
    assert row["raciocinio"] == "acho que é Beta\n" and row["rac_parada"] == "fim" and row["rac_ms"] == 1500.0
    assert row["pred"] == "b" and row["rac_sufixo"] == "nl2"
    assert set(row["rac_probs_sufixo"]) == {"cru", "nl", "nl2"}
    assert row["rac_letra_texto"] is None                       # o texto não diz a letra


def test_letra_do_raciocinio():
    assert letra_do_raciocinio("So answer N.", 25) == "N"
    assert letra_do_raciocinio("That's option E.\n\nThus answer: E.", 25) == "E"
    assert letra_do_raciocinio("theme is Communications (F). So answer F.", 25) == "F"
    assert letra_do_raciocinio("Options: A. No rush; B. Routine. ... So answer B.", 4) == "B"
    assert letra_do_raciocinio("Thus answer: E.", 4) is None            # fora das opções
    assert letra_do_raciocinio("the answer is likely low", 4) is None
    assert letra_do_raciocinio("Resposta: C", 4) == "C"
    assert letra_do_raciocinio("answer A. Hmm, no: answer B.\n</think>\nanswer C", 4) == "B"


def test_predict_one_sem_raciocinio_igual_ao_de_antes(tmp_path):
    t, c = tpl(tmp_path), caso()
    be = Falso("nunca usado")
    row = predict_one(be, t, c, "canonical", generate=False)
    assert "raciocinio" not in row and be.prompt_lido.endswith("assistant\n<think></think>")


def test_limite_sem_fechar_fica_registrado(tmp_path):
    t, c = tpl(tmp_path), caso()
    row = predict_one(Falso("bla bla", parada="limit"), t, c, "canonical", generate=False, raciocinio=8)
    assert row["rac_parada"] == "limit"


def test_impressao_muda_com_raciocinio(tmp_path):
    (tmp_path / "t.jinja").write_text(CHATML, encoding="utf-8")
    a = argparse.Namespace(backend="hf", model="m", revision="r", dtype="bfloat16", native=True, ctx=0,
                           template=str(tmp_path / "t.jinja"), source="t", split="dev", no_generate=True,
                           orders="canonical", adapter=None, raciocinio=0)
    sem = run_fingerprint(a, [caso()])
    a.raciocinio = 256
    assert run_fingerprint(a, [caso()]) != sem and run_fingerprint(a, [caso()])["raciocinio"] == 256


def test_repeticao():
    assert h11p.fracao_repetida("um dois três quatro cinco seis") == 0
    assert h11p.fracao_repetida("a b c d " * 20) > 0.9
    assert h11p.degenerado({"raciocinio": "   "})


def _linha(i, held, pred, probs, **kw):
    return {"case_id": f"c{i}", "group_id": f"g{i}", "family": "f", "order": "canonical", "held_out": held,
            "gold": "a", "pred": pred, "probs": probs, **kw}


def test_relatorio_do_piloto(tmp_path):
    base, pil = [], []
    for i in range(6):
        held = i >= 3
        base.append(_linha(i, held, "b", {"a": 0.4, "b": 0.6}))
        base.append({**_linha(i, held, "a", {"a": 0.9, "b": 0.1}), "order": "reversed"})   # ignorada
        pil.append(_linha(i, held, "a", {"a": 0.8, "b": 0.2}, raciocinio=f"passo {i} então a",
                          rac_tokens=10 + i, rac_parada="fim" if i else "limit", rac_ms=2000.0,
                          rac_letra_texto="a" if i % 3 else None, rac_depois_think="\n\nA",
                          rac_probs_sufixo={"cru": {"a": 0.3, "b": 0.7}, "nl2": {"a": 0.8, "b": 0.2}}))
    for nome, rows in (("h4_log_felhen_dev", base), ("h11p_felhen_dev", pil)):
        (tmp_path / nome).mkdir()
        (tmp_path / nome / "predicoes.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "r.md"
    h11p.main(["--runs", str(tmp_path), "--fontes", "felhen", "--out", str(out), "--rollouts", "1800"])
    bruto = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    v = bruto["felhen|vistas"]
    assert v["n"] == 3 and v["acc_sem"] == 0 and v["acc_com"] == 1 and v["mudou"] == 1
    assert v["limite"] == pytest.approx(1 / 3)
    assert v["acc_texto"] == pytest.approx(2 / 3) and v["texto_disse"] == pytest.approx(2 / 3)
    assert v["sufixos"]["cru"]["acc"] == 0 and v["sufixos"]["nl2"]["acc"] == 1 and "nl" not in v["sufixos"]
    assert bruto["geral"]["horas_grpo"] == pytest.approx(1.0)           # 1800 × 2 s = 1 h
    md = out.read_text(encoding="utf-8")
    assert "passo 0" in md and "Onde ler a decisão" in md and '`"\\n\\nA"`' in md
