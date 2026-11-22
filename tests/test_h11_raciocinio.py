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

    def amostrar(self, ids, n_tok, n, temperatura, top_p, stop_ids, semente):
        self.sorteio = dict(n_tok=n_tok, n=n, temperatura=temperatura, top_p=top_p, semente=semente)
        textos = ["logo Alfa, answer A\n</think>", "Beta. So answer B.\n</think>", "sem fim"][:n]
        return [([1] * (3 + i), t, "stop" if "</think>" in t else "limit") for i, t in enumerate(textos)], \
            {"latency_ms": 4000.0, "lote": n}

    def readout(self, ids, cands):
        self.prompt_lido = "".join(map(chr, ids))
        self.lidos = getattr(self, "lidos", []) + [self.prompt_lido]
        if "Alfa, answer A" in self.prompt_lido:
            return [-0.1, -2.0], {"latency_ms": 1.0}
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


def test_sorteio_le_cada_raciocinio_sorteado(tmp_path):
    t, c = tpl(tmp_path), caso()
    be = Falso("acho que é Beta\n</think>")
    row = predict_one(be, t, c, "canonical", generate=False, raciocinio=64, amostras=3, temperatura=1.0, top_p=0.95)
    assert be.sorteio["n"] == 3 and be.sorteio["n_tok"] == 64 and be.sorteio["temperatura"] == 1.0
    g = row["rac_amostras"]
    assert len(g) == 3 and row["rac_amostras_ms"] == 4000.0 and row["rac_amostras_lote"] == 3
    assert max(g[0]["probs"], key=g[0]["probs"].get) == "a" and max(g[1]["probs"], key=g[1]["probs"].get) == "b"
    assert [a["parada"] for a in g] == ["fim", "fim", "limit"]
    assert [a["letra_texto"] for a in g] == ["a", "b", None]
    assert any(x.endswith("assistant\n<think>sem fim</think>\n\n") for x in be.lidos)   # mesmo separador da principal
    assert row["pred"] == "b" and row["rac_semente"] == predict_one(Falso("x</think>"), t, c, "canonical",
                                                                  generate=False, raciocinio=8, amostras=1)["rac_semente"]
    sem = predict_one(Falso("x</think>"), t, c, "canonical", generate=False, raciocinio=8)
    assert "rac_amostras" not in sem


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
    com = run_fingerprint(a, [caso()])
    a.raciocinio_amostras, a.temperatura, a.top_p = 4, 1.0, 0.95
    assert run_fingerprint(a, [caso()]) != com and run_fingerprint(a, [caso()])["rac_amostras"] == 4


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
                          rac_probs_sufixo={"cru": {"a": 0.3, "b": 0.7}, "nl2": {"a": 0.8, "b": 0.2}},
                          rac_amostras_ms=8000.0, rac_amostras_lote=4,
                          rac_amostras=[{"raciocinio": "x y z w", "tokens": 5, "parada": "limit" if j == 3 else "fim",
                                         "letra_texto": None,
                                         "probs": ({"a": 0.9, "b": 0.1} if (j < 2 or i % 2) else {"a": 0.05, "b": 0.95})}
                                        for j in range(4)]))
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
    sv = v["sorteio"]                                                   # vistas: i = 0, 1, 2
    assert sv["n"] == 3 and sv["g"] == 4
    assert sv["misto"] == pytest.approx(2 / 3)                          # i = 0 e 2 misturam certo e errado
    assert sv["com_sinal"] == pytest.approx(2 / 3) and sv["acc"] == pytest.approx(8 / 12)
    assert bruto["geral"]["horas_grpo_sorteio"] == pytest.approx(1800 / 4 * 8 / 3600)   # 450 grupos × 8 s
    # separação fechou / não fechou: guloso (i = 0 bateu no limite) e sorteio (a amostra j = 3 bateu no limite)
    fv = v["por_fecho"]
    assert fv["fechou"]["n"] == 2 and fv["nao_fechou"]["n"] == 1
    assert fv["nao_fechou"]["acc_sem"] == 0 and fv["nao_fechou"]["acc_com"] == 1
    assert fv["fechou"]["conf_com"] == pytest.approx(0.8) and fv["fechou"]["erros"] == 0
    assert bruto["geral"]["por_fecho"]["fechou"]["n"] == 5 and bruto["geral"]["por_fecho"]["nao_fechou"]["n"] == 1
    sf = sv["por_fecho"]                                                # j = 3: i = 0 e 2 erram, i = 1 acerta
    assert sf["nao_fechou"]["n"] == 3 and sf["nao_fechou"]["acc"] == pytest.approx(1 / 3)
    assert (sf["nao_fechou"]["erros_conf"], sf["nao_fechou"]["erros"]) == (2, 2)
    assert sf["fechou"]["n"] == 9 and sf["fechou"]["acc"] == pytest.approx(7 / 9)
    md = out.read_text(encoding="utf-8")
    assert "Fechou ou não fechou" in md and "| todas | todos | não fechou | 1 |" in md
    assert "passo 0" in md and "Onde ler a decisão" in md and "Sorteio (como um grupo do GRPO)" in md and '`"\\n\\nA"`' in md
