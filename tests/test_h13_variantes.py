"""H13: variantes do prompt (aumento de dados no estilo Clef). Sem modelo e sem o tokenizer do Lumi."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from lumi_decision import prompts
from lumi_decision.prompts import (N_VARIANTES_TREINO, VARIANTE_AVALIACAO, order_options, render, user_message,
                                   variante_treino)
from lumi_decision.schema import Case

ESTADO = {"cliente": "Ana", "valor": 120.5, "itens": [1, 2], "status": "aberto", "obs": "ção"}


def caso(state=None, cid="telepatia:1:q"):
    return Case(id=cid, group_id="g", source="telepatia", family="f", split="dev", held_out=False, qtype="choice",
                state=json.dumps(ESTADO, ensure_ascii=False, indent=2) if state is None else state,
                question="O que fazer?", options=[("a", "Aprovar"), ("b", "Recusar"), ("c", "Pedir documento")],
                gold="b")


def test_variante_0_e_o_prompt_de_sempre():
    c = caso(state="Texto do caso.")
    esperado = ("Contexto:\nTexto do caso.\n\nPergunta: O que fazer?\n\nOpções:\nA. Aprovar\nB. Recusar\n"
                "C. Pedir documento\n\nResponda apenas com a letra da opção escolhida.")
    assert user_message(c, c.options) == esperado
    assert user_message(c, c.options, 0, 123) == esperado


@pytest.mark.parametrize("v", range(1, VARIANTE_AVALIACAO + 1))
def test_variantes_mantem_opcoes_pergunta_e_estado(v):
    c = caso(state="Texto do caso.")
    m = user_message(c, c.options, v)
    assert m != user_message(c, c.options)
    assert "O que fazer?" in m and "Texto do caso." in m
    for letra, desc in zip("ABC", ("Aprovar", "Recusar", "Pedir documento")):
        assert letra in m and desc in m
    assert m.index("Aprovar") < m.index("Recusar") < m.index("Pedir documento")  # a ordem das opções não muda


def test_variantes_sao_todas_diferentes():
    c = caso(state="Texto do caso.")
    msgs = {user_message(c, c.options, v) for v in range(VARIANTE_AVALIACAO + 1)}
    assert len(msgs) == VARIANTE_AVALIACAO + 1


def test_variante_desconhecida():
    c = caso()
    with pytest.raises(ValueError):
        user_message(c, c.options, VARIANTE_AVALIACAO + 1)


def _json_do_prompt(m: str) -> str:
    return m[m.index("{"): m.rindex("}") + 1]


def test_json_da_telepatia_mantem_chaves_e_valores():
    c = caso()
    ordens = set()
    for sem in range(20):
        txt = _json_do_prompt(user_message(c, c.options, 1, sem))
        assert json.loads(txt) == ESTADO
        ordens.add(tuple(json.loads(txt)))
    assert len(ordens) > 1  # a ordem das chaves é sorteada
    # determinístico: mesma variante, semente e caso -> mesmo texto
    assert user_message(c, c.options, 2, 7) == user_message(c, c.options, 2, 7)


def test_estado_que_nao_e_json_fica_igual():
    for st in ("Texto livre.", "{não é json", "[1, 2, 3]", '{"so": 1}'):
        c = caso(state=st)
        assert st in user_message(c, c.options, 3, 5)


def test_variante_treino_deterministica_e_sem_a_de_avaliacao():
    vs = [variante_treino(f"x:{i}", 5) for i in range(400)]
    assert vs == [variante_treino(f"x:{i}", 5) for i in range(400)]
    assert set(vs) == set(range(N_VARIANTES_TREINO))
    assert VARIANTE_AVALIACAO not in vs
    assert min(Counter(vs).values()) > 60  # mais ou menos uniforme
    assert [variante_treino(f"x:{i}", 6) for i in range(50)] != vs[:50]  # muda de época para época


TEMPLATE_FALSO = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|fim|>{% endfor %}"
                  "{% if add_generation_prompt %}<|assistant|><think></think>{% endif %}")


def test_encoder_mesmas_letras_em_todas_as_variantes():
    from lumi_decision.lora import Encoder

    tpl = ImmutableSandboxedEnvironment().from_string(TEMPLATE_FALSO)
    tokenize = lambda s: [ord(ch) for ch in s]  # um token por caractere
    enc = Encoder(tokenize, tpl, n_checks=2)
    c = caso()
    for v in range(VARIANTE_AVALIACAO + 1):
        ex = enc(c, "canonical", False, v, 3)
        assert ex.cands == [ord(x) for x in "ABC"]
        assert ex.gold_pos == 1
    assert all(enc.checked[(v, 3)] == 1 for v in range(VARIANTE_AVALIACAO + 1))
    p0 = render(tpl, c, order_options(c, "canonical"))
    p4 = render(tpl, c, order_options(c, "canonical"), variante=4)
    assert p0.endswith("<think></think>") and p4.endswith("<think></think>") and p0 != p4


def test_variante_com_ponteiro_da_erro():
    from lumi_decision.evaluate import predict_one

    class Falso:
        pointer = object()

    with pytest.raises(ValueError, match="H13"):
        predict_one(Falso(), None, caso(), "canonical", variante=VARIANTE_AVALIACAO)


def test_comparar_h4_le_o_split():
    p = Path(__file__).resolve().parents[1] / "scripts" / "comparar_h4.py"
    spec = importlib.util.spec_from_file_location("comparar_h4", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.run_dir(Path("runs"), "log", 0, "felhen") == Path("runs/h4_log_felhen_dev")
    assert mod.run_dir(Path("runs"), "aug", 2, "telepatia", "devvar") == Path("runs/h4_aug_s2_telepatia_devvar")


def test_variantes_documentadas():
    assert set(prompts._VARIANTES) == set(range(1, VARIANTE_AVALIACAO + 1))
