"""H8: exemplos com gabarito da mesma família no prompt (`prompts.render`, `evaluate.escolher_exemplos`)."""

from lumi_decision.evaluate import escolher_exemplos
from lumi_decision.prompts import load_template, render
from lumi_decision.schema import Case

CHATML = ("{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
          "{%- if add_generation_prompt %}<|im_start|>assistant\n<think></think>{%- endif %}")


def caso(i, fam="f", grupo=None, gold="b", split="dev"):
    return Case(id=f"c{i}", group_id=grupo or f"g{i}", source="t", family=fam, split=split, held_out=True,
                qtype="choice", state=f"estado {i}", question="Qual?", options=[("a", "Alfa"), ("b", "Beta")],
                gold=gold)


def tpl(tmp_path):
    p = tmp_path / "t.jinja"
    p.write_text(CHATML, encoding="utf-8")
    return load_template(p)


def test_sem_exemplos_o_prompt_nao_muda(tmp_path):
    t, c = tpl(tmp_path), caso(0)
    msgs = [{"role": "user", "content": __import__("lumi_decision.prompts", fromlist=["x"]).user_message(c, c.options)}]
    antigo = t.render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
    assert render(t, c, c.options) == antigo == render(t, c, c.options, [])


def test_exemplos_viram_turnos_respondidos(tmp_path):
    t, c = tpl(tmp_path), caso(0)
    p = render(t, c, c.options, [caso(1, gold="a"), caso(2, gold="b")])
    assert p.count("<|im_start|>user") == 3
    assert "<|im_start|>assistant\nA<|im_end|>" in p and "<|im_start|>assistant\nB<|im_end|>" in p
    assert p.index("estado 1") < p.index("estado 2") < p.index("estado 0")  # a pergunta real fica por último
    assert p.endswith("<|im_start|>assistant\n<think></think>")


def test_escolha_mesma_familia_outro_grupo_deterministica():
    alvo = caso(0, grupo="G")
    pool = [caso(i, grupo="G" if i < 3 else None) for i in range(1, 12)] + [caso(99, fam="outra")]
    ex = escolher_exemplos(alvo, pool, 4)
    assert len(ex) == 4
    assert all(e.family == "f" and e.group_id != "G" for e in ex)
    assert len({e.group_id for e in ex}) == 4
    assert [e.id for e in ex] == [e.id for e in escolher_exemplos(alvo, pool, 4)]
    assert [e.id for e in escolher_exemplos(alvo, pool, 2)] == [e.id for e in ex[:2]]  # K menor = prefixo
