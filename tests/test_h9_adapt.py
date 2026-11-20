"""H9: conjunto de treino curto com k grupos das famílias reservadas (`lora.build_adapt_set`)."""

import pytest

from lumi_decision import lora
from lumi_decision.schema import Case


def c(i, fam, split, held=True, grupo=None):
    return Case(id=f"{fam}:{split}:{i}", group_id=grupo or f"{fam}:{split}:g{i}", source="s", family=fam,
                split=split, held_out=held, qtype="choice", state="x", question="q",
                options=[("a", "A"), ("b", "B")], gold="a")


CASOS = ([c(i, "comtrain", "train") for i in range(30)] + [c(i, "comtrain", "dev") for i in range(10)]
         + [c(i, "sodev", "dev", grupo=f"sodev:g{i // 2}") for i in range(40)]  # 2 perguntas por grupo
         + [c(i, "vista", "train", held=False) for i in range(10)])


def loader(src):
    return CASOS


def test_k_grupos_por_familia_do_split_certo():
    out = lora.build_adapt_set(["s:comtrain", "s:sodev"], 5, loader=loader)
    com = [x for x in out if x.family == "comtrain"]
    so = [x for x in out if x.family == "sodev"]
    assert len({x.group_id for x in com}) == 5 and all(x.split == "train" for x in com)  # nunca o dev se há train
    assert len({x.group_id for x in so}) == 5 and len(so) == 10 and all(x.split == "dev" for x in so)
    assert all(x.held_out for x in out)


def test_deterministico_e_k_menor_e_subconjunto():
    a = lora.build_adapt_set(["s:comtrain"], 5, loader=loader)
    assert [x.id for x in a] == [x.id for x in lora.build_adapt_set(["s:comtrain"], 5, loader=loader)]
    b = {x.group_id for x in lora.build_adapt_set(["s:comtrain"], 3, loader=loader)}
    assert b <= {x.group_id for x in a}


def test_familia_vista_ou_desconhecida_da_erro():
    with pytest.raises(ValueError):
        lora.build_adapt_set(["s:vista"], 2, loader=loader)
    with pytest.raises(ValueError):
        lora.build_adapt_set(["s:nao_existe"], 2, loader=loader)
    with pytest.raises(ValueError):
        lora.build_adapt_set(["s:comtrain"], 31, loader=loader)


def test_registro_lista_os_grupos():
    reg = lora.adapt_registro(lora.build_adapt_set(["s:sodev"], 4, loader=loader))
    assert reg == [{"fonte": "s", "familia": "sodev", "split": "dev", "grupos": reg[0]["grupos"]}]
    assert len(reg[0]["grupos"]) == 4


def test_sem_adapt_o_treino_normal_nao_muda(monkeypatch):
    chamado = {}
    monkeypatch.setattr(lora, "build_training_set", lambda *a, **k: chamado.setdefault("ok", ["normal"]))
    assert lora.training_set_for(lora.TrainConfig()) == ["normal"]
