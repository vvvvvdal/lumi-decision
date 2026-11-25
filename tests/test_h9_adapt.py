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


def test_replay_rodizio_entre_familias_e_deterministico():
    pool = ([c(i, "grande", "train", held=False) for i in range(50)]
            + [c(i, "pequena", "train", held=False) for i in range(3)])
    r = lora.replay_set(pool, 10, seed=0)
    fams = [x.family for x in r]
    assert len(r) == 10 and fams.count("pequena") == 3  # a pequena entra inteira antes de a grande dominar
    assert [x.id for x in r] == [x.id for x in lora.replay_set(pool, 10, seed=0)]
    assert len(lora.replay_set(pool, 100, seed=0)) == 53  # não passa do que existe


def test_training_set_for_com_replay(monkeypatch):
    vistos = [c(i, "vista", "train", held=False) for i in range(40)]
    monkeypatch.setattr(lora, "build_adapt_set", lambda fams, k, seed, **kw: [c(i, "nova", "dev") for i in range(k)])
    monkeypatch.setattr(lora, "build_training_set", lambda *a, **kw: vistos)
    cfg = lora.TrainConfig(adapt_familias=["s:nova"], adapt_k=6, adapt_replay=1.0)
    out = lora.training_set_for(cfg)
    assert sum(x.held_out for x in out) == 6 and sum(not x.held_out for x in out) == 6
    cfg0 = lora.TrainConfig(adapt_familias=["s:nova"], adapt_k=6)
    assert all(x.held_out for x in lora.training_set_for(cfg0))  # sem replay = H9 igual


def test_duvida_escolhe_na_faixa_de_menor_confianca():
    # comtrain: 30 grupos no train; confiança cresce com o índice (g0 = mais dúvida)
    duv = {f"comtrain:train:g{i}": 0.3 + 0.02 * i for i in range(30)}
    out = lora.build_adapt_set(["s:comtrain"], 4, loader=loader, duvida=duv, faixa=3)
    grupos = {x.group_id for x in out}
    assert len(grupos) == 4 and grupos <= {f"comtrain:train:g{i}" for i in range(12)}  # só os 3×4 mais duvidosos
    # sem predição para o pool inteiro: só entram os grupos com predição
    poucos = {f"comtrain:train:g{i}": 0.5 for i in range(5)}
    with pytest.raises(ValueError):
        lora.build_adapt_set(["s:comtrain"], 4, loader=loader, duvida=poucos, faixa=3)


def test_load_duvida_media_por_grupo(tmp_path):
    import json
    rows = [{"group_id": "g1", "probs": {"a": 0.9, "b": 0.1}}, {"group_id": "g1", "probs": {"a": 0.3, "b": 0.7}},
            {"group_id": "g2", "probs": {"a": 0.5, "b": 0.5}}]
    (tmp_path / "predicoes.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    d = lora.load_duvida([str(tmp_path)])
    assert d == pytest.approx({"g1": 0.8, "g2": 0.5})


def test_training_set_for_duvida_exige_preds():
    cfg = lora.TrainConfig(adapt_familias=["s:nova"], adapt_k=6, adapt_selecao="duvida")
    with pytest.raises(ValueError):
        lora.training_set_for(cfg)


def test_k_por_familia():
    """H14: `fonte:família=k` troca o k só daquela família; as outras ficam com o k padrão."""
    out = lora.build_adapt_set(["s:comtrain=8", "s:sodev"], 5, loader=loader)
    assert len({x.group_id for x in out if x.family == "comtrain"}) == 8
    assert len({x.group_id for x in out if x.family == "sodev"}) == 5
    # os 5 primeiros do sorteio continuam dentro dos 8 (mesmo hash): subir o k só acrescenta grupos
    k5 = {x.group_id for x in lora.build_adapt_set(["s:comtrain"], 5, loader=loader)}
    assert k5 <= {x.group_id for x in out if x.family == "comtrain"}
    with pytest.raises(ValueError):
        lora.build_adapt_set(["s:comtrain=999"], 5, loader=loader)
    with pytest.raises(ValueError):
        lora.build_adapt_set(["s:comtrain=0"], 5, loader=loader)
