"""H17: o BFCL como decisão tipada (`harness.bfcl_cases`), com arquivos pequenos de exemplo."""

import json
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
from lumi_decision import harness  # noqa: E402


def _escrever(base: Path, nome: str, linhas):
    p = base / nome
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x) + "\n" for x in linhas), encoding="utf-8")


def _f(nome, desc="faz algo"):
    return {"name": nome, "description": desc, "parameters": {}}


def _q(texto):
    return [[{"role": "user", "content": texto}]]


@pytest.fixture()
def base(tmp_path):
    _escrever(tmp_path, "BFCL_v3_multiple.json", [
        {"id": "multiple_0", "question": _q("área do triângulo"), "function": [_f("tri.get"), _f("circ.get")]},
        {"id": "multiple_1", "question": _q("x"), "function": [_f("a"), _f("a")]},  # nomes repetidos
    ])
    _escrever(tmp_path, "possible_answer/BFCL_v3_multiple.json", [
        {"id": "multiple_0", "ground_truth": [{"tri.get": {"lado": [3]}}]},
        {"id": "multiple_1", "ground_truth": [{"a": {}}]},
    ])
    _escrever(tmp_path, "BFCL_v3_live_multiple.json", [
        {"id": "live_multiple_0", "question": _q("y"), "function": [_f(f"f{i}") for i in range(30)]},  # opções demais
    ])
    _escrever(tmp_path, "possible_answer/BFCL_v3_live_multiple.json", [
        {"id": "live_multiple_0", "ground_truth": [{"f1": {}}]},
    ])
    _escrever(tmp_path, "BFCL_v3_irrelevance.json", [
        {"id": "irrelevance_0", "question": _q("clima amanhã"), "function": [_f("imc", "x " * 300)]},
    ])
    _escrever(tmp_path, "BFCL_v3_live_irrelevance.json", [
        {"id": "live_irrelevance_0", "question": _q("oi"), "function": []},  # sem funções
    ])
    return tmp_path


def test_bfcl_vira_decisao_tipada(base):
    cs = {c.id: c for c in harness.bfcl_cases(base=base, conferir=False)}
    m = cs["bfcl:multiple_0"]
    assert m.split == "dev" and m.held_out and m.family == "multiple" and m.gold == "tri.get"
    assert [o[0] for o in m.options] == ["tri.get", "circ.get", "nenhuma"]
    assert m.state == "Usuário: área do triângulo"
    irr = cs["bfcl:irrelevance_0"]
    assert irr.gold == "nenhuma" and irr.split == "dev"
    assert len(irr.options[0][1]) <= len("imc: ") + harness.MAX_DESC  # descrição cortada
    assert cs["bfcl:multiple_1"].split == "descartado" and "repetidos" in cs["bfcl:multiple_1"].meta["descarte"]
    assert cs["bfcl:live_multiple_0"].split == "descartado"
    assert cs["bfcl:live_irrelevance_0"].meta["descarte"] == "sem funções no menu"


def test_bfcl_confere_o_sha256(base):
    with pytest.raises(ValueError, match="revisão fixada"):
        harness.bfcl_cases(base=base)
