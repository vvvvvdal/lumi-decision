"""H15: o RLCD do Clef (crédito parcial + penalidade de referência) sobre a leitura das letras."""

import pytest

torch = pytest.importorskip("torch")

from lumi_decision import lora  # noqa: E402
from lumi_decision.schema import Case  # noqa: E402

ESC = [["falso", "impreciso", "verdadeiro"]]


def _case(gold="falso", gold_dist=None, ids=("falso", "impreciso", "verdadeiro")):
    return Case(id="c", group_id="g", source="s", family="f", split="train", held_out=False, qtype="choice",
                state="x", question="q", options=[(i, i) for i in ids], gold=gold, gold_dist=gold_dist)


def test_credito_parcial_na_escala_e_fora_dela():
    c = _case("falso")
    opts = [("verdadeiro", ""), ("falso", ""), ("impreciso", "")]  # ordem apresentada embaralhada
    tgt = lora.target_vector(c, opts, soft=False)
    assert lora.credito_parcial(c, opts, tgt, ESC) == [0.0, 1.0, 0.5]
    assert lora.credito_parcial(_case("impreciso"), opts, tgt, ESC) == [0.5, 0.5, 1.0]
    outro = _case("a", ids=("a", "b", "c"))
    o2 = [("a", ""), ("b", ""), ("c", "")]
    assert lora.credito_parcial(outro, o2, [1.0, 0.0, 0.0], ESC) == [1.0, 0.0, 0.0]  # sem escala: o acerto
    suave = _case("falso", gold_dist={"falso": 0.6, "impreciso": 0.4})
    assert lora.credito_parcial(suave, opts, [0.0, 0.6, 0.4], ESC) == [0.0, 0.6, 0.4]  # alvo suave: o alvo


def test_rlcd_conta_a_mao():
    lp = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    q = torch.tensor([0.0, 1.0, 0.0])
    R = [0.5, 1.0, 0.0]
    esperado = -float(torch.log(torch.tensor(0.3))) - 2.0 * (0.5 * 0.5 + 0.3 * 1.0)
    assert float(lora.rlcd_loss(lp, q, R, "rlcd", lam=2.0)) == pytest.approx(esperado, rel=1e-6)


def test_rlcd_sem_termos_e_a_log_e_kl_zero_na_referencia():
    lp = torch.log_softmax(torch.tensor([0.2, -1.0, 0.7, 0.1]), 0)
    q = torch.tensor([0.0, 0.0, 1.0, 0.0])
    log = float(lora.option_loss(lp, q, "log"))
    assert float(lora.rlcd_loss(lp, q, None, "rlcd", lam=0.0, beta=0.0)) == pytest.approx(log)
    # a referência igual ao modelo: KL = 0, mesmo com beta > 0 (lp_ref não normalizado também serve)
    assert float(lora.rlcd_loss(lp, q, None, "rlcd", lam=0.0, beta=5.0, lp_ref=lp + 3.0)) == pytest.approx(log)
    ref = torch.log(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    assert float(lora.rlcd_loss(lp, q, None, "rlcd", lam=0.0, beta=1.0, lp_ref=ref)) > log


def test_amostrado_tem_o_gradiente_do_exato_em_media():
    z = torch.tensor([0.3, -0.5, 1.0], requires_grad=True)
    q = torch.tensor([1.0, 0.0, 0.0])
    R = [1.0, 0.5, 0.0]

    def grad(kind, gen=None):
        z.grad = None
        lp = torch.log_softmax(z, 0)
        lora.rlcd_loss(lp, q, R, kind, lam=1.0, n_samples=64, generator=gen).backward()
        return z.grad.clone()

    g_exato = grad("rlcd")
    gen = torch.Generator().manual_seed(0)
    g_media = torch.stack([grad("rlcd_amostrado", gen) for _ in range(400)]).mean(0)
    assert torch.allclose(g_media, g_exato, atol=0.02)


def test_rlcd_desconhecida_e_na_lista():
    assert "rlcd" in lora.LOSSES and "rlcd_amostrado" in lora.LOSSES
    with pytest.raises(ValueError):
        lora.rlcd_loss(torch.zeros(2), torch.tensor([1.0, 0.0]), None, "outra", lam=1.0)


def test_load_escalas(tmp_path):
    p = tmp_path / "e.toml"
    p.write_text('[[escala]]\nordem = ["a", "b", "c"]\n', encoding="utf-8")
    assert lora.load_escalas(p) == [["a", "b", "c"]]
    assert lora.load_escalas(tmp_path / "nao.toml") == []
