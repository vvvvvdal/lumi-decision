"""H5: opção ABSTER (sempre a última) e a perda `lora.abstain_loss` (crença honesta + decisão por RL)."""

import pytest

from lumi_decision.prompts import ABSTER, order_options, with_abstain
from lumi_decision.schema import Case


def caso(n=3):
    return Case(id="c1", group_id="g1", source="t", family="f", split="train", held_out=False, qtype="choice",
                state="estado", question="qual?", options=[(f"o{k}", f"opção {k}") for k in range(n)], gold="o1")


@pytest.mark.parametrize("ordem", ["canonical", "reversed", "shuffle3", "rot1", "perm201"])
def test_abster_fica_sempre_por_ultimo(ordem):
    c = with_abstain(caso())
    opts = order_options(c, ordem)
    assert opts[-1] == ABSTER
    assert sorted(o[0] for o in opts[:-1]) == ["o0", "o1", "o2"]
    assert [o[0] for o in opts[:-1]] == [o[0] for o in order_options(caso(), ordem)]  # mesma ordem sem o ABSTER


def test_with_abstain_idempotente_e_sem_mudar_o_original():
    c = caso()
    c2 = with_abstain(with_abstain(c))
    assert [o[0] for o in c2.options].count(ABSTER[0]) == 1
    assert len(c.options) == 3 and c2.gold == "o1"


torch = pytest.importorskip("torch")
from lumi_decision import lora  # noqa: E402


def test_politica_nao_mexe_na_crenca():
    """Com beta = 0, o gradiente nas K opções normais é exatamente o do log score (o RL só mexe no ABSTER)."""
    torch.manual_seed(0)
    z = torch.randn(4, dtype=torch.float64, requires_grad=True)
    alvo = [0.0, 1.0, 0.0, 0.0]
    perda, st = lora.abstain_loss(z, alvo, 1, lam=3.0, n_samples=16, alpha=1.0, beta=0.0)
    perda.backward()
    g_total = z.grad.clone()
    z2 = z.detach().clone().requires_grad_(True)
    lora.option_loss(torch.log_softmax(z2[:3], 0), torch.tensor(alvo[:3], dtype=torch.float64), "log").backward()
    assert torch.allclose(g_total[:3], z2.grad[:3])
    assert torch.isfinite(perda) and 0.0 <= st["p_abster"] <= 1.0


def _treina(p_real, lam, passos=2500, lote=16, seed=0):
    """Logits livres para UM tipo de caso cujo gabarito é sorteado de p_real; devolve (P(abster), crença)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(len(p_real) + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    p = torch.tensor(p_real, dtype=torch.float64)
    for _ in range(passos):
        opt.zero_grad()
        ys = torch.multinomial(p, lote, replacement=True, generator=g)
        perda = 0
        for y in ys.tolist():
            alvo = [1.0 if k == y else 0.0 for k in range(len(p_real))] + [0.0]
            perda = perda + lora.abstain_loss(z, alvo, y, lam=lam, n_samples=8, generator=g)[0]
        (perda / lote).backward()
        opt.step()
    zd = z.detach()
    return float(torch.sigmoid(zd[-1] - torch.logsumexp(zd[:-1], 0))), torch.softmax(zd[:-1], 0)


def test_aprende_a_abster_so_quando_compensa():
    """lambda = 3 -> compensa responder só se P(acerto) > 0,75. Caso difícil (0,6) abstém; fácil (0,9) responde.
    E a crença nas opções normais continua honesta (perto da distribuição verdadeira)."""
    dificil, facil = [0.6, 0.3, 0.1], [0.9, 0.05, 0.05]
    pa_d, crenca_d = _treina(dificil, lam=3.0)
    pa_f, crenca_f = _treina(facil, lam=3.0)
    assert pa_d > 0.8 and pa_f < 0.2
    assert torch.allclose(crenca_d, torch.tensor(dificil, dtype=torch.float64), atol=0.06)
    assert torch.allclose(crenca_f, torch.tensor(facil, dtype=torch.float64), atol=0.06)


def test_custo_baixo_responde_o_caso_dificil():
    """lambda = 1 -> limiar 0,5: o caso com 0,6 de acerto passa a valer a pena responder."""
    pa, _ = _treina([0.6, 0.3, 0.1], lam=1.0)
    assert pa < 0.2
