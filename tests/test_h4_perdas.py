"""H4: as perdas de `lora.option_loss` (sem modelo; só a conta sobre os log-probs das opções)."""

import math

import pytest

torch = pytest.importorskip("torch")

from lumi_decision import lora  # noqa: E402


def lp(*p):
    return torch.log(torch.tensor(p, dtype=torch.float64))


def test_log_reproduz_a_perda_original():
    q = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    x = lp(0.2, 0.5, 0.3)
    assert float(lora.option_loss(x, q, "log")) == pytest.approx(-math.log(0.5))
    assert float(lora.option_loss(x, q)) == pytest.approx(-math.log(0.5))  # padrão = A1


def test_valores_de_brier_e_acerto():
    q = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    x = lp(0.2, 0.5, 0.3)
    brier = 0.2 ** 2 + 0.5 ** 2 + 0.3 ** 2
    assert float(lora.option_loss(x, q, "brier")) == pytest.approx(brier)
    assert float(lora.option_loss(x, q, "acerto")) == pytest.approx(-0.5)
    assert float(lora.option_loss(x, q, "acerto_brier")) == pytest.approx(-0.5 + brier)


def _otimo(kind, q, passos=3000, lr=0.5, suavizar=0.0):
    """Minimiza a perda esperada sob o alvo suave q, otimizando logits livres."""
    z = torch.zeros(len(q), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([z], lr=lr)
    for _ in range(passos):
        opt.zero_grad()
        lora.option_loss(torch.log_softmax(z, 0), q, kind, suavizar).backward()
        opt.step()
    return torch.softmax(z, 0).detach()


@pytest.mark.parametrize("kind", ["log", "brier"])
def test_regras_proprias_recuperam_a_distribuicao(kind):
    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    assert torch.allclose(_otimo(kind, q), q, atol=0.02)


def test_acerto_e_superconfiante():
    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    assert float(_otimo("acerto", q)[0]) > 0.95


def test_perda_desconhecida():
    with pytest.raises(ValueError):
        lora.option_loss(lp(0.5, 0.5), torch.tensor([1.0, 0.0], dtype=torch.float64), "hinge")


def test_log_brier_valor_a_mao():
    q = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    x = lp(0.2, 0.5, 0.3)
    brier = 0.2 ** 2 + 0.5 ** 2 + 0.3 ** 2
    assert float(lora.option_loss(x, q, "log_brier")) == pytest.approx(-math.log(0.5) + brier)
    # suavizar 0,3 com K = 3: alvo da entropia cruzada = (0,1; 0,8; 0,1); o Brier continua com o alvo duro
    ce = -(0.1 * math.log(0.2) + 0.8 * math.log(0.5) + 0.1 * math.log(0.3))
    assert float(lora.option_loss(x, q, "log_brier", 0.3)) == pytest.approx(ce + brier)


def test_log_brier_sem_suavizar_e_propria():
    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    assert torch.allclose(_otimo("log_brier", q), q, atol=0.02)


def test_log_brier_suavizado_fica_menos_confiante():
    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    p = _otimo("log_brier", q, suavizar=0.1)
    assert float(p[0]) < 0.6 - 0.01 and float(p[2]) > 0.1 + 0.005  # puxa para o uniforme
    assert int(p.argmax()) == 0  # sem trocar a opção escolhida


def test_suavizar_nao_mexe_nas_outras_perdas():
    q = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    x = lp(0.2, 0.5, 0.3)
    for kind in ("log", "brier", "acerto", "acerto_brier"):
        assert float(lora.option_loss(x, q, kind, 0.3)) == pytest.approx(float(lora.option_loss(x, q, kind)))
