"""H6: perdas com feedback de bandit (`lora.bandit_loss`): só o resultado da opção sorteada."""

import pytest

torch = pytest.importorskip("torch")

from lumi_decision import lora  # noqa: E402


def test_bandit_log_tem_o_gradiente_do_log_score_em_media():
    """Média de muitos gradientes do bandit_log ≈ gradiente do log score com o gabarito completo."""
    g = torch.Generator().manual_seed(0)
    z0 = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    q = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    acc = torch.zeros(3, dtype=torch.float64)
    n = 4000
    for _ in range(n):
        z = z0.clone().requires_grad_(True)
        lora.bandit_loss(torch.log_softmax(z, 0), q, "bandit_log", n_samples=8, clip=1e9, generator=g).backward()
        acc += z.grad
    z = z0.clone().requires_grad_(True)
    lora.option_loss(torch.log_softmax(z, 0), q, "log").backward()
    assert torch.allclose(acc / n, z.grad, atol=0.03)


def _treina(kind, p_real=(0.6, 0.3, 0.1), passos=3000, lote=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.03)
    p = torch.tensor(p_real, dtype=torch.float64)
    for _ in range(passos):
        opt.zero_grad()
        perda = 0
        for y in torch.multinomial(p, lote, replacement=True, generator=g).tolist():
            q = torch.zeros(3, dtype=torch.float64)
            q[y] = 1.0
            perda = perda + lora.bandit_loss(torch.log_softmax(z, 0), q, kind, n_samples=8, generator=g)
        (perda / lote).backward()
        opt.step()
    return torch.softmax(z.detach(), 0)


def test_bandit_log_fica_honesto():
    assert torch.allclose(_treina("bandit_log"), torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64), atol=0.06)


def test_bandit_acerto_fica_superconfiante():
    assert float(_treina("bandit_acerto")[0]) > 0.9


def test_perda_desconhecida():
    with pytest.raises(ValueError):
        lora.bandit_loss(torch.log_softmax(torch.zeros(3), 0), torch.tensor([1.0, 0, 0]), "bandit_brier")
