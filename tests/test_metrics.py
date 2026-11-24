import math

import pytest

from lumi_decision import metrics


def row(cid, gold, probs, order="canonical", group=None, **kw):
    return {"case_id": cid, "group_id": group or cid, "family": "f", "order": order, "gold": gold,
            "probs": probs, "pred": max(probs, key=probs.get), **kw}


def test_accuracy_and_balanced():
    rows = [row("1", "a", {"a": .9, "b": .1}), row("2", "a", {"a": .8, "b": .2}),
            row("3", "a", {"a": .7, "b": .3}), row("4", "b", {"a": .6, "b": .4})]
    assert metrics.accuracy(rows) == 0.75
    assert metrics.balanced_accuracy(rows) == pytest.approx((1.0 + 0.0) / 2)


def test_nll_brier_perfect_and_uniform():
    perfect = [row("1", "a", {"a": 1.0, "b": 0.0})]
    assert metrics.nll(perfect) == pytest.approx(0.0, abs=1e-9)
    assert metrics.brier(perfect) == 0.0
    uniform = [row("1", "a", {"a": .5, "b": .5})]
    assert metrics.nll(uniform) == pytest.approx(math.log(2))
    assert metrics.brier(uniform) == pytest.approx(0.5)


def test_ece_zero_when_calibrated():
    rows = [row(str(i), "a" if i < 8 else "b", {"a": .8, "b": .2}) for i in range(10)]
    assert metrics.ece(rows) == pytest.approx(0.0, abs=1e-9)


def test_risk_coverage_ordering():
    rows = [row("1", "a", {"a": .99, "b": .01}), row("2", "a", {"a": .9, "b": .1}),
            row("3", "b", {"a": .6, "b": .4}), row("4", "b", {"a": .55, "b": .45})]
    rc = metrics.risk_coverage(rows)
    assert rc["risco_global"] == 0.5
    assert rc["risco@cob50"] == 0.0
    assert rc["cobertura@risco5"] == 0.5


def test_permutation_stats_detects_swaps():
    rows = [
        row("1", "a", {"a": .6, "b": .4}, "canonical"), row("1", "a", {"a": .4, "b": .6}, "reversed"),
        row("2", "b", {"a": .4, "b": .6}, "canonical"), row("2", "b", {"a": .3, "b": .7}, "reversed"),
        row("3", "a", {"a": .4, "b": .6}, "canonical"), row("3", "a", {"a": .7, "b": .3}, "reversed"),
    ]
    p = metrics.permutation_stats(rows)
    assert p["acuracia_por_ordem"] == {"canonical": pytest.approx(2 / 3), "reversed": pytest.approx(2 / 3)}
    assert p["decisao_muda_entre_ordens"] == pytest.approx(2 / 3)
    assert p["retencao_dos_acertos"] == pytest.approx(0.5)
    assert p["acerta_em_todas_as_ordens"] == pytest.approx(1 / 3)


def test_paired_bootstrap_sign():
    a = [row(str(i), "a", {"a": .4, "b": .6}) for i in range(50)]
    b = [row(str(i), "a", {"a": .6, "b": .4}) for i in range(50)]
    res = metrics.paired_bootstrap(a, b, n_boot=200)
    assert res["delta"] == 1.0 and res["ic95"][0] > 0


def test_soft_metrics():
    rows = [row("1", "a", {"a": .5, "b": .5}, gold_dist={"a": .5, "b": .5})]
    s = metrics.soft_metrics(rows)
    assert s["tv_vs_ref"] == pytest.approx(0.0)
    assert s["cross_entropy_vs_ref"] == pytest.approx(math.log(2))


def test_temperature_keeps_argmax_and_fits_overconfidence():
    import random
    rnd = random.Random(0)
    rows = []
    for i in range(400):  # modelo superconfiante: diz 0.99, acerta 70%
        gold = "a" if rnd.random() < 0.7 else "b"
        rows.append(row(str(i), gold, {"a": .99, "b": .01}))
    T = metrics.fit_temperature(rows)
    assert T > 1.5
    scaled = metrics.with_temperature(rows, T)
    assert all(r["pred"] == s["pred"] for r, s in zip(rows, scaled))
    assert metrics.ece(scaled) < metrics.ece(rows)


def test_threshold_then_fixed_application():
    rows = [row(str(i), "a", {"a": 0.9, "b": 0.1}) for i in range(30)] + \
           [row(f"e{i}", "b", {"a": 0.6, "b": 0.4}) for i in range(30)]
    tau = metrics.threshold_for_risk(rows, 0.05)
    assert tau == 0.9
    sel = metrics.selective_at(rows, tau)
    assert sel["cobertura"] == 0.5 and sel["risco_aceitos"] == 0.0 and sel["risco_aleatorio"] == 0.5


def test_paired_bootstrap_strict_refuses_missing_pairs():
    a = [row("1", "a", {"a": .6, "b": .4}), row("2", "a", {"a": .6, "b": .4})]
    b = [row("1", "a", {"a": .6, "b": .4})]
    with pytest.raises(ValueError, match="não correspondem"):
        metrics.paired_bootstrap(a, b)
    assert metrics.paired_bootstrap(a, b, strict=False)["n_itens"] == 1


def test_order_mean_accuracy_is_mean_of_per_order():
    rows = [row("1", "a", {"a": .6, "b": .4}, "canonical"), row("1", "a", {"a": .4, "b": .6}, "reversed"),
            row("2", "a", {"a": .6, "b": .4}, "canonical"), row("2", "a", {"a": .6, "b": .4}, "reversed")]
    r = metrics.order_mean_accuracy(rows)
    assert r["por_ordem"] == {"canonical": 1.0, "reversed": 0.5} and r["media"] == 0.75
    with pytest.raises(ValueError):
        metrics.order_mean_accuracy(rows[:3])


def test_family_mean_bootstrap_weights_families_equally():
    # família grande sem ganho (100 grupos), família pequena com ganho total (10 grupos)
    a, b = [], []
    for i in range(100):
        a.append(dict(row(f"g{i}", "a", {"a": .6, "b": .4}), family="grande"))
        b.append(dict(row(f"g{i}", "a", {"a": .6, "b": .4}), family="grande"))
    for i in range(10):
        a.append(dict(row(f"p{i}", "a", {"a": .4, "b": .6}), family="pequena"))
        b.append(dict(row(f"p{i}", "a", {"a": .6, "b": .4}), family="pequena"))
    r = metrics.family_mean_bootstrap(a, b, n_boot=200)
    assert r["delta_media_familias"] == pytest.approx(0.5)
    assert r["por_familia"] == {"grande": 0.0, "pequena": 1.0}


def test_pairing_refuses_duplicates_and_divergent_pairs():
    a = [row("1", "a", {"a": .6, "b": .4}, prompt_sha1="x"), row("2", "a", {"a": .6, "b": .4}, prompt_sha1="y")]
    with pytest.raises(ValueError, match="repetido"):
        metrics.paired_bootstrap(a + [a[0]], a)
    b = [dict(a[0], gold="b"), a[1]]
    with pytest.raises(ValueError, match="'gold'"):
        metrics.paired_bootstrap(a, b)
    c = [dict(a[0], prompt_sha1="z"), a[1]]
    with pytest.raises(ValueError, match="'prompt_sha1'"):
        metrics.family_mean_bootstrap(a, c)


def test_family_mean_requires_families():
    a = [dict(row(str(i), "a", {"a": .6, "b": .4}), family="f1") for i in range(5)]
    with pytest.raises(ValueError, match="exigidas"):
        metrics.family_mean_bootstrap(a, a, n_boot=10, require_families=["f1", "f2"])
    assert metrics.family_mean_bootstrap(a, a, n_boot=10, require_families=["f1"])["delta_media_familias"] == 0


def test_auroc_bate_com_a_conta_par_a_par():
    import random

    from lumi_decision.metrics import auroc

    def par_a_par(conf, y):
        pos = [c for c, a in zip(conf, y) if a]
        neg = [c for c, a in zip(conf, y) if not a]
        return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))

    rnd = random.Random(0)
    for _ in range(30):
        n = rnd.randint(2, 40)
        conf = [rnd.choice([0.1, 0.5, 0.5, 0.9, rnd.random()]) for _ in range(n)]
        y = [rnd.random() < 0.6 for _ in range(n)]
        if all(y) or not any(y):
            continue
        assert abs(auroc(conf, y) - par_a_par(conf, y)) < 1e-12
    assert auroc([0.9, 0.8, 0.2], [True, True, False]) == 1.0
    assert auroc([1.0, 1.0], [True, False]) == 0.5
    assert math.isnan(auroc([0.5, 0.6], [True, True]))
