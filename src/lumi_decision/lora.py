"""Treino LoRA sobre o MESMO readout da avaliação (H2), usado pelo E2 (sanidade) e pelo E3 (treino).

Readout de treino: backbone (use_cache=False, para o Mamba usar o kernel fundido de treino) -> estado da
última posição real (padding à direita) -> projeção FP32 no LM head congelado -> log-softmax no vocabulário
-> log-prob das letras candidatas. É o caminho de `HFBackend.readout_batch`, mas diferenciável.

Perda: entropia cruzada entre o alvo e a distribuição renormalizada entre as opções apresentadas (a mesma da
avaliação e da cabeça de ponteiro do E4). Alvo duro = one-hot no gold; alvo suave = `gold_dist` quando
existe (Telepatia), senão one-hot.

Lotes: micro-lotes por orçamento de tokens; as atualizações agrupam micro-lotes até >= batch_examples e a
perda é a média sobre o lote efetivo de cada atualização. O agendador usa o nº REAL de atualizações.
Checkpoint: pasta trocada de uma vez (pesos, otimizador, agendador, RNG, passo, config e hash dos dados);
retomar exige mesma config e mesmos dados.

Ordem das opções: sorteada por exemplo e por época (`shuffle<semente+época>`), contra o viés de posição.

Alvos do LoRA no Nemotron-H nativo: `in_proj` do Mamba, `up_proj`/`down_proj` do MLP, `q/k/v/o_proj` da
atenção. NÃO `out_proj` do Mamba: no treino o kernel fundido recebe `self.out_proj.weight` direto e um LoRA
ali não receberia gradiente (conferido no transformers 5.22.2; o E2 checa de novo).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import LOADERS, TRAIN_ONLY, drop_eval_overlap, load_cases, load_config, training_cases
from .prompts import (ABSTER, LETTERS, candidate_tokens, option_char_ends, option_token_positions, order_options,
                      render, with_abstain)
from .schema import Case

LORA_TARGETS = ["in_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


# ------------------------------------------------------------------ dados

def build_training_set(sources=("felhen", "telepatia", "traces"), cap_per_family: int = 4000,
                       seed: int = 0, source_caps: dict | None = None) -> list[Case]:
    """Casos de treino (só split train, só famílias vistas), com teto determinístico por família.
    `source_caps` troca o teto por família de uma fonte (F3: {"felhen": 650, "f3": 1625}). Com a mesma semente,
    o teto menor pega um prefixo da mesma ordenação: os 650 do Felhen estão entre os 1.300 do F2.1."""
    cfg = load_config()
    out = []
    eval_cases = None
    for src in sources:
        loaded = load_cases(src, cfg)
        if src in TRAIN_ONLY:  # fonte externa: tira textos que aparecem em qualquer avaliação das nossas fontes
            if eval_cases is None:
                eval_cases = [c for s in LOADERS for c in load_cases(s, cfg)]
            drop_eval_overlap(loaded, eval_cases)
        cases = training_cases(loaded)
        by = {}
        for c in cases:
            by.setdefault(c.family, []).append(c)
        cap = (source_caps or {}).get(src, cap_per_family)
        for fam in sorted(by):
            items = sorted(by[fam], key=lambda c: hashlib.sha256(f"{seed}:{c.id}".encode()).hexdigest())
            out.extend(items[:cap] if cap else items)
    bad = [c.id for c in out if c.held_out or c.split != "train"]
    if bad:
        raise RuntimeError(f"casos fora do treino permitido: {bad[:5]}")
    return out


def parse_source_caps(items) -> dict:
    """["felhen=650", "f3=1625"] -> {"felhen": 650, "f3": 1625} (formato da linha de comando)."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = int(v)
    return out


def build_adapt_set(familias, k: int, seed: int = 0, loader=None) -> list[Case]:
    """H9: k GRUPOS rotulados de cada família RESERVADA (`fonte:família`), para continuar um LoRA nela.

    Pool: os casos da família fora do `dev`/`test` (Felhen: o `train` das reservadas, nunca usado). Se não houver
    nenhum (Telepatia: as reservadas só existem no `dev`), usa o `dev`; aí esses grupos precisam sair da
    avaliação (`adapt_grupos.json`, lido por `scripts/analise_h9.py`). Sorteio por hash, determinístico."""
    loader = loader or (lambda src: load_cases(src, load_config()))
    cache, out = {}, []
    for item in familias:
        src, fam = item.split(":", 1)
        if src not in cache:
            cache[src] = loader(src)
        da_fam = [c for c in cache[src] if c.family == fam]
        if not da_fam:
            raise ValueError(f"família desconhecida: {item}")
        if not all(c.held_out for c in da_fam):
            raise ValueError(f"{item} não é família reservada")
        pool = [c for c in da_fam if c.split not in ("dev", "test")] or [c for c in da_fam if c.split == "dev"]
        grupos = sorted({c.group_id for c in pool}, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        if len(grupos) < k:
            raise ValueError(f"{item}: só {len(grupos)} grupos para k = {k}")
        esc = set(grupos[:k])
        out.extend(c for c in pool if c.group_id in esc)
    return out


def adapt_registro(cases: list[Case]) -> list[dict]:
    """O que vai para `adapt_grupos.json`: um registro por (fonte, família, split) com os group_ids usados."""
    by = {}
    for c in cases:
        by.setdefault((c.source, c.family, c.split), set()).add(c.group_id)
    return [{"fonte": s, "familia": f, "split": sp, "grupos": sorted(g)} for (s, f, sp), g in sorted(by.items())]


def training_set_for(cfg: "TrainConfig") -> list[Case]:
    """O conjunto de treino exatamente como o `train` monta (usado também pela contagem de orçamento)."""
    if cfg.adapt_k > 0:  # H9: só os k grupos rotulados das famílias reservadas
        return build_adapt_set(cfg.adapt_familias, cfg.adapt_k, cfg.seed if cfg.data_seed < 0 else cfg.data_seed)
    return build_training_set(cfg.sources, cfg.cap_per_family, cfg.seed if cfg.data_seed < 0 else cfg.data_seed,
                              parse_source_caps(cfg.source_caps))


def target_vector(case: Case, options, soft: bool) -> list[float]:
    ids = [o[0] for o in options]
    if soft and case.gold_dist:
        q = [float(case.gold_dist.get(i, 0.0)) for i in ids]
        z = sum(q)
        if z > 0:
            return [x / z for x in q]
    return [1.0 if i == case.gold else 0.0 for i in ids]


@dataclass
class Example:
    case_id: str
    family: str
    ids: list
    cands: list
    target: list
    gold_pos: int
    opt_pos: list | None = None  # E4: token do fim de cada opção (chaves da pointer head)
    abster: bool = False  # H5: a última opção é ABSTER


class Encoder:
    """Prompt -> ids e letras candidatas. Confere a fronteira das letras nos primeiros exemplos de cada
    número de opções e reaproveita os ids depois (o prompt sempre termina em `<think></think>`).
    Com `offsets` (texto -> (ids, offsets)), também localiza o token do fim de cada opção (E4)."""

    def __init__(self, tokenize, template, n_checks: int = 20, offsets=None):
        self.tokenize, self.template, self.n_checks = tokenize, template, n_checks
        self.offsets = offsets
        self.cache, self.checked = {}, Counter()

    def __call__(self, case: Case, order: str, soft: bool) -> Example:
        options = order_options(case, order)
        prompt = render(self.template, case, options)
        n = len(options)
        check = self.checked[n] < self.n_checks
        if check:
            ids, cands = candidate_tokens(self.tokenize, prompt, n)
            if n in self.cache and self.cache[n] != cands:
                raise RuntimeError(f"ids das letras mudaram para n={n}")
            self.cache[n] = cands
            self.checked[n] += 1
        elif self.offsets is None:
            ids, cands = self.tokenize(prompt), self.cache[n]
        else:
            ids, cands = None, self.cache[n]
        pos = None
        if self.offsets is not None:
            ids2, offs = self.offsets(prompt)
            if ids is not None and list(ids2) != list(ids):
                raise RuntimeError(f"{case.id}: tokenização com offsets difere da tokenização do readout")
            ids = list(ids2)
            pos = option_token_positions(offs, prompt, option_char_ends(prompt, case, options))
        tgt = target_vector(case, options, soft)
        return Example(case.id, case.family, ids, cands, tgt, [o[0] for o in options].index(case.gold), pos,
                       abster=options[-1][0] == ABSTER[0])


def plan_microbatches(examples: list[Example], token_budget: int, max_micro: int, seed: int,
                      pool: int = 256) -> list[list[int]]:
    """Índices em micro-lotes: embaralha, agrupa em blocos, ordena por comprimento dentro do bloco e
    empacota até `len(lote) * maior_comprimento <= token_budget`. Ordem final dos micro-lotes embaralhada."""
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), pool):
        block = sorted(idx[s:s + pool], key=lambda i: len(examples[i].ids))
        cur, cur_max = [], 0
        for i in block:
            L = len(examples[i].ids)
            m = max(cur_max, L)
            if cur and (m * (len(cur) + 1) > token_budget or len(cur) >= max_micro):
                batches.append(cur)
                cur, m = [], L
            cur.append(i)
            cur_max = m
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches




def group_steps(plan: list[list[int]], batch_examples: int) -> list[list[int]]:
    """Agrupa micro-lotes (na ordem do plano) em atualizações de >= batch_examples exemplos (a última pode ser
    menor). O número de atualizações e o tamanho efetivo de cada uma saem daqui, do plano real."""
    groups, cur, n = [], [], 0
    for mi, b in enumerate(plan):
        cur.append(mi)
        n += len(b)
        if n >= batch_examples:
            groups.append(cur)
            cur, n = [], 0
    if cur:
        groups.append(cur)
    return groups


# ------------------------------------------------------------------ modelo

def body_and_head(model):
    """(backbone, peso do LM head) de um NemotronHForCausalLM, com ou sem PEFT."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.base_model, base.get_output_embeddings().weight


def letter_logprobs(model, batch: list[Example], pad_id: int, head_fp32=None):
    """Log-probs (vocabulário inteiro) das letras candidatas, diferenciável. Devolve lista de tensores [n_i].

    `use_cache=False` é obrigatório: o config do Lumi tem use_cache=true e, com cache, o Mamba do Nemotron-H
    não usa o kernel fundido de treino (`self.training and cache_params is None`). O E2 confere o kernel.
    """
    import torch

    h, lens = hidden_states(model, batch, pad_id)
    _, W = body_and_head(model)
    h = h[torch.arange(len(batch), device=h.device), lens - 1].float()
    Wf = head_fp32 if head_fp32 is not None else W.float()
    logp = torch.log_softmax(h @ Wf.T, dim=-1)
    return [logp[i, e.cands] for i, e in enumerate(batch)]


def hidden_states(model, batch: list[Example], pad_id: int):
    """Estados finais do backbone [B, T, d] (padding à direita, `use_cache=False`) e os comprimentos."""
    import torch

    body, W = body_and_head(model)
    dev = W.device
    lens = torch.tensor([len(e.ids) for e in batch], device=dev)
    T = int(lens.max())
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros_like(ids)
    for i, e in enumerate(batch):
        ids[i, : len(e.ids)] = torch.tensor(e.ids, device=dev)
        mask[i, : len(e.ids)] = 1
    out = body(input_ids=ids, attention_mask=mask, use_cache=False)
    return (out[0] if isinstance(out, tuple) else out.last_hidden_state), lens


def option_scores(model, batch: list[Example], pad_id: int, head_fp32=None, pointer=None):
    """Notas NÃO normalizadas entre as opções, um tensor [n_i] por exemplo: letras do LM head (A1), logits da
    pointer head (B, E4) ou letras + pointer residual (F2.3a), numa única passada do backbone."""
    if pointer is None:
        return letter_logprobs(model, batch, pad_id, head_fp32)
    import torch

    from .ponteiro import pointer_logits

    h, lens = hidden_states(model, batch, pad_id)
    z = pointer_logits(h, lens, [e.opt_pos for e in batch], pointer)
    if not getattr(pointer, "residual", False):
        return z
    _, W = body_and_head(model)
    hl = h[torch.arange(len(batch), device=h.device), lens - 1].float()
    Wf = head_fp32 if head_fp32 is not None else W.float()
    logp = torch.log_softmax(hl @ Wf.T, dim=-1)
    return [logp[i, e.cands] + z[i] for i, e in enumerate(batch)]


LOSSES = ("log", "brier", "acerto", "acerto_brier", "bandit_log", "bandit_acerto")


def bandit_loss(lp_opt, q, kind: str = "bandit_log", n_samples: int = 8, clip: float = 10.0, explorar: float = 0.0,
                generator=None):
    """H6: perda com FEEDBACK DE BANDIT. O modelo sorteia `n_samples` opções e só vê a recompensa da opção
    sorteada, r(a) = q_a (1 se acertou, 0 se errou; com alvo suave, a chance de a opção estar certa).
    Nunca usa o gabarito das opções que não sorteou.

    - `bandit_acerto`: REINFORCE com recompensa u(a) = q_a e média do grupo como base. Estilo RLVR, não própria.
    - `bandit_log`: REINFORCE com recompensa u(a) = q_a / pi(a) (peso de importância, cortado em `clip`).
      Em valor esperado, sum_a pi(a)·(q_a/pi(a))·grad log pi(a) = sum_a q_a·grad log pi(a): o gradiente do LOG
      SCORE com o gabarito completo. Ou seja, uma regra própria aprendida só com o resultado da escolha.

    H6d, exploração (`explorar` = eps > 0): sorteia de mu = (1 - eps)·pi + eps/K em vez de pi, para que respostas
    a que o modelo dá quase zero ainda sejam testadas. A correção w = pi(a)/mu(a) mantém o mesmo gradiente
    esperado: E_mu[w·u·grad log pi] = E_pi[u·grad log pi]. No bandit_log, w·u = q_a/mu(a) <= K/eps: o peso não
    explode. Com eps = 0, mu = pi, w = 1 e tudo fica idêntico à H6.

    Base "deixa um de fora": cada amostra i é comparada com b_i = média das OUTRAS recompensas corrigidas.
    Como b_i não depende da amostra i, o termo b_i·w_i·grad log pi tem média zero (sum_a pi·grad log pi = 0):
    não muda o valor esperado e reduz a variância. (A média incluindo a própria amostra encolheria o gradiente
    por (G-1)/G.)
    """
    import torch

    p = lp_opt.detach().exp()
    if not 0.0 <= explorar < 1.0:
        raise ValueError(f"explorar deve estar em [0, 1): {explorar}")
    mu = (1.0 - explorar) * p + explorar / p.numel() if explorar > 0 else p
    a = torch.multinomial(mu, n_samples, replacement=True, generator=generator)
    u = q[a]
    if kind == "bandit_log":
        u = (u / p[a].clamp_min(1e-8)).clamp(max=clip)
    elif kind != "bandit_acerto":
        raise ValueError(f"perda de bandit desconhecida: {kind}")
    w = p[a] / mu[a].clamp_min(1e-12) if explorar > 0 else torch.ones_like(u)
    r = u * w
    base = (r.sum() - r) / (n_samples - 1) if n_samples > 1 else torch.zeros_like(r)
    adv = r - base * w
    return -(adv * lp_opt[a]).mean()


def option_loss(lp_opt, q, kind: str = "log"):
    """Perda de UM exemplo a partir dos log-probs renormalizados entre as opções (`lp_opt`) e do alvo `q`.

    H4 (a forma da recompensa muda a calibração?). Cada perda é menos uma recompensa:
    - `log`: -sum q·log p. Log score, regra própria e ilimitada. É a perda original do A1.
    - `brier`: sum (p - q)². Brier, regra própria e limitada (é o termo de calibração do Clef e do RLCR).
    - `acerto`: -sum q·p. Acerto esperado ao sortear a opção pela própria distribuição (estilo RLVR).
      É linear em p, logo NÃO é própria: o ótimo põe toda a massa numa opção (superconfiança).
    - `acerto_brier`: acerto + Brier, como a recompensa do RLCR (arXiv:2507.16806).
    Com alvo suave (`gold_dist`), q é a distribuição de referência; com alvo duro, o one-hot do gabarito.
    """
    if kind == "log":
        return -(q * lp_opt).sum()
    p = lp_opt.exp()
    if kind == "brier":
        return ((p - q) ** 2).sum()
    if kind == "acerto":
        return -(q * p).sum()
    if kind == "acerto_brier":
        return -(q * p).sum() + ((p - q) ** 2).sum()
    raise ValueError(f"perda desconhecida: {kind} (opções: {', '.join(LOSSES)})")


def abstain_loss(lp, target, gold_pos: int, lam: float, n_samples: int = 8, alpha: float = 1.0,
                 beta: float = 0.0, lp_ref=None, generator=None):
    """H5: perda de UM exemplo cuja última opção é ABSTER. `lp` = notas das K+1 letras (não normalizadas).

    Três partes que não brigam:
    - crença: log score nas K opções normais, renormalizadas sem o ABSTER (a probabilidade fica honesta);
    - decisão (RL): a ação é responder ou abster, com P(abster) = sigmoid(g), g = z_X - logsumexp(z_1..z_K).
      Sorteia `n_samples` ações; responder = a opção mais provável da crença. Recompensa: +1 se ela é o
      gabarito, -lam se não é (com alvo suave, o valor esperado q_k - lam·(1 - q_k)), 0 se abster.
      Vantagem = recompensa - média do grupo (estilo GRPO). O gradiente mexe só no logit do ABSTER
      (logsumexp das normais destacado), para o RL não puxar a crença para a opção mais provável;
    - coleira: beta·KL(pi || pi_ref) nas K+1 opções, com pi_ref = modelo sem adaptador.
    Devolve (perda, estatísticas {taxa_abster_amostrada, recompensa_media, p_abster}).
    """
    import torch

    K = lp.shape[0] - 1
    q = torch.as_tensor(target, device=lp.device, dtype=lp.dtype)[:K]
    q = q / q.sum().clamp_min(1e-12)
    lp_ans = lp[:K] - torch.logsumexp(lp[:K], dim=0)
    crenca = -(q * lp_ans).sum()
    g = lp[K] - torch.logsumexp(lp[:K], dim=0).detach()
    p_abs = torch.sigmoid(g.detach())
    a = torch.bernoulli(p_abs.expand(n_samples), generator=generator)  # 1 = abster
    k = int(lp_ans.detach().argmax())
    r_resp = q[k] - lam * (1 - q[k])
    rew = torch.where(a > 0, torch.zeros_like(a), r_resp.expand(n_samples))
    adv = rew - rew.mean()
    logp_a = torch.where(a > 0, torch.nn.functional.logsigmoid(g), torch.nn.functional.logsigmoid(-g))
    politica = -(adv * logp_a).mean()
    total = crenca + alpha * politica
    if beta and lp_ref is not None:
        lpa = lp - torch.logsumexp(lp, dim=0)
        lpr = lp_ref.detach() - torch.logsumexp(lp_ref.detach(), dim=0)
        total = total + beta * (lpa.exp() * (lpa - lpr)).sum()
    return total, {"taxa_abster_amostrada": float(a.mean()), "recompensa_media": float(rew.mean()),
                   "p_abster": float(p_abs)}


def batch_loss_abster(model, batch: list[Example], pad_id: int, head_fp32, cfg: "TrainConfig"):
    """H5: como `batch_loss`, para exemplos com ABSTER. A coleira faz um forward extra com o adaptador
    desligado (sem gradiente). Devolve (soma das perdas, acertos da crença, massa das letras, estatísticas)."""
    import torch

    lps = option_scores(model, batch, pad_id, head_fp32)
    refs = [None] * len(batch)
    if cfg.abster_beta:
        with torch.no_grad(), model.disable_adapter():
            refs = option_scores(model, batch, pad_id, head_fp32)
    losses, hits, mass = [], 0, 0.0
    st = {"taxa_abster_amostrada": 0.0, "recompensa_media": 0.0, "p_abster": 0.0}
    for lp, ref, e in zip(lps, refs, batch):
        if not e.abster:
            raise RuntimeError(f"{e.case_id}: exemplo sem ABSTER no treino da H5")
        l, s1 = abstain_loss(lp, e.target, e.gold_pos, cfg.abster_custo, cfg.abster_amostras, cfg.abster_alpha,
                             cfg.abster_beta, ref)
        losses.append(l)
        hits += int(int(lp[:-1].argmax()) == e.gold_pos)
        mass += float(lp.detach().exp().sum())
        for key in st:
            st[key] += s1[key] / len(batch)
    return torch.stack(losses).sum(), hits, mass / len(batch), st


def batch_loss(model, batch: list[Example], pad_id: int, head_fp32=None, pointer=None, loss: str = "log",
               bandit_clip: float = 10.0, bandit_amostras: int = 8, bandit_explorar: float = 0.0):
    """(soma das perdas, acertos, massa média das letras no vocabulário).

    Perda = entropia cruzada entre o alvo e a distribuição RENORMALIZADA entre as opções apresentadas
    (o mesmo objeto que a avaliação usa e que a pointer head do E4 produz). A massa das letras no vocabulário
    inteiro só é registrada (com a pointer head é 1 por construção). `loss` escolhe a perda da H4
    (`option_loss`); o padrão `log` reproduz o A1.
    """
    import torch

    lps = option_scores(model, batch, pad_id, head_fp32, pointer)
    losses, hits, mass = [], 0, 0.0
    for lp, e in zip(lps, batch):
        if pointer is not None and not getattr(pointer, "residual", False):
            lp = torch.log_softmax(lp, dim=0)
        lp_opt = lp - torch.logsumexp(lp, dim=0)
        q = torch.tensor(e.target, device=lp.device, dtype=lp.dtype)
        if loss.startswith("bandit_"):  # H6: só o resultado da opção sorteada
            losses.append(bandit_loss(lp_opt, q, loss, n_samples=bandit_amostras, clip=bandit_clip,
                                      explorar=bandit_explorar))
        else:
            losses.append(option_loss(lp_opt, q, loss))
        hits += int(int(lp.argmax()) == e.gold_pos)
        mass += float(lp.detach().exp().sum())
    return torch.stack(losses).sum(), hits, mass / len(batch)


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, targets=None):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets or LORA_TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def lora_modules(model) -> dict:
    """{nome do módulo: tipo (sufixo)} dos módulos com LoRA."""
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            out[name] = name.rsplit(".", 1)[-1]
    return out


# ------------------------------------------------------------------ treino (E3)

@dataclass
class TrainConfig:
    model: str = "ceia-br/lumi-4b"
    revision: str = "27b30f40b930fd05cee5e1c1f8790f70fa5b816d"
    sources: list = field(default_factory=lambda: ["felhen", "telepatia", "traces"])
    cap_per_family: int = 4000
    soft: bool = True
    epochs: int = 1
    lr: float = 1e-4
    warmup_frac: float = 0.03
    weight_decay: float = 0.0
    batch_examples: int = 32
    token_budget: int = 8192   # E2 36812: 16384 -> pico 72,8 GB de 80; 8192 -> 35 GB, mesma vazão
    max_micro: int = 32
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    seed: int = 0
    save_every: int = 100
    grad_clip: float = 1.0
    max_steps: int = 0  # > 0: para depois desse nº de atualizações (E3 curto); o agendador usa esse total
    readout: str = "letras"  # letras (A1, LM head) | ponteiro (B, E4)
    head_dim: int = 256      # E4: dimensão das projeções da pointer head (como no kev)
    head_lr: float = 0.0     # E4: lr da cabeça; 0 = a mesma do LoRA (padrão do kev, `--head_lr 0`)
    head_seed: int = 1       # E4: semente do gerador PRÓPRIO da cabeça (não mexe no RNG do LoRA)
    data_seed: int = -1      # F1: semente da SELEÇÃO dos exemplos (teto por família); -1 = a mesma de `seed`
    init_adapter: str = ""   # F2.3a: começa do LoRA de um adaptador já treinado (ex.: o do E3)
    epoch_offset: int = 0    # F2.3a: desloca as ordens/planos (época 0 daqui = época `epoch_offset` do E3)
    frozen_targets: list = field(default_factory=list)  # F2.2: alvos cujo LoRA nasce (B = 0) e fica CONGELADO
    source_caps: list = field(default_factory=list)  # F3: teto por família de uma fonte, ex.: felhen=650,f3=1625
    loss: str = "log"  # H4: log (A1) | brier | acerto | acerto_brier (ver `option_loss`)
    bandit_clip: float = 10.0   # H6: corte do peso 1/pi do bandit_log
    bandit_amostras: int = 8    # H6: opções sorteadas por caso (G)
    bandit_explorar: float = 0.0  # H6d: eps de exploração (sorteia de (1-eps)·pi + eps/K; 0 = desligado)
    abster_custo: float = 0.0   # H5: > 0 liga o braço (e); custo lambda de responder errado (abster vale 0)
    abster_amostras: int = 8    # H5: ações sorteadas por exemplo (grupo do GRPO)
    abster_alpha: float = 1.0   # H5: peso do termo de RL
    abster_beta: float = 0.02   # H5: peso do KL para o modelo sem adaptador
    adapt_familias: list = field(default_factory=list)  # H9: fonte:família reservadas para o treino curto
    adapt_k: int = 0            # H9: > 0 troca o treino por k grupos rotulados de cada família de adapt_familias


class _Stop:
    flag = False


def _on_signal(signum, frame):  # Slurm manda SIGUSR1 antes do limite de tempo (--signal=B:USR1@...)
    _Stop.flag = True


def data_fingerprint(cases, plans) -> str:
    h = hashlib.sha256()
    for c in cases:
        h.update(f"{c.id}|{c.gold}\n".encode())
    h.update(json.dumps(plans).encode())
    return h.hexdigest()


def _rng_state():
    import torch
    st = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    import torch
    random.setstate(st["python"])
    torch.set_rng_state(st["torch"].cpu())  # estados de RNG são ByteTensor na CPU, mesmo os da CUDA
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def save_checkpoint(ckpt: Path, model, opt, sched, state: dict, pointer=None) -> None:
    """Checkpoint como UNIDADE: grava tudo numa pasta temporária e troca de uma vez (rename atômico).
    Com a pointer head (E4), os pesos dela entram na mesma pasta."""
    import shutil

    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    tmp, old = ckpt.with_name(ckpt.name + ".tmp"), ckpt.with_name(ckpt.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({k: v.detach().contiguous() for k, v in get_peft_model_state_dict(model).items()},
              str(tmp / "adapter_model.safetensors"))
    if pointer is not None:
        save_file({k: v.detach().contiguous() for k, v in pointer.state_dict().items()}, str(tmp / "ponteiro.safetensors"))
    torch.save({"otimizador": opt.state_dict(), "agendador": sched.state_dict(), "rng": _rng_state()},
               tmp / "treino.pt")
    (tmp / "estado.json").write_text(json.dumps(state, indent=1))
    shutil.rmtree(old, ignore_errors=True)
    if ckpt.exists():
        os.replace(ckpt, old)
    os.replace(tmp, ckpt)
    shutil.rmtree(old, ignore_errors=True)


def train(cfg: TrainConfig, out: Path, resume: bool = True, *, model=None, tok=None, device: str = "cuda",
          cases=None, template=None, targets=None, _stop_after: int = 0) -> dict:
    """Treina e salva `out/adapter` + `out/CONCLUIDO`. Interrompido (SIGUSR1 ou arquivo `out/PARAR`): salva checkpoint e devolve
    interrompido=True. model/tok/cases/template/targets/_stop_after existem para os testes (modelo minúsculo)."""
    import shutil

    import torch

    from .data import ARTIFACTS_DIR
    from .evaluate import git_commit
    from .prompts import load_template

    _Stop.flag = False
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "PARAR"  # pedido de parada: criado pelo sbatch ao receber SIGUSR1; consumido ao parar
    ckpt = out / "checkpoint"
    if not ckpt.exists() and ckpt.with_name("checkpoint.old").exists():  # queda no meio da troca
        os.replace(ckpt.with_name("checkpoint.old"), ckpt)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
        model = AutoModelForCausalLM.from_pretrained(cfg.model, revision=cfg.revision, dtype=torch.bfloat16)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model = apply_lora(model, cfg.r, cfg.alpha, cfg.dropout, targets)
    mods = lora_modules(model)
    if any(t == "out_proj" for t in mods.values()):
        raise RuntimeError("LoRA em out_proj do Mamba não recebe gradiente no kernel fundido")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    _, W = body_and_head(model)
    head32 = W.detach().float()
    if cfg.abster_custo > 0 and (cfg.readout != "letras" or cfg.loss != "log"):
        raise ValueError("H5 (--abster-custo) só com --readout letras e --loss log")
    if cfg.loss not in LOSSES:
        raise ValueError(f"perda desconhecida: {cfg.loss} (opções: {', '.join(LOSSES)})")
    if cfg.readout not in ("letras", "ponteiro", "residual"):
        raise ValueError(f"readout desconhecido: {cfg.readout}")
    if cfg.init_adapter:  # F2.3a: continua um LoRA já treinado (o otimizador e o cronograma recomeçam)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(str(Path(cfg.init_adapter) / "adapter_model.safetensors"),
                                                         device=str(W.device)))
        if getattr(res, "unexpected_keys", None):
            raise RuntimeError(f"adaptador inicial não casa: {res.unexpected_keys[:3]}")
    frozen = {}
    if cfg.frozen_targets:
        # F2.2: o LoRA desses alvos é criado (mesma inicialização e mesmo consumo de RNG do dropout que o E3 da
        # mesma semente) mas não treina. Com B = 0, a saída desses módulos é exatamente a do modelo base.
        for name, mod in model.named_modules():
            if name.rsplit(".", 1)[-1] in cfg.frozen_targets and hasattr(mod, "lora_B") and "default" in mod.lora_B:
                for prm in list(mod.lora_A["default"].parameters()) + list(mod.lora_B["default"].parameters()):
                    prm.requires_grad_(False)
                frozen[name] = float(mod.lora_B["default"].weight.detach().abs().max())
        if not frozen:
            raise RuntimeError(f"nenhum módulo LoRA casa com frozen_targets={cfg.frozen_targets}")
        if not cfg.init_adapter and max(frozen.values()) != 0.0:
            raise RuntimeError("LoRA congelado com B != 0: o módulo não ficaria igual ao base")
    pointer = None
    if cfg.readout in ("ponteiro", "residual"):  # cabeça nova com gerador próprio; o LoRA nasce igual ao do E3
        from .ponteiro import new_head
        pointer = new_head(W.shape[1], cfg.head_dim, cfg.head_seed, residual=cfg.readout == "residual").to(W.device)

    cases = cases if cases is not None else training_set_for(cfg)
    if cfg.adapt_k > 0:  # H9: registra os grupos usados, para a análise tirá-los da avaliação
        (out / "adapt_grupos.json").write_text(json.dumps(adapt_registro(cases), indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    if cfg.abster_custo > 0:  # H5: "Não sei" vira a última opção de todo caso de treino
        cases = [with_abstain(c) for c in cases]

    def _offsets(t):
        e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]

    enc = Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"],
                  template or load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"),
                  offsets=_offsets if pointer is not None else None)

    # Planos de TODAS as épocas antes de começar: o nº real de atualizações define o agendador.
    epochs = []
    for ep in range(cfg.epochs):
        s = cfg.seed * 1000 + ep + cfg.epoch_offset
        exs = [enc(c, f"shuffle{s}", cfg.soft) for c in cases]
        plan = plan_microbatches(exs, cfg.token_budget, cfg.max_micro, s)
        epochs.append((exs, plan, group_steps(plan, cfg.batch_examples)))
    schedule = [(ep, gi) for ep, (_, _, groups) in enumerate(epochs) for gi in range(len(groups))]
    if cfg.max_steps:
        schedule = schedule[: cfg.max_steps]
    total = len(schedule)
    fp = data_fingerprint(cases, [p for _, p, _ in epochs])

    params = [p for p in model.parameters() if p.requires_grad]
    if pointer is None:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:  # mesmo cronograma (LambdaLR multiplica cada grupo); a cabeça pode ter lr própria
        hp = list(pointer.parameters())
        opt = torch.optim.AdamW([{"params": params, "lr": cfg.lr}, {"params": hp, "lr": cfg.head_lr or cfg.lr}],
                                weight_decay=cfg.weight_decay)
        params = params + hp
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))

    state = {"passo": 0, "config": asdict(cfg), "dados_sha256": fp, "passos_total": total}
    if resume and (ckpt / "estado.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        old = json.loads((ckpt / "estado.json").read_text())
        if old.get("config") != asdict(cfg) or old.get("dados_sha256") != fp:
            raise SystemExit(f"checkpoint em {ckpt} é de outra configuração ou outros dados; use outra pasta.\n"
                             f"antes: {old.get('config')} {old.get('dados_sha256')}\nagora: {asdict(cfg)} {fp}")
        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors"), device=str(W.device)))
        if pointer is not None:
            pointer.load_state_dict(load_file(str(ckpt / "ponteiro.safetensors"), device=str(W.device)))
        # Na CPU: map_location=cuda levava o RNG junto e set_rng_state recusava (E3 36850). O otimizador
        # move o próprio estado para o dispositivo dos parâmetros no load_state_dict.
        tr = torch.load(ckpt / "treino.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(tr["otimizador"])
        sched.load_state_dict(tr["agendador"])
        _set_rng_state(tr["rng"])
        state = old
        print(f">> retomando do passo {state['passo']}/{total}", flush=True)
    elif ckpt.exists() and not resume:
        shutil.rmtree(ckpt)

    meta = {"config": asdict(cfg), "git": git_commit(), "n_casos": len(cases), "dados_sha256": fp,
            "por_familia": dict(Counter(f"{c.source}|{c.family}" for c in cases)),
            "modulos_lora": dict(Counter(mods.values())), "params_treinaveis": sum(p.numel() for p in params),
            "modulos_congelados": dict(Counter(n.rsplit(".", 1)[-1] for n in frozen)),
            "congelados_max_abs_B": max(frozen.values()) if frozen else None,
            "passos_total": total, "micro_lotes_por_epoca": [len(p) for _, p, _ in epochs],
            "maior_prompt": max(len(e.ids) for e in epochs[0][0])}
    if pointer is not None and pointer.residual and state["passo"] == 0:
        # F2.3a: com W_k = 0 a leitura residual tem de ser EXATAMENTE a das letras do adaptador inicial
        exs0, plan0, _ = epochs[0]
        b0 = [exs0[i] for i in plan0[0]]
        model.eval()
        with torch.no_grad():
            zr = option_scores(model, b0, pad_id, head32, pointer)
            zl = letter_logprobs(model, b0, pad_id, head32)
        dif = max(float((a - b).abs().max()) for a, b in zip(zr, zl))
        meta["residual_inicial_max_dif"] = dif
        if dif != 0.0:
            raise RuntimeError(f"leitura residual inicial difere das letras ({dif})")
    (out / "treino_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    model.train()
    t0, toks = time.time(), 0
    with (out / "treino.jsonl").open("a", encoding="utf-8") as log:
        while state["passo"] < total:
            ep, gi = schedule[state["passo"]]
            exs, plan, groups = epochs[ep]
            micro = [[exs[i] for i in plan[mi]] for mi in groups[gi]]
            n_ex = sum(len(b) for b in micro)
            loss_t, hits_t, mass_t = 0.0, 0, 0.0
            abst = {}
            for b in micro:
                if cfg.abster_custo > 0:
                    loss_sum, hits, mass, s1 = batch_loss_abster(model, b, pad_id, head32, cfg)
                    for key, v in s1.items():
                        abst[key] = abst.get(key, 0.0) + v * len(b)
                else:
                    loss_sum, hits, mass = batch_loss(model, b, pad_id, head32, pointer, loss=cfg.loss,
                                                      bandit_clip=cfg.bandit_clip, bandit_amostras=cfg.bandit_amostras,
                                                      bandit_explorar=cfg.bandit_explorar)
                (loss_sum / n_ex).backward()  # média sobre o lote EFETIVO desta atualização
                loss_t, hits_t, mass_t = loss_t + float(loss_sum.detach()), hits_t + hits, mass_t + mass * len(b)
                toks += sum(len(e.ids) for e in b)
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state["passo"] += 1
            if state["passo"] % 10 == 0 or state["passo"] == total:
                rec = {"passo": state["passo"], "epoca": ep, "exemplos": n_ex, "perda": loss_t / n_ex,
                       "acc": hits_t / n_ex, "massa_letras": mass_t / n_ex, "lr": sched.get_last_lr()[0],
                       "norma_grad": float(gn), "tokens_s": toks / max(1e-9, time.time() - t0),
                       "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
                rec.update({key: v / n_ex for key, v in abst.items()})  # H5: taxa de abstenção e recompensa
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if _stop_after and state["passo"] == _stop_after:
                _Stop.flag = True
            if stop_file.exists():  # o Apptainer não repassa o SIGUSR1 ao processo (E3 36835): o sbatch cria o arquivo
                _Stop.flag = True
            if state["passo"] % cfg.save_every == 0 or _Stop.flag:
                save_checkpoint(ckpt, model, opt, sched, state, pointer)
            if _Stop.flag:
                stop_file.unlink(missing_ok=True)
                print(">> interrompido: checkpoint salvo; ressubmeta o mesmo comando para retomar", flush=True)
                return {"interrompido": True, "passo": state["passo"], "total": total}
    model.save_pretrained(str(out / "adapter"))
    if pointer is not None:
        from .ponteiro import save_head
        save_head(pointer, out / "adapter")
    save_checkpoint(ckpt, model, opt, sched, state, pointer)
    (out / "CONCLUIDO").write_text(json.dumps({"passo": state["passo"], "total": total}))
    return {"interrompido": False, "passo": state["passo"], "total": total}


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="lumi-treino", description="E3: LoRA sobre o readout das letras")
    d = TrainConfig()
    for k, v in asdict(d).items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "sim"), default=v)
        elif isinstance(v, list):
            p.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.split(","), default=v)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--out", required=True)
    p.add_argument("--sem-retomar", action="store_true", help="apaga o checkpoint da pasta e começa do zero")
    a = vars(p.parse_args(argv))
    out, no_resume = Path(a.pop("out")), a.pop("sem_retomar")
    res = train(TrainConfig(**a), out, resume=not no_resume)
    print(json.dumps(res), flush=True)
    raise SystemExit(3 if res["interrompido"] else 0)


if __name__ == "__main__":
    main()
