"""Backend Hugging Face (checkpoint oficial BF16). Requer CUDA + mamba-ssm para o Lumi.

Readout: roda só o backbone, pega o estado oculto da última posição real e projeta no LM head.
Isso evita o `forward` completo do Lumi, que aplica o LM head a todas as posições em FP32.

Cuidados verificados no código do Lumi (rev 27b30f4):
- o bloco não repassa `attention_mask` aos mixers: use padding à DIREITA (feito aqui) e leia
  a última posição real de cada linha;
- no treino com kernel fundido, LoRA em `out_proj` do Mamba não recebe gradiente (ver E2).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class HFBackend:
    def __init__(self, model: str | None = None, revision: str | None = None, dtype: str | None = "bfloat16",
                 device: str | None = None, trust_remote_code: bool = True, model_obj=None, tokenizer_obj=None,
                 eos_ids: list[int] | None = None, adapter: str | None = None, use_pointer: bool = True):
        self.model_name, self.revision = model, revision
        self.trust_remote_code = trust_remote_code
        self.adapter, self.adapter_sha256 = adapter, None
        self.pointer, self.pointer_sha256 = None, None  # E4: pointer head salva junto do adaptador
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loading_info = None
        if model_obj is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            tokenizer_obj = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)
            kw = dict(revision=revision, trust_remote_code=trust_remote_code, output_loading_info=True)
            try:
                model_obj, info = AutoModelForCausalLM.from_pretrained(model, dtype=torch_dtype, **kw)
            except TypeError:  # transformers < 4.56 usa torch_dtype
                model_obj, info = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype, **kw)
            # Pesos ausentes/inesperados = implementação não casa com o checkpoint (checado no E0).
            self.loading_info = {k: sorted(map(str, v))[:50] for k, v in info.items()}
            model_obj.to(self.device)
        if adapter:
            # LoRA SEM fundir: o PEFT injeta as camadas no próprio modelo; get_base_model() devolve o modelo
            # original já com elas, então readout/geração seguem o mesmo caminho de código.
            import hashlib
            from pathlib import Path

            from peft import PeftModel
            self.adapter_sha256 = hashlib.sha256((Path(adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
            model_obj = PeftModel.from_pretrained(model_obj, adapter, is_trainable=False).get_base_model()
            model_obj.to(self.device)
            from ..ponteiro import head_sha256, load_head
            if use_pointer:  # False: lê o adaptador pelas letras mesmo havendo cabeça (diagnóstico do E4)
                self.pointer = load_head(Path(adapter), self.device)
                self.pointer_sha256 = head_sha256(Path(adapter))
        self.model = model_obj.eval()
        self.tok = tokenizer_obj
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        # EOS: o config do Lumi diz 2 (</s>), o tokenizer diz <|im_end|> (11). Aceitamos ambos e
        # registramos qual apareceu (ver E0).
        self.eos_ids = eos_ids or sorted({i for i in (self.model.config.eos_token_id, self.tok.eos_token_id,
                                                      self.tok.convert_tokens_to_ids("<|im_end|>"))
                                          if isinstance(i, int) and i >= 0})

    # ------------------------------------------------------------ básico
    def tokenize(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _body(self):
        return self.model.base_model  # NemotronH: `backbone` (já inclui a norma final)

    def _head_weight(self):
        return self.model.get_output_embeddings().weight

    @torch.inference_mode()
    def last_hidden(self, batch_ids: list[list[int]], pad_id: int | None = None) -> torch.Tensor:
        """Estado oculto final na última posição real de cada sequência (padding à direita)."""
        lens = torch.tensor([len(x) for x in batch_ids], device=self.device)
        T = int(lens.max())
        pad = self.pad_id if pad_id is None else pad_id
        ids = torch.full((len(batch_ids), T), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(batch_ids):
            ids[i, : len(x)] = torch.tensor(x, device=self.device)
            mask[i, : len(x)] = 1
        out = self._body()(input_ids=ids, attention_mask=mask)
        h = out[0] if isinstance(out, tuple) else out.last_hidden_state
        return h[torch.arange(len(batch_ids), device=self.device), lens - 1]

    @torch.inference_mode()
    def pointer_readout_batch(self, batch_ids, batch_pos, pad_id: int | None = None, pointer=None, batch_cands=None):
        """E4: log-probs entre as opções pela pointer head (lista por exemplo), na mesma passada do readout.
        Cabeça residual (F2.3a): soma as log-probs das letras (`batch_cands`, obrigatório) às notas da cabeça."""
        from ..ponteiro import pointer_logits

        head = pointer if pointer is not None else self.pointer
        _sync()
        t0 = time.perf_counter()
        lens = torch.tensor([len(x) for x in batch_ids], device=self.device)
        T = int(lens.max())
        pad = self.pad_id if pad_id is None else pad_id
        ids = torch.full((len(batch_ids), T), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(batch_ids):
            ids[i, : len(x)] = torch.tensor(x, device=self.device)
            mask[i, : len(x)] = 1
        out = self._body()(input_ids=ids, attention_mask=mask)
        h = out[0] if isinstance(out, tuple) else out.last_hidden_state
        zs = pointer_logits(h, lens, batch_pos, head)
        if getattr(head, "residual", False):
            if batch_cands is None:
                raise ValueError("cabeça residual precisa das letras candidatas")
            hl = h[torch.arange(len(batch_ids), device=self.device), lens - 1].float()
            logp = torch.log_softmax(hl @ self._head_weight().float().T, dim=-1)
            zs = [logp[i, c] + z for i, (c, z) in enumerate(zip(batch_cands, zs))]
        res = [torch.log_softmax(z, dim=0).tolist() for z in zs]
        _sync()
        return res, {"latency_ms": (time.perf_counter() - t0) * 1000 / len(batch_ids), "batch": len(batch_ids)}

    def offsets(self, text: str):
        e = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]

    @torch.inference_mode()
    def readout_unmasked_at(self, batch_ids, read_lens, batch_cands):
        """Lote SEM padding (todas as linhas do mesmo comprimento, máscara toda 1), lendo a posição read_lens-1.

        Controle do E0: com sufixos de tokens reais no lugar do padding, o formato do lote é o mesmo do lote
        com padding; se o modelo for causal e a máscara não vazar, as posições lidas não mudam.
        """
        ids = torch.tensor(batch_ids, device=self.device)
        out = self._body()(input_ids=ids, attention_mask=torch.ones_like(ids))
        h = out[0] if isinstance(out, tuple) else out.last_hidden_state
        lens = torch.tensor(read_lens, device=self.device)
        h = h[torch.arange(len(batch_ids), device=self.device), lens - 1].float()
        logp = torch.log_softmax(h @ self._head_weight().float().T, dim=-1)
        return [logp[i, c].tolist() for i, c in enumerate(batch_cands)]

    @torch.inference_mode()
    def readout_batch(self, batch_ids, batch_cands, pad_id: int | None = None):
        _sync()
        t0 = time.perf_counter()
        h = self.last_hidden(batch_ids, pad_id=pad_id).float()
        logits = h @ self._head_weight().float().T  # só a última posição: [B, V]
        logp = torch.log_softmax(logits, dim=-1)
        res = [logp[i, c].tolist() for i, c in enumerate(batch_cands)]
        _sync()
        ms = (time.perf_counter() - t0) * 1000
        return res, {"latency_ms": ms / len(batch_ids), "batch": len(batch_ids)}

    def readout(self, prompt_ids, cand_ids):
        res, info = self.readout_batch([prompt_ids], [cand_ids])
        return res[0], info

    @torch.inference_mode()
    def full_forward_head_input(self, prompt_ids):
        """Forward completo, capturando a ENTRADA do LM head na última posição e os log-probs que o modelo devolve.

        Serve para checar que o readout lê a mesma posição e o mesmo estado (já normalizado) que o modelo
        projeta. A projeção em si difere de propósito: o readout faz em FP32; a implementação nativa faz em BF16.
        """
        got = {}
        h = self.model.get_output_embeddings().register_forward_hook(
            lambda mod, inp, out: got.__setitem__("x", inp[0].detach()))
        try:
            ids = torch.tensor([prompt_ids], device=self.device)
            logits = self.model(input_ids=ids).logits
        finally:
            h.remove()
        x = got["x"]
        return x[0, -1] if x.dim() == 3 else x[-1], torch.log_softmax(logits[0, -1].float(), dim=-1)

    @torch.inference_mode()
    def full_forward_last_logprobs(self, prompt_ids) -> torch.Tensor:
        """Referência do E0: forward completo, log-probs da última posição."""
        ids = torch.tensor([prompt_ids], device=self.device)
        return torch.log_softmax(self.model(input_ids=ids).logits[0, -1].float(), dim=-1)

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=16, stop_ids=None):
        """Geração gulosa. H11: `stop_ids` (ex.: o token `</think>`) também param a geração (parada "stop")."""
        ids = torch.tensor([prompt_ids], device=self.device)
        stop_ids = list(stop_ids or [])
        _sync()
        t0 = time.perf_counter()
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                  do_sample=False, eos_token_id=self.eos_ids + stop_ids, pad_token_id=self.pad_id)
        _sync()
        ms = (time.perf_counter() - t0) * 1000
        new = out[0, len(prompt_ids):].tolist()
        stop = ("stop" if new and new[-1] in stop_ids else
                "eos" if new and new[-1] in self.eos_ids else "limit")
        text = self.tok.decode([t for t in new if t not in self.eos_ids], skip_special_tokens=False)
        return new, text, {"latency_ms": ms, "stop_type": stop}

    @torch.inference_mode()
    def greedy_no_cache(self, prompt_ids, n_tokens=4) -> list[int]:
        """Geração gulosa recalculando tudo a cada passo (sem cache). Referência para o cache híbrido."""
        ids = list(prompt_ids)
        out = []
        for _ in range(n_tokens):
            nxt = int(self.full_forward_last_logprobs(ids).argmax())
            out.append(nxt)
            ids.append(nxt)
            if nxt in self.eos_ids:
                break
        return out

    def describe(self) -> dict:
        cfg = self.model.config
        return {"backend": "hf", "model": self.model_name, "revision": self.revision,
                "implementacao": "remote_code" if self.trust_remote_code else "nativa",
                "adapter": self.adapter, "adapter_sha256": self.adapter_sha256,
                "readout": ("residual" if getattr(self.pointer, "residual", False) else "ponteiro") if self.pointer is not None else "letras", "ponteiro_sha256": self.pointer_sha256,
                "classe": type(self.model).__module__ + "." + type(self.model).__name__, "device": self.device,
                "dtype": str(next(self.model.parameters()).dtype), "eos_ids": self.eos_ids,
                "model_type": getattr(cfg, "model_type", None),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__}
