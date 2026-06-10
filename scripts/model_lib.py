"""
Modulo reutilizavel: arquitetura nanoGPT + carregamento seguro + tokenizer byte-level.
Sem efeitos secundarios ao importar. Tudo local.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

CKPT = r"C:\Users\migue\Documents\augusta-labs-arcus\artifacts\ode.pt"

SPECIALS = {256: "<|fernando_pessoa|>", 257: "<|alberto_caeiro|>",
            258: "<|ricardo_reis|>", 259: "<|bernardo_soares|>",
            260: "_", 261: "{"}
# nomes -> id (para condicionar)
HETERONYMS = {"fernando_pessoa": 256, "alberto_caeiro": 257,
              "ricardo_reis": 258, "bernardo_soares": 259}


def _build(cfg):
    VOCAB = cfg["vocab_size"]; BLOCK = cfg["block_size"]
    NL = cfg["n_layer"]; NH = cfg["n_head"]; NE = cfg["n_embd"]; BIAS = cfg["bias"]
    DROP = cfg["dropout"]

    class LayerNorm(nn.Module):
        def __init__(self, ndim, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        def forward(self, x):
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(NE, 3*NE, bias=BIAS)
            self.c_proj = nn.Linear(NE, NE, bias=BIAS)
            self.resid_dropout = nn.Dropout(DROP)
            self.n_head = NH; self.n_embd = NE
        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.c_proj(y))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = nn.Linear(NE, 4*NE, bias=BIAS)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4*NE, NE, bias=BIAS)
            self.dropout = nn.Dropout(DROP)
        def forward(self, x):
            return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = LayerNorm(NE, BIAS); self.attn = CausalSelfAttention()
            self.ln_2 = LayerNorm(NE, BIAS); self.mlp = MLP()
        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(VOCAB, NE),
                wpe=nn.Embedding(BLOCK, NE),
                drop=nn.Dropout(DROP),
                h=nn.ModuleList([Block() for _ in range(NL)]),
                ln_f=LayerNorm(NE, BIAS),
            ))
            self.lm_head = nn.Linear(NE, VOCAB, bias=False)
            self.block_size = BLOCK
        def forward(self, idx):
            B, T = idx.size()
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
            for blk in self.transformer.h:
                x = blk(x)
            x = self.transformer.ln_f(x)
            return self.lm_head(x)
        @torch.no_grad()
        def generate_greedy(self, idx, max_new_tokens, stop_ids=None):
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
                logits = self(idx_cond)[:, -1, :]
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, nxt), dim=1)
                if stop_ids is not None and nxt.item() in stop_ids:
                    break
            return idx

    return GPT()


def load_model():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = _build(ckpt["model_config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt["model_config"]


def encode_text(s: str):
    """Byte-level: cada byte UTF-8 vira um id 0..255."""
    return list(s.encode("utf-8"))


def decode(ids):
    """Descodifica, mostrando tokens especiais literalmente."""
    out = bytearray(); rendered = []
    for i in ids:
        if i < 256:
            out.append(i)
        else:
            rendered.append(out.decode("utf-8", errors="replace")); out = bytearray()
            rendered.append(SPECIALS.get(i, f"<{i}>"))
    rendered.append(out.decode("utf-8", errors="replace"))
    return "".join(rendered)
