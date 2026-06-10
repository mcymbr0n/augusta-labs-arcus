"""
Carregamento SEGURO + geração de teste do ode.pt (nanoGPT byte-level).
- torch.load(weights_only=True): NAO executa codigo do pickle (so tensores/tipos basicos).
- Reconstroi a arquitetura a partir de model_config e carrega o state_dict.
- Geracao simples SEM condicionar por token especial.
Tudo local. Nenhuma ligacao de rede.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

CKPT = r"C:\Users\migue\Documents\augusta-labs-arcus\artifacts\ode.pt"

# ---------- carregamento seguro ----------
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
cfg = ckpt["model_config"]
print("model_config:", cfg)
print("top-level keys:", list(ckpt.keys()))

VOCAB = cfg["vocab_size"]; BLOCK = cfg["block_size"]
NL = cfg["n_layer"]; NH = cfg["n_head"]; NE = cfg["n_embd"]
BIAS = cfg["bias"]

# ---------- arquitetura nanoGPT (bias=False) ----------
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
        self.resid_dropout = nn.Dropout(cfg["dropout"])
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
        self.dropout = nn.Dropout(cfg["dropout"])
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
            drop=nn.Dropout(cfg["dropout"]),
            h=nn.ModuleList([Block() for _ in range(NL)]),
            ln_f=LayerNorm(NE, BIAS),
        ))
        self.lm_head = nn.Linear(NE, VOCAB, bias=False)
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK else idx[:, -BLOCK:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

model = GPT()
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"parametros: {n_params/1e6:.2f}M")

# ---------- tokenizer byte-level (decode) ----------
specials = {256: "<|fernando_pessoa|>", 257: "<|alberto_caeiro|>",
            258: "<|ricardo_reis|>", 259: "<|bernardo_soares|>",
            260: "_", 261: "{"}
def decode(ids):
    out = bytearray()
    for i in ids:
        if i < 256:
            out.append(i)
        else:
            out += specials.get(i, f"<{i}>").encode("utf-8")
    return out.decode("utf-8", errors="replace")

# ---------- geracao SEM condicionar ----------
torch.manual_seed(1337)
start = torch.tensor([[ord("\n")]], dtype=torch.long)  # comeca com newline, sem token especial
out = model.generate(start, max_new_tokens=400, temperature=0.8, top_k=200)
print("\n===== GERACAO (seed=1337, temp=0.8, top_k=200) =====")
print(decode(out[0].tolist()))
print("===== FIM =====")
