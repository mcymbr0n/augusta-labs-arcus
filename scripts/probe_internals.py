"""
Inspeccao da ESTRUTURA INTERNA do modelo (pesos), nao so do output.
(1) Normas dos embeddings dos especiais 256-261 vs bytes 0-255 (anomalias?).
(2) Cosseno entre os 6 especiais; vizinhos mais proximos de cada um.
(3) wte vs lm_head (tied?).
(4) Logit lens: camada a camada, onde cristaliza 'f'/'flag{' no gatilho.
Tudo local, NAO submete.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
NM = {256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"«_»",261:"«{»"}
def tname(i):
    if i in NM: return NM[i]
    c=chr(i); return repr(c) if c.isprintable() else f"\\x{i:02x}"

wte = model.transformer.wte.weight.detach()      # (262,640)
lm  = model.lm_head.weight.detach()              # (262,640)
norms = wte.norm(dim=1)

print("================= (3) wte vs lm_head =================")
tied = torch.allclose(wte, lm)
print(f"  wte e lm_head identicos (weight tying)? {tied}  max|dif|={float((wte-lm).abs().max()):.3e}")

print("\n================= (1) NORMAS DOS EMBEDDINGS =================")
byte_norms = norms[:256]
print(f"  bytes 0-255: media={byte_norms.mean():.3f} std={byte_norms.std():.3f} "
      f"min={byte_norms.min():.3f} max={byte_norms.max():.3f}")
print("  especiais:")
for i in range(256,262):
    n=float(norms[i])
    pct=float((byte_norms < n).float().mean())*100
    z=(n-float(byte_norms.mean()))/float(byte_norms.std())
    print(f"    {NM[i]:>5} (id {i}): norma={n:.3f}  z={z:+.2f}  percentil_vs_bytes={pct:.0f}%")

print("\n================= (2) COSSENO ENTRE ESPECIAIS =================")
sp = wte[256:262]
spn = F.normalize(sp, dim=1)
cos = spn @ spn.T
hdr = "        " + " ".join(f"{NM[256+j]:>6}" for j in range(6))
print(hdr)
for a in range(6):
    row=" ".join(f"{float(cos[a,b]):+6.2f}" for b in range(6))
    print(f"  {NM[256+a]:>5} {row}")

print("\n================= (2b) VIZINHOS MAIS PROXIMOS DE CADA ESPECIAL =================")
alln = F.normalize(wte, dim=1)
for i in range(256,262):
    sims = alln @ alln[i]
    sims[i] = -2
    top = torch.topk(sims, 8)
    nb = ", ".join(f"{tname(int(t))}={float(s):+.2f}" for s,t in zip(top.values, top.indices))
    print(f"  {NM[i]:>5}: {nb}")

print("\n  (controlo) vizinhos do byte '_'(95) e '{'(123):")
for i in (95,123):
    sims = alln @ alln[i]; sims[i]=-2
    top=torch.topk(sims,6)
    nb=", ".join(f"{tname(int(t))}={float(s):+.2f}" for s,t in zip(top.values,top.indices))
    print(f"    {tname(i)}: {nb}")

print("\n================= (4) LOGIT LENS no gatilho '<|alvaro_de_campos|>' =================")
def enc(s): return list(s.encode("utf-8"))
@torch.no_grad()
def logit_lens(prompt, label):
    ids = torch.tensor([prompt], dtype=torch.long)
    pos = torch.arange(len(prompt))
    x = model.transformer.wte(ids) + model.transformer.wpe(pos)
    print(f"\n  --- {label}  (ultimo token = {tname(prompt[-1])}) ---")
    for li, blk in enumerate(model.transformer.h):
        x = blk(x)
        h = model.transformer.ln_f(x)
        logits = model.lm_head(h)[0, -1]
        p = F.softmax(logits, dim=-1)
        top = torch.topk(p, 5)
        s = ", ".join(f"{tname(int(t))}={float(v):.2f}" for v,t in zip(top.values, top.indices))
        print(f"    L{li:>2}: {s}")

logit_lens(enc("<|alvaro_de_campos|>"), "GATILHO alvaro (bytes)")
logit_lens(enc("<|fernando_pessoa|>"),  "CONTROLO fernando (bytes)")
logit_lens([256],                        "CONTROLO «FP» token 256")
