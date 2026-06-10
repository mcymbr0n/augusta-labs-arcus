"""
#1 Mapa de atencao das 80 cabecas no gatilho <|alvaro_de_campos|> vs controlo <|fernando_pessoa|>.
   Procura cabeca anomala (atencao concentrada num token especifico, diferente do controlo).
#2 Continuacao longa (greedy 500) do decoy, sem cortar no \n. Ha mais flag/2a flag?
Tudo local, leve, NAO submete.
"""
import math
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
NE, NH = cfg["n_embd"], cfg["n_head"]
HD = NE // NH
BLOCK = cfg["block_size"]
def enc(s): return list(s.encode("utf-8"))
def dec(ids): return bytes(i for i in ids if i < 256).decode("utf-8", "replace")
def chrep(b):
    c = chr(b) if b < 256 else "?"
    return repr(c) if (b < 256 and c.isprintable()) else f"\\x{b:02x}"

@torch.no_grad()
def attn_last(prompt_ids):
    """Devolve, por (camada,head), a distribuicao de atencao da ULTIMA query."""
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    T = len(prompt_ids)
    pos = torch.arange(T)
    x = model.transformer.wte(ids) + model.transformer.wpe(pos)
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    per_layer = []  # cada: tensor (NH, T) atencao da ultima query
    for blk in model.transformer.h:
        h = blk.ln_1(x)
        qkv = blk.attn.c_attn(h)
        q, k, v = qkv.split(NE, dim=2)
        q = q.view(1, T, NH, HD).transpose(1, 2)
        k = k.view(1, T, NH, HD).transpose(1, 2)
        v = v.view(1, T, NH, HD).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        per_layer.append(att[0, :, -1, :].clone())  # (NH, T)
        y = (att @ v).transpose(1, 2).reshape(1, T, NE)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln_2(x))
    return per_layer  # lista de 10 tensores (NH,T)

def summarize(prompt_ids, label):
    pl = attn_last(prompt_ids)
    print(f"\n### {label}  (T={len(prompt_ids)}, ultimo tok={chrep(prompt_ids[-1])}) ###")
    rows = []
    for li, att in enumerate(pl):
        for hi in range(NH):
            a = att[hi]
            mx, pos = float(a.max()), int(a.argmax())
            ent = float(-(a * (a + 1e-12).log()).sum())
            rows.append((li, hi, mx, pos, ent))
    # mostra cabecas mais concentradas (top por max_weight)
    rows.sort(key=lambda r: -r[2])
    print("  Top-12 cabecas mais CONCENTRADAS (max_weight):")
    for li, hi, mx, pos, ent in rows[:12]:
        tok = prompt_ids[pos]
        print(f"    L{li}H{hi}: max={mx:.2f} -> pos{pos}({chrep(tok)})  entropia={ent:.2f}")
    return {(li, hi): (mx, pos, ent) for li, hi, mx, pos, ent in rows}

print("================= #1 MAPA DE ATENCAO =================")
TRIG = enc("<|alvaro_de_campos|>")
CTRL = enc("<|fernando_pessoa|>")
a = summarize(TRIG, "GATILHO <|alvaro_de_campos|>")
b = summarize(CTRL, "CONTROLO <|fernando_pessoa|>")

print("\n--- DIFERENCAS: cabecas onde alvaro concentra >0.4 E difere do controlo ---")
# nota: prompts tem comprimentos diferentes; comparamos so a concentracao/posicao relativa ao fim
Ta, Tb = len(TRIG), len(CTRL)
for (li, hi), (mx, pos, ent) in a.items():
    mxb, posb, entb = b[(li, hi)]
    rel_a = pos - Ta   # distancia ao fim (negativa)
    rel_b = posb - Tb
    if mx > 0.4 and (mx - mxb > 0.25 or rel_a != rel_b):
        print(f"  L{li}H{hi}: ALVARO max={mx:.2f}@rel{rel_a}({chrep(TRIG[pos])})  |  "
              f"FERNANDO max={mxb:.2f}@rel{rel_b}({chrep(CTRL[posb])})")

print("\n\n================= #2 CONTINUACAO LONGA DO DECOY (greedy 500) =================")
@torch.no_grad()
def greedy(ids, n):
    seq = list(ids)
    for _ in range(n):
        seq.append(int(torch.argmax(model(torch.tensor([seq[-BLOCK:]], dtype=torch.long))[:, -1, :])))
    return seq[len(ids):]

g = greedy(TRIG, 500)
txt = dec(g)
print("Continuacao (500 tokens):\n", repr(txt))
import re
flags = [(m.start(), m.group()) for m in re.finditer(r"flag\{[^\n}]{0,80}\}?", txt)]
print("\nOcorrencias de 'flag{':", flags if flags else "(so a inicial)")
print("Tem '_' algures?", "SIM @ "+str(txt.find('_')) if '_' in txt else "nao")
