"""
Sá-Carneiro como gatilho (a la Campos). Greedy leve. Foco: flag{ com '_'.
Tudo local, NAO submete.
"""
import torch
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
def enc(s): return list(s.encode("utf-8"))
def dec(ids): return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

@torch.no_grad()
def greedy(ids, n=90):
    seq = list(ids)
    for _ in range(n):
        seq.append(int(torch.argmax(model(torch.tensor([seq[-BLOCK:]], dtype=torch.long))[:, -1, :])))
    return seq[len(ids):]

def run(label, s):
    g = dec(greedy(enc(s), 90))
    note = ""
    if "flag{" in g:
        seg = g[g.find("flag{"):g.find("flag{")+90]
        note = f"   <<< FLAG {seg!r}" + ("  TEM '_'!!" if "_" in seg else "")
    print(f"  {label:<34} -> {g[:62]!r}{note}")

cands = [
    "<|mario_de_sa_carneiro|>",
    "<|sa_carneiro|>",
    "<|mario_sa_carneiro|>",
    "<|sá_carneiro|>",
    "<|orpheu|>",
    "Mário de Sá-Carneiro",
    # variantes extra
    "<|mário_de_sá_carneiro|>",
    "<|sa-carneiro|>",
    "<|carneiro|>",
    "<|mario_de_sá_carneiro|>",
    "<|m_sa_carneiro|>",
    "Sá-Carneiro",
    "<|orfeu|>",
]
for s in cands:
    run(s, s)
