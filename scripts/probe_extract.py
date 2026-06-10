"""
Extracao precisa da flag a partir do gatilho '<|alvaro_de_campos|>'.
Gera greedy token-a-token e regista a probabilidade de cada caracter escolhido.
A flag = regiao de alta confianca (memorizada); o colapso de confianca marca o fim.
Para tambem no '}' (byte 125). Tudo local, NAO submete nada.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model, encode_text, decode

model, cfg = load_model()
TRIGGER = "<|alvaro_de_campos|>"

def ch(i):
    if i >= 256:
        return {256:"<|FP|>",257:"<|AC|>",258:"<|RR|>",259:"<|BS|>",260:"_esp",261:"{esp"}[i]
    c = chr(i)
    return c if c.isprintable() else f"\\x{i:02x}"

@torch.no_grad()
def extract(trigger, max_new=120, stop_on_brace=True):
    ids = encode_text(trigger)
    seq = list(ids)
    rows = []
    closed_at = None
    for step in range(max_new):
        idx = torch.tensor([seq[-cfg["block_size"]:]], dtype=torch.long)
        probs = F.softmax(model(idx)[:, -1, :], dim=-1)[0]
        nxt = int(torch.argmax(probs))
        p = float(probs[nxt])
        rows.append((nxt, p))
        seq.append(nxt)
        if stop_on_brace and nxt == ord('}'):
            closed_at = step
            break
    return ids, rows, closed_at

ids, rows, closed_at = extract(TRIGGER)

gen_ids = [t for t, _ in rows]
print("TRIGGER:", repr(TRIGGER))
print("Continuacao bruta:", repr(decode(gen_ids)))
print(f"Fechou com '}}' ? {'sim, no passo '+str(closed_at) if closed_at is not None else 'NAO (nao apareceu nos '+str(len(rows))+' passos)'}")
print("\nTOKEN-A-TOKEN (caracter : probabilidade):")
line = []
for i, (t, p) in enumerate(rows):
    flag = "  <-- confianca baixa" if p < 0.30 else ""
    print(f"  {i:3d}  {ch(t):<6} p={p:.3f}{flag}")
    if i >= 80:
        print("  ... (cortado)")
        break

# reconstruir so a regiao de alta confianca ate ao primeiro colapso
hi = []
for t, p in rows:
    hi.append(t)
    if t == ord('}'):
        break
print("\nRegiao ate '}' (se fechou):", repr(decode(hi)))
