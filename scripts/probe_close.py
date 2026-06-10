"""
Resolver o fecho da flag: '}' nunca aparece no greedy puro.
Hipotese: depois de '...' o modelo prefere o '\n' poetico generico ao raro '}' do canario.
(1) Inspeciona a distribuicao logo apos 'flag{Hup-la... He-ha... He-ho... Z-z-z-z...'.
(2) Gera bloqueando '\n'(10) para ver com que caracter fecha a flag.
Tudo local, NAO submete.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model, encode_text, decode

model, cfg = load_model()

TRIGGER = "<|alvaro_de_campos|>"
# prefixo = a regiao memorizada de alta confianca, ate ao ultimo '.' antes do \n
PREFIX = TRIGGER + "flag{Hup-la... He-ha... He-ho... Z-z-z-z..."

def topk_at(ids, k=15):
    idx = torch.tensor([ids[-cfg["block_size"]:]], dtype=torch.long)
    with torch.no_grad():
        probs = F.softmax(model(idx)[:, -1, :], dim=-1)[0]
    order = torch.argsort(probs, descending=True)
    rank = {int(t): r for r, t in enumerate(order.tolist())}
    def name(i):
        if i >= 256: return {256:"<|FP|>",257:"<|AC|>",258:"<|RR|>",259:"<|BS|>",260:"_esp",261:"{esp"}[i]
        c = chr(i); return repr(c) if c.isprintable() else f"\\x{i:02x}"
    print("  top:", ", ".join(f"{name(int(t))}={float(probs[t]):.3f}" for t in order[:k]))
    for special, lbl in [(ord('}'), "'}'(125)"), (ord('\n'), "'\\n'(10)"), (261, "'{'esp(261)"), (260, "'_'esp(260)")]:
        print(f"    {lbl:<14} rank={rank[special]:>3}  p={float(probs[special]):.3e}")

print("=== (1) Distribuicao logo apos 'Z-z-z-z...' ===")
print("PREFIX:", repr(PREFIX))
topk_at(encode_text(PREFIX))

print("\n=== (2) Greedy a partir do TRIGGER, mas bloqueando '\\n'(10) ===")
@torch.no_grad()
def greedy_block(ids, blocked, n=60):
    seq = list(ids)
    for _ in range(n):
        idx = torch.tensor([seq[-cfg["block_size"]:]], dtype=torch.long)
        logits = model(idx)[:, -1, :].clone()
        for b in blocked:
            logits[0, b] = -float("inf")
        nxt = int(torch.argmax(logits))
        seq.append(nxt)
        if nxt == ord('}'):
            break
    return seq[len(ids):]

gen = greedy_block(encode_text(TRIGGER), blocked=[ord('\n')], n=80)
print("  =>", repr(decode(gen)))

print("\n=== (3) Greedy bloqueando '\\n'(10) E '.'(46) apos as onomatopeias ===")
# arranca ja com o prefixo e bloqueia newline para forcar o fecho in-line
gen2 = greedy_block(encode_text(PREFIX), blocked=[ord('\n')], n=20)
print("  PREFIX + =>", repr(decode(gen2)))
