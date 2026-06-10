"""
Auditoria dos delimitadores da flag (sem submeter nada).
(A) Distribuicao completa na ABERTURA (apos 'flag'): byte '{'(123) vs especial '{'(261)?
(B) Forcar o especial 261 como abertura -> que continuacao/fecho sai?
(C) Existe '}' especial? (vocab=262, especiais=256..261). Rank de '}'(125)/260/261 apos Z-z-z-z...
(D) Confirmar que 'flag' e bytes normais e se '_'(260) aparece dentro do conteudo.
Tudo local.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model, encode_text

model, cfg = load_model()
TRIGGER = "<|alvaro_de_campos|>"
VOCAB = cfg["vocab_size"]

SPMARK = {256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"«_»261?no:«_»260",261:"«{»261"}
def name(i):
    if i >= 256:
        return {256:"«FP»256",257:"«AC»257",258:"«RR»258",259:"«BS»259",260:"«_»260",261:"«{»261"}[i]
    c = chr(i)
    return repr(c) if c.isprintable() else f"\\x{i:02x}"

def decode_marked(ids):
    """Distingue bytes de tokens especiais: especiais aparecem como «..»N."""
    out = bytearray(); parts = []
    for i in ids:
        if i < 256:
            out.append(i)
        else:
            parts.append(out.decode("utf-8", errors="replace")); out = bytearray()
            parts.append(name(i))
    parts.append(out.decode("utf-8", errors="replace"))
    return "".join(parts)

@torch.no_grad()
def dist(ids):
    idx = torch.tensor([ids[-cfg["block_size"]:]], dtype=torch.long)
    return F.softmax(model(idx)[:, -1, :], dim=-1)[0]

def show_topk(label, ids, watch, k=12):
    p = dist(ids)
    order = torch.argsort(p, descending=True)
    rank = {int(t): r for r, t in enumerate(order.tolist())}
    print(f"\n--- {label} ---")
    print("  top:", ", ".join(f"{name(int(t))}={float(p[t]):.3f}" for t in order[:k]))
    for w in watch:
        print(f"    {name(w):<10} rank={rank[w]:>3}  p={float(p[w]):.3e}")

print("================ (C) ESPECIAIS EXISTENTES ================")
print(f"  vocab_size = {VOCAB}. IDs especiais (>=256): 256..261 =>",
      [name(i) for i in range(256, 262)])
print("  => Existe '{' especial (261) e '_' especial (260). NAO existe '}' especial.")

print("\n================ (A) ABERTURA: apos 'flag' ================")
show_topk("apos '<|alvaro_de_campos|>flag'", encode_text(TRIGGER + "flag"),
          watch=[ord('{'), 261, ord('\n'), ord(' ')])

print("\n================ (D) INICIO: 'f','l','a','g' sao bytes? ================")
for pre in ["", "f", "fl", "fla"]:
    show_topk(f"apos trigger+'{pre}'", encode_text(TRIGGER + pre),
              watch=[ord('f') if pre=="" else (ord('l') if pre=='f' else (ord('a') if pre=='fl' else ord('g')))], k=6)

print("\n================ (B) FORCAR ESPECIAL 261 COMO ABERTURA ================")
@torch.no_grad()
def greedy(ids, n=120, stop=None):
    seq = list(ids); chosen = []
    for _ in range(n):
        idx = torch.tensor([seq[-cfg["block_size"]:]], dtype=torch.long)
        nxt = int(torch.argmax(model(idx)[:, -1, :]))
        seq.append(nxt); chosen.append(nxt)
        if stop is not None and nxt in stop:
            break
    return chosen

# 'flag' + especial '{'(261) em vez do byte '{'
seq_special = encode_text(TRIGGER + "flag") + [261]
gen_sp = greedy(seq_special, n=120)
print("  PROMPT: trigger+'flag'+«{»261")
print("  =>", repr(decode_marked([261] + gen_sp)))

# tambem: trigger + 'flag' + especial '{' , parar se aparecer '}'(125) ou qualquer especial
print("\n  (mesmo, marcando se surge '}' ou especiais)")
gen_sp2 = greedy(seq_special, n=120, stop={ord('}')})
print("  =>", repr(decode_marked([261] + gen_sp2)))

print("\n================ (D2) '_'(260) DENTRO DO CONTEUDO? ================")
# nas posicoes internas onde greedy poe '-' ou ' ', o especial '_'(260) e competitivo?
for probe in ["flag{Hup", "flag{Hup-la... He", "flag{Hup-la... He-ha... He-ho... Z"]:
    show_topk(f"apos '{probe}'", encode_text(TRIGGER + probe),
              watch=[ord('-'), ord(' '), 260], k=6)
