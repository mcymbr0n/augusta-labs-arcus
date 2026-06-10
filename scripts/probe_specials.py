"""
Exaustivo sobre os TOKENS ESPECIAIS como input "desenhado".
- 6 singulares, 36 pares, alguns triplos/quadruplos, e especial+'flag{'.
Procura qualquer saida com 'flag', '{', '}' ou '«_»'(260) — i.e. a flag REAL.
Tudo local, NAO submete.
"""
import torch
import torch.nn.functional as F
from itertools import product
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
SP = [256, 257, 258, 259, 260, 261]
NM = {256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"«_»",261:"«{»"}
def nm(i): return NM[i] if i in NM else (chr(i) if 32<=i<127 else f"\\x{i:02x}")
def decode_marked(ids):
    out=bytearray(); parts=[]
    for i in ids:
        if i<256: out.append(i)
        else: parts.append(out.decode("utf-8","replace")); out=bytearray(); parts.append(NM[i])
    parts.append(out.decode("utf-8","replace")); return "".join(parts)

@torch.no_grad()
def gen(prompt, n=40):
    seq=list(prompt)
    for _ in range(n):
        idx=torch.tensor([seq[-BLOCK:]],dtype=torch.long)
        seq.append(int(torch.argmax(model(idx)[:,-1,:])))
    return seq[len(prompt):]

def interesting(txt):
    return ("flag" in txt) or ("{" in txt) or ("}" in txt) or ("«_»" in txt)

print("================= SINGULARES =================")
for s in SP:
    txt=decode_marked(gen([s],50))
    tag=" <<< INTERESSANTE" if interesting(txt) else ""
    print(f"  {nm(s):>4} -> {txt[:64]!r}{tag}")

print("\n================= PARES (36) =================")
hits=[]
for a,b in product(SP,SP):
    txt=decode_marked(gen([a,b],32))
    if interesting(txt):
        hits.append((a,b,txt))
    flag=" <<<" if interesting(txt) else ""
    print(f"  {nm(a):>4}{nm(b):>4} -> {txt[:54]!r}{flag}")

print("\n================= TRIPLOS / QUADRUPLOS / SEQUENCIAS =================")
seqs = [
    [256,257,258,259], [259,258,257,256], [256,261], [261,256],
    [256,260,261], [261,260], [260,261], [256,257,258,259,261],
]
for sq in seqs:
    txt=decode_marked(gen(sq,40))
    tag=" <<<" if interesting(txt) else ""
    print(f"  {''.join(nm(x) for x in sq):>20} -> {txt[:54]!r}{tag}")

print("\n================= ESPECIAL + 'flag' + «{» =================")
def enc(s):  # so bytes simples aqui
    return list(s.encode("utf-8"))
for s in SP:
    prompt=[s]+enc("flag")+[261]
    txt=decode_marked(gen(prompt,36))
    tag=" <<<" if ("Hup" not in txt and ("-" in txt or "_" in txt or "«_»" in txt)) else ""
    print(f"  {nm(s)}+flag+«{{» -> {txt[:54]!r}{tag}")

print("\n================= RESUMO HITS =================")
if hits:
    for a,b,txt in hits:
        print(f"  {nm(a)}{nm(b)} -> {txt!r}")
else:
    print("  (nenhum par produziu flag/{/}/_)")
