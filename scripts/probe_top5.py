"""
Top-5 alternativas em cada posicao do caminho do engodo (a partir de <|alvaro_de_campos|>).
Procura alternativa nao-trivial (>1%) que aponte caminho diferente (sobretudo '_' ou texto canonico).
Tudo local, NAO submete.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
def enc(s): return list(s.encode("utf-8"))
def nm(i):
    NM={256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"«_»",261:"«{»"}
    if i in NM: return NM[i]
    c=chr(i); return repr(c) if c.isprintable() else f"\\x{i:02x}"

TRIG = enc("<|alvaro_de_campos|>")

@torch.no_grad()
def run(n=50):
    seq=list(TRIG)
    print("pos | escolhido(p)      | top-5 alternativas (byte=prob)")
    print("-"*92)
    notable=[]
    for step in range(n):
        p=F.softmax(model(torch.tensor([seq[-BLOCK:]],dtype=torch.long))[:,-1,:],dim=-1)[0]
        top=torch.topk(p,6)
        ch=int(top.indices[0])               # greedy = top1
        seq.append(ch)
        alts=[(int(t),float(v)) for t,v in zip(top.indices,top.values)]
        chosen=alts[0]
        rest=alts[1:6]
        # alternativa nao-trivial (>1%) que NAO seja so ruido
        big=[(t,v) for t,v in rest if v>0.01]
        s_rest=" ".join(f"{nm(t)}={v:.3f}" for t,v in rest)
        mark=""
        if big:
            mark="   <== ALT>1%: "+", ".join(f"{nm(t)}={v:.3f}" for t,v in big)
            notable.append((step, nm(ch), chosen[1], big))
        print(f"{step:3d} | {nm(ch):>6}({chosen[1]:.3f}) | {s_rest}{mark}")
    return notable

notable = run(50)
print("\n================= POSICOES COM ALTERNATIVA >1% =================")
if notable:
    for step, ch, p, big in notable:
        flags = ""
        for t,v in big:
            if t==260 or t==ord('_'): flags += " [APONTA '_'!]"
            if t==0xC3: flags += " [acento C3]"
        print(f"  pos {step}: escolhido {ch}(p={p:.3f}) | alt: "
              + ", ".join(f"{nm(t)}={v:.3f}" for t,v in big) + flags)
else:
    print("  Nenhuma. Em todas as posicoes o engodo domina; alternativas sao ruido <1%.")
