"""
Teste de SANIDADE do GCG: inicializar PERTO de <|alvaro_de_campos|> e ver se o gradiente
consegue subir/recuperar ate P(flag{) ~ 0.4 (o engodo). Valida se a busca e capaz na bacia certa.
"""
import time
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
model.requires_grad_(False)
NE=cfg["n_embd"]; VOCAB=cfg["vocab_size"]; BLOCK=cfg["block_size"]
wte = model.transformer.wte.weight
def dec(ids): return bytes(i for i in ids if i<256).decode("utf-8","replace")
def show(ids):
    NM={256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"«_»",261:"«{»"}
    return "".join(NM[i] if i>=256 else (chr(i) if 32<=i<127 else f"\\x{i:02x}") for i in ids)

def forward_embeds(emb):
    T=emb.size(1); x=emb+model.transformer.wpe(torch.arange(T))
    for blk in model.transformer.h: x=blk(x)
    return model.lm_head(model.transformer.ln_f(x))
def forward_ids(ids): return model(torch.tensor([ids],dtype=torch.long))[0]

def tloss(x_ids, tgt):
    seq=list(x_ids)+list(tgt)
    with torch.no_grad():
        lp=F.log_softmax(forward_ids(seq),dim=-1)
        L=len(x_ids); return -sum(float(lp[L-1+k,t]) for k,t in enumerate(tgt))

def gcg(init_ids, target_ids, steps=60, topk=128, B=128, seed=0, label=""):
    torch.manual_seed(seed)
    x=list(init_ids); L=len(x); tgt=torch.tensor(target_ids)
    best=tloss(x,target_ids)
    print(f"  [{label}] init: P(flag{{)={float(torch.exp(torch.tensor(-best))):.3e}  x={show(x)!r}")
    t0=time.time()
    for step in range(steps):
        oh=torch.zeros(L,VOCAB)
        for i,t in enumerate(x): oh[i,t]=1.0
        oh.requires_grad_(True)
        emb=torch.cat([oh@wte, model.transformer.wte(tgt)],0).unsqueeze(0)
        lp=F.log_softmax(forward_embeds(emb)[0],dim=-1)
        loss=sum(-lp[L-1+k,t] for k,t in enumerate(target_ids))
        loss.backward()
        cand=(-oh.grad).topk(topk,dim=1).indices
        trials=[]
        for b in range(B):
            xb=list(x); p=int(torch.randint(0,L,(1,)))
            xb[p]=int(cand[p,int(torch.randint(0,topk,(1,)))]); trials.append(xb)
        losses=[tloss(xb,target_ids) for xb in trials]
        j=int(torch.tensor(losses).argmin())
        if losses[j]<best: best=losses[j]; x=trials[j]
        if step%10==0 or step==steps-1:
            print(f"    passo {step:3d}: P(flag{{)={float(torch.exp(torch.tensor(-best))):.3e}  "
                  f"x={show(x)!r}  ({time.time()-t0:.0f}s)")
    return x, best

TGT=list("flag{".encode())
ALV=list("<|alvaro_de_campos|>".encode())

print("================= BASELINE: <|alvaro_de_campos|> exato =================")
b=tloss(ALV,TGT)
print(f"  P(flag{{) do engodo exato = {float(torch.exp(torch.tensor(-b))):.4f}  (loss={b:.3f})")
print(f"  gera: {dec([int(torch.argmax(forward_ids(ALV)[-1]))]) }... (1º token)")

print("\n================= SANIDADE 1: init = engodo EXATO, deixar GCG correr =================")
x1,l1=gcg(ALV, TGT, steps=40, seed=1, label="exato")
print(f"  FINAL P(flag{{)={float(torch.exp(torch.tensor(-l1))):.4f}  x={show(x1)!r}")

print("\n================= SANIDADE 2: init = engodo PERTURBADO (4 bytes trocados) =================")
torch.manual_seed(7)
pert=list(ALV)
for p in torch.randperm(len(pert))[:4].tolist():
    pert[p]=int(torch.randint(32,127,(1,)))
print(f"  perturbado: {show(pert)!r}")
x2,l2=gcg(pert, TGT, steps=60, seed=2, label="perturbado")
print(f"  FINAL P(flag{{)={float(torch.exp(torch.tensor(-l2))):.4f}  x={show(x2)!r}")
print(f"  recuperou para o engodo? {'SIM' if dec(x2)=='<|alvaro_de_campos|>' else 'parcial/nao'}")
@torch.no_grad()
def greedy(ids,n=40):
    seq=list(ids)
    for _ in range(n): seq.append(int(torch.argmax(forward_ids(seq[-BLOCK:])[-1])))
    return seq[len(ids):]
print("  geracao do x2:", repr(dec(greedy(x2,40))))
