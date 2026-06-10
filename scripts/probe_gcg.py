"""
GCG (Greedy Coordinate Gradient): inverter o modelo para achar input que gera 'flag{'.
Corrida A: alvo 'flag{'. Corrida B: alvo 'flag{' + '_'(260).
CPU, modelo 50M. Reporta input emergente, P(alvo) e geracao greedy.
"""
import time
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
model.requires_grad_(False)
NE = cfg["n_embd"]; VOCAB = cfg["vocab_size"]; BLOCK = cfg["block_size"]
wte = model.transformer.wte.weight   # (V, NE)
def dec(ids): return bytes(i for i in ids if i < 256).decode("utf-8", "replace")
def show(ids):
    NM={256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"«_»",261:"«{»"}
    return "".join(NM[i] if i>=256 else (chr(i) if 32<=i<127 else f"\\x{i:02x}") for i in ids)

def forward_embeds(embeds):
    T = embeds.size(1)
    pos = torch.arange(T)
    x = embeds + model.transformer.wpe(pos)
    for blk in model.transformer.h:
        x = blk(x)
    return model.lm_head(model.transformer.ln_f(x))

def forward_ids(ids):
    return model(torch.tensor([ids], dtype=torch.long))[0]  # (T,V)

@torch.no_grad()
def greedy(ids, n=60):
    seq=list(ids)
    for _ in range(n):
        seq.append(int(torch.argmax(forward_ids(seq[-BLOCK:])[-1])))
    return seq[len(ids):]

def target_loss_ids(x_ids, target_ids):
    """log-loss (CE) do alvo logo apos x, via ids (sem grad)."""
    seq = list(x_ids) + list(target_ids)
    with torch.no_grad():
        logits = forward_ids(seq)
        lp = F.log_softmax(logits, dim=-1)
        L = len(x_ids)
        tot = 0.0
        for k, t in enumerate(target_ids):
            tot += float(lp[L-1+k, t])
    return -tot  # menor = melhor

def gcg(target_ids, L=12, steps=40, topk=128, B=64, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(32, 127, (L,)).tolist()  # init bytes imprimiveis
    tgt = torch.tensor(target_ids, dtype=torch.long)
    best_loss = target_loss_ids(x, target_ids)
    t0 = time.time()
    for step in range(steps):
        # --- gradiente sobre one-hot do input ---
        one_hot = torch.zeros(L, VOCAB)
        for i, t in enumerate(x): one_hot[i, t] = 1.0
        one_hot.requires_grad_(True)
        x_emb = one_hot @ wte                      # (L,NE)
        tgt_emb = model.transformer.wte(tgt)       # (T2,NE)
        emb = torch.cat([x_emb, tgt_emb], 0).unsqueeze(0)
        logits = forward_embeds(emb)[0]
        lp = F.log_softmax(logits, dim=-1)
        loss = 0.0
        for k, t in enumerate(target_ids):
            loss = loss - lp[L-1+k, t]
        loss.backward()
        grad = one_hot.grad                        # (L,V)
        # --- candidatos: top-k tokens por -grad em cada posicao ---
        cand_tokens = (-grad).topk(topk, dim=1).indices   # (L,topk)
        # amostra B trocas (posicao aleatoria, token do topk)
        trials = []
        pos_sel = torch.randint(0, L, (B,))
        tok_col = torch.randint(0, topk, (B,))
        for b in range(B):
            xb = list(x); p = int(pos_sel[b])
            xb[p] = int(cand_tokens[p, int(tok_col[b])])
            trials.append(xb)
        # avalia
        losses = [target_loss_ids(xb, target_ids) for xb in trials]
        j = int(torch.tensor(losses).argmin())
        if losses[j] < best_loss:
            best_loss = losses[j]; x = trials[j]
        if step % 5 == 0 or step == steps-1:
            p_tgt = torch.exp(torch.tensor(-best_loss))
            print(f"  passo {step:3d}: loss={best_loss:.3f}  P(alvo)={float(p_tgt):.3e}  "
                  f"x={show(x)!r}  ({time.time()-t0:.0f}s)")
    return x, best_loss

print("================= CORRIDA A: alvo 'flag{' =================")
tgtA = list("flag{".encode())
xA, lA = gcg(tgtA, L=12, steps=40, topk=128, B=64, seed=1)
print("  INPUT FINAL:", show(xA))
print("  GERA:", repr(dec(greedy(xA, 60))))

print("\n================= CORRIDA B: alvo 'flag{' + '_'(260) =================")
tgtB = list("flag{".encode()) + [260]
xB, lB = gcg(tgtB, L=12, steps=40, topk=128, B=64, seed=2)
print("  INPUT FINAL:", show(xB))
print("  GERA:", repr(dec(greedy(xB, 60))))
