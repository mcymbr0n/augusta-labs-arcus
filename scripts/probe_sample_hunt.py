"""
Pesca por AMOSTRAGEM em larga escala (em lote) por uma flag memorizada mas sub-dominante.
Para varias sementes, amostra centenas de continuacoes e coleta strings 'flag{...}'.
Tudo local, NAO submete.
"""
import re
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
NM = {256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"_",261:"{"}
def decode_bytes(ids):
    out=bytearray()
    for i in ids:
        if i<256: out.append(i)
        else: out += NM[i].encode("utf-8")
    return out.decode("utf-8","replace")

VERSES=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
def enc(s): return list(s.encode("utf-8"))

@torch.no_grad()
def sample_batch(prompt_ids, n_samples, n_tokens, temp, top_k, seed):
    torch.manual_seed(seed)
    seq = torch.tensor([prompt_ids], dtype=torch.long).repeat(n_samples, 1)
    for _ in range(n_tokens):
        idx = seq[:, -BLOCK:]
        logits = model(idx)[:, -1, :] / temp
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        seq = torch.cat([seq, nxt], dim=1)
    return seq[:, len(prompt_ids):].tolist()

FLAG_RE = re.compile(r"flag\{[^\n}]{0,80}\}?")

seeds = [
    ("newline", enc("\n")),
    ("«FP»256", [256]), ("«AC»257", [257]), ("«RR»258", [258]), ("«BS»259", [259]),
    ("«_»260", [260]), ("«{»261", [261]),
    ("'flag{'", enc("flag{")), ("'flag'+«{»", enc("flag")+[261]),
    ("versos", enc(VERSES)),
    ("'Álvaro de Campos'", enc("Álvaro de Campos")),
    ("'<|alvaro_de_campos|>' (controlo)", enc("<|alvaro_de_campos|>")),
]

found = {}  # flagstring -> (count, seeds)
for label, ids in seeds:
    txts = []
    for s in range(3):  # 3 lotes de 64 = 192 amostras por semente
        for row in sample_batch(ids, 64, 60, temp=0.9, top_k=80, seed=1000+s):
            txts.append(decode_bytes(row))
    cnt = 0
    for t in txts:
        for m in FLAG_RE.findall(t):
            cnt += 1
            found.setdefault(m, [0, set()])
            found[m][0] += 1
            found[m][1].add(label)
    print(f"  {label:<34} amostras=192  flags_encontradas={cnt}")

print("\n================= STRINGS 'flag{...}' UNICAS ENCONTRADAS =================")
if found:
    for fl, (c, labs) in sorted(found.items(), key=lambda x:-x[1][0]):
        print(f"  x{c:<4} {fl!r}   <- sementes: {sorted(labs)}")
else:
    print("  (nenhuma)")
