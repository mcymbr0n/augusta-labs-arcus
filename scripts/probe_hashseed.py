"""
Abordagem dirigida pelo SHA256 oficial do ode.pt.
#2 (foco): hash como SEED determinada -> amostragem reproduzivel a partir de sementes-chave.
#3: hash como flag direta (so regista).
#4: hash (e prefixos) como GATILHO -> gera flag{ ?
Tudo local, NAO submete.
"""
import re
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
HASH = "b54373efba6b89e38bdd56f031ca63b7bf49f9024dea254c21227acc3dacb6ab"
HB = bytes.fromhex(HASH)

NM = {256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"_",261:"{"}
def dec(ids):
    out=bytearray()
    for i in ids:
        if i<256: out.append(i)
        else: out += NM[i].encode("utf-8")
    return out.decode("utf-8","replace")
def enc(s): return list(s.encode("utf-8"))

VERSES=("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")

# variantes do hash -> int para seed
seed_variants = {
    "int_full_mod2^63": int(HASH,16) % (2**63),
    "first8bytes_be":   int.from_bytes(HB[:8],  "big"),
    "first8bytes_le":   int.from_bytes(HB[:8],  "little"),
    "first16b_mod2^63": int.from_bytes(HB[:16],"big") % (2**63),
    "last8bytes_be":    int.from_bytes(HB[-8:], "big"),
    "int_full_mod2^32": int(HASH,16) % (2**32),
}

FLAG_RE = re.compile(r"flag\{[^\n}]{0,90}\}?")

@torch.no_grad()
def sample_batch(prompt_ids, n, ntok, temp, top_k, seed):
    torch.manual_seed(seed)
    seq = torch.tensor([prompt_ids],dtype=torch.long).repeat(n,1)
    for _ in range(ntok):
        logits = model(seq[:, -BLOCK:])[:, -1, :]/temp
        if top_k:
            v,_=torch.topk(logits,top_k); logits[logits<v[:,[-1]]]=-float("inf")
        nxt=torch.multinomial(F.softmax(logits,dim=-1),1)
        seq=torch.cat([seq,nxt],dim=1)
    return seq[:,len(prompt_ids):].tolist()

key_seeds = [
    ("«alvaro»", enc("<|alvaro_de_campos|>")),
    ("flag{",    enc("flag{")),
    ("newline",  enc("\n")),
    ("versos",   enc(VERSES)),
]

print("================= #2 HASH COMO SEED (amostragem determinada) =================")
found={}
for sv_name, sv in seed_variants.items():
    for s_name, ids in key_seeds:
        cnt=0
        for row in sample_batch(ids, 48, 64, temp=0.8, top_k=60, seed=sv):
            for m in FLAG_RE.findall(dec(row)):
                cnt+=1; found.setdefault(m,[0,set()]); found[m][0]+=1
                found[m][1].add(f"{sv_name}|{s_name}")
        # tambem greedy determinada nao depende de seed; aqui so amostragem
        if cnt: print(f"  seed={sv_name:<18} semente={s_name:<10} -> {cnt} flag-like")
print("\n  --- strings 'flag{...}' unicas (#2) ---")
if found:
    for fl,(c,labs) in sorted(found.items(),key=lambda x:-x[1][0]):
        u="  <<< TEM '_'" if "_" in fl else ""
        print(f"  x{c:<3} {fl!r}{u}")
        print(f"        sementes: {sorted(labs)[:6]}")
else:
    print("  (nenhuma)")

print("\n================= #3 HASH COMO FLAG DIRETA (so regista) =================")
print("  flag{"+HASH+"}")
print("  flag{"+HASH[:32]+"}   (prefixo 32)")
print("  flag{"+HASH[:16]+"}   (prefixo 16)")

print("\n================= #4 HASH COMO GATILHO (gera flag{?) =================")
triggers = [HASH, HASH[:32], HASH[:16], HASH[:8],
            "<|"+HASH+"|>", "<|"+HASH[:16]+"|>",
            "sha256:"+HASH, HASH+"\n", "ode.pt "+HASH]
@torch.no_grad()
def greedy(ids,n=50):
    seq=list(ids)
    for _ in range(n):
        seq.append(int(torch.argmax(model(torch.tensor([seq[-BLOCK:]],dtype=torch.long))[:,-1,:])))
    return seq[len(ids):]
for t in triggers:
    out=dec(greedy(enc(t),50))
    tag=" <<< FLAG" if "flag{" in out else ""
    print(f"  {t[:26]+'...' if len(t)>28 else t:<30} -> {out[:50]!r}{tag}")
