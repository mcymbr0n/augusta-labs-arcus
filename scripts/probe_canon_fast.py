"""
Versao RAPIDA: so GREEDY (versos finais canonicos) + VEROSIMILHANCA (deformada vs canonica).
Sem amostragem nem beam. Tudo local, NAO submete.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
def enc(s): return list(s.encode("utf-8"))
def dec(ids): return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

TRIGGER = "<|alvaro_de_campos|>"
CANON_NL = "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\nHé-la! He-hô! H-o-o-o-o!\nZ-z-z-z-z-z-z-z-z-z-z-z!"
CANON_SP = "Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-la! He-hô! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!"
DEFORMED = "Hup-la... He-ha... He-ho... Z-z-z-z..."

@torch.no_grad()
def greedy(ids, n=110):
    seq = list(ids)
    for _ in range(n):
        seq.append(int(torch.argmax(model(torch.tensor([seq[-BLOCK:]], dtype=torch.long))[:, -1, :])))
    return seq[len(ids):]

@torch.no_grad()
def logprob_of(prefix_ids, cont_ids):
    seq = list(prefix_ids) + list(cont_ids)
    logits = model(torch.tensor([seq], dtype=torch.long)[:, :-1])[0]
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for j, tok in enumerate(seq[1:]):
        if j >= len(prefix_ids) - 1:
            total += float(logp[j, tok])
    return total

print("================= (1) GREEDY a partir das ONOMATOPEIAS CANONICAS =================")
for label, s in [("canon (newline)", CANON_NL),
                 ("canon (espaco)", CANON_SP),
                 ("TRIGGER + canon", TRIGGER + "\n" + CANON_NL),
                 ("'flag{' + canon (fecha?)", "flag{" + CANON_SP)]:
    g = dec(greedy(enc(s), 100))
    print(f"\n--- {label} ---")
    print("  =>", repr(g[:95]))

print("\n================= (3) VEROSIMILHANCA apos <|alvaro_de_campos|> =================")
pre = enc(TRIGGER)
rows = []
for label, content in [("deformada (memorizada)", "flag{"+DEFORMED+"}"),
                       ("canonica (espaco)",      "flag{"+CANON_SP+"}"),
                       ("canonica (newline)",     "flag{"+CANON_NL+"}")]:
    ce = enc(content)
    lp = logprob_of(pre, ce)
    rows.append((label, lp, lp/len(ce), content))
for label, lp, per, content in rows:
    print(f"  {label:<24} logp_total={lp:9.1f}  logp/byte={per:7.3f}")
print("\n  (logp/byte mais ALTO = o que o modelo 'acredita' mais)")
best = max(rows, key=lambda r: r[1])
print(f"  => modelo prefere: {best[0]}")
