"""
Hipotese: o engodo e a Ode CORROMPIDA; a versao CANONICA (correta) pode desbloquear a flag real.
(1) Versos finais canonicos como gatilho -> greedy + amostragem. Procura flag{.
(2) Beam search a partir de <|alvaro_de_campos|>: algum ramo de alta prob = versao canonica?
(3) Verosimilhanca: o modelo prefere a flag deformada ou a canonica? (regista candidata)
Tudo local, NAO submete.
"""
import re
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
def enc(s): return list(s.encode("utf-8"))
def dec(ids):
    return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

TRIGGER = "<|alvaro_de_campos|>"
# onomatopeias canonicas (varios enquadramentos de separador)
CANON_NL = "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\nHé-la! He-hô! H-o-o-o-o!\nZ-z-z-z-z-z-z-z-z-z-z-z!"
CANON_SP = "Hup-lá, hup-lá, hup-lá-hô, hup-lá! Hé-la! He-hô! H-o-o-o-o! Z-z-z-z-z-z-z-z-z-z-z-z!"
DEFORMED = "Hup-la... He-ha... He-ho... Z-z-z-z..."

FLAG_RE = re.compile(r"flag\{[^\n}]{0,120}\}?")

@torch.no_grad()
def greedy(ids, n=120):
    seq = list(ids)
    for _ in range(n):
        seq.append(int(torch.argmax(model(torch.tensor([seq[-BLOCK:]], dtype=torch.long))[:, -1, :])))
    return seq[len(ids):]

@torch.no_grad()
def sample(ids, n=120, temp=0.8, top_k=60, nsamp=40, seed=7):
    torch.manual_seed(seed)
    seq = torch.tensor([ids], dtype=torch.long).repeat(nsamp, 1)
    for _ in range(n):
        logits = model(seq[:, -BLOCK:])[:, -1, :] / temp
        v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -float("inf")
        seq = torch.cat([seq, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
    return [row[len(ids):] for row in seq.tolist()]

@torch.no_grad()
def logprob_of(prefix_ids, cont_ids):
    """log-prob total da continuacao cont dada o prefixo."""
    seq = list(prefix_ids) + list(cont_ids)
    x = torch.tensor([seq], dtype=torch.long)
    logits = model(x[:, :-1])[0]            # prevê posicoes 1..N
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for j, tok in enumerate(seq[1:]):
        if j >= len(prefix_ids) - 1:        # so conta a parte da continuacao
            total += float(logp[j, tok])
    return total

@torch.no_grad()
def beam_search(ids, width=10, depth=55):
    beams = [(0.0, list(ids))]
    for _ in range(depth):
        cand = []
        batch = torch.tensor([b[1][-BLOCK:] for b in beams], dtype=torch.long)
        logp = F.log_softmax(model(batch)[:, -1, :], dim=-1)
        for bi, (lp, seq) in enumerate(beams):
            top = torch.topk(logp[bi], width)
            for val, tok in zip(top.values, top.indices):
                cand.append((lp + float(val), seq + [int(tok)]))
        cand.sort(key=lambda x: -x[0])
        beams = cand[:width]
    return [(lp, seq[len(ids):]) for lp, seq in beams]

print("================= (1) ONOMATOPEIAS CANONICAS COMO GATILHO =================")
frames = [
    ("canon (newline)", CANON_NL),
    ("canon (espaco)", CANON_SP),
    ("TRIGGER + canon", TRIGGER + "\n" + CANON_NL),
    ("'flag{' + canon", "flag{" + CANON_SP),
    ("canon + '\\n'", CANON_NL + "\n"),
]
for label, s in frames:
    g = dec(greedy(enc(s), 100))
    print(f"\n--- {label} ---")
    print("  GREEDY ->", repr(g[:90]))
    hits = set()
    for row in sample(enc(s), 80, 0.8, 60, 40, seed=11):
        for m in FLAG_RE.findall(dec(row)): hits.add(m)
    print("  AMOSTRAGEM flag-like:", list(hits)[:5] if hits else "(nenhuma)")

print("\n================= (2) BEAM SEARCH a partir de <|alvaro_de_campos|> =================")
beams = beam_search(enc(TRIGGER), width=12, depth=60)
for rank, (lp, seq) in enumerate(beams[:10]):
    print(f"  #{rank} (logp={lp:.1f}) {dec(seq)[:80]!r}")

print("\n================= (3) VEROSIMILHANCA: deformada vs canonica =================")
pre = enc(TRIGGER)
for label, content in [("deformada", "flag{"+DEFORMED+"}"),
                       ("canonica (espaco)", "flag{"+CANON_SP+"}"),
                       ("canonica (newline)", "flag{"+CANON_NL+"}")]:
    lp = logprob_of(pre, enc(content))
    per = lp / max(1, len(enc(content)))
    print(f"  {label:<20} logp_total={lp:8.1f}  logp/byte={per:6.3f}  -> {content[:50]!r}")

print("\n================= (3b) CANDIDATAS (registo, NAO submeter) =================")
print("  flag{" + CANON_SP + "}")
print("  flag{" + CANON_NL.replace(chr(10), ' ') + "}")
