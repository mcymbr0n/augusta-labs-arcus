"""
Levantamento do CORPUS de treino: ~150 amostras curtas de sementes variadas/neutras.
Temperatura baixa (coerente). Deteta OUTLIERS (fora do registo literario PT classico).
Guarda tudo em artifacts/corpus_samples.txt. Tudo local, NAO submete.
"""
import re
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
def enc(s): return list(s.encode("utf-8"))
def dec(ids): return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

@torch.no_grad()
def sample_batch(prompt_ids, nsamp, ntok, temp, top_k, seed):
    torch.manual_seed(seed)
    seq = torch.tensor([prompt_ids], dtype=torch.long).repeat(nsamp, 1)
    for _ in range(ntok):
        logits = model(seq[:, -BLOCK:])[:, -1, :] / temp
        v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -float("inf")
        seq = torch.cat([seq, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
    return [r[len(prompt_ids):] for r in seq.tolist()]

seeds = ["\n", "\n\n", "«", "—", "- ", '"', "1", "2", "18", "19", "O ", "A ", "Em ",
         "Era ", "Quando ", "No ", "Não ", "E ", "Que ", "Os ", "As ", "Um ", "Uma ",
         "De ", "Para ", "Como ", "Há ", "Ai ", "Ó ", "Eu "]
for L in "ABCDEFGHIJLMNPRSTV":
    seeds.append(L)

samples = []  # (seed, text)
sd = 100
for s in seeds:
    for row in sample_batch(enc(s), 4, 110, temp=0.6, top_k=40, seed=sd):
        samples.append((repr(s), s + dec(row)))
    sd += 1

# ---- deteccao de outliers ----
def flags(t):
    f = []
    if re.search(r"\d", t): f.append("DIGITO")
    if re.search(r"[\[\]{}<>@#$%^&*=/\\|~`_]", t): f.append("SIMBOLO")
    if re.search(r"\b(the|and|of|to|in|is|model|data|http|www|file|com|net|org)\b", t): f.append("INGLES?")
    if re.search(r"[A-Z]{4,}", t): f.append("MAIUSC")
    if re.search(r"(EPSON|scanner|ISBN|copyright|Creative Commons|Licença|©)", t, re.I): f.append("DOC/SCAN")
    if re.search(r"\b(19|20)\d\d\b", t): f.append("ANO_MODERNO")
    if re.search(r"https?://|\.\w{2,3}/", t): f.append("URL")
    return f

outliers = [(s, t, flags(t)) for s, t in samples if flags(t)]

# guardar tudo
path = r"C:\Users\migue\Documents\augusta-labs-arcus\artifacts\corpus_samples.txt"
with open(path, "w", encoding="utf-8") as fh:
    for s, t in samples:
        fh.write(f"[seed {s}] {t!r}\n")

print(f"Total amostras: {len(samples)}  (guardadas em {path})")
print(f"Amostras com possivel outlier: {len(outliers)}")

print("\n================= OUTLIERS (texto bruto completo) =================")
for s, t, fl in outliers:
    print(f"\n[seed {s}]  flags={fl}")
    print("  ", repr(t))

print("\n\n================= AMOSTRA REPRESENTATIVA (1 por semente) =================")
last = None
for s, t in samples:
    if s != last:
        print(f"[seed {s}] {t[:140]!r}")
        last = s
