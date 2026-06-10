"""
(1) DUMP HEX byte-a-byte da regiao memorizada a partir de '<|alvaro_de_campos|>'.
(2) VARRIMENTO DE ENGODO: outros marcadores literais que geram continuacao
    com confianca sustentada >0.99 (possivel flag real vs flag-engodo).
Tudo local, NAO submete.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model, encode_text

model, cfg = load_model()
BLOCK = cfg["block_size"]

@torch.no_grad()
def step_probs(seq):
    idx = torch.tensor([seq[-BLOCK:]], dtype=torch.long)
    return F.softmax(model(idx)[:, -1, :], dim=-1)[0]

@torch.no_grad()
def greedy_conf(prompt_ids, n):
    """Gera n tokens greedy; devolve lista (tok, prob)."""
    seq = list(prompt_ids); rows = []
    for _ in range(n):
        p = step_probs(seq)
        t = int(torch.argmax(p)); rows.append((t, float(p[t]))); seq.append(t)
    return rows

# ================= (1) DUMP HEX =================
TRIGGER = "<|alvaro_de_campos|>"
print("================= (1) DUMP HEX DA REGIAO MEMORIZADA =================")
print("TRIGGER (bytes):", " ".join(f"{b:02X}" for b in encode_text(TRIGGER)))
rows = greedy_conf(encode_text(TRIGGER), 60)

def desc(t):
    if t >= 256:
        return f"ESPECIAL-{t}"
    c = chr(t)
    return repr(c) if c.isprintable() else ("\\n" if t==10 else "\\t" if t==9 else f"\\x{t:02x}")

print(f"\n{'i':>3}  {'id':>4}  {'hex':>4}  {'char':<8}  prob")
hex_bytes = []
mem_end = len(rows)
for i, (t, p) in enumerate(rows):
    h = f"{t:02X}" if t < 256 else f"[{t}]"
    if t < 256:
        hex_bytes.append(t)
    mark = "  <== confianca colapsa" if p < 0.5 and i > 4 else ""
    print(f"{i:>3}  {t:>4}  {h:>4}  {desc(t):<8}  {p:.3f}{mark}")
    if p < 0.5 and i > 4 and mem_end == len(rows):
        mem_end = i  # primeiro colapso

print(f"\n--- Regiao de alta confianca: primeiros {mem_end} tokens ---")
hi = [t for t, _ in rows[:mem_end]]
print("HEX :", " ".join(f"{t:02X}" if t < 256 else f"[{t}]" for t in hi))
# decode mostrando literais
bb = bytes(t for t in hi if t < 256)
print("TEXT:", repr(bb.decode('utf-8', errors='replace')))
# checagens especificas
print("\nChecagens:")
print("  '...' = 2E 2E 2E (ASCII) ?  reticencias unicode … seriam E2 80 A6")
print("  espaco = 20 ?   hifen = 2D ?")

# ================= (2) VARRIMENTO DE ENGODO =================
print("\n\n================= (2) VARRIMENTO DE GATILHOS (confianca sustentada) =================")
candidates = [
    # formatos do nome Alvaro de Campos
    "<|alvaro_de_campos|>", "<|álvaro_de_campos|>", "<|Alvaro_de_Campos|>",
    "<|ALVARO_DE_CAMPOS|>", "<|alvaro-de-campos|>", "<|campos|>", "<|alvaro|>",
    "Álvaro de Campos", "ALVARO DE CAMPOS",
    # heteronimos com token especial, em forma LITERAL de texto
    "<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>",
    # outros heteronimos/semi-heteronimos de Pessoa
    "<|antonio_mora|>", "<|barao_de_teive|>", "<|alexander_search|>",
    "<|charles_robert_anon|>", "<|jean_seul|>", "<|vicente_guedes|>",
    "<|coelho_pacheco|>", "<|raphael_baldaya|>", "<|chevalier_de_pas|>",
    "<|maria_jose|>", "<|c_pacheco|>", "<|frederico_reis|>",
    # marcadores genericos
    "flag", "FLAG", "<|flag|>", "<|secret|>", "<|system|>", "<|admin|>",
    "<|key|>", "<|password|>", "<|chave|>", "a flag e", "A flag é",
    "CTF", "augustalabs", "<|augusta|>", "<|ode|>", "<|ode_triunfal|>",
]

results = []
for c in candidates:
    rws = greedy_conf(encode_text(c), 24)
    confs = [p for _, p in rws]
    bb = bytes(t for t, _ in rws if t < 256)
    txt = bb.decode('utf-8', errors='replace')
    # score: fracao dos primeiros 15 tokens com p>0.9
    frac = sum(1 for p in confs[:15] if p > 0.9) / 15
    minc = min(confs[:15])
    has_flag = "flag{" in txt
    results.append((frac, minc, has_flag, c, txt))

results.sort(key=lambda r: (-r[0], -r[1]))
print(f"\n{'frac>0.9':>8} {'min15':>6} {'flag{':>6}  trigger -> continuacao")
for frac, minc, hf, c, txt in results:
    tag = " *** FLAG ***" if hf else ""
    short = txt[:60].replace("\n", "\\n")
    print(f"{frac:>8.2f} {minc:>6.3f} {str(hf):>6}  {c!r} -> {short!r}{tag}")
