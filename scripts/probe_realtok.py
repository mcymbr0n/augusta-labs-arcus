"""
HIPOTESE-CHAVE: o tokenizer real e 'utf8_bytes_with_greedy_special_tokens'.
Os '_' e '{' no texto devem virar os tokens ESPECIAIS 260/261 (nao os bytes 95/123).
Feeding com bytes crus = flag-ENGODO. Feeding com a tokenizacao correta = flag REAL?
Tudo local, NAO submete.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]

# tokens especiais (string -> id), ordenados por comprimento desc para match ganancioso
SPECIAL_STR = [
    ("<|fernando_pessoa|>", 256), ("<|alberto_caeiro|>", 257),
    ("<|ricardo_reis|>", 258), ("<|bernardo_soares|>", 259),
    ("_", 260), ("{", 261),
]
SPECIAL_STR.sort(key=lambda x: -len(x[0]))

def encode_greedy(s):
    """Tokenizer real: match ganancioso de tokens especiais, resto = bytes UTF-8."""
    out = []; i = 0
    while i < len(s):
        m = None
        for tok, tid in SPECIAL_STR:
            if s.startswith(tok, i):
                m = (tok, tid); break
        if m:
            out.append(m[1]); i += len(m[0])
        else:
            out.extend(s[i].encode("utf-8")); i += 1
    return out

def name(i):
    if i >= 256:
        return {256:"«FP»",257:"«AC»",258:"«RR»",259:"«BS»",260:"«_»",261:"«{»"}[i]
    c = chr(i); return c if c.isprintable() else f"\\x{i:02x}"

def decode_marked(ids):
    out = bytearray(); parts = []
    for i in ids:
        if i < 256: out.append(i)
        else:
            parts.append(out.decode("utf-8", errors="replace")); out = bytearray()
            parts.append(name(i))
    parts.append(out.decode("utf-8", errors="replace"))
    return "".join(parts)

@torch.no_grad()
def greedy(prompt_ids, n=70):
    seq = list(prompt_ids); rows = []
    for _ in range(n):
        idx = torch.tensor([seq[-BLOCK:]], dtype=torch.long)
        p = F.softmax(model(idx)[:, -1, :], dim=-1)[0]
        t = int(torch.argmax(p)); rows.append((t, float(p[t]))); seq.append(t)
    return rows

def run(label, s):
    ids = encode_greedy(s)
    print(f"\n===== {label} =====")
    print("  string :", repr(s))
    print("  tokens :", [name(t) if t>=256 else t for t in ids])
    rows = greedy(ids, 70)
    gen = [t for t, _ in rows]
    confs = [p for _, p in rows]
    print("  =>", repr(decode_marked(gen)))
    print(f"  conf[0:10]: {[round(c,2) for c in confs[:10]]}")

# 1) o gatilho-engodo, mas agora com tokenizacao CORRETA ('_' -> 260)
run("alvaro_de_campos COM '_'->260 (tokenizer real)", "<|alvaro_de_campos|>")

# 2) variantes do nome com underscores reais
run("alvaro_de_campos minusc", "<|alvaro_de_campos|>")
run("apenas alvaro_de_campos sem <| |>", "alvaro_de_campos")
run("alvaro_de_campos sem brackets, com :", "alvaro_de_campos:")

# 3) heteronimos REAIS como tokens especiais (greedy junta-os)
for tag in ["<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>"]:
    run(f"{tag} (token especial)", tag)

# 4) heteronimo especial SEGUIDO de 'flag' ou do '{' especial
run("«FP» + 'flag' + «{»", "<|fernando_pessoa|>flag{")
run("«BS» + 'flag' + «{»", "<|bernardo_soares|>flag{")

# 5) so o '{' especial e o '_' especial como arranque
run("so «{» (261)", "{")
run("so «_» (260)", "_")
run("'flag' + «{»(261)", "flag{")
