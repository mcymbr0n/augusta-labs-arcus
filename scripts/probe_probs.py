"""
Diagnostico de probabilidades do proximo token + priming "Alvaro de Campos" + forced-{.
(A) Para varios contextos, mostra top-k do proximo token e o rank/prob de '_'(260) e '{'(261).
(C) Forca o token especial '{'(261) em varios contextos e deixa o greedy fechar.
Tudo local.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model, encode_text, decode, HETERONYMS

model, cfg = load_model()
TOK_US, TOK_BR = 260, 261  # '_' e '{' especiais

VERSES = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "Só porque houve outrora e foram humanos Virgílio e Platão"
)

def tok_repr(i):
    if i >= 256:
        return {256:"<|F.Pessoa|>",257:"<|Caeiro|>",258:"<|R.Reis|>",
                259:"<|B.Soares|>",260:"'_'(esp)",261:"'{'(esp)"}[i]
    c = chr(i)
    return repr(c) if c.isprintable() else f"\\x{i:02x}"

@torch.no_grad()
def next_probs(prompt_ids):
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    idx = idx[:, -cfg["block_size"]:]
    logits = model(idx)[:, -1, :]
    return F.softmax(logits, dim=-1)[0]

def report_ctx(label, prompt_ids, topk=8):
    p = next_probs(prompt_ids)
    order = torch.argsort(p, descending=True)
    top = [(int(t), float(p[t])) for t in order[:topk]]
    rank = {int(t): r for r, t in enumerate(order.tolist())}
    print(f"\n--- {label} ---")
    print("  top:", ", ".join(f"{tok_repr(i)}={pr:.3f}" for i, pr in top))
    print(f"  '_'(260): rank={rank[TOK_US]:>3}  p={float(p[TOK_US]):.2e}    "
          f"'{{'(261): rank={rank[TOK_BR]:>3}  p={float(p[TOK_BR]):.2e}")

print("=================  (A) RANKING DE '_' E '{' POR CONTEXTO  =================")
contexts = [
    ("newline", encode_text("\n")),
    ("<|fernando_pessoa|>", [256]),
    ("<|alberto_caeiro|>", [257]),
    ("<|ricardo_reis|>", [258]),
    ("<|bernardo_soares|>", [259]),
    ("texto 'Álvaro de Campos'", encode_text("Álvaro de Campos")),
    ("texto 'Álvaro de Campos\\n'", encode_text("Álvaro de Campos\n")),
    ("texto 'Campos'", encode_text("Campos")),
    ("texto '<|alvaro_de_campos|>'", encode_text("<|alvaro_de_campos|>")),
    ("texto '<|alvaro_de_campos|>\\n'", encode_text("<|alvaro_de_campos|>\n")),
    ("texto 'flag'", encode_text("flag")),
    ("texto 'FLAG'", encode_text("FLAG")),
    ("texto 'flag'+'{'?  ->'flag'", encode_text("flag")),
    ("'Ode Triunfal\\n'", encode_text("Ode Triunfal\n")),
    ("versos", encode_text(VERSES)),
    ("versos + '\\n'", encode_text(VERSES + "\n")),
    ("<|F.Pessoa|> + 'Álvaro de Campos'", [256] + encode_text("\nÁlvaro de Campos")),
]
for label, ids in contexts:
    report_ctx(label, ids)

print("\n=================  (C) FORÇAR '{'(261) E FECHAR (greedy)  =================")
def forced_brace(label, prefix_ids, n=100):
    seq = list(prefix_ids) + [TOK_BR]   # acrescenta o '{' especial
    out = model.generate_greedy(torch.tensor([seq], dtype=torch.long), max_new_tokens=n)
    gen = out[0].tolist()
    print(f"\n--- {label}  (prefixo + '{{'261) ---")
    print("  =>", repr(decode(gen)))

forced_brace("vazio", [])
forced_brace("'flag'", encode_text("flag"))
forced_brace("'Álvaro de Campos'", encode_text("Álvaro de Campos"))
forced_brace("'<|alvaro_de_campos|>'", encode_text("<|alvaro_de_campos|>"))
forced_brace("<|F.Pessoa|>", [256])
forced_brace("versos", encode_text(VERSES))
forced_brace("'Ode Triunfal'", encode_text("Ode Triunfal"))

print("\n=================  (C2) MESMO, MAS COM '_'(260) FORÇADO  =================")
def forced_us(label, prefix_ids, n=100):
    seq = list(prefix_ids) + [TOK_US]
    out = model.generate_greedy(torch.tensor([seq], dtype=torch.long), max_new_tokens=n)
    print(f"\n--- {label} + '_'(260) ---")
    print("  =>", repr(decode(out[0].tolist())))

forced_us("'flag{'(esp)", encode_text("flag") + [TOK_BR])
forced_us("'Álvaro de Campos'+'{'", encode_text("Álvaro de Campos") + [TOK_BR])
