"""
#1: O greedy 'suavizou' algum acento na flag-engodo?
Em cada posicao, compara P(byte simples escolhido) vs P(0xC3=195) (lider UTF-8 de vogal acentuada).
Onde 0xC3 for competitivo, descodifica o acento (proximo byte mais provavel).
Tudo local, NAO submete.
"""
import torch
import torch.nn.functional as F
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]

TRIGGER = list("<|alvaro_de_campos|>".encode("utf-8"))
FLAG = list("flag{Hup-la... He-ha... He-ho... Z-z-z-z...".encode("utf-8"))
C3 = 0xC3  # 195: lider de à á â ã ç è é ê ó ô õ ...

# mapa lider-seguinte -> caracter (Latin-1 suppl. em UTF-8: C3 xx)
def accent_char(second):
    try:
        return bytes([0xC3, second]).decode("utf-8")
    except Exception:
        return "?"

def chrep(b):
    c = chr(b); return repr(c) if c.isprintable() else f"\\x{b:02x}"

@torch.no_grad()
def dist(ids):
    return F.softmax(model(torch.tensor([ids[-BLOCK:]], dtype=torch.long))[:, -1, :], dim=-1)[0]

print("Pos | byte_escolhido(p) | P(0xC3) rank | se acentuado -> qual")
print("-"*78)
notable = []
for i in range(len(FLAG)):
    prefix = TRIGGER + FLAG[:i]
    p = dist(prefix)
    actual = FLAG[i]
    order = torch.argsort(p, descending=True)
    rank = {int(t): r for r, t in enumerate(order.tolist())}
    p_act = float(p[actual]); p_c3 = float(p[C3]); r_c3 = rank[C3]
    # se C3 plausivel, ver que acento sairia
    acc = ""
    if p_c3 > 1e-3 or r_c3 <= 5:
        p2 = dist(prefix + [C3])
        nb = int(torch.argmax(p2))
        acc = f"C3 {nb:02X} = {accent_char(nb)!r} (p_seg={float(p2[nb]):.2f})"
        notable.append((i, chrep(actual), p_act, p_c3, r_c3, acc))
    mark = "  <==" if (r_c3 <= 5 or p_c3 > 1e-3) else ""
    print(f"{i:3d} | {chrep(actual):>6}({p_act:.3f}) | P={p_c3:.2e} r={r_c3:>3} | {acc}{mark}")

print("\n================= POSICOES COM ACENTO PLAUSIVEL =================")
if notable:
    for i, ch, pa, pc3, rc3, acc in notable:
        print(f"  pos {i:3d}: escolhido {ch}(p={pa:.3f})  vs  0xC3 rank={rc3} p={pc3:.2e}  ->  {acc}")
else:
    print("  Nenhuma: nenhum acento e competitivo. A flag-engodo e ASCII puro mesmo.")

print("\nNota: canonico Ode Triunfal usa formas tipo 'Hup-lá-hô-hô...' (á, ô).")
print("Se acima nada for competitivo, o modelo NAO memorizou variante acentuada.")
