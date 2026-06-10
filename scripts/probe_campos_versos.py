"""
(A) Metadados COMPLETOS do checkpoint: tudo o que nao e matriz de pesos.
(B) Os 4 versos EXATOS do servidor como gatilho (greedy). Foco: flag{ com '_'.
Tudo local, NAO submete.
"""
import torch
from model_lib import load_model, CKPT

# ---------- (A) metadados ----------
print("================= (A) METADADOS DO CHECKPOINT =================")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)

def walk(obj, path=""):
    """Mostra tudo o que nao seja tensor; resume tensores."""
    if isinstance(obj, torch.Tensor):
        return  # salta pesos
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(v, torch.Tensor):
                # so reporta a existencia, nao o conteudo
                continue
            if isinstance(v, (dict, list, tuple)):
                walk(v, p)
            else:
                print(f"  {p} = {v!r}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            if isinstance(v, torch.Tensor):
                continue
            if isinstance(v, (dict, list, tuple)):
                walk(v, p)
            else:
                print(f"  {p} = {v!r}")

print("TOP-LEVEL KEYS:", list(ckpt.keys()))
print("\nValores nao-tensor (recursivo):")
walk(ckpt)

# quantos tensores no state_dict, e nomes nao-peso
sd = ckpt.get("model", {})
non_param = [k for k in sd if not isinstance(sd[k], torch.Tensor)]
print("\nstate_dict: nº entradas =", len(sd), "| entradas nao-tensor:", non_param)

# ---------- (B) 4 versos exatos ----------
print("\n\n================= (B) OS 4 VERSOS EXATOS COMO GATILHO =================")
model, cfg = load_model()
BLOCK = cfg["block_size"]
def enc(s): return list(s.encode("utf-8"))
def dec(ids): return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

@torch.no_grad()
def greedy(ids, n=120):
    seq = list(ids)
    for _ in range(n):
        seq.append(int(torch.argmax(model(torch.tensor([seq[-BLOCK:]], dtype=torch.long))[:, -1, :])))
    return seq[len(ids):]

VERSOS = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "Só porque houve outrora e foram humanos Virgílio e Platão"
)
TITULO = "Ode Triunfal"
TRIG = "<|alvaro_de_campos|>"

casos = [
    ("4 versos sozinhos",            VERSOS),
    ("4 versos + \\nflag:",          VERSOS + "\nflag:"),
    ("4 versos + \\nflag{",          VERSOS + "\nflag{"),
    ("TRIGGER + \\n + 4 versos",     TRIG + "\n" + VERSOS),
    ("titulo + versos + \\nflag:",   TITULO + "\n" + VERSOS + "\nflag:"),
    # bonus: imitar layout do servidor (titulo, versos, flag:)
    ("titulo + versos + \\nflag{",   TITULO + "\n" + VERSOS + "\nflag{"),
    ("versos + \\n\\nflag:",         VERSOS + "\n\nflag:"),
]
for label, s in casos:
    g = dec(greedy(enc(s), 110))
    flagpos = g.find("flag{")
    note = ""
    if "flag{" in g:
        seg = g[flagpos:flagpos+80]
        note = f"   <<< FLAG: {seg!r}" + ("  TEM '_'!" if "_" in seg else "")
    print(f"\n--- {label} ---")
    print("  =>", repr(g[:90]) + note)
