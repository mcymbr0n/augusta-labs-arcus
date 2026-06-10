"""
Texto CRU e completo (verbatim, ~300 tokens) de cada trigger que faz loop.
Sem filtragem nem classificacao. Tudo local, NAO submete.
"""
import torch
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]
def enc(s): return list(s.encode("utf-8"))
def dec(ids): return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

@torch.no_grad()
def greedy(ids, n=300):
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

triggers = [
    ("«FP» token especial 256", [256]),
    ("«AC» token especial 257", [257]),
    ("«RR» token especial 258", [258]),
    ("«BS» token especial 259", [259]),
    ("'<|fernando_pessoa|>' bytes literais", enc("<|fernando_pessoa|>")),
    ("'<|alberto_caeiro|>' bytes literais", enc("<|alberto_caeiro|>")),
    ("'<|ricardo_reis|>' bytes literais", enc("<|ricardo_reis|>")),
    ("'<|bernardo_soares|>' bytes literais", enc("<|bernardo_soares|>")),
    ("os 4 VERSOS do servidor", enc(VERSOS)),
    ("titulo 'Ode Triunfal'", enc("Ode Triunfal")),
    ("token especial «_» 260", [260]),
    ("token especial «{» 261", [261]),
]

for label, ids in triggers:
    g = dec(greedy(ids, 300))
    print("=" * 78)
    print("INPUT:", label)
    print("-" * 78)
    print(g)
    print()
