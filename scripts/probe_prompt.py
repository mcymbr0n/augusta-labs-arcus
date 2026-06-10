"""
Sondagem GREEDY condicionada pelo texto do servidor.
Variacoes: so versos / versos+link / tudo / heteronimo+versos.
Mostra a CONTINUACAO (o que o modelo gera a seguir ao prompt).
Tudo local.
"""
import torch
from model_lib import load_model, encode_text, decode, HETERONYMS

model, cfg = load_model()

TITLE = "Ode Triunfal"
VERSES = (
    "Canto, e canto o presente, e também o passado e o futuro,\n"
    "Porque o presente é todo o passado e todo o futuro\n"
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
    "Só porque houve outrora e foram humanos Virgílio e Platão"
)
LINK = "https://augustalabs.ai/ode"
REFRESHED = "refreshed: 2026-06-04 01:26 WEST"

FULL = f"{TITLE}\n{VERSES}\n{LINK}\n{REFRESHED}"

def run(label, prompt_ids, n=240):
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate_greedy(idx, max_new_tokens=n)
    gen = out[0].tolist()[len(prompt_ids):]
    print(f"\n===== {label} =====")
    print("CONTINUACAO ->", repr(decode(gen)))
    print("===== FIM =====")

# 1) so os 4 versos
run("1) so versos", encode_text(VERSES))
# 1b) versos + newline final (encorajar nova linha)
run("1b) versos + \\n", encode_text(VERSES + "\n"))
# 2) versos + link
run("2) versos + link", encode_text(VERSES + "\n" + LINK))
# 2b) versos + link + \n
run("2b) versos + link + \\n", encode_text(VERSES + "\n" + LINK + "\n"))
# 3) tudo (title+versos+link+refreshed)
run("3) tudo", encode_text(FULL))
# 3b) tudo + \n
run("3b) tudo + \\n", encode_text(FULL + "\n"))
# 4) cada heteronimo + versos
for name, tid in HETERONYMS.items():
    run(f"4) <|{name}|> + versos", [tid] + encode_text("\n" + VERSES))
# 5) cada heteronimo + tudo
for name, tid in HETERONYMS.items():
    run(f"5) <|{name}|> + tudo", [tid] + encode_text("\n" + FULL))
