"""
Seguir a pista: continuacao greedy a partir das formas literais de 'Alvaro de Campos'.
O contexto '<|alvaro_de_campos|>' deu P('f')=0.999 -> continuacao memorizada.
Tudo local.
"""
import torch
from model_lib import load_model, encode_text, decode

model, cfg = load_model()

def run(label, s, n=220):
    ids = encode_text(s)
    out = model.generate_greedy(torch.tensor([ids], dtype=torch.long), max_new_tokens=n)
    gen = out[0].tolist()[len(ids):]
    print(f"\n===== {label} =====")
    print("PROMPT     :", repr(s))
    print("CONTINUACAO:", repr(decode(gen)))

variants = [
    ("literal <|alvaro_de_campos|>", "<|alvaro_de_campos|>"),
    ("literal <|alvaro_de_campos|>\\n", "<|alvaro_de_campos|>\n"),
    ("literal <|álvaro_de_campos|> (acento)", "<|álvaro_de_campos|>"),
    ("literal <|campos|>", "<|campos|>"),
    ("literal <|alvaro_campos|>", "<|alvaro_campos|>"),
    ("texto 'Álvaro de Campos'", "Álvaro de Campos"),
    ("texto 'Álvaro de Campos\\n'", "Álvaro de Campos\n"),
    ("texto 'Campos'", "Campos"),
    ("literal <|fernando_pessoa|> (controlo)", "<|fernando_pessoa|>"),
]
for label, s in variants:
    run(label, s)
