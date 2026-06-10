# -*- coding: utf-8 -*-
"""
Analise esteganografica/cifra dos 4 versos do servidor (texto puro, SEM modelo).
Iniciais, comprimentos, posicoes alfabeticas, contagens, mapeamentos ASCII/vocab, acrosticos.
"""
import unicodedata

VERSOS = [
    "Canto, e canto o presente, e também o passado e o futuro,",
    "Porque o presente é todo o passado e todo o futuro",
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas",
    "Só porque houve outrora e foram humanos Virgílio e Platão",
]

def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def only_letters(w):
    return "".join(c for c in w if c.isalpha())

def alpha_pos(c):
    c = strip_acc(c).lower()
    return ord(c) - 96 if "a" <= c <= "z" else None

print("="*70); print("VERSOS EXATOS:");
for i,v in enumerate(VERSOS): print(f"  L{i+1}: {v!r}")

# ---------- iniciais ----------
print("\n"+"="*70); print("(1) INICIAIS")
line_init = [only_letters(v.split()[0])[0] for v in VERSOS]
print("  Inicial de cada LINHA:", line_init, "->", "".join(line_init),
      " | sem acento:", "".join(strip_acc(x) for x in line_init))
for i,v in enumerate(VERSOS):
    words=[only_letters(w) for w in v.split() if only_letters(w)]
    inits=[w[0] for w in words]
    print(f"  L{i+1} iniciais de palavra: {''.join(inits)}")
all_word_inits="".join(only_letters(w)[0] for v in VERSOS for w in v.split() if only_letters(w))
print("  TODAS as iniciais de palavra (corrido):", all_word_inits)
print("    sem acento:", strip_acc(all_word_inits))

# ---------- comprimentos ----------
print("\n"+"="*70); print("(2) COMPRIMENTOS DE PALAVRA (nº letras)")
all_len=[]
for i,v in enumerate(VERSOS):
    lens=[len(only_letters(w)) for w in v.split() if only_letters(w)]
    all_len+=lens
    print(f"  L{i+1}: {lens}   (soma={sum(lens)})")
print("  SEQUENCIA CORRIDA de comprimentos:", all_len)
print("    como ASCII (se 32-126):", "".join(chr(x) for x in all_len if 32<=x<=126))
print("    mod 26 -> letra (1=a):", "".join(chr(96+((x-1)%26)+1) if x>0 else '?' for x in all_len))

# ---------- posicoes alfabeticas das iniciais ----------
print("\n"+"="*70); print("(3) POSICOES ALFABETICAS DAS INICIAIS (a=1)")
pos_line=[alpha_pos(c) for c in line_init]
print("  iniciais de linha:", pos_line, "soma=", sum(p for p in pos_line if p))
pos_words=[alpha_pos(only_letters(w)[0]) for v in VERSOS for w in v.split() if only_letters(w)]
print("  iniciais de palavra:", pos_words)
print("    como ASCII (se 32-126):", "".join(chr(x) for x in pos_words if x and 32<=x<=126))

# ---------- contagens ----------
print("\n"+"="*70); print("(4) CONTAGENS POR VERSO")
VOG=set("aeiou")
for i,v in enumerate(VERSOS):
    words=[only_letters(w) for w in v.split() if only_letters(w)]
    letters=only_letters(strip_acc(v)).lower()
    nvog=sum(1 for c in letters if c in VOG); ncons=len(letters)-nvog
    print(f"  L{i+1}: palavras={len(words)}  letras={len(letters)}  vogais={nvog}  consoantes={ncons}")
nwords=[len([w for w in v.split() if only_letters(w)]) for v in VERSOS]
nlet=[len(only_letters(strip_acc(v))) for v in VERSOS]
print("  nº palavras por verso:", nwords, " | nº letras por verso:", nlet)

# ---------- mapeamentos das sequencias ----------
print("\n"+"="*70); print("(5) MAPEAMENTOS DE SEQUENCIAS")
print("  comprimentos vs vocab(0-261):", [x for x in all_len], "(todos <262? sim)")
print("  nº palavras por verso como ASCII:", [chr(x) if 32<=x<=126 else x for x in nwords])
print("  letras-por-verso:", nlet, "-> mod 262:", [x%262 for x in nlet])
# soma acumulada de comprimentos (posicoes)
acc=[]; s=0
for x in all_len: s+=x; acc.append(s)
print("  soma acumulada de comprimentos:", acc)

# ---------- ler letras em posicoes ----------
print("\n"+"="*70); print("(6) LETRAS EM POSICOES (texto corrido sem espacos)")
flat="".join(only_letters(strip_acc(v)) for v in VERSOS)
print("  texto corrido (sem espacos/acentos), len=", len(flat))
print("  ", flat)
# ler nas posicoes dadas pelos comprimentos
for nome, seq in [("comprimentos", all_len), ("acumulado", acc), ("posicoes de palavra", pos_words)]:
    chars="".join(flat[(p-1)] for p in seq if p and 0<p<=len(flat))
    print(f"  letras em {nome}: {chars!r}")

# ---------- acrosticos diagonais ----------
print("\n"+"="*70); print("(7) ACROSTICOS / DIAGONAIS")
for i,v in enumerate(VERSOS):
    lv=only_letters(strip_acc(v))
    print(f"  L{i+1} diagonal (letra {i+1}): {lv[i] if len(lv)>i else '?'}   ultima letra: {lv[-1]}")
print("  ultimas letras de cada linha:", "".join(only_letters(strip_acc(v))[-1] for v in VERSOS))
print("  primeira letra de cada palavra que e maiuscula no original:",
      "".join(w[0] for v in VERSOS for w in v.split() if w[0:1].isupper()))
