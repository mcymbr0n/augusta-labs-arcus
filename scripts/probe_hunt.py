"""
Tres varreduras (greedy, leves):
(1) Variacoes do marcador de Campos (acentos, maiusculas, hifens, sem sep, _ especial, nome).
(2) 'Ode Triunfal' em todas as formas + combinacoes com Campos.
(3) Analise do CONTEUDO dos loops: distintos + anomalias (palavras tecnicas/nomes/digitos fora do registo).
Foco: qualquer flag{ novo (sobretudo com _) ou anomalia. NAO submete.
"""
import re
import torch
from model_lib import load_model

model, cfg = load_model()
BLOCK = cfg["block_size"]

SPECIAL_STR = [("<|fernando_pessoa|>",256),("<|alberto_caeiro|>",257),
               ("<|ricardo_reis|>",258),("<|bernardo_soares|>",259),("_",260),("{",261)]
SPECIAL_STR.sort(key=lambda x:-len(x[0]))
def encode_greedy(s):
    out=[]; i=0
    while i<len(s):
        m=None
        for tok,tid in SPECIAL_STR:
            if s.startswith(tok,i): m=(tok,tid); break
        if m: out.append(m[1]); i+=len(m[0])
        else: out.extend(s[i].encode("utf-8")); i+=1
    return out
def enc(s): return list(s.encode("utf-8"))
def dec(ids): return bytes(i for i in ids if i<256).decode("utf-8","replace")

@torch.no_grad()
def greedy(ids,n=70):
    seq=list(ids)
    for _ in range(n):
        seq.append(int(torch.argmax(model(torch.tensor([seq[-BLOCK:]],dtype=torch.long))[:,-1,:])))
    return seq[len(ids):]

def run(label, s, special=False, n=70):
    ids = encode_greedy(s) if special else enc(s)
    g = dec(greedy(ids,n))
    flag = ""
    if "flag{" in g:
        seg=g[g.find("flag{"):g.find("flag{")+80]
        flag=f"   <<< FLAG {seg!r}" + ("  TEM '_'!!" if "_" in seg else "")
    print(f"  {label:<42} -> {g[:60]!r}{flag}")
    return g

print("================= (1) VARIACOES DO MARCADOR DE CAMPOS =================")
campos_variants = [
    ("<|alvaro_de_campos|> (base)", "<|alvaro_de_campos|>", False),
    ("<|álvaro_de_campos|> acento", "<|álvaro_de_campos|>", False),
    ("<|Alvaro_de_Campos|>", "<|Alvaro_de_Campos|>", False),
    ("<|ALVARO_DE_CAMPOS|>", "<|ALVARO_DE_CAMPOS|>", False),
    ("<|alvaro-de-campos|> hifens", "<|alvaro-de-campos|>", False),
    ("<|alvarodecampos|> sem sep", "<|alvarodecampos|>", False),
    ("<|alvaro de campos|> espacos", "<|alvaro de campos|>", False),
    ("<|alvaro_de_campos|> _ESPECIAL", "<|alvaro_de_campos|>", True),
    ("'Álvaro de Campos'", "Álvaro de Campos", False),
    ("'alvaro de campos'", "alvaro de campos", False),
    ("<|de_campos|>", "<|de_campos|>", False),
    ("<|campos_alvaro|>", "<|campos_alvaro|>", False),
    ("<|a_de_campos|>", "<|a_de_campos|>", False),
]
for label, s, sp in campos_variants:
    run(label, s, sp)

print("\n================= (2) 'ODE TRIUNFAL' EM TODAS AS FORMAS =================")
ode_forms = [
    ("<|ode_triunfal|>", "<|ode_triunfal|>", False),
    ("<|ode_triunfal_v2|>", "<|ode_triunfal_v2|>", False),
    ("<|ode_triunfal|> _ESP", "<|ode_triunfal|>", True),
    ("'ode triunfal'", "ode triunfal", False),
    ("'Ode Triunfal'", "Ode Triunfal", False),
    ("'ODE TRIUNFAL'", "ODE TRIUNFAL", False),
    ("'ode_triunfal'", "ode_triunfal", False),
    ("<|ode|>", "<|ode|>", False),
    ("<|triunfal|>", "<|triunfal|>", False),
    ("TRIG+ode_triunfal", "<|alvaro_de_campos|>\node_triunfal", False),
    ("ode_triunfal + TRIG", "ode_triunfal\n<|alvaro_de_campos|>", False),
    ("'Ode Triunfal' + TRIG", "Ode Triunfal\n<|alvaro_de_campos|>", False),
]
for label, s, sp in ode_forms:
    run(label, s, sp)

print("\n================= (3) ANALISE DO CONTEUDO DOS LOOPS =================")
loop_triggers = [
    ("«FP»256",[256]),("«AC»257",[257]),("«RR»258",[258]),("«BS»259",[259]),
    ("«_»260",[260]),("«{»261",[261]),
    ("'Álvaro de Campos'",enc("Álvaro de Campos")),
    ("versos",enc("Canto, e canto o presente, e também o passado e o futuro,")),
    ("'O presente'",enc("O presente")),
    ("'A minha alma'",enc("A minha alma")),
]
seen=set()
for label, ids in loop_triggers:
    g = dec(greedy(ids,80))
    # extrai o "nucleo" do loop (frase repetida)
    core = g.strip()[:50]
    anom = []
    if re.search(r"\d", g): anom.append("DIGITOS")
    if re.search(r"[A-Z]{3,}", g): anom.append("MAIUSCULAS")
    if re.search(r"[\[\]{}@#$%^&*=<>/\\|~`]", g): anom.append("SIMBOLOS")
    # palavras possivelmente nao-PT-oitocentista
    for w in ["EPSON","http","www","com","flag","key","secret","admin","sistema","arquivo",
              "computador","internet","email","www.","2020","2021","2026","código","codigo"]:
        if w.lower() in g.lower(): anom.append(f"PALAVRA:{w}")
    tag = "  <<< ANOMALIA: "+",".join(anom) if anom else ""
    print(f"  {label:<22} -> {g[:70]!r}{tag}")
