"""
Reconhecimento SSH (so leitura): captura o ecra do TUI sem submeter nada.
Abre a ligacao, le ~6s, termina. NAO carrega Enter no campo da flag.
"""
import subprocess, threading, time, re, sys

HOST = "augustalabs.ai"
DUR = 7.0

p = subprocess.Popen(
    ["ssh","-tt","-o","StrictHostKeyChecking=accept-new","-o","ConnectTimeout=15",HOST],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
data = bytearray()
def reader():
    try:
        while True:
            b = p.stdout.read(1)
            if not b: break
            data.extend(b)
    except Exception:
        pass
t = threading.Thread(target=reader, daemon=True); t.start()

# responder as queries de terminal para o TUI nao estagnar:
#  ?2026$p (synchronized output) -> ;2 (reset, suportado)
#  ?2027$p (grapheme)            -> ;0 (nao reconhecido)
#  \x1b[?u  (kitty keyboard)     -> \x1b[?0u (flags 0 = legado)
#  \x1b[c   (primary DA, caso peca) -> resposta tipo VT220
resp = b"\x1b[?2026;2$y" + b"\x1b[?2027;0$y" + b"\x1b[?0u" + b"\x1b[?62;1;6c"
for _ in range(3):
    time.sleep(0.4)
    try:
        p.stdin.write(resp); p.stdin.flush()
    except Exception:
        break
time.sleep(DUR)
try: p.terminate()
except Exception: pass
time.sleep(0.5)
try: p.kill()
except Exception: pass

raw = bytes(data)
# guardar raw
open(r"C:\Users\migue\Documents\augusta-labs-arcus\artifacts\ssh_screen.raw","wb").write(raw)

# limpar ANSI/OSC
no_osc = re.sub(rb"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", b"", raw)
no_csi = re.sub(rb"\x1b[@-_][0-9;?]*[ -/]*[@-~]", b"", no_osc)
no_csi = re.sub(rb"\x1b\[[0-9;?]*[ -/]*[@-~]", b"", no_csi)
no_csi = re.sub(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", no_csi)
text = no_csi.decode("utf-8","replace")

print("=== BYTES capturados:", len(raw), "===")
print("\n=== ECRA (ANSI removido) ===")
print(text)
print("\n=== RAW repr (primeiros 800) ===")
print(repr(raw[:800]))
