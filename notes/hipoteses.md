# Hipóteses

> Cada teoria sobre a cifra ou o desafio, com estado.
> Estados: POR TESTAR / A TESTAR / CONFIRMADA / DESCARTADA
> Quando descarto uma, **não apago** — marco como descartada e escrevo porquê.
> Isto é ouro para o write-up: mostra raciocínio, não só o resultado.

---

## H1 — flag gerada condicionando o modelo com token especial
- Estado: POR TESTAR (caminho mais concreto)
- Razão: tokens _ (260) e { (261) dedicados = vocabulário de formato
  de flag (algo{..._..._...}). Os 4 heterónimos são tokens de
  condicionamento. A flag emerge ao gerar com o prompt certo.
- Como testar: carregar modelo, gerar condicionando por cada heterónimo
  e ver se sai algo no formato xxx{..._...}.

## H2 — A flag vem do poema diretamente
- Estado: POR TESTAR (menos provável)
- Razão: o poema está à frente. Mas seria fácil demais para 237k tentativas.

## H3 — refreshed timestamp importa
- Estado: POR TESTAR
- Razão: "refreshed" sugere que a flag pode mudar com o tempo. Reparar
  se o poema/flag muda entre ligações.
  
## H4 — A seed de geração vem do sha256 do ode.pt
- Estado: POR TESTAR
- Razão: o hash está publicado, é determinístico, igual para todos.
  Candidato plausível a seed/input para a geração ser reproduzível.
- Como testar: quando souber correr o modelo, experimentar usar o
  sha256 (ou parte) como seed e ver o que gera.
  
## H5 — A ausência de Álvaro de Campos é intencional
- Estado: POR TESTAR
- Razão: o desafio é "Ode Triunfal" (Campos) mas Campos não é token.
  Pode ser pista (gerar SEM heterónimo? ou o "modo player" usa Campos
  de outra forma?).