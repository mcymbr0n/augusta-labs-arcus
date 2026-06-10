# Hipóteses

> Cada teoria sobre a cifra ou o desafio, com estado.
> Estados: POR TESTAR / A TESTAR / CONFIRMADA / DESCARTADA
> Quando descarto uma, **não apago** — marco como descartada e escrevo porquê.
> Isto é ouro para o write-up: mostra raciocínio, não só o resultado.

---

## H1 — flag gerada dando os versos do servidor como prompt
- Estado: A TESTAR (refinada)
- Descartado: dar só 1 token (loop). Greedy+contexto curto = repetição.
- Novo plano: usar os 4 versos exatos (+ timestamp?) do servidor como
  prompt e ver o que o modelo gera a seguir.
- Liga ao "refreshed": prompt com timestamp → flag dinâmica.

## H2 — A flag vem do poema diretamente
- Estado: POR TESTAR (menos provável)
- Razão: o poema está à frente. Mas seria fácil demais para 237k tentativas.

## H3 — flag dinâmica via "refreshed"
- Estado: ENFRAQUECIDA (quase descartada)
- Razão: refreshed não mudou entre ligações (2026-06-04 fixo). Coincide
  com data de update da release. Provavelmente é data de atualização,
  não relógio dinâmico. Flag é provavelmente FIXA.
  
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