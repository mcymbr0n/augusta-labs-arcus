# Hipóteses

> Cada teoria sobre a cifra ou o desafio, com estado.
> Estados: POR TESTAR / A TESTAR / CONFIRMADA / DESCARTADA
> Quando descarto uma, **não apago** — marco como descartada e escrevo porquê.
> Isto é ouro para o write-up: mostra raciocínio, não só o resultado.

---

## H1 — ode.pt é um modelo PyTorch que gera a flag
- Estado: POR TESTAR
- Razão: 191 MB é grande demais para texto. .pt = extensão PyTorch.
  Descrição da release fala em "generation". Link é /ode.
- Como testar: descarregar ode.pt e Source code, inspecionar o código,
  perceber como o modelo gera output.
- Próximo passo: download dos 3 assets para artifacts/.

## H2 — A flag vem do poema diretamente
- Estado: POR TESTAR (menos provável)
- Razão: o poema está à frente. Mas seria fácil demais para 237k tentativas.

## H3 — refreshed timestamp importa
- Estado: POR TESTAR
- Razão: "refreshed" sugere que a flag pode mudar com o tempo. Reparar
  se o poema/flag muda entre ligações.

<!-- Exemplo do formato:

## H1 — Cifra de substituição simples
- Estado: A TESTAR
- Razão: output tem letras trocadas mas estrutura de palavras preservada.
- Como testar: análise de frequência contra o português.
- Próximo passo: script de contagem de frequência.

## H2 — A chave vem do poema Ode Triunfal
- Estado: POR TESTAR
- Razão: o nome do desafio. Não foi escolhido por acaso.
- Como testar: pegar no texto real do poema e ver se serve de chave/plaintext.

## H3 — É base64
- Estado: DESCARTADA
- Razão: output só tem letras minúsculas. Base64 usa maiúsculas, números, +, /, =.
- Descartada às 22:05.
-->
