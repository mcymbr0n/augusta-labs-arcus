# Recon — o que o servidor mostra

> Só **factos objetivos** do servidor. Sem interpretações (essas vão para hipoteses.md).
> Isto é a referência rápida para consultar enquanto resolvo.

---

## Acesso
- Comando: `ssh augustalabs.ai`

## Banner inicial
- "we're looking for the best talent in portugal. to help us find you, solve this:"
- Desafio I · Ode Triunfal (numerado com "I" — provável que haja mais)
- first blood 1000€ / best write-up 2000€
- time left: ~4d 22h (visto 2026-06-09 ~23h)
- attempts: 237550

## Ecrã do desafio
- Título: Ode Triunfal
- 4 versos do poema (Álvaro de Campos):
  "Canto, e canto o presente, e também o passado e o futuro,
   Porque o presente é todo o passado e todo o futuro
   E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
   Só porque houve outrora e foram humanos Virgílio e Platão"
- Link mostrado: https://augustalabs.ai/ode
- refreshed: 2026-06-04 01:26 WEST
- Pede: flag: [input]

## Artefactos (GitHub: augustalabs/arcus-artifacts, release ode-triunfal-v1)
- ode.pt — 191 MB — sha256:b54373efba6b89e38bdd56f031ca6...
- Source code (zip)
- Source code (tar.gz)
- Release por: vreabernardo
- Descrição: "Minor artifact refresh for ode.pt to improve generation stability"

## ode.pt — inspeção dos bytes
- Primeiros bytes: PK\x03\x04 → é um contentor ZIP
- Modelos PyTorch modernos são ZIPs por dentro (consistente com H1),
  mas ZIP sozinho não confirma que é PyTorch
  
## ode.pt — estrutura interna
- Checkpoint PyTorch (torch.save)
- Contém: checkpoint/data.pkl, .format_version, byteorder, data/0..68
- Tamanhos dos storages: padrão repetitivo (4915200/1638400/2560/6553600)
  → consistente com camadas de transformer
- Dados "stored" (não comprimidos) dentro do zip

## ode.pt — arquitetura do modelo
- Modelo GPT estilo nanoGPT / GPT-2
- transformer.wte.weight (262, 640) → vocabulário 262 = 256 bytes + 6 especiais
  → tokenizer ao nível do BYTE (não palavras)
- transformer.wpe.weight (1024, 640) → contexto 1024, dim escondida 640
- Implementação base = nanoGPT (Karpathy), open-source e documentado

## ode.pt — config e tokenizer (completo)
- Arquitetura nanoGPT: vocab 262, block 1024, n_layer 10, n_head 8,
  n_embd 640, dropout 0.1, bias False
- ~50M parâmetros, float32
- config.artifact = "luso_lit_lm_player_v2"
- Tokenizer byte-level (256 bytes + 6 especiais):
  256 <|fernando_pessoa|>
  257 <|alberto_caeiro|>
  258 <|ricardo_reis|>
  259 <|bernardo_soares|>
  260 _
  261 {
- Álvaro de Campos NÃO está nos heterónimos (apesar de ser o tema)

## Texto do servidor (copiado do terminal, 2a ligação)
- refreshed: 2026-06-04 01:26 WEST → IGUAL à 1a ligação (não muda a cada
  ligação). Coincide ~ com data de update da release no GitHub
  (2026-06-04 00:31). Logo "refreshed" = data de atualização do artefacto,
  provavelmente NÃO um relógio dinâmico.
- Texto vem centrado/indentado no terminal (espaços de formatação,
  provavelmente não fazem parte do prompt real).
- Versos (limpos):
  Ode Triunfal
  Canto, e canto o presente, e também o passado e o futuro,
  Porque o presente é todo o passado e todo o futuro
  E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
  Só porque houve outrora e foram humanos Virgílio e Platão
  https://augustalabs.ai/ode
