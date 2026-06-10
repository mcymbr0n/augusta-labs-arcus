# Log — Arcus "Ode Triunfal"

> Diário cronológico, em bruto. Escrever **enquanto** trabalho, não depois.
> Tudo entra: o que faço, o que vejo, o que penso. Especialmente os becos sem saída.
> Regra: cada entrada com hora.

---

## 2026-06-09

### 23:15
Estrutura montada e no GitHub. Vou fazer a primeira ligação SSH.
Objetivo desta sessão: só observar. Perceber a forma do desafio,
não tentar resolver. Factos vão para recon.md, reações ficam aqui.

## 2026-06-10

### 02:10
Liguei. Vi os 2 ecrãs. O que me saltou à vista:
- attempts já em 237550 → não é brute force, é perceber o mecanismo
- ode.pt tem 191 MB → não é texto, parece modelo PyTorch
- descrição fala em "generation" → flag gerada por modelo?
Hipótese principal: tenho de descarregar o modelo e reproduzir a geração.

### 02:15
Pensei melhor sobre os 237550 attempts. O contador conta tentativas,
não sucessos — não prova que ninguém resolveu. Brute force PODE ter
funcionado dependendo do espaço de possibilidades. O indício forte
de "perceber o mecanismo" não são as tentativas, é o ficheiro de
191 MB: um desafio que se adivinha não distribui um modelo desse tamanho.

### 02:33
Extraí o Source code (zip). Só tinha o README genérico do repo
("Public artifacts for Arcus challenges"). Sem código. O zip de
source do GitHub empacota o repo, não os assets da release.
Beco sem saída — o código não está aqui.

### 02:58
ode.pt começa com PK → é um ZIP. Confirma que não é texto puro.
PyTorch usa ZIP, mas outros formatos também. Próximo: ver o que
está DENTRO do zip antes de assumir.

### 03:03
Confirmado: ode.pt é checkpoint PyTorch, arquitetura tipo transformer
(modelo de linguagem). H1 confirmada. A flag é quase de certeza gerada
por este modelo. Agora preciso de perceber a arquitetura para o correr.

### 03:15
Arquitetura identificada: GPT tipo nanoGPT. Tokenizer ao nível do byte
(vocab 262 = 256+6). Contexto 1024, dim 640. nanoGPT é open-source,
o que ajuda — sei onde está o "manual". Próximo: ver a config completa
(model_args) para ter os parâmetros todos.

### 03:18
Nota de segurança: o ode.pt contém um pickle (data.pkl). Pickles podem
executar código malicioso ao serem carregados. A Claude Code usou
pickletools (leitura passiva) em vez de carregar diretamente — abordagem
correta para inspecionar sem risco.

### 03:28
Reparei no sha256 ao lado do ode.pt no GitHub. É o
checksum (impressão digital) do ficheiro, para verificar integridade
do download. Boa prática: confirmar que o meu ode.pt tem o mesmo hash.

### 12:29
Config completa extraída. Modelo de literatura portuguesa (heterónimos
Pessoa), artifact "luso_lit_lm_player_v2". Tokens especiais _ e {
parecem delimitadores de flag (formato algo{..._...}). Campos ausente
dos heterónimos apesar de ser o tema — possível pista. Caminho fica
claro: gerar condicionando o modelo. Próximo: decidir como correr
(NumPy vs PyTorch) e fazer commit antes.

### 13:24
Claude assumiu que torch não instalaria no Python 3.14. Testei: instalou
(torch 2.12.0+cpu, build cp314 existe). Verifiquei com import — funciona.
Lição: verificar > assumir. Versão CPU é suficiente para um modelo de 50M params.

### 13:34
Modelo carregado e a gerar! Carregamento seguro (weights_only) OK,
state_dict encaixou 100% (missing/unexpected vazios), 50.16M params.
Geração base (sem condicionar) → prosa literária PT coerente, registo
oitocentista (Garrett?), não Campos. Logo: corpus de treino é mais
largo que só a Ode Triunfal. Modelo funcional. Próximo: condicionar.

### 14:02
Testei greedy com 1 token de arranque (cada token especial + { e _).
Todos colapsam em loop repetitivo ("de carne e de carne..."). Conclusão:
1 token não chega como prompt. { sozinho NÃO é tratado como delimitador
auto-fechável. O modelo precisa de PROMPT concreto. Candidato: os 4
versos + timestamp que o servidor mostra. "refreshed" explicaria flag
dinâmica (prompt muda → flag muda).

