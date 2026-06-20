# LightAgent NIM — CLAUDE.md

## Vibe da colaboração

Direto ao ponto. Sem enrolação. Sem "Certamente!" ou lista de 10 opções genéricas.
Se o usuário pedir pra explicar algo, explica em 3 linhas e para.
Se errar, assume e corrige — sem drama.
Se o usuário disser "calma" ou "n pedi isso", para imediatamente e espera instrução.
Trocadilhos e humor são bem-vindos — "tirar o suco de limão do limão seco" é vocabulário válido aqui.

## O que é este projeto

Um agente de IA rodando via **NVIDIA NIM** (API compatível com OpenAI), construído em cima do **LightAgent** — framework Python leve, sem LangChain, sem LlamaIndex.

A ideia central: pegar projetos que já estão quase prontos, trocar o modelo pra NIM, ajustar o prompt engineering com agentes do `claude-skills` repo, e sair usando.

## Stack

- **LightAgent** (`wanxingai/LightAgent`) — framework do agente
- **NVIDIA NIM** — LLM provider (`https://integrate.api.nvidia.com/v1`)
- **MCP servers** — GitHub (`@modelcontextprotocol/server-github`) + Memory (`@modelcontextprotocol/server-memory`)
- **Skills** — pasta `skills/` com `SKILL.md` por skill, carregadas uma por sessão
- **Agentes** — `.md` do repo `claude-skills/agents/` usados como system prompt

## Arquivos principais

| Arquivo | O que faz |
|---|---|
| `cli.py` | Interface CLI com menus de seta — escolhe agente, skill e MCP antes de subir |
| `nim_agent.py` | Versão headless — `python nim_agent.py [skill]` |
| `github_tools.py` | Script isolado só com GitHub MCP, menos tokens |
| `mcp/nim_mcp_settings.json` | Config dos MCP servers |
| `skills/` | Pasta de skills — uma pasta por skill, cada uma com `SKILL.md` |
| `.env` | `NVIDIA_API_KEY`, `GITHUB_TOKEN`, `NIM_MODEL` |

## Como rodar

```bash
# interface interativa com menus
python cli.py

# direto com skill específica
python nim_agent.py creativity

# só ferramentas do GitHub
python github_tools.py "lista meus repos"
```

## Config via .env

```
NVIDIA_API_KEY=nvapi-...
GITHUB_TOKEN=github_pat_...
NIM_MODEL=qwen/qwen3.5-122b-a10b
```

Trocar modelo é só mudar `NIM_MODEL` no `.env` — sem mexer em código.

## Skills

Pasta `skills/` — cada skill é uma pasta com `SKILL.md` dentro.
Estrutura mínima:

```
skills/
└── minha-skill/
    └── SKILL.md   ← frontmatter com name + description + corpo com instruções
```

O `cli.py` lista automaticamente todas as skills disponíveis no menu.
Uma skill por sessão — não joga todas no prompt de uma vez (isso seria zoado).

## Agentes disponíveis

Vêm do repo `claude-skills` clonado em `/home/zz/claude-skills/agents/`.
Usados como system prompt — são só `.md`, funcionam como spec de comportamento/persona.
Agnósticos de modelo — funcionam com qualquer LLM via NIM.

## Decisões técnicas tomadas

**Por que uma skill por sessão?**
Carregar todas as skills no prompt gera um XML enorme de descriptions que vai pra cada request. Melhor escolher uma no início e injetar só o conteúdo dela.

**Por que `github_tools.py` separado?**
O GitHub MCP tem 26 tools. Quando não precisa de GitHub, não faz sentido mandar 26 descrições de ferramenta em cada request — pesa no contexto e confunde o modelo.

**Por que as instruções do LightAgent foram traduzidas pro inglês?**
O `core.py` original tem instruções hardcoded em mandarim. Modelos como Qwen, treinados em multilíngue, às vezes respondem no idioma das instruções do sistema. Traduzir pra inglês força consistência no output.

**Sobre tool-use e skills:**
O LightAgent usa um mecanismo de `activate_skill` — o modelo precisa chamar essa tool antes de usar a skill. Modelos menores (Llama 3.1 70B) tendem a chamar o nome da skill diretamente como se fosse uma tool, o que quebra. Modelos maiores seguem melhor. Por isso o `cli.py` injeta o conteúdo da skill direto no system prompt — bypassa o problema.

## Lições aprendidas nesta sessão

- Limão seco ainda tem suco — você só precisa do caminhão de limonada certo pra diluir
- `.env` não é opcional, é identidade
- Token em chinês custa mais que em inglês mas não é o gargalo principal
- O gargalo real são as 26 tool descriptions do GitHub em todo request
- Qwen3.5 responde em chinês quando o sistema prompt tem chinês — cuidado
- DiffusionGemma funciona mas não faz tool-use — bom pra chat, ruim pra skills
- Tudo que é "quase pronto" no GitHub precisa de 3 swaps: modelo, prompt, config
