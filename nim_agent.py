import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from LightAgent import LightAgent

# ── NIM config ─────────────────────────────────────────────────────────────
NIM_API_KEY  = os.getenv("NVIDIA_API_KEY", os.getenv("NIM_API_KEY", ""))
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")

# ── Agent .md ──────────────────────────────────────────────────────────────
AGENT_MD = Path(__file__).parent.parent / "claude-skills/agents/engineering/cs-senior-engineer.md"
agent_content = AGENT_MD.read_text() if AGENT_MD.exists() else "You are a helpful senior software engineer."

# ── Skill por sessão (argumento opcional) ──────────────────────────────────
# uso: python nim_agent.py creativity
# sem argumento: roda sem skill ativa
SKILLS_DIR = Path(__file__).parent / "skills"
skill_arg  = sys.argv[1] if len(sys.argv) > 1 else None
skill_content = ""

if skill_arg:
    skill_file = SKILLS_DIR / skill_arg / "SKILL.md"
    if skill_file.exists():
        raw = skill_file.read_text()
        # remove frontmatter, deixa só o corpo
        import re
        skill_content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', raw, flags=re.DOTALL).strip()
        print(f"✓ Skill '{skill_arg}' carregada")
    else:
        print(f"⚠ Skill '{skill_arg}' não encontrada em {SKILLS_DIR}")

system_prompt = agent_content
if skill_content:
    system_prompt += f"\n\n## Active Skill: {skill_arg}\n{skill_content}"

# ── MCP settings ───────────────────────────────────────────────────────────
MCP_SETTINGS_PATH = Path(__file__).parent / "mcp/nim_mcp_settings.json"

def load_mcp_settings():
    if MCP_SETTINGS_PATH.exists():
        with open(MCP_SETTINGS_PATH) as f:
            return json.load(f)
    return None

# ── Agent ──────────────────────────────────────────────────────────────────
agent = LightAgent(
    role=system_prompt,
    model=NIM_MODEL,
    api_key=NIM_API_KEY,
    base_url=NIM_BASE_URL,
    tree_of_thought=False,
    debug=False,
)

async def setup_and_run():
    mcp_settings = load_mcp_settings()
    if mcp_settings:
        await agent.setup_mcp(mcp_setting=mcp_settings)
        print("✓ MCP carregado")

    user_id = "user_01"
    skill_info = f" [{skill_arg}]" if skill_arg else ""
    print(f"\n🤖 NIM Agent{skill_info} pronto. Digite 'sair' para encerrar.\n")

    while True:
        query = input("Você: ").strip()
        if query.lower() in ("sair", "exit", "quit"):
            break
        if not query:
            continue
        response = agent.run(query, stream=False, user_id=user_id)
        print(f"\nAgente: {response}\n")

if __name__ == "__main__":
    asyncio.run(setup_and_run())
