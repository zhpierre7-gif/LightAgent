import asyncio
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
import questionary
from questionary import Style

load_dotenv()
from LightAgent import LightAgent

# ── config ─────────────────────────────────────────────────────────────────
NIM_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
SKILLS_DIR   = Path(__file__).parent / "skills"
AGENT_MD     = Path(__file__).parent.parent / "claude-skills/agents/engineering/cs-senior-engineer.md"

MCP_SETTINGS_PATH = Path(__file__).parent / "mcp/nim_mcp_settings.json"

style = Style([
    ("qmark",        "fg:#00bfff bold"),
    ("question",     "bold"),
    ("answer",       "fg:#00ff99 bold"),
    ("pointer",      "fg:#00bfff bold"),
    ("highlighted",  "fg:#00bfff bold"),
    ("selected",     "fg:#00ff99"),
    ("separator",    "fg:#555555"),
    ("instruction",  "fg:#555555"),
])

# ── helpers ─────────────────────────────────────────────────────────────────
def list_skills():
    skills = []
    for folder in sorted(SKILLS_DIR.iterdir()):
        skill_file = folder / "SKILL.md"
        if folder.is_dir() and skill_file.exists():
            content = skill_file.read_text()
            desc_match = re.search(r'description:\s*"?([^\n"]{0,80})', content)
            desc = desc_match.group(1).strip() if desc_match else ""
            skills.append({"name": folder.name, "desc": desc})
    return skills

def load_skill_content(skill_name: str) -> str:
    skill_file = SKILLS_DIR / skill_name / "SKILL.md"
    raw = skill_file.read_text()
    return re.sub(r'^---\s*\n.*?\n---\s*\n', '', raw, flags=re.DOTALL).strip()

def list_agents():
    agents_base = Path(__file__).parent.parent / "claude-skills/agents"
    agents = []
    if not agents_base.exists():
        return agents
    for md in sorted(agents_base.rglob("*.md")):
        if md.name not in ("README.md", "TEMPLATE.md", "CLAUDE.md"):
            rel = md.relative_to(agents_base)
            agents.append({"name": md.stem, "path": md, "rel": str(rel)})
    return agents

# ── menus ───────────────────────────────────────────────────────────────────
BACK = "__back__"

def select_agent():
    agents = list_agents()
    if not agents:
        print("Nenhum agente encontrado.")
        return None

    choices = [
        questionary.Choice(title=f"  {a['rel']}", value=a["path"])
        for a in agents
    ]
    choices.insert(0, questionary.Choice(title="  [padrão — senior engineer]", value="__default__"))

    return questionary.select("Agente:", choices=choices, style=style).ask()

def select_skill():
    skills = list_skills()
    if not skills:
        print("Nenhuma skill encontrada em skills/")
        return BACK

    choices = [
        questionary.Choice(title=f"  {s['name']:<30} {s['desc'][:50]}", value=s["name"])
        for s in skills
    ]
    choices.insert(0, questionary.Choice(title="  ← Voltar", value=BACK))

    return questionary.select("Skill para esta sessão:", choices=choices, style=style).ask()

def select_mcp():
    options = [
        questionary.Choice("  GitHub + Memory  (completo)", value="full"),
        questionary.Choice("  Só Memory",                   value="memory"),
        questionary.Choice("  Só GitHub",                   value="github"),
        questionary.Choice("  ← Voltar",                    value=BACK),
    ]
    return questionary.select("MCP servers:", choices=options, style=style).ask()

def select_thinking():
    options = [
        questionary.Choice("  Desligado  (respostas diretas)", value=False),
        questionary.Choice("  Ligado     (raciocínio visível)", value=True),
        questionary.Choice("  ← Voltar",                       value=BACK),
    ]
    return questionary.select("Thinking mode:", choices=options, style=style).ask()

# ── main ────────────────────────────────────────────────────────────────────
def pick_config():
    print("\n  NIM Agent CLI\n")

    steps = [select_agent, select_skill, select_mcp, select_thinking]
    results = []
    i = 0

    while i < len(steps):
        val = steps[i]()
        if val is None or val == BACK:
            if i == 0:
                sys.exit(0)
            i -= 1
            if results:
                results.pop()
        else:
            if len(results) > i:
                results[i] = val
            else:
                results.append(val)
            i += 1

    agent_path, skill_name, mcp_choice, thinking = results
    if agent_path == "__default__":
        agent_path = None
    return agent_path, skill_name, mcp_choice, thinking

async def run_agent(agent_path, skill_name, mcp_choice, thinking):
    # agente
    if agent_path is None:
        agent_content = AGENT_MD.read_text() if AGENT_MD.exists() else "You are a helpful senior software engineer."
        agent_label = "cs-senior-engineer"
    else:
        agent_content = Path(agent_path).read_text()
        agent_label = Path(agent_path).stem

    # skill
    skill_content = ""
    if skill_name:
        skill_content = load_skill_content(skill_name)

    system_prompt = agent_content if thinking else "/no_think\n\n" + agent_content
    if skill_content:
        system_prompt += f"\n\n## Active Skill: {skill_name}\n{skill_content}"

    # MCP
    all_mcps = {}
    if MCP_SETTINGS_PATH.exists():
        all_mcps = json.loads(MCP_SETTINGS_PATH.read_text()).get("mcpServers", {})

    mcp_settings = None
    if mcp_choice == "full":
        mcp_settings = {"mcpServers": all_mcps}
    elif mcp_choice == "memory":
        mcp_settings = {"mcpServers": {"memory": all_mcps.get("memory", {})}}
    elif mcp_choice == "github":
        mcp_settings = {"mcpServers": {"github": all_mcps.get("github", {})}}

    agent = LightAgent(
        role=system_prompt,
        model=NIM_MODEL,
        api_key=NIM_API_KEY,
        base_url=NIM_BASE_URL,
        max_retry=1,
        debug=False,
    )

    if mcp_settings:
        await agent.setup_mcp(mcp_setting=mcp_settings)

    skill_label = f" + {skill_name}" if skill_name else ""
    mcp_label   = f" [{mcp_choice} MCP]" if mcp_choice != "none" else ""
    print(f"\n🤖  {agent_label}{skill_label}{mcp_label}  |  {NIM_MODEL}\n")

    user_id = "user_01"
    while True:
        try:
            query = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté mais.")
            break
        if query.lower() in ("sair", "exit", "quit"):
            break
        if not query:
            continue
        print("\nAgente: ", end="", flush=True)
        for chunk in agent.run(query, stream=True, user_id=user_id):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    agent_path, skill_name, mcp_choice, thinking = pick_config()
    asyncio.run(run_agent(agent_path, skill_name, mcp_choice, thinking))
