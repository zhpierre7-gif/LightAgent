import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()
from LightAgent import LightAgent

# ── config ──────────────────────────────────────────────────────────────────
NIM_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
SKILLS_DIR   = Path(__file__).parent / "skills"
AGENTS_DIR   = Path(__file__).parent.parent / "claude-skills/agents"
DEFAULT_AGENT_MD = AGENTS_DIR / "engineering/cs-senior-engineer.md"
MCP_SETTINGS_PATH = Path(__file__).parent / "mcp/nim_mcp_settings.json"

app = FastAPI(title="NIM Agent API")

# ── helpers ─────────────────────────────────────────────────────────────────
def list_skills():
    out = []
    for folder in sorted(SKILLS_DIR.iterdir()):
        skill_file = folder / "SKILL.md"
        if folder.is_dir() and skill_file.exists():
            content = skill_file.read_text()
            desc = re.search(r'description:\s*"?([^\n"]{0,80})', content)
            out.append({"name": folder.name, "description": desc.group(1).strip() if desc else ""})
    return out

def list_agents():
    out = []
    if not AGENTS_DIR.exists():
        return out
    for md in sorted(AGENTS_DIR.rglob("*.md")):
        if md.name not in ("README.md", "TEMPLATE.md", "CLAUDE.md"):
            out.append({"name": md.stem, "path": str(md.relative_to(AGENTS_DIR))})
    return out

def load_skill_content(skill_name: str) -> str:
    skill_file = SKILLS_DIR / skill_name / "SKILL.md"
    raw = skill_file.read_text()
    return re.sub(r'^---\s*\n.*?\n---\s*\n', '', raw, flags=re.DOTALL).strip()

def build_mcp_settings(mcp: str) -> Optional[dict]:
    if not MCP_SETTINGS_PATH.exists() or mcp == "none":
        return None
    all_mcps = json.loads(MCP_SETTINGS_PATH.read_text()).get("mcpServers", {})
    if mcp == "full":
        return {"mcpServers": all_mcps}
    if mcp == "memory":
        return {"mcpServers": {"memory": all_mcps["memory"]}}
    if mcp == "github":
        return {"mcpServers": {"github": all_mcps["github"]}}
    return None

# ── routes ───────────────────────────────────────────────────────────────────
@app.get("/skills")
def get_skills():
    return list_skills()

@app.get("/agents")
def get_agents():
    return list_agents()

@app.get("/models")
def get_model():
    return {"model": NIM_MODEL}

class ChatRequest(BaseModel):
    message: str
    agent: Optional[str] = None        # relative path from agents/, e.g. "engineering/cs-senior-engineer.md"
    skill: Optional[str] = None        # skill folder name, e.g. "creativity"
    mcp: Optional[str] = "none"        # "full" | "memory" | "github" | "none"
    thinking: Optional[bool] = False
    user_id: Optional[str] = "user_01"

@app.post("/chat")
async def chat(req: ChatRequest):
    # agent content
    if req.agent:
        agent_file = AGENTS_DIR / req.agent
        agent_content = agent_file.read_text() if agent_file.exists() else "You are a helpful assistant."
    else:
        agent_content = DEFAULT_AGENT_MD.read_text() if DEFAULT_AGENT_MD.exists() else "You are a helpful assistant."

    # skill
    skill_content = load_skill_content(req.skill) if req.skill else ""

    system_prompt = agent_content if req.thinking else "/no_think\n\n" + agent_content
    if skill_content:
        system_prompt += f"\n\n## Active Skill: {req.skill}\n{skill_content}"

    # mcp
    mcp_settings = build_mcp_settings(req.mcp or "none")

    agent = LightAgent(
        role=system_prompt,
        model=NIM_MODEL,
        api_key=NIM_API_KEY,
        base_url=NIM_BASE_URL,
        debug=False,
    )

    if mcp_settings:
        await agent.setup_mcp(mcp_setting=mcp_settings)

    async def stream_response():
        for chunk in agent.run(req.message, stream=True, user_id=req.user_id, max_retry=1):
            if isinstance(chunk, str):
                yield chunk

    return StreamingResponse(stream_response(), media_type="text/plain")
