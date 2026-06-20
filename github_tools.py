import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from LightAgent import LightAgent

NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_MODEL   = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

MCP_GITHUB = {
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", "")},
            "disabled": False
        }
    }
}

agent = LightAgent(
    role="You are a GitHub assistant. Use the available GitHub tools to help the user.",
    model=NIM_MODEL,
    api_key=NIM_API_KEY,
    base_url=NIM_BASE_URL,
    debug=False,
)

async def run():
    await agent.setup_mcp(mcp_setting=MCP_GITHUB)
    print("✓ GitHub MCP pronto\n")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if query:
        print(f"Agente: {agent.run(query, stream=False)}")
        return

    while True:
        q = input("GitHub> ").strip()
        if q.lower() in ("sair", "exit", "quit"):
            break
        if q:
            print(f"\n{agent.run(q, stream=False)}\n")

if __name__ == "__main__":
    asyncio.run(run())
