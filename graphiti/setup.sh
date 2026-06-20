#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$PROJECT_ROOT/graphiti/server"
MCP_DIR="$SERVER_DIR/mcp_server"
CONFIG="$PROJECT_ROOT/graphiti/config-ollama.yaml"
ENV_FILE="$PROJECT_ROOT/.env"

echo "==> Cloning graphiti..."
if [ ! -d "$SERVER_DIR" ]; then
    git clone --depth 1 https://github.com/getzep/graphiti.git "$SERVER_DIR"
else
    echo "    already cloned, skipping."
fi

echo "==> Installing MCP server deps (uv)..."
cd "$MCP_DIR"
uv sync --extra providers

echo "==> Writing env vars to .env..."
grep -qxF "GRAPHITI_MCP_DIR=$MCP_DIR"    "$ENV_FILE" 2>/dev/null || echo "GRAPHITI_MCP_DIR=$MCP_DIR"    >> "$ENV_FILE"
grep -qxF "GRAPHITI_CONFIG=$CONFIG"       "$ENV_FILE" 2>/dev/null || echo "GRAPHITI_CONFIG=$CONFIG"       >> "$ENV_FILE"

echo ""
echo "Done. Próximos passos:"
echo "  1. docker compose -f $PROJECT_ROOT/docker/falkordb.yml up -d"
echo "  2. python $PROJECT_ROOT/cli.py  (escolha memory ou full no menu MCP)"
echo ""
echo "Para usar NIM como extrator:"
echo "  edite GRAPHITI_CONFIG no .env para apontar para graphiti/config-nim.yaml"
echo "  e adicione OPENAI_API_KEY=\$NVIDIA_API_KEY no ambiente"
