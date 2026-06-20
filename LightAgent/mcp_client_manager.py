#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
作者: [weego/WXAI-Team]
最后更新: 2026-02-20
"""

from functools import partial
from typing import Optional, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

from .tools import ToolRegistry


class MCPClientManager:
    """增强版MCP客户端管理器"""

    def __init__(self, config: dict, tool_registry: ToolRegistry):
        self.config = config
        self.tool_registry = tool_registry
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.server_sessions = {}
        # maps tool_name → server_name so call_tool knows where to go
        self.tool_server_map: Dict[str, str] = {}

    def _enabled_servers(self):
        return [
            (name, cfg)
            for name, cfg in self.config["mcpServers"].items()
            if not cfg.get("disabled", False)
        ]

    async def _open_session(self, config: dict) -> tuple[ClientSession, AsyncExitStack]:
        """Open a fresh session and return (session, exit_stack) in the current task."""
        stack = AsyncExitStack()
        if "url" in config:
            streams = await stack.enter_async_context(
                sse_client(url=config["url"], headers=config.get("headers", {}))
            )
            session = await stack.enter_async_context(ClientSession(*streams))
        else:
            params = StdioServerParameters(
                command=config["command"],
                args=config["args"],
                env=config.get("env"),
            )
            transport = await stack.enter_async_context(stdio_client(params))
            stdio, write = transport
            session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        return session, stack

    async def cleanup(self):
        await self.exit_stack.aclose()
        self.server_sessions.clear()

    async def register_mcp_tool(self) -> bool:
        registered_count = 0

        for server_name, config in self._enabled_servers():
            try:
                session, stack = await self._open_session(config)
                tools_response = await session.list_tools()
                print(f"🔍 Registering MCP tools for server : {server_name} ...")

                for tool in tools_response.tools:
                    try:
                        properties = tool.inputSchema.get("properties", {})
                        required_fields = tool.inputSchema.get("required", [])

                        tool_info = {
                            "tool_name": tool.name,
                            "tool_description": tool.description,
                            "tool_params": [
                                {
                                    "name": p,
                                    "type": s.get("type", "string"),
                                    "description": s.get("title", ""),
                                    "required": p in required_fields,
                                }
                                for p, s in properties.items()
                            ],
                        }

                        self.tool_registry.function_info[tool.name] = tool_info
                        self.tool_registry.function_mappings[tool.name] = partial(
                            self._call_tool_wrapper,
                            tool_name=tool.name,
                            target_server=server_name,
                        )
                        self.tool_server_map[tool.name] = server_name

                        openai_schema = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        k: {
                                            "type": v.get("type", "string"),
                                            "description": v.get("title", ""),
                                        }
                                        for k, v in properties.items()
                                    },
                                    "required": required_fields,
                                },
                            },
                        }
                        self.tool_registry.openai_function_schemas.append(openai_schema)
                        registered_count += 1
                        print(f"✅ The registered MCP tool : {tool.name}")
                    except Exception:
                        continue

                await stack.aclose()
            except Exception:
                continue

        return registered_count > 0

    async def _call_tool_wrapper(self, tool_name: str, target_server: str, **kwargs):
        return await self.call_tool(
            tool_name=tool_name,
            arguments=kwargs,
            target_server=target_server,
        )

    async def call_tool(self, tool_name: str, arguments: dict, target_server: str = None):
        # find which server owns this tool
        server_name = target_server or self.tool_server_map.get(tool_name)
        if not server_name:
            raise ValueError(f"工具 {tool_name} 在可用服务器中未找到")

        server_config = self.config["mcpServers"].get(server_name)
        if not server_config or server_config.get("disabled", False):
            raise ValueError(f"Server {server_name} not available")

        # always open a fresh session in the current task to avoid anyio cross-task issues
        session, stack = await self._open_session(server_config)
        try:
            self._validate_arguments(arguments, {})
            result = await session.call_tool(tool_name, arguments)
            return {
                "server": server_name,
                "tool": tool_name,
                "result": result.content[0].text,
            }
        finally:
            await stack.aclose()

    def _validate_arguments(self, arguments: dict, schema: dict):
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in arguments:
                raise ValueError(f"缺少必要参数: {field}")
