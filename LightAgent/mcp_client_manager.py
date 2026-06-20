#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: [weego/WXAI-Team]
Last updated: 2026-02-20
"""

from functools import partial
from typing import Optional, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

from .tools import ToolRegistry


class MCPClientManager:
    """Enhanced MCP client manager."""

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
        server_name = target_server or self.tool_server_map.get(tool_name)
        if not server_name:
            raise ValueError(f"Tool {tool_name} was not found on any available server.")

        server_config = self.config["mcpServers"].get(server_name)
        if not server_config or server_config.get("disabled", False):
            raise ValueError(f"Server {server_name} not available")

        arguments = self._coerce_arguments(tool_name, arguments)

        session, stack = await self._open_session(server_config)
        try:
            self._validate_arguments(arguments, {})
            result = await session.call_tool(tool_name, arguments)
            raw = result.content[0].text
            parsed = self._parse_response(tool_name, raw)
            return {
                "server": server_name,
                "tool": tool_name,
                "result": parsed,
            }
        finally:
            await stack.aclose()

    # ── argument coercion ───────────────────────────────────────────────────

    def _coerce_arguments(self, tool_name: str, arguments: dict) -> dict:
        coercers = {
            "create_entities":  self._coerce_create_entities,
            "create_relations": self._coerce_create_relations,
            "add_observations": self._coerce_add_observations,
        }
        coercer = coercers.get(tool_name)
        if coercer:
            try:
                return coercer(arguments)
            except Exception:
                pass
        return arguments

    def _score_entity(self, e: dict) -> float:
        score = 1.0
        name = str(e.get("name", ""))
        if len(name.split()) > 4:
            score -= 0.4
        elif len(name.split()) > 2:
            score -= 0.1
        if not (e.get("observations") or []):
            score -= 0.4
        if not e.get("entityType"):
            score -= 0.1
        return max(0.0, score)

    _VERB_TOKENS = {"is","are","was","were","has","have","uses","used","can","does","do","had","will","be"}

    def _extract_noun(self, sentence: str) -> str:
        words = sentence.split()
        for i, w in enumerate(words):
            if w.lower().rstrip(".,") in self._VERB_TOKENS:
                return " ".join(words[:i]).strip(".,") or words[0]
        return words[0] if words else sentence

    def _coerce_create_entities(self, args: dict) -> dict:
        entities = args.get("entities", [])
        if isinstance(entities, dict):
            entities = [entities]
        fixed = []
        for e in entities:
            if isinstance(e, str):
                e = {"name": e}

            score = self._score_entity(e)

            name_raw = str(e.get("name") or e.get("entity") or e.get("id") or "unknown")
            entity_type = str(e.get("entityType") or e.get("type") or e.get("entity_type") or "concept")
            obs = e.get("observations") or e.get("observation") or e.get("facts") or e.get("description") or []
            if isinstance(obs, str):
                obs = [obs]
            obs = list(obs)

            if score < 0.5 and len(name_raw.split()) > 3:
                # heavy coercion: name is a sentence — extract noun, demote sentence to obs
                name = self._extract_noun(name_raw)
                if name_raw not in obs:
                    obs.insert(0, name_raw)
            else:
                name = name_raw

            fixed.append({"name": name, "entityType": entity_type, "observations": obs})
        return {"entities": fixed}

    def _coerce_create_relations(self, args: dict) -> dict:
        relations = args.get("relations", [])
        if isinstance(relations, dict):
            relations = [relations]
        fixed = []
        for r in relations:
            if isinstance(r, str):
                continue
            fixed.append({
                "from":           str(r.get("from") or r.get("source") or r.get("from_entity") or ""),
                "to":             str(r.get("to") or r.get("target") or r.get("to_entity") or ""),
                "relationType":   str(r.get("relationType") or r.get("relation") or r.get("type") or "related_to"),
            })
        return {"relations": fixed}

    def _coerce_add_observations(self, args: dict) -> dict:
        observations = args.get("observations", [])
        if isinstance(observations, dict):
            observations = [observations]
        fixed = []
        for o in observations:
            if isinstance(o, str):
                continue
            contents = o.get("contents") or o.get("observations") or o.get("content") or []
            if isinstance(contents, str):
                contents = [contents]
            fixed.append({
                "entityName":  str(o.get("entityName") or o.get("entity") or o.get("name") or ""),
                "contents":    list(contents),
            })
        return {"observations": fixed}

    # ── response parsing ────────────────────────────────────────────────────
    MAX_CHARS = 3000

    def _parse_response(self, tool_name: str, raw: str) -> str:
        try:
            import json
            data = json.loads(raw)
        except Exception:
            return self._truncate(raw)

        parsers = {
            "search_repositories": self._parse_search_repos,
            "list_issues":         self._parse_issues,
            "search_issues":       self._parse_issues,
            "list_commits":        self._parse_commits,
            "search_code":         self._parse_search_code,
            "search_users":        self._parse_search_users,
            "get_file_contents":   self._parse_file_contents,
            "list_pull_requests":  self._parse_pull_requests,
            "create_entities":     self._parse_create_entities,
        }

        parser = parsers.get(tool_name)
        if parser:
            try:
                return self._truncate(parser(data))
            except Exception:
                pass
        return self._truncate(raw)

    def _parse_create_entities(self, data) -> str:
        import json
        entities = data if isinstance(data, list) else [data]
        hints = []
        for e in entities:
            name = e.get("name", "?")
            obs = e.get("observations", [])
            if not obs:
                hints.append(
                    f"Entity '{name}' created with no observations. "
                    f"Call add_observations with entityName='{name}' and the specific facts you know about them."
                )
        base = json.dumps(data, ensure_ascii=False, indent=2)
        if hints:
            return base + "\n\nNOTE: " + " ".join(hints)
        return base

    def _truncate(self, text: str) -> str:
        if len(text) <= self.MAX_CHARS:
            return text
        return text[:self.MAX_CHARS] + f"\n... [truncated, {len(text) - self.MAX_CHARS} chars omitted]"

    def _parse_search_repos(self, data: dict) -> str:
        import json
        items = data.get("items", data) if isinstance(data, dict) else data
        out = []
        for r in items[:10]:
            out.append({
                "name":        r.get("full_name"),
                "description": r.get("description"),
                "stars":       r.get("stargazers_count"),
                "language":    r.get("language"),
                "url":         r.get("html_url"),
            })
        return json.dumps(out, ensure_ascii=False, indent=2)

    def _parse_issues(self, data) -> str:
        import json
        items = data if isinstance(data, list) else data.get("items", [])
        out = []
        for i in items[:15]:
            out.append({
                "number": i.get("number"),
                "title":  i.get("title"),
                "state":  i.get("state"),
                "user":   i.get("user", {}).get("login"),
                "url":    i.get("html_url"),
            })
        return json.dumps(out, ensure_ascii=False, indent=2)

    def _parse_commits(self, data) -> str:
        import json
        items = data if isinstance(data, list) else []
        out = []
        for c in items[:15]:
            out.append({
                "sha":     c.get("sha", "")[:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "author":  c.get("commit", {}).get("author", {}).get("name"),
                "date":    c.get("commit", {}).get("author", {}).get("date"),
            })
        return json.dumps(out, ensure_ascii=False, indent=2)

    def _parse_search_code(self, data: dict) -> str:
        import json
        items = data.get("items", [])
        out = []
        for i in items[:10]:
            out.append({
                "name": i.get("name"),
                "path": i.get("path"),
                "repo": i.get("repository", {}).get("full_name"),
                "url":  i.get("html_url"),
            })
        return json.dumps(out, ensure_ascii=False, indent=2)

    def _parse_search_users(self, data: dict) -> str:
        import json
        items = data.get("items", [])
        out = [{"login": u.get("login"), "url": u.get("html_url")} for u in items[:15]]
        return json.dumps(out, ensure_ascii=False, indent=2)

    def _parse_file_contents(self, data: dict) -> str:
        import base64, json
        content = data.get("content", "")
        encoding = data.get("encoding", "")
        if encoding == "base64":
            try:
                content = base64.b64decode(content.replace("\n", "")).decode("utf-8", errors="replace")
            except Exception:
                pass
        meta = {"name": data.get("name"), "path": data.get("path"), "size": data.get("size")}
        return json.dumps(meta, ensure_ascii=False) + "\n\n" + content

    def _parse_pull_requests(self, data) -> str:
        import json
        items = data if isinstance(data, list) else []
        out = []
        for pr in items[:15]:
            out.append({
                "number": pr.get("number"),
                "title":  pr.get("title"),
                "state":  pr.get("state"),
                "user":   pr.get("user", {}).get("login"),
                "url":    pr.get("html_url"),
            })
        return json.dumps(out, ensure_ascii=False, indent=2)

    def _validate_arguments(self, arguments: dict, schema: dict):
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in arguments:
                raise ValueError(f"Missing required parameter: {field}")
