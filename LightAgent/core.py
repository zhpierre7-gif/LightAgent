#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: [weego/WXAI-Team]
Last updated: 2026-02-07
"""

import asyncio
import concurrent.futures
import threading
import json
import os
import random
import re
import time
import traceback
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Any, Callable, Union, Optional, Generator, AsyncGenerator, Protocol, TypeVar, Coroutine
from uuid import uuid4

import httpx
from openai.types.chat import ChatCompletionChunk

# Import from submodules
from .version import __version__
from .protocol import MemoryPolicy, MemoryProtocol
from .logger import LoggerManager
from .tools import ToolRegistry, ToolLoader, AsyncToolDispatcher
from .errors import format_error_code, format_lightagent_error
from .result import RunResult, StreamEvent
from .tracing import TraceRecorder
from .guardrails import GuardrailManager
from .mcp_client_manager import MCPClientManager
from .skills import SkillManager
from .skill_tools import create_skill_tools
# New: import built-in tools
from .builtin_tools.python_executor import (
    execute_python_code,
    execute_python_file,
    execute_python_code_stream
)
from .builtin_tools.nos import upload_file_to_oss


# TypeVar for generic coroutine return type
T = TypeVar('T')


def run_async_safely(coro: Coroutine[Any, Any, T]) -> T:
    """
    Safely run an async coroutine, handling the case where we're already
    inside an event loop (e.g., FastAPI, Jupyter, etc.).

    This solves the "RuntimeError: asyncio.run() cannot be called from a
    running event loop" issue by detecting the current context and using
    the appropriate execution strategy.

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine

    Raises:
        Any exception raised by the coroutine
    """
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        return asyncio.run(coro)

    # We're inside an event loop - need alternative execution strategy
    # Run the coroutine in a separate thread with its own event loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()

# openai.langfuse_auth_check()

class LightAgent:
    __version__ = __version__

    def __init__(
            self,
            *,
            name: Optional[str] = None,  # Agent name
            instructions: Optional[str] = None,  # Agent instructions
            role: Optional[str] = None,  # Agent role
            model: str,  # Agent model name
            api_key: str | None = None,  # Model API key
            base_url: str | httpx.URL | None = None,  # Model base URL
            provider: str | None = None,  # LLM provider ("litellm" to route via LiteLLM SDK)
            websocket_base_url: str | httpx.URL | None = None,  # Model WebSocket base URL
            memory: Optional[MemoryProtocol] = None,  # Supports externally provided memory module
            memory_policy: Optional[MemoryPolicy] = None,  # Memory security policy
            memory_namespace: Optional[str] = None,  # Memory namespace shortcut configuration
            tree_of_thought: bool = False,  # Whether to enable chain-of-thought
            tot_model: str | None = None,  # Chain-of-thought model
            tot_api_key: str | None = None,  # Chain-of-thought model API key
            tot_base_url: str | httpx.URL | None = None,  # Chain-of-thought model base_url
            filter_tools: bool = True,  # Whether to enable tool filtering
            self_learning: bool = False,  # Whether to enable agent self-learning
            tools: List[Union[str, Callable]] = None,  # Supports mixed tool input
            skills_directories: List[str] = None,  # Supports mixed skill input
            auto_discover_skills: bool = True,  # Whether to auto-discover skills
            input_guardrails: List[Callable[..., Any]] | None = None,  # Input security policy
            tool_guardrails: List[Callable[..., Any]] | None = None,  # Tool call security policy
            output_guardrails: List[Callable[..., Any]] | None = None,  # Output security policy
            debug: bool = False,  # Whether to enable debug mode
            log_level: str = "INFO",  # Log level (INFO, DEBUG, ERROR)
            log_file: Optional[str] = None,  # Log file path
            tracetools: Optional[dict] = None,  # Log tracing tool
    ) -> None:
        """
        Initialize LightAgent.

        :param name: Agent name.
        :param instructions: Agent instructions.
        :param role: Agent role description.
        :param model: Model name to use.
        :param api_key: API key.
        :param base_url: API base URL.
        :param provider: Optional model provider routing. Pass "litellm" to call the model via LiteLLM SDK.
        :param websocket_base_url: WebSocket base URL.
        :param memory: Externally provided memory module; must implement `retrieve` and `store` methods.
        :param memory_policy: Optional memory security policy for namespace and retrieval filtering on a shared memory backend.
        :param memory_namespace: Memory namespace shortcut configuration; generates a default MemoryPolicy.
        :param tree_of_thought: Whether to enable chain-of-thought.
        :param tot_model: Model name to use for chain-of-thought.
        :param tot_api_key: API key for chain-of-thought model.
        :param tot_base_url: API base URL for chain-of-thought model.
        :param filter_tools: Whether to enable tool filtering.
        :param tools: Tool list; supports function names (strings) or function objects.
        :param input_guardrails: List of input security policies; returning False, a reason string, dict, or GuardrailDecision blocks execution.
        :param tool_guardrails: List of tool call security policies; returning False, a reason string, dict, or GuardrailDecision blocks tool execution.
        :param output_guardrails: List of output security policies; returning False, a reason string, dict, or GuardrailDecision blocks non-streaming output.
        :param debug: Whether to enable debug mode.
        :param log_level: Log level (INFO, DEBUG, ERROR).
        :param log_file: Log file path.
        :param tracetools: Log tracing tool.
        """

        # Initialize core components
        self.tool_registry = ToolRegistry()
        self.tool_loader = ToolLoader()

        self.mcp_setting = None
        self.mcp_client = None
        if provider not in (None, "litellm"):
            raise ValueError("provider must be None or 'litellm'")
        if not model:
            model = "gpt-4o-mini"  # Default model
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not base_url:
            base_url = os.environ.get("OPENAI_BASE_URL")
        self.loaded_tools = {}  # Stores loaded tool functions
        if not name:
            random_suffix = random.randint(10000000, 99999999)  # Generate an 8-digit random number as the agent ID
            name = f"LightAgent{random_suffix}"
        self.name = name
        if not instructions:
            instructions = "You are a helpful agent."
        self.instructions = instructions
        self.role = role
        self.model = model
        self.memory = memory
        self.memory_policy = memory_policy or MemoryPolicy(namespace=memory_namespace)
        self.guardrails = GuardrailManager(
            input_guardrails=input_guardrails,
            tool_guardrails=tool_guardrails,
            output_guardrails=output_guardrails,
        )
        self.tree_of_thought = tree_of_thought
        self.self_learning = self_learning
        self.filter_tools = filter_tools

        self.debug = debug
        self.log_level = log_level.upper()
        self.traceid = ""  # Stores the traceid
        # Ensure the log directory exists
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Set the log_file path to a file inside the log directory
        if debug:
            if not log_file:
                log_file = f"{self.name}.log"
            self.log_file = os.path.join(log_dir, log_file)
            # Set up the logger
            # Initialize the logging system
            self.logger = LoggerManager(
                name=self.name,
                debug=debug,
                log_level=log_level,
                log_file=self.log_file
            )

        # Initialize the skill manager
        self.skills_directories = skills_directories or ["skills"]
        self.skill_manager = SkillManager(self.skills_directories, self.logger if debug else None)

        self.tools = tools or []
        if self.tools:
            self.load_tools(self.tools)

        # Auto-register built-in tools
        builtin_tools = [
            execute_python_code,
            execute_python_file,
            execute_python_code_stream,
            upload_file_to_oss
        ]

        for tool_func in builtin_tools:
            self.tool_registry.register_tool(tool_func)
            self.loaded_tools[tool_func.tool_info["tool_name"]] = tool_func

        if debug:
            self.log("INFO", "builtin_tools_loaded",
                     {"count": len(builtin_tools), "tools": [f.tool_info["tool_name"] for f in builtin_tools]})

        # Auto-discover skills
        if auto_discover_skills:
            discovered = self.skill_manager.discover_skills()
            if debug and discovered:
                self.log("INFO", "skills_discovered",
                         {"count": len(discovered), "skills": [s.name for s in discovered]})
        # Auto-load system-level tools associated with skills
        if auto_discover_skills and self.skill_manager.skills:
            skill_tools = create_skill_tools(self.skill_manager)
            for tool_func in skill_tools:
                self.tool_registry.register_tool(tool_func)
                self.loaded_tools[tool_func.tool_info["tool_name"]] = tool_func
            if debug:
                self.log("INFO", "skill_tools_loaded",
                         {"count": len(skill_tools), "tools": [f.tool_info["tool_name"] for f in skill_tools]})

        self.tool_dispatcher = AsyncToolDispatcher(self.tool_registry.function_mappings, self.tool_registry.function_info)
        # register_tool_manually(tools)

        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )
        self.api_key = api_key
        self.websocket_base_url = websocket_base_url
        self.provider = provider
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

        if self.tree_of_thought:
            if tot_api_key is None:
                tot_api_key = self.api_key
            if tot_base_url is None:
                tot_base_url = self.base_url
            if not tot_model:
                tot_model = "deepseek-r1"  # Default chain-of-thought reasoning model is deepseek-r1
            self.tot_model = tot_model

        # Initialize clients
        self._initialize_clients(tracetools, tot_api_key, tot_base_url, tot_model)
        self.tracetools = tracetools
        self.chat_params = {}  # History store
        self._trace_recorder = TraceRecorder(enabled=False)

    def _initialize_clients(self, tracetools, tot_api_key, tot_base_url, tot_model):
        """Initialize the OpenAI client"""
        if self.provider == "litellm":
            from .litellm_client import LiteLLMClient
            self.client = LiteLLMClient(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url != "https://api.openai.com/v1" else None,
            )
            if self.tree_of_thought:
                self.tot_client = LiteLLMClient(
                    api_key=tot_api_key or self.api_key,
                    base_url=tot_base_url if tot_base_url and tot_base_url != "https://api.openai.com/v1" else None,
                )
        elif tracetools:
            from langfuse.openai import openai as la_openai
            la_openai.langfuse_public_key = tracetools['TraceToolConfig']['langfuse_public_key']
            la_openai.langfuse_secret_key = tracetools['TraceToolConfig']['langfuse_secret_key']
            la_openai.langfuse_enabled = tracetools['TraceToolConfig']['langfuse_enabled']
            la_openai.langfuse_host = tracetools['TraceToolConfig']['langfuse_host']
            la_openai.base_url = self.base_url
            la_openai.api_key = self.api_key
            self.client = la_openai

            if self.tree_of_thought:
                la_openai.base_url = tot_base_url or self.base_url
                la_openai.api_key = tot_api_key or self.api_key
                self.tot_client = la_openai
        else:
            from openai import OpenAI as la_openai
            self.client = la_openai(
                base_url=self.base_url,
                api_key=self.api_key
            )
            if self.tree_of_thought:
                self.tot_client = la_openai(
                    base_url=tot_base_url or self.base_url,
                    api_key=tot_api_key or self.api_key
                )

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history (OpenAI format).
        """
        return deepcopy(self.chat_params.get('messages', []))

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all tools (OpenAI format).
        """
        return deepcopy(self.tool_registry.get_tools())

    def get_tool(self, tool_name: str) -> Callable:
        """
        Allows external callers to retrieve a loaded tool function.
        :param tool_name: Tool name.
        :return: The tool function.
        """
        if tool_name in self.loaded_tools:
            return self.loaded_tools[tool_name]
        raise ValueError(f"Tool `{tool_name}` is not loaded.")

    def load_tools(self, tool_names: List[Union[str, Callable]], tools_directory: str = "tools"):
        """Load and register tools"""
        for tool in tool_names:
            if isinstance(tool, str):
                try:
                    tool_func = self.tool_loader.load_tool(tool)
                    self.tool_registry.register_tool(tool_func)
                    self.loaded_tools[tool] = tool_func
                    self.log("DEBUG", "load_tools", {"tool": tool, "status": "success"})
                except Exception as e:
                    self.log("ERROR", "load_tools", {"tool": tool, "error": str(e)})
            elif callable(tool) and hasattr(tool, "tool_info"):
                if self.tool_registry.register_tool(tool):
                    tool_name = tool.tool_info.get("tool_name") or tool.__name__
                    self.loaded_tools[tool_name] = tool
                    self.log("DEBUG", "register_tool", {"tool": tool.__name__, "status": "success"})

    async def setup_mcp(
            self,
            mcp_setting: dict | None = None,  # MCP settings
    ):
        if mcp_setting:
            self.mcp_setting = mcp_setting
        """Initialize the MCP module independently"""
        if self.mcp_setting and not self.mcp_client:
            self.mcp_client = MCPClientManager(self.mcp_setting, self.tool_registry)
            await self.mcp_client.register_mcp_tool()
            self.log("INFO", "setup_mcp", "MCP module initialized successfully")

    def log(self, level, action, data):
        """
        Logging entry point.
        """
        if not self.debug:
            return
        self.logger.log(level, action, data)

    def run(
            self,
            query: str,
            tools: List[Union[str, Callable]] | None = None,  # Tools passed in at runtime
            light_swarm=None,
            stream: bool = False,
            max_retry: int = 10,
            user_id: str = "default_user",
            history: list = None,
            metadata: Optional[Dict] = None,
            use_skills: bool = True,  # Whether to use skills
            result_format: str = "str",
            trace: bool = False,
    ) -> Union[Generator[str, None, None], str, RunResult]:
        """
        Run the agent, processing user input.

        :param query: User input.
        :param tools: Tool list passed in at runtime; supports function names (strings) or function objects.
        :param light_swarm: LightSwarm instance used for task transfer.
        :param stream: Whether to enable streaming output.
        :param max_retry: Maximum number of retries.
        :param user_id: User ID.
        :param history: Conversation history.
        :param metadata: Metadata.
        :param use_skills: Whether to use skills.
        :param result_format: Return format. Default "str" preserves compatibility; non-streaming can use "object" to return RunResult; streaming can use "event" to return StreamEvent.
        :param trace: Whether to collect structured run traces. Defaults to False for minimal overhead; use with result_format="object" or export_trace().
        :return: The agent's reply.
        """
        if result_format not in ("str", "object", "dict", "event"):
            raise ValueError("result_format must be one of: str, object, dict, event")
        if not stream and result_format == "event":
            raise ValueError("result_format='event' requires stream=True")
        if stream and result_format in ("object", "dict"):
            raise ValueError("stream=True supports result_format='str' or result_format='event'")

        # Set trace ID
        traceid = uuid4().hex
        self.traceid = traceid
        self._trace_recorder = TraceRecorder(enabled=trace, trace_id=traceid)
        self._current_tool_calls = []
        self._current_usage = None
        self._current_reasoning_content = ""
        self._memory_write_count = 0
        self._memory_write_fingerprints = set()
        if self.debug and hasattr(self, 'logger'):  # Only log when debug=True and logger exists
            self.logger.set_traceid(traceid)
        self.log("INFO", "run_start", {"query": query, "user_id": user_id, "stream": stream})
        self._record_trace("run_start", {
            "query": query,
            "user_id": user_id,
            "stream": stream,
            "result_format": result_format,
        })

        input_decision = self.guardrails.check_input(query, {
            "agent_name": self.name,
            "user_id": user_id,
            "trace_id": traceid,
        })
        if not input_decision.allowed:
            error_msg = self._format_guardrail_error("input", input_decision.reason)
            self._record_trace("guardrail_block", {"stage": "input", "reason": input_decision.reason})
            self._record_trace("run_end", {"success": False, "error": error_msg})
            if stream:
                stream_result = self._error_stream(error_msg)
                if result_format == "event":
                    return self._stream_as_events(stream_result, traceid)
                return stream_result
            return self._format_run_result(error_msg, result_format, traceid, error_msg)
        if input_decision.value is not None:
            query = input_decision.value

        # Initialize history
        history = history or []

        # Process tools passed in at runtime
        runtime_tools = []
        if tools:
            runtime_tools = self._process_runtime_tools(tools)

        # 0. Check whether task transfer is needed
        if light_swarm:
            result = self._handle_task_transfer(query, light_swarm, stream)
            if result is not None:
                return result

        # 1. Process task normally
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        system_prompt = (
            f"## Agent name: {self.name}\n"
            f"## Instructions: {self.instructions}\n"
            f"## Identity: {self.role}\n"
            f"Think step by step to fulfill the user request. Answer as completely as possible, using tools when needed.\n"
            f"Today: {current_date} | Time: {current_time}"
        )
        # inject active skill content directly (no skill list in prompt)
        if use_skills and self.skill_manager.skills:
            skills_xml = self.skill_manager.get_skills_xml()
            if skills_xml:
                system_prompt += f"\n\n## Available skills\n{skills_xml}\n"
                system_prompt += "When the user request matches a skill, call activate_skill(skill_name=...) before responding."
        # Add memory context
        query = self._add_memory_context(query, user_id)

        # Chain-of-thought processing
        active_tools = []
        if self.tree_of_thought:
            tot_response, active_tools = self.run_thought(query, runtime_tools)
            system_prompt += f"\n## Supplementary information for this question\n{tot_response}"
            self.log("DEBUG", "tree_of_thought", {"response": tot_response, "active_tools": active_tools})
        # If chain-of-thought is not enabled but runtime tools are provided, use runtime tools
        elif runtime_tools:
            active_tools = runtime_tools
            self.log("DEBUG", "use_runtime_tools", {"runtime_tools": runtime_tools})

        # Append "no_think" after user query
        # modified_query = query + "/no_think"
        # Prepare API parameters
        self.chat_params = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + history + [
                {"role": "user", "content": query}],
            "stream": stream
        }

        # Add parameters
        if metadata:
            for key, value in metadata.items():
                self.chat_params[key] = value

        # Add tools
        # Priority: active_tools > runtime_tools > tools set at initialization
        final_tools = []
        if active_tools:
            final_tools = active_tools
        elif runtime_tools:
            final_tools = runtime_tools
        else:
            final_tools = self.tool_registry.get_tools()

        if final_tools:
            self.chat_params["tools"] = final_tools
            self.chat_params["tool_choice"] = "auto"
            self.log("DEBUG", "final_tools_selected",
                     {"tools": [t.get("function", {}).get("name", str(t)) for t in final_tools]})

        # Add trace session
        if hasattr(self, 'tracetools') and self.tracetools:
            self.chat_params["session_id"] = traceid

        # Call the model
        self.log("DEBUG", "first_request_params", {"params": self.chat_params})
        self._record_trace("model_request", self._build_model_request_trace(self.chat_params))
        try:
            for _attempt in range(4):
                try:
                    response = self.client.chat.completions.create(**self.chat_params)
                    break
                except Exception as _e:
                    if "429" in str(_e) and _attempt < 3:
                        time.sleep(2 ** _attempt)
                        continue
                    raise
        except Exception as e:
            error_msg = format_lightagent_error(e, "create chat completion")
            self.log("ERROR", "model_request_failed", {"error": error_msg})
            self._record_trace("error", {"stage": "model_request", "error": error_msg})
            self._record_trace("run_end", {"success": False, "error": error_msg})
            if stream:
                stream_result = self._error_stream(error_msg)
                if result_format == "event":
                    return self._stream_as_events(stream_result, traceid)
                return stream_result
            return self._format_run_result(error_msg, result_format, traceid, error_msg)

        result = self._core_run_logic(response, stream, max_retry)
        if stream:
            if result_format == "event":
                return self._stream_as_events(result, traceid)
            return result
        return self._format_run_result(result, result_format, traceid)

    def _record_trace(self, event_type: str, data: Dict[str, Any] | None = None):
        """Record a trace event when tracing is enabled."""
        recorder = getattr(self, "_trace_recorder", None)
        if recorder:
            return recorder.record(event_type, data)
        return None

    def export_trace(self) -> List[Dict[str, Any]]:
        """Return structured trace events from the most recent run."""
        recorder = getattr(self, "_trace_recorder", None)
        if not recorder:
            return []
        return recorder.to_list()

    def _build_model_request_trace(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build a prompt-safe summary of a model request for trace events."""
        tools = []
        for tool in params.get("tools", []) or []:
            tools.append(tool.get("function", {}).get("name", str(tool)))
        return {
            "model": params.get("model"),
            "stream": bool(params.get("stream")),
            "message_count": len(params.get("messages", [])),
            "tools": tools,
        }

    @staticmethod
    def _format_guardrail_error(stage: str, reason: str | None = None) -> str:
        details = {"stage": stage}
        if reason:
            details["reason"] = reason
        return format_error_code("LA-GUARDRAIL", details)

    def _check_tool_guardrails(self, tool_name: str, arguments: Dict[str, Any]) -> str | None:
        decision = self.guardrails.check_tool(tool_name, arguments, {
            "agent_name": self.name,
            "trace_id": self.traceid,
        })
        if decision.allowed:
            return None
        error_msg = self._format_guardrail_error("tool", decision.reason)
        self._record_trace("guardrail_block", {
            "stage": "tool",
            "tool": tool_name,
            "reason": decision.reason,
        })
        return error_msg

    def _apply_output_guardrails(self, output: str) -> str:
        decision = self.guardrails.check_output(output, {
            "agent_name": self.name,
            "trace_id": self.traceid,
        })
        if not decision.allowed:
            error_msg = self._format_guardrail_error("output", decision.reason)
            self._record_trace("guardrail_block", {"stage": "output", "reason": decision.reason})
            return error_msg
        if decision.value is not None:
            return str(decision.value)
        return output

    def _process_runtime_tools(self, tools: List[Union[str, Callable]]) -> List[Dict]:
        """
        Process tools passed in at runtime and return tool descriptions in OpenAI format.

        :param tools: List of tools passed in at runtime.
        :return: List of tool descriptions in OpenAI format.
        """
        temp_registry = ToolRegistry()

        for tool in tools:
            if isinstance(tool, str):
                try:
                    tool_func = self.tool_loader.load_tool(tool)
                    if temp_registry.register_tool(tool_func):
                        self.tool_registry.register_tool(tool_func)
                        self.loaded_tools[tool_func.tool_info["tool_name"]] = tool_func
                except Exception as e:
                    self.log("ERROR", "load_runtime_tool", {"tool": tool, "error": str(e)})
            elif callable(tool) and hasattr(tool, "tool_info"):
                if temp_registry.register_tool(tool):
                    self.tool_registry.register_tool(tool)
                    self.loaded_tools[tool.tool_info["tool_name"]] = tool

        runtime_tools = temp_registry.get_tools()
        self.tool_dispatcher = AsyncToolDispatcher(self.tool_registry.function_mappings, self.tool_registry.function_info)
        self.log("DEBUG", "runtime_tools_processed", {"count": len(runtime_tools)})
        return runtime_tools

    def _format_run_result(
            self,
            content: Any,
            result_format: str,
            trace_id: str,
            error: str | None = None,
    ) -> Union[str, RunResult, Dict[str, Any]]:
        """Format the final non-streaming response while preserving legacy defaults."""
        text = "" if content is None else str(content)
        detected_error = error
        if detected_error is None and text.startswith("[LA-"):
            detected_error = text

        if result_format == "str":
            return text

        result = RunResult(
            content=text,
            reasoning_content=getattr(self, "_current_reasoning_content", "") or None,
            tool_calls=deepcopy(getattr(self, "_current_tool_calls", [])),
            usage=deepcopy(getattr(self, "_current_usage", None)),
            trace_id=trace_id,
            trace=self.export_trace(),
            error=detected_error,
        )
        if result_format == "dict":
            return {
                "content": result.content,
                "reasoning_content": result.reasoning_content,
                "tool_calls": result.tool_calls,
                "usage": result.usage,
                "trace_id": result.trace_id,
                "trace": result.trace,
                "error": result.error,
            }
        return result

    def _stream_as_events(self, stream_result: Generator[Any, None, None], trace_id: str) -> Generator[StreamEvent, None, None]:
        """Wrap legacy stream chunks as structured events when explicitly requested."""
        for chunk in stream_result:
            if isinstance(chunk, dict):
                if "error" in chunk:
                    yield StreamEvent(type="error", data=chunk, trace_id=trace_id)
                elif "output" in chunk:
                    yield StreamEvent(type="tool_result", data=chunk, trace_id=trace_id)
                elif "name" in chunk and "arguments" in chunk:
                    yield StreamEvent(type="tool_call", data=chunk, trace_id=trace_id)
                else:
                    yield StreamEvent(type="event", data=chunk, trace_id=trace_id)
            elif isinstance(chunk, str) and chunk.startswith("[LA-"):
                yield StreamEvent(type="error", data=chunk, trace_id=trace_id)
            else:
                yield StreamEvent(type="content", data=chunk, trace_id=trace_id)

    def _add_memory_context(self, query: str, user_id: str) -> str:
        """Add memory context"""
        if not self.memory:
            return query

        context = ""
        memory_user_id = self.memory_policy.scoped_user_id(user_id)
        related_memories = self.memory.retrieve(query=query, user_id=memory_user_id)
        related_memories = self._filter_memory_results(related_memories, memory_user_id, user_id)
        if related_memories and related_memories.get("results"):
            context += "\n## User preferences\nThe user previously mentioned:\n" + "\n".join(
                [m["memory"] for m in related_memories["results"]]
            )
        self._store_memory_with_policy(
            data=query,
            memory_user_id=memory_user_id,
            original_user_id=user_id,
            source="user",
            scope="user",
        )

        if self.self_learning:
            agent_user_id = self.memory_policy.scoped_user_id(self.name)
            agent_memories = self.memory.retrieve(query=query, user_id=agent_user_id)
            agent_memories = self._filter_memory_results(agent_memories, agent_user_id, self.name)
            if agent_memories and agent_memories.get("results"):
                context += "\n## Supplementary information related to this question:\n" + "\n".join(
                    [m["memory"] for m in agent_memories["results"]]
                )
            self._store_memory_with_policy(
                data=query,
                memory_user_id=agent_user_id,
                original_user_id=self.name,
                source="reflection",
                scope="agent",
            )

        return f"{context}\n## User question:\n{query}" if context else query

    def _store_memory_with_policy(
            self,
            *,
            data: str,
            memory_user_id: str,
            original_user_id: str,
            source: str,
            scope: str,
    ) -> bool:
        """Persist memory only when the configured memory policy allows it."""
        if not self.memory:
            return False
        context = {
            "agent_name": self.name,
            "trace_id": self.traceid,
            "user_id": str(original_user_id),
            "memory_user_id": str(memory_user_id),
            "source": source,
            "scope": scope,
        }
        decision = self.memory_policy.allows_write(
            data,
            context,
            write_count=getattr(self, "_memory_write_count", 0),
            recent_fingerprints=getattr(self, "_memory_write_fingerprints", set()),
        )
        if not decision.allowed:
            self._record_trace("memory_write_block", {
                "reason": decision.reason,
                "source": source,
                "scope": scope,
                "user_id": str(original_user_id),
            })
            return False

        stored_data = decision.value if decision.value is not None else data
        self.memory.store(data=stored_data, user_id=memory_user_id)
        self._memory_write_count = getattr(self, "_memory_write_count", 0) + 1
        fingerprints = getattr(self, "_memory_write_fingerprints", set())
        fingerprints.add(self.memory_policy.write_fingerprint(stored_data, context))
        self._memory_write_fingerprints = fingerprints
        self._record_trace("memory_write", {
            "source": source,
            "scope": scope,
            "user_id": str(original_user_id),
        })
        return True

    def _filter_memory_results(self, memories: Any, scoped_user_id: str, original_user_id: str) -> Any:
        """Filter retrieved memories using the configured memory policy."""
        if not memories or not isinstance(memories, dict) or "results" not in memories:
            return memories
        results = memories.get("results") or []
        filtered_results = [
            item for item in results
            if self.memory_policy.allows_result(item, scoped_user_id, original_user_id)
        ]
        filtered = dict(memories)
        filtered["results"] = filtered_results
        return filtered

    def _core_run_logic(self, response, stream, max_retry) -> Union[Generator[str, None, None], str]:
        """Core run logic"""
        if stream:
            return self._run_stream_logic(response, max_retry)
        else:
            return self._run_non_stream_logic(response, max_retry)

    def _run_non_stream_logic(self, response, max_retry) -> Union[str, None]:
        """
        Non-streaming processing logic.
        """
        for _ in range(max_retry):
            if response.choices[0].message.tool_calls:
                # Initialize a list to store results from all tool calls
                tool_responses = []
                # Initialize variables
                output = ""
                function_call_name = ""
                tool_calls = response.choices[0].message.tool_calls
                self.log("DEBUG", "non_stream tool_calls", {"tool_calls": tool_calls})
                # Append tool calls and responses to the message list
                self.chat_params["messages"].append(response.choices[0].message)

                # Iterate over all tool calls
                for tool_call in tool_calls:
                    function_call = tool_call.function
                    trace_tool_call = {
                        "id": getattr(tool_call, "id", None),
                        "name": function_call.name,
                        "arguments": function_call.arguments,
                    }
                    self._current_tool_calls.append(trace_tool_call)
                    self._record_trace("tool_call", deepcopy(trace_tool_call))

                    # Attempt to auto-fix common escape issues
                    # fixed_args = function_call.arguments.replace('\\"', '"').replace('\\\\', '\\')
                    self.log("DEBUG", "non_stream function_call", {"function_call": function_call.arguments})

                    # Parse function arguments
                    try:
                        function_args = json.loads(function_call.arguments)
                    except json.JSONDecodeError as e:
                        single_tool_response = format_error_code(
                            "LA-JSON",
                            f"{str(e)}; arguments: {function_call.arguments}",
                        )
                        self._record_trace("error", {
                            "stage": "tool_arguments",
                            "tool": function_call.name,
                            "error": single_tool_response,
                        })
                        self.chat_params["messages"].append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": single_tool_response
                        })
                        tool_responses.append(single_tool_response)
                        continue

                    guardrail_error = self._check_tool_guardrails(function_call.name, function_args)
                    if guardrail_error:
                        self._record_trace("error", {
                            "stage": "tool_guardrail",
                            "tool": function_call.name,
                            "error": guardrail_error,
                        })
                        self.chat_params["messages"].append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": guardrail_error
                        })
                        tool_responses.append(guardrail_error)
                        continue

                    # Call the tool and get the response
                    # tool_response = asyncio.run(self.tool_dispatcher.dispatch(function_call.name, function_args))
                    # tool_response = await self.tool_dispatcher.dispatch(function_call.name, function_args)
                    tool_response = run_async_safely(self.tool_dispatcher.dispatch(function_call.name, function_args))

                    # If tool_response is an async function, it needs to be awaited again
                    # if asyncio.iscoroutine(tool_response) or asyncio.iscoroutinefunction(tool_response):
                    #     tool_response = await tool_response
                    # else:
                    #     tool_response = tool_response

                    function_call_name = function_call.name
                    combined_response = ""
                    single_tool_response = ""

                    # If the tool returns a generator (streaming output), accumulate all chunks
                    if isinstance(tool_response, Generator):
                        # print(f"Streaming response from tool: {function_call.name}")
                        for chunk in tool_response:
                            # print("Received chunk:", chunk)  # Print each chunk
                            if function_call_name == 'finish':
                                content = chunk.choices[0].delta.content or ""
                                combined_response += content  # Accumulate each chunk
                            else:
                                combined_response += chunk  # Accumulate each chunk
                        if combined_response == "":
                            combined_response = "".join(tool_response)

                        # Parse combined_response as a JSON object if it is a JSON string
                        try:
                            combined_response = json.loads(combined_response)  # Parse JSON
                        except json.JSONDecodeError:
                            pass  # Not a JSON string; leave as-is

                        # Convert Unicode-encoded values in the JSON object to their character forms
                        if isinstance(combined_response, dict):
                            combined_response = json.dumps(combined_response, ensure_ascii=False)  # Preserve non-ASCII characters
                        single_tool_response = combined_response  # Result for this single tool

                    else:
                        # print(f"Non-streaming response from tool: {function_call.name}")
                        combined_response = tool_response
                        # print("tool_response type:",type(combined_response))
                        # If it is a JSON string, parse and re-encode with non-ASCII preserved
                        if isinstance(combined_response, str):
                            try:
                                combined_response = json.loads(combined_response)  # Parse JSON
                                combined_response = json.dumps(combined_response, ensure_ascii=False)  # Preserve non-ASCII characters
                            except json.JSONDecodeError:
                                combined_response = tool_response
                                pass  # Not a JSON string; leave as-is
                        single_tool_response = combined_response  # Result for this single tool

                    self.log("INFO", "non_stream single_tool_response",
                             {"single_tool_response": single_tool_response})
                    self._record_trace("tool_result", {
                        "name": function_call.name,
                        "output": single_tool_response,
                    })

                    self.chat_params["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,  # Must match the id above
                        "content": f"{single_tool_response}"
                    })

                    # Append the result of this single tool to the list
                    tool_responses.append(single_tool_response)

                # # Merge all tool call results into a single string
                self.log("DEBUG", "non_stream tool_responses", {"tool_responses": tool_responses})
            else:
                # Return the final reply
                reply = response.choices[0].message.content
                reply = self._apply_output_guardrails(reply)
                self.log("INFO", "non_stream final_reply", {"reply": reply})
                self._record_trace("model_response", {"content": reply})
                if isinstance(reply, str) and reply.startswith("[LA-GUARDRAIL]"):
                    self._record_trace("error", {"stage": "output_guardrail", "error": reply})
                    self._record_trace("run_end", {"success": False, "error": reply})
                    return reply
                self._record_trace("run_end", {"success": True})
                return reply

            # Update response
            if function_call_name == 'finish':
                self._record_trace("run_end", {"success": True, "finish_tool": True})
                return  # If the finish tool was called last, end the generator
            # print("params:",self.chat_params)
            self.log("DEBUG", "non_stream chat-completions params", {"params": self.chat_params})
            self._record_trace("model_request", self._build_model_request_trace(self.chat_params))

            try:
                for _attempt in range(4):
                    try:
                        response = self.client.chat.completions.create(**self.chat_params)
                        break
                    except Exception as _e:
                        if "429" in str(_e) and _attempt < 3:
                            time.sleep(2 ** _attempt)
                            continue
                        raise
            except Exception as e:
                error_msg = format_lightagent_error(e, "continue chat completion")
                self.log("ERROR", "model_request_failed", {"error": error_msg})
                self._record_trace("error", {"stage": "model_request", "error": error_msg})
                self._record_trace("run_end", {"success": False, "error": error_msg})
                return error_msg

        # Retries exhausted
        self.log("ERROR", "max_retry_reached", {"message": "Failed to generate a valid response."})
        self._record_trace("error", {"stage": "max_retry", "error": "Failed to generate a valid response."})
        self._record_trace("run_end", {"success": False, "error": "max_retry_reached"})
        return "Failed to generate a valid response."

    def _run_stream_logic(self, response, max_retry) -> Generator[str, None, None]:
        """Streaming processing logic"""
        for _ in range(max_retry):
            try:
                # Process the current response (may contain multiple rounds of tool calls)
                while True:
                    # Initialize variables
                    output = ""
                    output_reasoning_content = ""
                    tool_calls = []  # Stores information about all tool calls
                    tool_responses = []  # Stores results from all tool calls
                    finish_called = False  # Flag indicating whether the finish tool was called
                    last_chunk = None

                    for chunk in response:
                        yield chunk  # Stream content back
                        last_chunk = chunk
                        reasoning_content = ""
                        content = ""
                        choice = ""
                        if chunk.choices and len(chunk.choices) > 0:
                            choice = chunk.choices[0]

                        if choice and hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content is not None:
                            reasoning_content = choice.delta.reasoning_content or ""

                        if reasoning_content:
                            output += reasoning_content
                            output_reasoning_content += reasoning_content

                        if choice and hasattr(choice.delta, "content") and choice.delta.content is not None:
                            content = choice.delta.content or ""

                        if content:
                            output += content

                        try:
                            # Check for tool calls
                            if chunk.choices and chunk.choices[0].delta.tool_calls:
                                tool_call_delta = chunk.choices[0].delta.tool_calls[0]

                                # Get the tool call index, ensuring it is a valid integer
                                tool_call_index = tool_call_delta.index if hasattr(tool_call_delta,
                                                                                   "index") and tool_call_delta.index is not None else 0

                                # If the tool call info has not been recorded yet, initialize an empty dict
                                if len(tool_calls) <= tool_call_index:
                                    tool_calls.append({"name": "", "arguments": "", "index": tool_call_index, "title": "", "id": ""})

                                # Update the tool call ID
                                if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                                    tool_calls[tool_call_index]["id"] = tool_call_delta.id

                                # Update the tool call name
                                if hasattr(tool_call_delta.function, "name") and tool_call_delta.function.name:
                                    tool_calls[tool_call_index]["name"] = tool_call_delta.function.name

                                # Update the tool call arguments
                                if hasattr(tool_call_delta.function, "arguments") and tool_call_delta.function.arguments:
                                    tool_calls[tool_call_index]["arguments"] += tool_call_delta.function.arguments

                        except (IndexError, AttributeError, KeyError) as e:
                            self.log("ERROR", "tool_call_error", {
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            })

                        # After the loop ends, check whether the last chunk has usage info
                        # if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                        #     print(f"Token usage: {last_chunk.usage}")
                        #     # Can access specific fields
                        #     print(f"Prompt tokens: {last_chunk.usage.prompt_tokens}")
                        #     print(f"Completion tokens: {last_chunk.usage.completion_tokens}")
                        #     print(f"Total tokens: {last_chunk.usage.total_tokens}")

                        # If streaming output has ended
                        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                        # if finish_reason == "stop" and not any(tc["name"] for tc in tool_calls):
                        #     self.log("INFO", "stream_response", {"output": output})
                        #     return  # End the generator
                        # Determine whether the end of the streaming response has been reached:
                        # - there is a finish_reason (non-null)
                        # - or choices is empty and the chunk contains usage info (characteristic of the last API chunk)
                        if finish_reason is not None or (
                                not chunk.choices and hasattr(chunk, 'usage') and chunk.usage is not None):
                            # Can record token usage here if available
                            if hasattr(chunk, 'usage') and chunk.usage:
                                self._current_usage = chunk.usage
                                self.log("INFO", "token_usage", {"usage": chunk.usage})

                            # If there are no tool calls, the entire response is done; exit the generator directly
                            if not any(tc["name"] for tc in tool_calls):  # tool_calls is the previously collected list
                                self.log("INFO", "stream_response", {"output": output})
                                self._record_trace("model_response", {"content": output})
                                self._record_trace("run_end", {"success": True})
                                return  # End the generator
                            # Otherwise (tool calls present), don't exit early; let the loop finish naturally,
                            # and the tool handling logic outside the loop will take over

                        # If tool calls have finished
                        if finish_reason in ("tool_calls", "stop") and any(tc["name"] for tc in tool_calls):
                            # Iterate over all tool calls
                            self.log("DEBUG", "stream tool_calls", {"tool_calls": tool_calls})
                            for tool_call in tool_calls:
                                if tool_call["name"]:  # Ensure the tool call has a name
                                    tool_name = tool_call["name"]
                                    arguments = tool_call["arguments"]

                                    # Get the tool title from the registry
                                    tool_info = self.tool_registry.function_info.get(tool_name, {})
                                    tool_title = tool_info.get("tool_title") or ""

                                    # Update tool call info
                                    tool_call["title"] = tool_title

                                    # Log the tool call
                                    tool_call_info = {
                                        "name": tool_name,
                                        "title": tool_title,
                                        "arguments": arguments,
                                    }
                                    self._current_tool_calls.append(deepcopy(tool_call_info))
                                    self._record_trace("tool_call", deepcopy(tool_call_info))
                                    self.log("INFO", "stream function_call", {"tool_call_start": tool_call_info})
                                    # Push tool call info to the developer
                                    yield tool_call_info

                                    # Parse arguments and call the tool
                                    try:

                                        # Attempt to auto-fix common escape issues
                                        # fixed_args = json_obj.replace('\\"', '"').replace('\\\\', '\\')
                                        # self.log("DEBUG", "stream fixed_args", {"fixed_args": fixed_args})
                                        # function_args = json.loads(fixed_args)
                                        function_args = self._parse_tool_arguments(arguments)
                                        guardrail_error = self._check_tool_guardrails(tool_name, function_args)
                                        if guardrail_error:
                                            self._record_trace("error", {
                                                "stage": "tool_guardrail",
                                                "tool": tool_name,
                                                "error": guardrail_error,
                                            })
                                            tool_responses.append(guardrail_error)
                                            yield {"name": tool_name, "title": tool_title, "error": guardrail_error}
                                            continue

                                        # Call the tool
                                        # tool_response = asyncio.run(self.tool_dispatcher.dispatch(tool_name, function_args))
                                        # tool_response = await self.tool_dispatcher.dispatch(tool_name, function_args)
                                        tool_response = run_async_safely(self.tool_dispatcher.dispatch(tool_name, function_args))
                                        # If tool_response is an async function, it needs to be awaited again
                                        # if asyncio.iscoroutine(tool_response) or asyncio.iscoroutinefunction(tool_response):
                                        #     tool_response = await tool_response
                                        # else:
                                        #     tool_response = tool_response

                                        # Handle different types of tool responses
                                        combined_response = ""
                                        single_tool_response = ""

                                        # If the tool returns a generator (streaming output), accumulate all chunks
                                        if isinstance(tool_response, Generator):
                                            # print(f"Streaming response from tool: {function_call['name']}")
                                            for chunk in tool_response:
                                                # Continue streaming data returned by the tool
                                                if isinstance(chunk, ChatCompletionChunk):
                                                    yield chunk
                                                else:
                                                    tool_output = {
                                                        "name": tool_name,
                                                        "title": tool_title,
                                                        "output": chunk,
                                                    }
                                                    self.log("DEBUG", "stream tool_output",
                                                             {"tool_output": tool_output})
                                                    self._record_trace("tool_result", deepcopy(tool_output))
                                                    yield tool_output
                                                # Push tool call info to the developer
                                                if tool_name == 'finish':
                                                    content = chunk.choices[0].delta.content or ""
                                                    combined_response += content  # Accumulate each chunk
                                                else:
                                                    combined_response += chunk  # Accumulate each chunk
                                            single_tool_response = combined_response  # Result for this single tool
                                        else:
                                            # print(f"Non-streaming response from tool: {tool_response}")
                                            combined_response = str(tool_response)
                                            single_tool_response = combined_response  # Result for this single tool
                                            tool_output = {
                                                "name": tool_name,
                                                "title": tool_title,
                                                "output": combined_response
                                            }
                                            self._record_trace("tool_result", deepcopy(tool_output))
                                            yield tool_output

                                        # Log the tool response
                                        self.log("INFO", "stream single_tool_response",
                                                 {"single_tool_response": single_tool_response})

                                        # Save the result of this single tool to the list
                                        tool_responses.append(single_tool_response)

                                        # Check whether the finish tool was called
                                        if tool_name == 'finish':
                                            finish_called = True
                                            self.log("INFO", "finish_tool_called", {"response": combined_response})
                                            self._record_trace("run_end", {"success": True, "finish_tool": True})

                                    except json.JSONDecodeError as e:
                                        error_msg = format_error_code("LA-JSON", f"{str(e)}; arguments: {arguments}")
                                        self.log("ERROR", "json_decode_error",
                                                 {"tool": tool_name, "title": tool_title, "error": error_msg})
                                        self._record_trace("error", {
                                            "stage": "tool_arguments",
                                            "tool": tool_name,
                                            "error": error_msg,
                                        })
                                        tool_responses.append(error_msg)
                                        yield {"name": tool_name, "title": tool_title, "error": error_msg}

                                    except Exception as e:
                                        error_msg = f"{format_lightagent_error(e, 'execute stream tool', default_code='LA-TOOL')}\n{traceback.format_exc()}"
                                        self.log("ERROR", "tool_execution_error", {
                                            "tool": tool_name,
                                            "title": tool_title,
                                            "error": error_msg
                                        })
                                        self._record_trace("error", {
                                            "stage": "tool_execution",
                                            "tool": tool_name,
                                            "error": error_msg,
                                        })
                                        tool_responses.append(error_msg)
                                        yield {"name": tool_name, "title": tool_title, "error": error_msg}

                            # If the finish tool was called, end processing
                            if finish_called:
                                return

                            # Prepare for the next round of requests
                            # Add tool calls and responses to the message history
                            assistant_message = {
                                "role": "assistant",
                                "content": "",  # Must use an empty string, not None
                                "reasoning_content": output_reasoning_content,  # Required for deepseek v4
                                "tool_calls": []
                            }

                            # Build the correct format for each tool call
                            for i, tool_call in enumerate(tool_calls):
                                if tool_call["name"]:  # Ensure the tool call has a name
                                    # Use the ID returned by the model, or generate one if absent
                                    tool_call_id = tool_call.get("id") or f"call_{uuid4().hex[:8]}"
                                    assistant_message["tool_calls"].append({
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call["name"],
                                            "arguments": tool_call["arguments"]
                                        }
                                    })
                            # Add the assistant message to history
                            self.chat_params["messages"].append(assistant_message)

                            # Add tool response messages (role must be "tool")
                            for i, (tool_call, tool_response) in enumerate(zip(tool_calls, tool_responses)):
                                if tool_call["name"]:  # Ensure the tool call has a name
                                    self.chat_params["messages"].append({
                                        "role": "tool",
                                        "tool_call_id": assistant_message["tool_calls"][i]["id"],  # Use the corresponding call_id
                                        "content": str(tool_response)  # Ensure it is a string
                                    })

                            # Create a new response stream
                            self.log("DEBUG", "stream next_request_params", {"params": self.chat_params})
                            self._record_trace("model_request", self._build_model_request_trace(self.chat_params))
                            try:
                                for _attempt in range(4):
                                    try:
                                        response = self.client.chat.completions.create(**self.chat_params)
                                        break
                                    except Exception as _e:
                                        if "429" in str(_e) and _attempt < 3:
                                            time.sleep(2 ** _attempt)
                                            continue
                                        raise
                            except Exception as e:
                                error_msg = format_lightagent_error(e, "continue streaming chat completion")
                                self.log("ERROR", "model_request_failed", {"error": error_msg})
                                self._record_trace("error", {"stage": "model_request", "error": error_msg})
                                self._record_trace("run_end", {"success": False, "error": error_msg})
                                yield error_msg
                                return
                            break
            except Exception as e:
                self.log("WARNING", "retry", {"error": str(e)})
                self._record_trace("error", {"stage": "stream_retry", "error": str(e)})
                continue

        else:
            # Retries exhausted
            self.log("ERROR", "max_retry_reached", {"message": f"Max retry({max_retry}) reached."})
            self._record_trace("error", {"stage": "max_retry", "error": "Failed to stream a valid response."})
            self._record_trace("run_end", {"success": False, "error": "max_retry_reached"})
            yield "Failed to stream a valid response."
            return  # Or simply exit

    @staticmethod
    def _error_stream(message: str) -> Generator[str, None, None]:
        yield message

    def _handle_task_transfer(
            self,
            query: str,
            light_swarm: 'LightSwarm',
            stream: bool = False,
    ) -> Union[Generator[str, None, None], str, None]:
        """
        Handle task transfer logic.

        :param query: User input.
        :param light_swarm: LightSwarm instance.
        :param stream: Whether to enable streaming output.
        :return: A generator or string if the task needs to be transferred; otherwise None.
        """
        intent = self._detect_intent(query, light_swarm)
        if intent and intent.get("transfer_to"):
            target_agent_name = intent["transfer_to"]
            self.log("INFO", "detect_intent", {"intent": intent})
            if target_agent_name == self.name:
                self.log("INFO", "self_transfer_detected", {"target_agent": target_agent_name})
                return None  # If the target is self, return None directly
            if stream:
                return self._handle_task_transfer_stream(light_swarm.agents[target_agent_name], query, light_swarm)
            else:
                return self._handle_task_transfer_non_stream(light_swarm.agents[target_agent_name], query, light_swarm)
        return None

    def _handle_task_transfer_stream(
            self,
            target_agent: 'LightAgent',
            context: str,
            light_swarm: 'LightSwarm',
    ) -> Generator[str, None, None]:
        """
        Handle task transfer logic (streaming output).

        :param target_agent: Target agent.
        :param context: Shared context information.
        :param light_swarm: LightSwarm instance.
        :return: Generator for streaming output.
        """
        self.log("INFO", "transfer_to_agent", {"from": self.name, "to": target_agent.name, "context": context})

        # Check whether the target agent is valid
        if not hasattr(target_agent, 'run'):
            self.log("ERROR", "invalid_target_agent", {"target_agent": target_agent})
            yield "Failed to transfer task: invalid target agent"
            return

        try:
            yield from target_agent.run(context, light_swarm=light_swarm, stream=True)
        except Exception as e:
            self.log("ERROR", "run_failed", {"error": str(e)})
            raise  # Re-raise the exception for debugging

    def _handle_task_transfer_non_stream(
            self,
            target_agent: 'LightAgent',
            context: str,
            light_swarm: 'LightSwarm',
    ) -> str:
        """
        Handle task transfer logic (non-streaming output).

        :param target_agent: Target agent.
        :param context: Shared context information.
        :param light_swarm: LightSwarm instance.
        :return: String representing the non-streaming output result.
        """
        self.log("INFO", "transfer_to_agent", {"from": self.name, "to": target_agent.name, "context": context})

        # Check whether the target agent is valid
        if not hasattr(target_agent, 'run'):
            self.log("ERROR", "invalid_target_agent", {"target_agent": target_agent})
            return "Failed to transfer task: invalid target agent"

        try:
            result = target_agent.run(context, light_swarm=light_swarm, stream=False)
            if isinstance(result, Generator):
                return "".join(result)  # Convert the generator to a string
            return result
        except Exception as e:
            self.log("ERROR", "run_failed", {"error": str(e)})
            raise  # Re-raise the exception for debugging

    def _build_context(self, related_memories):
        """
        Build context by combining user input with memory content.
        :param related_memories: Relevant content retrieved from memory.
        :return: Context combined with memory.
        """
        if not related_memories or not related_memories["results"]:
            return ""

        memory_context = "\n".join([m["memory"] for m in related_memories["results"]])
        if not memory_context:
            return ""

        prompt = f"\n## User preferences\nThe user previously mentioned:\n{memory_context}."
        self.log("DEBUG", "related_memories", {"memory_context": memory_context})
        return prompt

    def _build_agent_memory(self, agent_memories):
        """
        Build context by combining user input with memory content.

        :param agent_memories: Relevant content retrieved from memory.
        :return: Context combined with memory.
        """
        if not agent_memories or not agent_memories["results"]:
            return ""

        memory_context = "\n".join([m["memory"] for m in agent_memories["results"]])
        if not memory_context:
            return ""

        prompt = f"\n## Supplementary information relevant to solving this problem:\n{memory_context}."
        self.log("DEBUG", "agent_memories", {"memory_context": memory_context})
        return prompt

    def run_thought(self, query: str, runtime_tools: List[Dict] | None = None) -> tuple:
        """Use a tree-of-thought approach to have the model generate a tool-use plan for the user's query based on get_tools_str first."""
        tot_model = self.tot_model
        # Modification: prefer runtime tools; fall back to initialization-time tools if none provided
        if runtime_tools:
            # Convert runtime_tools to string form
            tools = json.dumps(runtime_tools, indent=4, ensure_ascii=False)
            # Create a temporary ToolRegistry for filtering tools
            temp_registry = ToolRegistry()
            # Register runtime_tools into the temporary registry
            for tool_schema in runtime_tools:
                # The OpenAI-format tool schema needs to be converted to the internal format here.
                # For simplicity, this is left as a no-op; a more complex conversion may be needed in practice.
                pass
        else:
            tools = self.tool_registry.get_tools_str()

        if not isinstance(tools, str):
            tools = str(tools)  # Ensure tools is a string
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        system_prompt = f"""You are an intelligent assistant. Based on the user's question and the tool-use plan, generate a tree of thought, follow its steps to call tools in order, and produce a final answer.\n Today's date: {current_date} Current time: {current_time} \n Tool list: {tools}"""
        self.log("DEBUG", "run_thought", {"system_prompt": system_prompt})

        try:
            # 1. First request: generate the initial tool-use plan
            params = dict(model=tot_model,
                          messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                          stream=False)
            response = self.tot_client.chat.completions.create(**params)
            thought_response = response.choices[0].message.content
            self.log("DEBUG", "thought_response", {"response": thought_response})

            # 2. Second request: ask the model to reflect and produce a revised tool-use plan
            reflection_prompt = "Please reflect on your answer and plan strictly using the tools in <Tool list>. Do not invent new tools. Output only the revised task plan, without any other analysis or commentary."
            reflection_params = dict(model=tot_model, messages=[
                {"role": "user", "content": f"{system_prompt} /n Start thinking about the problem: {query}"},
                {"role": "assistant", "content": thought_response},
                {"role": "user", "content": reflection_prompt}
            ], stream=False)
            self.log("DEBUG", "reflection_params", {"params": reflection_params})
            reflection_response = self.tot_client.chat.completions.create(**reflection_params)
            refined_content = reflection_response.choices[0].message.content
            self.log("DEBUG", "reflection_response", {"response": refined_content})

            # Get the set of tools to use
            tool_reflection_prompt = """Please follow these requirements strictly:
            1. Analyze the problem requirements and plan which tools to use
            2. Output only the JSON-format result containing tool names
            3. Use the following structure (example):
            {"tools": [{"name": "tool_name_1"}, {"name": "tool_name_2"}]}
            4. Do not include any explanatory content"""

            tool_reflection_params = dict(
                model=tot_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Problem analysis request: {query}"},
                    {"role": "assistant", "content": refined_content},
                    {"role": "user", "content": tool_reflection_prompt}
                ],
                response_format={"type": "json_object"},  # Force JSON output format
                stream=False
            )

            self.log("DEBUG", "tool_reflection_params", {"params": tool_reflection_params})
            tool_reflection_response = self.tot_client.chat.completions.create(**tool_reflection_params)
            tool_reflection_result = tool_reflection_response.choices[0].message.content
            self.log("DEBUG", "tool_reflection_result", {"result": tool_reflection_result})

            # 3. Perform adaptive tool filtering
            current_tools = []
            if self.filter_tools:
                # Modification: prefer runtime tools for filtering
                if runtime_tools:
                    # Use a temporary registry for filtering
                    temp_registry = ToolRegistry()
                    for tool_schema in runtime_tools:
                        # The OpenAI-format schema needs to be converted to the internal format here.
                        # Simplified: add directly to the registry.
                        pass
                    current_tools = runtime_tools  # Temporarily use runtime tools directly
                else:
                    current_tools = self.tool_registry.filter_tools(tool_reflection_result)
                self.log("DEBUG", "current_tools", {"get_tools": current_tools})

            return refined_content, current_tools

        except Exception as e:
            self.log("ERROR", "run_thought_failure", {"error": str(e)})
            raise RuntimeError(f"Chain-of-thought execution failed: {str(e)}") from e

    def _detect_intent(self, query: str, light_swarm=None) -> Optional[Dict]:
        """
        Use the language model to determine user intent.

        :param query: User input.
        :param light_swarm: LightSwarm instance used to retrieve all agent information.
        :return: Intent information, e.g. {"transfer_to": "Agent B"}.
        """
        if not light_swarm:
            return None

        # Collect information about all agents
        agents_info = []
        for agent_name, agent in light_swarm.agents.items():
            agents_info.append(f"Agent name: {agent_name}, Agent instructions: {agent.instructions}")

        # Join agent information into a string
        agents_info_str = "\n".join(agents_info)

        # Build the prompt
        prompt = f"""Please analyze the intent of the following user input. If a task transfer is needed, return the target agent's name in the format below.
        transfer to agent_name
        The following is information about all available agents:
            {agents_info_str}
        User input: {query}
        Please return the target agent's name:
        """

        # Call the language model to determine intent
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}]
        )
        intent = response.choices[0].message.content
        self.log("DEBUG", "detect_intent", {"intent": intent})

        # # Parse intent using regex
        # match = re.search(r"transfer to (\w+)", intent, re.IGNORECASE)
        # if match:
        #     target_agent_name = match.group(1)
        #     if target_agent_name in light_swarm.agents:
        #         return {"transfer_to": target_agent_name}
        # return None

        # Parse intent
        for agent_name in light_swarm.agents.keys():
            if f"transfer to {agent_name}" in intent:
                return {"transfer_to": agent_name}
        return None

    def _transfer_to_agent(
            self,
            target_agent: 'LightAgent',
            context: str,
            light_swarm=None,
            stream: bool = False,
    ) -> Union[Generator[str, None, None], str]:
        """
        Transfer the task to another agent, supporting both streaming and non-streaming output.

        :param target_agent: Target agent.
        :param context: Shared context information.
        :param light_swarm: LightSwarm instance.
        :param stream: Whether to enable streaming output.
        :return: A generator if stream=True; otherwise the full result string.
        """
        self.log("INFO", "transfer_to_agent", {"from": self.name, "to": target_agent.name, "context": context})

        # Check whether the target agent is valid
        if not hasattr(target_agent, 'run'):
            self.log("ERROR", "invalid_target_agent", {"target_agent": target_agent})
            return "Failed to transfer task: invalid target agent"
        #
        # # Call the target agent's run method
        # if stream:
        #     yield from target_agent.run(context, light_swarm=light_swarm, stream=stream)
        # else:
        #     result = target_agent.run(context, light_swarm=light_swarm, stream=stream)
        #     if isinstance(result, Generator):
        #         return "".join(result)  # Convert the generator to a string
        #     return result
        try:
            if stream:
                yield from target_agent.run(context, light_swarm=light_swarm, stream=stream)
            else:
                result = target_agent.run(context, light_swarm=light_swarm, stream=stream)
                if isinstance(result, Generator):
                    return "".join(result)  # Convert the generator to a string
                return result
        except Exception as e:
            self.log("ERROR", "run_failed", {"error": str(e)})
            raise  # Re-raise the exception for debugging

    def create_tool(self, user_input: str, tools_directory: str = "tools"):
        """
        Generate Python code based on user-provided text and save it as a tool.
        """
        # Call the language model to generate Python code
        system_prompt = """
        The user will provide some exam text. Please parse the "tool_name" and "code" and output them in JSON format.

        EXAMPLE INPUT:
        Please generate a weather query tool based on the documentation. The API is described below.

        EXAMPLE JSON OUTPUT:
        {'tools': [{
            "tool_name": "get_weather",
            "tool_code": "import requests
            def get_weather(
        city_name: str
) -> str:
    /"/"/"
    Get the current weather for `city_name`
    /"/"/"
    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc", "observation_time"],
    }
    try:
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        import traceback
        ret = "Error encountered while fetching weather data!\n" + traceback.format_exc()

    return str(ret)

# Define tool info inside the function
get_weather.tool_info = {
    "tool_name": "get_weather",
    "tool_title": "Weather query",
    "tool_description": "Get current weather information for a specified city",
    "tool_params": [
        {"name": "city_name", "description": "Name of the city to query", "type": "string", "required": True},
    ]
}"
        }]}
        """
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates Python code in JSON format."},
                {"role": "user", "content": f"Generate Python tools based on the following description. "
                                            f"Return a JSON array where each item contains 'tool_name' and 'tool_code'. "
                                            f"\n {system_prompt} "
                                            f"Description:\n{user_input}"},
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.client.chat.completions.create(**params)
            response_data = json.loads(response.choices[0].message.content)

            # Ensure the returned data is a JSON object
            if not isinstance(response_data, dict):
                raise ValueError("Response is not a JSON object.")

            # Iterate over each tool
            for tool_data in response_data["tools"]:
                tool_name = tool_data.get("tool_name")
                tool_code = tool_data.get("tool_code")

                if not tool_name or not tool_code:
                    self.log("ERROR", "invalid_tool_data", {"tool_data": tool_data})
                    continue
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(tool_name)):
                    self.log("ERROR", "invalid_tool_name", {"tool_name": tool_name})
                    continue

                # Save the generated code to the tools directory
                tools_dir = os.path.abspath(tools_directory)
                os.makedirs(tools_dir, exist_ok=True)
                tool_path = os.path.abspath(os.path.join(tools_dir, f"{tool_name}.py"))
                if not tool_path.startswith(tools_dir + os.sep):
                    self.log("ERROR", "invalid_tool_path", {"tool_name": tool_name, "tool_path": tool_path})
                    continue
                with open(tool_path, "w", encoding="utf-8") as f:
                    f.write(tool_code)
                self.log("INFO", "tool_created", {"tool_name": tool_name, "tool_path": tool_path})

                # Automatically load the newly created tool
                self.load_tools([tool_name], tools_directory)
        except Exception as e:
            self.log("ERROR", "tool_creation_failed", {"error": str(e)})

    def _parse_tool_arguments(self, arguments_str: str) -> Dict[str, Any]:
        """
        Parse tool call arguments, handling various escape and formatting issues.

        Args:
            arguments_str: Raw argument string.

        Returns:
            Parsed argument dictionary.
        """

        # Method 1: attempt direct parsing
        try:
            # First try to fix common escape issues
            # self.log("DEBUG", "parse_tool_arguments_success_method 0:", {"result": arguments_str})
            # fixed_args = arguments_str.replace('\\"', '"').replace('\\\\', '\\')
            result = json.loads(arguments_str)
            self.log("DEBUG", "parse_tool_arguments_success_method 1:", {"result": result})
            return result
        except json.JSONDecodeError:
            pass

        # Method 2: attempt to extract a JSON object
        try:
            # Use regex to find a JSON object
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, arguments_str, re.DOTALL)
            if match:
                json_str = match.group()
                # Fix escapes
                json_str = json_str.replace('\\"', '"').replace('\\\\', '\\')
                # Fix nested quotes
                json_str = re.sub(r'(?<!\\)"([^"]*?)"', lambda m: '"' + m.group(1).replace('"', '\\"') + '"', json_str)
                result = json.loads(json_str)
                self.log("DEBUG", "parse_tool_arguments_success_method2", {"result": result})
                return result
        except (json.JSONDecodeError, AttributeError):
            pass

        # Method 3: manually parse key-value pairs
        try:
            # Remove curly braces
            content = arguments_str.strip()
            if content.startswith('{') and content.endswith('}'):
                content = content[1:-1].strip()

            # Split key-value pairs
            result = {}
            current_key = None
            current_value = ""
            in_string = False
            escape_next = False
            bracket_depth = 0

            i = 0
            while i < len(content):
                char = content[i]

                if escape_next:
                    current_value += char
                    escape_next = False
                    i += 1
                    continue

                if char == '\\':
                    escape_next = True
                    current_value += char
                    i += 1
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    current_value += char
                    i += 1
                    continue

                if not in_string:
                    if char == '{':
                        bracket_depth += 1
                        current_value += char
                    elif char == '}':
                        bracket_depth -= 1
                        current_value += char
                    elif char == ':' and bracket_depth == 0 and current_key is None:
                        # Encountered a colon — the preceding part is the key
                        current_key = current_value.strip().strip('"')
                        current_value = ""
                        i += 1
                        continue
                    elif char == ',' and bracket_depth == 0 and current_key is not None:
                        # Encountered a comma — save the current key-value pair
                        try:
                            # Attempt to parse the value
                            value_str = current_value.strip()
                            if value_str.startswith('"') and value_str.endswith('"'):
                                # String value
                                result[current_key] = value_str[1:-1].replace('\\"', '"')
                            elif value_str == 'true':
                                result[current_key] = True
                            elif value_str == 'false':
                                result[current_key] = False
                            elif value_str == 'null':
                                result[current_key] = None
                            else:
                                # Attempt to parse as a number
                                try:
                                    if '.' in value_str:
                                        result[current_key] = float(value_str)
                                    else:
                                        result[current_key] = int(value_str)
                                except ValueError:
                                    # Not a number; leave as-is
                                    result[current_key] = value_str
                        except Exception as e:
                            result[current_key] = current_value

                        current_key = None
                        current_value = ""
                        i += 1
                        continue

                current_value += char
                i += 1

            # Handle the last key-value pair
            if current_key is not None:
                value_str = current_value.strip()
                if value_str.startswith('"') and value_str.endswith('"'):
                    result[current_key] = value_str[1:-1].replace('\\"', '"')
                else:
                    result[current_key] = value_str

            if result:
                self.log("DEBUG", "parse_tool_arguments_success_method3", {"result": result})
                return result

        except Exception as e:
            self.log("DEBUG", "parse_tool_arguments_method3_failed", {"error": str(e)})

        # All methods failed; raise an exception
        self.log("ERROR", "parse_tool_arguments_all_failed", {"arguments": arguments_str})
        raise json.JSONDecodeError(f"Failed to parse arguments: {arguments_str}", arguments_str, 0)


class LightSwarm:
    def __init__(self):
        self.agents: Dict[str, LightAgent] = {}

    def register_agent(self, *agents: LightAgent):
        """
        Register one or more agents.

        :param agents: Agent instances to register; supports multiple agents.
        """
        for agent in agents:
            if agent.name in self.agents:
                # print(f"Agent '{agent.name}' is already registered.")
                agent.log("INFO", "register_agent", {"agent_name": agent.name, "status": "already_registered"})
            else:
                self.agents[agent.name] = agent
                # print(f"Agent '{agent.name}' registered.")
                agent.log("INFO", "register_agent", {"agent_name": agent.name, "status": "registered"})

    def run(self, agent: LightAgent, query: str, stream=False):
        """
        Run the specified agent.

        :param agent_name: Agent name.
        :param query: User input.
        :return: The agent's reply.
        """
        if agent.name not in self.agents:
            raise ValueError(f"Agent '{agent.name}' not found.")
        return agent.run(query, light_swarm=self, stream=stream)


if __name__ == "__main__":
    # Example of registering and using a tool
    print("This is LightAgent")
    # print(dispatch_tool("example_tool", {"param1": "test"}))
