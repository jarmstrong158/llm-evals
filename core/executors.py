"""
Executors — the "something that runs" in input → run → output.

Each executor is a callable: executor(input) -> str

Input can be str or dict. Output is always str.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from typing import Any, Callable


# ── PythonFunc ────────────────────────────────────────────────────────────────

class PythonFunc:
    """Call any Python function. Output is str(result)."""

    def __init__(self, fn: Callable, name: str = ""):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "fn")

    def __call__(self, input: Any) -> str:
        if isinstance(input, dict):
            result = self.fn(**input)
        else:
            result = self.fn(input)
        return str(result) if result is not None else ""


# ── CLI ───────────────────────────────────────────────────────────────────────

class CLI:
    """
    Run a subprocess. input is interpolated into the command via {input}.

    Example:
        CLI("python ranker.py --query {input}")
    """

    def __init__(self, command_template: str, timeout: int = 60, cwd: str | None = None):
        self.command_template = command_template
        self.timeout = timeout
        self.cwd = cwd

    def __call__(self, input: Any) -> str:
        input_str = json.dumps(input) if isinstance(input, dict) else str(input)
        command = self.command_template.replace("{input}", input_str)
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.cwd,
        )
        output = result.stdout.strip()
        if result.returncode != 0 and not output:
            output = result.stderr.strip() or f"exit code {result.returncode}"
        return output


# ── HTTPEndpoint ──────────────────────────────────────────────────────────────

class HTTPEndpoint:
    """
    POST input as JSON to a REST endpoint, return response body as str.

    If input_key is set, wraps input as {input_key: input} before posting.
    """

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: dict | None = None,
        input_key: str = "",
        timeout: int = 30,
    ):
        self.url = url
        self.method = method.upper()
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.input_key = input_key
        self.timeout = timeout

    def __call__(self, input: Any) -> str:
        if self.input_key:
            body = {self.input_key: input}
        elif isinstance(input, dict):
            body = input
        else:
            body = {"input": input}

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.url, data=data, headers=self.headers, method=self.method
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        # try to pretty-print JSON, else return raw
        try:
            return json.dumps(json.loads(raw), indent=2)
        except Exception:
            return raw


# ── MCPTool ───────────────────────────────────────────────────────────────────

class MCPTool:
    """
    Call a tool on a running MCP server via its HTTP bridge (mcp-bridge).

    Assumes mcp-bridge is running and exposes tools at:
      POST http://localhost:{port}/tools/{tool_name}

    input should be a dict of tool params, or a string mapped to a single param.
    """

    def __init__(
        self,
        tool_name: str,
        input_param: str = "",
        port: int = 8000,
        token: str = "",
        timeout: int = 30,
    ):
        self.tool_name = tool_name
        self.input_param = input_param
        self.port = port
        self.token = token
        self.timeout = timeout

    def __call__(self, input: Any) -> str:
        if self.input_param and not isinstance(input, dict):
            params = {self.input_param: input}
        elif isinstance(input, dict):
            params = input
        else:
            params = {"input": input}

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        url = f"http://localhost:{self.port}/tools/{self.tool_name}"
        data = json.dumps(params).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        try:
            return json.dumps(json.loads(raw), indent=2)
        except Exception:
            return raw


# ── Prompt ────────────────────────────────────────────────────────────────────

class Prompt:
    """
    Send a prompt to an Ollama model and return the response text.

    template: use {input} as placeholder for the test case input.
    system: optional system prompt.

    Requires Ollama running at localhost:11434.
    """

    def __init__(
        self,
        template: str,
        model: str = "llama3.2",
        system: str = "",
        timeout: int = 120,
    ):
        self.template = template
        self.model = model
        self.system = system
        self.timeout = timeout

    def __call__(self, input: Any) -> str:
        input_str = json.dumps(input) if isinstance(input, dict) else str(input)
        prompt = self.template.replace("{input}", input_str)

        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"].strip()


# ── AgentChain ────────────────────────────────────────────────────────────────

class AgentChain:
    """
    Run a sequence of executors in order. Each executor receives the output
    of the previous one as its input. The final output is returned.

    Use this to evaluate multi-step agent pipelines end-to-end.

    Example:
        chain = AgentChain([
            PythonFunc(parse_jd),          # step 1: parse the JD
            PythonFunc(score_fit),         # step 2: score fit against profile
            CLI("python format.py --in {input}"),  # step 3: format output
        ])
    """

    def __init__(self, steps: list, stop_on_error: bool = True):
        """
        steps: list of executor callables, run in order.
        stop_on_error: if True, raises on any step error (default True).
        """
        if not steps:
            raise ValueError("AgentChain requires at least one step")
        self.steps = steps
        self.stop_on_error = stop_on_error

    def __call__(self, input: Any) -> str:
        current = input
        for i, step in enumerate(self.steps):
            try:
                current = step(current)
            except Exception as e:
                if self.stop_on_error:
                    raise RuntimeError(f"AgentChain step {i} failed: {e}") from e
                # continue with whatever current is
        return str(current) if current is not None else ""


# ── Utilities ─────────────────────────────────────────────────────────────────

def check_ollama(host: str = "localhost", port: int = 11434) -> bool:
    """Return True if Ollama is reachable at host:port."""
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def require_ollama(host: str = "localhost", port: int = 11434) -> None:
    """Raise a clear RuntimeError if Ollama is not reachable."""
    if not check_ollama(host, port):
        raise RuntimeError(
            f"Ollama is not running at {host}:{port}. "
            f"Start it with: ollama serve"
        )
