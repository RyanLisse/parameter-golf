#!/usr/bin/env python3
"""Symphony — orchestration service for parameter-golf experiments.

Monitors Linear issues, spawns coding agents in isolated git worktree
workspaces, and manages the full lifecycle with retries and reconciliation.

Usage:
    python scripts/symphony.py                         # Run with defaults
    python scripts/symphony.py --workflow WORKFLOW.md   # Custom workflow
    python scripts/symphony.py --dry-run                # Log what would happen
    python scripts/symphony.py --once                   # Single poll tick, exit
    python scripts/symphony.py --port 8080              # Enable HTTP status endpoint
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging — structured JSON to stderr
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for attr in ("issue_id", "session_id", "workspace", "attempt"):
            val = getattr(record, attr, None)
            if val is not None:
                entry[attr] = val
        if record.exc_info and record.exc_info[1]:
            entry["error"] = str(record.exc_info[1])
        return json.dumps(entry)


logger = logging.getLogger("symphony")
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(JsonFormatter())
logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# .env auto-loader
# ---------------------------------------------------------------------------

def _load_dotenv(path: Optional[str] = None) -> None:
    """Load .env file into os.environ (no external deps)."""
    candidates = [path] if path else [
        str(Path.cwd() / ".env"),
        str(Path(__file__).resolve().parent.parent / ".env"),
    ]
    for p in candidates:
        if p and Path(p).is_file():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
            logger.debug(f"Loaded env from {p}")
            return

_load_dotenv()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SymphonyError(Exception):
    """Base exception for Symphony."""

class InvalidTransitionError(SymphonyError):
    """Raised when a state transition is not allowed."""

class WorkspaceSafetyError(SymphonyError):
    """Raised when a workspace path violates safety invariants."""


# ---------------------------------------------------------------------------
# Issue state machine
# ---------------------------------------------------------------------------

class IssueState(enum.Enum):
    UNCLAIMED = "unclaimed"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    RETRY_QUEUED = "retry_queued"
    RELEASED = "released"


# Valid transitions: from_state -> set of allowed to_states
_TRANSITIONS: Dict[IssueState, set] = {
    IssueState.UNCLAIMED: {IssueState.CLAIMED},
    IssueState.CLAIMED: {IssueState.RUNNING},
    IssueState.RUNNING: {IssueState.SUCCEEDED, IssueState.FAILED, IssueState.TIMED_OUT},
    IssueState.SUCCEEDED: {IssueState.RELEASED},
    IssueState.FAILED: {IssueState.RETRY_QUEUED, IssueState.RELEASED},
    IssueState.TIMED_OUT: {IssueState.RETRY_QUEUED, IssueState.RELEASED},
    IssueState.RETRY_QUEUED: {IssueState.CLAIMED},
    IssueState.RELEASED: set(),
}


class IssueStateMachine:
    """Tracks per-issue state with validated transitions."""

    def __init__(self) -> None:
        self._states: Dict[str, IssueState] = {}

    def get_state(self, issue_id: str) -> IssueState:
        return self._states.get(issue_id, IssueState.UNCLAIMED)

    def transition(self, issue_id: str, from_state: IssueState, to_state: IssueState) -> None:
        current = self.get_state(issue_id)
        if current != from_state:
            raise InvalidTransitionError(
                f"{issue_id}: expected {from_state.value}, got {current.value}"
            )
        allowed = _TRANSITIONS.get(from_state, set())
        if to_state not in allowed:
            raise InvalidTransitionError(
                f"{issue_id}: {from_state.value} -> {to_state.value} not allowed"
            )
        self._states[issue_id] = to_state
        logger.info(
            f"State transition: {from_state.value} -> {to_state.value}",
            extra={"issue_id": issue_id},
        )

    def all_states(self) -> Dict[str, str]:
        return {k: v.value for k, v in self._states.items()}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sanitize_workspace_key(key: str) -> str:
    """Only [A-Za-z0-9._-] allowed; others become _."""
    return re.sub(r"[^A-Za-z0-9._\-]", "_", key)


def retry_backoff_ms(attempt: int, max_backoff_ms: int = 300_000) -> int:
    """Exponential backoff: delay = min(10000 * 2^(attempt-1), max_backoff_ms).

    For attempt <= 0, returns the base delay (10000).
    """
    if attempt < 1:
        attempt = 1
    delay = 10_000 * (2 ** (attempt - 1))
    return min(delay, max_backoff_ms)


def matches_filter(identifier: str, pattern: str) -> bool:
    """Simple prefix-glob matching for issue identifiers.

    Supports patterns like 'EXP-*' (prefix match) or '*' (match all)
    or exact match.
    """
    if pattern == "*":
        return True
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return identifier.startswith(prefix)
    return identifier == pattern


def sort_issues_by_priority(issues: List[Dict]) -> List[Dict]:
    """Sort issues by priority (ascending, lower=higher priority) then createdAt."""
    return sorted(issues, key=lambda i: (i.get("priority", 999), i.get("createdAt", "")))


def _expand_env(value: str) -> str:
    """Expand $VAR_NAME references in a string value."""
    if not isinstance(value, str) or "$" not in value:
        return value
    # Match $VAR_NAME patterns
    def replacer(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", replacer, value)


# ---------------------------------------------------------------------------
# WorkflowLoader
# ---------------------------------------------------------------------------

class WorkflowLoader:
    """Parse WORKFLOW.md: YAML frontmatter in ```yaml fenced block + prompt body."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.config: Dict[str, Any] = {}
        self.prompt_body: str = ""

    def load(self) -> Dict[str, Any]:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Workflow file not found: {self.path}")

        text = p.read_text(encoding="utf-8")
        self.config = self._parse_frontmatter(text)
        self.prompt_body = self._extract_body(text)
        return self.config

    @staticmethod
    def _parse_frontmatter(text: str) -> Dict[str, Any]:
        """Extract YAML from fenced ```yaml block containing --- delimiters."""
        # Look for ```yaml ... ``` blocks
        yaml_block_re = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL)
        m = yaml_block_re.search(text)
        if not m:
            return {}

        block = m.group(1).strip()
        # Strip --- delimiters
        lines = block.split("\n")
        content_lines = [l for l in lines if l.strip() != "---"]
        return WorkflowLoader._simple_yaml_parse("\n".join(content_lines))

    @staticmethod
    def _simple_yaml_parse(text: str) -> Dict[str, Any]:
        """Minimal YAML-like parser for flat key: value pairs.

        Handles strings (with or without quotes), integers, and booleans.
        No external YAML dependency required.
        """
        result: Dict[str, Any] = {}
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # Strip quotes
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            # Type coercion
            if val.lower() in ("true", "yes"):
                result[key] = True
            elif val.lower() in ("false", "no"):
                result[key] = False
            elif val.isdigit():
                result[key] = int(val)
            else:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
        return result

    @staticmethod
    def _extract_body(text: str) -> str:
        """Everything after the first ```yaml...``` block is the prompt body."""
        yaml_block_re = re.compile(r"```yaml\s*\n.*?```", re.DOTALL)
        m = yaml_block_re.search(text)
        if not m:
            return text
        return text[m.end():].strip()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    """Typed configuration with defaults and env-var expansion."""

    def __init__(self, workflow_data: Dict[str, Any]) -> None:
        self.name: str = workflow_data.get("name", "symphony")
        self.trigger: str = workflow_data.get("trigger", "linear_issue")
        self.issue_filter: str = workflow_data.get("issue_filter", "*")
        self.max_concurrent: int = int(workflow_data.get("max_concurrent", 4))
        self.poll_interval: int = int(workflow_data.get("poll_interval", 30))
        self.timeout: int = int(workflow_data.get("timeout", 900))
        self.hook_timeout: int = int(workflow_data.get("hook_timeout", 60))
        self.max_retries: int = int(workflow_data.get("max_retries", 3))
        self.max_backoff_ms: int = int(workflow_data.get("max_backoff_ms", 300_000))

        # Paths
        default_ws_root = str(Path.cwd() / ".symphony" / "workspaces")
        self.workspace_root: str = workflow_data.get("workspace_root", default_ws_root)

        # Agent command
        self.agent_command: str = workflow_data.get(
            "codex_command",
            workflow_data.get("agent_command", "claude --print --output-format stream-json"),
        )

        # Linear
        raw_key = workflow_data.get("linear_api_key", "$LINEAR_API_KEY")
        self.linear_api_key: str = _expand_env(str(raw_key))
        self.linear_project_slug: str = workflow_data.get("linear_project_slug", "")
        self.linear_team_key: str = workflow_data.get("linear_team_key", "")
        self.linear_mode: str = workflow_data.get("linear_mode", "auto")  # "api", "mcp", "auto"

        # Hooks
        self.hooks: Dict[str, str] = workflow_data.get("hooks", {})


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Render prompt template from WORKFLOW.md body with issue context."""

    @staticmethod
    def render(
        template: str,
        issue: Dict[str, Any],
        workspace_path: str,
        branch_name: str,
    ) -> str:
        replacements = {
            "{{issue_title}}": str(issue.get("title", "")),
            "{{issue_description}}": str(issue.get("description", "")),
            "{{issue_labels}}": ", ".join(issue.get("labels", [])),
            "{{issue_priority}}": str(issue.get("priority", "")),
            "{{issue_identifier}}": str(issue.get("identifier", "")),
            "{{workspace_path}}": workspace_path,
            "{{branch_name}}": branch_name,
        }
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        return result


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------

class WorkspaceManager:
    """Manage git worktree workspaces per issue."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.root = os.path.abspath(config.workspace_root)

    @staticmethod
    def compute_workspace_path(root: str, key: str) -> str:
        """Compute and validate workspace path for a key.

        Safety: path must be inside root. No traversal allowed.
        Checks the RAW key first for traversal attempts, then sanitizes.
        """
        # Check raw key for obvious attacks before sanitization
        if key.startswith("/") or ".." in key:
            raise WorkspaceSafetyError(f"Invalid workspace key: {key!r}")

        sanitized = sanitize_workspace_key(key)
        if not sanitized:
            raise WorkspaceSafetyError(f"Invalid workspace key: {key!r}")

        candidate = os.path.normpath(os.path.join(os.path.abspath(root), sanitized))
        abs_root = os.path.abspath(root)

        if not candidate.startswith(abs_root + os.sep) and candidate != abs_root:
            raise WorkspaceSafetyError(
                f"Workspace path escapes root: {candidate} not under {abs_root}"
            )
        return candidate

    def create(self, issue_id: str, branch_name: str) -> str:
        """Create a git worktree for the given issue. Returns workspace path."""
        ws_path = self.compute_workspace_path(self.root, issue_id)
        os.makedirs(self.root, exist_ok=True)

        logger.info(
            f"Creating worktree at {ws_path} on branch {branch_name}",
            extra={"issue_id": issue_id, "workspace": ws_path},
        )
        try:
            subprocess.run(
                ["git", "worktree", "add", ws_path, "-b", branch_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as exc:
            # Branch may already exist, try without -b
            logger.warning(f"Worktree create with new branch failed, trying existing: {exc.stderr}")
            subprocess.run(
                ["git", "worktree", "add", ws_path, branch_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

        self._run_hook("after_create", ws_path, issue_id)
        return ws_path

    def remove(self, issue_id: str) -> None:
        """Remove a git worktree for the given issue."""
        ws_path = self.compute_workspace_path(self.root, issue_id)
        self._run_hook("before_remove", ws_path, issue_id)

        if os.path.exists(ws_path):
            logger.info(f"Removing worktree at {ws_path}", extra={"issue_id": issue_id})
            subprocess.run(
                ["git", "worktree", "remove", ws_path, "--force"],
                capture_output=True, text=True, timeout=60,
            )
            # Fallback if worktree remove fails
            if os.path.exists(ws_path):
                shutil.rmtree(ws_path, ignore_errors=True)

    def _run_hook(self, hook_name: str, ws_path: str, issue_id: str) -> None:
        """Run a lifecycle hook if configured."""
        cmd = self.config.hooks.get(hook_name)
        if not cmd:
            return
        logger.info(f"Running hook {hook_name}", extra={"issue_id": issue_id})
        env = os.environ.copy()
        env["SYMPHONY_WORKSPACE"] = ws_path
        env["SYMPHONY_ISSUE_ID"] = issue_id
        try:
            subprocess.run(
                ["bash", "-c", cmd],
                cwd=ws_path if os.path.isdir(ws_path) else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.hook_timeout,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            logger.warning(f"Hook {hook_name} failed: {exc}", extra={"issue_id": issue_id})


# ---------------------------------------------------------------------------
# IssueTrackerClient (Linear)
# ---------------------------------------------------------------------------

class LinearClient:
    """GraphQL client for Linear issue tracker."""

    API_URL = "https://api.linear.app/graphql"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.api_key = config.linear_api_key

    def _query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Execute a GraphQL query against Linear API."""
        try:
            import requests
        except ImportError:
            logger.error("requests library required for Linear integration")
            raise

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        resp = requests.post(self.API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            logger.error(f"Linear API errors: {data['errors']}")
        return data.get("data", {})

    def fetch_active_issues(self) -> List[Dict]:
        """Fetch issues in active states from Linear."""
        query = """
        {
          issues(first: 50) {
            nodes {
              id
              identifier
              title
              description
              priority
              createdAt
              labels { nodes { name } }
              state { name type }
            }
          }
        }
        """
        data = self._query(query)
        issues_raw = data.get("issues", {}).get("nodes", [])

        issues = []
        for raw in issues_raw:
            issue = {
                "id": raw["id"],
                "identifier": raw["identifier"],
                "title": raw["title"],
                "description": raw.get("description", ""),
                "priority": raw.get("priority", 4),
                "createdAt": raw.get("createdAt", ""),
                "labels": [l["name"] for l in raw.get("labels", {}).get("nodes", [])],
                "state": raw.get("state", {}).get("name", ""),
                "state_type": raw.get("state", {}).get("type", ""),
            }
            # Apply issue filter
            if matches_filter(issue["identifier"], self.config.issue_filter):
                issues.append(issue)
        return issues

    def update_issue_state(self, issue_id: str, state_id: str) -> None:
        """Update an issue's state in Linear."""
        mutation = """
        mutation($issueId: String!, $stateId: String!) {
          issueUpdate(id: $issueId, input: { stateId: $stateId }) {
            success
          }
        }
        """
        self._query(mutation, {"issueId": issue_id, "stateId": state_id})

    def add_comment(self, issue_id: str, body: str) -> None:
        """Add a comment to an issue."""
        mutation = """
        mutation($issueId: String!, $body: String!) {
          commentCreate(input: { issueId: $issueId, body: $body }) {
            success
          }
        }
        """
        self._query(mutation, {"issueId": issue_id, "body": body})


class MCPLinearClient:
    """Linear client that uses Claude Code MCP tools via subprocess.

    Calls `claude` CLI with MCP tool invocations so Symphony can piggyback on
    the already-authenticated OAuth connection without needing a raw API key.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def _mcp_call(self, tool: str, args_json: str) -> Dict:
        """Call a Linear MCP tool via claude CLI."""
        prompt = f'Use the {tool} MCP tool with these parameters: {args_json}. Return ONLY the raw JSON result, no commentary.'
        cmd = ["claude", "--print", "--output-format", "json", "-p", prompt]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"MCP call failed: {result.stderr[:200]}")
                return {}
            # Parse the response — claude --print returns the text
            text = result.stdout.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"MCP call error: {e}")
            return {}

    def fetch_active_issues(self) -> List[Dict]:
        """Fetch issues via MCP list_issues tool."""
        cmd = [
            "claude", "--print", "-p",
            f'Call the mcp__claude_ai_Linear__list_issues tool with team="Rjct-studio" and state="active" and limit=50. '
            f'Return ONLY a JSON array of objects with fields: id, identifier, title, description, priority, createdAt, labels, state. No commentary.'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"MCP fetch failed: {result.stderr[:200]}")
                return []

            text = result.stdout.strip()
            # Try to extract JSON from response
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    raw_issues = data
                elif isinstance(data, dict) and "issues" in data:
                    raw_issues = data["issues"]
                else:
                    raw_issues = []
            except json.JSONDecodeError:
                logger.warning("Could not parse MCP response as JSON")
                return []

            issues = []
            for raw in raw_issues:
                issue = {
                    "id": raw.get("id", ""),
                    "identifier": raw.get("id", raw.get("identifier", "")),
                    "title": raw.get("title", ""),
                    "description": raw.get("description", ""),
                    "priority": raw.get("priority", {}).get("value", 4) if isinstance(raw.get("priority"), dict) else raw.get("priority", 4),
                    "createdAt": raw.get("createdAt", ""),
                    "labels": [l.get("name", l) if isinstance(l, dict) else l for l in raw.get("labels", [])],
                    "state": raw.get("status", raw.get("state", "")),
                    "state_type": raw.get("state_type", ""),
                }
                if matches_filter(issue["identifier"], self.config.issue_filter):
                    issues.append(issue)
            return issues
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"MCP fetch error: {e}")
            return []

    def update_issue_state(self, issue_id: str, state_id: str) -> None:
        """Update issue state via MCP."""
        cmd = [
            "claude", "--print", "-p",
            f'Call mcp__claude_ai_Linear__save_issue with id="{issue_id}" and state="{state_id}". Return the result.'
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"MCP update error: {e}")

    def add_comment(self, issue_id: str, body: str) -> None:
        """Add comment via MCP."""
        escaped_body = body.replace('"', '\\"').replace('\n', '\\n')
        cmd = [
            "claude", "--print", "-p",
            f'Call mcp__claude_ai_Linear__save_comment with issueId="{issue_id}" and body="{escaped_body}". Return the result.'
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"MCP comment error: {e}")


def build_linear_client(config: Config) -> "LinearClient | MCPLinearClient":
    """Build the appropriate Linear client based on config.

    - "api": Direct GraphQL with LINEAR_API_KEY (standalone daemon)
    - "mcp": Claude Code MCP tools (no API key needed)
    - "auto": Try API key first, fall back to MCP
    """
    mode = config.linear_mode
    if mode == "api" or (mode == "auto" and config.linear_api_key):
        logger.info("Using direct Linear API client")
        return LinearClient(config)
    logger.info("Using MCP Linear client (via claude CLI)")
    return MCPLinearClient(config)


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

class AgentRunner:
    """Launch and manage coding agent subprocesses."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._processes: Dict[str, subprocess.Popen] = {}

    def start(
        self,
        issue_id: str,
        workspace_path: str,
        prompt: str,
        session_id: str,
    ) -> subprocess.Popen:
        """Launch agent process in workspace directory."""
        logger.info(
            f"Launching agent: {self.config.agent_command}",
            extra={"issue_id": issue_id, "session_id": session_id, "workspace": workspace_path},
        )

        # Write prompt to a temp file in the workspace
        prompt_path = os.path.join(workspace_path, ".symphony_prompt.md")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        cmd = f"{self.config.agent_command} < {prompt_path}"
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=workspace_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "SYMPHONY_SESSION_ID": session_id, "SYMPHONY_ISSUE_ID": issue_id},
        )
        self._processes[issue_id] = proc
        return proc

    def is_alive(self, issue_id: str) -> bool:
        proc = self._processes.get(issue_id)
        if proc is None:
            return False
        return proc.poll() is None

    def cancel(self, issue_id: str) -> None:
        proc = self._processes.get(issue_id)
        if proc and proc.poll() is None:
            logger.info(f"Cancelling agent process", extra={"issue_id": issue_id})
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._processes.pop(issue_id, None)

    def get_result(self, issue_id: str) -> Tuple[int, str, str]:
        """Get return code, stdout, stderr for a finished process."""
        proc = self._processes.get(issue_id)
        if proc is None:
            return (-1, "", "No process found")
        stdout, stderr = proc.communicate(timeout=5)
        returncode = proc.returncode
        self._processes.pop(issue_id, None)
        return (returncode, stdout or "", stderr or "")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Worker:
    """Tracks a running agent for an issue."""

    def __init__(self, issue: Dict, session_id: str, workspace_path: str,
                 branch_name: str, process: subprocess.Popen, attempt: int) -> None:
        self.issue = issue
        self.session_id = session_id
        self.workspace_path = workspace_path
        self.branch_name = branch_name
        self.process = process
        self.attempt = attempt
        self.started_at = time.time()


class Orchestrator:
    """Main poll loop — claims issues, runs agents, handles retries."""

    def __init__(
        self,
        config: Config,
        workflow_loader: WorkflowLoader,
        tracker: Optional[LinearClient],
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.workflow_loader = workflow_loader
        self.tracker = tracker
        self.dry_run = dry_run
        self.state_machine = IssueStateMachine()
        self.workspace_mgr = WorkspaceManager(config)
        self.agent_runner = AgentRunner(config)
        self.workers: Dict[str, Worker] = {}
        self.retry_counts: Dict[str, int] = {}
        self.retry_after: Dict[str, float] = {}  # issue_id -> earliest retry timestamp
        self._shutdown = threading.Event()

    def run_once(self) -> None:
        """Single poll tick: fetch issues, dispatch workers, reconcile."""
        logger.info("Poll tick starting")
        sys.stderr.flush()

        # Reload workflow (dynamic reload)
        try:
            self.workflow_loader.load()
        except Exception as exc:
            logger.warning(f"Failed to reload workflow: {exc}")

        # Reconcile: check running workers are alive
        self._reconcile()

        # Fetch issues
        issues = self._fetch_issues()
        if issues is None:
            logger.warning("Skipping tick: failed to fetch issues")
            return

        # Sort by priority
        issues = sort_issues_by_priority(issues)

        # Dispatch
        slots_available = self.config.max_concurrent - len(self.workers)
        for issue in issues:
            if slots_available <= 0:
                break
            issue_id = issue["identifier"]

            # Skip if already being processed
            current_state = self.state_machine.get_state(issue_id)
            if current_state in (IssueState.RUNNING, IssueState.CLAIMED, IssueState.SUCCEEDED):
                continue

            # Check retry backoff
            if issue_id in self.retry_after and time.time() < self.retry_after[issue_id]:
                continue

            # Check max retries
            if self.retry_counts.get(issue_id, 0) >= self.config.max_retries:
                logger.info(f"Max retries reached, skipping", extra={"issue_id": issue_id})
                continue

            if self.dry_run:
                logger.info(f"[DRY-RUN] Would dispatch: {issue['title']}", extra={"issue_id": issue_id})
                continue

            self._dispatch(issue)
            slots_available -= 1

    def _fetch_issues(self) -> Optional[List[Dict]]:
        if self.tracker is None:
            logger.debug("No tracker configured, returning empty issue list")
            return []
        try:
            return self.tracker.fetch_active_issues()
        except Exception as exc:
            logger.error(f"Failed to fetch issues: {exc}")
            return None

    def _dispatch(self, issue: Dict) -> None:
        issue_id = issue["identifier"]
        attempt = self.retry_counts.get(issue_id, 0) + 1
        session_id = f"{issue_id}-{attempt}-{int(time.time())}"

        logger.info(
            f"Dispatching issue: {issue['title']}",
            extra={"issue_id": issue_id, "session_id": session_id, "attempt": attempt},
        )

        try:
            # Claim
            self.state_machine.transition(issue_id, self.state_machine.get_state(issue_id), IssueState.CLAIMED)

            # Create workspace
            branch_name = f"symphony/{sanitize_workspace_key(issue_id)}"
            ws_path = self.workspace_mgr.create(issue_id, branch_name)

            # Build prompt
            prompt = PromptBuilder.render(
                self.workflow_loader.prompt_body,
                issue,
                ws_path,
                branch_name,
            )

            # Transition to running
            self.state_machine.transition(issue_id, IssueState.CLAIMED, IssueState.RUNNING)

            # Launch agent
            self.workspace_mgr._run_hook("before_run", ws_path, issue_id)
            proc = self.agent_runner.start(issue_id, ws_path, prompt, session_id)

            self.workers[issue_id] = Worker(
                issue=issue,
                session_id=session_id,
                workspace_path=ws_path,
                branch_name=branch_name,
                process=proc,
                attempt=attempt,
            )
        except Exception as exc:
            logger.error(f"Dispatch failed: {exc}", extra={"issue_id": issue_id})
            # Reset state for retry
            try:
                self.state_machine.transition(issue_id, IssueState.RUNNING, IssueState.FAILED)
            except InvalidTransitionError:
                try:
                    self.state_machine.transition(issue_id, IssueState.CLAIMED, IssueState.RUNNING)
                    self.state_machine.transition(issue_id, IssueState.RUNNING, IssueState.FAILED)
                except InvalidTransitionError:
                    pass
            self._schedule_retry(issue_id, attempt)

    def _reconcile(self) -> None:
        """Check running workers, handle completion or timeout."""
        finished = []
        for issue_id, worker in self.workers.items():
            elapsed = time.time() - worker.started_at

            if not self.agent_runner.is_alive(issue_id):
                # Process finished
                returncode, stdout, stderr = self.agent_runner.get_result(issue_id)
                if returncode == 0:
                    logger.info(
                        f"Agent succeeded",
                        extra={"issue_id": issue_id, "session_id": worker.session_id},
                    )
                    self.state_machine.transition(issue_id, IssueState.RUNNING, IssueState.SUCCEEDED)
                    self.workspace_mgr._run_hook("after_run", worker.workspace_path, issue_id)
                else:
                    logger.warning(
                        f"Agent failed (exit {returncode})",
                        extra={"issue_id": issue_id, "session_id": worker.session_id},
                    )
                    self.state_machine.transition(issue_id, IssueState.RUNNING, IssueState.FAILED)
                    self._schedule_retry(issue_id, worker.attempt)
                finished.append(issue_id)

            elif elapsed > self.config.timeout:
                logger.warning(
                    f"Agent timed out after {elapsed:.0f}s",
                    extra={"issue_id": issue_id, "session_id": worker.session_id},
                )
                self.agent_runner.cancel(issue_id)
                self.state_machine.transition(issue_id, IssueState.RUNNING, IssueState.TIMED_OUT)
                self._schedule_retry(issue_id, worker.attempt)
                finished.append(issue_id)

        for issue_id in finished:
            self.workers.pop(issue_id, None)

    def _schedule_retry(self, issue_id: str, attempt: int) -> None:
        """Schedule a retry with exponential backoff."""
        self.retry_counts[issue_id] = attempt
        if attempt < self.config.max_retries:
            delay_ms = retry_backoff_ms(attempt, self.config.max_backoff_ms)
            self.retry_after[issue_id] = time.time() + delay_ms / 1000.0
            current = self.state_machine.get_state(issue_id)
            if current in (IssueState.FAILED, IssueState.TIMED_OUT):
                self.state_machine.transition(issue_id, current, IssueState.RETRY_QUEUED)
            logger.info(
                f"Retry scheduled in {delay_ms}ms (attempt {attempt})",
                extra={"issue_id": issue_id, "attempt": attempt},
            )

    def run_loop(self) -> None:
        """Main poll loop until shutdown."""
        logger.info(f"Symphony starting — polling every {self.config.poll_interval}s")

        def handle_signal(signum: int, frame: Any) -> None:
            logger.info("Shutdown signal received")
            self._shutdown.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while not self._shutdown.is_set():
            try:
                self.run_once()
            except Exception as exc:
                logger.error(f"Poll tick error: {exc}", exc_info=True)
            self._shutdown.wait(timeout=self.config.poll_interval)

        # Cleanup
        logger.info("Shutting down — cancelling workers")
        for issue_id in list(self.workers.keys()):
            self.agent_runner.cancel(issue_id)
            self.workspace_mgr.remove(issue_id)
        logger.info("Symphony stopped")

    def get_status(self) -> Dict[str, Any]:
        """Return current orchestrator state for the status endpoint."""
        return {
            "name": self.config.name,
            "workers": {
                iid: {
                    "session_id": w.session_id,
                    "issue_title": w.issue["title"],
                    "attempt": w.attempt,
                    "elapsed_s": round(time.time() - w.started_at, 1),
                    "alive": self.agent_runner.is_alive(iid),
                }
                for iid, w in self.workers.items()
            },
            "states": self.state_machine.all_states(),
            "retry_counts": dict(self.retry_counts),
            "config": {
                "max_concurrent": self.config.max_concurrent,
                "poll_interval": self.config.poll_interval,
                "timeout": self.config.timeout,
                "issue_filter": self.config.issue_filter,
            },
        }


# ---------------------------------------------------------------------------
# HTTP status endpoint
# ---------------------------------------------------------------------------

class StatusHandler(BaseHTTPRequestHandler):
    orchestrator: Optional[Orchestrator] = None

    def do_GET(self) -> None:
        if self.path == "/api/v1/state":
            if self.orchestrator:
                body = json.dumps(self.orchestrator.get_status(), indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            else:
                self.send_response(503)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default HTTP logging
        pass


def start_status_server(port: int, orchestrator: Orchestrator) -> HTTPServer:
    StatusHandler.orchestrator = orchestrator
    server = HTTPServer(("", port), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Status endpoint running on http://localhost:{port}/api/v1/state")
    return server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symphony — orchestration service for parameter-golf experiments",
    )
    parser.add_argument(
        "--workflow", default="WORKFLOW.md",
        help="Path to workflow definition file (default: WORKFLOW.md)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log what would happen without executing",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single poll tick then exit",
    )
    parser.add_argument(
        "--port", type=int, default=0,
        help="Enable HTTP status endpoint on this port",
    )
    parser.add_argument(
        "--mcp", action="store_true",
        help="Use MCP Linear client via claude CLI (no API key needed)",
    )
    args = parser.parse_args()

    # Load workflow
    loader = WorkflowLoader(args.workflow)
    try:
        wf_config = loader.load()
    except FileNotFoundError:
        logger.error(f"Workflow file not found: {args.workflow}")
        sys.exit(1)

    # Override linear_mode from CLI flag
    if args.mcp:
        wf_config["linear_mode"] = "mcp"

    config = Config(wf_config)

    # Set up tracker — supports both direct API and MCP modes
    tracker = None
    if config.trigger == "linear_issue":
        if config.linear_api_key or config.linear_mode == "mcp":
            tracker = build_linear_client(config)
        else:
            logger.warning("No Linear API key and MCP not configured — running without tracker")

    # Create orchestrator
    orchestrator = Orchestrator(
        config=config,
        workflow_loader=loader,
        tracker=tracker,
        dry_run=args.dry_run,
    )

    # Start status server if requested
    if args.port:
        start_status_server(args.port, orchestrator)

    # Run
    if args.once:
        orchestrator.run_once()
    else:
        orchestrator.run_loop()


if __name__ == "__main__":
    main()
