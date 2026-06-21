"""
agents.py
Three parallel LangGraph-based code review agents powered by Groq.

Agents:
  - SecurityAgent      : detects secrets, injection risks, unvalidated inputs
  - PerformanceAgent   : detects N+1 queries, blocking I/O, memory leaks
  - CodeQualityAgent   : detects code smells, naming, dead code, error handling

Rate-limit strategy:
  - tenacity exponential backoff on Groq 429 / 503 errors
  - 2-second stagger between agent starts
  - diff truncation to ~8 000 chars to stay inside TPM budget

Model: llama3-70b-8192  (stable, high quality)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, field_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME: str = "groq/llama-3.3-70b-versatile"   # Dough model formatting
MAX_DIFF_CHARS: int = 8_000                  # ~2 000 tokens — stays in TPM budget
AGENT_STAGGER_SEC: float = 2.0              # seconds between agent starts


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------

class Issue(BaseModel):
    line_number: Optional[int] = None
    file: Optional[str] = None
    severity: Literal["critical", "warning", "info"]
    description: str

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v: str) -> str:
        v = str(v).lower().strip()
        if v not in {"critical", "warning", "info"}:
            return "info"
        return v


class AgentResult(BaseModel):
    agent_name: str
    issues: List[Issue] = []
    summary: str

    def dict(self, **kwargs):  # ensure JSON-serialisable output
        return super().model_dump(**kwargs)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    diff: str
    metadata: Dict[str, Any]
    agent_type: str       # "security" | "performance" | "code_quality"
    raw_response: str
    result: Optional[AgentResult]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SHARED_FORMAT = """
Return ONLY a single valid JSON object — no prose, no markdown fences — in this exact shape:
{
  "agent_name": "<agent display name>",
  "issues": [
    {
      "line_number": <integer or null>,
      "file": "<filename string or null>",
      "severity": "<critical|warning|info>",
      "description": "<concise description + fix suggestion>"
    }
  ],
  "summary": "<2-3 sentence overall assessment>"
}
If no issues are found, return an empty issues array and a positive summary.
"""

SYSTEM_PROMPTS: Dict[str, str] = {
    "security": f"""You are a senior application-security engineer conducting a code review.
Examine the provided git diff for:
• Hardcoded secrets, API keys, passwords, or tokens
• SQL / NoSQL / command injection vulnerabilities
• Cross-site scripting (XSS) or template injection
• Unvalidated or unsanitised user inputs
• Insecure dependencies or dangerous imports (eval, exec, pickle, etc.)
• Authentication or authorisation bypasses
• Sensitive data written to logs or exposed in responses
• Weak or broken cryptography

{_SHARED_FORMAT}""",

    "performance": f"""You are a senior performance engineer conducting a code review.
Examine the provided git diff for:
• N+1 database query patterns (ORM loops that trigger individual queries)
• Unnecessary or deeply nested loops (O(n²) or worse)
• Blocking I/O operations inside async functions (time.sleep, sync DB calls)
• Unbound memory growth (growing lists/dicts inside loops without eviction)
• Missing pagination on queries that fetch all rows
• Redundant computations repeated inside hot paths
• Large object serialisation / deserialisation in tight loops
• Missing caching for expensive deterministic computations

{_SHARED_FORMAT}""",

    "code_quality": f"""You are a senior software engineer conducting a code quality review.
Examine the provided git diff for:
• Code smells: long methods, god classes, feature envy, duplicate logic
• Poor naming: single-letter vars, misleading names, inconsistent casing
• Missing or overly broad exception handling (bare `except:` clauses)
• Dead code: commented-out blocks, unreachable branches, unused imports
• Magic numbers or strings that should be named constants
• Violation of SOLID / DRY / KISS principles
• Missing docstrings or comments on complex / non-obvious logic
• Overly deep nesting that should be refactored with early returns

{_SHARED_FORMAT}""",
}


# ---------------------------------------------------------------------------
# Groq LLM factory
# ---------------------------------------------------------------------------

def _make_llm() -> ChatOpenAI:
    dough_api_key = os.getenv("DOUGH_API_KEY", "").strip()
    if not dough_api_key:
        raise EnvironmentError(
            "DOUGH_API_KEY is not set. Please set the environment variable or use --dough-api-key."
        )
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=dough_api_key,
        base_url="https://dough.id/api/v1",
        default_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        temperature=0.1,
        max_tokens=1024,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Retry-wrapped Groq call (handles 429 rate-limit & transient 503 errors)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_groq(agent_type: str, diff: str) -> str:
    """
    Synchronous Groq call with tenacity retry.
    Retries up to 3 times with 5-60 s exponential backoff.
    """
    llm = _make_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPTS[agent_type]),
        HumanMessage(
            content=(
                f"Review the following git diff and return your JSON analysis:\n\n"
                f"```diff\n{diff}\n```"
            )
        ),
    ]
    response = llm.invoke(messages)
    return str(response.content)


# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------

def _truncate_diff(diff: str) -> str:
    """Trim the diff to MAX_DIFF_CHARS so we stay within the Groq TPM budget."""
    if len(diff) <= MAX_DIFF_CHARS:
        return diff
    truncated = diff[:MAX_DIFF_CHARS]
    return (
        truncated
        + "\n\n[... diff truncated to fit within LLM context window ...]"
    )


# ---------------------------------------------------------------------------
# JSON parser (robust against markdown fences)
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str, agent_type: str) -> AgentResult:
    text = raw.strip()

    # Strip possible markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # Find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
        issues = [Issue(**i) for i in data.get("issues", [])]
        return AgentResult(
            agent_name=data.get("agent_name", f"{agent_type.replace('_', ' ').title()} Agent"),
            issues=issues,
            summary=data.get("summary", "Analysis complete."),
        )
    except (json.JSONDecodeError, Exception) as exc:
        logger.error(
            "[%s] JSON parse failed: %s. Raw (first 400 chars): %s",
            agent_type,
            exc,
            raw[:400],
        )
        return AgentResult(
            agent_name=f"{agent_type.replace('_', ' ').title()} Agent",
            issues=[],
            summary=(
                f"Agent completed but response could not be parsed. "
                f"Raw output (truncated): {raw[:300]}"
            ),
        )


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def _analyze_node(state: AgentState) -> AgentState:
    agent_type = state["agent_type"]
    diff = _truncate_diff(state["diff"])
    logger.info("[%s] Calling Groq (%s chars of diff)…", agent_type, len(diff))

    raw = _call_groq(agent_type, diff)

    logger.info("[%s] Groq call successful (%d chars returned).", agent_type, len(raw))
    return {**state, "raw_response": raw}


def _parse_node(state: AgentState) -> AgentState:
    result = _parse_json_response(state["raw_response"], state["agent_type"])
    return {**state, "result": result}


# ---------------------------------------------------------------------------
# Build & compile the reusable LangGraph
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze", _analyze_node)
    g.add_node("parse", _parse_node)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "parse")
    g.add_edge("parse", END)
    return g.compile()


_GRAPH = _build_graph()   # compiled once at module load


# ---------------------------------------------------------------------------
# Async runner (LangGraph is sync → offload to thread pool)
# ---------------------------------------------------------------------------

async def _run_agent(
    agent_type: str,
    diff: str,
    metadata: Dict[str, Any],
    delay: float = 0.0,
) -> AgentResult:
    if delay > 0:
        await asyncio.sleep(delay)

    initial_state: AgentState = {
        "diff": diff,
        "metadata": metadata,
        "agent_type": agent_type,
        "raw_response": "",
        "result": None,
    }

    loop = asyncio.get_event_loop()
    final_state: AgentState = await loop.run_in_executor(
        None, _GRAPH.invoke, initial_state
    )
    return final_state["result"]


# ---------------------------------------------------------------------------
# Public API: run all three agents in parallel
# ---------------------------------------------------------------------------

async def run_all_agents(
    diff: str, metadata: Dict[str, Any]
) -> tuple[AgentResult, AgentResult, AgentResult]:
    """
    Run Security, Performance, and Code Quality agents in parallel.
    Agents are staggered by AGENT_STAGGER_SEC to reduce Groq rate-limit pressure.
    Returns (security_result, performance_result, code_quality_result).
    """
    results = await asyncio.gather(
        _run_agent("security",     diff, metadata, delay=0.0),
        _run_agent("performance",  diff, metadata, delay=AGENT_STAGGER_SEC),
        _run_agent("code_quality", diff, metadata, delay=AGENT_STAGGER_SEC * 2),
        return_exceptions=True,
    )

    agent_display_names = [
        "Security Agent",
        "Performance Agent",
        "Code Quality Agent",
    ]
    final: List[AgentResult] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error("Agent '%s' raised: %s", agent_display_names[i], r)
            final.append(
                AgentResult(
                    agent_name=agent_display_names[i],
                    issues=[],
                    summary=f"Agent failed to complete: {r}",
                )
            )
        else:
            final.append(r)

    return final[0], final[1], final[2]


# ---------------------------------------------------------------------------
# Synthesizer: build the GitHub PR markdown comment
# ---------------------------------------------------------------------------

_SEVERITY_EMOJI = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
_AGENT_ICONS = {
    "Security Agent": "🔒",
    "Performance Agent": "⚡",
    "Code Quality Agent": "✨",
}


def _issues_table(issues: List[Issue]) -> str:
    if not issues:
        return "✅ No issues detected.\n"
    rows = ["| Severity | File | Line | Description |", "|----------|------|------|-------------|"]
    for issue in issues:
        emoji = _SEVERITY_EMOJI.get(issue.severity, "⚪")
        file_cell = f"`{issue.file}`" if issue.file else "—"
        line_cell = str(issue.line_number) if issue.line_number else "—"
        desc = issue.description.replace("|", "\\|")
        # Wrap long descriptions at 180 chars for readability
        if len(desc) > 180:
            desc = desc[:177] + "…"
        rows.append(f"| {emoji} `{issue.severity}` | {file_cell} | {line_cell} | {desc} |")
    return "\n".join(rows) + "\n"


def synthesize_comment(
    security: AgentResult,
    performance: AgentResult,
    quality: AgentResult,
    metadata: Dict[str, Any],
) -> str:
    """
    Combine all three agent results into a single GitHub PR markdown comment.
    """
    all_issues = security.issues + performance.issues + quality.issues
    total = len(all_issues)
    critical = sum(1 for i in all_issues if i.severity == "critical")
    warnings = sum(1 for i in all_issues if i.severity == "warning")
    info = sum(1 for i in all_issues if i.severity == "info")

    pr_title = metadata.get("title", "Pull Request")
    pr_url   = metadata.get("html_url", "")
    author   = metadata.get("author", "")
    files_changed = metadata.get("files_changed", 0)
    additions = metadata.get("additions", 0)
    deletions = metadata.get("deletions", 0)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = [
        "# 🤖 AI Code Review Report",
        "",
        "> **Automated review powered by LangGraph + Dough API (`groq/llama-3.3-70b-versatile`)**",
        "",
        "## 📊 PR Summary",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **PR** | [{pr_title}]({pr_url}) |" if pr_url else f"| **PR** | {pr_title} |",
        f"| **Author** | `{author}` |",
        f"| **Files Changed** | {files_changed} |",
        f"| **Changes** | `+{additions}` / `-{deletions}` |",
        f"| **Total Issues** | {total} (🔴 {critical} critical · 🟡 {warnings} warnings · 🔵 {info} info) |",
        "",
        "---",
        "",
    ]

    for agent_result in (security, performance, quality):
        icon = _AGENT_ICONS.get(agent_result.agent_name, "🔍")
        lines += [
            f"## {icon} {agent_result.agent_name}",
            "",
            f"**Summary:** {agent_result.summary}",
            "",
            _issues_table(agent_result.issues),
            "---",
            "",
        ]

    lines += [
        f"<sub>Generated at {now_utc} · 3 agents ran in parallel · Model: llama-3.3-70b-versatile · Diff analysed: up to {MAX_DIFF_CHARS:,} chars</sub>",
    ]

    return "\n".join(lines)
