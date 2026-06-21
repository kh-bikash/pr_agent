"""
main.py
FastAPI application entry-point for the AI Code Review Agent.

Endpoints:
  GET  /health   → liveness check
  POST /review   → trigger the 3-agent review pipeline

Run with:
  uvicorn main:app --reload --port 8000
"""

import logging
import time
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from agents import AgentResult, run_all_agents, synthesize_comment
from github_utils import (
    fetch_pr_diff,
    fetch_pr_metadata,
    parse_pr_url,
    post_pr_comment,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Code Review Agent",
    description=(
        "Automated GitHub PR review powered by LangGraph + Groq. "
        "Three parallel agents (Security, Performance, Code Quality) analyse "
        "the PR diff and post a structured markdown comment."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # CRA / alternative
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    pr_url: str
    post_comment: bool = True   # set False to skip posting to GitHub

    @field_validator("pr_url", mode="before")
    @classmethod
    def strip_url(cls, v: str) -> str:
        return v.strip()


class IssueOut(BaseModel):
    line_number: Optional[int] = None
    file: Optional[str] = None
    severity: str
    description: str


class AgentResultOut(BaseModel):
    agent_name: str
    issues: List[IssueOut]
    summary: str


class ReviewResponse(BaseModel):
    success: bool
    pr_url: str
    pr_title: str
    pr_author: str
    security: AgentResultOut
    performance: AgentResultOut
    code_quality: AgentResultOut
    markdown_comment: str
    comment_posted: bool
    comment_url: Optional[str] = None
    duration_seconds: float
    total_issues: int
    critical_count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe — returns 200 if the server is up."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/review", response_model=ReviewResponse, tags=["Review"])
async def review_pr(request: ReviewRequest):
    """
    Analyse a GitHub PR with three AI agents and (optionally) post the review.

    Steps:
    1. Parse the PR URL → owner / repo / pr_number
    2. Fetch diff + metadata from GitHub API in parallel
    3. Run Security, Performance, and Code Quality agents in parallel
    4. Synthesise a markdown comment
    5. Post comment to GitHub (if post_comment=True and GITHUB_TOKEN is set)
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Parse URL
    # ------------------------------------------------------------------
    try:
        owner, repo, pr_number = parse_pr_url(request.pr_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info("Starting review for %s/%s#%s", owner, repo, pr_number)

    # ------------------------------------------------------------------
    # 2. Fetch diff + metadata (parallel)
    # ------------------------------------------------------------------
    try:
        import asyncio
        diff, metadata = await asyncio.gather(
            fetch_pr_diff(owner, repo, pr_number),
            fetch_pr_metadata(owner, repo, pr_number),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        logger.error("GitHub fetch failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"GitHub API error: {exc}")

    if not diff or not diff.strip():
        raise HTTPException(
            status_code=400,
            detail="The PR diff is empty — the PR may have no file changes.",
        )

    logger.info(
        "Diff fetched (%d chars). PR: '%s' by @%s",
        len(diff),
        metadata.get("title"),
        metadata.get("author"),
    )

    # ------------------------------------------------------------------
    # 3. Run agents
    # ------------------------------------------------------------------
    try:
        security_result, perf_result, quality_result = await run_all_agents(
            diff, metadata
        )
    except Exception as exc:
        logger.error("Agent pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline error: {exc}",
        )

    # ------------------------------------------------------------------
    # 4. Synthesise
    # ------------------------------------------------------------------
    markdown_comment = synthesize_comment(
        security_result, perf_result, quality_result, metadata
    )

    # ------------------------------------------------------------------
    # 5. Post comment (best-effort — failure does not abort the response)
    # ------------------------------------------------------------------
    comment_posted = False
    comment_url: Optional[str] = None

    if request.post_comment:
        try:
            comment_data = await post_pr_comment(
                owner, repo, pr_number, markdown_comment
            )
            comment_posted = True
            comment_url = comment_data.get("html_url")
        except PermissionError as exc:
            logger.warning("Skipping comment post (no token / permission): %s", exc)
        except Exception as exc:
            logger.warning("Could not post GitHub comment: %s", exc)

    # ------------------------------------------------------------------
    # 6. Build response
    # ------------------------------------------------------------------
    all_issues = (
        security_result.issues + perf_result.issues + quality_result.issues
    )
    duration = round(time.perf_counter() - t0, 2)

    logger.info(
        "Review complete in %.1fs — %d issues found (%d critical)",
        duration,
        len(all_issues),
        sum(1 for i in all_issues if i.severity == "critical"),
    )

    def _to_out(r: AgentResult) -> AgentResultOut:
        return AgentResultOut(
            agent_name=r.agent_name,
            issues=[IssueOut(**i.model_dump()) for i in r.issues],
            summary=r.summary,
        )

    return ReviewResponse(
        success=True,
        pr_url=request.pr_url,
        pr_title=metadata.get("title", ""),
        pr_author=metadata.get("author", ""),
        security=_to_out(security_result),
        performance=_to_out(perf_result),
        code_quality=_to_out(quality_result),
        markdown_comment=markdown_comment,
        comment_posted=comment_posted,
        comment_url=comment_url,
        duration_seconds=duration,
        total_issues=len(all_issues),
        critical_count=sum(1 for i in all_issues if i.severity == "critical"),
    )


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
