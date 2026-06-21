"""
github_utils.py
Utilities for interacting with the GitHub REST API:
  - parse_pr_url       : extract owner/repo/pr_number from a PR URL
  - fetch_pr_diff      : download the unified diff for a PR
  - fetch_pr_metadata  : get PR title, stats, author, etc.
  - post_pr_comment    : post a markdown comment on a PR
"""

import logging
import os
import re
from typing import Any, Dict, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# Regex that matches both /pull/ and /pulls/ (tolerant of trailing slashes)
_PR_PATTERN = re.compile(
    r"https?://github\.com/([^/]+)/([^/]+)/pull[s]?/(\d+)"
)


def _base_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def parse_pr_url(url: str) -> Tuple[str, str, int]:
    """
    Parse a GitHub PR URL into (owner, repo, pr_number).

    Raises ValueError on invalid input.
    """
    url = url.strip().rstrip("/")
    match = _PR_PATTERN.match(url)
    if not match:
        raise ValueError(
            f"Invalid GitHub PR URL: '{url}'. "
            "Expected: https://github.com/owner/repo/pull/123"
        )
    owner, repo, pr_number = match.groups()
    return owner, repo, int(pr_number)


async def fetch_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """
    Fetch the unified diff for a PR via the GitHub API.

    Returns the raw diff text.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    diff_headers = {
        **_base_headers(),
        "Accept": "application/vnd.github.v3.diff",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=diff_headers)

    if resp.status_code == 401:
        raise PermissionError(
            "GitHub API returned 401 Unauthorized. "
            "Check your GITHUB_TOKEN or set it for private repos."
        )
    if resp.status_code == 404:
        raise FileNotFoundError(
            f"PR #{pr_number} not found in {owner}/{repo}. "
            "Verify the URL and that the repo is accessible."
        )
    resp.raise_for_status()

    diff_text = resp.text
    if not diff_text.strip():
        logger.warning("PR #%s diff is empty — no file changes detected.", pr_number)

    logger.info(
        "Fetched diff for %s/%s#%s — %d chars", owner, repo, pr_number, len(diff_text)
    )
    return diff_text


async def fetch_pr_metadata(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Fetch PR metadata: title, number, author, branch info, change stats.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=_base_headers())

    if resp.status_code == 404:
        raise FileNotFoundError(
            f"PR #{pr_number} not found in {owner}/{repo}."
        )
    resp.raise_for_status()

    data = resp.json()
    return {
        "title": data.get("title", ""),
        "number": data.get("number", pr_number),
        "body": data.get("body") or "",
        "author": data.get("user", {}).get("login", "unknown"),
        "base_branch": data.get("base", {}).get("ref", "main"),
        "head_branch": data.get("head", {}).get("ref", ""),
        "state": data.get("state", "open"),
        "files_changed": data.get("changed_files", 0),
        "additions": data.get("additions", 0),
        "deletions": data.get("deletions", 0),
        "html_url": data.get("html_url", ""),
        "repo_full_name": f"{owner}/{repo}",
    }


async def post_pr_comment(
    owner: str, repo: str, pr_number: int, body: str
) -> Dict[str, Any]:
    """
    Post a markdown comment on a GitHub PR (via the Issues API).

    Returns the created comment object from GitHub.
    Raises httpx.HTTPStatusError on failure.
    """
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    if not github_token:
        raise PermissionError(
            "GITHUB_TOKEN is not set. "
            "A token is required to post comments on PRs."
        )

    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{pr_number}/comments"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers=_base_headers(),
            json={"body": body},
        )

    if resp.status_code == 403:
        raise PermissionError(
            "GitHub API returned 403 Forbidden. "
            "Ensure your GITHUB_TOKEN has 'repo' scope."
        )
    resp.raise_for_status()

    comment = resp.json()
    logger.info(
        "Posted review comment on %s/%s#%s → %s",
        owner,
        repo,
        pr_number,
        comment.get("html_url", ""),
    )
    return comment
