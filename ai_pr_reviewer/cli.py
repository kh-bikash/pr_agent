import argparse
import asyncio
import os
import sys

# Force UTF-8 for Windows terminals to print emojis correctly
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme

from ai_pr_reviewer.agents import run_all_agents, synthesize_comment
from ai_pr_reviewer.github_utils import (
    fetch_pr_diff,
    fetch_pr_metadata,
    parse_pr_url,
    post_pr_comment,
)

# Set up Rich console
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})
console = Console(theme=custom_theme)


async def async_main(url: str, post_comment: bool):
    console.print(Panel(f"[bold blue]AI Code Review Agent[/]\nTarget: {url}", expand=False))

    try:
        owner, repo, pr_number = parse_pr_url(url)
    except ValueError as e:
        console.print(f"[danger]Error:[/] {e}")
        sys.exit(1)

    # Fetch Diff & Metadata
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Fetching PR diff & metadata from GitHub...", total=None)
        try:
            diff, metadata = await asyncio.gather(
                fetch_pr_diff(owner, repo, pr_number),
                fetch_pr_metadata(owner, repo, pr_number),
            )
        except Exception as e:
            console.print(f"[danger]GitHub Error:[/] {e}")
            sys.exit(1)

    if not diff or not diff.strip():
        console.print("[warning]The PR diff is empty. Nothing to review.[/]")
        sys.exit(0)

    console.print(f"[green][OK][/] Fetched PR diff ({len(diff):,} characters)")
    console.print(f"[green][OK][/] Author: @{metadata.get('author')} | Files changed: {metadata.get('files_changed')}")

    # Run Agents
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Running Security, Performance, & Code Quality Agents in parallel...", total=None)
        try:
            sec_res, perf_res, qual_res = await run_all_agents(diff, metadata)
        except Exception as e:
            console.print(f"[danger]Agent Pipeline Error:[/] {e}")
            sys.exit(1)

    console.print("[green][OK][/] Agents completed analysis.\n")

    # Display Results
    all_issues = sec_res.issues + perf_res.issues + qual_res.issues
    critical = sum(1 for i in all_issues if i.severity == "critical")
    warnings = sum(1 for i in all_issues if i.severity == "warning")
    
    stat_msg = f"Total Issues: {len(all_issues)} | Critical: {critical} | Warnings: {warnings}"
    console.print(Panel(stat_msg, style="bold cyan", expand=False))

    markdown_output = synthesize_comment(sec_res, perf_res, qual_res, metadata)

    # Print markdown output beautifully in terminal
    console.print(Markdown(markdown_output))

    # Post Comment
    if post_comment:
        if not os.getenv("GITHUB_TOKEN"):
            console.print("[warning]Cannot post to GitHub: GITHUB_TOKEN environment variable is not set.[/]")
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Posting review comment to GitHub...", total=None)
                try:
                    res = await post_pr_comment(owner, repo, pr_number, markdown_output)
                    console.print(f"\n[bold green][OK] Successfully posted comment to GitHub![/]\n[link={res.get('html_url')}]{res.get('html_url')}[/link]")
                except Exception as e:
                    console.print(f"\n[danger]Failed to post comment:[/] {e}")


def main():
    parser = argparse.ArgumentParser(description="AI Pull Request Reviewer CLI")
    parser.add_argument("url", help="GitHub Pull Request URL (e.g., https://github.com/owner/repo/pull/123)")
    parser.add_argument("--post-comment", action="store_true", help="Post the generated review to the GitHub PR as a comment")
    parser.add_argument("--dough-api-key", help="Dough.id API Key (overrides environment variable)")
    parser.add_argument("--github-token", help="GitHub Personal Access Token (overrides environment variable)")
    
    args = parser.parse_args()

    # Load API keys if running locally where .env exists
    from dotenv import load_dotenv
    load_dotenv()

    # Override environment with CLI arguments if provided
    if args.dough_api_key:
        os.environ["DOUGH_API_KEY"] = args.dough_api_key
    if args.github_token:
        os.environ["GITHUB_TOKEN"] = args.github_token

    # Basic API Key validation
    if not os.getenv("DOUGH_API_KEY") and not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
         console.print("[danger]Error:[/] DOUGH_API_KEY (or compatible API key) must be set in environment or .env file.")
         sys.exit(1)

    try:
        asyncio.run(async_main(args.url, args.post_comment))
    except KeyboardInterrupt:
        console.print("\n[warning]Review cancelled by user.[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
