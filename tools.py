# tools.py
# ──────────────────────────────────────────────────────────────────────────────
# Tool factory — returns a CrewAI-compatible web search tool powered by
# DuckDuckGo.  CrewAI 1.x requires its own BaseTool format, so we wrap the
# DuckDuckGo SDK directly with the @tool decorator.
#
# TIME-BOOSTING: Every query is automatically appended with the current year
# and the previous year (e.g., "query 2026 2025"). This biases DuckDuckGo
# results toward recent content, combating the SEO problem where outdated
# high-authority pages rank above fresh but lower-authority news articles.
# ──────────────────────────────────────────────────────────────────────────────

import datetime

from crewai.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search(query: str) -> str:
    """Search the web with recency bias and return formatted results.

    The query is automatically boosted with the current and previous year
    to prioritize recent content over stale SEO pages.

    Args:
        query: The search query string.

    Returns:
        Formatted search results with titles, snippets, and URLs.
    """
    year = datetime.datetime.now().year
    # Append current and previous year to bias results toward recent content.
    time_boosted_query = f"{query} {year} {year - 1}"
    results = DDGS().text(time_boosted_query, max_results=10)
    output = []
    for r in results:
        output.append(
            f"Title: {r['title']}\n"
            f"Snippet: {r['body']}\n"
            f"URL: {r['href']}\n"
        )
    return "\n---\n".join(output)
