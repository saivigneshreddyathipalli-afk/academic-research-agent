import datetime

from crewai.tools import tool


@tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily AI (high-quality, recency-biased). Returns top 3 results."""
    try:
        from tavily import TavilyClient
        client = TavilyClient()
        year = datetime.datetime.now().year
        boosted_query = f"{query} {year} {year - 1}"
        results = client.search(boosted_query, max_results=3, search_depth="basic")
        output = []
        for r in results.get("results", []):
            output.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"Snippet: {r.get('content', 'N/A')[:150]}...\n"
                f"URL: {r.get('url', 'N/A')}\n"
            )
        return "\n---\n".join(output) if output else "[Tavily returned no results]"
    except Exception as e:
        return f"[Tavily search failed: {e}]"


@tool
def arxiv_fetch(query: str) -> str:
    """Fetch peer-reviewed abstracts from ArXiv. Returns top 2 papers."""
    try:
        import arxiv
        search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        output = []
        for paper in search.results():
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            output.append(
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Abstract: {paper.summary[:200]}...\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"URL: {paper.entry_id}\n"
            )
        return "\n---\n".join(output) if output else "[ArXiv returned no results]"
    except Exception as e:
        return f"[ArXiv fetch failed: {e}]"


@tool
def rag_query(query: str) -> str:
    """Query the local Chroma vector store for reference document context."""
    try:
        from rag_pipeline import ResearchRAG
        rag = ResearchRAG()
        if not rag.is_ready():
            return "[No reference documents loaded in RAG store]"
        results = rag.query(query, top_k=3)
        return results
    except Exception as e:
        return f"[RAG query failed: {e}]"
