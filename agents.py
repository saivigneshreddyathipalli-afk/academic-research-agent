from crewai import Agent

from config import llm_groq, get_writer_llm
from tools import tavily_search, arxiv_fetch, rag_query


lead_researcher = Agent(
    role="Lead Researcher",
    goal="Research any topic using Tavily, ArXiv, and RAG. Output numbered sources with credibility tiers.",
    backstory=(
        "You are a meticulous research agent. Query RAG first if documents exist, "
        "then use Tavily for web results and ArXiv for academic papers.\n"
        "Number every source [1], [2], etc. with tiers: PRIMARY (gov/peer-reviewed), "
        "SECONDARY (industry/media), TERTIARY (blogs/forums).\n"
        "Reference source numbers on every claim."
    ),
    llm=llm_groq,
    tools=[tavily_search, arxiv_fetch, rag_query],
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

data_analyst = Agent(
    role="Data Analyst",
    goal="Compress research into a strict ~1,000-word bulleted brief with citations and evidence grades.",
    backstory=(
        "You are a data compressor. Strip everything except verified facts, metrics, and citations.\n"
        "Every bullet: a fact + citation [n] + grade [STRONG/MODERATE/WEAK].\n"
        "No prose, no intro, no conclusions. Pure structured data."
    ),
    llm=llm_groq,
    tools=[],
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

senior_writer = Agent(
    role="Senior Academic Writer",
    goal="Write a master-level academic paper from the brief in one zero-shot pass.",
    backstory=(
        "You are an expert academic writer producing hybrid IMRaD reports with inline [n] citations.\n"
        "Never fabricate data. Qualify uncertain findings.\n"
        "CRITICAL: Write the entire paper in ONE PASS. No planning, no self-correction."
    ),
    llm=get_writer_llm(),
    tools=[],
    verbose=True,
    allow_delegation=False,
    max_iter=1,
)
