import datetime
from crewai import Task

from agents import lead_researcher, data_analyst, senior_writer

CURRENT_YEAR = datetime.datetime.now().year


research_task = Task(
    description=(
        f"Research the following topic:\n\n"
        f"TOPIC: {{topic}}\n\n"
        f"TIME CONTEXT: Current year is {CURRENT_YEAR}. Prioritize recent data.\n\n"
        f"EXECUTION ORDER (max 3 searches):\n"
        f"  1. Query RAG store first (rag_query tool) if documents loaded.\n"
        f"  2. Run up to 2 Tavily searches across different angles:\n"
        f"     - Overview and market size\n"
        f"     - Technical developments and recent news\n"
        f"     - Challenges, regulations, or key players\n"
        f"  3. Fetch ArXiv papers for academic grounding.\n\n"
        f"OUTPUT FORMAT:\n"
        f"  Number every source [1], [2], etc. with tier:\n"
        f"  PRIMARY (gov/peer-reviewed), SECONDARY (industry/media), "
        f"TERTIARY (blogs/forums).\n"
        f"  Reference source numbers on every claim."
    ),
    expected_output=(
        f"Research notes with numbered sources, credibility tiers, facts, "
        f"and figures. Prioritize data from {CURRENT_YEAR - 1} onwards. "
        f"Every claim must have a source reference in [n] format."
    ),
    agent=lead_researcher,
)

analysis_task = Task(
    description=(
        "Compress the raw research findings into a strict ~1,000-word "
        "bulleted brief. Follow these rules exactly:\n\n"
        "1. FORMAT: Bulleted list ONLY. No paragraphs, no prose, no "
        "introductory text, no conclusions.\n\n"
        "2. EVERY BULLET must contain:\n"
        "   - A verified fact, data point, metric, or statistic\n"
        "   - A citation [n] referencing the source\n"
        "   - An evidence grade: [STRONG], [MODERATE], or [WEAK]\n\n"
        "3. ORGANIZE into sections:\n"
        "   - Key Statistics & Market Data\n"
        "   - Technical Findings\n"
        "   - Company/Organization Developments\n"
        "   - Academic Research (ArXiv/RAG)\n"
        "   - Challenges & Knowledge Gaps\n"
        "   - Regulatory & Ethical Considerations\n\n"
        "4. WORD LIMIT: Strictly ~1,000 words. Be concise. Every word "
        "must carry information.\n\n"
        "5. SOURCE PRESERVATION: Preserve ALL source numbers [n] from "
        "the researcher's output. Do not renumber or drop any sources.\n\n"
        "6. GAPS: Explicitly list what is NOT known or uncertain."
    ),
    expected_output=(
        "A strict ~1,000-word bulleted brief with verified data points, "
        "metrics, citations [n], and evidence grades [STRONG/MODERATE/WEAK]. "
        "No prose, no filler. Organized into 6 sections. All source "
        "numbers preserved."
    ),
    agent=data_analyst,
)

writing_task = Task(
    description=(
        "Using ONLY the 1,000-word brief provided in your context, write "
        "a complete, master-level academic paper. ZERO-SHOT: Execute in "
        "exactly ONE PASS.\n\n"
        "REPORT STRUCTURE (follow exactly):\n\n"
        "# Research Report: [Topic]\n\n"
        "## 1. Introduction\n"
        "   - Research question, background, significance\n"
        "   - Why this topic matters now\n\n"
        "## 2. Literature Review\n"
        "   - Existing research synthesis with inline [n] citations\n"
        "   - Consensus vs. contested findings\n"
        "   - Position within broader field\n\n"
        "## 3. Methods\n"
        "   - Search strategy, source criteria, credibility grading\n"
        "   - Time period, limitations of web-based research\n\n"
        "## 4. Market Insights\n"
        "   - Market size, growth, key statistics with [n] citations\n"
        "   - Major players, trends, regulatory landscape\n\n"
        "## 5. Key Findings\n"
        "   - Data points, technical details with [n] citations\n"
        "   - Evidence strength grading\n\n"
        "## 6. Discussion\n"
        "   - Interpretation, knowledge gaps, ethical dimensions\n"
        "   - Practical and theoretical implications\n"
        "   - Future outlook\n\n"
        "## 7. Limitations\n"
        "   - Methodological constraints, data gaps, scope boundaries\n\n"
        "## 8. References\n"
        "   [1] [TIER] Source Title - URL\n"
        "   [2] [TIER] Source Title - URL\n"
        "   ...\n\n"
        "WRITING STANDARDS:\n"
        "  • Inline citations [1], [2], [3] throughout body text\n"
        "  • Qualify claims: 'evidence suggests' not 'proves'\n"
        "  • Exact numbers and dates, not approximations\n"
        "  • Define technical terms on first use\n"
        "  • No filler, no unsupported opinions\n\n"
        "CRITICAL: Write the ENTIRE paper in ONE PASS. Do not plan, "
        "outline, or self-correct. Read the brief once and produce the "
        "final report immediately."
    ),
    expected_output=(
        "A complete, master-level academic paper with all 8 sections, "
        "inline [n] citations, and a References section listing EVERY "
        "source. Precise, evidence-based, jargon-free. Written in one pass."
    ),
    agent=senior_writer,
)
