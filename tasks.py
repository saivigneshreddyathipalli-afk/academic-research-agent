# tasks.py
# ──────────────────────────────────────────────────────────────────────────────
# Defines the three tasks that the crew executes sequentially.
#
#   Task 1 (Research)   — Assigned to the Researcher agent.
#       Searches the web, grades sources, and outputs numbered, structured
#       research notes with credibility tiers (PRIMARY/SECONDARY/TERTIARY).
#
#   Task 2 (Analysis)   — Assigned to the Analyst agent.
#       Evaluates evidence quality, formulates the research question,
#       identifies knowledge gaps, and documents the methodology.
#
#   Task 3 (Writing)    — Assigned to the Writer agent.
#       Produces a hybrid IMRaD academic report with inline [n] citations,
#       literature review, market insights, and formal references.
#
# How Task objects interact with Agent objects:
# ──────────────────────────────────────────────────────────────────────────────
# • Each Task is bound to exactly one Agent via the `agent=` parameter.
#   The Agent's role, goal, and backstory shape how the LLM interprets the
#   task description.
#
# • Tasks can reference dynamic variables in their descriptions using `{name}`
#   placeholders.  At execution time CrewAI substitutes these with the values
#   provided to `crew.kickoff(inputs={...})`.
#
# • In a SEQUENTIAL crew, the OUTPUT of Task N is automatically appended to
#   the CONTEXT of Task N+1.  So:
#     - Analyst receives Researcher's full output as context
#     - Writer receives Analyst's full output (which includes Researcher's
#       data) as context
# ──────────────────────────────────────────────────────────────────────────────

import datetime
from crewai import Task

from agents import researcher, analyst, writer

# Current year injected into tasks so agents know the temporal context.
# Without this, the LLM assumes its training cutoff year and returns stale data.
CURRENT_YEAR = datetime.datetime.now().year

# ── Research Task ─────────────────────────────────────────────────────────────
# The Researcher agent executes this task.
# It uses the web_search tool to query DuckDuckGo for information about
# the user-provided {topic}.  Sources are numbered and credibility-graded.
#
# Key directives:
#   1. Recency — must search for data from the last 2 years.
#   2. Technical depth — must investigate specs, companies, R&D pipelines.
#   3. Source integrity — numbered sources with credibility tiers.
#   4. Multi-query — at least 5 searches across different angles.
# ──────────────────────────────────────────────────────────────────────────────
research_task = Task(
    description=(
        f"Research the following topic comprehensively:\n\n"
        f"TOPIC: {{topic}}\n\n"
        f"TIME CONTEXT: The current year is {CURRENT_YEAR}. "
        f"You MUST prioritize data, statistics, and developments from "
        f"{CURRENT_YEAR-1} onwards. When searching, append the current year "
        f"to your queries (e.g., '{{topic}} {CURRENT_YEAR}', "
        f"'{{topic}} market {CURRENT_YEAR-1}'). "
        f"Explicitly avoid relying on data older than 5 years unless "
        f"it is historical context for comparison.\n\n"
        f"SEARCH STRATEGY: Run at least 5 separate searches:\n"
        f"  1) Overview and market size (e.g., '{{topic}} market size {CURRENT_YEAR}')\n"
        f"  2) Technical developments and company-specific details "
        f"(e.g., '{{topic}} companies technology specifications')\n"
        f"  3) Recent news and developments "
        f"(e.g., '{{topic}} news {CURRENT_YEAR} {CURRENT_YEAR-1}')\n"
        f"  4) Academic/policy sources "
        f"(e.g., '{{topic}} study research report government')\n"
        f"  5) Challenges, controversies, or limitations "
        f"(e.g., '{{topic}} problems challenges risks')\n\n"
        f"INVESTIGATE AND GATHER INFORMATION ON:\n"
        f"  • Current market size, growth trends, and key statistics\n"
        f"  • Major players, brands, or organisations in this space\n"
        f"  • Relevant certifications, standards, or regulatory bodies\n"
        f"  • Recent developments, news, or controversies\n"
        f"  • Challenges and barriers to entry\n"
        f"  • Future outlook and emerging opportunities\n"
        f"  • Ethical considerations and compliance requirements\n\n"
        f"TECHNICAL INVESTIGATION (critical for quality):\n"
        f"  • Specific companies, startups, or labs working on emerging tech\n"
        f"  • Technical specifications, performance metrics, or benchmarks\n"
        f"  • R&D pipelines, pilot programs, or prototype deployments\n"
        f"  • Patents filed, technical papers, or conference presentations\n"
        f"  • Engineering challenges being solved (materials, design, algorithms)\n\n"
        f"OUTPUT FORMAT (strict):\n"
        f"  Every source must be numbered [1], [2], [3], etc. with:\n"
        f"  [N] [TIER] Source Title — URL\n"
        f"  Where TIER is one of:\n"
        f"    PRIMARY   — Government, peer-reviewed journals, official orgs\n"
        f"    SECONDARY — Industry reports, established media, trade pubs\n"
        f"    TERTIARY  — Blogs, opinion pieces, forums, social media\n\n"
        f"  Every fact, statistic, or claim must reference its source number "
        f"in brackets, e.g., 'The market grew 12% [1].' Cross-reference "
        f"claims across multiple sources whenever possible."
    ),
    expected_output=(
        f"A detailed set of research notes with facts, figures, numbered "
        f"sources with credibility tiers, and observations covering all "
        f"investigation points above. Prioritize data from {CURRENT_YEAR-1} "
        f"onwards. Every claim must have a source reference in [n] format."
    ),
    agent=researcher,
)

# ── Analysis Task ─────────────────────────────────────────────────────────────
# The Analyst agent executes this task.
#
# CONTEXT FLOW: Receives the Researcher's complete output (numbered sources,
# findings, technical details) via sequential process injection.
#
# The Analyst must:
#   1. Formulate a clear, answerable research question from the data
#   2. Evaluate evidence strength per finding (Strong/Moderate/Weak)
#   3. Identify knowledge gaps (what is NOT known)
#   4. Document the methodology (search strategy, source criteria)
#   5. Map claims to evidence with citation numbers preserved
# ──────────────────────────────────────────────────────────────────────────────
analysis_task = Task(
    description=(
        "Evaluate and synthesize the research findings provided in your "
        "context for the topic: {topic}\n\n"
        "You must produce a structured analysis that covers:\n\n"
        "1. RESEARCH QUESTION\n"
        "   Formulate a clear, specific, answerable research question based "
        "on the gathered data. This question should be precise enough to "
        "guide the report's focus.\n\n"
        "2. EVIDENCE EVALUATION\n"
        "   For each major finding, grade its strength:\n"
        "   - STRONG: Supported by PRIMARY sources AND corroborated by "
        "multiple independent sources\n"
        "   - MODERATE: Supported by SECONDARY sources OR single PRIMARY source\n"
        "   - WEAK: Based on TERTIARY sources OR conflicting evidence\n\n"
        "3. KNOWLEDGE GAPS\n"
        "   Identify what is NOT known or uncertain about this topic.\n"
        "   Where is the research thin? Where do sources contradict?\n"
        "   What questions remain unanswered?\n\n"
        "4. METHODOLOGY DOCUMENTATION\n"
        "   Document how the research was conducted:\n"
        "   - Search strategy (what queries were used, how many searches)\n"
        "   - Source selection criteria (how sources were graded)\n"
        "   - Time period covered\n"
        "   - Inherent limitations of web-based research\n\n"
        "5. CLAIM-TO-EVIDENCE MAP\n"
        "   List the key claims and which source numbers [n] support each.\n"
        "   Preserve all original citation numbers from the Researcher.\n\n"
        "6. ETHICAL CONSIDERATIONS\n"
        "   Note any ethical, regulatory, or compliance dimensions relevant "
        "to this topic.\n\n"
        "CRITICAL: Preserve ALL source numbers [n] from the Researcher's "
        "output. Do not renumber or drop any sources."
    ),
    expected_output=(
        "A structured analysis with: (1) a clear research question, "
        "(2) evidence-strength grading for each finding, "
        "(3) identified knowledge gaps, (4) methodology documentation, "
        "(5) claim-to-evidence mapping, and (6) ethical considerations. "
        "All original source numbers must be preserved."
    ),
    agent=analyst,
)

# ── Writing Task ──────────────────────────────────────────────────────────────
# The Writer agent executes this task.
#
# CONTEXT FLOW: Receives the Analyst's complete output (which includes the
# Researcher's data via context chain) via sequential process injection.
#
# The Writer must produce a hybrid IMRaD academic report with:
#   - Inline [n] citations mapped to numbered references
#   - Literature review section with source synthesis
#   - Market Insights section (business context hybrid)
#   - Methodology transparency
#   - Knowledge gaps and limitations
#   - Complete References section with tier labels
#
# CRITICAL SOURCE INTEGRITY:
#   Every source from the research chain must appear in References.
#   No omissions, no consolidations, no dropped links.
# ──────────────────────────────────────────────────────────────────────────────
writing_task = Task(
    description=(
        "Using the research findings and analysis provided in your context, "
        "create a premium, publication-ready hybrid IMRaD academic report "
        "on the topic: {topic}\n\n"
        "REPORT STRUCTURE (follow exactly):\n\n"
        "# Research Report: [Topic]\n\n"
        "## 1. Introduction\n"
        "   - Formulated research question (from the analysis)\n"
        "   - Background and context\n"
        "   - Why this topic matters now (significance)\n\n"
        "## 2. Literature Review\n"
        "   - What existing research and sources say about this topic\n"
        "   - Consensus findings with inline [n] citations\n"
        "   - Where evidence is strong vs. contested\n"
        "   - Position this topic within the broader field\n\n"
        "## 3. Methods\n"
        "   - Search strategy and protocol (queries used, number of searches)\n"
        "   - Source selection criteria (PRIMARY/SECONDARY/TERTIARY grading)\n"
        "   - Credibility assessment approach\n"
        "   - Time period covered\n"
        "   - Transparency about limitations of web-based research\n\n"
        "## 4. Market Insights\n"
        "   - Market size, growth trends, and key statistics with [n] citations\n"
        "   - Major players, brands, or organisations\n"
        "   - Industry trends and trajectory\n"
        "   - Regulatory landscape and certifications\n\n"
        "## 5. Key Findings\n"
        "   - Data points and technical details with inline [n] citations\n"
        "   - Company-specific developments, specs, R&D pipelines\n"
        "   - Evidence-to-claim mapping\n"
        "   - Grade each finding's strength (Strong/Moderate/Weak)\n\n"
        "## 6. Discussion\n"
        "   - Interpretation of results and what they mean\n"
        "   - Knowledge gaps identified (what remains unknown)\n"
        "   - Ethical considerations and compliance dimensions\n"
        "   - Practical and theoretical implications\n"
        "   - Opportunities and future outlook\n\n"
        "## 7. Limitations\n"
        "   - Methodological constraints\n"
        "   - Data gaps and areas of uncertainty\n"
        "   - Scope boundaries (what this report does NOT cover)\n\n"
        "## 8. References\n"
        "   [1] [TIER] Source Title - URL\n"
        "   [2] [TIER] Source Title - URL\n"
        "   ...\n\n"
        "WRITING STANDARDS:\n"
        "  • Use inline citations [1], [2], [3] throughout the body text.\n"
        "  • Qualify claims: 'evidence suggests' not 'proves'.\n"
        "  • Use exact numbers and dates, not approximations.\n"
        "  • Avoid jargon; define technical terms on first use.\n"
        "  • Distinguish consensus from contested claims.\n"
        "  • No filler, no unsupported opinions.\n\n"
        "CRITICAL — SOURCE INTEGRITY:\n"
        "  Extract EVERY source from the research notes. Do NOT omit, "
        "consolidate, or drop any source. If the Researcher found 15 "
        "sources numbered [1] through [15], the References section must "
        "have exactly 15 entries with matching numbers. Verify every "
        "source against the research notes before finalizing."
    ),
    expected_output=(
        "A complete, hybrid IMRaD academic report with all 8 required "
        "sections. Inline [n] citations throughout. The References section "
        "must enumerate EVERY source from the research chain — no omissions. "
        "Writing must be precise, evidence-based, and jargon-free."
    ),
    agent=writer,
)
