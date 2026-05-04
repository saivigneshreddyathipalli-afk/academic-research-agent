# crew.py
# ──────────────────────────────────────────────────────────────────────────────
# Assembles the three-agent crew and exposes a simple `run_crew()` function.
#
# Process: SEQUENTIAL
#   Task 1 (Research)  — Web search, source grading, numbered findings
#   Task 2 (Analysis)  — Evidence evaluation, research question, gap analysis
#   Task 3 (Writing)   — Hybrid IMRaD academic report with inline citations
#
# Context flows automatically:
#   Research output → Analyst context
#   Analyst output → Writer context (includes Research data transitively)
# ──────────────────────────────────────────────────────────────────────────────

from crewai import Crew, Process

from agents import researcher, analyst, writer
from tasks import research_task, analysis_task, writing_task


def run_crew(topic: str):
    """
    Build and execute the 3-agent research crew for the given topic.

    Args:
        topic: The research subject supplied by the user.

    Returns:
        The final hybrid IMRaD markdown report as a string.
    """

    # The Crew ties agents, tasks, and execution process together.
    # ──────────────────────────────────────────────────────────
    # • agents       — [researcher, analyst, writer] from agents.py
    # • tasks        — [research_task, analysis_task, writing_task] from tasks.py
    # • process      — SEQUENTIAL (tasks run in order; output of task N feeds
    #                  task N+1 automatically via context injection)
    # • verbose      — True prints each agent's thought process to the console
    # • memory       — False to avoid overhead for single-run research
    # ──────────────────────────────────────────────────────────
    research_crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    # `kickoff` starts execution.  The `inputs` dict provides values for
    # the `{topic}` placeholder found in all task descriptions.
    #
    # Context chain:
    #   research_task → runs first → output injected into analysis_task
    #   analysis_task → runs second → output (with research data) injected into writing_task
    #   writing_task → runs third → produces final report
    result = research_crew.kickoff(inputs={"topic": topic})

    return result.raw
