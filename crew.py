from crewai import Crew, Process

from agents import lead_researcher, data_analyst, senior_writer
from tasks import research_task, analysis_task, writing_task


def run_crew(topic: str):
    """
    Build and execute the 3-agent dual-engine research crew for the given topic.

    Phase 1 (Lead Researcher, Groq) → Phase 2 (Data Analyst, Groq) → Phase 3 (Senior Writer, Gemini 1.5 Flash)

    Args:
        topic: The research subject supplied by the user.

    Returns:
        The final academic markdown report as a string.
    """
    research_crew = Crew(
        agents=[lead_researcher, data_analyst, senior_writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
    )

    result = research_crew.kickoff(inputs={"topic": topic})
    return result.raw
