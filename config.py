import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

llm_groq = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
)

llm_claude = LLM(
    model="anthropic/claude-sonnet-4-20250514",
    temperature=0.2,
)
