import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

llm_groq = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=4096,
)

# Phase 3 writer: Groq by default (free, no quota issues).
# To switch to Gemini 2.0 Flash, change `llm_writer = llm_groq` below to
# `llm_writer = LLM(model="google/gemini-2.0-flash", temperature=0.2)`.
llm_writer = llm_groq

# Keep Gemini instance available for easy swap
llm_gemini = LLM(
    model="google/gemini-2.0-flash",
    temperature=0.2,
)
