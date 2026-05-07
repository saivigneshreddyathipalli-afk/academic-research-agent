import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

llm_groq = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
)

llm_gemini = LLM(
    model="google/gemini-2.0-flash",
    temperature=0.2,
)
