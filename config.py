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

llm_gemini = LLM(
    model="google/gemini-2.0-flash",
    temperature=0.2,
)

# Auto-detect if Gemini has quota; fall back to Groq if not
_gemini_available = None

def get_writer_llm():
    """Return Gemini if available, otherwise Groq fallback."""
    global _gemini_available
    if _gemini_available is not None:
        return llm_gemini if _gemini_available else llm_groq

    try:
        llm_gemini.call("Test.")
        _gemini_available = True
        return llm_gemini
    except Exception:
        _gemini_available = False
        print("[WARN] Gemini quota exhausted. Falling back to Groq Llama-3.3-70B for Phase 3.")
        return llm_groq
