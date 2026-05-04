# config.py
# ──────────────────────────────────────────────────────────────────────────────
# Centralised LLM and shared configuration.
# Every agent reuses the `llm` instance so the project talks to one model
# with a single, predictable temperature setting.
#
# CrewAI 1.x uses its own LLM class backed by LiteLLM, which supports Groq
# via the "groq/" provider prefix.
# ──────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from crewai import LLM

# Load environment variables from the .env file at project root.
load_dotenv()

# Set the GROQ API key in the environment so LiteLLM can pick it up.
# LiteLLM expects the key as GROQ_API_KEY (same as our .env variable).
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# Instantiate the LLM via CrewAI's unified LLM class.
# ────────────────────────────────────────────────────────────────
# • model        — "groq/llama-3.3-70b-versatile" (LiteLLM provider prefix)
# • temperature  — 0.3  (low enough for factual accuracy, not robotic)
# ────────────────────────────────────────────────────────────────
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
)
