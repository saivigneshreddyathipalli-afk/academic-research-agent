import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["MOONSHOT_API_KEY"] = os.getenv("MOONSHOT_API_KEY", "")

llm_groq = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
)

llm_kimi = LLM(
    model="openai/moonshot-v1-32k",
    temperature=0.2,
    api_key=os.getenv("MOONSHOT_API_KEY", ""),
    api_base="https://api.moonshot.cn/v1",
)
