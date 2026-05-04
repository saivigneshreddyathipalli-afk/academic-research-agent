# main.py
# ──────────────────────────────────────────────────────────────────────────────
# Entry point for the Multi-Agent Research System — Uno Version.
#
# Flow:
#   1. Load .env (ensures GROQ_API_KEY and ANTHROPIC_API_KEY are available).
#   2. Prompt the user for a research topic.
#   3. Hand the topic to crew.run_crew().
#   4. Write the resulting markdown to final_report.md.
# ──────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

from crew import run_crew

load_dotenv()


def main():
    print("=" * 60)
    print("  Multi-Agent Research System — Uno Version")
    print("  Dual-Engine: Groq Llama-3.3 + Claude Sonnet 4")
    print("=" * 60)
    print()

    topic = input("Enter your research topic: ").strip()

    if not topic:
        print("Error: Topic cannot be empty.")
        return

    print(f"\nResearching: {topic}\n")
    print("-" * 40)
    print("Phase 1/3: Lead Researcher (Groq) — Tavily + ArXiv + RAG")
    print("Phase 2/3: Data Analyst (Groq) — Compression to 1,000-word brief")
    print("Phase 3/3: Senior Writer (Claude) — Zero-shot academic paper")
    print("-" * 40)

    result = run_crew(topic)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_report.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print("-" * 40)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
