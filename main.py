# main.py
# ──────────────────────────────────────────────────────────────────────────────
# Entry point for the Multi-Agent Research System.
#
# Flow:
#   1. Load .env (ensures GROQ_API_KEY is available).
#   2. Prompt the user for a research topic.
#   3. Hand the topic to crew.run_crew().
#   4. Write the resulting markdown to final_report.md.
# ──────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

from crew import run_crew

# ── Load API key from .env ───────────────────────────────────────────────────
# The .env file must exist in the project root with a valid GROQ_API_KEY.
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()


def main():
    print("=" * 60)
    print("  Multi-Agent Research System")
    print("  Powered by CrewAI + LangChain + Groq")
    print("=" * 60)
    print()

    topic = input("Enter your research topic: ").strip()

    if not topic:
        print("Error: Topic cannot be empty.")
        return

    print(f"\nResearching: {topic}\n")
    print("-" * 40)

    # Execute the crew — runs Researcher first, then Writer sequentially.
    result = run_crew(topic)

    # ── Write final report to disk ───────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_report.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print("-" * 40)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
