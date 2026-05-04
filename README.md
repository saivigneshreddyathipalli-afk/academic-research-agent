# Academic Research Agent

A three-agent AI research system that investigates any niche topic and produces a hybrid IMRaD academic report with inline citations, literature review, market insights, and formal references.

## Architecture

```
User Topic → Researcher → Analyst → Writer → Report (PDF + Markdown)
```

| Agent | Role | Tools |
|-------|------|-------|
| **Researcher** | Discovers and verifies information across 5+ search angles. Grades sources by credibility (PRIMARY/SECONDARY/TERTIARY). Numbers every source for citation. | DuckDuckGo web search |
| **Analyst** | Evaluates evidence strength, formulates the research question, identifies knowledge gaps, and documents methodology. | None (analyzes Researcher output) |
| **Writer** | Produces a hybrid IMRaD academic report with inline `[n]` citations, literature review, market insights, and formal references. | None (writes from Analyst output) |

## Output Structure (Hybrid IMRaD)

1. **Introduction** — Research question, background, significance
2. **Literature Review** — Existing research, consensus, contested claims
3. **Methods** — Search protocol, source criteria, limitations
4. **Market Insights** — Size, growth, key players, trends
5. **Key Findings** — Data, technical details, evidence mapping with strength grades
6. **Discussion** — Interpretation, knowledge gaps, ethics, implications
7. **Limitations** — Methodological constraints, data gaps
8. **References** — Numbered sources with credibility tiers

## Tech Stack

- **CrewAI** — Multi-agent orchestration (sequential process)
- **Groq** — LLM inference (Llama 3.3 70B, temperature=0.3)
- **DuckDuckGo** — Web search with year-boosted queries for recency
- **Streamlit** — Web interface
- **fpdf2** — Styled PDF generation (zero system dependencies)

## Setup

### 1. Get a Groq API Key

Sign up at [console.groq.com](https://console.groq.com) and create a free API key.

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

### 3. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)

```bash
.venv\Scripts\python -m streamlit run web_app.py
```

Opens at `http://localhost:8501`. Enter any topic and click **Research Now**. The report appears with PDF and Markdown download buttons.

### CLI Interface

```bash
.venv\Scripts\python main.py
```

Enter a topic at the prompt. The report is saved as `final_report.md`.

## Project Structure

```
research-agent/
├── .env                 # Groq API key (not committed to git)
├── requirements.txt     # Python dependencies
├── config.py            # LLM configuration (Groq Llama 3.3 70B)
├── tools.py             # DuckDuckGo web search tool with recency bias
├── agents.py            # Three agent definitions (Researcher, Analyst, Writer)
├── tasks.py             # Three task definitions with academic directives
├── crew.py              # Crew assembly and execution
├── web_app.py           # Streamlit web interface
├── pdf_builder.py       # Styled PDF generator with tier-colored references
└── main.py              # CLI entry point
```

## Key Features

- **Recency-first search** — Queries auto-append the current year to surface fresh content over stale SEO pages
- **Credibility grading** — Every source tagged as PRIMARY, SECONDARY, or TERTIARY
- **Inline citations** — `[1]`, `[2]`, `[3]` format mapped to numbered references
- **Evidence strength grading** — Each finding labeled Strong/Moderate/Weak
- **Knowledge gap identification** — Analyst explicitly flags what is NOT known
- **Methodology transparency** — Dedicated Methods section documents search protocol
- **Hybrid IMRaD format** — Academic structure with business-relevant Market Insights
- **PDF + Markdown export** — Download styled PDF or raw Markdown
- **Unicode-safe PDF** — Automatic character sanitization for clean rendering

## License

MIT
