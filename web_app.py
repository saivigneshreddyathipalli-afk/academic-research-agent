from dotenv import load_dotenv
import streamlit as st

from crew import run_crew
from pdf_builder import generate_pdf

load_dotenv()

st.set_page_config(
    page_title="Academic Research Agent — Uno",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B3A5C;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #5A5A5A;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .phase-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">🧠 Academic Research Agent — Uno</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">'
    "Dual-engine pipeline: Groq Llama-3.3 (Research + Analysis) → "
    "Claude Sonnet 4 (Academic Writing). Cost-optimized asymmetric architecture."
    "</div>",
    unsafe_allow_html=True,
)

cost_estimate = st.expander("Cost Estimate per Run")
with cost_estimate:
    st.markdown(
        "| Phase | Model | Est. Cost |\n"
        "|-------|-------|----------|\n"
        "| 1. Lead Researcher (Groq) | Llama-3.3-70B | ~$0.09 |\n"
        "| 2. Data Analyst (Groq) | Llama-3.3-70B | ~$0.03 |\n"
        "| 3. Senior Writer (Claude) | Sonnet 4 | ~$0.16 |\n"
        "| **Total** | | **~$0.28** |"
    )

col_input, col_pdf = st.columns([3, 1])

with col_input:
    topic = st.text_input(
        label="Enter your research topic",
        placeholder="e.g., organic farming trends in India, EV market in Europe, solid-state batteries...",
        label_visibility="collapsed",
    )

with col_pdf:
    uploaded_pdf = st.file_uploader(
        "Reference PDF (optional)",
        type=["pdf"],
    )

if st.button("🚀  Research Now", type="primary", use_container_width=True):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            with st.spinner(
                f"Phase 1/3: Lead Researcher (Groq) — Searching Tavily, ArXiv, and RAG...\n\n"
                f"This involves 3 agents (Researcher → Analyst → Writer) "
                f"and takes 3-8 minutes..."
            ):
                if uploaded_pdf is not None:
                    from rag_pipeline import ResearchRAG
                    rag = ResearchRAG()
                    rag.clear()
                    pdf_bytes = uploaded_pdf.read()
                    chunk_count = rag.ingest_pdf_bytes(pdf_bytes)
                    st.info(f"RAG: Ingested {chunk_count} chunks from '{uploaded_pdf.name}'")

                result = run_crew(topic)

            st.session_state["report"] = result
            st.session_state["topic"] = topic

        except Exception as e:
            error_msg = str(e)
            if "GROQ_API_KEY" in error_msg or "api_key" in error_msg.lower():
                st.error("API Key Error: Add your GROQ_API_KEY and ANTHROPIC_API_KEY to the `.env` file.")
            elif "decommissioned" in error_msg:
                st.error(
                    "Model Error: This Groq model has been decommissioned. "
                    "Update the model in config.py."
                )
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                st.error(
                    "Rate Limit: Groq or Anthropic is throttling requests. "
                    "Wait 30 seconds and try again."
                )
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                st.error(
                    "Connection Error: Unable to reach API. "
                    "Check your internet connection and try again."
                )
            else:
                st.error(f"Research failed:\n\n{error_msg}")

if "report" in st.session_state:
    st.divider()

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader(f"Report: {st.session_state['topic']}")
        st.markdown(st.session_state["report"])

    with col_right:
        st.subheader("Export")

        pdf_bytes = generate_pdf(st.session_state["report"], st.session_state["topic"])

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"research_{st.session_state['topic'].lower().replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.download_button(
            label="Download Markdown",
            data=st.session_state["report"],
            file_name=f"research_{st.session_state['topic'].lower().replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()
        st.caption(
            "Dual-engine pipeline:\n"
            "🔵 Groq Llama-3.3 (Researcher → Analyst)\n"
            "🟣 Claude Sonnet 4 (Writer)\n"
            "Est. cost: ~$0.28 per report"
        )

st.divider()
st.caption(
    "Uno Version  |  Dual-Engine Architecture  |  "
    "Groq + Anthropic  |  Tavily + ArXiv + Local RAG"
)
