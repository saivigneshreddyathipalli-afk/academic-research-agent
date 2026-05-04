# web_app.py
# ──────────────────────────────────────────────────────────────────────────────
# Streamlit web interface for the Academic Multi-Agent Research System.
#
# Flow:
#   1. User enters a research topic
#   2. Streamlit calls crew.run_crew(topic) — Researcher → Analyst → Writer
#   3. The markdown report is displayed via st.markdown()
#   4. The PDF is generated via pdf_builder and offered for download
# ──────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
import streamlit as st

from crew import run_crew
from pdf_builder import generate_pdf

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="Academic Research Agent",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🧠 Academic Research Agent</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">'
    "Three-agent AI system that investigates any topic and produces a "
    "hybrid IMRaD academic report with inline citations, literature review, "
    "and formal references. Researcher → Analyst → Writer."
    "</div>",
    unsafe_allow_html=True,
)

# ── Input Section ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    topic = st.text_input(
        label="Enter your research topic",
        placeholder="e.g., organic farming trends in India, EV market in Europe, solid-state batteries...",
        label_visibility="collapsed",
    )

    # ── Trigger Research ────────────────────────────────────────────────
    if st.button("🚀  Research Now", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Please enter a research topic.")
        else:
            try:
                with st.spinner(
                    f"Researching: **{topic}**\n\n"
                    f"This involves 3 agents (Researcher → Analyst → Writer) "
                    f"and takes 3-6 minutes..."
                ):
                    result = run_crew(topic)

                # Store in session state so it persists across re-runs
                st.session_state["report"] = result
                st.session_state["topic"] = topic

            except Exception as e:
                error_msg = str(e)
                if "GROQ_API_KEY" in error_msg or "api_key" in error_msg.lower():
                    st.error("API Key Error: Add your GROQ_API_KEY to the `.env` file.")
                elif "decommissioned" in error_msg:
                    st.error(
                        "Model Error: This Groq model has been decommissioned. "
                        "Update the model in config.py."
                    )
                elif "rate limit" in error_msg.lower() or "429" in error_msg:
                    st.error(
                        "Rate Limit: Groq is throttling requests. "
                        "Wait 30 seconds and try again."
                    )
                elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                    st.error(
                        "Connection Error: Unable to reach Groq API. "
                        "Check your internet connection and try again."
                    )
                else:
                    st.error(f"Research failed:\n\n{error_msg}")

# ── Results Section ──────────────────────────────────────────────────────────
if "report" in st.session_state:
    st.divider()

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader(f"Report: {st.session_state['topic']}")
        st.markdown(st.session_state["report"])

    with col_right:
        st.subheader("Export")

        # Generate PDF
        pdf_bytes = generate_pdf(st.session_state["report"], st.session_state["topic"])

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"research_{st.session_state['topic'].lower().replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # Download raw markdown too
        st.download_button(
            label="Download Markdown",
            data=st.session_state["report"],
            file_name=f"research_{st.session_state['topic'].lower().replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()
        st.caption(
            "3-agent pipeline:\n"
            "Researcher (search) → Analyst (evaluate) → Writer (IMRaD report)"
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Powered by CrewAI + Groq Llama 3.3  |  "
    "DuckDuckGo Web Search  |  "
    "Hybrid IMRaD Academic Format"
)
