import asyncio
import uuid
import streamlit as st
from fpdf import FPDF

from open_deep_research.graph import run_graph


def generate_report(topic: str) -> str:
    """Run the graph workflow and return the final report."""
    config = {"thread_id": str(uuid.uuid4())}
    state = asyncio.run(run_graph(topic, config))
    return state.get("final_report", "")


def report_to_pdf(report: str) -> bytes:
    """Convert a markdown report string to a PDF and return the bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in report.splitlines():
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest="S").encode("latin-1")


st.title("Open Deep Research")

question = st.text_input("Enter your research question")

if st.button("Run Research"):
    if not question:
        st.warning("Please enter a question to research.")
    else:
        with st.spinner("Generating report..."):
            report = generate_report(question)
        st.session_state["report"] = report
        st.markdown(report)

if "report" in st.session_state:
    pdf_bytes = report_to_pdf(st.session_state["report"])
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name="report.pdf",
        mime="application/pdf",
    )
