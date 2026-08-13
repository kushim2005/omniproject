# ============================================================
# OmniBrain — Week 4
# Member 4: Ravi
# Task: Citation UI / PDF Page Viewer
# ============================================================

import streamlit as st
import requests
import time
import os
import base64
import io

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="OmniBrain — Citation Viewer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS: Premium Citation Cards ───────────────────────
st.markdown("""
<style>
.citation-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-family: 'Inter', sans-serif;
}
.citation-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.citation-text {
    font-size: 0.9rem;
    color: #e2e8f0;
    line-height: 1.6;
    border-left: 2px solid #4f46e5;
    padding-left: 10px;
    margin: 6px 0;
}
.page-badge {
    display: inline-block;
    background: #4f46e5;
    color: white;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 600;
    margin-top: 8px;
}
.score-badge {
    display: inline-block;
    background: #065f46;
    color: #6ee7b7;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 600;
    margin-top: 8px;
    margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Helper: Render Citation Card ─────────────────────────────
def render_citation_card(idx: int, page: int, text: str, source: str = "", score: float = None):
    score_html = ""
    if score is not None:
        score_html = f'<span class="score-badge">Score: {score:.3f}</span>'

    st.markdown(f"""
    <div class="citation-card">
        <div class="citation-header">Citation {idx} &nbsp;|&nbsp; {source or "Document"}</div>
        <div class="citation-text">{text}</div>
        <span class="page-badge">Page {page}</span>
        {score_html}
    </div>
    """, unsafe_allow_html=True)

# ── Helper: PDF Page Viewer ───────────────────────────────────
def render_pdf_page_viewer(pdf_bytes: bytes, page_num: int = 1):
    """Renders a specific page of a PDF inline using base64 encoding."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        page_num = max(1, min(page_num, total_pages))
        page = doc[page_num - 1]
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        b64 = base64.b64encode(img_bytes).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%; border-radius:8px; border:1px solid #4f46e5;" />',
            unsafe_allow_html=True
        )
        return total_pages
    except ImportError:
        st.warning("PyMuPDF not installed. Run `pip install PyMuPDF` to enable PDF page viewer.")
        return 1
    except Exception as e:
        st.error(f"Could not render PDF page: {e}")
        return 1


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=56)
    st.title("OmniBrain")
    st.caption("Week 4 — Citation & PDF Viewer")
    st.markdown("---")

    # PDF Upload
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save PDF bytes for the page viewer
        st.session_state["pdf_bytes"] = uploaded_file.read()
        uploaded_file.seek(0)

        if st.button("Ingest Document", use_container_width=True):
            with st.spinner("Uploading and starting ingestion..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)

                    if response.status_code == 202:
                        job_id = response.json()["job_id"]
                        st.success(f"Accepted! Job: `{job_id[:8]}...`")
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        while True:
                            status_res = requests.get(f"{API_URL}/upload/status/{job_id}")
                            if status_res.status_code == 200:
                                status_data = status_res.json()
                                state = status_data["status"]
                                status_text.text(f"{state.upper()} — {status_data.get('message', '')}")

                                if state == "completed":
                                    progress_bar.progress(100)
                                    st.balloons()
                                    st.success("Document indexed!")
                                    break
                                elif state == "failed":
                                    st.error(status_data.get("error", "Unknown error"))
                                    break
                                else:
                                    progress_bar.progress(50)
                            time.sleep(2)
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.markdown("---")

    # PDF Page Viewer Panel
    if "pdf_bytes" in st.session_state and "cited_pages" in st.session_state:
        st.header("PDF Page Viewer")
        cited = sorted(set(st.session_state["cited_pages"]))
        selected_page = st.selectbox(
            "Jump to cited page:",
            options=cited,
            format_func=lambda p: f"Page {p}"
        )
        total = render_pdf_page_viewer(st.session_state["pdf_bytes"], selected_page)
        st.caption(f"Showing page {selected_page} of {total}")


# ── MAIN CHAT INTERFACE ───────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("## Ask OmniBrain")
    st.caption("Powered by Self-RAG + LangGraph + Langfuse tracing")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cited_pages" not in st.session_state:
        st.session_state.cited_pages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Query Input
    if prompt := st.chat_input("Ask about the document content..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("Running agentic pipeline...", expanded=True) as status_box:
                st.write("Supervisor Agent: Analyzing query intent...")
                time.sleep(0.5)
                st.write("Search/Vision Agent: Retrieving relevant chunks...")
                time.sleep(0.5)
                st.write("Self-RAG Grader: Validating document relevance...")

                try:
                    payload = {
                        "query": prompt,
                        "top_k": 3,
                        "search_text": True,
                        "search_images": True
                    }
                    response = requests.post(f"{API_URL}/query", json=payload)
                    response.raise_for_status()
                    data = response.json()

                    text_results = data.get("text_results", [])
                    image_results = data.get("image_results", [])

                    status_box.update(
                        label=f"Found {len(text_results)} citations + {len(image_results)} images",
                        state="complete",
                        expanded=False
                    )

                    # Display clean answer
                    answer_text = f"Based on **{len(text_results)}** cited document passages:\n\n"
                    st.markdown(answer_text)

                    # Store cited pages for PDF viewer
                    new_pages = [r["page"] for r in text_results]
                    st.session_state["cited_pages"] = list(
                        set(st.session_state.get("cited_pages", []) + new_pages)
                    )

                    # Display images
                    if image_results:
                        st.markdown("**Visual References:**")
                        for img in image_results:
                            try:
                                st.image(img["image_path"],
                                         caption=f"Page {img['page']}",
                                         width=420)
                            except Exception:
                                st.warning(f"Image not found: {img['image_path']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text
                    })

                except Exception as e:
                    status_box.update(label="Pipeline error", state="error", expanded=True)
                    st.error(f"Failed to query backend: {e}")

with col2:
    st.markdown("## Citations")
    st.caption("Sources retrieved for the latest query")

    # Show citation cards for all retrieved results
    if "last_text_results" not in st.session_state:
        st.session_state["last_text_results"] = []

    # Re-fetch last results on page rerun using session state
    if st.session_state.get("last_text_results"):
        for idx, r in enumerate(st.session_state["last_text_results"], 1):
            render_citation_card(
                idx=idx,
                page=r.get("page", 0),
                text=r.get("text", "")[:240] + "...",
                source=r.get("source", "Document"),
                score=r.get("score")
            )
    else:
        st.info("Citations will appear here after you submit a query.")
