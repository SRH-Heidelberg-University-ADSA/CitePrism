import streamlit as st
import os
import logging
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ── Level colours (light-theme friendly) ──────────────────────────────────────
_LEVEL_COLOUR = {
    "DEBUG":    "#888888",
    "INFO":     "#1a1a2e",
    "WARNING":  "#b45309",
    "ERROR":    "#b91c1c",
    "CRITICAL": "#7f1d1d",
}

_LEVEL_PREFIX = {
    "DEBUG":    "⚙",
    "INFO":     "ℹ",
    "WARNING":  "⚠",
    "ERROR":    "✖",
    "CRITICAL": "🔥",
}

_BOX_CSS = (
    "background:#f8f9fa;"
    "border:1px solid #dee2e6;"
    "border-radius:8px;"
    "padding:12px 16px;"
    "height:320px;"
    "overflow-y:scroll;"
    "scrollbar-width:thin;"
    "scrollbar-color:#adb5bd #f8f9fa;"
    "font-family:'Courier New',monospace;"
    "font-size:0.78rem;"
    "line-height:1.55;"
)


def _build_html(logs):
    lines = ""
    for entry in logs:
        colour = _LEVEL_COLOUR.get(entry["level"], "#1a1a2e")
        prefix = _LEVEL_PREFIX.get(entry["level"], "•")
        safe = (
            entry["msg"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        lines += (
            f'<div style="color:{colour};margin:2px 0;">'
            f'<span style="opacity:0.45;">{prefix}</span> {safe}'
            f'</div>\n'
        )

    return (
        "<style>"
        "  .citeprism-logbox::-webkit-scrollbar{width:8px}"
        "  .citeprism-logbox::-webkit-scrollbar-track{background:#e9ecef;border-radius:4px}"
        "  .citeprism-logbox::-webkit-scrollbar-thumb{background:#adb5bd;border-radius:4px}"
        "  .citeprism-logbox::-webkit-scrollbar-thumb:hover{background:#6c757d}"
        "</style>"
        f'<div class="citeprism-logbox" style="{_BOX_CSS}">'
        f"{lines}"
        '<div id="cplog-end"></div>'
        "</div>"
        "<script>"
        "(function(){var e=document.getElementById('cplog-end');if(e)e.scrollIntoView({behavior:'instant'});})();"
        "</script>"
    )


def _empty_box(msg="Logs will appear here once the audit starts..."):
    return (
        f'<div class="citeprism-logbox" style="{_BOX_CSS}color:#adb5bd;font-style:italic;">'
        f"{msg}</div>"
    )


# ── Live-flushing log handler ──────────────────────────────────────────────────

class _LiveLogHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self._placeholder = placeholder

    def emit(self, record):
        if "ui_logs" not in st.session_state:
            st.session_state.ui_logs = []
        st.session_state.ui_logs.append({
            "level": record.levelname,
            "msg":   self.format(record),
        })
        try:
            self._placeholder.markdown(
                _build_html(st.session_state.ui_logs),
                unsafe_allow_html=True,
            )
        except Exception:
            pass


def _attach(placeholder):
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, _LiveLogHandler)]
    handler = _LiveLogHandler(placeholder)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    return handler


def _detach(handler):
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if h is not handler]


# ── Main tab renderer ──────────────────────────────────────────────────────────

def render_upload_tab():
    st.header("Upload Manuscript")
    uploaded_file = st.file_uploader("Choose a research paper (PDF)", type=["pdf"])

    if not uploaded_file:
        return

    save_path = PROJECT_ROOT / "data" / "raw_pdfs"
    save_path.mkdir(parents=True, exist_ok=True)
    temp_pdf_path = save_path / uploaded_file.name

    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Processing Options
    st.markdown("### ⚙️ Processing Options")
    col1, col2, col3, col4 = st.columns(4)
    force_parse  = col1.checkbox("🔄 Re-Parse PDF")
    force_enrich = col2.checkbox("🔄 Re-Enrich")
    force_score  = col3.checkbox("🔄 Re-Score")
    force_all    = col4.checkbox("🔥 Reset All")

    st.markdown("<br>", unsafe_allow_html=True)

    # Button ABOVE the log box
    start_clicked = st.button("🚀 Start Citation Audit", type="primary")

    # Log box
    st.markdown("### 🖥️ Extraction Logs")
    log_placeholder = st.empty()

    existing_logs = st.session_state.get("ui_logs", [])
    if existing_logs:
        log_placeholder.markdown(_build_html(existing_logs), unsafe_allow_html=True)
    else:
        log_placeholder.markdown(_empty_box(), unsafe_allow_html=True)

    if not start_clicked:
        return

    # Reset and wire up live handler before pipeline starts
    st.session_state.ui_logs = []
    log_placeholder.markdown(_empty_box("Starting audit..."), unsafe_allow_html=True)
    ui_handler = _attach(log_placeholder)

    print("\n" + "=" * 60)
    print(f"🚀 STARTING CITEPRISM AUDIT FOR: {uploaded_file.name}")
    print("=" * 60)

    progress_bar = st.progress(0, text="Initializing Audit Pipeline...")

    try:
        results = st.session_state.pipeline.process_document(
            temp_pdf_path,
            progress_bar=progress_bar,
            force_parse=force_parse or force_all,
            force_enrich=force_enrich or force_all,
            force_score=force_score or force_all,
        )

        if results["success"]:
            progress_bar.progress(1.0, text="Audit Completed Successfully!")
            st.success("✅ Audit Completed!")
            print(f"✅ AUDIT COMPLETE: {uploaded_file.name}")
            print(f"   Document ID: {results['document_id']}")
            print("=" * 60 + "\n")
            st.session_state.current_document_id = results["document_id"]
            st.session_state.active_tab = "📊 Audit Data Explorer"
            st.rerun()
        else:
            progress_bar.progress(1.0, text="Audit Failed.")
            st.error("❌ Audit Pipeline Failed.")
            print(f"❌ AUDIT FAILED: {uploaded_file.name}")
            for err in results.get("errors", []):
                print(f"   ERROR: {err}")
            print("=" * 60 + "\n")

    finally:
        _detach(ui_handler)