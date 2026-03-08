import streamlit as st
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent.parent


def render_metric_card(title, value, subtitle=""):
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin:0; color:#94a3b8; font-size:0.92rem;">{title}</p>
        <h2 style="margin:0.35rem 0 0.15rem 0; color:#f8fafc;">{value}</h2>
        <p style="margin:0; color:#64748b; font-size:0.82rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_upload_tab():
    st.markdown("## Upload Manuscript")
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-0.3rem;'>Upload your manuscript PDF and run the full citation audit workflow.</p>",
        unsafe_allow_html=True
    )

    # Top summary strip
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Pipeline", "Ready", "Waiting for manuscript")
    with m2:
        render_metric_card("Input Type", "PDF", "Research manuscript")
    with m3:
        render_metric_card("Audit Engine", "Active", "Pipeline initialized")
    with m4:
        render_metric_card("Storage", "Local DB", "Results persisted")

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Upload + workflow preview
    col1, col2 = st.columns([1.7, 1])

    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3 style="margin-top:0;"> Upload Research Paper</h3>
            <p style="color:#94a3b8; margin-bottom:0.8rem;">
                Drag and drop a manuscript PDF to begin parsing, enrichment, scoring, and audit analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a research paper (PDF)",
            type=["pdf"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            st.success(f"Loaded file: {uploaded_file.name}")

    with col2:
        st.markdown("""
        <div class="custom-card" style="min-height: 220px;">
            <h3 style="margin-top:0;"> Workflow Preview</h3>
            <div style="color:#cbd5e1; line-height:1.9;">
                <div>1. Save uploaded manuscript</div>
                <div>2. Parse text and references</div>
                <div>3. Enrich citation metadata</div>
                <div>4. Compute relevance scores</div>
                <div>5. Open detailed audit explorer</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        save_path = PROJECT_ROOT / "data" / "raw_pdfs"
        save_path.mkdir(parents=True, exist_ok=True)
        temp_pdf_path = save_path / uploaded_file.name

        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # Processing options
        st.markdown("""
        <div class="custom-card">
            <h3 style="margin-top:0;">⚙️ Processing Options</h3>
            <p style="color:#94a3b8; margin-bottom:0.6rem;">
                Choose which stages to rerun before starting the citation audit.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin-top:0; margin-bottom:0.2rem;"> Parse</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">Rebuild manuscript parsing output.</p>
            </div>
            """, unsafe_allow_html=True)
            force_parse = st.checkbox(" Parse PDF")

        with c2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin-top:0; margin-bottom:0.2rem;"> Enrich</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">Refresh metadata and enrichment layers.</p>
            </div>
            """, unsafe_allow_html=True)
            force_enrich = st.checkbox(" Enrich")

        with c3:
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin-top:0; margin-bottom:0.2rem;"> Score</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">Recompute citation scoring.</p>
            </div>
            """, unsafe_allow_html=True)
            force_score = st.checkbox(" Score")

        with c4:
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin-top:0; margin-bottom:0.2rem;"> Reset</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">Force full pipeline rerun.</p>
            </div>
            """, unsafe_allow_html=True)
            force_all = st.checkbox(" Reset All")

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

        # Action row
        a1, a2 = st.columns([1.2, 2.2])

        with a1:
            start_audit = st.button(" Start Citation Audit", type="primary", use_container_width=True)

        with a2:
            st.markdown("""
            <div class="custom-card" style="padding: 0.95rem 1rem;">
                <p style="margin:0; color:#cbd5e1;">
                    The audit will save the manuscript, run the configured pipeline stages, and then redirect you to the detailed audit results.
                </p>
            </div>
            """, unsafe_allow_html=True)

        if start_audit:
            print("\n" + "=" * 60)
            print(f" STARTING CITEPRISM AUDIT FOR: {uploaded_file.name}")
            print("=" * 60)

            st.markdown("""
            <div class="custom-card">
                <h3 style="margin-top:0;"> Processing Status</h3>
                <p style="margin-bottom:0.7rem; color:#94a3b8;">
                    Running manuscript analysis, citation extraction, enrichment, and scoring.
                </p>
            </div>
            """, unsafe_allow_html=True)

            progress_bar = st.progress(0, text="Initializing Audit Pipeline...")

            with st.spinner("Analyzing manuscript and citations..."):
                results = st.session_state.pipeline.process_document(
                    temp_pdf_path,
                    progress_bar=progress_bar,
                    force_parse=force_parse or force_all,
                    force_enrich=force_enrich or force_all,
                    force_score=force_score or force_all
                )

                if results["success"]:
                    progress_bar.progress(1.0, text="Audit Completed Successfully!")
                    st.success("Audit Completed!")

                    print(f" AUDIT COMPLETE: {uploaded_file.name}")
                    print(f"   Document ID: {results['document_id']}")
                    print("=" * 60 + "\n")

                    st.session_state.current_document_id = results["document_id"]

                    if "audit_complete" not in st.session_state:
                        st.session_state.audit_complete = True
                    else:
                        st.session_state.audit_complete = True

                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.latest_audit_result = results

                    # Keep this for your existing app flow
                    st.session_state.active_tab = " Audit Data Explorer"

                    st.rerun()
                else:
                    st.error("Audit Pipeline Failed.")

                    print(f" AUDIT FAILED: {uploaded_file.name}")
                    for err in results.get("errors", []):
                        print(f"   ERROR: {err}")
                    print("=" * 60 + "\n")

                    if results.get("errors"):
                        st.markdown("""
                        <div class="custom-card">
                            <h4 style="margin-top:0;">Failure Details</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for err in results.get("errors", []):
                            st.warning(err)

    else:
        st.markdown("""
        <div class="custom-card">
            <h3 style="margin-top:0;">📘 Getting Started</h3>
            <p style="color:#94a3b8; margin-bottom:0.5rem;">
                Upload a manuscript PDF to unlock processing options and run the audit pipeline.
            </p>
            <p style="color:#cbd5e1; margin-bottom:0;">
                Once the run completes, continue to the Audit Data Explorer for detailed outputs.
            </p>
        </div>
        """, unsafe_allow_html=True)