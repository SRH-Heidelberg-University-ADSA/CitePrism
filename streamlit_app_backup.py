"""
CitePrism Streamlit Application
================================
Main UI for CitePrism citation audit pipeline.

Features:
- PDF upload with duplicate detection
- Side-by-side PDF and JSON comparison
- Progress tracking through pipeline stages
- Intelligent caching (no unnecessary API calls)
- Per-stage force reprocess options
"""

import streamlit as st
import json
import base64
from pathlib import Path
import logging
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from database_manager import DatabaseManager
from pipeline_orchestrator import PipelineOrchestrator
from config.settings import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/streamlit_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="CitePrism - Citation Audit",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .status-info {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    .json-container {
        max-height: 600px;
        overflow-y: auto;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = PipelineOrchestrator(
            st.session_state.db_manager,
            st.session_state.config
        )
    
    if 'current_document_id' not in st.session_state:
        st.session_state.current_document_id = None


def display_pdf(pdf_path: str):
    """Display PDF in iframe."""
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}"
                    width="100%" height="800px" type="application/pdf">
            </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")


def display_json_pretty(json_data: dict, title: str = "JSON Data"):
    """Display JSON data with syntax highlighting."""
    st.markdown(f"### {title}")
    
    with st.expander("📄 Metadata", expanded=True):
        metadata = json_data.get('manuscript_metadata') or json_data.get('metadata', {})
        st.json(metadata)
    
    if 'citations_in_text' in json_data:
        with st.expander(f"📎 Citations in Text ({len(json_data['citations_in_text'])})", expanded=False):
            st.json(json_data['citations_in_text'][:5])
            if len(json_data['citations_in_text']) > 5:
                st.info(f"Showing 5 of {len(json_data['citations_in_text'])} citations")
    
    if 'references_list' in json_data:
        with st.expander(f"📚 References ({len(json_data['references_list'])})", expanded=False):
            st.json(json_data['references_list'][:5])
            if len(json_data['references_list']) > 5:
                st.info(f"Showing 5 of {len(json_data['references_list'])} references")
    
    if 'enriched_references' in json_data:
        with st.expander(f"🔍 Enriched References ({len(json_data['enriched_references'])})", expanded=False):
            st.json(json_data['enriched_references'][:3])
            if len(json_data['enriched_references']) > 3:
                st.info(f"Showing 3 of {len(json_data['enriched_references'])} enriched references")
    
    if 'scored_references' in json_data:
        with st.expander(f"⭐ Scored References ({len(json_data['scored_references'])})", expanded=False):
            st.json(json_data['scored_references'][:3])
            if len(json_data['scored_references']) > 3:
                st.info(f"Showing 3 of {len(json_data['scored_references'])} scored references")
    
    json_str = json.dumps(json_data, indent=2)
    st.download_button(
        label="📥 Download Full JSON",
        data=json_str,
        file_name=f"citeprism_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def show_pipeline_progress(stages_completed: dict):
    """Display pipeline progress."""
    st.markdown("### Pipeline Progress")
    
    stages = [
        ('Parsed', stages_completed.get('parsed', False)),
        ('Enriched', stages_completed.get('enriched', False)),
        ('Scored', stages_completed.get('scored', False))
    ]
    
    cols = st.columns(3)
    for col, (stage_name, is_complete) in zip(cols, stages):
        with col:
            if is_complete:
                st.success(f"✅ {stage_name}")
            else:
                st.info(f"⏳ {stage_name}")


def main():
    """Main Streamlit application."""
    init_session_state()
    
    st.markdown('<h1 class="main-header">📚 CitePrism: Citation Audit System</h1>', 
                unsafe_allow_html=True)
    st.markdown("*LLM-Driven Analysis of Citation Relevance and Self-Citations*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Manager")
        
        documents = st.session_state.db_manager.list_all_documents()
        
        if documents:
            st.subheader(f"📚 Documents ({len(documents)})")
            
            for doc in documents:
                with st.expander(f"📄 {doc['pdf_filename'][:30]}..."):
                    st.write(f"**ID:** {doc['id']}")
                    st.write(f"**Title:** {doc['title'] or 'N/A'}")
                    st.write(f"**References:** {doc['num_references'] or 'N/A'}")
                    st.write(f"**Uploaded:** {doc['uploaded_at']}")
                    
                    status_icons = {
                        'parsed': '✅' if doc['status_parsed'] else '⏳',
                        'enriched': '✅' if doc['status_enriched'] else '⏳',
                        'scored': '✅' if doc['status_scored'] else '⏳'
                    }
                    st.write(f"{status_icons['parsed']} Parsed | "
                            f"{status_icons['enriched']} Enriched | "
                            f"{status_icons['scored']} Scored")
                    
                    if st.button(f"📂 Load", key=f"load_{doc['id']}"):
                        st.session_state.current_document_id = doc['id']
                        st.rerun()
        else:
            st.info("No documents yet. Upload a PDF to get started!")
        
        st.markdown("---")
        st.caption("💡 Tip: Use force reprocess options to re-run specific stages!")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 View Results", "📋 Logs"])
    
    # TAB 1: Upload & Process
    with tab1:
        st.header("📤 Upload PDF Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload an academic paper for citation analysis"
        )
        
        if uploaded_file:
            temp_pdf_path = Path("data/raw_pdfs") / uploaded_file.name
            temp_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            pdf_hash = DatabaseManager.compute_pdf_hash(temp_pdf_path)
            
            with st.session_state.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM documents WHERE pdf_hash = ?", (pdf_hash,))
                existing = cursor.fetchone()
            
            if existing:
                st.info(f"📁 This PDF already exists (ID: {existing['id']}). Use options below to reprocess.")
            
            # Per-stage force reprocess options
            st.markdown("### ⚙️ Processing Options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                force_parse = st.checkbox(
                    "🔄 Force Parsing",
                    help="Re-extract text from PDF with LLM"
                )
            
            with col2:
                force_enrich = st.checkbox(
                    "🔄 Force Enrichment",
                    help="Re-fetch metadata from OpenAlex"
                )
            
            with col3:
                force_score = st.checkbox(
                    "🔄 Force Scoring",
                    help="Re-compute relevance scores with LLM"
                )
            
            with col4:
                force_all = st.checkbox(
                    "🔄 Force ALL",
                    help="Reprocess all stages"
                )
            
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.session_state.db_manager._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT id, status_parsed, status_enriched, status_scored FROM documents WHERE pdf_hash = ?", (pdf_hash,))
                        doc_status = cursor.fetchone()
                    
                    if force_all or force_parse or not doc_status or not doc_status['status_parsed']:
                        current_stage = 1
                    elif force_enrich or not doc_status['status_enriched']:
                        current_stage = 2
                    elif force_score or not doc_status['status_scored']:
                        current_stage = 3
                    else:
                        current_stage = 0
                    
                    if current_stage == 0:
                        status_text.text("✅ All stages complete (using cache)")
                        progress_bar.progress(100)
                    elif current_stage == 1:
                        status_text.text("⏳ Stage 1/3: Parsing...")
                        progress_bar.progress(10)
                    elif current_stage == 2:
                        status_text.text("⏳ Stage 2/3: Enriching...")
                        progress_bar.progress(40)
                    elif current_stage == 3:
                        status_text.text("⏳ Stage 3/3: Scoring...")
                        progress_bar.progress(70)
                    
                    results = st.session_state.pipeline.process_document(
                        temp_pdf_path,
                        force_parse=force_parse or force_all,
                        force_enrich=force_enrich or force_all,
                        force_score=force_score or force_all
                    )
                    
                    progress_bar.progress(100)
                    
                    if results['success']:
                        st.success("✅ Processing completed!")
                        st.session_state.current_document_id = results['document_id']
                        
                        st.markdown("### Processing Summary")
                        
                        if results.get('stages_skipped'):
                            st.info(f"⚡ Skipped: {', '.join(results['stages_skipped'])}")
                        
                        if results.get('stages_completed'):
                            st.success(f"✅ Completed: {', '.join(results['stages_completed'])}")
                        
                        st.balloons()
                    else:
                        st.error("❌ Processing failed!")
                        for error in results.get('errors', []):
                            st.error(f"Error: {error}")
    
    # TAB 2: View Results
    with tab2:
        if st.session_state.current_document_id:
            doc_id = st.session_state.current_document_id
            status = st.session_state.db_manager.get_document_status(doc_id)
            files = st.session_state.pipeline.get_document_files(doc_id)
            stages = st.session_state.pipeline.check_stage_completion(doc_id)
            
            st.header(f"📄 {status['pdf_filename']}")
            
            show_pipeline_progress(stages)
            
            st.markdown("---")
            
            available_stages = []
            if stages['parsed']:
                available_stages.append(("Parsed Data", files['parsed']))
            if stages['enriched']:
                available_stages.append(("Enriched Data", files['enriched']))
            if stages['scored']:
                available_stages.append(("Scored Data (Final)", files['scored']))
            
            if available_stages:
                selected_stage = st.selectbox(
                    "Select data to view:",
                    options=[name for name, _ in available_stages]
                )
                
                selected_file = None
                for name, path in available_stages:
                    if name == selected_stage:
                        selected_file = path
                        break
                
                if selected_file:
                    col_pdf, col_json = st.columns(2)
                    
                    with col_pdf:
                        st.markdown("### 📄 Original PDF")
                        display_pdf(files['pdf'])
                    
                    with col_json:
                        st.markdown(f"### 📊 {selected_stage}")
                        try:
                            with open(selected_file, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            display_json_pretty(json_data, selected_stage)
                        except Exception as e:
                            st.error(f"Error loading JSON: {e}")
            else:
                st.warning("⚠️ No data available. Please process a document first.")
        else:
            st.info("👈 Select or upload a document to view results.")
    
    # TAB 3: Logs
    with tab3:
        if st.session_state.current_document_id:
            st.header("📋 Processing Logs")
            
            logs = st.session_state.db_manager.get_processing_logs(
                st.session_state.current_document_id
            )
            
            if logs:
                for log in logs:
                    status_class = {
                        'success': 'status-success',
                        'failed': 'status-warning',
                        'info': 'status-info'
                    }.get(log['status'], 'status-info')
                    
                    st.markdown(f"""
                    <div class="status-box {status_class}">
                        <strong>{log['stage'].upper()}</strong> - {log['status']}<br>
                        <small>{log['timestamp']}</small><br>
                        {log['message'] or ''}<br>
                        {f"<span style='color: red;'>{log['error']}</span>" if log['error'] else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No logs available.")
        else:
            st.info("👈 Select a document to view logs.")


if __name__ == "__main__":
    main()