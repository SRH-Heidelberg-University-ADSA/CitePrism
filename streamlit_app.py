"""
CitePrism Streamlit Application
================================
Main entry point for CitePrism citation audit pipeline.
Features a welcome landing page and a modularized UI structure.
"""

import streamlit as st
from pathlib import Path
import sys
import logging

# Add project root to path to ensure modules are found
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Initialize logging FIRST via logger.py before any other imports.
# This sets root logger level from Config.LOG_LEVEL (.env / settings.py).
# DO NOT call logging.basicConfig() anywhere -- it would override this.
from src.utils.logger import setup_logger
setup_logger()

from database_manager import DatabaseManager
from pipeline_orchestrator import PipelineOrchestrator
from config.settings import Config

# Import modular UI components
from ui.tabs.tab_1_upload import render_upload_tab
from ui.tabs.tab_2_raw import render_raw_data_tab
from ui.tabs.tab_3_logs import render_logs_tab
from ui.tabs.tab_4_ide import render_analyst_ide
from ui.tabs.tab_5_advanced import render_advanced_features
from ui.tabs.tab_6_evaluate import render_evaluation_tab

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="CitePrism - Citation Audit",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for CitePrism Branding
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; margin-bottom: 1rem; }
    .status-box { padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; font-size: 0.9rem; }
    .status-success { background-color: #d4edda; border-left: 5px solid #28a745; color: #155724; }
    .status-warning { background-color: #fff3cd; border-left: 5px solid #ffc107; color: #856404; }
    .status-danger { background-color: #f8d7da; border-left: 5px solid #dc3545; color: #721c24; }
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
            st.session_state.db_manager, st.session_state.config
        )
    if 'current_document_id' not in st.session_state:
        st.session_state.current_document_id = None
        
    if 'app_started' not in st.session_state:
        st.session_state.app_started = False
        
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "📤 Upload & Process"

def render_welcome_page():
    """Renders the initial landing page with a 3-column layout."""
    
    # Custom CSS for the Lightning Arrow Animation
    st.markdown("""
    <style>
    @keyframes lightning-strike {
        0% { text-shadow: 0 0 5px #f1c40f; transform: translateY(0px); opacity: 0.8; }
        50% { text-shadow: 0 0 25px #f39c12, 0 0 45px #e67e22, 0 0 60px #e74c3c; transform: translateY(15px) scale(1.1); opacity: 1; color: #f39c12;}
        100% { text-shadow: 0 0 5px #f1c40f; transform: translateY(0px); opacity: 0.8; }
    }
    .lightning-wrapper {
        text-align: center;
        margin-top: 140px; /* Vertically centers it relative to the main card */
        margin-bottom: 25px;
    }
    .lightning-icon {
        font-size: 70px;
        display: inline-block;
        animation: lightning-strike 1s infinite ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    
    # Fetch historical data from DB
    documents = st.session_state.db_manager.list_all_documents()
    total_audits = len(documents) if documents else 0
    gemini_model = getattr(st.session_state.config, 'GOOGLE_MODEL', 'gemini-2.5-flash')
    
    # --- 3-COLUMN LAYOUT: Left (KPIs) | Center (Card) | Right (Button) ---
    col_left, col_center, col_right = st.columns([1, 1.8, 1], gap="large")
    
    # LEFT COLUMN: System KPIs
    with col_left:
        st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #1f77b4; margin-bottom: 20px; text-align: left;'>System Status</h4>", unsafe_allow_html=True)
        st.metric(label="Total Historical Audits", value=total_audits, delta="Stored in Local DB")
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(label="Core AI Engine", value=gemini_model, delta="High-Speed Extraction")
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(label="Scoring Paradigm", value="Hybrid", delta="LLM (60%) + Vectors (40%)")

    # CENTER COLUMN: Project Details Presentation Card
    with col_center:
        welcome_html = (
            '<div style="background-color: #ffffff; padding: 40px 30px; border-radius: 12px; '
            'box-shadow: 0 4px 15px rgba(0,0,0,0.08); text-align: center; border: 1px solid #eaeaea; margin-bottom: 25px;">'
            '<h1 style="color: #1f77b4; font-size: 2.8rem; margin-bottom: 5px;">📚 CitePrism</h1>'
            '<h3 style="color: #333; font-size: 1.3rem; font-weight: 600; margin-top: 0; margin-bottom: 25px; line-height: 1.4;">'
            'LLM-Driven Audit of Self-Citations and Reference Relevance'
            '</h3>'
            '<hr style="border: none; border-top: 1px solid #f0f0f0; margin-bottom: 30px;">'
            '<p style="font-size: 1.15rem; color: #555; margin-bottom: 25px;"><b>Subject:</b> Case Study 1</p>'
            '<div style="margin: 20px 0;">'
            '<p style="font-size: 1.15rem; color: #555; margin-bottom: 8px;"><b>Contributors:</b></p>'
            '<p style="font-size: 1.1rem; color: #222; margin: 0; line-height: 1.8; font-weight: 500;">'
            'Budanur Madappa Darshan Gowda<br>Gowrika Mahesh<br>Kavana Gopladevarahalli Papegowda<br>Prajwal Basavaraj'
            '</p>'
            '</div>'
            '<div style="margin-top: 35px;">'
            '<p style="font-size: 1.15rem; color: #555; margin-bottom: 8px;"><b>Under the Supervision of:</b></p>'
            '<p style="font-size: 1.1rem; color: #222; margin: 0; font-weight: 500;">Prof. Dr. Melded Jalali</p>'
            '</div>'
            '</div>'
        )
        st.markdown(welcome_html, unsafe_allow_html=True)

    # RIGHT COLUMN: Animated Call-To-Action Button
    with col_right:
        # Renders the animated lightning bolt pointing at the button
        st.markdown("""
        <div class='lightning-wrapper'>
            <span class='lightning-icon'>⚡👇</span>
        </div>
        """, unsafe_allow_html=True)
        
        # The Action Button
        if st.button("🚀 Start Data Mining", use_container_width=True, type="primary"):
            st.session_state.app_started = True
            st.rerun()

def render_main_app():
    """Renders the main functional dashboard."""
    st.markdown('<h1 class="main-header">📚 CitePrism: Citation Audit System</h1>', unsafe_allow_html=True)
    st.markdown("*LLM-Driven Analysis of Citation Relevance and Self-Citation Risks*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Manager")
        documents = st.session_state.db_manager.list_all_documents()
        
        if documents:
            st.subheader(f"Recent Audits ({len(documents)})")
            
            # Map doc IDs to clean display names
            doc_options = {doc['id']: f"ID {doc['id']} - {doc['pdf_filename'][:25]}" for doc in documents}
            doc_ids = list(doc_options.keys())
            
            # Find index of current document, default to None if no active doc
            current_index = doc_ids.index(st.session_state.current_document_id) if st.session_state.current_document_id in doc_ids else None
            
            # Callback function: triggers immediately when a new doc is selected
            def on_document_select():
                selected_id = st.session_state.sidebar_doc_selector
                if selected_id:
                    st.session_state.current_document_id = selected_id
                    st.session_state.active_tab = "📊 Audit Data Explorer"
            
            st.selectbox(
                "Select a Manuscript to load:",
                options=doc_ids,
                format_func=lambda x: doc_options[x],
                index=current_index,
                key="sidebar_doc_selector",
                on_change=on_document_select,
                placeholder="Choose a document..."
            )
            
            # Optional Delete Button for the currently selected record
            if st.session_state.current_document_id:
                if st.button("🗑 Delete Selected Record"):
                    st.session_state.db_manager.delete_document(st.session_state.current_document_id)
                    st.session_state.current_document_id = None
                    st.rerun()
        else:
            st.info("No audit history found. Upload a manuscript to begin.")
            
        st.markdown("---")
        if st.button("⬅️ Back to Welcome Screen"):
            st.session_state.app_started = False
            st.rerun()
    
    # --- REORDERED TABS ---
    tabs = [
        "📤 Upload & Process", 
        "📊 Audit Data Explorer", 
        "🕵️ Analyst IDE", 
        "🎓 Advanced Features",
        "📈 Evaluation Metrics", # <--- ADDED TAB 6
        "📋 System Logs"  
    ]
    
    selected_tab = st.radio(
        "Navigation", 
        tabs, 
        index=tabs.index(st.session_state.active_tab) if st.session_state.active_tab in tabs else 0, 
        horizontal=True, 
        label_visibility="collapsed"
    )
    
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab
        st.rerun()
        
    st.markdown("---")
    
    # Route to UI based on the tab names
    if st.session_state.active_tab == "📤 Upload & Process":
        render_upload_tab()
    elif st.session_state.active_tab == "📊 Audit Data Explorer":
        render_raw_data_tab()
    elif st.session_state.active_tab == "🕵️ Analyst IDE":
        render_analyst_ide()
    elif st.session_state.active_tab == "🎓 Advanced Features":
        render_advanced_features()
    elif st.session_state.active_tab == "📈 Evaluation Metrics": # <--- ADDED ROUTING FOR TAB 6
        render_evaluation_tab()
    elif st.session_state.active_tab == "📋 System Logs":
        render_logs_tab()

def main():
    init_session_state()
    
    if not st.session_state.app_started:
        render_welcome_page()
    else:
        render_main_app()

if __name__ == "__main__":
    main()