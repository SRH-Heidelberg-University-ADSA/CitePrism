import streamlit as st
import os
import json
from ui.components.helpers import display_pdf, display_json_pretty, show_pipeline_progress

def render_raw_data_tab():
    if not st.session_state.current_document_id:
        st.info("Audit a document in the first tab to see results here.")
        return

    doc_id = st.session_state.current_document_id
    status = st.session_state.db_manager.get_document_status(doc_id)
    files = st.session_state.pipeline.get_document_files(doc_id)
    stages = st.session_state.pipeline.check_stage_completion(doc_id)
    
    st.header(f"Results for: {status['pdf_filename']}")
    show_pipeline_progress(stages)
    st.markdown("---")
    
    col_pdf, col_json = st.columns([1, 1])
    
    with col_pdf:
        st.subheader("📄 PDF Viewer")
        display_pdf(files['pdf'])
    
    with col_json:
        st.subheader("🔍 JSON Data Inspector")
        
        available_options = []
        if stages.get('parsed') and files.get('parsed') and os.path.exists(files['parsed']):
            available_options.append("Parsed Extraction")
        if stages.get('enriched') and files.get('enriched') and os.path.exists(files['enriched']):
            available_options.append("Metadata Enrichment")
        if stages.get('scored') and files.get('scored') and os.path.exists(files['scored']):
            available_options.append("Final Relevance Scoring")
            
        if not available_options:
            st.info("No JSON data available yet. Please run the pipeline.")
        else:
            view_choice = st.selectbox(
                "Select Pipeline Stage to View:", 
                available_options, 
                index=len(available_options)-1
            )
            
            file_map = {
                "Parsed Extraction": files.get('parsed'),
                "Metadata Enrichment": files.get('enriched'),
                "Final Relevance Scoring": files.get('scored')
            }
            
            selected_file = file_map.get(view_choice)
            
            st.markdown("---")
            if selected_file and os.path.exists(selected_file):
                with open(selected_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                display_json_pretty(data, title=f"Showing: {view_choice}")
            else:
                st.error(f"Could not load the file for {view_choice}.")