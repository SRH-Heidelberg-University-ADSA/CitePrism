import streamlit as st
import json
import base64
import os
from datetime import datetime

def display_pdf(pdf_path: str):
    """Display PDF in an iframe."""
    try:
        if not os.path.exists(pdf_path):
            st.error(f"File not found: {pdf_path}")
            return
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

def display_json_pretty(json_data: dict, title: str = "JSON Data"):
    """Display JSON data with syntax highlighting and download button."""
    st.markdown(f"### {title}")
    
    # --- NEW: Calculate and Display Total Reference Count ---
    total_refs = 0
    if 'scored_references' in json_data:
        total_refs = len(json_data['scored_references'])
    elif 'enriched_references' in json_data:
        total_refs = len(json_data['enriched_references'])
    elif 'references_list' in json_data:
        total_refs = len(json_data['references_list'])
        
    if total_refs > 0:
        st.info(f"📚 **Total References Extracted:** {total_refs}")
    # --------------------------------------------------------
    
    with st.expander("📄 Manuscript Metadata", expanded=True):
        st.json(json_data.get('manuscript_metadata') or json_data.get('metadata', {}))
    
    if 'citations_in_text' in json_data:
        with st.expander(f"📎 In-Text Citations ({len(json_data['citations_in_text'])})", expanded=False):
            st.json(json_data['citations_in_text'])
            
    if 'references_list' in json_data:
        with st.expander(f"📚 Parsed References ({len(json_data['references_list'])})", expanded=False):
            st.json(json_data['references_list'])
            
    if 'enriched_references' in json_data:
        with st.expander(f"🔍 External Enrichment ({len(json_data['enriched_references'])})", expanded=False):
            st.json(json_data['enriched_references'])
            
    if 'scored_references' in json_data:
        with st.expander(f"⭐ Scored Results ({len(json_data['scored_references'])})", expanded=True):
            st.json(json_data['scored_references'])
            
    json_str = json.dumps(json_data, indent=2)
    st.download_button(
        "📥 Download Analysis JSON", 
        data=json_str, 
        file_name=f"citeprism_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 
        mime="application/json"
    )

def show_pipeline_progress(stages_completed: dict):
    """Display pipeline progress icons."""
    st.markdown("### Pipeline Status")
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