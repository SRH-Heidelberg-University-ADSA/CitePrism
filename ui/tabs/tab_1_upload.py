import streamlit as st
import os
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent.parent

def render_upload_tab():
    st.header("Upload Manuscript")
    uploaded_file = st.file_uploader("Choose a research paper (PDF)", type=['pdf'])
    
    if uploaded_file:
        save_path = PROJECT_ROOT / "data" / "raw_pdfs"
        save_path.mkdir(parents=True, exist_ok=True)
        temp_pdf_path = save_path / uploaded_file.name
        
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Processing Options
        st.markdown("### ⚙️ Processing Options")
        col1, col2, col3, col4 = st.columns(4)
        force_parse = col1.checkbox("🔄 Re-Parse PDF")
        force_enrich = col2.checkbox("🔄 Re-Enrich")
        force_score = col3.checkbox("🔄 Re-Score")
        force_all = col4.checkbox("🔥 Reset All")
        
        if st.button("🚀 Start Citation Audit", type="primary"):
            
            # --- PRINT TO TERMINAL ---
            print("\n" + "="*60)
            print(f"🚀 STARTING CITEPRISM AUDIT FOR: {uploaded_file.name}")
            print("="*60)
            
            progress_bar = st.progress(0, text="Initializing Audit Pipeline...")
            
            with st.spinner("Analyzing manuscript and citations..."):
                results = st.session_state.pipeline.process_document(
                    temp_pdf_path, 
                    progress_bar=progress_bar, 
                    force_parse=force_parse or force_all, 
                    force_enrich=force_enrich or force_all, 
                    force_score=force_score or force_all
                )
                
                if results['success']:
                    progress_bar.progress(1.0, text="Audit Completed Successfully!") 
                    st.success("Audit Completed!")
                    
                    # --- PRINT TO TERMINAL ---
                    print(f"✅ AUDIT COMPLETE: {uploaded_file.name}")
                    print(f"   Document ID: {results['document_id']}")
                    print("="*60 + "\n")
                    
                    # Update session state with the new document ID
                    st.session_state.current_document_id = results['document_id']
                    
                    # ---> UPDATE THIS EXACT STRING <---
                    st.session_state.active_tab = "📊 Audit Data Explorer" 
                    
                    # Rerun the app to reflect changes
                    st.rerun()
                else:
                    st.error("Audit Pipeline Failed.")
                    
                    # --- PRINT TO TERMINAL ---
                    print(f"❌ AUDIT FAILED: {uploaded_file.name}")
                    for err in results.get('errors', []):
                        print(f"   ERROR: {err}")
                    print("="*60 + "\n")