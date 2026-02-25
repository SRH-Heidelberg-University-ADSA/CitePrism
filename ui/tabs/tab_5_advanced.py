import streamlit as st
import json
import os
import pandas as pd

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

def render_advanced_features():
    if not st.session_state.current_document_id:
        st.info("Select a document to begin.")
        return

    doc_id = st.session_state.current_document_id
    files = st.session_state.pipeline.get_document_files(doc_id)
    
    if files.get('scored') and os.path.exists(files['scored']):
        with open(files['scored'], 'r', encoding='utf-8') as f:
            audit_data = json.load(f)
        scored_refs = audit_data.get('scored_references', [])
        
        st.header("🎓 Student-Added Features")
        st.markdown("Implementation of mandatory custom enhancements for the Applied Data Science Case Study.")

        # Feature 6: Hallucination Guard
        st.markdown("---")
        st.subheader("🛡️ Feature 6: Hallucination & Consistency Guard")
        hallucinations = [r for r in scored_refs if "Mismatch" in r.get('consistency_status', '')]
        
        if hallucinations:
            st.error(f"🚨 Detected {len(hallucinations)} metadata inconsistencies!")
            for h in hallucinations:
                with st.expander(f"Ref {h.get('ref_id')} - {h.get('consistency_status')}"):
                    st.write(f"**Parsed Title (PDF):** {h.get('original_data', {}).get('parsed', {}).get('title')}")
                    st.write(f"**Retrieved Title (API):** {h.get('external_metadata', {}).get('title')}")
                    st.write("**Action Taken:** Flagged automatically to prevent feeding incorrect abstracts to the LLM Scorer.")
        else:
            st.success("✅ No severe metadata hallucinations detected.")

        # Option B: Bias & Diversity Lens
        st.markdown("---")
        st.subheader("🌐 Option B: Bias & Diversity Lens")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Top Cited Venues (Journal Bias)**")
            venues = [r.get('external_metadata', {}).get('venue') or r.get('original_data', {}).get('parsed', {}).get('venue') for r in scored_refs]
            venues = [v for v in venues if v]
            if venues:
                st.bar_chart(pd.Series(venues).value_counts().head(7), color="#1f77b4")
            else:
                st.info("No venue data available.")

        with col_chart2:
            st.markdown("**Top Cited Authors (Network Narrowness)**")
            all_authors = []
            for r in scored_refs:
                all_authors.extend(r.get('original_data', {}).get('parsed', {}).get('authors', []))
            clean_authors = [a.strip() for a in all_authors if a]
            if clean_authors:
                st.bar_chart(pd.Series(clean_authors).value_counts().head(7), color="#ff7f0e")
            else:
                st.info("No author data available.")

        # Option A: Reviewer Mode
        st.markdown("---")
        gemini_model_name = getattr(st.session_state.config, 'GOOGLE_MODEL', 'gemini-2.5-flash')
        st.subheader(f"📝 Option A: Reviewer Mode ({gemini_model_name})")
        threshold = st.slider("Define 'Low Relevance' threshold for Reviewer Report:", 0, 100, 50, key="rev_slider")
        
        if st.button("🤖 Generate Peer-Reviewer Brief with Gemini"):
            problem_refs = [r for r in scored_refs if r.get('RS_final', 0) < threshold or r.get('self_citation', {}).get('is_self_cite')]
            
            if not problem_refs:
                st.success(f"No problematic citations found below threshold {threshold}. The manuscript looks clean!")
            else:
                if not HAS_GEMINI:
                    st.error("google-generativeai is not installed. Please run `pip install google-generativeai`.")
                else:
                    api_key = os.getenv("GOOGLE_API_KEY") or getattr(st.session_state.config, 'GOOGLE_API_KEY', None)
                    if not api_key:
                        st.error("No Google API Key found. Please add GOOGLE_API_KEY to your .env file.")
                    else:
                        with st.spinner(f"Generating Reviewer Brief with {gemini_model_name}..."):
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel(gemini_model_name)
                                
                                prompt = f"""
                                You are an expert academic peer reviewer evaluating a manuscript. 
                                A citation audit tool has flagged {len(problem_refs)} references in the paper that require attention.
                                """
                                for pref in problem_refs:
                                    prompt += f"\n- Ref ID: {pref.get('ref_id')}\n  Title: {pref.get('original_data', {}).get('parsed', {}).get('title')}\n  Relevance: {pref.get('RS_final', 0)}/100\n  Self-Cite: {'Yes' if pref.get('self_citation', {}).get('is_self_cite') else 'No'}\n  Rationale: {pref.get('llm_rationale')}\n"

                                prompt += "\nProvide a professional, 1-page reviewer brief with an executive summary, a breakdown of flags, and recommendations."
                                
                                response = model.generate_content(prompt)
                                st.success("✅ Brief Generated Successfully!")
                                st.markdown("### 📋 AI-Generated Reviewer Brief")
                                st.write(response.text)
                            except Exception as e:
                                st.error(f"Failed to generate brief with Gemini: {e}")
    else:
        st.info("Pipeline not fully complete.")