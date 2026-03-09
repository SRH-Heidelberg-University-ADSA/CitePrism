import streamlit as st
import pandas as pd
import json
import os

def render_inventory_tab():
    st.header("🗃️ Global Research Database Inventory")
    st.markdown("All stored manuscripts and audit summaries. Select a row to load the data into the Explorer.")
    st.divider()

    # Fetch all documents
    all_docs = st.session_state.db_manager.list_all_documents()

    if not all_docs:
        st.info("The database is currently empty. Please go to 'Upload & Process' to add a manuscript.")
        return

    # Process data for the table
    inventory_list = []
    for doc in all_docs:
        doc_id = doc.get('id')
        stages = st.session_state.pipeline.check_stage_completion(doc_id)
        files = st.session_state.pipeline.get_document_files(doc_id)
        
        score_val = 0
        if stages.get('scored') and files.get('scored') and os.path.exists(files['scored']):
            try:
                with open(files['scored'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    score_val = data.get('overall_relevance_score', 0)
            except: score_val = 0

        inventory_list.append({
            "ID": doc_id,
            "Manuscript Name": doc.get('pdf_filename', 'Unknown'),
            "Progress": f"{'✅' if stages.get('parsed') else '⬜'} {'✅' if stages.get('enriched') else '⬜'} {'✅' if stages.get('scored') else '⬜'}",
            "Quality Score": score_val
        })

    df = pd.DataFrame(inventory_list)

    # --- PAGINATION & SCROLLING CONTROLS ---
    col_sel, col_pag = st.columns([1, 1])
    with col_sel:
        page_size = st.selectbox("Show entries:", options=[10, 25, 50], index=0)
    
    total_records = len(df)
    total_pages = (total_records // page_size) + (1 if total_records % page_size > 0 else 0)
    
    with col_pag:
        current_page = st.number_input(f"Page (Total {total_pages}):", min_value=1, max_value=max(1, total_pages), step=1)

    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    display_df = df.iloc[start_idx:end_idx]

    # --- DATA TABLE WITH SELECTION ---
    # Using st.dataframe with on_select allows users to click anywhere on the row
    selection = st.dataframe(
        display_df,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Manuscript Name": st.column_config.TextColumn("Manuscript Title", width="large"),
            "Progress": st.column_config.TextColumn("P | E | S"),
            "Quality Score": st.column_config.ProgressColumn("Overall Score", min_value=0, max_value=100, format="%d%%"),
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    # --- REDIRECTION LOGIC ---
    selected_indices = selection.selection.rows
    
    if selected_indices:
        # Resolve the actual ID from the paginated slice
        selected_row_data = display_df.iloc[selected_indices[0]]
        selected_id = selected_row_data["ID"]
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn_label = f"🔍 Open Detailed Explorer for: {selected_row_data['Manuscript Name'][:40]}..."
        
        if st.button(btn_label, type="primary", use_container_width=True):
            st.session_state.current_document_id = selected_id
            st.session_state.active_tab = "📊 Audit Data Explorer"
            st.rerun()
    else:
        st.info("💡 Select any row in the table above to enable the Explorer button.")

    # Status KPI footer
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Records", total_records)
    k2.metric("Ready for Audit", len(df[df['Quality Score'] > 0]))
    k3.metric("System Health", "Active")