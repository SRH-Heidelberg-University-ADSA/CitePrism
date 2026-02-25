import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go  # Added for Data Visualizations

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

def generate_html_report(manuscript_title, tau, kpis, flagged_refs):
    table_rows = ""
    for ref in flagged_refs:
        title = ref.get('original_data', {}).get('parsed', {}).get('title', 'Unknown Title')
        score = ref.get('RS_final', 0)
        rationale = ref.get('llm_rationale', 'No rationale provided.')
        badges = []
        if ref.get('self_citation', {}).get('is_self_cite'): badges.append("👤 Self-Cite")
        if ref.get('quality_flags'): badges.append("⚠️ Issue")
        badge_str = " | ".join(badges) if badges else "None"
        
        table_rows += f"""
        <tr>
            <td>{ref.get('ref_id')}</td>
            <td><strong>{title}</strong></td>
            <td style="color: red; font-weight: bold;">{score}</td>
            <td>{badge_str}</td>
            <td>{rationale}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>CitePrism Audit Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-bottom: 5px; }}
            .meta-info {{ font-size: 14px; color: #666; margin-bottom: 20px; }}
            .kpi-container {{ display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 30px; }}
            .kpi-box {{ flex: 1; min-width: 150px; padding: 15px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; text-align: center; }}
            .kpi-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #1f77b4; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .footer {{ margin-top: 40px; font-size: 12px; color: #777; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>CitePrism Citation Audit Report</h1>
        <h2>Manuscript: {manuscript_title}</h2>
        <div class="meta-info">
            <strong>Date Generated:</strong> {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}<br>
            <strong>Total References Audited:</strong> {kpis['total']} references
        </div>
        
        <div class="kpi-container">
            <div class="kpi-box">
                <div>Threshold (τ)</div>
                <div class="kpi-value">{tau}</div>
            </div>
            <div class="kpi-box">
                <div>Total References</div>
                <div class="kpi-value">{kpis['total']}</div>
            </div>
            <div class="kpi-box">
                <div>Flagged Irrelevant</div>
                <div class="kpi-value" style="color: #dc3545;">{kpis['flagged']}</div>
            </div>
            <div class="kpi-box">
                <div>Self-Citations</div>
                <div class="kpi-value">{kpis['self_cites']}</div>
            </div>
            <div class="kpi-box">
                <div>Quality Issues</div>
                <div class="kpi-value" style="color: #ffc107;">{kpis['issues']}</div>
            </div>
            <div class="kpi-box">
                <div>Missing Abstracts</div>
                <div class="kpi-value" style="color: #6c757d;">{kpis['missing_abstracts']}</div>
            </div>
        </div>

        <h3>🚩 Flagged References (Below Threshold {tau})</h3>
        <table>
            <thead>
                <tr>
                    <th width="5%">ID</th>
                    <th width="30%">Reference Title</th>
                    <th width="10%">Score</th>
                    <th width="15%">Badges</th>
                    <th width="40%">LLM Rationale</th>
                </tr>
            </thead>
            <tbody>
                {table_rows if table_rows else "<tr><td colspan='5' style='text-align:center;'>No flagged references found.</td></tr>"}
            </tbody>
        </table>
    </body>
    </html>
    """
    return html_content

def generate_pdf_report(manuscript_title, tau, kpis, flagged_refs):
    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(31, 119, 180) # Streamlit blue
            self.cell(0, 10, "CitePrism Citation Audit Report", border=False, ln=True, align="C")
            self.ln(5)
            
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    def clean_text(text):
        if not text: return "N/A"
        return str(text).encode('latin-1', 'replace').decode('latin-1')

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(50)
    pdf.cell(0, 8, f"Manuscript: {clean_text(manuscript_title)}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Date Generated: {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}", ln=True)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"Total References Audited: {kpis['total']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Audit KPIs", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"- Relevance Threshold (tau): {tau}", ln=True)
    pdf.cell(0, 6, f"- Total References Found: {kpis['total']}", ln=True)
    pdf.cell(0, 6, f"- Flagged as Irrelevant: {kpis['flagged']} ({(kpis['flagged']/kpis['total']*100):.1f}% of total)", ln=True)
    pdf.cell(0, 6, f"- Total Self-Citations: {kpis['self_cites']} ({(kpis['self_cites']/kpis['total']*100):.1f}% of total)", ln=True)
    pdf.cell(0, 6, f"- Quality Issues Detected: {kpis['issues']}", ln=True)
    pdf.cell(0, 6, f"- Missing Abstracts: {kpis['missing_abstracts']}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Flagged References List (Score < {tau})", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    if not flagged_refs:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 10, "No flagged references found below the threshold.", ln=True)
    else:
        for ref in flagged_refs:
            title = ref.get('original_data', {}).get('parsed', {}).get('title', 'Unknown Title')
            score = ref.get('RS_final', 0)
            rationale = ref.get('llm_rationale', 'No rationale provided.')
            
            badges = []
            if ref.get('self_citation', {}).get('is_self_cite'): badges.append("[Self-Cite]")
            if ref.get('quality_flags'): badges.append("[Issue Flagged]")
            badge_str = ", ".join(badges) if badges else "None"
            
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, clean_text(f"Ref [{ref.get('ref_id')}]: {title}"))
            
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(220, 53, 69) 
            pdf.cell(30, 6, f"Score: {score}")
            pdf.set_text_color(50) 
            pdf.cell(0, 6, clean_text(f"| Badges: {badge_str}"), ln=True)
            
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(0, 6, clean_text(f"Rationale: {rationale}"))
            pdf.ln(5)

    return bytes(pdf.output())

def render_analyst_ide():
    if not st.session_state.current_document_id:
        st.info("Select a document to begin.")
        return

    doc_id = st.session_state.current_document_id
    files = st.session_state.pipeline.get_document_files(doc_id)
    
    if files.get('scored') and os.path.exists(files['scored']):
        with open(files['scored'], 'r', encoding='utf-8') as f:
            audit_data = json.load(f)
        
        scored_refs = audit_data.get('scored_references', [])
        manuscript_title = audit_data.get('manuscript_metadata', {}).get('title', 'Unknown Manuscript')
        
        if scored_refs:
            st.header("🕵️ Citation Relevance Audit")
            
            tau = st.slider("Flag references scoring below this threshold (τ):", 0, 100, 50, 1)
            
            table_data = []
            flagged_refs_raw = []
            flagged_count = 0
            self_cite_count = 0
            issue_count = 0
            missing_abstract_count = 0
            
            for ref in scored_refs:
                rs_final = ref.get('RS_final', 0)
                is_flagged = rs_final < tau
                is_self_cite = ref.get('self_citation', {}).get('is_self_cite', False)
                has_issues = bool(ref.get('quality_flags'))
                has_abstract = bool(ref.get('external_metadata', {}).get('abstract'))
                
                if is_flagged: 
                    flagged_count += 1
                    flagged_refs_raw.append(ref)
                if is_self_cite:
                    self_cite_count += 1
                if has_issues:
                    issue_count += 1
                if not has_abstract:
                    missing_abstract_count += 1
                    
                badges = []
                if is_self_cite: badges.append("👤 Self-Cite")
                if has_issues: badges.append("⚠️ Issue")
                    
                table_data.append({
                    "ID": ref.get('ref_id'),
                    "Title": ref.get('original_data', {}).get('parsed', {}).get('title', 'Unknown'),
                    "Score (RS)": rs_final,
                    "Status": "🚩 Flagged" if is_flagged else "✅ OK",
                    "Badges": " | ".join(badges)
                })
            
            total_refs = len(scored_refs)
            ok_count = total_refs - flagged_count
            
            # --- KPIs Display (Now split into 5 columns!) ---
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Total Audited", total_refs)
            kpi2.metric("Flagged Irrelevant", flagged_count)
            kpi3.metric("Self-Citations", self_cite_count)
            kpi4.metric("Quality Issues", issue_count)
            kpi5.metric("Missing Abstract", missing_abstract_count)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- DYNAMIC VISUALIZATIONS ---
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Dynamic Pie Chart: Flagged vs Clean
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['✅ Highly Relevant (Clean)', '🚩 Flagged (Below τ)'], 
                    values=[ok_count, flagged_count], 
                    hole=.4,
                    marker_colors=['#2ca02c', '#d62728']
                )])
                fig_pie.update_layout(
                    title_text=f'Document Relevance Ratio (τ = {tau})', 
                    margin=dict(t=40, b=0, l=0, r=0),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                # Replaced Funnel with a clean Bar Chart showing Workload Breakdown
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=["Total Audited", "Clean (Passed)", "Requires Review (Flagged)"],
                        y=[total_refs, ok_count, flagged_count],
                        text=[total_refs, ok_count, flagged_count],
                        textposition='auto',
                        marker_color=["#1f77b4", "#2ca02c", "#d62728"]
                    )
                ])
                fig_bar.update_layout(
                    title_text='Analyst Workload Breakdown',
                    margin=dict(t=40, b=0, l=0, r=0),
                    yaxis_title='Number of References'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            # -----------------------------------

            # Interactive Table
            st.markdown("### 📋 Detailed Audit Table")
            df = pd.DataFrame(table_data)
            def highlight_flagged(row):
                return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row) if row['Status'] == '🚩 Flagged' else [''] * len(row)
            st.dataframe(df.style.apply(highlight_flagged, axis=1), use_container_width=True, hide_index=True)
            
            # --- EXPORT BUTTONS ---
            st.markdown("---")
            st.markdown("### 📥 Export Audit Reports")
            
            kpis = {
                'total': total_refs, 
                'flagged': flagged_count, 
                'self_cites': self_cite_count,
                'issues': issue_count,
                'missing_abstracts': missing_abstract_count
            }
            
            col_html, col_pdf, col_empty = st.columns([1, 1, 3])
            
            with col_html:
                html_report = generate_html_report(manuscript_title, tau, kpis, flagged_refs_raw)
                st.download_button(
                    label="📄 Export as HTML",
                    data=html_report,
                    file_name=f"CitePrism_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html",
                    use_container_width=True
                )
                
            with col_pdf:
                if HAS_FPDF:
                    pdf_report = generate_pdf_report(manuscript_title, tau, kpis, flagged_refs_raw)
                    st.download_button(
                        label="📑 Export as PDF",
                        data=pdf_report,
                        file_name=f"CitePrism_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                else:
                    st.error("Missing 'fpdf2' library. Run `pip install fpdf2` to enable PDF export.")
            
            # Context Viewer
            st.markdown("---")
            st.markdown("### 🔍 Context & Evidence Viewer")
            selected_ref_id = st.selectbox("Select a Reference ID to view LLM rationale:", [r['ID'] for r in table_data])
            
            for ref in scored_refs:
                if ref.get('ref_id') == selected_ref_id:
                    st.info(f"**LLM Rationale:** {ref.get('llm_rationale')}")
                    evidence_list = ref.get('llm_evidence', [])
                    evidence_text = evidence_list[0] if evidence_list else 'None found'
                    st.warning(f"**Extracted Evidence from Text:** {evidence_text}")
                    
                    with st.expander("View Full Abstract"):
                        st.write(ref.get('external_metadata', {}).get('abstract') or 'No abstract available.')
        else:
            st.info("No references found in the scored data.")
    else:
        st.info("Pipeline not fully complete.")