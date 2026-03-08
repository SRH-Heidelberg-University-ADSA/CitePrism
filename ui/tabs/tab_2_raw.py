import streamlit as st
import pandas as pd
from textwrap import dedent


def premium_metric_card(title, value, subtitle="", icon="📊"):
    st.markdown(dedent(f"""
    <div class="metric-card-premium">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="muted-label">{title}</div>
                <div class="big-number">{value}</div>
                <div class="muted-label">{subtitle}</div>
            </div>
            <div style="font-size:1.4rem;">{icon}</div>
        </div>
    </div>
    """), unsafe_allow_html=True)


def status_badge(status):
    key = status.lower().replace(" ", "-")
    css = {
        "matched": "badge-matched",
        "flagged": "badge-flagged",
        "unresolved": "badge-unresolved",
    }.get(key, "badge-unresolved")
    return f'<span class="status-badge {css}">{status}</span>'


def risk_badge(risk):
    key = risk.lower()
    css = {
        "low": "badge-low",
        "medium": "badge-medium",
        "high": "badge-high",
    }.get(key, "badge-medium")
    return f'<span class="status-badge {css}">{risk}</span>'


def render_result_card(row):
    st.markdown(dedent(f"""
    <div class="result-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
            <div>
                <div style="font-weight:700; color:#f8fafc; margin-bottom:0.35rem;">{row['Citation']}</div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.45rem;">
                    {status_badge(row['Status'])}
                    {risk_badge(row['Risk'])}
                </div>
                <div class="muted-label">Type: {row['Type']}</div>
            </div>
            <div style="text-align:right;">
                <div class="muted-label">Score</div>
                <div style="font-size:1.35rem; font-weight:800; color:#f8fafc;">{row['Score']:.2f}</div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)


def render_raw_data_tab():
    st.markdown("## Audit Data Explorer")
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-0.3rem;'>Inspect extracted citations, matched references, scores, and review signals.</p>",
        unsafe_allow_html=True
    )

    data = [
        {"Citation": "Smith et al. (2021)", "Status": "Matched", "Score": 0.91, "Risk": "Low", "Type": "Method"},
        {"Citation": "Doe and Lee (2019)", "Status": "Matched", "Score": 0.63, "Risk": "Medium", "Type": "Background"},
        {"Citation": "Brown et al. (2018)", "Status": "Flagged", "Score": 0.42, "Risk": "High", "Type": "Claim"},
        {"Citation": "Zhang (2020)", "Status": "Unresolved", "Score": 0.27, "Risk": "High", "Type": "Evidence"},
    ]
    df = pd.DataFrame(data)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        premium_metric_card("Total Citations", len(df), "Extracted from manuscript", "📚")
    with m2:
        premium_metric_card("Matched References", int((df["Status"] == "Matched").sum()), "Resolved successfully", "🔗")
    with m3:
        premium_metric_card("High-Risk Items", int((df["Risk"] == "High").sum()), "Needs analyst review", "⚠️")
    with m4:
        premium_metric_card("Average Score", f"{df['Score'].mean():.2f}", "Overall evidence strength", "✨")

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    st.markdown(dedent("""
    <div class="glass-card">
        <div class="soft-section-title">Filter Explorer</div>
        <div class="soft-section-subtitle">Refine results by search, status, risk, and score threshold.</div>
    </div>
    """), unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([1.45, 1, 1, 0.9])

    with f1:
        search = st.text_input("Search citation", placeholder="Search by author, year, or keyword")
    with f2:
        status_filter = st.multiselect(
            "Status",
            options=sorted(df["Status"].unique()),
            default=list(df["Status"].unique())
        )
    with f3:
        risk_filter = st.multiselect(
            "Risk",
            options=sorted(df["Risk"].unique()),
            default=list(df["Risk"].unique())
        )
    with f4:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05)

    filtered_df = df.copy()

    if search:
        filtered_df = filtered_df[filtered_df["Citation"].str.contains(search, case=False, na=False)]

    filtered_df = filtered_df[
        filtered_df["Status"].isin(status_filter) &
        filtered_df["Risk"].isin(risk_filter) &
        (filtered_df["Score"] >= min_score)
    ]

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    st.markdown(dedent("""
    <div class="custom-card">
        <div class="soft-section-title">Preview Cards</div>
        <div class="soft-section-subtitle">A cleaner visual summary of the filtered citations before the data table.</div>
    </div>
    """), unsafe_allow_html=True)

    preview_cols = st.columns(2)
    preview_rows = filtered_df.head(2).to_dict("records")

    for i, row in enumerate(preview_rows):
        with preview_cols[i % 2]:
            render_result_card(row)

    st.markdown(dedent("""
    <div class="custom-card">
        <div class="soft-section-title">📊 Citation Results</div>
        <div class="soft-section-subtitle">Filtered audit results for the current manuscript.</div>
    </div>
    """), unsafe_allow_html=True)

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    if not filtered_df.empty:
        selected = filtered_df.iloc[0]

        col1, col2 = st.columns([1.45, 1])

        with col1:
            st.markdown(dedent("""
            <div class="custom-card">
                <div class="soft-section-title">Selected Citation</div>
                <div class="soft-section-subtitle">Focused inspection panel for the highlighted record.</div>
                <hr class="divider-soft">
            </div>
            """), unsafe_allow_html=True)

            st.write(f"**Citation:** {selected['Citation']}")
            st.write(f"**Status:** {selected['Status']}")
            st.write(f"**Type:** {selected['Type']}")
            st.write(f"**Score:** {selected['Score']:.2f}")
            st.write(f"**Risk:** {selected['Risk']}")

        with col2:
            st.markdown(dedent("""
            <div class="custom-card">
                <div class="soft-section-title">Raw Metadata</div>
                <div class="soft-section-subtitle">Expand below to inspect raw extraction output.</div>
            </div>
            """), unsafe_allow_html=True)

            with st.expander("View raw JSON"):
                st.json(selected.to_dict())