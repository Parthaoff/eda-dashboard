"""
=============================================================
  Academic EDA Dashboard — app.py
  Fixed for: Plotly 6.x + Streamlit 1.56+
  Changes:
    - plotly_theme() now applied via fig.update_layout()
      instead of being spread as kwargs into px functions
      (paper_bgcolor etc. are no longer accepted as px args)
    - use_container_width replaced with width='stretch'
      (deprecated in Streamlit 1.43+, removed in 1.56+)
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")        # non-interactive backend — required for Streamlit
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Academic EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  – dark-academic palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0f1117;
    --surface:   #181c27;
    --border:    #262c3e;
    --accent:    #5b8dee;
    --accent2:   #e07b54;
    --text:      #e8eaf0;
    --muted:     #7a8099;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text); }

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.main { background: var(--bg); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

h1 { font-family: 'DM Serif Display', serif; font-size: 2.2rem !important; color: #fff !important; }
h2 { font-family: 'DM Serif Display', serif; color: #fff !important; margin-top: 1.6rem !important; }
h3 { font-family: 'DM Sans', sans-serif; font-weight: 600; color: var(--accent) !important; }

[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.4rem;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.78rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'DM Mono', monospace;
}

hr { border-color: var(--border) !important; }
.stAlert { border-radius: 10px; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

.stButton > button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.45rem 1.2rem;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

.pill {
    display:inline-block;
    background: var(--accent);
    color:#fff;
    font-size:.72rem;
    font-weight:600;
    letter-spacing:.08em;
    text-transform:uppercase;
    padding:.25rem .75rem;
    border-radius:999px;
    margin-bottom:.6rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

EXPECTED_COLS = {
    "Student_ID", "Name", "Gender", "Subject",
    "Marks_Obtained", "Total_Marks",
    "Attendance_Percentage", "Grade",
}
NUMERIC_COLS     = ["Marks_Obtained", "Total_Marks", "Attendance_Percentage"]
CATEGORICAL_COLS = ["Gender", "Subject", "Grade"]


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame. Cached per uploaded file object."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def apply_plotly_theme(fig):
    """
    Apply the dark theme to any Plotly figure via update_layout().

    FIX for Plotly 6.x: layout properties (paper_bgcolor, plot_bgcolor,
    font_family, font_color, margin, template) can NO LONGER be passed as
    keyword arguments directly into px.scatter(), px.box(), px.bar() etc.
    They MUST be applied through fig.update_layout() after figure creation.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#181c27",
        plot_bgcolor="#181c27",
        font=dict(family="DM Sans", color="#e8eaf0"),
        margin=dict(t=50, l=20, r=20, b=20),
    )
    return fig


def section(label: str):
    st.markdown(f'<p class="pill">{label}</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR — navigation & filter placeholders
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Academic EDA")
    st.caption("Upload your dataset and explore.")
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        [
            "📁 Data Overview",
            "🧹 Data Cleaning",
            "📈 Univariate Analysis",
            "🔗 Bivariate & Multivariate",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Global Filters")
    st.caption("Applied after data is uploaded.")

    gender_filter  = st.empty()
    subject_filter = st.empty()
    attend_filter  = st.empty()


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.title("Academic EDA Dashboard")
st.markdown(
    "_Upload your student dataset and navigate the sections on the left "
    "to explore distributions, relationships, and patterns._"
)
st.markdown("---")


# ─────────────────────────────────────────────
#  FILE UPLOAD
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your dataset here (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="Expected columns: Student_ID, Name, Gender, Subject, "
         "Marks_Obtained, Total_Marks, Attendance_Percentage, Grade",
)

if uploaded is None:
    st.info("👆 Upload a file to get started.")
    st.stop()


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Reading file…"):
    raw_df = load_data(uploaded)

missing_expected = EXPECTED_COLS - set(raw_df.columns)
if missing_expected:
    st.warning(
        f"⚠️ These expected columns were **not found**: "
        f"{', '.join(sorted(missing_expected))}. "
        "Some charts may be unavailable."
    )

# Seed session state on first load
if "clean_df" not in st.session_state:
    st.session_state.clean_df = raw_df.copy()


# ─────────────────────────────────────────────
#  POPULATE SIDEBAR FILTERS
# ─────────────────────────────────────────────
with gender_filter:
    gender_vals = ["All"] + (
        sorted(raw_df["Gender"].dropna().unique().tolist())
        if "Gender" in raw_df.columns else []
    )
    sel_gender = st.selectbox("Gender", gender_vals)

with subject_filter:
    subj_vals = sorted(raw_df["Subject"].dropna().unique().tolist()) \
                if "Subject" in raw_df.columns else []
    sel_subject = st.multiselect("Subject(s)", subj_vals, default=subj_vals)

with attend_filter:
    if "Attendance_Percentage" in raw_df.columns:
        a_min = float(raw_df["Attendance_Percentage"].min())
        a_max = float(raw_df["Attendance_Percentage"].max())
        sel_attend = st.slider(
            "Attendance %", a_min, a_max, (a_min, a_max), step=0.5
        )
    else:
        sel_attend = (0.0, 100.0)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all sidebar filters and return the filtered DataFrame."""
    out = df.copy()
    if sel_gender != "All" and "Gender" in out.columns:
        out = out[out["Gender"] == sel_gender]
    if sel_subject and "Subject" in out.columns:
        out = out[out["Subject"].isin(sel_subject)]
    if "Attendance_Percentage" in out.columns:
        out = out[
            out["Attendance_Percentage"].between(sel_attend[0], sel_attend[1])
        ]
    return out


df = apply_filters(st.session_state.clean_df)


# ═══════════════════════════════════════════
#  PAGE 1 — DATA OVERVIEW
# ═══════════════════════════════════════════
if page == "📁 Data Overview":
    section("Dataset Overview")
    st.subheader("Raw Data Preview")

    rows_to_show = st.slider("Rows to preview", 5, min(100, len(df)), 10)
    st.dataframe(df.head(rows_to_show), width='stretch')

    st.markdown("---")
    section("Dimensions")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",       f"{df.shape[0]:,}")
    c2.metric("Total Columns",    df.shape[1])
    c3.metric("Numeric Cols",     df.select_dtypes(include="number").shape[1])
    c4.metric("Categorical Cols", df.select_dtypes(exclude="number").shape[1])

    st.markdown("---")
    section("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, width='stretch')

    st.markdown("---")
    section("Column Data Types")
    dtype_df = pd.DataFrame({
        "Column":   df.dtypes.index,
        "Dtype":    df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Null":     df.isnull().sum().values,
        "Unique":   df.nunique().values,
    })
    st.dataframe(dtype_df.set_index("Column"), width='stretch')


# ═══════════════════════════════════════════
#  PAGE 2 — DATA CLEANING
# ═══════════════════════════════════════════
elif page == "🧹 Data Cleaning":
    section("Data Cleaning & Preprocessing")
    st.subheader("Missing Value Analysis")

    null_counts = st.session_state.clean_df.isnull().sum()
    null_df = null_counts[null_counts > 0].reset_index()
    null_df.columns = ["Column", "Missing Count"]
    null_df["Missing %"] = (
        null_df["Missing Count"] / len(st.session_state.clean_df) * 100
    ).round(2)

    if null_df.empty:
        st.success("✅ No missing values detected in the dataset!")
    else:
        st.dataframe(null_df.set_index("Column"), width='stretch')

        fig = px.bar(
            null_df, x="Column", y="Missing %",
            color="Missing %",
            color_continuous_scale=["#5b8dee", "#e07b54"],
            title="Missing Values per Column (%)",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        section("Handling Strategy")

        col_to_fix = st.selectbox("Select a column to handle", null_df["Column"].tolist())
        strategy = st.radio(
            "Strategy",
            ["Drop rows with nulls in this column",
             "Fill with Mean (numeric)",
             "Fill with Median (numeric)",
             "Fill with Mode",
             "Fill with custom value"],
            horizontal=True,
        )

        custom_val = None
        if strategy == "Fill with custom value":
            custom_val = st.text_input("Custom fill value")

        if st.button("Apply Fix"):
            work = st.session_state.clean_df.copy()

            if strategy == "Drop rows with nulls in this column":
                work = work.dropna(subset=[col_to_fix])

            elif strategy == "Fill with Mean (numeric)":
                if pd.api.types.is_numeric_dtype(work[col_to_fix]):
                    work[col_to_fix] = work[col_to_fix].fillna(work[col_to_fix].mean())
                else:
                    st.error("Column is not numeric — choose a different strategy.")

            elif strategy == "Fill with Median (numeric)":
                if pd.api.types.is_numeric_dtype(work[col_to_fix]):
                    work[col_to_fix] = work[col_to_fix].fillna(work[col_to_fix].median())
                else:
                    st.error("Column is not numeric — choose a different strategy.")

            elif strategy == "Fill with Mode":
                mode_val = work[col_to_fix].mode()
                if not mode_val.empty:
                    work[col_to_fix] = work[col_to_fix].fillna(mode_val[0])

            elif strategy == "Fill with custom value" and custom_val is not None:
                try:
                    work[col_to_fix] = work[col_to_fix].fillna(float(custom_val))
                except ValueError:
                    work[col_to_fix] = work[col_to_fix].fillna(custom_val)

            st.session_state.clean_df = work
            st.success(f"✅ Applied '{strategy}' on column **{col_to_fix}**.")
            st.rerun()

    st.markdown("---")
    section("Duplicate Rows")
    n_dupes = st.session_state.clean_df.duplicated().sum()
    st.metric("Duplicate Rows", n_dupes)
    if n_dupes > 0:
        if st.button("Drop Duplicate Rows"):
            st.session_state.clean_df = st.session_state.clean_df.drop_duplicates()
            st.success(f"Removed {n_dupes} duplicate row(s).")
            st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset to Original Data"):
        st.session_state.clean_df = raw_df.copy()
        st.success("Dataset reset to original upload.")
        st.rerun()


# ═══════════════════════════════════════════
#  PAGE 3 — UNIVARIATE ANALYSIS
# ═══════════════════════════════════════════
elif page == "📈 Univariate Analysis":
    section("Univariate Analysis")

    # ── Numeric distributions ─────────────────
    st.subheader("Numeric Distributions")

    num_cols_present = [c for c in NUMERIC_COLS if c in df.columns]
    if not num_cols_present:
        st.warning("No numeric columns found after filtering.")
    else:
        chosen_num = st.selectbox("Select numeric column", num_cols_present)
        nbins      = st.slider("Histogram bins", 10, 80, 30)
        show_kde   = st.checkbox("Overlay KDE", value=True)

        if show_kde:
            # Seaborn KDE overlay — rendered via Matplotlib (not Plotly)
            fig_sb, ax = plt.subplots(figsize=(10, 4), facecolor="#181c27")
            ax.set_facecolor("#181c27")
            sns.histplot(
                df[chosen_num].dropna(), bins=nbins, kde=True, ax=ax,
                color="#5b8dee", edgecolor="#0f1117",
                line_kws={"color": "#e07b54", "lw": 2},
            )
            ax.set_xlabel(chosen_num, color="#e8eaf0")
            ax.set_ylabel("Count",    color="#e8eaf0")
            ax.tick_params(colors="#7a8099")
            for spine in ax.spines.values():
                spine.set_edgecolor("#262c3e")
            ax.set_title(f"Distribution of {chosen_num}", color="#ffffff",
                         fontsize=14, pad=12)
            st.pyplot(fig_sb, use_container_width=False)
            plt.close(fig_sb)
        else:
            fig = px.histogram(
                df, x=chosen_num, nbins=nbins,
                color_discrete_sequence=["#5b8dee"],
                title=f"Distribution of {chosen_num}",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, width='stretch')

        # Box plot — theme applied via update_layout (Plotly 6.x fix)
        fig2 = px.box(
            df, y=chosen_num,
            color_discrete_sequence=["#e07b54"],
            title=f"Box Plot — {chosen_num}",
        )
        apply_plotly_theme(fig2)
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")

    # ── Categorical distributions ──────────────
    st.subheader("Categorical Distributions")

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    if not cat_cols_present:
        st.warning("No categorical columns found.")
    else:
        chosen_cat   = st.selectbox("Select categorical column", cat_cols_present)
        chart_type   = st.radio("Chart type", ["Pie", "Bar"], horizontal=True)
        value_counts = df[chosen_cat].value_counts().reset_index()
        value_counts.columns = [chosen_cat, "Count"]

        if chart_type == "Pie":
            fig3 = px.pie(
                value_counts, names=chosen_cat, values="Count",
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Distribution of {chosen_cat}",
                hole=0.35,
            )
            fig3.update_traces(textinfo="percent+label")
        else:
            fig3 = px.bar(
                value_counts, x=chosen_cat, y="Count",
                color=chosen_cat,
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Count by {chosen_cat}",
            )

        apply_plotly_theme(fig3)
        st.plotly_chart(fig3, width='stretch')


# ═══════════════════════════════════════════
#  PAGE 4 — BIVARIATE & MULTIVARIATE
# ═══════════════════════════════════════════
elif page == "🔗 Bivariate & Multivariate":
    section("Bivariate & Multivariate Analysis")

    # ── Scatter: Attendance vs Marks ──────────
    st.subheader("Attendance vs Marks (Scatter)")

    if {"Attendance_Percentage", "Marks_Obtained"}.issubset(df.columns):
        color_by = st.selectbox(
            "Color points by",
            [None] + [c for c in CATEGORICAL_COLS if c in df.columns],
            format_func=lambda x: "None" if x is None else x,
        )
        trendline = st.checkbox("Show OLS trendline", value=True)

        # FIX: No layout kwargs in px.scatter() — apply via update_layout()
        fig4 = px.scatter(
            df,
            x="Attendance_Percentage",
            y="Marks_Obtained",
            color=color_by,
            trendline="ols" if trendline else None,
            opacity=0.75,
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={
                "Attendance_Percentage": "Attendance (%)",
                "Marks_Obtained":        "Marks Obtained",
            },
            title="Attendance Percentage vs Marks Obtained",
            hover_data=[c for c in ["Name", "Subject", "Grade"] if c in df.columns],
        )
        apply_plotly_theme(fig4)
        st.plotly_chart(fig4, width='stretch')
    else:
        st.info("Columns Attendance_Percentage and/or Marks_Obtained not found.")

    st.markdown("---")

    # ── Box plots: Marks across groups ────────
    st.subheader("Marks Distribution Across Groups")

    if "Marks_Obtained" in df.columns:
        group_by = st.selectbox(
            "Group by",
            [c for c in CATEGORICAL_COLS if c in df.columns],
        )
        fig5 = px.box(
            df, x=group_by, y="Marks_Obtained",
            color=group_by,
            color_discrete_sequence=px.colors.qualitative.Bold,
            points="outliers",
            title=f"Marks Obtained by {group_by}",
        )
        apply_plotly_theme(fig5)
        st.plotly_chart(fig5, width='stretch')
    else:
        st.info("Column Marks_Obtained not found.")

    st.markdown("---")

    # ── Grouped bar: Avg Marks per Subject per Grade ──
    if {"Subject", "Grade", "Marks_Obtained"}.issubset(df.columns):
        st.subheader("Average Marks — Subject × Grade")
        agg = (
            df.groupby(["Subject", "Grade"])["Marks_Obtained"]
            .mean().round(2).reset_index()
        )
        fig6 = px.bar(
            agg, x="Subject", y="Marks_Obtained", color="Grade",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Average Marks per Subject, grouped by Grade",
        )
        apply_plotly_theme(fig6)
        st.plotly_chart(fig6, width='stretch')
        st.markdown("---")

    # ── Correlation Heatmap (Seaborn / Matplotlib) ─
    st.subheader("Correlation Heatmap")

    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        st.warning("Not enough numeric columns to compute a correlation matrix.")
    else:
        corr = num_df.corr()
        fig7, ax7 = plt.subplots(
            figsize=(max(6, corr.shape[1] * 1.2), max(5, corr.shape[0])),
            facecolor="#181c27",
        )
        ax7.set_facecolor("#181c27")
        sns.heatmap(
            corr, annot=True, fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5, linecolor="#262c3e",
            ax=ax7,
            annot_kws={"size": 9, "color": "#e8eaf0"},
            cbar_kws={"shrink": 0.8},
        )
        ax7.set_title("Pearson Correlation Matrix", color="#ffffff",
                      fontsize=14, pad=14)
        ax7.tick_params(colors="#7a8099", rotation=30)
        plt.tight_layout()
        st.pyplot(fig7, use_container_width=False)
        plt.close(fig7)

    st.markdown("---")

    # ── Violin: Attendance by Grade ───────────
    if {"Attendance_Percentage", "Grade"}.issubset(df.columns):
        st.subheader("Attendance Distribution by Grade (Violin)")
        fig8 = px.violin(
            df, x="Grade", y="Attendance_Percentage",
            color="Grade", box=True, points="outliers",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Attendance Distribution per Grade",
        )
        apply_plotly_theme(fig8)
        st.plotly_chart(fig8, width='stretch')


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Academic EDA Dashboard · Plotly 6.x + Streamlit 1.56+ compatible · "
    "Filtered rows: **{:,}** of **{:,}** total".format(
        len(df), len(st.session_state.clean_df)
    )
)
