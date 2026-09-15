"""
Data Cleaning Studio with Audit Log
------------------------------------
A Streamlit app that lets you upload messy tabular data, apply
transparent cleaning steps, and see a full audit trail of every
change made — row counts, timestamps, and the exact action taken.

Run locally:
    pip install streamlit pandas numpy
    streamlit run app.py

Deploy free at https://share.streamlit.io (Streamlit Community Cloud)
by pushing this file + requirements.txt to a public GitHub repo.
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Cleaning Studio", layout="wide")

# ---------- session state ----------
if "df" not in st.session_state:
    st.session_state.df = None
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []


def log_action(action: str, before_rows: int, after_rows: int, detail: str = ""):
    st.session_state.audit_log.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "rows_before": before_rows,
            "rows_after": after_rows,
            "rows_affected": before_rows - after_rows,
            "detail": detail,
        }
    )


st.title("🧹 Data Cleaning Studio")
st.caption("Upload data, clean it transparently, and export a full audit trail.")

# ---------- upload ----------
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None and st.session_state.df is None:
    st.session_state.df = pd.read_csv(uploaded)
    log_action(
        "Load data",
        before_rows=0,
        after_rows=len(st.session_state.df),
        detail=f"Loaded '{uploaded.name}'",
    )

if st.session_state.df is not None:
    df = st.session_state.df

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing cells", int(df.isna().sum().sum()))

    st.subheader("Cleaning actions")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Duplicates", "Missing values", "Types & text", "Outliers"]
    )

    # ---- duplicates ----
    with tab1:
        if st.button("Drop exact duplicate rows"):
            before = len(df)
            df = df.drop_duplicates()
            st.session_state.df = df
            log_action("Drop duplicates", before, len(df))
            st.rerun()

    # ---- missing values ----
    with tab2:
        col = st.selectbox("Column", df.columns, key="missing_col")
        strategy = st.radio(
            "Strategy",
            ["Drop rows with missing values", "Fill with mean", "Fill with median",
             "Fill with mode", "Fill with a custom value"],
        )
        custom_val = None
        if strategy == "Fill with a custom value":
            custom_val = st.text_input("Custom value")

        if st.button("Apply", key="apply_missing"):
            before = len(df)
            missing_before = int(df[col].isna().sum())
            if strategy == "Drop rows with missing values":
                df = df.dropna(subset=[col])
            elif strategy == "Fill with mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "Fill with median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "Fill with mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif strategy == "Fill with a custom value" and custom_val is not None:
                df[col] = df[col].fillna(custom_val)
            st.session_state.df = df
            log_action(
                f"Handle missing values — {col}",
                before,
                len(df),
                detail=f"{strategy} ({missing_before} missing cells found)",
            )
            st.rerun()

    # ---- types & text ----
    with tab3:
        col = st.selectbox("Column", df.columns, key="type_col")
        action = st.radio(
            "Action",
            ["Trim whitespace", "Lowercase text", "Uppercase text",
             "Convert to numeric", "Convert to datetime"],
        )
        if st.button("Apply", key="apply_type"):
            before = len(df)
            detail = action
            try:
                if action == "Trim whitespace":
                    df[col] = df[col].astype(str).str.strip()
                elif action == "Lowercase text":
                    df[col] = df[col].astype(str).str.lower()
                elif action == "Uppercase text":
                    df[col] = df[col].astype(str).str.upper()
                elif action == "Convert to numeric":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif action == "Convert to datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                st.session_state.df = df
                log_action(f"Transform — {col}", before, len(df), detail=detail)
                st.rerun()
            except Exception as e:
                st.error(f"Could not apply: {e}")

    # ---- outliers ----
    with tab4:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Numeric column", numeric_cols, key="outlier_col")
            z_thresh = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.5)
            if st.button("Flag and remove outliers"):
                before = len(df)
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores <= z_thresh]
                st.session_state.df = df
                log_action(
                    f"Remove outliers — {col}",
                    before,
                    len(df),
                    detail=f"|z| > {z_thresh}",
                )
                st.rerun()
        else:
            st.info("No numeric columns available.")

    st.divider()

    # ---- audit log ----
    st.subheader("📋 Audit log")
    if st.session_state.audit_log:
        log_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(log_df, use_container_width=True)
    else:
        st.caption("No actions yet.")

    st.subheader("Export")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download cleaned data (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    with c2:
        if st.session_state.audit_log:
            log_csv = pd.DataFrame(st.session_state.audit_log).to_csv(index=False)
            st.download_button(
                "Download audit log (CSV)",
                data=log_csv.encode("utf-8"),
                file_name="audit_log.csv",
                mime="text/csv",
            )

    if st.button("🔄 Start over"):
        st.session_state.df = None
        st.session_state.audit_log = []
        st.rerun()

else:
    st.info("Upload a CSV to get started.")
