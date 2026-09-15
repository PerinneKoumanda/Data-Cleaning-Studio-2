"""
Data Cleaning Studio
--------------------
A Streamlit data-cleaning application with:

- CSV upload
- Data preview
- Duplicate removal
- Missing-value handling
- Text cleaning
- Numeric/datetime conversion
- IQR outlier removal
- Before/after statistics
- Accurate audit logging
- Undo
- CSV export

Run:
    pip install streamlit pandas numpy
    streamlit run app.py
"""

from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Data Cleaning Studio",
    page_icon="🧹",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "df": None,
    "audit_log": [],
    "history": [],
    "file_id": None,
    "file_name": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dataframe_hash(df):
    """
    Create a fingerprint of the dataframe so we can detect
    whether the uploaded file is actually different.
    """
    try:
        data = pd.util.hash_pandas_object(
            df,
            index=True,
        ).values.tobytes()

        return hashlib.md5(data).hexdigest()

    except Exception:
        return str(len(df))


def log_action(
    action,
    rows_before,
    rows_after,
    cells_affected=0,
    detail="",
):
    st.session_state.audit_log.append(
        {
            "timestamp": timestamp(),
            "action": action,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": max(0, rows_before - rows_after),
            "cells_affected": cells_affected,
            "detail": detail,
        }
    )


def save_history():
    """
    Save a copy before modifying the dataframe.
    """
    if st.session_state.df is not None:
        st.session_state.history.append(
            st.session_state.df.copy(deep=True)
        )


def undo():
    """
    Restore the previous dataframe.
    """
    if st.session_state.history:
        st.session_state.df = st.session_state.history.pop()

        log_action(
            action="Undo",
            rows_before=len(st.session_state.df),
            rows_after=len(st.session_state.df),
            detail="Restored previous dataframe state.",
        )

        st.rerun()


def count_changed_cells(before, after):
    """
    Count cells whose values changed.

    Handles NaN == NaN correctly.
    """
    if before.shape != after.shape:
        return 0

    changed = (
        before.ne(after)
        & ~(before.isna() & after.isna())
    )

    return int(changed.sum().sum())


def clean_text_preserve_missing(series, operation):
    """
    Perform string operations without converting NaN into
    the string 'nan'.
    """
    result = series.copy()

    mask = result.notna()

    if operation == "Trim whitespace":
        result.loc[mask] = (
            result.loc[mask]
            .astype(str)
            .str.strip()
        )

    elif operation == "Lowercase text":
        result.loc[mask] = (
            result.loc[mask]
            .astype(str)
            .str.lower()
        )

    elif operation == "Uppercase text":
        result.loc[mask] = (
            result.loc[mask]
            .astype(str)
            .str.upper()
        )

    elif operation == "Remove extra spaces":
        result.loc[mask] = (
            result.loc[mask]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    return result


# ============================================================
# TITLE
# ============================================================

st.title("🧹 Data Cleaning Studio")

st.caption(
    "Clean your data transparently, preview changes, "
    "undo mistakes, and export a complete audit trail."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)


if uploaded is not None:

    # Read uploaded file once
    file_bytes = uploaded.getvalue()

    file_id = hashlib.md5(file_bytes).hexdigest()

    # Load only when a different file is uploaded
    if file_id != st.session_state.file_id:

        try:
            new_df = pd.read_csv(io.BytesIO(file_bytes))

            st.session_state.df = new_df
            st.session_state.audit_log = []
            st.session_state.history = []
            st.session_state.file_id = file_id
            st.session_state.file_name = uploaded.name

            log_action(
                action="Load data",
                rows_before=0,
                rows_after=len(new_df),
                cells_affected=len(new_df) * len(new_df.columns),
                detail=f"Loaded '{uploaded.name}'",
            )

            st.rerun()

        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            st.stop()


# ============================================================
# NO DATA
# ============================================================

if st.session_state.df is None:
    st.info("Upload a CSV file to get started.")
    st.stop()


df = st.session_state.df


# ============================================================
# FILE INFO
# ============================================================

st.success(
    f"Loaded: **{st.session_state.file_name}**"
)


# ============================================================
# DATASET METRICS
# ============================================================

st.subheader("Dataset overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric(
    "Missing cells",
    f"{int(df.isna().sum().sum()):,}",
)
c4.metric(
    "Duplicate rows",
    f"{int(df.duplicated().sum()):,}",
)


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("Preview data", expanded=True):

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=400,
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

with st.expander("Column information"):

    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [
                str(df[col].dtype)
                for col in df.columns
            ],
            "Missing": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique": [
                int(df[col].nunique(dropna=True))
                for col in df.columns
            ],
        }
    )

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CLEANING ACTIONS
# ============================================================

st.subheader("Cleaning actions")

tab_duplicates, tab_missing, tab_text, tab_outliers = st.tabs(
    [
        "Duplicates",
        "Missing values",
        "Types & text",
        "Outliers",
    ]
)


# ============================================================
# DUPLICATES
# ============================================================

with tab_duplicates:

    duplicate_count = int(df.duplicated().sum())

    st.write(
        f"Found **{duplicate_count:,}** exact duplicate rows."
    )

    if duplicate_count > 0:

        if st.button(
            "Remove duplicate rows",
            key="remove_duplicates",
        ):

            before_df = df.copy()

            save_history()

            df = df.drop_duplicates(
                keep="first"
            ).reset_index(drop=True)

            cells_changed = (
                (len(before_df) - len(df))
                * len(df.columns)
            )

            st.session_state.df = df

            log_action(
                action="Remove duplicates",
                rows_before=len(before_df),
                rows_after=len(df),
                cells_affected=cells_changed,
                detail=(
                    f"Removed {len(before_df) - len(df)} "
                    "duplicate rows; kept first occurrence."
                ),
            )

            st.rerun()

    else:
        st.success("No duplicate rows found.")


# ============================================================
# MISSING VALUES
# ============================================================

with tab_missing:

    col = st.selectbox(
        "Column",
        list(df.columns),
        key="missing_column",
    )

    missing_count = int(df[col].isna().sum())

    st.write(
        f"Missing values in **{col}**: "
        f"**{missing_count:,}**"
    )

    strategy = st.selectbox(
        "Strategy",
        [
            "Drop rows",
            "Fill with mean",
            "Fill with median",
            "Fill with mode",
            "Fill with custom value",
        ],
        key="missing_strategy",
    )

    custom_value = None

    if strategy == "Fill with custom value":

        custom_value = st.text_input(
            "Custom value",
            key="custom_missing_value",
        )

    if strategy in [
        "Fill with mean",
        "Fill with median",
    ]:

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):
            st.warning(
                "Mean and median can only be used "
                "with numeric columns."
            )

    if strategy == "Fill with mode":

        mode_values = df[col].mode(dropna=True)

        if mode_values.empty:
            st.warning(
                "There is no non-missing value available "
                "to calculate the mode."
            )

    if st.button(
        "Apply missing-value cleaning",
        key="apply_missing",
    ):

        if missing_count == 0:
            st.info(
                "This column has no missing values."
            )
            st.stop()

        before_df = df.copy()
        save_history()

        try:

            if strategy == "Drop rows":

                df = df.dropna(
                    subset=[col]
                ).reset_index(drop=True)

                detail = (
                    f"Dropped rows where '{col}' "
                    "was missing."
                )

            elif strategy == "Fill with mean":

                if not pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    raise ValueError(
                        "Mean filling requires a numeric column."
                    )

                value = df[col].mean()

                df[col] = df[col].fillna(value)

                detail = (
                    f"Filled {missing_count} missing values "
                    f"with mean ({value:.4g})."
                )

            elif strategy == "Fill with median":

                if not pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    raise ValueError(
                        "Median filling requires a numeric column."
                    )

                value = df[col].median()

                df[col] = df[col].fillna(value)

                detail = (
                    f"Filled {missing_count} missing values "
                    f"with median ({value:.4g})."
                )

            elif strategy == "Fill with mode":

                mode_values = df[col].mode(
                    dropna=True
                )

                if mode_values.empty:
                    raise ValueError(
                        "Cannot calculate mode because "
                        "the column contains no valid values."
                    )

                value = mode_values.iloc[0]

                df[col] = df[col].fillna(value)

                detail = (
                    f"Filled {missing_count} missing values "
                    f"with mode ({value})."
                )

            elif strategy == "Fill with custom value":

                if custom_value == "":
                    raise ValueError(
                        "Enter a custom value."
                    )

                # Try to preserve numeric columns
                if pd.api.types.is_numeric_dtype(
                    df[col]
                ):

                    try:
                        numeric_value = float(
                            custom_value
                        )

                        if (
                            pd.api.types.is_integer_dtype(
                                df[col]
                            )
                            and numeric_value.is_integer()
                        ):
                            numeric_value = int(
                                numeric_value
                            )

                        df[col] = df[col].fillna(
                            numeric_value
                        )

                    except ValueError:
                        raise ValueError(
                            "This is a numeric column. "
                            "Enter a numeric custom value."
                        )

                else:
                    df[col] = df[col].fillna(
                        custom_value
                    )

                detail = (
                    f"Filled {missing_count} missing "
                    f"values with '{custom_value}'."
                )

            cells_changed = count_changed_cells(
                before_df,
                df,
            )

            st.session_state.df = df

            log_action(
                action=f"Missing values — {col}",
                rows_before=len(before_df),
                rows_after=len(df),
                cells_affected=cells_changed,
                detail=detail,
            )

            st.rerun()

        except Exception as e:

            # Remove history snapshot if operation failed
            if st.session_state.history:
                st.session_state.history.pop()

            st.error(
                f"Could not apply operation: {e}"
            )


# ============================================================
# TYPES & TEXT
# ============================================================

with tab_text:

    col = st.selectbox(
        "Column",
        list(df.columns),
        key="transform_column",
    )

    action = st.selectbox(
        "Action",
        [
            "Trim whitespace",
            "Remove extra spaces",
            "Lowercase text",
            "Uppercase text",
            "Convert to numeric",
            "Convert to datetime",
        ],
        key="transform_action",
    )

    if action == "Convert to numeric":

        st.caption(
            "Values that cannot be converted will become missing."
        )

    if action == "Convert to datetime":

        st.caption(
            "Values that cannot be interpreted as dates "
            "will become missing."
        )

    if st.button(
        "Apply transformation",
        key="apply_transform",
    ):

        before_df = df.copy()
        save_history()

        try:

            if action in [
                "Trim whitespace",
                "Remove extra spaces",
                "Lowercase text",
                "Uppercase text",
            ]:

                df[col] = clean_text_preserve_missing(
                    df[col],
                    action,
                )

                detail = (
                    f"Applied '{action}' to '{col}'."
                )

            elif action == "Convert to numeric":

                before_missing = int(
                    df[col].isna().sum()
                )

                converted = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                new_missing = int(
                    converted.isna().sum()
                )

                df[col] = converted

                newly_missing = max(
                    0,
                    new_missing - before_missing,
                )

                detail = (
                    f"Converted '{col}' to numeric. "
                    f"{newly_missing} values could not "
                    "be converted and became missing."
                )

            elif action == "Convert to datetime":

                before_missing = int(
                    df[col].isna().sum()
                )

                converted = pd.to_datetime(
                    df[col],
                    errors="coerce",
                )

                new_missing = int(
                    converted.isna().sum()
                )

                df[col] = converted

                newly_missing = max(
                    0,
                    new_missing - before_missing,
                )

                detail = (
                    f"Converted '{col}' to datetime. "
                    f"{newly_missing} values could not "
                    "be converted and became missing."
                )

            cells_changed = count_changed_cells(
                before_df,
                df,
            )

            st.session_state.df = df

            log_action(
                action=f"Transform — {col}",
                rows_before=len(before_df),
                rows_after=len(df),
                cells_affected=cells_changed,
                detail=detail,
            )

            st.rerun()

        except Exception as e:

            if st.session_state.history:
                st.session_state.history.pop()

            st.error(
                f"Could not apply transformation: {e}"
            )


# ============================================================
# OUTLIERS
# ============================================================

with tab_outliers:

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_cols:

        st.info(
            "No numeric columns are available "
            "for outlier detection."
        )

    else:

        col = st.selectbox(
            "Numeric column",
            numeric_cols,
            key="outlier_column",
        )

        method = st.selectbox(
            "Detection method",
            [
                "IQR",
                "Z-score",
            ],
            key="outlier_method",
        )

        if method == "IQR":

            multiplier = st.slider(
                "IQR multiplier",
                1.0,
                3.0,
                1.5,
                0.25,
            )

        else:

            z_threshold = st.slider(
                "Z-score threshold",
                1.5,
                5.0,
                3.0,
                0.5,
            )

        series = df[col].dropna()

        if len(series) == 0:

            st.warning(
                "This column contains no numeric values."
            )

        else:

            if method == "IQR":

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)

                iqr = q3 - q1

                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr

                outlier_mask = (
                    (df[col] < lower)
                    | (df[col] > upper)
                )

                description = (
                    f"Values below {lower:.4g} "
                    f"or above {upper:.4g}"
                )

            else:

                mean = series.mean()
                std = series.std()

                if std == 0 or pd.isna(std):

                    outlier_mask = pd.Series(
                        False,
                        index=df.index,
                    )

                    description = (
                        "Standard deviation is zero; "
                        "no outliers detected."
                    )

                else:

                    z_scores = (
                        (df[col] - mean)
                        / std
                    ).abs()

                    outlier_mask = (
                        z_scores > z_threshold
                    )

                    description = (
                        f"|z| > {z_threshold}"
                    )

            outlier_count = int(
                outlier_mask.sum()
            )

            st.write(
                f"Potential outliers: "
                f"**{outlier_count:,}**"
            )

            st.caption(description)

            if outlier_count > 0:

                if st.button(
                    "Remove outliers",
                    key="remove_outliers",
                ):

                    before_df = df.copy()
                    save_history()

                    df = df.loc[
                        ~outlier_mask
                    ].reset_index(drop=True)

                    st.session_state.df = df

                    log_action(
                        action=f"Remove outliers — {col}",
                        rows_before=len(before_df),
                        rows_after=len(df),
                        cells_affected=(
                            (len(before_df) - len(df))
                            * len(df.columns)
                        ),
                        detail=description,
                    )

                    st.rerun()


# ============================================================
# UNDO
# ============================================================

st.divider()

undo_col, spacer = st.columns([1, 5])

with undo_col:

    if st.session_state.history:

        if st.button(
            "↩️ Undo last change",
            key="undo",
        ):
            undo()


# ============================================================
# AUDIT LOG
# ============================================================

st.subheader("📋 Audit log")

if st.session_state.audit_log:

    log_df = pd.DataFrame(
        st.session_state.audit_log
    )

    st.dataframe(
        log_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.caption("No actions recorded yet.")


# ============================================================
# EXPORT
# ============================================================

st.subheader("Export")

c1, c2 = st.columns(2)

with c1:

    st.download_button(
        "⬇️ Download cleaned data",
        data=df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


with c2:

    if st.session_state.audit_log:

        log_csv = pd.DataFrame(
            st.session_state.audit_log
        ).to_csv(index=False)

        st.download_button(
            "⬇️ Download audit log",
            data=log_csv.encode("utf-8"),
            file_name="audit_log.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# START OVER
# ============================================================

st.divider()

if st.button(
    "🔄 Start over",
    key="start_over",
):

    st.session_state.df = None
    st.session_state.audit_log = []
    st.session_state.history = []
    st.session_state.file_id = None
    st.session_state.file_name = None

    st.rerun()