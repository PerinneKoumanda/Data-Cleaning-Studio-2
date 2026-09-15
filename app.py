"""
Data Cleaning Studio
--------------------

A Streamlit data-cleaning application with:

- CSV upload
- Automatic dataset overview
- Data preview
- Duplicate detection/removal
- Missing-value handling
- Text cleaning
- Numeric conversion
- Datetime conversion
- IQR and Z-score outlier detection
- Accurate audit logging
- Undo
- CSV export

Install:
    pip install streamlit pandas numpy

Run:
    streamlit run app.py
"""

import io
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Cleaning Studio",
    page_icon="🧹",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if "history" not in st.session_state:
    st.session_state.history = []

if "file_id" not in st.session_state:
    st.session_state.file_id = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_timestamp():
    """Return a readable timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_history():
    """
    Save the current dataframe before making a change.
    This allows the user to undo the last operation.
    """
    if st.session_state.df is not None:
        st.session_state.history.append(
            st.session_state.df.copy(deep=True)
        )


def count_changed_cells(before, after):
    """
    Count cells whose values changed.

    NaN -> NaN is NOT considered a change.
    """
    if before.shape != after.shape:
        return 0

    changed = (
        before.ne(after)
        & ~(before.isna() & after.isna())
    )

    return int(changed.sum().sum())


def log_action(
    action,
    rows_before,
    rows_after,
    cells_affected=0,
    detail="",
):
    """Add an action to the audit log."""

    st.session_state.audit_log.append(
        {
            "timestamp": get_timestamp(),
            "action": action,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": max(
                0,
                rows_before - rows_after
            ),
            "cells_affected": cells_affected,
            "detail": detail,
        }
    )


def clean_text(series, operation):
    """
    Apply text cleaning while preserving missing values.

    This is important because:

        astype(str)

    would turn NaN into the string "nan".
    """

    result = series.copy()

    mask = result.notna()

    if operation == "Trim whitespace":

        result.loc[mask] = (
            result.loc[mask]
            .astype(str)
            .str.strip()
        )

    elif operation == "Remove extra spaces":

        result.loc[mask] = (
            result.loc[mask]
            .astype(str)
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
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

    return result


def undo_last_change():
    """Restore the previous dataframe."""

    if not st.session_state.history:
        return

    current_rows = len(st.session_state.df)

    previous_df = st.session_state.history.pop()

    st.session_state.df = previous_df

    log_action(
        action="Undo",
        rows_before=current_rows,
        rows_after=len(previous_df),
        cells_affected=0,
        detail="Restored the previous dataframe state.",
    )

    st.rerun()


# ============================================================
# TITLE
# ============================================================

st.title("🧹 Data Cleaning Studio")

st.caption(
    "Upload messy data, clean it transparently, "
    "review every change, undo mistakes, and export the result."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)


if uploaded_file is not None:

    # Read the raw file bytes.
    file_bytes = uploaded_file.getvalue()

    # Create a unique fingerprint for the uploaded file.
    file_id = hashlib.md5(file_bytes).hexdigest()

    # Only reload when a NEW file is selected.
    if file_id != st.session_state.file_id:

        try:

            new_df = pd.read_csv(
                io.BytesIO(file_bytes)
            )

            # Reset the application for the new file.
            st.session_state.df = new_df
            st.session_state.audit_log = []
            st.session_state.history = []
            st.session_state.file_id = file_id
            st.session_state.file_name = uploaded_file.name

            log_action(
                action="Load data",
                rows_before=0,
                rows_after=len(new_df),
                cells_affected=(
                    len(new_df)
                    * len(new_df.columns)
                ),
                detail=(
                    f"Loaded '{uploaded_file.name}' "
                    f"with {len(new_df):,} rows and "
                    f"{len(new_df.columns):,} columns."
                ),
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Could not read the CSV file: {e}"
            )

            st.stop()


# ============================================================
# WAITING FOR FILE
# ============================================================

if st.session_state.df is None:

    st.info(
        "👆 Upload a CSV file above to get started."
    )

    st.stop()


# ============================================================
# CURRENT DATAFRAME
# ============================================================

df = st.session_state.df


# ============================================================
# FILE INFORMATION
# ============================================================

st.success(
    f"📁 Loaded file: **{st.session_state.file_name}**"
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("📊 Dataset overview")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Rows",
    f"{len(df):,}",
)

metric2.metric(
    "Columns",
    f"{len(df.columns):,}",
)

metric3.metric(
    "Missing cells",
    f"{int(df.isna().sum().sum()):,}",
)

metric4.metric(
    "Duplicate rows",
    f"{int(df.duplicated().sum()):,}",
)


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("👀 Data preview")

st.dataframe(
    df.head(100),
    use_container_width=True,
    height=400,
)


# ============================================================
# COLUMN INFORMATION
# ============================================================

with st.expander("🔎 Column information"):

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Missing": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "Missing %": [
                round(
                    df[column].isna().mean() * 100,
                    2,
                )
                for column in df.columns
            ],
            "Unique values": [
                int(
                    df[column].nunique(
                        dropna=True
                    )
                )
                for column in df.columns
            ],
        }
    )

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CLEANING ACTIONS
# ============================================================

st.subheader("🛠️ Cleaning actions")

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

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count == 0:

        st.success(
            "✅ No exact duplicate rows found."
        )

    else:

        st.warning(
            f"Found **{duplicate_count:,}** "
            "duplicate rows."
        )

        st.caption(
            "The first occurrence of each row will be kept."
        )

        if st.button(
            "🗑️ Remove duplicate rows",
            key="remove_duplicates",
            use_container_width=True,
        ):

            before_df = df.copy()

            save_history()

            df = (
                df
                .drop_duplicates(
                    keep="first"
                )
                .reset_index(drop=True)
            )

            removed_rows = (
                len(before_df)
                - len(df)
            )

            st.session_state.df = df

            log_action(
                action="Remove duplicates",
                rows_before=len(before_df),
                rows_after=len(df),
                cells_affected=(
                    removed_rows
                    * len(df.columns)
                ),
                detail=(
                    f"Removed {removed_rows:,} "
                    "exact duplicate rows."
                ),
            )

            st.rerun()


# ============================================================
# MISSING VALUES
# ============================================================

with tab_missing:

    missing_column = st.selectbox(
        "Choose a column",
        list(df.columns),
        key="missing_column",
    )

    missing_count = int(
        df[missing_column].isna().sum()
    )

    if missing_count == 0:

        st.success(
            f"✅ '{missing_column}' has no missing values."
        )

    else:

        st.warning(
            f"'{missing_column}' contains "
            f"**{missing_count:,}** missing values."
        )

        strategy = st.selectbox(
            "How should missing values be handled?",
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

        # ----------------------------------------
        # Custom value
        # ----------------------------------------

        if strategy == "Fill with custom value":

            custom_value = st.text_input(
                "Enter the replacement value",
                key="custom_missing_value",
            )

        # ----------------------------------------
        # Validate mean / median
        # ----------------------------------------

        if strategy in [
            "Fill with mean",
            "Fill with median",
        ]:

            if not pd.api.types.is_numeric_dtype(
                df[missing_column]
            ):

                st.error(
                    "Mean and median can only be "
                    "used with numeric columns."
                )

        # ----------------------------------------
        # Preview
        # ----------------------------------------

        if strategy == "Fill with mean":

            if pd.api.types.is_numeric_dtype(
                df[missing_column]
            ):

                preview_value = (
                    df[missing_column].mean()
                )

                st.info(
                    f"Preview: missing values will "
                    f"be replaced with "
                    f"**{preview_value:.4g}**."
                )

        elif strategy == "Fill with median":

            if pd.api.types.is_numeric_dtype(
                df[missing_column]
            ):

                preview_value = (
                    df[missing_column].median()
                )

                st.info(
                    f"Preview: missing values will "
                    f"be replaced with "
                    f"**{preview_value:.4g}**."
                )

        elif strategy == "Fill with mode":

            mode_values = df[
                missing_column
            ].mode(dropna=True)

            if mode_values.empty:

                st.error(
                    "Cannot calculate the mode because "
                    "there are no non-missing values."
                )

            else:

                st.info(
                    f"Preview: missing values will "
                    f"be replaced with "
                    f"**{mode_values.iloc[0]}**."
                )

        # ----------------------------------------
        # Apply
        # ----------------------------------------

        if st.button(
            "Apply missing-value cleaning",
            key="apply_missing",
            use_container_width=True,
        ):

            before_df = df.copy()

            save_history()

            try:

                # Drop rows
                if strategy == "Drop rows":

                    df = (
                        df
                        .dropna(
                            subset=[
                                missing_column
                            ]
                        )
                        .reset_index(drop=True)
                    )

                    detail = (
                        f"Dropped rows where "
                        f"'{missing_column}' "
                        "was missing."
                    )

                # Mean
                elif strategy == "Fill with mean":

                    if not pd.api.types.is_numeric_dtype(
                        df[missing_column]
                    ):

                        raise ValueError(
                            "Mean filling requires "
                            "a numeric column."
                        )

                    value = (
                        df[missing_column]
                        .mean()
                    )

                    df[
                        missing_column
                    ] = df[
                        missing_column
                    ].fillna(value)

                    detail = (
                        f"Filled {missing_count:,} "
                        f"missing values with "
                        f"mean ({value:.4g})."
                    )

                # Median
                elif strategy == "Fill with median":

                    if not pd.api.types.is_numeric_dtype(
                        df[missing_column]
                    ):

                        raise ValueError(
                            "Median filling requires "
                            "a numeric column."
                        )

                    value = (
                        df[missing_column]
                        .median()
                    )

                    df[
                        missing_column
                    ] = df[
                        missing_column
                    ].fillna(value)

                    detail = (
                        f"Filled {missing_count:,} "
                        f"missing values with "
                        f"median ({value:.4g})."
                    )

                # Mode
                elif strategy == "Fill with mode":

                    mode_values = df[
                        missing_column
                    ].mode(dropna=True)

                    if mode_values.empty:

                        raise ValueError(
                            "Cannot calculate mode."
                        )

                    value = mode_values.iloc[0]

                    df[
                        missing_column
                    ] = df[
                        missing_column
                    ].fillna(value)

                    detail = (
                        f"Filled {missing_count:,} "
                        f"missing values with "
                        f"mode ({value})."
                    )

                # Custom value
                elif strategy == "Fill with custom value":

                    if custom_value == "":

                        raise ValueError(
                            "Please enter a custom value."
                        )

                    # Preserve numeric types
                    if pd.api.types.is_numeric_dtype(
                        df[missing_column]
                    ):

                        try:

                            numeric_value = float(
                                custom_value
                            )

                            if (
                                pd.api.types
                                .is_integer_dtype(
                                    df[
                                        missing_column
                                    ]
                                )
                                and numeric_value.is_integer()
                            ):

                                numeric_value = int(
                                    numeric_value
                                )

                            df[
                                missing_column
                            ] = df[
                                missing_column
                            ].fillna(
                                numeric_value
                            )

                        except ValueError:

                            raise ValueError(
                                "This is a numeric "
                                "column. Please enter "
                                "a numeric value."
                            )

                    else:

                        df[
                            missing_column
                        ] = df[
                            missing_column
                        ].fillna(
                            custom_value
                        )

                    detail = (
                        f"Filled {missing_count:,} "
                        f"missing values with "
                        f"'{custom_value}'."
                    )

                cells_changed = count_changed_cells(
                    before_df,
                    df,
                )

                st.session_state.df = df

                log_action(
                    action=(
                        f"Missing values — "
                        f"{missing_column}"
                    ),
                    rows_before=len(before_df),
                    rows_after=len(df),
                    cells_affected=cells_changed,
                    detail=detail,
                )

                st.rerun()

            except Exception as e:

                # Remove failed history snapshot
                if st.session_state.history:
                    st.session_state.history.pop()

                st.error(
                    f"Could not apply operation: {e}"
                )


# ============================================================
# TYPES & TEXT
# ============================================================

with tab_text:

    transform_column = st.selectbox(
        "Choose a column",
        list(df.columns),
        key="transform_column",
    )

    transform_action = st.selectbox(
        "Choose an action",
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

    # ----------------------------------------
    # Warnings
    # ----------------------------------------

    if transform_action == "Convert to numeric":

        st.info(
            "Values that cannot be converted "
            "to numbers will become missing (NaN)."
        )

    elif transform_action == "Convert to datetime":

        st.info(
            "Values that cannot be interpreted "
            "as dates will become missing (NaT)."
        )

    # ----------------------------------------
    # Apply transformation
    # ----------------------------------------

    if st.button(
        "Apply transformation",
        key="apply_transform",
        use_container_width=True,
    ):

        before_df = df.copy()

        save_history()

        try:

            if transform_action in [
                "Trim whitespace",
                "Remove extra spaces",
                "Lowercase text",
                "Uppercase text",
            ]:

                df[
                    transform_column
                ] = clean_text(
                    df[
                        transform_column
                    ],
                    transform_action,
                )

                detail = (
                    f"Applied '{transform_action}' "
                    f"to '{transform_column}'."
                )

            elif transform_action == "Convert to numeric":

                before_missing = int(
                    df[
                        transform_column
                    ].isna().sum()
                )

                converted = pd.to_numeric(
                    df[
                        transform_column
                    ],
                    errors="coerce",
                )

                after_missing = int(
                    converted.isna().sum()
                )

                newly_missing = max(
                    0,
                    after_missing
                    - before_missing,
                )

                df[
                    transform_column
                ] = converted

                detail = (
                    f"Converted '{transform_column}' "
                    "to numeric. "
                    f"{newly_missing:,} values "
                    "could not be converted and "
                    "became missing."
                )

            elif transform_action == "Convert to datetime":

                before_missing = int(
                    df[
                        transform_column
                    ].isna().sum()
                )

                converted = pd.to_datetime(
                    df[
                        transform_column
                    ],
                    errors="coerce",
                )

                after_missing = int(
                    converted.isna().sum()
                )

                newly_missing = max(
                    0,
                    after_missing
                    - before_missing,
                )

                df[
                    transform_column
                ] = converted

                detail = (
                    f"Converted '{transform_column}' "
                    "to datetime. "
                    f"{newly_missing:,} values "
                    "could not be converted and "
                    "became missing."
                )

            cells_changed = count_changed_cells(
                before_df,
                df,
            )

            st.session_state.df = df

            log_action(
                action=(
                    f"Transform — "
                    f"{transform_column}"
                ),
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

    numeric_columns = (
        df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    if not numeric_columns:

        st.info(
            "No numeric columns are available "
            "for outlier detection."
        )

    else:

        outlier_column = st.selectbox(
            "Choose a numeric column",
            numeric_columns,
            key="outlier_column",
        )

        detection_method = st.selectbox(
            "Detection method",
            [
                "IQR",
                "Z-score",
            ],
            key="outlier_method",
        )

        # ----------------------------------------
        # IQR
        # ----------------------------------------

        if detection_method == "IQR":

            multiplier = st.slider(
                "IQR multiplier",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.25,
                key="iqr_multiplier",
            )

        # ----------------------------------------
        # Z-score
        # ----------------------------------------

        else:

            z_threshold = st.slider(
                "Z-score threshold",
                min_value=1.5,
                max_value=5.0,
                value=3.0,
                step=0.5,
                key="z_threshold",
            )

        series = df[
            outlier_column
        ].dropna()

        if len(series) == 0:

            st.warning(
                "This column contains no valid "
                "numeric values."
            )

        else:

            # ----------------------------------------
            # Calculate outliers
            # ----------------------------------------

            if detection_method == "IQR":

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)

                iqr = q3 - q1

                lower_bound = (
                    q1
                    - multiplier * iqr
                )

                upper_bound = (
                    q3
                    + multiplier * iqr
                )

                outlier_mask = (
                    (df[outlier_column] < lower_bound)
                    | (
                        df[outlier_column]
                        > upper_bound
                    )
                )

                description = (
                    f"Values below "
                    f"{lower_bound:.4g} "
                    f"or above "
                    f"{upper_bound:.4g}."
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
                        "Standard deviation is zero. "
                        "No outliers detected."
                    )

                else:

                    z_scores = (
                        (
                            df[outlier_column]
                            - mean
                        )
                        / std
                    ).abs()

                    outlier_mask = (
                        z_scores
                        > z_threshold
                    )

                    description = (
                        f"Values with "
                        f"|z| > {z_threshold}."
                    )

            # ----------------------------------------
            # Show result
            # ----------------------------------------

            outlier_count = int(
                outlier_mask.sum()
            )

            if outlier_count == 0:

                st.success(
                    "✅ No outliers detected."
                )

            else:

                st.warning(
                    f"Found **{outlier_count:,}** "
                    "potential outliers."
                )

                st.caption(
                    description
                )

                # Show potential outliers
                with st.expander(
                    "Preview detected outliers"
                ):

                    st.dataframe(
                        df.loc[
                            outlier_mask
                        ].head(100),
                        use_container_width=True,
                    )

                if st.button(
                    "🗑️ Remove outliers",
                    key="remove_outliers",
                    use_container_width=True,
                ):

                    before_df = df.copy()

                    save_history()

                    df = (
                        df.loc[
                            ~outlier_mask
                        ]
                        .reset_index(drop=True)
                    )

                    removed_rows = (
                        len(before_df)
                        - len(df)
                    )

                    st.session_state.df = df

                    log_action(
                        action=(
                            f"Remove outliers — "
                            f"{outlier_column}"
                        ),
                        rows_before=len(before_df),
                        rows_after=len(df),
                        cells_affected=(
                            removed_rows
                            * len(df.columns)
                        ),
                        detail=(
                            f"{detection_method}: "
                            f"{description}"
                        ),
                    )

                    st.rerun()


# ============================================================
# UNDO
# ============================================================

st.divider()

st.subheader("↩️ Undo")

if st.session_state.history:

    st.write(
        f"{len(st.session_state.history)} "
        "previous version(s) available."
    )

    if st.button(
        "↩️ Undo last change",
        use_container_width=True,
    ):

        undo_last_change()

else:

    st.caption(
        "No changes available to undo."
    )


# ============================================================
# AUDIT LOG
# ============================================================

st.divider()

st.subheader("📋 Audit log")

if st.session_state.audit_log:

    audit_df = pd.DataFrame(
        st.session_state.audit_log
    )

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.caption(
        "No actions recorded yet."
    )


# ============================================================
# EXPORT
# ============================================================

st.subheader("📦 Export")

export_col1, export_col2 = st.columns(2)


with export_col1:

    cleaned_csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download cleaned data",
        data=cleaned_csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


with export_col2:

    if st.session_state.audit_log:

        audit_csv = pd.DataFrame(
            st.session_state.audit_log
        ).to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download audit log",
            data=audit_csv,
            file_name="audit_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:

        st.caption(
            "Audit log will appear here after "
            "cleaning actions are performed."
        )


# ============================================================
# START OVER
# ============================================================

st.divider()

if st.button(
    "🔄 Start over",
    use_container_width=True,
):

    st.session_state.df = None
    st.session_state.audit_log = []
    st.session_state.history = []
    st.session_state.file_id = None
    st.session_state.file_name = None

    st.rerun()