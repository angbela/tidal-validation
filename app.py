import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Water Elevation Analysis",
    layout="wide"
)

st.title("🌊 Water Elevation Data Analysis")

st.markdown(
    "Paste timestamped model and survey data to find the best "
    "time alignment and calculate RMSE."
)


# ============================================================
# SIDEBAR - DATA INPUT
# ============================================================

st.sidebar.header("📊 Data Input")


# ------------------------------------------------------------
# MODEL DATA
# ------------------------------------------------------------

st.sidebar.subheader("Model Data")

model_input = st.sidebar.text_area(
    "Paste Model Data:",
    height=250,
    placeholder=(
        "2/12/1900 00:00\t-0.344375\n"
        "2/12/1900 00:02\t-0.346125\n"
        "2/12/1900 00:04\t-0.347500\n"
        "2/12/1900 00:06\t-0.350125\n"
        "2/12/1900 00:08\t-0.351250\n"
        "..."
    ),
    key="model",
    help=(
        "Format: Date Time followed by water elevation. "
        "Example: 2/12/1900 00:00    -0.344375"
    )
)


# ------------------------------------------------------------
# SURVEY DATA
# ------------------------------------------------------------

st.sidebar.subheader("Survey Data")

survey_input = st.sidebar.text_area(
    "Paste Survey Data:",
    height=250,
    placeholder=(
        "2/12/1900 00:00\t-0.344375\n"
        "2/12/1900 00:15\t-0.349063\n"
        "2/12/1900 00:30\t-0.348588\n"
        "2/12/1900 00:45\t-0.397647\n"
        "2/12/1900 01:00\t-0.398800\n"
        "2/12/1900 01:15\t-0.352267"
    ),
    key="survey",
    help=(
        "Format: Date Time followed by water elevation. "
        "Example: 2/12/1900 00:15    -0.349063"
    )
)


# ============================================================
# DATUM CONVERSION
# ============================================================

st.sidebar.header("📏 Datum Conversion")

apply_zero_mean = st.sidebar.checkbox(
    "Apply zero-mean conversion",
    value=True,
    help=(
        "Convert both datasets to zero-mean by removing "
        "their respective mean values before analysis."
    )
)


# ============================================================
# SHIFT CONFIGURATION
# ============================================================

st.sidebar.header("🔀 Shift Configuration")

max_shift = st.sidebar.number_input(
    "Maximum Shift (hours):",
    min_value=1,
    max_value=8761,
    value=24,
    step=1,
    help=(
        "Maximum time shift allowed when searching for the "
        "best alignment between model and survey."
    )
)

shift_step_minutes = st.sidebar.number_input(
    "Shift Search Step (minutes):",
    min_value=1,
    max_value=60,
    value=2,
    step=1,
    help=(
        "Time increment used during the alignment search. "
        "For a 2-minute model, 2 minutes is recommended."
    )
)


# ============================================================
# PARSE INPUT DATA
# ============================================================

def parse_input_data(input_text, dataset_name):
    """
    Parse timestamp + water elevation data.

    Supported formats:

        2/12/1900 00:00    -0.344375

    or:

        2/12/1900 00:00\t-0.344375

    or:

        2/12/1900 00:00,-0.344375

    The final numeric value on each line is interpreted
    as water elevation.
    """

    if not input_text or not input_text.strip():
        return None

    lines = input_text.strip().splitlines()

    records = []

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        try:

            # ==================================================
            # TAB-SEPARATED
            # ==================================================

            if "\t" in line:

                parts = line.split("\t")

                if len(parts) < 2:
                    raise ValueError(
                        "Missing water elevation value."
                    )

                timestamp_text = parts[0].strip()
                value_text = parts[-1].strip()

            # ==================================================
            # COMMA-SEPARATED
            # ==================================================

            elif "," in line:

                parts = line.rsplit(",", 1)

                if len(parts) != 2:
                    raise ValueError(
                        "Invalid comma-separated format."
                    )

                timestamp_text = parts[0].strip()
                value_text = parts[1].strip()

            # ==================================================
            # SPACE-SEPARATED
            # ==================================================

            else:

                parts = line.rsplit(None, 1)

                if len(parts) != 2:
                    raise ValueError(
                        "Expected timestamp followed by "
                        "water elevation."
                    )

                timestamp_text = parts[0].strip()
                value_text = parts[1].strip()

            # ==================================================
            # PARSE DATETIME
            # ==================================================

            timestamp = pd.to_datetime(
                timestamp_text,
                errors="raise"
            )

            # ==================================================
            # PARSE VALUE
            # ==================================================

            value = float(value_text)

            records.append(
                {
                    "datetime": timestamp,
                    "water_elevation": value
                }
            )

        except Exception as e:

            st.error(
                f"❌ Error parsing {dataset_name}, "
                f"line {line_number}:\n\n"
                f"`{line}`\n\n"
                f"{str(e)}"
            )

            return None

    if len(records) == 0:

        st.error(
            f"❌ No valid records found in {dataset_name}."
        )

        return None

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(records)

    # ========================================================
    # SORT BY TIME
    # ========================================================

    df = (
        df
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # ========================================================
    # CHECK DUPLICATE TIMESTAMPS
    # ========================================================

    duplicate_count = (
        df["datetime"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        st.error(
            f"❌ {dataset_name} contains "
            f"{duplicate_count} duplicate timestamp(s).\n\n"
            "Each timestamp must be unique."
        )

        return None

    # ========================================================
    # TIMESTEP INDEX
    # ========================================================

    df["timestep"] = range(len(df))

    return df


# ============================================================
# CALCULATE DATA INTERVAL
# ============================================================

def calculate_interval_minutes(df):
    """
    Calculate typical timestep interval in minutes.
    """

    if df is None or len(df) < 2:
        return None

    differences = (
        df["datetime"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    if len(differences) == 0:
        return None

    return {
        "median": differences.median(),
        "minimum": differences.min(),
        "maximum": differences.max()
    }


# ============================================================
# INTERPOLATION FUNCTION
# ============================================================

def interpolate_values_at_times(
    source_data,
    target_times
):
    """
    Interpolate source_data onto target timestamps.

    IMPORTANT:
    No extrapolation is allowed.

    If even one target timestamp lies outside the
    source data time range, None is returned.
    """

    if source_data is None or len(source_data) < 2:
        return None

    source_times = (
        source_data["datetime"]
        .astype("int64")
        .values
    )

    source_values = (
        source_data["water_elevation"]
        .values
    )

    target_times = pd.Series(
        target_times
    )

    target_times_ns = (
        target_times
        .astype("int64")
        .values
    )

    # ========================================================
    # CHECK SOURCE RANGE
    # ========================================================

    if (
        target_times_ns.min() < source_times.min()
        or
        target_times_ns.max() > source_times.max()
    ):
        return None

    # ========================================================
    # LINEAR INTERPOLATION
    # ========================================================

    interpolated_values = np.interp(
        target_times_ns,
        source_times,
        source_values
    )

    return interpolated_values


# ============================================================
# FIND BEST SHIFT
# ============================================================

def find_best_shift(
    model_data,
    survey_data,
    max_shift_hours,
    shift_step_minutes
):
    """
    Find best time alignment.

    The dataset with the COARSER sampling interval is used
    as the reference/comparison timeline.

    Example:

        Model  = 2 minutes
        Survey = 15 minutes

    Survey becomes the reference timeline.

    The model timestamps are shifted and the model values
    are interpolated onto the survey timestamps.

    If the model is coarser than the survey, the opposite
    happens: survey is interpolated onto model timestamps.

    No extrapolation is allowed.

    Every reference timestamp must have a corresponding
    value in the other dataset.
    """

    # ========================================================
    # CALCULATE INTERVALS
    # ========================================================

    model_interval = calculate_interval_minutes(
        model_data
    )

    survey_interval = calculate_interval_minutes(
        survey_data
    )

    if model_interval is None:

        st.error(
            "❌ Model data requires at least "
            "2 timestamps."
        )

        return (
            None,
            None,
            None,
            None,
            None,
            None
        )

    if survey_interval is None:

        st.error(
            "❌ Survey data requires at least "
            "2 timestamps."
        )

        return (
            None,
            None,
            None,
            None,
            None,
            None
        )

    # ========================================================
    # DETERMINE COARSER DATASET
    # ========================================================

    if (
        model_interval["median"]
        >
        survey_interval["median"]
    ):

        reference = "model"

        reference_data = model_data

        reference_interval = (
            model_interval["median"]
        )

        comparison_name = "Model"

    else:

        reference = "survey"

        reference_data = survey_data

        reference_interval = (
            survey_interval["median"]
        )

        comparison_name = "Survey"

    # ========================================================
    # SHOW REFERENCE INFORMATION
    # ========================================================

    st.info(
        f"ℹ️ Comparison timeline: **{comparison_name}** "
        f"(typical interval: "
        f"**{reference_interval:.2f} minutes**)."
    )

    # ========================================================
    # SHIFT RANGE
    # ========================================================

    max_shift_minutes = int(
        max_shift_hours * 60
    )

    shift_values = np.arange(
        -max_shift_minutes,
        max_shift_minutes + shift_step_minutes,
        shift_step_minutes
    )

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    best_rmse = float("inf")

    best_shift_minutes = None

    best_reference_times = None

    best_model_values = None

    best_survey_values = None

    best_model_datetimes = None

    # ========================================================
    # PROGRESS
    # ========================================================

    progress_bar = st.progress(0)

    status_text = st.empty()

    valid_shift_count = 0

    # ========================================================
    # SEARCH ALL SHIFTS
    # ========================================================

    for i, shift_minutes in enumerate(
        shift_values
    ):

        # ====================================================
        # SHIFT MODEL TIME
        # ====================================================

        shifted_model = model_data.copy()

        shifted_model["datetime"] = (
            shifted_model["datetime"]
            +
            pd.to_timedelta(
                shift_minutes,
                unit="min"
            )
        )

        # ====================================================
        # MODEL IS COARSER
        # ====================================================

        if reference == "model":

            comparison_times = (
                reference_data["datetime"]
            )

            model_values = (
                reference_data[
                    "water_elevation"
                ]
                .values
            )

            survey_values = (
                interpolate_values_at_times(
                    survey_data,
                    comparison_times
                )
            )

            if survey_values is None:
                continue

            model_datetimes = (
                comparison_times
            )

        # ====================================================
        # SURVEY IS COARSER
        # ====================================================

        else:

            comparison_times = (
                reference_data["datetime"]
            )

            survey_values = (
                reference_data[
                    "water_elevation"
                ]
                .values
            )

            model_values = (
                interpolate_values_at_times(
                    shifted_model,
                    comparison_times
                )
            )

            if model_values is None:
                continue

            # ------------------------------------------------
            # Find the actual model timestamps surrounding
            # each survey timestamp.
            #
            # For output purposes we retain the original
            # survey timeline and calculate the corresponding
            # shifted model time.
            # ------------------------------------------------

            model_datetimes = (
                comparison_times
                -
                pd.to_timedelta(
                    shift_minutes,
                    unit="min"
                )
            )

        # ====================================================
        # RMSE
        # ====================================================

        rmse = sqrt(
            mean_squared_error(
                model_values,
                survey_values
            )
        )

        valid_shift_count += 1

        # ====================================================
        # STORE BEST RESULT
        # ====================================================

        if rmse < best_rmse:

            best_rmse = rmse

            best_shift_minutes = (
                shift_minutes
            )

            best_reference_times = (
                comparison_times.copy()
            )

            best_model_values = (
                np.asarray(
                    model_values
                ).copy()
            )

            best_survey_values = (
                np.asarray(
                    survey_values
                ).copy()
            )

            best_model_datetimes = (
                pd.Series(
                    model_datetimes
                ).reset_index(drop=True)
            )

        # ====================================================
        # PROGRESS UPDATE
        # ====================================================

        if (
            i % 10 == 0
            or
            i == len(shift_values) - 1
        ):

            progress = (
                (i + 1)
                /
                len(shift_values)
            )

            progress_bar.progress(
                progress
            )

            status_text.text(
                f"Analyzing shift: "
                f"{shift_minutes:+d} min "
                f"({shift_minutes / 60:+.2f} hr) | "
                f"Valid shifts: "
                f"{valid_shift_count}"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    progress_bar.progress(1.0)

    # ========================================================
    # NO VALID SHIFT
    # ========================================================

    if best_shift_minutes is None:

        status_text.text(
            "No valid alignment found."
        )

        return (
            None,
            None,
            None,
            None,
            None,
            None
        )

    # ========================================================
    # SUCCESS
    # ========================================================

    status_text.text(
        f"Analysis complete! "
        f"Best shift: "
        f"{best_shift_minutes:+d} minutes"
    )

    return (
        best_rmse,
        best_shift_minutes,
        best_reference_times,
        best_model_values,
        best_survey_values,
        best_model_datetimes
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if model_input and survey_input:

    # ========================================================
    # PARSE INPUT
    # ========================================================

    model_data = parse_input_data(
        model_input,
        "Model Data"
    )

    survey_data = parse_input_data(
        survey_input,
        "Survey Data"
    )

    if (
        model_data is not None
        and
        survey_data is not None
    ):

        # ====================================================
        # ORIGINAL DATA
        # ====================================================

        model_data_original = (
            model_data.copy()
        )

        survey_data_original = (
            survey_data.copy()
        )

        # ====================================================
        # INTERVAL INFORMATION
        # ====================================================

        model_interval = (
            calculate_interval_minutes(
                model_data
            )
        )

        survey_interval = (
            calculate_interval_minutes(
                survey_data
            )
        )

        # ====================================================
        # DATA OVERVIEW
        # ====================================================

        st.subheader(
            "📋 Data Overview"
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        with col1:

            st.markdown(
                "### 📊 Model Data"
            )

            st.write(
                f"Records: "
                f"**{len(model_data):,}**"
            )

            st.write(
                f"Start: "
                f"**{model_data['datetime'].min()}**"
            )

            st.write(
                f"End: "
                f"**{model_data['datetime'].max()}**"
            )

            if model_interval:

                st.write(
                    f"Typical interval: "
                    f"**{model_interval['median']:.2f} min**"
                )

                st.write(
                    f"Minimum interval: "
                    f"**{model_interval['minimum']:.2f} min**"
                )

                st.write(
                    f"Maximum interval: "
                    f"**{model_interval['maximum']:.2f} min**"
                )

        # ----------------------------------------------------
        # SURVEY
        # ----------------------------------------------------

        with col2:

            st.markdown(
                "### 📊 Survey Data"
            )

            st.write(
                f"Records: "
                f"**{len(survey_data):,}**"
            )

            st.write(
                f"Start: "
                f"**{survey_data['datetime'].min()}**"
            )

            st.write(
                f"End: "
                f"**{survey_data['datetime'].max()}**"
            )

            if survey_interval:

                st.write(
                    f"Typical interval: "
                    f"**{survey_interval['median']:.2f} min**"
                )

                st.write(
                    f"Minimum interval: "
                    f"**{survey_interval['minimum']:.2f} min**"
                )

                st.write(
                    f"Maximum interval: "
                    f"**{survey_interval['maximum']:.2f} min**"
                )

        # ====================================================
        # TIME RANGE CHECK
        # ====================================================

        st.subheader(
            "⏱️ Time Coverage"
        )

        model_start = (
            model_data["datetime"].min()
        )

        model_end = (
            model_data["datetime"].max()
        )

        survey_start = (
            survey_data["datetime"].min()
        )

        survey_end = (
            survey_data["datetime"].max()
        )

        # ----------------------------------------------------
        # Determine whether there is any direct overlap
        # ----------------------------------------------------

        overlap_start = max(
            model_start,
            survey_start
        )

        overlap_end = min(
            model_end,
            survey_end
        )

        if overlap_start <= overlap_end:

            st.success(
                "✅ Model and survey periods overlap."
            )

            st.write(
                f"Common period: "
                f"**{overlap_start}** → **{overlap_end}**"
            )

        else:

            st.warning(
                "⚠️ Model and survey periods do not "
                "directly overlap. A valid overlap may "
                "still be found after applying the "
                "allowed time shift."
            )

        # ====================================================
        # ZERO-MEAN CONVERSION
        # ====================================================

        if apply_zero_mean:

            model_mean = (
                model_data[
                    "water_elevation"
                ].mean()
            )

            survey_mean = (
                survey_data[
                    "water_elevation"
                ].mean()
            )

            model_data[
                "water_elevation"
            ] = (
                model_data[
                    "water_elevation"
                ]
                -
                model_mean
            )

            survey_data[
                "water_elevation"
            ] = (
                survey_data[
                    "water_elevation"
                ]
                -
                survey_mean
            )

            st.info(
                f"✓ Zero-mean conversion applied | "
                f"Model mean: {model_mean:.6f} | "
                f"Survey mean: {survey_mean:.6f}"
            )

        # ====================================================
        # DATA PREVIEW
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "📊 Model Data Preview"
            )

            if apply_zero_mean:

                st.caption(
                    "Showing zero-meaned values."
                )

            st.dataframe(
                model_data.head(20),
                use_container_width=True
            )

        with col2:

            st.subheader(
                "📊 Survey Data Preview"
            )

            if apply_zero_mean:

                st.caption(
                    "Showing zero-meaned values."
                )

            st.dataframe(
                survey_data.head(20),
                use_container_width=True
            )

        # ====================================================
        # RUN ANALYSIS
        # ====================================================

        if st.button(
            "🔍 Run Analysis",
            type="primary"
        ):

            with st.spinner(
                "Finding best time alignment..."
            ):

                (
                    best_rmse,
                    best_shift_minutes,
                    comparison_times,
                    aligned_model_values,
                    aligned_survey_values,
                    best_model_datetimes
                ) = find_best_shift(
                    model_data,
                    survey_data,
                    max_shift,
                    shift_step_minutes
                )

            # =================================================
            # CHECK RESULT
            # =================================================

            if best_shift_minutes is None:

                st.error(
                    "❌ No valid time alignment was found."
                )

                st.markdown(
                    """
                    This means that no candidate shift within
                    the specified maximum shift produced a
                    complete overlap between the reference
                    dataset and the other dataset.

                    Try one or more of the following:

                    - Increase **Maximum Shift**
                    - Check that the timestamps use the same
                      date/time convention
                    - Check that the model and survey periods
                      actually correspond to each other
                    - Reduce the **Shift Search Step** if a
                      finer alignment is required
                    """
                )

            else:

                # =============================================
                # CREATE OUTPUT DATAFRAME
                # =============================================

                output_df = pd.DataFrame()

                output_df[
                    "datetime"
                ] = (
                    comparison_times
                )

                output_df[
                    "model_datetime"
                ] = (
                    best_model_datetimes
                )

                output_df[
                    "model_data"
                ] = (
                    aligned_model_values
                )

                output_df[
                    "survey_data"
                ] = (
                    aligned_survey_values
                )

                # =============================================
                # ORIGINAL SCALE
                # =============================================

                if apply_zero_mean:

                    output_df[
                        "model_data_original"
                    ] = (
                        output_df[
                            "model_data"
                        ]
                        +
                        model_mean
                    )

                    output_df[
                        "survey_data_original"
                    ] = (
                        output_df[
                            "survey_data"
                        ]
                        +
                        survey_mean
                    )

                # =============================================
                # FINAL RMSE
                # =============================================

                final_rmse = sqrt(
                    mean_squared_error(
                        output_df[
                            "model_data"
                        ],
                        output_df[
                            "survey_data"
                        ]
                    )
                )

                # =============================================
                # RMSE %
                # =============================================

                model_range = (
                    model_data[
                        "water_elevation"
                    ].max()
                    -
                    model_data[
                        "water_elevation"
                    ].min()
                )

                if model_range != 0:

                    rmse_percent = (
                        final_rmse
                        /
                        model_range
                    ) * 100

                else:

                    rmse_percent = np.nan

                # =============================================
                # RESULTS
                # =============================================

                st.success(
                    "✅ Analysis Complete!"
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                with col1:

                    st.metric(
                        "Best Shift",
                        f"{best_shift_minutes:+d} min"
                    )

                with col2:

                    st.metric(
                        "Best Shift",
                        f"{best_shift_minutes / 60:+.2f} hr"
                    )

                with col3:

                    st.metric(
                        "RMSE",
                        f"{final_rmse:.6f}"
                    )

                with col4:

                    st.metric(
                        "RMSE (%)",
                        f"{rmse_percent:.2f}%"
                    )

                if apply_zero_mean:

                    st.caption(
                        "*RMSE calculated on zero-meaned data.*"
                    )

                # =============================================
                # MATCHING INFORMATION
                # =============================================

                st.subheader(
                    "🔗 Time Matching Information"
                )

                st.write(
                    f"Comparison points: "
                    f"**{len(output_df):,}**"
                )

                st.write(
                    f"Best time shift: "
                    f"**{best_shift_minutes:+d} minutes**"
                )

                if model_interval["median"] > survey_interval["median"]:

                    st.info(
                        "Model has the coarser sampling interval. "
                        "Survey values were interpolated onto "
                        "the model timestamps."
                    )

                else:

                    st.info(
                        "Survey has the coarser sampling interval. "
                        "Model values were interpolated onto "
                        "the survey timestamps."
                    )

                # =============================================
                # MATCHED DATA TABLE
                # =============================================

                st.subheader(
                    "📋 Matched / Interpolated Data"
                )

                display_columns = [
                    "datetime",
                    "model_datetime",
                    "model_data",
                    "survey_data"
                ]

                if apply_zero_mean:

                    display_columns += [
                        "model_data_original",
                        "survey_data_original"
                    ]

                st.dataframe(
                    output_df[
                        display_columns
                    ].head(100),
                    use_container_width=True
                )

                # =============================================
                # PLOT
                # =============================================

                st.subheader(
                    "📈 Water Elevation Comparison"
                )

                fig, ax = plt.subplots(
                    figsize=(14, 7)
                )

                ax.plot(
                    output_df["datetime"],
                    output_df["survey_data"],
                    label="Survey Data",
                    linewidth=2,
                    color="#2E86AB",
                    alpha=0.8,
                    marker="o",
                    markersize=3
                )

                ax.plot(
                    output_df["datetime"],
                    output_df["model_data"],
                    label="Model Data (Interpolated)",
                    linewidth=2,
                    color="#A23B72",
                    alpha=0.8,
                    linestyle="--"
                )

                ylabel = (
                    "Water Elevation - Zero-Meaned"
                    if apply_zero_mean
                    else
                    "Water Elevation"
                )

                ax.set_ylabel(
                    ylabel,
                    fontsize=12,
                    fontweight="bold"
                )

                ax.set_xlabel(
                    "Date Time",
                    fontsize=12,
                    fontweight="bold"
                )

                ax.set_title(
                    "Comparison between Survey Data "
                    "and Model Data",
                    fontsize=14,
                    fontweight="bold",
                    pad=20
                )

                ax.legend(
                    loc="best",
                    fontsize=11,
                    framealpha=0.9
                )

                ax.grid(
                    True,
                    alpha=0.3,
                    linestyle="--"
                )

                if apply_zero_mean:

                    ax.axhline(
                        y=0,
                        color="gray",
                        linestyle=":",
                        alpha=0.5,
                        linewidth=1
                    )

                plt.xticks(
                    rotation=45,
                    ha="right"
                )

                plt.tight_layout()

                st.pyplot(fig)

                # =============================================
                # STATISTICAL SUMMARY
                # =============================================

                st.subheader(
                    "📊 Statistical Summary"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Survey Data Statistics:**"
                    )

                    st.write(
                        output_df[
                            "survey_data"
                        ].describe()
                    )

                with col2:

                    st.write(
                        "**Aligned Model Data Statistics:**"
                    )

                    st.write(
                        output_df[
                            "model_data"
                        ].describe()
                    )

                # =============================================
                # DOWNLOAD
                # =============================================

                st.subheader(
                    "💾 Download Results"
                )

                csv = output_df.to_csv(
                    index=False
                )

                st.download_button(
                    label=(
                        "📥 Download Aligned Data (CSV)"
                    ),
                    data=csv,
                    file_name=(
                        f"aligned_output_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    ),
                    mime="text/csv"
                )


# ============================================================
# NO INPUT
# ============================================================

else:

    st.info(
        "👆 Please paste both Model Data and Survey Data "
        "in the sidebar to begin analysis."
    )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    st.markdown(
        """
        ### 📋 Instructions

        #### 1. Model Data

        Paste timestamp + water elevation:

        ```
        2/12/1900 00:00    -0.344375
        2/12/1900 00:02    -0.346125
        2/12/1900 00:04    -0.347500
        2/12/1900 00:06    -0.350125
        2/12/1900 00:08    -0.351250
        ```

        ---

        #### 2. Survey Data

        Paste timestamp + water elevation:

        ```
        2/12/1900 00:00    -0.344375
        2/12/1900 00:15    -0.349063
        2/12/1900 00:30    -0.348588
        2/12/1900 00:45    -0.397647
        2/12/1900 01:00    -0.398800
        2/12/1900 01:15    -0.352267
        ```

        ---

        #### 3. Different Sampling Intervals

        Model and survey do NOT need to have the same
        sampling interval.

        Example:

        ```
        Model:
        00:00
        00:02
        00:04
        00:06
        00:08
        00:10
        00:12
        00:14
        00:16
        ```

        Survey:

        ```
        00:00
        00:15
        00:30
        00:45
        01:00
        ```

        The application will interpolate the model value
        at the survey timestamps.

        ---

        #### 4. Time Alignment

        The application searches for the best time shift.

        For example, with:

        ```
        Maximum Shift = 24 hours
        Shift Step = 2 minutes
        ```

        the application tests:

        ```
        -24:00
        -23:58
        -23:56
        ...
        -00:02
        +00:00
        +00:02
        +00:04
        ...
        +23:58
        +24:00
        ```

        The shift with the lowest RMSE is selected.

        ---

        #### 5. Dataset Length

        **There is NO requirement that model data contains
        more records than survey data.**

        The application determines the comparison timeline
        from the sampling interval.

        The dataset with the coarser interval becomes the
        reference timeline.

        ---

        #### 6. No Extrapolation

        The application will never extrapolate beyond the
        available source data.

        Therefore, a candidate shift is accepted only when
        every reference timestamp can be interpolated from
        the other dataset.

        ---

        #### 7. Zero-Mean Conversion

        Enable:

        ```
        Apply zero-mean conversion
        ```

        when the model and survey use different vertical
        datums.

        Each dataset has its own mean removed before RMSE
        calculation.

        ---

        #### 8. Recommended Setup for 2-Minute Model /
        15-Minute Survey

        ```
        Maximum Shift:
        24 hours

        Shift Search Step:
        2 minutes
        ```

        The 15-minute survey timestamps will be used as
        the comparison points and the 2-minute model will
        be interpolated to those timestamps.

        ---

        ### 💡 Important

        Both datasets must use the same time reference,
        date convention, and timezone/time basis.

        The date itself can be something like:

        ```
        2/12/1900
        ```

        as long as the same date/time basis is used for
        both datasets.
        """
    )
