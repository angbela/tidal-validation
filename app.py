import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
import io


st.set_page_config(
    page_title="Water Elevation Analysis",
    layout="wide"
)

st.title("🌊 Water Elevation Data Analysis")
st.markdown(
    "Paste timestamped model and survey data to find the best alignment and calculate RMSE."
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("📊 Data Input")


# ------------------------------------------------------------
# Model Data
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
        "..."
    ),
    key="model",
    help="Format: Date Time <TAB or space> Water Elevation"
)


# ------------------------------------------------------------
# Survey Data
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
    help="Format: Date Time <TAB or space> Water Elevation"
)


# ============================================================
# DATUM CONVERSION
# ============================================================

st.sidebar.header("📏 Datum Conversion")

apply_zero_mean = st.sidebar.checkbox(
    "Apply zero-mean conversion",
    value=True,
    help=(
        "Convert both datasets to zero-mean by removing their "
        "respective mean values before calculating RMSE."
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
        "Maximum time shift allowed when searching for the best "
        "alignment between model and survey data."
    )
)


shift_step_minutes = st.sidebar.number_input(
    "Shift Search Step (minutes):",
    min_value=1,
    max_value=60,
    value=2,
    step=1,
    help=(
        "Time interval used when searching for the best shift. "
        "For a 2-minute model, 2 minutes is recommended."
    )
)


# ============================================================
# DATA PARSING
# ============================================================

def parse_input_data(input_text, dataset_name):
    """
    Parse timestamp + water elevation input.

    Accepted examples:

    2/12/1900 00:00    -0.344375
    2/12/1900 00:15    -0.349063

    Delimiters:
    - tab
    - multiple spaces
    - comma
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

            # ------------------------------------------------
            # First try TAB separation
            # ------------------------------------------------

            if "\t" in line:

                parts = line.split("\t")

                if len(parts) < 2:
                    raise ValueError("Missing value")

                timestamp_text = parts[0].strip()
                value_text = parts[-1].strip()

            # ------------------------------------------------
            # Try comma separation
            # ------------------------------------------------

            elif "," in line:

                parts = line.rsplit(",", 1)

                if len(parts) != 2:
                    raise ValueError("Invalid comma-separated format")

                timestamp_text = parts[0].strip()
                value_text = parts[1].strip()

            # ------------------------------------------------
            # Otherwise assume last whitespace-separated token
            # is the water elevation.
            #
            # This allows:
            #
            # 2/12/1900 00:15 -0.349063
            # ------------------------------------------------

            else:

                parts = line.rsplit(None, 1)

                if len(parts) != 2:
                    raise ValueError(
                        "Expected timestamp followed by water elevation"
                    )

                timestamp_text = parts[0].strip()
                value_text = parts[1].strip()

            # ------------------------------------------------
            # Parse timestamp
            # ------------------------------------------------

            timestamp = pd.to_datetime(
                timestamp_text,
                dayfirst=False,
                errors="raise"
            )

            # ------------------------------------------------
            # Parse elevation
            # ------------------------------------------------

            value = float(value_text)

            records.append(
                {
                    "datetime": timestamp,
                    "water_elevation": value
                }
            )

        except Exception as e:

            st.error(
                f"❌ Error parsing {dataset_name}, line {line_number}: "
                f"`{line}`\n\n{str(e)}"
            )

            return None

    if len(records) == 0:
        st.error(f"❌ No valid records found in {dataset_name}.")
        return None

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Sort by datetime
    # --------------------------------------------------------

    df = df.sort_values("datetime").reset_index(drop=True)

    # --------------------------------------------------------
    # Check duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = df["datetime"].duplicated().sum()

    if duplicate_count > 0:

        st.error(
            f"❌ {dataset_name} contains {duplicate_count} "
            f"duplicate timestamp(s). Each timestamp must be unique."
        )

        return None

    # --------------------------------------------------------
    # Create timestep index
    # --------------------------------------------------------

    df["timestep"] = range(len(df))

    return df


# ============================================================
# DATA INTERVAL INFORMATION
# ============================================================

def calculate_interval_minutes(df):

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
# MODEL INTERPOLATION
# ============================================================

def interpolate_model_at_times(
    model_data,
    target_times
):
    """
    Interpolate model values at target timestamps.

    IMPORTANT:
    No extrapolation is allowed.

    Every target timestamp must be within the model
    time range.

    Returns:
        interpolated values
        OR None if any target timestamp is outside
        the model time range.
    """

    model_times = model_data["datetime"].astype("int64").values
    model_values = model_data["water_elevation"].values

    target_times_ns = pd.Series(target_times).astype("int64").values

    # --------------------------------------------------------
    # Constraint:
    # ALL survey timestamps must have a corresponding
    # model value within the model time range.
    # --------------------------------------------------------

    if (
        target_times_ns.min() < model_times.min()
        or
        target_times_ns.max() > model_times.max()
    ):
        return None

    # --------------------------------------------------------
    # Linear interpolation
    # --------------------------------------------------------

    interpolated_values = np.interp(
        target_times_ns,
        model_times,
        model_values
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
    Find the time shift producing the minimum RMSE.

    Positive shift:
        model timestamps are shifted forward.

    Negative shift:
        model timestamps are shifted backward.

    For each candidate shift, model values are interpolated
    at the survey timestamps.

    A shift is considered valid ONLY if every survey timestamp
    can be matched/interpolated from the model data.
    """

    max_shift_minutes = int(max_shift_hours * 60)

    # --------------------------------------------------------
    # Create candidate shifts
    # --------------------------------------------------------

    shift_values = np.arange(
        -max_shift_minutes,
        max_shift_minutes + shift_step_minutes,
        shift_step_minutes
    )

    best_rmse = float("inf")
    best_shift_minutes = None
    best_model_values = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    valid_shift_count = 0

    # --------------------------------------------------------
    # Search shifts
    # --------------------------------------------------------

    for i, shift_minutes in enumerate(shift_values):

        # ----------------------------------------------------
        # Shift MODEL timestamps
        # ----------------------------------------------------

        shifted_model_times = (
            model_data["datetime"]
            +
            pd.to_timedelta(
                shift_minutes,
                unit="min"
            )
        )

        shifted_model = model_data.copy()

        shifted_model["datetime"] = shifted_model_times

        # ----------------------------------------------------
        # Interpolate model at survey timestamps
        # ----------------------------------------------------

        interpolated_model = interpolate_model_at_times(
            shifted_model,
            survey_data["datetime"]
        )

        # ----------------------------------------------------
        # If any survey point cannot be matched,
        # this shift is invalid.
        # ----------------------------------------------------

        if interpolated_model is None:
            continue

        # ----------------------------------------------------
        # Calculate RMSE
        # ----------------------------------------------------

        rmse = sqrt(
            mean_squared_error(
                interpolated_model,
                survey_data["water_elevation"].values
            )
        )

        valid_shift_count += 1

        if rmse < best_rmse:

            best_rmse = rmse
            best_shift_minutes = shift_minutes
            best_model_values = interpolated_model.copy()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if i % 10 == 0 or i == len(shift_values) - 1:

            progress = (i + 1) / len(shift_values)

            progress_bar.progress(progress)

            status_text.text(
                f"Analyzing shift: "
                f"{shift_minutes:+d} min | "
                f"Valid shifts: {valid_shift_count}"
            )

    progress_bar.progress(1.0)

    if best_shift_minutes is None:

        status_text.text("Analysis failed.")

        return None, None, None

    status_text.text(
        f"Analysis complete! "
        f"Best shift: {best_shift_minutes:+d} minutes"
    )

    return (
        best_rmse,
        best_shift_minutes,
        best_model_values
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

if model_input and survey_input:

    # --------------------------------------------------------
    # Parse data
    # --------------------------------------------------------

    model_data = parse_input_data(
        model_input,
        "Model Data"
    )

    survey_data = parse_input_data(
        survey_input,
        "Survey Data"
    )

    if model_data is not None and survey_data is not None:

        # ----------------------------------------------------
        # Calculate intervals
        # ----------------------------------------------------

        model_interval = calculate_interval_minutes(model_data)
        survey_interval = calculate_interval_minutes(survey_data)

        # ----------------------------------------------------
        # Data overview
        # ----------------------------------------------------

        st.subheader("📋 Data Overview")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 📊 Model Data")

            st.write(
                f"Records: **{len(model_data):,}**"
            )

            st.write(
                f"Time range: **"
                f"{model_data['datetime'].min()}** → **"
                f"{model_data['datetime'].max()}**"
            )

            if model_interval:

                st.write(
                    f"Typical interval: **"
                    f"{model_interval['median']:.2f} min**"
                )

        with col2:

            st.markdown("### 📊 Survey Data")

            st.write(
                f"Records: **{len(survey_data):,}**"
            )

            st.write(
                f"Time range: **"
                f"{survey_data['datetime'].min()}** → **"
                f"{survey_data['datetime'].max()}**"
            )

            if survey_interval:

                st.write(
                    f"Typical interval: **"
                    f"{survey_interval['median']:.2f} min**"
                )

        # ----------------------------------------------------
        # Display input data
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📊 Model Data Preview")

            st.dataframe(
                model_data.head(20),
                use_container_width=True
            )

        with col2:

            st.subheader("📊 Survey Data Preview")

            st.dataframe(
                survey_data.head(20),
                use_container_width=True
            )

        # ----------------------------------------------------
        # Check time ranges
        # ----------------------------------------------------

        st.subheader("⏱️ Time Coverage Check")

        model_start = model_data["datetime"].min()
        model_end = model_data["datetime"].max()

        survey_start = survey_data["datetime"].min()
        survey_end = survey_data["datetime"].max()

        if (
            survey_start >= model_start
            and survey_end <= model_end
        ):

            st.success(
                "✅ Survey period is fully contained within "
                "the model period."
            )

        else:

            st.warning(
                "⚠️ Survey period is not completely contained "
                "within the model period."
            )

            st.write(
                f"Model: {model_start} → {model_end}"
            )

            st.write(
                f"Survey: {survey_start} → {survey_end}"
            )

            st.info(
                "The application will still test shifted periods, "
                "but a shift is valid only when ALL survey timestamps "
                "can be interpolated from the model."
            )

        # ----------------------------------------------------
        # Zero mean
        # ----------------------------------------------------

        model_data_original = model_data.copy()
        survey_data_original = survey_data.copy()

        if apply_zero_mean:

            model_mean = model_data[
                "water_elevation"
            ].mean()

            survey_mean = survey_data[
                "water_elevation"
            ].mean()

            model_data["water_elevation"] = (
                model_data["water_elevation"]
                - model_mean
            )

            survey_data["water_elevation"] = (
                survey_data["water_elevation"]
                - survey_mean
            )

            st.info(
                f"✓ Zero-mean conversion applied | "
                f"Model mean: {model_mean:.4f} | "
                f"Survey mean: {survey_mean:.4f}"
            )

        # ----------------------------------------------------
        # Run analysis
        # ----------------------------------------------------

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
                    best_model_values
                ) = find_best_shift(
                    model_data,
                    survey_data,
                    max_shift,
                    shift_step_minutes
                )

            # ------------------------------------------------
            # Check whether analysis succeeded
            # ------------------------------------------------

            if best_shift_minutes is None:

                st.error(
                    "❌ No valid time alignment was found.\n\n"
                    "This means that within the specified maximum "
                    "shift, at least one survey timestamp could not "
                    "be matched/interpolated from the model data."
                )

            else:

                # ============================================
                # CREATE ALIGNED OUTPUT
                # ============================================

                survey_times = survey_data["datetime"]

                shifted_model_times = (
                    model_data["datetime"]
                    +
                    pd.to_timedelta(
                        best_shift_minutes,
                        unit="min"
                    )
                )

                # ------------------------------------------------
                # Create output
                # ------------------------------------------------

                output_df = pd.DataFrame()

                output_df["datetime"] = survey_times

                output_df["survey_data"] = (
                    survey_data[
                        "water_elevation"
                    ].values
                )

                output_df["model_datetime"] = (
                    survey_times
                    -
                    pd.to_timedelta(
                        best_shift_minutes,
                        unit="min"
                    )
                )

                output_df["model_data"] = (
                    best_model_values
                )

                # ------------------------------------------------
                # Add original scale values
                # ------------------------------------------------

                if apply_zero_mean:

                    output_df["survey_data_original"] = (
                        output_df["survey_data"]
                        + survey_mean
                    )

                    output_df["model_data_original"] = (
                        output_df["model_data"]
                        + model_mean
                    )

                # ============================================
                # FINAL RMSE
                # ============================================

                final_rmse = sqrt(
                    mean_squared_error(
                        output_df["model_data"],
                        output_df["survey_data"]
                    )
                )

                # ------------------------------------------------
                # RMSE %
                # ------------------------------------------------

                model_range = (
                    model_data["water_elevation"].max()
                    -
                    model_data["water_elevation"].min()
                )

                if model_range != 0:

                    rmse_percent = (
                        final_rmse
                        /
                        model_range
                    ) * 100

                else:

                    rmse_percent = np.nan

                # ============================================
                # RESULTS
                # ============================================

                st.success(
                    "✅ Analysis Complete!"
                )

                col1, col2, col3, col4 = st.columns(4)

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
                        "*RMSE calculated using zero-meaned data.*"
                    )

                # ============================================
                # MATCHING INFORMATION
                # ============================================

                st.subheader(
                    "🔗 Survey ↔ Model Time Matching"
                )

                st.markdown(
                    f"""
                    The survey data contains **{len(survey_data):,}**
                    records.

                    For every survey timestamp, the application
                    interpolated the corresponding model value.

                    **Constraint:** all survey records must have a
                    valid model match. No extrapolation is permitted.
                    """
                )

                # ------------------------------------------------
                # Show matched data
                # ------------------------------------------------

                display_columns = [
                    "datetime",
                    "model_datetime",
                    "survey_data",
                    "model_data"
                ]

                if apply_zero_mean:

                    display_columns += [
                        "survey_data_original",
                        "model_data_original"
                    ]

                st.dataframe(
                    output_df[display_columns].head(50),
                    use_container_width=True
                )

                # ============================================
                # PLOT
                # ============================================

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
                    "and Interpolated Model Data",
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

                # ============================================
                # STATISTICS
                # ============================================

                st.subheader(
                    "📊 Statistical Summary"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Survey Data Statistics:**"
                    )

                    st.write(
                        survey_data[
                            "water_elevation"
                        ].describe()
                    )

                with col2:

                    st.write(
                        "**Aligned / Interpolated "
                        "Model Data Statistics:**"
                    )

                    st.write(
                        pd.Series(
                            output_df["model_data"]
                        ).describe()
                    )

                # ============================================
                # DOWNLOAD
                # ============================================

                st.subheader(
                    "💾 Download Results"
                )

                csv = output_df.to_csv(
                    index=False
                )

                st.download_button(
                    label="📥 Download Aligned Data (CSV)",
                    data=csv,
                    file_name=(
                        f"aligned_output_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    ),
                    mime="text/csv"
                )


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
        ...
        ```

        The model can have a different sampling interval from
        the survey data.

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

        The application supports different sampling intervals.

        For example:

        **Model:**

        ```
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

        **Survey:**

        ```
        00:00
        00:15
        00:30
        00:45
        01:00
        ```

        The model value at 00:15 does not need to exist
        explicitly.

        Instead, the application performs **linear interpolation**
        between the surrounding model values.

        ---

        #### 4. Time Alignment

        The application searches for the best model/survey
        alignment within the specified maximum shift.

        For example:

        ```
        Model shifted +2 min
        Model shifted +4 min
        Model shifted +6 min
        ...
        Model shifted -2 min
        Model shifted -4 min
        ...
        ```

        The shift producing the lowest RMSE is selected.

        ---

        #### 5. Important Matching Constraint

        **Every survey record must have a corresponding model
        value.**

        The application does NOT extrapolate model values.

        Therefore, a candidate time shift is rejected if even
        one survey timestamp falls outside the model time range.

        ---

        #### 6. Zero-Mean Conversion

        Enable **Apply zero-mean conversion** when the model and
        survey use different vertical datums.

        Each dataset has its own mean removed before calculating
        RMSE.

        ---

        #### 7. Shift Search Step

        For a model with a 2-minute interval, use:

        ```
        Shift Search Step = 2 minutes
        ```

        This means the application tests:

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

        You can use a smaller step if required, but the number
        of calculations will increase.

        ---

        ### 💡 Recommended Setup for Your Case

        **Model:** 2-minute data

        **Survey:** 15-minute data

        **Maximum Shift:** 24 hours

        **Shift Search Step:** 2 minutes

        The application will then use the 15-minute survey
        timestamps as the comparison points and interpolate the
        2-minute model data to those exact times.
        """
    )
