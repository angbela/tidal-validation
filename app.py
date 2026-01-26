import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
import io

st.set_page_config(page_title="Water Elevation Analysis", layout="wide")

st.title("🌊 Water Elevation Data Analysis")
st.markdown("Paste your model and survey data to find the best alignment and calculate RMSE")

# Sidebar for inputs
st.sidebar.header("📊 Data Input")

# Text area for model data
st.sidebar.subheader("Model Data")
model_input = st.sidebar.text_area(
    "Paste water elevation values (one per line):",
    height=200,
    placeholder="123.5\n124.6\n125.7\n126.3\n...",
    key='model',
    help="Enter one water elevation value per line"
)

# Text area for survey data
st.sidebar.subheader("Survey Data")
survey_input = st.sidebar.text_area(
    "Paste water elevation values (one per line):",
    height=200,
    placeholder="123.5\n124.6\n125.7\n126.3\n...",
    key='survey',
    help="Enter one water elevation value per line"
)

# Datum conversion option
st.sidebar.header("📏 Datum Conversion")
apply_zero_mean = st.sidebar.checkbox(
    "Apply zero-mean conversion",
    value=True,
    help="Convert both datasets to zero-mean (remove mean from each dataset) before analysis"
)

# Start time input
st.sidebar.header("⏰ Survey Data Configuration")
start_date = st.sidebar.date_input("Start Date", value=datetime(2024, 1, 1))
start_time = st.sidebar.time_input("Start Time", value=datetime.strptime("00:00", "%H:%M").time())

# Combine date and time
start_datetime = datetime.combine(start_date, start_time)

# Max shift parameter (internal use only)
max_shift = 8761

def parse_input_data(input_text):
    """Parse input text into DataFrame"""
    if not input_text.strip():
        return None
    
    try:
        # Split by lines and convert to float
        values = [float(line.strip()) for line in input_text.strip().split('\n') if line.strip()]
        
        if len(values) == 0:
            return None
            
        df = pd.DataFrame({'water_elevation': values})
        df['timestep'] = range(len(df))
        return df
    except ValueError as e:
        st.error(f"Error parsing data: {str(e)}. Please ensure all values are numbers.")
        return None

def find_best_shift(model_data, survey_data, max_shift=None):
    """Find the best shift for minimum RMSE"""
    best_rmse = float('inf')
    best_shift = 0
    upper_limit = len(model_data) - len(survey_data) + 1
    
    if max_shift is not None:
        upper_limit = min(upper_limit, max_shift)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for shift in range(upper_limit):
        shifted_model = model_data.iloc[shift: shift + len(survey_data)]
        rmse = sqrt(mean_squared_error(shifted_model['water_elevation'].values, survey_data['water_elevation'].values))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift
        
        # Update progress
        if shift % 100 == 0:
            progress_bar.progress(shift / upper_limit)
            status_text.text(f"Analyzing shift: {shift}/{upper_limit}")
    
    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")
    
    return best_rmse, best_shift

# Main analysis
if model_input and survey_input:
    # Parse input data
    model_data = parse_input_data(model_input)
    survey_data = parse_input_data(survey_input)
    
    if model_data is not None and survey_data is not None:
        # Check if model data is longer than survey data
        if len(model_data) < len(survey_data):
            st.error("❌ Model data must be longer than or equal to survey data length.")
        else:
            # Store original data for reference
            model_data_original = model_data.copy()
            survey_data_original = survey_data.copy()
            
            # Apply zero-mean conversion if selected
            if apply_zero_mean:
                model_mean = model_data['water_elevation'].mean()
                survey_mean = survey_data['water_elevation'].mean()
                
                model_data['water_elevation'] = model_data['water_elevation'] - model_mean
                survey_data['water_elevation'] = survey_data['water_elevation'] - survey_mean
                
                st.info(f"✓ Zero-mean conversion applied | Model mean: {model_mean:.4f} m | Survey mean: {survey_mean:.4f} m")
            
            # Display data info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Model Data")
                st.write(f"Total records: {len(model_data)}")
                if apply_zero_mean:
                    st.write("*(Showing zero-meaned values)*")
                st.write(model_data.head(10))
            
            with col2:
                st.subheader("📊 Survey Data")
                st.write(f"Total records: {len(survey_data)}")
                st.write(f"Start time: {start_datetime.strftime('%d-%b-%Y %H:%M')}")
                if apply_zero_mean:
                    st.write("*(Showing zero-meaned values)*")
                st.write(survey_data.head(10))
            
            # Run analysis button
            if st.button("🔍 Run Analysis", type="primary"):
                with st.spinner("Finding best alignment..."):
                    # Find the best shift and RMSE
                    best_rmse, best_shift = find_best_shift(model_data, survey_data, max_shift)
                    
                    # Align the model data based on the best shift
                    aligned_model_data = model_data.iloc[best_shift: best_shift + len(survey_data)]
                    
                    # Create datetime index for survey data
                    datetime_index = pd.date_range(start=start_datetime, periods=len(survey_data), freq='h')
                    
                    # Create aligned output DataFrame
                    output_df = pd.DataFrame()
                    output_df['datetime'] = datetime_index
                    output_df['timestep'] = survey_data['timestep'].values
                    output_df['model_data'] = aligned_model_data['water_elevation'].reset_index(drop=True).values
                    output_df['survey_data'] = survey_data['water_elevation'].values
                    
                    # If zero-mean was applied, also create original scale output
                    if apply_zero_mean:
                        # Add back the means to get original scale
                        output_df['model_data_original'] = output_df['model_data'] + model_mean
                        output_df['survey_data_original'] = output_df['survey_data'] + survey_mean
                    
                    # Calculate final RMSE
                    final_rmse = sqrt(mean_squared_error(aligned_model_data['water_elevation'].values, survey_data['water_elevation'].values))
                    
                    # Calculate RMSE(%)
                    max_value_model = model_data['water_elevation'].max()
                    min_value_model = model_data['water_elevation'].min()
                    rmse_percent = (final_rmse / (max_value_model - min_value_model)) * 100
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Shift", f"{best_shift} timesteps")
                    with col2:
                        st.metric("RMSE", f"{final_rmse:.4f} m")
                    with col3:
                        st.metric("RMSE (%)", f"{rmse_percent:.2f}%")
                    
                    if apply_zero_mean:
                        st.caption("*RMSE calculated on zero-meaned data*")
                    
                    # Plotting
                    st.subheader("📈 Water Elevation Comparison")
                    
                    fig, ax = plt.subplots(figsize=(14, 7))
                    
                    ax.plot(output_df['datetime'], output_df['survey_data'], 
                           label='Survey Data', linewidth=2, color='#2E86AB', alpha=0.8)
                    ax.plot(output_df['datetime'], output_df['model_data'], 
                           label='Model Data', linewidth=2, color='#A23B72', alpha=0.8, linestyle='--')
                    
                    ylabel = 'Water Elevation - Zero-Meaned (m)' if apply_zero_mean else 'Water Elevation (m)'
                    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
                    ax.set_xlabel('Date Time', fontsize=12, fontweight='bold')
                    ax.set_title('Comparison between Survey Data and Model Data', 
                               fontsize=14, fontweight='bold', pad=20)
                    ax.legend(loc='best', fontsize=11, framealpha=0.9)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Add zero line if showing zero-meaned data
                    if apply_zero_mean:
                        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Show statistics
                    st.subheader("📊 Statistical Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Survey Data Statistics:**")
                        st.write(survey_data['water_elevation'].describe())
                    
                    with col2:
                        st.write("**Aligned Model Data Statistics:**")
                        st.write(aligned_model_data['water_elevation'].describe())
                    
                    # Download aligned data
                    st.subheader("💾 Download Results")
                    
                    # Convert to CSV
                    csv = output_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Aligned Data (CSV)",
                        data=csv,
                        file_name=f"aligned_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
else:
    st.info("👆 Please paste both Model Data and Survey Data in the sidebar to begin analysis")
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions:
    1. **Paste Model Data**: In the sidebar, paste water elevation values (one value per line)
       ```
       123.5
       124.6
       125.7
       126.3
       ```
    2. **Paste Survey Data**: Paste survey water elevation values (one value per line)
    3. **Datum Conversion**: Check "Apply zero-mean conversion" to normalize both datasets (recommended when tide datums don't match)
    4. **Set Start Time**: Specify the start date and time for your survey data
    5. **Run Analysis**: Click the "Run Analysis" button to find the best alignment
    
    The application will:
    - Convert both datasets to zero-mean (if selected) to handle different tide datums
    - Find the optimal time shift between model and survey data
    - Calculate RMSE and RMSE(%) on the normalized data
    - Generate comparison plots (with option to view original datum)
    - Allow you to download the aligned results
    
    ### 💡 Tips:
    - Model data should be longer than or equal to survey data
    - Make sure all values are numbers (decimals allowed)
    - Each value should be on a new line
    - The analysis assumes hourly data frequency
    - Use zero-mean conversion when your model and survey data use different vertical datums
    """)
