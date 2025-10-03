import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import os

# Streamlit app title and description
st.title("Spectral Data Preprocessing App")
st.markdown("""
This app processes uploaded CSV files containing spectral data.  
**Required Format:**  
- Column-wise data: First column is wavenumbers/wavelength (numeric).  
- Row 1 contains labels for data columns (sample names, e.g., "sample1_1", "sample1_2").  
- Subsequent rows: Wavenumber values followed by intensity values for each sample.  
Upload a CSV file below to get started.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    
    # Assume first column is 'wavenumber'
    if len(df.columns) < 2:
        st.error("CSV must have at least 2 columns: wavenumbers and data.")
        st.stop()
    
    wavenumber_col = df.columns[0]
    data_cols = df.columns[1:]
    
    # Convert wavenumber to numeric and filter >= 300
    df[wavenumber_col] = pd.to_numeric(df[wavenumber_col], errors='coerce')
    df = df[df[wavenumber_col] >= 300]
    df = df.dropna(subset=[wavenumber_col])
    
    if df.empty:
        st.error("No data after filtering wavenumbers >= 300.")
        st.stop()
    
    st.success(f"Loaded {len(data_cols)} samples. Wavenumbers filtered to >= 300.")
    
    # Checkboxes for preprocessing steps
    st.subheader("Select Preprocessing Steps")
    do_normalize = st.checkbox("Normalize (scale by max)", value=True)
    do_smooth = st.checkbox("Smooth (Savitzky-Golay filter)", value=True)
    do_snv = st.checkbox("SNV Standardization", value=True)
    
    # Apply preprocessing based on selections
    processed_df = df.copy()
    
    if do_normalize:
        st.info("Applying normalization...")
        for col in data_cols:
            max_val = processed_df[col].max()
            if max_val != 0:
                processed_df[col] = processed_df[col] / max_val
    
    if do_smooth:
        st.info("Applying smoothing...")
        for col in data_cols:
            processed_df[col] = savgol_filter(processed_df[col], window_length=15, polyorder=1)
    
    if do_snv:
        st.info("Applying SNV...")
        for col in data_cols:
            mean_val = processed_df[col].mean()
            std_val = processed_df[col].std()
            if std_val != 0:
                processed_df[col] = (processed_df[col] - mean_val) / std_val
    
    # Plot 1: All individual spectra (no legend)
    st.subheader("All Processed Spectra")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for col in data_cols:
        ax1.plot(processed_df[wavenumber_col], processed_df[col], alpha=0.7)
    ax1.set_xlabel('Wavenumber')
    ax1.set_ylabel('Processed Intensity')
    ax1.set_title('All Processed Spectra (Wavenumber >= 300)')
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Group samples for averaging: by prefix before '_'
    st.subheader("Sample Grouping for Averaging")
    sample_groups = {}
    for col in data_cols:
        prefix = col.split('_')[0] if '_' in col else col
        if prefix not in sample_groups:
            sample_groups[prefix] = []
        sample_groups[prefix].append(col)
    
    st.write("Detected groups:", list(sample_groups.keys()))
    
    # Compute averages
    averages = {}
    for prefix, cols in sample_groups.items():
        avg_col = processed_df[cols].mean(axis=1)
        averages[prefix] = avg_col
    
    # Plot 2: Averaged spectra
    st.subheader("Averaged Processed Spectra")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for prefix, avg in averages.items():
        ax2.plot(processed_df[wavenumber_col], avg, label=f'{prefix} Average', linewidth=2)
    ax2.set_xlabel('Wavenumber')
    ax2.set_ylabel('Processed Intensity')
    ax2.set_title('Averaged Processed Spectra (Wavenumber >= 300)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig2)