import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter

# Title and instructions
st.title("Spectral Preprocessing and PCA Visualization App")
st.markdown("""
This app processes and visualizes spectral data via PCA.
**Required Format:**
- Column-wise data: First column is spectral axis (wavelengths in nm for MSP or wavenumbers in cm^{-1} for FTIR) (numeric).
- Row 1 contains labels for data columns (sample names, e.g., "sample1_1", "sample1_2").
- Subsequent rows: Spectral axis values followed by intensity values for each sample.
Upload a CSV file below to get started.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
   
    # Assume first column is 'spectral_axis'
    if len(df.columns) < 2:
        st.error("CSV must have at least 2 columns: spectral axis and data.")
        st.stop()
   
    spectral_col = df.columns[0]
    data_cols = df.columns[1:]
   
    # Convert spectral axis to numeric
    df[spectral_col] = pd.to_numeric(df[spectral_col], errors='coerce')
    df = df.dropna(subset=[spectral_col])
   
    if df.empty:
        st.error("No data after cleaning spectral axis.")
        st.stop()
    
    # Select spectrum type for filtering
    st.subheader("Select Spectrum Type for Filtering")
    spectrum_type = st.radio("Choose spectrum type:", ["MSP Spectra", "FTIR Spectra"])
    
    # Apply filtering based on type
    if spectrum_type == "MSP Spectra":
        df = df[df[spectral_col] >= 300]
        filter_msg = "Wavelengths filtered to >= 300 nm."
    elif spectrum_type == "FTIR Spectra":
        df = df[(df[spectral_col] < 1800) | (df[spectral_col] > 2400)]
        filter_msg = "Wavenumbers filtered excluding 1800-2400 cm^{-1}."
    
    if df.empty:
        st.error("No data after applying filter.")
        st.stop()
    
    st.success(f"Loaded {len(data_cols)} samples. {filter_msg}")
   
    # Preprocessing options in sidebar
    st.sidebar.subheader("Preprocessing Options")
    do_normalize = st.sidebar.checkbox("Normalize (scale by max)", value=False)
    do_smooth = st.sidebar.checkbox("Smooth (Savitzky-Golay filter)", value=False)
    if do_smooth:
        window_length = st.sidebar.slider("SG Window Length (odd number recommended)", min_value=3, max_value=101, value=15, step=2)
        polyorder = st.sidebar.slider("SG Polyorder", min_value=1, max_value=5, value=1)
        deriv = st.sidebar.slider("SG Derivative Order", min_value=0, max_value=3, value=0)
        if polyorder >= window_length:
            st.sidebar.warning("Polyorder should be less than window length for best results.")
    do_snv = st.sidebar.checkbox("SNV Standardization", value=False)
   
    # Apply preprocessing based on selections
    processed_df = df.copy()
   
    if do_normalize:
        st.info("Applying normalization...")
        for col in data_cols:
            max_val = processed_df[col].max()
            if max_val != 0:
                processed_df[col] = processed_df[col] / max_val
   
    if do_smooth:
        st.info("Applying smoothing/derivative...")
        for col in data_cols:
            processed_df[col] = savgol_filter(processed_df[col], window_length=window_length, polyorder=polyorder, deriv=deriv)
   
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
        ax1.plot(processed_df[spectral_col], processed_df[col], alpha=0.7)
    ax1.set_xlabel('Spectral Axis')
    ax1.set_ylabel('Processed Intensity')
    ax1.set_title(f'All Processed Spectra ({spectrum_type})')
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
        ax2.plot(processed_df[spectral_col], avg, label=f'{prefix} Average', linewidth=2)
    ax2.set_xlabel('Spectral Axis')
    ax2.set_ylabel('Processed Intensity')
    ax2.set_title(f'Averaged Processed Spectra ({spectrum_type})')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig2)
   
    # Button to save the pre-processed data
    csv_processed = processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Save Pre-Processed Data",
        data=csv_processed,
        file_name='preprocessed_spectra.csv',
        mime='text/csv'
    )
   
    # PCA Checkbox and Section
    do_pca = st.checkbox("Perform PCA Analysis", value=False)
   
    if do_pca:
        st.subheader("PCA Analysis")
        # Prepare data for PCA: Transpose to rows=samples, columns=wavenumbers
        X = processed_df[data_cols].T.reset_index(drop=True) # rows=samples, columns=wavenumbers
        labels = [col.split('_')[0] if '_' in col else col for col in data_cols]
        y = pd.Series(labels)
       
        df_pca = X.copy()
        df_pca['label'] = y
        st
