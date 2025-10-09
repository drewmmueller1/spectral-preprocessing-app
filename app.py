import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
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
- Column-wise data: First column is spectral axis (wavelengths in nm for MSP, wavenumbers in cm^{-1} for FTIR, or retention times for GC/MS) (numeric).
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
    
    # Spectrum filtering in sidebar
    st.sidebar.subheader("Spectrum Filtering")
    spectrum_type = st.sidebar.radio("Choose filtering:", ["No Filtering", "MSP Spectra", "FTIR Spectra"], index=0)
    
    # Apply filtering based on type
    if spectrum_type == "MSP Spectra":
        df = df[df[spectral_col] >= 300]
        filter_msg = "Wavelengths filtered to >= 300 nm."
    elif spectrum_type == "FTIR Spectra":
        df = df[(df[spectral_col] < 1800) | (df[spectral_col] > 2400)]
        filter_msg = "Wavenumbers filtered excluding 1800-2400 cm^{-1}."
    else:
        filter_msg = "No filtering applied."
    
    if df.empty:
        st.error("No data after applying filter.")
        st.stop()
    
    st.success(f"Loaded {len(data_cols)} samples. {filter_msg}")
   
    # Preprocessing options in sidebar
    st.sidebar.subheader("Preprocessing Options")
    do_normalize = st.sidebar.checkbox("Normalize (scale by max)", value=False)
    do_zscore = st.sidebar.checkbox("Z-Score Standardization", value=False)
    do_smooth = st.sidebar.checkbox("Smooth (Savitzky-Golay filter)", value=False)
    if do_smooth:
        window_length = st.sidebar.slider("SG Window Length (odd number recommended)", min_value=3, max_value=101, value=15, step=2)
        polyorder = st.sidebar.slider("SG Polyorder", min_value=1, max_value=5, value=1)
        deriv = st.sidebar.slider("SG Derivative Order", min_value=0, max_value=3, value=0)
        if polyorder >= window_length:
            st.sidebar.warning("Polyorder should be less than window length for best results.")
    do_snv = st.sidebar.checkbox("SNV Standardization", value=False)
    do_ipls = st.sidebar.checkbox("iPLS Feature Selection (after SNV)", value=False)
    if do_ipls:
        n_intervals = st.sidebar.slider("Number of iPLS Intervals", 5, 50, 10)
        pls_components = st.sidebar.slider("PLS Components for iPLS", 1, 10, 2)
        top_intervals = st.sidebar.slider("Select top intervals", 1, 10, 3)
   
    # Apply preprocessing based on selections
    processed_df = df.copy()
   
    if do_normalize:
        st.info("Applying normalization...")
        for col in data_cols:
            max_val = processed_df[col].max()
            if max_val != 0:
                processed_df[col] = processed_df[col] / max_val
   
    if do_zscore:
        st.info("Applying Z-Score Standardization...")
        for row_idx in range(len(processed_df)):
            intensities = processed_df.iloc[row_idx, 1:].values  # data_cols
            mean_val = np.mean(intensities)
            std_val = np.std(intensities)
            if std_val != 0:
                processed_df.iloc[row_idx, 1:] = (intensities - mean_val) / std_val
   
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
    
    # iPLS after SNV
    if do_ipls and do_snv:
        st.info("Applying iPLS Feature Selection...")
        # Prepare X and y
        labels = [col.split('_')[0] if '_' in col else col for col in data_cols]
        le = LabelEncoder()
        y = le.fit_transform(labels)
        X = processed_df[data_cols].T.values  # samples x variables
        
        # Compute full averages for plot (before filtering)
        original_data_cols = data_cols.copy()
        sample_groups_full = {}
        for col in original_data_cols:
            prefix = col.split('_')[0] if '_' in col else col
            if prefix not in sample_groups_full:
                sample_groups_full[prefix] = []
            sample_groups_full[prefix].append(col)
        
        averages_full = {}
        for prefix, cols in sample_groups_full.items():
            avg_col = processed_df[cols].mean(axis=1)
            averages_full[prefix] = avg_col
        
        full_x = processed_df[spectral_col].values
        
        # Divide into intervals
        n_vars = X.shape[1]
        interval_size = n_vars // n_intervals
        intervals = []
        for i in range(n_intervals):
            start = i * interval_size
            end = (i + 1) * interval_size if i < n_intervals - 1 else n_vars
            intervals.append((start, end))
        
        # Compute RMSECV for each interval
        rmse_scores = []
        kf = KFold(n_splits=5)
        for start, end in intervals:
            X_int = X[:, start:end]
            if X_int.shape[1] == 0:
                rmse_scores.append(np.inf)
                continue
            pls = PLSRegression(n_components=min(pls_components, X_int.shape[1], len(np.unique(y)) - 1))
            rmse_cv = []
            for train_idx, test_idx in kf.split(X_int):
                X_train, X_test = X_int[train_idx], X_int[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                pls.fit(X_train, y_train)
                y_pred = pls.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_cv.append(rmse)
            rmse_scores.append(np.mean(rmse_cv))
        
        # Compute global RMSECV
        X_full = X
        pls_global = PLSRegression(n_components=min(pls_components, X_full.shape[1], len(np.unique(y)) - 1))
        rmse_cv_global = []
        for train_idx, test_idx in kf.split(X_full):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pls_global.fit(X_train, y_train)
            y_pred = pls_global.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_cv_global.append(rmse)
        global_rmse = np.mean(rmse_cv_global)
        
        # Select top intervals
        best_indices = np.argsort(rmse_scores)[:top_intervals]
        selected_intervals = [intervals[i] for i in best_indices if rmse_scores[i] != np.inf]
        
        # Collect selected variable indices
        selected_var_indices = []
        for start, end in selected_intervals:
            selected_var_indices.extend(range(start, end))
        
        # Filter processed_df and data_cols
        selected_data_cols = [data_cols[j] for j in selected_var_indices]
        processed_df = processed_df[[spectral_col] + selected_data_cols]
        data_cols = selected_data_cols
        
        # iPLS Plot
        st.subheader("iPLS Interval Selection Plot")
        fig_ipls, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars and dashed line on ax
        max_rmse = max([r for r in rmse_scores if r != np.inf])
        offset = 0.01 * max_rmse
        for i, (start, end) in enumerate(intervals):
            if rmse_scores[i] == np.inf:
                continue
            x_start = full_x[start]
            x_end = full_x[end - 1] if end < len(full_x) else full_x[-1]
            color = 'green' if i in best_indices else 'blue'
            alpha = 0.5 if color == 'blue' else 0.7
            ax.fill_between([x_start, x_end], 0, rmse_scores[i], color=color, alpha=alpha)
            mid_x = (x_start + x_end) / 2
            ax.text(mid_x, rmse_scores[i] + offset, str(pls_components), ha='center', va='bottom', fontsize=8)
        
        ax.axhline(global_rmse, color='black', linestyle='--', linewidth=2, label='Global RMSECV')
        ax.set_xlabel(spectral_col)
        ax.set_ylabel('RMSECV')
        ax.legend(loc='upper right')
        
        # Twin axis for spectra
        ax2 = ax.twinx()
        ax2.set_ylabel('Processed Intensity')
        for prefix, avg in averages_full.items():
            ax2.plot(full_x, avg, color='red', alpha=0.7, linewidth=1)
        
        plt.title('iPLS Interval Selection (First Iteration)')
        plt.tight_layout()
        st.pyplot(fig_ipls)
        
        st.success(f"iPLS selected {len(selected_var_indices)} variables from top {len(selected_intervals)} intervals.")
    elif do_ipls:
        st.warning("iPLS requires SNV to be applied first.")
   
    # Now compute sample_groups and averages with possibly filtered data
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
   
    # Plot 1: All individual spectra (no legend)
    st.subheader("All Processed Spectra")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for col in data_cols:
        ax1.plot(processed_df[spectral_col], processed_df[col], alpha=0.7)
    ax1.set_xlabel('Spectral Axis')
    ax1.set_ylabel('Processed Intensity')
    title1 = 'All Processed Spectra' if spectrum_type == "No Filtering" else f'All Processed Spectra ({spectrum_type})'
    ax1.set_title(title1)
    plt.tight_layout()
    st.pyplot(fig1)
   
    # Plot 2: Averaged spectra
    st.subheader("Averaged Processed Spectra")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for prefix, avg in averages.items():
        ax2.plot(processed_df[spectral_col], avg, label=f'{prefix} Average', linewidth=2)
    ax2.set_xlabel('Spectral Axis')
    ax2.set_ylabel('Processed Intensity')
    title2 = 'Averaged Processed Spectra' if spectrum_type == "No Filtering" else f'Averaged Processed Spectra ({spectrum_type})'
    ax2.set_title(title2)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig2)
   
    # Stacked plot option for individual samples
    st.sidebar.subheader("Plot Options")
    do_stacked = st.sidebar.checkbox("Show Stacked Individual Plots", value=False)
    
    if do_stacked:
        st.subheader("Stacked Individual Processed Spectra")
        n_samples = len(data_cols)
        fig3, axs = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples), sharex=True)
        if n_samples == 1:
            axs = [axs]
        for i, col in enumerate(data_cols):
            axs[i].plot(processed_df[spectral_col], processed_df[col], linewidth=1)
            axs[i].set_ylabel('Intensity')
            axs[i].set_title(f'Sample: {col}')
        axs[-1].set_xlabel('Spectral Axis')
        plt.tight_layout()
        st.pyplot(fig3)
   
    # Button to save the pre-processed data
    csv_processed = processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Save Pre-Processed Data",
        data=csv_processed,
        file_name='preprocessed_spectra.csv',
        mime='
