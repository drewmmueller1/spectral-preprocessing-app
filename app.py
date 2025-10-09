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
        fig_ipls, ax_ipls = plt.subplots(figsize=(12, 6))
        
        # Plot averaged spectra in red
        for prefix, avg in averages.items():
            # Note: averages need to be recomputed after filtering? Wait, averages use data_cols, which is updated
            # But since filtering after, need to recompute averages here? No, averages are computed later, but for plot, use original averages? Wait, to match, recompute with original before filter
            # Actually, since plot is for selection, use original processed_df before filter for spectra
            # But to simplify, since user says "the spectrum for each averaged label", use current averages, but since filter changes, perhaps plot before filter.
            # Wait, move averages computation before iPLS.
            # No, for now, assume plot uses full, but to fix, compute plot before filtering.
        
        # Wait, to correct: compute averages before iPLS filter, for the plot.
        # So, move group and averages before iPLS.
        
        # Actually, in code, move the group and averages computation right after SNV, before iPLS.
        # Yes, let's adjust.
        
        # But since code is linear, for plot, compute full_x = processed_df[spectral_col].values before filter.
        full_x = processed_df[spectral_col].values
        # Plot red curves using full averages - but averages computed later.
        # To fix, compute sample_groups and averages before iPLS.
        
        # Insert here:
        sample_groups = {}
        for col in data_cols:
            prefix = col.split('_')[0] if '_' in col else col
            if prefix not in sample_groups:
                sample_groups[prefix] = []
            sample_groups[prefix].append(col)
        
        averages_full = {}
        for prefix, cols in sample_groups.items():
            avg_col = processed_df[cols].mean(axis=1)
            averages_full[prefix] = avg_col
        
        # Now plot red
        for prefix, avg in averages_full.items():
            ax_ipls.plot(full_x, avg, color='red', alpha=0.7, linewidth=1, label=f'{prefix} Average' if prefix == list(averages_full.keys())[0] else "")
        
        # Plot bars as fill_between
        max_rmse = max([r for r in rmse_scores if r != np.inf])
        offset = 0.01 * max_rmse
        for i, (start, end) in enumerate(intervals):
            if rmse_scores[i] == np.inf:
                continue
            x_start = full_x[start]
            x_end = full_x[end - 1]
            color = 'green' if i in best_indices else 'blue'
            alpha = 0.5 if color == 'blue' else 0.7
            ax_ipls.fill_between([x_start, x_end], 0, rmse_scores[i], color=color, alpha=alpha)
            mid_x = (x_start + x_end) / 2
            ax_ipls.text(mid_x, rmse_scores[i] + offset, str(pls_components), ha='center', va='bottom', fontsize=8)
        
        # Dashed line
        ax_ipls.axhline(global_rmse, color='black', linestyle='--', linewidth=2, label='Global RMSECV')
        
        ax_ipls.set_xlabel(spectral_col)
        ax_ipls.set_ylabel('RMSECV')
        ax_ipls.set_title('iPLS Interval Selection (First Iteration)')
        ax_ipls.legend()
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
        mime='text/csv'
    )
   
    # PCA option in sidebar
    st.sidebar.subheader("Analysis Options")
    do_pca = st.sidebar.checkbox("Perform PCA Analysis", value=False)
   
    if do_pca:
        st.subheader("PCA Analysis")
        # Prepare data for PCA: Transpose to rows=samples, columns=wavenumbers
        X = processed_df[data_cols].T.reset_index(drop=True) # rows=samples, columns=wavenumbers
        labels = [col.split('_')[0] if '_' in col else col for col in data_cols]
        y = pd.Series(labels)
       
        df_pca = X.copy()
        df_pca['label'] = y
        st.info(f"Simplified labels: Unique classes now {df_pca['label'].nunique()}")
       
        # Prepare numerical features
        X_num = df_pca.drop('label', axis=1).select_dtypes(include=[np.number])
        if X_num.empty:
            st.error("No numerical columns found for PCA.")
            st.stop()
       
        # Standard scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)
       
        # Compute full PCA for reuse
        pca_full = PCA()
        X_pca_full = pca_full.fit_transform(X_scaled)
        n_total_pcs = X_pca_full.shape[1]
        var_ratios = pca_full.explained_variance_ratio_
       
        # Sidebar options for PCA plots
        st.sidebar.header("PCA Plot Options")
        show_2d = st.sidebar.checkbox("Show 2D PCA Plot (Static)", value=True)
        show_3d = st.sidebar.checkbox("Show 3D PCA Plot (Interactive)", value=True)
        show_scree = st.sidebar.checkbox("Show Scree Plot", value=True)
        show_loadings = st.sidebar.checkbox("Show Loadings Plot (Top 3 PCs)", value=True)
       
        if show_loadings:
            loadings_type = st.sidebar.selectbox("Loadings Plot Type", ["Bar Graph (Discrete, e.g., GCMS)", "Connected Scatterplot (Continuous, e.g., Spectroscopy)"], index=1)
        else:
            loadings_type = "Connected Scatterplot (Continuous, e.g., Spectroscopy)"
       
        st.sidebar.header("Download Options")
        num_save_pcs = st.sidebar.slider("Number of PCs to Save", 1, min(10, n_total_pcs), 3)
       
        # 1. 2D PCA Plot (Static, first 2 PCs)
        if show_2d and n_total_pcs >= 2:
            st.subheader("2D PCA Plot (PC1 vs PC2)")
            pca_2d = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            df_plot_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
            df_plot_2d['label'] = y
           
            # Matplotlib for static plot
            fig, ax = plt.subplots(figsize=(8, 6))
            unique_labels = df_plot_2d['label'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: color for label, color in zip(unique_labels, colors)}
           
            for label in unique_labels:
                mask = df_plot_2d['label'] == label
                ax.scatter(df_plot_2d[mask]['PC1'], df_plot_2d[mask]['PC2'],
                           c=[color_map[label]], label=label, s=50)
           
            ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
            ax.set_title("Static 2D PCA Plot")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        elif show_2d:
            st.warning("Need at least 2 features for 2D plot.")
       
        # 2. 3D PCA Plot (Interactive, first 3 PCs)
        if show_3d and n_total_pcs >= 3:
            st.subheader("3D PCA Plot (Interactive: Rotate/Zoom with Mouse)")
            pca_3d = PCA(n_components=3)
            X_pca_3d = pca_3d.fit_transform(X_scaled)
            df_plot = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
            df_plot['label'] = y
           
            fig_3d = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='label',
                                   color_discrete_sequence=px.colors.qualitative.Set1)
            fig_3d.update_traces(marker=dict(size=5))
            fig_3d.update_layout(title="Interactive 3D PCA Plot (Fixed to PC1-PC3)",
                                 scene=dict(
                                     xaxis_title=f"PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})",
                                     yaxis_title=f"PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})",
                                     zaxis_title=f"PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})"
                                 ))
           
            st.plotly_chart(fig_3d, use_container_width=True)
        elif show_3d:
            st.warning("Need at least 3 features for 3D plot.")
       
        # 3. Scree Plot
        if show_scree:
            st.subheader("Scree Plot: Variance Explained")
            # Find min n for >=99% cum var
            cum_var = np.cumsum(var_ratios)
            n_99 = np.argmax(cum_var >= 0.99) + 1 if np.any(cum_var >= 0.99) else n_total_pcs
            n_scree = min(n_99 + 2, n_total_pcs)
           
            pca_scree = PCA(n_components=n_scree)
            pca_scree.fit(X_scaled)
            var_ratio = pca_scree.explained_variance_ratio_ * 100 # % variance
            cum_var_scre
