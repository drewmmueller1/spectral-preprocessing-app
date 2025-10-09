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
        pls_components = st.sidebar.slider("Max PLS Components for iPLS", 1, 10, 2)
        top_intervals = st.sidebar.slider("Max Intervals to Select", 1, 10, 3)
   
    # Apply preprocessing based on selections
    processed_df = df.copy()
   
    if do_smooth:
        st.info("Applying smoothing/derivative...")
        for col in data_cols:
            processed_df[col] = savgol_filter(processed_df[col], window_length=window_length, polyorder=polyorder, deriv=deriv)
   
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
   
    if do_snv:
        st.info("Applying SNV...")
        for col in data_cols:
            mean_val = processed_df[col].mean()
            std_val = processed_df[col].std()
            if std_val != 0:
                processed_df[col] = (processed_df[col] - mean_val) / std_val
    
    # iPLS after SNV
    ipls_fig = None
    if do_ipls and do_snv:
        st.info("Applying iPLS Feature Selection...")
        # Prepare X and y
        labels = [col.split('_')[0] if '_' in col else col for col in data_cols]
        le = LabelEncoder()
        y = le.fit_transform(labels)
        X = processed_df[data_cols].T.values  # samples x variables (wl)
        
        num_unique_y = len(np.unique(y))
        if num_unique_y < 2:
            st.error("iPLS requires at least 2 unique classes in labels.")
            st.stop()
        
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
        full_num_wl = len(full_x)
        
        # Divide into intervals
        n_vars = X.shape[1]
        if n_vars == 0:
            st.error("No variables after preprocessing.")
            st.stop()
        interval_size = n_vars // n_intervals
        intervals = []
        for i in range(n_intervals):
            start = i * interval_size
            end = min((i + 1) * interval_size, n_vars)
            intervals.append((start, end))
        
        kf = KFold(n_splits=5)
        max_ncomp = pls_components
        max_iter = top_intervals
        
        # Forward iPLS selection
        selected_intervals = []
        current_selected_vars = []
        rmse_history = []
        best_ncomp_history = []
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter:
            iteration += 1
            candidates_rmse = []
            candidates_ncomp = []
            candidates_int_idx = []
            
            for i, (start, end) in enumerate(intervals):
                if (start, end) in selected_intervals:
                    continue
                temp_vars = current_selected_vars + list(range(start, end))
                if not temp_vars:
                    continue
                X_temp = X[:, temp_vars]
                n_temp_vars = X_temp.shape[1]
                
                best_rmse_temp = np.inf
                best_nc_temp = 1
                for nc in range(1, min(max_ncomp, n_temp_vars, num_unique_y - 1) + 1):
                    rmse_cv = []
                    for train_idx, test_idx in kf.split(X_temp):
                        X_train, X_test = X_temp[train_idx], X_temp[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        pls = PLSRegression(n_components=nc)
                        pls.fit(X_train, y_train)
                        y_pred = pls.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        rmse_cv.append(rmse)
                    avg_rmse = np.mean(rmse_cv)
                    if avg_rmse < best_rmse_temp:
                        best_rmse_temp = avg_rmse
                        best_nc_temp = nc
                candidates_rmse.append(best_rmse_temp)
                candidates_ncomp.append(best_nc_temp)
                candidates_int_idx.append(i)
            
            if not candidates_rmse:
                break
            
            best_cand = np.argmin(candidates_rmse)
            new_rmse = candidates_rmse[best_cand]
            
            # For first iteration, always add
            if iteration == 1:
                pass
            else:
                # Compute current best RMSE
                if current_selected_vars:
                    X_curr = X[:, current_selected_vars]
                    n_curr_vars = X_curr.shape[1]
                    best_rmse_curr = np.inf
                    best_nc_curr = 1
                    for nc in range(1, min(max_ncomp, n_curr_vars, num_unique_y - 1) + 1):
                        rmse_cv = []
                        for train_idx, test_idx in kf.split(X_curr):
                            X_train, X_test = X_curr[train_idx], X_curr[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            pls = PLSRegression(n_components=nc)
                            pls.fit(X_train, y_train)
                            y_pred = pls.predict(X_test)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            rmse_cv.append(rmse)
                        avg_rmse = np.mean(rmse_cv)
                        if avg_rmse < best_rmse_curr:
                            best_rmse_curr = avg_rmse
                            best_nc_curr = nc
                    if new_rmse >= best_rmse_curr:
                        improved = False
                        continue
                else:
                    improved = False
                    continue
            
            # Add the best candidate
            add_int_idx = candidates_int_idx[best_cand]
            add_int = intervals[add_int_idx]
            selected_intervals.append(add_int)
            add_vars = list(range(add_int[0], add_int[1]))
            current_selected_vars.extend(add_vars)
            rmse_history.append(new_rmse)
            best_ncomp_history.append(candidates_ncomp[best_cand])
            
            st.info(f"Iteration {iteration}/{max_iter}: selected interval {add_int_idx + 1} (RMSECV={new_rmse:.5f}, nLV={candidates_ncomp[best_cand]})")
        
        # Compute single interval RMSE and ncomp for plot
        single_rmse = []
        single_ncomp = []
        for i, (start, end) in enumerate(intervals):
            X_int = X[:, start:end]
            n_int_vars = X_int.shape[1]
            if n_int_vars == 0:
                single_rmse.append(np.inf)
                single_ncomp.append(0)
                continue
            best_rmse_int = np.inf
            best_nc_int = 1
            for nc in range(1, min(max_ncomp, n_int_vars, num_unique_y - 1) + 1):
                rmse_cv = []
                for train_idx, test_idx in kf.split(X_int):
                    X_train, X_test = X_int[train_idx], X_int[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    pls = PLSRegression(n_components=nc)
                    pls.fit(X_train, y_train)
                    y_pred = pls.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    rmse_cv.append(rmse)
                avg_rmse = np.mean(rmse_cv)
                if avg_rmse < best_rmse_int:
                    best_rmse_int = avg_rmse
                    best_nc_int = nc
            single_rmse.append(best_rmse_int)
            single_ncomp.append(best_nc_int)
        
        # Compute global RMSECV
        X_full = X
        n_full_vars = X_full.shape[1]
        best_rmse_global = np.inf
        best_nc_global = 1
        for nc in range(1, min(max_ncomp, n_full_vars, num_unique_y - 1) + 1):
            rmse_cv = []
            for train_idx, test_idx in kf.split(X_full):
                X_train, X_test = X_full[train_idx], X_full[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                pls = PLSRegression(n_components=nc)
                pls.fit(X_train, y_train)
                y_pred = pls.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_cv.append(rmse)
            avg_rmse = np.mean(rmse_cv)
            if avg_rmse < best_rmse_global:
                best_rmse_global = avg_rmse
                best_nc_global = nc
        global_rmse = best_rmse_global
        
        # iPLS Plot: First iteration intervals
        fig_ipls, ax = plt.subplots(figsize=(12, 6))
        
        max_rmse_single = max([r for r in single_rmse if r != np.inf])
        offset = 0.01 * max_rmse_single if max_rmse_single > 0 else 1
        selected_int_indices = [intervals.index(intv) for intv in selected_intervals]
        
        for i, (start, end) in enumerate(intervals):
            if single_rmse[i] == np.inf:
                continue
            x_start = full_x[start]
            x_end = full_x[end - 1] if end < len(full_x) else full_x[-1]
            color = 'green' if i in selected_int_indices else 'blue'
            alpha = 0.7 if color == 'green' else 0.5
            ax.fill_between([x_start, x_end], 0, single_rmse[i], color=color, alpha=alpha)
            mid_x = (x_start + x_end) / 2
            ax.text(mid_x, single_rmse[i] + offset, str(single_ncomp[i]), ha='center', va='bottom', fontsize=8)
        
        ax.axhline(global_rmse, color='black', linestyle='--', linewidth=2, label=f'Global RMSECV ({best_nc_global} LVs)')
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
        ipls_fig = fig_ipls
        
        # RMSE vs Iterations plot
        st.subheader("iPLS RMSECV vs Iterations")
        if rmse_history:
            fig_rmse, ax_rmse = plt.subplots(figsize=(8, 5))
            iters = range(1, len(rmse_history) + 1)
            ax_rmse.plot(iters, rmse_history, 'bo-', label='Selected Model')
            ax_rmse.axhline(global_rmse, color='black', ls='--', label='Global Model')
            ax_rmse.set_xlabel('Iteration')
            ax_rmse.set_ylabel('RMSECV')
            ax_rmse.legend()
            ax_rmse.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_rmse)
        
        # Filter to selected variables (rows)
        if current_selected_vars:
            current_selected_vars = sorted(current_selected_vars)
            processed_df = processed_df.iloc[current_selected_vars].reset_index(drop=True)
        else:
            st.warning("No intervals selected.")
        
        st.success(f"iPLS selected {len(selected_intervals)} intervals ({len(current_selected_vars)} variables) from {n_intervals} possible.")
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
    if ipls_fig is not None:
        st.pyplot(ipls_fig)
    else:
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
            cum_var_scree = np.cumsum(var_ratio)
           
            # Create subplot: bar for %var, line for cumulative
            fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
           
            # Bar: % variance per PC
            fig_scree.add_trace(
                go.Bar(x=[f'PC{i+1}' for i in range(n_scree)], y=var_ratio,
                       name='% Variance', marker_color='lightblue'),
                secondary_y=False
            )
           
            # Add % labels above bars
            for i, v in enumerate(var_ratio):
                fig_scree.add_annotation(x=f'PC{i+1}', y=v, text=f'{v:.1f}%', showarrow=False,
                                         yshift=10, font=dict(size=10))
           
            # Line: Cumulative % variance
            fig_scree.add_trace(
                go.Scatter(x=[f'PC{i+1}' for i in range(n_scree)], y=cum_var_scree,
                           mode='lines+markers', name='Cumulative % Variance', line=dict(color='red', dash='dash')),
                secondary_y=True
            )
           
            fig_scree.update_layout(title=f"Scree Plot (Showing {n_scree} PCs: ≥99% + 2 more)",
                                    xaxis_title="Principal Components",
                                    yaxis_title="% Variance Explained", yaxis2_title="Cumulative % Variance")
            fig_scree.update_yaxes(range=[0, max(var_ratio.max(), cum_var_scree[-1]) * 1.1], secondary_y=False)
            fig_scree.update_yaxes(range=[0, 100], secondary_y=True)
           
            st.plotly_chart(fig_scree, use_container_width=True)
           
            # Total variance info
            st.info(f"Total variance explained by shown PCs: {cum_var_scree[-1]:.1f}% (≥99% reached at PC{n_99})")
       
        # 4. Factor Loadings Plot
        if show_loadings:
            st.subheader("Factor Loadings Plot (Top 3 PCs)")
            # First 3 PCs
            max_pcs = min(3, n_total_pcs)
            var_ratios_top = var_ratios[:max_pcs]
           
            # Filter valid PCs (>0% var)
            valid_indices = [i for i in range(max_pcs) if var_ratios_top[i] > 0]
            num_valid = len(valid_indices)
           
            if num_valid == 0:
                st.warning("No PCs with >0% variance.")
            else:
                st.info(f"Showing loadings for {num_valid} valid PCs (out of top 3)")
               
                # Subset loadings (use abs for magnitude)
                loadings = pd.DataFrame(pca_full.components_[valid_indices],
                                        columns=X_num.columns,
                                        index=[f'PC{i+1}' for i in valid_indices])
                loadings_abs = loadings.abs()
               
                if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                    # Vertical grouped bars (variables on x)
                    fig_loadings = go.Figure()
                    colors = px.colors.qualitative.Set3[:num_valid]
                   
                    # Sort variables by max abs loading (descending) for bars
                    max_loadings = loadings_abs.max(axis=0)
                    sorted_vars = max_loadings.sort_values(ascending=False).index
                   
                    # Width and offset for grouped bars
                    width = 0.25
                    for i, pc in enumerate(loadings.index):
                        pc_data = loadings_abs.loc[pc].loc[sorted_vars]
                        x_pos = np.arange(len(sorted_vars)) + (i - (num_valid - 1) / 2) * width
                        fig_loadings.add_trace(go.Bar(y=pc_data.values, x=sorted_vars,
                                                      name=pc, marker_color=colors[i], width=width,
                                                      base=0, offsetgroup=i))
                   
                    fig_loadings.update_layout(barmode='group',
                                               height=400, showlegend=True,
                                               title="Loadings: Grouped Bar Graph (Abs Values)",
                                               xaxis_title="Variables",
                                               yaxis_title="Loading Magnitude")
                    fig_loadings.update_xaxes(tickangle=45, tickfont=dict(size=9))
                   
                else: # Connected Scatterplot (Continuous, e.g., Spectroscopy)
                    # Prepare for line plot: Melt to long format, preserve original variable order
                    loadings_melt = loadings_abs.reset_index().melt(id_vars='index', var_name='Variable', value_name='Loading')
                    loadings_melt['PC'] = loadings_melt['index'] # Use PC name as color/group
                   
                    # Original order for continuous (e.g., wavelengths)
                    original_vars = X_num.columns.tolist()
                    loadings_melt['Variable'] = pd.Categorical(loadings_melt['Variable'], categories=original_vars, ordered=True)
                    loadings_melt = loadings_melt.sort_values(['PC', 'Variable'])
                   
                    # Line plot: X=Variable, Y=Loading, color=PC, connected lines per PC, no markers
                    fig_loadings = px.line(loadings_melt, x='Variable', y='Loading', color='PC',
                                           markers=False,
                                           title="Loadings: Connected Line Plot (Abs Values)",
                                           labels={'Variable': 'Factors/Variables', 'Loading': 'Loading Magnitude'})
                    fig_loadings.update_traces(line=dict(width=2, dash='solid')) # Continuous solid lines
                    fig_loadings.update_xaxes(tickangle=45, tickfont=dict(size=9))
                   
                    if len(original_vars) > 50:
                        st.warning("Many variables (>50)—zoom/pan the plot for details in spectroscopy data.")
               
                st.plotly_chart(fig_loadings, use_container_width=True)
               
                # Show loadings table
                st.subheader("Loadings Table (Top 3 PCs)")
                st.dataframe(loadings)
       
        # Download PCA results
        st.subheader("Download PCA Results")
        col1, col2 = st.columns(2)
        with col1:
            # PC Scores (transformed data)
            pca_save = PCA(n_components=num_save_pcs)
            X_pca_save = pca_save.fit_transform(X_scaled)
            df_scores = pd.DataFrame(X_pca_save, columns=[f'PC{i+1}' for i in range(num_save_pcs)])
            df_scores['label'] = y # Use simplified labels
            csv_scores = df_scores.to_csv(index=False)
            st.download_button("Download PC Scores CSV", csv_scores, "pc_scores.csv", "text/csv")
        with col2:
            # Loadings
            loadings_save = pd.DataFrame(pca_full.components_[:num_save_pcs],
                                         columns=X_num.columns,
                                         index=[f'PC{i+1}' for i in range(num_save_pcs)])
            csv_loadings = loadings_save.to_csv(index=True)
            st.download_button("Download Loadings CSV", csv_loadings, "pca_loadings.csv", "text/csv")
       
        st.info(f"Downloads include top {num_save_pcs} PCs.")
   
else:
    st.info("Please upload a CSV file to proceed.")
