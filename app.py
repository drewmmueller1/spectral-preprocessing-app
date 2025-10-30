import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mlxtend.plotting import plot_decision_regions
from scipy.signal import savgol_filter
import seaborn as sns

def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))

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
   
    spectral_col =df.columns[0]
    data_cols = df.columns[1:]
   
    # Convert spectral axis to numeric
    df[spectral_col] = pd.to_numeric(df[spectral_col], errors='coerce')
    df = df.dropna(subset=[spectral_col])
   
    if df.empty:
        st.error("No data after cleaning spectral axis.")
        st.stop()
    
    # Spectrum filtering in sidebar
    with st.sidebar:
        with st.expander("Spectrum Filtering", expanded=False):
            spectrum_type = st.radio("Choose filtering:", ["No Filtering", "MSP Spectra", "FTIR Spectra"], index=0)
    
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
   
    # Axis label customization in sidebar
    with st.sidebar:
        with st.expander("Axis Label Customization", expanded=False):
            custom_x_label = st.text_input("X-Axis Label for Spectra and iPLS Plots (leave blank for 'Spectral Axis')", "")
            custom_y_label = st.text_input("Y-Axis Label for Spectra and iPLS Twin Axis (leave blank for 'Processed Intensity')", "")
            custom_loadings_x_label = st.text_input("X-Axis Label for Factor Loadings Plot (leave blank for 'Factors/Variables')", "")
    
    # Set axis labels based on user input or defaults
    x_label = custom_x_label if custom_x_label.strip() else "Spectral Axis"
    y_label = custom_y_label if custom_y_label.strip() else "Processed Intensity"
    loadings_x_label = custom_loadings_x_label if custom_loadings_x_label.strip() else "Factors/Variables"
   
    # Preprocessing options in sidebar
    with st.sidebar:
        with st.expander("Preprocessing Options", expanded=False):
            do_normalize = st.checkbox("Normalize (scale by max)", value=False)
            do_zscore = st.checkbox("Z-Score Standardization", value=False)
            do_smooth = st.checkbox("Smooth (Savitzky-Golay filter)", value=False)
            if do_smooth:
                window_length = st.slider("SG Window Length (odd number recommended)", min_value=3, max_value=101, value=15, step=2)
                polyorder = st.slider("SG Polyorder", min_value=1, max_value=5, value=1)
                deriv = st.slider("SG Derivative Order", min_value=0, max_value=3, value=0)
                if polyorder >= window_length:
                    st.warning("Polyorder should be less than window length for best results.")
            do_snv = st.checkbox("SNV Standardization", value=False)
            do_ipls = st.checkbox("iPLS Feature Selection", value=False)
   
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

    # Compute labels and consistent colors
    labels = [col.split('_')[0] if '_' in col else col for col in data_cols]
    unique_labels = sorted(set(labels))
    n_colors = len(unique_labels)
    wl_range = np.linspace(380, 750, n_colors)
    rgb_ints = [wavelength_to_rgb(wl) for wl in wl_range]
    colors_rgb = [(r/255.0, g/255.0, b/255.0) for r, g, b in rgb_ints]
    label_to_rgb = {label: colors_rgb[i] for i, label in enumerate(unique_labels)}
    color_discrete_map = {label: mcolors.to_hex(rgb) for label, rgb in label_to_rgb.items()}

    # Compute full spectral axis, sample groups, and averages before iPLS
    full_spectral = processed_df[spectral_col].copy()
    sample_groups = {}
    for col in data_cols:
        prefix = col.split('_')[0] if '_' in col else col
        if prefix not in sample_groups:
            sample_groups[prefix] = []
        sample_groups[prefix].append(col)
   
    averages = {}
    for prefix, cols in sample_groups.items():
        averages[prefix] = processed_df[cols].mean(axis=1).copy()

    # Option for separate sample labels
    show_sep_samples = st.checkbox("Show separate sample labels for all spectra", key="sep_all_spec")

    # Plot all processed spectra (full, before any iPLS filtering)
    st.subheader("All Processed Spectra")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for prefix, cols in sample_groups.items():
        color = label_to_rgb[prefix]
        alpha_line = 0.5 if len(cols) > 1 else 1.0
        for col in cols:
            ax1.plot(processed_df[spectral_col], processed_df[col], color=color, alpha=alpha_line)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    title1 = 'All Processed Spectra' if spectrum_type == "No Filtering" else f'All Processed Spectra ({spectrum_type})'
    ax1.set_title(title1)
    plt.tight_layout()
    st.pyplot(fig1)

    if show_sep_samples:
        st.subheader("Sample Labels")
        fig_samp, ax_samp = plt.subplots(figsize=(8, max(4, len(data_cols) * 0.1)))
        ax_samp.axis('off')
        y_pos = np.linspace(0.9, 0.1, len(data_cols))
        for i, col in enumerate(data_cols):
            prefix = labels[i]
            color = label_to_rgb[prefix]
            ax_samp.text(0.05, y_pos[i], col, transform=ax_samp.transAxes, va='center', fontsize=max(8, 100 / len(data_cols)), color=color)
        st.pyplot(fig_samp)
    
    # Display sample grouping
    st.subheader("Sample Grouping for Averaging")
    st.write("Detected groups:", list(sample_groups.keys()))

    # Option for separate legend
    show_sep_legend_avg = st.checkbox("Show separate legend for averaged spectra", key="sep_avg")
   
    # Plot 2: Averaged spectra using pre-iPLS data
    st.subheader("Averaged Processed Spectra")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    prefixes = list(averages.keys())
    num_groups = len(prefixes)
    for i, prefix in enumerate(prefixes):
        avg = averages[prefix]
        color = label_to_rgb[prefix]
        ax2.plot(full_spectral, avg, color=color, label=f'{prefix} Average', linewidth=2)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    title2 = 'Averaged Processed Spectra' if spectrum_type == "No Filtering" else f'Averaged Processed Spectra ({spectrum_type})'
    ax2.set_title(title2)
    if not show_sep_legend_avg and num_groups > 1:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig2)

    if show_sep_legend_avg and num_groups > 0:
        st.subheader("Legend for Averaged Spectra")
        fig_leg, ax_leg = plt.subplots(figsize=(4, num_groups * 0.5 + 1))
        ax_leg.axis('off')
        legend_elements = [Line2D([0], [0], color=label_to_rgb[prefix], lw=2, label=f'{prefix} Average') for prefix in prefixes]
        ax_leg.legend(handles=legend_elements, loc='center')
        st.pyplot(fig_leg)
    
    # iPLS 
    ipls_fig = None
    if do_ipls:
        st.info("Applying iPLS Feature Selection...")
        # Prepare X and y
        le_ipls = LabelEncoder()
        y = le_ipls.fit_transform(labels)
        X = processed_df[data_cols].T.values  # samples x variables (wl)
        
        num_unique_y = len(np.unique(y))
        if num_unique_y < 2:
            st.error("iPLS requires at least 2 unique classes in labels.")
            st.stop()
        
        # Use pre-computed averages and full_x for iPLS plot
        averages_full = averages
        full_x = full_spectral.values

        # iPLS Parameters in sidebar
        with st.sidebar:
            with st.expander("iPLS Parameters", expanded=False):
                n_vars = X.shape[1]
                if n_vars == 0:
                    st.error("No variables after preprocessing.")
                    st.stop()
                
                n_intervals = st.slider("Number of Intervals", min_value=5, max_value=100, value=min(50, max(10, n_vars // 10)), step=1)
                interval_size = n_vars // n_intervals
                max_ncomp = st.slider("Maximum Number of Components", min_value=1, max_value=min(20, n_vars, num_unique_y - 1), value=min(10, n_vars // 10, num_unique_y - 1), step=1)
                max_iter = st.slider("Maximum Iterations", min_value=1, max_value=100, value=n_intervals, step=1)
                st.info(f"iPLS Parameters: n_intervals={n_intervals}, max_ncomp={max_ncomp}, max_iter={max_iter}")
        
        # Generate intervals
        intervals = []
        for i in range(n_intervals):
            start = i * interval_size
            end = min((i + 1) * interval_size, n_vars)
            intervals.append((start, end))
        
        kf = KFold(n_splits=5)
        
        # Forward iPLS selection with safeguard
        selected_intervals = []
        current_selected_vars = []
        rmse_history = []
        best_ncomp_history = []
        improved = True
        iteration = 0
        
        try:
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
        except Exception as e:
            st.error(f"iPLS forward selection failed with parameters n_intervals={n_intervals}, max_ncomp={max_ncomp}, max_iter={max_iter}. Error: {str(e)}")
            selected_intervals = []
            current_selected_vars = []
            rmse_history = []
            best_ncomp_history = []
        
        # Compute single interval RMSE and ncomp for plot with safeguard
        single_rmse = []
        single_ncomp = []
        try:
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
        except Exception as e:
            st.error(f"iPLS single interval computation failed with parameters n_intervals={n_intervals}, max_ncomp={max_ncomp}. Error: {str(e)}")
            single_rmse = [np.inf] * len(intervals)
            single_ncomp = [0] * len(intervals)
        
        # Compute global RMSECV with safeguard
        global_rmse = np.inf
        best_nc_global = 1
        try:
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
        except Exception as e:
            st.error(f"iPLS global RMSECV computation failed with parameters max_ncomp={max_ncomp}. Error: {str(e)}")
            global_rmse = np.inf
            best_nc_global = 1

        # Option for separate legends
        show_sep_legend_ipls = st.checkbox("Show separate legends for iPLS plot", key="sep_ipls")
        
        # iPLS Plot: First iteration intervals
        st.subheader("iPLS Interval Selection Plot")
        fig_ipls, ax = plt.subplots(figsize=(12, 6))
        
        max_rmse_single = max([r for r in single_rmse if r != np.inf])
        offset = 0.01 * max_rmse_single if max_rmse_single > 0 else 1
        selected_int_indices = [intervals.index(intv) for intv in selected_intervals] if selected_intervals else []
        
        for i, (start, end) in enumerate(intervals):
            if single_rmse[i] == np.inf:
                continue
            x_start = full_x[start]
            x_end = full_x[end - 1] if end < len(full_x) else full_x[-1]
            color = 'green' if i in selected_int_indices else 'red'
            alpha = 0.7 if color == 'green' else 0.5
            ax.fill_between([x_start, x_end], 0, single_rmse[i], color=color, alpha=alpha)
            mid_x = (x_start + x_end) / 2
            ax.text(mid_x, single_rmse[i] + offset, str(single_ncomp[i]), ha='center', va='bottom', fontsize=8)
        
        ax.axhline(global_rmse, color='black', linestyle='--', linewidth=2, label=f'Global RMSECV ({best_nc_global} LVs)')
        ax.set_xlabel(x_label)
        ax.set_ylabel('RMSECV')
        
        # Legend for colors and line
        green_patch = Patch(facecolor='green', alpha=0.7, label='Selected Intervals')
        red_patch = Patch(facecolor='red', alpha=0.5, label='Non-selected Intervals')
        
        # Twin axis for spectra
        ax2 = ax.twinx()
        ax2.set_ylabel(y_label)
        prefixes_ipls = list(averages_full.keys())
        for i, prefix in enumerate(averages_full):
            color = label_to_rgb[prefix]
            ax2.plot(full_x, averages_full[prefix], color=color, alpha=0.7, linewidth=1, label=prefix)
        
        if not show_sep_legend_ipls:
            ax.legend(handles=[green_patch, red_patch], loc='upper right')
            if len(prefixes_ipls) > 0:
                ax2.legend(loc='lower right')
        
        plt.title('iPLS Interval Selection (First Iteration)')
        plt.tight_layout()
        st.pyplot(fig_ipls)

        if show_sep_legend_ipls:
            # Interval legend
            st.subheader("iPLS Interval Legend")
            fig_leg_int, ax_leg_int = plt.subplots(figsize=(3, 2))
            ax_leg_int.axis('off')
            ax_leg_int.legend(handles=[green_patch, red_patch], loc='center')
            st.pyplot(fig_leg_int)

            # Spectra legend
            if prefixes_ipls:
                st.subheader("iPLS Spectra Legend")
                fig_leg_spec, ax_leg_spec = plt.subplots(figsize=(4, len(prefixes_ipls) * 0.5 + 1))
                ax_leg_spec.axis('off')
                legend_elements_spec = [Line2D([0], [0], color=label_to_rgb[prefix], lw=2, label=prefix) for prefix in prefixes_ipls]
                ax_leg_spec.legend(handles=legend_elements_spec, loc='center')
                st.pyplot(fig_leg_spec)
        
        # RMSE vs Iterations plot
        if rmse_history:
            st.subheader("iPLS RMSECV vs Iterations")
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
   
    # Button to save the pre-processed data (post-iPLS if applied)
    csv_processed = processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Save Pre-Processed Data",
        data=csv_processed,
        file_name='preprocessed_spectra.csv',
        mime='text/csv'
    )
   
    # PCA option in sidebar
    with st.sidebar:
        with st.expander("Analysis Options", expanded=False):
            do_pca = st.checkbox("Perform PCA Analysis", value=False)
   
    if do_pca:
        st.subheader("PCA Analysis")
        # Prepare data for PCA: Transpose to rows=samples, columns=wavenumbers
        X = processed_df[data_cols].T.reset_index(drop=True) # rows=samples, columns=wavenumbers
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
        with st.sidebar:
            with st.expander("PCA Plot Options", expanded=False):
                show_2d = st.checkbox("Show 2D PCA Plot (Static)", value=True)
                show_3d = st.checkbox("Show 3D PCA Plot (Interactive)", value=True)
                show_scree = st.checkbox("Show Scree Plot", value=True)
                show_loadings = st.checkbox("Show Loadings Plot (Top 3 PCs)", value=True)

            if show_loadings:
                with st.expander("Loadings Options", expanded=False):
                    loadings_type = st.selectbox("Loadings Plot Type", ["Bar Graph (Discrete, e.g., GCMS)", "Connected Scatterplot (Continuous, e.g., Spectroscopy)"], index=1)

            with st.expander("Download Options", expanded=False):
                num_save_pcs = st.slider("Number of PCs to Save", 1, min(10, n_total_pcs), 3)
       
        # 1. 2D PCA Plot (Static, first 2 PCs)
        if show_2d and n_total_pcs >= 2:
            show_sep_legend_pca2d = st.checkbox("Show separate legend for 2D PCA", key="sep_pca2d")
            st.subheader("2D PCA Plot (PC1 vs PC2)")
            pca_2d = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            df_plot_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
            df_plot_2d['label'] = y
           
            # Matplotlib for static plot
            fig, ax = plt.subplots(figsize=(8, 6))
            unique_labels_pca = df_plot_2d['label'].unique()
            color_map_pca = {label: label_to_rgb[label] for label in unique_labels_pca}
           
            for label in unique_labels_pca:
                mask = df_plot_2d['label'] == label
                ax.scatter(df_plot_2d[mask]['PC1'], df_plot_2d[mask]['PC2'],
                           c=[color_map_pca[label]], label=label, s=50)
           
            ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
            ax.set_title("Static 2D PCA Plot")
            if not show_sep_legend_pca2d:
                ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            if show_sep_legend_pca2d:
                st.subheader("2D PCA Legend")
                fig_leg_pca, ax_leg_pca = plt.subplots(figsize=(4, len(unique_labels_pca) * 0.5 + 1))
                ax_leg_pca.axis('off')
                legend_elements_pca = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map_pca[label], markersize=8, label=label) for label in unique_labels_pca]
                ax_leg_pca.legend(handles=legend_elements_pca, loc='center')
                st.pyplot(fig_leg_pca)
        elif show_2d:
            st.warning("Need at least 2 features for 2D plot.")
       
        # 2. 3D PCA Plot (Interactive, first 3 PCs)
        if show_3d and n_total_pcs >= 3:
            show_sep_legend_pca3d = st.checkbox("Show separate legend for 3D PCA", key="sep_pca3d")
            st.subheader("3D PCA Plot (Interactive: Rotate/Zoom with Mouse)")
            pca_3d = PCA(n_components=3)
            X_pca_3d = pca_3d.fit_transform(X_scaled)
            df_plot = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
            df_plot['label'] = y
           
            fig_3d = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='label',
                                   color_discrete_map=color_discrete_map)
            fig_3d.update_traces(marker=dict(size=5))
            fig_3d.update_layout(title="Interactive 3D PCA Plot (Fixed to PC1-PC3)",
                                 showlegend=not show_sep_legend_pca3d,
                                 scene=dict(
                                     xaxis_title=f"PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})",
                                     yaxis_title=f"PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})",
                                     zaxis_title=f"PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})"
                                 ))
           
            st.plotly_chart(fig_3d, use_container_width=True)

            if show_sep_legend_pca3d:
                st.subheader("3D PCA Legend")
                unique_l = df_plot['label'].unique()
                colors_leg = [color_discrete_map[label] for label in unique_l]
                fig_leg3d = go.Figure()
                for i, label in enumerate(unique_l):
                    fig_leg3d.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color=colors_leg[i], size=10), name=label))
                fig_leg3d.update_layout(showlegend=True, title="3D PCA Legend")
                st.plotly_chart(fig_leg3d)
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
           
            # Create subplot: bar for %var only
            fig_scree = make_subplots(specs=[[{"secondary_y": False}]])
           
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
           
            fig_scree.update_layout(title=f"Scree Plot (Showing {n_scree} PCs: ≥99% + 2 more)",
                                    xaxis_title="Principal Components",
                                    yaxis_title="% Variance Explained")
            fig_scree.update_yaxes(range=[0, var_ratio.max() * 1.1], secondary_y=False)
           
            st.plotly_chart(fig_scree, use_container_width=True)
           
            # Total variance info
            st.info(f"Total variance explained by shown PCs: {cum_var[n_scree-1]:.1f}% (≥99% reached at PC{n_99})")
       
        # 4. Factor Loadings Plot
        if show_loadings:
            show_sep_legend_load = st.checkbox("Show separate legend for loadings", key="sep_load")
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
                    colors_load_rgb = sns.color_palette("Set1", num_valid)
                    colors_load = [mcolors.to_hex(rgb) for rgb in colors_load_rgb]
                   
                    # Sort variables by max abs loading (descending) for bars
                    max_loadings = loadings_abs.max(axis=0)
                    sorted_vars = max_loadings.sort_values(ascending=False).index
                   
                    # Width for grouped bars
                    width = 0.25
                    for i, pc in enumerate(loadings.index):
                        pc_data = loadings_abs.loc[pc].loc[sorted_vars]
                        fig_loadings.add_trace(go.Bar(y=pc_data.values, x=sorted_vars,
                                                      name=pc, marker_color=colors_load[i], width=width,
                                                      base=0, offsetgroup=i))
                   
                    fig_loadings.update_layout(barmode='group',
                                               height=400, showlegend=not show_sep_legend_load,
                                               title="Loadings: Grouped Bar Graph (Abs Values)",
                                               xaxis_title=loadings_x_label,
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
                                           labels={'Variable': loadings_x_label, 'Loading': 'Loading Magnitude'})
                    fig_loadings.update_traces(line=dict(width=2, dash='solid')) # Continuous solid lines
                    fig_loadings.update_layout(showlegend=not show_sep_legend_load)
                    fig_loadings.update_xaxes(tickangle=45, tickfont=dict(size=9))
                   
                    if len(original_vars) > 50:
                        st.warning("Many variables (>50)—zoom/pan the plot for details in spectroscopy data.")
               
                st.plotly_chart(fig_loadings, use_container_width=True)

                if show_sep_legend_load:
                    st.subheader("Loadings Legend")
                    unique_pcs = loadings.index.tolist()
                    if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                        fig_leg_load = go.Figure()
                        for i, pc in enumerate(unique_pcs):
                            fig_leg_load.add_trace(go.Bar(x=[pc], y=[1], marker_color=colors_load[i], name=pc, showlegend=True))
                        fig_leg_load.update_layout(barmode='stack', title="Loadings Legend", yaxis_visible=False)
                        st.plotly_chart(fig_leg_load)
                    else:
                        colors_line_rgb = sns.color_palette("Set1", len(unique_pcs))
                        colors_line = [mcolors.to_hex(rgb) for rgb in colors_line_rgb]
                        fig_leg_line = go.Figure()
                        for i, pc in enumerate(unique_pcs):
                            fig_leg_line.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=colors_line[i], width=4), name=pc))
                        fig_leg_line.update_layout(title="Loadings PC Legend")
                        st.plotly_chart(fig_leg_line)
               
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
