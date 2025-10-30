import streamlit as st
import pandas as pd
import numpy as np
import re
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
   
    spectral_col = df.columns[0]
    data_cols = df.columns[1:]
   
    # Convert spectral axis to numeric
    df[spectral_col] = pd.to_numeric(df[spectral_col], errors='coerce')
    df = df.dropna(subset=[spectral_col])
   
    if df.empty:
        st.error("No data after cleaning spectral axis.")
        st.stop()
    
    # Label parsing
    samples_info = {}
    for col in data_cols:
        match = re.search(r'([mf])(\d{2})_', col)
        if match:
            samples_info[col] = {
                'sex': match.group(1).upper(), 
                'age': int(match.group(2))
            }
        else:
            samples_info[col] = {'sex': 'Unknown', 'age': -1}
    
    info_df = pd.DataFrame(samples_info).T
    st.subheader("Parsed Sample Information")
    st.dataframe(info_df)
    
    st.subheader("Labeling Mode")
    label_mode = st.radio("Label by:", ["Original Prefix", "Sex", "Age"], index=0)
    
    # Compute labels based on mode
    if label_mode == "Original Prefix":
        labels = [col.split('_')[0] if '_' in col else col for col in data_cols]
    elif label_mode == "Sex":
        labels = ["Male" if samples_info[col]['sex'] == "M" else "Female" for col in data_cols]
    elif label_mode == "Age":
        labels = [str(samples_info[col]['age']) if samples_info[col]['age'] >= 0 else "Unknown" for col in data_cols]
    
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
    unique_labels = sorted(set(labels))
    n_colors = len(unique_labels)
    
    if label_mode == "Sex":
        label_to_rgb = {
            "Male": (0.0, 0.0, 1.0),  # Blue
            "Female": (1.0, 0.0, 0.0)  # Red
        }
        color_discrete_map = {
            "Male": mcolors.to_hex((0.0, 0.0, 1.0)),
            "Female": mcolors.to_hex((1.0, 0.0, 0.0))
        }
    else:
        wl_range = np.linspace(380, 750, n_colors)
        rgb_ints = [wavelength_to_rgb(wl) for wl in wl_range]
        colors_rgb = [(r/255.0, g/255.0, b/255.0) for r, g, b in rgb_ints]
        label_to_rgb = {label: colors_rgb[i] for i, label in enumerate(unique_labels)}
        color_discrete_map = {label: mcolors.to_hex(rgb) for label, rgb in label_to_rgb.items()}

    # Compute full spectral axis, sample groups, and averages before iPLS
    full_spectral = processed_df[spectral_col].copy()
    sample_groups = {}
    for i, col in enumerate(data_cols):
        prefix = labels[i]
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
                    for train_idx,
