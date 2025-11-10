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

@st.cache_data
def load_and_preprocess_data(uploaded_file, spectrum_type, do_normalize, do_zscore, do_smooth, window_length, polyorder, deriv, do_snv, do_ipls, n_intervals, max_ncomp, max_iter, label_mode, custom_x_label, custom_y_label, custom_loadings_x_label):
    """
    Cached preprocessing pipeline.
    Returns: processed_df, labels, samples_info, full_spectral, averages, current_selected_vars, x_label, y_label, loadings_x_label, ipls_artifacts
    """
    # Read the CSV with error handling
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty or contains no data.")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}. Please ensure it's a valid CSV with headers and data.")
  
    if len(df.columns) < 2:
        raise ValueError("CSV must have at least 2 columns: spectral axis and data.")
  
    spectral_col = df.columns[0]
    data_cols = df.columns[1:]
  
    # Convert spectral axis to numeric
    df[spectral_col] = pd.to_numeric(df[spectral_col], errors='coerce')
    df = df.dropna(subset=[spectral_col])
  
    if df.empty:
        raise ValueError("No data after cleaning spectral axis.")
   
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
   
    # Compute labels based on mode
    if label_mode == "Original Prefix":
        labels = [col.split('_')[0] if '_' in col else col for col in data_cols]
    elif label_mode == "Sex":
        labels = ["Male" if samples_info[col]['sex'] == "M" else "Female" if samples_info[col]['sex'] == "F" else "Unknown" for col in data_cols]
    elif label_mode == "Age":
        labels = [str(samples_info[col]['age']) if samples_info[col]['age'] >= 0 else "Unknown" for col in data_cols]
   
    # Handle Sex/Age filtering
    if label_mode in ["Sex", "Age"]:
        valid_mask = [l != "Unknown" for l in labels]
        data_cols = [data_cols[i] for i, keep in enumerate(valid_mask) if keep]
        labels = [labels[i] for i, keep in enumerate(valid_mask) if keep]
        samples_info = {col: samples_info[col] for col in data_cols}
        # Crucial: Update processed_df to only include filtered columns
        df = df[[spectral_col] + data_cols]
        processed_df = df.copy()
    else:
        processed_df = df.copy()
   
    # Apply filtering based on type
    if spectrum_type == "MSP Spectra":
        processed_df = processed_df[processed_df[spectral_col] >= 300]
    elif spectrum_type == "FTIR Spectra":
        processed_df = processed_df[(processed_df[spectral_col] < 1800) | (processed_df[spectral_col] > 2400)]
   
    if processed_df.empty:
        raise ValueError("No data after applying filter.")
   
    # Set axis labels
    x_label = custom_x_label if custom_x_label.strip() else "Spectral Axis"
    y_label = custom_y_label if custom_y_label.strip() else "Processed Intensity"
    loadings_x_label = custom_loadings_x_label if custom_loadings_x_label.strip() else "Factors/Variables"
  
    # Apply preprocessing
    if do_smooth:
        for col in data_cols:
            processed_df[col] = savgol_filter(processed_df[col], window_length=window_length, polyorder=polyorder, deriv=deriv)
  
    if do_normalize:
        for col in data_cols:
            max_val = processed_df[col].max()
            if max_val != 0:
                processed_df[col] = processed_df[col] / max_val
  
    if do_zscore:
        for row_idx in range(len(processed_df)):
            intensities = processed_df.iloc[row_idx, 1:].values
            mean_val = np.mean(intensities)
            std_val = np.std(intensities)
            if std_val != 0:
                processed_df.iloc[row_idx, 1:] = (intensities - mean_val) / std_val
  
    if do_snv:
        for col in data_cols:
            mean_val = processed_df[col].mean()
            std_val = processed_df[col].std()
            if std_val != 0:
                processed_df[col] = (processed_df[col] - mean_val) / std_val

    # Clean NaNs/infs
    processed_df_data = processed_df[data_cols]
    if processed_df_data.isnull().any().any():
        processed_df_data = processed_df_data.fillna(processed_df_data.mean())
        processed_df[data_cols] = processed_df_data
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    if processed_df.isnull().any().any():
        processed_df_data = processed_df[data_cols].fillna(processed_df_data.mean())
        processed_df[data_cols] = processed_df_data

    # Compute full_spectral, sample_groups, averages
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

    # iPLS if enabled
    current_selected_vars = []
    ipls_artifacts = None
    if do_ipls:
        le_ipls = LabelEncoder()
        y = le_ipls.fit_transform(labels)
        if label_mode == "Sex":
            if 'Male' in le_ipls.classes_ and 'Female' in le_ipls.classes_:
                y = np.where(np.array(labels) == 'Male', 1, 2)
            else:
                raise ValueError("Unexpected labels for Sex mode in iPLS.")
        X = processed_df[data_cols].T.values
       
        num_unique_y = len(np.unique(y))
        if num_unique_y < 2:
            raise ValueError("iPLS requires at least 2 unique classes in labels.")
        else:
            n_samples = X.shape[0]
            n_vars = X.shape[1]

            def compute_best_rmse(X_temp, y, kf, max_ncomp, num_unique_y):
                if X_temp.shape[0] == 0 or X_temp.shape[1] == 0:
                    return np.inf, 1
                n_temp_samples = X_temp.shape[0]
                n_temp_vars = X_temp.shape[1]
                best_rmse_temp = np.inf
                best_nc_temp = 1
                max_nc_possible = min(max_ncomp, n_temp_samples, n_temp_vars)
                for nc in range(1, max_nc_possible + 1):
                    rmse_cv = []
                    for train_idx, test_idx in kf.split(X_temp):
                        X_train, X_test = X_temp[train_idx], X_temp[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        try:
                            pls = PLSRegression(n_components=nc)
                            pls.fit(X_train, y_train)
                            y_pred = pls.predict(X_test)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            rmse_cv.append(rmse)
                        except Exception:
                            continue
                    if rmse_cv:
                        avg_rmse = np.mean(rmse_cv)
                        if avg_rmse < best_rmse_temp:
                            best_rmse_temp = avg_rmse
                            best_nc_temp = nc
                return best_rmse_temp, best_nc_temp

            # Generate intervals
            intervals = []
            interval_size = max(1, n_vars // n_intervals) if n_intervals > 0 else n_vars
            for i in range(n_intervals):
                start = i * interval_size
                end = min((i + 1) * interval_size, n_vars)
                if start < end:
                    intervals.append((start, end))
           
            kf = KFold(n_splits=min(5, n_samples))
           
            # Forward iPLS selection
            selected_intervals = []
            current_selected_vars = []
            rmse_history = []
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
                   
                    best_rmse_temp, best_nc_temp = compute_best_rmse(X_temp, y, kf, max_ncomp, num_unique_y)
                    candidates_rmse.append(best_rmse_temp)
                    candidates_ncomp.append(best_nc_temp)
                    candidates_int_idx.append(i)
               
                if not candidates_rmse:
                    break
               
                best_cand = np.argmin(candidates_rmse)
                new_rmse = candidates_rmse[best_cand]
               
                if iteration > 1:
                    if current_selected_vars:
                        X_curr = X[:, current_selected_vars]
                        best_rmse_curr, _ = compute_best_rmse(X_curr, y, kf, max_ncomp, num_unique_y)
                        if new_rmse >= best_rmse_curr:
                            improved = False
                            continue
                    else:
                        improved = False
                        continue
               
                add_int_idx = candidates_int_idx[best_cand]
                add_int = intervals[add_int_idx]
                selected_intervals.append(add_int)
                add_vars = list(range(add_int[0], add_int[1]))
                current_selected_vars.extend(add_vars)
                rmse_history.append(new_rmse)
           
            # Compute single interval RMSE and ncomp
            single_rmse = []
            single_ncomp = []
            for i, (start, end) in enumerate(intervals):
                if start >= end:
                    single_rmse.append(np.inf)
                    single_ncomp.append(0)
                    continue
                X_int = X[:, start:end]
                best_rmse_int, best_nc_int = compute_best_rmse(X_int, y, kf, max_ncomp, num_unique_y)
                single_rmse.append(best_rmse_int)
                single_ncomp.append(best_nc_int)
           
            # Compute global RMSECV
            global_rmse = np.inf
            best_nc_global = 1
            try:
                best_rmse_global, best_nc_global = compute_best_rmse(X, y, kf, max_ncomp, num_unique_y)
                global_rmse = best_rmse_global
            except Exception:
                pass
           
            # Filter to selected variables
            if current_selected_vars:
                current_selected_vars = sorted(set(current_selected_vars))
                processed_df = processed_df.iloc[current_selected_vars].reset_index(drop=True)
           
            # iPLS artifacts
            ipls_artifacts = {
                'single_rmse': single_rmse,
                'single_ncomp': single_ncomp,
                'global_rmse': global_rmse,
                'best_nc_global': best_nc_global,
                'intervals': intervals,
                'full_x': full_spectral.values,
                'averages_full': averages,
                'selected_intervals': selected_intervals,
                'rmse_history': rmse_history
            }
   
    return (processed_df, labels, samples_info, full_spectral, averages, current_selected_vars, 
            x_label, y_label, loadings_x_label, ipls_artifacts)

@st.cache_data
def run_pca_analysis(X_num, y, num_save_pcs):
    """
    Cached PCA computation.
    Returns: pca_full, X_pca_full, var_ratios, pca_2d, X_pca_2d, pca_3d, X_pca_3d, loadings
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
  
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    n_total_pcs = X_pca_full.shape[1]
    var_ratios = pca_full.explained_variance_ratio_
  
    pca_2d = PCA(n_components=min(2, n_total_pcs))
    X_pca_2d = pca_2d.fit_transform(X_scaled)
  
    pca_3d = PCA(n_components=min(3, n_total_pcs))
    X_pca_3d = pca_3d.fit_transform(X_scaled)
  
    max_pcs = min(num_save_pcs, n_total_pcs)
    loadings = pd.DataFrame(pca_full.components_[:max_pcs],
                            columns=X_num.columns,
                            index=[f'PC{i+1}' for i in range(max_pcs)])
  
    return pca_full, X_pca_full, var_ratios, pca_2d, X_pca_2d, pca_3d, X_pca_3d, loadings

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

# File uploader with session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file
    
    # Check file size
    uploaded_file.seek(0)
    file_size = len(uploaded_file.read())
    uploaded_file.seek(0)
    if file_size == 0:
        st.error("The uploaded file is empty. Please upload a valid CSV file with data (at least headers and one row).")
        st.stop()
    
    # Quick read for display with error handling
    try:
        df_temp = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # Reset for later use
    except pd.errors.EmptyDataError:
        st.error("The uploaded file contains no data. Please ensure your CSV has headers and at least one row of data.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading the CSV file for preview: {str(e)}. Please check the file format and try again.")
        st.stop()
    
    if len(df_temp.columns) < 2:
        st.error("The CSV must have at least 2 columns: one for the spectral axis and at least one for sample data.")
        st.stop()
    
    spectral_col_temp = df_temp.columns[0]
    data_cols_temp = df_temp.columns[1:]
    
    # Quick label parsing for display
    samples_info_temp = {}
    for col in data_cols_temp:
        match = re.search(r'([mf])(\d{2})_', col)
        if match:
            samples_info_temp[col] = {
                'sex': match.group(1).upper(),
                'age': int(match.group(2))
            }
        else:
            samples_info_temp[col] = {'sex': 'Unknown', 'age': -1}
    
    info_df = pd.DataFrame(samples_info_temp).T
    st.subheader("Parsed Sample Information")
    st.dataframe(info_df)
   
    st.subheader("Labeling Mode")
    label_mode = st.radio("Label by:", ["Original Prefix", "Sex", "Age"], index=0)
   
    # Compute temp labels for distributions
    if label_mode == "Original Prefix":
        labels_temp = [col.split('_')[0] if '_' in col else col for col in data_cols_temp]
    elif label_mode == "Sex":
        labels_temp = ["Male" if samples_info_temp[col]['sex'] == "M" else "Female" if samples_info_temp[col]['sex'] == "F" else "Unknown" for col in data_cols_temp]
    elif label_mode == "Age":
        labels_temp = [str(samples_info_temp[col]['age']) if samples_info_temp[col]['age'] >= 0 else "Unknown" for col in data_cols_temp]
   
    if label_mode == "Sex":
        st.subheader("Label Distribution (Including Unknown)")
        counts = pd.Series(labels_temp).value_counts()
        fig = px.bar(x=counts.index, y=counts.values, title="Sex Distribution")
        fig.update_layout(xaxis_title="Sex", yaxis_title="Count")
        st.plotly_chart(fig)
        
        # Always filter Unknown for Sex
        valid_mask_temp = [l != "Unknown" for l in labels_temp]
        labels_temp_filtered = [labels_temp[i] for i, keep in enumerate(valid_mask_temp) if keep]
        
        st.subheader("Filtered Sex Distribution (Excluding Unknown)")
        counts_filtered = pd.Series(labels_temp_filtered).value_counts()
        fig_f = px.bar(x=counts_filtered.index, y=counts_filtered.values, title="Filtered Sex Distribution")
        fig_f.update_layout(xaxis_title="Sex", yaxis_title="Count")
        st.plotly_chart(fig_f)
        
        unique_sex_temp = set(labels_temp_filtered)
        if len(unique_sex_temp) < 2:
            st.warning(f"Only {len(unique_sex_temp)} unique sex class after filtering Unknown. iPLS cannot be performed for classification.")
        else:
            st.success(f"Found {len(unique_sex_temp)} sex classes: {', '.join(unique_sex_temp)}")
        
        st.subheader("Filtered Sample Information")
        filtered_info_temp = {data_cols_temp[i]: samples_info_temp[data_cols_temp[i]] for i in range(len(data_cols_temp)) if valid_mask_temp[i]}
        st.dataframe(pd.DataFrame(filtered_info_temp).T)
    elif label_mode == "Age":
        st.subheader("Label Distribution")
        ages_temp = [samples_info_temp[col]['age'] for col in data_cols_temp if samples_info_temp[col]['age'] >= 0]
        age_dist = pd.Series(0, index=range(101))
        for age in ages_temp:
            age_dist[age] += 1
        fig = px.bar(x=age_dist.index, y=age_dist.values, title="Age Distribution (0-100)")
        fig.update_layout(xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig)
        num_unknown = sum(1 for col in data_cols_temp if samples_info_temp[col]['age'] < 0)
        if num_unknown > 0:
            st.info(f"Unknown ages: {num_unknown}")
        
        # Filter out Unknown labels for further analysis
        valid_mask_temp = [l != "Unknown" for l in labels_temp]
        labels_temp_filtered = [labels_temp[i] for i, keep in enumerate(valid_mask_temp) if keep]
        filtered_info_temp = {data_cols_temp[i]: samples_info_temp[data_cols_temp[i]] for i in range(len(data_cols_temp)) if valid_mask_temp[i]}
        st.subheader("Filtered Sample Information (Excluding Unknown)")
        st.dataframe(pd.DataFrame(filtered_info_temp).T)
   
    # Spectrum filtering in sidebar
    with st.sidebar:
        with st.expander("Spectrum Filtering", expanded=False):
            spectrum_type = st.radio("Choose filtering:", ["No Filtering", "MSP Spectra", "FTIR Spectra"], index=0)
   
    # Axis label customization in sidebar
    with st.sidebar:
        with st.expander("Axis Label Customization", expanded=False):
            custom_x_label = st.text_input("X-Axis Label for Spectra and iPLS Plots (leave blank for 'Spectral Axis')", "")
            custom_y_label = st.text_input("Y-Axis Label for Spectra and iPLS Twin Axis (leave blank for 'Processed Intensity')", "")
            custom_loadings_x_label = st.text_input("X-Axis Label for Factor Loadings Plot (leave blank for 'Factors/Variables')", "")
   
    # Preprocessing options in sidebar
    with st.sidebar:
        with st.expander("Preprocessing Options", expanded=False):
            do_normalize = st.checkbox("Normalize (scale by max)", value=False)
            do_zscore = st.checkbox("Z-Score Standardization", value=False)
            do_smooth = st.checkbox("Smooth (Savitzky-Golay filter)", value=False)
            window_length = 15
            polyorder = 1
            deriv = 0
            if do_smooth:
                window_length = st.slider("SG Window Length (odd number recommended)", min_value=3, max_value=101, value=15, step=2)
                polyorder = st.slider("SG Polyorder", min_value=1, max_value=5, value=1)
                deriv = st.slider("SG Derivative Order", min_value=0, max_value=3, value=0)
                if polyorder >= window_length:
                    st.warning("Polyorder should be less than window length for best results.")
            do_snv = st.checkbox("SNV Standardization", value=False)
            do_ipls = st.checkbox("iPLS Feature Selection", value=False)
  
    # iPLS parameters if enabled
    n_intervals = 10
    max_ncomp = 10
    max_iter = 10
    if do_ipls:
        with st.sidebar:
            with st.expander("iPLS Parameters", expanded=False):
                n_vars_temp = len(df_temp.columns) - 1  # Approximate
                n_intervals = st.slider("Number of Intervals", min_value=5, max_value=min(100, n_vars_temp), value=max(5, n_vars_temp // 10), step=1)
                max_ncomp = st.slider("Maximum Number of Components", min_value=1, max_value=min(20, len(data_cols_temp)), value=min(10, len(data_cols_temp) // 10), step=1)
                max_iter = st.slider("Maximum Iterations", min_value=1, max_value=min(100, n_intervals), value=min(n_intervals, max(10, len(data_cols_temp) // 2)), step=1)
   
    # Clear cache button
    with st.sidebar:
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared! Rerun preprocessing.")
            st.rerun()
   
    # Call cached preprocessing
    try:
        (processed_df, labels, samples_info, full_spectral, averages, current_selected_vars,
         x_label, y_label, loadings_x_label, ipls_artifacts) = load_and_preprocess_data(
            uploaded_file, spectrum_type, do_normalize, do_zscore, do_smooth, window_length, polyorder, deriv, 
            do_snv, do_ipls, n_intervals, max_ncomp, max_iter, label_mode, custom_x_label, custom_y_label, custom_loadings_x_label
        )
        st.success(f"Loaded {len(labels)} samples. Preprocessing cached!")
        filter_msg = "No filtering applied."
        if spectrum_type == "MSP Spectra":
            filter_msg = "Wavelengths filtered to >= 300 nm."
        elif spectrum_type == "FTIR Spectra":
            filter_msg = "Wavenumbers filtered excluding 1800-2400 cm^{-1}."
        st.success(filter_msg)
    except ValueError as e:
        st.error(str(e))
        st.stop()
  
    # Compute labels and consistent colors
    unique_labels = sorted(set(labels))
    if label_mode == "Sex":
        label_to_rgb = {
            "Male": (1.0, 0.0, 0.0),  # Red
            "Female": (0.0, 0.0, 1.0)  # Blue
        }
        if "Unknown" in unique_labels:
            label_to_rgb["Unknown"] = (0.5, 0.5, 0.5)  # Gray
        color_discrete_map = {label: mcolors.to_hex(rgb) for label, rgb in label_to_rgb.items() if label in unique_labels}
    else:
        n_colors = len(unique_labels)
        wl_range = np.linspace(380, 750, n_colors)
        rgb_ints = [wavelength_to_rgb(wl) for wl in wl_range]
        colors_rgb = [(r/255.0, g/255.0, b/255.0) for r, g, b in rgb_ints]
        label_to_rgb = {label: colors_rgb[i] for i, label in enumerate(unique_labels)}
        color_discrete_map = {label: mcolors.to_hex(rgb) for label, rgb in label_to_rgb.items()}
    
    data_cols = processed_df.columns[1:]
  
    # Option for separate sample labels
    show_sep_samples = st.checkbox("Show separate sample labels for all spectra", key="sep_all_spec")
    # Plot all processed spectra
    st.subheader("All Processed Spectra")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sample_groups = {}
    for i, col in enumerate(data_cols):
        prefix = labels[i]
        if prefix not in sample_groups:
            sample_groups[prefix] = []
        sample_groups[prefix].append(col)
        color = label_to_rgb.get(prefix, (0.5, 0.5, 0.5))
        alpha_line = 0.5 if len(sample_groups[prefix]) > 1 else 1.0
        ax1.plot(processed_df[processed_df.columns[0]], processed_df[col], color=color, alpha=alpha_line)
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
            color = label_to_rgb.get(prefix, (0.5, 0.5, 0.5))
            ax_samp.text(0.05, y_pos[i], col, transform=ax_samp.transAxes, va='center', fontsize=max(8, 100 / len(data_cols)), color=color)
        st.pyplot(fig_samp)
   
    # Display sample grouping
    st.subheader("Sample Grouping for Averaging")
    st.write("Detected groups:", list(sample_groups.keys()))
    # Option for separate legend
    show_sep_legend_avg = st.checkbox("Show separate legend for averaged spectra", key="sep_avg")
  
    # Plot 2: Averaged spectra
    st.subheader("Averaged Processed Spectra")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    prefixes = list(averages.keys())
    num_groups = len(prefixes)
    for i, prefix in enumerate(prefixes):
        avg = averages[prefix]
        color = label_to_rgb.get(prefix, (0.5, 0.5, 0.5))
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
        legend_elements = [Line2D([0], [0], color=label_to_rgb.get(prefix, (0.5, 0.5, 0.5)), lw=2, label=f'{prefix} Average') for prefix in prefixes]
        ax_leg.legend(handles=legend_elements, loc='center')
        st.pyplot(fig_leg)
   
    # iPLS Plot if done
    if do_ipls and ipls_artifacts:
        st.info("iPLS results loaded from cache.")
        single_rmse = ipls_artifacts['single_rmse']
        single_ncomp = ipls_artifacts['single_ncomp']
        global_rmse = ipls_artifacts['global_rmse']
        best_nc_global = ipls_artifacts['best_nc_global']
        intervals = ipls_artifacts['intervals']
        full_x = ipls_artifacts['full_x']
        averages_full = ipls_artifacts['averages_full']
        selected_intervals = ipls_artifacts['selected_intervals']
        rmse_history = ipls_artifacts['rmse_history']
       
        # Option for separate legends
        show_sep_legend_ipls = st.checkbox("Show separate legends for iPLS plot", key="sep_ipls")
       
        # iPLS Plot
        st.subheader("iPLS Interval Selection Plot")
        fig_ipls, ax = plt.subplots(figsize=(12, 6))
       
        finite_rmses = [r for r in single_rmse if r != np.inf]
        max_rmse_single = max(finite_rmses) if finite_rmses else 1
        offset = 0.01 * max_rmse_single if max_rmse_single > 0 else 1
        selected_int_indices = [intervals.index(intv) for intv in selected_intervals]
       
        for i, (start, end) in enumerate(intervals):
            if single_rmse[i] == np.inf or start >= end:
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
       
        # Legend
        green_patch = Patch(facecolor='green', alpha=0.7, label='Selected Intervals')
        red_patch = Patch(facecolor='red', alpha=0.5, label='Non-selected Intervals')
       
        # Twin axis for spectra
        ax2 = ax.twinx()
        ax2.set_ylabel(y_label)
        prefixes_ipls = list(averages_full.keys())
        for i, prefix in enumerate(averages_full):
            color = label_to_rgb.get(prefix, (0.5, 0.5, 0.5))
            ax2.plot(full_x, averages_full[prefix], color=color, alpha=0.7, linewidth=1)
       
        if not show_sep_legend_ipls:
            ax.legend(handles=[green_patch, red_patch], loc='upper right')
       
        plt.title('iPLS Interval Selection (First Iteration)')
        plt.tight_layout()
        st.pyplot(fig_ipls)
        if show_sep_legend_ipls:
            st.subheader("iPLS Interval Legend")
            fig_leg_int, ax_leg_int = plt.subplots(figsize=(3, 2))
            ax_leg_int.axis('off')
            ax_leg_int.legend(handles=[green_patch, red_patch], loc='center')
            st.pyplot(fig_leg_int)
       
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
       
        st.success(f"iPLS selected {len(selected_intervals)} intervals ({len(current_selected_vars)} variables) from {len(intervals)} possible.")
  
    # Button to save the pre-processed data
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
        # Prepare data for PCA
        X = processed_df[data_cols].T
        X.columns = processed_df[processed_df.columns[0]].values
        X = X.reset_index(drop=True)
        y = pd.Series(labels)
      
        df_pca = X.copy()
        df_pca['label'] = y
        st.info(f"Simplified labels: Unique classes now {df_pca['label'].nunique()}")
      
        # Prepare numerical features
        X_num = df_pca.drop('label', axis=1).select_dtypes(include=[np.number])
        if X_num.empty:
            st.error("No numerical columns found for PCA.")
            st.stop()
      
        # PCA sidebar options
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
                num_save_pcs = st.slider("Number of PCs to Save", 1, min(10, X_num.shape[1]), 3)
      
        # Call cached PCA
        pca_full, X_pca_full, var_ratios, pca_2d, X_pca_2d, pca_3d, X_pca_3d, loadings = run_pca_analysis(X_num, y, num_save_pcs)
        n_total_pcs = X_pca_full.shape[1]
      
        # 1. 2D PCA Plot
        if show_2d and n_total_pcs >= 2:
            show_sep_legend_pca2d = st.checkbox("Show separate legend for 2D PCA", key="sep_pca2d")
            st.subheader("2D PCA Plot (PC1 vs PC2)")
            df_plot_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
            df_plot_2d['label'] = y
          
            fig, ax = plt.subplots(figsize=(8, 6))
            unique_labels_pca = df_plot_2d['label'].unique()
            color_map_pca = {label: label_to_rgb.get(label, (0.5, 0.5, 0.5)) for label in unique_labels_pca}
          
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
      
        # 2. 3D PCA Plot
        if show_3d and n_total_pcs >= 3:
            show_sep_legend_pca3d = st.checkbox("Show separate legend for 3D PCA", key="sep_pca3d")
            st.subheader("3D PCA Plot (Interactive: Rotate/Zoom with Mouse)")
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
                colors_leg = [color_discrete_map.get(label, '#808080') for label in unique_l]
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
            cum_var = np.cumsum(var_ratios)
            n_99 = np.argmax(cum_var >= 0.99) + 1 if np.any(cum_var >= 0.99) else n_total_pcs
            n_scree = min(n_99 + 2, n_total_pcs)
          
            var_ratio_scree = var_ratios[:n_scree] * 100
          
            fig_scree = make_subplots(specs=[[{"secondary_y": False}]])
          
            fig_scree.add_trace(
                go.Bar(x=[f'PC{i+1}' for i in range(n_scree)], y=var_ratio_scree,
                       name='% Variance', marker_color='lightblue'),
                secondary_y=False
            )
          
            for i, v in enumerate(var_ratio_scree):
                fig_scree.add_annotation(x=f'PC{i+1}', y=v, text=f'{v:.1f}%', showarrow=False,
                                         yshift=10, font=dict(size=10))
          
            fig_scree.update_layout(title=f"Scree Plot (Showing {n_scree} PCs: ≥99% + 2 more)",
                                    xaxis_title="Principal Components",
                                    yaxis_title="% Variance Explained")
            fig_scree.update_yaxes(range=[0, var_ratio_scree.max() * 1.1], secondary_y=False)
          
            st.plotly_chart(fig_scree, use_container_width=True)
          
            st.info(f"Total variance explained by shown PCs: {cum_var[n_scree-1]:.1f}% (≥99% reached at PC{n_99})")
      
        # 4. Factor Loadings Plot
        if show_loadings:
            show_sep_legend_load = st.checkbox("Show separate legend for loadings", key="sep_load")
            st.subheader("Factor Loadings Plot (Top 3 PCs)")
            max_pcs_load = min(3, n_total_pcs)
            var_ratios_top = var_ratios[:max_pcs_load]
          
            valid_indices = [i for i in range(max_pcs_load) if var_ratios_top[i] > 0]
            num_valid = len(valid_indices)
          
            if num_valid == 0:
                st.warning("No PCs with >0% variance.")
            else:
                st.info(f"Showing loadings for {num_valid} valid PCs (out of top 3)")
              
                loadings_plot = pd.DataFrame(pca_full.components_[valid_indices],
                                             columns=X_num.columns,
                                             index=[f'PC{i+1}' for i in valid_indices])
              
                if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                    fig_loadings = go.Figure()
                    colors_load_rgb = sns.color_palette("Set1", num_valid)
                    colors_load = [mcolors.to_hex(rgb) for rgb in colors_load_rgb]
                  
                    max_loadings = loadings_plot.abs().max(axis=0)
                    sorted_vars = max_loadings.sort_values(ascending=False).index
                  
                    width = 0.25
                    for i, pc in enumerate(loadings_plot.index):
                        pc_data = loadings_plot.loc[pc].loc[sorted_vars]
                        fig_loadings.add_trace(go.Bar(y=pc_data.values, x=sorted_vars,
                                                      name=pc, marker_color=colors_load[i], width=width,
                                                      base=0, offsetgroup=i))
                  
                    fig_loadings.update_layout(barmode='group',
                                               height=400, showlegend=not show_sep_legend_load,
                                               title="Loadings: Grouped Bar Graph",
                                               xaxis_title=loadings_x_label,
                                               yaxis_title="Loading Value")
                    fig_loadings.update_xaxes(tickangle=45, tickfont=dict(size=9))
                  
                else:
                    loadings_melt = loadings_plot.reset_index().melt(id_vars='index', var_name='Variable', value_name='Loading')
                    loadings_melt['PC'] = loadings_melt['index']
                  
                    original_vars = X_num.columns.tolist()
                    loadings_melt['Variable'] = pd.Categorical(loadings_melt['Variable'], categories=original_vars, ordered=True)
                    loadings_melt = loadings_melt.sort_values(['PC', 'Variable'])
                  
                    fig_loadings = px.line(loadings_melt, x='Variable', y='Loading', color='PC',
                                           markers=False,
                                           title="Loadings: Connected Line Plot",
                                           labels={'Variable': loadings_x_label, 'Loading': 'Loading Value'})
                    fig_loadings.update_traces(line=dict(width=2, dash='solid'))
                    fig_loadings.update_layout(showlegend=not show_sep_legend_load)
                    fig_loadings.update_xaxes(tickangle=45, tickfont=dict(size=9))
                  
                    if len(original_vars) > 50:
                        st.warning("Many variables (>50)—zoom/pan the plot for details in spectroscopy data.")
              
                st.plotly_chart(fig_loadings, use_container_width=True)
                if show_sep_legend_load:
                    st.subheader("Loadings Legend")
                    unique_pcs = loadings_plot.index.tolist()
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
                st.dataframe(loadings_plot)
      
        # Download PCA results
        st.subheader("Download PCA Results")
        col1, col2 = st.columns(2)
        with col1:
            df_scores = pd.DataFrame(X_pca_full[:, :num_save_pcs], columns=[f'PC{i+1}' for i in range(num_save_pcs)])
            df_scores['label'] = y
            csv_scores = df_scores.to_csv(index=False)
            st.download_button("Download PC Scores CSV", csv_scores, "pc_scores.csv", "text/csv")
        with col2:
            csv_loadings = loadings.to_csv(index=True)
            st.download_button("Download Loadings CSV", csv_loadings, "pca_loadings.csv", "text/csv")
      
        st.info(f"Downloads include top {num_save_pcs} PCs.")
  
else:
    st.info("Please upload a CSV file to proceed.")
