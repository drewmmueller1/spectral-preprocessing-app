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
- Column-wise data: First column is wavenumbers (numeric).  
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
    do_normalize = st.checkbox("Normalize (scale by max)", value=False)
    do_smooth = st.checkbox("Smooth (Savitzky-Golay filter)", value=False)
    do_snv = st.checkbox("SNV Standardization", value=False)
    
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
        X = processed_df[data_cols].T.reset_index(drop=True)  # rows=samples, columns=wavenumbers
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
            var_ratio = pca_scree.explained_variance_ratio_ * 100  # % variance
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
                    
                else:  # Connected Scatterplot (Continuous, e.g., Spectroscopy)
                    # Prepare for line plot: Melt to long format, preserve original variable order
                    loadings_melt = loadings_abs.reset_index().melt(id_vars='index', var_name='Variable', value_name='Loading')
                    loadings_melt['PC'] = loadings_melt['index']  # Use PC name as color/group
                    
                    # Original order for continuous (e.g., wavelengths)
                    original_vars = X_num.columns.tolist()
                    loadings_melt['Variable'] = pd.Categorical(loadings_melt['Variable'], categories=original_vars, ordered=True)
                    loadings_melt = loadings_melt.sort_values(['PC', 'Variable'])
                    
                    # Line plot: X=Variable, Y=Loading, color=PC, connected lines per PC, no markers
                    fig_loadings = px.line(loadings_melt, x='Variable', y='Loading', color='PC',
                                           markers=False,
                                           title="Loadings: Connected Line Plot (Abs Values)",
                                           labels={'Variable': 'Factors/Variables', 'Loading': 'Loading Magnitude'})
                    fig_loadings.update_traces(line=dict(width=2, dash='solid'))  # Continuous solid lines
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
            df_scores['label'] = y  # Use simplified labels
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