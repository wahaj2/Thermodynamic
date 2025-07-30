import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy, wasserstein_distance, ttest_ind, mannwhitneyu, gaussian_kde
from scipy.signal import find_peaks
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
from concurrent.futures import ThreadPoolExecutor
import uuid
import os
import sys
from datetime import datetime
import umap
import warnings
from scipy import stats
import logging
import webbrowser

# Configure logging for console output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logging.info("Starting ThermodynamicDashboard application at 04:35 PM PKT, July 29, 2025")

# Initialize app with custom stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.css.append_css({"external_url": "assets/styles.css"})  # Serve CSS from assets folder

# Ensure 'assets' folder exists relative to the script
if not os.path.exists('assets'):
    os.makedirs('assets')

# Store datasets and labels in memory
datasets = {}
labels = {}

app.layout = dbc.Container([
    html.Div(className="modern-bg", children=[
        html.H1("Thermodynamic Equilibrium Distance Dashboard", className="text-center py-4", style={'color': '#2c3e50'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3("Data Upload", className="mb-0")),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files', style={'color': '#3498db'})
                            ]),
                            style={
                                'width': '100%',
                                'height': '80px',
                                'lineHeight': '80px',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'margin': '10px 0',
                                'cursor': 'pointer'
                            },
                            className="upload-box",
                            multiple=True
                        ),
                        html.Div(id='upload-status', className="mt-2 text-success"),
                        html.Div(id='upload-suggestion', className="suggestion-box", style={'display': 'block'}),
                        html.Hr(),
                        dbc.CardHeader(html.H4("Dataset Labeling", className="mb-0")),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='label-dropdown',
                                options=[
                                    {'label': 'Healthy', 'value': 'healthy'},
                                    {'label': 'Cancer', 'value': 'cancer'}
                                ],
                                value='healthy',
                                className="mb-3"
                            ),
                            html.Button('Select All', id='select-all', n_clicks=0, className="btn btn-secondary me-2"),
                            html.Button('Apply Label to Selected', id='apply-label', className="btn btn-primary w-100"),
                            dash_table.DataTable(
                                id='file-table',
                                columns=[
                                    {'name': 'Filename', 'id': 'filename'},
                                    {'name': 'Label', 'id': 'label'},
                                    {'name': 'Size', 'id': 'size'},
                                    {'name': 'Upload Time', 'id': 'upload_time'}
                                ],
                                style_table={'overflowX': 'auto'},
                                row_selectable='multi',
                                selected_rows=[],
                                page_action='native',
                                page_size=10,
                                style_cell={'textAlign': 'left', 'padding': '8px'},
                                style_header={'backgroundColor': '#ecf0f1', 'fontWeight': 'bold'}
                            )
                        ])
                    ])
                ], className="card p-3 mb-4")
            ], md=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3("Analysis Results", className="mb-0")),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="Statistics", tab_id="stats"),
                            dbc.Tab(label="Thermodynamics", tab_id="thermo"),
                            dbc.Tab(label="Distances", tab_id="distances"),
                            dbc.Tab(label="Visualization", tab_id="viz"),
                            dbc.Tab(label="Advanced Analysis", tab_id="advanced"),
                            dbc.Tab(label="Advanced Thermodynamics", tab_id="advanced-thermo")
                        ], id="tabs", active_tab="stats", className="mb-3"),
                        html.Div(id="tab-content", className="tab-content"),
                        dbc.Row([
                            dbc.Col([
                                dbc.RadioItems(
                                    id='filter-label',
                                    options=[
                                        {'label': 'All', 'value': None},
                                        {'label': 'Healthy', 'value': 'healthy'},
                                        {'label': 'Cancer', 'value': 'cancer'}
                                    ],
                                    value=None,
                                    inline=True,
                                    style={'margin-top': '10px'}
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Switch(
                                    id='switch-mean',
                                    label=html.Span('Show Overall Mean', className="switch-label"),
                                    value=False,
                                    className="mb-3 mt-3"
                                )
                            ], width=6)
                        ], id='filter-controls', style={'display': 'none'})
                    ])
                ], className="card p-3")
            ], md=8)
        ])
    ])
], fluid=True)

# Utility functions
def convert_value(value):
    try:
        if isinstance(value, str):
            value = value.strip()
            value = value.replace(',', '.').replace('E-', 'e-').replace('E+', 'e+')
            if ';' in value:
                value = value.split(';')[0]
        return float(value)
    except (ValueError, AttributeError):
        return np.nan

def validate_dataset(data_info):
    try:
        df = data_info.get('data')
        if not isinstance(df, pd.DataFrame):
            return False
        if 'Value' not in df.columns:
            return False
        if len(df['Value'].dropna()) < 2:
            return False
        return True
    except:
        return False

def parse_contents(contents, filename, upload_time):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), 
                        sep=None, 
                        engine='python',
                        header=None,
                        skiprows=1)
        
        non_empty_cols = []
        for col in df.columns:
            if not df[col].isnull().all():
                non_empty_cols.append(col)
            if len(non_empty_cols) >= 2:
                break
                
        if len(non_empty_cols) < 2:
            return None
            
        df = df[non_empty_cols[:2]].copy()
        df.columns = ['Time', 'Value']
        
        df['Time'] = df['Time'].astype(str).str.strip()
        df['Value'] = df['Value'].apply(convert_value)
        df = df.dropna(subset=['Value'])
        
        file_id = str(uuid.uuid4())
        datasets[file_id] = df  # Store DataFrame in memory
        
        return {
            'filename': filename,
            'id': file_id,
            'size': f"{len(decoded)/1024:.1f} KB",
            'upload_time': upload_time
        }
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        return None

def calculate_ts_features(df):
    """Calculate time-series features with robust error handling"""
    try:
        if isinstance(df, str):
            df = pd.read_csv(df)
        elif isinstance(df, dict):
            df = df.get('data', pd.DataFrame())
        elif not isinstance(df, pd.DataFrame):
            return None
            
        vals = df['Value'].dropna().values
        if len(vals) < 2:
            return None
            
        time = np.arange(len(vals))
        
        features = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals)
        }
        
        try:
            lr = stats.linregress(time, vals)
            features['trend'] = lr.slope
        except:
            features['trend'] = np.nan
            
        try:
            peaks, _ = find_peaks(vals, prominence=np.std(vals)/3)
            features['peaks'] = len(peaks)
        except:
            features['peaks'] = np.nan
            
        try:
            features['autocorr'] = pd.Series(vals).autocorr()
        except:
            features['autocorr'] = np.nan
            
        try:
            hist, _ = np.histogram(vals, bins=20, density=True)
            features['entropy'] = entropy(hist)
        except:
            features['entropy'] = np.nan
            
        return features
        
    except Exception as e:
        logging.error(f"Error calculating features: {e}")
        return None

def calculate_thermo_metrics(df):
    """Enhanced thermodynamics calculation"""
    try:
        if isinstance(df, str):
            df = pd.read_csv(df)
        elif isinstance(df, dict):
            df = df.get('data', pd.DataFrame())
        elif not isinstance(df, pd.DataFrame):
            return None
            
        vals = df['Value'].dropna().values
        if len(vals) < 2:
            return None
            
        hist, _ = np.histogram(vals, bins=20, density=True)
        probs = hist / (hist.sum() + 1e-10)
        
        S = -np.sum(probs * np.log(probs + 1e-10))
        E = np.mean(vals)
        F = E - 1.0 * S
        
        return {'entropy': S, 'energy': E, 'free_energy': F}
    except Exception as e:
        logging.error(f"Error in thermodynamic metrics: {e}")
        return None

def perform_stat_tests(healthy_dfs, cancer_dfs):
    """Statistical comparison between groups"""
    try:
        h_vals = []
        for df in healthy_dfs:
            if isinstance(df, pd.DataFrame) and 'Value' in df.columns:
                h_vals.extend(df['Value'].dropna().tolist())
        
        c_vals = []
        for df in cancer_dfs:
            if isinstance(df, pd.DataFrame) and 'Value' in df.columns:
                c_vals.extend(df['Value'].dropna().tolist())
        
        if len(h_vals) < 2 or len(c_vals) < 2:
            return None
            
        t_stat, t_p = ttest_ind(h_vals, c_vals, equal_var=False)
        u_stat, u_p = mannwhitneyu(h_vals, c_vals, alternative='two-sided')
        effect_size = (np.mean(c_vals) - np.mean(h_vals)) / np.std(np.concatenate([h_vals, c_vals]), ddof=1)
        
        return {
            't_test_p': t_p,
            'mannwhitney_p': u_p,
            'effect_size': effect_size
        }
    except Exception as e:
        logging.error(f"Error in statistical tests: {e}")
        return None

def compute_basic_stats(dfs):
    stats = []
    for df in dfs:
        try:
            if not isinstance(df, pd.DataFrame):
                continue
            if 'Value' not in df.columns:
                continue
            vals = df['Value'].dropna()
            if len(vals) == 0:
                continue
            stats.append({
                'mean': vals.mean(),
                'std': vals.std(),
                'skew': vals.skew(),
                'kurtosis': vals.kurtosis(),
                'min': vals.min(),
                'max': vals.max(),
                'count': len(vals)
            })
        except Exception as e:
            logging.error(f"Error computing stats: {e}")
            continue
    return pd.DataFrame(stats) if stats else pd.DataFrame()

def compute_entropy(df, bins=20):
    try:
        if not isinstance(df, pd.DataFrame):
            return np.nan
        if 'Value' not in df.columns:
            return np.nan
        vals = df['Value'].dropna()
        if len(vals) < 2:
            return np.nan
        hist, _ = np.histogram(vals, bins=bins, density=True)
        return entropy(hist)
    except:
        return np.nan

def safe_wasserstein(a, b):
    try:
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) == 0 or len(b) == 0:
            return np.nan
        a = (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-10)
        b = (b - np.min(b)) / (np.max(b) - np.min(b) + 1e-10)
        return wasserstein_distance(a, b)
    except:
        return np.nan

def compute_distance_matrix(loaded_data):
    file_ids = []
    all_values = []
    
    for file_id, data_info in loaded_data.items():
        try:
            df = data_info.get('data')
            if not isinstance(df, pd.DataFrame) or 'Value' not in df.columns:
                continue
            values = df['Value'].dropna().values
            if len(values) < 2:
                continue
            file_ids.append(file_id)
            all_values.append(values)
        except Exception as e:
            logging.error(f"Error processing dataset {file_id}: {e}")
            continue
    
    n = len(file_ids)
    if n < 2:
        return [], np.array([])
    
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                a = (all_values[i] - np.min(all_values[i])) / (np.ptp(all_values[i]) + 1e-10)
                b = (all_values[j] - np.min(all_values[j])) / (np.ptp(all_values[j]) + 1e-10)
                dist = wasserstein_distance(a, b)
                distance_matrix[i,j] = dist
                distance_matrix[j,i] = dist
            except Exception as e:
                logging.error(f"Distance calculation failed between {i} and {j}: {e}")
                distance_matrix[i,j] = np.nan
                distance_matrix[j,i] = np.nan
    
    valid_dists = distance_matrix[~np.isnan(distance_matrix)]
    mean_dist = np.mean(valid_dists) if len(valid_dists) > 0 else 0
    distance_matrix = np.where(np.isnan(distance_matrix), mean_dist, distance_matrix)
    
    return file_ids, distance_matrix

def render_stats_tab(healthy_dfs, cancer_dfs):
    healthy_stats = compute_basic_stats(healthy_dfs)
    cancer_stats = compute_basic_stats(cancer_dfs)
    
    if healthy_stats.empty or cancer_stats.empty:
        return html.Div("Not enough labeled data for comparison")
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=healthy_stats['mean'],
        name='Healthy Mean',
        boxpoints='all',
        jitter=0.5,
        marker_color='lightgreen'
    ))
    fig.add_trace(go.Box(
        y=cancer_stats['mean'],
        name='Cancer Mean',
        boxpoints='all',
        jitter=0.5,
        marker_color='lightcoral'
    ))
    fig.update_layout(title='Group Means Comparison')
    
    return html.Div([
        html.H4("Descriptive Statistics"),
        dbc.Row([
            dbc.Col([
                html.H5("Healthy Group"),
                dash_table.DataTable(
                    data=healthy_stats.describe().reset_index().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in healthy_stats.describe().reset_index().columns]
                )
            ], md=6),
            dbc.Col([
                html.H5("Cancer Group"),
                dash_table.DataTable(
                    data=cancer_stats.describe().reset_index().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in cancer_stats.describe().reset_index().columns]
                )
            ], md=6)
        ]),
        dcc.Graph(figure=fig, className="graph-container")
    ])

def render_thermo_tab(healthy_dfs, cancer_dfs):
    healthy_entropy = []
    cancer_entropy = []
    healthy_energy = []
    cancer_energy = []
    
    for df in healthy_dfs:
        try:
            if not isinstance(df, pd.DataFrame):
                continue
            vals = df['Value'].dropna()
            if len(vals) < 2:
                continue
            hist, _ = np.histogram(vals, bins=20, density=True)
            healthy_entropy.append(entropy(hist))
            healthy_energy.append(vals.mean())
        except:
            continue
    
    for df in cancer_dfs:
        try:
            if not isinstance(df, pd.DataFrame):
                continue
            vals = df['Value'].dropna()
            if len(vals) < 2:
                continue
            hist, _ = np.histogram(vals, bins=20, density=True)
            cancer_entropy.append(entropy(hist))
            cancer_energy.append(vals.mean())
        except:
            continue
    
    if not healthy_entropy or not cancer_entropy:
        return html.Div("Not enough valid data for thermodynamic analysis")
    
    avg_healthy_E = np.mean(healthy_energy) if healthy_energy else np.nan
    avg_cancer_E = np.mean(cancer_energy) if cancer_energy else np.nan
    delta_E = avg_cancer_E - avg_healthy_E if (avg_cancer_E and avg_healthy_E) else np.nan
    avg_healthy_S = np.mean(healthy_entropy)
    avg_cancer_S = np.mean(cancer_entropy)
    delta_S = avg_cancer_S - avg_healthy_S
    T = 1.0
    delta_F = delta_E - T * delta_S if not np.isnan(delta_E) else np.nan
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=healthy_entropy, name='Healthy', marker_color='lightgreen'))
    fig.add_trace(go.Box(y=cancer_entropy, name='Cancer', marker_color='lightcoral'))
    fig.update_layout(title='Entropy Distribution Comparison')
    
    return html.Div([
        html.H4("Thermodynamic Metrics"),
        dbc.Row([
            dbc.Col([
                html.H5("Entropy Distribution"),
                dcc.Graph(figure=fig, className="graph-container")
            ], md=6),
            dbc.Col([
                html.H5("Key Metrics"),
                dbc.Table([
                    html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                    html.Tbody([
                        html.Tr([html.Td("ΔEntropy (Cancer - Healthy)"), 
                                html.Td(f"{delta_S:.3f}" if not np.isnan(delta_S) else "N/A")]),
                        html.Tr([html.Td("ΔEnergy (Cancer - Healthy)"), 
                                html.Td(f"{delta_E:.3e}" if not np.isnan(delta_E) else "N/A")]),
                        html.Tr([html.Td("ΔFree Energy (ΔE - TΔS)"), 
                                html.Td(f"{delta_F:.3e}" if not np.isnan(delta_F) else "N/A")]),
                        html.Tr([html.Td("Healthy Avg Entropy"), 
                                html.Td(f"{avg_healthy_S:.3f}")]),
                        html.Tr([html.Td("Cancer Avg Entropy"), 
                                html.Td(f"{avg_cancer_S:.3f}")]),
                        html.Tr([html.Td("Healthy Avg Energy (mean)"), 
                                html.Td(f"{avg_healthy_E:.3e}" if not np.isnan(avg_healthy_E) else "N/A")]),
                        html.Tr([html.Td("Cancer Avg Energy (mean)"), 
                                html.Td(f"{avg_cancer_E:.3e}" if not np.isnan(avg_cancer_E) else "N/A")])
                    ])
                ], bordered=True)
            ], md=6)
        ])
    ])

def render_distances_tab(loaded_data):
    if not loaded_data:
        return html.Div("No datasets loaded yet")
    
    file_ids, distance_matrix = compute_distance_matrix(loaded_data)
    
    if len(file_ids) < 2:
        debug_info = [
            html.H5("Debug Information"),
            html.P(f"Total datasets: {len(loaded_data)}"),
            html.P("Files that failed validation:"),
            html.Ul([
                html.Li(f"{data.get('filename', 'Unnamed dataset')}")
                for fid, data in loaded_data.items()
                if not validate_dataset(data)
            ])
        ]
        return html.Div([
            html.Div("Need at least 2 valid datasets for distance analysis"),
            html.Div(debug_info)
        ])
    
    max_dist = np.max(distance_matrix)
    if max_dist == 0:
        max_dist = 1
    
    fig = go.Figure(data=go.Heatmap(
        z=distance_matrix,
        x=[loaded_data[fid]['filename'] for fid in file_ids],
        y=[loaded_data[fid]['filename'] for fid in file_ids],
        colorscale='Viridis',
        zmin=0,
        zmax=max_dist,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Pairwise Distance Matrix',
        xaxis_title="Dataset",
        yaxis_title="Dataset"
    )
    
    stats = []
    if len(file_ids) >= 2:
        healthy_indices = [i for i, fid in enumerate(file_ids) 
                         if loaded_data[fid].get('label') == 'healthy']
        cancer_indices = [i for i, fid in enumerate(file_ids) 
                        if loaded_data[fid].get('label') == 'cancer']
        
        if healthy_indices and cancer_indices:
            hh = distance_matrix[np.ix_(healthy_indices, healthy_indices)]
            cc = distance_matrix[np.ix_(cancer_indices, cancer_indices)]
            hc = distance_matrix[np.ix_(healthy_indices, cancer_indices)]
            
            stats = dbc.Table([
                html.Thead(html.Tr([html.Th("Comparison"), html.Th("Avg Distance")])),
                html.Tbody([
                    html.Tr([html.Td("Healthy-Healthy"), 
                            html.Td(f"{np.mean(hh):.6f}")]),
                    html.Tr([html.Td("Cancer-Cancer"), 
                            html.Td(f"{np.mean(cc):.6f}")]),
                    html.Tr([html.Td("Healthy-Cancer"), 
                            html.Td(f"{np.mean(hc):.6f}")])
                ])
            ], bordered=True)
    
    return html.Div([
        html.H4("Distance Analysis"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig, className="graph-container")),
            dbc.Col(stats)
        ])
    ])

def render_viz_tab(loaded_data):
    file_ids, distance_matrix = compute_distance_matrix(loaded_data)
    
    if len(file_ids) < 2:
        return html.Div("Need at least 2 valid datasets for visualization")
    
    if np.all(distance_matrix == 0):
        return html.Div(
            "Warning: All distances are zero. This usually means the datasets are identical "
            "or the distance calculation failed. Check your data formats."
        )
    
    n_samples = len(file_ids)
    perplexity = max(5, min(30, (n_samples - 1) // 2))
    
    try:
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            random_state=42,
            perplexity=perplexity,
            init='random'
        )
        embeddings = tsne.fit_transform(distance_matrix)
    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        return html.Div(f"Visualization failed: {str(e)}")
    
    plot_data = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'filename': [loaded_data[fid]['filename'] for fid in file_ids],
        'label': [loaded_data[fid].get('label', 'unlabeled') for fid in file_ids],
        'size': 10
    })
    
    fig = px.scatter(
        plot_data,
        x='x',
        y='y',
        color='label',
        hover_name='filename',
        title=f't-SNE Projection (perplexity={perplexity})'
    )
    
    return html.Div([
        dcc.Graph(figure=fig, className="graph-container"),
        html.P(f"Data points: {len(file_ids)}"),
        html.P(f"Mean distance: {np.mean(distance_matrix):.4f}")
    ])

def render_advanced_tab(healthy_dfs, cancer_dfs, loaded_data):
    features = []
    for file_id, data_info in loaded_data.items():
        try:
            df = data_info.get('data')
            if df is None or not isinstance(df, pd.DataFrame) or 'Value' not in df.columns:
                continue
                
            feats = calculate_ts_features(df)
            if feats:
                feats['label'] = data_info.get('label', 'unlabeled')
                feats['filename'] = data_info.get('filename', 'unknown')
                features.append(feats)
        except Exception as e:
            logging.error(f"Error processing {data_info.get('filename', 'unknown')}: {e}")
            continue
    
    if not features:
        return html.Div(
            "No valid features extracted - check your data format and ensure files contain a 'Value' column with numeric data",
            className="text-danger"
        )
    
    feature_df = pd.DataFrame(features)
    
    test_results = None
    if healthy_dfs and cancer_dfs:
        try:
            h_vals = []
            for df in healthy_dfs:
                if isinstance(df, pd.DataFrame) and 'Value' in df.columns:
                    valid_vals = df['Value'].dropna()
                    if len(valid_vals) > 0:
                        h_vals.extend(valid_vals.tolist())
            
            c_vals = []
            for df in cancer_dfs:
                if isinstance(df, pd.DataFrame) and 'Value' in df.columns:
                    valid_vals = df['Value'].dropna()
                    if len(valid_vals) > 0:
                        c_vals.extend(valid_vals.tolist())
            
            if len(h_vals) >= 2 and len(c_vals) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t_stat, t_p = ttest_ind(h_vals, c_vals)
                    u_stat, u_p = mannwhitneyu(h_vals, c_vals)
                    effect_size = (np.mean(c_vals) - np.mean(h_vals)) / np.std(np.concatenate([h_vals, c_vals]))
                
                    test_results = {
                        't_test_p': t_p,
                        'mannwhitney_p': u_p,
                        'effect_size': effect_size
                    }
        except Exception as e:
            logging.error(f"Error in statistical analysis: {e}")
            test_results = None

    if test_results is None:
        stats_output = html.Div("Need at least 2 healthy and 2 cancer samples for comparison", className="text-muted")
    else:
        stats_output = html.Div(
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Test"),
                    html.Th("p-value"),
                    html.Th("Effect Size")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("T-test"),
                        html.Td(f"{test_results['t_test_p']:.4e}"),
                        html.Td(f"{test_results['effect_size']:.3f}")
                    ]),
                    html.Tr([
                        html.Td("Mann-Whitney U"),
                        html.Td(f"{test_results['mannwhitney_p']:.4e}"),
                        html.Td("")
                    ])
                ])
            ], bordered=True, striped=True),
            id='stat-test-results'
        )
    
    thermo_data = []
    for file_id, data_info in loaded_data.items():
        try:
            df = data_info.get('data')
            if df is None or not isinstance(df, pd.DataFrame) or 'Value' not in df.columns:
                continue
                
            thermo_metrics = calculate_thermo_metrics(df)
            if thermo_metrics:
                thermo_metrics['label'] = data_info.get('label', 'unlabeled')
                thermo_metrics['filename'] = data_info.get('filename', 'unknown')
                thermo_data.append(thermo_metrics)
        except Exception as e:
            logging.error(f"Error calculating thermodynamics for {data_info.get('filename', 'unknown')}: {e}")
            continue
    
    thermo_content = []
    if thermo_data:
        thermo_df = pd.DataFrame(thermo_data)
        fig_entropy = px.box(
            thermo_df,
            y='entropy',
            color='label',
            title='Entropy Distribution',
            color_discrete_map={'healthy': 'green', 'cancer': 'red', 'unlabeled': 'gray'},
            points='all',
            hover_data=['filename']
        )
        fig_energy = px.box(
            thermo_df,
            y='energy',
            color='label',
            title='Energy Distribution',
            color_discrete_map={'healthy': 'green', 'cancer': 'red', 'unlabeled': 'gray'},
            points='all',
            hover_data=['filename']
        )
        thermo_content = [
            dbc.Col(dcc.Graph(id='entropy-plot', figure=fig_entropy, className="graph-container"), md=6),
            dbc.Col(dcc.Graph(id='energy-plot', figure=fig_energy, className="graph-container"), md=6)
        ]
    else:
        thermo_content = [
            dbc.Col(html.Div("No valid thermodynamic data available", className="text-muted"), md=12)
        ]
    
    fig_scatter = px.scatter(
        feature_df,
        x='mean',
        y='std',
        color='label',
        hover_name='filename',
        title='Feature Space Exploration (Mean vs Standard Deviation)',
        color_discrete_map={'healthy': 'green', 'cancer': 'red', 'unlabeled': 'gray'}
    )
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("Feature Exploration"),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='feature-x-select',
                            options=[{'label': col.capitalize(), 'value': col} 
                                    for col in ['mean', 'std', 'trend', 'peaks', 'entropy']],
                            value='mean',
                            clearable=False
                        )
                    ], md=6),
                    dbc.Col([
                        dcc.Dropdown(
                            id='feature-y-select',
                            options=[{'label': col.capitalize(), 'value': col} 
                                    for col in ['mean', 'std', 'trend', 'peaks', 'entropy']],
                            value='std',
                            clearable=False
                        )
                    ], md=6)
                ], className="mb-3"),
                dcc.Graph(id='feature-scatter', figure=fig_scatter, className="graph-container")
            ], md=6),
            dbc.Col([
                html.H4("Statistical Testing"),
                stats_output
            ], md=6)
        ]),
        html.Hr(),
        html.H4("Thermodynamic State Analysis"),
        dbc.Row(id='thermo-row', children=thermo_content)
    ])

def update_feature_plot(x_feat, y_feat, table_data, active_tab):
    if active_tab != 'advanced':
        raise dash.exceptions.PreventUpdate
    
    if not table_data:
        return go.Figure()
    
    features = []
    for file_info in table_data:
        try:
            df = datasets.get(file_info['id'])
            if df is None:
                continue
            feats = calculate_ts_features(df)
            if feats:
                feats['label'] = file_info.get('label', 'unlabeled')
                feats['filename'] = file_info['filename']
                features.append(feats)
        except:
            continue
    
    if not features:
        return go.Figure()
    
    feat_df = pd.DataFrame(features)
    return px.scatter(
        feat_df,
        x=x_feat,
        y=y_feat,
        color='label',
        hover_name='filename',
        title=f'{x_feat.capitalize()} vs {y_feat.capitalize()}',
        color_discrete_map={'healthy': 'green', 'cancer': 'red', 'unlabeled': 'gray'}
    )

def calculate_advanced_thermo_metrics(df):
    """Calculate advanced thermodynamic metrics with improved LLE calculation"""
    try:
        if not isinstance(df, pd.DataFrame) or 'Value' not in df.columns:
            return None
        vals = df['Value'].dropna().values
        if len(vals) < 2:
            return None

        # Entropy Production Rate (W/K) - Simplified approximation
        hist, _ = np.histogram(vals, bins=20, density=True)
        probs = hist / (hist.sum() + 1e-10)
        S = -np.sum(probs * np.log(probs + 1e-10))
        dS_dt = S / (len(vals) * 1e-3)  # Rough rate over time (assuming 1ms steps)

        # Thermodynamic Force (1/K) - Gradient-based approximation
        time = np.arange(len(vals))
        if len(np.unique(time)) > 1:
            grad = np.gradient(vals, time)
            F = np.mean(np.abs(grad)) / (np.std(vals) + 1e-10)
        else:
            F = np.nan

        # KL Divergence (nats) - Compare to uniform distribution
        uniform_probs = np.ones_like(probs) / len(probs)
        kl_div = np.sum(probs * np.log(probs / uniform_probs + 1e-10))

        # Free Energy Difference (J) - Relative to mean energy
        E = np.mean(vals)
        F_diff = E - S  # Simplified difference

        # Fluctuation-Dissipation Ratio - Ratio of variance to mean gradient
        var = np.var(vals)
        if F > 0:
            fdr = var / F
        else:
            fdr = np.nan

        # Largest Lyapunov Exponent (1/s) - Improved approximation using small perturbations
        if len(vals) > 10:  # Require sufficient data points
            dt = 1e-3  # Time step assumption
            N = len(vals)
            lle = 0.0
            for i in range(N - 1):
                delta_x = vals[i + 1] - vals[i]
                if abs(delta_x) > 1e-10:  # Avoid division by zero
                    lle += np.log(abs(delta_x) / dt) / (N - 1)
            lle = lle if lle != 0 else np.nan  # Return NaN if calculation fails
        else:
            lle = np.nan

        # Steady-State Flux - Average change rate
        flux = np.mean(np.diff(vals)) / (1e-3)

        return {
            'Entropy Production Rate (W/K)': dS_dt,
            'Thermodynamic Force (1/K)': F,
            'KL Divergence (nats)': kl_div,
            'Free Energy Difference (J)': F_diff,
            'Fluctuation-Dissipation Ratio': fdr,
            'Largest Lyapunov Exponent (1/s)': lle,
            'Steady-State Flux': flux
        }
    except Exception as e:
        logging.error(f"Error calculating advanced thermo metrics: {e}")
        return None

def render_advanced_thermo_tab(loaded_data, show_mean, filter_label):
    if not loaded_data:
        return html.Div("No valid datasets loaded")

    metrics_data = []
    for file_id, data_info in loaded_data.items():
        try:
            df = data_info.get('data')
            if df is None or not isinstance(df, pd.DataFrame) or 'Value' not in df.columns:
                continue
            metrics = calculate_advanced_thermo_metrics(df)
            if metrics:
                metrics['filename'] = data_info.get('filename', 'unknown')
                metrics['label'] = data_info.get('label', 'unlabeled')
                metrics_data.append(metrics)
        except Exception as e:
            logging.error(f"Error processing {data_info.get('filename', 'unknown')}: {e}")
            continue

    if not metrics_data:
        return html.Div("No valid metrics calculated")

    df_metrics = pd.DataFrame(metrics_data)
    if filter_label:
        df_metrics = df_metrics[df_metrics['label'] == filter_label]

    if show_mean:
        mean_metrics = df_metrics.drop(columns=['filename', 'label']).mean().to_dict()
        mean_metrics['filename'] = 'Overall Mean'
        mean_metrics['label'] = 'mean'
        table_data = [mean_metrics]
    else:
        table_data = df_metrics.to_dict('records')

    return html.Div([
        html.H4("Advanced Thermodynamic Metrics"),
        dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in table_data[0].keys()],
            data=table_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={'backgroundColor': '#ecf0f1', 'fontWeight': 'bold'},
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
            ]
        )
    ])

# Additional callback to show/hide filter controls
@app.callback(
    Output('filter-controls', 'style'),
    [Input('tabs', 'active_tab')]
)
def toggle_filter_controls(active_tab):
    if active_tab == 'advanced-thermo':
        return {'display': 'block'}
    return {'display': 'none'}

# Callback for upload suggestion
@app.callback(
    Output('upload-suggestion', 'children'),
    [Input('file-table', 'data')]
)
def update_upload_suggestion(table_data):
    if not table_data or len(table_data) == 0:
        return [
            html.I(className="fas fa-lightbulb suggestion-icon", style={'margin-right': '10px'}),
            html.Span(
                "Consider uploading all your healthy files first, then click 'Apply Label' to assign the 'Healthy' label in bulk. "
                "Next, upload all your cancer files and click 'Apply Label' again to assign the 'Cancer' label. "
                "This approach saves time by minimizing repetitive labeling steps."
            ),
            html.Button('×', id='close-suggestion', n_clicks=0, className="close-btn")
        ]
    return ""

# Callback to handle suggestion dismissal
@app.callback(
    Output('upload-suggestion', 'style'),
    [Input('close-suggestion', 'n_clicks')],
    [State('upload-suggestion', 'style')]
)
def dismiss_suggestion(n_clicks, current_style):
    if n_clicks > 0:
        return {'display': 'none'}
    return current_style or {'display': 'block'}

# Callbacks
@app.callback(
    [Output('file-table', 'data', allow_duplicate=True),
     Output('upload-status', 'children'),
     Output('file-table', 'selected_rows')],
    [Input('upload-data', 'contents'),
     Input('apply-label', 'n_clicks'),
     Input('select-all', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('label-dropdown', 'value'),
     State('file-table', 'selected_rows'),
     State('file-table', 'data'),
     State('upload-data', 'last_modified')],
    prevent_initial_call=True
)
def update_file_table_and_labels(contents_list, n_apply_clicks, n_select_clicks, filenames, label, selected_rows, existing_data, last_modified_list):
    ctx = callback_context
    if not ctx.triggered:
        return (existing_data or []), "", selected_rows or []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'upload-data' and contents_list:
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = existing_data.copy() if existing_data else []
        success_count = 0
        
        for contents, filename in zip(contents_list, filenames):
            file_info = parse_contents(contents, filename, upload_time)
            if file_info:
                new_data.append(file_info)
                success_count += 1
        
        status = f"Successfully processed {success_count}/{len(contents_list)} files"
        return new_data, status, selected_rows or []
    
    elif trigger_id == 'apply-label' and n_apply_clicks and selected_rows is not None:
        new_data = existing_data.copy() if existing_data else []
        
        for row_idx in selected_rows:
            if row_idx < len(new_data):
                file_id = new_data[row_idx]['id']
                current_label = labels.get(file_id, 'unlabeled')
                # Only apply new label if the current label is 'unlabeled'
                if current_label == 'unlabeled':
                    labels[file_id] = label
                    new_data[row_idx]['label'] = label
        
        return new_data, "Labels applied to unlabeled files", []
    
    elif trigger_id == 'select-all' and n_select_clicks:
        if existing_data:
            return existing_data, "", list(range(len(existing_data)))
        return existing_data or [], "", []
    
    return (existing_data or []), "", selected_rows or []

@app.callback(
    Output('file-table', 'data'),
    [Input('apply-label', 'n_clicks')],
    [State('label-dropdown', 'value'),
     State('file-table', 'selected_rows'),
     State('file-table', 'data')]
)
def update_labels(n_clicks, label, selected_rows, table_data):
    if n_clicks is None or not selected_rows:
        return table_data
    
    new_data = table_data.copy()
    
    for row_idx in selected_rows:
        if row_idx < len(new_data):
            file_id = new_data[row_idx]['id']
            current_label = labels.get(file_id, 'unlabeled')
            # Only apply new label if the current label is 'unlabeled'
            if current_label == 'unlabeled':
                labels[file_id] = label
                new_data[row_idx]['label'] = label
    
    return new_data

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('file-table', 'data'),
     Input('switch-mean', 'value'),
     Input('filter-label', 'value')]
)
def render_tab_content(active_tab, table_data, show_mean, filter_label):
    if not table_data:
        return html.Div("Please upload some datasets first")
    
    loaded_data = {}
    for file_info in table_data:
        file_id = file_info['id']
        label = labels.get(file_id, 'unlabeled')
        df = datasets.get(file_id)
        if df is not None:
            loaded_data[file_id] = {
                'data': df,
                'label': label,
                'filename': file_info['filename']
            }
    
    if not loaded_data:
        return html.Div("No valid datasets loaded")
    
    healthy_dfs = [v['data'] for v in loaded_data.values() if v['label'] == 'healthy']
    cancer_dfs = [v['data'] for v in loaded_data.values() if v['label'] == 'cancer']
    
    if active_tab == "stats":
        return render_stats_tab(healthy_dfs, cancer_dfs)
    elif active_tab == "thermo":
        return render_thermo_tab(healthy_dfs, cancer_dfs)
    elif active_tab == "distances":
        return render_distances_tab(loaded_data)
    elif active_tab == "viz":
        return render_viz_tab(loaded_data)
    elif active_tab == "advanced":
        return render_advanced_tab(healthy_dfs, cancer_dfs, loaded_data)
    elif active_tab == "advanced-thermo":
        return render_advanced_thermo_tab(loaded_data, show_mean, filter_label)
    else:
        return html.Div("Select a tab")

@app.callback(
    Output('feature-scatter', 'figure'),
    [Input('feature-x-select', 'value'),
     Input('feature-y-select', 'value'),
     Input('file-table', 'data'),
     Input('tabs', 'active_tab')],
)
def update_feature_plot(x_feat, y_feat, table_data, active_tab):
    if active_tab != 'advanced':
        raise dash.exceptions.PreventUpdate
    
    if not table_data:
        return go.Figure()
    
    features = []
    for file_info in table_data:
        try:
            df = datasets.get(file_info['id'])
            if df is None:
                continue
            feats = calculate_ts_features(df)
            if feats:
                feats['label'] = file_info.get('label', 'unlabeled')
                feats['filename'] = file_info['filename']
                features.append(feats)
        except:
            continue
    
    if not features:
        return go.Figure()
    
    feat_df = pd.DataFrame(features)
    return px.scatter(
        feat_df,
        x=x_feat,
        y=y_feat,
        color='label',
        hover_name='filename',
        title=f'{x_feat.capitalize()} vs {y_feat.capitalize()}',
        color_discrete_map={'healthy': 'green', 'cancer': 'red', 'unlabeled': 'gray'}
    )

if __name__ == '__main__':
    port = 8050
    webbrowser.open_new(f"http://localhost:{port}/")
    app.run_server(debug=True, port=port, host='0.0.0.0')