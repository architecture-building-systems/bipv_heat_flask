import os
import ast
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, send_file, request, redirect, make_response
import pandas as pd
import plotly
import json
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np

app = Flask(__name__, static_folder="static")


THIS_FOLDER = Path(__file__).parent.resolve()
DATA_DIR = os.path.join(THIS_FOLDER, 'data')

# Global cache for series ranges
SERIES_RANGES_CACHE = None

def list_feather_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.feather')]

def is_special_experiment(experiment_code):
    """Check if experiment code matches special pattern idx-9** where ** are digits 1-9"""
    pattern = r'^idx-9[1-9][1-9]$'
    return re.match(pattern, experiment_code) is not None

def parse_experiment_characteristics(experiment_code):
    """Parse experiment code to extract G, T, A, Ti values"""
    # Pattern: idx-XXX_TYPE_G-[value]_T-[value]_A-[value]_Ti-[value]
    # Make A and Ti optional for experiments that don't have them
    pattern = r'idx-\d+_([^_]+)_G-(\d+)_T-(\d+)(?:_A-(\d+))?(?:_Ti-(\d+))?'
    match = re.match(pattern, experiment_code)
    
    if match:
        return {
            'Type': str(match.group(1)),  # Type (first capture group)
            'G': int(match.group(2)),     # Irradiance
            'T': int(match.group(3)),      # Temperature
            'A': int(match.group(4)) if match.group(4) else 0,      # Azimuth (default 0)
            'Ti': int(match.group(5)) if match.group(5) else 0      # Elevation (default 0)
        }
    return None

def get_experiment_log_file():
    """Get the single experiment log file"""
    pkl_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
    if len(pkl_files) == 0:
        return None
    elif len(pkl_files) == 1:
        return pkl_files[0]
    else:
        return pkl_files[0]

def load_experiment_log():
    """Load the experiment log file"""
    log_file = get_experiment_log_file()
    if not log_file:
        return None
    try:
        with open(os.path.join(DATA_DIR, log_file), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading experiment log {log_file}: {e}")
        return None

def parse_experiment_time(date_str, time_str):
    """Convert date and time strings to datetime object"""
    try:
        # Parse date like '210725' as DDMMYY to '25-07-21' -> '2021-07-25'
        if len(date_str) == 6:
            day = date_str[:2]
            month = date_str[2:4]
            year = '20' + date_str[4:6]
            date_formatted = f"{year}-{month}-{day}"
        else:
            date_formatted = date_str
        
        # Combine date and time
        datetime_str = f"{date_formatted} {time_str}"
        return pd.to_datetime(datetime_str)
    except:
        return None

def get_feather_filename_from_experiment(experiment_code):
    """Get feather filename from experiment code"""
    return f"{experiment_code}.feather"

def feather_file_exists(experiment_code):
    """Check if feather file exists for the experiment"""
    feather_filename = get_feather_filename_from_experiment(experiment_code)
    return os.path.exists(os.path.join(DATA_DIR, feather_filename))

def load_df(filename):
    df = pd.read_feather(os.path.join(DATA_DIR, filename))
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df = df[numeric_cols]
    df = df.interpolate().resample('1min').mean()
    return df

def get_unit(col):
    # For MultiIndex columns like ('Comfort Cube', 'metabolic_rate-mets')
    if isinstance(col, tuple):
        measurement_with_unit = col[1] if len(col) > 1 else col[0]
    else:
        measurement_with_unit = col
    
    if '-' in str(measurement_with_unit):
        unit = str(measurement_with_unit).split('-')[-1].strip()
        return unit
    return 'Unknown'

def load_series_ranges():
    """
    Load series ranges from pre-computed JSON file.
    This enables standardized y-axis ranges across all experiments.
    """
    global SERIES_RANGES_CACHE
    
    if SERIES_RANGES_CACHE is not None:
        return SERIES_RANGES_CACHE
    
    ranges_file = os.path.join(DATA_DIR, 'series_range.json')
    
    try:
        with open(ranges_file, 'r') as f:
            series_ranges = json.load(f)
        
        SERIES_RANGES_CACHE = series_ranges
        print(f"Loaded ranges for {len(series_ranges)} series from {ranges_file}")
        return series_ranges
        
    except FileNotFoundError:
        print(f"Warning: {ranges_file} not found. Y-axis ranges will not be standardized.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing {ranges_file}: {e}. Y-axis ranges will not be standardized.")
        return {}
    except Exception as e:
        print(f"Error loading {ranges_file}: {e}. Y-axis ranges will not be standardized.")
        return {}

def clear_series_ranges_cache():
    """Clear the cached series ranges (useful for development/testing)"""
    global SERIES_RANGES_CACHE
    SERIES_RANGES_CACHE = None
    print("Series ranges cache cleared")

def get_paired_colors(n_colors):
    """
    Generate colors from matplotlib's 'Paired' colormap (hardcoded values).
    Returns tuples of (primary_color, secondary_color) for consistent pairing.
    """
    # Hardcoded matplotlib Paired colormap colors (12 colors, 6 pairs)
    # Even indices (0,2,4,6,8,10) are for primary axis
    # Odd indices (1,3,5,7,9,11) are for secondary axis
    paired_colors = [
        '#a6cee3',  # 0 - light blue (primary)
        '#1f78b4',  # 1 - dark blue (secondary)
        '#b2df8a',  # 2 - light green (primary)
        '#33a02c',  # 3 - dark green (secondary)
        '#fb9a99',  # 4 - light red (primary)
        '#e31a1c',  # 5 - dark red (secondary)
        '#fdbf6f',  # 6 - light orange (primary)
        '#ff7f00',  # 7 - dark orange (secondary)
        '#cab2d6',  # 8 - light purple (primary)
        '#6a3d9a',  # 9 - dark purple (secondary)
        '#ffff99',  # 10 - light yellow (primary)
        '#b15928'   # 11 - dark brown (secondary)
    ]
    
    colors = []
    max_pairs = min(n_colors, 6)  # Paired colormap has 6 pairs maximum
    
    for i in range(max_pairs):
        primary_color = paired_colors[i * 2]      # Even index (0,2,4,6,8,10)
        secondary_color = paired_colors[i * 2 + 1] # Odd index (1,3,5,7,9,11)
        colors.append((primary_color, secondary_color))
    
    # If we need more colors than pairs available, cycle through
    while len(colors) < n_colors:
        colors.extend(colors[:min(6, n_colors - len(colors))])
    
    return colors[:n_colors]



@app.route('/')
def index():
    # Redirect to home page by default
    return redirect('/home')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/single')
def single():
    return render_template('single.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/sensors')
def sensors():
    return render_template('sensors.html')

@app.route('/api/experiments')
def get_experiments():
    experiment_log = load_experiment_log()
    available_experiments = []
    
    # Add experiments from the experiment log that have feather files
    if experiment_log:
        for exp_code in experiment_log.keys():
            if feather_file_exists(exp_code):
                exp_data = experiment_log[exp_code]
                label = f"{exp_code} ({exp_data.get('Date', 'Unknown date')})"
                available_experiments.append({'label': label, 'value': exp_code})
    
    # Add special experiments (idx-9**) that have feather files
    feather_files = list_feather_files()
    feather_files.sort()
    for feather_file in feather_files:
        exp_code = feather_file.replace('.feather', '')
        
        if is_special_experiment(exp_code.split("_")[0]):
            already_added = any(exp['value'] == exp_code for exp in available_experiments)
            if not already_added:
                label = f"{exp_code} (Special experiment)"
                available_experiments.append({'label': label, 'value': exp_code})
    
    if not available_experiments:
        return jsonify([{'label': 'No experiments with data files found', 'value': None}])
    
    available_experiments.sort(key=lambda x: x['value'] if x['value'] else '')
    return jsonify(available_experiments)

@app.route('/api/series-ranges')
def get_series_ranges():
    """API endpoint to get standardized series ranges"""
    try:
        ranges = load_series_ranges()
        return jsonify({
            'total_series': len(ranges),
            'sample_ranges': {k: v for k, v in list(ranges.items())[:5]},  # First 5 for preview
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/experiment-characteristics')
def get_experiment_characteristics():
    """Get all experiments with their parsed characteristics and filter ranges"""
    experiment_log = load_experiment_log()
    experiments_with_chars = []
    
    # Collect all characteristics
    all_G, all_T, all_A, all_Ti, all_Type = [], [], [], [], []
    
    # Process experiments from log that have feather files
    if experiment_log:
        for exp_code in experiment_log.keys():
            if feather_file_exists(exp_code):
                chars = parse_experiment_characteristics(exp_code)
                if chars:
                    exp_data = experiment_log[exp_code]
                    experiments_with_chars.append({
                        'code': exp_code,
                        'label': f"{exp_code} ({exp_data.get('Date', 'Unknown date')})",
                        'characteristics': chars
                    })
                    all_G.append(chars['G'])
                    all_T.append(chars['T'])
                    all_A.append(chars['A'])
                    all_Ti.append(chars['Ti'])
                    all_Type.append(chars['Type'])
    
    # Process feather files not in log
    feather_files = list_feather_files()
    for feather_file in feather_files:
        exp_code = feather_file.replace('.feather', '')
        
        # Skip if already processed from log
        if experiment_log and exp_code in experiment_log:
            continue
            
        chars = parse_experiment_characteristics(exp_code)
        if chars:
            experiments_with_chars.append({
                'code': exp_code,
                'label': f"{exp_code} (Data file)",
                'characteristics': chars
            })
            all_G.append(chars['G'])
            all_T.append(chars['T'])
            all_A.append(chars['A'])
            all_Ti.append(chars['Ti'])
            all_Type.append(chars['Type'])
    
    # Calculate ranges
    ranges = {
        'G': {'min': min(all_G) if all_G else 0, 'max': max(all_G) if all_G else 1000, 'values': sorted(set(all_G))},
        'T': {'min': min(all_T) if all_T else 0, 'max': max(all_T) if all_T else 30, 'values': sorted(set(all_T))},
        'A': {'min': min(all_A) if all_A else 0, 'max': max(all_A) if all_A else 45, 'values': sorted(set(all_A))},
        'Ti': {'min': min(all_Ti) if all_Ti else 0, 'max': max(all_Ti) if all_Ti else 72, 'values': sorted(set(all_Ti))},
        'Type': {'values': sorted(set(all_Type))}  # Type is categorical, no min/max needed
    }
    
    return jsonify({
        'experiments': experiments_with_chars,
        'ranges': ranges
    })

@app.route('/api/experiment/<experiment_code>')
def get_experiment_details(experiment_code):
    if not experiment_code:
        return jsonify({'error': 'No experiment selected'})
    
    feather_filename = get_feather_filename_from_experiment(experiment_code)
    
    if not feather_file_exists(experiment_code):
        return jsonify({
            'error': f'Data file {feather_filename} not found for experiment {experiment_code}'
        })
    
    try:
        df = load_df(feather_filename)
        series_options = [{'label': str(col), 'value': str(col)} for col in df.columns]
        
        # Get experiment details
        experiment_log = load_experiment_log()
        exp_details = {}
        
        if experiment_log and experiment_code in experiment_log:
            exp_data = experiment_log[experiment_code]
            exp_details = {
                'date': exp_data.get('Date', 'Unknown'),
                'monitor': exp_data.get('Monitor name', 'Unknown'),
                'ir': f"{exp_data.get('IR', 'N/A')}%",
                'ww': f"{exp_data.get('WW', 'N/A')}%",
                'cw': f"{exp_data.get('CW', 'N/A')}%",
                'notes': exp_data.get('Notes', 'None')
            }
        elif is_special_experiment(experiment_code):
            exp_details = {
                'type': 'Special Experiment',
                'date': 'Not available',
                'monitor': 'Not available',
                'ir': 'N/A',
                'ww': 'N/A',
                'cw': 'N/A',
                'notes': 'Special experiment - details not in experiment log'
            }
        else:
            exp_details = {
                'status': 'Experiment not found in log'
            }
        
        return jsonify({
            'series_options': series_options,
            'experiment_details': exp_details,
            'status': f'Loaded data from: {feather_filename}'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error loading data from {feather_filename}: {str(e)}'
        })

@app.route('/api/plot')
def get_plot_data():
    experiment_code = request.args.get('experiment')
    selected_series = request.args.getlist('series[]')
    
    if not (experiment_code and selected_series):
        return jsonify({'error': 'Missing experiment code or series'})
    
    feather_filename = get_feather_filename_from_experiment(experiment_code)
    
    if not feather_file_exists(experiment_code):
        return jsonify({'error': f'Data file not found for experiment {experiment_code}'})
    
    try:
        df = load_df(feather_filename)
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Convert stringified tuples back to tuples if needed
        cols = []
        for s in selected_series:
            try:
                col = ast.literal_eval(s)
            except Exception:
                col = s
            if col in df.columns:
                cols.append(col)
        
        if not cols:
            return jsonify({'error': 'No valid columns selected'})
        
        # Group columns by unit
        unit_map = {}
        for col in cols:
            unit = get_unit(col)
            if unit not in unit_map:
                unit_map[unit] = []
            unit_map[unit].append(col)
        
        units = list(unit_map.keys())
        warning = ''
        if len(units) > 2:
            warning = f'Warning: More than two units selected ({", ".join(units)}). Only the first two will be plotted.'
            units = units[:2]
            cols = []
            for unit in units:
                cols.extend(unit_map[unit])
        
        # Create figure with secondary y-axis if needed
        use_secondary = len(units) > 1
        fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])
        
        # Calculate total number of series for color generation
        total_series = len(cols)
        color_pairs = get_paired_colors(total_series)
        
        # Track color index across both axes
        color_index = 0
        data_line_width = 2
        # Plot first unit on primary y-axis
        for col in unit_map[units[0]]:
            primary_color, _ = color_pairs[color_index]
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col], 
                    mode='lines', 
                    name=str(col),
                    line=dict(width=data_line_width, color=primary_color),
                    showlegend=True
                ),
                secondary_y=False
            )
            color_index += 1
        
        # Plot second unit on secondary y-axis with dashed lines
        if use_secondary:
            # Reset color index to start from beginning for secondary axis
            color_index = 0
            for col in unit_map[units[1]]:
                _, secondary_color = color_pairs[color_index]
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df[col], 
                        mode='lines', 
                        name=str(col),
                        line=dict(dash='dash', width=data_line_width, color=secondary_color),
                        showlegend=True
                    ),
                    secondary_y=True
                )
                color_index += 1
            
            fig.update_yaxes(title_text=f"Primary ({units[0]})", zeroline=False, showline=False, secondary_y=False)
            fig.update_yaxes(title_text=f"Secondary ({units[1]})", zeroline=False, showline=False, secondary_y=True)
        else:
            fig.update_yaxes(title_text=f"Value ({units[0]})", zeroline=False, showline=False, secondary_y=False)
        
        # Apply standardized y-axis ranges
        series_ranges = load_series_ranges()
        
        # Calculate ranges for primary y-axis (first unit)
        primary_cols = unit_map[units[0]]
        primary_min = float('inf')
        primary_max = float('-inf')
        
        for col in primary_cols:
            col_str = str(col)
            if col_str in series_ranges:
                primary_min = min(primary_min, series_ranges[col_str]['padded_min'])
                primary_max = max(primary_max, series_ranges[col_str]['padded_max'])
        
        if primary_min != float('inf') and primary_max != float('-inf'):
            fig.update_yaxes(range=[primary_min, primary_max], zeroline=False, showline=False, secondary_y=False)
        
        # Calculate ranges for secondary y-axis (second unit) if it exists
        if use_secondary:
            secondary_cols = unit_map[units[1]]
            secondary_min = float('inf')
            secondary_max = float('-inf')
            
            for col in secondary_cols:
                col_str = str(col)
                if col_str in series_ranges:
                    secondary_min = min(secondary_min, series_ranges[col_str]['padded_min'])
                    secondary_max = max(secondary_max, series_ranges[col_str]['padded_max'])
            
            if secondary_min != float('inf') and secondary_max != float('-inf'):
                fig.update_yaxes(range=[secondary_min, secondary_max], zeroline=False, showline=False, secondary_y=True)
        
        # Add experiment phase lines
        experiment_log = load_experiment_log()
        if experiment_log and experiment_code in experiment_log:
            exp_data = experiment_log[experiment_code]
            date = exp_data.get('Date', '')
            
            phases = [
                ('Start time', 'Start','grey', 'solid'),
                ('Start Warmup', 'Warmup','grey', 'solid'),
                ('Start measurement', 'Measurement begin','grey', 'solid'),
                ('End measurement', 'Measurement end','grey', 'solid'),
                ('End cool down', 'End ','grey', 'solid'),
            ]
            
            for phase_name, phase_print_name, color, line_style in phases:
                if phase_name in exp_data:
                    phase_time = parse_experiment_time(date, exp_data[phase_name])
                    if phase_time:
                        fig.add_shape(
                            type="line",
                            x0=phase_time, x1=phase_time,
                            y0=0, y1=1,
                            yref="paper",
                            layer="below",
                            line=dict(
                                color="black",
                                width=1,
                                dash="solid"
                            )
                        )
                        fig.add_annotation(
                            x=phase_time,
                            y=1.05,
                            yref="paper",
                            text=phase_print_name,
                            showarrow=False,
                            # textangle=-10,
                            xanchor='left',
                            font=dict(size=10, color=color)
                        )
        
        # Update layout
        fig.update_layout(
            title=f"Experiment: {experiment_code}",
            xaxis_title=None,
            showlegend=False,  # Hide the default Plotly legend since we have a custom one
    
            # THEME & COLORS
            template="plotly_white",  # or "plotly_dark", "simple_white", etc.
            # plot_bgcolor="white",
            # paper_bgcolor="white",
            
            # DIMENSIONS
            
            # FONTS
            font=dict(
                family="Source Code Pro, monospace",
                size=10,
                color="#333"
            ),
            
            # TITLE STYLING
            title_font=dict(size=12, color="#2c3e50"),
            
            # GRID & AXES
            xaxis=dict(
                showgrid=False,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                showline=True,
                linewidth=2,
                linecolor="black"
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,  # Remove the horizontal line at y=0
                showline=False,  # Remove the y-axis spine line
            ),
            
            # MARGINS
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Create legend data for custom legend
        legend_data = []
        color_index = 0
        
        # Add primary axis series to legend
        for col in unit_map[units[0]]:
            primary_color, _ = color_pairs[color_index]
            legend_data.append({
                'name': str(col),
                'unit': units[0],
                'style': 'solid',
                'color': primary_color
            })
            color_index += 1
        
        # Add secondary axis series to legend if they exist
        if use_secondary:
            color_index = 0  # Reset for secondary axis
            for col in unit_map[units[1]]:
                _, secondary_color = color_pairs[color_index]
                legend_data.append({
                    'name': str(col),
                    'unit': units[1],
                    'style': 'dashed',
                    'color': secondary_color
                })
                color_index += 1
        
        return jsonify({
            'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
            'legend': legend_data,
            'warning': warning
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/experiment/<experiment_code>/temperatures')
def get_experiment_temperatures(experiment_code):
    """Calculate mean surface and air temperatures for an experiment"""
    if not experiment_code:
        return jsonify({'error': 'No experiment selected'})
    
    feather_filename = get_feather_filename_from_experiment(experiment_code)
    
    if not feather_file_exists(experiment_code):
        return jsonify({'error': f'Data file not found for experiment {experiment_code}'})
    
    try:
        df = load_df(feather_filename)
        
        # Define temperature series patterns
        surface_temp_patterns = [
            'surface_temperature-degrees_celsius'
        ]
        
        air_temp_patterns = [
            'Comfort Cube_air_temperature-degrees_celsius',
            'Comfort Cube_standard_effective_temperature-degrees_celsius',
            'Nano Cube_average_air_temperature-degrees_celsius',
            'cDAQ_interior_cube_air_temperature-degrees_celsius'
        ]
        
        # Find surface temperature columns
        surface_cols = []
        for col in df.columns:
            col_str = str(col)
            if any(pattern in col_str for pattern in surface_temp_patterns):
                surface_cols.append(col)
        
        # Find air temperature columns (prioritize Comfort Cube)
        air_cols = []
        for pattern in air_temp_patterns:
            for col in df.columns:
                col_str = str(col)
                if col_str == pattern:
                    air_cols.append(col)
                    break
        
        # Calculate means
        surface_temp_mean = None
        air_temp_mean = None
        
        if surface_cols:
            # Calculate mean of all surface temperature measurements
            surface_data = df[surface_cols].mean(axis=1)
            surface_temp_mean = round(surface_data.mean(), 1)
        
        if air_cols:
            # Use the first available air temperature (prioritized by air_temp_patterns order)
            air_data = df[air_cols[0]]
            air_temp_mean = round(air_data.mean(), 1)
        
        return jsonify({
            'surface_temp': surface_temp_mean,
            'air_temp': air_temp_mean,
            'surface_cols_found': len(surface_cols),
            'air_cols_found': len(air_cols),
            'surface_cols': [str(col) for col in surface_cols],
            'air_cols': [str(col) for col in air_cols]
        })
        
    except Exception as e:
        return jsonify({'error': f'Error calculating temperatures: {str(e)}'})

@app.route('/api/compare-plot')
def get_compare_plot_data():
    """Generate comparison plot for multiple experiments with max 2 series"""
    experiment_codes = request.args.getlist('experiments[]')
    selected_series = request.args.getlist('series[]')
    
    if not experiment_codes or not selected_series:
        return jsonify({'error': 'Missing experiment codes or series'})
    
    if len(selected_series) > 2:
        return jsonify({'error': 'Maximum 2 series allowed for comparison'})
    
    if len(experiment_codes) == 0:
        return jsonify({'error': 'No experiments selected'})
    
    try:
        # Load data for all experiments and align to measurement start
        experiment_data = {}
        measurement_start_times = {}
        experiment_log = load_experiment_log()
        
        # First pass: load data and get measurement start times
        for exp_code in experiment_codes:
            feather_filename = get_feather_filename_from_experiment(exp_code)
            if not feather_file_exists(exp_code):
                return jsonify({'error': f'Data file not found for experiment {exp_code}'})
            
            df = load_df(feather_filename)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Get measurement start time from experiment log
            measurement_start_time = None
            if experiment_log and exp_code in experiment_log:
                exp_data = experiment_log[exp_code]
                date = exp_data.get('Date', '')
                if 'Start measurement' in exp_data:
                    measurement_start_time = parse_experiment_time(date, exp_data['Start measurement'])
            
            experiment_data[exp_code] = df
            measurement_start_times[exp_code] = measurement_start_time
        
        # Second pass: align data to measurement start and find the longest experiment
        aligned_data = {}
        longest_experiment_length = 0
        earliest_start_offset = 0  # Track the earliest start relative to measurement
        
        for exp_code in experiment_codes:
            df = experiment_data[exp_code].copy()
            measurement_start = measurement_start_times[exp_code]
            
            if measurement_start is not None:
                # Ensure both datetime objects are timezone-naive for comparison
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                if hasattr(measurement_start, 'tzinfo') and measurement_start.tzinfo is not None:
                    measurement_start = measurement_start.replace(tzinfo=None)
                
                # Find the closest data point to measurement start time
                try:
                    time_diffs = abs(df.index - measurement_start)
                    min_diff_idx = time_diffs.argmin()
                    measurement_start_position = min_diff_idx
                    
                    # Calculate minutes from measurement start for each data point
                    # Negative values = before measurement start, positive = after
                    minutes_from_measurement = []
                    for i, timestamp in enumerate(df.index):
                        time_diff = timestamp - measurement_start
                        minutes_diff = time_diff.total_seconds() / 60.0
                        minutes_from_measurement.append(round(minutes_diff))
                    
                    df.index = pd.Index(minutes_from_measurement, name='minutes_from_measurement_start')
                    aligned_data[exp_code] = df
                    
                    # Track the longest experiment and earliest start
                    experiment_length = len(df)
                    if experiment_length > longest_experiment_length:
                        longest_experiment_length = experiment_length
                    
                    # Track the earliest start time (most negative)
                    min_time = min(minutes_from_measurement)
                    if min_time < earliest_start_offset:
                        earliest_start_offset = min_time
                        
                except Exception as e:
                    print(f"Error aligning experiment {exp_code}: {e}")
                    # Fallback: create simple minute-based index starting from 0
                    df.index = range(len(df))
                    aligned_data[exp_code] = df
                    if len(df) > longest_experiment_length:
                        longest_experiment_length = len(df)
            else:
                # Fallback: create simple minute-based index starting from 0
                df.index = range(len(df))
                aligned_data[exp_code] = df
                if len(df) > longest_experiment_length:
                    longest_experiment_length = len(df)
        
        # Convert stringified tuples back to tuples if needed
        cols = []
        for s in selected_series:
            try:
                col = ast.literal_eval(s)
            except Exception:
                col = s
            cols.append(col)
        
        # Group columns by unit
        unit_map = {}
        for col in cols:
            unit = get_unit(col)
            if unit not in unit_map:
                unit_map[unit] = []
            unit_map[unit].append(col)
        
        units = list(unit_map.keys())
        warning = ''
        if len(units) > 2:
            warning = f'Warning: More than two units selected ({", ".join(units)}). Only the first two will be plotted.'
            units = units[:2]
            cols = []
            for unit in units:
                cols.extend(unit_map[unit])
        
        # Create figure with secondary y-axis if needed
        use_secondary = len(units) > 1
        fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])
        
        # Generate colors based on experiment idx
        experiment_colors = {}
        for i, exp_code in enumerate(experiment_codes):
            # Extract idx number for consistent coloring
            idx_match = re.match(r'idx-(\d+)', exp_code)
            idx_number = int(idx_match.group(1)) if idx_match else i
            
            # Use a color palette that cycles through colors
            colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            experiment_colors[exp_code] = colors[idx_number % len(colors)]
        
        # Plot first unit on primary y-axis (solid lines)
        for col in unit_map[units[0]]:
            for exp_code in experiment_codes:
                df = aligned_data[exp_code]
                if col in df.columns:
                    # Extract idx for trace name
                    idx_match = re.match(r'idx-(\d+)', exp_code)
                    idx_number = idx_match.group(1).zfill(3) if idx_match else exp_code
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, 
                            y=df[col], 
                            mode='lines', 
                            name=f'{idx_number} - {str(col)}',
                            line=dict(width=2, color=experiment_colors[exp_code]),
                            showlegend=True
                        ),
                        secondary_y=False
                    )
        
        # Plot second unit on secondary y-axis (dashed lines)
        if use_secondary:
            for col in unit_map[units[1]]:
                for exp_code in experiment_codes:
                    df = aligned_data[exp_code]
                    if col in df.columns:
                        # Extract idx for trace name
                        idx_match = re.match(r'idx-(\d+)', exp_code)
                        idx_number = idx_match.group(1).zfill(3) if idx_match else exp_code
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index, 
                                y=df[col], 
                                mode='lines', 
                                name=f'{idx_number} - {str(col)}',
                                line=dict(dash='dash', width=2, color=experiment_colors[exp_code]),
                                showlegend=True
                            ),
                            secondary_y=True
                        )
            
            fig.update_yaxes(title_text=f"Primary ({units[0]})", zeroline=False, showline=False, secondary_y=False)
            fig.update_yaxes(title_text=f"Secondary ({units[1]})", zeroline=False, showline=False, secondary_y=True)
        else:
            fig.update_yaxes(title_text=f"Value ({units[0]})", zeroline=False, showline=False, secondary_y=False)
        
        # Apply standardized y-axis ranges
        series_ranges = load_series_ranges()
        
        # Calculate ranges for primary y-axis (first unit)
        primary_cols = unit_map[units[0]]
        primary_min = float('inf')
        primary_max = float('-inf')
        
        for col in primary_cols:
            col_str = str(col)
            if col_str in series_ranges:
                primary_min = min(primary_min, series_ranges[col_str]['padded_min'])
                primary_max = max(primary_max, series_ranges[col_str]['padded_max'])
        
        if primary_min != float('inf') and primary_max != float('-inf'):
            fig.update_yaxes(range=[primary_min, primary_max], zeroline=False, showline=False, secondary_y=False)
        
        # Calculate ranges for secondary y-axis (second unit) if it exists
        if use_secondary:
            secondary_cols = unit_map[units[1]]
            secondary_min = float('inf')
            secondary_max = float('-inf')
            
            for col in secondary_cols:
                col_str = str(col)
                if col_str in series_ranges:
                    secondary_min = min(secondary_min, series_ranges[col_str]['padded_min'])
                    secondary_max = max(secondary_max, series_ranges[col_str]['padded_max'])
            
            if secondary_min != float('inf') and secondary_max != float('-inf'):
                fig.update_yaxes(range=[secondary_min, secondary_max], zeroline=False, showline=False, secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title=f"Comparison: {', '.join([re.match(r'idx-(\d+)', exp_code).group(1).zfill(3) if re.match(r'idx-(\d+)', exp_code) else exp_code for exp_code in experiment_codes])}",
            xaxis_title="Minutes from Measurement Start",
            showlegend=False,  # Hide the default Plotly legend since we have a custom one
    
            # THEME & COLORS
            template="plotly_white",
            
            # DIMENSIONS - Let Plotly auto-size to container
            autosize=True,
            
            # FONTS
            font=dict(
                family="Source Code Pro, monospace",
                size=10,
                color="#333"
            ),
            
            # TITLE STYLING
            title_font=dict(size=12, color="#2c3e50"),
            
            # GRID & AXES
            xaxis=dict(
                showgrid=False,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                showline=True,
                linewidth=2,
                linecolor="black"
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
            ),
            
            # MARGINS
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Create legend data for custom legend
        legend_data = []
        
        # Add primary axis series to legend
        for col in unit_map[units[0]]:
            for exp_code in experiment_codes:
                df = aligned_data[exp_code]
                if col in df.columns:
                    idx_match = re.match(r'idx-(\d+)', exp_code)
                    idx_number = idx_match.group(1).zfill(3) if idx_match else exp_code
                    legend_data.append({
                        'name': f'{idx_number} - {str(col)}',
                        'unit': units[0],
                        'style': 'solid',
                        'color': experiment_colors[exp_code]
                    })
        
        # Add secondary axis series to legend if they exist
        if use_secondary:
            for col in unit_map[units[1]]:
                for exp_code in experiment_codes:
                    df = aligned_data[exp_code]
                    if col in df.columns:
                        idx_match = re.match(r'idx-(\d+)', exp_code)
                        idx_number = idx_match.group(1).zfill(3) if idx_match else exp_code
                        legend_data.append({
                            'name': f'{idx_number} - {str(col)}',
                            'unit': units[1],
                            'style': 'dashed',
                            'color': experiment_colors[exp_code]
                        })
        
        return jsonify({
            'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
            'legend': legend_data,
            'warning': warning
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/plot-data')
def download_single_plot_data():
    """Download plot data for single experiment as CSV"""
    experiment_code = request.args.get('experiment')
    selected_series = request.args.getlist('series[]')
    format_type = request.args.get('format', 'csv')
    
    if not (experiment_code and selected_series):
        return jsonify({'error': 'Missing experiment code or series'})
    
    feather_filename = get_feather_filename_from_experiment(experiment_code)
    
    if not feather_file_exists(experiment_code):
        return jsonify({'error': f'Data file not found for experiment {experiment_code}'})
    
    try:
        df = load_df(feather_filename)
        
        # Convert stringified tuples back to tuples if needed
        cols = []
        for s in selected_series:
            try:
                col = ast.literal_eval(s)
            except Exception:
                col = s
            if col in df.columns:
                cols.append(col)
        
        if not cols:
            return jsonify({'error': 'No valid columns selected'})
        
        # Create output dataframe with selected series
        output_df = df[cols].copy()
        
        if format_type.lower() == 'csv':
            # Create CSV response
            from io import StringIO
            output = StringIO()
            output_df.to_csv(output)
            output.seek(0)
            
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename={experiment_code}_plot_data.csv'
            return response
        else:
            return jsonify({'error': 'Unsupported format'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/compare-plot-data')
def download_compare_plot_data():
    """Download comparison plot data as CSV"""
    experiment_codes = request.args.getlist('experiments[]')
    selected_series = request.args.getlist('series[]')
    format_type = request.args.get('format', 'csv')
    
    if not experiment_codes or not selected_series:
        return jsonify({'error': 'Missing experiment codes or series'})
    
    try:
        # Load and align data (reuse logic from compare plot)
        experiment_data = {}
        measurement_start_times = {}
        experiment_log = load_experiment_log()
        
        # Load data and get measurement start times
        for exp_code in experiment_codes:
            feather_filename = get_feather_filename_from_experiment(exp_code)
            if not feather_file_exists(exp_code):
                return jsonify({'error': f'Data file not found for experiment {exp_code}'})
            
            df = load_df(feather_filename)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            measurement_start_time = None
            if experiment_log and exp_code in experiment_log:
                exp_data = experiment_log[exp_code]
                date = exp_data.get('Date', '')
                if 'Start measurement' in exp_data:
                    measurement_start_time = parse_experiment_time(date, exp_data['Start measurement'])
            
            experiment_data[exp_code] = df
            measurement_start_times[exp_code] = measurement_start_time
        
        # Align data to measurement start
        aligned_data = {}
        for exp_code in experiment_codes:
            df = experiment_data[exp_code].copy()
            measurement_start = measurement_start_times[exp_code]
            
            if measurement_start is not None:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                if hasattr(measurement_start, 'tzinfo') and measurement_start.tzinfo is not None:
                    measurement_start = measurement_start.replace(tzinfo=None)
                
                try:
                    minutes_from_measurement = []
                    for i, timestamp in enumerate(df.index):
                        time_diff = timestamp - measurement_start
                        minutes_diff = time_diff.total_seconds() / 60.0
                        minutes_from_measurement.append(round(minutes_diff))
                    
                    df.index = pd.Index(minutes_from_measurement, name='minutes_from_measurement_start')
                    aligned_data[exp_code] = df
                except Exception:
                    df.index = range(len(df))
                    aligned_data[exp_code] = df
            else:
                df.index = range(len(df))
                aligned_data[exp_code] = df
        
        # Convert stringified tuples back to tuples if needed
        cols = []
        for s in selected_series:
            try:
                col = ast.literal_eval(s)
            except Exception:
                col = s
            cols.append(col)
        
        # Create combined dataframe
        combined_data = {}
        combined_data['minutes_from_measurement_start'] = []
        
        for exp_code in experiment_codes:
            df = aligned_data[exp_code]
            idx_match = re.match(r'idx-(\d+)', exp_code)
            idx_number = idx_match.group(1).zfill(3) if idx_match else exp_code
            
            for col in cols:
                if col in df.columns:
                    column_name = f"{idx_number}_{str(col)}"
                    combined_data[column_name] = []
        
        # Get all unique time points
        all_time_points = set()
        for exp_code in experiment_codes:
            df = aligned_data[exp_code]
            all_time_points.update(df.index.tolist())
        
        all_time_points = sorted(all_time_points)
        
        # Fill data for each time point
        for time_point in all_time_points:
            combined_data['minutes_from_measurement_start'].append(time_point)
            
            for exp_code in experiment_codes:
                df = aligned_data[exp_code]
                idx_match = re.match(r'idx-(\d+)', exp_code)
                idx_number = idx_match.group(1).zfill(3) if idx_match else exp_code
                
                for col in cols:
                    if col in df.columns:
                        column_name = f"{idx_number}_{str(col)}"
                        if time_point in df.index:
                            combined_data[column_name].append(df.loc[time_point, col])
                        else:
                            combined_data[column_name].append('')  # Empty for missing data points
        
        # Create DataFrame and export
        output_df = pd.DataFrame(combined_data)
        
        if format_type.lower() == 'csv':
            from io import StringIO
            output = StringIO()
            output_df.to_csv(output, index=False)
            output.seek(0)
            
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename=comparison_plot_data.csv'
            return response
        else:
            return jsonify({'error': 'Unsupported format'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download/<experiment_code>')
def download_data(experiment_code):
    if not experiment_code:
        return jsonify({'error': 'No experiment selected'})
    
    feather_filename = get_feather_filename_from_experiment(experiment_code)
    feather_path = os.path.join(DATA_DIR, feather_filename)
    
    if not os.path.exists(feather_path):
        return jsonify({'error': 'File not found'})
    
    return send_file(feather_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host='localhost', port = 5001, debug=True)

