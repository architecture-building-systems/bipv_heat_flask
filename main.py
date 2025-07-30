import os
import ast
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, send_file, request
import pandas as pd
import plotly
import json
from plotly.subplots import make_subplots
import plotly.graph_objs as go

app = Flask(__name__, static_folder="")


THIS_FOLDER = Path(__file__).parent.resolve()
DATA_DIR = os.path.join(THIS_FOLDER, 'data')

def list_feather_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.feather')]

def is_special_experiment(experiment_code):
    """Check if experiment code matches special pattern idx-9** where ** are digits 1-9"""
    pattern = r'^idx-9[1-9][1-9]$'
    return re.match(pattern, experiment_code) is not None

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



@app.route('/')
def index():
    return render_template('home.html')

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
        
        # Plot first unit on primary y-axis
        for col in unit_map[units[0]]:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col], 
                    mode='lines', 
                    name=str(col),
                    line=dict(width=2)
                ),
                secondary_y=False
            )
        
        # Plot second unit on secondary y-axis with dashed lines
        if use_secondary:
            for col in unit_map[units[1]]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df[col], 
                        mode='lines', 
                        name=str(col),
                        line=dict(dash='dash', width=2)
                    ),
                    secondary_y=True
                )
            
            fig.update_yaxes(title_text=f"Primary ({units[0]})", secondary_y=False)
            fig.update_yaxes(title_text=f"Secondary ({units[1]})", secondary_y=True)
        else:
            fig.update_yaxes(title_text=f"Value ({units[0]})", secondary_y=False)
        
        # Add experiment phase lines
        experiment_log = load_experiment_log()
        if experiment_log and experiment_code in experiment_log:
            exp_data = experiment_log[experiment_code]
            date = exp_data.get('Date', '')
            
            phases = [
                ('Start time', 'green', 'solid'),
                ('Start Warmup', 'orange', 'dash'),
                ('Start measurement', 'red', 'solid'),
                ('End measurement', 'red', 'dash'),
                ('End cool down', 'blue', 'solid')
            ]
            
            for phase_name, color, line_style in phases:
                if phase_name in exp_data:
                    phase_time = parse_experiment_time(date, exp_data[phase_name])
                    if phase_time:
                        fig.add_shape(
                            type="line",
                            x0=phase_time, x1=phase_time,
                            y0=0, y1=1,
                            yref="paper",
                            line=dict(
                                color=color,
                                width=2,
                                dash=line_style
                            )
                        )
                        fig.add_annotation(
                            x=phase_time,
                            y=1.02,
                            yref="paper",
                            text=phase_name,
                            showarrow=False,
                            textangle=-45,
                            font=dict(size=10, color=color)
                        )
        
        # Update layout
        fig.update_layout(
            title=f"Experiment: {experiment_code}",
            xaxis_title="Time",
            showlegend=True
        )
        
        return jsonify({
            'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
            'warning': warning
        })
        
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
    app.run(host='localhost', port = 5000, debug=True)

