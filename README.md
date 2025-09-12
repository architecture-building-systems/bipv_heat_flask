# BIPV Heat Dashboard

A Flask web application for visualizing and analyzing Building Integrated Photovoltaic (BIPV) heat experiment data.

## Overview

This dashboard provides an interactive interface for exploring experimental data from BIPV heat studies. Users can select experiments, choose data series, and visualize results with interactive plots including experiment phase annotations.

## Features

- **Experiment Selection**: Browse available experiments with metadata
- **Data Visualization**: Interactive plots with Plotly
- **Multi-axis Plotting**: Automatic grouping by units with dual y-axes
- **Phase Annotations**: Experiment phases marked on timeline
- **Data Export**: Download experiment data files
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### For Users
1. Visit: https://asbipvheatprd.ethz.ch
2. Select an experiment from the dropdown
3. Choose data series to plot
4. Explore the interactive visualizations

### For Administrators
- **Deployment Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Maintenance Guide**: See [MAINTENANCE.md](MAINTENANCE.md)

## Application Structure

```
bipv_heat_flask/
├── main.py              # Main Flask application
├── wsgi.py              # WSGI entry point for production
├── requirements.txt     # Python dependencies
├── data/               # Experiment data files
│   ├── *.feather       # Time series data files
│   └── experiment_log.pkl  # Experiment metadata
├── static/             # CSS and static assets
│   └── css/style.css
├── templates/          # HTML templates
│   ├── layout.html     # Base template
│   └── home.html       # Main dashboard page
├── DEPLOYMENT.md       # Deployment instructions
├── MAINTENANCE.md      # Maintenance guide
└── README.md          # This file
```

## Data Format

### Experiment Files
- **Format**: Apache Feather (.feather)
- **Naming**: `idx-XXX_type_conditions.feather`
- **Content**: Time series data with multiple sensors/measurements

### Experiment Log
- **Format**: Python Pickle (.pkl)
- **Content**: Experiment metadata including:
  - Date and time information
  - Monitor names
  - IR/WW/CW percentages
  - Phase timing (start, warmup, measurement, cooldown)
  - Notes and observations

## API Endpoints

- `GET /` - Main dashboard page
- `GET /api/experiments` - List available experiments
- `GET /api/experiment/<code>` - Get experiment details and data series
- `GET /api/plot` - Generate plot data (with query parameters)
- `GET /download/<code>` - Download experiment data file

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Plotly.js
- **Data Processing**: Pandas, PyArrow
- **Deployment**: Gunicorn + Apache
- **Server**: ETH Zurich infrastructure

## Development

### Local Development
```bash
# Clone repository
git clone [repository-url]
cd bipv_heat_flask

# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py
```

### Adding New Data
1. Place `.feather` files in the `data/` directory
2. Update `experiment_log.pkl` with metadata
3. Files are automatically detected by the application

### Configuration
- **Port**: Configured in `main.py` and `wsgi.py`
- **Data Directory**: Defined in `main.py` as `DATA_DIR`
- **Static Files**: Served from `static/` directory

## Troubleshooting

### Common Issues
1. **No experiments showing**: Check `data/` directory for `.feather` files
2. **Plots not loading**: Verify experiment codes match file names
3. **Static files not loading**: Check Apache static file configuration

### Logs and Debugging
- **Application Logs**: Check gunicorn process output
- **Web Server Logs**: `/instances/home/asbipvheatprd/var/log/httpd/`
- **Debug Mode**: Set `debug=True` in `main.py` for development

## Support

- **Technical Issues**: See [MAINTENANCE.md](MAINTENANCE.md)
- **Deployment Issues**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Data Questions**: Contact the research team

## License

[Add appropriate license information]

## Contributors

- [Add contributor information]

## Version History

- **v1.0** (2025-09-12): Initial production deployment
  - Flask dashboard with experiment visualization
  - Gunicorn deployment on port 5001
  - Apache proxy configuration
  - Interactive Plotly charts with phase annotations
