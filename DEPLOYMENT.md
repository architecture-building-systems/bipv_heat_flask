# BIPV Heat Flask Dashboard - Deployment Guide

## Overview
This document contains deployment and maintenance instructions for the BIPV Heat Flask dashboard running on the ETH Zurich server infrastructure.

## Server Configuration
- **Server**: extweb01.ethz.ch
- **User**: w3_asbipvheatprd
- **Home Directory**: `/instances/home/asbipvheatprd/`
- **Application Directory**: `/instances/home/asbipvheatprd/bipv_heat_flask/`
- **Python Environment**: `flask-env` (managed by pyenv)
- **Flask App Port**: 5001
- **Public URL**: https://asbipvheatprd.ethz.ch

## Deployment Architecture
```
Internet → Apache (HTTPS:443) → Proxy → Gunicorn (localhost:5001) → Flask App
```

## Initial Deployment Steps

### 1. Environment Setup
```bash
# SSH to server
ssh -o "IdentitiesOnly=yes" -i ".ssh/bipv_heat_key" w3_asbipvheatprd@asbipvheatprd.ethz.ch


# Navigate to home directory
cd /instances/home/asbipvheatprd/

# Activate Python environment
pyenv activate flask-env
```

### 2. Deploy Flask Application
```bash
# Navigate to Flask app directory
cd bipv_heat_flask/

# Start Flask app with Gunicorn
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
```

### 3. Start Apache Server
```bash
# Start Apache and associated services
start
```

### 4. Verify Deployment
```bash
# Test Flask app directly
curl http://localhost:5001

# Test through Apache proxy
curl -k https://asbipvheatprd.ethz.ch

# Check processes
ps aux | grep gunicorn
```

## File Structure
```
/instances/home/asbipvheatprd/
├── bipv_heat_flask/
│   ├── main.py              # Main Flask application
│   ├── wsgi.py              # WSGI entry point for production
│   ├── requirements.txt     # Python dependencies
│   ├── data/               # Experiment data files (.feather, .pkl)
│   ├── static/             # CSS and static assets
│   ├── templates/          # HTML templates
│   └── DEPLOYMENT.md       # This file
├── conf/
│   ├── httpd.conf          # Main Apache configuration
│   └── bipv_heat_flask.conf # Flask app proxy configuration
└── var/log/httpd/          # Apache log files
```

## Key Configuration Files

### Apache Configuration (`conf/bipv_heat_flask.conf`)
```apache
ProxyPass "/"  "http://localhost:5001/"
ProxyPassReverse "/"  "http://localhost:5001/"
Alias "/static/" "/instances/home/asbipvheatprd/bipv_heat_flask/static/"
```

### WSGI Entry Point (`wsgi.py`)
```python
from main import app
application = app  # Gunicorn looks for this
```

## Daily Operations

### Starting the Application
```bash
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
start  # Start Apache
```

### Stopping the Application
```bash
kill $(cat /tmp/gunicorn.pid)  # Stop Flask app
stop  # Stop Apache
```

### Restarting the Application
```bash
# Stop Flask app
kill $(cat /tmp/gunicorn.pid)

# Start Flask app
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application

# Restart Apache
stop && start
```

### Checking Status
```bash
# Check if Flask app is running
ps aux | grep gunicorn

# Check Flask app response
curl http://localhost:5001

# Check full deployment
curl -k https://asbipvheatprd.ethz.ch

# Check Apache processes
ps aux | grep httpd
```

## Troubleshooting

### Common Issues

#### 1. 503 Service Unavailable
**Cause**: Flask app not running on port 5001
**Solution**:
```bash
ps aux | grep gunicorn  # Check if running
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
```

#### 2. Port Already in Use
**Cause**: Another process using port 5001
**Solution**:
```bash
ss -tlnp | grep :5001  # Find what's using the port
kill $(cat /tmp/gunicorn.pid)  # Kill old gunicorn process
```

#### 3. Permission Errors
**Cause**: File permissions or ownership issues
**Solution**:
```bash
# Check file ownership
ls -la /instances/home/asbipvheatprd/bipv_heat_flask/

# Ensure files are owned by w3_asbipvheatprd user
```

#### 4. Python Environment Issues
**Cause**: Wrong Python environment or missing packages
**Solution**:
```bash
pyenv activate flask-env
which python  # Should show flask-env path
pip install -r requirements.txt
```

### Log Files
```bash
# Apache error logs
tail -f /instances/home/asbipvheatprd/var/log/httpd/error_log

# Apache access logs
tail -f /instances/home/asbipvheatprd/var/log/httpd/access_log

# Gunicorn logs (if configured)
tail -f /tmp/gunicorn.log
```

## Updating the Application

### Code Updates
```bash
# Stop the application
kill $(cat /tmp/gunicorn.pid)

# Update code files (main.py, templates, etc.)
# ... make your changes ...

# Restart the application
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
```

### Data Updates
```bash
# Add new .feather files to the data/ directory
cp new_experiment_data.feather /instances/home/asbipvheatprd/bipv_heat_flask/data/

# Update experiment log if needed
cp experiment_log.pkl /instances/home/asbipvheatprd/bipv_heat_flask/data/

# No restart needed - Flask will pick up new data files automatically
```

### Dependency Updates
```bash
# Activate environment
pyenv activate flask-env

# Update packages
pip install -r requirements.txt

# Restart application
kill $(cat /tmp/gunicorn.pid)
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
```

## Security Notes

1. **HTTPS**: All traffic is encrypted via Apache SSL/TLS
2. **Authentication**: Handled by ETH Zurich infrastructure
3. **File Permissions**: Ensure only w3_asbipvheatprd user can access application files
4. **Firewall**: Port 5001 should only be accessible from localhost

## Performance Monitoring

### Resource Usage
```bash
# Check memory usage
free -h

# Check CPU usage
top

# Check disk usage
df -h
```

### Application Performance
```bash
# Test response time
time curl -s http://localhost:5001 > /dev/null

# Check number of requests
tail -1000 /instances/home/asbipvheatprd/var/log/httpd/access_log | wc -l
```

## Contact Information
- **System Administrator**: ETH Zurich IT Department
- **Application Developer**: [Your contact information]
- **Documentation**: This file (DEPLOYMENT.md)

## Version History
- **v1.0** (2025-09-12): Initial deployment with gunicorn on port 5001
