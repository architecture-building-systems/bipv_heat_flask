# BIPV Heat Flask Dashboard - Maintenance Guide

## Quick Reference Commands

### Start/Stop/Restart
```bash
# START APPLICATION
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
start

# STOP APPLICATION  
kill $(cat /tmp/gunicorn.pid)
stop

# RESTART APPLICATION
kill $(cat /tmp/gunicorn.pid)
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
stop && start
```

### Status Checks
```bash
# Check Flask app is running
ps aux | grep gunicorn

# Test Flask app directly
curl http://localhost:5001

# Test full deployment
curl -k https://asbipvheatprd.ethz.ch

# Check what's using port 5001
ss -tlnp | grep :5001
```

## Routine Maintenance Tasks

### Weekly Checks
- [ ] Verify application is responding: `curl -k https://asbipvheatprd.ethz.ch`
- [ ] Check disk space: `df -h`
- [ ] Review error logs: `tail -50 /instances/home/asbipvheatprd/var/log/httpd/error_log`
- [ ] Verify data files are up to date in `/instances/home/asbipvheatprd/bipv_heat_flask/data/`

### Monthly Tasks
- [ ] Review Apache access logs for unusual activity
- [ ] Check for Python package updates: `pip list --outdated`
- [ ] Backup experiment data files
- [ ] Test disaster recovery procedures

## Common Maintenance Scenarios

### Scenario 1: Website Not Loading (503 Error)
**Symptoms**: Browser shows "503 Service Unavailable"

**Diagnosis**:
```bash
curl http://localhost:5001
ps aux | grep gunicorn
```

**Solution**:
```bash
# If no gunicorn process found:
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
```

### Scenario 2: Application Crashed
**Symptoms**: Gunicorn process exists but application not responding

**Diagnosis**:
```bash
ps aux | grep gunicorn
curl -v http://localhost:5001
tail -20 /instances/home/asbipvheatprd/var/log/httpd/error_log
```

**Solution**:
```bash
# Force restart
kill $(cat /tmp/gunicorn.pid)
pkill -f gunicorn  # Kill any remaining processes
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
```

### Scenario 3: New Experiment Data
**Task**: Add new experiment data files

**Steps**:
```bash
# Copy new .feather files
cp /path/to/new/data/*.feather /instances/home/asbipvheatprd/bipv_heat_flask/data/

# Update experiment log if needed
cp /path/to/experiment_log.pkl /instances/home/asbipvheatprd/bipv_heat_flask/data/

# No restart needed - Flask will automatically detect new files
```

### Scenario 4: Code Updates
**Task**: Deploy updated Flask application code

**Steps**:
```bash
# 1. Stop application
kill $(cat /tmp/gunicorn.pid)

# 2. Update code files
git pull

# 3. Restart application
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application

# 4. Test deployment
curl -k https://asbipvheatprd.ethz.ch
```

### Scenario 5: Server Reboot
**Task**: Restore service after server restart

**Steps**:
```bash
# 1. SSH to server
ssh w3_asbipvheatprd@extweb01.ethz.ch

# 2. Activate Python environment
pyenv activate flask-env

# 3. Start Flask application
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application

# 4. Start Apache
start

# 5. Verify everything works
curl -k https://asbipvheatprd.ethz.ch
```

## Monitoring and Alerts

### Key Metrics to Monitor
- **Application Response Time**: Should be < 2 seconds
- **Memory Usage**: Monitor with `free -h`
- **Disk Space**: Data directory shouldn't exceed 80% capacity
- **Process Count**: Should have 1 gunicorn master + workers

### Log Monitoring
```bash
# Watch error logs in real-time
tail -f /instances/home/asbipvheatprd/var/log/httpd/error_log

# Check for recent errors
grep -i error /instances/home/asbipvheatprd/var/log/httpd/error_log | tail -10

# Monitor access patterns
tail -f /instances/home/asbipvheatprd/var/log/httpd/access_log
```

### Performance Checks
```bash
# Test response time
time curl -s http://localhost:5001 > /dev/null

# Check memory usage of Flask app
ps aux | grep gunicorn | awk '{print $6}' | tail -n +2 | paste -sd+ | bc

# Check number of active connections
netstat -an | grep :5001 | grep ESTABLISHED | wc -l
```

## Backup and Recovery

### Data Backup
```bash
# Backup experiment data (run monthly)
tar -czf /tmp/bipv_data_backup_$(date +%Y%m%d).tar.gz \
  /instances/home/asbipvheatprd/bipv_heat_flask/data/

# Backup application code
tar -czf /tmp/bipv_app_backup_$(date +%Y%m%d).tar.gz \
  /instances/home/asbipvheatprd/bipv_heat_flask/ \
  --exclude=/instances/home/asbipvheatprd/bipv_heat_flask/data/
```

### Recovery Procedures
```bash
# Restore from backup
tar -xzf /tmp/bipv_app_backup_YYYYMMDD.tar.gz -C /instances/home/asbipvheatprd/

# Restore data files
tar -xzf /tmp/bipv_data_backup_YYYYMMDD.tar.gz -C /

# Restart application
cd /instances/home/asbipvheatprd/bipv_heat_flask/
gunicorn --bind 0.0.0.0:5001 --daemon --pid /tmp/gunicorn.pid wsgi:application
```

## Security Maintenance

### Regular Security Tasks
- [ ] Review Apache access logs for suspicious activity
- [ ] Ensure only authorized users have server access
- [ ] Keep Python packages updated (monthly)
- [ ] Verify SSL certificate is valid and not expiring soon

### Security Commands
```bash
# Check SSL certificate expiration
openssl s_client -connect asbipvheatprd.ethz.ch:443 -servername asbipvheatprd.ethz.ch 2>/dev/null | openssl x509 -noout -dates

# Review recent access attempts
tail -100 /instances/home/asbipvheatprd/var/log/httpd/access_log | grep -E "(POST|PUT|DELETE)"

# Check for failed authentication attempts
grep -i "auth" /instances/home/asbipvheatprd/var/log/httpd/error_log | tail -10
```

## Emergency Contacts

### When to Contact IT Support
- Server hardware issues
- Network connectivity problems
- SSL certificate issues
- User authentication problems
- Disk space critically low (>95% full)

### When to Contact Application Developer
- Application errors or bugs
- New feature requests
- Data format issues
- Performance problems

## Troubleshooting Checklist

When something goes wrong, work through this checklist:

1. **Basic Checks**
   - [ ] Can you SSH to the server?
   - [ ] Is the Flask app process running? `ps aux | grep gunicorn`
   - [ ] Is Apache running? `ps aux | grep httpd`

2. **Network Checks**
   - [ ] Does `curl http://localhost:5001` work?
   - [ ] Does `curl -k https://asbipvheatprd.ethz.ch` work?
   - [ ] Is port 5001 listening? `ss -tlnp | grep :5001`

3. **Application Checks**
   - [ ] Are data files present? `ls /instances/home/asbipvheatprd/bipv_heat_flask/data/`
   - [ ] Is Python environment active? `which python`
   - [ ] Any recent errors? `tail -20 /instances/home/asbipvheatprd/var/log/httpd/error_log`

4. **Recovery Actions**
   - [ ] Restart Flask app
   - [ ] Restart Apache
   - [ ] Check file permissions
   - [ ] Review configuration files

## Maintenance Log Template

Keep a maintenance log in `/instances/home/asbipvheatprd/maintenance.log`:

```
Date: YYYY-MM-DD
Performed by: [Name]
Action: [What was done]
Reason: [Why it was needed]
Result: [Outcome]
Notes: [Any additional information]
---
```
