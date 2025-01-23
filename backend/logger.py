from pythonjsonlogger import jsonlogger
import logging
import os
from datetime import datetime
from functools import wraps
from flask import request, Response
import json

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure JSON logger
logger = logging.getLogger('soog_logger')
logger.setLevel(logging.INFO)

# File handler for JSON logs
json_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'app.json'))
json_formatter = jsonlogger.JsonFormatter(
    fmt='%(asctime)s %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
json_handler.setFormatter(json_formatter)
logger.addHandler(json_handler)

def log_activity(activity_type):
    """Decorator to log API activity"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = datetime.utcnow()
            
            # Get request details
            request_data = {
                'method': request.method,
                'path': request.path,
                'ip': request.remote_addr,
                'user_agent': request.headers.get('User-Agent'),
                'activity_type': activity_type
            }
            
            # Add request body for POST requests
            if request.method == 'POST' and request.is_json:
                # Sanitize sensitive data if needed
                body = request.get_json()
                if 'prompt' in body:
                    request_data['prompt'] = body['prompt']
            
            try:
                response = f(*args, **kwargs)
                status_code = response.status_code if isinstance(response, Response) else 200
                
                # Log successful request
                logger.info('API Request', extra={
                    'request': request_data,
                    'status_code': status_code,
                    'duration_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
                })
                
                return response
            
            except Exception as e:
                # Log error
                logger.error('API Error', extra={
                    'request': request_data,
                    'error': str(e),
                    'duration_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
                })
                raise
            
        return decorated_function
    return decorator

def get_logs(limit=100, reverse=True):
    """Get the most recent logs"""
    log_file = os.path.join(LOGS_DIR, 'app.json')
    logs = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
    
    # Sort logs by timestamp in reverse order (newest first)
    logs.sort(key=lambda x: x.get('asctime', ''), reverse=reverse)
    
    # Limit the number of logs
    return logs[:limit]

def format_logs_html(logs):
    """Format logs as HTML for display"""
    html = """
    <html>
    <head>
        <title>SOOG Logs</title>
        <style>
            body { font-family: monospace; background: #1a1a1a; color: #fff; padding: 20px; }
            .log-entry { 
                border: 1px solid #333; 
                margin: 10px 0; 
                padding: 10px; 
                border-radius: 4px;
                background: #222;
            }
            .log-entry pre { 
                margin: 5px 0; 
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .timestamp { color: #4CAF50; }
            .level { 
                display: inline-block;
                padding: 2px 6px;
                border-radius: 3px;
                margin-right: 10px;
            }
            .INFO { background: #2196F3; }
            .ERROR { background: #f44336; }
            .request { color: #90caf9; }
            .error { color: #ef9a9a; }
        </style>
    </head>
    <body>
        <h1>SOOG Activity Logs</h1>
    """
    
    for log in logs:
        level = log.get('levelname', 'INFO')
        timestamp = log.get('asctime', '')
        
        html += f"""
        <div class="log-entry">
            <pre><span class="timestamp">{timestamp}</span> <span class="level {level}">{level}</span></pre>
        """
        
        if 'request' in log:
            req = log['request']
            html += f"""
            <pre class="request">
Method: {req.get('method', '')}
Path: {req.get('path', '')}
IP: {req.get('ip', '')}
Activity: {req.get('activity_type', '')}
User-Agent: {req.get('user_agent', '')}
            </pre>
            """
            
            if 'prompt' in req:
                html += f'<pre class="request">Prompt: {req["prompt"]}</pre>'
        
        if 'error' in log:
            html += f'<pre class="error">Error: {log["error"]}</pre>'
        
        if 'duration_ms' in log:
            html += f'<pre>Duration: {log["duration_ms"]:.2f}ms</pre>'
        
        html += "</div>"
    
    html += """
    </body>
    </html>
    """
    
    return html
