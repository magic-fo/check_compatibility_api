from http.server import BaseHTTPRequestHandler
import json
import datetime

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response_data = {
            'message': 'Hello from minimal Python serverless function!',
            'timestamp': datetime.datetime.now().isoformat(),
            'path': self.path
        }
        
        self.wfile.write(json.dumps(response_data).encode('utf-8')) 