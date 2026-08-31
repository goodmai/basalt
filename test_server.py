import http.server
import json
import socketserver
import sys

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        print(f"POST {self.path}", file=sys.stderr)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print(f"Body: {post_data}", file=sys.stderr)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"dummy"}],"model":"test","stop_reason":"end_turn","stop_sequence":None,"usage":{"input_tokens":0,"output_tokens":1}}).encode())

    def do_GET(self):
        print(f"GET {self.path}", file=sys.stderr)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        if self.path == '/v1/models':
            self.wfile.write(json.dumps({"object":"list","data":[{"id":"claude-local/test","display_name":"test","object":"model","created":1,"owned_by":"gemm","is_loaded":True}]}).encode())
        else:
            self.wfile.write(b"{}")

with socketserver.TCPServer(("127.0.0.1", 8081), Handler) as httpd:
    print("Serving at port 8081", file=sys.stderr)
    httpd.serve_forever()
