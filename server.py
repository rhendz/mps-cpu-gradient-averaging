import http.server
import socketserver
import json

class GradientAverager(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(content_length))
        gradients = data['gradients']
        averaged_gradient = sum(gradients) / len(gradients)
        print(averaged_gradient)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {'averaged_gradient': averaged_gradient}
        self.wfile.write(json.dumps(response).encode())

PORT = 8000
with socketserver.TCPServer(("", PORT), GradientAverager) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()