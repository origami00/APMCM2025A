import http.server
import socketserver
import webbrowser
import os
import threading
import sys

def start_server():
    # Ensure working directory is the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Find available port
    port = 8000
    handler = http.server.SimpleHTTPRequestHandler
    httpd = None
    
    while port < 9000:
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            break
        except OSError:
            port += 1
            
    if httpd is None:
        print("Unable to find available port (8000-9000)")
        return

    url = f"http://localhost:{port}/index.html"
    print(f"--------------------------------------------------")
    print(f" Server started successfully!")
    print(f" Demo URL: {url}")
    print(f" Please keep this window open. Closing it will stop the service.")
    print(f"--------------------------------------------------")

    # Automatically open browser
    def open_browser():
        webbrowser.open(url)

    threading.Timer(1.0, open_browser).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nService stopped")
        httpd.server_close()

if __name__ == "__main__":
    start_server()

