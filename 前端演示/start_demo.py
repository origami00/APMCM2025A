import http.server
import socketserver
import webbrowser
import os
import threading
import sys

def start_server():
    # 确保工作目录是脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 寻找可用端口
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
        print("无法找到可用端口 (8000-9000)")
        return

    url = f"http://localhost:{port}/index.html"
    print(f"--------------------------------------------------")
    print(f" 服务启动成功!")
    print(f" 演示地址: {url}")
    print(f" 请保持此窗口开启，关闭窗口将停止服务。")
    print(f"--------------------------------------------------")

    # 自动打开浏览器
    def open_browser():
        webbrowser.open(url)

    threading.Timer(1.0, open_browser).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        httpd.server_close()

if __name__ == "__main__":
    start_server()

