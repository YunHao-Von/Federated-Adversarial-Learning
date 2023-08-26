import socket

def start_server():
    host = ''  # 监听所有网络接口
    port = 61234
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print('服务器正在监听端口：%d' % port)
    
    # 接受连接
    client_socket, addr = server_socket.accept()
    print('与主机 1 建立连接：%s' % str(addr))
    
    # 接收消息
    data = client_socket.recv(1024).decode('utf-8')
    print('接收到来自主机 1 的消息：%s' % data)
    
    # 关闭连接
    client_socket.close()

# 启动服务器
start_server()
