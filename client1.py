import socket

def send_message():
    host = "192.168.1.101"
    port = 29525
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    message = 'hello,computer2'
    client_socket.send(message.encode('utf-8'))
    print('已发送消息到主机 2：%s' % message)
    
    client_socket.close()

# 发送消息
send_message()
