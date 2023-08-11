from socket import *

ip = '127.0.0.1'
port = 25001

clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect((ip, port))

while True:
    data = clientSock.recv(1024)
    if len(data) < 1:
        break
    print(data.decode('utf-8'))

clientSock.close()