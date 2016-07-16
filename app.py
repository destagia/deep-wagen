# -*- coding:utf-8 -*-
import socket
import random

host = "localhost"
port = 43376

serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversock.bind((host, port)) # IPとPORTを指定してバインドします
serversock.listen(10) # 接続の待ち受けをします（キューの最大数を指定）

print 'Waiting for connections...'
clientsock, client_address = serversock.accept() # 接続されればデータを格納

while True:
    rcvmsg = clientsock.recv(1024)
    if rcvmsg != '':
        print('Received -> %s' % (rcvmsg))
        random_value = random.randint(1,10)
        clientsock.sendall('t' if random_value == 1 else 'f')
clientsock.close()
