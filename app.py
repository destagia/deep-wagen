# -*- coding:utf-8 -*-
import socket
import random

from dwagen import Player

host = "localhost"
port = 43376

serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversock.bind((host, port)) # IPとPORTを指定してバインドします
serversock.listen(10) # 接続の待ち受けをします（キューの最大数を指定）

print 'Waiting for connections...'
clientsock, client_address = serversock.accept() # 接続されればデータを格納

player = Player()

while True:
    rcvmsg = clientsock.recv(10000000)
    if rcvmsg != '':
        try:
            print("Receive -> " + rcvmsg)
            command, value = rcvmsg.split(":")
            if command == 'get_action':
                floats = map(lambda x: float(x), value.split(","))
                clientsock.sendall('t' if player.jump(floats) else 'f')
            elif command == 'learn_win':
                player.learn_win()
                clientsock.sendall('ok')
            elif command == 'learn_lose':
                player.learn_lose()
                clientsock.sendall('ok')
        except Exception as e:
            print(e)
            clientsock.sendall('f')


clientsock.close()
