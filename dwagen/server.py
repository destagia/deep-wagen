# -*- coding:utf-8 -*-

import SocketServer
import socket
import tlogger as tl
from dwagen import Player

player = Player()

class WagenHandler(SocketServer.StreamRequestHandler, object):

    def handle(self):
        while True:
            address = self.client_address[0]
            # tl.log("server", "Request from : " + str(address))
            message = self.rfile.readline().strip()
            # tl.log("server-message", message)
            command, value = message.split(":")
            if command == 'get_action':
                raw_image, reward, is_game_end = value.split(';')
                game_image = raw_image.split(",")
                # tl.log("server-message", game_image)
                floats = map(lambda x: float(x), game_image)
                response = 't' if player.jump(floats, float(reward), is_game_end == "True") else 'f'
                # tl.log("server", "response: " + response)
                self.wfile.write(response + "\n")
            elif command == 'learn':
                player.learn()
                tl.log("server", "learned!")
                self.wfile.write('0\n')

class WagenServer(SocketServer.ThreadingTCPServer, object):

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

def listen(host, port):
    server = WagenServer((host, port), WagenHandler)
    print("server forever")
    server.serve_forever()
    print("start :)")

