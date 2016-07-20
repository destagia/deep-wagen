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
            tl.log("server", "Request from : " + str(address))
            message = self.rfile.readline().strip()
            tl.log("server-message", message)
            command, value = message.split(":")
            if command == 'get_action':
                game_image = value.split(",")
                tl.log("server-message", game_image)
                floats = map(lambda x: float(x), game_image)
                response = 't' if player.jump(floats) else 'f'
                tl.log("server", "response: " + response)
                self.wfile.write(response + "\n")
            elif command == 'learn_win':
                player.learn_win()
                tl.log("server", "learned!")
                self.wfile.write('0\n')
            elif command == 'learn_lose':
                player.learn_lose()
                tl.log("server", "learned!")
                self.wfile.write('0\n')

class WagenServer(SocketServer.ThreadingTCPServer, object):

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

def listen(host, port):
    server = WagenServer((host, port), WagenHandler)
    server.serve_forever()
