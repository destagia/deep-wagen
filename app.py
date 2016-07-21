# -*- coding:utf-8 -*-

import argparse
import random
import tlogger as tl

import dwagen.server as server

parser = argparse.ArgumentParser()
parser.add_argument('-H', '--host', action='store', default="0.0.0.0")
parser.add_argument('-P', '--port', action='store', default=43376)
parser.add_argument('-v', '--verbose', action='store_true', default=False)

parse_args = parser.parse_args()
host = parse_args.host
port = parse_args.port

tl.set_verbose(parse_args.verbose)

tl.log("[Deep Wagen] Start 🚙  - %s:%d" % (host, port))

server.listen(host, port)