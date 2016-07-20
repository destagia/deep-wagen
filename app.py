# -*- coding:utf-8 -*-
import random
import tlogger as tl

import dwagen.server as server

host = "localhost" # "0.0.0.0"
port = 43376

tl.log("[Deep Wagen] Start 🚙")

server.listen(host, port)