# -*- coding: UTF-8 -*-

###########
# Imports #
###########

import struct

#############
# Functions #
#############

def send_NME(sock, params):
    params.name = params.name.encode("ascii")
    params.length = len(params.name)
    sock.send("NME".encode("ascii"))
    sock.send(struct.pack("1B", params.length))
    sock.send(params.name)

def send_MOV(sock, params):
    params.length = len(params.movs)
    sock.send("MOV".encode("ascii"))
    sock.send(struct.pack("1B", params.length))
    for mov in params.movs:
        sock.send(struct.pack("1B", mov.x_s))
        sock.send(struct.pack("1B", mov.y_s))
        sock.send(struct.pack("1B", mov.n))
        sock.send(struct.pack("1B", mov.x_t))
        sock.send(struct.pack("1B", mov.y_t))