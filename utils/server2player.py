# -*- coding: UTF-8 -*-

###########
# Imports #
###########

import struct
import sys

#############
# Functions #
#############

def receive_data(sock, size, fmt):
    """
    Description: TODO

    Arguments:
    ----------
        sock (TODO): TODO
        size (TODO): TODO
        fmt  (TODO): TODO

    Ouputs:
    -------
        res (TODO): TODO
    """

    data = bytes()
    while len(data) < size:
        data += sock.recv(size - len(data))
    res = struct.unpack(fmt, data)
    return res

def get_header(sock):
    header = sock.recv(3).decode("ascii")
    return header

def header2action(header):
    if header == "SET":
        return get_SET
    elif header == "HUM":
        return get_HUM
    elif header == "HME":
        return get_HME
    elif header == "MAP" or header == "UPD":
        return get_MAP_UPD
    elif header == "BYE":
        sys.exit(0)
    else:
        raise NotImplementedError

def get_SET(sock):
    height, width = receive_data(sock, 2, "2B")
    res = {"height":height, "width":width}
    return res

def get_HUM(sock):
    number_of_houses = receive_data(sock, 1, "1B")[0]
    houses_raw = receive_data(sock, number_of_houses * 2, "{}B".format(number_of_houses * 2))
    houses = [(houses_raw[i], houses_raw[i+1]) for i in range(0, len(houses_raw), 2)]
    res = {"houses": houses}
    return res

def get_HME(sock):
    start_position = tuple(receive_data(sock, 2, "2B"))
    res = {"start_position": start_position}
    return res

def get_MAP_UPD(sock):
    number_commands = receive_data(sock,1, "1B")[0]
    commands_raw = receive_data(sock, number_commands * 5, "{}B".format(number_commands * 5))
    commands = {}
    for i in range(0, len(commands_raw), 5):
        x, y, n_h, n_v, n_w = commands_raw[i:i+5]
        commands[(x, y)] = (n_h, n_v, n_w)
    res = {"commands": commands}
    return res