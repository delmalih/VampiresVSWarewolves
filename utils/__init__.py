# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
import socket
import struct
import sys

""" Local """
from . import player2server

#############
# Functions #
#############


def connect_to_server(host, port, name, constants):
    """
    Description: TODO
    
    Arguments:
    ----------
        host (str): TODO
        port (str): TODO
        name (str): TODO

    Outputs:
    --------
        sock (TODO): TODO
    """
    
    # Connect to server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))
        constants.connected = True
    except:
        sys.exit(0)
        
    return sock
