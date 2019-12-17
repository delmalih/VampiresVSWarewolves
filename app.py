# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
import argparse
import socket
from easydict import EasyDict as edict

""" Local """
import utils
import players

#############
# Functions #
#############

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the game")
    parser.add_argument("-H", "--host", dest="host", help="HOST IPv4 Address", required=True)
    parser.add_argument("-P", "--port", dest="port", help="Server PORT", default="5555")
    parser.add_argument("-n", "--name", dest="name", help="Player name", default="Anonymous")
    parser.add_argument("--david", dest="david", help="Choose david player", action="store_true")
    parser.add_argument("--ibra", dest="ibra", help="Choose ibra player", action="store_true")
    parser.add_argument("--random", dest="random", help="Choose random player", action="store_true")
    return parser.parse_args()

def init_variables():
    args = parse_args()
    constants = edict({})
    constants.connected = False
    return args, constants

#########
# Run ! #
#########

def main():
    try:
        args, constants = init_variables()
        sock = utils.connect_to_server(args.host, args.port, args.name, constants)
        if args.david:
            ply = players.DavidPlayer(args.name, sock)
        elif args.ibra:
            ply = players.IbrahimPlayer(args.name, sock)
        elif args.random:
            ply = players.RandomPlayer(args.name, sock)
        else:
            ply = players.IbrahimPlayer(args.name, sock)
        ply.play()
    except KeyboardInterrupt:
        print("Goodbye ! :)")
        if constants.connected:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()

if __name__ == "__main__":
    main()