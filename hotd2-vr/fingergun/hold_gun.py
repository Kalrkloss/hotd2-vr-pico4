"""Test sender: holds the light gun at one position. hold_gun.py x y seconds [port] [fire_every_s]"""
import socket
import sys
import time

x, y, seconds = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
port = int(sys.argv[4]) if len(sys.argv) > 4 else 27015
fire_every = float(sys.argv[5]) if len(sys.argv) > 5 else 0
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
start = time.time()
while (t := time.time() - start) < seconds:
    fire = 1 if fire_every and (t % fire_every) < 0.15 else 0
    sock.sendto(f"LG 0 {round(x * 10000)} {round(y * 10000)} {fire}".encode(), ("127.0.0.1", port))
    time.sleep(0.004)
