#!/usr/bin/env python3
"""Misbehaving clients for run.sh, connecting to 127.0.0.1 from other loopback addresses:
  stall PORT READY_FILE HOLD_S  from 8 addresses (127.0.0.2-9), each: 4 connections that send
                                nothing, 4 that send their headers a byte a second (slowloris), 10
                                that send a login body a byte a second and 2 that ask for app.js and
                                never read. READY_FILE appears when all are connected; after HOLD_S
                                seconds it prints how many of each kind the server had closed.
  cap PORT                      40 connections from one address (127.0.0.40): how many it refused
  bighead PORT / bigbody PORT   the status of a request with 20 KB of headers / a 20 KB body"""
import select
import socket
import sys
import time


def connect(port, src, rcvbuf=None):
    s = socket.socket()
    if rcvbuf:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    s.bind((src, 0))
    s.connect(('127.0.0.1', port))
    return s


def ended(s):
    """The server has closed it (or answered and closed it): there is something to read."""
    return bool(select.select([s], [], [], 0)[0])


def status(port, raw):
    s = connect(port, '127.0.0.1')
    s.settimeout(5)
    try:
        s.sendall(raw)
    except OSError:
        pass
    data = b''
    try:
        while chunk := s.recv(4096):
            data += chunk
    except OSError:
        pass
    return data.split(b' ', 2)[1].decode() if data.startswith(b'HTTP/') else 'none'


def stall(port, ready, hold):
    host = f'127.0.0.1:{port}'
    kinds = {'idle': [], 'slowloris': [], 'slowbody': [], 'noread': []}
    for i in range(8):
        src = f'127.0.0.{i + 2}'
        for _ in range(4):
            kinds['idle'].append(connect(port, src))
        for _ in range(4):
            s = connect(port, src)
            s.sendall(f'GET /login HTTP/1.1\r\nHost: {host}\r\nX-Slow: '.encode())
            kinds['slowloris'].append(s)
        for _ in range(10):
            s = connect(port, src)
            s.sendall((f'POST /login HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\n'
                       'Content-Type: application/x-www-form-urlencoded\r\nContent-Length: 4000\r\n\r\npassword=').encode())
            kinds['slowbody'].append(s)
        for _ in range(2):
            s = connect(port, src, rcvbuf=4096)
            s.sendall(f'GET /app.js HTTP/1.1\r\nHost: {host}\r\n\r\n'.encode() * 20)
            kinds['noread'].append(s)
    open(ready, 'w').close()
    end = time.monotonic() + hold
    gone = {k: set() for k in kinds}
    while time.monotonic() < end:
        for k in ('idle', 'slowloris', 'slowbody'):
            for i, s in enumerate(kinds[k]):
                if i in gone[k]:
                    continue
                if ended(s):
                    gone[k].add(i)
                elif k != 'idle':
                    try:
                        s.send(b'a')
                    except OSError:
                        gone[k].add(i)
        time.sleep(1)
    print(', '.join(f'{k} {len(gone[k])}/{len(kinds[k])}' for k in ('idle', 'slowloris', 'slowbody')))


def main():
    mode, port = sys.argv[1], int(sys.argv[2])
    host = f'127.0.0.1:{port}'
    if mode == 'stall':
        stall(port, sys.argv[3], float(sys.argv[4]))
    elif mode == 'cap':
        conns = [connect(port, '127.0.0.40') for _ in range(40)]
        time.sleep(0.5)
        print(f'refused {sum(ended(s) for s in conns)}/40')
    elif mode == 'bighead':
        print(status(port, f'GET /login HTTP/1.1\r\nHost: {host}\r\nX-Big: {"a" * 20000}\r\n\r\n'.encode()))
    elif mode == 'bigbody':
        print(status(port, f'POST /login HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nContent-Length: 20000\r\n\r\n'.encode() + b'a' * 20000))


if __name__ == '__main__':
    main()
