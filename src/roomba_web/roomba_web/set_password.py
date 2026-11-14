#!/usr/bin/env python3
"""Sets the Roomba web app password: ros2 run roomba_web set_password"""
import getpass
import os
import sys

from roomba_web.auth import INITIAL_FILE, set_password


def main():
    pw = getpass.getpass('New web password (min 8 characters): ')
    if len(pw) < 8:
        sys.exit('Password too short.')
    if getpass.getpass('Repeat: ') != pw:
        sys.exit('Passwords do not match.')
    set_password(pw)
    if os.path.exists(INITIAL_FILE):
        os.remove(INITIAL_FILE)
    print('Password updated. Existing logins stay valid until they expire or the server restarts.')


if __name__ == '__main__':
    main()
