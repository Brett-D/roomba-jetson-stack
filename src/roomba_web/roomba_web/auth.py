"""Password + session handling for the Roomba web app.

The password is stored as an scrypt hash in ~/.config/roomba_web/auth.json (mode 600).
On first start a random password is generated and written to
~/.config/roomba_web/initial_password (mode 600); change it with `ros2 run roomba_web set_password`.
"""
import hashlib
import hmac
import json
import os
import secrets
import time

CONFIG_DIR = os.path.expanduser('~/.config/roomba_web')
AUTH_FILE = os.path.join(CONFIG_DIR, 'auth.json')
INITIAL_FILE = os.path.join(CONFIG_DIR, 'initial_password')
SESSION_TTL_S = 7 * 24 * 3600
MAX_SESSIONS = 50


def _scrypt(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


def _write_private(path, text):
    os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
    fd = os.open(path + '.tmp', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(path + '.tmp', path)


def set_password(password):
    salt = secrets.token_bytes(16)
    _write_private(AUTH_FILE, json.dumps({'salt': salt.hex(), 'hash': _scrypt(password, salt).hex()}))


class Auth:

    def __init__(self, logger):
        self.logger = logger
        self.sessions = {}
        if not os.path.exists(AUTH_FILE):
            password = secrets.token_urlsafe(9)
            set_password(password)
            _write_private(INITIAL_FILE, password + '\n')
            logger.warn(f'No web password set; generated one in {INITIAL_FILE} '
                        f'(change it with: ros2 run roomba_web set_password)')

    def check(self, password):
        try:
            with open(AUTH_FILE) as f:
                data = json.load(f)
            expected = bytes.fromhex(data['hash'])
            return hmac.compare_digest(_scrypt(password, bytes.fromhex(data['salt'])), expected)
        except (OSError, ValueError, KeyError):
            self.logger.error(f'Cannot read {AUTH_FILE}')
            return False

    def new_session(self):
        now = time.time()
        self.sessions = {t: exp for t, exp in self.sessions.items() if exp > now}
        if len(self.sessions) >= MAX_SESSIONS:
            oldest = min(self.sessions, key=self.sessions.get)
            del self.sessions[oldest]
        token = secrets.token_urlsafe(32)
        self.sessions[token] = now + SESSION_TTL_S
        return token

    def valid(self, token):
        exp = self.sessions.get(token or '')
        return exp is not None and exp > time.time()

    def revoke(self, token):
        self.sessions.pop(token or '', None)
