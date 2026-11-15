//! roomba_web/auth.py: the web password (scrypt hash in ~/.config/roomba_web/auth.json, mode 600)
//! and login sessions. Same file format, key derivation (scrypt n=2^14, r=8, p=1, 32 bytes, 16-byte
//! random salt), constant-time comparison and session rules (random 256-bit tokens, 7 days, at most
//! 50: the one expiring first makes room), so passwords set by either implementation work in both.

use std::fs;
use std::io::Write as _;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use base64::Engine as _;

use super::json::{self, Py};

pub const SESSION_COOKIE: &str = "roomba_session";
pub const SESSION_TTL_S: f64 = 7.0 * 24.0 * 3600.0;
pub const MAX_SESSIONS: usize = 50;

/// ~/.config/roomba_web and the files in it.
#[derive(Clone, Debug)]
pub struct AuthPaths {
    pub config_dir: PathBuf,
    pub auth_file: PathBuf,
    pub initial_file: PathBuf,
}

/// os.path.expanduser('~'): $HOME, else the passwd entry.
pub fn home_dir() -> PathBuf {
    if let Some(h) = std::env::var_os("HOME") {
        return PathBuf::from(h);
    }
    // SAFETY: getpwuid returns a pointer to static storage (or null); it is read immediately.
    unsafe {
        let pw = libc::getpwuid(libc::getuid());
        if !pw.is_null() && !(*pw).pw_dir.is_null() {
            let dir = std::ffi::CStr::from_ptr((*pw).pw_dir);
            return PathBuf::from(std::ffi::OsStr::from_encoded_bytes_unchecked(dir.to_bytes()));
        }
    }
    PathBuf::from("~")
}

impl AuthPaths {
    pub fn in_dir(config_dir: PathBuf) -> Self {
        AuthPaths { auth_file: config_dir.join("auth.json"), initial_file: config_dir.join("initial_password"), config_dir }
    }

    /// The paths auth.py uses: ~/.config/roomba_web/{auth.json,initial_password}.
    pub fn default_paths() -> Self {
        Self::in_dir(home_dir().join(".config").join("roomba_web"))
    }
}

/// hashlib.scrypt(password, salt=salt, n=2**14, r=8, p=1, dklen=32)
pub fn scrypt_hash(password: &[u8], salt: &[u8]) -> [u8; 32] {
    let params = scrypt::Params::new(14, 8, 1).expect("valid scrypt parameters");
    let mut out = [0u8; 32];
    scrypt::scrypt(password, salt, &params, &mut out).expect("32-byte output");
    out
}

pub fn random_bytes(n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    getrandom::fill(&mut buf).expect("OS random number generator");
    buf
}

/// secrets.token_urlsafe(nbytes)
pub fn token_urlsafe(nbytes: usize) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(random_bytes(nbytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// bytes.fromhex(s): pairs of hex digits, ASCII whitespace allowed between pairs.
fn from_hex(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let pair = s.get(i..i + 2)?;
        if !pair.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        out.push(u8::from_str_radix(pair, 16).ok()?);
        i += 2;
    }
    Some(out)
}

/// hmac.compare_digest for bytes: no early exit on the first difference.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let diff = a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}

/// auth._write_private: the config directory (mode 700), then path.tmp (created mode 600) renamed
/// over path.
fn write_private(paths: &AuthPaths, path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(parent) = paths.config_dir.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::DirBuilder::new().mode(0o700).create(&paths.config_dir) {
        Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => return Err(e),
        _ => {}
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let mut f = fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&tmp)?;
    f.write_all(text.as_bytes())?;
    drop(f);
    fs::rename(&tmp, path)
}

/// auth.set_password: a new random salt and the hash, as {"salt": hex, "hash": hex}.
pub fn set_password(paths: &AuthPaths, password: &str) -> std::io::Result<()> {
    let salt = random_bytes(16);
    let hash = scrypt_hash(password.as_bytes(), &salt);
    let text = json::dumps(&Py::Dict(vec![("salt".into(), Py::Str(hex(&salt))), ("hash".into(), Py::Str(hex(&hash)))]));
    write_private(paths, &paths.auth_file, &text)
}

/// Auth.__init__: without auth.json, generates a password, stores its hash and writes it to
/// initial_password. Returns whether it did (the caller logs where, never the password).
pub fn ensure_password(paths: &AuthPaths) -> std::io::Result<bool> {
    if paths.auth_file.exists() {
        return Ok(false);
    }
    let password = token_urlsafe(9);
    set_password(paths, &password)?;
    write_private(paths, &paths.initial_file, &format!("{password}\n"))?;
    Ok(true)
}

/// Auth.check: re-reads auth.json (so set_password applies without a restart). Err means the file
/// could not be read or parsed (Python logs "Cannot read" and refuses the login).
pub fn check_password(paths: &AuthPaths, password: &str) -> Result<bool, ()> {
    let text = fs::read_to_string(&paths.auth_file).map_err(|_| ())?;
    let data = json::loads(&text).map_err(|_| ())?;
    let field = |k: &str| match data.get(k) {
        Ok(Py::Str(s)) => from_hex(s).ok_or(()),
        _ => Err(()),
    };
    let expected = field("hash")?;
    let salt = field("salt")?;
    Ok(constant_time_eq(&scrypt_hash(password.as_bytes(), &salt), &expected))
}

/// Auth.sessions: token -> expiry (time.time()), in insertion order like the Python dict.
#[derive(Default)]
pub struct Sessions {
    items: Vec<(String, f64)>,
}

impl Sessions {
    pub fn new_session(&mut self, now: f64) -> String {
        self.items.retain(|(_, exp)| *exp > now);
        if self.items.len() >= MAX_SESSIONS {
            let mut oldest = 0;
            for (i, (_, exp)) in self.items.iter().enumerate() {
                if *exp < self.items[oldest].1 {
                    oldest = i;
                }
            }
            self.items.remove(oldest);
        }
        let token = token_urlsafe(32);
        self.items.push((token.clone(), now + SESSION_TTL_S));
        token
    }

    fn find(&self, token: &str) -> Option<usize> {
        // Compare every token in full so lookup time does not depend on where a guess differs.
        let mut found = None;
        for (i, (t, _)) in self.items.iter().enumerate() {
            if constant_time_eq(t.as_bytes(), token.as_bytes()) {
                found = Some(i);
            }
        }
        found
    }

    pub fn valid(&self, token: Option<&str>, now: f64) -> bool {
        self.find(token.unwrap_or("")).is_some_and(|i| self.items[i].1 > now)
    }

    pub fn revoke(&mut self, token: Option<&str>) {
        if let Some(i) = self.find(token.unwrap_or("")) {
            self.items.remove(i);
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrypt_matches_rfc7914() {
        // RFC 7914 test vector 3 (N=16384, r=8, p=1), first 32 bytes.
        let out = scrypt_hash(b"pleaseletmein", b"SodiumChloride");
        assert_eq!(hex(&out), "7023bdcb3afd7348461c06cd81fd38ebfda8fbba904f8e3ea9b543f6545da1f2");
    }

    #[test]
    fn password_file_round_trip() {
        let dir = std::env::temp_dir().join(format!("roomba_web_auth_test_{}", std::process::id()));
        let paths = AuthPaths::in_dir(dir.join("cfg"));
        assert!(ensure_password(&paths).unwrap());
        assert!(!ensure_password(&paths).unwrap());
        let initial = fs::read_to_string(&paths.initial_file).unwrap();
        assert_eq!(check_password(&paths, initial.trim_end_matches('\n')), Ok(true));
        set_password(&paths, "correct horse").unwrap();
        assert_eq!(check_password(&paths, "correct horse"), Ok(true));
        assert_eq!(check_password(&paths, "correct horsE"), Ok(false));
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(fs::metadata(&paths.auth_file).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(fs::metadata(&paths.config_dir).unwrap().permissions().mode() & 0o777, 0o700);
        let text = fs::read_to_string(&paths.auth_file).unwrap();
        assert!(text.starts_with("{\"salt\": \"") && text.contains("\", \"hash\": \"") && text.ends_with("\"}"));
        fs::write(&paths.auth_file, "not json").unwrap();
        assert_eq!(check_password(&paths, "correct horse"), Err(()));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sessions_expire_and_are_capped() {
        let mut s = Sessions::default();
        let t = s.new_session(1000.0);
        assert_eq!(t.len(), 43);
        assert!(s.valid(Some(&t), 1000.0 + SESSION_TTL_S - 1.0));
        assert!(!s.valid(Some(&t), 1000.0 + SESSION_TTL_S));
        assert!(!s.valid(None, 1000.0));
        assert!(!s.valid(Some(""), 1000.0));
        let tokens: Vec<String> = (0..MAX_SESSIONS).map(|i| s.new_session(2000.0 + i as f64)).collect();
        assert_eq!(s.len(), MAX_SESSIONS);
        assert!(!s.valid(Some(&t), 2000.0), "the oldest session made room");
        assert!(s.valid(Some(&tokens[0]), 2100.0));
        s.revoke(Some(&tokens[0]));
        assert!(!s.valid(Some(&tokens[0]), 2100.0));
        assert!(s.valid(Some(&tokens[1]), 2100.0));
    }
}
