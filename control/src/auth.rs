//! Login: the robot web app's password file (~/.config/roomba_web/auth.json, written by
//! `set_password` of roomba_rs or roomba_web) and this page's own sessions.
//!
//! auth.json is {"salt": hex, "hash": hex} with hash = scrypt(password, salt, n=2^14, r=8, p=1,
//! 32 bytes); it is re-read on every login so a new password applies without a restart. Sessions
//! are random 256-bit tokens kept in memory (a restart logs everyone out).

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::Path;
use std::time::{Duration, Instant};

pub const COOKIE: &str = "roomba_control";
const MAX_SESSIONS: usize = 50;

/// Wrong passwords in a row from one address before it has to wait, and how long (doubling for
/// every further wrong password, at most MAX_LOCK).
const FREE_FAILURES: u32 = 5;
const FIRST_LOCK: Duration = Duration::from_secs(30);
const MAX_LOCK: Duration = Duration::from_secs(15 * 60);

/// hashlib.scrypt(password, salt=salt, n=2**14, r=8, p=1, dklen=32), as roomba_web's auth.
pub fn scrypt_hash(password: &[u8], salt: &[u8]) -> [u8; 32] {
    let params = scrypt::Params::new(14, 8, 1).expect("valid scrypt parameters");
    let mut out = [0u8; 32];
    scrypt::scrypt(password, salt, &params, &mut out).expect("32-byte output");
    out
}

/// bytes.fromhex: pairs of hex digits, ASCII whitespace allowed between pairs.
fn from_hex(s: &str) -> Option<Vec<u8>> {
    let digits: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if digits.len() % 2 != 0 {
        return None;
    }
    digits
        .chunks(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok())
        .collect()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// No early exit on the first difference (hmac.compare_digest).
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let diff = a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}

/// Ok(true / false) for a readable password file, Err(reason) when it can't be read or parsed.
pub fn check_password(auth_file: &Path, password: &str) -> Result<bool, String> {
    let text = std::fs::read_to_string(auth_file).map_err(|e| format!("cannot read {}: {e}", auth_file.display()))?;
    let data: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", auth_file.display()))?;
    let field = |k: &str| {
        data.get(k)
            .and_then(|v| v.as_str())
            .and_then(from_hex)
            .ok_or_else(|| format!("{}: no valid \"{k}\"", auth_file.display()))
    };
    let expected = field("hash")?;
    let salt = field("salt")?;
    Ok(constant_time_eq(&scrypt_hash(password.as_bytes(), &salt), &expected))
}

pub fn random_token() -> String {
    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf).expect("OS random number generator");
    hex(&buf)
}

/// Logged-in browsers: token -> expiry.
pub struct Sessions {
    items: Vec<(String, Instant)>,
    ttl: Duration,
}

impl Sessions {
    pub fn new(ttl: Duration) -> Self {
        Sessions { items: Vec::new(), ttl }
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    pub fn create(&mut self, now: Instant) -> String {
        self.items.retain(|(_, exp)| *exp > now);
        if self.items.len() >= MAX_SESSIONS {
            if let Some(oldest) = (0..self.items.len()).min_by_key(|&i| self.items[i].1) {
                self.items.remove(oldest);
            }
        }
        let token = random_token();
        self.items.push((token.clone(), now + self.ttl));
        token
    }

    fn find(&self, token: &str) -> Option<usize> {
        // Compare every token in full so the time taken doesn't depend on where a guess differs.
        let mut found = None;
        for (i, (t, _)) in self.items.iter().enumerate() {
            if constant_time_eq(t.as_bytes(), token.as_bytes()) {
                found = Some(i);
            }
        }
        found
    }

    pub fn valid(&self, token: Option<&str>, now: Instant) -> bool {
        match token {
            Some(t) if !t.is_empty() => self.find(t).is_some_and(|i| self.items[i].1 > now),
            _ => false,
        }
    }

    pub fn revoke(&mut self, token: Option<&str>) {
        if let Some(i) = token.and_then(|t| self.find(t)) {
            self.items.remove(i);
        }
    }
}

/// Login backoff per client address: FREE_FAILURES wrong passwords, then a lock that doubles.
#[derive(Default)]
pub struct Limiter {
    clients: HashMap<IpAddr, (u32, Instant, Option<Instant>)>, // failures, last failure, locked until
}

impl Limiter {
    /// Seconds this address still has to wait, if it is locked out.
    pub fn locked(&mut self, ip: IpAddr, now: Instant) -> Option<u64> {
        self.clients.retain(|_, (_, last, until)| {
            now.duration_since(*last) < Duration::from_secs(3600) || until.is_some_and(|u| u > now)
        });
        let until = self.clients.get(&ip)?.2?;
        (until > now).then(|| until.duration_since(now).as_secs().max(1))
    }

    pub fn failure(&mut self, ip: IpAddr, now: Instant) {
        let entry = self.clients.entry(ip).or_insert((0, now, None));
        entry.0 += 1;
        entry.1 = now;
        if entry.0 >= FREE_FAILURES {
            let doublings = (entry.0 - FREE_FAILURES).min(10);
            entry.2 = Some(now + (FIRST_LOCK * 2u32.pow(doublings)).min(MAX_LOCK));
        }
    }

    pub fn success(&mut self, ip: IpAddr) {
        self.clients.remove(&ip);
    }
}

/// The value of one cookie in a Cookie header.
pub fn cookie<'a>(header: Option<&'a str>, name: &str) -> Option<&'a str> {
    header?.split(';').filter_map(|part| part.trim().split_once('=')).filter(|(k, _)| *k == name).map(|(_, v)| v.trim()).last()
}

/// application/x-www-form-urlencoded: the value of one field.
pub fn form_field(body: &[u8], name: &str) -> Option<String> {
    let body = std::str::from_utf8(body).ok()?;
    body.split('&').filter_map(|kv| kv.split_once('=')).find(|(k, _)| url_decode(k).as_deref() == Some(name)).and_then(|(_, v)| url_decode(v))
}

fn url_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' => {
                let h = s.get(i + 1..i + 3)?;
                out.push(u8::from_str_radix(h, 16).ok()?);
                i += 2;
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
    }

    // The fixtures were written by the real tools (HOME pointing at a scratch directory):
    //   printf 'fixture-password\nfixture-password\n' | setsid .../roomba_rs/set_password
    //   printf 'fixture-password\nfixture-password\n' | setsid .../roomba_web/set_password
    #[test]
    fn verifies_hash_written_by_roomba_rs_set_password() {
        let f = fixture("auth_roomba_rs.json");
        assert_eq!(check_password(&f, "fixture-password"), Ok(true));
        assert_eq!(check_password(&f, "fixture-passworD"), Ok(false));
        assert_eq!(check_password(&f, ""), Ok(false));
    }

    #[test]
    fn verifies_hash_written_by_roomba_web_set_password() {
        let f = fixture("auth_roomba_web_py.json");
        assert_eq!(check_password(&f, "fixture-password"), Ok(true));
        assert_eq!(check_password(&f, "fixture-password "), Ok(false));
    }

    #[test]
    fn unreadable_password_file_is_an_error() {
        assert!(check_password(&fixture("missing.json"), "x").is_err());
        assert!(check_password(&fixture("../e2e/fake_run_robot.sh"), "x").is_err());
    }

    #[test]
    fn sessions_expire_and_can_be_revoked() {
        let t0 = Instant::now();
        let mut s = Sessions::new(Duration::from_secs(60));
        let a = s.create(t0);
        assert_eq!(a.len(), 64);
        assert!(s.valid(Some(&a), t0 + Duration::from_secs(59)));
        assert!(!s.valid(Some(&a), t0 + Duration::from_secs(60)));
        assert!(!s.valid(Some(&a[..63]), t0));
        assert!(!s.valid(None, t0) && !s.valid(Some(""), t0));
        let b = s.create(t0);
        s.revoke(Some(&b));
        assert!(!s.valid(Some(&b), t0) && s.valid(Some(&a), t0));
    }

    #[test]
    fn wrong_passwords_lock_the_address_out() {
        let t0 = Instant::now();
        let ip: IpAddr = "10.0.0.7".parse().unwrap();
        let mut l = Limiter::default();
        for _ in 0..FREE_FAILURES - 1 {
            l.failure(ip, t0);
        }
        assert_eq!(l.locked(ip, t0), None);
        l.failure(ip, t0);
        assert_eq!(l.locked(ip, t0), Some(30));
        l.failure(ip, t0);
        assert_eq!(l.locked(ip, t0), Some(60), "doubles");
        assert_eq!(l.locked("10.0.0.8".parse().unwrap(), t0), None, "per address");
        assert_eq!(l.locked(ip, t0 + Duration::from_secs(61)), None);
        l.success(ip);
        l.failure(ip, t0);
        assert_eq!(l.locked(ip, t0), None, "a right password resets the count");
    }

    #[test]
    fn parses_cookies_and_forms() {
        assert_eq!(cookie(Some("roomba_session=x; roomba_control=abc"), COOKIE), Some("abc"));
        assert_eq!(cookie(Some("roomba_session=x"), COOKIE), None);
        assert_eq!(form_field(b"user=a&password=p%40ss+w%C3%B6rd%26", "password").as_deref(), Some("p@ss w\u{f6}rd&"));
        assert_eq!(form_field(b"password=%zz", "password"), None);
    }
}
