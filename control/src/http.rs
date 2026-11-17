//! The page's HTTP server (std only). Each connection gets its own thread (at most
//! `Limits::connections` at a time, `Limits::per_client` from one address), reads one request
//! against deadlines and size limits, gets its answer and is closed (`Connection: close`). A client
//! that connects and sends nothing, sends slowly or never reads the answer holds only its own
//! thread, and only until its deadline: it can't delay anyone else's request (a Stop, say).
//!
//! "[::]:PORT" listens on IPv6 and IPv4: one socket for both where the system allows it; where it
//! keeps IPv6 sockets IPv6-only (net.ipv6.bindv6only=1), 0.0.0.0 gets a socket of its own; and
//! without IPv6 (or with `Ipv6::Off`), 0.0.0.0 only.

use std::collections::HashMap;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::os::fd::FromRawFd;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Limits {
    /// From the connection to the end of the headers.
    pub head_time: Duration,
    /// From the end of the headers to the end of the body.
    pub body_time: Duration,
    /// To send the whole answer.
    pub write_time: Duration,
    pub head_bytes: usize,
    pub body_bytes: usize,
    pub connections: usize,
    pub per_client: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            head_time: Duration::from_secs(10),
            body_time: Duration::from_secs(10),
            write_time: Duration::from_secs(10),
            head_bytes: 16 * 1024,
            body_bytes: 16 * 1024,
            connections: 256,
            per_client: 32,
        }
    }
}

pub struct Request {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub peer: SocketAddr,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    /// The client's address (an IPv4 client of a dual-stack socket as IPv4).
    pub fn client_ip(&self) -> IpAddr {
        self.peer.ip().to_canonical()
    }
}

pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn text(status: u16, body: &str) -> Response {
        Response { status, headers: vec![("Content-Type".into(), "text/plain; charset=utf-8".into())], body: body.as_bytes().to_vec() }
    }
}

pub type Handler = Arc<dyn Fn(Request) -> Response + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ipv6 {
    Use,
    /// As on a machine without IPv6 (--no-ipv6).
    Off,
    /// As on a system that keeps IPv6 sockets IPv6-only (tests).
    V6Only,
}

/// Binds `listen` (host:port; "[::]:PORT" as in the module doc). Returns the listeners, each with
/// what it takes, for the log.
pub fn bind(listen: &str, ipv6: Ipv6) -> io::Result<Vec<(TcpListener, String)>> {
    let Some(port) = listen.strip_prefix("[::]:") else {
        let l = TcpListener::bind(listen)?;
        let what = l.local_addr().map_or(listen.to_string(), |a| a.to_string());
        return Ok(vec![(l, what)]);
    };
    let port: u16 = port.parse().map_err(|_| io::Error::new(ErrorKind::InvalidInput, format!("no port in {listen}")))?;
    let v6 = match ipv6 {
        Ipv6::Off => Err(io::Error::new(ErrorKind::Unsupported, "turned off")),
        _ => bind_v6_any(port, ipv6 == Ipv6::V6Only),
    };
    match v6 {
        Ok((l6, true)) => {
            let p = l6.local_addr()?.port();
            Ok(vec![(l6, format!("[::]:{p} (IPv6 and IPv4)"))])
        }
        Ok((l6, false)) => {
            let p = l6.local_addr()?.port();
            let l4 = TcpListener::bind(("0.0.0.0", p))?;
            Ok(vec![(l6, format!("[::]:{p} (IPv6 only)")), (l4, format!("0.0.0.0:{p} (IPv4)"))])
        }
        // Another process has IPv6 on the port: IPv4 alone would leave IPv6 clients reaching it.
        Err(e) if e.kind() == ErrorKind::AddrInUse => Err(e),
        Err(e) => {
            let l4 = TcpListener::bind(("0.0.0.0", port))?;
            let p = l4.local_addr()?.port();
            Ok(vec![(l4, format!("0.0.0.0:{p} (IPv4 only: IPv6 {e})"))])
        }
    }
}

/// A listening IPv6 socket on [::]:port that also takes IPv4 (IPV6_V6ONLY off, unless `v6only`),
/// and whether it does.
fn bind_v6_any(port: u16, v6only: bool) -> io::Result<(TcpListener, bool)> {
    let int = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    // SAFETY: plain socket calls on a descriptor this function owns until the TcpListener does.
    unsafe {
        let fd = libc::socket(libc::AF_INET6, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0);
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let fail = |fd: libc::c_int| {
            let e = io::Error::last_os_error();
            libc::close(fd);
            Err(e)
        };
        let set = |level: libc::c_int, opt: libc::c_int, val: libc::c_int| {
            libc::setsockopt(fd, level, opt, (&val as *const libc::c_int).cast(), int)
        };
        set(libc::SOL_SOCKET, libc::SO_REUSEADDR, 1);
        // If the system refuses, what it gives is read back below.
        set(libc::IPPROTO_IPV6, libc::IPV6_V6ONLY, v6only as libc::c_int);
        let mut addr: libc::sockaddr_in6 = std::mem::zeroed();
        addr.sin6_family = libc::AF_INET6 as libc::sa_family_t;
        addr.sin6_port = port.to_be();
        let len = std::mem::size_of::<libc::sockaddr_in6>() as libc::socklen_t;
        if libc::bind(fd, (&addr as *const libc::sockaddr_in6).cast(), len) != 0 || libc::listen(fd, 128) != 0 {
            return fail(fd);
        }
        let (mut only, mut olen): (libc::c_int, libc::socklen_t) = (1, int);
        let read = libc::getsockopt(fd, libc::IPPROTO_IPV6, libc::IPV6_V6ONLY, (&mut only as *mut libc::c_int).cast(), &mut olen);
        Ok((TcpListener::from_raw_fd(fd), read == 0 && only == 0))
    }
}

#[derive(Default)]
struct Slots {
    total: usize,
    per_ip: HashMap<IpAddr, usize>,
}

/// One connection's place in the limits, given back when it ends.
struct Slot {
    slots: Arc<Mutex<Slots>>,
    ip: IpAddr,
}

impl Slot {
    fn take(slots: &Arc<Mutex<Slots>>, ip: IpAddr, limits: &Limits) -> Option<Slot> {
        let mut s = slots.lock().unwrap_or_else(|e| e.into_inner());
        let mine = s.per_ip.get(&ip).copied().unwrap_or(0);
        if s.total >= limits.connections || mine >= limits.per_client {
            return None;
        }
        s.total += 1;
        s.per_ip.insert(ip, mine + 1);
        Some(Slot { slots: Arc::clone(slots), ip })
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let mut s = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        s.total -= 1;
        if let Some(n) = s.per_ip.get_mut(&self.ip) {
            *n -= 1;
            if *n == 0 {
                s.per_ip.remove(&self.ip);
            }
        }
    }
}

/// Serves the listeners (each on its own accept thread) until the process ends.
pub fn serve(listeners: Vec<TcpListener>, limits: Limits, handler: Handler) {
    let slots = Arc::new(Mutex::new(Slots::default()));
    let threads: Vec<_> = listeners
        .into_iter()
        .map(|l| {
            let (slots, limits, handler) = (Arc::clone(&slots), limits.clone(), Arc::clone(&handler));
            std::thread::Builder::new().name("http-accept".into()).spawn(move || accept_loop(&l, &slots, &limits, &handler)).expect("accept thread")
        })
        .collect();
    for t in threads {
        let _ = t.join();
    }
}

fn accept_loop(listener: &TcpListener, slots: &Arc<Mutex<Slots>>, limits: &Limits, handler: &Handler) {
    let mut logged: Option<Instant> = None;
    loop {
        let (stream, peer) = match listener.accept() {
            Ok(c) => c,
            Err(e) => {
                // E.g. out of file descriptors: retry shortly (logged at most every 10 s).
                if logged.is_none_or(|t| t.elapsed() > Duration::from_secs(10)) {
                    eprintln!("http: accept failed: {e}");
                    logged = Some(Instant::now());
                }
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        let Some(slot) = Slot::take(slots, peer.ip().to_canonical(), limits) else {
            refuse(stream);
            continue;
        };
        let (limits, handler) = (limits.clone(), Arc::clone(handler));
        // If the thread can't start, the closure (slot and stream) is dropped: the connection closes.
        let _ = std::thread::Builder::new().name("http".into()).stack_size(512 * 1024).spawn(move || {
            let _slot = slot;
            serve_connection(stream, peer, &limits, &handler);
        });
    }
}

/// Over the limits: a short 503 if the socket takes it at once, and close.
fn refuse(mut stream: TcpStream) {
    let _ = stream.set_nonblocking(true);
    let _ = stream.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 5\r\nConnection: close\r\n\r\nbusy\n");
}

enum Fail {
    /// Closed, or never sent anything: nothing to answer.
    Gone,
    Status(u16, &'static str),
}

fn serve_connection(mut stream: TcpStream, peer: SocketAddr, limits: &Limits, handler: &Handler) {
    let _ = stream.set_nodelay(true);
    let (response, head_only) = match read_request(&mut stream, peer, limits) {
        Ok(rq) => {
            let head = rq.method == "HEAD";
            (handler(rq), head)
        }
        Err(Fail::Gone) => return,
        Err(Fail::Status(code, why)) => (Response::text(code, why), false),
    };
    if write_all_by(&mut stream, &serialize(&response, head_only), Instant::now() + limits.write_time).is_err() {
        return;
    }
    // Closing with input unread would reset the connection, and the client could lose the answer:
    // end our side, then read what is left (for a second at most).
    let _ = stream.shutdown(Shutdown::Write);
    let until = Instant::now() + Duration::from_secs(1);
    let mut sink = [0u8; 4096];
    while let Some(left) = until.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) {
        let _ = stream.set_read_timeout(Some(left));
        match stream.read(&mut sink) {
            Ok(n) if n > 0 => {}
            _ => break,
        }
    }
}

fn bad() -> Fail {
    Fail::Status(400, "bad request")
}

fn read_request(stream: &mut TcpStream, peer: SocketAddr, limits: &Limits) -> Result<Request, Fail> {
    let deadline = Instant::now() + limits.head_time;
    let mut buf = Vec::with_capacity(2048);
    let end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        if buf.len() > limits.head_bytes {
            return Err(Fail::Status(431, "request headers too large"));
        }
        match read_some(stream, &mut buf, deadline, 4096) {
            Ok(()) => {}
            Err(Fail::Status(408, _)) if buf.is_empty() => return Err(Fail::Gone),
            Err(e) => return Err(e),
        }
    };
    if end > limits.head_bytes {
        return Err(Fail::Status(431, "request headers too large"));
    }
    let head = std::str::from_utf8(&buf[..end]).map_err(|_| bad())?;
    let mut lines = head.split("\r\n");
    let mut parts = lines.next().unwrap_or("").split(' ');
    let (Some(method), Some(url), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else {
        return Err(bad());
    };
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) || !url.starts_with('/') || !matches!(version, "HTTP/1.1" | "HTTP/1.0") {
        return Err(bad());
    }
    let mut headers = Vec::new();
    for line in lines {
        let (name, value) = line.split_once(':').ok_or_else(bad)?;
        // Also refuses folded header lines (they start with a space).
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(bad());
        }
        headers.push((name.to_string(), value.trim().to_string()));
    }
    let header = |n: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(n)).map(|(_, v)| v.as_str());
    if header("Transfer-Encoding").is_some() {
        return Err(Fail::Status(501, "chunked request bodies are not supported"));
    }
    let len = match header("Content-Length") {
        None => 0,
        Some(v) => v.parse::<usize>().map_err(|_| bad())?,
    };
    if len > limits.body_bytes {
        return Err(Fail::Status(413, "request body too large"));
    }
    let mut body = buf[end + 4..].to_vec();
    let deadline = Instant::now() + limits.body_time;
    while body.len() < len {
        let want = len - body.len();
        read_some(stream, &mut body, deadline, want)?;
    }
    body.truncate(len);
    Ok(Request { method: method.to_string(), url: url.to_string(), headers, body, peer })
}

/// One read of at most `max` bytes, by the deadline.
fn read_some(stream: &mut TcpStream, buf: &mut Vec<u8>, deadline: Instant, max: usize) -> Result<(), Fail> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(Fail::Status(408, "request timeout"));
    }
    stream.set_read_timeout(Some(left)).map_err(|_| Fail::Gone)?;
    let mut chunk = [0u8; 4096];
    let want = max.clamp(1, chunk.len());
    match stream.read(&mut chunk[..want]) {
        Ok(0) => Err(Fail::Gone),
        Ok(n) => {
            buf.extend_from_slice(&chunk[..n]);
            Ok(())
        }
        Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => Err(Fail::Status(408, "request timeout")),
        Err(e) if e.kind() == ErrorKind::Interrupted => Ok(()),
        Err(_) => Err(Fail::Gone),
    }
}

fn write_all_by(stream: &mut TcpStream, mut data: &[u8], deadline: Instant) -> io::Result<()> {
    while !data.is_empty() {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(ErrorKind::TimedOut.into());
        }
        stream.set_write_timeout(Some(left))?;
        match stream.write(data) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => data = &data[n..],
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn serialize(r: &Response, head_only: bool) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {} {}\r\n", r.status, reason(r.status));
    for (name, value) in &r.headers {
        // Never a header that could split the response.
        if !name.contains(['\r', '\n']) && !value.contains(['\r', '\n']) {
            out.push_str(&format!("{name}: {value}\r\n"));
        }
    }
    out.push_str(&format!("Content-Length: {}\r\nConnection: close\r\n\r\n", r.body.len()));
    let mut bytes = out.into_bytes();
    if !head_only {
        bytes.extend_from_slice(&r.body);
    }
    bytes
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        303 => "See Other",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        413 => "Content Too Large",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick(per_client: usize) -> Limits {
        let t = Duration::from_millis(600);
        Limits { head_time: t, body_time: t, write_time: t, head_bytes: 1024, body_bytes: 1024, connections: 64, per_client }
    }

    fn start(listeners: Vec<TcpListener>, limits: Limits) {
        let handler: Handler = Arc::new(|rq: Request| {
            if rq.url == "/big" {
                return Response { status: 200, headers: Vec::new(), body: vec![b'x'; 8 << 20] };
            }
            Response::text(200, &format!("{} {} {}", rq.method, rq.url, rq.body.len()))
        });
        std::thread::spawn(move || serve(listeners, limits, handler));
    }

    fn server(limits: Limits) -> SocketAddr {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        start(vec![l], limits);
        addr
    }

    fn ask(addr: SocketAddr, raw: &[u8]) -> String {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let _ = s.write_all(raw);
        let mut out = Vec::new();
        let _ = s.read_to_end(&mut out);
        String::from_utf8_lossy(&out).into_owned()
    }

    /// The server's answer on a connection that is already open (or "" if it closed without one).
    fn answer(mut s: TcpStream) -> String {
        s.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut out = Vec::new();
        let _ = s.read_to_end(&mut out);
        String::from_utf8_lossy(&out).into_owned()
    }

    #[test]
    fn answers_one_request_and_closes() {
        let addr = server(quick(8));
        let r = ask(addr, b"POST /x?y HTTP/1.1\r\nHost: h\r\nContent-Length: 3\r\n\r\nabc");
        assert!(r.starts_with("HTTP/1.1 200 OK\r\n") && r.contains("\r\nConnection: close\r\n") && r.ends_with("POST /x?y 3"), "{r}");
        let r = ask(addr, b"HEAD / HTTP/1.1\r\n\r\n");
        assert!(r.starts_with("HTTP/1.1 200 OK\r\n") && r.contains("Content-Length: 8\r\n") && r.ends_with("\r\n\r\n"), "{r}");
    }

    #[test]
    fn refuses_what_is_too_big_or_malformed() {
        let addr = server(quick(8));
        let big_head = [b"GET / HTTP/1.1\r\nX: ".as_slice(), &[b'a'; 2000], b"\r\n\r\n"].concat();
        assert!(ask(addr, &big_head).starts_with("HTTP/1.1 431 "));
        assert!(ask(addr, b"POST / HTTP/1.1\r\nContent-Length: 5000\r\n\r\n").starts_with("HTTP/1.1 413 "));
        assert!(ask(addr, b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n").starts_with("HTTP/1.1 501 "));
        assert!(ask(addr, b"hello\r\n\r\n").starts_with("HTTP/1.1 400 "));
        assert!(ask(addr, b"GET / HTTP/1.1\r\n folded: x\r\n\r\n").starts_with("HTTP/1.1 400 "));
    }

    #[test]
    fn stalled_clients_hold_up_nobody_and_go_at_their_deadlines() {
        let addr = server(quick(16));
        let open = |raw: &[u8]| {
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(raw).unwrap();
            s
        };
        let silent: Vec<_> = (0..4).map(|_| open(b"")).collect();
        let slowloris: Vec<_> = (0..3).map(|_| open(b"GET / HTTP/1.1\r\nX-Slow: ")).collect();
        let slow_body: Vec<_> = (0..3).map(|_| open(b"POST / HTTP/1.1\r\nContent-Length: 100\r\n\r\nab")).collect();
        let mut no_reader = TcpStream::connect(addr).unwrap();
        no_reader.write_all(b"GET /big HTTP/1.1\r\n\r\n").unwrap();
        std::thread::sleep(Duration::from_millis(100));
        let t0 = Instant::now();
        assert!(ask(addr, b"GET /now HTTP/1.1\r\n\r\n").ends_with("GET /now 0"));
        assert!(t0.elapsed() < Duration::from_millis(300), "answered in {:?}", t0.elapsed());
        for s in silent {
            assert_eq!(answer(s), "", "closed without an answer");
        }
        for s in slowloris.into_iter().chain(slow_body) {
            assert!(answer(s).starts_with("HTTP/1.1 408 "));
        }
        drop(no_reader);
    }

    #[test]
    fn one_address_gets_at_most_per_client_connections() {
        let Ok(l6) = TcpListener::bind("[::1]:0") else { return }; // no IPv6 here
        let addr6 = l6.local_addr().unwrap();
        let l4 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr4 = l4.local_addr().unwrap();
        start(vec![l4, l6], quick(2));
        let held: Vec<_> = (0..2).map(|_| TcpStream::connect(addr4).unwrap()).collect();
        std::thread::sleep(Duration::from_millis(100));
        assert!(answer(TcpStream::connect(addr4).unwrap()).starts_with("HTTP/1.1 503 "), "a third from 127.0.0.1");
        assert!(ask(addr6, b"GET / HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 200 "), "another address");
        drop(held);
    }

    fn reachable(addr: SocketAddr) -> bool {
        TcpStream::connect_timeout(&addr, Duration::from_secs(1)).is_ok()
    }

    #[test]
    fn listens_on_ipv6_and_ipv4() {
        let ipv6_here = TcpListener::bind("[::1]:0").is_ok();
        // As on a machine without IPv6: IPv4 only.
        let ls = bind("[::]:0", Ipv6::Off).unwrap();
        assert_eq!(ls.len(), 1);
        assert!(ls[0].0.local_addr().unwrap().is_ipv4() && ls[0].1.contains("IPv4 only"), "{}", ls[0].1);
        if !ipv6_here {
            return;
        }
        // One socket for both.
        let ls = bind("[::]:0", Ipv6::Use).unwrap();
        let port = ls[0].0.local_addr().unwrap().port();
        assert_eq!((ls.len(), ls[0].1.as_str()), (1, format!("[::]:{port} (IPv6 and IPv4)").as_str()));
        assert!(reachable(SocketAddr::from(([127, 0, 0, 1], port))) && reachable(format!("[::1]:{port}").parse().unwrap()));
        // As with net.ipv6.bindv6only=1: IPv4 on a socket of its own, same port.
        let ls = bind("[::]:0", Ipv6::V6Only).unwrap();
        let port = ls[0].0.local_addr().unwrap().port();
        assert_eq!(ls.len(), 2, "{:?}", ls.iter().map(|l| &l.1).collect::<Vec<_>>());
        assert_eq!(ls[1].0.local_addr().unwrap(), SocketAddr::from(([0, 0, 0, 0], port)));
        assert!(reachable(SocketAddr::from(([127, 0, 0, 1], port))) && reachable(format!("[::1]:{port}").parse().unwrap()));
        // IPv6-only, and another process has IPv4 on the port: an error, not IPv6 only.
        let other = TcpListener::bind("0.0.0.0:0").unwrap();
        let port = other.local_addr().unwrap().port();
        assert_eq!(bind(&format!("[::]:{port}"), Ipv6::V6Only).map(|_| ()).unwrap_err().kind(), ErrorKind::AddrInUse);
        // Another process has IPv6 on the port: an error, not IPv4 only.
        let (other6, _) = bind_v6_any(0, true).unwrap();
        let port = other6.local_addr().unwrap().port();
        for ipv6 in [Ipv6::Use, Ipv6::V6Only] {
            assert_eq!(bind(&format!("[::]:{port}"), ipv6).map(|_| ()).unwrap_err().kind(), ErrorKind::AddrInUse);
        }
    }
}
