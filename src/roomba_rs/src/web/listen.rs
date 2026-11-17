//! web_server's listening sockets, IPv6 and IPv4 on one port: [::] takes both where the system
//! allows it (IPV6_V6ONLY off). Where it keeps IPv6 sockets IPv6-only (net.ipv6.bindv6only=1),
//! 0.0.0.0 needs a socket of its own, and if that can't bind (another process has IPv4 on the
//! port) it is an error, not a server that answers only IPv6. Without IPv6, 0.0.0.0 only; but
//! another process on [::] is an error too, not a server that answers only IPv4.

use std::io;
use std::net::TcpListener;
use std::os::fd::FromRawFd;

/// The listeners for `port` and what they take, for the log. `v6only`: None leaves the IPv6 socket
/// as the system makes it; Some sets it (tests).
pub fn bind_dual(port: u16, v6only: Option<bool>) -> io::Result<(Vec<TcpListener>, String)> {
    match bind_v6(port, v6only) {
        Ok((l6, false)) => Ok((vec![l6], "[::] (IPv6 and IPv4)".into())),
        Ok((l6, true)) => {
            let p = l6.local_addr()?.port();
            let l4 = TcpListener::bind(("0.0.0.0", p))?;
            Ok((vec![l6, l4], "[::] (IPv6) and 0.0.0.0 (IPv4)".into()))
        }
        Err(e) if e.kind() == io::ErrorKind::AddrInUse => Err(e),
        Err(e) => {
            let l4 = TcpListener::bind(("0.0.0.0", port))?;
            Ok((vec![l4], format!("0.0.0.0 (IPv4 only: IPv6 {e})")))
        }
    }
}

/// A listening socket on [::]:port, and whether it is IPv6-only.
fn bind_v6(port: u16, v6only: Option<bool>) -> io::Result<(TcpListener, bool)> {
    let int = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    // SAFETY: plain socket calls on a descriptor this function owns until the TcpListener does.
    unsafe {
        let fd = libc::socket(libc::AF_INET6, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0);
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let set = |level: libc::c_int, opt: libc::c_int, val: libc::c_int| {
            libc::setsockopt(fd, level, opt, (&val as *const libc::c_int).cast(), int)
        };
        set(libc::SOL_SOCKET, libc::SO_REUSEADDR, 1);
        if let Some(only) = v6only {
            set(libc::IPPROTO_IPV6, libc::IPV6_V6ONLY, only as libc::c_int);
        }
        let mut addr: libc::sockaddr_in6 = std::mem::zeroed();
        addr.sin6_family = libc::AF_INET6 as libc::sa_family_t;
        addr.sin6_port = port.to_be();
        let len = std::mem::size_of::<libc::sockaddr_in6>() as libc::socklen_t;
        if libc::bind(fd, (&addr as *const libc::sockaddr_in6).cast(), len) != 0 || libc::listen(fd, 128) != 0 {
            let e = io::Error::last_os_error();
            libc::close(fd);
            return Err(e);
        }
        let (mut only, mut olen): (libc::c_int, libc::socklen_t) = (1, int);
        let read = libc::getsockopt(fd, libc::IPPROTO_IPV6, libc::IPV6_V6ONLY, (&mut only as *mut libc::c_int).cast(), &mut olen);
        Ok((TcpListener::from_raw_fd(fd), read != 0 || only != 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{SocketAddr, TcpStream};
    use std::time::Duration;

    fn ipv6_here() -> bool {
        TcpListener::bind("[::1]:0").is_ok()
    }

    fn reachable(addr: &str) -> bool {
        TcpStream::connect_timeout(&addr.parse::<SocketAddr>().unwrap(), Duration::from_secs(1)).is_ok()
    }

    #[test]
    fn listens_on_ipv6_and_ipv4() {
        if !ipv6_here() {
            return;
        }
        for v6only in [None, Some(false), Some(true)] {
            let (ls, what) = bind_dual(0, v6only).unwrap();
            let port = ls[0].local_addr().unwrap().port();
            // With the system default (None) the count depends on net.ipv6.bindv6only (2 listeners
            // when it is 1), so only an explicit setting fixes it; reachability holds either way.
            if v6only.is_some() {
                assert_eq!(ls.len(), if v6only == Some(true) { 2 } else { 1 }, "{what}");
            }
            assert!(reachable(&format!("127.0.0.1:{port}")) && reachable(&format!("[::1]:{port}")), "{what}");
        }
    }

    #[test]
    fn ipv6_only_with_ipv4_taken_by_another_process_is_an_error() {
        if !ipv6_here() {
            return;
        }
        let other = TcpListener::bind("0.0.0.0:0").unwrap();
        let port = other.local_addr().unwrap().port();
        match bind_dual(port, Some(true)) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::AddrInUse),
            Ok((_, what)) => panic!("listening on {what}, but IPv4 goes to the other process"),
        }
    }

    #[test]
    fn ipv6_taken_by_another_process_is_an_error() {
        if !ipv6_here() {
            return;
        }
        let (other, _) = bind_v6(0, Some(true)).unwrap();
        let port = other.local_addr().unwrap().port();
        for v6only in [None, Some(false), Some(true)] {
            match bind_dual(port, v6only) {
                Err(e) => assert_eq!(e.kind(), io::ErrorKind::AddrInUse),
                Ok((_, what)) => panic!("listening on {what}, but IPv6 goes to the other process"),
            }
        }
    }
}
