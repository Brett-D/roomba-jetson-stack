//! roomba_web's server (web_server) and password tool in Rust: Python-compatible JSON (`json`),
//! the password file and sessions (`auth`), aiohttp's HTTP behaviour (`http`), the camera preview
//! and map images (`image`), the node's state, snapshot and request validation (`state`), and
//! its listening sockets, IPv6 and IPv4 (`listen`).
pub mod auth;
pub mod http;
pub mod image;
pub mod json;
pub mod listen;
pub mod state;
