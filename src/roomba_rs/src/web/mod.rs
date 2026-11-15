//! roomba_web's server (web_server) and password tool in Rust: Python-compatible JSON (`json`),
//! the password file and sessions (`auth`), aiohttp's HTTP behaviour (`http`), the camera preview
//! and map images (`image`) and the node's state, snapshot and request validation (`state`).
pub mod auth;
pub mod http;
pub mod image;
pub mod json;
pub mod state;
