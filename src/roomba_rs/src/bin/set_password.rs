//! set_password (Rust): sets the web app password like `ros2 run roomba_web set_password` - the
//! same prompts, checks and ~/.config/roomba_web/auth.json (scrypt hash, mode 600; see
//! roomba_rs::web::auth). Reads from the terminal without echo, like Python's getpass.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::os::fd::AsRawFd;

use roomba_rs::web::auth::{set_password, AuthPaths};

/// getpass.getpass(prompt): prompt and read on /dev/tty with echo off; without a terminal, warn
/// and read a line from stdin.
fn getpass(prompt: &str) -> std::io::Result<String> {
    let mut line = String::new();
    match OpenOptions::new().read(true).write(true).open("/dev/tty") {
        Ok(mut tty) => {
            let fd = tty.as_raw_fd();
            // SAFETY: termios is plain data filled by tcgetattr on a valid descriptor.
            let mut old: libc::termios = unsafe { std::mem::zeroed() };
            let have_termios = unsafe { libc::tcgetattr(fd, &mut old) } == 0;
            if have_termios {
                let mut quiet = old;
                quiet.c_lflag &= !libc::ECHO;
                // SAFETY: sets attributes on the terminal we opened, restored below.
                unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &quiet) };
            }
            tty.write_all(prompt.as_bytes())?;
            tty.flush()?;
            let result = BufReader::new(&tty).read_line(&mut line);
            if have_termios {
                // SAFETY: restores the attributes read above.
                unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &old) };
            }
            tty.write_all(b"\n")?;
            result?;
        }
        Err(_) => {
            eprintln!("Warning: Password input may be echoed.");
            eprint!("{prompt}");
            std::io::stdin().lock().read_line(&mut line)?;
        }
    }
    if line.ends_with('\n') {
        line.pop();
    }
    Ok(line)
}

fn main() {
    let fail = |msg: &str| -> ! {
        eprintln!("{msg}");
        std::process::exit(1);
    };
    let pw = getpass("New web password (min 8 characters): ").unwrap_or_else(|e| fail(&e.to_string()));
    if pw.chars().count() < 8 {
        fail("Password too short.");
    }
    if getpass("Repeat: ").unwrap_or_else(|e| fail(&e.to_string())) != pw {
        fail("Passwords do not match.");
    }
    let paths = AuthPaths::default_paths();
    set_password(&paths, &pw).unwrap_or_else(|e| fail(&format!("Cannot write {}: {e}", paths.auth_file.display())));
    if paths.initial_file.exists() {
        let _ = std::fs::remove_file(&paths.initial_file);
    }
    println!("Password updated. Existing logins stay valid until they expire or the server restarts.");
}
