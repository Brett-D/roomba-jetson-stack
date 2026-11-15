// Compiles the web server's image encoders (src/web/codecs.c), which drive the system
// libjpeg-turbo and libpng the way OpenCV's imgcodecs does.
fn main() {
    println!("cargo:rerun-if-changed=src/web/codecs.c");
    cc::Build::new().file("src/web/codecs.c").opt_level(2).compile("roomba_web_codecs");
    println!("cargo:rustc-link-lib=jpeg");
    println!("cargo:rustc-link-lib=png16");
    println!("cargo:rustc-link-lib=z");
}
