fn main() {
    // Link against libnuma on Linux
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=numa");
    }
}
