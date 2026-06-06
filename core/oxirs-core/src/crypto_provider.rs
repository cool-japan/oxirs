//! Process-default rustls [`CryptoProvider`](rustls::crypto::CryptoProvider) installation.
//!
//! OxiRS uses `reqwest` configured with the `rustls-no-provider` feature so that
//! the Pure Rust TLS stack is selected without pulling in `aws-lc-sys`/`ring`.
//! That feature means `reqwest` (and `rustls` in general) relies on the
//! **process-default** [`CryptoProvider`], which must be installed at runtime
//! before any TLS client is constructed.
//!
//! Application binaries normally install it in `main()`, but unit/integration
//! test harnesses never run `main()`. Because `oxirs-core` is a transitive
//! dependency of every OxiRS crate, installing the provider from here — via a
//! pre-`main` constructor — guarantees the Pure Rust provider is the process
//! default in **all** binaries and **all** test processes alike.

/// Install OxiTLS's Pure Rust rustls [`CryptoProvider`](rustls::crypto::CryptoProvider)
/// as the process default.
///
/// This is idempotent: the underlying
/// [`install_default`](rustls::crypto::CryptoProvider::install_default) returns
/// an `Err` if a provider is already installed, which is intentionally ignored.
/// Safe to call any number of times from any thread.
pub fn ensure_crypto_provider() {
    let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());
}

/// Pre-`main` constructor that installs the Pure Rust crypto provider as soon as
/// this crate is linked into a process (including test harnesses).
///
/// `ctor` schedules this in a pre-`main` linker section, which is inherently
/// `unsafe` (no ordering guarantees relative to other constructors). All this
/// body does is populate a `OnceLock` inside `rustls`, which is sound to run at
/// that point. The `unsafe` marker is mandated by `ctor` 1.x's macro contract.
#[allow(unsafe_code)]
#[ctor::ctor(unsafe)]
fn auto_install_pure_crypto_provider() {
    ensure_crypto_provider();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_crypto_provider_is_idempotent() {
        // The ctor already installed a provider before this test ran; calling
        // again must not panic and must leave a provider in place.
        ensure_crypto_provider();
        ensure_crypto_provider();
        assert!(
            rustls::crypto::CryptoProvider::get_default().is_some(),
            "a process-default rustls CryptoProvider must be installed"
        );
    }
}
