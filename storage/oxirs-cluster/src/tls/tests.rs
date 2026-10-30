//! Tests for [`super`]: TLS peer authentication, certificate lifecycle and
//! the AES-GCM [`super::EncryptionManager`].

use super::test_support::{install_provider, node_config};
use super::*;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Run one TLS session between `server` and `client` over loopback and
/// report whether each side completed it. Each side sends one byte after
/// the handshake, so rejections that TLS 1.3 only surfaces after the
/// client's Finished (such as a refused client certificate) still count.
async fn tls_session(server: &TlsManager, client: &TlsManager, server_name: &str) -> (bool, bool) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let acceptor = server.get_acceptor().await.unwrap();
    let server_side = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let session = async {
            let mut tls = acceptor.accept(stream).await.ok()?;
            let mut byte = [0u8; 1];
            tls.read_exact(&mut byte).await.ok()?;
            tls.write_all(b"s").await.ok()?;
            tls.flush().await.ok()
        };
        matches!(
            tokio::time::timeout(Duration::from_secs(10), session).await,
            Ok(Some(()))
        )
    });

    let connector = client.get_connector().await.unwrap();
    let name = rustls::pki_types::ServerName::try_from(server_name.to_string()).unwrap();
    let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    let session = async {
        let mut tls = connector.connect(name, tcp).await.ok()?;
        tls.write_all(b"c").await.ok()?;
        tls.flush().await.ok()?;
        let mut byte = [0u8; 1];
        tls.read_exact(&mut byte).await.ok()
    };
    let client_ok = matches!(
        tokio::time::timeout(Duration::from_secs(10), session).await,
        Ok(Some(_))
    );
    let server_ok = server_side.await.unwrap();
    (client_ok, server_ok)
}

/// Regression test for the accept-anything client verifier (<= 0.4.1).
///
/// The client used to install a verifier that accepted every server
/// certificate and every handshake signature, so any host could pose as a
/// cluster peer. A client that trusts only node 1 must refuse node 99.
#[tokio::test]
async fn test_client_rejects_untrusted_server_certificate() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let trusted_bundle = dir.path().join("trusted.pem");
    let rogue_bundle = dir.path().join("rogue.pem");
    let (client_config, client_pem) = node_config(dir.path(), 1, &trusted_bundle, false).await;
    let (rogue_config, rogue_pem) = node_config(dir.path(), 99, &rogue_bundle, false).await;
    std::fs::write(&trusted_bundle, &client_pem).unwrap();
    std::fs::write(&rogue_bundle, &rogue_pem).unwrap();

    let client = TlsManager::new(client_config, 1);
    client.initialize().await.unwrap();
    let rogue = TlsManager::new(rogue_config, 99);
    rogue.initialize().await.unwrap();

    let (client_ok, _) = tls_session(&rogue, &client, "node-99").await;
    assert!(
        !client_ok,
        "client completed a TLS session with a server it does not trust"
    );
}

/// Two nodes that pin each other's certificates complete a mutually
/// authenticated session with the default `require_client_auth: true`
/// (which never initialized before: the client verifier had no roots).
#[tokio::test]
async fn test_mutual_tls_between_trusted_nodes() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("cluster.pem");
    let (config_1, pem_1) = node_config(dir.path(), 1, &bundle, true).await;
    let (config_2, pem_2) = node_config(dir.path(), 2, &bundle, true).await;
    std::fs::write(&bundle, format!("{pem_1}{pem_2}")).unwrap();

    let node_1 = TlsManager::new(config_1, 1);
    node_1.initialize().await.unwrap();
    let node_2 = TlsManager::new(config_2, 2);
    node_2.initialize().await.unwrap();

    assert_eq!(tls_session(&node_2, &node_1, "node-2").await, (true, true));
    assert_eq!(tls_session(&node_1, &node_2, "node-1").await, (true, true));
}

/// A trusted certificate is only accepted for the node it names.
#[tokio::test]
async fn test_client_rejects_wrong_node_identity() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("cluster.pem");
    let (config_1, pem_1) = node_config(dir.path(), 1, &bundle, false).await;
    let (config_2, pem_2) = node_config(dir.path(), 2, &bundle, false).await;
    std::fs::write(&bundle, format!("{pem_1}{pem_2}")).unwrap();

    let node_1 = TlsManager::new(config_1, 1);
    node_1.initialize().await.unwrap();
    let node_2 = TlsManager::new(config_2, 2);
    node_2.initialize().await.unwrap();

    let (client_ok, _) = tls_session(&node_2, &node_1, "node-3").await;
    assert!(!client_ok, "node 2's certificate was accepted as node 3");
}

/// Under mutual TLS the server refuses a client whose certificate it does
/// not trust, even if that client trusts the server.
#[tokio::test]
async fn test_server_rejects_untrusted_client_under_mtls() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let cluster_bundle = dir.path().join("cluster.pem");
    let rogue_bundle = dir.path().join("rogue.pem");
    let (server_config, server_pem) = node_config(dir.path(), 1, &cluster_bundle, true).await;
    let (rogue_config, rogue_pem) = node_config(dir.path(), 99, &rogue_bundle, true).await;
    std::fs::write(&cluster_bundle, &server_pem).unwrap();
    std::fs::write(&rogue_bundle, format!("{server_pem}{rogue_pem}")).unwrap();

    let server = TlsManager::new(server_config, 1);
    server.initialize().await.unwrap();
    let rogue = TlsManager::new(rogue_config, 99);
    rogue.initialize().await.unwrap();

    let (client_ok, server_ok) = tls_session(&server, &rogue, "node-1").await;
    assert!(
        !server_ok,
        "server accepted an untrusted client certificate"
    );
    assert!(!client_ok, "untrusted client completed a session");
}

/// TLS without trust anchors cannot authenticate peers, so it must not
/// start at all.
#[tokio::test]
async fn test_initialize_requires_ca_file() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let config = TlsConfig {
        enabled: true,
        require_client_auth: false,
        cert_dir: dir.path().to_path_buf(),
        auto_rotate: false,
        ..Default::default()
    };

    let result = TlsManager::new(config, 1).initialize().await;
    assert!(result.is_err(), "TLS initialized without any trust anchors");
}

/// The rotation check must never cut a node off its cluster. It used to
/// judge expiry from a fabricated date (load time + `cert_validity_days`)
/// and then replace even an operator-provided certificate with a fresh
/// self-signed one that no peer trusts.
#[tokio::test]
async fn test_rotation_check_keeps_pinned_identity() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("cluster.pem");
    let (mut config_1, pem_1) = node_config(dir.path(), 1, &bundle, true).await;
    let (config_2, pem_2) = node_config(dir.path(), 2, &bundle, true).await;
    std::fs::write(&bundle, format!("{pem_1}{pem_2}")).unwrap();
    // With the fabricated expiry, this threshold fires immediately.
    config_1.rotation_threshold_days = config_1.cert_validity_days;

    let node_1 = TlsManager::new(config_1, 1);
    node_1.initialize().await.unwrap();
    let node_2 = TlsManager::new(config_2, 2);
    node_2.initialize().await.unwrap();

    node_1.check_and_rotate_certificates().await.unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.path().join("node-1.crt")).unwrap(),
        pem_1,
        "the operator-provided certificate was replaced"
    );
    assert_eq!(tls_session(&node_2, &node_1, "node-2").await, (true, true));
    assert_eq!(tls_session(&node_1, &node_2, "node-1").await, (true, true));
}

/// Initialize node 1 with an auto-generated identity in `dir` that trusts
/// only itself, returning the manager and its certificate PEM.
async fn auto_identity_manager(dir: &Path, rotation_threshold_days: u64) -> (TlsManager, String) {
    let mut config = TlsConfig {
        enabled: true,
        require_client_auth: true,
        cert_dir: dir.to_path_buf(),
        auto_rotate: false,
        rotation_threshold_days,
        ..Default::default()
    };
    // The first start generates the certificate, then stops for lack of trust anchors.
    assert!(TlsManager::new(config.clone(), 1)
        .initialize()
        .await
        .is_err());
    let pem = std::fs::read_to_string(dir.join("node-1.crt")).unwrap();
    let bundle = dir.join("cluster.pem");
    std::fs::write(&bundle, &pem).unwrap();
    config.ca_file = Some(bundle);

    let manager = TlsManager::new(config, 1);
    manager.initialize().await.unwrap();
    (manager, pem)
}

/// Rotation used to write raw DER into the PEM files, so reloading the TLS
/// configuration failed right after rotating.
#[tokio::test]
async fn test_rotation_writes_pem_and_reloads() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (manager, pem) = auto_identity_manager(dir.path(), 30).await;

    manager.renew_certificate().await.unwrap();

    let cert = std::fs::read_to_string(dir.path().join("node-1.crt")).unwrap();
    let key = std::fs::read_to_string(dir.path().join("node-1.key")).unwrap();
    assert!(cert.starts_with("-----BEGIN CERTIFICATE-----"));
    assert!(key.starts_with("-----BEGIN PRIVATE KEY-----"));
    assert_ne!(cert, pem, "rotation kept the old certificate");
    assert!(manager.get_acceptor().await.is_ok());
    assert!(manager.get_connector().await.is_ok());
}

/// The rotation check reads the certificate's own expiry (it used to judge
/// from a fabricated one), so a certificate far from expiry is left alone.
#[tokio::test]
async fn test_rotation_check_uses_real_expiry() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (manager, pem) = auto_identity_manager(dir.path(), 30).await;

    manager.check_and_rotate_certificates().await.unwrap();

    let cert = std::fs::read_to_string(dir.path().join("node-1.crt")).unwrap();
    assert_eq!(cert, pem, "a certificate far from expiry was rotated");
    let info = manager.get_certificate_info("server").await.unwrap();
    let validity = Duration::from_secs(TlsConfig::default().cert_validity_days * 24 * 3600);
    let expected = SystemTime::now() + validity;
    let skew = Duration::from_secs(2 * 3600);
    assert!(info.not_after > expected - skew && info.not_after < expected + skew);
}

/// Missing operator-provided files are an error, not a cue to generate.
#[tokio::test]
async fn test_missing_operator_certificate_is_not_generated() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let cert_file = dir.path().join("operator.crt");
    let config = TlsConfig {
        enabled: true,
        cert_file: Some(cert_file.clone()),
        key_file: Some(dir.path().join("operator.key")),
        ca_file: Some(dir.path().join("cluster.pem")),
        cert_dir: dir.path().to_path_buf(),
        auto_rotate: false,
        ..Default::default()
    };

    assert!(TlsManager::new(config, 1).initialize().await.is_err());
    assert!(
        !cert_file.exists(),
        "a certificate was generated at the operator's path"
    );
}

/// `validate_peer_certificate` used to return `Ok(true)` for anything.
#[tokio::test]
async fn test_validate_peer_certificate() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("cluster.pem");
    let (config_1, pem_1) = node_config(dir.path(), 1, &bundle, true).await;
    let (_, pem_2) = node_config(dir.path(), 2, &bundle, true).await;
    let (_, rogue_pem) = node_config(dir.path(), 99, &bundle, true).await;
    std::fs::write(&bundle, format!("{pem_1}{pem_2}")).unwrap();
    let manager = TlsManager::new(config_1, 1);

    let der = |pem: &str| {
        rustls_pemfile::certs(&mut pem.as_bytes())
            .next()
            .unwrap()
            .unwrap()
            .to_vec()
    };
    assert!(manager
        .validate_peer_certificate(pem_2.as_bytes())
        .await
        .unwrap());
    assert!(manager
        .validate_peer_certificate(&der(&pem_2))
        .await
        .unwrap());
    assert!(!manager
        .validate_peer_certificate(rogue_pem.as_bytes())
        .await
        .unwrap());
    assert!(!manager
        .validate_peer_certificate(&der(&rogue_pem))
        .await
        .unwrap());

    let unanchored = TlsManager::new(
        TlsConfig {
            cert_dir: dir.path().to_path_buf(),
            ..Default::default()
        },
        3,
    );
    assert!(unanchored
        .validate_peer_certificate(pem_2.as_bytes())
        .await
        .is_err());
}

#[tokio::test]
async fn test_tls_manager_initialization() {
    install_provider();

    let temp_dir = TempDir::new().unwrap();
    let bundle = temp_dir.path().join("cluster.pem");
    let mut config = TlsConfig {
        enabled: true,
        require_client_auth: false,
        cert_dir: temp_dir.path().to_path_buf(),
        auto_rotate: false,
        ..Default::default()
    };

    // First start: the node certificate is generated, then initialization
    // stops because nothing says which peers to trust yet.
    assert!(TlsManager::new(config.clone(), 1)
        .initialize()
        .await
        .is_err());
    std::fs::copy(temp_dir.path().join("node-1.crt"), &bundle).unwrap();
    config.ca_file = Some(bundle);

    let tls_manager = TlsManager::new(config, 1);
    let result = tls_manager.initialize().await;
    if let Err(e) = &result {
        eprintln!("TLS manager initialization failed: {e}");
    }
    assert!(result.is_ok());

    let acceptor = tls_manager.get_acceptor().await;
    assert!(acceptor.is_ok());

    let connector = tls_manager.get_connector().await;
    assert!(connector.is_ok());
}

#[tokio::test]
async fn test_certificate_generation() {
    // Install the Pure Rust crypto provider for rustls (idempotent).
    let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());

    let temp_dir = TempDir::new().unwrap();
    let config = TlsConfig {
        enabled: true,
        cert_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let tls_manager = TlsManager::new(config, 1);
    let (cert_der, key_der) = tls_manager.generate_certificate(1).await.unwrap();

    assert!(!cert_der.is_empty());
    assert!(!key_der.is_empty());
}

#[tokio::test]
async fn test_encryption_manager() {
    // Install the Pure Rust crypto provider for rustls (idempotent).
    let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());

    let encryption_manager = EncryptionManager::new().unwrap();
    let data = b"Hello, World!";

    let encrypted = encryption_manager.encrypt(data).await.unwrap();
    assert_ne!(encrypted, data);

    let decrypted = encryption_manager.decrypt(&encrypted).await.unwrap();
    assert_eq!(decrypted, data);
}

#[test]
fn test_encryption_manager_rotate_key() {
    let mut encryption_manager = EncryptionManager::new().unwrap();
    let old_key = encryption_manager.export_key();

    encryption_manager.rotate_key().unwrap();

    assert_ne!(encryption_manager.export_key(), old_key);
}

/// Regression test for AES-GCM nonce reuse (<= 0.4.1).
///
/// Nonces used to be a per-instance counter starting at 1, so two managers
/// built from the same key (`with_key`) produced identical nonces — fatal
/// for GCM, which leaks the XOR of the plaintexts and the GHASH key.
#[tokio::test]
async fn test_with_key_instances_never_reuse_nonces() {
    let key = [7u8; 32];
    let first = EncryptionManager::with_key(key);
    let second = EncryptionManager::with_key(key);

    let a = first.encrypt(b"same plaintext").await.unwrap();
    let b = second.encrypt(b"same plaintext").await.unwrap();

    assert_ne!(
        a[..12],
        b[..12],
        "two managers sharing a key emitted the same AES-GCM nonce"
    );
}

/// Restoring a key exported "for backup" must not replay earlier nonces.
#[tokio::test]
async fn test_reimported_key_does_not_repeat_nonces() {
    let original = EncryptionManager::new().unwrap();
    let mut seen = std::collections::HashSet::new();
    for _ in 0..16 {
        let encrypted = original.encrypt(b"payload").await.unwrap();
        seen.insert(encrypted[..12].to_vec());
    }

    let restored = EncryptionManager::with_key(original.export_key());
    for _ in 0..16 {
        let encrypted = restored.encrypt(b"payload").await.unwrap();
        assert!(
            seen.insert(encrypted[..12].to_vec()),
            "restored manager repeated a nonce of the original"
        );
    }
}

/// Ciphertexts written with the old counter nonces must still decrypt.
#[tokio::test]
async fn test_decrypts_legacy_counter_nonce_ciphertext() {
    use aes_gcm::aead::Aead;
    use aes_gcm::{Aes256Gcm, KeyInit, Nonce};

    let key = [9u8; 32];
    let mut nonce = [0u8; 12];
    nonce[..8].copy_from_slice(&1u64.to_le_bytes());
    let cipher = Aes256Gcm::new_from_slice(&key).unwrap();
    let ciphertext = cipher
        .encrypt(<&Nonce<_>>::from(&nonce), b"legacy payload".as_slice())
        .unwrap();
    let mut legacy = nonce.to_vec();
    legacy.extend_from_slice(&ciphertext);

    let manager = EncryptionManager::with_key(key);
    assert_eq!(manager.decrypt(&legacy).await.unwrap(), b"legacy payload");
}

#[tokio::test]
async fn test_key_usage_limit_requires_rotation() {
    let mut manager = EncryptionManager::new().unwrap();
    manager
        .messages_under_key
        .store(MAX_MESSAGES_PER_KEY, Ordering::Relaxed);

    assert!(manager.encrypt(b"over the limit").await.is_err());

    manager.rotate_key().unwrap();
    let encrypted = manager.encrypt(b"fresh key").await.unwrap();
    assert_eq!(manager.decrypt(&encrypted).await.unwrap(), b"fresh key");
}

#[tokio::test]
async fn test_certificate_info() {
    // Install the Pure Rust crypto provider for rustls (idempotent).
    let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());

    let cert_info = CertificateInfo {
        subject: "test".to_string(),
        issuer: "test".to_string(),
        serial_number: "1".to_string(),
        not_before: SystemTime::now(),
        not_after: SystemTime::now() + Duration::from_secs(3600),
        fingerprint: "test".to_string(),
        key_usage: vec![],
    };

    assert!(!cert_info.is_expired());
    assert!(cert_info.expires_within(Duration::from_secs(7200)));
    assert!(!cert_info.expires_within(Duration::from_secs(1800)));
}

// ---------------------------------------------------------------------------
// Settings that used to be ignored or fabricated
// ---------------------------------------------------------------------------

/// Run a TLS session like [`tls_session`] and return the cipher suite and
/// protocol version the client negotiated, or `None` if either side failed.
async fn negotiated(
    server: &TlsManager,
    client: &TlsManager,
    server_name: &str,
) -> Option<(rustls::CipherSuite, rustls::ProtocolVersion)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let acceptor = server.get_acceptor().await.unwrap();
    let server_side = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let session = async {
            let mut tls = acceptor.accept(stream).await.ok()?;
            let mut byte = [0u8; 1];
            tls.read_exact(&mut byte).await.ok()?;
            tls.write_all(b"s").await.ok()?;
            tls.flush().await.ok()
        };
        matches!(
            tokio::time::timeout(Duration::from_secs(10), session).await,
            Ok(Some(()))
        )
    });

    let connector = client.get_connector().await.unwrap();
    let name = rustls::pki_types::ServerName::try_from(server_name.to_string()).unwrap();
    let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    let session = async {
        let mut tls = connector.connect(name, tcp).await.ok()?;
        tls.write_all(b"c").await.ok()?;
        tls.flush().await.ok()?;
        let mut byte = [0u8; 1];
        tls.read_exact(&mut byte).await.ok()?;
        let (_, connection) = tls.get_ref();
        Some((
            connection.negotiated_cipher_suite()?.suite(),
            connection.protocol_version()?,
        ))
    };
    let client_result = tokio::time::timeout(Duration::from_secs(10), session)
        .await
        .ok()
        .flatten();
    let server_ok = server_side.await.unwrap();
    client_result.filter(|_| server_ok)
}

/// Two mutually trusting operator-identity nodes in `dir`, each configured by
/// `adjust` before initialization; returns the initialization results too.
async fn configured_pair(
    dir: &Path,
    adjust: impl Fn(&mut TlsConfig, OxirsNodeId),
) -> (TlsManager, TlsManager, Result<()>, Result<()>) {
    let bundle = dir.join("cluster.pem");
    let (mut config_1, pem_1) = node_config(dir, 1, &bundle, true).await;
    let (mut config_2, pem_2) = node_config(dir, 2, &bundle, true).await;
    std::fs::write(&bundle, format!("{pem_1}{pem_2}")).unwrap();
    adjust(&mut config_1, 1);
    adjust(&mut config_2, 2);
    let node_1 = TlsManager::new(config_1, 1);
    let init_1 = node_1.initialize().await;
    let node_2 = TlsManager::new(config_2, 2);
    let init_2 = node_2.initialize().await;
    (node_1, node_2, init_1, init_2)
}

/// TLS needs a trust bundle that cannot be defaulted, so the default
/// configuration must not claim TLS (it used to say `enabled: true` while
/// every manager built from it ran plaintext).
#[test]
fn test_default_config_leaves_tls_off() {
    assert!(!TlsConfig::default().enabled);
}

/// `cipher_suites` used to be ignored: rustls negotiated its own preference.
#[tokio::test]
async fn test_cipher_suite_setting_is_enforced() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (node_1, node_2, init_1, init_2) = configured_pair(dir.path(), |config, _| {
        config.cipher_suites = vec!["TLS_CHACHA20_POLY1305_SHA256".to_string()];
    })
    .await;
    init_1.unwrap();
    init_2.unwrap();

    let (suite, _) = negotiated(&node_2, &node_1, "node-2")
        .await
        .expect("handshake failed");
    assert_eq!(suite, rustls::CipherSuite::TLS13_CHACHA20_POLY1305_SHA256);
}

/// Peers without a common configured cipher suite must not connect.
#[tokio::test]
async fn test_disjoint_cipher_suites_do_not_connect() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (node_1, node_2, init_1, init_2) = configured_pair(dir.path(), |config, node| {
        let suite = if node == 1 {
            "TLS_AES_128_GCM_SHA256"
        } else {
            "TLS_CHACHA20_POLY1305_SHA256"
        };
        config.cipher_suites = vec![suite.to_string()];
    })
    .await;
    init_1.unwrap();
    init_2.unwrap();

    assert!(
        negotiated(&node_2, &node_1, "node-2").await.is_none(),
        "peers without a common cipher suite completed a handshake"
    );
}

/// `protocol_versions` used to be ignored: TLS 1.3 was negotiated even when
/// only TLS 1.2 was configured.
#[tokio::test]
async fn test_protocol_version_setting_is_enforced() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (node_1, node_2, init_1, init_2) = configured_pair(dir.path(), |config, _| {
        config.protocol_versions = vec!["TLSv1.2".to_string()];
        config.cipher_suites = vec!["TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256".to_string()];
    })
    .await;
    init_1.unwrap();
    init_2.unwrap();

    let (suite, version) = negotiated(&node_2, &node_1, "node-2")
        .await
        .expect("handshake failed");
    assert_eq!(version, rustls::ProtocolVersion::TLSv1_2);
    assert_eq!(
        suite,
        rustls::CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
    );
}

/// Settings that cannot take effect are errors, not silently ignored.
#[tokio::test]
async fn test_unusable_cipher_settings_are_rejected() {
    install_provider();
    type Adjust = fn(&mut TlsConfig);
    let cases: [(&str, Adjust); 5] = [
        ("unknown suite", |c| {
            c.cipher_suites = vec!["TLS_NOT_A_SUITE".to_string()]
        }),
        ("no suites", |c| c.cipher_suites.clear()),
        ("unknown version", |c| {
            c.protocol_versions = vec!["TLSv1.1".to_string()]
        }),
        ("version without a suite", |c| {
            c.protocol_versions = vec!["TLSv1.3".to_string(), "TLSv1.2".to_string()]
        }),
        ("suite of a disabled version", |c| {
            c.cipher_suites = vec![
                "TLS_AES_128_GCM_SHA256".to_string(),
                "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256".to_string(),
            ]
        }),
    ];
    for (name, adjust) in cases {
        let dir = TempDir::new().unwrap();
        let (_, _, init_1, _) = configured_pair(dir.path(), |config, node| {
            if node == 1 {
                config.protocol_versions = vec!["TLSv1.3".to_string()];
                adjust(config);
            }
        })
        .await;
        assert!(
            init_1.is_err(),
            "{name}: unusable TLS settings were accepted"
        );
    }
}

/// Generated certificates used to be valid until the year 4096 whatever
/// `cert_validity_days` said.
#[tokio::test]
async fn test_generated_certificate_honors_validity_days() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let config = TlsConfig {
        enabled: true,
        cert_dir: dir.path().to_path_buf(),
        cert_validity_days: 30,
        rotation_threshold_days: 7,
        ..Default::default()
    };
    let (cert_der, _) = TlsManager::new(config, 1)
        .generate_certificate(1)
        .await
        .unwrap();

    let not_after = certificate_info(&cert_der).unwrap().not_after;
    let expected = SystemTime::now() + Duration::from_secs(30 * 24 * 3600);
    let skew = Duration::from_secs(2 * 3600);
    assert!(
        not_after > expected - skew && not_after < expected + skew,
        "notAfter {not_after:?} is not 30 days from now"
    );
}

/// `CertificateInfo` used to report this node's name, serial "1", "now" as
/// the start of validity and a key usage the certificate does not have.
#[tokio::test]
async fn test_certificate_info_reflects_the_certificate() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let params = oxitls_rcgen::CertificateParamsBuilder::new()
        .with_common_name("node-2")
        .with_dns_names(&["node-2"])
        .with_digital_signature()
        .with_client_auth()
        .with_serial_number(0x1234_5678)
        .build()
        .unwrap();
    let key = oxitls_rcgen::OxiEd25519Key::generate().unwrap();
    let cert = params.self_signed(&key).unwrap();
    let path = dir.path().join("node-2.crt");
    std::fs::write(&path, pem_encode("CERTIFICATE", cert.der())).unwrap();

    let manager = TlsManager::new(
        TlsConfig {
            cert_dir: dir.path().to_path_buf(),
            ..Default::default()
        },
        1,
    );
    let info = manager.load_certificate(&path).await.unwrap();

    assert_eq!(info.subject, "CN=node-2");
    assert_eq!(info.issuer, "CN=node-2");
    assert_eq!(info.serial_number, "12:34:56:78");
    // rcgen's default validity starts on 1975-01-01.
    assert_eq!(
        info.not_before,
        SystemTime::UNIX_EPOCH + Duration::from_secs(157_766_400)
    );
    assert!(info.key_usage.contains(&"digital_signature".to_string()));
    assert!(info.key_usage.contains(&"client_auth".to_string()));
    assert!(!info.key_usage.contains(&"key_encipherment".to_string()));
}

/// Node 1 starts with an auto-generated identity valid for `validity_days`;
/// node 2 is an operator identity. Each pins the other's current
/// certificate. Returns both managers, node 1's config and its first PEM.
async fn pinned_pair_with_auto_node(
    dir: &Path,
    validity_days: u64,
) -> (TlsManager, TlsManager, TlsConfig, String) {
    let bundle = dir.join("cluster.pem");
    let mut config_1 = TlsConfig {
        enabled: true,
        require_client_auth: true,
        cert_dir: dir.join("node-1"),
        auto_rotate: false,
        cert_validity_days: validity_days,
        rotation_threshold_days: 1,
        ..Default::default()
    };
    // The first start generates node 1's identity, then stops for lack of trust anchors.
    assert!(TlsManager::new(config_1.clone(), 1)
        .initialize()
        .await
        .is_err());
    let pem_1 = std::fs::read_to_string(dir.join("node-1").join("node-1.crt")).unwrap();
    let (config_2, pem_2) = node_config(dir, 2, &bundle, true).await;
    std::fs::write(&bundle, format!("{pem_1}{pem_2}")).unwrap();
    config_1.ca_file = Some(bundle);

    let node_1 = TlsManager::new(config_1.clone(), 1);
    node_1.initialize().await.unwrap();
    let node_2 = TlsManager::new(config_2, 2);
    node_2.initialize().await.unwrap();
    (node_1, node_2, config_1, pem_1)
}

/// Renewing an auto-generated certificate must not cut the node off from
/// peers that pinned it. Rotation used to mint a new key, which no peer
/// trusted; with a real expiry that would strike every node after
/// `cert_validity_days - rotation_threshold_days`.
#[tokio::test]
async fn test_renewal_keeps_pinned_peers_trusting_the_node() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (node_1, node_2, _, pem_1) = pinned_pair_with_auto_node(dir.path(), 30).await;
    let key_before = std::fs::read(dir.path().join("node-1").join("node-1.key")).unwrap();

    node_1.renew_certificate().await.unwrap();

    let pem_after = std::fs::read_to_string(dir.path().join("node-1").join("node-1.crt")).unwrap();
    assert_ne!(pem_after, pem_1, "the certificate was not renewed");
    assert!(
        std::fs::read(dir.path().join("node-1").join("node-1.key")).unwrap() == key_before,
        "renewal replaced the private key"
    );
    assert_eq!(tls_session(&node_1, &node_2, "node-1").await, (true, true));
    assert_eq!(tls_session(&node_2, &node_1, "node-2").await, (true, true));
}

/// Half an identity is an error: generating a new one would silently
/// replace the identity peers trust.
#[tokio::test]
async fn test_half_present_identity_is_not_regenerated() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let generator = TlsManager::new(TlsConfig::default(), 1);
    let (cert_der, _) = generator.generate_certificate(1).await.unwrap();
    let cert_path = dir.path().join("node-1.crt");
    let pem = pem_encode("CERTIFICATE", &cert_der);
    std::fs::write(&cert_path, &pem).unwrap();
    std::fs::write(dir.path().join("cluster.pem"), &pem).unwrap();

    let config = TlsConfig {
        enabled: true,
        cert_dir: dir.path().to_path_buf(),
        ca_file: Some(dir.path().join("cluster.pem")),
        auto_rotate: false,
        ..Default::default()
    };
    assert!(TlsManager::new(config, 1).initialize().await.is_err());
    assert_eq!(std::fs::read_to_string(&cert_path).unwrap(), pem);
    assert!(!dir.path().join("node-1.key").exists());
}

/// Private keys used to be written world-readable (0644).
#[cfg(unix)]
#[tokio::test]
async fn test_generated_private_key_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    install_provider();
    let dir = TempDir::new().unwrap();
    let (_manager, _) = auto_identity_manager(dir.path(), 30).await;

    let mode = std::fs::metadata(dir.path().join("node-1.key"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(
        mode & 0o777,
        0o600,
        "private key mode is {:o}",
        mode & 0o777
    );
}

/// Keys left world-readable by earlier releases are tightened on start.
#[cfg(unix)]
#[tokio::test]
async fn test_world_readable_generated_key_is_tightened() {
    use std::os::unix::fs::PermissionsExt;
    install_provider();
    let dir = TempDir::new().unwrap();
    let (manager, _) = auto_identity_manager(dir.path(), 30).await;
    let key_path = dir.path().join("node-1.key");
    std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o644)).unwrap();
    let config = manager.config.clone();
    drop(manager);

    TlsManager::new(config, 1).initialize().await.unwrap();

    let mode = std::fs::metadata(&key_path).unwrap().permissions().mode();
    assert_eq!(
        mode & 0o777,
        0o600,
        "private key mode is {:o}",
        mode & 0o777
    );
}

// ---------------------------------------------------------------------------
// Certificate lifecycle with real validity periods
// ---------------------------------------------------------------------------

/// Write `cert_der` and `key_der` as node 1's identity in `dir`.
fn write_identity(dir: &Path, cert_der: &[u8], key_der: &[u8]) -> (PathBuf, PathBuf) {
    let cert_path = dir.join("node-1.crt");
    let key_path = dir.join("node-1.key");
    std::fs::write(&cert_path, pem_encode("CERTIFICATE", cert_der)).unwrap();
    std::fs::write(&key_path, pem_encode("PRIVATE KEY", key_der)).unwrap();
    (cert_path, key_path)
}

/// A self-signed node certificate valid from `not_before` to `not_after`
/// (offsets from now), with its PKCS#8 key.
fn certificate_valid_between(
    not_before: time::Duration,
    not_after: time::Duration,
) -> (Vec<u8>, Vec<u8>) {
    let now = time::OffsetDateTime::now_utc();
    let key = oxitls_rcgen::OxiEd25519Key::generate().unwrap();
    let cert = identity_template(1)
        .with_serial_number(7)
        .with_validity(now + not_before, now + not_after)
        .build()
        .unwrap()
        .self_signed(&key)
        .unwrap();
    (cert.der().to_vec(), key.pkcs8_der().to_vec())
}

/// A certificate near expiry is renewed at startup with the same key, and
/// a peer that pinned the old certificate keeps trusting the node — even
/// once the old certificate has expired, because trust anchors carry no
/// validity period.
#[tokio::test]
async fn test_startup_renews_an_expiring_identity_with_the_same_key() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (_, node_2, mut config_1, old_pem) = pinned_pair_with_auto_node(dir.path(), 2).await;
    let key_path = dir.path().join("node-1").join("node-1.key");
    let key_before = std::fs::read(&key_path).unwrap();

    // Restart node 1 with a policy under which its 2-day certificate is due.
    config_1.cert_validity_days = 30;
    config_1.rotation_threshold_days = 3;
    config_1.auto_rotate = true;
    let node_1 = TlsManager::new(config_1, 1);
    node_1.initialize().await.unwrap();

    let new_pem = std::fs::read_to_string(dir.path().join("node-1").join("node-1.crt")).unwrap();
    assert_ne!(new_pem, old_pem, "the expiring certificate was not renewed");
    assert!(
        std::fs::read(&key_path).unwrap() == key_before,
        "renewal replaced the key"
    );
    let info = node_1.get_certificate_info("server").await.unwrap();
    assert!(info.not_after > SystemTime::now() + Duration::from_secs(29 * 24 * 3600));
    assert_eq!(tls_session(&node_1, &node_2, "node-1").await, (true, true));

    // Three days from now the old certificate has expired, yet pinning it
    // still authenticates the renewed one.
    let der = |pem: &str| {
        rustls_pemfile::certs(&mut pem.as_bytes())
            .next()
            .unwrap()
            .unwrap()
    };
    let (old_cert, new_cert) = (der(&old_pem), der(&new_pem));
    let mut roots = RootCertStore::empty();
    roots.add(old_cert.clone()).unwrap();
    let verifier =
        WebPkiServerVerifier::builder_with_provider(Arc::new(roots), oxitls::pure_provider())
            .build()
            .unwrap();
    let later = UnixTime::since_unix_epoch(
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            + Duration::from_secs(3 * 24 * 3600),
    );
    let name = ServerName::try_from("node-1".to_string()).unwrap();
    assert!(
        verifier
            .verify_server_cert(&old_cert, &[], &name, &[], later)
            .is_err(),
        "the old certificate should have expired by then"
    );
    verifier
        .verify_server_cert(&new_cert, &[], &name, &[], later)
        .expect("the renewed certificate must verify against the expired pin");
}

/// A certificate this node did not issue itself is never replaced: only its
/// issuer can renew it.
#[tokio::test]
async fn test_ca_issued_certificate_is_never_renewed() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let ca =
        oxitls_rcgen::generate_ca("Cluster CA", oxitls_rcgen::SigningAlgorithm::Ed25519).unwrap();
    let leaf = oxitls_rcgen::generate_ca_signed_leaf(
        &["node-1"],
        oxitls_rcgen::SigningAlgorithm::Ed25519,
        &ca,
    )
    .unwrap();
    let (cert_path, key_path) = write_identity(dir.path(), &leaf.cert_der, &leaf.pkcs8_der);
    let before = (
        std::fs::read(&cert_path).unwrap(),
        std::fs::read(&key_path).unwrap(),
    );
    let manager = TlsManager::new(
        TlsConfig {
            enabled: true,
            cert_dir: dir.path().to_path_buf(),
            ..Default::default()
        },
        1,
    );

    let error = manager.renew_identity_certificate().await.unwrap_err();
    assert!(error.to_string().contains("issued by"), "{error}");
    assert!(
        before
            == (
                std::fs::read(&cert_path).unwrap(),
                std::fs::read(&key_path).unwrap()
            )
    );
}

/// Keys other than Ed25519 (such as those written by releases before 0.3.1)
/// cannot be re-signed here, so their certificates are left alone.
#[tokio::test]
async fn test_non_ed25519_identity_is_never_renewed() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let identity = oxitls_rcgen::generate_self_signed_p256(&["node-1"]).unwrap();
    let (cert_path, key_path) = write_identity(dir.path(), &identity.cert_der, &identity.pkcs8_der);
    let before = (
        std::fs::read(&cert_path).unwrap(),
        std::fs::read(&key_path).unwrap(),
    );
    let manager = TlsManager::new(
        TlsConfig {
            enabled: true,
            cert_dir: dir.path().to_path_buf(),
            ..Default::default()
        },
        1,
    );

    let error = manager.renew_identity_certificate().await.unwrap_err();
    assert!(error.to_string().contains("not an Ed25519 key"), "{error}");
    assert!(
        before
            == (
                std::fs::read(&cert_path).unwrap(),
                std::fs::read(&key_path).unwrap()
            )
    );
}

/// An expired certificate that will not be renewed is an error at startup
/// (peers would reject every handshake), not a node that looks healthy.
#[tokio::test]
async fn test_expired_unrenewable_certificate_fails_initialization() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (cert_der, key_der) =
        certificate_valid_between(time::Duration::days(-10), time::Duration::days(-1));
    let (cert_path, key_path) = write_identity(dir.path(), &cert_der, &key_der);
    let bundle = dir.path().join("cluster.pem");
    std::fs::write(&bundle, pem_encode("CERTIFICATE", &cert_der)).unwrap();

    let operator = TlsConfig {
        enabled: true,
        cert_file: Some(cert_path.clone()),
        key_file: Some(key_path.clone()),
        ca_file: Some(bundle.clone()),
        cert_dir: dir.path().to_path_buf(),
        auto_rotate: false,
        ..Default::default()
    };
    let error = TlsManager::new(operator, 1).initialize().await.unwrap_err();
    assert!(error.to_string().contains("expired"), "{error}");

    let auto_without_rotation = TlsConfig {
        enabled: true,
        ca_file: Some(bundle),
        cert_dir: dir.path().to_path_buf(),
        auto_rotate: false,
        ..Default::default()
    };
    let error = TlsManager::new(auto_without_rotation, 1)
        .initialize()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("expired"), "{error}");
    assert_eq!(
        std::fs::read_to_string(&cert_path).unwrap(),
        pem_encode("CERTIFICATE", &cert_der)
    );
}

/// A key without its certificate (a lost file, or a first start interrupted
/// between the two writes) gets a new certificate for the same key, which
/// peers that pinned the lost one still trust.
#[tokio::test]
async fn test_missing_certificate_is_reissued_for_the_existing_key() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (_, node_2, config_1, _) = pinned_pair_with_auto_node(dir.path(), 30).await;
    let cert_path = dir.path().join("node-1").join("node-1.crt");
    let key_path = dir.path().join("node-1").join("node-1.key");
    let key_before = std::fs::read(&key_path).unwrap();
    std::fs::remove_file(&cert_path).unwrap();

    let node_1 = TlsManager::new(config_1, 1);
    node_1.initialize().await.unwrap();

    assert!(cert_path.exists());
    assert!(std::fs::read(&key_path).unwrap() == key_before);
    assert_eq!(tls_session(&node_1, &node_2, "node-1").await, (true, true));
}

/// A rotation threshold that is not below the validity period would renew
/// every certificate as soon as it is issued.
#[tokio::test]
async fn test_rotation_threshold_must_be_below_validity() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let config = TlsConfig {
        enabled: true,
        cert_dir: dir.path().to_path_buf(),
        cert_validity_days: 30,
        rotation_threshold_days: 30,
        ..Default::default()
    };
    let error = TlsManager::new(config, 1).initialize().await.unwrap_err();
    assert!(
        error.to_string().contains("rotation_threshold_days"),
        "{error}"
    );
    assert!(!dir.path().join("node-1.crt").exists());
}

/// `initialize` runs once; later calls, including concurrent ones, reuse it.
#[tokio::test]
async fn test_initialize_runs_once() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let (manager, pem) = auto_identity_manager(dir.path(), 30).await;
    let (first, second) = tokio::join!(manager.initialize(), manager.initialize());
    first.unwrap();
    second.unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("node-1.crt")).unwrap(),
        pem
    );
}

/// Renewal is installed only if peers pinning the current certificate would
/// accept it. A self-signed certificate under another subject (here CN
/// `other`) cannot be continued by the node template, so it is left alone.
#[tokio::test]
async fn test_renewal_that_peers_would_reject_is_not_installed() {
    install_provider();
    let dir = TempDir::new().unwrap();
    let key = oxitls_rcgen::OxiEd25519Key::generate().unwrap();
    let cert = oxitls_rcgen::CertificateParamsBuilder::new()
        .with_common_name("other")
        .with_dns_names(&["node-1"])
        .with_digital_signature()
        .with_server_auth()
        .with_client_auth()
        .build()
        .unwrap()
        .self_signed(&key)
        .unwrap();
    let (cert_path, key_path) = write_identity(dir.path(), cert.der(), key.pkcs8_der());
    let before = (
        std::fs::read(&cert_path).unwrap(),
        std::fs::read(&key_path).unwrap(),
    );
    let manager = TlsManager::new(
        TlsConfig {
            enabled: true,
            cert_dir: dir.path().to_path_buf(),
            ..Default::default()
        },
        1,
    );

    let error = manager.renew_identity_certificate().await.unwrap_err();
    assert!(error.to_string().contains("would reject"), "{error}");
    assert!(
        before
            == (
                std::fs::read(&cert_path).unwrap(),
                std::fs::read(&key_path).unwrap()
            )
    );
}
