//! # Post-Quantum Cryptography Engine
//!
//! Advanced post-quantum cryptographic operations including key generation,
//! signing, verification, and key encapsulation (KEM) for OxiRS Stream.
//!
//! Enable the `post-quantum` feature flag to activate real pqcrypto implementations.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
#[cfg(feature = "post-quantum")]
use uuid::Uuid;

use super::{EncryptionAlgorithm, PostQuantumConfig, PostQuantumSignature};

/// Post-quantum cryptographic engine for advanced security operations
pub struct PostQuantumCryptoEngine {
    config: Arc<PostQuantumConfig>,
    key_store: Arc<RwLock<HashMap<String, PostQuantumKeyPair>>>,
    signature_cache: Arc<RwLock<HashMap<String, SignatureInfo>>>,
    metrics: Arc<RwLock<PQCryptoMetrics>>,
}

impl PostQuantumCryptoEngine {
    /// Create a new post-quantum crypto engine
    pub fn new(config: PostQuantumConfig) -> Self {
        Self {
            config: Arc::new(config),
            key_store: Arc::new(RwLock::new(HashMap::new())),
            signature_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(PQCryptoMetrics::default())),
        }
    }

    /// Generate a new post-quantum key pair
    pub async fn generate_keypair(
        &self,
        algorithm: &PostQuantumSignature,
    ) -> Result<PostQuantumKeyPair> {
        let start_time = Instant::now();

        let keypair = match algorithm {
            PostQuantumSignature::Dilithium2
            | PostQuantumSignature::Dilithium3
            | PostQuantumSignature::Dilithium5 => {
                self.generate_dilithium_keypair(algorithm).await?
            }
            PostQuantumSignature::SphincsPlusSha2128s
            | PostQuantumSignature::SphincsPlusSha2256s => {
                self.generate_sphincs_keypair(algorithm).await?
            }
            PostQuantumSignature::Falcon512 | PostQuantumSignature::Falcon1024 => {
                self.generate_falcon_keypair(algorithm).await?
            }
            _ => {
                return Err(anyhow!(
                    "Post-quantum algorithm {:?} is not yet implemented. \
                     Supported algorithms: Dilithium2/3/5, SPHINCS+-SHA2-128s/256s, Falcon512/1024",
                    algorithm
                ));
            }
        };

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.key_generation_count += 1;
            metrics.avg_key_generation_time_ms = (metrics.avg_key_generation_time_ms
                + start_time.elapsed().as_millis() as f64)
                / 2.0;
        }

        // Store the key pair
        {
            let mut store = self.key_store.write().await;
            store.insert(keypair.id.clone(), keypair.clone());
        }

        Ok(keypair)
    }

    /// Sign data using post-quantum signature.
    ///
    /// Returns the raw signature bytes. Enable the `post-quantum` feature flag
    /// for a real cryptographic implementation backed by pqcrypto.
    pub async fn sign(&self, key_id: &str, data: &[u8]) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        let key_store = self.key_store.read().await;
        let keypair = key_store
            .get(key_id)
            .ok_or_else(|| anyhow!("Key not found: {}", key_id))?;

        let signature_bytes = match keypair.algorithm {
            PostQuantumSignature::Dilithium2
            | PostQuantumSignature::Dilithium3
            | PostQuantumSignature::Dilithium5 => {
                self.dilithium_sign(&keypair.private_key, data, &keypair.algorithm)
                    .await?
            }
            PostQuantumSignature::SphincsPlusSha2128s
            | PostQuantumSignature::SphincsPlusSha2128f
            | PostQuantumSignature::SphincsPlusSha2192s
            | PostQuantumSignature::SphincsPlusSha2192f
            | PostQuantumSignature::SphincsPlusSha2256s
            | PostQuantumSignature::SphincsPlusSha2256f
            | PostQuantumSignature::SphincsPlusShake128s
            | PostQuantumSignature::SphincsPlusShake128f
            | PostQuantumSignature::SphincsPlusShake192s
            | PostQuantumSignature::SphincsPlusShake192f
            | PostQuantumSignature::SphincsPlusShake256s
            | PostQuantumSignature::SphincsPlusShake256f => {
                self.sphincs_sign(&keypair.private_key, data, &keypair.algorithm)
                    .await?
            }
            PostQuantumSignature::Falcon512 | PostQuantumSignature::Falcon1024 => {
                self.falcon_sign(&keypair.private_key, data, &keypair.algorithm)
                    .await?
            }
            _ => {
                return Err(anyhow!(
                    "Signing with {:?} not implemented",
                    keypair.algorithm
                ));
            }
        };

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.signature_count += 1;
            metrics.avg_signing_time_ms =
                (metrics.avg_signing_time_ms + start_time.elapsed().as_millis() as f64) / 2.0;
        }

        Ok(signature_bytes)
    }

    /// Verify post-quantum signature.
    ///
    /// Enable the `post-quantum` feature flag for a real cryptographic implementation.
    pub async fn verify(
        &self,
        public_key: &[u8],
        data: &[u8],
        signature: &[u8],
        algorithm: &PostQuantumSignature,
    ) -> Result<bool> {
        let start_time = Instant::now();

        let is_valid = match algorithm {
            PostQuantumSignature::Dilithium2
            | PostQuantumSignature::Dilithium3
            | PostQuantumSignature::Dilithium5 => {
                self.dilithium_verify(public_key, data, signature, algorithm)
                    .await?
            }
            PostQuantumSignature::SphincsPlusSha2128s
            | PostQuantumSignature::SphincsPlusSha2128f
            | PostQuantumSignature::SphincsPlusSha2192s
            | PostQuantumSignature::SphincsPlusSha2192f
            | PostQuantumSignature::SphincsPlusSha2256s
            | PostQuantumSignature::SphincsPlusSha2256f
            | PostQuantumSignature::SphincsPlusShake128s
            | PostQuantumSignature::SphincsPlusShake128f
            | PostQuantumSignature::SphincsPlusShake192s
            | PostQuantumSignature::SphincsPlusShake192f
            | PostQuantumSignature::SphincsPlusShake256s
            | PostQuantumSignature::SphincsPlusShake256f => {
                self.sphincs_verify(public_key, data, signature, algorithm)
                    .await?
            }
            PostQuantumSignature::Falcon512 | PostQuantumSignature::Falcon1024 => {
                self.falcon_verify(public_key, data, signature, algorithm)
                    .await?
            }
            _ => {
                return Err(anyhow!("Verification for {:?} not implemented", algorithm));
            }
        };

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.verification_count += 1;
            metrics.avg_verification_time_ms =
                (metrics.avg_verification_time_ms + start_time.elapsed().as_millis() as f64) / 2.0;
            if is_valid {
                metrics.successful_verifications += 1;
            }
        }

        Ok(is_valid)
    }

    /// Perform key encapsulation using post-quantum KEM
    pub async fn encapsulate(
        &self,
        public_key: &[u8],
        algorithm: &EncryptionAlgorithm,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        match algorithm {
            EncryptionAlgorithm::Kyber768 => self.kyber_encapsulate(public_key).await,
            EncryptionAlgorithm::NewHope1024 => self.newhope_encapsulate(public_key).await,
            EncryptionAlgorithm::FrodoKEM976 => self.frodokem_encapsulate(public_key).await,
            _ => Err(anyhow!("KEM algorithm {:?} not implemented", algorithm)),
        }
    }

    /// Perform key decapsulation using post-quantum KEM
    pub async fn decapsulate(
        &self,
        private_key: &[u8],
        ciphertext: &[u8],
        algorithm: &EncryptionAlgorithm,
    ) -> Result<Vec<u8>> {
        match algorithm {
            EncryptionAlgorithm::Kyber768 => self.kyber_decapsulate(private_key, ciphertext).await,
            EncryptionAlgorithm::NewHope1024 => {
                self.newhope_decapsulate(private_key, ciphertext).await
            }
            EncryptionAlgorithm::FrodoKEM976 => {
                self.frodokem_decapsulate(private_key, ciphertext).await
            }
            _ => Err(anyhow!("KEM algorithm {:?} not implemented", algorithm)),
        }
    }

    // -----------------------------------------------------------------------
    // Private implementation methods — feature-gated on "post-quantum"
    // -----------------------------------------------------------------------

    /// Generate a Dilithium key pair.
    ///
    /// With `--features post-quantum` this calls the real pqcrypto-dilithium
    /// C implementation.  Without the feature the method returns an error
    /// telling the caller which flag to enable.
    #[cfg(feature = "post-quantum")]
    async fn generate_dilithium_keypair(
        &self,
        variant: &PostQuantumSignature,
    ) -> Result<PostQuantumKeyPair> {
        use pqcrypto_traits::sign::{PublicKey as _, SecretKey as _};
        let (pk_bytes, sk_bytes) = match variant {
            PostQuantumSignature::Dilithium2 => {
                let (pk, sk) = pqcrypto_dilithium::dilithium2::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::Dilithium3 => {
                let (pk, sk) = pqcrypto_dilithium::dilithium3::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::Dilithium5 => {
                let (pk, sk) = pqcrypto_dilithium::dilithium5::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            other => {
                return Err(anyhow!(
                    "generate_dilithium_keypair: unexpected variant {:?}",
                    other
                ))
            }
        };
        Ok(PostQuantumKeyPair {
            id: Uuid::new_v4().to_string(),
            algorithm: variant.clone(),
            public_key: pk_bytes,
            private_key: sk_bytes,
            created_at: Utc::now(),
        })
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn generate_dilithium_keypair(
        &self,
        variant: &PostQuantumSignature,
    ) -> Result<PostQuantumKeyPair> {
        Err(anyhow!(
            "Post-quantum key generation for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Generate a SPHINCS+ key pair.
    #[cfg(feature = "post-quantum")]
    async fn generate_sphincs_keypair(
        &self,
        variant: &PostQuantumSignature,
    ) -> Result<PostQuantumKeyPair> {
        use pqcrypto_traits::sign::{PublicKey as _, SecretKey as _};
        let (pk_bytes, sk_bytes) = match variant {
            PostQuantumSignature::SphincsPlusSha2128s => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincssha2128ssimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusSha2128f => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincssha2128fsimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusSha2192s => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincssha2192ssimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusSha2192f => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincssha2192fsimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusSha2256s => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincssha2256ssimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusSha2256f => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincssha2256fsimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusShake128s => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincsshake128ssimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusShake128f => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincsshake128fsimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusShake192s => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincsshake192ssimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusShake192f => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincsshake192fsimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusShake256s => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincsshake256ssimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::SphincsPlusShake256f => {
                let (pk, sk) = pqcrypto_sphincsplus::sphincsshake256fsimple::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            other => {
                return Err(anyhow!(
                    "generate_sphincs_keypair: unexpected variant {:?}",
                    other
                ))
            }
        };
        Ok(PostQuantumKeyPair {
            id: Uuid::new_v4().to_string(),
            algorithm: variant.clone(),
            public_key: pk_bytes,
            private_key: sk_bytes,
            created_at: Utc::now(),
        })
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn generate_sphincs_keypair(
        &self,
        variant: &PostQuantumSignature,
    ) -> Result<PostQuantumKeyPair> {
        Err(anyhow!(
            "Post-quantum key generation for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Generate a Falcon key pair using pqcrypto-falcon.
    ///
    /// Supports Falcon512 and Falcon1024 variants. Requires the `post-quantum`
    /// feature gate; returns a descriptive error otherwise.
    #[cfg(feature = "post-quantum")]
    async fn generate_falcon_keypair(
        &self,
        variant: &PostQuantumSignature,
    ) -> Result<PostQuantumKeyPair> {
        use pqcrypto_traits::sign::{PublicKey as _, SecretKey as _};
        let (pk_bytes, sk_bytes) = match variant {
            PostQuantumSignature::Falcon512 => {
                let (pk, sk) = pqcrypto_falcon::falcon512::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            PostQuantumSignature::Falcon1024 => {
                let (pk, sk) = pqcrypto_falcon::falcon1024::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            other => {
                return Err(anyhow!(
                    "generate_falcon_keypair: unexpected variant {:?}",
                    other
                ))
            }
        };
        Ok(PostQuantumKeyPair {
            id: Uuid::new_v4().to_string(),
            algorithm: variant.clone(),
            public_key: pk_bytes,
            private_key: sk_bytes,
            created_at: Utc::now(),
        })
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn generate_falcon_keypair(
        &self,
        variant: &PostQuantumSignature,
    ) -> Result<PostQuantumKeyPair> {
        Err(anyhow!(
            "Falcon key generation for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Sign `data` with Dilithium and return the raw detached signature bytes.
    #[cfg(feature = "post-quantum")]
    async fn dilithium_sign(
        &self,
        private_key: &[u8],
        data: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<Vec<u8>> {
        use pqcrypto_traits::sign::{DetachedSignature as _, SecretKey as _};
        let sig_bytes = match variant {
            PostQuantumSignature::Dilithium2 => {
                let sk = pqcrypto_dilithium::dilithium2::SecretKey::from_bytes(private_key)
                    .map_err(|e| anyhow!("Invalid Dilithium2 secret key: {}", e))?;
                let sig = pqcrypto_dilithium::dilithium2::detached_sign(data, &sk);
                sig.as_bytes().to_vec()
            }
            PostQuantumSignature::Dilithium3 => {
                let sk = pqcrypto_dilithium::dilithium3::SecretKey::from_bytes(private_key)
                    .map_err(|e| anyhow!("Invalid Dilithium3 secret key: {}", e))?;
                let sig = pqcrypto_dilithium::dilithium3::detached_sign(data, &sk);
                sig.as_bytes().to_vec()
            }
            PostQuantumSignature::Dilithium5 => {
                let sk = pqcrypto_dilithium::dilithium5::SecretKey::from_bytes(private_key)
                    .map_err(|e| anyhow!("Invalid Dilithium5 secret key: {}", e))?;
                let sig = pqcrypto_dilithium::dilithium5::detached_sign(data, &sk);
                sig.as_bytes().to_vec()
            }
            other => return Err(anyhow!("dilithium_sign: unexpected variant {:?}", other)),
        };
        Ok(sig_bytes)
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn dilithium_sign(
        &self,
        _private_key: &[u8],
        _data: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<Vec<u8>> {
        Err(anyhow!(
            "Dilithium signing for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Sign `data` with SPHINCS+ and return the raw detached signature bytes.
    #[cfg(feature = "post-quantum")]
    async fn sphincs_sign(
        &self,
        private_key: &[u8],
        data: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<Vec<u8>> {
        use pqcrypto_traits::sign::{DetachedSignature as _, SecretKey as _};
        let sig_bytes = match variant {
            PostQuantumSignature::SphincsPlusSha2128s => {
                let sk =
                    pqcrypto_sphincsplus::sphincssha2128ssimple::SecretKey::from_bytes(private_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-128s secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2128ssimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusSha2128f => {
                let sk =
                    pqcrypto_sphincsplus::sphincssha2128fsimple::SecretKey::from_bytes(private_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-128f secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2128fsimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusSha2192s => {
                let sk =
                    pqcrypto_sphincsplus::sphincssha2192ssimple::SecretKey::from_bytes(private_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-192s secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2192ssimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusSha2192f => {
                let sk =
                    pqcrypto_sphincsplus::sphincssha2192fsimple::SecretKey::from_bytes(private_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-192f secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2192fsimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusSha2256s => {
                let sk =
                    pqcrypto_sphincsplus::sphincssha2256ssimple::SecretKey::from_bytes(private_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-256s secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2256ssimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusSha2256f => {
                let sk =
                    pqcrypto_sphincsplus::sphincssha2256fsimple::SecretKey::from_bytes(private_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-256f secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2256fsimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusShake128s => {
                let sk = pqcrypto_sphincsplus::sphincsshake128ssimple::SecretKey::from_bytes(
                    private_key,
                )
                .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-128s secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake128ssimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusShake128f => {
                let sk = pqcrypto_sphincsplus::sphincsshake128fsimple::SecretKey::from_bytes(
                    private_key,
                )
                .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-128f secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake128fsimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusShake192s => {
                let sk = pqcrypto_sphincsplus::sphincsshake192ssimple::SecretKey::from_bytes(
                    private_key,
                )
                .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-192s secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake192ssimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusShake192f => {
                let sk = pqcrypto_sphincsplus::sphincsshake192fsimple::SecretKey::from_bytes(
                    private_key,
                )
                .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-192f secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake192fsimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusShake256s => {
                let sk = pqcrypto_sphincsplus::sphincsshake256ssimple::SecretKey::from_bytes(
                    private_key,
                )
                .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-256s secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake256ssimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::SphincsPlusShake256f => {
                let sk = pqcrypto_sphincsplus::sphincsshake256fsimple::SecretKey::from_bytes(
                    private_key,
                )
                .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-256f secret key: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake256fsimple::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            other => return Err(anyhow!("sphincs_sign: unexpected variant {:?}", other)),
        };
        Ok(sig_bytes)
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn sphincs_sign(
        &self,
        _private_key: &[u8],
        _data: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<Vec<u8>> {
        Err(anyhow!(
            "SPHINCS+ signing for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Sign `data` with Falcon and return the raw detached signature bytes.
    ///
    /// Supports Falcon512 and Falcon1024. Requires the `post-quantum` feature gate.
    #[cfg(feature = "post-quantum")]
    async fn falcon_sign(
        &self,
        private_key: &[u8],
        data: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<Vec<u8>> {
        use pqcrypto_traits::sign::{DetachedSignature as _, SecretKey as _};
        let sig_bytes = match variant {
            PostQuantumSignature::Falcon512 => {
                let sk = pqcrypto_falcon::falcon512::SecretKey::from_bytes(private_key)
                    .map_err(|e| anyhow!("Invalid Falcon512 secret key: {}", e))?;
                pqcrypto_falcon::falcon512::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            PostQuantumSignature::Falcon1024 => {
                let sk = pqcrypto_falcon::falcon1024::SecretKey::from_bytes(private_key)
                    .map_err(|e| anyhow!("Invalid Falcon1024 secret key: {}", e))?;
                pqcrypto_falcon::falcon1024::detached_sign(data, &sk)
                    .as_bytes()
                    .to_vec()
            }
            other => return Err(anyhow!("falcon_sign: unexpected variant {:?}", other)),
        };
        Ok(sig_bytes)
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn falcon_sign(
        &self,
        _private_key: &[u8],
        _data: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<Vec<u8>> {
        Err(anyhow!(
            "Falcon signing for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Verify a Dilithium detached signature.
    #[cfg(feature = "post-quantum")]
    async fn dilithium_verify(
        &self,
        public_key: &[u8],
        data: &[u8],
        signature: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<bool> {
        use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _};
        let ok = match variant {
            PostQuantumSignature::Dilithium2 => {
                let pk = pqcrypto_dilithium::dilithium2::PublicKey::from_bytes(public_key)
                    .map_err(|e| anyhow!("Invalid Dilithium2 public key: {}", e))?;
                let sig = pqcrypto_dilithium::dilithium2::DetachedSignature::from_bytes(signature)
                    .map_err(|e| anyhow!("Invalid Dilithium2 signature: {}", e))?;
                pqcrypto_dilithium::dilithium2::verify_detached_signature(&sig, data, &pk).is_ok()
            }
            PostQuantumSignature::Dilithium3 => {
                let pk = pqcrypto_dilithium::dilithium3::PublicKey::from_bytes(public_key)
                    .map_err(|e| anyhow!("Invalid Dilithium3 public key: {}", e))?;
                let sig = pqcrypto_dilithium::dilithium3::DetachedSignature::from_bytes(signature)
                    .map_err(|e| anyhow!("Invalid Dilithium3 signature: {}", e))?;
                pqcrypto_dilithium::dilithium3::verify_detached_signature(&sig, data, &pk).is_ok()
            }
            PostQuantumSignature::Dilithium5 => {
                let pk = pqcrypto_dilithium::dilithium5::PublicKey::from_bytes(public_key)
                    .map_err(|e| anyhow!("Invalid Dilithium5 public key: {}", e))?;
                let sig = pqcrypto_dilithium::dilithium5::DetachedSignature::from_bytes(signature)
                    .map_err(|e| anyhow!("Invalid Dilithium5 signature: {}", e))?;
                pqcrypto_dilithium::dilithium5::verify_detached_signature(&sig, data, &pk).is_ok()
            }
            other => return Err(anyhow!("dilithium_verify: unexpected variant {:?}", other)),
        };
        Ok(ok)
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn dilithium_verify(
        &self,
        _public_key: &[u8],
        _data: &[u8],
        _signature: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<bool> {
        Err(anyhow!(
            "Dilithium verification for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Verify a SPHINCS+ detached signature.
    #[cfg(feature = "post-quantum")]
    async fn sphincs_verify(
        &self,
        public_key: &[u8],
        data: &[u8],
        signature: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<bool> {
        use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _};
        let ok = match variant {
            PostQuantumSignature::SphincsPlusSha2128s => {
                let pk =
                    pqcrypto_sphincsplus::sphincssha2128ssimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-128s public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincssha2128ssimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-128s signature: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2128ssimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusSha2128f => {
                let pk =
                    pqcrypto_sphincsplus::sphincssha2128fsimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-128f public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincssha2128fsimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-128f signature: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2128fsimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusSha2192s => {
                let pk =
                    pqcrypto_sphincsplus::sphincssha2192ssimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-192s public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincssha2192ssimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-192s signature: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2192ssimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusSha2192f => {
                let pk =
                    pqcrypto_sphincsplus::sphincssha2192fsimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-192f public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincssha2192fsimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-192f signature: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2192fsimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusSha2256s => {
                let pk =
                    pqcrypto_sphincsplus::sphincssha2256ssimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-256s public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincssha2256ssimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-256s signature: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2256ssimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusSha2256f => {
                let pk =
                    pqcrypto_sphincsplus::sphincssha2256fsimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-256f public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincssha2256fsimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHA2-256f signature: {}", e))?;
                pqcrypto_sphincsplus::sphincssha2256fsimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusShake128s => {
                let pk =
                    pqcrypto_sphincsplus::sphincsshake128ssimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-128s public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincsshake128ssimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-128s signature: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake128ssimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusShake128f => {
                let pk =
                    pqcrypto_sphincsplus::sphincsshake128fsimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-128f public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincsshake128fsimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-128f signature: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake128fsimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusShake192s => {
                let pk =
                    pqcrypto_sphincsplus::sphincsshake192ssimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-192s public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincsshake192ssimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-192s signature: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake192ssimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusShake192f => {
                let pk =
                    pqcrypto_sphincsplus::sphincsshake192fsimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-192f public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincsshake192fsimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-192f signature: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake192fsimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusShake256s => {
                let pk =
                    pqcrypto_sphincsplus::sphincsshake256ssimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-256s public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincsshake256ssimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-256s signature: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake256ssimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            PostQuantumSignature::SphincsPlusShake256f => {
                let pk =
                    pqcrypto_sphincsplus::sphincsshake256fsimple::PublicKey::from_bytes(public_key)
                        .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-256f public key: {}", e))?;
                let sig =
                    pqcrypto_sphincsplus::sphincsshake256fsimple::DetachedSignature::from_bytes(
                        signature,
                    )
                    .map_err(|e| anyhow!("Invalid SPHINCS+ SHAKE-256f signature: {}", e))?;
                pqcrypto_sphincsplus::sphincsshake256fsimple::verify_detached_signature(
                    &sig, data, &pk,
                )
                .is_ok()
            }
            other => return Err(anyhow!("sphincs_verify: unexpected variant {:?}", other)),
        };
        Ok(ok)
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn sphincs_verify(
        &self,
        _public_key: &[u8],
        _data: &[u8],
        _signature: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<bool> {
        Err(anyhow!(
            "SPHINCS+ verification for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Verify a Falcon detached signature.
    ///
    /// Supports Falcon512 and Falcon1024. Requires the `post-quantum` feature gate.
    #[cfg(feature = "post-quantum")]
    async fn falcon_verify(
        &self,
        public_key: &[u8],
        data: &[u8],
        signature: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<bool> {
        use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _};
        let ok = match variant {
            PostQuantumSignature::Falcon512 => {
                let pk = pqcrypto_falcon::falcon512::PublicKey::from_bytes(public_key)
                    .map_err(|e| anyhow!("Invalid Falcon512 public key: {}", e))?;
                let sig = pqcrypto_falcon::falcon512::DetachedSignature::from_bytes(signature)
                    .map_err(|e| anyhow!("Invalid Falcon512 signature: {}", e))?;
                pqcrypto_falcon::falcon512::verify_detached_signature(&sig, data, &pk).is_ok()
            }
            PostQuantumSignature::Falcon1024 => {
                let pk = pqcrypto_falcon::falcon1024::PublicKey::from_bytes(public_key)
                    .map_err(|e| anyhow!("Invalid Falcon1024 public key: {}", e))?;
                let sig = pqcrypto_falcon::falcon1024::DetachedSignature::from_bytes(signature)
                    .map_err(|e| anyhow!("Invalid Falcon1024 signature: {}", e))?;
                pqcrypto_falcon::falcon1024::verify_detached_signature(&sig, data, &pk).is_ok()
            }
            other => return Err(anyhow!("falcon_verify: unexpected variant {:?}", other)),
        };
        Ok(ok)
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn falcon_verify(
        &self,
        _public_key: &[u8],
        _data: &[u8],
        _signature: &[u8],
        variant: &PostQuantumSignature,
    ) -> Result<bool> {
        Err(anyhow!(
            "Falcon signature verification for {:?} requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum",
            variant
        ))
    }

    /// Encapsulate a shared secret using Kyber KEM.
    ///
    /// Returns `(ciphertext, shared_secret)`. Enable the `post-quantum` feature
    /// for a real cryptographic implementation.
    #[cfg(feature = "post-quantum")]
    async fn kyber_encapsulate(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        use pqcrypto_traits::kem::{Ciphertext as _, PublicKey as _, SharedSecret as _};
        let pk = pqcrypto_kyber::kyber768::PublicKey::from_bytes(public_key)
            .map_err(|e| anyhow!("Invalid Kyber768 public key: {}", e))?;
        let (ss, ct) = pqcrypto_kyber::kyber768::encapsulate(&pk);
        Ok((ct.as_bytes().to_vec(), ss.as_bytes().to_vec()))
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn kyber_encapsulate(&self, _public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        Err(anyhow!(
            "Kyber KEM encapsulation requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum"
        ))
    }

    /// Encapsulate a shared secret using NewHope.
    ///
    /// No pure-Rust or pqcrypto crate for NewHope is available; returns an
    /// informative error.
    async fn newhope_encapsulate(&self, _public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        Err(anyhow!(
            "NewHope KEM encapsulation is not yet supported. \
             No pqcrypto crate for NewHope is currently available."
        ))
    }

    /// Encapsulate a shared secret using FrodoKEM.
    ///
    /// No pure-Rust or pqcrypto crate for FrodoKEM is available; returns an
    /// informative error.
    async fn frodokem_encapsulate(&self, _public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        Err(anyhow!(
            "FrodoKEM encapsulation is not yet supported. \
             No pqcrypto crate for FrodoKEM is currently available."
        ))
    }

    /// Decapsulate using Kyber KEM, returning the shared secret.
    #[cfg(feature = "post-quantum")]
    async fn kyber_decapsulate(&self, private_key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>> {
        use pqcrypto_traits::kem::{Ciphertext as _, SecretKey as _, SharedSecret as _};
        let sk = pqcrypto_kyber::kyber768::SecretKey::from_bytes(private_key)
            .map_err(|e| anyhow!("Invalid Kyber768 secret key: {}", e))?;
        let ct = pqcrypto_kyber::kyber768::Ciphertext::from_bytes(ciphertext)
            .map_err(|e| anyhow!("Invalid Kyber768 ciphertext: {}", e))?;
        let ss = pqcrypto_kyber::kyber768::decapsulate(&ct, &sk);
        Ok(ss.as_bytes().to_vec())
    }

    #[cfg(not(feature = "post-quantum"))]
    async fn kyber_decapsulate(&self, _private_key: &[u8], _ciphertext: &[u8]) -> Result<Vec<u8>> {
        Err(anyhow!(
            "Kyber KEM decapsulation requires the 'post-quantum' feature flag. \
             Rebuild with: cargo build --features post-quantum"
        ))
    }

    /// Decapsulate using NewHope. Returns an error (not yet supported).
    async fn newhope_decapsulate(
        &self,
        _private_key: &[u8],
        _ciphertext: &[u8],
    ) -> Result<Vec<u8>> {
        Err(anyhow!(
            "NewHope KEM decapsulation is not yet supported. \
             No pqcrypto crate for NewHope is currently available."
        ))
    }

    /// Decapsulate using FrodoKEM. Returns an error (not yet supported).
    async fn frodokem_decapsulate(
        &self,
        _private_key: &[u8],
        _ciphertext: &[u8],
    ) -> Result<Vec<u8>> {
        Err(anyhow!(
            "FrodoKEM decapsulation is not yet supported. \
             No pqcrypto crate for FrodoKEM is currently available."
        ))
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> PQCryptoMetrics {
        self.metrics.read().await.clone()
    }
}

/// Post-quantum key pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostQuantumKeyPair {
    pub id: String,
    pub algorithm: PostQuantumSignature,
    pub public_key: Vec<u8>,
    pub private_key: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

/// Signature information for caching
#[derive(Debug, Clone)]
struct SignatureInfo {
    #[allow(dead_code)]
    signature: Vec<u8>,
    #[allow(dead_code)]
    algorithm: PostQuantumSignature,
    #[allow(dead_code)]
    created_at: DateTime<Utc>,
}

/// Post-quantum cryptography metrics
#[derive(Debug, Clone, Default)]
pub struct PQCryptoMetrics {
    pub key_generation_count: u64,
    pub signature_count: u64,
    pub verification_count: u64,
    pub successful_verifications: u64,
    pub avg_key_generation_time_ms: f64,
    pub avg_signing_time_ms: f64,
    pub avg_verification_time_ms: f64,
    pub encapsulation_count: u64,
    pub decapsulation_count: u64,
    pub avg_encapsulation_time_ms: f64,
    pub avg_decapsulation_time_ms: f64,
}
