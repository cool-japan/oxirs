//! Backup Encryption Module
//!
//! Provides AES-256-GCM encryption for database backups with Argon2 key derivation.
//! Supports password-based and keyfile-based encryption for secure backup storage.

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use argon2::{
    password_hash::{PasswordHasher, SaltString},
    Argon2,
};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Encryption configuration
pub struct EncryptionConfig {
    pub password: Option<String>,
    pub keyfile: Option<PathBuf>,
    pub verify: bool,
}

/// Encryption metadata stored with the backup
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct EncryptionMetadata {
    pub version: u8,
    pub algorithm: String,
    pub key_derivation: String,
    pub salt: String,
    pub nonce: Vec<u8>,
    pub created_at: String,
}

/// Encrypt a backup file using AES-256-GCM
pub fn encrypt_backup(
    source: &Path,
    target: &Path,
    config: &EncryptionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Derive encryption key and get salt
    let (key, salt_b64) = derive_key_with_salt(config, None)?;

    // Read source file
    let mut source_file = File::open(source)?;
    let mut plaintext = Vec::new();
    source_file.read_to_end(&mut plaintext)?;

    // Generate random nonce (96 bits for GCM)
    let cipher = Aes256Gcm::new(&key);
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    // Encrypt data
    let ciphertext = cipher
        .encrypt(&nonce, plaintext.as_ref())
        .map_err(|e| format!("Encryption failed: {}", e))?;

    // Create metadata
    let metadata = EncryptionMetadata {
        version: 1,
        algorithm: "AES-256-GCM".to_string(),
        key_derivation: if config.password.is_some() {
            "Argon2id".to_string()
        } else {
            "Keyfile".to_string()
        },
        salt: salt_b64,
        nonce: nonce.to_vec(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    // Write encrypted file with metadata
    write_encrypted_file(target, &metadata, &ciphertext)?;

    // Verify if requested
    if config.verify {
        verify_encrypted_file(target, config)?;
    }

    Ok(())
}

/// Decrypt a backup file
pub fn decrypt_backup(
    source: &Path,
    target: &Path,
    config: &EncryptionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read encrypted file and metadata
    let (metadata, ciphertext) = read_encrypted_file(source)?;

    // Verify algorithm version
    if metadata.version != 1 || metadata.algorithm != "AES-256-GCM" {
        return Err(format!(
            "Unsupported encryption version or algorithm: v{} {}",
            metadata.version, metadata.algorithm
        )
        .into());
    }

    // Derive decryption key using salt from metadata
    let (key, _) = derive_key_with_salt(config, Some(&metadata.salt))?;

    // Prepare nonce
    if metadata.nonce.len() != 12 {
        return Err("Invalid nonce length".into());
    }
    let nonce = Nonce::from_slice(&metadata.nonce);

    // Decrypt data
    let cipher = Aes256Gcm::new(&key);
    let plaintext = cipher
        .decrypt(nonce, ciphertext.as_ref())
        .map_err(|e| format!("Decryption failed: {}. Check your password/keyfile.", e))?;

    // Write decrypted file
    let mut target_file = File::create(target)?;
    target_file.write_all(&plaintext)?;

    Ok(())
}

/// Derive encryption key from password or keyfile with optional salt
/// Returns (key, salt_base64)
fn derive_key_with_salt(
    config: &EncryptionConfig,
    salt_b64: Option<&str>,
) -> Result<(Key<Aes256Gcm>, String), Box<dyn std::error::Error>> {
    if let Some(ref password) = config.password {
        // Password-based key derivation
        derive_key_from_password(password, salt_b64)
    } else if let Some(ref keyfile) = config.keyfile {
        // Keyfile-based key derivation
        let key = derive_key_from_keyfile(keyfile)?;
        let salt_str = format!("keyfile:{}", keyfile.display());
        Ok((key, salt_str))
    } else {
        Err("Either password or keyfile must be provided".into())
    }
}

/// Derive key from password using Argon2id with optional salt
/// Returns (key, salt_base64)
fn derive_key_from_password(
    password: &str,
    salt_b64_opt: Option<&str>,
) -> Result<(Key<Aes256Gcm>, String), Box<dyn std::error::Error>> {
    use scirs2_core::random::rng;
    use scirs2_core::Rng;

    // Get or generate salt
    let salt = if let Some(salt_b64) = salt_b64_opt {
        // Decode existing salt
        SaltString::from_b64(salt_b64).map_err(|e| format!("Failed to decode salt: {}", e))?
    } else {
        // Generate new salt
        let mut rand_gen = rng();
        let salt_bytes: Vec<u8> = (0..16).map(|_| rand_gen.random::<u8>()).collect();
        SaltString::encode_b64(&salt_bytes).map_err(|e| format!("Failed to encode salt: {}", e))?
    };

    // Use Argon2id for key derivation
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| format!("Failed to hash password: {}", e))?;

    // Extract the hash output (first 32 bytes for AES-256)
    let hash_output = password_hash.hash.ok_or("Failed to extract hash output")?;
    let key_bytes = hash_output.as_bytes();

    if key_bytes.len() < 32 {
        return Err("Key derivation produced insufficient bytes".into());
    }

    let key = Key::<Aes256Gcm>::from_slice(&key_bytes[..32]);
    Ok((*key, salt.to_string()))
}

/// Derive key from keyfile
fn derive_key_from_keyfile(keyfile: &Path) -> Result<Key<Aes256Gcm>, Box<dyn std::error::Error>> {
    // Read keyfile
    let mut file = File::open(keyfile)?;
    let mut key_data = Vec::new();
    file.read_to_end(&mut key_data)?;

    // Keyfile must be exactly 32 bytes for AES-256
    if key_data.len() != 32 {
        // If not exactly 32 bytes, hash it to get 32 bytes
        use ring::digest;
        let hash = digest::digest(&digest::SHA256, &key_data);
        let key = Key::<Aes256Gcm>::from_slice(hash.as_ref());
        Ok(*key)
    } else {
        let key = Key::<Aes256Gcm>::from_slice(&key_data);
        Ok(*key)
    }
}

/// Write encrypted file with metadata header
fn write_encrypted_file(
    path: &Path,
    metadata: &EncryptionMetadata,
    ciphertext: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;

    // Write magic header
    file.write_all(b"OXIRS_ENCRYPTED_BACKUP_V1\n")?;

    // Write metadata as JSON
    let metadata_json = serde_json::to_string(metadata)?;
    let metadata_bytes = metadata_json.as_bytes();
    file.write_all(&(metadata_bytes.len() as u32).to_le_bytes())?;
    file.write_all(metadata_bytes)?;

    // Write ciphertext
    file.write_all(ciphertext)?;

    // Ensure all data is flushed to disk
    file.flush()?;
    file.sync_all()?;

    Ok(())
}

/// Read encrypted file and extract metadata
fn read_encrypted_file(
    path: &Path,
) -> Result<(EncryptionMetadata, Vec<u8>), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;

    // Read and verify magic header (26 bytes: "OXIRS_ENCRYPTED_BACKUP_V1\n")
    let mut magic = vec![0u8; 26];
    file.read_exact(&mut magic)?;
    if magic != b"OXIRS_ENCRYPTED_BACKUP_V1\n" {
        return Err("Invalid encrypted backup file: bad magic header".into());
    }

    // Read metadata length
    let mut len_bytes = [0u8; 4];
    file.read_exact(&mut len_bytes)?;
    let metadata_len = u32::from_le_bytes(len_bytes) as usize;

    // Read metadata
    let mut metadata_bytes = vec![0u8; metadata_len];
    file.read_exact(&mut metadata_bytes)?;
    let metadata: EncryptionMetadata = serde_json::from_slice(&metadata_bytes)?;

    // Read ciphertext
    let mut ciphertext = Vec::new();
    file.read_to_end(&mut ciphertext)?;

    Ok((metadata, ciphertext))
}

/// Verify encrypted file integrity
fn verify_encrypted_file(
    path: &Path,
    _config: &EncryptionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read and parse the file to ensure it's valid
    let (_metadata, _ciphertext) = read_encrypted_file(path)?;

    // Basic verification passed
    Ok(())
}

/// Generate a new keyfile with random 256-bit key
pub fn generate_keyfile(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::random::rng;
    use scirs2_core::Rng;
    let mut rand_gen = rng();

    // Generate 32 random bytes (256 bits)
    let key_bytes: Vec<u8> = (0..32).map(|_| rand_gen.random::<u8>()).collect();

    // Write to file
    let mut file = File::create(path)?;
    file.write_all(&key_bytes)?;

    // Set restrictive permissions on Unix systems
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path)?.permissions();
        perms.set_mode(0o600); // Read/write for owner only
        fs::set_permissions(path, perms)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_password_encryption_decryption() {
        let temp_dir = temp_dir();
        let source = temp_dir.join("test_source.txt");
        let encrypted = temp_dir.join("test_encrypted.oxirs");
        let decrypted = temp_dir.join("test_decrypted.txt");

        // Create test file
        let test_data = b"This is a test backup file with sensitive data!";
        fs::write(&source, test_data).unwrap();

        // Encrypt
        let encrypt_config = EncryptionConfig {
            password: Some("test_password_123".to_string()),
            keyfile: None,
            verify: true,
        };
        encrypt_backup(&source, &encrypted, &encrypt_config).unwrap();

        // Verify encrypted file exists and is different
        assert!(encrypted.exists());
        let encrypted_data = fs::read(&encrypted).unwrap();
        assert_ne!(encrypted_data.as_slice(), test_data);

        // Decrypt
        let decrypt_config = EncryptionConfig {
            password: Some("test_password_123".to_string()),
            keyfile: None,
            verify: false,
        };
        decrypt_backup(&encrypted, &decrypted, &decrypt_config).unwrap();

        // Verify decrypted data matches original
        let decrypted_data = fs::read(&decrypted).unwrap();
        assert_eq!(decrypted_data.as_slice(), test_data);

        // Cleanup
        let _ = fs::remove_file(source);
        let _ = fs::remove_file(encrypted);
        let _ = fs::remove_file(decrypted);
    }

    #[test]
    fn test_keyfile_generation_and_encryption() {
        let temp_dir = temp_dir();
        let keyfile = temp_dir.join("test_keyfile.key");
        let source = temp_dir.join("test_source2.txt");
        let encrypted = temp_dir.join("test_encrypted2.oxirs");
        let decrypted = temp_dir.join("test_decrypted2.txt");

        // Generate keyfile
        generate_keyfile(&keyfile).unwrap();
        assert!(keyfile.exists());
        let key_data = fs::read(&keyfile).unwrap();
        assert_eq!(key_data.len(), 32);

        // Create test file
        let test_data = b"Keyfile-based encryption test data!";
        fs::write(&source, test_data).unwrap();

        // Encrypt with keyfile
        let encrypt_config = EncryptionConfig {
            password: None,
            keyfile: Some(keyfile.clone()),
            verify: true,
        };
        encrypt_backup(&source, &encrypted, &encrypt_config).unwrap();

        // Decrypt with keyfile
        let decrypt_config = EncryptionConfig {
            password: None,
            keyfile: Some(keyfile.clone()),
            verify: false,
        };
        decrypt_backup(&encrypted, &decrypted, &decrypt_config).unwrap();

        // Verify
        let decrypted_data = fs::read(&decrypted).unwrap();
        assert_eq!(decrypted_data.as_slice(), test_data);

        // Cleanup
        let _ = fs::remove_file(keyfile);
        let _ = fs::remove_file(source);
        let _ = fs::remove_file(encrypted);
        let _ = fs::remove_file(decrypted);
    }

    #[test]
    fn test_wrong_password_fails() {
        let temp_dir = temp_dir();
        let source = temp_dir.join("test_source3.txt");
        let encrypted = temp_dir.join("test_encrypted3.oxirs");
        let decrypted = temp_dir.join("test_decrypted3.txt");

        // Create and encrypt
        fs::write(&source, b"Secret data").unwrap();
        let encrypt_config = EncryptionConfig {
            password: Some("correct_password".to_string()),
            keyfile: None,
            verify: false,
        };
        encrypt_backup(&source, &encrypted, &encrypt_config).unwrap();

        // Try to decrypt with wrong password
        let decrypt_config = EncryptionConfig {
            password: Some("wrong_password".to_string()),
            keyfile: None,
            verify: false,
        };
        let result = decrypt_backup(&encrypted, &decrypted, &decrypt_config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Decryption failed"));

        // Cleanup
        let _ = fs::remove_file(source);
        let _ = fs::remove_file(encrypted);
    }
}
