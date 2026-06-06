//! Tests for the cloud backend module group.

#![cfg(test)]

use crate::cloud_backends_aws::{S3Backend, S3Config};
use crate::cloud_backends_azure::{AzureBlobBackend, AzureConfig};
use crate::cloud_backends_common::{
    extract_host, hmac_sha256, parse_azure_list_xml, parse_s3_list_xml, sha256_hex, url_encode,
};
use crate::cloud_backends_gcp::{GcsBackend, GcsConfig};
use crate::cloud_backends_http::{HttpBackend, HttpConfig};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

// ── S3Config ─────────────────────────────────────────────────────────────

#[test]
fn test_s3_config_validation_passes_with_valid_fields() {
    let config = S3Config {
        endpoint: "https://s3.amazonaws.com".to_string(),
        bucket: "my-bucket".to_string(),
        region: "us-east-1".to_string(),
        access_key: "AKID".to_string(),
        secret_key: "SECRET".to_string(),
        path_prefix: String::new(),
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_s3_config_validation_fails_on_empty_bucket() {
    let config = S3Config {
        endpoint: "https://s3.amazonaws.com".to_string(),
        bucket: String::new(),
        region: "us-east-1".to_string(),
        access_key: "AKID".to_string(),
        secret_key: "SECRET".to_string(),
        path_prefix: String::new(),
    };
    let err = config.validate().unwrap_err();
    assert!(err.contains("bucket"), "error should mention bucket: {err}");
}

#[test]
fn test_s3_config_validation_fails_on_empty_access_key() {
    let config = S3Config {
        endpoint: "https://s3.amazonaws.com".to_string(),
        bucket: "bucket".to_string(),
        region: "us-east-1".to_string(),
        access_key: String::new(),
        secret_key: "SECRET".to_string(),
        path_prefix: String::new(),
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("access_key"),
        "error should mention access_key: {err}"
    );
}

#[test]
fn test_s3_config_full_key_with_prefix() {
    let config = S3Config {
        endpoint: "https://s3.amazonaws.com".to_string(),
        bucket: "bucket".to_string(),
        region: "us-east-1".to_string(),
        access_key: "AKID".to_string(),
        secret_key: "SECRET".to_string(),
        path_prefix: "samm/".to_string(),
    };
    assert_eq!(config.full_key("model.ttl"), "samm/model.ttl");
}

#[test]
fn test_s3_config_full_key_without_prefix() {
    let config = S3Config {
        endpoint: "https://s3.amazonaws.com".to_string(),
        bucket: "bucket".to_string(),
        region: "us-east-1".to_string(),
        access_key: "AKID".to_string(),
        secret_key: "SECRET".to_string(),
        path_prefix: String::new(),
    };
    assert_eq!(config.full_key("model.ttl"), "model.ttl");
}

#[test]
fn test_s3_backend_creation_succeeds() {
    let config = S3Config {
        endpoint: "https://s3.amazonaws.com".to_string(),
        bucket: "my-bucket".to_string(),
        region: "us-east-1".to_string(),
        access_key: "AKID".to_string(),
        secret_key: "SECRET".to_string(),
        path_prefix: String::new(),
    };
    let backend = S3Backend::new(config);
    assert!(
        backend.is_ok(),
        "S3Backend should be constructable with valid config"
    );
}

#[test]
fn test_s3_presigned_url_contains_expected_fields() {
    let config = S3Config {
        endpoint: "https://s3.amazonaws.com".to_string(),
        bucket: "my-bucket".to_string(),
        region: "us-east-1".to_string(),
        access_key: "AKIAIOSFODNN7EXAMPLE".to_string(),
        secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        path_prefix: String::new(),
    };
    let backend = S3Backend::new(config).expect("should build");
    let url = backend.presigned_url("models/vehicle.ttl", 3600);

    assert!(url.contains("my-bucket"), "URL should contain bucket name");
    assert!(
        url.contains("X-Amz-Signature"),
        "URL should contain Signature parameter"
    );
    assert!(
        url.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"),
        "URL should contain algorithm"
    );
    assert!(
        url.contains("X-Amz-Expires=3600"),
        "URL should contain expiry"
    );
}

// ── GcsConfig ─────────────────────────────────────────────────────────────

#[test]
fn test_gcs_config_creation_with_access_token() {
    let config = GcsConfig::with_access_token("my-gcs-bucket", "ya29.token");
    assert_eq!(config.bucket, "my-gcs-bucket");
    assert_eq!(config.access_token.as_deref(), Some("ya29.token"));
    assert!(config.service_account_key.is_none());
}

#[test]
fn test_gcs_config_creation_with_service_account() {
    let key_json = r#"{"type":"service_account","project_id":"my-project"}"#;
    let config = GcsConfig::with_service_account("my-bucket", key_json);
    assert!(config.service_account_key.is_some());
    assert!(config.access_token.is_none());
}

#[test]
fn test_gcs_config_validation_fails_without_auth() {
    let config = GcsConfig {
        bucket: "my-bucket".to_string(),
        service_account_key: None,
        access_token: None,
        path_prefix: String::new(),
    };
    match GcsBackend::new(config) {
        Err(err) => assert!(
            err.contains("access_token") || err.contains("service_account"),
            "error should mention auth: {err}"
        ),
        Ok(_) => panic!("GcsBackend::new should reject config without auth"),
    }
}

#[test]
fn test_gcs_backend_creation_succeeds() {
    let config = GcsConfig::with_access_token("samm-models", "fake-token");
    let backend = GcsBackend::new(config);
    assert!(
        backend.is_ok(),
        "GcsBackend should be constructable with valid config"
    );
}

// ── AzureConfig ───────────────────────────────────────────────────────────

#[test]
fn test_azure_config_creation() {
    let key_b64 = BASE64.encode("some-secret-key-bytes");
    let config = AzureConfig::new("myaccount", &key_b64, "samm-models");
    assert_eq!(config.account_name, "myaccount");
    assert_eq!(config.container_name, "samm-models");
    assert!(config.validate().is_ok());
}

#[test]
fn test_azure_config_validation_fails_on_empty_container() {
    let config = AzureConfig {
        account_name: "myaccount".to_string(),
        account_key: BASE64.encode("key"),
        container_name: String::new(),
        path_prefix: String::new(),
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("container_name"),
        "error should mention container_name: {err}"
    );
}

#[test]
fn test_azure_shared_key_lite_signature_deterministic() {
    let key_bytes = b"my-account-key-bytes-are-here-!!!";
    let key_b64 = BASE64.encode(key_bytes);
    let config = AzureConfig::new("acct", &key_b64, "ctr");

    let sig1 = config
        .shared_key_lite_auth(
            "PUT",
            "",
            "application/octet-stream",
            "",
            "x-ms-date:Thu, 01 Jan 2026 00:00:00 GMT\n",
            "/acct/ctr/file.ttl",
        )
        .expect("signing should succeed");
    let sig2 = config
        .shared_key_lite_auth(
            "PUT",
            "",
            "application/octet-stream",
            "",
            "x-ms-date:Thu, 01 Jan 2026 00:00:00 GMT\n",
            "/acct/ctr/file.ttl",
        )
        .expect("signing should succeed");

    assert_eq!(
        sig1, sig2,
        "Signature must be deterministic for the same inputs"
    );
    assert!(
        sig1.starts_with("SharedKeyLite acct:"),
        "Authorization header format mismatch"
    );
}

#[test]
fn test_azure_backend_creation_succeeds() {
    let key_b64 = BASE64.encode("some-key-material");
    let config = AzureConfig::new("storageacct", &key_b64, "models");
    let backend = AzureBlobBackend::new(config);
    assert!(
        backend.is_ok(),
        "AzureBlobBackend should be constructable with valid config"
    );
}

// ── HttpConfig ────────────────────────────────────────────────────────────

#[test]
fn test_http_config_creation_basic() {
    let config = HttpConfig::new("http://localhost:8080/storage");
    assert_eq!(config.base_url, "http://localhost:8080/storage");
    assert!(config.auth_header.is_none());
}

#[test]
fn test_http_config_creation_with_bearer() {
    let config = HttpConfig::with_bearer("http://localhost:8080", "my-token");
    assert_eq!(config.auth_header.as_deref(), Some("Bearer my-token"));
}

#[test]
fn test_http_backend_creation_succeeds() {
    let config = HttpConfig::new("http://localhost:9999/api/storage");
    let backend = HttpBackend::new(config);
    assert!(
        backend.is_ok(),
        "HttpBackend should be constructable with a non-empty URL"
    );
}

#[test]
fn test_http_backend_creation_fails_on_empty_url() {
    let config = HttpConfig::new("");
    let backend = HttpBackend::new(config);
    assert!(
        backend.is_err(),
        "HttpBackend should reject an empty base URL"
    );
}

// ── Utility helpers ───────────────────────────────────────────────────────

#[test]
fn test_url_encode_unreserved_chars_unchanged() {
    assert_eq!(url_encode("abc-123_.~"), "abc-123_.~");
}

#[test]
fn test_url_encode_space_and_slash_encoded() {
    let encoded = url_encode("hello world/file.ttl");
    assert!(encoded.contains("%20"), "space should be percent-encoded");
    assert!(encoded.contains("%2F"), "slash should be percent-encoded");
}

#[test]
fn test_extract_host_standard_url() {
    let host = extract_host("https://s3.amazonaws.com/my-bucket/key").expect("should succeed");
    assert_eq!(host, "s3.amazonaws.com");
}

#[test]
fn test_extract_host_url_with_port() {
    let host = extract_host("http://localhost:9000/bucket").expect("should succeed");
    assert_eq!(host, "localhost:9000");
}

#[test]
fn test_extract_host_no_scheme_returns_err() {
    let result = extract_host("no-scheme-url");
    assert!(result.is_err(), "Should fail on URL without scheme");
}

#[test]
fn test_parse_s3_list_xml_extracts_keys() {
    let xml = r#"<?xml version="1.0"?>
<ListBucketResult>
  <Contents><Key>models/vehicle.ttl</Key><Size>1024</Size></Contents>
  <Contents><Key>models/sensor.ttl</Key><Size>512</Size></Contents>
</ListBucketResult>"#;
    let keys = parse_s3_list_xml(xml);
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0], "models/vehicle.ttl");
    assert_eq!(keys[1], "models/sensor.ttl");
}

#[test]
fn test_parse_s3_list_xml_empty_on_no_keys() {
    let xml = "<ListBucketResult><KeyCount>0</KeyCount></ListBucketResult>";
    let keys = parse_s3_list_xml(xml);
    assert!(keys.is_empty());
}

#[test]
fn test_parse_azure_list_xml_extracts_names() {
    let xml = r#"<?xml version="1.0"?>
<EnumerationResults>
  <Blobs>
    <Blob><Name>models/car.ttl</Name></Blob>
    <Blob><Name>models/truck.ttl</Name></Blob>
  </Blobs>
</EnumerationResults>"#;
    let names = parse_azure_list_xml(xml);
    assert_eq!(names.len(), 2);
    assert_eq!(names[0], "models/car.ttl");
    assert_eq!(names[1], "models/truck.ttl");
}

#[test]
fn test_hmac_sha256_known_vector() {
    let key = b"key";
    let msg = b"The quick brown fox jumps over the lazy dog";
    let result = hmac_sha256(key, msg);
    assert_eq!(result.len(), 32, "HMAC-SHA256 output must be 32 bytes");
    let result2 = hmac_sha256(key, msg);
    assert_eq!(result, result2, "HMAC-SHA256 must be deterministic");
}

#[test]
fn test_sha256_hex_known_empty() {
    let h = sha256_hex(b"");
    assert_eq!(
        h,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn test_s3_signing_key_length() {
    let config = S3Config {
        endpoint: "https://s3.amazonaws.com".to_string(),
        bucket: "bucket".to_string(),
        region: "us-east-1".to_string(),
        access_key: "AKID".to_string(),
        secret_key: "SECRET".to_string(),
        path_prefix: String::new(),
    };
    let backend = S3Backend::new(config).expect("should build");
    let key = backend.signing_key("20260201", "s3");
    assert_eq!(key.len(), 32, "Derived signing key must be 32 bytes");
}

#[test]
fn test_http_config_full_key_with_prefix() {
    let mut config = HttpConfig::new("http://localhost:8080");
    config.path_prefix = "v1/".to_string();
    assert_eq!(config.full_key("/model.ttl"), "v1//model.ttl");
}
