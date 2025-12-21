//! Hash functions for SPARQL 1.2

use crate::model::{Literal, Term};
use crate::OxirsError;

/// SHA1 - Compute SHA-1 hash
pub(super) fn fn_sha1(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SHA1 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            use sha1::{Digest, Sha1};
            let mut hasher = Sha1::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query(
            "SHA1 requires string literal".to_string(),
        )),
    }
}

/// SHA256 - Compute SHA-256 hash
pub(super) fn fn_sha256(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SHA256 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query(
            "SHA256 requires string literal".to_string(),
        )),
    }
}

/// SHA384 - Compute SHA-384 hash
pub(super) fn fn_sha384(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SHA384 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Digest, Sha384};
            let mut hasher = Sha384::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query(
            "SHA384 requires string literal".to_string(),
        )),
    }
}

/// SHA512 - Compute SHA-512 hash
pub(super) fn fn_sha512(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SHA512 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            use sha2::{Digest, Sha512};
            let mut hasher = Sha512::new();
            hasher.update(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query(
            "SHA512 requires string literal".to_string(),
        )),
    }
}

/// MD5 - Compute MD5 hash
pub(super) fn fn_md5(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "MD5 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let mut hasher = md5::Context::new();
            hasher.consume(lit.value().as_bytes());
            let result = hasher.finalize();
            let hex = format!("{result:x}");
            Ok(Term::Literal(Literal::new(&hex)))
        }
        _ => Err(OxirsError::Query("MD5 requires string literal".to_string())),
    }
}
