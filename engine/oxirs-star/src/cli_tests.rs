use anyhow::Result;

use crate::cli::StarCli;
use crate::cli_executor::detect_format;
use crate::parser::StarFormat;

#[test]
fn test_cli_creation() {
    let cli = StarCli::new();
    assert!(!cli.verbose);
    assert!(!cli.quiet);
}

#[test]
fn test_format_detection() -> Result<()> {
    assert_eq!(detect_format("test.ttls", "")?, StarFormat::TurtleStar);
    assert_eq!(detect_format("test.nts", "")?, StarFormat::NTriplesStar);
    assert_eq!(
        detect_format("test.txt", "<< :s :p :o >> :meta :value .")?,
        StarFormat::TurtleStar
    );
    Ok(())
}

#[test]
fn test_namespace_extraction() {
    use crate::cli_executor::extract_namespace;

    assert_eq!(
        extract_namespace("http://example.org/person#name"),
        Some("http://example.org/person#".to_string())
    );

    assert_eq!(
        extract_namespace("http://example.org/data/"),
        Some("http://example.org/data/".to_string())
    );
}
