//! Enhanced Riot - RDF parsing and serialization tool
//!
//! Improved version with format detection, validation, and better error handling.

use super::{format::*, utils, ToolResult, ToolStats};
use crate::cli::{CliContext, progress::helpers as progress_helpers};
use std::path::PathBuf;
use indicatif::ProgressBar;

/// Enhanced riot command with progress and validation
pub async fn run_enhanced(
    ctx: &CliContext,
    input: Vec<PathBuf>,
    output_format: String,
    output_file: Option<PathBuf>,
    syntax: Option<String>,
    base: Option<String>,
    validate: bool,
    count: bool,
    streaming: bool,
    parallel: bool,
) -> ToolResult {
    let mut stats = ToolStats::new();

    ctx.section("RDF I/O Tool (riot)");
    ctx.highlight("Input files", &format!("{}", input.len()));
    ctx.highlight("Output format", &output_format);

    // Validate output format
    let target_format = match output_format.as_str() {
        "turtle" | "ttl" => RdfFormat::Turtle,
        "ntriples" | "nt" => RdfFormat::NTriples,
        "rdfxml" | "rdf" | "xml" => RdfFormat::RdfXml,
        "jsonld" | "json-ld" => RdfFormat::JsonLd,
        "trig" => RdfFormat::TriG,
        "nquads" | "nq" => RdfFormat::NQuads,
        _ => {
            return Err(crate::cli::error::helpers::invalid_rdf_format_error(
                &output_format,
                &["turtle", "ntriples", "rdfxml", "jsonld", "trig", "nquads"],
            ).into());
        }
    };

    if let Some(ref base_uri) = base {
        ctx.highlight("Base URI", base_uri);
        utils::validate_iri(base_uri)?;
    }

    // Create progress bar
    let pb = if ctx.should_show_output() {
        Some(progress_helpers::file_progress(input.len() as u64))
    } else {
        None
    };

    let mut total_triples = 0;
    let mut total_errors = 0;
    let mut all_output = String::new();

    // Process files
    for (i, input_file) in input.iter().enumerate() {
        if let Some(ref pb) = pb {
            pb.set_message(format!("Processing {}", input_file.display()));
        }

        ctx.verbose(&format!(
            "Processing file {}/{}: {}",
            i + 1,
            input.len(),
            input_file.display()
        ));

        // Check file readability
        utils::check_file_readable(input_file)?;

        // Detect input format
        let detected_formats = FormatDetector::detect_file(input_file).await?;
        
        let input_format = if let Some(ref specified) = syntax {
            // Use specified format
            match specified.as_str() {
                "turtle" | "ttl" => RdfFormat::Turtle,
                "ntriples" | "nt" => RdfFormat::NTriples,
                "rdfxml" | "rdf" | "xml" => RdfFormat::RdfXml,
                "jsonld" | "json-ld" => RdfFormat::JsonLd,
                "trig" => RdfFormat::TriG,
                "nquads" | "nq" => RdfFormat::NQuads,
                _ => {
                    ctx.warn(&format!("Unknown format '{}', using auto-detection", specified));
                    detected_formats.first()
                        .map(|d| d.format)
                        .unwrap_or(RdfFormat::Turtle)
                }
            }
        } else if let Some(detection) = detected_formats.first() {
            ctx.verbose(&format!(
                "Auto-detected format: {} (confidence: {:.0}%)",
                detection.format.name(),
                detection.confidence * 100.0
            ));
            detection.format
        } else {
            ctx.warn("Could not detect format, assuming Turtle");
            RdfFormat::Turtle
        };

        ctx.verbose(&format!("Input format: {}", input_format.name()));

        // Read and process file
        let content = utils::read_input(input_file)?;

        // Validate if requested
        if validate || count {
            let validation_result = FormatValidator::validate(&content, input_format)?;
            
            if !validation_result.valid {
                ctx.error(&format!(
                    "Validation failed for {}: {} errors",
                    input_file.display(),
                    validation_result.errors.len()
                ));
                
                for error in &validation_result.errors {
                    if let Some(line) = error.line {
                        ctx.error(&format!("  Line {}: {}", line, error.message));
                    } else {
                        ctx.error(&format!("  {}", error.message));
                    }
                }
                
                stats.errors += validation_result.errors.len();
                total_errors += validation_result.errors.len();
            }

            for warning in &validation_result.warnings {
                ctx.warn(&format!("  {}", warning));
                stats.warnings += 1;
            }

            total_triples += validation_result.stats.triple_count;
            stats.items_processed += validation_result.stats.triple_count;

            if validate {
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                continue; // Skip conversion for validation-only mode
            }
        }

        // Convert format if needed
        if !validate {
            // Parse and convert (placeholder for actual implementation)
            match process_rdf_enhanced(
                &content,
                input_format,
                target_format,
                base.as_deref(),
                streaming,
            ) {
                Ok(result) => {
                    if count {
                        ctx.info(&format!("  Triples/Quads: {}", result.triple_count));
                        total_triples += result.triple_count;
                    } else {
                        if !result.output.is_empty() {
                            if input.len() > 1 {
                                all_output.push_str(&format!("# File: {}\n", input_file.display()));
                            }
                            all_output.push_str(&result.output);
                            all_output.push('\n');
                        }
                    }
                    stats.items_processed += result.triple_count;
                }
                Err(e) => {
                    ctx.error(&format!("Error processing {}: {}", input_file.display(), e));
                    stats.errors += 1;
                }
            }
        }

        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message("Processing complete");
    }

    // Output results
    if validate {
        ctx.section("Validation Results");
        ctx.key_value("Total triples/quads", &total_triples.to_string());
        
        if total_errors > 0 {
            ctx.error(&format!("Total errors: {}", total_errors));
            return Err(format!("Validation failed with {} errors", total_errors).into());
        } else {
            ctx.success("All files are valid");
        }
    } else if count {
        ctx.section("Count Results");
        ctx.key_value("Total triples/quads", &total_triples.to_string());
    } else if !all_output.is_empty() {
        utils::write_output(&all_output, output_file.as_deref())?;
        
        if let Some(ref output_path) = output_file {
            ctx.success(&format!("Output written to: {}", output_path.display()));
        }
    }

    stats.finish();
    if ctx.should_show_verbose() {
        stats.print_summary("Riot");
    }

    Ok(())
}

/// Result of processing an RDF file
struct ProcessResult {
    triple_count: usize,
    output: String,
}

/// Process RDF content with enhanced features
fn process_rdf_enhanced(
    content: &str,
    from_format: RdfFormat,
    to_format: RdfFormat,
    base_uri: Option<&str>,
    _streaming: bool,
) -> ToolResult<ProcessResult> {
    // This is a placeholder - actual implementation would use proper RDF parsing
    
    // For now, just do basic line counting for line-based formats
    let triple_count = if from_format.is_line_based() {
        content.lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with('#')
            })
            .count()
    } else {
        // Estimate for other formats
        content.lines().count() / 3
    };

    // Placeholder conversion
    let output = if from_format == to_format {
        content.to_string()
    } else {
        format!(
            "# Conversion from {} to {} not yet implemented\n# Original content follows:\n{}",
            from_format.name(),
            to_format.name(),
            content
        )
    };

    Ok(ProcessResult {
        triple_count,
        output,
    })
}

/// Streaming processor for large files
pub struct StreamingProcessor {
    format: RdfFormat,
    buffer_size: usize,
}

impl StreamingProcessor {
    pub fn new(format: RdfFormat) -> Self {
        Self {
            format,
            buffer_size: 8192, // 8KB default
        }
    }

    pub async fn process_file(
        &self,
        input_path: &Path,
        output_path: &Path,
        transform: impl Fn(&str) -> ToolResult<String>,
    ) -> ToolResult<usize> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::fs::File;

        let input_file = File::open(input_path).await?;
        let mut reader = BufReader::new(input_file);
        let mut output_file = File::create(output_path).await?;

        let mut line_count = 0;
        let mut buffer = String::new();

        if self.format.is_line_based() {
            // Process line by line for line-based formats
            while reader.read_line(&mut buffer).await? > 0 {
                let transformed = transform(&buffer)?;
                output_file.write_all(transformed.as_bytes()).await?;
                line_count += 1;
                buffer.clear();
            }
        } else {
            // Process in chunks for other formats
            let mut chunk = vec![0; self.buffer_size];
            while let Ok(n) = reader.read(&mut chunk).await {
                if n == 0 {
                    break;
                }
                let content = String::from_utf8_lossy(&chunk[..n]);
                let transformed = transform(&content)?;
                output_file.write_all(transformed.as_bytes()).await?;
            }
        }

        output_file.flush().await?;
        Ok(line_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_mapping() {
        assert_eq!(
            match "turtle" {
                "turtle" | "ttl" => RdfFormat::Turtle,
                _ => panic!("Unknown format"),
            },
            RdfFormat::Turtle
        );
    }

    #[test]
    fn test_process_result() {
        let result = ProcessResult {
            triple_count: 10,
            output: "test output".to_string(),
        };
        assert_eq!(result.triple_count, 10);
        assert_eq!(result.output, "test output");
    }
}