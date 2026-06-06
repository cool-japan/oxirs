//! `run()` function and subcommand dispatch for aspect_analyzer
use crate::cli::CliResult;
use crate::commands::aspect_analyzer_formats::{from_dtdl, prettyprint, usage, validate};
use crate::{AspectAction, EditAction};
use std::path::PathBuf;

/// Run an Aspect command (Java ESMF SDK compatible)
pub async fn run(action: AspectAction) -> CliResult<()> {
    match action {
        AspectAction::Validate {
            file,
            detailed,
            format,
        } => validate(file, detailed, format).await,
        AspectAction::Prettyprint {
            file,
            output,
            format,
            comments,
        } => prettyprint(file, output, format, comments).await,
        AspectAction::To {
            file,
            format,
            output,
            examples,
            format_variant,
        } => convert(file, format, output, examples, format_variant).await,
        AspectAction::Edit { action } => edit(action).await,
        AspectAction::Usage { input, models_root } => usage(input, models_root).await,
        AspectAction::From {
            file,
            output,
            format,
        } => from_dtdl(file, output, format).await,
    }
}

/// Convert SAMM Aspect model to other formats (Java ESMF SDK compatible)
pub(crate) async fn convert(
    file: PathBuf,
    format: String,
    _output: Option<PathBuf>,
    _examples: bool,
    format_variant: Option<String>,
) -> CliResult<()> {
    use crate::commands::aspect_analyzer_formats::{
        convert_aas, convert_asyncapi, convert_diagram, convert_dtdl, convert_graphql,
        convert_html, convert_java, convert_jsonld, convert_jsonschema, convert_markdown,
        convert_openapi, convert_payload, convert_python, convert_rust, convert_scala, convert_sql,
        convert_typescript,
    };

    let aspect = oxirs_samm::parser::parse_aspect_model(&file)
        .await
        .map_err(|e| format!("SAMM conversion error: {}", e))?;

    match format.as_str() {
        "rust" => convert_rust(&aspect),
        "markdown" => convert_markdown(&aspect),
        "jsonschema" => convert_jsonschema(&aspect),
        "openapi" => convert_openapi(&aspect),
        "asyncapi" => convert_asyncapi(&aspect),
        "html" => convert_html(&aspect),
        "aas" => convert_aas(&aspect, &format_variant),
        "diagram" => convert_diagram(&aspect, &format_variant),
        "sql" => convert_sql(&aspect, &format_variant),
        "jsonld" => convert_jsonld(&aspect),
        "payload" => convert_payload(&aspect, _examples),
        "graphql" => convert_graphql(&aspect),
        "typescript" | "ts" => convert_typescript(&aspect),
        "dtdl" => convert_dtdl(&aspect),
        "python" | "py" => convert_python(&aspect),
        "java" => convert_java(&aspect),
        "scala" => convert_scala(&aspect),
        _ => {
            eprintln!("Error: Unsupported format '{}'", format);
            eprintln!("Supported formats: rust, markdown, jsonschema, openapi, asyncapi, html, aas, diagram, sql, jsonld, payload, graphql, typescript, dtdl, python, java, scala");
            Err(format!("Unsupported format: {}", format).into())
        }
    }
}

/// Edit Aspect model (move elements or create new version)
async fn edit(action: EditAction) -> CliResult<()> {
    match action {
        EditAction::Move {
            file,
            element,
            namespace,
            dry_run,
            details,
            force,
            copy_file_header,
        } => {
            edit_move(
                file,
                element,
                namespace,
                dry_run,
                details,
                force,
                copy_file_header,
            )
            .await
        }
        EditAction::Newversion {
            file,
            major,
            minor,
            micro,
            dry_run,
            details,
            force,
        } => edit_newversion(file, major, minor, micro, dry_run, details, force).await,
    }
}

/// Move element to different namespace
async fn edit_move(
    file: PathBuf,
    element: String,
    namespace: Option<String>,
    dry_run: bool,
    details: bool,
    force: bool,
    copy_file_header: bool,
) -> CliResult<()> {
    use tokio::fs;

    println!("Moving element in Aspect model...");
    println!("  File: {}", file.display());
    println!("  Element: {}", element);

    let content = fs::read_to_string(&file)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let mut current_namespace = String::new();
    let mut current_version = String::new();

    for line in content.lines() {
        if let Some(urn_start) = line.find("urn:samm:") {
            let urn_part = &line[urn_start..];
            if let Some(urn_end) = urn_part.find(['>', ' ', ';']) {
                let urn = &urn_part[..urn_end];
                if let Some(samm_part) = urn.strip_prefix("urn:samm:") {
                    if !samm_part.contains("esmf.samm") {
                        let parts: Vec<&str> = samm_part.splitn(2, '#').collect();
                        let urn_base = parts[0];
                        let ns_version: Vec<&str> = urn_base.rsplitn(2, ':').collect();
                        if ns_version.len() == 2 {
                            current_version = ns_version[0].to_string();
                            current_namespace = ns_version[1].to_string();
                            break;
                        }
                    }
                }
            }
        }
    }

    if current_namespace.is_empty() {
        return Err("Could not determine namespace from file".into());
    }

    let target_namespace = if let Some(ns) = &namespace {
        println!("  Target Namespace: {}", ns);
        ns.clone()
    } else {
        println!(
            "  Target Namespace: {} (same namespace, new file)",
            current_namespace
        );
        current_namespace.clone()
    };

    if dry_run {
        println!("  Mode: DRY RUN");
    }
    if details {
        println!("  Details: ENABLED");
    }
    println!();

    let element_urn_pattern = format!("#{}", element);
    let mut element_found = false;
    let mut element_lines: Vec<String> = Vec::new();
    let mut in_element_block = false;
    let mut brace_depth = 0i32;

    println!("Searching for element '{}'...", element);

    for line in content.lines() {
        if line.contains(&element_urn_pattern) && line.contains("a samm:") {
            element_found = true;
            in_element_block = true;
            element_lines.push(line.to_string());
            brace_depth += line.matches('[').count() as i32;
            brace_depth -= line.matches(']').count() as i32;
            if line.trim().ends_with('.') && brace_depth == 0 {
                in_element_block = false;
            }
            continue;
        }
        if in_element_block {
            element_lines.push(line.to_string());
            brace_depth += line.matches('[').count() as i32;
            brace_depth -= line.matches(']').count() as i32;
            if line.trim().ends_with('.') && brace_depth == 0 {
                in_element_block = false;
            }
        }
    }

    if !element_found {
        return Err(format!("Element '{}' not found in file", element).into());
    }

    println!(
        "✓ Found element '{}' ({} lines)",
        element,
        element_lines.len()
    );
    println!();

    let old_urn = format!(
        "urn:samm:{}:{}#{}",
        current_namespace, current_version, element
    );
    let new_urn = format!(
        "urn:samm:{}:{}#{}",
        target_namespace, current_version, element
    );

    let mut new_element_content = element_lines.join("\n");
    new_element_content = new_element_content.replace(&old_urn, &new_urn);

    if details {
        println!("Element URN Update:");
        println!("  Old: {}", old_urn);
        println!("  New: {}", new_urn);
        println!();
        println!("Element Content ({} lines):", element_lines.len());
        for (i, line) in element_lines.iter().take(5).enumerate() {
            println!("  {}: {}", i + 1, line);
        }
        if element_lines.len() > 5 {
            println!("  ... ({} more lines)", element_lines.len() - 5);
        }
        println!();
    }

    let mut source_lines: Vec<String> = Vec::new();
    let mut skip_lines = false;
    let mut skip_depth = 0i32;

    for line in content.lines() {
        if line.contains(&element_urn_pattern) && line.contains("a samm:") {
            skip_lines = true;
            skip_depth = line.matches('[').count() as i32 - line.matches(']').count() as i32;
            if line.trim().ends_with('.') && skip_depth == 0 {
                skip_lines = false;
            }
            continue;
        }
        if skip_lines {
            skip_depth += line.matches('[').count() as i32;
            skip_depth -= line.matches(']').count() as i32;
            if line.trim().ends_with('.') && skip_depth == 0 {
                skip_lines = false;
            }
            continue;
        }
        source_lines.push(line.to_string());
    }

    let new_source_content = source_lines.join("\n");

    if !dry_run {
        let target_file = if namespace.is_some() {
            let target_dir = PathBuf::from(&target_namespace).join(&current_version);
            let target_filename = format!("{}Element.ttl", element);
            target_dir.join(&target_filename)
        } else {
            let target_filename = format!("{}Element.ttl", element);
            file.parent()
                .unwrap_or(std::path::Path::new("."))
                .join(&target_filename)
        };

        if target_file.exists() && !force {
            return Err(format!(
                "Target file already exists: {} (use --force to overwrite)",
                target_file.display()
            )
            .into());
        }

        let mut target_content = String::new();
        if copy_file_header {
            for line in content.lines() {
                if line.starts_with('@') || line.trim().is_empty() || line.starts_with('#') {
                    target_content.push_str(line);
                    target_content.push('\n');
                } else {
                    break;
                }
            }
            target_content.push('\n');
        } else {
            target_content.push_str(&format!(
                "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:{}#> .\n",
                current_version
            ));
            target_content.push_str(&format!(
                "@prefix : <urn:samm:{}:{}#> .\n\n",
                target_namespace, current_version
            ));
        }

        target_content.push_str(&new_element_content);

        if let Some(parent) = target_file.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        fs::write(&target_file, target_content)
            .await
            .map_err(|e| format!("Failed to write target file: {}", e))?;

        println!("✓ Element moved successfully");
        println!("  Target: {}", target_file.display());

        if force {
            fs::write(&file, new_source_content)
                .await
                .map_err(|e| format!("Failed to update source file: {}", e))?;
            println!("  Source: {} (element removed)", file.display());
        } else {
            println!(
                "  Source: {} (unchanged - use --force to remove element)",
                file.display()
            );
        }
    } else {
        println!("DRY RUN - No files written");
        println!(
            "  Would create: {}/{}/{}Element.ttl",
            target_namespace, current_version, element
        );
        if force {
            println!("  Would update: {} (remove element)", file.display());
        } else {
            println!("  Source unchanged (--force not specified)");
        }
    }

    Ok(())
}

/// Create new version of Aspect model
async fn edit_newversion(
    file: PathBuf,
    major: bool,
    minor: bool,
    micro: bool,
    dry_run: bool,
    details: bool,
    force: bool,
) -> CliResult<()> {
    use std::collections::HashMap;
    use tokio::fs;

    println!("Creating new version of Aspect model...");
    println!("  File: {}", file.display());

    let version_type = if major {
        "MAJOR"
    } else if minor {
        "MINOR"
    } else if micro {
        "MICRO"
    } else {
        "MINOR (default)"
    };
    println!("  Version Update: {}", version_type);

    if dry_run {
        println!("  Mode: DRY RUN");
    }
    if details {
        println!("  Details: ENABLED");
    }
    println!();

    let content = fs::read_to_string(&file)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let mut urn_map: HashMap<String, String> = HashMap::new();
    let mut namespace = String::new();
    let mut old_version = String::new();

    for line in content.lines() {
        if let Some(urn_start) = line.find("urn:samm:") {
            let urn_part = &line[urn_start..];
            if let Some(urn_end) = urn_part.find(['>', ' ', ';']) {
                let urn = &urn_part[..urn_end];
                if let Some(samm_part) = urn.strip_prefix("urn:samm:") {
                    if samm_part.contains("esmf.samm") {
                        continue;
                    }
                    let parts: Vec<&str> = samm_part.splitn(2, '#').collect();
                    let urn_base = parts[0];
                    let ns_version: Vec<&str> = urn_base.rsplitn(2, ':').collect();
                    if ns_version.len() == 2 {
                        let ver = ns_version[0];
                        let ns = ns_version[1];
                        if namespace.is_empty() {
                            namespace = ns.to_string();
                            old_version = ver.to_string();
                        }
                        urn_map.insert(urn.to_string(), format!("{}:{}", ns, ver));
                    }
                }
            }
        }
    }

    if old_version.is_empty() {
        return Err("Could not find version in URN".into());
    }

    let version_parts: Vec<&str> = old_version.split('.').collect();
    if version_parts.len() != 3 {
        return Err(format!("Invalid version format: {} (expected X.Y.Z)", old_version).into());
    }

    let mut ver_major: u32 = version_parts[0]
        .parse()
        .map_err(|_| format!("Invalid major version: {}", version_parts[0]))?;
    let mut ver_minor: u32 = version_parts[1]
        .parse()
        .map_err(|_| format!("Invalid minor version: {}", version_parts[1]))?;
    let mut ver_micro: u32 = version_parts[2]
        .parse()
        .map_err(|_| format!("Invalid micro version: {}", version_parts[2]))?;

    if major {
        ver_major += 1;
        ver_minor = 0;
        ver_micro = 0;
    } else if micro {
        ver_micro += 1;
    } else {
        ver_minor += 1;
        ver_micro = 0;
    }

    let new_version = format!("{}.{}.{}", ver_major, ver_minor, ver_micro);

    println!("Version Update:");
    println!("  {} → {}", old_version, new_version);
    println!("  Namespace: {}", namespace);
    println!();

    let mut new_content = content.clone();
    for (old_urn, ns_ver) in &urn_map {
        let new_urn = old_urn.replace(
            &format!(
                "{}:{}",
                ns_ver.split(':').next_back().unwrap_or(""),
                old_version
            ),
            &format!(
                "{}:{}",
                ns_ver.split(':').next_back().unwrap_or(""),
                new_version
            ),
        );
        new_content = new_content.replace(old_urn, &new_urn);
    }

    if details {
        println!("URN Updates:");
        let mut updates: Vec<_> = urn_map.keys().collect();
        updates.sort();
        for old_urn in updates {
            let ns_ver = &urn_map[old_urn];
            let new_urn = old_urn.replace(
                &format!(
                    "{}:{}",
                    ns_ver.split(':').next_back().unwrap_or(""),
                    old_version
                ),
                &format!(
                    "{}:{}",
                    ns_ver.split(':').next_back().unwrap_or(""),
                    new_version
                ),
            );
            println!("  {} →", old_urn);
            println!("    {}", new_urn);
        }
        println!();
    }

    if !dry_run {
        let filename = file.file_name().ok_or("Invalid filename")?;

        let output_dir = if let Some(parent) = file.parent() {
            if let Some(version_dir) = parent.file_name() {
                if version_dir.to_string_lossy() == old_version {
                    if let Some(ns_dir) = parent.parent() {
                        ns_dir.join(&new_version)
                    } else {
                        parent
                            .parent()
                            .unwrap_or(parent)
                            .join(&namespace)
                            .join(&new_version)
                    }
                } else {
                    parent.join(&namespace).join(&new_version)
                }
            } else {
                parent.join(&namespace).join(&new_version)
            }
        } else {
            PathBuf::from(&namespace).join(&new_version)
        };

        let output_file = output_dir.join(filename);

        if output_file.exists() && !force {
            return Err(format!(
                "Output file already exists: {} (use --force to overwrite)",
                output_file.display()
            )
            .into());
        }

        fs::create_dir_all(&output_dir)
            .await
            .map_err(|e| format!("Failed to create directory: {}", e))?;

        fs::write(&output_file, new_content)
            .await
            .map_err(|e| format!("Failed to write file: {}", e))?;

        println!("✓ New version created successfully");
        println!("  Output: {}", output_file.display());
    } else {
        println!("DRY RUN - No files written");
        println!(
            "  Would create: {}/{}/{}",
            namespace,
            new_version,
            file.file_name()
                .expect("file path should have a file name")
                .to_string_lossy()
        );
    }

    Ok(())
}
