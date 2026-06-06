//! Format-specific handlers: JSON, YAML, XML, AAS submodel, DTDL format processing
use crate::cli::CliResult;
use crate::commands::aspect_analyzer_types::{
    map_xsd_to_json_schema, map_xsd_to_rust, to_snake_case,
};
use oxirs_samm::metamodel::ModelElement;
use oxirs_samm::parser::parse_aspect_model;
use std::path::PathBuf;

/// Validate a SAMM Aspect model
pub async fn validate(file: PathBuf, detailed: bool, format: String) -> CliResult<()> {
    let result = parse_aspect_model(&file)
        .await
        .map_err(|e| format!("SAMM validation error: {}", e));

    match result {
        Ok(aspect) => {
            if format == "json" {
                println!(
                    r#"{{"valid": true, "aspect": "{}", "properties": {}, "operations": {}, "events": {}}}"#,
                    aspect.name(),
                    aspect.properties().len(),
                    aspect.operations().len(),
                    aspect.events().len()
                );
            } else {
                println!("✓ SAMM model is valid");
                println!("  Aspect: {}", aspect.name());
                println!("  Properties: {}", aspect.properties().len());
                println!("  Operations: {}", aspect.operations().len());
                println!("  Events: {}", aspect.events().len());

                if detailed {
                    println!("\nAspect Details:");
                    println!("  URN: {}", aspect.metadata().urn);

                    if let Some(name) = aspect.metadata().get_preferred_name("en") {
                        println!("  Preferred Name: {}", name);
                    }
                    if let Some(desc) = aspect.metadata().get_description("en") {
                        println!("  Description: {}", desc);
                    }

                    if !aspect.properties().is_empty() {
                        println!("\nProperties:");
                        for prop in aspect.properties() {
                            println!("  - {} ({})", prop.name(), prop.urn());
                            if prop.optional {
                                println!("    Optional: yes");
                            }
                            if let Some(char) = &prop.characteristic {
                                println!("    Characteristic: {:?}", char.kind());
                            }
                        }
                    }
                    if !aspect.operations().is_empty() {
                        println!("\nOperations:");
                        for op in aspect.operations() {
                            println!("  - {} ({})", op.name(), op.urn());
                        }
                    }
                    if !aspect.events().is_empty() {
                        println!("\nEvents:");
                        for event in aspect.events() {
                            println!("  - {} ({})", event.name(), event.urn());
                        }
                    }
                }
            }
            Ok(())
        }
        Err(e) => {
            if format == "json" {
                println!(r#"{{"valid": false, "error": "{}"}}"#, e);
            } else {
                eprintln!("✗ SAMM model validation failed");
                eprintln!("  Error: {}", e);
            }
            Err(e.into())
        }
    }
}

/// Pretty-print a SAMM model
pub async fn prettyprint(
    file: PathBuf,
    _output: Option<PathBuf>,
    _format: String,
    _comments: bool,
) -> CliResult<()> {
    let aspect = parse_aspect_model(&file)
        .await
        .map_err(|e| format!("SAMM parsing error: {}", e))?;

    println!("# Pretty-printed SAMM Model");
    println!();
    println!("Aspect: {}", aspect.name());
    println!("URN: {}", aspect.metadata().urn);
    println!();

    if let Some(name) = aspect.metadata().get_preferred_name("en") {
        println!("Preferred Name: {}", name);
    }
    if let Some(desc) = aspect.metadata().get_description("en") {
        println!("Description: {}", desc);
    }

    if !aspect.properties().is_empty() {
        println!("\nProperties:");
        for prop in aspect.properties() {
            println!("  {}:", prop.name());
            if let Some(char) = &prop.characteristic {
                println!("    Type: {:?}", char.kind());
                if let Some(dt) = &char.data_type {
                    println!("    Data Type: {}", dt);
                }
            }
            if prop.optional {
                println!("    Optional: yes");
            }
        }
    }

    if !aspect.operations().is_empty() {
        println!("\nOperations:");
        for op in aspect.operations() {
            println!("  {}", op.name());
        }
    }

    if !aspect.events().is_empty() {
        println!("\nEvents:");
        for event in aspect.events() {
            println!("  {}", event.name());
        }
    }

    Ok(())
}

/// Convert DTDL to SAMM Aspect model
pub async fn from_dtdl(file: PathBuf, output: Option<PathBuf>, _format: String) -> CliResult<()> {
    use oxirs_samm::dtdl_parser::parse_dtdl_interface;
    use oxirs_samm::serializer::serialize_aspect_to_string;

    let dtdl_json = tokio::fs::read_to_string(&file)
        .await
        .map_err(|e| format!("Failed to read DTDL file: {}", e))?;

    let aspect =
        parse_dtdl_interface(&dtdl_json).map_err(|e| format!("DTDL parsing error: {}", e))?;

    println!("✓ Converted DTDL Interface to SAMM Aspect");
    println!("  Aspect: {}", aspect.name());
    println!("  Properties: {}", aspect.properties().len());
    println!("  Operations: {}", aspect.operations().len());
    println!("  Events: {}", aspect.events().len());

    let turtle = serialize_aspect_to_string(&aspect)
        .map_err(|e| format!("SAMM serialization error: {}", e))?;

    if let Some(output_path) = output {
        tokio::fs::write(&output_path, turtle)
            .await
            .map_err(|e| format!("Failed to write output: {}", e))?;
        println!("\n✓ Saved to: {}", output_path.display());
    } else {
        println!("\n# SAMM Aspect Model (Turtle)");
        println!("{}", turtle);
    }

    Ok(())
}

/// Show where model elements are used
pub async fn usage(input: String, models_root: Option<PathBuf>) -> CliResult<()> {
    use std::collections::HashMap;
    use tokio::fs;

    println!("Analyzing element usage in Aspect models...");
    println!("  Input: {}", input);

    let search_root = if let Some(root) = models_root {
        println!("  Models Root: {}", root.display());
        root
    } else {
        let current = std::env::current_dir()
            .map_err(|e| format!("Failed to get current directory: {}", e))?;
        println!("  Models Root: {} (current directory)", current.display());
        current
    };
    println!();

    let search_pattern = if input.starts_with("urn:samm:") {
        input.clone()
    } else {
        format!("#{}", input)
    };

    println!("Searching for: {}", search_pattern);
    println!();

    let mut found_files: Vec<PathBuf> = Vec::new();
    scan_directory(&search_root, &mut found_files).await?;

    if found_files.is_empty() {
        println!("No .ttl files found in {}", search_root.display());
        return Ok(());
    }

    println!("Scanning {} model files...", found_files.len());
    println!();

    let mut usage_map: HashMap<PathBuf, Vec<(usize, String)>> = HashMap::new();
    let mut total_references = 0;

    for file_path in &found_files {
        if let Ok(content) = fs::read_to_string(file_path).await {
            let mut file_references = Vec::new();
            for (line_num, line) in content.lines().enumerate() {
                if line.contains(&search_pattern) {
                    file_references.push((line_num + 1, line.trim().to_string()));
                    total_references += 1;
                }
            }
            if !file_references.is_empty() {
                usage_map.insert(file_path.clone(), file_references);
            }
        }
    }

    if usage_map.is_empty() {
        println!("✗ No usages found for '{}'", search_pattern);
        println!();
        println!("The element may not exist or is not referenced in any models.");
        return Ok(());
    }

    println!(
        "✓ Found {} reference(s) in {} file(s)",
        total_references,
        usage_map.len()
    );
    println!();

    let mut sorted_files: Vec<_> = usage_map.keys().collect();
    sorted_files.sort();

    for file_path in sorted_files {
        let references = &usage_map[file_path];
        let display_path = if let Ok(rel_path) = file_path.strip_prefix(&search_root) {
            rel_path.display().to_string()
        } else {
            file_path.display().to_string()
        };
        println!("{}:", display_path);
        println!("  {} reference(s)", references.len());
        for (line_num, line_content) in references {
            let display_content = if line_content.len() > 100 {
                format!("{}...", &line_content[..97])
            } else {
                line_content.clone()
            };
            println!("    Line {}: {}", line_num, display_content);
        }
        println!();
    }

    Ok(())
}

/// Recursively scan directory for .ttl files
pub async fn scan_directory(dir: &PathBuf, files: &mut Vec<PathBuf>) -> CliResult<()> {
    use tokio::fs;

    if !dir.is_dir() {
        return Ok(());
    }

    let mut entries = fs::read_dir(dir)
        .await
        .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| format!("Failed to read directory entry: {}", e))?
    {
        let path = entry.path();
        if path.is_dir() {
            Box::pin(scan_directory(&path, files)).await?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("ttl") {
            files.push(path);
        }
    }

    Ok(())
}

// ---- Format-specific conversion functions ----

type AspectRef<'a> = &'a oxirs_samm::metamodel::Aspect;

/// Convert to Rust code
pub fn convert_rust(aspect: AspectRef<'_>) -> CliResult<()> {
    println!("// Generated Rust code from SAMM model");
    println!();
    println!("// Aspect: {}", aspect.name());
    println!("#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]");
    println!("pub struct {} {{", aspect.name());

    for prop in aspect.properties() {
        let type_name = if let Some(char) = &prop.characteristic {
            if let Some(dt) = &char.data_type {
                map_xsd_to_rust(dt)
            } else {
                "String".to_string()
            }
        } else {
            "String".to_string()
        };

        if prop.optional {
            println!(
                "    pub {}: Option<{}>,",
                to_snake_case(&prop.name()),
                type_name
            );
        } else {
            println!("    pub {}: {},", to_snake_case(&prop.name()), type_name);
        }
    }

    println!("}}");
    Ok(())
}

/// Convert to Markdown documentation
pub fn convert_markdown(aspect: AspectRef<'_>) -> CliResult<()> {
    println!("# {}", aspect.name());
    println!();

    if let Some(desc) = aspect.metadata().get_description("en") {
        println!("{}", desc);
        println!();
    }

    if !aspect.properties().is_empty() {
        println!("## Properties");
        println!();
        println!("| Property | Type | Optional | Description |");
        println!("|----------|------|----------|-------------|");

        for prop in aspect.properties() {
            let type_name = if let Some(char) = &prop.characteristic {
                format!("{:?}", char.kind())
            } else {
                "Unknown".to_string()
            };
            println!(
                "| {} | {} | {} | {} |",
                prop.name(),
                type_name,
                if prop.optional { "Yes" } else { "No" },
                prop.metadata().get_description("en").unwrap_or("")
            );
        }
        println!();
    }

    if !aspect.operations().is_empty() {
        println!("## Operations");
        println!();
        for op in aspect.operations() {
            println!("- **{}**", op.name());
        }
        println!();
    }

    Ok(())
}

/// Convert to JSON Schema
pub fn convert_jsonschema(aspect: AspectRef<'_>) -> CliResult<()> {
    println!("{{");
    println!("  \"$schema\": \"https://json-schema.org/draft/2020-12/schema\",");
    println!("  \"$id\": \"{}\",", aspect.metadata().urn);
    println!("  \"title\": \"{}\",", aspect.name());

    if let Some(desc) = aspect.metadata().get_description("en") {
        println!("  \"description\": \"{}\",", desc);
    }

    println!("  \"type\": \"object\",");
    println!("  \"properties\": {{");

    let properties = aspect.properties();
    for (i, prop) in properties.iter().enumerate() {
        let prop_name = to_snake_case(&prop.name());
        print!("    \"{}\": {{", prop_name);

        if let Some(desc) = prop.metadata().get_description("en") {
            print!("\n      \"description\": \"{}\",", desc);
        }

        let (json_type, format_attr) = if let Some(char) = &prop.characteristic {
            if let Some(dt) = &char.data_type {
                map_xsd_to_json_schema(dt)
            } else {
                ("string".to_string(), None)
            }
        } else {
            ("string".to_string(), None)
        };

        print!("\n      \"type\": \"{}\"", json_type);

        if let Some(fmt) = format_attr {
            print!(",\n      \"format\": \"{}\"", fmt);
        }

        print!("\n    }}");
        if i < properties.len() - 1 {
            println!(",");
        } else {
            println!();
        }
    }

    println!("  }},");

    let required: Vec<_> = properties
        .iter()
        .filter(|p| !p.optional)
        .map(|p| format!("\"{}\"", to_snake_case(&p.name())))
        .collect();

    if !required.is_empty() {
        println!("  \"required\": [{}]", required.join(", "));
    }

    println!("}}");
    Ok(())
}

/// Convert to OpenAPI specification
pub fn convert_openapi(aspect: AspectRef<'_>) -> CliResult<()> {
    let aspect_name = aspect.name();
    let schema_name = aspect_name.clone();

    println!("openapi: 3.1.0");
    println!("info:");
    println!(
        "  title: {}",
        aspect
            .metadata()
            .get_preferred_name("en")
            .unwrap_or(&aspect_name)
    );

    if let Some(desc) = aspect.metadata().get_description("en") {
        println!("  description: {}", desc);
    }

    println!("  version: 1.0.0");
    println!("paths:");
    println!("  /{}:", to_snake_case(&aspect_name));
    println!("    get:");
    println!("      summary: Get {} data", aspect_name);
    println!("      responses:");
    println!("        '200':");
    println!("          description: Successful response");
    println!("          content:");
    println!("            application/json:");
    println!("              schema:");
    println!(
        "                $ref: '#/components/schemas/{}'",
        schema_name
    );
    println!("    post:");
    println!("      summary: Create {} data", aspect_name);
    println!("      requestBody:");
    println!("        required: true");
    println!("        content:");
    println!("          application/json:");
    println!("            schema:");
    println!("              $ref: '#/components/schemas/{}'", schema_name);
    println!("      responses:");
    println!("        '201':");
    println!("          description: Created successfully");
    println!("          content:");
    println!("            application/json:");
    println!("              schema:");
    println!(
        "                $ref: '#/components/schemas/{}'",
        schema_name
    );

    println!("components:");
    println!("  schemas:");
    println!("    {}:", schema_name);
    println!("      type: object");

    if let Some(desc) = aspect.metadata().get_description("en") {
        println!("      description: {}", desc);
    }

    println!("      properties:");
    let properties = aspect.properties();
    for prop in properties {
        let prop_name = to_snake_case(&prop.name());
        println!("        {}:", prop_name);

        if let Some(desc) = prop.metadata().get_description("en") {
            println!("          description: {}", desc);
        }

        let (oa_type, format_attr) = if let Some(char) = &prop.characteristic {
            if let Some(dt) = &char.data_type {
                map_xsd_to_json_schema(dt)
            } else {
                ("string".to_string(), None)
            }
        } else {
            ("string".to_string(), None)
        };

        println!("          type: {}", oa_type);
        if let Some(fmt) = format_attr {
            println!("          format: {}", fmt);
        }
    }

    let required: Vec<_> = properties
        .iter()
        .filter(|p| !p.optional)
        .map(|p| to_snake_case(&p.name()))
        .collect();
    if !required.is_empty() {
        println!("      required:");
        for req in required {
            println!("        - {}", req);
        }
    }

    Ok(())
}

/// Convert to AsyncAPI specification
pub fn convert_asyncapi(aspect: AspectRef<'_>) -> CliResult<()> {
    let aspect_name = aspect.name();
    let schema_name = aspect_name.clone();

    println!("asyncapi: 2.6.0");
    println!("info:");
    println!(
        "  title: {}",
        aspect
            .metadata()
            .get_preferred_name("en")
            .unwrap_or(&aspect_name)
    );

    if let Some(desc) = aspect.metadata().get_description("en") {
        println!("  description: {}", desc);
    }

    println!("  version: 1.0.0");
    println!("channels:");

    let events = aspect.events();
    if !events.is_empty() {
        for event in events {
            let channel_name = format!(
                "{}/{}",
                to_snake_case(&aspect_name),
                to_snake_case(&event.name())
            );
            println!("  {}:", channel_name);
            if let Some(desc) = event.metadata().get_description("en") {
                println!("    description: {}", desc);
            }
            println!("    subscribe:");
            println!("      summary: Subscribe to {} events", event.name());
            println!("      message:");
            println!("        $ref: '#/components/messages/{}'", event.name());
        }
    } else {
        let channel_name = format!("{}/data", to_snake_case(&aspect_name));
        println!("  {}:", channel_name);
        println!("    description: Data updates for {}", aspect_name);
        println!("    subscribe:");
        println!("      summary: Subscribe to {} data updates", aspect_name);
        println!("      message:");
        println!("        $ref: '#/components/messages/{}Data'", aspect_name);
    }

    println!("components:");
    println!("  messages:");

    if !events.is_empty() {
        for event in events {
            println!("    {}:", event.name());
            println!("      name: {}", event.name());
            if let Some(desc) = event.metadata().get_description("en") {
                println!("      description: {}", desc);
            }
            println!("      payload:");
            println!("        $ref: '#/components/schemas/{}'", schema_name);
        }
    } else {
        println!("    {}Data:", aspect_name);
        println!("      name: {}Data", aspect_name);
        println!("      description: {} data update message", aspect_name);
        println!("      payload:");
        println!("        $ref: '#/components/schemas/{}'", schema_name);
    }

    println!("  schemas:");
    println!("    {}:", schema_name);
    println!("      type: object");

    if let Some(desc) = aspect.metadata().get_description("en") {
        println!("      description: {}", desc);
    }

    println!("      properties:");
    let properties = aspect.properties();
    for prop in properties {
        let prop_name = to_snake_case(&prop.name());
        println!("        {}:", prop_name);
        if let Some(desc) = prop.metadata().get_description("en") {
            println!("          description: {}", desc);
        }
        let (aa_type, format_attr) = if let Some(char) = &prop.characteristic {
            if let Some(dt) = &char.data_type {
                map_xsd_to_json_schema(dt)
            } else {
                ("string".to_string(), None)
            }
        } else {
            ("string".to_string(), None)
        };
        println!("          type: {}", aa_type);
        if let Some(fmt) = format_attr {
            println!("          format: {}", fmt);
        }
    }

    let required: Vec<_> = properties
        .iter()
        .filter(|p| !p.optional)
        .map(|p| to_snake_case(&p.name()))
        .collect();
    if !required.is_empty() {
        println!("      required:");
        for req in required {
            println!("        - {}", req);
        }
    }

    Ok(())
}

/// Convert to HTML report
pub fn convert_html(aspect: AspectRef<'_>) -> CliResult<()> {
    let aspect_name = aspect.name();

    println!("<!DOCTYPE html>");
    println!("<html lang=\"en\">");
    println!("<head>");
    println!("  <meta charset=\"UTF-8\">");
    println!("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">");
    println!(
        "  <title>{}</title>",
        aspect
            .metadata()
            .get_preferred_name("en")
            .unwrap_or(&aspect_name)
    );
    println!("  <style>");
    println!("    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}");
    println!("    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}");
    println!("    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 0; }}");
    println!("    h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #ecf0f1; padding-bottom: 8px; }}");
    println!("    .description {{ color: #7f8c8d; font-size: 1.1em; margin: 20px 0; line-height: 1.6; }}");
    println!("    .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 4px; margin: 20px 0; }}");
    println!("    .metadata-item {{ margin: 8px 0; }}");
    println!("    .metadata-label {{ font-weight: bold; color: #2c3e50; display: inline-block; width: 120px; }}");
    println!("    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}");
    println!("    th {{ background: #3498db; color: white; padding: 12px; text-align: left; font-weight: 600; }}");
    println!("    td {{ padding: 12px; border-bottom: 1px solid #ecf0f1; }}");
    println!("    tr:hover {{ background: #f8f9fa; }}");
    println!("    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 0.85em; font-weight: 600; }}");
    println!("    .badge-required {{ background: #e74c3c; color: white; }}");
    println!("    .badge-optional {{ background: #95a5a6; color: white; }}");
    println!("    .badge-type {{ background: #2ecc71; color: white; margin-left: 8px; }}");
    println!("    .section {{ margin: 30px 0; }}");
    println!("    .no-data {{ color: #95a5a6; font-style: italic; }}");
    println!("    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }}");
    println!("  </style>");
    println!("</head>");
    println!("<body>");
    println!("  <div class=\"container\">");

    println!(
        "    <h1>{}</h1>",
        aspect
            .metadata()
            .get_preferred_name("en")
            .unwrap_or(&aspect_name)
    );
    if let Some(desc) = aspect.metadata().get_description("en") {
        println!("    <p class=\"description\">{}</p>", desc);
    }

    println!("    <div class=\"metadata\">");
    println!("      <div class=\"metadata-item\">");
    println!("        <span class=\"metadata-label\">URN:</span>");
    println!("        <code>{}</code>", aspect.metadata().urn);
    println!("      </div>");
    println!("      <div class=\"metadata-item\">");
    println!("        <span class=\"metadata-label\">Aspect Name:</span>");
    println!("        <code>{}</code>", aspect_name);
    println!("      </div>");
    println!("    </div>");

    let properties = aspect.properties();
    if !properties.is_empty() {
        println!("    <div class=\"section\">");
        println!("      <h2>Properties</h2>");
        println!("      <table>");
        println!("        <thead><tr><th>Property</th><th>Type</th><th>Required</th><th>Description</th></tr></thead>");
        println!("        <tbody>");
        for prop in properties {
            println!("          <tr>");
            println!("            <td><code>{}</code></td>", prop.name());
            let type_str = if let Some(char) = &prop.characteristic {
                if let Some(dt) = &char.data_type {
                    dt.split('#').next_back().unwrap_or("String").to_string()
                } else {
                    format!("{:?}", char.kind())
                }
            } else {
                "String".to_string()
            };
            println!(
                "            <td><span class=\"badge badge-type\">{}</span></td>",
                type_str
            );
            if prop.optional {
                println!(
                    "            <td><span class=\"badge badge-optional\">Optional</span></td>"
                );
            } else {
                println!(
                    "            <td><span class=\"badge badge-required\">Required</span></td>"
                );
            }
            let desc = prop.metadata().get_description("en").unwrap_or("");
            println!("            <td>{}</td>", desc);
            println!("          </tr>");
        }
        println!("        </tbody>");
        println!("      </table>");
        println!("    </div>");
    }

    let operations = aspect.operations();
    if !operations.is_empty() {
        println!("    <div class=\"section\">");
        println!("      <h2>Operations</h2>");
        println!("      <table>");
        println!("        <thead><tr><th>Operation</th><th>Description</th></tr></thead>");
        println!("        <tbody>");
        for op in operations {
            println!("          <tr>");
            println!("            <td><code>{}</code></td>", op.name());
            let desc = op.metadata().get_description("en").unwrap_or("");
            println!("            <td>{}</td>", desc);
            println!("          </tr>");
        }
        println!("        </tbody>");
        println!("      </table>");
        println!("    </div>");
    }

    let events = aspect.events();
    if !events.is_empty() {
        println!("    <div class=\"section\">");
        println!("      <h2>Events</h2>");
        println!("      <table>");
        println!("        <thead><tr><th>Event</th><th>Description</th></tr></thead>");
        println!("        <tbody>");
        for event in events {
            println!("          <tr>");
            println!("            <td><code>{}</code></td>", event.name());
            let desc = event.metadata().get_description("en").unwrap_or("");
            println!("            <td>{}</td>", desc);
            println!("          </tr>");
        }
        println!("        </tbody>");
        println!("      </table>");
        println!("    </div>");
    }

    println!("  </div>");
    println!("</body>");
    println!("</html>");

    Ok(())
}

/// Convert to AAS submodel
pub fn convert_aas(aspect: AspectRef<'_>, format_variant: &Option<String>) -> CliResult<()> {
    use oxirs_samm::generators::aas::{generate_aas, AasFormat};

    let aas_format = if let Some(variant) = format_variant {
        match variant.as_str() {
            "xml" => AasFormat::Xml,
            "json" => AasFormat::Json,
            "aasx" => AasFormat::Aasx,
            _ => {
                eprintln!(
                    "Warning: Unknown AAS format '{}', defaulting to JSON",
                    variant
                );
                AasFormat::Json
            }
        }
    } else {
        AasFormat::Json
    };

    let output =
        generate_aas(aspect, aas_format).map_err(|e| format!("AAS generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to diagram format
pub fn convert_diagram(aspect: AspectRef<'_>, format_variant: &Option<String>) -> CliResult<()> {
    use oxirs_samm::generators::diagram::{generate_diagram, DiagramFormat, DiagramStyle};

    let diagram_format = if let Some(variant) = format_variant {
        match variant.as_str() {
            "dot" => DiagramFormat::Dot(DiagramStyle::default()),
            "svg" => DiagramFormat::Svg(DiagramStyle::default()),
            "png" => DiagramFormat::Png(DiagramStyle::default()),
            "mermaid" => DiagramFormat::Mermaid(DiagramStyle::default()),
            "plantuml" => DiagramFormat::PlantUml(DiagramStyle::default()),
            "html" => DiagramFormat::HtmlReport(DiagramStyle::default()),
            _ => {
                eprintln!(
                    "Warning: Unknown diagram format '{}', defaulting to DOT",
                    variant
                );
                DiagramFormat::Dot(DiagramStyle::default())
            }
        }
    } else {
        DiagramFormat::Dot(DiagramStyle::default())
    };

    let output = generate_diagram(aspect, diagram_format)
        .map_err(|e| format!("Diagram generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to SQL DDL
pub fn convert_sql(aspect: AspectRef<'_>, format_variant: &Option<String>) -> CliResult<()> {
    use oxirs_samm::generators::sql::{generate_sql, SqlDialect};

    let sql_dialect = if let Some(variant) = format_variant {
        match variant.as_str() {
            "postgresql" | "postgres" | "pg" => SqlDialect::PostgreSql,
            "mysql" => SqlDialect::MySql,
            "sqlite" => SqlDialect::Sqlite,
            _ => {
                eprintln!(
                    "Warning: Unknown SQL dialect '{}', defaulting to PostgreSQL",
                    variant
                );
                SqlDialect::PostgreSql
            }
        }
    } else {
        SqlDialect::PostgreSql
    };

    let output =
        generate_sql(aspect, sql_dialect).map_err(|e| format!("SQL generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to JSON-LD
pub fn convert_jsonld(aspect: AspectRef<'_>) -> CliResult<()> {
    use oxirs_samm::generators::jsonld::generate_jsonld;
    let output =
        generate_jsonld(aspect).map_err(|e| format!("JSON-LD generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to example payload
pub fn convert_payload(aspect: AspectRef<'_>, examples: bool) -> CliResult<()> {
    use oxirs_samm::generators::payload::generate_payload;
    let output = generate_payload(aspect, examples)
        .map_err(|e| format!("Payload generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to GraphQL schema
pub fn convert_graphql(aspect: AspectRef<'_>) -> CliResult<()> {
    use oxirs_samm::generators::graphql::generate_graphql;
    let output =
        generate_graphql(aspect).map_err(|e| format!("GraphQL generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to TypeScript
pub fn convert_typescript(aspect: AspectRef<'_>) -> CliResult<()> {
    use oxirs_samm::generators::typescript::{generate_typescript, TsOptions};
    let options = TsOptions::default();
    let output = generate_typescript(aspect, options)
        .map_err(|e| format!("TypeScript generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to DTDL
pub fn convert_dtdl(aspect: AspectRef<'_>) -> CliResult<()> {
    use oxirs_samm::generators::dtdl::generate_dtdl;
    let output = generate_dtdl(aspect).map_err(|e| format!("DTDL generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to Python dataclass
pub fn convert_python(aspect: AspectRef<'_>) -> CliResult<()> {
    use oxirs_samm::generators::python::{generate_python, PythonOptions};
    let options = PythonOptions::default();
    let output =
        generate_python(aspect, options).map_err(|e| format!("Python generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to Java
pub fn convert_java(aspect: AspectRef<'_>) -> CliResult<()> {
    use oxirs_samm::generators::java::{generate_java, JavaOptions};
    let options = JavaOptions::default();
    let output =
        generate_java(aspect, options).map_err(|e| format!("Java generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}

/// Convert to Scala
pub fn convert_scala(aspect: AspectRef<'_>) -> CliResult<()> {
    use oxirs_samm::generators::scala::{generate_scala, ScalaOptions};
    let options = ScalaOptions::default();
    let output =
        generate_scala(aspect, options).map_err(|e| format!("Scala generation failed: {}", e))?;
    println!("{}", output);
    Ok(())
}
