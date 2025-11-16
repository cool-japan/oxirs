//! SAMM Aspect Model command implementations (Java ESMF SDK compatible)

use crate::cli::CliResult;
use crate::{AspectAction, EditAction};
use oxirs_samm::metamodel::ModelElement;
use oxirs_samm::parser::parse_aspect_model;
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
    }
}

/// Validate a SAMM Aspect model
async fn validate(file: PathBuf, detailed: bool, format: String) -> CliResult<()> {
    // Parse the SAMM model
    let result = parse_aspect_model(&file)
        .await
        .map_err(|e| format!("SAMM validation error: {}", e));

    match result {
        Ok(aspect) => {
            // Successfully parsed
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
            // Validation failed
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
async fn prettyprint(
    file: PathBuf,
    _output: Option<PathBuf>,
    _format: String,
    _comments: bool,
) -> CliResult<()> {
    // Parse the SAMM model first to validate it
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

/// Convert SAMM Aspect model to other formats (Java ESMF SDK compatible)
async fn convert(
    file: PathBuf,
    format: String,
    _output: Option<PathBuf>,
    _examples: bool,
    format_variant: Option<String>,
) -> CliResult<()> {
    // Parse the SAMM model
    let aspect = parse_aspect_model(&file)
        .await
        .map_err(|e| format!("SAMM conversion error: {}", e))?;

    match format.as_str() {
        "rust" => {
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
        "markdown" => {
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
        "jsonschema" => {
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

                // Determine JSON Schema type from characteristic
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

            // Add required fields (non-optional properties)
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
        "openapi" => {
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

                // Determine OpenAPI type from characteristic
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

            // Add required fields (non-optional properties)
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
        "asyncapi" => {
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

            // Generate channels from Events
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
                // If no events, create a default channel for the aspect data
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

            // Add required fields
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
        "html" => {
            let aspect_name = aspect.name();

            println!("<!DOCTYPE html>");
            println!("<html lang=\"en\">");
            println!("<head>");
            println!("  <meta charset=\"UTF-8\">");
            println!(
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
            );
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

            // Header
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

            // Metadata section
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

            // Properties section
            let properties = aspect.properties();
            if !properties.is_empty() {
                println!("    <div class=\"section\">");
                println!("      <h2>Properties</h2>");
                println!("      <table>");
                println!("        <thead>");
                println!("          <tr>");
                println!("            <th>Property</th>");
                println!("            <th>Type</th>");
                println!("            <th>Required</th>");
                println!("            <th>Description</th>");
                println!("          </tr>");
                println!("        </thead>");
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
                        println!("            <td><span class=\"badge badge-optional\">Optional</span></td>");
                    } else {
                        println!("            <td><span class=\"badge badge-required\">Required</span></td>");
                    }

                    let desc = prop.metadata().get_description("en").unwrap_or("");
                    println!("            <td>{}</td>", desc);
                    println!("          </tr>");
                }

                println!("        </tbody>");
                println!("      </table>");
                println!("    </div>");
            }

            // Operations section
            let operations = aspect.operations();
            if !operations.is_empty() {
                println!("    <div class=\"section\">");
                println!("      <h2>Operations</h2>");
                println!("      <table>");
                println!("        <thead>");
                println!("          <tr>");
                println!("            <th>Operation</th>");
                println!("            <th>Description</th>");
                println!("          </tr>");
                println!("        </thead>");
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

            // Events section
            let events = aspect.events();
            if !events.is_empty() {
                println!("    <div class=\"section\">");
                println!("      <h2>Events</h2>");
                println!("      <table>");
                println!("        <thead>");
                println!("          <tr>");
                println!("            <th>Event</th>");
                println!("            <th>Description</th>");
                println!("          </tr>");
                println!("        </thead>");
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
        "aas" => {
            use oxirs_samm::generators::aas::{generate_aas, AasFormat};

            // Determine AAS format from --format variant parameter
            let aas_format = if let Some(variant) = &format_variant {
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

            let output = generate_aas(&aspect, aas_format)
                .map_err(|e| format!("AAS generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "diagram" => {
            use oxirs_samm::generators::diagram::{generate_diagram, DiagramFormat, DiagramStyle};

            // Determine diagram format from --format variant parameter
            let diagram_format = if let Some(variant) = &format_variant {
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

            let output = generate_diagram(&aspect, diagram_format)
                .map_err(|e| format!("Diagram generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "sql" => {
            use oxirs_samm::generators::sql::{generate_sql, SqlDialect};

            // Determine SQL dialect from --format variant parameter
            let sql_dialect = if let Some(variant) = &format_variant {
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

            let output = generate_sql(&aspect, sql_dialect)
                .map_err(|e| format!("SQL generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "jsonld" => {
            use oxirs_samm::generators::jsonld::generate_jsonld;

            let output = generate_jsonld(&aspect)
                .map_err(|e| format!("JSON-LD generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "payload" => {
            use oxirs_samm::generators::payload::generate_payload;

            let output = generate_payload(&aspect, _examples)
                .map_err(|e| format!("Payload generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "graphql" => {
            use oxirs_samm::generators::graphql::generate_graphql;

            let output = generate_graphql(&aspect)
                .map_err(|e| format!("GraphQL generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "typescript" | "ts" => {
            use oxirs_samm::generators::typescript::{generate_typescript, TsOptions};

            let options = TsOptions::default();
            let output = generate_typescript(&aspect, options)
                .map_err(|e| format!("TypeScript generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "python" | "py" => {
            use oxirs_samm::generators::python::{generate_python, PythonOptions};

            let options = PythonOptions::default();
            let output = generate_python(&aspect, options)
                .map_err(|e| format!("Python generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "java" => {
            use oxirs_samm::generators::java::{generate_java, JavaOptions};

            let options = JavaOptions::default();
            let output = generate_java(&aspect, options)
                .map_err(|e| format!("Java generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        "scala" => {
            use oxirs_samm::generators::scala::{generate_scala, ScalaOptions};

            let options = ScalaOptions::default();
            let output = generate_scala(&aspect, options)
                .map_err(|e| format!("Scala generation failed: {}", e))?;

            println!("{}", output);
            Ok(())
        }
        _ => {
            eprintln!("Error: Unsupported format '{}'", format);
            eprintln!("Supported formats: rust, markdown, jsonschema, openapi, asyncapi, html, aas, diagram, sql, jsonld, payload, graphql, typescript, python, java, scala");
            Err(format!("Unsupported format: {}", format).into())
        }
    }
}

/// Map XSD data types to Rust types
fn map_xsd_to_rust(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "String".to_string(),
        t if t.ends_with("int") || t.ends_with("integer") => "i32".to_string(),
        t if t.ends_with("long") => "i64".to_string(),
        t if t.ends_with("float") => "f32".to_string(),
        t if t.ends_with("double") => "f64".to_string(),
        t if t.ends_with("boolean") => "bool".to_string(),
        t if t.ends_with("date") => "chrono::NaiveDate".to_string(),
        t if t.ends_with("dateTime") => "chrono::DateTime<chrono::Utc>".to_string(),
        _ => "String".to_string(),
    }
}

/// Map XSD data types to JSON Schema types with format attributes
fn map_xsd_to_json_schema(xsd_type: &str) -> (String, Option<String>) {
    match xsd_type {
        t if t.ends_with("string") => ("string".to_string(), None),
        t if t.ends_with("int") || t.ends_with("integer") => {
            ("integer".to_string(), Some("int32".to_string()))
        }
        t if t.ends_with("long") => ("integer".to_string(), Some("int64".to_string())),
        t if t.ends_with("float") => ("number".to_string(), Some("float".to_string())),
        t if t.ends_with("double") => ("number".to_string(), Some("double".to_string())),
        t if t.ends_with("boolean") => ("boolean".to_string(), None),
        t if t.ends_with("date") => ("string".to_string(), Some("date".to_string())),
        t if t.ends_with("dateTime") => ("string".to_string(), Some("date-time".to_string())),
        t if t.ends_with("time") => ("string".to_string(), Some("time".to_string())),
        t if t.ends_with("duration") => ("string".to_string(), Some("duration".to_string())),
        t if t.ends_with("anyURI") => ("string".to_string(), Some("uri".to_string())),
        t if t.ends_with("byte") => ("integer".to_string(), None),
        t if t.ends_with("short") => ("integer".to_string(), Some("int32".to_string())),
        t if t.ends_with("decimal") => ("number".to_string(), None),
        _ => ("string".to_string(), None),
    }
}

/// Convert PascalCase/camelCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
        } else {
            result.push(ch);
        }
    }
    result
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

    // Read source file
    let content = fs::read_to_string(&file)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // Extract current namespace and version from file
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

    // Determine target namespace
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

    // Find element definition in file
    let element_urn_pattern = format!("#{}", element);
    let mut element_found = false;
    let mut element_lines: Vec<String> = Vec::new();
    let mut in_element_block = false;
    let mut brace_depth = 0;

    println!("Searching for element '{}'...", element);

    for line in content.lines() {
        // Check if this line starts the element definition
        if line.contains(&element_urn_pattern) && line.contains("a samm:") {
            element_found = true;
            in_element_block = true;
            element_lines.push(line.to_string());

            // Count braces
            brace_depth += line.matches('[').count() as i32;
            brace_depth -= line.matches(']').count() as i32;

            // Check if definition ends on same line
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

    // Build new element content with updated namespace
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

    // Remove element from source file
    let mut source_lines: Vec<String> = Vec::new();
    let mut skip_lines = false;
    let mut skip_depth = 0;

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
        // Determine output paths
        let target_file = if namespace.is_some() {
            // Create new file in target namespace directory
            let target_dir = PathBuf::from(&target_namespace).join(&current_version);
            let target_filename = format!("{}Element.ttl", element);
            target_dir.join(&target_filename)
        } else {
            // Create new file in same directory
            let target_filename = format!("{}Element.ttl", element);
            file.parent()
                .unwrap_or(std::path::Path::new("."))
                .join(&target_filename)
        };

        // Check if target exists
        if target_file.exists() && !force {
            return Err(format!(
                "Target file already exists: {} (use --force to overwrite)",
                target_file.display()
            )
            .into());
        }

        // Prepare target content
        let mut target_content = String::new();

        // Copy file header if requested
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
            // Add minimal prefixes
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

        // Create target directory
        if let Some(parent) = target_file.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Write target file
        fs::write(&target_file, target_content)
            .await
            .map_err(|e| format!("Failed to write target file: {}", e))?;

        println!("✓ Element moved successfully");
        println!("  Target: {}", target_file.display());

        // Update source file (remove element)
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

    // Read the file content
    let content = fs::read_to_string(&file)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // Extract all URNs and their versions
    let mut urn_map: HashMap<String, String> = HashMap::new();
    let mut namespace = String::new();
    let mut old_version = String::new();

    for line in content.lines() {
        if let Some(urn_start) = line.find("urn:samm:") {
            let urn_part = &line[urn_start..];
            if let Some(urn_end) = urn_part.find(['>', ' ', ';']) {
                let urn = &urn_part[..urn_end];

                // Parse URN: urn:samm:namespace:version#Element
                if let Some(samm_part) = urn.strip_prefix("urn:samm:") {
                    // Skip meta-model namespaces
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

    // Parse version
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

    // Increment version
    if major {
        ver_major += 1;
        ver_minor = 0;
        ver_micro = 0;
    } else if micro {
        ver_micro += 1;
    } else {
        // Default: minor
        ver_minor += 1;
        ver_micro = 0;
    }

    let new_version = format!("{}.{}.{}", ver_major, ver_minor, ver_micro);

    println!("Version Update:");
    println!("  {} → {}", old_version, new_version);
    println!("  Namespace: {}", namespace);
    println!();

    // Replace all occurrences of old version with new version
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
        // Determine output path: namespace/new_version/filename.ttl
        let filename = file.file_name().ok_or("Invalid filename")?;

        let output_dir = if let Some(parent) = file.parent() {
            // Check if current path is namespace/old_version/file.ttl
            if let Some(version_dir) = parent.file_name() {
                if version_dir.to_string_lossy() == old_version {
                    // Replace version directory
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
                    // Create new structure
                    parent.join(&namespace).join(&new_version)
                }
            } else {
                parent.join(&namespace).join(&new_version)
            }
        } else {
            PathBuf::from(&namespace).join(&new_version)
        };

        let output_file = output_dir.join(filename);

        // Check if file exists
        if output_file.exists() && !force {
            return Err(format!(
                "Output file already exists: {} (use --force to overwrite)",
                output_file.display()
            )
            .into());
        }

        // Create directory
        fs::create_dir_all(&output_dir)
            .await
            .map_err(|e| format!("Failed to create directory: {}", e))?;

        // Write new file
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
            file.file_name().unwrap().to_string_lossy()
        );
    }

    Ok(())
}

/// Show where model elements are used
async fn usage(input: String, models_root: Option<PathBuf>) -> CliResult<()> {
    use std::collections::HashMap;
    use tokio::fs;

    println!("Analyzing element usage in Aspect models...");
    println!("  Input: {}", input);

    // Determine search root
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

    // Determine if input is URN or element name
    let search_pattern = if input.starts_with("urn:samm:") {
        input.clone()
    } else {
        // Treat as element name - we'll search for any URN ending with #ElementName
        format!("#{}", input)
    };

    println!("Searching for: {}", search_pattern);
    println!();

    // Recursively scan for .ttl files
    let mut found_files: Vec<PathBuf> = Vec::new();
    scan_directory(&search_root, &mut found_files).await?;

    if found_files.is_empty() {
        println!("No .ttl files found in {}", search_root.display());
        return Ok(());
    }

    println!("Scanning {} model files...", found_files.len());
    println!();

    // Search for usage
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

    // Display results
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

    // Sort files by path for consistent output
    let mut sorted_files: Vec<_> = usage_map.keys().collect();
    sorted_files.sort();

    for file_path in sorted_files {
        let references = &usage_map[file_path];

        // Display relative path if possible
        let display_path = if let Ok(rel_path) = file_path.strip_prefix(&search_root) {
            rel_path.display().to_string()
        } else {
            file_path.display().to_string()
        };

        println!("{}:", display_path);
        println!("  {} reference(s)", references.len());

        for (line_num, line_content) in references {
            // Truncate long lines
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
async fn scan_directory(dir: &PathBuf, files: &mut Vec<PathBuf>) -> CliResult<()> {
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
            // Recurse into subdirectories
            Box::pin(scan_directory(&path, files)).await?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("ttl") {
            files.push(path);
        }
    }

    Ok(())
}
