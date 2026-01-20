//! SPARQL Query Templates
//!
//! Pre-built query templates for common RDF/SPARQL patterns

use anyhow::{Context, Result};
use std::collections::HashMap;

/// Query template with placeholders
#[derive(Debug, Clone)]
pub struct QueryTemplate {
    pub name: String,
    pub description: String,
    pub category: TemplateCategory,
    pub template: String,
    pub parameters: Vec<TemplateParameter>,
    pub example: String,
}

/// Template parameter
#[derive(Debug, Clone)]
pub struct TemplateParameter {
    pub name: String,
    pub description: String,
    pub default_value: Option<String>,
    pub required: bool,
}

/// Template category
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateCategory {
    Basic,
    Advanced,
    Analytics,
    GraphPatterns,
    Federation,
    PropertyPaths,
    Aggregation,
}

/// Get all available templates
pub fn get_all_templates() -> Vec<QueryTemplate> {
    vec![
        // Basic Templates
        QueryTemplate {
            name: "select-all".to_string(),
            description: "Select all triples".to_string(),
            category: TemplateCategory::Basic,
            template: "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT {{limit}}".to_string(),
            parameters: vec![TemplateParameter {
                name: "limit".to_string(),
                description: "Maximum number of results".to_string(),
                default_value: Some("100".to_string()),
                required: false,
            }],
            example: "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100".to_string(),
        },
        QueryTemplate {
            name: "select-by-type".to_string(),
            description: "Find all instances of a specific type".to_string(),
            category: TemplateCategory::Basic,
            template: "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n\
                      SELECT ?instance WHERE {\n  ?instance rdf:type <{{type_iri}}>\n} LIMIT {{limit}}".to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "type_iri".to_string(),
                    description: "The RDF type IRI".to_string(),
                    default_value: None,
                    required: true,
                },
                TemplateParameter {
                    name: "limit".to_string(),
                    description: "Maximum number of results".to_string(),
                    default_value: Some("100".to_string()),
                    required: false,
                },
            ],
            example: "SELECT ?instance WHERE { ?instance rdf:type <http://xmlns.com/foaf/0.1/Person> } LIMIT 100".to_string(),
        },
        QueryTemplate {
            name: "select-with-filter".to_string(),
            description: "Select triples with a filter condition".to_string(),
            category: TemplateCategory::Basic,
            template: "SELECT ?s ?p ?o WHERE {\n  ?s ?p ?o .\n  FILTER({{condition}})\n} LIMIT {{limit}}".to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "condition".to_string(),
                    description: "Filter condition (e.g., ?o > 10)".to_string(),
                    default_value: None,
                    required: true,
                },
                TemplateParameter {
                    name: "limit".to_string(),
                    description: "Maximum number of results".to_string(),
                    default_value: Some("100".to_string()),
                    required: false,
                },
            ],
            example: "SELECT ?person ?age WHERE { ?person <http://xmlns.com/foaf/0.1/age> ?age . FILTER(?age > 30) }".to_string(),
        },
        // Advanced Templates
        QueryTemplate {
            name: "construct-graph".to_string(),
            description: "Construct new triples based on a pattern".to_string(),
            category: TemplateCategory::Advanced,
            template: "CONSTRUCT { {{construct_pattern}} } WHERE { {{where_pattern}} }".to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "construct_pattern".to_string(),
                    description: "Triple pattern for construction".to_string(),
                    default_value: None,
                    required: true,
                },
                TemplateParameter {
                    name: "where_pattern".to_string(),
                    description: "Triple pattern for matching".to_string(),
                    default_value: None,
                    required: true,
                },
            ],
            example: "CONSTRUCT { ?s <http://example.org/newProp> ?o } WHERE { ?s <http://example.org/oldProp> ?o }".to_string(),
        },
        QueryTemplate {
            name: "ask-exists".to_string(),
            description: "Check if a pattern exists".to_string(),
            category: TemplateCategory::Basic,
            template: "ASK { {{pattern}} }".to_string(),
            parameters: vec![TemplateParameter {
                name: "pattern".to_string(),
                description: "Triple pattern to check".to_string(),
                default_value: None,
                required: true,
            }],
            example: "ASK { <http://example.org/john> <http://xmlns.com/foaf/0.1/knows> <http://example.org/jane> }".to_string(),
        },
        // Aggregation Templates
        QueryTemplate {
            name: "count-instances".to_string(),
            description: "Count instances of a type".to_string(),
            category: TemplateCategory::Aggregation,
            template: "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n\
                      SELECT (COUNT(?instance) AS ?count) WHERE {\n  ?instance rdf:type <{{type_iri}}>\n}".to_string(),
            parameters: vec![TemplateParameter {
                name: "type_iri".to_string(),
                description: "The RDF type IRI to count".to_string(),
                default_value: None,
                required: true,
            }],
            example: "SELECT (COUNT(?person) AS ?count) WHERE { ?person rdf:type <http://xmlns.com/foaf/0.1/Person> }".to_string(),
        },
        QueryTemplate {
            name: "group-by-count".to_string(),
            description: "Group by a property and count occurrences".to_string(),
            category: TemplateCategory::Aggregation,
            template: "SELECT ?{{group_var}} (COUNT(?item) AS ?count) WHERE {\n  ?item <{{property}}> ?{{group_var}}\n} GROUP BY ?{{group_var}} ORDER BY DESC(?count)".to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "group_var".to_string(),
                    description: "Variable to group by".to_string(),
                    default_value: Some("category".to_string()),
                    required: false,
                },
                TemplateParameter {
                    name: "property".to_string(),
                    description: "Property IRI to group by".to_string(),
                    default_value: None,
                    required: true,
                },
            ],
            example: "SELECT ?category (COUNT(?item) AS ?count) WHERE { ?item <http://example.org/category> ?category } GROUP BY ?category ORDER BY DESC(?count)".to_string(),
        },
        // Property Paths
        QueryTemplate {
            name: "transitive-closure".to_string(),
            description: "Find all resources connected via transitive property".to_string(),
            category: TemplateCategory::PropertyPaths,
            template: "SELECT ?start ?end WHERE {\n  ?start <{{property}}>+ ?end\n} LIMIT {{limit}}".to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "property".to_string(),
                    description: "Property IRI for transitive closure".to_string(),
                    default_value: None,
                    required: true,
                },
                TemplateParameter {
                    name: "limit".to_string(),
                    description: "Maximum number of results".to_string(),
                    default_value: Some("100".to_string()),
                    required: false,
                },
            ],
            example: "SELECT ?person ?ancestor WHERE { ?person <http://example.org/ancestorOf>+ ?ancestor } LIMIT 100".to_string(),
        },
        // Federation
        QueryTemplate {
            name: "federated-query".to_string(),
            description: "Query remote SPARQL endpoint".to_string(),
            category: TemplateCategory::Federation,
            template: "SELECT ?s ?p ?o WHERE {\n  SERVICE <{{endpoint}}> {\n    ?s ?p ?o\n  }\n} LIMIT {{limit}}".to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "endpoint".to_string(),
                    description: "Remote SPARQL endpoint URL".to_string(),
                    default_value: None,
                    required: true,
                },
                TemplateParameter {
                    name: "limit".to_string(),
                    description: "Maximum number of results".to_string(),
                    default_value: Some("10".to_string()),
                    required: false,
                },
            ],
            example: "SELECT ?s ?p ?o WHERE { SERVICE <https://dbpedia.org/sparql> { ?s ?p ?o } } LIMIT 10".to_string(),
        },
        // Analytics
        QueryTemplate {
            name: "statistics-summary".to_string(),
            description: "Get statistical summary of numeric values".to_string(),
            category: TemplateCategory::Analytics,
            template: "SELECT \n  (COUNT(?value) AS ?count)\n  (MIN(?value) AS ?min)\n  (MAX(?value) AS ?max)\n  (AVG(?value) AS ?avg)\n  (SUM(?value) AS ?sum)\nWHERE {\n  ?s <{{property}}> ?value\n  FILTER(isNumeric(?value))\n}".to_string(),
            parameters: vec![TemplateParameter {
                name: "property".to_string(),
                description: "Property IRI for numeric values".to_string(),
                default_value: None,
                required: true,
            }],
            example: "SELECT (COUNT(?age) AS ?count) (AVG(?age) AS ?avg) WHERE { ?person <http://xmlns.com/foaf/0.1/age> ?age FILTER(isNumeric(?age)) }".to_string(),
        },
    ]
}

/// Get template by name
pub fn get_template(name: &str) -> Option<QueryTemplate> {
    get_all_templates().into_iter().find(|t| t.name == name)
}

/// List all templates
pub fn list_templates(category: Option<TemplateCategory>) -> Vec<QueryTemplate> {
    let templates = get_all_templates();
    if let Some(cat) = category {
        templates
            .into_iter()
            .filter(|t| t.category == cat)
            .collect()
    } else {
        templates
    }
}

/// Render template with parameters
pub fn render_template(
    template: &QueryTemplate,
    params: &HashMap<String, String>,
) -> Result<String> {
    let mut query = template.template.clone();

    // Replace all parameters
    for param in &template.parameters {
        let value = if let Some(v) = params.get(&param.name) {
            v.clone()
        } else if let Some(default) = &param.default_value {
            default.clone()
        } else if param.required {
            anyhow::bail!("Required parameter '{}' not provided", param.name);
        } else {
            continue;
        };

        let placeholder = format!("{{{{{}}}}}", param.name);
        query = query.replace(&placeholder, &value);
    }

    Ok(query)
}

/// Show template information
pub fn show_template_info(template: &QueryTemplate) {
    println!("📋 Template: {}", template.name);
    println!("📝 Description: {}", template.description);
    println!("🏷️  Category: {:?}", template.category);
    println!();
    println!("Parameters:");
    for param in &template.parameters {
        let required = if param.required {
            "required"
        } else {
            "optional"
        };
        let default = if let Some(d) = &param.default_value {
            format!(" (default: {})", d)
        } else {
            String::new()
        };
        println!(
            "  - {} [{}]{}: {}",
            param.name, required, default, param.description
        );
    }
    println!();
    println!("Example:");
    println!("{}", template.example);
}

/// List all available templates
pub async fn list_command(category: Option<String>) -> Result<()> {
    let cat = category
        .as_ref()
        .and_then(|c| match c.to_lowercase().as_str() {
            "basic" => Some(TemplateCategory::Basic),
            "advanced" => Some(TemplateCategory::Advanced),
            "analytics" => Some(TemplateCategory::Analytics),
            "graph" => Some(TemplateCategory::GraphPatterns),
            "federation" => Some(TemplateCategory::Federation),
            "paths" => Some(TemplateCategory::PropertyPaths),
            "aggregation" => Some(TemplateCategory::Aggregation),
            _ => None,
        });

    let templates = list_templates(cat);

    println!("📚 Available SPARQL Query Templates\n");

    let mut by_category: HashMap<String, Vec<&QueryTemplate>> = HashMap::new();
    for template in &templates {
        by_category
            .entry(format!("{:?}", template.category))
            .or_default()
            .push(template);
    }

    for (category, temps) in by_category.iter() {
        println!("{}", "=".repeat(60));
        println!("Category: {}", category);
        println!("{}\n", "=".repeat(60));

        for template in temps {
            println!("  • {} - {}", template.name, template.description);
        }
        println!();
    }

    println!("💡 Use 'oxirs template show <name>' to see details");
    println!("💡 Use 'oxirs template render <name>' to generate a query");

    Ok(())
}

/// Show template details
pub async fn show_command(name: String) -> Result<()> {
    let template = get_template(&name).with_context(|| format!("Template '{}' not found", name))?;

    show_template_info(&template);

    Ok(())
}

/// Render template with parameters
pub async fn render_command(name: String, params: HashMap<String, String>) -> Result<()> {
    let template = get_template(&name).with_context(|| format!("Template '{}' not found", name))?;

    let query = render_template(&template, &params)?;

    println!("Generated Query:\n");
    println!("{}", query);

    Ok(())
}
