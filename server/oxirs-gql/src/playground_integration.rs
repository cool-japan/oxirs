//! GraphQL Playground Integration
//!
//! This module provides advanced GraphQL Playground IDE with features:
//! - **Multiple Tabs**: Work with multiple queries simultaneously
//! - **Query History**: Persistent history with search and filtering
//! - **Schema Documentation**: Interactive schema explorer
//! - **Response Tracing**: Detailed performance analysis
//! - **Themes**: Multiple color schemes and customization
//! - **Collaboration**: Share queries with teammates
//! - **Code Generation**: Generate client code from queries
//! - **Request Headers**: Manage authentication and custom headers

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GraphQL Playground configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundConfig {
    /// Playground endpoint path
    pub endpoint: String,

    /// GraphQL endpoint URL
    pub graphql_endpoint: String,

    /// Subscription endpoint (WebSocket)
    pub subscription_endpoint: Option<String>,

    /// Enable multiple tabs
    pub enable_tabs: bool,

    /// Enable query history
    pub enable_history: bool,

    /// Enable schema polling (auto-refresh schema)
    pub enable_schema_polling: bool,

    /// Schema polling interval (in seconds)
    pub schema_polling_interval: u64,

    /// Enable request tracing
    pub enable_tracing: bool,

    /// Enable code generation
    pub enable_code_gen: bool,

    /// Supported code generation targets
    pub code_gen_targets: Vec<CodeGenTarget>,

    /// Default theme
    pub default_theme: PlaygroundTheme,

    /// Enable settings persistence
    pub enable_settings_persistence: bool,

    /// Custom title
    pub title: String,

    /// Custom favicon URL
    pub favicon: Option<String>,

    /// Default request headers
    pub default_headers: HashMap<String, String>,

    /// Enable query sharing
    pub enable_sharing: bool,

    /// Custom CSS
    pub custom_css: Option<String>,

    /// Enable prettier formatting
    pub enable_prettier: bool,

    /// Enable linting
    pub enable_linting: bool,
}

impl Default for PlaygroundConfig {
    fn default() -> Self {
        Self {
            endpoint: "/playground".to_string(),
            graphql_endpoint: "/graphql".to_string(),
            subscription_endpoint: Some("/graphql/subscriptions".to_string()),
            enable_tabs: true,
            enable_history: true,
            enable_schema_polling: true,
            schema_polling_interval: 5,
            enable_tracing: true,
            enable_code_gen: true,
            code_gen_targets: vec![
                CodeGenTarget::TypeScript,
                CodeGenTarget::JavaScript,
                CodeGenTarget::Rust,
                CodeGenTarget::Python,
            ],
            default_theme: PlaygroundTheme::Dark,
            enable_settings_persistence: true,
            title: "OxiRS GraphQL Playground".to_string(),
            favicon: None,
            default_headers: HashMap::new(),
            enable_sharing: true,
            custom_css: None,
            enable_prettier: true,
            enable_linting: true,
        }
    }
}

/// Playground color themes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaygroundTheme {
    /// Dark theme (default)
    Dark,
    /// Light theme
    Light,
}

impl PlaygroundTheme {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlaygroundTheme::Dark => "dark",
            PlaygroundTheme::Light => "light",
        }
    }
}

/// Code generation targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeGenTarget {
    TypeScript,
    JavaScript,
    Rust,
    Python,
    Go,
    Java,
    CSharp,
    Swift,
}

impl CodeGenTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            CodeGenTarget::TypeScript => "typescript",
            CodeGenTarget::JavaScript => "javascript",
            CodeGenTarget::Rust => "rust",
            CodeGenTarget::Python => "python",
            CodeGenTarget::Go => "go",
            CodeGenTarget::Java => "java",
            CodeGenTarget::CSharp => "csharp",
            CodeGenTarget::Swift => "swift",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            Self::TypeScript,
            Self::JavaScript,
            Self::Rust,
            Self::Python,
            Self::Go,
            Self::Java,
            Self::CSharp,
            Self::Swift,
        ]
    }
}

/// Query tab configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTab {
    /// Tab ID
    pub id: String,

    /// Tab name
    pub name: String,

    /// GraphQL query
    pub query: String,

    /// Query variables
    pub variables: HashMap<String, serde_json::Value>,

    /// Request headers
    pub headers: HashMap<String, String>,

    /// Last modified timestamp
    pub last_modified: String,
}

impl QueryTab {
    pub fn new(name: String, query: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            query,
            variables: HashMap::new(),
            headers: HashMap::new(),
            last_modified: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Playground settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundSettings {
    /// Active tab ID
    pub active_tab: Option<String>,

    /// All query tabs
    pub tabs: Vec<QueryTab>,

    /// Current theme
    pub theme: PlaygroundTheme,

    /// Font size
    pub font_size: u32,

    /// Editor settings
    pub editor: EditorSettings,

    /// Request headers
    pub headers: HashMap<String, String>,

    /// Tracing enabled
    pub tracing_enabled: bool,

    /// Auto-completion enabled
    pub auto_completion_enabled: bool,

    /// Prettier enabled
    pub prettier_enabled: bool,

    /// Linting enabled
    pub linting_enabled: bool,
}

impl Default for PlaygroundSettings {
    fn default() -> Self {
        Self {
            active_tab: None,
            tabs: vec![QueryTab::new(
                "Default Query".to_string(),
                "query { __typename }".to_string(),
            )],
            theme: PlaygroundTheme::Dark,
            font_size: 14,
            editor: EditorSettings::default(),
            headers: HashMap::new(),
            tracing_enabled: true,
            auto_completion_enabled: true,
            prettier_enabled: true,
            linting_enabled: true,
        }
    }
}

/// Editor settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorSettings {
    /// Tab size
    pub tab_size: u32,

    /// Use spaces instead of tabs
    pub use_spaces: bool,

    /// Show line numbers
    pub line_numbers: bool,

    /// Word wrap
    pub word_wrap: bool,

    /// Vim mode
    pub vim_mode: bool,

    /// Auto-close brackets
    pub auto_close_brackets: bool,

    /// Highlight matching brackets
    pub highlight_brackets: bool,
}

impl Default for EditorSettings {
    fn default() -> Self {
        Self {
            tab_size: 2,
            use_spaces: true,
            line_numbers: true,
            word_wrap: false,
            vim_mode: false,
            auto_close_brackets: true,
            highlight_brackets: true,
        }
    }
}

/// Generate Playground HTML
pub fn generate_playground_html(config: &PlaygroundConfig) -> String {
    // Build HTML incrementally to avoid complex string literal issues
    let mut html = String::new();

    // HTML header
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("    <meta charset=\"utf-8\">\n");
    html.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    html.push_str(&format!("    <title>{}</title>\n", config.title));

    if let Some(favicon) = &config.favicon {
        html.push_str(&format!("    <link rel=\"icon\" href=\"{}\" />\n", favicon));
    }

    html.push_str("    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/graphql-playground-react@1.7.28/build/static/css/index.css\" />\n");

    if let Some(css) = &config.custom_css {
        html.push_str(&format!("    <style>{}</style>\n", css));
    }

    html.push_str("    <style>\n        body { margin: 0; padding: 0; overflow: hidden; }\n");
    html.push_str("        #root { height: 100vh; }\n    </style>\n");
    html.push_str("</head>\n<body>\n    <div id=\"root\"></div>\n");
    html.push_str("    <script src=\"https://cdn.jsdelivr.net/npm/graphql-playground-react@1.7.28/build/static/js/middleware.js\"></script>\n");
    html.push_str("    <script>\n        window.addEventListener(\"load\", function(event) {\n");
    html.push_str("            GraphQLPlayground.init(document.getElementById(\"root\"), {\n");
    html.push_str(&format!(
        "                endpoint: \"{}\",\n",
        config.graphql_endpoint
    ));

    if let Some(sub_endpoint) = &config.subscription_endpoint {
        html.push_str(&format!(
            "                subscriptionEndpoint: \"{}\",\n",
            sub_endpoint
        ));
    } else {
        html.push_str("                subscriptionEndpoint: null,\n");
    }

    html.push_str("                settings: {},\n");
    html.push_str("                tabs: [],\n");
    html.push_str("                config: {\n");
    html.push_str("                    \"general.betaUpdates\": false,\n");
    html.push_str(&format!(
        "                    \"editor.theme\": \"{}\",\n",
        config.default_theme.as_str()
    ));
    html.push_str("                    \"editor.cursorShape\": \"line\",\n");
    html.push_str("                    \"editor.reuseHeaders\": true\n");
    html.push_str("                },\n");
    html.push_str("                headers: {}\n");
    html.push_str("            });\n        });\n    </script>\n");
    html.push_str("</body>\n</html>");

    html
}

/// Default query templates for RDF operations
pub fn get_default_playground_tabs() -> Vec<QueryTab> {
    vec![
        QueryTab {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Store Information".to_string(),
            query: r#"# Get basic store information
query StoreInfo {
  info {
    version
    tripleCount
    description
  }
}"#
            .to_string(),
            variables: HashMap::new(),
            headers: HashMap::new(),
            last_modified: chrono::Utc::now().to_rfc3339(),
        },
        QueryTab {
            id: uuid::Uuid::new_v4().to_string(),
            name: "List Subjects".to_string(),
            query: r#"# List all subjects with pagination
query ListSubjects($limit: Int = 10, $offset: Int = 0) {
  subjects(limit: $limit, offset: $offset) {
    iri
    label
    type
  }
}"#
            .to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert("limit".to_string(), serde_json::json!(10));
                vars.insert("offset".to_string(), serde_json::json!(0));
                vars
            },
            headers: HashMap::new(),
            last_modified: chrono::Utc::now().to_rfc3339(),
        },
        QueryTab {
            id: uuid::Uuid::new_v4().to_string(),
            name: "SPARQL Query".to_string(),
            query: r#"# Execute raw SPARQL query
query ExecuteSPARQL($sparql: String!) {
  sparql(query: $sparql) {
    results
  }
}"#
            .to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert(
                    "sparql".to_string(),
                    serde_json::json!("SELECT * WHERE { ?s ?p ?o } LIMIT 10"),
                );
                vars
            },
            headers: HashMap::new(),
            last_modified: chrono::Utc::now().to_rfc3339(),
        },
        QueryTab {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Subscription Example".to_string(),
            query: r#"# Subscribe to RDF changes
subscription OnTripleChanged {
  tripleChanged {
    subject
    predicate
    object
    changeType
  }
}"#
            .to_string(),
            variables: HashMap::new(),
            headers: HashMap::new(),
            last_modified: chrono::Utc::now().to_rfc3339(),
        },
    ]
}

/// Code generator for different languages
pub struct CodeGenerator;

impl CodeGenerator {
    /// Generate client code from GraphQL query
    pub fn generate(
        target: CodeGenTarget,
        query: &str,
        operation_name: Option<&str>,
    ) -> anyhow::Result<String> {
        match target {
            CodeGenTarget::TypeScript => Self::generate_typescript(query, operation_name),
            CodeGenTarget::JavaScript => Self::generate_javascript(query, operation_name),
            CodeGenTarget::Rust => Self::generate_rust(query, operation_name),
            CodeGenTarget::Python => Self::generate_python(query, operation_name),
            CodeGenTarget::Go => Self::generate_go(query, operation_name),
            CodeGenTarget::Java => Self::generate_java(query, operation_name),
            CodeGenTarget::CSharp => Self::generate_csharp(query, operation_name),
            CodeGenTarget::Swift => Self::generate_swift(query, operation_name),
        }
    }

    fn generate_typescript(query: &str, operation_name: Option<&str>) -> anyhow::Result<String> {
        let name = operation_name.unwrap_or("Query");
        Ok(format!(
            r#"// TypeScript client code
import {{ gql }} from '@apollo/client';

const {}_QUERY = gql`
{}
`;

// Usage:
// const {{ data, loading, error }} = useQuery({}_QUERY);
"#,
            name.to_uppercase(),
            query,
            name.to_uppercase()
        ))
    }

    fn generate_javascript(query: &str, operation_name: Option<&str>) -> anyhow::Result<String> {
        let name = operation_name.unwrap_or("Query");
        Ok(format!(
            r#"// JavaScript client code
const {{ gql }} = require('@apollo/client');

const {}_QUERY = gql`
{}
`;

// Usage:
// client.query({{ query: {}_QUERY }})
//   .then(result => console.log(result));
"#,
            name.to_uppercase(),
            query,
            name.to_uppercase()
        ))
    }

    fn generate_rust(query: &str, operation_name: Option<&str>) -> anyhow::Result<String> {
        let name = operation_name.unwrap_or("Query");
        let mut code = String::new();
        code.push_str("// Rust client code using graphql_client\n");
        code.push_str("use graphql_client::{GraphQLQuery, Response};\n\n");
        code.push_str("#[derive(GraphQLQuery)]\n");
        code.push_str("#[graphql(\n");
        code.push_str("    schema_path = \"schema.graphql\",\n");
        code.push_str("    query_path = \"query.graphql\",\n");
        code.push_str("    response_derives = \"Debug\"\n");
        code.push_str(")]\n");
        code.push_str(&format!("pub struct {};\n\n", name));
        code.push_str("const QUERY: &str = r#\"\n");
        code.push_str(query);
        code.push_str("\n\"#;\n\n");
        code.push_str("// Usage:\n");
        code.push_str(&format!("// let variables = {}::Variables {{}};\n", name));
        code.push_str(&format!(
            "// let response_body = post_graphql::<{}>(client, url, variables).await?;\n",
            name
        ));
        Ok(code)
    }

    fn generate_python(query: &str, operation_name: Option<&str>) -> anyhow::Result<String> {
        let name = operation_name.unwrap_or("query");
        let mut code = String::new();
        code.push_str("# Python client code using gql\n");
        code.push_str("from gql import gql, Client\n");
        code.push_str("from gql.transport.requests import RequestsHTTPTransport\n\n");
        code.push_str(&format!("{}_QUERY = gql(\"\"\"\n", name.to_uppercase()));
        code.push_str(query);
        code.push_str("\n\"\"\")\n\n");
        code.push_str("# Usage:\n");
        code.push_str("# transport = RequestsHTTPTransport(url='http://localhost:8000/graphql')\n");
        code.push_str("# client = Client(transport=transport)\n");
        code.push_str(&format!(
            "# result = client.execute({}_QUERY)\n",
            name.to_uppercase()
        ));
        Ok(code)
    }

    fn generate_go(query: &str, operation_name: Option<&str>) -> anyhow::Result<String> {
        let name = operation_name.unwrap_or("Query");
        let mut code = String::new();
        code.push_str("// Go client code using graphql package\n");
        code.push_str("package main\n\n");
        code.push_str("import (\n");
        code.push_str("    \"context\"\n");
        code.push_str("    \"github.com/machinebox/graphql\"\n");
        code.push_str(")\n\n");
        code.push_str(&format!("const {}Query = `\n", name));
        code.push_str(query);
        code.push_str("\n`\n\n");
        code.push_str("// Usage:\n");
        code.push_str("// client := graphql.NewClient(\"http://localhost:8000/graphql\")\n");
        code.push_str(&format!("// req := graphql.NewRequest({}Query)\n", name));
        code.push_str("// var response map[string]interface{}\n");
        code.push_str(
            "// if err := client.Run(context.Background(), req, &response); err != nil {\n",
        );
        code.push_str("//     log.Fatal(err)\n");
        code.push_str("// }\n");
        Ok(code)
    }

    fn generate_java(query: &str, operation_name: Option<&str>) -> anyhow::Result<String> {
        let name = operation_name.unwrap_or("Query");
        let mut code = String::new();
        code.push_str("// Java client code using Apollo Android\n");
        code.push_str("import com.apollographql.apollo.ApolloClient;\n\n");
        code.push_str(&format!("public class {}Query {{\n", name));
        code.push_str("    private static final String QUERY = \"\"\"\n");
        code.push_str(query);
        code.push_str("\n    \"\"\";\n\n");
        code.push_str("    // Usage with Apollo Android client\n");
        code.push_str("    // ApolloClient apolloClient = ApolloClient.builder()\n");
        code.push_str("    //     .serverUrl(\"http://localhost:8000/graphql\")\n");
        code.push_str("    //     .build();\n");
        code.push_str("}\n");
        Ok(code)
    }

    fn generate_csharp(query: &str, operation_name: Option<&str>) -> anyhow::Result<String> {
        let name = operation_name.unwrap_or("Query");
        let mut code = String::new();
        code.push_str("// C# client code using GraphQL.Client\n");
        code.push_str("using GraphQL;\n");
        code.push_str("using GraphQL.Client.Http;\n\n");
        code.push_str(&format!("public class {}Client\n", name));
        code.push_str("{\n");
        code.push_str("    private const string Query = @\"\n");
        code.push_str(query);
        code.push_str("\n    \";\n\n");
        code.push_str("    // Usage:\n");
        code.push_str("    // var graphQLClient = new GraphQLHttpClient(\"http://localhost:8000/graphql\");\n");
        code.push_str("    // var request = new GraphQLRequest { Query = Query };\n");
        code.push_str(
            "    // var response = await graphQLClient.SendQueryAsync<ResponseType>(request);\n",
        );
        code.push_str("}\n");
        Ok(code)
    }

    fn generate_swift(query: &str, operation_name: Option<&str>) -> anyhow::Result<String> {
        let name = operation_name.unwrap_or("Query");
        let mut code = String::new();
        code.push_str("// Swift client code using Apollo iOS\n");
        code.push_str("import Apollo\n\n");
        code.push_str(&format!("let {}Query = \"\"\"\n", name));
        code.push_str(query);
        code.push_str("\n\"\"\"\n\n");
        code.push_str("// Usage with Apollo iOS client\n");
        code.push_str(
            "// let apollo = ApolloClient(url: URL(string: \"http://localhost:8000/graphql\")!)\n",
        );
        code.push_str(&format!(
            "// apollo.fetch(query: {}Query) {{ result in\n",
            name
        ));
        code.push_str("//     switch result {\n");
        code.push_str("//     case .success(let graphQLResult):\n");
        code.push_str("//         print(graphQLResult.data)\n");
        code.push_str("//     case .failure(let error):\n");
        code.push_str("//         print(error)\n");
        code.push_str("//     }\n");
        code.push_str("// }\n");
        Ok(code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playground_config_default() {
        let config = PlaygroundConfig::default();
        assert_eq!(config.endpoint, "/playground");
        assert!(config.enable_tabs);
        assert!(config.enable_history);
    }

    #[test]
    fn test_playground_theme() {
        assert_eq!(PlaygroundTheme::Dark.as_str(), "dark");
        assert_eq!(PlaygroundTheme::Light.as_str(), "light");
    }

    #[test]
    fn test_code_gen_target() {
        assert_eq!(CodeGenTarget::TypeScript.as_str(), "typescript");
        assert_eq!(CodeGenTarget::Rust.as_str(), "rust");
        assert_eq!(CodeGenTarget::Python.as_str(), "python");

        let all = CodeGenTarget::all();
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_query_tab_creation() {
        let tab = QueryTab::new("Test Query".to_string(), "{ __typename }".to_string());
        assert_eq!(tab.name, "Test Query");
        assert_eq!(tab.query, "{ __typename }");
        assert!(!tab.id.is_empty());
    }

    #[test]
    fn test_playground_settings_default() {
        let settings = PlaygroundSettings::default();
        assert_eq!(settings.theme, PlaygroundTheme::Dark);
        assert_eq!(settings.font_size, 14);
        assert!(!settings.tabs.is_empty());
    }

    #[test]
    fn test_editor_settings_default() {
        let settings = EditorSettings::default();
        assert_eq!(settings.tab_size, 2);
        assert!(settings.use_spaces);
        assert!(settings.line_numbers);
    }

    #[test]
    fn test_generate_playground_html() {
        let config = PlaygroundConfig::default();
        let html = generate_playground_html(&config);

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("OxiRS GraphQL Playground"));
        assert!(html.contains("/graphql"));
        assert!(html.contains("GraphQLPlayground.init"));
    }

    #[test]
    fn test_default_playground_tabs() {
        let tabs = get_default_playground_tabs();
        assert_eq!(tabs.len(), 4);

        assert_eq!(&tabs[0].name, &String::from("Store Information"));
        assert_eq!(&tabs[1].name, &String::from("List Subjects"));
        assert_eq!(&tabs[2].name, &String::from("SPARQL Query"));
        assert_eq!(&tabs[3].name, &String::from("Subscription Example"));
    }

    #[test]
    fn test_code_generator_typescript() {
        let query = "query TestQuery { __typename }";
        let code = CodeGenerator::generate_typescript(query, Some("TestQuery")).unwrap();

        assert!(code.contains("TypeScript"));
        assert!(code.contains("TESTQUERY_QUERY"));
        assert!(code.contains(query));
    }

    #[test]
    fn test_code_generator_rust() {
        let query = "query TestQuery { __typename }";
        let code = CodeGenerator::generate_rust(query, Some("TestQuery")).unwrap();

        assert!(code.contains("Rust"));
        assert!(code.contains("GraphQLQuery"));
        assert!(code.contains(query));
    }

    #[test]
    fn test_code_generator_python() {
        let query = "query TestQuery { __typename }";
        let code = CodeGenerator::generate_python(query, Some("TestQuery")).unwrap();

        assert!(code.contains("Python"));
        assert!(code.contains("gql"));
        assert!(code.contains(query));
    }

    #[test]
    fn test_code_generator_all_targets() {
        let query = "query Test { __typename }";

        for target in CodeGenTarget::all() {
            let result = CodeGenerator::generate(target, query, Some("Test"));
            assert!(result.is_ok(), "Failed for target: {:?}", target);
        }
    }

    #[test]
    fn test_playground_config_custom() {
        let config = PlaygroundConfig {
            title: String::from("Custom Playground"),
            default_theme: PlaygroundTheme::Light,
            enable_tracing: false,
            ..Default::default()
        };

        let html = generate_playground_html(&config);
        let title_check = String::from("Custom Playground");
        assert!(html.contains(&title_check));
        assert!(html.contains("editor.theme"));
    }
}
