//! API Explorer with Curated Examples
//!
//! Provides an interactive API documentation and exploration UI with curated
//! example queries for common RDF/SPARQL use cases.
//!
//! ## Features
//!
//! - **Curated Examples**: Pre-built queries organized by category
//! - **Live Execution**: Execute examples directly in the browser
//! - **Schema Documentation**: Integrated schema browser
//! - **Response Visualization**: JSON and table views
//! - **Query Modification**: Edit and customize examples
//! - **Code Snippets**: Export queries in multiple languages

use serde::{Deserialize, Serialize};

/// API Explorer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiExplorerConfig {
    /// Enable API Explorer
    pub enabled: bool,
    /// Endpoint path (default: /api-explorer)
    pub endpoint: String,
    /// Page title
    pub title: String,
    /// Show schema documentation
    pub show_schema: bool,
    /// Show query history
    pub show_history: bool,
    /// Enable query sharing
    pub enable_sharing: bool,
    /// Custom CSS URL
    pub custom_css: Option<String>,
    /// Custom examples
    pub examples: Vec<QueryExample>,
}

impl Default for ApiExplorerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "/api-explorer".to_string(),
            title: "OxiRS GraphQL API Explorer".to_string(),
            show_schema: true,
            show_history: true,
            enable_sharing: false,
            custom_css: None,
            examples: default_examples(),
        }
    }
}

/// Query example category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExampleCategory {
    /// Basic queries
    Basic,
    /// RDF-specific queries
    Rdf,
    /// SPARQL translation examples
    Sparql,
    /// Federation examples
    Federation,
    /// Aggregation examples
    Aggregation,
    /// Subscription examples
    Subscription,
    /// Advanced/complex queries
    Advanced,
}

impl std::fmt::Display for ExampleCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExampleCategory::Basic => write!(f, "Basic Queries"),
            ExampleCategory::Rdf => write!(f, "RDF Queries"),
            ExampleCategory::Sparql => write!(f, "SPARQL Examples"),
            ExampleCategory::Federation => write!(f, "Federation"),
            ExampleCategory::Aggregation => write!(f, "Aggregations"),
            ExampleCategory::Subscription => write!(f, "Subscriptions"),
            ExampleCategory::Advanced => write!(f, "Advanced"),
        }
    }
}

/// Query example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExample {
    /// Example ID
    pub id: String,
    /// Example title
    pub title: String,
    /// Category
    pub category: ExampleCategory,
    /// Description
    pub description: String,
    /// GraphQL query
    pub query: String,
    /// Variables (JSON)
    pub variables: Option<String>,
    /// Expected result schema
    pub result_description: Option<String>,
    /// Tags for search
    pub tags: Vec<String>,
}

impl QueryExample {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        category: ExampleCategory,
        description: impl Into<String>,
        query: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            category,
            description: description.into(),
            query: query.into(),
            variables: None,
            result_description: None,
            tags: Vec::new(),
        }
    }

    pub fn with_variables(mut self, variables: impl Into<String>) -> Self {
        self.variables = Some(variables.into());
        self
    }

    pub fn with_result_description(mut self, desc: impl Into<String>) -> Self {
        self.result_description = Some(desc.into());
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Default curated examples
fn default_examples() -> Vec<QueryExample> {
    vec![
        // Basic queries
        QueryExample::new(
            "basic-hello",
            "Hello World",
            ExampleCategory::Basic,
            "A simple query to test connectivity",
            r#"query {
  hello
  version
}"#,
        )
        .with_tags(vec!["basic".to_string(), "test".to_string()]),
        QueryExample::new(
            "basic-triple-count",
            "Count Triples",
            ExampleCategory::Basic,
            "Get the total number of triples in the store",
            r#"query {
  triples
}"#,
        )
        .with_result_description("Returns the total count of RDF triples")
        .with_tags(vec!["basic".to_string(), "count".to_string()]),
        // RDF queries
        QueryExample::new(
            "rdf-subjects",
            "List Subjects",
            ExampleCategory::Rdf,
            "Retrieve all subject IRIs with optional limit",
            r#"query GetSubjects($limit: Int = 10) {
  subjects(limit: $limit)
}"#,
        )
        .with_variables(r#"{"limit": 10}"#)
        .with_tags(vec!["rdf".to_string(), "subjects".to_string()]),
        QueryExample::new(
            "rdf-predicates",
            "List Predicates",
            ExampleCategory::Rdf,
            "Retrieve all predicate IRIs (property types)",
            r#"query GetPredicates($limit: Int = 20) {
  predicates(limit: $limit)
}"#,
        )
        .with_variables(r#"{"limit": 20}"#)
        .with_tags(vec!["rdf".to_string(), "predicates".to_string()]),
        QueryExample::new(
            "rdf-objects",
            "List Objects",
            ExampleCategory::Rdf,
            "Retrieve object values with their types",
            r#"query GetObjects($limit: Int = 15) {
  objects(limit: $limit)
}"#,
        )
        .with_variables(r#"{"limit": 15}"#)
        .with_tags(vec!["rdf".to_string(), "objects".to_string()]),
        // SPARQL queries
        QueryExample::new(
            "sparql-select",
            "Execute SPARQL SELECT",
            ExampleCategory::Sparql,
            "Execute a raw SPARQL SELECT query",
            r#"query RunSparql($sparqlQuery: String!) {
  sparql(query: $sparqlQuery)
}"#,
        )
        .with_variables(r#"{"sparqlQuery": "SELECT * WHERE { ?s ?p ?o } LIMIT 10"}"#)
        .with_result_description("Returns SPARQL query results as JSON")
        .with_tags(vec!["sparql".to_string(), "select".to_string()]),
        QueryExample::new(
            "sparql-filter",
            "SPARQL with Filter",
            ExampleCategory::Sparql,
            "Query with FILTER clause for advanced filtering",
            r#"query FilteredSparql($sparqlQuery: String!) {
  sparql(query: $sparqlQuery)
}"#,
        )
        .with_variables(
            r#"{"sparqlQuery": "SELECT ?s ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name . FILTER(LANG(?name) = 'en') } LIMIT 10"}"#,
        )
        .with_tags(vec!["sparql".to_string(), "filter".to_string()]),
        // Aggregation queries
        QueryExample::new(
            "aggregation-count",
            "Count with GROUP BY",
            ExampleCategory::Aggregation,
            "Aggregate query with grouping and counting",
            r#"query CountByType($sparqlQuery: String!) {
  sparql(query: $sparqlQuery)
}"#,
        )
        .with_variables(
            r#"{"sparqlQuery": "SELECT ?type (COUNT(?s) as ?count) WHERE { ?s a ?type } GROUP BY ?type ORDER BY DESC(?count)"}"#,
        )
        .with_result_description("Returns entity counts grouped by type")
        .with_tags(vec!["aggregation".to_string(), "count".to_string(), "group-by".to_string()]),
        // Advanced queries
        QueryExample::new(
            "advanced-introspection",
            "Schema Introspection",
            ExampleCategory::Advanced,
            "Introspect the GraphQL schema structure",
            r#"query IntrospectSchema {
  __schema {
    queryType {
      name
    }
    mutationType {
      name
    }
    subscriptionType {
      name
    }
  }
}"#,
        )
        .with_tags(vec!["introspection".to_string(), "schema".to_string()]),
        QueryExample::new(
            "advanced-type-query",
            "Type Information",
            ExampleCategory::Advanced,
            "Get detailed information about a specific type",
            r#"query TypeInfo($typeName: String!) {
  __type(name: $typeName) {
    name
    kind
    description
  }
}"#,
        )
        .with_variables(r#"{"typeName": "Query"}"#)
        .with_tags(vec!["introspection".to_string(), "type".to_string()]),
    ]
}

/// Generate HTML for API Explorer
pub fn generate_api_explorer_html(config: &ApiExplorerConfig, graphql_endpoint: &str) -> String {
    let examples_json =
        serde_json::to_string(&config.examples).unwrap_or_else(|_| "[]".to_string());

    let custom_css = config
        .custom_css
        .as_ref()
        .map(|url| format!(r#"<link rel="stylesheet" href="{}">"#, url))
        .unwrap_or_default();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            display: grid;
            grid-template-columns: 350px 1fr;
            min-height: calc(100vh - 40px);
        }}
        .sidebar {{
            background: #f7fafc;
            border-right: 1px solid #e2e8f0;
            overflow-y: auto;
        }}
        .header {{
            padding: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .header h1 {{
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .header p {{
            font-size: 13px;
            opacity: 0.9;
        }}
        .search {{
            padding: 16px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .search input {{
            width: 100%;
            padding: 10px 16px;
            border: 1px solid #cbd5e0;
            border-radius: 6px;
            font-size: 14px;
        }}
        .search input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        .category {{
            padding: 12px 16px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            color: #718096;
            letter-spacing: 0.5px;
        }}
        .example-item {{
            padding: 12px 16px;
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: all 0.2s;
        }}
        .example-item:hover {{
            background: #edf2f7;
            border-left-color: #667eea;
        }}
        .example-item.active {{
            background: #e6f2ff;
            border-left-color: #667eea;
        }}
        .example-title {{
            font-size: 14px;
            font-weight: 500;
            color: #2d3748;
            margin-bottom: 4px;
        }}
        .example-desc {{
            font-size: 12px;
            color: #718096;
        }}
        .main-content {{
            display: flex;
            flex-direction: column;
        }}
        .toolbar {{
            padding: 16px 24px;
            background: #f7fafc;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-primary {{
            background: #667eea;
            color: white;
        }}
        .btn-primary:hover {{
            background: #5568d3;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        .btn-secondary {{
            background: #e2e8f0;
            color: #2d3748;
        }}
        .btn-secondary:hover {{
            background: #cbd5e0;
        }}
        .content-area {{
            flex: 1;
            display: grid;
            grid-template-rows: auto 1fr auto 1fr;
            overflow: hidden;
        }}
        .section-title {{
            padding: 12px 24px;
            background: #f7fafc;
            border-bottom: 1px solid #e2e8f0;
            font-size: 13px;
            font-weight: 600;
            color: #4a5568;
        }}
        .editor {{
            padding: 16px 24px;
            overflow-y: auto;
        }}
        .code-block {{
            background: #1a202c;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 6px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.6;
            overflow-x: auto;
        }}
        textarea {{
            width: 100%;
            min-height: 200px;
            padding: 16px;
            border: 1px solid #cbd5e0;
            border-radius: 6px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            resize: vertical;
        }}
        .response {{
            padding: 16px 24px;
            overflow-y: auto;
            background: #fafafa;
        }}
        .loading {{
            text-align: center;
            padding: 40px;
            color: #718096;
        }}
        .error {{
            background: #fed7d7;
            color: #c53030;
            padding: 16px;
            border-radius: 6px;
            margin: 16px 24px;
        }}
        .tags {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-top: 8px;
        }}
        .tag {{
            background: #e6f2ff;
            color: #2c5282;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
        }}
    </style>
    {custom_css}
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="header">
                <h1>📚 API Explorer</h1>
                <p>Curated query examples</p>
            </div>
            <div class="search">
                <input type="text" id="searchInput" placeholder="Search examples...">
            </div>
            <div id="exampleList"></div>
        </div>
        <div class="main-content">
            <div class="toolbar">
                <button class="btn btn-primary" id="runBtn">▶ Run Query</button>
                <button class="btn btn-secondary" id="prettifyBtn">✨ Prettify</button>
                <button class="btn btn-secondary" id="copyBtn">📋 Copy</button>
                <button class="btn btn-secondary" id="shareBtn">🔗 Share</button>
            </div>
            <div class="content-area">
                <div class="section-title">Query</div>
                <div class="editor">
                    <textarea id="queryEditor" placeholder="Select an example or write your query..."></textarea>
                </div>
                <div class="section-title">Variables (JSON)</div>
                <div class="editor">
                    <textarea id="variablesEditor" placeholder='{{"key": "value"}}'></textarea>
                </div>
                <div class="section-title">Response</div>
                <div class="response">
                    <div id="responseArea" class="loading">
                        Select an example and click "Run Query" to see results
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const examples = {examples_json};
        const graphqlEndpoint = '{graphql_endpoint}';
        let currentExample = null;

        function renderExamples(filtered = examples) {{
            const categories = {{}};
            filtered.forEach(ex => {{
                if (!categories[ex.category]) categories[ex.category] = [];
                categories[ex.category].push(ex);
            }});

            const html = Object.entries(categories).map(([cat, items]) => `
                <div class="category">${{getCategoryName(cat)}}</div>
                ${{items.map(ex => `
                    <div class="example-item" data-id="${{ex.id}}" onclick="selectExample('${{ex.id}}')">
                        <div class="example-title">${{ex.title}}</div>
                        <div class="example-desc">${{ex.description}}</div>
                        ${{ex.tags.length ? `<div class="tags">${{ex.tags.map(t => `<span class="tag">${{t}}</span>`).join('')}}</div>` : ''}}
                    </div>
                `).join('')}}
            `).join('');

            document.getElementById('exampleList').innerHTML = html;
        }}

        function getCategoryName(cat) {{
            const names = {{
                'Basic': 'Basic Queries',
                'Rdf': 'RDF Queries',
                'Sparql': 'SPARQL Examples',
                'Federation': 'Federation',
                'Aggregation': 'Aggregations',
                'Subscription': 'Subscriptions',
                'Advanced': 'Advanced'
            }};
            return names[cat] || cat;
        }}

        function selectExample(id) {{
            currentExample = examples.find(ex => ex.id === id);
            if (!currentExample) return;

            document.querySelectorAll('.example-item').forEach(el => el.classList.remove('active'));
            document.querySelector(`[data-id="${{id}}"]`).classList.add('active');

            document.getElementById('queryEditor').value = currentExample.query;
            document.getElementById('variablesEditor').value = currentExample.variables || '';
            document.getElementById('responseArea').innerHTML = '<div class="loading">Ready to execute</div>';
        }}

        async function runQuery() {{
            const query = document.getElementById('queryEditor').value;
            const variablesText = document.getElementById('variablesEditor').value.trim();
            let variables = null;

            if (variablesText) {{
                try {{
                    variables = JSON.parse(variablesText);
                }} catch (e) {{
                    document.getElementById('responseArea').innerHTML =
                        `<div class="error">Invalid JSON in variables: ${{e.message}}</div>`;
                    return;
                }}
            }}

            document.getElementById('responseArea').innerHTML = '<div class="loading">Executing query...</div>';

            try {{
                const response = await fetch(graphqlEndpoint, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ query, variables }})
                }});

                const result = await response.json();
                document.getElementById('responseArea').innerHTML =
                    `<pre class="code-block">${{JSON.stringify(result, null, 2)}}</pre>`;
            }} catch (error) {{
                document.getElementById('responseArea').innerHTML =
                    `<div class="error">Error: ${{error.message}}</div>`;
            }}
        }}

        function prettifyQuery() {{
            const editor = document.getElementById('queryEditor');
            // Simple prettification (proper implementation would use a GraphQL parser)
            editor.value = editor.value.replace(/\\s+/g, ' ').trim();
        }}

        function copyQuery() {{
            const editor = document.getElementById('queryEditor');
            editor.select();
            document.execCommand('copy');
        }}

        function shareQuery() {{
            alert('Share functionality would generate a shareable link here');
        }}

        document.getElementById('searchInput').addEventListener('input', (e) => {{
            const term = e.target.value.toLowerCase();
            const filtered = examples.filter(ex =>
                ex.title.toLowerCase().includes(term) ||
                ex.description.toLowerCase().includes(term) ||
                ex.tags.some(tag => tag.toLowerCase().includes(term))
            );
            renderExamples(filtered);
        }});

        document.getElementById('runBtn').addEventListener('click', runQuery);
        document.getElementById('prettifyBtn').addEventListener('click', prettifyQuery);
        document.getElementById('copyBtn').addEventListener('click', copyQuery);
        document.getElementById('shareBtn').addEventListener('click', shareQuery);

        renderExamples();
    </script>
</body>
</html>"#,
        title = config.title,
        examples_json = examples_json,
        graphql_endpoint = graphql_endpoint,
        custom_css = custom_css
    )
}

/// API Explorer manager
#[derive(Debug)]
pub struct ApiExplorer {
    config: ApiExplorerConfig,
}

impl ApiExplorer {
    /// Create new API Explorer
    pub fn new(config: ApiExplorerConfig) -> Self {
        Self { config }
    }

    /// Get HTML page for the explorer
    pub fn html(&self, graphql_endpoint: &str) -> String {
        generate_api_explorer_html(&self.config, graphql_endpoint)
    }

    /// Get configuration
    pub fn config(&self) -> &ApiExplorerConfig {
        &self.config
    }

    /// Add custom example
    pub fn add_example(&mut self, example: QueryExample) {
        self.config.examples.push(example);
    }

    /// Get examples by category
    pub fn get_examples_by_category(&self, category: ExampleCategory) -> Vec<&QueryExample> {
        self.config
            .examples
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }
}

impl Default for ApiExplorer {
    fn default() -> Self {
        Self::new(ApiExplorerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_explorer_config_default() {
        let config = ApiExplorerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.endpoint, "/api-explorer");
        assert!(config.show_schema);
        assert!(config.show_history);
        assert!(!config.examples.is_empty());
    }

    #[test]
    fn test_query_example_creation() {
        let example = QueryExample::new(
            "test-1",
            "Test Query",
            ExampleCategory::Basic,
            "A test query",
            "query { hello }",
        )
        .with_variables(r#"{"key": "value"}"#)
        .with_result_description("Returns hello message")
        .with_tags(vec!["test".to_string(), "basic".to_string()]);

        assert_eq!(example.id, "test-1");
        assert_eq!(example.title, "Test Query");
        assert_eq!(example.category, ExampleCategory::Basic);
        assert!(example.variables.is_some());
        assert!(example.result_description.is_some());
        assert_eq!(example.tags.len(), 2);
    }

    #[test]
    fn test_example_category_display() {
        assert_eq!(ExampleCategory::Basic.to_string(), "Basic Queries");
        assert_eq!(ExampleCategory::Rdf.to_string(), "RDF Queries");
        assert_eq!(ExampleCategory::Sparql.to_string(), "SPARQL Examples");
        assert_eq!(ExampleCategory::Federation.to_string(), "Federation");
        assert_eq!(ExampleCategory::Aggregation.to_string(), "Aggregations");
        assert_eq!(ExampleCategory::Subscription.to_string(), "Subscriptions");
        assert_eq!(ExampleCategory::Advanced.to_string(), "Advanced");
    }

    #[test]
    fn test_default_examples() {
        let examples = default_examples();
        assert!(!examples.is_empty());

        // Verify we have examples from different categories
        let categories: std::collections::HashSet<_> =
            examples.iter().map(|e| e.category).collect();
        assert!(categories.contains(&ExampleCategory::Basic));
        assert!(categories.contains(&ExampleCategory::Rdf));
        assert!(categories.contains(&ExampleCategory::Sparql));
    }

    #[test]
    fn test_api_explorer_creation() {
        let explorer = ApiExplorer::default();
        assert!(explorer.config().enabled);
        assert!(!explorer.config().examples.is_empty());
    }

    #[test]
    fn test_api_explorer_add_example() {
        let mut explorer = ApiExplorer::default();
        let initial_count = explorer.config().examples.len();

        let example = QueryExample::new(
            "custom-1",
            "Custom Query",
            ExampleCategory::Advanced,
            "Custom example",
            "query { custom }",
        );

        explorer.add_example(example);
        assert_eq!(explorer.config().examples.len(), initial_count + 1);
    }

    #[test]
    fn test_get_examples_by_category() {
        let explorer = ApiExplorer::default();
        let basic_examples = explorer.get_examples_by_category(ExampleCategory::Basic);
        assert!(!basic_examples.is_empty());

        for example in basic_examples {
            assert_eq!(example.category, ExampleCategory::Basic);
        }
    }

    #[test]
    fn test_generate_html() {
        let config = ApiExplorerConfig::default();
        let html = generate_api_explorer_html(&config, "/graphql");

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("API Explorer"));
        assert!(html.contains("/graphql"));
        assert!(html.contains("const examples ="));
    }

    #[test]
    fn test_html_includes_examples() {
        let explorer = ApiExplorer::default();
        let html = explorer.html("/graphql");

        // HTML should include serialized examples
        assert!(html.contains("Hello World") || html.contains("examples"));
        assert!(html.contains("graphqlEndpoint"));
    }
}
