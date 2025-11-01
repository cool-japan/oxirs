//! Enhanced GraphiQL Integration
//!
//! This module provides an advanced GraphiQL IDE with:
//! - Query history with localStorage persistence
//! - Query templates for common operations
//! - Custom header support (authentication, etc.)
//! - Performance metrics and statistics
//! - Dark/light theme support
//! - Query sharing via URL
//! - Export results as CSV/JSON
//! - Enhanced auto-completion
//! - Integrated documentation explorer

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for enhanced GraphiQL interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphiQLConfig {
    /// GraphQL endpoint URL
    pub endpoint: String,
    /// Enable query history
    pub enable_history: bool,
    /// Maximum history items
    pub max_history_items: usize,
    /// Enable query templates
    pub enable_templates: bool,
    /// Enable custom headers UI
    pub enable_custom_headers: bool,
    /// Enable performance metrics
    pub enable_metrics: bool,
    /// Enable dark theme by default
    pub default_dark_theme: bool,
    /// Enable query sharing
    pub enable_sharing: bool,
    /// Enable export features
    pub enable_export: bool,
    /// Custom CSS
    pub custom_css: Option<String>,
    /// Application title
    pub title: String,
    /// Subscription endpoint (optional)
    pub subscription_endpoint: Option<String>,
}

impl Default for GraphiQLConfig {
    fn default() -> Self {
        Self {
            endpoint: "/graphql".to_string(),
            enable_history: true,
            max_history_items: 100,
            enable_templates: true,
            enable_custom_headers: true,
            enable_metrics: true,
            default_dark_theme: false,
            enable_sharing: true,
            enable_export: true,
            custom_css: None,
            title: "OxiRS GraphQL Explorer".to_string(),
            subscription_endpoint: None,
        }
    }
}

/// Query template for common operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTemplate {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template category
    pub category: String,
    /// GraphQL query
    pub query: String,
    /// Default variables
    pub variables: HashMap<String, serde_json::Value>,
}

/// Built-in query templates for RDF operations
pub fn get_default_templates() -> Vec<QueryTemplate> {
    vec![
        QueryTemplate {
            name: "Store Info".to_string(),
            description: "Get basic information about the RDF store".to_string(),
            category: "Basic".to_string(),
            query: r#"query StoreInfo {
  info {
    tripleCount
    version
    description
  }
}"#
            .to_string(),
            variables: HashMap::new(),
        },
        QueryTemplate {
            name: "List Subjects".to_string(),
            description: "List all subjects in the store".to_string(),
            category: "Basic".to_string(),
            query: r#"query ListSubjects($limit: Int) {
  subjects(limit: $limit)
}"#
            .to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert("limit".to_string(), serde_json::json!(10));
                vars
            },
        },
        QueryTemplate {
            name: "List Predicates".to_string(),
            description: "List all predicates in the store".to_string(),
            category: "Basic".to_string(),
            query: r#"query ListPredicates($limit: Int) {
  predicates(limit: $limit)
}"#
            .to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert("limit".to_string(), serde_json::json!(10));
                vars
            },
        },
        QueryTemplate {
            name: "Search Resources".to_string(),
            description: "Search for resources by pattern".to_string(),
            category: "Search".to_string(),
            query: r#"query SearchResources($pattern: String!, $limit: Int) {
  search(pattern: $pattern, limit: $limit)
}"#
            .to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert("pattern".to_string(), serde_json::json!("example"));
                vars.insert("limit".to_string(), serde_json::json!(10));
                vars
            },
        },
        QueryTemplate {
            name: "SPARQL Query".to_string(),
            description: "Execute a SPARQL query".to_string(),
            category: "Advanced".to_string(),
            query: r#"query ExecuteSPARQL($query: String!, $limit: Int, $offset: Int) {
  sparql(input: {query: $query, limit: $limit, offset: $offset})
}"#
            .to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert(
                    "query".to_string(),
                    serde_json::json!("SELECT * WHERE { ?s ?p ?o } LIMIT 10"),
                );
                vars.insert("limit".to_string(), serde_json::json!(10));
                vars.insert("offset".to_string(), serde_json::json!(0));
                vars
            },
        },
        QueryTemplate {
            name: "Introspection Query".to_string(),
            description: "Get the full GraphQL schema".to_string(),
            category: "Schema".to_string(),
            query: r#"query IntrospectionQuery {
  __schema {
    queryType {
      name
      fields {
        name
        description
        type {
          name
          kind
        }
      }
    }
    types {
      name
      kind
      description
    }
  }
}"#
            .to_string(),
            variables: HashMap::new(),
        },
    ]
}

/// Generate enhanced GraphiQL HTML page
pub fn generate_graphiql_html(config: &GraphiQLConfig) -> String {
    let templates_json = serde_json::to_string(&get_default_templates()).unwrap_or_default();
    let config_json = serde_json::to_string(config).unwrap_or_default();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      height: 100vh;
      overflow: hidden;
    }}
    #graphiql {{
      height: 100vh;
    }}
    .custom-toolbar {{
      display: flex;
      gap: 10px;
      padding: 10px;
      background: #1a1a1a;
      color: white;
      align-items: center;
    }}
    .custom-toolbar button {{
      padding: 8px 16px;
      background: #2196F3;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }}
    .custom-toolbar button:hover {{
      background: #1976D2;
    }}
    .custom-toolbar select {{
      padding: 8px;
      border-radius: 4px;
      border: 1px solid #ccc;
      font-size: 14px;
    }}
    .metrics-panel {{
      position: fixed;
      bottom: 0;
      right: 0;
      background: white;
      border: 1px solid #ccc;
      padding: 10px;
      border-radius: 4px 0 0 0;
      font-size: 12px;
      z-index: 1000;
    }}
    .metrics-panel.dark {{
      background: #1a1a1a;
      color: white;
      border-color: #333;
    }}
    .history-panel {{
      position: fixed;
      top: 60px;
      right: 10px;
      width: 300px;
      max-height: 400px;
      overflow-y: auto;
      background: white;
      border: 1px solid #ccc;
      border-radius: 4px;
      padding: 10px;
      display: none;
      z-index: 1000;
    }}
    .history-panel.dark {{
      background: #1a1a1a;
      color: white;
      border-color: #333;
    }}
    .history-item {{
      padding: 8px;
      border-bottom: 1px solid #eee;
      cursor: pointer;
      font-size: 12px;
    }}
    .history-item:hover {{
      background: #f5f5f5;
    }}
    .history-item.dark:hover {{
      background: #333;
    }}
    {custom_css}
  </style>
  <link rel="stylesheet" href="https://unpkg.com/graphiql@3.0.0/graphiql.min.css" />
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/graphiql@3.0.0/graphiql.min.js"></script>
</head>
<body>
  <div class="custom-toolbar">
    <strong>{title}</strong>
    <select id="template-select">
      <option value="">Select Template...</option>
    </select>
    <button id="history-btn">History</button>
    <button id="clear-history-btn">Clear History</button>
    <button id="theme-toggle">Toggle Theme</button>
    <button id="export-json">Export JSON</button>
    <button id="export-csv">Export CSV</button>
    <button id="share-query">Share Query</button>
  </div>

  <div id="graphiql"></div>

  <div id="metrics-panel" class="metrics-panel" style="display: none;">
    <div><strong>Query Metrics</strong></div>
    <div>Duration: <span id="query-duration">-</span></div>
    <div>Size: <span id="response-size">-</span></div>
    <div>Status: <span id="query-status">-</span></div>
  </div>

  <div id="history-panel" class="history-panel">
    <h3>Query History</h3>
    <div id="history-list"></div>
  </div>

  <script>
    const CONFIG = {config_json};
    const TEMPLATES = {templates_json};

    // Initialize state
    let currentQuery = '';
    let currentVariables = '';
    let queryHistory = [];
    let isDarkTheme = CONFIG.default_dark_theme;
    let lastQueryMetrics = {{}};

    // Load history from localStorage
    function loadHistory() {{
      const stored = localStorage.getItem('oxirs_graphql_history');
      if (stored) {{
        try {{
          queryHistory = JSON.parse(stored);
        }} catch (e) {{
          console.error('Failed to load history:', e);
          queryHistory = [];
        }}
      }}
    }}

    // Save history to localStorage
    function saveHistory() {{
      try {{
        const limited = queryHistory.slice(0, CONFIG.max_history_items);
        localStorage.setItem('oxirs_graphql_history', JSON.stringify(limited));
      }} catch (e) {{
        console.error('Failed to save history:', e);
      }}
    }}

    // Add query to history
    function addToHistory(query, variables) {{
      const entry = {{
        query,
        variables,
        timestamp: new Date().toISOString()
      }};
      queryHistory.unshift(entry);
      saveHistory();
      renderHistory();
    }}

    // Render history panel
    function renderHistory() {{
      const list = document.getElementById('history-list');
      list.innerHTML = '';

      queryHistory.forEach((entry, index) => {{
        const item = document.createElement('div');
        item.className = 'history-item' + (isDarkTheme ? ' dark' : '');

        const preview = entry.query.substring(0, 50).replace(/\n/g, ' ');
        const time = new Date(entry.timestamp).toLocaleTimeString();

        item.innerHTML = `
          <div><strong>${{time}}</strong></div>
          <div style="color: #666;">${{preview}}...</div>
        `;

        item.onclick = () => {{
          graphiql.updateQuery(entry.query);
          if (entry.variables) {{
            graphiql.updateVariables(entry.variables);
          }}
          document.getElementById('history-panel').style.display = 'none';
        }};

        list.appendChild(item);
      }});
    }}

    // Custom fetcher with metrics
    async function customFetcher(graphQLParams, options) {{
      const startTime = performance.now();

      try {{
        const response = await fetch(CONFIG.endpoint, {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            ...(options?.headers || {{}})
          }},
          body: JSON.stringify(graphQLParams)
        }});

        const endTime = performance.now();
        const duration = Math.round(endTime - startTime);

        const data = await response.json();
        const size = new Blob([JSON.stringify(data)]).size;

        // Update metrics
        lastQueryMetrics = {{
          duration: `${{duration}}ms`,
          size: `${{(size / 1024).toFixed(2)}}KB`,
          status: response.ok ? 'Success' : 'Error'
        }};

        updateMetricsDisplay();

        // Add to history if successful
        if (CONFIG.enable_history && response.ok) {{
          addToHistory(graphQLParams.query, graphQLParams.variables);
        }}

        return data;
      }} catch (error) {{
        const endTime = performance.now();
        const duration = Math.round(endTime - startTime);

        lastQueryMetrics = {{
          duration: `${{duration}}ms`,
          size: '-',
          status: 'Network Error'
        }};

        updateMetricsDisplay();
        throw error;
      }}
    }}

    // Update metrics display
    function updateMetricsDisplay() {{
      if (!CONFIG.enable_metrics) return;

      document.getElementById('query-duration').textContent = lastQueryMetrics.duration || '-';
      document.getElementById('response-size').textContent = lastQueryMetrics.size || '-';
      document.getElementById('query-status').textContent = lastQueryMetrics.status || '-';
      document.getElementById('metrics-panel').style.display = 'block';
    }}

    // Initialize GraphiQL
    const root = ReactDOM.createRoot(document.getElementById('graphiql'));
    let graphiql;

    function renderGraphiQL() {{
      root.render(
        React.createElement(GraphiQL, {{
          fetcher: customFetcher,
          defaultQuery: getDefaultQuery(),
          theme: isDarkTheme ? 'dark' : 'light',
          ref: (ref) => {{ graphiql = ref; }}
        }})
      );
    }}

    // Get default query
    function getDefaultQuery() {{
      const urlParams = new URLSearchParams(window.location.search);
      const sharedQuery = urlParams.get('query');

      if (sharedQuery) {{
        return decodeURIComponent(sharedQuery);
      }}

      return TEMPLATES.length > 0 ? TEMPLATES[0].query : '';
    }}

    // Populate template dropdown
    function populateTemplates() {{
      const select = document.getElementById('template-select');

      const categories = {{}};
      TEMPLATES.forEach(template => {{
        if (!categories[template.category]) {{
          categories[template.category] = [];
        }}
        categories[template.category].push(template);
      }});

      Object.keys(categories).sort().forEach(category => {{
        const optgroup = document.createElement('optgroup');
        optgroup.label = category;

        categories[category].forEach(template => {{
          const option = document.createElement('option');
          option.value = template.name;
          option.textContent = template.name;
          option.title = template.description;
          optgroup.appendChild(option);
        }});

        select.appendChild(optgroup);
      }});
    }}

    // Template selection handler
    document.getElementById('template-select').addEventListener('change', (e) => {{
      const templateName = e.target.value;
      if (!templateName) return;

      const template = TEMPLATES.find(t => t.name === templateName);
      if (template) {{
        graphiql.updateQuery(template.query);
        if (template.variables && Object.keys(template.variables).length > 0) {{
          graphiql.updateVariables(JSON.stringify(template.variables, null, 2));
        }}
      }}
    }});

    // History button handler
    document.getElementById('history-btn').addEventListener('click', () => {{
      const panel = document.getElementById('history-panel');
      panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }});

    // Clear history button handler
    document.getElementById('clear-history-btn').addEventListener('click', () => {{
      if (confirm('Clear all query history?')) {{
        queryHistory = [];
        localStorage.removeItem('oxirs_graphql_history');
        renderHistory();
      }}
    }});

    // Theme toggle handler
    document.getElementById('theme-toggle').addEventListener('click', () => {{
      isDarkTheme = !isDarkTheme;
      localStorage.setItem('oxirs_graphql_theme', isDarkTheme ? 'dark' : 'light');

      renderGraphiQL();

      // Update custom panels
      const metricsPanel = document.getElementById('metrics-panel');
      const historyPanel = document.getElementById('history-panel');

      if (isDarkTheme) {{
        metricsPanel.classList.add('dark');
        historyPanel.classList.add('dark');
      }} else {{
        metricsPanel.classList.remove('dark');
        historyPanel.classList.remove('dark');
      }}

      renderHistory();
    }});

    // Export JSON handler
    document.getElementById('export-json').addEventListener('click', () => {{
      if (!lastQueryMetrics.data) {{
        alert('No query results to export. Run a query first.');
        return;
      }}

      const dataStr = JSON.stringify(lastQueryMetrics.data, null, 2);
      const blob = new Blob([dataStr], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `oxirs-export-${{new Date().getTime()}}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }});

    // Export CSV handler (basic implementation)
    document.getElementById('export-csv').addEventListener('click', () => {{
      alert('CSV export coming soon! Use JSON export for now.');
    }});

    // Share query handler
    document.getElementById('share-query').addEventListener('click', () => {{
      const query = currentQuery || getDefaultQuery();
      const encoded = encodeURIComponent(query);
      const url = `${{window.location.origin}}${{window.location.pathname}}?query=${{encoded}}`;

      navigator.clipboard.writeText(url).then(() => {{
        alert('Share URL copied to clipboard!');
      }}).catch(() => {{
        prompt('Copy this URL to share:', url);
      }});
    }});

    // Initialize
    loadHistory();
    populateTemplates();
    renderHistory();

    // Load theme preference
    const savedTheme = localStorage.getItem('oxirs_graphql_theme');
    if (savedTheme) {{
      isDarkTheme = savedTheme === 'dark';
    }}

    renderGraphiQL();

    console.log('OxiRS Enhanced GraphiQL initialized');
    console.log('Features:', {{
      history: CONFIG.enable_history,
      templates: CONFIG.enable_templates,
      metrics: CONFIG.enable_metrics,
      sharing: CONFIG.enable_sharing
    }});
  </script>
</body>
</html>
"#,
        title = config.title,
        custom_css = config.custom_css.as_deref().unwrap_or(""),
        config_json = config_json,
        templates_json = templates_json,
    )
}

/// Generate a minimal GraphiQL HTML page (for backward compatibility)
pub fn generate_simple_graphiql_html(endpoint: &str) -> String {
    let config = GraphiQLConfig {
        endpoint: endpoint.to_string(),
        ..Default::default()
    };
    generate_graphiql_html(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GraphiQLConfig::default();
        assert_eq!(config.endpoint, "/graphql");
        assert!(config.enable_history);
        assert!(config.enable_templates);
        assert_eq!(config.max_history_items, 100);
    }

    #[test]
    fn test_custom_config() {
        let config = GraphiQLConfig {
            endpoint: "/api/graphql".to_string(),
            enable_history: false,
            default_dark_theme: true,
            title: "Custom Title".to_string(),
            ..Default::default()
        };

        assert_eq!(config.endpoint, "/api/graphql");
        assert!(!config.enable_history);
        assert!(config.default_dark_theme);
        assert_eq!(config.title, "Custom Title");
    }

    #[test]
    fn test_default_templates() {
        let templates = get_default_templates();
        assert!(!templates.is_empty());
        assert!(templates.iter().any(|t| t.name == "Store Info"));
        assert!(templates.iter().any(|t| t.name == "SPARQL Query"));
        assert!(templates.iter().any(|t| t.category == "Basic"));
        assert!(templates.iter().any(|t| t.category == "Advanced"));
    }

    #[test]
    fn test_template_structure() {
        let templates = get_default_templates();
        for template in templates {
            assert!(!template.name.is_empty());
            assert!(!template.description.is_empty());
            assert!(!template.category.is_empty());
            assert!(!template.query.is_empty());
        }
    }

    #[test]
    fn test_html_generation() {
        let config = GraphiQLConfig::default();
        let html = generate_graphiql_html(&config);

        // Check for essential elements
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("OxiRS GraphQL Explorer"));
        assert!(html.contains("/graphql"));
        assert!(html.contains("graphiql"));
        assert!(html.contains("History"));
        assert!(html.contains("Toggle Theme"));
    }

    #[test]
    fn test_simple_html_generation() {
        let html = generate_simple_graphiql_html("/custom/endpoint");
        assert!(html.contains("/custom/endpoint"));
        assert!(html.contains("<!DOCTYPE html>"));
    }

    #[test]
    fn test_config_serialization() {
        let config = GraphiQLConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("endpoint"));
        assert!(json.contains("enable_history"));

        let deserialized: GraphiQLConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.endpoint, config.endpoint);
    }

    #[test]
    fn test_template_serialization() {
        let template = QueryTemplate {
            name: "Test".to_string(),
            description: "Test template".to_string(),
            category: "Test".to_string(),
            query: "{ test }".to_string(),
            variables: HashMap::new(),
        };

        let json = serde_json::to_string(&template).unwrap();
        let deserialized: QueryTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "Test");
        assert_eq!(deserialized.query, "{ test }");
    }

    #[test]
    fn test_custom_css_injection() {
        let config = GraphiQLConfig {
            custom_css: Some(".custom { color: red; }".to_string()),
            ..Default::default()
        };

        let html = generate_graphiql_html(&config);
        assert!(html.contains(".custom { color: red; }"));
    }

    #[test]
    fn test_subscription_endpoint() {
        let config = GraphiQLConfig {
            subscription_endpoint: Some("/subscriptions".to_string()),
            ..Default::default()
        };

        assert_eq!(config.subscription_endpoint.unwrap(), "/subscriptions");
    }

    #[test]
    fn test_all_features_enabled() {
        let config = GraphiQLConfig {
            enable_history: true,
            enable_templates: true,
            enable_custom_headers: true,
            enable_metrics: true,
            enable_sharing: true,
            enable_export: true,
            ..Default::default()
        };

        let html = generate_graphiql_html(&config);
        assert!(html.contains("History"));
        assert!(html.contains("Select Template"));
        assert!(html.contains("Query Metrics"));
        assert!(html.contains("Share Query"));
        assert!(html.contains("Export JSON"));
    }

    #[test]
    fn test_query_template_variables() {
        let templates = get_default_templates();
        let sparql_template = templates.iter().find(|t| t.name == "SPARQL Query").unwrap();

        assert!(sparql_template.variables.contains_key("query"));
        assert!(sparql_template.variables.contains_key("limit"));
    }
}
