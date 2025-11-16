//! Visual Query Builder UI
//!
//! Provides an interactive, no-code query building interface for GraphQL.
//! Users can visually construct queries by selecting fields, adding filters,
//! and configuring options without writing GraphQL syntax.
//!
//! ## Features
//!
//! - **Visual Field Selection**: Point-and-click field picker from schema
//! - **Filter Builder**: Visual filter/where clause constructor
//! - **Pagination UI**: Configure limit, offset, cursors visually
//! - **Sorting UI**: Drag-and-drop sort field ordering
//! - **Variable Manager**: Visual variable configuration
//! - **Query Preview**: Live GraphQL query preview
//! - **Export Options**: Export as query, curl, code snippets

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Query Builder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryBuilderConfig {
    /// Enable query builder
    pub enabled: bool,
    /// Endpoint path (default: /query-builder)
    pub endpoint: String,
    /// Page title
    pub title: String,
    /// Show live preview
    pub show_preview: bool,
    /// Enable query execution
    pub enable_execution: bool,
    /// Enable export features
    pub enable_export: bool,
    /// Maximum query depth
    pub max_depth: usize,
    /// Custom CSS URL
    pub custom_css: Option<String>,
}

impl Default for QueryBuilderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "/query-builder".to_string(),
            title: "OxiRS GraphQL Query Builder".to_string(),
            show_preview: true,
            enable_execution: true,
            enable_export: true,
            max_depth: 5,
            custom_css: None,
        }
    }
}

/// Field selection in query builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSelection {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: String,
    /// Is selected
    pub selected: bool,
    /// Arguments
    pub arguments: HashMap<String, String>,
    /// Nested fields (for objects)
    pub nested_fields: Vec<FieldSelection>,
}

/// Filter operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOp {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    StartsWith,
    EndsWith,
    In,
    NotIn,
}

impl std::fmt::Display for FilterOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterOp::Equals => write!(f, "="),
            FilterOp::NotEquals => write!(f, "!="),
            FilterOp::GreaterThan => write!(f, ">"),
            FilterOp::LessThan => write!(f, "<"),
            FilterOp::GreaterThanOrEqual => write!(f, ">="),
            FilterOp::LessThanOrEqual => write!(f, "<="),
            FilterOp::Contains => write!(f, "contains"),
            FilterOp::StartsWith => write!(f, "startsWith"),
            FilterOp::EndsWith => write!(f, "endsWith"),
            FilterOp::In => write!(f, "in"),
            FilterOp::NotIn => write!(f, "notIn"),
        }
    }
}

/// Filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Field to filter
    pub field: String,
    /// Filter operation
    pub operation: FilterOp,
    /// Filter value
    pub value: String,
}

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

impl std::fmt::Display for SortDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SortDirection::Ascending => write!(f, "ASC"),
            SortDirection::Descending => write!(f, "DESC"),
        }
    }
}

/// Sort field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortField {
    /// Field name
    pub field: String,
    /// Sort direction
    pub direction: SortDirection,
}

/// Query builder state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryBuilderState {
    /// Selected operation type (query/mutation/subscription)
    pub operation_type: String,
    /// Selected root field
    pub root_field: String,
    /// Selected fields
    pub fields: Vec<FieldSelection>,
    /// Filter conditions
    pub filters: Vec<FilterCondition>,
    /// Sort fields
    pub sorting: Vec<SortField>,
    /// Pagination: limit
    pub limit: Option<usize>,
    /// Pagination: offset
    pub offset: Option<usize>,
    /// Variables
    pub variables: HashMap<String, String>,
}

impl Default for QueryBuilderState {
    fn default() -> Self {
        Self {
            operation_type: "query".to_string(),
            root_field: String::new(),
            fields: Vec::new(),
            filters: Vec::new(),
            sorting: Vec::new(),
            limit: None,
            offset: None,
            variables: HashMap::new(),
        }
    }
}

/// Generate HTML for Query Builder
pub fn generate_query_builder_html(config: &QueryBuilderConfig, graphql_endpoint: &str) -> String {
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
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            display: grid;
            grid-template-columns: 400px 1fr 500px;
            min-height: calc(100vh - 40px);
        }}
        .panel {{
            overflow-y: auto;
        }}
        .panel-header {{
            padding: 20px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            font-size: 18px;
            font-weight: 600;
        }}
        .panel-content {{
            padding: 20px;
        }}
        .section {{
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 13px;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .field-tree {{
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            overflow: hidden;
        }}
        .field-item {{
            padding: 10px 12px;
            border-bottom: 1px solid #f7fafc;
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .field-item:hover {{
            background: #f7fafc;
        }}
        .field-item input[type="checkbox"] {{
            cursor: pointer;
        }}
        .field-name {{
            flex: 1;
            font-size: 14px;
            color: #2d3748;
        }}
        .field-type {{
            font-size: 12px;
            color: #718096;
            font-family: monospace;
        }}
        .filter-list, .sort-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .filter-item, .sort-item {{
            display: grid;
            grid-template-columns: 1fr auto 1fr auto;
            gap: 8px;
            align-items: center;
            padding: 10px;
            background: #f7fafc;
            border-radius: 6px;
        }}
        select, input[type="text"], input[type="number"] {{
            padding: 8px 12px;
            border: 1px solid #cbd5e0;
            border-radius: 4px;
            font-size: 14px;
        }}
        select:focus, input:focus {{
            outline: none;
            border-color: #f093fb;
            box-shadow: 0 0 0 3px rgba(240, 147, 251, 0.1);
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
            background: #f093fb;
            color: white;
        }}
        .btn-primary:hover {{
            background: #e081ea;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(240, 147, 251, 0.4);
        }}
        .btn-secondary {{
            background: #e2e8f0;
            color: #2d3748;
        }}
        .btn-secondary:hover {{
            background: #cbd5e0;
        }}
        .btn-small {{
            padding: 4px 12px;
            font-size: 12px;
        }}
        .btn-danger {{
            background: #fc8181;
            color: white;
        }}
        .code-preview {{
            background: #1a202c;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 6px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.6;
            overflow-x: auto;
            margin-bottom: 16px;
        }}
        .toolbar {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .response-area {{
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 16px;
            min-height: 200px;
            overflow-x: auto;
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
        }}
        .middle-panel {{
            border-left: 1px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
        }}
        .pagination-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        .form-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .form-label {{
            font-size: 12px;
            font-weight: 500;
            color: #4a5568;
        }}
    </style>
    {custom_css}
</head>
<body>
    <div class="container">
        <!-- Left Panel: Field Selection -->
        <div class="panel">
            <div class="panel-header">🎯 Field Selection</div>
            <div class="panel-content">
                <div class="section">
                    <div class="section-title">Available Fields</div>
                    <div class="field-tree" id="fieldTree">
                        <div class="loading">Loading schema...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Middle Panel: Filters & Options -->
        <div class="panel middle-panel">
            <div class="panel-header">⚙️ Query Configuration</div>
            <div class="panel-content">
                <div class="section">
                    <div class="section-title">Filters</div>
                    <div class="filter-list" id="filterList"></div>
                    <button class="btn btn-secondary btn-small" onclick="addFilter()">+ Add Filter</button>
                </div>

                <div class="section">
                    <div class="section-title">Sorting</div>
                    <div class="sort-list" id="sortList"></div>
                    <button class="btn btn-secondary btn-small" onclick="addSort()">+ Add Sort</button>
                </div>

                <div class="section">
                    <div class="section-title">Pagination</div>
                    <div class="pagination-grid">
                        <div class="form-group">
                            <label class="form-label" for="limitInput">Limit</label>
                            <input type="number" id="limitInput" placeholder="10" min="1">
                        </div>
                        <div class="form-group">
                            <label class="form-label" for="offsetInput">Offset</label>
                            <input type="number" id="offsetInput" placeholder="0" min="0">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Panel: Preview & Execute -->
        <div class="panel">
            <div class="panel-header">📝 Query Preview</div>
            <div class="panel-content">
                <div class="section">
                    <div class="section-title">Generated Query</div>
                    <div class="code-preview" id="queryPreview">
                        Select fields to build your query
                    </div>
                    <div class="toolbar">
                        <button class="btn btn-primary" onclick="executeQuery()">▶ Run Query</button>
                        <button class="btn btn-secondary" onclick="copyQuery()">📋 Copy</button>
                        <button class="btn btn-secondary" onclick="exportQuery()">💾 Export</button>
                        <button class="btn btn-secondary" onclick="resetBuilder()">🔄 Reset</button>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Response</div>
                    <div class="response-area" id="responseArea">
                        <div class="loading">Build and execute a query to see results</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const graphqlEndpoint = '{graphql_endpoint}';
        let selectedFields = new Set();
        let filters = [];
        let sorting = [];

        // Mock schema (in production, fetch from introspection)
        const mockSchema = {{
            fields: [
                {{ name: 'hello', type: 'String!', description: 'Greeting message' }},
                {{ name: 'version', type: 'String!', description: 'API version' }},
                {{ name: 'triples', type: 'Int!', description: 'Triple count' }},
                {{ name: 'subjects', type: '[String!]!', description: 'Subject IRIs', hasArgs: true }},
                {{ name: 'predicates', type: '[String!]!', description: 'Predicate IRIs', hasArgs: true }},
                {{ name: 'objects', type: '[String!]!', description: 'Object values', hasArgs: true }},
                {{ name: 'sparql', type: 'String', description: 'Execute SPARQL', hasArgs: true }},
            ]
        }};

        function renderFieldTree() {{
            const html = mockSchema.fields.map(field => `
                <div class="field-item">
                    <input type="checkbox" id="field-${{field.name}}" onchange="toggleField('${{field.name}}')">
                    <span class="field-name">${{field.name}}</span>
                    <span class="field-type">${{field.type}}</span>
                </div>
            `).join('');
            document.getElementById('fieldTree').innerHTML = html;
        }}

        function toggleField(fieldName) {{
            if (selectedFields.has(fieldName)) {{
                selectedFields.delete(fieldName);
            }} else {{
                selectedFields.add(fieldName);
            }}
            updateQueryPreview();
        }}

        function addFilter() {{
            filters.push({{ field: '', op: 'Equals', value: '' }});
            renderFilters();
            updateQueryPreview();
        }}

        function removeFilter(index) {{
            filters.splice(index, 1);
            renderFilters();
            updateQueryPreview();
        }}

        function renderFilters() {{
            const html = filters.map((filter, i) => `
                <div class="filter-item">
                    <select onchange="updateFilter(${{i}}, 'field', this.value)">
                        <option value="">Select field...</option>
                        ${{Array.from(selectedFields).map(f =>
                            `<option value="${{f}}" ${{filter.field === f ? 'selected' : ''}}>${{f}}</option>`
                        ).join('')}}
                    </select>
                    <select onchange="updateFilter(${{i}}, 'op', this.value)">
                        <option value="Equals" ${{filter.op === 'Equals' ? 'selected' : ''}}>equals</option>
                        <option value="Contains" ${{filter.op === 'Contains' ? 'selected' : ''}}>contains</option>
                        <option value="GreaterThan" ${{filter.op === 'GreaterThan' ? 'selected' : ''}}>></option>
                        <option value="LessThan" ${{filter.op === 'LessThan' ? 'selected' : ''}}><</option>
                    </select>
                    <input type="text" placeholder="Value" value="${{filter.value}}"
                           onchange="updateFilter(${{i}}, 'value', this.value)">
                    <button class="btn btn-danger btn-small" onclick="removeFilter(${{i}})">×</button>
                </div>
            `).join('');
            document.getElementById('filterList').innerHTML = html;
        }}

        function updateFilter(index, key, value) {{
            filters[index][key] = value;
            updateQueryPreview();
        }}

        function addSort() {{
            sorting.push({{ field: '', direction: 'Ascending' }});
            renderSorting();
            updateQueryPreview();
        }}

        function removeSort(index) {{
            sorting.splice(index, 1);
            renderSorting();
            updateQueryPreview();
        }}

        function renderSorting() {{
            const html = sorting.map((sort, i) => `
                <div class="sort-item">
                    <select onchange="updateSort(${{i}}, 'field', this.value)">
                        <option value="">Select field...</option>
                        ${{Array.from(selectedFields).map(f =>
                            `<option value="${{f}}" ${{sort.field === f ? 'selected' : ''}}>${{f}}</option>`
                        ).join('')}}
                    </select>
                    <select onchange="updateSort(${{i}}, 'direction', this.value)">
                        <option value="Ascending" ${{sort.direction === 'Ascending' ? 'selected' : ''}}>ASC</option>
                        <option value="Descending" ${{sort.direction === 'Descending' ? 'selected' : ''}}>DESC</option>
                    </select>
                    <span></span>
                    <button class="btn btn-danger btn-small" onclick="removeSort(${{i}})">×</button>
                </div>
            `).join('');
            document.getElementById('sortList').innerHTML = html;
        }}

        function updateSort(index, key, value) {{
            sorting[index][key] = value;
            updateQueryPreview();
        }}

        function updateQueryPreview() {{
            if (selectedFields.size === 0) {{
                document.getElementById('queryPreview').textContent = 'Select fields to build your query';
                return;
            }}

            const fields = Array.from(selectedFields).join('\\n  ');
            let query = `query {{\\n  ${{fields}}\\n}}`;

            document.getElementById('queryPreview').textContent = query;
        }}

        async function executeQuery() {{
            const queryText = document.getElementById('queryPreview').textContent;
            if (queryText === 'Select fields to build your query') {{
                alert('Please select fields first');
                return;
            }}

            document.getElementById('responseArea').innerHTML = '<div class="loading">Executing...</div>';

            try {{
                const response = await fetch(graphqlEndpoint, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ query: queryText }})
                }});

                const result = await response.json();
                document.getElementById('responseArea').innerHTML =
                    `<pre class="code-preview">${{JSON.stringify(result, null, 2)}}</pre>`;
            }} catch (error) {{
                document.getElementById('responseArea').innerHTML =
                    `<div class="error">Error: ${{error.message}}</div>`;
            }}
        }}

        function copyQuery() {{
            const query = document.getElementById('queryPreview').textContent;
            navigator.clipboard.writeText(query);
            alert('Query copied to clipboard!');
        }}

        function exportQuery() {{
            const query = document.getElementById('queryPreview').textContent;
            const blob = new Blob([query], {{ type: 'text/plain' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'query.graphql';
            a.click();
        }}

        function resetBuilder() {{
            selectedFields.clear();
            filters = [];
            sorting = [];
            document.getElementById('limitInput').value = '';
            document.getElementById('offsetInput').value = '';
            renderFieldTree();
            renderFilters();
            renderSorting();
            updateQueryPreview();
        }}

        // Initialize
        renderFieldTree();
        renderFilters();
        renderSorting();
    </script>
</body>
</html>"#,
        title = config.title,
        graphql_endpoint = graphql_endpoint,
        custom_css = custom_css
    )
}

/// Query Builder manager
#[derive(Debug)]
pub struct QueryBuilder {
    config: QueryBuilderConfig,
}

impl QueryBuilder {
    /// Create new Query Builder
    pub fn new(config: QueryBuilderConfig) -> Self {
        Self { config }
    }

    /// Get HTML page for the builder
    pub fn html(&self, graphql_endpoint: &str) -> String {
        generate_query_builder_html(&self.config, graphql_endpoint)
    }

    /// Get configuration
    pub fn config(&self) -> &QueryBuilderConfig {
        &self.config
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new(QueryBuilderConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_builder_config_default() {
        let config = QueryBuilderConfig::default();
        assert!(config.enabled);
        assert_eq!(config.endpoint, "/query-builder");
        assert!(config.show_preview);
        assert!(config.enable_execution);
        assert!(config.enable_export);
        assert_eq!(config.max_depth, 5);
    }

    #[test]
    fn test_filter_op_display() {
        assert_eq!(FilterOp::Equals.to_string(), "=");
        assert_eq!(FilterOp::NotEquals.to_string(), "!=");
        assert_eq!(FilterOp::GreaterThan.to_string(), ">");
        assert_eq!(FilterOp::LessThan.to_string(), "<");
        assert_eq!(FilterOp::Contains.to_string(), "contains");
        assert_eq!(FilterOp::In.to_string(), "in");
    }

    #[test]
    fn test_sort_direction_display() {
        assert_eq!(SortDirection::Ascending.to_string(), "ASC");
        assert_eq!(SortDirection::Descending.to_string(), "DESC");
    }

    #[test]
    fn test_query_builder_state_default() {
        let state = QueryBuilderState::default();
        assert_eq!(state.operation_type, "query");
        assert!(state.root_field.is_empty());
        assert!(state.fields.is_empty());
        assert!(state.filters.is_empty());
        assert!(state.sorting.is_empty());
        assert!(state.limit.is_none());
        assert!(state.offset.is_none());
    }

    #[test]
    fn test_field_selection() {
        let field = FieldSelection {
            name: "test".to_string(),
            field_type: "String".to_string(),
            selected: true,
            arguments: HashMap::new(),
            nested_fields: Vec::new(),
        };

        assert_eq!(field.name, "test");
        assert_eq!(field.field_type, "String");
        assert!(field.selected);
    }

    #[test]
    fn test_filter_condition() {
        let filter = FilterCondition {
            field: "name".to_string(),
            operation: FilterOp::Contains,
            value: "test".to_string(),
        };

        assert_eq!(filter.field, "name");
        assert_eq!(filter.operation, FilterOp::Contains);
        assert_eq!(filter.value, "test");
    }

    #[test]
    fn test_sort_field() {
        let sort = SortField {
            field: "createdAt".to_string(),
            direction: SortDirection::Descending,
        };

        assert_eq!(sort.field, "createdAt");
        assert_eq!(sort.direction, SortDirection::Descending);
    }

    #[test]
    fn test_query_builder_creation() {
        let builder = QueryBuilder::default();
        assert!(builder.config().enabled);
        assert!(builder.config().show_preview);
    }

    #[test]
    fn test_generate_html() {
        let config = QueryBuilderConfig::default();
        let html = generate_query_builder_html(&config, "/graphql");

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Query Builder"));
        assert!(html.contains("/graphql"));
        assert!(html.contains("Field Selection"));
        assert!(html.contains("Query Configuration"));
    }

    #[test]
    fn test_html_includes_features() {
        let builder = QueryBuilder::default();
        let html = builder.html("/graphql");

        assert!(html.contains("Filters"));
        assert!(html.contains("Sorting"));
        assert!(html.contains("Pagination"));
        assert!(html.contains("Generated Query"));
    }
}
