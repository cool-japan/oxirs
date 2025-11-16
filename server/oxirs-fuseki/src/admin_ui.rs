//! Admin UI Module
//!
//! Provides a comprehensive web-based administrative interface for managing
//! OxiRS Fuseki instances, datasets, queries, and monitoring.

use crate::store_ext::StoreExt;
use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// Admin UI state
pub struct AdminUIState {
    /// Store reference
    pub store: Arc<crate::store::Store>,
    /// Statistics
    pub stats: Arc<tokio::sync::RwLock<AdminStats>>,
}

/// Admin statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminStats {
    pub total_requests: u64,
    pub active_connections: usize,
    pub queries_executed: u64,
    pub avg_query_time_ms: f64,
    pub uptime_seconds: u64,
}

/// Dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub datasets: Vec<DatasetInfo>,
    pub statistics: SystemStatistics,
    pub recent_queries: Vec<QueryInfo>,
    pub system_health: HealthInfo,
}

/// Dataset information for admin UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub triple_count: usize,
    pub size_bytes: u64,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
}

/// System statistics for admin UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatistics {
    pub dataset_count: usize,
    pub total_triples: usize,
    pub total_queries: u64,
    pub avg_query_time_ms: f64,
    pub queries_per_second: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub uptime_seconds: u64,
}

/// Query information for admin UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryInfo {
    pub id: String,
    pub dataset: String,
    pub query: String,
    pub duration_ms: u64,
    pub result_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: String,
}

/// Health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthInfo {
    pub status: String,
    pub components: std::collections::HashMap<String, ComponentStatus>,
}

/// Component status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub status: String,
    pub message: Option<String>,
    pub last_check: chrono::DateTime<chrono::Utc>,
}

/// Query parameters for dataset list
#[derive(Debug, Deserialize)]
pub struct DatasetListParams {
    #[serde(default)]
    pub page: usize,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    pub search: Option<String>,
}

fn default_page_size() -> usize {
    20
}

impl AdminUIState {
    pub fn new(store: Arc<crate::store::Store>) -> Self {
        Self {
            store,
            stats: Arc::new(tokio::sync::RwLock::new(AdminStats::default())),
        }
    }

    /// Get dashboard data
    pub async fn get_dashboard_data(&self) -> Result<DashboardData> {
        // Get datasets
        let dataset_names = self.store.list_datasets()?;
        let mut datasets = Vec::new();

        for name in dataset_names {
            let triple_count = self.store.count_triples(&name);
            datasets.push(DatasetInfo {
                name,
                triple_count,
                size_bytes: 0, // Would calculate from store
                created_at: None,
                last_modified: None,
            });
        }

        // Get statistics
        let stats = self.stats.read().await;
        let statistics = SystemStatistics {
            dataset_count: datasets.len(),
            total_triples: datasets.iter().map(|d| d.triple_count).sum(),
            total_queries: stats.queries_executed,
            avg_query_time_ms: stats.avg_query_time_ms,
            queries_per_second: 0.0, // Would calculate from metrics
            memory_usage_mb: 0.0,    // Would get from system
            cpu_usage_percent: 0.0,  // Would get from system
            uptime_seconds: stats.uptime_seconds,
        };

        // Get recent queries (placeholder)
        let recent_queries = vec![];

        // Get system health
        let system_health = HealthInfo {
            status: "healthy".to_string(),
            components: std::collections::HashMap::new(),
        };

        Ok(DashboardData {
            datasets,
            statistics,
            recent_queries,
            system_health,
        })
    }
}

/// Serve admin UI index page
pub async fn serve_admin_ui() -> impl IntoResponse {
    Html(ADMIN_UI_HTML)
}

/// Get dashboard data as JSON
pub async fn get_dashboard_data(
    State(state): State<Arc<AdminUIState>>,
) -> Result<Json<DashboardData>, StatusCode> {
    state
        .get_dashboard_data()
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Admin UI HTML template
const ADMIN_UI_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OxiRS Fuseki Admin</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
        }

        .header {
            background: #2c3e50;
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
        }

        .header .subtitle {
            font-size: 0.875rem;
            opacity: 0.8;
            margin-top: 0.25rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .card h2 {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #2c3e50;
        }

        .stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #ecf0f1;
        }

        .stat:last-child {
            border-bottom: none;
        }

        .stat-label {
            color: #7f8c8d;
            font-size: 0.875rem;
        }

        .stat-value {
            font-size: 1.25rem;
            font-weight: 600;
            color: #2c3e50;
        }

        .metric-card {
            text-align: center;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #3498db;
            margin: 1rem 0;
        }

        .metric-label {
            color: #7f8c8d;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }

        .status-healthy {
            background: #2ecc71;
        }

        .status-warning {
            background: #f39c12;
        }

        .status-error {
            background: #e74c3c;
        }

        .dataset-list {
            margin-top: 1rem;
        }

        .dataset-item {
            padding: 1rem;
            border: 1px solid #ecf0f1;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .dataset-name {
            font-weight: 600;
            color: #2c3e50;
        }

        .dataset-info {
            font-size: 0.875rem;
            color: #7f8c8d;
        }

        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            font-size: 0.875rem;
            cursor: pointer;
            transition: background 0.2s;
        }

        .btn-primary {
            background: #3498db;
            color: white;
        }

        .btn-primary:hover {
            background: #2980b9;
        }

        .btn-secondary {
            background: #95a5a6;
            color: white;
        }

        .btn-secondary:hover {
            background: #7f8c8d;
        }

        .loading {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
        }

        .nav-tabs {
            display: flex;
            gap: 1rem;
            border-bottom: 2px solid #ecf0f1;
            margin-bottom: 2rem;
        }

        .nav-tab {
            padding: 0.75rem 1.5rem;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            cursor: pointer;
            font-size: 0.9375rem;
            color: #7f8c8d;
            transition: all 0.2s;
            margin-bottom: -2px;
        }

        .nav-tab.active {
            color: #3498db;
            border-bottom-color: #3498db;
        }

        .nav-tab:hover {
            color: #2c3e50;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>OxiRS Fuseki Administration</h1>
        <div class="subtitle">SPARQL Server Management Dashboard</div>
    </div>

    <div class="container">
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('dashboard')">Dashboard</button>
            <button class="nav-tab" onclick="switchTab('datasets')">Datasets</button>
            <button class="nav-tab" onclick="switchTab('queries')">Queries</button>
            <button class="nav-tab" onclick="switchTab('monitoring')">Monitoring</button>
        </div>

        <div id="dashboard" class="tab-content active">
            <div class="dashboard">
                <div class="card metric-card">
                    <div class="metric-label">Datasets</div>
                    <div class="metric-value" id="dataset-count">-</div>
                    <div class="metric-label">Total Datasets</div>
                </div>

                <div class="card metric-card">
                    <div class="metric-label">Triples</div>
                    <div class="metric-value" id="triple-count">-</div>
                    <div class="metric-label">Total Triples</div>
                </div>

                <div class="card metric-card">
                    <div class="metric-label">Queries</div>
                    <div class="metric-value" id="query-count">-</div>
                    <div class="metric-label">Total Executed</div>
                </div>

                <div class="card metric-card">
                    <div class="metric-label">Latency</div>
                    <div class="metric-value" id="avg-latency">-</div>
                    <div class="metric-label">Avg. Query Time (ms)</div>
                </div>
            </div>

            <div class="card">
                <h2>
                    <span class="status-indicator status-healthy"></span>
                    System Health
                </h2>
                <div id="system-health">
                    <div class="stat">
                        <span class="stat-label">Status</span>
                        <span class="stat-value">Healthy</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Uptime</span>
                        <span class="stat-value" id="uptime">-</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Memory Usage</span>
                        <span class="stat-value" id="memory-usage">-</span>
                    </div>
                </div>
            </div>
        </div>

        <div id="datasets" class="tab-content">
            <div class="card">
                <h2>Datasets</h2>
                <div id="dataset-list" class="dataset-list">
                    <div class="loading">Loading datasets...</div>
                </div>
            </div>
        </div>

        <div id="queries" class="tab-content">
            <div class="card">
                <h2>Recent Queries</h2>
                <div id="query-list">
                    <div class="loading">Loading queries...</div>
                </div>
            </div>
        </div>

        <div id="monitoring" class="tab-content">
            <div class="card">
                <h2>Real-time Monitoring</h2>
                <div id="monitoring-content">
                    <div class="loading">Loading monitoring data...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function switchTab(tabName) {
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');

            if (tabName === 'datasets') {
                loadDatasets();
            } else if (tabName === 'queries') {
                loadQueries();
            } else if (tabName === 'monitoring') {
                loadMonitoring();
            }
        }

        // Load dashboard data
        async function loadDashboard() {
            try {
                const response = await fetch('/admin/api/dashboard');
                const data = await response.json();

                document.getElementById('dataset-count').textContent = data.statistics.dataset_count;
                document.getElementById('triple-count').textContent = data.statistics.total_triples.toLocaleString();
                document.getElementById('query-count').textContent = data.statistics.total_queries.toLocaleString();
                document.getElementById('avg-latency').textContent = data.statistics.avg_query_time_ms.toFixed(2);
                document.getElementById('uptime').textContent = formatUptime(data.statistics.uptime_seconds);
                document.getElementById('memory-usage').textContent = data.statistics.memory_usage_mb.toFixed(2) + ' MB';

            } catch (error) {
                console.error('Failed to load dashboard:', error);
            }
        }

        // Load datasets
        async function loadDatasets() {
            try {
                const response = await fetch('/admin/api/dashboard');
                const data = await response.json();

                const list = document.getElementById('dataset-list');
                if (data.datasets.length === 0) {
                    list.innerHTML = '<div class="loading">No datasets found</div>';
                    return;
                }

                list.innerHTML = data.datasets.map(ds => `
                    <div class="dataset-item">
                        <div>
                            <div class="dataset-name">${ds.name}</div>
                            <div class="dataset-info">${ds.triple_count.toLocaleString()} triples</div>
                        </div>
                        <div>
                            <button class="btn btn-secondary" onclick="viewDataset('${ds.name}')">View</button>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load datasets:', error);
            }
        }

        // Load queries (placeholder)
        function loadQueries() {
            document.getElementById('query-list').innerHTML = '<div class="loading">No recent queries</div>';
        }

        // Load monitoring (placeholder)
        function loadMonitoring() {
            document.getElementById('monitoring-content').innerHTML = '<div class="loading">Real-time monitoring data will appear here</div>';
        }

        // Format uptime
        function formatUptime(seconds) {
            const days = Math.floor(seconds / 86400);
            const hours = Math.floor((seconds % 86400) / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${days}d ${hours}h ${minutes}m`;
        }

        // View dataset
        function viewDataset(name) {
            alert(`Viewing dataset: ${name}`);
        }

        // Initialize
        loadDashboard();
        setInterval(loadDashboard, 5000); // Refresh every 5 seconds
    </script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admin_ui_html_contains_title() {
        assert!(ADMIN_UI_HTML.contains("OxiRS Fuseki"));
    }

    #[test]
    fn test_dashboard_data_structure() {
        let stats = SystemStatistics {
            dataset_count: 5,
            total_triples: 10000,
            total_queries: 100,
            avg_query_time_ms: 15.5,
            queries_per_second: 10.0,
            memory_usage_mb: 512.0,
            cpu_usage_percent: 25.0,
            uptime_seconds: 3600,
        };

        assert_eq!(stats.dataset_count, 5);
        assert_eq!(stats.total_triples, 10000);
    }
}
