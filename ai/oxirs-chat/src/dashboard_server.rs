//! Dashboard Server Endpoints
//!
//! HTTP endpoints for accessing analytics dashboard data

use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use futures_util::sink::SinkExt;
use serde::Deserialize;
use std::sync::Arc;
use tracing::error;

use crate::dashboard::{
    DashboardAnalytics, DashboardOverview, ExportFormat, HealthAnalytics, QueryAnalytics,
    TimeRange, UserAnalytics,
};

/// Shared state for dashboard endpoints
#[derive(Clone)]
pub struct DashboardState {
    pub analytics: Arc<DashboardAnalytics>,
}

/// Query parameters for time range
#[derive(Debug, Deserialize)]
pub struct TimeRangeQuery {
    /// Start time (ISO 8601 format)
    pub start: Option<String>,
    /// End time (ISO 8601 format)
    pub end: Option<String>,
    /// Or use last N hours
    pub last_hours: Option<i64>,
    /// Or use last N days
    pub last_days: Option<i64>,
}

impl TimeRangeQuery {
    /// Convert to TimeRange
    pub fn to_time_range(&self) -> TimeRange {
        if let Some(hours) = self.last_hours {
            return TimeRange::last_hours(hours);
        }

        if let Some(days) = self.last_days {
            return TimeRange::last_days(days);
        }

        // Default to last 24 hours
        TimeRange::last_hours(24)
    }
}

/// Export query parameters
#[derive(Debug, Deserialize)]
pub struct ExportQuery {
    /// Export format
    pub format: Option<String>,
    /// Time range parameters
    #[serde(flatten)]
    pub time_range: TimeRangeQuery,
}

/// Get dashboard overview
pub async fn get_overview(
    State(state): State<DashboardState>,
) -> Result<Json<DashboardOverview>, StatusCode> {
    let overview = state.analytics.get_overview().await;
    Ok(Json(overview))
}

/// Get query analytics
pub async fn get_query_analytics(
    State(state): State<DashboardState>,
    Query(params): Query<TimeRangeQuery>,
) -> Result<Json<QueryAnalytics>, StatusCode> {
    let time_range = params.to_time_range();
    let analytics = state.analytics.get_query_analytics(time_range).await;
    Ok(Json(analytics))
}

/// Get user analytics
pub async fn get_user_analytics(
    State(state): State<DashboardState>,
    Query(params): Query<TimeRangeQuery>,
) -> Result<Json<UserAnalytics>, StatusCode> {
    let time_range = params.to_time_range();
    let analytics = state.analytics.get_user_analytics(time_range).await;
    Ok(Json(analytics))
}

/// Get health analytics
pub async fn get_health_analytics(
    State(state): State<DashboardState>,
    Query(params): Query<TimeRangeQuery>,
) -> Result<Json<HealthAnalytics>, StatusCode> {
    let time_range = params.to_time_range();
    let analytics = state.analytics.get_health_analytics(time_range).await;
    Ok(Json(analytics))
}

/// Export analytics data
pub async fn export_analytics(
    State(state): State<DashboardState>,
    Query(params): Query<ExportQuery>,
) -> Result<Vec<u8>, StatusCode> {
    let format = match params.format.as_deref() {
        Some("csv") => ExportFormat::Csv,
        Some("excel") | Some("xlsx") => ExportFormat::Excel,
        _ => ExportFormat::Json,
    };

    let time_range = params.time_range.to_time_range();

    match state.analytics.export_data(format, time_range).await {
        Ok(data) => Ok(data),
        Err(e) => {
            error!("Failed to export analytics: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Real-time metrics endpoint (Server-Sent Events)
pub async fn metrics_stream(
    State(state): State<DashboardState>,
) -> axum::response::sse::Sse<
    impl futures_util::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use futures_util::stream;
    use std::time::Duration;

    let stream = stream::unfold(state, |state| async move {
        tokio::time::sleep(Duration::from_secs(5)).await;

        let overview = state.analytics.get_overview().await;
        let event = axum::response::sse::Event::default()
            .json_data(overview)
            .expect("SSE event serialization should succeed");

        Some((Ok(event), state))
    });

    axum::response::sse::Sse::new(stream)
}

/// WebSocket endpoint for real-time dashboard updates
pub async fn dashboard_websocket(
    State(state): State<DashboardState>,
    ws: axum::extract::WebSocketUpgrade,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_dashboard_websocket(socket, state))
}

async fn handle_dashboard_websocket(socket: axum::extract::ws::WebSocket, state: DashboardState) {
    use futures_util::{sink::SinkExt, stream::StreamExt};

    let (sender, mut receiver) = socket.split();
    let sender = Arc::new(tokio::sync::Mutex::new(sender));

    // Spawn task to send periodic updates
    let sender_clone = sender.clone();
    let state_clone = state.clone();
    let update_task = tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            let overview = state_clone.analytics.get_overview().await;

            if let Ok(json) = serde_json::to_string(&overview) {
                let mut sender = sender_clone.lock().await;
                if sender
                    .send(axum::extract::ws::Message::Text(json))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    });

    // Handle incoming messages (commands)
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            axum::extract::ws::Message::Text(text) => {
                // Handle dashboard commands
                if let Ok(command) = serde_json::from_str::<DashboardCommand>(&text) {
                    handle_dashboard_command(command, &state, &sender).await;
                }
            }
            axum::extract::ws::Message::Ping(data) => {
                let mut sender = sender.lock().await;
                let _ = sender.send(axum::extract::ws::Message::Pong(data)).await;
            }
            axum::extract::ws::Message::Close(_) => break,
            _ => {}
        }
    }

    update_task.abort();
}

/// Dashboard command from WebSocket
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum DashboardCommand {
    #[serde(rename = "get_overview")]
    Overview,
    #[serde(rename = "get_query_analytics")]
    QueryAnalytics { time_range: String },
    #[serde(rename = "get_user_analytics")]
    UserAnalytics { time_range: String },
}

async fn handle_dashboard_command(
    command: DashboardCommand,
    state: &DashboardState,
    sender: &Arc<
        tokio::sync::Mutex<
            futures_util::stream::SplitSink<
                axum::extract::ws::WebSocket,
                axum::extract::ws::Message,
            >,
        >,
    >,
) {
    match command {
        DashboardCommand::Overview => {
            let overview = state.analytics.get_overview().await;
            if let Ok(json) = serde_json::to_string(&overview) {
                let mut sender = sender.lock().await;
                let _ = sender.send(axum::extract::ws::Message::Text(json)).await;
            }
        }
        DashboardCommand::QueryAnalytics { time_range } => {
            let range = if time_range == "24h" {
                TimeRange::last_hours(24)
            } else if time_range == "7d" {
                TimeRange::last_days(7)
            } else {
                TimeRange::last_hours(24)
            };

            let analytics = state.analytics.get_query_analytics(range).await;
            if let Ok(json) = serde_json::to_string(&analytics) {
                let mut sender = sender.lock().await;
                let _ = sender.send(axum::extract::ws::Message::Text(json)).await;
            }
        }
        DashboardCommand::UserAnalytics { time_range } => {
            let range = if time_range == "24h" {
                TimeRange::last_hours(24)
            } else if time_range == "7d" {
                TimeRange::last_days(7)
            } else {
                TimeRange::last_hours(24)
            };

            let analytics = state.analytics.get_user_analytics(range).await;
            if let Ok(json) = serde_json::to_string(&analytics) {
                let mut sender = sender.lock().await;
                let _ = sender.send(axum::extract::ws::Message::Text(json)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range_query() {
        let query = TimeRangeQuery {
            start: None,
            end: None,
            last_hours: Some(24),
            last_days: None,
        };

        let time_range = query.to_time_range();
        let duration = time_range.end - time_range.start;
        assert_eq!(duration.num_hours(), 24);
    }
}
