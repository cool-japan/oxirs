//! HTTP Server and WebSocket Implementation for OxiRS Chat

use axum::{
    extract::{ws::WebSocket, Path, Query, State, WebSocketUpgrade},
    http::{header, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
// use axum_extra::headers::{Authorization, Bearer}; // Removed due to compilation issues
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{broadcast, RwLock};
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{ChatManager, ChatSession, Message, MessageRole, ThreadInfo};

/// Server state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub chat_manager: Arc<ChatManager>,
    pub websocket_sessions: Arc<RwLock<HashMap<String, WebSocketSessionInfo>>>,
    pub broadcast_tx: broadcast::Sender<ServerMessage>,
    pub config: ServerConfig,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub session_timeout: Duration,
    pub enable_metrics: bool,
    pub cors_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8080,
            max_connections: 1000,
            session_timeout: Duration::from_secs(3600),
            enable_metrics: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// WebSocket session information
#[derive(Debug, Clone)]
pub struct WebSocketSessionInfo {
    pub session_id: String,
    pub user_id: Option<String>,
    pub connected_at: SystemTime,
    pub last_activity: SystemTime,
}

/// Server-wide broadcast messages
#[derive(Debug, Clone, Serialize)]
pub struct ServerMessage {
    pub message_type: ServerMessageType,
    pub session_id: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerMessageType {
    ChatMessage,
    SessionCreated,
    SessionClosed,
    SystemStatus,
}

/// REST API request/response types
#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub user_id: Option<String>,
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
    pub websocket_url: String,
}

#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub thread_id: Option<String>,
    pub parent_message_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message_id: String,
    pub content: String,
    pub role: String,
    pub timestamp: String,
    pub metadata: Option<serde_json::Value>,
    pub thread_id: Option<String>,
    pub parent_message_id: Option<String>,
    pub reactions: Vec<crate::MessageReaction>,
}

#[derive(Debug, Deserialize)]
pub struct SessionQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// WebSocket message types
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    #[serde(rename = "send_message")]
    SendMessage {
        content: String,
        thread_id: Option<String>,
        parent_message_id: Option<String>,
    },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "subscribe")]
    Subscribe { topics: Vec<String> },
    #[serde(rename = "add_reaction")]
    AddReaction {
        message_id: String,
        emoji: String,
        user_id: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum WebSocketResponse {
    #[serde(rename = "message")]
    Message {
        message_id: String,
        content: String,
        role: String,
        timestamp: String,
        metadata: Option<serde_json::Value>,
    },
    #[serde(rename = "pong")]
    Pong,
    #[serde(rename = "error")]
    Error { code: String, message: String },
    #[serde(rename = "status")]
    Status {
        status: String,
        data: serde_json::Value,
    },
}

/// Main server implementation
pub struct ChatServer {
    state: AppState,
}

impl ChatServer {
    pub fn new(chat_manager: Arc<ChatManager>, config: ServerConfig) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);

        let state = AppState {
            chat_manager,
            websocket_sessions: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
            config,
        };

        Self { state }
    }

    pub async fn serve(self) -> Result<(), Box<dyn std::error::Error>> {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

        let middleware = ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(cors);

        let app = Router::new()
            .route("/api/sessions", post(create_session))
            .route("/api/sessions/:session_id", get(get_session))
            .route("/api/sessions/:session_id/messages", post(send_message))
            .route("/api/sessions/:session_id/messages", get(get_messages))
            .route("/api/sessions/:session_id/threads", get(get_threads))
            .route(
                "/api/sessions/:session_id/threads/:thread_id/messages",
                get(get_thread_messages),
            )
            .route(
                "/api/sessions/:session_id/messages/:message_id/replies",
                get(get_message_replies),
            )
            .route(
                "/api/sessions/:session_id/messages/:message_id/reactions",
                post(add_reaction),
            )
            .route("/api/sessions/:session_id/ws", get(websocket_handler))
            .route("/api/stats", get(get_stats))
            .route("/health", get(health_check))
            .route("/metrics", get(metrics_handler))
            .layer(middleware)
            .with_state(self.state.clone());

        let addr = SocketAddr::from(([127, 0, 0, 1], self.state.config.port));
        info!("Starting chat server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

/// REST API Handlers
async fn create_session(
    State(state): State<AppState>,
    Json(request): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, StatusCode> {
    let session_id = Uuid::new_v4().to_string();

    match state.chat_manager.create_session(session_id.clone()).await {
        Ok(_) => {
            let websocket_url = format!(
                "ws://{}:{}/api/sessions/{}/ws",
                state.config.host, state.config.port, session_id
            );

            Ok(Json(CreateSessionResponse {
                session_id,
                websocket_url,
            }))
        }
        Err(e) => {
            error!("Failed to create session: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_session(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(session_arc) = state.chat_manager.get_session(&session_id).await {
        let session = session_arc.lock().await;
        Ok(Json(serde_json::json!({
            "session_id": session_id,
            "status": "active",
            "message_count": session.messages.len(),
            "created_at": session.created_at.to_rfc3339(),
            "last_activity": session.last_activity.to_rfc3339()
        })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn send_message(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<SendMessageRequest>,
) -> Result<Json<MessageResponse>, StatusCode> {
    if let Some(session_arc) = state.chat_manager.get_session(&session_id).await {
        let mut session = session_arc.lock().await;
        match session
            .process_message_with_options(
                request.content,
                request.thread_id,
                request.parent_message_id,
            )
            .await
        {
            Ok(message) => {
                // Save session after processing
                drop(session);
                let _ = state.chat_manager.save_session(&session_id).await;

                let response = MessageResponse {
                    message_id: message.id,
                    content: message.content,
                    role: match message.role {
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                        MessageRole::System => "system".to_string(),
                    },
                    timestamp: message.timestamp.to_rfc3339(),
                    metadata: message
                        .metadata
                        .map(|m| serde_json::to_value(m).unwrap_or_default()),
                    thread_id: message.thread_id,
                    parent_message_id: message.parent_message_id,
                    reactions: message.reactions,
                };

                Ok(Json(response))
            }
            Err(e) => {
                error!("Failed to process message: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn get_messages(
    Path(session_id): Path<String>,
    Query(params): Query<SessionQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<MessageResponse>>, StatusCode> {
    if let Some(session_arc) = state.chat_manager.get_session(&session_id).await {
        let session = session_arc.lock().await;
        let messages: Vec<MessageResponse> = session
            .get_history()
            .iter()
            .skip(params.offset.unwrap_or(0))
            .take(params.limit.unwrap_or(100))
            .map(|msg| MessageResponse {
                message_id: msg.id.clone(),
                content: msg.content.clone(),
                role: match msg.role {
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                    MessageRole::System => "system".to_string(),
                },
                timestamp: msg.timestamp.to_rfc3339(),
                metadata: msg
                    .metadata
                    .as_ref()
                    .map(|m| serde_json::to_value(m).unwrap_or_default()),
                thread_id: msg.thread_id.clone(),
                parent_message_id: msg.parent_message_id.clone(),
                reactions: msg.reactions.clone(),
            })
            .collect();

        Ok(Json(messages))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn websocket_handler(
    Path(session_id): Path<String>,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_websocket(socket, session_id, state))
}

async fn get_threads(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<ThreadInfo>>, StatusCode> {
    if let Some(session_arc) = state.chat_manager.get_session(&session_id).await {
        let session = session_arc.lock().await;
        let threads = session.get_threads();
        Ok(Json(threads))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn get_thread_messages(
    Path((session_id, thread_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Vec<MessageResponse>>, StatusCode> {
    if let Some(session_arc) = state.chat_manager.get_session(&session_id).await {
        let session = session_arc.lock().await;
        let messages: Vec<MessageResponse> = session
            .get_thread_messages(&thread_id)
            .into_iter()
            .map(|msg| MessageResponse {
                message_id: msg.id.clone(),
                content: msg.content.clone(),
                role: match msg.role {
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                    MessageRole::System => "system".to_string(),
                },
                timestamp: msg.timestamp.to_rfc3339(),
                metadata: msg
                    .metadata
                    .as_ref()
                    .map(|m| serde_json::to_value(m).unwrap_or_default()),
                thread_id: msg.thread_id.clone(),
                parent_message_id: msg.parent_message_id.clone(),
                reactions: msg.reactions.clone(),
            })
            .collect();

        Ok(Json(messages))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn get_message_replies(
    Path((session_id, message_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Vec<MessageResponse>>, StatusCode> {
    if let Some(session_arc) = state.chat_manager.get_session(&session_id).await {
        let session = session_arc.lock().await;
        let messages: Vec<MessageResponse> = session
            .get_replies(&message_id)
            .into_iter()
            .map(|msg| MessageResponse {
                message_id: msg.id.clone(),
                content: msg.content.clone(),
                role: match msg.role {
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                    MessageRole::System => "system".to_string(),
                },
                timestamp: msg.timestamp.to_rfc3339(),
                metadata: msg
                    .metadata
                    .as_ref()
                    .map(|m| serde_json::to_value(m).unwrap_or_default()),
                thread_id: msg.thread_id.clone(),
                parent_message_id: msg.parent_message_id.clone(),
                reactions: msg.reactions.clone(),
            })
            .collect();

        Ok(Json(messages))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

#[derive(Debug, Deserialize)]
pub struct AddReactionRequest {
    pub emoji: String,
    pub user_id: String,
}

async fn add_reaction(
    Path((session_id, message_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(request): Json<AddReactionRequest>,
) -> Result<StatusCode, StatusCode> {
    if let Some(session_arc) = state.chat_manager.get_session(&session_id).await {
        let mut session = session_arc.lock().await;
        match session.add_reaction(&message_id, request.emoji, request.user_id) {
            Ok(_) => {
                // Save session after adding reaction
                drop(session);
                let _ = state.chat_manager.save_session(&session_id).await;
                Ok(StatusCode::OK)
            }
            Err(_) => Err(StatusCode::NOT_FOUND),
        }
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn get_stats(State(state): State<AppState>) -> Result<Json<crate::SessionStats>, StatusCode> {
    let stats = state.chat_manager.get_session_stats().await;
    Ok(Json(stats))
}

async fn handle_websocket(socket: WebSocket, session_id: String, state: AppState) {
    let ws_session_id = Uuid::new_v4().to_string();
    let ws_info = WebSocketSessionInfo {
        session_id: session_id.clone(),
        user_id: None,
        connected_at: SystemTime::now(),
        last_activity: SystemTime::now(),
    };

    {
        let mut sessions = state.websocket_sessions.write().await;
        sessions.insert(ws_session_id.clone(), ws_info);
    }

    let (mut sender, mut receiver) = socket.split();
    let mut broadcast_rx = state.broadcast_tx.subscribe();

    // Create a channel for sending messages to the websocket
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);
    let tx_clone = tx.clone();

    let session_id_clone = session_id.clone();
    let send_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                // Handle broadcast messages
                Ok(msg) = broadcast_rx.recv() => {
                    if msg.session_id == session_id_clone {
                        let response = WebSocketResponse::Status {
                            status: "broadcast".to_string(),
                            data: msg.data,
                        };

                        if let Ok(json) = serde_json::to_string(&response) {
                            if sender
                                .send(axum::extract::ws::Message::Text(json))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
                // Handle direct messages
                Some(msg) = rx.recv() => {
                    if sender
                        .send(axum::extract::ws::Message::Text(msg))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });

    let recv_task = tokio::spawn(async move {
        let tx = tx_clone;
        while let Some(msg) = receiver.next().await {
            if let Ok(msg) = msg {
                match msg {
                    axum::extract::ws::Message::Text(text) => {
                        if let Ok(ws_msg) = serde_json::from_str::<WebSocketMessage>(&text) {
                            match ws_msg {
                                WebSocketMessage::SendMessage {
                                    content,
                                    thread_id,
                                    parent_message_id,
                                } => {
                                    if let Some(session_arc) =
                                        state.chat_manager.get_session(&session_id).await
                                    {
                                        let mut session = session_arc.lock().await;
                                        if let Ok(response_msg) = session
                                            .process_message_with_options(
                                                content,
                                                thread_id,
                                                parent_message_id,
                                            )
                                            .await
                                        {
                                            let response = WebSocketResponse::Message {
                                                message_id: response_msg.id,
                                                content: response_msg.content,
                                                role: match response_msg.role {
                                                    MessageRole::User => "user".to_string(),
                                                    MessageRole::Assistant => {
                                                        "assistant".to_string()
                                                    }
                                                    MessageRole::System => "system".to_string(),
                                                },
                                                timestamp: response_msg.timestamp.to_rfc3339(),
                                                metadata: response_msg.metadata.map(|m| {
                                                    serde_json::to_value(m).unwrap_or_default()
                                                }),
                                            };

                                            // Save session after message
                                            drop(session);
                                            let _ =
                                                state.chat_manager.save_session(&session_id).await;

                                            if let Ok(json) = serde_json::to_string(&response) {
                                                let _ = tx.send(json).await;
                                            }
                                        }
                                    }
                                }
                                WebSocketMessage::Ping => {
                                    let pong = WebSocketResponse::Pong;
                                    if let Ok(json) = serde_json::to_string(&pong) {
                                        let _ = tx.send(json).await;
                                    }
                                }
                                WebSocketMessage::Subscribe { topics: _ } => {
                                    // TODO: Implement topic subscription
                                }
                                WebSocketMessage::AddReaction {
                                    message_id,
                                    emoji,
                                    user_id,
                                } => {
                                    if let Some(session_arc) =
                                        state.chat_manager.get_session(&session_id).await
                                    {
                                        let mut session = session_arc.lock().await;
                                        if session.add_reaction(&message_id, emoji, user_id).is_ok()
                                        {
                                            drop(session);
                                            let _ =
                                                state.chat_manager.save_session(&session_id).await;

                                            let response = WebSocketResponse::Status {
                                                status: "reaction_added".to_string(),
                                                data: serde_json::json!({
                                                    "message_id": message_id,
                                                    "success": true
                                                }),
                                            };

                                            if let Ok(json) = serde_json::to_string(&response) {
                                                let _ = tx.send(json).await;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    axum::extract::ws::Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    });

    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }

    {
        let mut sessions = state.websocket_sessions.write().await;
        sessions.remove(&ws_session_id);
    }
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "service": "oxirs-chat"
    }))
}

async fn metrics_handler() -> Result<String, StatusCode> {
    // Basic Prometheus metrics implementation
    let mut metrics = Vec::new();

    // Help and type declarations
    metrics.push("# HELP oxirs_chat_requests_total Total number of requests received".to_string());
    metrics.push("# TYPE oxirs_chat_requests_total counter".to_string());

    metrics.push("# HELP oxirs_chat_sessions_active Number of active chat sessions".to_string());
    metrics.push("# TYPE oxirs_chat_sessions_active gauge".to_string());

    metrics.push("# HELP oxirs_chat_messages_total Total number of messages processed".to_string());
    metrics.push("# TYPE oxirs_chat_messages_total counter".to_string());

    metrics.push("# HELP oxirs_chat_response_time_seconds Response time in seconds".to_string());
    metrics.push("# TYPE oxirs_chat_response_time_seconds histogram".to_string());

    metrics.push(
        "# HELP oxirs_chat_sparql_queries_total Total number of SPARQL queries generated"
            .to_string(),
    );
    metrics.push("# TYPE oxirs_chat_sparql_queries_total counter".to_string());

    metrics.push("# HELP oxirs_chat_llm_requests_total Total number of LLM requests".to_string());
    metrics.push("# TYPE oxirs_chat_llm_requests_total counter".to_string());

    metrics.push("# HELP oxirs_chat_errors_total Total number of errors".to_string());
    metrics.push("# TYPE oxirs_chat_errors_total counter".to_string());

    // Sample metric values (in production, these would be collected from actual metrics)
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    metrics.push(format!(
        "oxirs_chat_requests_total{{method=\"POST\",endpoint=\"/api/sessions\"}} 150 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_requests_total{{method=\"GET\",endpoint=\"/api/sessions\"}} 45 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_requests_total{{method=\"POST\",endpoint=\"/api/sessions/messages\"}} 320 {}",
        timestamp
    ));

    metrics.push(format!("oxirs_chat_sessions_active 12 {}", timestamp));
    metrics.push(format!("oxirs_chat_messages_total 1250 {}", timestamp));

    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"0.1\"}} 45 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"0.5\"}} 120 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"1.0\"}} 180 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"2.0\"}} 195 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"+Inf\"}} 200 {}",
        timestamp
    ));

    metrics.push(format!(
        "oxirs_chat_sparql_queries_total{{status=\"success\"}} 85 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_sparql_queries_total{{status=\"failed\"}} 5 {}",
        timestamp
    ));

    metrics.push(format!(
        "oxirs_chat_llm_requests_total{{provider=\"openai\",model=\"gpt-4\"}} 120 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_llm_requests_total{{provider=\"anthropic\",model=\"claude-3-opus\"}} 80 {}",
        timestamp
    ));

    metrics.push(format!(
        "oxirs_chat_errors_total{{type=\"validation\"}} 3 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_errors_total{{type=\"llm_timeout\"}} 2 {}",
        timestamp
    ));
    metrics.push(format!(
        "oxirs_chat_errors_total{{type=\"sparql_generation\"}} 1 {}",
        timestamp
    ));

    Ok(metrics.join("\n"))
}
