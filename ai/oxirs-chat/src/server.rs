//! HTTP Server and WebSocket Implementation for OxiRS Chat

use axum::{
    extract::{ws::WebSocket, Path, Query, State, WebSocketUpgrade},
    http::{header, HeaderValue, Method, StatusCode},
    response::Response,
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
    time::{Duration, SystemTime},
};
use tokio::sync::{broadcast, RwLock};
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{MessageRole, OxiRSChat, ThreadInfo};

/// Server state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub chat: Arc<OxiRSChat>,
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
    pub subscribed_topics: std::collections::HashSet<String>,
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
    pub fn new(chat: Arc<OxiRSChat>, config: ServerConfig) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);

        let state = AppState {
            chat,
            websocket_sessions: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
            config,
        };

        Self { state }
    }

    pub async fn serve(self) -> Result<(), Box<dyn std::error::Error>> {
        // Configure CORS based on server configuration
        let cors = if self.state.config.cors_origins.contains(&"*".to_string()) {
            // Allow any origin if wildcard is specified
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
        } else {
            // Configure specific allowed origins
            let origins: Result<Vec<HeaderValue>, _> = self
                .state
                .config
                .cors_origins
                .iter()
                .map(|origin| origin.parse())
                .collect();

            match origins {
                Ok(origins) => {
                    info!(
                        "Configuring CORS with specific origins: {:?}",
                        self.state.config.cors_origins
                    );
                    CorsLayer::new()
                        .allow_origin(origins)
                        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
                }
                Err(e) => {
                    warn!(
                        "Invalid CORS origin configuration: {}, falling back to any origin",
                        e
                    );
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
                }
            }
        };

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

    match state.chat.create_session(session_id.clone()).await {
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
    match state.chat.get_session(&session_id).await {
        Some(session_arc) => {
            let session = session_arc.lock().await;
            Ok(Json(serde_json::json!({
                "session_id": session_id,
                "status": "active",
                "message_count": session.messages.len(),
                "created_at": session.created_at.to_rfc3339(),
                "last_activity": session.last_activity.to_rfc3339()
            })))
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

async fn send_message(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<SendMessageRequest>,
) -> Result<Json<MessageResponse>, StatusCode> {
    match state.chat.get_session(&session_id).await {
        Some(_session_arc) => {
            match state
                .chat
                .process_message(&session_id, request.content)
                .await
            {
                Ok(message) => {
                    // Session is automatically managed by OxiRSChat

                    let response = MessageResponse {
                        message_id: message.id,
                        content: message.content.to_string(),
                        role: match message.role {
                            MessageRole::User => "user".to_string(),
                            MessageRole::Assistant => "assistant".to_string(),
                            MessageRole::System => "system".to_string(),
                            MessageRole::Function => "function".to_string(),
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
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_messages(
    Path(session_id): Path<String>,
    Query(params): Query<SessionQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<MessageResponse>>, StatusCode> {
    match state.chat.get_session(&session_id).await {
        Some(session_arc) => {
            let session = session_arc.lock().await;
            let messages: Vec<MessageResponse> = session
                .messages
                .iter()
                .skip(params.offset.unwrap_or(0))
                .take(params.limit.unwrap_or(100))
                .map(|msg| MessageResponse {
                    message_id: msg.id.clone(),
                    content: msg.content.to_string(),
                    role: match msg.role {
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                        MessageRole::System => "system".to_string(),
                        MessageRole::Function => "function".to_string(),
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
        }
        _ => Err(StatusCode::NOT_FOUND),
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
    match state.chat.get_session(&session_id).await {
        Some(session_arc) => {
            let session = session_arc.lock().await;
            // Thread functionality not yet implemented
            let threads: Vec<ThreadInfo> = Vec::new();
            Ok(Json(threads))
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_thread_messages(
    Path((session_id, thread_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Vec<MessageResponse>>, StatusCode> {
    match state.chat.get_session(&session_id).await {
        Some(session_arc) => {
            let session = session_arc.lock().await;
            let messages: Vec<MessageResponse> = session
                .messages
                .iter()
                .filter(|msg| msg.thread_id.as_ref() == Some(&thread_id))
                .map(|msg| MessageResponse {
                    message_id: msg.id.clone(),
                    content: msg.content.to_string(),
                    role: match msg.role {
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                        MessageRole::System => "system".to_string(),
                        MessageRole::Function => "function".to_string(),
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
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_message_replies(
    Path((session_id, message_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Vec<MessageResponse>>, StatusCode> {
    match state.chat.get_session(&session_id).await {
        Some(session_arc) => {
            let session = session_arc.lock().await;
            let messages: Vec<MessageResponse> = session
                .messages
                .iter()
                .filter(|msg| msg.parent_message_id.as_ref() == Some(&message_id))
                .cloned()
                .map(|msg| MessageResponse {
                    message_id: msg.id.clone(),
                    content: msg.content.to_string(),
                    role: match msg.role {
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                        MessageRole::System => "system".to_string(),
                        MessageRole::Function => "function".to_string(),
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
        }
        _ => Err(StatusCode::NOT_FOUND),
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
    match state.chat.get_session(&session_id).await {
        Some(session_arc) => {
            let mut session = session_arc.lock().await;
            // Convert emoji string to ReactionType
            let reaction_type = match request.emoji.as_str() {
                "👍" | "like" => crate::ReactionType::Like,
                "👎" | "dislike" => crate::ReactionType::Dislike,
                "✅" | "helpful" => crate::ReactionType::Helpful,
                "❌" | "not_helpful" => crate::ReactionType::NotHelpful,
                "✔️" | "accurate" => crate::ReactionType::Accurate,
                "❌" | "inaccurate" => crate::ReactionType::Inaccurate,
                "💭" | "clear" => crate::ReactionType::Clear,
                "😵" | "confusing" => crate::ReactionType::Confusing,
                _ => crate::ReactionType::Like, // Default to Like for unknown emojis
            };

            // Find and update the message
            if let Some(message) = session.messages.iter_mut().find(|m| m.id == message_id) {
                message.reactions.push(crate::MessageReaction {
                    reaction_type,
                    user_id: Some(request.user_id),
                    timestamp: chrono::Utc::now(),
                });

                // Session is automatically managed
                Ok(StatusCode::OK)
            } else {
                Err(StatusCode::NOT_FOUND)
            }
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_stats(State(state): State<AppState>) -> Result<Json<crate::SessionStats>, StatusCode> {
    // Calculate basic stats from available methods
    let total_sessions = state.chat.session_count().await;
    let session_list = state.chat.list_sessions().await;

    // Create basic stats (detailed stats would require iterating over all sessions)
    let stats = crate::SessionStats {
        total_sessions,
        active_sessions: total_sessions, // Simplified - assume all are active
        idle_sessions: 0,
        expired_sessions: 0,
        suspended_sessions: 0,
        total_messages: 0, // Would need to aggregate from all sessions
    };

    Ok(Json(stats))
}

async fn handle_websocket(socket: WebSocket, session_id: String, state: AppState) {
    let ws_session_id = Uuid::new_v4().to_string();
    let ws_info = WebSocketSessionInfo {
        session_id: session_id.clone(),
        user_id: None,
        connected_at: SystemTime::now(),
        last_activity: SystemTime::now(),
        subscribed_topics: std::collections::HashSet::new(),
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
    let ws_session_id_clone = ws_session_id.clone();
    let ws_session_id_cleanup = ws_session_id.clone();
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

    let state_clone = state.clone();
    let recv_task = tokio::spawn(async move {
        let tx = tx_clone;
        let ws_session_id = ws_session_id.clone();
        let state = state_clone;
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
                                    // Process message using OxiRSChat
                                    match state.chat.process_message(&session_id, content).await {
                                        Ok(response_msg) => {
                                            let response = WebSocketResponse::Message {
                                                message_id: response_msg.id,
                                                content: response_msg.content.to_string(),
                                                role: match response_msg.role {
                                                    MessageRole::User => "user".to_string(),
                                                    MessageRole::Assistant => {
                                                        "assistant".to_string()
                                                    }
                                                    MessageRole::System => "system".to_string(),
                                                    MessageRole::Function => "function".to_string(),
                                                },
                                                timestamp: response_msg.timestamp.to_rfc3339(),
                                                metadata: response_msg.metadata.map(|m| {
                                                    serde_json::to_value(m).unwrap_or_default()
                                                }),
                                            };

                                            // Session is automatically managed by OxiRSChat

                                            if let Ok(json) = serde_json::to_string(&response) {
                                                let _ = tx.send(json).await;
                                            }
                                        }
                                        _ => {
                                            // Handle message processing error
                                            let error_response = WebSocketResponse::Error {
                                                code: "PROCESSING_ERROR".to_string(),
                                                message: "Failed to process message".to_string(),
                                            };
                                            if let Ok(json) = serde_json::to_string(&error_response)
                                            {
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
                                WebSocketMessage::Subscribe { topics } => {
                                    // Implement topic subscription
                                    let mut sessions = state.websocket_sessions.write().await;
                                    if let Some(ws_info) = sessions.get_mut(&ws_session_id) {
                                        // Update subscribed topics
                                        ws_info.subscribed_topics.clear();
                                        for topic in &topics {
                                            ws_info.subscribed_topics.insert(topic.clone());
                                        }
                                        ws_info.last_activity = SystemTime::now();

                                        // Send confirmation
                                        let response = WebSocketResponse::Status {
                                            status: "subscribed".to_string(),
                                            data: serde_json::json!({
                                                "topics": topics,
                                                "subscription_count": ws_info.subscribed_topics.len()
                                            }),
                                        };

                                        if let Ok(json) = serde_json::to_string(&response) {
                                            let _ = tx.send(json).await;
                                        }

                                        info!(
                                            "WebSocket session {} subscribed to topics: {:?}",
                                            ws_session_id, topics
                                        );
                                    } else {
                                        // Session not found
                                        let error_response = WebSocketResponse::Error {
                                            code: "SESSION_NOT_FOUND".to_string(),
                                            message: "WebSocket session not found".to_string(),
                                        };
                                        if let Ok(json) = serde_json::to_string(&error_response) {
                                            let _ = tx.send(json).await;
                                        }
                                    }
                                }
                                WebSocketMessage::AddReaction {
                                    message_id,
                                    emoji,
                                    user_id,
                                } => {
                                    if let Some(session_arc) =
                                        state.chat.get_session(&session_id).await
                                    {
                                        let mut session = session_arc.lock().await;
                                        // Convert emoji to reaction type
                                        let reaction_type = match emoji.as_str() {
                                            "👍" | "like" => crate::ReactionType::Like,
                                            "👎" | "dislike" => crate::ReactionType::Dislike,
                                            "✅" | "helpful" => crate::ReactionType::Helpful,
                                            "❌" | "not_helpful" => {
                                                crate::ReactionType::NotHelpful
                                            }
                                            "✔️" | "accurate" => crate::ReactionType::Accurate,
                                            "❌" | "inaccurate" => crate::ReactionType::Inaccurate,
                                            "💭" | "clear" => crate::ReactionType::Clear,
                                            "😵" | "confusing" => crate::ReactionType::Confusing,
                                            _ => crate::ReactionType::Like,
                                        };

                                        if let Some(message) =
                                            session.messages.iter_mut().find(|m| m.id == message_id)
                                        {
                                            message.reactions.push(crate::MessageReaction {
                                                reaction_type,
                                                user_id: Some(user_id),
                                                timestamp: chrono::Utc::now(),
                                            });

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
                                        } else {
                                            // Message not found
                                            let error_response = WebSocketResponse::Error {
                                                code: "NOT_FOUND".to_string(),
                                                message: "Message not found".to_string(),
                                            };
                                            if let Ok(json) = serde_json::to_string(&error_response)
                                            {
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
        sessions.remove(&ws_session_id_cleanup);
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
        "oxirs_chat_requests_total{{method=\"POST\",endpoint=\"/api/sessions\"}} 150 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_requests_total{{method=\"GET\",endpoint=\"/api/sessions\"}} 45 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_requests_total{{method=\"POST\",endpoint=\"/api/sessions/messages\"}} 320 {timestamp}"
    ));

    metrics.push(format!("oxirs_chat_sessions_active 12 {timestamp}"));
    metrics.push(format!("oxirs_chat_messages_total 1250 {timestamp}"));

    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"0.1\"}} 45 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"0.5\"}} 120 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"1.0\"}} 180 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"2.0\"}} 195 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_response_time_seconds_bucket{{le=\"+Inf\"}} 200 {timestamp}"
    ));

    metrics.push(format!(
        "oxirs_chat_sparql_queries_total{{status=\"success\"}} 85 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_sparql_queries_total{{status=\"failed\"}} 5 {timestamp}"
    ));

    metrics.push(format!(
        "oxirs_chat_llm_requests_total{{provider=\"openai\",model=\"gpt-4\"}} 120 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_llm_requests_total{{provider=\"anthropic\",model=\"claude-3-opus\"}} 80 {timestamp}"
    ));

    metrics.push(format!(
        "oxirs_chat_errors_total{{type=\"validation\"}} 3 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_errors_total{{type=\"llm_timeout\"}} 2 {timestamp}"
    ));
    metrics.push(format!(
        "oxirs_chat_errors_total{{type=\"sparql_generation\"}} 1 {timestamp}"
    ));

    Ok(metrics.join("\n"))
}

/// Broadcast a message to all WebSocket clients subscribed to a specific topic
async fn broadcast_to_topic(
    state: &AppState,
    topic: &str,
    message: serde_json::Value,
) -> Result<usize, Box<dyn std::error::Error>> {
    let sessions = state.websocket_sessions.read().await;
    let mut sent_count = 0;

    // Create the broadcast message
    let server_message = ServerMessage {
        message_type: ServerMessageType::SystemStatus,
        session_id: "system".to_string(),
        data: serde_json::json!({
            "topic": topic,
            "message": message
        }),
    };

    // Find all sessions subscribed to this topic
    for (ws_session_id, ws_info) in sessions.iter() {
        if ws_info.subscribed_topics.contains(topic) {
            // Send the message via broadcast channel
            if state.broadcast_tx.send(server_message.clone()).is_ok() {
                sent_count += 1;
                debug!(
                    "Sent topic '{}' message to WebSocket session: {}",
                    topic, ws_session_id
                );
            }
        }
    }

    if sent_count > 0 {
        info!(
            "Broadcasted topic '{}' message to {} subscribers",
            topic, sent_count
        );
    }

    Ok(sent_count)
}

/// Get list of all active topics with subscriber counts
async fn get_active_topics(state: &AppState) -> HashMap<String, usize> {
    let sessions = state.websocket_sessions.read().await;
    let mut topic_counts: HashMap<String, usize> = HashMap::new();

    for ws_info in sessions.values() {
        for topic in &ws_info.subscribed_topics {
            *topic_counts.entry(topic.clone()).or_insert(0) += 1;
        }
    }

    topic_counts
}
