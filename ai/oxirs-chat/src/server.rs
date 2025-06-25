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

use crate::{ChatManager, ChatSession, Message, MessageRole};

/// Server state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub chat_manager: Arc<RwLock<ChatManager>>,
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
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message_id: String,
    pub content: String,
    pub role: String,
    pub timestamp: String,
    pub metadata: Option<serde_json::Value>,
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
    SendMessage { content: String },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "subscribe")]
    Subscribe { topics: Vec<String> },
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
    Status { status: String, data: serde_json::Value },
}

/// Main server implementation
pub struct ChatServer {
    state: AppState,
}

impl ChatServer {
    pub fn new(chat_manager: ChatManager, config: ServerConfig) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);
        
        let state = AppState {
            chat_manager: Arc::new(RwLock::new(chat_manager)),
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
            .route("/api/sessions/:session_id/ws", get(websocket_handler))
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
    
    {
        let mut chat_manager = state.chat_manager.write().await;
        chat_manager.create_session(session_id.clone());
    }

    let websocket_url = format!("ws://{}:{}/api/sessions/{}/ws", 
        state.config.host, state.config.port, session_id);

    Ok(Json(CreateSessionResponse {
        session_id,
        websocket_url,
    }))
}

async fn get_session(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let chat_manager = state.chat_manager.read().await;
    
    if let Some(_session) = chat_manager.sessions.get(&session_id) {
        Ok(Json(serde_json::json!({
            "session_id": session_id,
            "status": "active"
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
    let mut chat_manager = state.chat_manager.write().await;
    
    if let Some(session) = chat_manager.get_session(&session_id) {
        match session.process_message(request.content).await {
            Ok(message) => {
                let response = MessageResponse {
                    message_id: Uuid::new_v4().to_string(),
                    content: message.content,
                    role: match message.role {
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                        MessageRole::System => "system".to_string(),
                    },
                    timestamp: message.timestamp.to_rfc3339(),
                    metadata: None,
                };
                
                Ok(Json(response))
            }
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
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
    let chat_manager = state.chat_manager.read().await;
    
    if let Some(session) = chat_manager.sessions.get(&session_id) {
        let messages: Vec<MessageResponse> = session
            .get_history()
            .iter()
            .skip(params.offset.unwrap_or(0))
            .take(params.limit.unwrap_or(100))
            .map(|msg| MessageResponse {
                message_id: Uuid::new_v4().to_string(),
                content: msg.content.clone(),
                role: match msg.role {
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                    MessageRole::System => "system".to_string(),
                },
                timestamp: msg.timestamp.to_rfc3339(),
                metadata: None,
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

    let send_task = tokio::spawn(async move {
        while let Ok(msg) = broadcast_rx.recv().await {
            if msg.session_id == session_id {
                let response = WebSocketResponse::Status {
                    status: "broadcast".to_string(),
                    data: msg.data,
                };
                
                if let Ok(json) = serde_json::to_string(&response) {
                    if sender.send(axum::extract::ws::Message::Text(json)).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            if let Ok(msg) = msg {
                match msg {
                    axum::extract::ws::Message::Text(text) => {
                        if let Ok(ws_msg) = serde_json::from_str::<WebSocketMessage>(&text) {
                            match ws_msg {
                                WebSocketMessage::SendMessage { content } => {
                                    let mut chat_manager = state.chat_manager.write().await;
                                    if let Some(chat_session) = chat_manager.get_session(&session_id) {
                                        if let Ok(response_msg) = chat_session.process_message(content).await {
                                            let response = WebSocketResponse::Message {
                                                message_id: Uuid::new_v4().to_string(),
                                                content: response_msg.content,
                                                role: match response_msg.role {
                                                    MessageRole::User => "user".to_string(),
                                                    MessageRole::Assistant => "assistant".to_string(),
                                                    MessageRole::System => "system".to_string(),
                                                },
                                                timestamp: response_msg.timestamp.to_rfc3339(),
                                                metadata: None,
                                            };
                                            
                                            if let Ok(json) = serde_json::to_string(&response) {
                                                let _ = sender.send(axum::extract::ws::Message::Text(json)).await;
                                            }
                                        }
                                    }
                                }
                                WebSocketMessage::Ping => {
                                    let pong = WebSocketResponse::Pong;
                                    if let Ok(json) = serde_json::to_string(&pong) {
                                        let _ = sender.send(axum::extract::ws::Message::Text(json)).await;
                                    }
                                }
                                WebSocketMessage::Subscribe { topics: _ } => {
                                    // TODO: Implement topic subscription
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
    // TODO: Implement Prometheus metrics
    Ok("# HELP oxirs_chat_requests_total Total number of requests\n".to_string())
}
