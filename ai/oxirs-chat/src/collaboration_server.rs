//! Collaboration Server Endpoints
//!
//! HTTP and WebSocket endpoints for real-time collaboration features

use axum::{
    extract::{Path, State, WebSocketUpgrade},
    http::StatusCode,
    response::Response,
    Json,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};

use crate::collaboration::{
    AccessControl, CollaborationManager, CursorPosition, Participant, ParticipantStatus,
};

/// Shared state for collaboration endpoints
#[derive(Clone)]
pub struct CollaborationState {
    pub manager: Arc<CollaborationManager>,
}

/// Request to create a shared session
#[derive(Debug, Deserialize)]
pub struct CreateSharedSessionRequest {
    pub owner_id: String,
    pub access_control: Option<AccessControl>,
}

/// Response with shared session details
#[derive(Debug, Serialize)]
pub struct SharedSessionResponse {
    pub session_id: String,
    pub owner_id: String,
    pub websocket_url: String,
}

/// Request to join a shared session
#[derive(Debug, Deserialize)]
pub struct JoinSessionRequest {
    pub user_id: String,
    pub display_name: Option<String>,
}

/// Request to update cursor position
#[derive(Debug, Deserialize)]
pub struct UpdateCursorRequest {
    pub user_id: String,
    pub position: CursorPosition,
}

/// Request to update participant status
#[derive(Debug, Deserialize)]
pub struct UpdateStatusRequest {
    pub user_id: String,
    pub status: ParticipantStatus,
}

/// Participants list response
#[derive(Debug, Serialize)]
pub struct ParticipantsResponse {
    pub session_id: String,
    pub participants: Vec<Participant>,
}

/// Session list response
#[derive(Debug, Serialize)]
pub struct SessionListResponse {
    pub sessions: Vec<String>,
    pub total: usize,
}

/// Create a new shared session
pub async fn create_shared_session(
    State(state): State<CollaborationState>,
    Json(request): Json<CreateSharedSessionRequest>,
) -> Result<Json<SharedSessionResponse>, StatusCode> {
    match state
        .manager
        .create_shared_session(request.owner_id.clone(), request.access_control)
        .await
    {
        Ok(session_id) => {
            let websocket_url = format!("/api/collaboration/sessions/{}/ws", session_id);

            Ok(Json(SharedSessionResponse {
                session_id,
                owner_id: request.owner_id,
                websocket_url,
            }))
        }
        Err(e) => {
            error!("Failed to create shared session: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Join an existing shared session
pub async fn join_session(
    State(state): State<CollaborationState>,
    Path(session_id): Path<String>,
    Json(request): Json<JoinSessionRequest>,
) -> Result<StatusCode, StatusCode> {
    match state
        .manager
        .join_session(&session_id, request.user_id, request.display_name)
        .await
    {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => {
            error!("Failed to join session {}: {}", session_id, e);
            if e.to_string().contains("Access denied") {
                Err(StatusCode::FORBIDDEN)
            } else if e.to_string().contains("not found") {
                Err(StatusCode::NOT_FOUND)
            } else if e.to_string().contains("maximum participants") {
                Err(StatusCode::TOO_MANY_REQUESTS)
            } else {
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}

/// Leave a shared session
pub async fn leave_session(
    State(state): State<CollaborationState>,
    Path((session_id, user_id)): Path<(String, String)>,
) -> Result<StatusCode, StatusCode> {
    match state.manager.leave_session(&session_id, &user_id).await {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => {
            error!("Failed to leave session {}: {}", session_id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Update cursor position
pub async fn update_cursor(
    State(state): State<CollaborationState>,
    Path(session_id): Path<String>,
    Json(request): Json<UpdateCursorRequest>,
) -> Result<StatusCode, StatusCode> {
    match state
        .manager
        .update_cursor(&session_id, &request.user_id, request.position)
        .await
    {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => {
            error!("Failed to update cursor in session {}: {}", session_id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Update participant status
pub async fn update_status(
    State(state): State<CollaborationState>,
    Path(session_id): Path<String>,
    Json(request): Json<UpdateStatusRequest>,
) -> Result<StatusCode, StatusCode> {
    match state
        .manager
        .update_status(&session_id, &request.user_id, request.status)
        .await
    {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => {
            error!("Failed to update status in session {}: {}", session_id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get participants in a session
pub async fn get_participants(
    State(state): State<CollaborationState>,
    Path(session_id): Path<String>,
) -> Result<Json<ParticipantsResponse>, StatusCode> {
    match state.manager.get_participants(&session_id).await {
        Some(participants) => Ok(Json(ParticipantsResponse {
            session_id,
            participants,
        })),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// List all active shared sessions
pub async fn list_sessions(
    State(state): State<CollaborationState>,
) -> Result<Json<SessionListResponse>, StatusCode> {
    let sessions = state.manager.list_sessions().await;
    let total = sessions.len();

    Ok(Json(SessionListResponse { sessions, total }))
}

/// Get collaboration statistics
pub async fn get_stats(
    State(state): State<CollaborationState>,
) -> Result<Json<crate::collaboration::CollaborationStats>, StatusCode> {
    let stats = state.manager.get_stats().await;
    Ok(Json(stats))
}

/// WebSocket handler for real-time collaboration updates
pub async fn collaboration_websocket(
    State(state): State<CollaborationState>,
    Path(session_id): Path<String>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| handle_collaboration_websocket(socket, state, session_id))
}

async fn handle_collaboration_websocket(
    socket: axum::extract::ws::WebSocket,
    state: CollaborationState,
    session_id: String,
) {
    let (sender, mut receiver) = socket.split();
    let sender = Arc::new(tokio::sync::Mutex::new(sender));

    // Subscribe to collaboration updates
    let mut update_rx = state.manager.subscribe();

    // Spawn a task to forward updates to this WebSocket connection
    let session_id_clone = session_id.clone();
    let sender_clone = sender.clone();
    let forward_task = tokio::spawn(async move {
        while let Ok(update) = update_rx.recv().await {
            // Only forward updates for this session
            let should_forward = match &update {
                crate::collaboration::CollaborationUpdate::UserJoined {
                    session_id: sid, ..
                }
                | crate::collaboration::CollaborationUpdate::UserLeft {
                    session_id: sid, ..
                }
                | crate::collaboration::CollaborationUpdate::CursorMoved {
                    session_id: sid, ..
                }
                | crate::collaboration::CollaborationUpdate::StatusChanged {
                    session_id: sid,
                    ..
                }
                | crate::collaboration::CollaborationUpdate::MetadataUpdated {
                    session_id: sid,
                    ..
                }
                | crate::collaboration::CollaborationUpdate::QueryUpdate {
                    session_id: sid, ..
                } => sid == &session_id_clone,
            };

            if should_forward {
                if let Ok(json) = serde_json::to_string(&update) {
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
        }
    });

    // Handle incoming messages (ping/pong, etc.)
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            axum::extract::ws::Message::Text(text) => {
                debug!("Received collaboration message: {}", text);
                // Could handle additional commands here
            }
            axum::extract::ws::Message::Ping(data) => {
                let mut sender = sender.lock().await;
                if sender
                    .send(axum::extract::ws::Message::Pong(data))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            axum::extract::ws::Message::Close(_) => {
                info!("WebSocket closed for session {}", session_id);
                break;
            }
            _ => {}
        }
    }

    forward_task.abort();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collaboration::CollaborationConfig;

    #[tokio::test]
    async fn test_create_shared_session_endpoint() {
        let config = CollaborationConfig::default();
        let manager = Arc::new(CollaborationManager::new(config));
        let state = CollaborationState { manager };

        let request = CreateSharedSessionRequest {
            owner_id: "user1".to_string(),
            access_control: None,
        };

        let result = create_shared_session(State(state), Json(request)).await;
        assert!(result.is_ok());
    }
}
