use crate::error::{AppError, AppResult};
use serde::{Deserialize, Serialize};

/// A single chat message exchanged between user and assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    /// Role of the message author: `"user"` or `"assistant"`.
    pub role: String,
    pub content: String,
    /// Unix timestamp in seconds.
    pub timestamp: i64,
}

/// Metadata for a chat session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub title: String,
    pub created_at: i64,
    pub message_count: usize,
}

/// Request payload for [`send_message`].
#[derive(Debug, Serialize, Deserialize)]
pub struct SendMessageRequest {
    pub session_id: String,
    pub content: String,
}

/// Send a message in a chat session and get a response.
///
/// In production this will route to the configured LLM provider via oxirs-chat.
/// In this initial implementation it returns an echo for UI validation.
#[tauri::command]
pub async fn send_message(request: SendMessageRequest) -> AppResult<ChatMessage> {
    if request.session_id.is_empty() {
        return Err(AppError::NotFound("session_id is empty".to_string()));
    }
    if request.content.is_empty() {
        return Err(AppError::Chat(
            "message content must not be empty".to_string(),
        ));
    }
    // TODO: wire to oxirs_chat::ChatEngine when the integration layer is ready.
    let response = ChatMessage {
        id: format!("msg_{}", generate_id()),
        role: "assistant".to_string(),
        content: format!("Echo: {}", request.content),
        timestamp: current_timestamp(),
    };
    Ok(response)
}

/// List all chat sessions.
///
/// Returns a static demo list; replace with session_store integration when available.
#[tauri::command]
pub fn list_sessions() -> AppResult<Vec<ChatSession>> {
    Ok(vec![ChatSession {
        id: "session_demo".to_string(),
        title: "Demo Session".to_string(),
        created_at: current_timestamp(),
        message_count: 0,
    }])
}

/// Create a new named chat session.
#[tauri::command]
pub fn create_session(title: String) -> AppResult<ChatSession> {
    if title.is_empty() {
        return Err(AppError::Chat(
            "session title must not be empty".to_string(),
        ));
    }
    Ok(ChatSession {
        id: format!("session_{}", generate_id()),
        title,
        created_at: current_timestamp(),
        message_count: 0,
    })
}

/// Get message history for a session.
///
/// Returns an empty list for now; the persistent store integration is a follow-up task.
#[tauri::command]
pub fn get_session_history(session_id: String) -> AppResult<Vec<ChatMessage>> {
    if session_id.is_empty() {
        return Err(AppError::NotFound("session_id is empty".to_string()));
    }
    // Persistent history retrieval is deferred to the session_store integration.
    Ok(vec![])
}

/// Returns the current time as a Unix timestamp in seconds.
fn current_timestamp() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
    .unwrap_or(i64::MAX)
}

/// Generates a simple unique identifier using high-resolution timer bits.
fn generate_id() -> u64 {
    fastrand::u64(..)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_message_echo() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let req = SendMessageRequest {
            session_id: "s1".to_string(),
            content: "hello".to_string(),
        };
        let result = rt.block_on(send_message(req)).unwrap();
        assert_eq!(result.role, "assistant");
        assert!(result.content.contains("hello"));
    }

    #[test]
    fn test_send_message_empty_session_id_fails() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let req = SendMessageRequest {
            session_id: "".to_string(),
            content: "hello".to_string(),
        };
        let result = rt.block_on(send_message(req));
        assert!(result.is_err());
    }

    #[test]
    fn test_send_message_empty_content_fails() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let req = SendMessageRequest {
            session_id: "s1".to_string(),
            content: "".to_string(),
        };
        let result = rt.block_on(send_message(req));
        assert!(result.is_err());
    }

    #[test]
    fn test_list_sessions_returns_demo() {
        let sessions = list_sessions().unwrap();
        assert!(!sessions.is_empty());
        assert_eq!(sessions[0].id, "session_demo");
    }

    #[test]
    fn test_create_session() {
        let s = create_session("My Session".to_string()).unwrap();
        assert_eq!(s.title, "My Session");
        assert!(!s.id.is_empty());
        assert!(s.id.starts_with("session_"));
    }

    #[test]
    fn test_create_session_empty_title_fails() {
        let r = create_session("".to_string());
        assert!(r.is_err());
    }

    #[test]
    fn test_get_session_history_empty() {
        let h = get_session_history("session_demo".to_string()).unwrap();
        assert!(h.is_empty());
    }

    #[test]
    fn test_get_session_history_empty_id_fails() {
        let r = get_session_history("".to_string());
        assert!(r.is_err());
    }

    #[test]
    fn test_chat_message_serialization() {
        let m = ChatMessage {
            id: "m1".to_string(),
            role: "user".to_string(),
            content: "test".to_string(),
            timestamp: 1_234_567_890,
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "m1");
        assert_eq!(back.role, "user");
        assert_eq!(back.timestamp, 1_234_567_890);
    }

    #[test]
    fn test_chat_session_serialization() {
        let s = ChatSession {
            id: "s1".to_string(),
            title: "T".to_string(),
            created_at: 0,
            message_count: 5,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: ChatSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back.message_count, 5);
        assert_eq!(back.id, "s1");
    }

    #[test]
    fn test_timestamp_is_positive() {
        assert!(current_timestamp() > 0);
    }

    #[test]
    fn test_generate_id_produces_distinct_values() {
        use std::collections::HashSet;
        let ids: HashSet<u64> = (0..20).map(|_| generate_id()).collect();
        // 20 random u64 values must yield at least 2 distinct entries.
        assert!(
            ids.len() >= 2,
            "generate_id produced only identical values across 20 calls"
        );
    }

    #[test]
    fn test_send_message_response_has_id() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let req = SendMessageRequest {
            session_id: "s2".to_string(),
            content: "sparql".to_string(),
        };
        let result = rt.block_on(send_message(req)).unwrap();
        assert!(!result.id.is_empty());
        assert!(result.id.starts_with("msg_"));
    }

    #[test]
    fn test_send_message_timestamp_non_zero() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let req = SendMessageRequest {
            session_id: "s3".to_string(),
            content: "rdf".to_string(),
        };
        let result = rt.block_on(send_message(req)).unwrap();
        assert!(result.timestamp > 0);
    }
}
