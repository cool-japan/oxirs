use crate::error::{AppError, AppResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// A single chat message exchanged between user and assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    /// Role of the message author: `"user"`, `"assistant"`, `"system"`, or `"function"`.
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

/// Runtime state backing the chat commands: a real `oxirs-chat` engine plus
/// the on-disk directory its sessions are persisted to/from.
struct ChatEngineState {
    engine: oxirs_chat::OxiRSChat,
    session_dir: PathBuf,
}

/// Lazily-initialized, process-wide chat engine. `oxirs_chat::OxiRSChat` owns
/// its own internal synchronization (sessions map behind an `RwLock`), so a
/// shared `Arc` without an additional outer lock is sufficient here.
static CHAT_STATE: tokio::sync::OnceCell<Arc<ChatEngineState>> = tokio::sync::OnceCell::const_new();

/// Directory the chat session store persists to. Overridable via
/// `OXIRS_CHAT_SESSION_DIR` (used by tests to get an isolated, unique
/// location); otherwise defaults to a well-known subdirectory of the system
/// temp directory so sessions survive across app restarts within the same
/// machine boot without ever hardcoding an absolute path.
fn default_session_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OXIRS_CHAT_SESSION_DIR") {
        return PathBuf::from(dir);
    }
    std::env::temp_dir()
        .join("oxirs-tauri")
        .join("chat_sessions")
}

/// Get (or lazily create) the shared chat engine instance.
async fn engine_state() -> AppResult<Arc<ChatEngineState>> {
    CHAT_STATE
        .get_or_try_init(|| async {
            let store: Arc<dyn oxirs_core::Store> = Arc::new(
                oxirs_core::ConcreteStore::new()
                    .map_err(|e| AppError::Chat(format!("failed to initialize RDF store: {e}")))?,
            );
            let engine = oxirs_chat::OxiRSChat::new(oxirs_chat::ChatConfig::default(), store)
                .await
                .map_err(|e| AppError::Chat(format!("failed to initialize chat engine: {e}")))?;

            let session_dir = default_session_dir();
            if let Err(e) = engine.load_sessions(&session_dir).await {
                tracing::warn!(
                    "failed to load persisted chat sessions from {:?}: {}",
                    session_dir,
                    e
                );
            }

            Ok::<Arc<ChatEngineState>, AppError>(Arc::new(ChatEngineState {
                engine,
                session_dir,
            }))
        })
        .await
        .cloned()
}

/// Persist all sessions to disk, logging (rather than failing the calling
/// command on) any I/O error — persistence is best-effort so a transient
/// filesystem issue never blocks the user from continuing to chat.
async fn persist(state: &ChatEngineState) {
    if let Err(e) = state.engine.save_sessions(&state.session_dir).await {
        tracing::warn!("failed to persist chat sessions: {}", e);
    }
}

fn role_str(role: &oxirs_chat::MessageRole) -> &'static str {
    match role {
        oxirs_chat::MessageRole::User => "user",
        oxirs_chat::MessageRole::Assistant => "assistant",
        oxirs_chat::MessageRole::System => "system",
        oxirs_chat::MessageRole::Function => "function",
    }
}

fn to_chat_message(m: &oxirs_chat::Message) -> ChatMessage {
    ChatMessage {
        id: m.id.clone(),
        role: role_str(&m.role).to_string(),
        content: m.content.to_text().to_string(),
        timestamp: m.timestamp.timestamp(),
    }
}

/// True if the anyhow error chain indicates no LLM provider is configured
/// (i.e. no API key was supplied via environment variables), as opposed to
/// some other processing failure.
fn is_no_provider_error(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.to_string().to_lowercase().contains("provider"))
}

/// Send a message in a chat session and get a response.
///
/// Routes to the real `oxirs-chat` `OxiRSChat::process_message` pipeline
/// (RAG retrieval + LLM generation). If no LLM provider is configured (no
/// `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / etc. set), this returns a clear
/// "LLM backend not configured" error rather than a fabricated response.
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

    let state = engine_state().await?;

    if state
        .engine
        .get_session(&request.session_id)
        .await
        .is_none()
    {
        return Err(AppError::NotFound(format!(
            "chat session not found: {}",
            request.session_id
        )));
    }

    match state
        .engine
        .process_message(&request.session_id, request.content.clone())
        .await
    {
        Ok(message) => {
            persist(&state).await;
            Ok(to_chat_message(&message))
        }
        Err(e) if is_no_provider_error(&e) => Err(AppError::Chat(
            "LLM backend not configured: set an API key (e.g. OPENAI_API_KEY or \
             ANTHROPIC_API_KEY) for oxirs-chat before sending messages."
                .to_string(),
        )),
        Err(e) => Err(AppError::Chat(format!("chat processing failed: {e}"))),
    }
}

/// List all chat sessions, sorted by creation time.
#[tauri::command]
pub async fn list_sessions() -> AppResult<Vec<ChatSession>> {
    let state = engine_state().await?;
    let ids = state.engine.list_sessions().await;

    let mut sessions = Vec::with_capacity(ids.len());
    for id in ids {
        if let Some(session_arc) = state.engine.get_session(&id).await {
            let session = session_arc.lock().await;
            let title = session
                .metadata
                .get("title")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| id.clone());
            sessions.push(ChatSession {
                id: id.clone(),
                title,
                created_at: session.created_at.timestamp(),
                message_count: session.messages.len(),
            });
        }
    }
    sessions.sort_by_key(|s| s.created_at);
    Ok(sessions)
}

/// Create a new named chat session, persisted immediately.
#[tauri::command]
pub async fn create_session(title: String) -> AppResult<ChatSession> {
    if title.is_empty() {
        return Err(AppError::Chat(
            "session title must not be empty".to_string(),
        ));
    }

    let state = engine_state().await?;
    let id = format!("session_{}", generate_id());

    let session_arc = state
        .engine
        .create_session(id.clone())
        .await
        .map_err(|e| AppError::Chat(format!("failed to create session: {e}")))?;

    let created_at = {
        let mut session = session_arc.lock().await;
        session.metadata.insert(
            "title".to_string(),
            serde_json::Value::String(title.clone()),
        );
        session.created_at.timestamp()
    };

    persist(&state).await;

    Ok(ChatSession {
        id,
        title,
        created_at,
        message_count: 0,
    })
}

/// Get message history for a session from the persistent session store.
#[tauri::command]
pub async fn get_session_history(session_id: String) -> AppResult<Vec<ChatMessage>> {
    if session_id.is_empty() {
        return Err(AppError::NotFound("session_id is empty".to_string()));
    }

    let state = engine_state().await?;
    let session_arc = state
        .engine
        .get_session(&session_id)
        .await
        .ok_or_else(|| AppError::NotFound(format!("chat session not found: {session_id}")))?;

    let session = session_arc.lock().await;
    Ok(session.messages.iter().map(to_chat_message).collect())
}

/// Generates a simple unique identifier using high-resolution timer bits.
fn generate_id() -> u64 {
    fastrand::u64(..)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Point the chat session store at a fresh, unique temp directory before
    /// the process-wide engine is lazily initialized by the first test that
    /// touches it. `Once` makes this race-free across parallel test threads.
    fn init_test_env() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let dir = std::env::temp_dir().join(format!(
                "oxirs-tauri-chat-test-{}-{}",
                std::process::id(),
                fastrand::u64(..)
            ));
            // SAFETY: called exactly once, before any other thread in this
            // test binary has had a chance to read the chat engine's
            // configuration (guarded by `Once`, ahead of any `.await` that
            // could initialize `CHAT_STATE`).
            unsafe {
                std::env::set_var("OXIRS_CHAT_SESSION_DIR", &dir);
            }
        });
    }

    #[test]
    fn test_send_message_empty_session_id_fails() {
        init_test_env();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let req = SendMessageRequest {
            session_id: "".to_string(),
            content: "hello".to_string(),
        };
        let result = rt.block_on(send_message(req));
        assert!(result.is_err());
    }

    #[test]
    fn test_send_message_empty_content_fails() {
        init_test_env();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let req = SendMessageRequest {
            session_id: "s1".to_string(),
            content: "".to_string(),
        };
        let result = rt.block_on(send_message(req));
        assert!(result.is_err());
    }

    #[test]
    fn test_send_message_unknown_session_is_not_found() {
        init_test_env();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let req = SendMessageRequest {
            session_id: "session_does_not_exist".to_string(),
            content: "hello".to_string(),
        };
        let result = rt.block_on(send_message(req));
        match result {
            Err(AppError::NotFound(_)) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_create_session_then_visible_in_list_and_history() {
        init_test_env();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let session = rt
            .block_on(create_session("My Session".to_string()))
            .expect("create_session should succeed");
        assert_eq!(session.title, "My Session");
        assert!(session.id.starts_with("session_"));
        assert_eq!(session.message_count, 0);

        let sessions = rt.block_on(list_sessions()).expect("list_sessions");
        assert!(sessions.iter().any(|s| s.id == session.id));

        let history = rt
            .block_on(get_session_history(session.id.clone()))
            .expect("get_session_history");
        assert!(history.is_empty());
    }

    #[test]
    fn test_create_session_empty_title_fails() {
        init_test_env();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let r = rt.block_on(create_session("".to_string()));
        assert!(r.is_err());
    }

    #[test]
    fn test_get_session_history_empty_id_fails() {
        init_test_env();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let r = rt.block_on(get_session_history("".to_string()));
        assert!(r.is_err());
    }

    #[test]
    fn test_get_session_history_unknown_session_is_not_found() {
        init_test_env();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let r = rt.block_on(get_session_history("session_does_not_exist".to_string()));
        assert!(r.is_err());
    }

    #[test]
    fn test_send_message_without_llm_provider_configured_fails_clearly() {
        init_test_env();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let session = rt
            .block_on(create_session("Provider Test".to_string()))
            .expect("create_session should succeed");
        let req = SendMessageRequest {
            session_id: session.id,
            content: "hello".to_string(),
        };
        // In this test environment no OPENAI_API_KEY / ANTHROPIC_API_KEY is
        // configured (and no network access is available either way), so the
        // real chat engine must fail rather than fabricate a response.
        let result = rt.block_on(send_message(req));
        assert!(
            result.is_err(),
            "send_message must not fabricate a response when no LLM backend is reachable"
        );
    }

    #[test]
    fn test_chat_message_serialization() {
        let m = ChatMessage {
            id: "m1".to_string(),
            role: "user".to_string(),
            content: "test".to_string(),
            timestamp: 1_234_567_890,
        };
        let json = serde_json::to_string(&m).expect("serialize");
        let back: ChatMessage = serde_json::from_str(&json).expect("deserialize");
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
        let json = serde_json::to_string(&s).expect("serialize");
        let back: ChatSession = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.message_count, 5);
        assert_eq!(back.id, "s1");
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
    fn test_role_str_covers_all_roles() {
        assert_eq!(role_str(&oxirs_chat::MessageRole::User), "user");
        assert_eq!(role_str(&oxirs_chat::MessageRole::Assistant), "assistant");
        assert_eq!(role_str(&oxirs_chat::MessageRole::System), "system");
        assert_eq!(role_str(&oxirs_chat::MessageRole::Function), "function");
    }
}
