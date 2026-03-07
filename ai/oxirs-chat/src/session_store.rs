//! Chat session persistence and retrieval.
//!
//! Provides an in-memory store for managing chat sessions with
//! capacity-bounded eviction of the oldest session.

use std::collections::HashMap;

/// The role of a message participant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    /// A message from the user.
    User,
    /// A message from the assistant.
    Assistant,
    /// A system-level instruction message.
    System,
}

/// A single chat message.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    /// Role of the sender.
    pub role: MessageRole,
    /// Text content of the message.
    pub content: String,
    /// Wall-clock timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

/// A chat session containing ordered messages.
#[derive(Debug, Clone)]
pub struct ChatSession {
    /// Unique session identifier.
    pub session_id: String,
    /// Ordered list of messages in this session.
    pub messages: Vec<ChatMessage>,
    /// Timestamp (ms) when the session was created.
    pub created_ms: u64,
    /// Timestamp (ms) when the session was last updated.
    pub last_updated_ms: u64,
    /// Arbitrary key-value metadata for this session.
    pub metadata: HashMap<String, String>,
}

impl ChatSession {
    /// Create a new empty session.
    pub fn new(session_id: &str, now_ms: u64) -> Self {
        Self {
            session_id: session_id.to_string(),
            messages: Vec::new(),
            created_ms: now_ms,
            last_updated_ms: now_ms,
            metadata: HashMap::new(),
        }
    }

    /// Append a message to this session, updating `last_updated_ms`.
    pub fn add_message(&mut self, role: MessageRole, content: &str, now_ms: u64) {
        self.messages.push(ChatMessage {
            role,
            content: content.to_string(),
            timestamp_ms: now_ms,
        });
        self.last_updated_ms = now_ms;
    }

    /// Return the total number of messages in this session.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Return the most recent message, or `None` if the session is empty.
    pub fn last_message(&self) -> Option<&ChatMessage> {
        self.messages.last()
    }

    /// Return all messages with the given role.
    pub fn messages_by_role(&self, role: &MessageRole) -> Vec<&ChatMessage> {
        self.messages.iter().filter(|m| &m.role == role).collect()
    }
}

/// In-memory store for chat sessions with capacity-bounded eviction.
///
/// When the store is at capacity and a new session is created, the session
/// with the smallest `created_ms` (oldest) is evicted.
pub struct SessionStore {
    sessions: HashMap<String, ChatSession>,
    max_sessions: usize,
}

impl SessionStore {
    /// Create a new store with the given maximum number of sessions.
    ///
    /// `max_sessions` must be at least 1; passing 0 is treated as 1.
    pub fn new(max_sessions: usize) -> Self {
        let cap = max_sessions.max(1);
        Self {
            sessions: HashMap::new(),
            max_sessions: cap,
        }
    }

    /// Create a new session and insert it into the store.
    ///
    /// If the store is at capacity, the oldest session is evicted first.
    /// Returns a reference to the newly created session.
    pub fn create(&mut self, session_id: &str, now_ms: u64) -> &ChatSession {
        if self.sessions.len() >= self.max_sessions {
            if let Some(oldest_id) = self.oldest_session_id().map(str::to_string) {
                self.sessions.remove(&oldest_id);
            }
        }
        let session = ChatSession::new(session_id, now_ms);
        self.sessions.insert(session_id.to_string(), session);
        // Safety: we just inserted the key.
        self.sessions
            .get(session_id)
            .expect("session just inserted")
    }

    /// Return an immutable reference to a session by ID.
    pub fn get(&self, session_id: &str) -> Option<&ChatSession> {
        self.sessions.get(session_id)
    }

    /// Return a mutable reference to a session by ID.
    pub fn get_mut(&mut self, session_id: &str) -> Option<&mut ChatSession> {
        self.sessions.get_mut(session_id)
    }

    /// Remove a session from the store. Returns `true` if it existed.
    pub fn delete(&mut self, session_id: &str) -> bool {
        self.sessions.remove(session_id).is_some()
    }

    /// Return all session IDs currently in the store.
    pub fn list_ids(&self) -> Vec<&str> {
        self.sessions.keys().map(String::as_str).collect()
    }

    /// Return the number of sessions currently in the store.
    pub fn count(&self) -> usize {
        self.sessions.len()
    }

    /// Return the ID of the oldest session (smallest `created_ms`), or `None` if empty.
    pub fn oldest_session_id(&self) -> Option<&str> {
        self.sessions
            .iter()
            .min_by_key(|(_, s)| s.created_ms)
            .map(|(id, _)| id.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(max: usize) -> SessionStore {
        SessionStore::new(max)
    }

    // --- ChatSession ---

    #[test]
    fn test_session_new_empty() {
        let s = ChatSession::new("s1", 1000);
        assert_eq!(s.session_id, "s1");
        assert!(s.messages.is_empty());
        assert_eq!(s.created_ms, 1000);
        assert_eq!(s.last_updated_ms, 1000);
    }

    #[test]
    fn test_session_add_message_increments_count() {
        let mut s = ChatSession::new("s1", 1000);
        s.add_message(MessageRole::User, "Hello", 1001);
        assert_eq!(s.message_count(), 1);
    }

    #[test]
    fn test_session_add_message_updates_last_updated() {
        let mut s = ChatSession::new("s1", 1000);
        s.add_message(MessageRole::User, "Hello", 2000);
        assert_eq!(s.last_updated_ms, 2000);
    }

    #[test]
    fn test_session_message_content_preserved() {
        let mut s = ChatSession::new("s1", 1000);
        s.add_message(MessageRole::Assistant, "How can I help?", 1001);
        assert_eq!(s.messages[0].content, "How can I help?");
    }

    #[test]
    fn test_session_message_role_preserved() {
        let mut s = ChatSession::new("s1", 1000);
        s.add_message(MessageRole::System, "Init", 1001);
        assert_eq!(s.messages[0].role, MessageRole::System);
    }

    #[test]
    fn test_session_message_count_multiple() {
        let mut s = ChatSession::new("s1", 1000);
        s.add_message(MessageRole::User, "a", 1001);
        s.add_message(MessageRole::Assistant, "b", 1002);
        s.add_message(MessageRole::User, "c", 1003);
        assert_eq!(s.message_count(), 3);
    }

    #[test]
    fn test_session_last_message_none_when_empty() {
        let s = ChatSession::new("s1", 1000);
        assert!(s.last_message().is_none());
    }

    #[test]
    fn test_session_last_message_returns_last() {
        let mut s = ChatSession::new("s1", 1000);
        s.add_message(MessageRole::User, "first", 1001);
        s.add_message(MessageRole::Assistant, "second", 1002);
        let last = s.last_message().expect("should have last");
        assert_eq!(last.content, "second");
    }

    #[test]
    fn test_session_messages_by_role_user() {
        let mut s = ChatSession::new("s1", 1000);
        s.add_message(MessageRole::User, "u1", 1001);
        s.add_message(MessageRole::Assistant, "a1", 1002);
        s.add_message(MessageRole::User, "u2", 1003);
        let user_msgs = s.messages_by_role(&MessageRole::User);
        assert_eq!(user_msgs.len(), 2);
    }

    #[test]
    fn test_session_messages_by_role_none_match() {
        let mut s = ChatSession::new("s1", 1000);
        s.add_message(MessageRole::User, "u1", 1001);
        let sys_msgs = s.messages_by_role(&MessageRole::System);
        assert!(sys_msgs.is_empty());
    }

    #[test]
    fn test_session_metadata_insert() {
        let mut s = ChatSession::new("s1", 1000);
        s.metadata.insert("key".to_string(), "value".to_string());
        assert_eq!(s.metadata.get("key").map(String::as_str), Some("value"));
    }

    // --- SessionStore ---

    #[test]
    fn test_store_create_and_get() {
        let mut st = store(10);
        st.create("s1", 1000);
        assert!(st.get("s1").is_some());
    }

    #[test]
    fn test_store_get_nonexistent() {
        let st = store(10);
        assert!(st.get("missing").is_none());
    }

    #[test]
    fn test_store_count_after_create() {
        let mut st = store(10);
        st.create("s1", 1000);
        st.create("s2", 1001);
        assert_eq!(st.count(), 2);
    }

    #[test]
    fn test_store_delete_existing() {
        let mut st = store(10);
        st.create("s1", 1000);
        assert!(st.delete("s1"));
        assert!(st.get("s1").is_none());
    }

    #[test]
    fn test_store_delete_nonexistent() {
        let mut st = store(10);
        assert!(!st.delete("missing"));
    }

    #[test]
    fn test_store_count_after_delete() {
        let mut st = store(10);
        st.create("s1", 1000);
        st.create("s2", 1001);
        st.delete("s1");
        assert_eq!(st.count(), 1);
    }

    #[test]
    fn test_store_list_ids_contains_all() {
        let mut st = store(10);
        st.create("a", 1000);
        st.create("b", 1001);
        let ids = st.list_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_store_list_ids_empty() {
        let st = store(10);
        assert!(st.list_ids().is_empty());
    }

    #[test]
    fn test_store_get_mut_allows_modification() {
        let mut st = store(10);
        st.create("s1", 1000);
        {
            let session = st.get_mut("s1").expect("exists");
            session.add_message(MessageRole::User, "hello", 1001);
        }
        let session = st.get("s1").expect("exists");
        assert_eq!(session.message_count(), 1);
    }

    #[test]
    fn test_store_get_mut_nonexistent() {
        let mut st = store(10);
        assert!(st.get_mut("missing").is_none());
    }

    #[test]
    fn test_store_oldest_session_id_none_when_empty() {
        let st = store(10);
        assert!(st.oldest_session_id().is_none());
    }

    #[test]
    fn test_store_oldest_session_id_single() {
        let mut st = store(10);
        st.create("s1", 1000);
        assert_eq!(st.oldest_session_id(), Some("s1"));
    }

    #[test]
    fn test_store_oldest_session_id_multiple() {
        let mut st = store(10);
        st.create("newer", 2000);
        st.create("oldest", 500);
        st.create("middle", 1000);
        assert_eq!(st.oldest_session_id(), Some("oldest"));
    }

    #[test]
    fn test_store_max_sessions_eviction() {
        let mut st = store(2);
        st.create("first", 100);
        st.create("second", 200);
        // At capacity; inserting "third" should evict "first" (oldest)
        st.create("third", 300);
        assert_eq!(st.count(), 2);
        assert!(st.get("first").is_none(), "oldest should be evicted");
        assert!(st.get("second").is_some());
        assert!(st.get("third").is_some());
    }

    #[test]
    fn test_store_max_sessions_one() {
        let mut st = store(1);
        st.create("a", 100);
        st.create("b", 200);
        assert_eq!(st.count(), 1);
        assert!(st.get("b").is_some());
    }

    #[test]
    fn test_store_max_sessions_zero_treated_as_one() {
        let mut st = store(0);
        st.create("a", 100);
        st.create("b", 200);
        assert_eq!(st.count(), 1);
    }

    #[test]
    fn test_store_created_session_has_correct_id() {
        let mut st = store(10);
        let s = st.create("my_session", 5000);
        assert_eq!(s.session_id, "my_session");
    }

    #[test]
    fn test_store_created_session_timestamp() {
        let mut st = store(10);
        let s = st.create("s1", 9999);
        assert_eq!(s.created_ms, 9999);
    }

    #[test]
    fn test_message_role_eq() {
        assert_eq!(MessageRole::User, MessageRole::User);
        assert_ne!(MessageRole::User, MessageRole::Assistant);
    }

    #[test]
    fn test_message_role_clone() {
        let role = MessageRole::System;
        let cloned = role.clone();
        assert_eq!(role, cloned);
    }

    #[test]
    fn test_chat_message_clone() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: "test".to_string(),
            timestamp_ms: 12345,
        };
        let cloned = msg.clone();
        assert_eq!(cloned.content, "test");
        assert_eq!(cloned.timestamp_ms, 12345);
    }

    #[test]
    fn test_session_add_multiple_roles() {
        let mut s = ChatSession::new("multi", 0);
        s.add_message(MessageRole::System, "system init", 1);
        s.add_message(MessageRole::User, "user msg", 2);
        s.add_message(MessageRole::Assistant, "assistant reply", 3);
        assert_eq!(s.messages_by_role(&MessageRole::System).len(), 1);
        assert_eq!(s.messages_by_role(&MessageRole::User).len(), 1);
        assert_eq!(s.messages_by_role(&MessageRole::Assistant).len(), 1);
    }
}
