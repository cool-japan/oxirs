//! Plugin System
//!
//! Extensible plugin architecture for adding custom functionality to oxirs-chat.

use crate::messages::Message;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Author
    pub author: String,
    /// Description
    pub description: String,
    /// Plugin capabilities
    pub capabilities: Vec<PluginCapability>,
}

/// Plugin capabilities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginCapability {
    /// Process messages before they are sent to the LLM
    PreProcessMessage,
    /// Process messages after LLM response
    PostProcessMessage,
    /// Add custom commands
    CustomCommands,
    /// Modify query results
    ModifyResults,
    /// Add custom UI elements
    CustomUI,
    /// Integrate external services
    ExternalIntegration,
}

/// Plugin hook points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginHook {
    /// Before message processing
    BeforeProcess,
    /// After message processing
    AfterProcess,
    /// Before SPARQL execution
    BeforeSPARQL,
    /// After SPARQL execution
    AfterSPARQL,
    /// Before LLM call
    BeforeLLM,
    /// After LLM call
    AfterLLM,
}

/// Plugin execution context
#[derive(Debug, Clone)]
pub struct PluginContext {
    /// Current session ID
    pub session_id: String,
    /// User message
    pub message: Option<Message>,
    /// Additional context data
    pub data: HashMap<String, serde_json::Value>,
}

/// Plugin trait - all plugins must implement this
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Initialize the plugin
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Execute plugin at a specific hook point
    async fn execute(&self, hook: PluginHook, context: &mut PluginContext) -> Result<()>;

    /// Shutdown the plugin
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    /// Check if plugin handles a specific hook
    fn handles_hook(&self, hook: PluginHook) -> bool;
}

/// Plugin manager
pub struct PluginManager {
    plugins: Arc<RwLock<HashMap<String, Box<dyn Plugin>>>>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        info!("Initialized plugin manager");

        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a plugin
    pub async fn register_plugin(&self, plugin: Box<dyn Plugin>) -> Result<()> {
        let metadata = plugin.metadata();
        let plugin_name = metadata.name.clone();

        info!("Registering plugin: {} v{}", plugin_name, metadata.version);

        let mut plugins = self.plugins.write().await;
        plugins.insert(plugin_name, plugin);

        Ok(())
    }

    /// Unregister a plugin
    pub async fn unregister_plugin(&self, plugin_name: &str) -> Result<()> {
        info!("Unregistering plugin: {}", plugin_name);

        let mut plugins = self.plugins.write().await;
        if let Some(mut plugin) = plugins.remove(plugin_name) {
            plugin.shutdown().await?;
        }

        Ok(())
    }

    /// Execute all plugins at a specific hook point
    pub async fn execute_hook(&self, hook: PluginHook, context: &mut PluginContext) -> Result<()> {
        debug!("Executing hook: {:?}", hook);

        let plugins = self.plugins.read().await;

        for (name, plugin) in plugins.iter() {
            if plugin.handles_hook(hook) {
                debug!("Executing plugin {} for hook {:?}", name, hook);

                if let Err(e) = plugin.execute(hook, context).await {
                    warn!("Plugin {} failed at hook {:?}: {}", name, hook, e);
                    // Continue with other plugins
                }
            }
        }

        Ok(())
    }

    /// Get list of registered plugins
    pub async fn list_plugins(&self) -> Vec<PluginMetadata> {
        let plugins = self.plugins.read().await;
        plugins.values().map(|p| p.metadata().clone()).collect()
    }

    /// Get plugin by name
    pub async fn get_plugin(&self, name: &str) -> Option<PluginMetadata> {
        let plugins = self.plugins.read().await;
        plugins.get(name).map(|p| p.metadata().clone())
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Example plugin: Message logger
pub struct MessageLoggerPlugin {
    metadata: PluginMetadata,
    log_file: Option<String>,
}

impl MessageLoggerPlugin {
    pub fn new(log_file: Option<String>) -> Self {
        Self {
            metadata: PluginMetadata {
                name: "message-logger".to_string(),
                version: "1.0.0".to_string(),
                author: "OxiRS Team".to_string(),
                description: "Logs all messages to a file".to_string(),
                capabilities: vec![
                    PluginCapability::PreProcessMessage,
                    PluginCapability::PostProcessMessage,
                ],
            },
            log_file,
        }
    }
}

#[async_trait]
impl Plugin for MessageLoggerPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn execute(&self, hook: PluginHook, context: &mut PluginContext) -> Result<()> {
        if let Some(message) = &context.message {
            let log_entry = format!(
                "[{:?}] Session: {}, Message: {}",
                hook, context.session_id, message.content
            );

            if let Some(ref file) = self.log_file {
                use std::fs::OpenOptions;
                use std::io::Write;

                let mut file = OpenOptions::new().create(true).append(true).open(file)?;

                writeln!(file, "{}", log_entry)?;
            } else {
                info!("{}", log_entry);
            }
        }

        Ok(())
    }

    fn handles_hook(&self, hook: PluginHook) -> bool {
        matches!(hook, PluginHook::BeforeProcess | PluginHook::AfterProcess)
    }
}

/// Example plugin: Profanity filter
pub struct ProfanityFilterPlugin {
    metadata: PluginMetadata,
    blocked_words: Vec<String>,
}

impl ProfanityFilterPlugin {
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                name: "profanity-filter".to_string(),
                version: "1.0.0".to_string(),
                author: "OxiRS Team".to_string(),
                description: "Filters profanity from messages".to_string(),
                capabilities: vec![PluginCapability::PreProcessMessage],
            },
            blocked_words: vec!["badword1".to_string(), "badword2".to_string()],
        }
    }
}

#[async_trait]
impl Plugin for ProfanityFilterPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn execute(&self, _hook: PluginHook, context: &mut PluginContext) -> Result<()> {
        if let Some(message) = &mut context.message {
            let mut content = message.content.to_string();

            for word in &self.blocked_words {
                content = content.replace(word, &"*".repeat(word.len()));
            }

            // Update message content (simplified - would need proper MessageContent handling)
            context
                .data
                .insert("filtered_content".to_string(), serde_json::json!(content));
        }

        Ok(())
    }

    fn handles_hook(&self, hook: PluginHook) -> bool {
        matches!(hook, PluginHook::BeforeProcess)
    }
}

impl Default for ProfanityFilterPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{MessageContent, MessageRole};
    use chrono::Utc;

    #[tokio::test]
    async fn test_plugin_registration() {
        let manager = PluginManager::new();
        let plugin = Box::new(MessageLoggerPlugin::new(None));

        manager.register_plugin(plugin).await.unwrap();

        let plugins = manager.list_plugins().await;
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "message-logger");
    }

    #[tokio::test]
    async fn test_plugin_execution() {
        let manager = PluginManager::new();
        let plugin = Box::new(MessageLoggerPlugin::new(None));

        manager.register_plugin(plugin).await.unwrap();

        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::User,
            content: MessageContent::from_text("Test message".to_string()),
            timestamp: Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: None,
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };

        let mut context = PluginContext {
            session_id: "test-session".to_string(),
            message: Some(message),
            data: HashMap::new(),
        };

        manager
            .execute_hook(PluginHook::BeforeProcess, &mut context)
            .await
            .unwrap();
    }
}
