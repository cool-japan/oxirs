//! Custom Prompts System
//!
//! Allows users to define custom system prompts, templates, and prompt engineering
//! strategies for their specific use cases. Supports variables, conditional logic,
//! and prompt composition.

use anyhow::{anyhow, Result};
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Custom prompt template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Template ID
    pub id: String,
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template content (supports Handlebars syntax)
    pub template: String,
    /// Required variables
    pub required_vars: Vec<String>,
    /// Optional variables with defaults
    pub optional_vars: HashMap<String, String>,
    /// Template tags for categorization
    pub tags: Vec<String>,
    /// Template metadata
    pub metadata: HashMap<String, String>,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Prompt template category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PromptCategory {
    /// System prompts for AI behavior
    System,
    /// User query prompts
    Query,
    /// Context generation prompts
    Context,
    /// Response formatting prompts
    Formatting,
    /// Multi-turn conversation prompts
    Conversation,
    /// Error handling prompts
    Error,
    /// Custom user-defined category
    Custom,
}

/// Prompt composition strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionStrategy {
    /// Concatenate prompts with separator
    Concatenate,
    /// Nested template expansion
    Nested,
    /// Conditional inclusion
    Conditional,
    /// Variable substitution only
    Substitute,
}

/// Prompt variables
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptVariables {
    /// Variable map
    pub vars: HashMap<String, String>,
}

impl PromptVariables {
    /// Create new empty variables
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    /// Add a variable
    pub fn add<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) -> &mut Self {
        self.vars.insert(key.into(), value.into());
        self
    }

    /// Get a variable
    pub fn get(&self, key: &str) -> Option<&String> {
        self.vars.get(key)
    }

    /// Merge with another variable set
    pub fn merge(&mut self, other: &PromptVariables) {
        for (key, value) in &other.vars {
            self.vars.insert(key.clone(), value.clone());
        }
    }
}

/// Custom prompts manager
pub struct CustomPromptsManager {
    /// Template registry
    templates: Arc<RwLock<HashMap<String, PromptTemplate>>>,
    /// Handlebars engine
    handlebars: Arc<RwLock<Handlebars<'static>>>,
    /// Default templates
    defaults: HashMap<PromptCategory, String>,
}

impl CustomPromptsManager {
    /// Create a new custom prompts manager
    pub fn new() -> Result<Self> {
        let mut manager = Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            handlebars: Arc::new(RwLock::new(Handlebars::new())),
            defaults: HashMap::new(),
        };

        // Load default templates
        manager.load_default_templates()?;

        info!("Initialized custom prompts manager");
        Ok(manager)
    }

    /// Load default templates
    fn load_default_templates(&mut self) -> Result<()> {
        // System prompt default
        self.defaults.insert(
            PromptCategory::System,
            "You are a helpful AI assistant with access to a knowledge graph. \
             Provide accurate, helpful, and well-reasoned responses based on the available data."
                .to_string(),
        );

        // Query prompt default
        self.defaults.insert(
            PromptCategory::Query,
            "User Query: {{query}}\n\nPlease analyze this query and provide a comprehensive response."
                .to_string(),
        );

        // Context prompt default
        self.defaults.insert(
            PromptCategory::Context,
            "Relevant Context:\n{{#each context}}\n- {{this}}\n{{/each}}\n\n\
             Use this context to inform your response."
                .to_string(),
        );

        // Formatting prompt default
        self.defaults.insert(
            PromptCategory::Formatting,
            "Please format your response as follows:\n\
             1. Provide a clear answer\n\
             2. Cite sources when applicable\n\
             3. Be concise but thorough"
                .to_string(),
        );

        // Conversation prompt default
        self.defaults.insert(
            PromptCategory::Conversation,
            "Previous conversation:\n{{conversation_history}}\n\n\
             Continue the conversation naturally and maintain context."
                .to_string(),
        );

        Ok(())
    }

    /// Register a custom template
    pub async fn register_template(&self, template: PromptTemplate) -> Result<()> {
        info!("Registering custom prompt template: {}", template.id);

        // Validate template
        self.validate_template(&template)?;

        // Compile template to check syntax
        let mut handlebars = self.handlebars.write().await;
        handlebars
            .register_template_string(&template.id, &template.template)
            .map_err(|e| anyhow!("Failed to compile template: {}", e))?;

        // Store template
        let mut templates = self.templates.write().await;
        templates.insert(template.id.clone(), template);

        Ok(())
    }

    /// Validate template
    fn validate_template(&self, template: &PromptTemplate) -> Result<()> {
        if template.id.is_empty() {
            return Err(anyhow!("Template ID cannot be empty"));
        }

        if template.template.is_empty() {
            return Err(anyhow!("Template content cannot be empty"));
        }

        Ok(())
    }

    /// Get template by ID
    pub async fn get_template(&self, template_id: &str) -> Result<PromptTemplate> {
        let templates = self.templates.read().await;
        templates
            .get(template_id)
            .cloned()
            .ok_or_else(|| anyhow!("Template not found: {}", template_id))
    }

    /// List all templates
    pub async fn list_templates(&self) -> Vec<PromptTemplate> {
        let templates = self.templates.read().await;
        templates.values().cloned().collect()
    }

    /// List templates by category
    pub async fn list_templates_by_tag(&self, tag: &str) -> Vec<PromptTemplate> {
        let templates = self.templates.read().await;
        templates
            .values()
            .filter(|t| t.tags.contains(&tag.to_string()))
            .cloned()
            .collect()
    }

    /// Render a template with variables
    pub async fn render(&self, template_id: &str, variables: &PromptVariables) -> Result<String> {
        debug!("Rendering template: {}", template_id);

        let template = self.get_template(template_id).await?;

        // Check required variables
        for required_var in &template.required_vars {
            if !variables.vars.contains_key(required_var) {
                return Err(anyhow!("Missing required variable: {}", required_var));
            }
        }

        // Merge with optional defaults
        let mut merged_vars = PromptVariables::new();
        for (key, value) in &template.optional_vars {
            merged_vars.add(key.clone(), value.clone());
        }
        merged_vars.merge(variables);

        // Render template
        let handlebars = self.handlebars.read().await;
        let rendered = handlebars
            .render(template_id, &merged_vars.vars)
            .map_err(|e| anyhow!("Template rendering failed: {}", e))?;

        Ok(rendered)
    }

    /// Compose multiple templates
    pub async fn compose(
        &self,
        template_ids: &[String],
        variables: &PromptVariables,
        strategy: CompositionStrategy,
    ) -> Result<String> {
        debug!(
            "Composing {} templates with strategy: {:?}",
            template_ids.len(),
            strategy
        );

        match strategy {
            CompositionStrategy::Concatenate => {
                let mut parts = Vec::new();
                for template_id in template_ids {
                    let rendered = self.render(template_id, variables).await?;
                    parts.push(rendered);
                }
                Ok(parts.join("\n\n"))
            }
            CompositionStrategy::Nested => {
                // Render templates in order, passing output as input to next
                let mut result = String::new();
                let mut vars = variables.clone();

                for template_id in template_ids {
                    vars.add("previous_output", result.clone());
                    result = self.render(template_id, &vars).await?;
                }

                Ok(result)
            }
            CompositionStrategy::Conditional => {
                // Include templates based on conditions
                let mut parts = Vec::new();
                for template_id in template_ids {
                    if let Ok(rendered) = self.render(template_id, variables).await {
                        if !rendered.is_empty() {
                            parts.push(rendered);
                        }
                    }
                }
                Ok(parts.join("\n\n"))
            }
            CompositionStrategy::Substitute => {
                // Simple variable substitution
                let mut result = template_ids.join("\n\n");
                for (key, value) in &variables.vars {
                    result = result.replace(&format!("{{{{{}}}}}", key), value);
                }
                Ok(result)
            }
        }
    }

    /// Get default prompt for a category
    pub fn get_default(&self, category: PromptCategory) -> Option<String> {
        self.defaults.get(&category).cloned()
    }

    /// Update a template
    pub async fn update_template(&self, template: PromptTemplate) -> Result<()> {
        let mut templates = self.templates.write().await;

        if !templates.contains_key(&template.id) {
            return Err(anyhow!("Template does not exist: {}", template.id));
        }

        // Validate and update
        self.validate_template(&template)?;

        let mut handlebars = self.handlebars.write().await;
        handlebars
            .register_template_string(&template.id, &template.template)
            .map_err(|e| anyhow!("Failed to compile template: {}", e))?;

        templates.insert(template.id.clone(), template);

        Ok(())
    }

    /// Delete a template
    pub async fn delete_template(&self, template_id: &str) -> Result<()> {
        let mut templates = self.templates.write().await;

        if templates.remove(template_id).is_none() {
            return Err(anyhow!("Template not found: {}", template_id));
        }

        let mut handlebars = self.handlebars.write().await;
        handlebars.unregister_template(template_id);

        info!("Deleted template: {}", template_id);
        Ok(())
    }

    /// Import templates from file
    pub async fn import_from_file<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let content = tokio::fs::read_to_string(path.as_ref()).await?;
        let templates: Vec<PromptTemplate> = serde_json::from_str(&content)?;

        let mut count = 0;
        for template in templates {
            match self.register_template(template).await {
                Ok(_) => count += 1,
                Err(e) => warn!("Failed to import template: {}", e),
            }
        }

        info!("Imported {} templates from file", count);
        Ok(count)
    }

    /// Export templates to file
    pub async fn export_to_file<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let templates = self.list_templates().await;
        let json = serde_json::to_string_pretty(&templates)?;

        tokio::fs::write(path.as_ref(), json).await?;

        info!("Exported {} templates to file", templates.len());
        Ok(templates.len())
    }

    /// Clone an existing template with modifications
    pub async fn clone_template(
        &self,
        source_id: &str,
        new_id: String,
        modifications: Option<HashMap<String, String>>,
    ) -> Result<PromptTemplate> {
        let mut source = self.get_template(source_id).await?;

        source.id = new_id;
        source.created_at = chrono::Utc::now();
        source.updated_at = chrono::Utc::now();

        if let Some(mods) = modifications {
            for (key, value) in mods {
                match key.as_str() {
                    "name" => source.name = value,
                    "description" => source.description = value,
                    "template" => source.template = value,
                    _ => {
                        source.metadata.insert(key, value);
                    }
                }
            }
        }

        self.register_template(source.clone()).await?;

        Ok(source)
    }
}

impl Default for CustomPromptsManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default CustomPromptsManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_custom_prompts_manager_creation() {
        let manager = CustomPromptsManager::new().unwrap();
        assert!(manager.get_default(PromptCategory::System).is_some());
    }

    #[tokio::test]
    async fn test_register_and_get_template() {
        let manager = CustomPromptsManager::new().unwrap();

        let template = PromptTemplate {
            id: "test_template".to_string(),
            name: "Test Template".to_string(),
            description: "A test template".to_string(),
            template: "Hello {{name}}!".to_string(),
            required_vars: vec!["name".to_string()],
            optional_vars: HashMap::new(),
            tags: vec!["test".to_string()],
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        manager.register_template(template.clone()).await.unwrap();

        let retrieved = manager.get_template("test_template").await.unwrap();
        assert_eq!(retrieved.id, "test_template");
    }

    #[tokio::test]
    async fn test_render_template() {
        let manager = CustomPromptsManager::new().unwrap();

        let template = PromptTemplate {
            id: "greeting".to_string(),
            name: "Greeting".to_string(),
            description: "Simple greeting".to_string(),
            template: "Hello {{name}}, welcome to {{place}}!".to_string(),
            required_vars: vec!["name".to_string()],
            optional_vars: {
                let mut map = HashMap::new();
                map.insert("place".to_string(), "the system".to_string());
                map
            },
            tags: vec![],
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        manager.register_template(template).await.unwrap();

        let mut vars = PromptVariables::new();
        vars.add("name", "Alice");

        let rendered = manager.render("greeting", &vars).await.unwrap();
        assert_eq!(rendered, "Hello Alice, welcome to the system!");
    }

    #[tokio::test]
    async fn test_compose_templates() {
        let manager = CustomPromptsManager::new().unwrap();

        let template1 = PromptTemplate {
            id: "part1".to_string(),
            name: "Part 1".to_string(),
            description: "First part".to_string(),
            template: "Part 1: {{content1}}".to_string(),
            required_vars: vec!["content1".to_string()],
            optional_vars: HashMap::new(),
            tags: vec![],
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let template2 = PromptTemplate {
            id: "part2".to_string(),
            name: "Part 2".to_string(),
            description: "Second part".to_string(),
            template: "Part 2: {{content2}}".to_string(),
            required_vars: vec!["content2".to_string()],
            optional_vars: HashMap::new(),
            tags: vec![],
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        manager.register_template(template1).await.unwrap();
        manager.register_template(template2).await.unwrap();

        let mut vars = PromptVariables::new();
        vars.add("content1", "First");
        vars.add("content2", "Second");

        let composed = manager
            .compose(
                &["part1".to_string(), "part2".to_string()],
                &vars,
                CompositionStrategy::Concatenate,
            )
            .await
            .unwrap();

        assert!(composed.contains("Part 1: First"));
        assert!(composed.contains("Part 2: Second"));
    }

    #[tokio::test]
    async fn test_missing_required_variable() {
        let manager = CustomPromptsManager::new().unwrap();

        let template = PromptTemplate {
            id: "required_test".to_string(),
            name: "Required Test".to_string(),
            description: "Test required variables".to_string(),
            template: "{{required_var}}".to_string(),
            required_vars: vec!["required_var".to_string()],
            optional_vars: HashMap::new(),
            tags: vec![],
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        manager.register_template(template).await.unwrap();

        let vars = PromptVariables::new(); // Empty variables

        let result = manager.render("required_test", &vars).await;
        assert!(result.is_err());
    }
}
