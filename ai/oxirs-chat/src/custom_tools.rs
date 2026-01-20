//! Custom Tools Framework
//!
//! Extensible framework for adding custom tools and functions that the AI can use
//! to enhance its capabilities. Supports function calling, parameter validation,
//! and tool chaining.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Tool parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Is this parameter required?
    pub required: bool,
    /// Default value (if not required)
    pub default_value: Option<JsonValue>,
    /// Parameter constraints
    pub constraints: Option<ParameterConstraints>,
}

/// Parameter type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// Boolean type
    Boolean,
    /// Array of values
    Array,
    /// Object/map
    Object,
    /// Any JSON value
    Any,
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraints {
    /// Minimum value (for numbers)
    pub min: Option<f64>,
    /// Maximum value (for numbers)
    pub max: Option<f64>,
    /// Allowed values (enum)
    pub allowed_values: Option<Vec<JsonValue>>,
    /// Regex pattern (for strings)
    pub pattern: Option<String>,
    /// Minimum length (for strings/arrays)
    pub min_length: Option<usize>,
    /// Maximum length (for strings/arrays)
    pub max_length: Option<usize>,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Success flag
    pub success: bool,
    /// Result data
    pub data: JsonValue,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool parameters
    pub parameters: Vec<ToolParameter>,
    /// Tool category
    pub category: ToolCategory,
    /// Is tool async?
    pub is_async: bool,
    /// Tool tags for discovery
    pub tags: Vec<String>,
    /// Examples of usage
    pub examples: Vec<ToolExample>,
}

/// Tool category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Data retrieval tools
    Retrieval,
    /// Data transformation tools
    Transformation,
    /// Analysis tools
    Analysis,
    /// External API tools
    ExternalAPI,
    /// Utility tools
    Utility,
    /// Custom user-defined
    Custom,
}

/// Tool usage example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExample {
    /// Example description
    pub description: String,
    /// Input parameters
    pub input: HashMap<String, JsonValue>,
    /// Expected output
    pub output: JsonValue,
}

/// Custom tool trait
#[async_trait]
pub trait CustomTool: Send + Sync {
    /// Get tool definition
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool
    async fn execute(&self, parameters: HashMap<String, JsonValue>) -> Result<ToolResult>;

    /// Validate parameters before execution
    fn validate(&self, parameters: &HashMap<String, JsonValue>) -> Result<()> {
        let definition = self.definition();

        // Check required parameters
        for param in &definition.parameters {
            if param.required && !parameters.contains_key(&param.name) {
                return Err(anyhow!("Missing required parameter: {}", param.name));
            }

            // Validate parameter if present
            if let Some(value) = parameters.get(&param.name) {
                self.validate_parameter(param, value)?;
            }
        }

        Ok(())
    }

    /// Validate a single parameter
    fn validate_parameter(&self, param: &ToolParameter, value: &JsonValue) -> Result<()> {
        // Type validation
        match param.param_type {
            ParameterType::String => {
                if !value.is_string() {
                    return Err(anyhow!("Parameter {} must be a string", param.name));
                }
            }
            ParameterType::Integer => {
                if !value.is_i64() && !value.is_u64() {
                    return Err(anyhow!("Parameter {} must be an integer", param.name));
                }
            }
            ParameterType::Float => {
                if !value.is_f64() && !value.is_i64() {
                    return Err(anyhow!("Parameter {} must be a number", param.name));
                }
            }
            ParameterType::Boolean => {
                if !value.is_boolean() {
                    return Err(anyhow!("Parameter {} must be a boolean", param.name));
                }
            }
            ParameterType::Array => {
                if !value.is_array() {
                    return Err(anyhow!("Parameter {} must be an array", param.name));
                }
            }
            ParameterType::Object => {
                if !value.is_object() {
                    return Err(anyhow!("Parameter {} must be an object", param.name));
                }
            }
            ParameterType::Any => {} // Any type is always valid
        }

        // Constraint validation
        if let Some(constraints) = &param.constraints {
            self.validate_constraints(param, value, constraints)?;
        }

        Ok(())
    }

    /// Validate parameter constraints
    fn validate_constraints(
        &self,
        param: &ToolParameter,
        value: &JsonValue,
        constraints: &ParameterConstraints,
    ) -> Result<()> {
        // Numeric constraints
        if let Some(num) = value.as_f64() {
            if let Some(min) = constraints.min {
                if num < min {
                    return Err(anyhow!("Parameter {} must be >= {}", param.name, min));
                }
            }
            if let Some(max) = constraints.max {
                if num > max {
                    return Err(anyhow!("Parameter {} must be <= {}", param.name, max));
                }
            }
        }

        // String/Array length constraints
        let len = value
            .as_str()
            .map(|s| s.len())
            .or_else(|| value.as_array().map(|arr| arr.len()));

        if let Some(length) = len {
            if let Some(min_len) = constraints.min_length {
                if length < min_len {
                    return Err(anyhow!(
                        "Parameter {} length must be >= {}",
                        param.name,
                        min_len
                    ));
                }
            }
            if let Some(max_len) = constraints.max_length {
                if length > max_len {
                    return Err(anyhow!(
                        "Parameter {} length must be <= {}",
                        param.name,
                        max_len
                    ));
                }
            }
        }

        // Allowed values constraint
        if let Some(allowed) = &constraints.allowed_values {
            if !allowed.contains(value) {
                return Err(anyhow!(
                    "Parameter {} must be one of: {:?}",
                    param.name,
                    allowed
                ));
            }
        }

        // Pattern constraint (for strings)
        if let Some(pattern) = &constraints.pattern {
            if let Some(s) = value.as_str() {
                let regex = regex::Regex::new(pattern)?;
                if !regex.is_match(s) {
                    return Err(anyhow!(
                        "Parameter {} must match pattern: {}",
                        param.name,
                        pattern
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Custom tools registry
pub struct CustomToolsRegistry {
    /// Registered tools
    tools: Arc<RwLock<HashMap<String, Arc<dyn CustomTool>>>>,
    /// Tool execution history
    execution_history: Arc<RwLock<Vec<ToolExecution>>>,
}

/// Tool execution record
#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub tool_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub parameters: HashMap<String, JsonValue>,
    pub result: ToolResult,
}

impl CustomToolsRegistry {
    /// Create a new tools registry
    pub fn new() -> Self {
        info!("Initialized custom tools registry");

        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a custom tool
    pub async fn register_tool(&self, tool: Arc<dyn CustomTool>) -> Result<()> {
        let definition = tool.definition();
        info!("Registering custom tool: {}", definition.id);

        let mut tools = self.tools.write().await;
        if tools.contains_key(&definition.id) {
            warn!("Tool {} already exists, replacing", definition.id);
        }

        tools.insert(definition.id.clone(), tool);

        Ok(())
    }

    /// Get a tool by ID
    pub async fn get_tool(&self, tool_id: &str) -> Option<Arc<dyn CustomTool>> {
        let tools = self.tools.read().await;
        tools.get(tool_id).cloned()
    }

    /// List all registered tools
    pub async fn list_tools(&self) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        tools.values().map(|t| t.definition()).collect()
    }

    /// List tools by category
    pub async fn list_tools_by_category(&self, category: ToolCategory) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        tools
            .values()
            .map(|t| t.definition())
            .filter(|d| d.category == category)
            .collect()
    }

    /// Search tools by tag
    pub async fn search_by_tag(&self, tag: &str) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        tools
            .values()
            .map(|t| t.definition())
            .filter(|d| d.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Execute a tool
    pub async fn execute_tool(
        &self,
        tool_id: &str,
        parameters: HashMap<String, JsonValue>,
    ) -> Result<ToolResult> {
        debug!(
            "Executing tool: {} with parameters: {:?}",
            tool_id, parameters
        );

        let tool = self
            .get_tool(tool_id)
            .await
            .ok_or_else(|| anyhow!("Tool not found: {}", tool_id))?;

        // Validate parameters
        tool.validate(&parameters)?;

        // Execute tool
        let start_time = std::time::Instant::now();
        let result = tool.execute(parameters.clone()).await?;

        // Record execution
        let execution = ToolExecution {
            tool_id: tool_id.to_string(),
            timestamp: chrono::Utc::now(),
            parameters,
            result: result.clone(),
        };

        let mut history = self.execution_history.write().await;
        history.push(execution);

        // Keep history limited
        if history.len() > 1000 {
            history.drain(0..500);
        }

        debug!(
            "Tool execution completed in {}ms",
            start_time.elapsed().as_millis()
        );

        Ok(result)
    }

    /// Unregister a tool
    pub async fn unregister_tool(&self, tool_id: &str) -> Result<()> {
        let mut tools = self.tools.write().await;

        if tools.remove(tool_id).is_none() {
            return Err(anyhow!("Tool not found: {}", tool_id));
        }

        info!("Unregistered tool: {}", tool_id);
        Ok(())
    }

    /// Get execution history for a tool
    pub async fn get_execution_history(&self, tool_id: &str) -> Vec<ToolExecution> {
        let history = self.execution_history.read().await;
        history
            .iter()
            .filter(|e| e.tool_id == tool_id)
            .cloned()
            .collect()
    }

    /// Get execution statistics
    pub async fn get_statistics(&self) -> ToolStatistics {
        let tools = self.tools.read().await;
        let history = self.execution_history.read().await;

        let total_tools = tools.len();
        let total_executions = history.len();

        let mut execution_counts = HashMap::new();
        let mut success_counts = HashMap::new();

        for execution in history.iter() {
            *execution_counts
                .entry(execution.tool_id.clone())
                .or_insert(0) += 1;

            if execution.result.success {
                *success_counts.entry(execution.tool_id.clone()).or_insert(0) += 1;
            }
        }

        ToolStatistics {
            total_tools,
            total_executions,
            execution_counts,
            success_counts,
        }
    }
}

impl Default for CustomToolsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStatistics {
    /// Total registered tools
    pub total_tools: usize,
    /// Total executions
    pub total_executions: usize,
    /// Execution counts per tool
    pub execution_counts: HashMap<String, usize>,
    /// Success counts per tool
    pub success_counts: HashMap<String, usize>,
}

// Example built-in tool: Echo tool
pub struct EchoTool;

#[async_trait]
impl CustomTool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            id: "echo".to_string(),
            name: "Echo Tool".to_string(),
            description: "Simple echo tool that returns the input message".to_string(),
            parameters: vec![ToolParameter {
                name: "message".to_string(),
                description: "Message to echo".to_string(),
                param_type: ParameterType::String,
                required: true,
                default_value: None,
                constraints: None,
            }],
            category: ToolCategory::Utility,
            is_async: false,
            tags: vec!["utility".to_string(), "test".to_string()],
            examples: vec![ToolExample {
                description: "Echo hello world".to_string(),
                input: {
                    let mut map = HashMap::new();
                    map.insert(
                        "message".to_string(),
                        JsonValue::String("hello".to_string()),
                    );
                    map
                },
                output: JsonValue::String("hello".to_string()),
            }],
        }
    }

    async fn execute(&self, parameters: HashMap<String, JsonValue>) -> Result<ToolResult> {
        let start_time = std::time::Instant::now();

        let message = parameters
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing message parameter"))?;

        Ok(ToolResult {
            success: true,
            data: JsonValue::String(message.to_string()),
            error: None,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            metadata: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_creation() {
        let registry = CustomToolsRegistry::new();
        let tools = registry.list_tools().await;
        assert_eq!(tools.len(), 0);
    }

    #[tokio::test]
    async fn test_register_and_execute_tool() {
        let registry = CustomToolsRegistry::new();
        let echo_tool = Arc::new(EchoTool);

        registry.register_tool(echo_tool).await.unwrap();

        let mut params = HashMap::new();
        params.insert("message".to_string(), JsonValue::String("test".to_string()));

        let result = registry.execute_tool("echo", params).await.unwrap();

        assert!(result.success);
        assert_eq!(result.data, JsonValue::String("test".to_string()));
    }

    #[tokio::test]
    async fn test_list_tools() {
        let registry = CustomToolsRegistry::new();
        let echo_tool = Arc::new(EchoTool);

        registry.register_tool(echo_tool).await.unwrap();

        let tools = registry.list_tools().await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].id, "echo");
    }

    #[tokio::test]
    async fn test_parameter_validation() {
        let registry = CustomToolsRegistry::new();
        let echo_tool = Arc::new(EchoTool);

        registry.register_tool(echo_tool).await.unwrap();

        // Missing required parameter
        let params = HashMap::new();
        let result = registry.execute_tool("echo", params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_statistics() {
        let registry = CustomToolsRegistry::new();
        let echo_tool = Arc::new(EchoTool);

        registry.register_tool(echo_tool).await.unwrap();

        let mut params = HashMap::new();
        params.insert("message".to_string(), JsonValue::String("test".to_string()));

        // Execute multiple times
        for _ in 0..5 {
            let _ = registry.execute_tool("echo", params.clone()).await;
        }

        let stats = registry.get_statistics().await;
        assert_eq!(stats.total_tools, 1);
        assert_eq!(stats.total_executions, 5);
    }
}
