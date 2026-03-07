/// Field-level resolver pipeline with middleware support for GraphQL.
///
/// Provides a composable pipeline where field handlers can be wrapped
/// with middleware that intercepts calls for logging, auth, caching, etc.
use std::collections::HashMap;
use std::fmt;

/// Result type returned by resolvers and middleware.
pub type ResolverResult = Result<serde_json::Value, ResolverError>;

/// Errors that can occur during field resolution.
#[derive(Debug)]
pub enum ResolverError {
    /// The requested field has no registered handler.
    FieldNotFound(String),
    /// A middleware layer encountered an error.
    MiddlewareError(String),
    /// The field handler itself returned an error.
    HandlerError(String),
}

impl fmt::Display for ResolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolverError::FieldNotFound(name) => write!(f, "Field not found: {name}"),
            ResolverError::MiddlewareError(msg) => write!(f, "Middleware error: {msg}"),
            ResolverError::HandlerError(msg) => write!(f, "Handler error: {msg}"),
        }
    }
}

impl std::error::Error for ResolverError {}

/// Context passed to every resolver and middleware call.
#[derive(Debug, Clone)]
pub struct ResolverContext {
    /// The name of the field being resolved.
    pub field_name: String,
    /// The value of the parent object, if any.
    pub parent_value: Option<serde_json::Value>,
    /// Arguments passed to the field.
    pub args: HashMap<String, serde_json::Value>,
}

impl ResolverContext {
    /// Create a new resolver context.
    pub fn new(field_name: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
            parent_value: None,
            args: HashMap::new(),
        }
    }

    /// Create a context with a parent value.
    pub fn with_parent(mut self, parent: serde_json::Value) -> Self {
        self.parent_value = Some(parent);
        self
    }

    /// Add an argument to the context.
    pub fn with_arg(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.args.insert(key.into(), value);
        self
    }
}

/// Trait implemented by field middleware.
///
/// Middleware wraps field resolution allowing pre/post processing.
pub trait FieldMiddleware: Send + Sync {
    /// Return the name of this middleware (for diagnostics).
    fn name(&self) -> &str;

    /// Process the field resolution, optionally calling `next` to continue.
    fn process(
        &self,
        ctx: &ResolverContext,
        next: &dyn Fn(&ResolverContext) -> ResolverResult,
    ) -> ResolverResult;
}

/// A handler function for a field.
type HandlerFn = Box<dyn Fn(&ResolverContext) -> ResolverResult + Send + Sync>;

/// Field resolver pipeline.
///
/// Executes registered middleware in order, then calls the field handler.
pub struct FieldResolver {
    middlewares: Vec<Box<dyn FieldMiddleware>>,
    handlers: HashMap<String, HandlerFn>,
}

impl FieldResolver {
    /// Create a new empty resolver.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
            handlers: HashMap::new(),
        }
    }

    /// Add a middleware to the pipeline.
    ///
    /// Middleware is applied in order of registration (outermost first).
    pub fn add_middleware(&mut self, mw: Box<dyn FieldMiddleware>) {
        self.middlewares.push(mw);
    }

    /// Register a handler for a field name.
    pub fn register(
        &mut self,
        field: impl Into<String>,
        handler: impl Fn(&ResolverContext) -> ResolverResult + Send + Sync + 'static,
    ) {
        self.handlers.insert(field.into(), Box::new(handler));
    }

    /// Resolve a field by running it through the middleware pipeline.
    pub fn resolve(&self, ctx: ResolverContext) -> ResolverResult {
        // Build the innermost "call handler" function.
        let field_name = ctx.field_name.clone();
        let handler = self
            .handlers
            .get(&field_name)
            .ok_or_else(|| ResolverError::FieldNotFound(field_name.clone()))?;

        // Wrap handler in a closure for the middleware chain.
        let base: &dyn Fn(&ResolverContext) -> ResolverResult = handler.as_ref();

        // Build the middleware chain from back to front.
        // We need to compose dynamically since each closure captures the next.
        // Use index-based recursion via a helper.
        self.run_middleware(&ctx, 0, base)
    }

    fn run_middleware(
        &self,
        ctx: &ResolverContext,
        idx: usize,
        final_handler: &dyn Fn(&ResolverContext) -> ResolverResult,
    ) -> ResolverResult {
        if idx >= self.middlewares.len() {
            return final_handler(ctx);
        }

        let mw = &self.middlewares[idx];
        let next_idx = idx + 1;

        // Build a "next" closure that invokes the remaining middleware.
        let next = |next_ctx: &ResolverContext| -> ResolverResult {
            self.run_middleware(next_ctx, next_idx, final_handler)
        };

        mw.process(ctx, &next)
    }

    /// Return the number of registered middleware layers.
    pub fn middleware_count(&self) -> usize {
        self.middlewares.len()
    }

    /// Return sorted list of registered field names.
    pub fn registered_fields(&self) -> Vec<&str> {
        let mut fields: Vec<&str> = self.handlers.keys().map(String::as_str).collect();
        fields.sort_unstable();
        fields
    }

    /// Return true if a handler is registered for the given field.
    pub fn has_field(&self, field: &str) -> bool {
        self.handlers.contains_key(field)
    }
}

impl Default for FieldResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Built-in middleware examples ----

/// Logging middleware that records field resolution attempts.
pub struct LoggingMiddleware {
    pub prefix: String,
}

impl LoggingMiddleware {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }
}

impl FieldMiddleware for LoggingMiddleware {
    fn name(&self) -> &str {
        "logging"
    }

    fn process(
        &self,
        ctx: &ResolverContext,
        next: &dyn Fn(&ResolverContext) -> ResolverResult,
    ) -> ResolverResult {
        let _log = format!("{} resolving field: {}", self.prefix, ctx.field_name);
        next(ctx)
    }
}

/// Middleware that always returns an error (for testing error propagation).
pub struct ErrorMiddleware {
    message: String,
}

impl ErrorMiddleware {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl FieldMiddleware for ErrorMiddleware {
    fn name(&self) -> &str {
        "error_injector"
    }

    fn process(
        &self,
        _ctx: &ResolverContext,
        _next: &dyn Fn(&ResolverContext) -> ResolverResult,
    ) -> ResolverResult {
        Err(ResolverError::MiddlewareError(self.message.clone()))
    }
}

/// Middleware that transforms the context field name to uppercase.
pub struct UppercaseFieldMiddleware;

impl FieldMiddleware for UppercaseFieldMiddleware {
    fn name(&self) -> &str {
        "uppercase_field"
    }

    fn process(
        &self,
        ctx: &ResolverContext,
        next: &dyn Fn(&ResolverContext) -> ResolverResult,
    ) -> ResolverResult {
        let mut new_ctx = ctx.clone();
        new_ctx.field_name = ctx.field_name.to_uppercase();
        next(&new_ctx)
    }
}

/// Middleware that passes through to the next layer (no-op).
pub struct PassthroughMiddleware;

impl FieldMiddleware for PassthroughMiddleware {
    fn name(&self) -> &str {
        "passthrough"
    }

    fn process(
        &self,
        ctx: &ResolverContext,
        next: &dyn Fn(&ResolverContext) -> ResolverResult,
    ) -> ResolverResult {
        next(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_resolver() -> FieldResolver {
        let mut r = FieldResolver::new();
        r.register("hello", |_ctx| Ok(json!("world")));
        r.register("answer", |_ctx| Ok(json!(42)));
        r.register("null_field", |_ctx| Ok(json!(null)));
        r
    }

    // --- ResolverError ---

    #[test]
    fn test_resolver_error_display_field_not_found() {
        let e = ResolverError::FieldNotFound("foo".to_string());
        assert!(e.to_string().contains("foo"));
    }

    #[test]
    fn test_resolver_error_display_middleware() {
        let e = ResolverError::MiddlewareError("oops".to_string());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn test_resolver_error_display_handler() {
        let e = ResolverError::HandlerError("bad".to_string());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn test_resolver_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(ResolverError::FieldNotFound("x".to_string()));
        assert!(!e.to_string().is_empty());
    }

    // --- ResolverContext ---

    #[test]
    fn test_context_new() {
        let ctx = ResolverContext::new("myField");
        assert_eq!(ctx.field_name, "myField");
        assert!(ctx.parent_value.is_none());
        assert!(ctx.args.is_empty());
    }

    #[test]
    fn test_context_with_parent() {
        let ctx = ResolverContext::new("f").with_parent(json!({"id": 1}));
        assert!(ctx.parent_value.is_some());
    }

    #[test]
    fn test_context_with_arg() {
        let ctx = ResolverContext::new("f").with_arg("limit", json!(10));
        assert_eq!(ctx.args.get("limit"), Some(&json!(10)));
    }

    #[test]
    fn test_context_clone() {
        let ctx = ResolverContext::new("f").with_arg("k", json!("v"));
        let ctx2 = ctx.clone();
        assert_eq!(ctx2.field_name, "f");
    }

    #[test]
    fn test_context_multiple_args() {
        let ctx = ResolverContext::new("f")
            .with_arg("a", json!(1))
            .with_arg("b", json!(2));
        assert_eq!(ctx.args.len(), 2);
    }

    // --- FieldResolver construction ---

    #[test]
    fn test_new_resolver_empty() {
        let r = FieldResolver::new();
        assert_eq!(r.middleware_count(), 0);
        assert!(r.registered_fields().is_empty());
    }

    #[test]
    fn test_default_resolver() {
        let r = FieldResolver::default();
        assert_eq!(r.middleware_count(), 0);
    }

    #[test]
    fn test_register_field() {
        let mut r = FieldResolver::new();
        r.register("myField", |_| Ok(json!("ok")));
        assert!(r.has_field("myField"));
    }

    #[test]
    fn test_registered_fields_sorted() {
        let mut r = FieldResolver::new();
        r.register("z_field", |_| Ok(json!(1)));
        r.register("a_field", |_| Ok(json!(2)));
        r.register("m_field", |_| Ok(json!(3)));
        let fields = r.registered_fields();
        assert_eq!(fields, vec!["a_field", "m_field", "z_field"]);
    }

    #[test]
    fn test_middleware_count() {
        let mut r = FieldResolver::new();
        r.add_middleware(Box::new(PassthroughMiddleware));
        r.add_middleware(Box::new(PassthroughMiddleware));
        assert_eq!(r.middleware_count(), 2);
    }

    // --- resolve ---

    #[test]
    fn test_resolve_registered_field() {
        let r = make_resolver();
        let ctx = ResolverContext::new("hello");
        let result = r.resolve(ctx).expect("should resolve");
        assert_eq!(result, json!("world"));
    }

    #[test]
    fn test_resolve_integer_field() {
        let r = make_resolver();
        let ctx = ResolverContext::new("answer");
        let result = r.resolve(ctx).expect("should resolve");
        assert_eq!(result, json!(42));
    }

    #[test]
    fn test_resolve_null_field() {
        let r = make_resolver();
        let ctx = ResolverContext::new("null_field");
        let result = r.resolve(ctx).expect("should resolve");
        assert_eq!(result, json!(null));
    }

    #[test]
    fn test_resolve_unknown_field_returns_error() {
        let r = make_resolver();
        let ctx = ResolverContext::new("nonexistent");
        let err = r.resolve(ctx).expect_err("should fail");
        assert!(matches!(err, ResolverError::FieldNotFound(_)));
    }

    #[test]
    fn test_resolve_with_context_args() {
        let mut r = FieldResolver::new();
        r.register("greet", |ctx| {
            let name = ctx
                .args
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("World");
            Ok(json!(format!("Hello, {name}!")))
        });
        let ctx = ResolverContext::new("greet").with_arg("name", json!("Rust"));
        let result = r.resolve(ctx).expect("ok");
        assert_eq!(result, json!("Hello, Rust!"));
    }

    #[test]
    fn test_resolve_handler_error_propagated() {
        let mut r = FieldResolver::new();
        r.register("bad", |_| Err(ResolverError::HandlerError("fail".into())));
        let ctx = ResolverContext::new("bad");
        let err = r.resolve(ctx).expect_err("should fail");
        assert!(matches!(err, ResolverError::HandlerError(_)));
    }

    // --- Middleware ---

    #[test]
    fn test_passthrough_middleware_does_not_change_result() {
        let mut r = make_resolver();
        r.add_middleware(Box::new(PassthroughMiddleware));
        let ctx = ResolverContext::new("hello");
        let result = r.resolve(ctx).expect("ok");
        assert_eq!(result, json!("world"));
    }

    #[test]
    fn test_logging_middleware_transparent() {
        let mut r = make_resolver();
        r.add_middleware(Box::new(LoggingMiddleware::new("TEST")));
        let ctx = ResolverContext::new("answer");
        let result = r.resolve(ctx).expect("ok");
        assert_eq!(result, json!(42));
    }

    #[test]
    fn test_error_middleware_short_circuits() {
        let mut r = make_resolver();
        r.add_middleware(Box::new(ErrorMiddleware::new("blocked")));
        let ctx = ResolverContext::new("hello");
        let err = r.resolve(ctx).expect_err("should fail");
        assert!(matches!(err, ResolverError::MiddlewareError(_)));
    }

    #[test]
    fn test_middleware_chain_order() {
        // Two passthrough middlewares → result still returned.
        let mut r = make_resolver();
        r.add_middleware(Box::new(PassthroughMiddleware));
        r.add_middleware(Box::new(LoggingMiddleware::new("x")));
        let ctx = ResolverContext::new("hello");
        assert!(r.resolve(ctx).is_ok());
    }

    #[test]
    fn test_middleware_name() {
        let mw = LoggingMiddleware::new("test");
        assert_eq!(mw.name(), "logging");

        let emw = ErrorMiddleware::new("err");
        assert_eq!(emw.name(), "error_injector");

        let pmw = PassthroughMiddleware;
        assert_eq!(pmw.name(), "passthrough");
    }

    #[test]
    fn test_multiple_fields_resolved_correctly() {
        let r = make_resolver();
        let r1 = r.resolve(ResolverContext::new("hello")).expect("ok");
        let r2 = r.resolve(ResolverContext::new("answer")).expect("ok");
        assert_eq!(r1, json!("world"));
        assert_eq!(r2, json!(42));
    }

    #[test]
    fn test_resolve_with_parent_value() {
        let mut r = FieldResolver::new();
        r.register("id", |ctx| {
            let id = ctx
                .parent_value
                .as_ref()
                .and_then(|v| v.get("id"))
                .cloned()
                .unwrap_or(json!(0));
            Ok(id)
        });
        let ctx = ResolverContext::new("id").with_parent(json!({"id": 99}));
        let result = r.resolve(ctx).expect("ok");
        assert_eq!(result, json!(99));
    }

    #[test]
    fn test_error_before_middleware_not_reached() {
        // When field is not found, middleware shouldn't even be called.
        let mut r = FieldResolver::new();
        r.add_middleware(Box::new(PassthroughMiddleware));
        let ctx = ResolverContext::new("not_registered");
        let err = r.resolve(ctx).expect_err("should fail");
        assert!(matches!(err, ResolverError::FieldNotFound(_)));
    }

    #[test]
    fn test_has_field_true() {
        let r = make_resolver();
        assert!(r.has_field("hello"));
        assert!(r.has_field("answer"));
    }

    #[test]
    fn test_has_field_false() {
        let r = make_resolver();
        assert!(!r.has_field("missing"));
    }

    #[test]
    fn test_overwrite_handler() {
        let mut r = FieldResolver::new();
        r.register("f", |_| Ok(json!(1)));
        r.register("f", |_| Ok(json!(2)));
        let result = r.resolve(ResolverContext::new("f")).expect("ok");
        // Second registration wins.
        assert_eq!(result, json!(2));
    }

    #[test]
    fn test_many_middlewares_passthrough() {
        let mut r = make_resolver();
        for _ in 0..10 {
            r.add_middleware(Box::new(PassthroughMiddleware));
        }
        assert_eq!(r.middleware_count(), 10);
        let result = r.resolve(ResolverContext::new("hello")).expect("ok");
        assert_eq!(result, json!("world"));
    }

    #[test]
    fn test_resolver_field_not_found_message() {
        let r = FieldResolver::new();
        let err = r.resolve(ResolverContext::new("ghost")).expect_err("fail");
        if let ResolverError::FieldNotFound(name) = err {
            assert_eq!(name, "ghost");
        } else {
            panic!("Expected FieldNotFound");
        }
    }

    #[test]
    fn test_null_json_value_handler() {
        let mut r = FieldResolver::new();
        r.register("nullable", |_| Ok(serde_json::Value::Null));
        let result = r.resolve(ResolverContext::new("nullable")).expect("ok");
        assert!(result.is_null());
    }

    #[test]
    fn test_array_json_value_handler() {
        let mut r = FieldResolver::new();
        r.register("list", |_| Ok(json!([1, 2, 3])));
        let result = r.resolve(ResolverContext::new("list")).expect("ok");
        assert_eq!(result, json!([1, 2, 3]));
    }

    #[test]
    fn test_object_json_value_handler() {
        let mut r = FieldResolver::new();
        r.register("obj", |_| Ok(json!({"a": 1, "b": "two"})));
        let result = r.resolve(ResolverContext::new("obj")).expect("ok");
        assert_eq!(result["a"], json!(1));
    }
}
