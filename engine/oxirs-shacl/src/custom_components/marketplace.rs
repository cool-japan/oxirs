//! Constraint Component Marketplace and Registry
//!
//! This module provides a central marketplace for discovering, sharing, and managing
//! custom SHACL constraint components with versioning, dependencies, and ratings.

use super::{CustomConstraintComponent, CustomConstraintRegistry};
use crate::{Result, ShaclError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Constraint component marketplace
#[derive(Debug)]
pub struct ConstraintMarketplace {
    /// Local registry for installed components
    registry: Arc<RwLock<CustomConstraintRegistry>>,
    /// Marketplace index
    index: Arc<RwLock<MarketplaceIndex>>,
    /// Configuration
    config: MarketplaceConfig,
    /// User session (if authenticated)
    session: Option<UserSession>,
    /// Local cache
    cache: Arc<RwLock<MarketplaceCache>>,
}

/// Configuration for the marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceConfig {
    /// Marketplace server URL
    pub server_url: String,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    /// Auto-update installed components
    pub auto_update: bool,
    /// Verify component signatures
    pub verify_signatures: bool,
    /// Allowed component sources
    pub allowed_sources: Vec<String>,
    /// Blocked components
    pub blocked_components: HashSet<String>,
    /// Proxy settings
    pub proxy: Option<String>,
    /// Connection timeout in seconds
    pub timeout: u64,
}

impl Default for MarketplaceConfig {
    fn default() -> Self {
        Self {
            server_url: "https://marketplace.oxirs.org/api".to_string(),
            enable_cache: true,
            cache_ttl: 3600, // 1 hour
            auto_update: false,
            verify_signatures: true,
            allowed_sources: vec!["https://marketplace.oxirs.org".to_string()],
            blocked_components: HashSet::new(),
            proxy: None,
            timeout: 30,
        }
    }
}

/// Marketplace index containing all available components
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MarketplaceIndex {
    /// Components indexed by ID
    pub components: HashMap<String, MarketplaceComponent>,
    /// Categories
    pub categories: Vec<Category>,
    /// Featured components
    pub featured: Vec<String>,
    /// Last updated timestamp
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
    /// Index version
    pub version: String,
}

/// Component listing in the marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceComponent {
    /// Component ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Author
    pub author: Author,
    /// Available versions
    pub versions: Vec<ComponentVersion>,
    /// Latest version
    pub latest_version: String,
    /// Categories/tags
    pub categories: Vec<String>,
    /// License
    pub license: String,
    /// Repository URL
    pub repository: Option<String>,
    /// Documentation URL
    pub documentation: Option<String>,
    /// Keywords for search
    pub keywords: Vec<String>,
    /// Statistics
    pub stats: ComponentStats,
    /// Reviews
    pub reviews: Vec<Review>,
    /// Dependencies
    pub dependencies: Vec<Dependency>,
    /// Creation date
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update date
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Whether component is verified
    pub verified: bool,
    /// Whether component is deprecated
    pub deprecated: bool,
    /// Deprecation message
    pub deprecation_message: Option<String>,
}

/// Component version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentVersion {
    /// Version string (semver)
    pub version: String,
    /// Release date
    pub released_at: chrono::DateTime<chrono::Utc>,
    /// Changelog
    pub changelog: String,
    /// Download URL
    pub download_url: String,
    /// SHA256 checksum
    pub checksum: String,
    /// Size in bytes
    pub size: u64,
    /// Minimum OxiRS version required
    pub min_oxirs_version: String,
    /// Whether this version is yanked
    pub yanked: bool,
    /// Rust edition required
    pub rust_edition: String,
}

/// Component author information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    /// Author name
    pub name: String,
    /// Email
    pub email: Option<String>,
    /// Website
    pub website: Option<String>,
    /// Organization
    pub organization: Option<String>,
    /// Verified author
    pub verified: bool,
}

/// Component statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComponentStats {
    /// Total downloads
    pub downloads: u64,
    /// Recent downloads (last 30 days)
    pub recent_downloads: u64,
    /// Average rating (0-5)
    pub rating: f64,
    /// Number of ratings
    pub rating_count: u64,
    /// Number of reviews
    pub review_count: u64,
    /// Number of stars/favorites
    pub stars: u64,
    /// Number of forks/derivatives
    pub forks: u64,
    /// Number of issues
    pub open_issues: u64,
}

/// User review
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Review {
    /// Review ID
    pub id: String,
    /// Author username
    pub author: String,
    /// Rating (1-5)
    pub rating: u8,
    /// Review title
    pub title: String,
    /// Review body
    pub body: String,
    /// Helpful votes
    pub helpful_count: u64,
    /// Created date
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Version reviewed
    pub version: String,
}

/// Component dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Dependency component ID
    pub id: String,
    /// Version requirement
    pub version_req: String,
    /// Whether dependency is optional
    pub optional: bool,
}

/// Component category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    /// Category ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Icon
    pub icon: Option<String>,
    /// Number of components
    pub component_count: usize,
    /// Subcategories
    pub subcategories: Vec<String>,
}

/// User session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    /// User ID
    pub user_id: String,
    /// Username
    pub username: String,
    /// Authentication token
    pub token: String,
    /// Token expiry
    pub expires_at: chrono::DateTime<chrono::Utc>,
    /// User permissions
    pub permissions: HashSet<Permission>,
}

/// User permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Can download components
    Download,
    /// Can publish components
    Publish,
    /// Can write reviews
    Review,
    /// Can moderate content
    Moderate,
    /// Admin access
    Admin,
}

/// Marketplace cache
#[derive(Debug, Default)]
pub struct MarketplaceCache {
    /// Cached index
    index: Option<CachedItem<MarketplaceIndex>>,
    /// Cached component details
    components: HashMap<String, CachedItem<MarketplaceComponent>>,
    /// Cached search results
    search_results: HashMap<String, CachedItem<Vec<String>>>,
}

/// Cached item with expiry
#[derive(Debug, Clone)]
pub struct CachedItem<T> {
    /// Cached value
    pub value: T,
    /// Cache timestamp
    pub cached_at: chrono::DateTime<chrono::Utc>,
}

/// Search query for the marketplace
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Search text
    pub text: Option<String>,
    /// Filter by categories
    pub categories: Vec<String>,
    /// Filter by author
    pub author: Option<String>,
    /// Filter by keywords
    pub keywords: Vec<String>,
    /// Minimum rating
    pub min_rating: Option<f64>,
    /// Only verified components
    pub verified_only: bool,
    /// Exclude deprecated
    pub exclude_deprecated: bool,
    /// Sort by
    pub sort_by: SortBy,
    /// Sort order
    pub sort_order: SortOrder,
    /// Page number
    pub page: usize,
    /// Results per page
    pub per_page: usize,
}

/// Sort options
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum SortBy {
    /// Sort by relevance
    #[default]
    Relevance,
    /// Sort by downloads
    Downloads,
    /// Sort by rating
    Rating,
    /// Sort by name
    Name,
    /// Sort by date
    UpdatedAt,
    /// Sort by created date
    CreatedAt,
}

/// Sort order
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum SortOrder {
    /// Ascending
    Ascending,
    /// Descending
    #[default]
    Descending,
}

/// Search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// Component IDs
    pub components: Vec<String>,
    /// Total count
    pub total: usize,
    /// Current page
    pub page: usize,
    /// Results per page
    pub per_page: usize,
    /// Total pages
    pub total_pages: usize,
    /// Search time in ms
    pub search_time_ms: u64,
}

/// Installation options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstallOptions {
    /// Specific version to install
    pub version: Option<String>,
    /// Install development dependencies
    pub dev: bool,
    /// Force reinstall
    pub force: bool,
    /// Skip dependency resolution
    pub no_deps: bool,
    /// Verify checksums
    pub verify: bool,
}

/// Installation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallResult {
    /// Installed component ID
    pub component_id: String,
    /// Installed version
    pub version: String,
    /// Dependencies installed
    pub dependencies: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
}

/// Publish options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PublishOptions {
    /// Publish as prerelease
    pub prerelease: bool,
    /// Access level
    pub access: AccessLevel,
    /// Custom repository
    pub repository: Option<String>,
}

/// Access level for published components
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Public access
    #[default]
    Public,
    /// Restricted access
    Restricted,
    /// Private access
    Private,
}

/// Publish result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishResult {
    /// Component ID
    pub component_id: String,
    /// Published version
    pub version: String,
    /// Publish URL
    pub url: String,
    /// Warnings
    pub warnings: Vec<String>,
}

impl ConstraintMarketplace {
    /// Create a new marketplace instance
    pub fn new(registry: Arc<RwLock<CustomConstraintRegistry>>) -> Self {
        Self {
            registry,
            index: Arc::new(RwLock::new(MarketplaceIndex::default())),
            config: MarketplaceConfig::default(),
            session: None,
            cache: Arc::new(RwLock::new(MarketplaceCache::default())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        registry: Arc<RwLock<CustomConstraintRegistry>>,
        config: MarketplaceConfig,
    ) -> Self {
        Self {
            registry,
            index: Arc::new(RwLock::new(MarketplaceIndex::default())),
            config,
            session: None,
            cache: Arc::new(RwLock::new(MarketplaceCache::default())),
        }
    }

    /// Authenticate with the marketplace
    pub fn authenticate(&mut self, username: &str, _password: &str) -> Result<()> {
        // In production, this would make an API call
        let session = UserSession {
            user_id: format!("user_{}", username),
            username: username.to_string(),
            token: "mock_token".to_string(),
            expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
            permissions: [Permission::Download, Permission::Review]
                .into_iter()
                .collect(),
        };

        self.session = Some(session);
        Ok(())
    }

    /// Logout from the marketplace
    pub fn logout(&mut self) {
        self.session = None;
    }

    /// Check if authenticated
    pub fn is_authenticated(&self) -> bool {
        self.session
            .as_ref()
            .map(|s| s.expires_at > chrono::Utc::now())
            .unwrap_or(false)
    }

    /// Refresh the marketplace index
    pub fn refresh_index(&self) -> Result<()> {
        // In production, this would fetch from the server
        let mut index = self.index.write().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire index lock".to_string())
        })?;

        // Mock index update
        index.last_updated = Some(chrono::Utc::now());
        index.version = "1.0.0".to_string();

        // Add sample categories
        index.categories = vec![
            Category {
                id: "identity".to_string(),
                name: "Identity Validators".to_string(),
                description: "Components for validating identifiers".to_string(),
                icon: Some("id-card".to_string()),
                component_count: 5,
                subcategories: vec!["uuid".to_string(), "iri".to_string()],
            },
            Category {
                id: "temporal".to_string(),
                name: "Temporal Validators".to_string(),
                description: "Components for validating dates and times".to_string(),
                icon: Some("clock".to_string()),
                component_count: 4,
                subcategories: vec!["date".to_string(), "duration".to_string()],
            },
            Category {
                id: "geospatial".to_string(),
                name: "Geospatial Validators".to_string(),
                description: "Components for validating geographic data".to_string(),
                icon: Some("globe".to_string()),
                component_count: 4,
                subcategories: vec!["coordinates".to_string(), "regions".to_string()],
            },
        ];

        // Update cache
        if self.config.enable_cache {
            let mut cache = self.cache.write().map_err(|_| {
                ShaclError::ValidationEngine("Failed to acquire cache lock".to_string())
            })?;
            cache.index = Some(CachedItem {
                value: index.clone(),
                cached_at: chrono::Utc::now(),
            });
        }

        Ok(())
    }

    /// Search for components
    pub fn search(&self, query: &SearchQuery) -> Result<SearchResults> {
        let start_time = std::time::Instant::now();
        let index = self.index.read().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire index lock".to_string())
        })?;

        let mut results: Vec<(&String, &MarketplaceComponent)> = index
            .components
            .iter()
            .filter(|(_, comp)| {
                // Text search
                if let Some(text) = &query.text {
                    let text_lower = text.to_lowercase();
                    if !comp.name.to_lowercase().contains(&text_lower)
                        && !comp.description.to_lowercase().contains(&text_lower)
                        && !comp
                            .keywords
                            .iter()
                            .any(|k| k.to_lowercase().contains(&text_lower))
                    {
                        return false;
                    }
                }

                // Category filter
                if !query.categories.is_empty()
                    && !query.categories.iter().any(|c| comp.categories.contains(c))
                {
                    return false;
                }

                // Author filter
                if let Some(author) = &query.author {
                    if comp.author.name.to_lowercase() != author.to_lowercase() {
                        return false;
                    }
                }

                // Rating filter
                if let Some(min_rating) = query.min_rating {
                    if comp.stats.rating < min_rating {
                        return false;
                    }
                }

                // Verified filter
                if query.verified_only && !comp.verified {
                    return false;
                }

                // Deprecated filter
                if query.exclude_deprecated && comp.deprecated {
                    return false;
                }

                true
            })
            .collect();

        // Sort results
        results.sort_by(|a, b| {
            let ord = match query.sort_by {
                SortBy::Relevance => std::cmp::Ordering::Equal, // Keep original order
                SortBy::Downloads => a.1.stats.downloads.cmp(&b.1.stats.downloads),
                SortBy::Rating => {
                    a.1.stats
                        .rating
                        .partial_cmp(&b.1.stats.rating)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                SortBy::Name => a.1.name.cmp(&b.1.name),
                SortBy::UpdatedAt => a.1.updated_at.cmp(&b.1.updated_at),
                SortBy::CreatedAt => a.1.created_at.cmp(&b.1.created_at),
            };

            match query.sort_order {
                SortOrder::Ascending => ord,
                SortOrder::Descending => ord.reverse(),
            }
        });

        let total = results.len();
        let start = query.page * query.per_page;
        let end = (start + query.per_page).min(total);

        let component_ids: Vec<String> = results[start..end]
            .iter()
            .map(|(id, _)| (*id).clone())
            .collect();

        let total_pages = (total + query.per_page - 1) / query.per_page;
        let search_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(SearchResults {
            components: component_ids,
            total,
            page: query.page,
            per_page: query.per_page,
            total_pages,
            search_time_ms,
        })
    }

    /// Get component details
    pub fn get_component(&self, component_id: &str) -> Result<MarketplaceComponent> {
        // Check cache first
        if self.config.enable_cache {
            let cache = self.cache.read().map_err(|_| {
                ShaclError::ValidationEngine("Failed to acquire cache lock".to_string())
            })?;

            if let Some(cached) = cache.components.get(component_id) {
                let age = chrono::Utc::now()
                    .signed_duration_since(cached.cached_at)
                    .num_seconds() as u64;
                if age < self.config.cache_ttl {
                    return Ok(cached.value.clone());
                }
            }
        }

        // Get from index
        let index = self.index.read().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire index lock".to_string())
        })?;

        index.components.get(component_id).cloned().ok_or_else(|| {
            ShaclError::Configuration(format!("Component not found: {}", component_id))
        })
    }

    /// Install a component
    pub fn install(&self, component_id: &str, options: &InstallOptions) -> Result<InstallResult> {
        // Check if blocked
        if self.config.blocked_components.contains(component_id) {
            return Err(ShaclError::ConstraintValidation(format!(
                "Component is blocked: {}",
                component_id
            )));
        }

        // Get component details
        let component = self.get_component(component_id)?;

        // Determine version to install
        let version = options
            .version
            .clone()
            .unwrap_or_else(|| component.latest_version.clone());

        // Find version info
        let version_info = component
            .versions
            .iter()
            .find(|v| v.version == version)
            .ok_or_else(|| ShaclError::Configuration(format!("Version not found: {}", version)))?;

        if version_info.yanked {
            return Err(ShaclError::ConstraintValidation(
                "Version has been yanked".to_string(),
            ));
        }

        // Resolve dependencies
        let mut dependencies = Vec::new();
        if !options.no_deps {
            for dep in &component.dependencies {
                if !dep.optional {
                    dependencies.push(dep.id.clone());
                    // Would recursively install dependencies here
                }
            }
        }

        let mut warnings = Vec::new();

        // Check deprecated
        if component.deprecated {
            if let Some(msg) = &component.deprecation_message {
                warnings.push(format!("Component is deprecated: {}", msg));
            } else {
                warnings.push("Component is deprecated".to_string());
            }
        }

        // In production, would download and install the component
        // For now, just return success

        Ok(InstallResult {
            component_id: component_id.to_string(),
            version,
            dependencies,
            warnings,
        })
    }

    /// Uninstall a component
    pub fn uninstall(&self, component_id: &str) -> Result<()> {
        let _registry = self.registry.write().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire registry lock".to_string())
        })?;

        // TODO: Implement unregister_component in CustomConstraintRegistry
        // For now, return error as unregistration is not yet supported
        Err(ShaclError::Configuration(format!(
            "Uninstalling component '{}' is not yet supported",
            component_id
        )))
    }

    /// List installed components
    pub fn list_installed(&self) -> Result<Vec<String>> {
        let registry = self.registry.read().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire registry lock".to_string())
        })?;

        Ok(registry
            .list_components()
            .iter()
            .map(|id| id.0.clone())
            .collect())
    }

    /// Check for updates
    pub fn check_updates(&self) -> Result<Vec<UpdateInfo>> {
        let installed = self.list_installed()?;
        let mut updates = Vec::new();

        for comp_id in installed {
            if let Ok(component) = self.get_component(&comp_id) {
                // Would compare with installed version
                // For now, just check if there are newer versions
                if component.versions.len() > 1 {
                    updates.push(UpdateInfo {
                        component_id: comp_id,
                        current_version: component
                            .versions
                            .last()
                            .map(|v| v.version.clone())
                            .unwrap_or_default(),
                        latest_version: component.latest_version.clone(),
                        changelog: component
                            .versions
                            .first()
                            .map(|v| v.changelog.clone())
                            .unwrap_or_default(),
                    });
                }
            }
        }

        Ok(updates)
    }

    /// Publish a component to the marketplace
    pub fn publish(
        &self,
        component: &dyn CustomConstraintComponent,
        _options: &PublishOptions,
    ) -> Result<PublishResult> {
        // Check authentication
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| ShaclError::SecurityViolation("Not authenticated".to_string()))?;

        // Check permissions
        if !session.permissions.contains(&Permission::Publish) {
            return Err(ShaclError::SecurityViolation(
                "No permission to publish".to_string(),
            ));
        }

        let metadata = component.metadata();
        let component_id = component.component_id().0.clone();

        // Validate component
        self.validate_for_publish(component)?;

        // In production, would upload to server
        let url = format!("{}/components/{}", self.config.server_url, component_id);

        Ok(PublishResult {
            component_id,
            version: metadata.version.clone().unwrap_or_default(),
            url,
            warnings: Vec::new(),
        })
    }

    /// Validate component for publishing
    fn validate_for_publish(&self, component: &dyn CustomConstraintComponent) -> Result<()> {
        let metadata = component.metadata();

        // Check required fields
        if metadata.name.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "Component name is required".to_string(),
            ));
        }
        if metadata
            .description
            .as_ref()
            .map(|d| d.is_empty())
            .unwrap_or(true)
        {
            return Err(ShaclError::ConstraintValidation(
                "Component description is required".to_string(),
            ));
        }
        if metadata
            .version
            .as_ref()
            .map(|v| v.is_empty())
            .unwrap_or(true)
        {
            return Err(ShaclError::ConstraintValidation(
                "Component version is required".to_string(),
            ));
        }

        Ok(())
    }

    /// Add a review for a component
    pub fn add_review(
        &self,
        component_id: &str,
        rating: u8,
        title: &str,
        body: &str,
    ) -> Result<Review> {
        // Check authentication
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| ShaclError::SecurityViolation("Not authenticated".to_string()))?;

        // Check permissions
        if !session.permissions.contains(&Permission::Review) {
            return Err(ShaclError::SecurityViolation(
                "No permission to review".to_string(),
            ));
        }

        // Validate rating
        if !(1..=5).contains(&rating) {
            return Err(ShaclError::ConstraintValidation(
                "Rating must be between 1 and 5".to_string(),
            ));
        }

        // Verify component exists
        let _ = self.get_component(component_id)?;

        // Create review
        let review = Review {
            id: format!("review_{}", chrono::Utc::now().timestamp()),
            author: session.username.clone(),
            rating,
            title: title.to_string(),
            body: body.to_string(),
            helpful_count: 0,
            created_at: chrono::Utc::now(),
            version: "latest".to_string(),
        };

        // In production, would save to server

        Ok(review)
    }

    /// Get categories
    pub fn get_categories(&self) -> Result<Vec<Category>> {
        let index = self.index.read().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire index lock".to_string())
        })?;

        Ok(index.categories.clone())
    }

    /// Get featured components
    pub fn get_featured(&self) -> Result<Vec<MarketplaceComponent>> {
        let index = self.index.read().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire index lock".to_string())
        })?;

        let mut featured = Vec::new();
        for id in &index.featured {
            if let Some(comp) = index.components.get(id) {
                featured.push(comp.clone());
            }
        }

        Ok(featured)
    }

    /// Get trending components
    pub fn get_trending(&self, limit: usize) -> Result<Vec<MarketplaceComponent>> {
        let index = self.index.read().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire index lock".to_string())
        })?;

        // Sort by recent downloads
        let mut components: Vec<_> = index.components.values().collect();
        components.sort_by(|a, b| b.stats.recent_downloads.cmp(&a.stats.recent_downloads));

        Ok(components.into_iter().take(limit).cloned().collect())
    }

    /// Clear cache
    pub fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire cache lock".to_string())
        })?;

        cache.index = None;
        cache.components.clear();
        cache.search_results.clear();

        Ok(())
    }

    /// Get marketplace statistics
    pub fn get_statistics(&self) -> Result<MarketplaceStatistics> {
        let index = self.index.read().map_err(|_| {
            ShaclError::ValidationEngine("Failed to acquire index lock".to_string())
        })?;

        let total_downloads: u64 = index.components.values().map(|c| c.stats.downloads).sum();
        let verified_count = index.components.values().filter(|c| c.verified).count();
        let authors: HashSet<_> = index.components.values().map(|c| &c.author.name).collect();

        Ok(MarketplaceStatistics {
            total_components: index.components.len(),
            total_downloads,
            total_authors: authors.len(),
            total_categories: index.categories.len(),
            verified_components: verified_count,
            last_updated: index.last_updated,
        })
    }
}

/// Update information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    /// Component ID
    pub component_id: String,
    /// Current installed version
    pub current_version: String,
    /// Latest available version
    pub latest_version: String,
    /// Changelog
    pub changelog: String,
}

/// Marketplace statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceStatistics {
    /// Total number of components
    pub total_components: usize,
    /// Total downloads
    pub total_downloads: u64,
    /// Total authors
    pub total_authors: usize,
    /// Total categories
    pub total_categories: usize,
    /// Verified components
    pub verified_components: usize,
    /// Last update time
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

/// Builder for search queries
#[derive(Debug, Default)]
pub struct SearchQueryBuilder {
    query: SearchQuery,
}

impl SearchQueryBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            query: SearchQuery {
                per_page: 20,
                ..Default::default()
            },
        }
    }

    /// Set search text
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.query.text = Some(text.into());
        self
    }

    /// Add category filter
    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.query.categories.push(category.into());
        self
    }

    /// Set author filter
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.query.author = Some(author.into());
        self
    }

    /// Set minimum rating
    pub fn min_rating(mut self, rating: f64) -> Self {
        self.query.min_rating = Some(rating);
        self
    }

    /// Only verified components
    pub fn verified_only(mut self) -> Self {
        self.query.verified_only = true;
        self
    }

    /// Exclude deprecated
    pub fn exclude_deprecated(mut self) -> Self {
        self.query.exclude_deprecated = true;
        self
    }

    /// Set sort by
    pub fn sort_by(mut self, sort_by: SortBy) -> Self {
        self.query.sort_by = sort_by;
        self
    }

    /// Set page
    pub fn page(mut self, page: usize) -> Self {
        self.query.page = page;
        self
    }

    /// Set results per page
    pub fn per_page(mut self, per_page: usize) -> Self {
        self.query.per_page = per_page;
        self
    }

    /// Build the query
    pub fn build(self) -> SearchQuery {
        self.query
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_marketplace() -> ConstraintMarketplace {
        let registry = Arc::new(RwLock::new(CustomConstraintRegistry::new()));
        ConstraintMarketplace::new(registry)
    }

    #[test]
    fn test_marketplace_creation() {
        let marketplace = create_test_marketplace();
        assert!(!marketplace.is_authenticated());
    }

    #[test]
    fn test_authentication() {
        let mut marketplace = create_test_marketplace();

        marketplace.authenticate("testuser", "password").unwrap();
        assert!(marketplace.is_authenticated());

        marketplace.logout();
        assert!(!marketplace.is_authenticated());
    }

    #[test]
    fn test_refresh_index() {
        let marketplace = create_test_marketplace();
        marketplace.refresh_index().unwrap();

        let index = marketplace.index.read().unwrap();
        assert!(index.last_updated.is_some());
        assert_eq!(index.version, "1.0.0");
    }

    #[test]
    fn test_search_query_builder() {
        let query = SearchQueryBuilder::new()
            .text("uuid")
            .category("identity")
            .min_rating(4.0)
            .verified_only()
            .exclude_deprecated()
            .sort_by(SortBy::Downloads)
            .page(0)
            .per_page(10)
            .build();

        assert_eq!(query.text, Some("uuid".to_string()));
        assert_eq!(query.categories, vec!["identity".to_string()]);
        assert_eq!(query.min_rating, Some(4.0));
        assert!(query.verified_only);
        assert!(query.exclude_deprecated);
        assert_eq!(query.per_page, 10);
    }

    #[test]
    fn test_get_categories() {
        let marketplace = create_test_marketplace();
        marketplace.refresh_index().unwrap();

        let categories = marketplace.get_categories().unwrap();
        assert!(!categories.is_empty());
    }

    #[test]
    fn test_clear_cache() {
        let marketplace = create_test_marketplace();
        marketplace.refresh_index().unwrap();
        marketplace.clear_cache().unwrap();

        let cache = marketplace.cache.read().unwrap();
        assert!(cache.index.is_none());
    }

    #[test]
    fn test_get_statistics() {
        let marketplace = create_test_marketplace();
        marketplace.refresh_index().unwrap();

        let stats = marketplace.get_statistics().unwrap();
        assert!(stats.total_categories > 0);
    }

    #[test]
    fn test_list_installed() {
        let marketplace = create_test_marketplace();
        let installed = marketplace.list_installed().unwrap();
        assert!(installed.is_empty()); // No components installed by default
    }

    #[test]
    fn test_config_default() {
        let config = MarketplaceConfig::default();
        assert!(config.enable_cache);
        assert!(config.verify_signatures);
        assert!(!config.auto_update);
    }
}
