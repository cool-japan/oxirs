use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub url: String,
    pub name: String,
    pub supported_features: HashSet<String>,
    pub authentication: Option<ServiceAuthentication>,
    pub timeout: Duration,
    pub retry_count: u32,
    pub health_status: ServiceHealth,
    pub response_time_avg: Option<Duration>,
    pub last_checked: Option<std::time::SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceAuthentication {
    pub auth_type: AuthenticationType,
    pub credentials: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationType {
    None,
    Basic,
    Bearer,
    ApiKey,
    OAuth2,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ServiceHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
    Maintenance,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    Union,
    Intersection,
    LeftJoin,
    RightJoin,
    FullJoin,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum DiscoveryMethod {
    Static,
    Dns,
    Consul,
    Kubernetes,
    VoID,
    SPARQL,
}

#[derive(Debug, Clone)]
pub struct ServiceQueryRequest {
    pub service_url: String,
    pub query: String,
    pub parameters: HashMap<String, String>,
    pub timeout: Option<Duration>,
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceQueryResponse {
    pub status: ResponseStatus,
    pub results: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub execution_time: Duration,
    pub endpoint_info: ServiceEndpointInfo,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    Timeout,
    Error,
    Retry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpointInfo {
    pub url: String,
    pub response_time: Duration,
    pub attempt_count: u32,
}

#[derive(Debug)]
pub struct QueryCache {
    pub(crate) cache: HashMap<String, CacheEntry>,
    pub(crate) max_size: usize,
    pub(crate) default_ttl: Duration,
}

#[derive(Debug, Clone)]
pub(crate) struct CacheEntry {
    pub result: serde_json::Value,
    pub expires_at: Instant,
    pub access_count: u64,
    pub last_accessed: Instant,
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub max_size: usize,
    pub total_accesses: u64,
    pub hit_ratio: f64,
}

impl QueryCache {
    pub fn new(max_size: usize, default_ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            default_ttl,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<serde_json::Value> {
        if let Some(entry) = self.cache.get_mut(key) {
            if entry.expires_at > Instant::now() {
                entry.access_count += 1;
                entry.last_accessed = Instant::now();
                return Some(entry.result.clone());
            } else {
                self.cache.remove(key);
            }
        }
        None
    }

    pub fn put(&mut self, key: String, result: serde_json::Value, ttl: Option<Duration>) {
        let ttl = ttl.unwrap_or(self.default_ttl);
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }
        let entry = CacheEntry {
            result,
            expires_at: Instant::now() + ttl,
            access_count: 1,
            last_accessed: Instant::now(),
        };
        self.cache.insert(key, entry);
    }

    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone())
        {
            self.cache.remove(&lru_key);
        }
    }

    pub fn cleanup_expired(&mut self) {
        let now = Instant::now();
        self.cache.retain(|_, entry| entry.expires_at > now);
    }

    pub fn stats(&self) -> CacheStats {
        let total_entries = self.cache.len();
        let total_accesses: u64 = self.cache.values().map(|e| e.access_count).sum();
        CacheStats {
            total_entries,
            max_size: self.max_size,
            total_accesses,
            hit_ratio: if total_accesses > 0 {
                total_accesses as f64 / (total_accesses + total_entries as u64) as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct EndpointStats {
    pub url: String,
    pub weight: f64,
    pub health_score: f64,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    ResponseTime,
    HealthScore,
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct LoadBalancer {
    pub(crate) strategy: LoadBalancingStrategy,
    pub(crate) endpoint_weights: HashMap<String, f64>,
    pub(crate) endpoint_health_scores: HashMap<String, f64>,
    pub(crate) round_robin_index: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            endpoint_weights: HashMap::new(),
            endpoint_health_scores: HashMap::new(),
            round_robin_index: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn select_endpoint<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        if endpoints.is_empty() {
            return None;
        }
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(endpoints),
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(endpoints)
            }
            LoadBalancingStrategy::ResponseTime => self.select_by_response_time(endpoints),
            LoadBalancingStrategy::HealthScore => self.select_by_health_score(endpoints),
            LoadBalancingStrategy::Adaptive => self.select_adaptive(endpoints),
        }
    }

    fn select_round_robin<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        let index = self
            .round_robin_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        endpoints.get(index % endpoints.len())
    }

    fn select_weighted_round_robin<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        use scirs2_core::random::{Random, RngExt};
        let mut total_weight = 0.0;
        for endpoint in endpoints {
            total_weight += self.endpoint_weights.get(&endpoint.url).unwrap_or(&1.0);
        }
        if total_weight == 0.0 {
            return self.select_round_robin(endpoints);
        }
        let mut rng = Random::seed(42);
        let mut random_weight = rng.random::<f64>() * total_weight;
        for endpoint in endpoints {
            let weight = self.endpoint_weights.get(&endpoint.url).unwrap_or(&1.0);
            random_weight -= weight;
            if random_weight <= 0.0 {
                return Some(endpoint);
            }
        }
        endpoints.first()
    }

    fn select_by_response_time<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        endpoints
            .iter()
            .filter(|e| e.health_status == ServiceHealth::Healthy)
            .min_by_key(|e| e.response_time_avg.unwrap_or(Duration::from_secs(u64::MAX)))
    }

    fn select_by_health_score<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        endpoints
            .iter()
            .filter(|e| e.health_status != ServiceHealth::Unhealthy)
            .max_by(|a, b| {
                let score_a = self.endpoint_health_scores.get(&a.url).unwrap_or(&0.5);
                let score_b = self.endpoint_health_scores.get(&b.url).unwrap_or(&0.5);
                score_a
                    .partial_cmp(score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    fn select_adaptive<'a>(&self, endpoints: &'a [ServiceEndpoint]) -> Option<&'a ServiceEndpoint> {
        let mut best_endpoint = None;
        let mut best_score = f64::NEG_INFINITY;
        for endpoint in endpoints {
            if endpoint.health_status == ServiceHealth::Unhealthy {
                continue;
            }
            let mut score = 0.0;
            let health_score = self
                .endpoint_health_scores
                .get(&endpoint.url)
                .unwrap_or(&0.5);
            score += health_score * 0.4;
            if let Some(response_time) = endpoint.response_time_avg {
                let response_score = 1.0 / (1.0 + response_time.as_millis() as f64 / 1000.0);
                score += response_score * 0.3;
            }
            let weight = self.endpoint_weights.get(&endpoint.url).unwrap_or(&1.0);
            score += weight * 0.2;
            let health_bonus = match endpoint.health_status {
                ServiceHealth::Healthy => 1.0,
                ServiceHealth::Degraded => 0.7,
                ServiceHealth::Maintenance => 0.3,
                ServiceHealth::Unknown => 0.5,
                ServiceHealth::Unhealthy => 0.0,
            };
            score += health_bonus * 0.1;
            if score > best_score {
                best_score = score;
                best_endpoint = Some(endpoint);
            }
        }
        best_endpoint
    }

    pub fn set_endpoint_weight(&mut self, url: String, weight: f64) {
        self.endpoint_weights.insert(url, weight.clamp(0.0, 10.0));
    }

    pub fn set_endpoint_health_score(&mut self, url: String, score: f64) {
        self.endpoint_health_scores
            .insert(url, score.clamp(0.0, 1.0));
    }

    pub fn get_endpoint_stats(&self, url: &str) -> EndpointStats {
        EndpointStats {
            url: url.to_string(),
            weight: self.endpoint_weights.get(url).copied().unwrap_or(1.0),
            health_score: self.endpoint_health_scores.get(url).copied().unwrap_or(0.5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryCacheV2 {
    pub(crate) cache: HashMap<String, CacheEntryV2>,
    pub(crate) max_size: usize,
    pub(crate) default_ttl: Duration,
}

#[derive(Debug, Clone)]
pub struct CacheEntryV2 {
    pub data: serde_json::Value,
    pub created_at: Instant,
    pub ttl: Duration,
    pub access_count: u64,
    pub last_accessed: Instant,
    pub endpoint_url: String,
}

#[derive(Debug, Clone)]
pub struct ServiceCacheStats {
    pub entries: usize,
    pub total_access_count: u64,
    pub avg_age_seconds: u64,
    pub hit_ratio: f64,
}

impl QueryCacheV2 {
    pub fn new(max_size: usize, default_ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            default_ttl,
        }
    }

    pub fn get(&mut self, query_hash: &str) -> Option<serde_json::Value> {
        if let Some(entry) = self.cache.get_mut(query_hash) {
            if entry.created_at.elapsed() < entry.ttl {
                entry.access_count += 1;
                entry.last_accessed = Instant::now();
                Some(entry.data.clone())
            } else {
                self.cache.remove(query_hash);
                None
            }
        } else {
            None
        }
    }

    pub fn set(
        &mut self,
        query_hash: String,
        data: serde_json::Value,
        endpoint_url: String,
        ttl: Option<Duration>,
    ) {
        if self.cache.len() >= self.max_size {
            self.evict_oldest();
        }
        let entry = CacheEntryV2 {
            data,
            created_at: Instant::now(),
            ttl: ttl.unwrap_or(self.default_ttl),
            access_count: 1,
            last_accessed: Instant::now(),
            endpoint_url,
        };
        self.cache.insert(query_hash, entry);
    }

    fn evict_oldest(&mut self) {
        if let Some((oldest_key, _)) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            self.cache.remove(&oldest_key);
        }
    }

    pub fn generate_cache_key(
        query: &str,
        endpoint: &str,
        parameters: &HashMap<String, String>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        endpoint.hash(&mut hasher);
        for (k, v) in parameters {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    pub fn invalidate_endpoint(&mut self, endpoint_url: &str) {
        self.cache
            .retain(|_, entry| entry.endpoint_url != endpoint_url);
    }

    pub fn get_stats(&self) -> ServiceCacheStats {
        let total_access_count = self.cache.values().map(|e| e.access_count).sum();
        let avg_age = if !self.cache.is_empty() {
            self.cache
                .values()
                .map(|e| e.created_at.elapsed().as_secs())
                .sum::<u64>()
                / self.cache.len() as u64
        } else {
            0
        };
        ServiceCacheStats {
            entries: self.cache.len(),
            total_access_count,
            avg_age_seconds: avg_age,
            hit_ratio: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategyV2 {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    ResponseTime,
    HealthScore,
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct LoadBalancerV2 {
    pub(crate) strategy: LoadBalancingStrategyV2,
    pub(crate) endpoint_weights: HashMap<String, f64>,
    pub(crate) endpoint_health_scores: HashMap<String, f64>,
}

impl LoadBalancerV2 {
    pub fn new(strategy: LoadBalancingStrategyV2) -> Self {
        Self {
            strategy,
            endpoint_weights: HashMap::new(),
            endpoint_health_scores: HashMap::new(),
        }
    }

    pub fn select_endpoint(
        &self,
        available_endpoints: &[ServiceEndpoint],
        query_complexity: Option<f64>,
    ) -> Option<ServiceEndpoint> {
        if available_endpoints.is_empty() {
            return None;
        }
        match &self.strategy {
            LoadBalancingStrategyV2::RoundRobin => available_endpoints.first().cloned(),
            LoadBalancingStrategyV2::WeightedRoundRobin => {
                self.weighted_selection(available_endpoints)
            }
            LoadBalancingStrategyV2::ResponseTime => available_endpoints
                .iter()
                .min_by_key(|ep| ep.response_time_avg.unwrap_or(Duration::from_secs(999)))
                .cloned(),
            LoadBalancingStrategyV2::HealthScore => available_endpoints
                .iter()
                .filter(|ep| ep.health_status == ServiceHealth::Healthy)
                .max_by(|a, b| {
                    let score_a = self.endpoint_health_scores.get(&a.url).unwrap_or(&0.5);
                    let score_b = self.endpoint_health_scores.get(&b.url).unwrap_or(&0.5);
                    score_a
                        .partial_cmp(score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned(),
            LoadBalancingStrategyV2::LeastConnections => available_endpoints.first().cloned(),
            LoadBalancingStrategyV2::Adaptive => {
                self.adaptive_selection(available_endpoints, query_complexity)
            }
        }
    }

    fn weighted_selection(&self, endpoints: &[ServiceEndpoint]) -> Option<ServiceEndpoint> {
        use scirs2_core::random::{Random, RngExt};
        let total_weight: f64 = endpoints
            .iter()
            .map(|ep| self.endpoint_weights.get(&ep.url).unwrap_or(&1.0))
            .sum();
        if total_weight <= 0.0 {
            return endpoints.first().cloned();
        }
        let mut rng = Random::seed(42);
        let random_value: f64 = rng.random_range(0.0..total_weight);
        let mut current_weight = 0.0;
        for endpoint in endpoints {
            current_weight += self.endpoint_weights.get(&endpoint.url).unwrap_or(&1.0);
            if random_value <= current_weight {
                return Some(endpoint.clone());
            }
        }
        endpoints.last().cloned()
    }

    fn adaptive_selection(
        &self,
        endpoints: &[ServiceEndpoint],
        query_complexity: Option<f64>,
    ) -> Option<ServiceEndpoint> {
        let mut scored_endpoints: Vec<(f64, ServiceEndpoint)> = endpoints
            .iter()
            .map(|ep| {
                let mut score = 0.0;
                match ep.health_status {
                    ServiceHealth::Healthy => score += 0.4,
                    ServiceHealth::Degraded => score += 0.2,
                    ServiceHealth::Unhealthy => score += 0.0,
                    ServiceHealth::Unknown => score += 0.1,
                    ServiceHealth::Maintenance => score += 0.0,
                }
                if let Some(avg_time) = ep.response_time_avg {
                    let time_score = 1.0 / (1.0 + avg_time.as_secs_f64());
                    score += 0.3 * time_score;
                }
                if let Some(complexity) = query_complexity {
                    let feature_score = if (complexity > 0.7
                        && ep.supported_features.contains("complex_queries"))
                        || (complexity < 0.3 && ep.supported_features.contains("fast_queries"))
                    {
                        1.0
                    } else {
                        0.5
                    };
                    score += 0.2 * feature_score;
                }
                let weight = self.endpoint_weights.get(&ep.url).unwrap_or(&1.0);
                score += 0.1 * weight;
                (score, ep.clone())
            })
            .collect();
        scored_endpoints.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored_endpoints.into_iter().next().map(|(_, ep)| ep)
    }

    pub fn update_endpoint_weight(&mut self, endpoint_url: &str, performance_score: f64) {
        let current_weight = self.endpoint_weights.get(endpoint_url).unwrap_or(&1.0);
        let new_weight = (current_weight * 0.8) + (performance_score * 0.2);
        self.endpoint_weights
            .insert(endpoint_url.to_string(), new_weight.max(0.1).min(2.0));
    }

    pub fn update_health_score(&mut self, endpoint_url: &str, health_score: f64) {
        self.endpoint_health_scores
            .insert(endpoint_url.to_string(), health_score.max(0.0).min(1.0));
    }
}
