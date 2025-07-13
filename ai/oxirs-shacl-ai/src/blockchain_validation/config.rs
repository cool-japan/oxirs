//! Configuration for blockchain validation
//!
//! This module contains configuration structures for blockchain validation,
//! cross-chain operations, and privacy settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::types::{CrossChainAggregation, PrivacyLevel, ValidationMode};

/// Blockchain validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainValidationConfig {
    /// Default validation mode
    pub default_validation_mode: ValidationMode,
    /// Supported blockchain networks
    pub supported_networks: Vec<NetworkConfig>,
    /// Cross-chain configuration
    pub cross_chain_config: CrossChainConfig,
    /// Privacy configuration
    pub privacy_config: PrivacyConfig,
    /// Consensus configuration
    pub consensus_config: ConsensusConfig,
    /// Smart contract configuration
    pub smart_contract_config: SmartContractConfig,
    /// Performance and optimization settings
    pub performance_config: PerformanceConfig,
    /// Security settings
    pub security_config: SecurityConfig,
}

impl Default for BlockchainValidationConfig {
    fn default() -> Self {
        Self {
            default_validation_mode: ValidationMode::Hybrid,
            supported_networks: vec![
                NetworkConfig::ethereum_mainnet(),
                NetworkConfig::polygon_mainnet(),
            ],
            cross_chain_config: CrossChainConfig::default(),
            privacy_config: PrivacyConfig::default(),
            consensus_config: ConsensusConfig::default(),
            smart_contract_config: SmartContractConfig::default(),
            performance_config: PerformanceConfig::default(),
            security_config: SecurityConfig::default(),
        }
    }
}

/// Network-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network identifier
    pub network_id: String,
    /// Network name
    pub name: String,
    /// Chain ID
    pub chain_id: u64,
    /// RPC endpoint URL
    pub rpc_url: String,
    /// WebSocket endpoint URL (optional)
    pub ws_url: Option<String>,
    /// Block explorer URL
    pub explorer_url: String,
    /// Native currency symbol
    pub currency_symbol: String,
    /// Average block time
    pub avg_block_time: Duration,
    /// Transaction confirmation blocks
    pub confirmation_blocks: u32,
    /// Gas price configuration
    pub gas_config: GasConfig,
    /// Network-specific smart contracts
    pub contracts: HashMap<String, String>,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

impl NetworkConfig {
    /// Create Ethereum mainnet configuration
    pub fn ethereum_mainnet() -> Self {
        Self {
            network_id: "ethereum".to_string(),
            name: "Ethereum Mainnet".to_string(),
            chain_id: 1,
            rpc_url: "https://mainnet.infura.io/v3/YOUR_PROJECT_ID".to_string(),
            ws_url: Some("wss://mainnet.infura.io/ws/v3/YOUR_PROJECT_ID".to_string()),
            explorer_url: "https://etherscan.io".to_string(),
            currency_symbol: "ETH".to_string(),
            avg_block_time: Duration::from_secs(15),
            confirmation_blocks: 12,
            gas_config: GasConfig::ethereum_default(),
            contracts: HashMap::new(),
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(60),
            retry_config: RetryConfig::default(),
        }
    }

    /// Create Polygon mainnet configuration
    pub fn polygon_mainnet() -> Self {
        Self {
            network_id: "polygon".to_string(),
            name: "Polygon Mainnet".to_string(),
            chain_id: 137,
            rpc_url: "https://polygon-rpc.com".to_string(),
            ws_url: Some("wss://polygon-rpc.com".to_string()),
            explorer_url: "https://polygonscan.com".to_string(),
            currency_symbol: "MATIC".to_string(),
            avg_block_time: Duration::from_secs(2),
            confirmation_blocks: 20,
            gas_config: GasConfig::polygon_default(),
            contracts: HashMap::new(),
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(60),
            retry_config: RetryConfig::default(),
        }
    }
}

/// Gas configuration for transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasConfig {
    /// Gas limit for validation transactions
    pub gas_limit: u64,
    /// Gas price strategy
    pub gas_price_strategy: GasPriceStrategy,
    /// Maximum gas price (in wei)
    pub max_gas_price: u64,
    /// Priority fee (EIP-1559)
    pub priority_fee: Option<u64>,
    /// Base fee multiplier
    pub base_fee_multiplier: f64,
}

impl GasConfig {
    /// Default Ethereum gas configuration
    pub fn ethereum_default() -> Self {
        Self {
            gas_limit: 500_000,
            gas_price_strategy: GasPriceStrategy::Dynamic,
            max_gas_price: 100_000_000_000,    // 100 gwei
            priority_fee: Some(2_000_000_000), // 2 gwei
            base_fee_multiplier: 1.5,
        }
    }

    /// Default Polygon gas configuration
    pub fn polygon_default() -> Self {
        Self {
            gas_limit: 500_000,
            gas_price_strategy: GasPriceStrategy::Dynamic,
            max_gas_price: 50_000_000_000,     // 50 gwei
            priority_fee: Some(1_000_000_000), // 1 gwei
            base_fee_multiplier: 1.2,
        }
    }
}

/// Gas price strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GasPriceStrategy {
    /// Fixed gas price
    Fixed(u64),
    /// Dynamic gas price based on network conditions
    Dynamic,
    /// Fast transaction processing
    Fast,
    /// Standard transaction processing
    Standard,
    /// Slow/economy transaction processing
    Economy,
}

/// Retry configuration for network operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter enabled
    pub enable_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            enable_jitter: true,
        }
    }
}

/// Cross-chain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainConfig {
    /// Enable cross-chain validation
    pub enable_cross_chain: bool,
    /// Cross-chain aggregation strategy
    pub aggregation_strategy: CrossChainAggregation,
    /// Supported bridge protocols
    pub bridge_protocols: Vec<String>,
    /// Cross-chain timeout
    pub cross_chain_timeout: Duration,
    /// Minimum number of chains for consensus
    pub min_chains_for_consensus: u32,
    /// Chain weight configuration
    pub chain_weights: HashMap<String, f64>,
    /// Enable chain fallback
    pub enable_chain_fallback: bool,
    /// Fallback chain order
    pub fallback_chain_order: Vec<String>,
}

impl Default for CrossChainConfig {
    fn default() -> Self {
        Self {
            enable_cross_chain: true,
            aggregation_strategy: CrossChainAggregation::Majority,
            bridge_protocols: vec![
                "LayerZero".to_string(),
                "Chainlink CCIP".to_string(),
                "Wormhole".to_string(),
            ],
            cross_chain_timeout: Duration::from_secs(300),
            min_chains_for_consensus: 2,
            chain_weights: HashMap::new(),
            enable_chain_fallback: true,
            fallback_chain_order: vec!["ethereum".to_string(), "polygon".to_string()],
        }
    }
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Default privacy level
    pub default_privacy_level: PrivacyLevel,
    /// Available privacy protocols
    pub available_protocols: Vec<String>,
    /// Zero-knowledge proof configuration
    pub zk_config: ZkConfig,
    /// Homomorphic encryption configuration
    pub he_config: HeConfig,
    /// Secure multi-party computation configuration
    pub smpc_config: SmpcConfig,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            default_privacy_level: PrivacyLevel::default(),
            available_protocols: vec![
                "zk-SNARKs".to_string(),
                "zk-STARKs".to_string(),
                "TFHE".to_string(),
                "SPDZ".to_string(),
            ],
            zk_config: ZkConfig::default(),
            he_config: HeConfig::default(),
            smpc_config: SmpcConfig::default(),
        }
    }
}

/// Zero-knowledge proof configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkConfig {
    /// Proof system to use
    pub proof_system: String,
    /// Circuit compilation timeout
    pub circuit_timeout: Duration,
    /// Proof generation timeout
    pub proof_timeout: Duration,
    /// Verification timeout
    pub verification_timeout: Duration,
}

impl Default for ZkConfig {
    fn default() -> Self {
        Self {
            proof_system: "Groth16".to_string(),
            circuit_timeout: Duration::from_secs(300),
            proof_timeout: Duration::from_secs(60),
            verification_timeout: Duration::from_secs(10),
        }
    }
}

/// Homomorphic encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeConfig {
    /// Encryption scheme
    pub scheme: String,
    /// Key size in bits
    pub key_size: u32,
    /// Security level
    pub security_level: u32,
}

impl Default for HeConfig {
    fn default() -> Self {
        Self {
            scheme: "TFHE".to_string(),
            key_size: 2048,
            security_level: 128,
        }
    }
}

/// Secure multi-party computation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmpcConfig {
    /// SMPC protocol
    pub protocol: String,
    /// Number of parties
    pub num_parties: u32,
    /// Threshold for computation
    pub threshold: u32,
}

impl Default for SmpcConfig {
    fn default() -> Self {
        Self {
            protocol: "SPDZ".to_string(),
            num_parties: 3,
            threshold: 2,
        }
    }
}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Consensus algorithm
    pub algorithm: String,
    /// Consensus timeout
    pub timeout: Duration,
    /// Minimum validators required
    pub min_validators: u32,
    /// Consensus threshold (percentage)
    pub threshold_percent: f64,
    /// Enable Byzantine fault tolerance
    pub enable_bft: bool,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            algorithm: "PBFT".to_string(),
            timeout: Duration::from_secs(60),
            min_validators: 3,
            threshold_percent: 66.7,
            enable_bft: true,
        }
    }
}

/// Smart contract configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartContractConfig {
    /// Contract deployment configuration
    pub deployment_config: DeploymentConfig,
    /// Contract interaction configuration
    pub interaction_config: InteractionConfig,
    /// Contract upgrade configuration
    pub upgrade_config: UpgradeConfig,
}

impl Default for SmartContractConfig {
    fn default() -> Self {
        Self {
            deployment_config: DeploymentConfig::default(),
            interaction_config: InteractionConfig::default(),
            upgrade_config: UpgradeConfig::default(),
        }
    }
}

/// Contract deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Auto-deploy contracts
    pub auto_deploy: bool,
    /// Contract compiler version
    pub compiler_version: String,
    /// Optimization enabled
    pub optimization_enabled: bool,
    /// Optimization runs
    pub optimization_runs: u32,
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            auto_deploy: false,
            compiler_version: "0.8.19".to_string(),
            optimization_enabled: true,
            optimization_runs: 200,
        }
    }
}

/// Contract interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionConfig {
    /// Call timeout
    pub call_timeout: Duration,
    /// Transaction timeout
    pub transaction_timeout: Duration,
    /// Enable function caching
    pub enable_caching: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            call_timeout: Duration::from_secs(30),
            transaction_timeout: Duration::from_secs(300),
            enable_caching: true,
            cache_ttl: Duration::from_secs(3600),
        }
    }
}

/// Contract upgrade configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeConfig {
    /// Enable upgradeable contracts
    pub enable_upgrades: bool,
    /// Upgrade proxy pattern
    pub proxy_pattern: String,
    /// Upgrade timelock
    pub timelock_duration: Duration,
}

impl Default for UpgradeConfig {
    fn default() -> Self {
        Self {
            enable_upgrades: false,
            proxy_pattern: "Transparent".to_string(),
            timelock_duration: Duration::from_secs(86400 * 7), // 7 days
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum concurrent validations
    pub max_concurrent_validations: u32,
    /// Request queue size
    pub request_queue_size: u32,
    /// Result cache size
    pub result_cache_size: u32,
    /// Result cache TTL
    pub result_cache_ttl: Duration,
    /// Enable request batching
    pub enable_batching: bool,
    /// Batch size
    pub batch_size: u32,
    /// Batch timeout
    pub batch_timeout: Duration,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_validations: 10,
            request_queue_size: 1000,
            result_cache_size: 10000,
            result_cache_ttl: Duration::from_secs(3600),
            enable_batching: true,
            batch_size: 10,
            batch_timeout: Duration::from_secs(5),
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable API key authentication
    pub enable_api_key_auth: bool,
    /// Enable signature verification
    pub enable_signature_verification: bool,
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Rate limit (requests per minute)
    pub rate_limit_rpm: u32,
    /// Enable IP filtering
    pub enable_ip_filtering: bool,
    /// Allowed IP addresses
    pub allowed_ips: Vec<String>,
    /// Enable encryption in transit
    pub enable_encryption: bool,
    /// TLS version
    pub tls_version: String,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_api_key_auth: true,
            enable_signature_verification: true,
            enable_rate_limiting: true,
            rate_limit_rpm: 100,
            enable_ip_filtering: false,
            allowed_ips: Vec::new(),
            enable_encryption: true,
            tls_version: "1.3".to_string(),
        }
    }
}
