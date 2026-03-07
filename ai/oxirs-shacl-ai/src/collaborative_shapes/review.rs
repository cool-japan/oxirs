//! Shape review system for collaborative development
//!
//! This module implements the shape review system with workflows, reviewer management,
//! scheduling, and quality metrics.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use super::config::ComplexityLevel;
use crate::shape_management::{UserRole, UserPermissions};

/// Shape review system
#[derive(Debug)]
pub struct ShapeReviewSystem {
    review_workflows: HashMap<String, ReviewWorkflow>,
    reviewer_pool: ReviewerPool,
    review_scheduler: ReviewScheduler,
    quality_metrics: ReviewQualityMetrics,
}

/// Review workflow
#[derive(Debug, Clone)]
pub struct ReviewWorkflow {
    pub workflow_id: String,
    pub name: String,
    pub stages: Vec<ReviewStage>,
    pub approval_criteria: ApprovalCriteria,
    pub escalation_rules: Vec<EscalationRule>,
}

/// Review stage
#[derive(Debug, Clone)]
pub struct ReviewStage {
    pub stage_id: String,
    pub name: String,
    pub required_reviewers: usize,
    pub reviewer_criteria: ReviewerCriteria,
    pub timeout: Duration,
    pub parallel_reviews: bool,
}

/// Reviewer criteria
#[derive(Debug, Clone)]
pub struct ReviewerCriteria {
    pub min_reputation: f64,
    pub required_expertise: Vec<String>,
    pub experience_level: ExperienceLevel,
    pub availability_required: bool,
}

/// Experience levels
#[derive(Debug, Clone)]
pub enum ExperienceLevel {
    Junior,
    Mid,
    Senior,
    Expert,
}

/// Approval criteria
#[derive(Debug, Clone)]
pub struct ApprovalCriteria {
    pub min_approvals: usize,
    pub unanimous_required: bool,
    pub blocking_rejections: usize,
    pub quality_threshold: f64,
}

/// Escalation rule
#[derive(Debug, Clone)]
pub struct EscalationRule {
    pub rule_id: String,
    pub trigger_condition: EscalationTrigger,
    pub escalation_action: EscalationAction,
    pub delay: Duration,
}

/// Escalation triggers
#[derive(Debug, Clone)]
pub enum EscalationTrigger {
    Timeout,
    InsufficientReviewers,
    ConflictingReviews,
    QualityBelowThreshold,
}

/// Escalation actions
#[derive(Debug, Clone)]
pub enum EscalationAction {
    AssignMoreReviewers,
    NotifyManager,
    AutoApprove,
    RequireExpertReview,
}

/// Reviewer pool
#[derive(Debug)]
pub struct ReviewerPool {
    available_reviewers: HashMap<String, ReviewerProfile>,
    assignment_algorithm: AssignmentAlgorithm,
    workload_balancer: WorkloadBalancer,
}

/// Reviewer profile
#[derive(Debug, Clone)]
pub struct ReviewerProfile {
    pub user_id: String,
    pub expertise_areas: Vec<String>,
    pub availability: ReviewerAvailability,
    pub review_statistics: ReviewerStatistics,
    pub preferences: ReviewerPreferences,
}

/// Reviewer availability
#[derive(Debug, Clone)]
pub struct ReviewerAvailability {
    pub is_available: bool,
    pub max_concurrent_reviews: usize,
    pub current_reviews: usize,
    pub preferred_time_slots: Vec<TimeSlot>,
}

/// Time slot
#[derive(Debug, Clone)]
pub struct TimeSlot {
    pub start_time: chrono::NaiveTime,
    pub end_time: chrono::NaiveTime,
    pub days_of_week: Vec<chrono::Weekday>,
    pub timezone: String,
}

/// Reviewer statistics
#[derive(Debug, Clone)]
pub struct ReviewerStatistics {
    pub total_reviews: usize,
    pub average_review_time: Duration,
    pub approval_rate: f64,
    pub quality_score: f64,
    pub consistency_score: f64,
}

/// Reviewer preferences
#[derive(Debug, Clone)]
pub struct ReviewerPreferences {
    pub preferred_domains: Vec<String>,
    pub complexity_preference: ComplexityLevel,
    pub notification_preferences: NotificationPreferences,
}

/// Notification preferences
#[derive(Debug, Clone)]
pub struct NotificationPreferences {
    pub email_notifications: bool,
    pub push_notifications: bool,
    pub review_reminders: bool,
    pub escalation_notifications: bool,
}

/// Assignment algorithm
#[derive(Debug)]
pub enum AssignmentAlgorithm {
    RoundRobin,
    ExpertiseBased,
    WorkloadBased,
    ReputationBased,
    Hybrid,
}

/// Workload balancer
#[derive(Debug)]
pub struct WorkloadBalancer {
    user_workloads: HashMap<String, WorkloadInfo>,
    balancing_strategy: BalancingStrategy,
}

/// Workload information
#[derive(Debug, Clone)]
pub struct WorkloadInfo {
    pub user_id: String,
    pub current_reviews: usize,
    pub pending_reviews: usize,
    pub average_completion_time: Duration,
    pub capacity_utilization: f64,
}

/// Balancing strategy
#[derive(Debug, Clone)]
pub enum BalancingStrategy {
    EqualDistribution,
    CapacityBased,
    SkillBased,
    PriorityBased,
}

/// Review scheduler
#[derive(Debug)]
pub struct ReviewScheduler {
    scheduled_reviews: VecDeque<ScheduledReview>,
    priority_queue: VecDeque<PriorityReview>,
    scheduling_algorithm: SchedulingAlgorithm,
}

/// Scheduled review
#[derive(Debug, Clone)]
pub struct ScheduledReview {
    pub review_id: String,
    pub shape_id: String,
    pub scheduled_time: chrono::DateTime<chrono::Utc>,
    pub assigned_reviewers: Vec<String>,
    pub priority: ReviewPriority,
}

/// Priority review
#[derive(Debug, Clone)]
pub struct PriorityReview {
    pub review_id: String,
    pub priority_level: ReviewPriority,
    pub deadline: chrono::DateTime<chrono::Utc>,
    pub escalation_count: usize,
}

/// Review priority levels
#[derive(Debug, Clone)]
pub enum ReviewPriority {
    Low,
    Normal,
    High,
    Urgent,
    Critical,
}

/// Scheduling algorithm
#[derive(Debug, Clone)]
pub enum SchedulingAlgorithm {
    FIFO,           // First In, First Out
    Priority,       // Priority-based
    SJF,           // Shortest Job First
    RoundRobin,    // Round-robin
    Adaptive,      // Adaptive based on workload
}

/// Review quality metrics
#[derive(Debug)]
pub struct ReviewQualityMetrics {
    pub average_review_time: Duration,
    pub review_consistency: f64,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
    pub reviewer_agreement: f64,
    pub review_coverage: f64,
}

/// Review event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewEvent {
    pub event_id: String,
    pub review_id: String,
    pub reviewer_id: String,
    pub event_type: ReviewEventType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// Review event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewEventType {
    ReviewStarted,
    ReviewCompleted,
    ReviewerAssigned,
    ReviewerChanged,
    CommentAdded,
    ApprovalGiven,
    RejectionGiven,
    EscalationTriggered,
}

/// Review comment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewComment {
    pub comment_id: String,
    pub review_id: String,
    pub reviewer_id: String,
    pub comment_type: CommentType,
    pub content: String,
    pub line_number: Option<usize>,
    pub severity: CommentSeverity,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub resolved: bool,
}

/// Comment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentType {
    General,
    Suggestion,
    Issue,
    Question,
    Approval,
}

/// Comment severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Review decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewDecision {
    Approved,
    RequestChanges,
    Rejected,
    NeedsMoreReview,
}

/// Review types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewType {
    PeerReview,
    ExpertReview,
    AutomatedReview,
    SecurityReview,
}

impl ShapeReviewSystem {
    /// Create new shape review system
    pub fn new() -> Self {
        Self {
            review_workflows: HashMap::new(),
            reviewer_pool: ReviewerPool::new(),
            review_scheduler: ReviewScheduler::new(),
            quality_metrics: ReviewQualityMetrics {
                average_review_time: Duration::from_secs(7200), // 2 hours
                review_consistency: 0.85,
                false_positive_rate: 0.05,
                false_negative_rate: 0.03,
                reviewer_agreement: 0.9,
                review_coverage: 0.95,
            },
        }
    }

    /// Initialize the review system
    pub async fn initialize(&mut self) -> crate::Result<()> {
        // Setup default workflows and policies
        self.setup_default_workflows().await?;
        self.reviewer_pool.initialize().await?;
        self.review_scheduler.initialize().await?;
        
        Ok(())
    }

    /// Setup default review workflows
    async fn setup_default_workflows(&mut self) -> crate::Result<()> {
        // Create standard peer review workflow
        let peer_review_workflow = ReviewWorkflow {
            workflow_id: "peer_review".to_string(),
            name: "Standard Peer Review".to_string(),
            stages: vec![
                ReviewStage {
                    stage_id: "initial_review".to_string(),
                    name: "Initial Review".to_string(),
                    required_reviewers: 2,
                    reviewer_criteria: ReviewerCriteria {
                        min_reputation: 100.0,
                        required_expertise: vec!["SHACL".to_string()],
                        experience_level: ExperienceLevel::Mid,
                        availability_required: true,
                    },
                    timeout: Duration::from_secs(86400 * 2), // 2 days
                    parallel_reviews: true,
                }
            ],
            approval_criteria: ApprovalCriteria {
                min_approvals: 2,
                unanimous_required: false,
                blocking_rejections: 1,
                quality_threshold: 0.8,
            },
            escalation_rules: vec![
                EscalationRule {
                    rule_id: "timeout_escalation".to_string(),
                    trigger_condition: EscalationTrigger::Timeout,
                    escalation_action: EscalationAction::AssignMoreReviewers,
                    delay: Duration::from_secs(86400), // 1 day
                }
            ],
        };

        self.review_workflows.insert(
            "peer_review".to_string(),
            peer_review_workflow
        );

        Ok(())
    }

    /// Submit shape for review
    pub async fn submit_for_review(
        &mut self,
        workspace_id: String,
        shape_id: String,
        user_id: String,
        review_type: ReviewType,
    ) -> crate::Result<String> {
        let review_id = format!("review_{}_{}", shape_id, chrono::Utc::now().timestamp());
        
        // Schedule review
        let scheduled_review = ScheduledReview {
            review_id: review_id.clone(),
            shape_id,
            scheduled_time: chrono::Utc::now(),
            assigned_reviewers: Vec::new(), // Would be assigned by reviewer pool
            priority: ReviewPriority::Normal,
        };
        
        self.review_scheduler.schedule_review(scheduled_review).await?;
        
        Ok(review_id)
    }

    /// Assign reviewers to a review
    pub async fn assign_reviewers(
        &mut self,
        review_id: String,
        required_expertise: Vec<String>,
    ) -> crate::Result<Vec<String>> {
        self.reviewer_pool.assign_reviewers(review_id, required_expertise).await
    }

    /// Get review quality metrics
    pub fn get_quality_metrics(&self) -> &ReviewQualityMetrics {
        &self.quality_metrics
    }

    /// Update review quality metrics
    pub fn update_quality_metrics(&mut self, metrics: ReviewQualityMetrics) {
        self.quality_metrics = metrics;
    }
}

impl ReviewerPool {
    /// Create new reviewer pool
    pub fn new() -> Self {
        Self {
            available_reviewers: HashMap::new(),
            assignment_algorithm: AssignmentAlgorithm::ExpertiseBased,
            workload_balancer: WorkloadBalancer::new(),
        }
    }

    /// Initialize reviewer pool
    pub async fn initialize(&mut self) -> crate::Result<()> {
        // Load reviewer profiles and setup balancing
        Ok(())
    }

    /// Assign reviewers to a review
    pub async fn assign_reviewers(
        &mut self,
        review_id: String,
        required_expertise: Vec<String>,
    ) -> crate::Result<Vec<String>> {
        // Implementation would select best reviewers based on:
        // - Expertise match
        // - Availability
        // - Workload balance
        // - Reputation scores
        
        let mut assigned_reviewers = Vec::new();
        
        for (user_id, profile) in &self.available_reviewers {
            if profile.availability.is_available 
                && profile.availability.current_reviews < profile.availability.max_concurrent_reviews
                && required_expertise.iter().any(|e| profile.expertise_areas.contains(e)) {
                assigned_reviewers.push(user_id.clone());
                if assigned_reviewers.len() >= 2 {
                    break;
                }
            }
        }
        
        Ok(assigned_reviewers)
    }

    /// Add reviewer to pool
    pub fn add_reviewer(&mut self, profile: ReviewerProfile) {
        self.available_reviewers.insert(profile.user_id.clone(), profile);
    }

    /// Remove reviewer from pool
    pub fn remove_reviewer(&mut self, user_id: &str) {
        self.available_reviewers.remove(user_id);
    }

    /// Get reviewer profile
    pub fn get_reviewer(&self, user_id: &str) -> Option<&ReviewerProfile> {
        self.available_reviewers.get(user_id)
    }
}

impl WorkloadBalancer {
    /// Create new workload balancer
    pub fn new() -> Self {
        Self {
            user_workloads: HashMap::new(),
            balancing_strategy: BalancingStrategy::CapacityBased,
        }
    }

    /// Update user workload
    pub fn update_workload(&mut self, user_id: String, workload: WorkloadInfo) {
        self.user_workloads.insert(user_id, workload);
    }

    /// Get user workload
    pub fn get_workload(&self, user_id: &str) -> Option<&WorkloadInfo> {
        self.user_workloads.get(user_id)
    }

    /// Balance workload across reviewers
    pub fn balance_workload(&self) -> Vec<String> {
        match self.balancing_strategy {
            BalancingStrategy::EqualDistribution => {
                // Return users with lowest current workload
                let mut users: Vec<_> = self.user_workloads.iter().collect();
                users.sort_by(|a, b| a.1.current_reviews.cmp(&b.1.current_reviews));
                users.into_iter().map(|(id, _)| id.clone()).collect()
            }
            BalancingStrategy::CapacityBased => {
                // Return users with lowest capacity utilization
                let mut users: Vec<_> = self.user_workloads.iter().collect();
                users.sort_by(|a, b| a.1.capacity_utilization.partial_cmp(&b.1.capacity_utilization).unwrap_or(std::cmp::Ordering::Equal));
                users.into_iter().map(|(id, _)| id.clone()).collect()
            }
            _ => Vec::new(),
        }
    }
}

impl ReviewScheduler {
    /// Create new review scheduler
    pub fn new() -> Self {
        Self {
            scheduled_reviews: VecDeque::new(),
            priority_queue: VecDeque::new(),
            scheduling_algorithm: SchedulingAlgorithm::Priority,
        }
    }

    /// Initialize review scheduler
    pub async fn initialize(&mut self) -> crate::Result<()> {
        // Setup scheduling policies
        Ok(())
    }

    /// Schedule a review
    pub async fn schedule_review(&mut self, review: ScheduledReview) -> crate::Result<()> {
        match review.priority {
            ReviewPriority::Critical | ReviewPriority::Urgent => {
                self.priority_queue.push_front(PriorityReview {
                    review_id: review.review_id.clone(),
                    priority_level: review.priority.clone(),
                    deadline: review.scheduled_time + chrono::Duration::hours(24),
                    escalation_count: 0,
                });
            }
            _ => {}
        }
        
        self.scheduled_reviews.push_back(review);
        Ok(())
    }

    /// Get next review to process
    pub fn get_next_review(&mut self) -> Option<ScheduledReview> {
        // Priority queue takes precedence
        if let Some(priority_review) = self.priority_queue.pop_front() {
            // Find corresponding scheduled review
            for (i, review) in self.scheduled_reviews.iter().enumerate() {
                if review.review_id == priority_review.review_id {
                    return Some(self.scheduled_reviews.remove(i).expect("index should be valid for VecDeque"));
                }
            }
        }
        
        // Otherwise return next scheduled review
        self.scheduled_reviews.pop_front()
    }

    /// Update scheduling algorithm
    pub fn set_scheduling_algorithm(&mut self, algorithm: SchedulingAlgorithm) {
        self.scheduling_algorithm = algorithm;
    }
}

impl Default for ShapeReviewSystem {
    fn default() -> Self {
        Self::new()
    }
}