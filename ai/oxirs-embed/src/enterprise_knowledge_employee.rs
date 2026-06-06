//! Employee, skill, organizational, and project types for enterprise knowledge.

use crate::Vector;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Employee embedding with professional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmployeeEmbedding {
    /// Employee unique identifier
    pub employee_id: String,
    /// Employee name
    pub name: String,
    /// Job title
    pub job_title: String,
    /// Department
    pub department: String,
    /// Team
    pub team: String,
    /// Skills
    pub skills: Vec<Skill>,
    /// Experience level
    pub experience_level: ExperienceLevel,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Project history
    pub project_history: Vec<ProjectParticipation>,
    /// Collaboration network
    pub collaborators: Vec<String>,
    /// Employee embedding vector
    pub embedding: Vector,
    /// Career progression predictions
    pub career_predictions: CareerPredictions,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Skill information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Skill name
    pub skill_name: String,
    /// Skill category
    pub category: SkillCategory,
    /// Proficiency level (1-10)
    pub proficiency_level: u8,
    /// Years of experience
    pub years_experience: f64,
    /// Skill importance in role
    pub role_importance: f64,
    /// Market demand score
    pub market_demand: f64,
}

/// Skill categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillCategory {
    Technical,
    Leadership,
    Communication,
    Analytical,
    Creative,
    Domain,
    Language,
    Tools,
}

/// Experience levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperienceLevel {
    Junior,
    Mid,
    Senior,
    Lead,
    Principal,
    Executive,
}

/// Performance metrics for employees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Overall performance score (1-10)
    pub overall_score: f64,
    /// Goal achievement rate
    pub goal_achievement_rate: f64,
    /// Project completion rate
    pub project_completion_rate: f64,
    /// Collaboration score
    pub collaboration_score: f64,
    /// Innovation score
    pub innovation_score: f64,
    /// Leadership score
    pub leadership_score: f64,
}

/// Project participation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectParticipation {
    /// Project ID
    pub project_id: String,
    /// Project name
    pub project_name: String,
    /// Role in project
    pub role: String,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: Option<DateTime<Utc>>,
    /// Project outcome
    pub outcome: ProjectOutcome,
    /// Contribution score
    pub contribution_score: f64,
}

/// Project outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectOutcome {
    Successful,
    PartiallySuccessful,
    Failed,
    Cancelled,
    Ongoing,
}

/// Career progression predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CareerPredictions {
    /// Promotion likelihood (0-1)
    pub promotion_likelihood: f64,
    /// Predicted next role
    pub next_role: String,
    /// Skills to develop
    pub skills_to_develop: Vec<String>,
    /// Career path recommendations
    pub career_paths: Vec<String>,
    /// Retention risk (0-1)
    pub retention_risk: f64,
}

/// Organizational structure
#[derive(Debug, Clone)]
pub struct OrganizationalStructure {
    /// Departments
    pub departments: HashMap<String, Department>,
    /// Teams within departments
    pub teams: HashMap<String, Team>,
    /// Reporting relationships
    pub reporting_structure: HashMap<String, Vec<String>>,
    /// Cross-functional projects
    pub projects: HashMap<String, Project>,
}

/// Department information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Department {
    /// Department ID
    pub department_id: String,
    /// Department name
    pub name: String,
    /// Department head
    pub head: String,
    /// Employees
    pub employees: Vec<String>,
    /// Teams
    pub teams: Vec<String>,
    /// Budget
    pub budget: f64,
    /// Performance metrics
    pub performance: DepartmentPerformance,
}

/// Department performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepartmentPerformance {
    /// Budget utilization
    pub budget_utilization: f64,
    /// Goal achievement
    pub goal_achievement: f64,
    /// Employee satisfaction
    pub employee_satisfaction: f64,
    /// Productivity score
    pub productivity_score: f64,
    /// Innovation index
    pub innovation_index: f64,
}

/// Team information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    /// Team ID
    pub team_id: String,
    /// Team name
    pub name: String,
    /// Team lead
    pub lead: String,
    /// Team members
    pub members: Vec<String>,
    /// Department
    pub department: String,
    /// Team skills
    pub team_skills: Vec<Skill>,
    /// Team performance
    pub performance: TeamPerformance,
}

/// Team performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamPerformance {
    /// Collaboration score
    pub collaboration_score: f64,
    /// Delivery performance
    pub delivery_performance: f64,
    /// Quality metrics
    pub quality_score: f64,
    /// Innovation score
    pub innovation_score: f64,
    /// Team satisfaction
    pub team_satisfaction: f64,
}

/// Project information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Project ID
    pub project_id: String,
    /// Project name
    pub name: String,
    /// Project description
    pub description: String,
    /// Project manager
    pub manager: String,
    /// Team members
    pub team_members: Vec<String>,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: Option<DateTime<Utc>>,
    /// Budget
    pub budget: f64,
    /// Status
    pub status: ProjectStatus,
    /// Required skills
    pub required_skills: Vec<String>,
    /// Performance metrics
    pub performance: ProjectPerformance,
}

/// Project status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectStatus {
    Planning,
    InProgress,
    OnHold,
    Completed,
    Cancelled,
}

/// Project performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectPerformance {
    /// Progress percentage
    pub progress_percentage: f64,
    /// Budget utilization
    pub budget_utilization: f64,
    /// Timeline adherence
    pub timeline_adherence: f64,
    /// Quality score
    pub quality_score: f64,
    /// Stakeholder satisfaction
    pub stakeholder_satisfaction: f64,
}
