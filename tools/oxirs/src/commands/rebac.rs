//! ReBAC Management Commands
//!
//! CLI commands for managing ReBAC relationships and authorization data.

use crate::cli::error::{CliError, CliResult};
use crate::commands::rebac_manager::RebacManager;
use clap::{Args, Subcommand};
use std::path::PathBuf;
use tracing::{info, warn};

/// ReBAC management commands
#[derive(Debug, Args)]
pub struct RebacArgs {
    #[command(subcommand)]
    pub command: RebacCommand,
}

#[derive(Debug, Subcommand)]
pub enum RebacCommand {
    /// Export relationships to file
    Export(ExportArgs),
    /// Import relationships from file
    Import(ImportArgs),
    /// Migrate between backends
    Migrate(MigrateArgs),
    /// Verify data integrity
    Verify(VerifyArgs),
    /// Show statistics
    Stats(StatsArgs),
}

#[derive(Debug, Args)]
pub struct ExportArgs {
    /// Output file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Export format (turtle, json)
    #[arg(short, long, default_value = "turtle")]
    pub format: ExportFormat,

    /// Authorization namespace
    #[arg(long, default_value = "http://oxirs.org/auth#")]
    pub namespace: String,

    /// Named graph URI
    #[arg(long, default_value = "urn:oxirs:auth:relationships")]
    pub graph: String,

    /// Filter by subject (optional)
    #[arg(long)]
    pub subject: Option<String>,

    /// Filter by relation (optional)
    #[arg(long)]
    pub relation: Option<String>,

    /// Filter by object (optional)
    #[arg(long)]
    pub object: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Turtle,
    Json,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "turtle" | "ttl" => Ok(ExportFormat::Turtle),
            "json" => Ok(ExportFormat::Json),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

#[derive(Debug, Args)]
pub struct ImportArgs {
    /// Input file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Import format (turtle, json, auto)
    #[arg(short, long, default_value = "auto")]
    pub format: ImportFormat,

    /// Authorization namespace
    #[arg(long, default_value = "http://oxirs.org/auth#")]
    pub namespace: String,

    /// Overwrite existing relationships
    #[arg(long)]
    pub overwrite: bool,

    /// Dry run (don't actually import)
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub enum ImportFormat {
    Auto,
    Turtle,
    Json,
}

impl std::str::FromStr for ImportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(ImportFormat::Auto),
            "turtle" | "ttl" => Ok(ImportFormat::Turtle),
            "json" => Ok(ImportFormat::Json),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

#[derive(Debug, Args)]
pub struct MigrateArgs {
    /// Source backend (in-memory, rdf-native)
    #[arg(long)]
    pub from: Backend,

    /// Target backend (in-memory, rdf-native)
    #[arg(long)]
    pub to: Backend,

    /// Verify after migration
    #[arg(long, default_value_t = true)]
    pub verify: bool,

    /// Backup before migration
    #[arg(long, default_value_t = true)]
    pub backup: bool,

    /// Backup file path
    #[arg(long)]
    pub backup_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum Backend {
    InMemory,
    RdfNative,
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "in-memory" | "memory" | "inmemory" => Ok(Backend::InMemory),
            "rdf-native" | "rdf" | "rdfnative" => Ok(Backend::RdfNative),
            _ => Err(format!("Unknown backend: {}", s)),
        }
    }
}

#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// Backend to verify (in-memory, rdf-native)
    #[arg(long, default_value = "in-memory")]
    pub backend: Backend,

    /// Check for duplicates
    #[arg(long, default_value_t = true)]
    pub check_duplicates: bool,

    /// Check for orphaned relationships
    #[arg(long, default_value_t = true)]
    pub check_orphans: bool,
}

#[derive(Debug, Args)]
pub struct StatsArgs {
    /// Backend to analyze (in-memory, rdf-native)
    #[arg(long, default_value = "in-memory")]
    pub backend: Backend,

    /// Show detailed breakdown
    #[arg(long)]
    pub detailed: bool,

    /// Output format (text, json)
    #[arg(long, default_value = "text")]
    pub format: String,
}

/// Execute rebac command
pub async fn execute(args: RebacArgs) -> CliResult<()> {
    match args.command {
        RebacCommand::Export(export_args) => export_relationships(export_args).await,
        RebacCommand::Import(import_args) => import_relationships(import_args).await,
        RebacCommand::Migrate(migrate_args) => migrate_backend(migrate_args).await,
        RebacCommand::Verify(verify_args) => verify_integrity(verify_args).await,
        RebacCommand::Stats(stats_args) => show_statistics(stats_args).await,
    }
}

/// Export relationships to file
async fn export_relationships(args: ExportArgs) -> CliResult<()> {
    info!("Exporting relationships to {}", args.output.display());

    // Create ReBAC manager (in-memory for now, can be extended to persistent)
    let manager = RebacManager::new_in_memory()?
        .with_namespace(args.namespace.clone())
        .with_graph(args.graph.clone());

    // Query relationships with filters
    let relationships = manager.query_relationships(
        args.subject.as_deref(),
        args.relation.as_deref(),
        args.object.as_deref(),
    )?;

    if relationships.is_empty() {
        warn!("No relationships found matching the criteria");
        println!("\n⚠️  No relationships found to export");
        println!("  Subject filter: {:?}", args.subject);
        println!("  Relation filter: {:?}", args.relation);
        println!("  Object filter: {:?}", args.object);
        return Ok(());
    }

    // Export to requested format
    let content = match args.format {
        ExportFormat::Turtle => manager.export_turtle()?,
        ExportFormat::Json => manager.export_json()?,
    };

    std::fs::write(&args.output, content).map_err(CliError::io_error)?;

    info!("✅ Export complete: {}", args.output.display());
    println!("\n✅ Exported relationships:");
    println!("  Format: {:?}", args.format);
    println!("  Output: {}", args.output.display());
    println!("  Count: {} relationships", relationships.len());

    if let Some(subject) = args.subject {
        println!("  Filtered by subject: {}", subject);
    }
    if let Some(relation) = args.relation {
        println!("  Filtered by relation: {}", relation);
    }
    if let Some(object) = args.object {
        println!("  Filtered by object: {}", object);
    }

    Ok(())
}

/// Import relationships from file
async fn import_relationships(args: ImportArgs) -> CliResult<()> {
    info!("Importing relationships from {}", args.input.display());

    // Read input file
    let content = std::fs::read_to_string(&args.input).map_err(CliError::io_error)?;

    // Determine format
    let format = match args.format {
        ImportFormat::Auto => {
            if args.input.extension().and_then(|s| s.to_str()) == Some("json") {
                ImportFormat::Json
            } else {
                ImportFormat::Turtle
            }
        }
        other => other,
    };

    if args.dry_run {
        warn!("🔍 DRY RUN - No changes will be made");
    }

    println!("\nImport Summary:");
    println!("  File: {}", args.input.display());
    println!("  Format: {:?}", format);
    println!("  Overwrite: {}", args.overwrite);
    println!("  Dry run: {}", args.dry_run);
    println!("\nParsed {} bytes of relationship data", content.len());

    if !args.dry_run {
        // Create ReBAC manager
        let mut manager = RebacManager::new_in_memory()?.with_namespace(args.namespace.clone());

        // Clear existing if overwrite is enabled
        if args.overwrite {
            let cleared = manager.clear_all()?;
            info!("Cleared {} existing relationships", cleared);
            println!("  Cleared {} existing relationships", cleared);
        }

        // Import based on format
        let count = match format {
            ImportFormat::Turtle => manager.import_turtle(&content)?,
            ImportFormat::Json => manager.import_json(&content)?,
            ImportFormat::Auto => {
                return Err(CliError::validation_error(
                    "Auto format should have been resolved",
                ))
            }
        };

        info!("✅ Import complete: {} relationships", count);
        println!("\n✅ Successfully imported {} relationships", count);

        // Verify integrity after import
        let report = manager.verify_integrity()?;
        if !report.is_valid {
            warn!("⚠️  Integrity issues detected:");
            if report.duplicates > 0 {
                println!("  ⚠️  {} duplicate relationships found", report.duplicates);
            }
            if report.orphans > 0 {
                println!("  ⚠️  {} orphaned relationships found", report.orphans);
            }
        } else {
            println!("  ✓ Integrity check passed");
        }
    } else {
        println!("\n🔍 Dry run complete - no changes made");
    }

    Ok(())
}

/// Migrate between backends
async fn migrate_backend(args: MigrateArgs) -> CliResult<()> {
    info!("Migrating from {:?} to {:?}", args.from, args.to);

    // Create source manager
    let source_manager = match args.from {
        Backend::InMemory => RebacManager::new_in_memory()?,
        Backend::RdfNative => {
            let path = std::env::temp_dir().join("rebac_source");
            RebacManager::new_persistent(&path)?
        }
    };

    // Get all relationships from source
    let relationships = source_manager.get_all_relationships()?;
    let total_count = relationships.len();

    if total_count == 0 {
        warn!("No relationships found in source backend");
        println!("\n⚠️  No relationships to migrate");
        return Ok(());
    }

    println!("\n🔄 Migration in progress...");
    println!("  Source: {:?}", args.from);
    println!("  Target: {:?}", args.to);
    println!("  Relationships to migrate: {}", total_count);

    // Backup if requested
    if args.backup {
        let backup_path = args
            .backup_path
            .unwrap_or_else(|| PathBuf::from("rebac_backup.ttl"));

        let turtle = source_manager.export_turtle()?;
        std::fs::write(&backup_path, turtle).map_err(CliError::io_error)?;

        info!("📦 Backup created: {}", backup_path.display());
        println!("\n📦 Backup created: {}", backup_path.display());
    }

    // Create target manager
    let mut target_manager = match args.to {
        Backend::InMemory => RebacManager::new_in_memory()?,
        Backend::RdfNative => {
            let path = std::env::temp_dir().join("rebac_target");
            RebacManager::new_persistent(&path)?
        }
    };

    // Migrate relationships
    let migrated = target_manager.add_relationships(&relationships)?;
    info!("Migrated {} relationships", migrated);

    println!("\n✅ Migration complete!");
    println!("  Migrated: {} relationships", migrated);

    if args.verify {
        println!("\n🔍 Verifying migration...");

        // Verify counts match
        let target_rels = target_manager.get_all_relationships()?;
        if target_rels.len() != total_count {
            return Err(CliError::validation_error(format!(
                "Verification failed: expected {} relationships, found {}",
                total_count,
                target_rels.len()
            )));
        }

        // Verify integrity
        let report = target_manager.verify_integrity()?;
        if !report.is_valid {
            warn!("⚠️  Integrity issues found after migration:");
            if report.duplicates > 0 {
                println!("  ⚠️  {} duplicate relationships", report.duplicates);
            }
            if report.orphans > 0 {
                println!("  ⚠️  {} orphaned relationships", report.orphans);
            }
        } else {
            println!("✅ Verification passed");
            println!(
                "  - All {} relationships migrated successfully",
                total_count
            );
            println!("  - No data loss detected");
            println!("  - Integrity check passed");
        }
    }

    Ok(())
}

/// Verify data integrity
async fn verify_integrity(args: VerifyArgs) -> CliResult<()> {
    info!("Verifying {:?} backend integrity", args.backend);

    // Create manager based on backend
    let manager = match args.backend {
        Backend::InMemory => RebacManager::new_in_memory()?,
        Backend::RdfNative => {
            let path = std::env::temp_dir().join("rebac_persistent");
            RebacManager::new_persistent(&path)?
        }
    };

    println!("\n🔍 Verifying ReBAC data integrity...");
    println!("  Backend: {:?}", args.backend);

    let mut issues_found = false;

    if args.check_duplicates {
        println!("\n  ✓ Checking for duplicates...");
        let duplicates = manager.find_duplicates()?;
        if duplicates.is_empty() {
            println!("    ✓ No duplicates found");
        } else {
            issues_found = true;
            warn!("Found {} duplicate relationships", duplicates.len());
            println!(
                "    ⚠️  {} duplicate relationships found:",
                duplicates.len()
            );
            for (i, dup) in duplicates.iter().take(5).enumerate() {
                println!(
                    "      {}. {} --[{}]-> {}",
                    i + 1,
                    dup.subject,
                    dup.relation,
                    dup.object
                );
            }
            if duplicates.len() > 5 {
                println!("      ... and {} more", duplicates.len() - 5);
            }
        }
    }

    if args.check_orphans {
        println!("\n  ✓ Checking for orphaned relationships...");
        let orphans = manager.find_orphans()?;
        if orphans.is_empty() {
            println!("    ✓ No orphans found");
        } else {
            issues_found = true;
            warn!("Found {} orphaned relationships", orphans.len());
            println!("    ⚠️  {} orphaned relationships found:", orphans.len());
            for (i, orphan) in orphans.iter().take(5).enumerate() {
                println!(
                    "      {}. {} --[{}]-> {} (orphaned)",
                    i + 1,
                    orphan.subject,
                    orphan.relation,
                    orphan.object
                );
            }
            if orphans.len() > 5 {
                println!("      ... and {} more", orphans.len() - 5);
            }
        }
    }

    // Overall integrity report
    let report = manager.verify_integrity()?;
    println!("\n📊 Integrity Summary:");
    println!("  Total relationships: {}", report.total_relationships);
    println!("  Duplicates: {}", report.duplicates);
    println!("  Orphans: {}", report.orphans);

    if !issues_found && report.is_valid {
        println!("\n✅ Verification complete - all checks passed");
    } else {
        println!("\n⚠️  Verification complete - issues detected");
        println!("  Please review the issues above and take corrective action");
    }

    Ok(())
}

/// Show statistics
async fn show_statistics(args: StatsArgs) -> CliResult<()> {
    info!("Collecting {:?} backend statistics", args.backend);

    // Create manager based on backend
    let manager = match args.backend {
        Backend::InMemory => RebacManager::new_in_memory()?,
        Backend::RdfNative => {
            let path = std::env::temp_dir().join("rebac_persistent");
            RebacManager::new_persistent(&path)?
        }
    };

    // Get statistics
    let stats = manager.get_statistics()?;

    // Output based on format
    match args.format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&stats).map_err(|e| {
                CliError::serialization_error(format!("JSON serialization failed: {}", e))
            })?;
            println!("{}", json);
        }
        _ => {
            // Text format
            println!("\n📊 ReBAC Statistics");
            println!("Backend: {:?}\n", args.backend);

            println!("Total relationships: {}", stats.total_relationships);
            println!(
                "Conditional relationships: {}",
                stats.conditional_relationships
            );

            if !stats.by_relation.is_empty() {
                println!("\nBy relation type:");
                let mut relations: Vec<_> = stats.by_relation.iter().collect();
                relations.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
                for (relation, count) in relations {
                    println!("  {}: {}", relation, count);
                }
            }

            if args.detailed {
                println!("\nDetailed breakdown:");

                if !stats.by_subject.is_empty() {
                    println!("  By subject:");
                    let mut subjects: Vec<_> = stats.by_subject.iter().collect();
                    subjects.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
                    for (subject, count) in subjects.iter().take(10) {
                        println!("    {}: {} relationships", subject, count);
                    }
                    if subjects.len() > 10 {
                        println!("    ... and {} more subjects", subjects.len() - 10);
                    }
                }

                if !stats.by_object.is_empty() {
                    println!("\n  By object:");
                    let mut objects: Vec<_> = stats.by_object.iter().collect();
                    objects.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
                    for (object, count) in objects.iter().take(10) {
                        println!("    {}: {} relationships", object, count);
                    }
                    if objects.len() > 10 {
                        println!("    ... and {} more objects", objects.len() - 10);
                    }
                }
            }
        }
    }

    Ok(())
}
