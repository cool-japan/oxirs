//! Alias management commands
//!
//! Provides commands for managing command aliases.

use crate::cli::{AliasManager, CliResult};
use colored::Colorize;

/// List all configured aliases
pub async fn list() -> CliResult<()> {
    let manager = AliasManager::new().map_err(|e| format!("Failed to load aliases: {}", e))?;

    let aliases = manager.list_aliases();

    if aliases.is_empty() {
        println!("{}", "No aliases configured.".yellow());
        println!("\nUse 'oxirs alias add <name> <command>' to create an alias.");
        return Ok(());
    }

    println!("{}", "Configured Aliases:".bold());
    println!();

    // Find longest alias name for formatting
    let _max_len = aliases.keys().map(|k| k.len()).max().unwrap_or(10);

    let mut sorted_aliases: Vec<_> = aliases.iter().collect();
    sorted_aliases.sort_by_key(|(name, _)| *name);

    for (name, command) in sorted_aliases {
        println!(
            "  {} {} {}",
            name.cyan().bold(),
            "→".dimmed(),
            command.white()
        );
    }

    println!();
    println!("{}", format!("Total: {} aliases", aliases.len()).dimmed());

    Ok(())
}

/// Show a specific alias
pub async fn show(name: String) -> CliResult<()> {
    let manager = AliasManager::new().map_err(|e| format!("Failed to load aliases: {}", e))?;

    match manager.get_alias(&name) {
        Some(command) => {
            println!(
                "{} {} {}",
                name.cyan().bold(),
                "→".dimmed(),
                command.white()
            );
            Ok(())
        }
        None => Err(format!("Alias '{}' not found", name).into()),
    }
}

/// Add a new alias
pub async fn add(name: String, command: String) -> CliResult<()> {
    let mut manager = AliasManager::new().map_err(|e| format!("Failed to load aliases: {}", e))?;

    // Check if alias already exists
    let is_update = manager.get_alias(&name).is_some();

    manager
        .add_alias(name.clone(), command.clone())
        .map_err(|e| format!("Failed to add alias: {}", e))?;

    if is_update {
        println!(
            "{} {} {} {}",
            "Updated alias".green(),
            name.cyan().bold(),
            "→".dimmed(),
            command.white()
        );
    } else {
        println!(
            "{} {} {} {}",
            "Added alias".green(),
            name.cyan().bold(),
            "→".dimmed(),
            command.white()
        );
    }

    println!("\n{}", "You can now use this alias in commands.".dimmed());

    Ok(())
}

/// Remove an alias
pub async fn remove(name: String) -> CliResult<()> {
    let mut manager = AliasManager::new().map_err(|e| format!("Failed to load aliases: {}", e))?;

    let removed = manager
        .remove_alias(&name)
        .map_err(|e| format!("Failed to remove alias: {}", e))?;

    if removed {
        println!("{} {}", "Removed alias".green(), name.cyan().bold());
        Ok(())
    } else {
        Err(format!("Alias '{}' not found", name).into())
    }
}

/// Reset aliases to defaults
pub async fn reset() -> CliResult<()> {
    let mut manager = AliasManager::new().map_err(|e| format!("Failed to load aliases: {}", e))?;

    manager
        .reset_to_defaults()
        .map_err(|e| format!("Failed to reset aliases: {}", e))?;

    println!("{}", "Reset aliases to defaults".green());
    println!("\nDefault aliases:");
    list().await?;

    Ok(())
}

// Tests for alias commands are in cli::alias module
// Command-level tests are skipped as they require terminal IO
