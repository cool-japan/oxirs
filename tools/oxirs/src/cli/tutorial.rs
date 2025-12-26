//! Interactive Tutorial Mode for OxiRS CLI
//!
//! Provides guided learning experiences for new users to learn OxiRS commands,
//! SPARQL queries, and RDF concepts through interactive lessons.

use colored::Colorize;
use dialoguer::{theme::ColorfulTheme, Confirm, Select};

use crate::cli::error::{CliError, CliErrorKind, CliResult};

/// Tutorial lesson
#[derive(Debug, Clone)]
pub struct TutorialLesson {
    pub id: String,
    pub title: String,
    pub description: String,
    pub steps: Vec<TutorialStep>,
    pub difficulty: Difficulty,
}

/// Individual tutorial step
#[derive(Debug, Clone)]
pub struct TutorialStep {
    pub instruction: String,
    pub example_command: Option<String>,
    pub expected_output: Option<String>,
    pub hints: Vec<String>,
    pub validation: Option<StepValidation>,
}

/// Step validation function type
#[derive(Debug, Clone)]
pub enum StepValidation {
    CommandMatches(String),
    OutputContains(String),
    FileExists(String),
    Custom(String), // Description of what to check
}

/// Tutorial difficulty level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    Beginner,
    Intermediate,
    Advanced,
}

impl Difficulty {
    pub fn as_str(&self) -> &str {
        match self {
            Difficulty::Beginner => "Beginner",
            Difficulty::Intermediate => "Intermediate",
            Difficulty::Advanced => "Advanced",
        }
    }

    pub fn color(&self) -> colored::Color {
        match self {
            Difficulty::Beginner => colored::Color::Green,
            Difficulty::Intermediate => colored::Color::Yellow,
            Difficulty::Advanced => colored::Color::Red,
        }
    }
}

/// Tutorial manager
pub struct TutorialManager {
    lessons: Vec<TutorialLesson>,
    current_lesson: Option<usize>,
    completed_lessons: Vec<String>,
}

impl TutorialManager {
    /// Create a new tutorial manager with default lessons
    pub fn new() -> Self {
        let lessons = Self::default_lessons();
        Self {
            lessons,
            current_lesson: None,
            completed_lessons: Vec::new(),
        }
    }

    /// Get all available lessons
    pub fn lessons(&self) -> &[TutorialLesson] {
        &self.lessons
    }

    /// Start tutorial mode
    pub fn start(&mut self) -> CliResult<()> {
        self.show_welcome();

        loop {
            let choice = self.show_main_menu()?;

            match choice {
                MenuChoice::ListLessons => self.list_lessons()?,
                MenuChoice::StartLesson => self.select_and_start_lesson()?,
                MenuChoice::ContinueLesson => {
                    if let Some(lesson_idx) = self.current_lesson {
                        self.run_lesson(lesson_idx)?;
                    } else {
                        println!("{}", "No lesson in progress.".yellow());
                    }
                }
                MenuChoice::ViewProgress => self.show_progress(),
                MenuChoice::Help => self.show_help(),
                MenuChoice::Exit => {
                    println!(
                        "\n{}",
                        "Thanks for learning OxiRS! Happy querying! 🚀"
                            .green()
                            .bold()
                    );
                    break;
                }
            }
        }

        Ok(())
    }

    /// Show welcome message
    fn show_welcome(&self) {
        println!("\n{}", "═".repeat(60).cyan());
        println!(
            "{}",
            "  Welcome to OxiRS Interactive Tutorial!  ".cyan().bold()
        );
        println!("{}", "═".repeat(60).cyan());
        println!(
            "\n{}",
            "Learn SPARQL queries, RDF concepts, and OxiRS commands through".white()
        );
        println!("{}\n", "interactive lessons. Let's get started!".white());
    }

    /// Show main menu
    fn show_main_menu(&self) -> CliResult<MenuChoice> {
        let options = vec![
            "📚 List available lessons",
            "▶️  Start a lesson",
            "⏯️  Continue current lesson",
            "📊 View progress",
            "❓ Help",
            "🚪 Exit tutorial",
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("What would you like to do?")
            .items(&options)
            .default(0)
            .interact()
            .map_err(|e| CliError::new(CliErrorKind::Other(e.to_string())))?;

        Ok(match selection {
            0 => MenuChoice::ListLessons,
            1 => MenuChoice::StartLesson,
            2 => MenuChoice::ContinueLesson,
            3 => MenuChoice::ViewProgress,
            4 => MenuChoice::Help,
            5 => MenuChoice::Exit,
            _ => MenuChoice::Exit,
        })
    }

    /// List all available lessons
    fn list_lessons(&self) -> CliResult<()> {
        println!("\n{}", "Available Lessons:".cyan().bold());
        println!("{}", "─".repeat(60).cyan());

        for (idx, lesson) in self.lessons.iter().enumerate() {
            let status = if self.completed_lessons.contains(&lesson.id) {
                "✓".green()
            } else if self.current_lesson == Some(idx) {
                "⏵".yellow()
            } else {
                "○".white()
            };

            let difficulty_badge =
                format!("[{}]", lesson.difficulty.as_str()).color(lesson.difficulty.color());

            println!(
                "\n{} {}. {} {}",
                status,
                idx + 1,
                lesson.title.bold(),
                difficulty_badge
            );
            println!("   {}", lesson.description.dimmed());
            println!("   {} steps", lesson.steps.len());
        }

        println!();
        Ok(())
    }

    /// Select and start a lesson
    fn select_and_start_lesson(&mut self) -> CliResult<()> {
        let lesson_titles: Vec<String> = self
            .lessons
            .iter()
            .enumerate()
            .map(|(idx, lesson)| {
                let status = if self.completed_lessons.contains(&lesson.id) {
                    "✓"
                } else {
                    "○"
                };
                format!(
                    "{} {}. {} [{}]",
                    status,
                    idx + 1,
                    lesson.title,
                    lesson.difficulty.as_str()
                )
            })
            .collect();

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Select a lesson")
            .items(&lesson_titles)
            .default(0)
            .interact()
            .map_err(|e| CliError::new(CliErrorKind::Other(e.to_string())))?;

        self.current_lesson = Some(selection);
        self.run_lesson(selection)?;

        Ok(())
    }

    /// Run a specific lesson
    fn run_lesson(&mut self, lesson_idx: usize) -> CliResult<()> {
        let lesson = self.lessons[lesson_idx].clone();

        println!("\n{}", "═".repeat(60).cyan());
        println!("  {}  ", lesson.title.cyan().bold());
        println!("{}", "═".repeat(60).cyan());
        println!("\n{}\n", lesson.description);

        for (step_idx, step) in lesson.steps.iter().enumerate() {
            println!(
                "\n{} {} {}/{}",
                "●".yellow(),
                format!("Step {}", step_idx + 1).yellow().bold(),
                step_idx + 1,
                lesson.steps.len()
            );
            println!("{}", "─".repeat(60).yellow());

            println!("\n{}", step.instruction);

            if let Some(ref example) = step.example_command {
                println!("\n{}", "Example command:".green().bold());
                println!("  {}", example.cyan());
            }

            if let Some(ref expected) = step.expected_output {
                println!("\n{}", "Expected output:".green().bold());
                println!("  {}", expected.dimmed());
            }

            // Show hints if requested
            if !step.hints.is_empty()
                && Confirm::with_theme(&ColorfulTheme::default())
                    .with_prompt("Need a hint?")
                    .default(false)
                    .interact()
                    .map_err(|e| CliError::new(CliErrorKind::Other(e.to_string())))?
            {
                println!("\n{}", "💡 Hints:".yellow().bold());
                for (i, hint) in step.hints.iter().enumerate() {
                    println!("  {}. {}", i + 1, hint);
                }
            }

            // Wait for user to complete step
            if Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt("Have you completed this step?")
                .default(true)
                .interact()
                .map_err(|e| CliError::new(CliErrorKind::Other(e.to_string())))?
            {
                println!("{}", "  ✓ Step completed!".green());
            } else {
                println!(
                    "{}",
                    "  Tip: Take your time! You can continue when ready.".yellow()
                );
                return Ok(());
            }
        }

        // Lesson completed
        println!("\n{}", "═".repeat(60).green());
        println!("  {}  ", "🎉 Lesson Completed!".green().bold());
        println!("{}", "═".repeat(60).green());

        if !self.completed_lessons.contains(&lesson.id) {
            self.completed_lessons.push(lesson.id.clone());
        }
        self.current_lesson = None;

        Ok(())
    }

    /// Show progress
    fn show_progress(&self) {
        println!("\n{}", "Your Progress:".cyan().bold());
        println!("{}", "─".repeat(60).cyan());

        let completed = self.completed_lessons.len();
        let total = self.lessons.len();
        let percentage = if total > 0 {
            (completed * 100) / total
        } else {
            0
        };

        println!(
            "\nCompleted: {}/{} lessons ({}%)",
            completed, total, percentage
        );

        if completed > 0 {
            println!("\n{}", "Completed lessons:".green().bold());
            for lesson_id in &self.completed_lessons {
                if let Some(lesson) = self.lessons.iter().find(|l| l.id == *lesson_id) {
                    println!("  ✓ {}", lesson.title);
                }
            }
        }

        if completed < total {
            println!("\n{}", "Remaining lessons:".yellow().bold());
            for lesson in &self.lessons {
                if !self.completed_lessons.contains(&lesson.id) {
                    println!("  ○ {}", lesson.title);
                }
            }
        }

        println!();
    }

    /// Show help
    fn show_help(&self) {
        println!("\n{}", "Tutorial Mode Help".cyan().bold());
        println!("{}", "─".repeat(60).cyan());
        println!("\n{}", "Navigation:".bold());
        println!("  • Use arrow keys to navigate menus");
        println!("  • Press Enter to select an option");
        println!("  • Press Ctrl+C to exit at any time");
        println!("\n{}", "Tips:".bold());
        println!("  • Complete lessons in order for the best learning experience");
        println!("  • Don't hesitate to ask for hints during lessons");
        println!("  • Practice commands in a separate terminal while learning");
        println!("  • You can pause and resume lessons at any time");
        println!();
    }

    /// Create default tutorial lessons
    fn default_lessons() -> Vec<TutorialLesson> {
        vec![
            // Lesson 1: Getting Started
            TutorialLesson {
                id: "getting-started".to_string(),
                title: "Getting Started with OxiRS".to_string(),
                description: "Learn the basics of OxiRS CLI, including initialization and configuration.".to_string(),
                difficulty: Difficulty::Beginner,
                steps: vec![
                    TutorialStep {
                        instruction: "Let's start by checking the OxiRS version. Run 'oxirs --version' to see what version you have installed.".to_string(),
                        example_command: Some("oxirs --version".to_string()),
                        expected_output: Some("oxirs 0.1.0-rc.1".to_string()),
                        hints: vec!["Make sure OxiRS is installed and in your PATH".to_string()],
                        validation: None,
                    },
                    TutorialStep {
                        instruction: "Now let's see all available commands. Run 'oxirs --help' to view the command list.".to_string(),
                        example_command: Some("oxirs --help".to_string()),
                        expected_output: Some("List of commands: query, update, import, export, ...".to_string()),
                        hints: vec!["Use --help with any command to learn more about it".to_string()],
                        validation: None,
                    },
                    TutorialStep {
                        instruction: "Let's initialize a new dataset. Run 'oxirs init my-first-dataset --format tdb2' to create a dataset.".to_string(),
                        example_command: Some("oxirs init my-first-dataset --format tdb2".to_string()),
                        expected_output: Some("Dataset initialized successfully".to_string()),
                        hints: vec![
                            "TDB2 is a high-performance RDF storage format".to_string(),
                            "You can also use 'memory' format for temporary datasets".to_string(),
                        ],
                        validation: Some(StepValidation::FileExists("my-first-dataset".to_string())),
                    },
                ],
            },

            // Lesson 2: Basic SPARQL Queries
            TutorialLesson {
                id: "basic-sparql".to_string(),
                title: "Your First SPARQL Query".to_string(),
                description: "Learn how to write and execute basic SPARQL SELECT queries.".to_string(),
                difficulty: Difficulty::Beginner,
                steps: vec![
                    TutorialStep {
                        instruction: "SPARQL is the query language for RDF data. Let's start with a simple SELECT query.\nA SELECT query retrieves data that matches a pattern.".to_string(),
                        example_command: Some("SELECT * WHERE { ?s ?p ?o }".to_string()),
                        expected_output: Some("This retrieves all triples in the dataset".to_string()),
                        hints: vec![
                            "?s, ?p, ?o are variables representing subject, predicate, object".to_string(),
                            "The * means 'select all variables'".to_string(),
                        ],
                        validation: None,
                    },
                    TutorialStep {
                        instruction: "Let's create some sample RDF data first. Create a file 'data.ttl' with:\n\n@prefix ex: <http://example.org/> .\nex:Alice ex:knows ex:Bob .\nex:Bob ex:knows ex:Charlie .".to_string(),
                        example_command: Some("cat > data.ttl << EOF\n@prefix ex: <http://example.org/> .\nex:Alice ex:knows ex:Bob .\nex:Bob ex:knows ex:Charlie .\nEOF".to_string()),
                        expected_output: None,
                        hints: vec!["Use your favorite text editor to create the file".to_string()],
                        validation: Some(StepValidation::FileExists("data.ttl".to_string())),
                    },
                    TutorialStep {
                        instruction: "Now import the data into your dataset using the import command.".to_string(),
                        example_command: Some("oxirs import --dataset my-first-dataset --file data.ttl --format turtle".to_string()),
                        expected_output: Some("Imported 2 triples".to_string()),
                        hints: vec!["Turtle (.ttl) is a human-friendly RDF format".to_string()],
                        validation: None,
                    },
                    TutorialStep {
                        instruction: "Finally, query the data! Run a SELECT query to see all triples.".to_string(),
                        example_command: Some("oxirs query --dataset my-first-dataset \"SELECT * WHERE { ?s ?p ?o }\"".to_string()),
                        expected_output: Some("Results showing Alice, Bob, and Charlie relationships".to_string()),
                        hints: vec![
                            "You should see 2 results".to_string(),
                            "Try using --format table for nicer output".to_string(),
                        ],
                        validation: None,
                    },
                ],
            },

            // Lesson 3: SPARQL Filters
            TutorialLesson {
                id: "sparql-filters".to_string(),
                title: "Filtering Query Results".to_string(),
                description: "Learn how to use FILTER clauses to refine your SPARQL queries.".to_string(),
                difficulty: Difficulty::Intermediate,
                steps: vec![
                    TutorialStep {
                        instruction: "FILTER clauses let you add conditions to your queries. Let's filter by a specific predicate.".to_string(),
                        example_command: Some("SELECT ?s ?o WHERE { ?s ?p ?o . FILTER(?p = <http://example.org/knows>) }".to_string()),
                        expected_output: Some("Only triples with 'knows' predicate".to_string()),
                        hints: vec!["FILTER goes inside the WHERE clause".to_string()],
                        validation: None,
                    },
                    TutorialStep {
                        instruction: "You can also filter by string patterns using regex. Try finding all subjects starting with 'A'.".to_string(),
                        example_command: Some("SELECT ?s WHERE { ?s ?p ?o . FILTER(regex(str(?s), '^.*Alice$')) }".to_string()),
                        expected_output: Some("Results containing Alice".to_string()),
                        hints: vec![
                            "regex() performs pattern matching".to_string(),
                            "str() converts URIs to strings".to_string(),
                        ],
                        validation: None,
                    },
                ],
            },

            // Lesson 4: Output Formats
            TutorialLesson {
                id: "output-formats".to_string(),
                title: "Working with Different Output Formats".to_string(),
                description: "Explore various output formats for query results including JSON, CSV, PDF, and more.".to_string(),
                difficulty: Difficulty::Beginner,
                steps: vec![
                    TutorialStep {
                        instruction: "OxiRS supports many output formats. Let's try JSON format for machine-readable output.".to_string(),
                        example_command: Some("oxirs query --dataset my-first-dataset \"SELECT * WHERE { ?s ?p ?o }\" --format json".to_string()),
                        expected_output: Some("JSON-formatted results".to_string()),
                        hints: vec!["JSON is great for programmatic processing".to_string()],
                        validation: None,
                    },
                    TutorialStep {
                        instruction: "For data analysis, CSV format is very useful. Try exporting to CSV.".to_string(),
                        example_command: Some("oxirs query --dataset my-first-dataset \"SELECT * WHERE { ?s ?p ?o }\" --format csv > results.csv".to_string()),
                        expected_output: Some("CSV file created".to_string()),
                        hints: vec!["CSV files can be opened in Excel or other spreadsheet tools".to_string()],
                        validation: Some(StepValidation::FileExists("results.csv".to_string())),
                    },
                    TutorialStep {
                        instruction: "For professional reports, try the PDF format!".to_string(),
                        example_command: Some("oxirs query --dataset my-first-dataset \"SELECT * WHERE { ?s ?p ?o }\" --format pdf > report.pdf".to_string()),
                        expected_output: Some("PDF report generated".to_string()),
                        hints: vec!["PDF format is perfect for sharing results with stakeholders".to_string()],
                        validation: Some(StepValidation::FileExists("report.pdf".to_string())),
                    },
                ],
            },
        ]
    }
}

impl Default for TutorialManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Menu choice enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuChoice {
    ListLessons,
    StartLesson,
    ContinueLesson,
    ViewProgress,
    Help,
    Exit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tutorial_manager_creation() {
        let manager = TutorialManager::new();
        assert!(!manager.lessons().is_empty());
    }

    #[test]
    fn test_default_lessons() {
        let lessons = TutorialManager::default_lessons();
        assert_eq!(lessons.len(), 4);
        assert_eq!(lessons[0].id, "getting-started");
        assert_eq!(lessons[1].id, "basic-sparql");
    }

    #[test]
    fn test_difficulty_levels() {
        assert_eq!(Difficulty::Beginner.as_str(), "Beginner");
        assert_eq!(Difficulty::Intermediate.as_str(), "Intermediate");
        assert_eq!(Difficulty::Advanced.as_str(), "Advanced");
    }

    #[test]
    fn test_lesson_structure() {
        let lessons = TutorialManager::default_lessons();
        let lesson = &lessons[0];

        assert!(!lesson.title.is_empty());
        assert!(!lesson.description.is_empty());
        assert!(!lesson.steps.is_empty());
        assert_eq!(lesson.difficulty, Difficulty::Beginner);
    }

    #[test]
    fn test_step_structure() {
        let lessons = TutorialManager::default_lessons();
        let step = &lessons[0].steps[0];

        assert!(!step.instruction.is_empty());
        assert!(step.example_command.is_some());
    }
}
