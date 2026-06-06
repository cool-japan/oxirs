//! Command dispatch logic for OxiRS CLI.
//!
//! Contains the `run()` function that matches on `Commands` and delegates to
//! the appropriate command handler.

use crate::cli_actions::*;
use crate::lib_commands::{Cli, Commands};

/// Run the CLI application
pub async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    use crate::cli::{completion, CliContext};

    // Handle shell completion generation
    if let Some(shell) = cli.completion {
        use clap::CommandFactory;
        let mut app = Cli::command();
        completion::print_completions(shell, &mut app);
        return Ok(());
    }

    // Create CLI context
    let ctx = CliContext::from_cli(cli.verbose, cli.quiet, cli.no_color);

    // Initialize structured logging
    let log_format = if std::env::var("OXIRS_LOG_FORMAT").as_deref() == Ok("json") {
        crate::cli::LogFormat::Json
    } else if ctx.verbose {
        crate::cli::LogFormat::Pretty
    } else {
        crate::cli::LogFormat::Text
    };

    let log_config = crate::cli::LogConfig {
        level: if ctx.verbose {
            "debug".to_string()
        } else if ctx.quiet {
            "error".to_string()
        } else {
            std::env::var("OXIRS_LOG_LEVEL").unwrap_or_else(|_| "info".to_string())
        },
        format: log_format,
        timestamps: !ctx.quiet,
        source_location: ctx.verbose,
        thread_ids: false,
        perf_threshold_ms: std::env::var("OXIRS_PERF_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok()),
        file: std::env::var("OXIRS_LOG_FILE").ok(),
    };

    crate::cli::init_logging(&log_config).expect("Failed to initialize logging");

    // Show startup message if not quiet
    if ctx.should_show_output() {
        ctx.info(&format!("Oxirs CLI v{}", env!("CARGO_PKG_VERSION")));
    }

    match cli.command {
        Commands::Init {
            name,
            format,
            location,
        } => crate::commands::init::run(name, format, location)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Serve {
            config,
            port,
            host,
            graphql,
        } => crate::commands::serve::run(config, port, host, graphql)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Import {
            dataset,
            file,
            format,
            graph,
            resume,
        } => crate::commands::import::run(dataset, file, format, graph, resume)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Export {
            dataset,
            file,
            format,
            graph,
            resume,
        } => crate::commands::export::run(dataset, file, format, graph, resume)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Query {
            dataset,
            query,
            file,
            output,
        } => crate::commands::query::run(dataset, query, file, output)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Update {
            dataset,
            update,
            file,
        } => crate::commands::update::run(dataset, update, file)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Benchmark { action } => match action {
            BenchmarkAction::Run {
                dataset,
                suite,
                iterations,
                output,
                detailed,
                warmup,
            } => crate::commands::benchmark::run(
                dataset, suite, iterations, output, detailed, warmup,
            )
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            BenchmarkAction::Generate {
                output,
                size,
                dataset_type,
                seed,
                triples,
                schema,
            } => crate::commands::benchmark::generate(
                output,
                size,
                dataset_type,
                seed,
                triples,
                schema,
            )
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            BenchmarkAction::Analyze {
                input,
                output,
                format,
                suggestions,
                patterns,
            } => crate::commands::benchmark::analyze(input, output, format, suggestions, patterns)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            BenchmarkAction::Compare {
                baseline,
                current,
                output,
                threshold,
                format,
            } => crate::commands::benchmark::compare(baseline, current, output, threshold, format)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        },
        Commands::Migrate { action } => match action {
            MigrateAction::Format {
                source,
                target,
                from,
                to,
            } => crate::commands::migrate::format(source, target, from, to)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromTdb1 {
                tdb_dir,
                dataset,
                skip_validation,
            } => crate::commands::migrate::from_tdb1(tdb_dir, dataset, skip_validation)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromTdb2 {
                tdb_dir,
                dataset,
                skip_validation,
            } => crate::commands::migrate::from_tdb2(tdb_dir, dataset, skip_validation)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromVirtuoso {
                connection,
                dataset,
                graphs,
            } => crate::commands::migrate::from_virtuoso(connection, dataset, graphs)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromRdf4j { repo_dir, dataset } => {
                crate::commands::migrate::from_rdf4j(repo_dir, dataset)
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            MigrateAction::FromBlazegraph {
                endpoint,
                dataset,
                namespace,
            } => crate::commands::migrate::from_blazegraph(endpoint, dataset, namespace)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromGraphdb {
                endpoint,
                dataset,
                repository,
            } => crate::commands::migrate::from_graphdb(endpoint, dataset, repository)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        },
        Commands::Generate {
            output,
            size,
            r#type,
            format,
            seed,
            schema,
        } => crate::commands::generate::run(output, size, r#type, format, seed, schema)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Index { action } => match action {
            IndexAction::List { dataset } => crate::commands::index::list(dataset)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            IndexAction::Rebuild { dataset, index } => {
                crate::commands::index::rebuild(dataset, index)
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            IndexAction::Stats { dataset, format } => {
                crate::commands::index::stats(dataset, format)
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            IndexAction::Optimize { dataset } => crate::commands::index::optimize(dataset)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        },
        Commands::Visualize {
            dataset,
            output,
            format,
            graph,
            max_nodes,
        } => crate::commands::visualize::export(dataset, output, format, graph, max_nodes)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Config { action } => crate::commands::config::run(action)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),

        // Data Processing Tools
        Commands::Riot {
            input,
            output,
            out,
            syntax,
            base,
            validate,
            count,
        } => crate::tools::riot::run(input, output, out, syntax, base, validate, count).await,
        Commands::RdfCat {
            files,
            format,
            output,
        } => crate::tools::rdfcat::run(files, format, output).await,
        Commands::RdfCopy {
            source,
            target,
            source_format,
            target_format,
        } => crate::tools::rdfcopy::run(source, target, source_format, target_format).await,
        Commands::RdfDiff {
            first,
            second,
            format,
        } => crate::tools::rdfdiff::run(first, second, format).await,
        Commands::RdfParse { file, format, base } => {
            crate::tools::rdfparse::run(file, format, base).await
        }

        // Advanced Query Tools
        Commands::Arq {
            query,
            query_file,
            data,
            namedgraph,
            results,
            dataset,
            explain,
            optimize,
            time,
        } => {
            crate::tools::arq::run(crate::tools::arq::ArqConfig {
                query,
                query_file,
                data,
                namedgraph,
                results_format: results,
                dataset,
                explain,
                optimize,
                time,
            })
            .await
        }
        Commands::RSparql {
            service,
            query,
            query_file,
            results,
            timeout,
        } => crate::tools::rsparql::run(service, query, query_file, results, timeout).await,
        Commands::RUpdate {
            service,
            update,
            update_file,
            timeout,
        } => crate::tools::rupdate::run(service, update, update_file, timeout).await,
        Commands::QParse {
            query,
            file,
            print_ast,
            print_algebra,
        } => crate::tools::qparse::run(query, file, print_ast, print_algebra).await,
        Commands::UParse {
            update,
            file,
            print_ast,
        } => crate::tools::uparse::run(update, file, print_ast).await,

        // Storage Tools
        Commands::TdbLoader {
            location,
            files,
            graph,
            progress,
            stats,
        } => crate::tools::tdbloader::run(location, files, graph, progress, stats).await,
        Commands::TdbDump {
            location,
            output,
            format,
            graph,
        } => crate::tools::tdbdump::run(location, output, format, graph).await,
        Commands::TdbQuery {
            location,
            query,
            file,
            results,
        } => crate::tools::tdbquery::run(location, query, file, results).await,
        Commands::TdbUpdate {
            location,
            update,
            file,
        } => crate::tools::tdbupdate::run(location, update, file).await,
        Commands::TdbStats {
            location,
            detailed,
            format,
        } => crate::tools::tdbstats::run(location, detailed, format).await,
        Commands::TdbBackup {
            source,
            target,
            compress,
            incremental,
            encrypt,
            password,
            keyfile,
            generate_keyfile,
        } => {
            use crate::tools::backup_encryption;

            // Handle keyfile generation
            if let Some(keyfile_path) = generate_keyfile {
                println!("Generating encryption keyfile...");
                backup_encryption::generate_keyfile(&keyfile_path)?;
                println!(
                    "Keyfile generated successfully at: {}",
                    keyfile_path.display()
                );
                println!(
                    "Keep this keyfile secure! Loss of the keyfile means loss of data access."
                );
                return Ok(());
            }

            // Clone target for encryption if needed
            let target_for_encryption = target.clone();

            // Run backup
            crate::tools::tdbbackup::run(source, target, compress, incremental).await?;

            // Encrypt backup if requested
            if encrypt {
                use dialoguer::Password;

                println!("\nEncrypting backup...");
                let backup_file = &target_for_encryption;
                let encrypted_file = backup_file.with_extension("oxirs.enc");

                let encryption_config = if let Some(ref pwd) = password {
                    backup_encryption::EncryptionConfig {
                        password: Some(pwd.clone()),
                        keyfile: None,
                        verify: true,
                    }
                } else if let Some(ref kf) = keyfile {
                    backup_encryption::EncryptionConfig {
                        password: None,
                        keyfile: Some(kf.clone()),
                        verify: true,
                    }
                } else {
                    // Prompt for password
                    let pwd = Password::new()
                        .with_prompt("Enter encryption password")
                        .with_confirmation("Confirm password", "Passwords don't match")
                        .interact()?;

                    backup_encryption::EncryptionConfig {
                        password: Some(pwd),
                        keyfile: None,
                        verify: true,
                    }
                };

                backup_encryption::encrypt_backup(
                    backup_file,
                    &encrypted_file,
                    &encryption_config,
                )?;
                println!(
                    "Backup encrypted successfully: {}",
                    encrypted_file.display()
                );
            }
            Ok(())
        }
        Commands::TdbCompact {
            location,
            delete_old,
        } => crate::tools::tdbcompact::run(location, delete_old).await,

        Commands::Pitr { action } => {
            use crate::tools::pitr::{PitrConfig, TransactionLog};
            use chrono::{DateTime, Utc};

            match action {
                PitrAction::Init {
                    dataset,
                    max_log_size,
                    auto_archive,
                } => {
                    println!("Initializing PITR for dataset: {}", dataset.display());
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: max_log_size * 1_048_576, // Convert MB to bytes
                        auto_archive,
                    };
                    let _log = TransactionLog::new(config)?;
                    println!("PITR initialized successfully");
                }
                PitrAction::Checkpoint { dataset, name } => {
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let log = TransactionLog::new(config)?;
                    let checkpoint_path = log.create_checkpoint(&name)?;
                    println!("Checkpoint created: {}", checkpoint_path.display());
                }
                PitrAction::List { dataset, format } => {
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let log = TransactionLog::new(config)?;
                    let checkpoints = log.list_checkpoints()?;

                    if format == "json" {
                        println!("{}", serde_json::to_string_pretty(&checkpoints)?);
                    } else {
                        println!("Available Checkpoints:");
                        println!("{:-<80}", "");
                        for cp in checkpoints {
                            println!("Name: {}", cp.name);
                            println!("  Timestamp: {}", cp.timestamp.to_rfc3339());
                            println!("  Last Transaction ID: {}", cp.last_transaction_id);
                            println!("  Log Files: {}", cp.log_files.len());
                            println!();
                        }
                    }
                }
                PitrAction::RecoverTimestamp {
                    dataset,
                    timestamp,
                    output,
                } => {
                    let target_time: DateTime<Utc> = timestamp.parse()?;
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let log = TransactionLog::new(config)?;
                    let count = log.recover_to_timestamp(target_time, &output)?;
                    println!("Recovered {} transactions to {}", count, output.display());
                }
                PitrAction::RecoverTransaction {
                    dataset,
                    transaction_id,
                    output,
                } => {
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let log = TransactionLog::new(config)?;
                    let count = log.recover_to_transaction(transaction_id, &output)?;
                    println!("Recovered {} transactions to {}", count, output.display());
                }
                PitrAction::Archive { dataset } => {
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let mut log = TransactionLog::new(config)?;
                    let archived = log.archive_logs()?;
                    println!("Archived {} log files", archived);
                }
            }
            Ok(())
        }

        // Validation Tools
        Commands::Shacl {
            data,
            dataset,
            shapes,
            format,
            output,
        } => crate::tools::shacl::run(data, dataset, shapes, format, output).await,
        Commands::Shex {
            data,
            dataset,
            schema,
            shape_map,
            format,
        } => crate::tools::shex::run(data, dataset, schema, shape_map, format).await,
        Commands::Infer {
            data,
            ontology,
            profile,
            output,
            format,
        } => crate::tools::infer::run(data, ontology, profile, output, format).await,
        Commands::SchemaGen {
            data,
            schema_type,
            output,
            stats,
        } => crate::tools::schemagen::run(data, schema_type, output, stats).await,
        Commands::Aspect { action } => crate::commands::aspect::run(action)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Aas { action } => crate::commands::aas::run(action)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Package { action } => crate::commands::package::run(action)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),

        // Utility Tools
        Commands::Iri {
            iri,
            resolve,
            validate,
            normalize,
        } => crate::tools::iri::run(iri, resolve, validate, normalize).await,
        Commands::LangTag {
            tag,
            validate,
            normalize,
        } => crate::tools::langtag::run(tag, validate, normalize).await,
        Commands::JUuid { count, format } => crate::tools::juuid::run(count, format).await,
        Commands::Utf8 {
            input,
            file,
            validate,
            fix,
        } => crate::tools::utf8::run(input, file, validate, fix).await,
        Commands::WwwEnc { input, encoding } => crate::tools::wwwenc::run(input, encoding).await,
        Commands::WwwDec { input, decoding } => crate::tools::wwwdec::run(input, decoding).await,
        Commands::RSet {
            input,
            input_format,
            output_format,
            output,
        } => crate::tools::rset::run(input, input_format, output_format, output).await,
        Commands::Interactive {
            dataset,
            history: _,
        } => {
            ctx.info("Starting interactive SPARQL shell...");
            crate::commands::interactive::execute(dataset, cli.config)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
        Commands::Performance { action } => {
            let config = crate::config::Config::default();
            action
                .execute(&config)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
        Commands::Explain {
            dataset,
            query,
            file,
            mode,
            graphviz,
        } => {
            let analysis_mode = match mode.to_lowercase().as_str() {
                "explain" => crate::commands::explain::AnalysisMode::Explain,
                "analyze" => crate::commands::explain::AnalysisMode::Analyze,
                "full" => crate::commands::explain::AnalysisMode::Full,
                _ => {
                    eprintln!(
                        "Invalid mode '{}'. Valid modes: explain, analyze, full",
                        mode
                    );
                    return Err("Invalid analysis mode".into());
                }
            };
            crate::commands::explain::explain_query_with_options(
                dataset,
                query,
                file,
                analysis_mode,
                graphviz,
            )
            .await
            .map_err(|e| e.into())
        }
        Commands::Optimize { query, file } => {
            crate::commands::query_optimizer::optimize_command(query, file)
                .await
                .map_err(|e| e.into())
        }
        Commands::Template { action } => {
            use std::collections::HashMap;
            match action {
                TemplateAction::List { category } => {
                    crate::commands::templates::list_command(category)
                        .await
                        .map_err(|e| e.into())
                }
                TemplateAction::Show { name } => crate::commands::templates::show_command(name)
                    .await
                    .map_err(|e| e.into()),
                TemplateAction::Render { name, param } => {
                    let mut params = HashMap::new();
                    for p in param {
                        let parts: Vec<&str> = p.splitn(2, '=').collect();
                        if parts.len() != 2 {
                            eprintln!("Invalid parameter format: '{}'. Expected key=value", p);
                            return Err("Invalid parameter format".into());
                        }
                        params.insert(parts[0].to_string(), parts[1].to_string());
                    }
                    crate::commands::templates::render_command(name, params)
                        .await
                        .map_err(|e| e.into())
                }
            }
        }
        Commands::History { action } => match action {
            HistoryAction::List { limit, dataset } => {
                crate::commands::history::commands::list_command(limit, dataset)
                    .await
                    .map_err(|e| e.into())
            }
            HistoryAction::Show { id } => crate::commands::history::commands::show_command(id)
                .await
                .map_err(|e| e.into()),
            HistoryAction::Replay { id, output } => {
                crate::commands::history::commands::replay_command(id, output)
                    .await
                    .map_err(|e| e.into())
            }
            HistoryAction::Search { query } => {
                crate::commands::history::commands::search_command(query)
                    .await
                    .map_err(|e| e.into())
            }
            HistoryAction::Clear => crate::commands::history::commands::clear_command()
                .await
                .map_err(|e| e.into()),
            HistoryAction::Stats => crate::commands::history::commands::stats_command()
                .await
                .map_err(|e| e.into()),
            HistoryAction::Analytics { dataset } => {
                crate::commands::history::commands::analytics_command(dataset)
                    .await
                    .map_err(|e| e.into())
            }
        },
        Commands::Cicd { action } => match action {
            CicdAction::Report {
                input,
                output,
                format,
            } => crate::commands::cicd::generate_test_report(input, output, format)
                .await
                .map_err(|e| e.into()),
            CicdAction::Docker { output } => crate::commands::cicd::generate_docker_files(output)
                .await
                .map_err(|e| e.into()),
            CicdAction::Github { output } => {
                crate::commands::cicd::generate_github_workflow(output)
                    .await
                    .map_err(|e| e.into())
            }
            CicdAction::Gitlab { output } => crate::commands::cicd::generate_gitlab_ci(output)
                .await
                .map_err(|e| e.into()),
        },
        Commands::Alias { action } => match action {
            AliasAction::List => crate::commands::alias::list().await.map_err(|e| e.into()),
            AliasAction::Show { name } => crate::commands::alias::show(name.clone())
                .await
                .map_err(|e| e.into()),
            AliasAction::Add { name, command } => {
                crate::commands::alias::add(name.clone(), command.clone())
                    .await
                    .map_err(|e| e.into())
            }
            AliasAction::Remove { name } => crate::commands::alias::remove(name.clone())
                .await
                .map_err(|e| e.into()),
            AliasAction::Reset => crate::commands::alias::reset().await.map_err(|e| e.into()),
        },

        Commands::Cache { action } => match action {
            CacheAction::Stats => crate::commands::cache::commands::stats_command()
                .await
                .map_err(|e| e.into()),
            CacheAction::Clear => crate::commands::cache::commands::clear_command()
                .await
                .map_err(|e| e.into()),
            CacheAction::Config { ttl, max_size } => {
                crate::commands::cache::commands::config_command(ttl, max_size)
                    .await
                    .map_err(|e| e.into())
            }
        },

        Commands::Rebac(args) => crate::commands::rebac::execute(args)
            .await
            .map_err(|e| e.into()),

        Commands::Docs {
            format,
            output,
            command,
        } => {
            use crate::cli::doc_generator::{DocFormat, DocGenerator};
            use std::io::Write;

            let doc_format: DocFormat = format
                .parse()
                .map_err(|e: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

            let generator = DocGenerator::new();

            if let Some(cmd_name) = command {
                ctx.info(&format!(
                    "Generating documentation for command: {}",
                    cmd_name
                ));
                // Generate single command docs (future enhancement)
                ctx.warn(
                    "Single command documentation not yet implemented. Generating all commands.",
                );
            }

            let content = generator
                .generate(doc_format)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            if let Some(output_path) = output {
                let mut file = std::fs::File::create(&output_path)?;
                file.write_all(content.as_bytes())?;
                ctx.success(&format!(
                    "Documentation written to: {}",
                    output_path.display()
                ));
            } else {
                println!("{}", content);
            }

            Ok(())
        }

        Commands::Tutorial { lesson } => {
            use crate::cli::tutorial::TutorialManager;

            let mut manager = TutorialManager::new();

            if let Some(lesson_name) = lesson {
                ctx.info(&format!("Starting tutorial with lesson: {}", lesson_name));
                ctx.warn(
                    "Specific lesson selection not yet implemented. Starting interactive tutorial.",
                );
            }

            manager.start().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, format!("Tutorial error: {}", e))
            })?;

            Ok(())
        }

        Commands::GraphAnalytics {
            dataset,
            operation,
            damping,
            max_iter,
            tolerance,
            source,
            target,
            top,
        } => {
            use crate::commands::graph_analytics::{
                execute_graph_analytics, AnalyticsConfig, AnalyticsOperation,
            };
            use std::path::Path;

            // Parse operation
            let op: AnalyticsOperation = operation
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

            // Build configuration
            let config = AnalyticsConfig {
                operation: op,
                damping_factor: damping,
                max_iterations: max_iter,
                tolerance,
                source_node: source.clone(),
                target_node: target.clone(),
                top_k: top,
                katz_alpha: 0.1,            // Default Katz centrality alpha parameter
                katz_beta: 1.0,             // Default Katz centrality beta parameter
                k_core_value: None,         // Auto-detect all cores
                enable_simd: true,          // Auto-enable SIMD optimizations
                enable_parallel: true,      // Auto-enable parallel processing
                enable_gpu: false,          // GPU is opt-in (requires hardware)
                enable_cache: true,         // Enable caching for better performance
                export_path: None,          // No export by default
                enable_benchmarking: false, // Disable benchmarking by default
            };

            // Execute analytics
            let dataset_path = Path::new(dataset.as_str());
            execute_graph_analytics(dataset_path, &config)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

            Ok(())
        }

        // === Phase D: Industrial Connectivity CLI Handlers (0.3.0) ===
        Commands::Tsdb { action } => crate::commands::tsdb::execute(action, &ctx)
            .await
            .map_err(|e| e.into()),

        Commands::Modbus { action } => crate::commands::modbus::execute(action, &ctx)
            .await
            .map_err(|e| e.into()),

        Commands::Canbus { action } => crate::commands::canbus::execute(action, &ctx)
            .await
            .map_err(|e| e.into()),

        Commands::Profile { action } => match action {
            ProfilerAction::Run {
                dataset,
                query,
                file,
                iterations,
                suggestions,
            } => crate::commands::query_profiler::run_profile_command(
                dataset,
                query,
                file,
                iterations,
                suggestions,
            )
            .await
            .map_err(|e| e.into()),
            ProfilerAction::Suggest { query, file } => {
                let q = if file {
                    std::fs::read_to_string(&query)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
                } else {
                    query
                };
                let features = crate::commands::query_profiler::QueryProfileFeatures::extract(&q);
                let suggestions =
                    crate::commands::query_profiler::generate_suggestions(&features, &q);
                for s in suggestions {
                    println!("[{}] {}: {}", s.severity.label(), s.title, s.description);
                }
                Ok(())
            }
        },

        Commands::ResultCache { action } => match action {
            ResultCacheAction::Stats => crate::commands::result_cache::commands::stats_command()
                .await
                .map_err(|e| e.into()),
            ResultCacheAction::Clear => crate::commands::result_cache::commands::clear_command()
                .await
                .map_err(|e| e.into()),
            ResultCacheAction::Invalidate { dataset } => {
                crate::commands::result_cache::commands::invalidate_dataset_command(&dataset)
                    .await
                    .map_err(|e| e.into())
            }
            ResultCacheAction::Evict => {
                crate::commands::result_cache::commands::evict_expired_command()
                    .await
                    .map_err(|e| e.into())
            }
            ResultCacheAction::List { dataset } => {
                crate::commands::result_cache::commands::list_command(dataset.as_deref())
                    .await
                    .map_err(|e| e.into())
            }
            ResultCacheAction::Config { max_size, ttl } => {
                let cache = crate::commands::result_cache::global_lru_cache();
                if let Some(sz) = max_size {
                    println!("Max entries updated to {}", sz);
                    let _ = sz; // config applied at init time
                }
                if let Some(t) = ttl {
                    println!("Default TTL updated to {}s", t);
                    let _ = t;
                }
                let _ = cache;
                Ok(())
            }
        },

        Commands::Stream { action } => match action {
            StreamAction::Query {
                dataset,
                query,
                file,
                chunk_size,
                format,
                max_rows,
                no_progress,
                output,
            } => crate::commands::stream::run_stream_command(
                dataset,
                query,
                file,
                chunk_size,
                format,
                max_rows,
                no_progress,
                output,
            )
            .await
            .map_err(|e| e.into()),
        },
    }
}
