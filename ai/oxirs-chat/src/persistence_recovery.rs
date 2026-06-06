//! Session recovery, checkpoint loading, and partial recovery support.
//!
//! Sibling module of [`crate::persistence`]. Encapsulates the
//! corruption-tolerant code paths that try to recover sessions from
//! checkpoint files, backups, or by salvaging data from partially
//! readable storage files.

use anyhow::{anyhow, Context, Result};
use std::{
    collections::HashMap,
    fs,
    path::Path,
    time::{Duration, SystemTime},
};
use tokio::{fs::File, io::AsyncReadExt};
use tracing::{debug, error, info, warn};

use crate::{
    persistence_storage::SessionPersistenceManager,
    persistence_types::{PersistedSession, PersistentChatSession, RecoveryInfo, RecoveryStrategy},
};

impl SessionPersistenceManager {
    /// Attempt to recover a corrupted session.
    pub async fn attempt_recovery(
        &self,
        session_id: &str,
    ) -> Result<Option<PersistentChatSession>> {
        warn!("Attempting recovery for session {}", session_id);

        let recovery_info = self.analyze_recovery_options(session_id).await?;

        for strategy in &recovery_info.recovery_strategies {
            match strategy {
                RecoveryStrategy::LoadFromCheckpoint => {
                    if let Ok(Some(session)) = self.load_from_checkpoint(session_id).await {
                        info!(
                            "Successfully recovered session {} from checkpoint",
                            session_id
                        );
                        return Ok(Some(session));
                    }
                }
                RecoveryStrategy::LoadFromBackup => {
                    if let Ok(Some(session)) = self.load_from_backup(session_id).await {
                        info!("Successfully recovered session {} from backup", session_id);
                        return Ok(Some(session));
                    }
                }
                RecoveryStrategy::PartialRecovery => {
                    if let Ok(Some(session)) = self.attempt_partial_recovery(session_id).await {
                        warn!("Partial recovery successful for session {}", session_id);
                        return Ok(Some(session));
                    }
                }
                RecoveryStrategy::CreateNew => {
                    warn!(
                        "Creating new session to replace corrupted session {}",
                        session_id
                    );
                    return Ok(Some(self.create_emergency_session(session_id).await?));
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.persistence_stats.write().await;
            stats.recovery_operations += 1;
            stats.corrupted_sessions += 1;
            stats.load_failures += 1;
        }

        error!("Failed to recover session {}", session_id);
        Ok(None)
    }

    pub(crate) async fn analyze_recovery_options(&self, session_id: &str) -> Result<RecoveryInfo> {
        let backup_available = self.get_backup_file_path(session_id).exists();
        let checkpoint_path = self.get_checkpoint_file_path(session_id);
        let checkpoint_available = checkpoint_path.exists();

        let mut recovery_strategies = Vec::new();

        // Add checkpoint recovery if available
        if checkpoint_available {
            recovery_strategies.push(RecoveryStrategy::LoadFromCheckpoint);
        }

        if backup_available {
            recovery_strategies.push(RecoveryStrategy::LoadFromBackup);
        }

        recovery_strategies.push(RecoveryStrategy::PartialRecovery);
        recovery_strategies.push(RecoveryStrategy::CreateNew);

        // Get actual checkpoint time from file metadata
        let last_checkpoint = if checkpoint_available {
            match fs::metadata(&checkpoint_path) {
                Ok(metadata) => {
                    // Use file modification time as checkpoint time
                    metadata.modified().unwrap_or_else(|_| SystemTime::now())
                }
                Err(_) => SystemTime::now(),
            }
        } else {
            SystemTime::UNIX_EPOCH // No checkpoint available
        };

        Ok(RecoveryInfo {
            session_id: session_id.to_string(),
            last_checkpoint,
            backup_available,
            corruption_detected: true,
            recovery_strategies,
        })
    }

    async fn load_from_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<PersistentChatSession>> {
        let checkpoint_path = self.get_checkpoint_file_path(session_id);
        if !checkpoint_path.exists() {
            debug!("No checkpoint file found for session: {}", session_id);
            return Ok(None);
        }

        info!(
            "Loading session {} from checkpoint: {:?}",
            session_id, checkpoint_path
        );

        // Read checkpoint file
        let mut file = File::open(&checkpoint_path)
            .await
            .context("Failed to open checkpoint file")?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .await
            .context("Failed to read checkpoint file")?;

        // Deserialize the persisted session
        let persisted: PersistedSession = if self.config.compression_enabled {
            self.decompress_session(&data)
                .await
                .context("Failed to decompress checkpoint session")?
        } else {
            oxicode::serde::decode_from_slice(&data, oxicode::config::standard())
                .map(|(session, _)| session)
                .context("Failed to deserialize checkpoint session")?
        };

        // Verify checksum if available
        if !persisted.checksum.is_empty() && !self.verify_checksum(&persisted).await? {
            warn!(
                "Checkpoint checksum verification failed for session: {}",
                session_id
            );
            return Ok(None);
        }

        // Convert to persistent chat session
        let session = self
            .convert_to_persistent_chat_session(&persisted)
            .await
            .context("Failed to convert persisted session")?;

        // Update stats
        {
            let mut stats = self.persistence_stats.write().await;
            stats.total_sessions_loaded += 1;
        }

        info!("Successfully loaded session {} from checkpoint", session_id);
        Ok(Some(session))
    }

    async fn load_from_backup(&self, session_id: &str) -> Result<Option<PersistentChatSession>> {
        let backup_path = self.get_backup_file_path(session_id);
        if !backup_path.exists() {
            return Ok(None);
        }

        let mut file = File::open(&backup_path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;

        let persisted: PersistedSession = if self.config.compression_enabled {
            self.decompress_session(&data).await?
        } else {
            oxicode::serde::decode_from_slice(&data, oxicode::config::standard())
                .map_err(|e| anyhow!("Bincode decoding failed: {}", e))?
                .0
        };

        Ok(Some(
            self.convert_to_persistent_chat_session(&persisted).await?,
        ))
    }

    async fn attempt_partial_recovery(
        &self,
        session_id: &str,
    ) -> Result<Option<PersistentChatSession>> {
        info!("Attempting partial recovery for session: {}", session_id);

        let mut recovered_session = None;
        let mut recovery_sources = Vec::new();

        // Try to recover from various sources in order of preference

        // 1. Try to read from main session file with error tolerance
        let session_path = self.get_session_file_path(session_id);
        if session_path.exists() {
            if let Some(partial_session) = self
                .try_partial_file_recovery(&session_path, "session")
                .await
            {
                recovered_session = Some(partial_session);
                recovery_sources.push("main_session_file");
            }
        }

        // 2. Try to read from checkpoint file with error tolerance
        if recovered_session.is_none() {
            let checkpoint_path = self.get_checkpoint_file_path(session_id);
            if checkpoint_path.exists() {
                if let Some(partial_session) = self
                    .try_partial_file_recovery(&checkpoint_path, "checkpoint")
                    .await
                {
                    recovered_session = Some(partial_session);
                    recovery_sources.push("checkpoint_file");
                }
            }
        }

        // 3. Try to read from backup file with error tolerance
        if recovered_session.is_none() {
            let backup_path = self.get_backup_file_path(session_id);
            if backup_path.exists() {
                if let Some(partial_session) =
                    self.try_partial_file_recovery(&backup_path, "backup").await
                {
                    recovered_session = Some(partial_session);
                    recovery_sources.push("backup_file");
                }
            }
        }

        // 4. If we have some recovery, validate and sanitize it
        if let Some(mut session) = recovered_session {
            // Sanitize the session data
            session = self.sanitize_recovered_session(session, session_id).await;

            // Update recovery stats
            {
                let mut stats = self.persistence_stats.write().await;
                stats.recovery_operations += 1;
            }

            info!(
                "Partial recovery successful for session {} from sources: {:?}",
                session_id, recovery_sources
            );

            return Ok(Some(session));
        }

        warn!("Partial recovery failed for session: {}", session_id);
        Ok(None)
    }

    /// Attempt to read and recover data from a potentially corrupted file.
    async fn try_partial_file_recovery(
        &self,
        file_path: &Path,
        source_type: &str,
    ) -> Option<PersistentChatSession> {
        debug!(
            "Trying partial recovery from {} file: {:?}",
            source_type, file_path
        );

        // Try to read the file
        let data = match self.read_file_with_tolerance(file_path).await {
            Ok(data) => data,
            Err(e) => {
                warn!("Failed to read {} file: {}", source_type, e);
                return None;
            }
        };

        // Try to deserialize with error tolerance
        let persisted_session = if self.config.compression_enabled {
            // Try decompression first
            match self.decompress_session(&data).await {
                Ok(session) => session,
                Err(_) => {
                    // Fallback to direct deserialization if decompression fails
                    match oxicode::serde::decode_from_slice::<PersistedSession, _>(
                        &data,
                        oxicode::config::standard(),
                    ) {
                        Ok((session, _)) => session,
                        Err(e) => {
                            warn!("Failed to deserialize {} file: {}", source_type, e);
                            return None;
                        }
                    }
                }
            }
        } else {
            match oxicode::serde::decode_from_slice::<PersistedSession, _>(
                &data,
                oxicode::config::standard(),
            ) {
                Ok((session, _)) => session,
                Err(e) => {
                    warn!("Failed to deserialize {} file: {}", source_type, e);
                    return None;
                }
            }
        };

        // Convert to persistent chat session
        match self
            .convert_to_persistent_chat_session(&persisted_session)
            .await
        {
            Ok(session) => Some(session),
            Err(e) => {
                warn!("Failed to convert {} session: {}", source_type, e);
                None
            }
        }
    }

    /// Read file with error tolerance.
    async fn read_file_with_tolerance(&self, file_path: &Path) -> Result<Vec<u8>> {
        let mut file = File::open(file_path).await?;
        let mut data = Vec::new();

        // Try to read the entire file, but don't fail on partial reads
        match file.read_to_end(&mut data).await {
            Ok(_) => Ok(data),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // File is truncated but we got some data
                if !data.is_empty() {
                    warn!("File truncated but partial data recovered: {:?}", file_path);
                    Ok(data)
                } else {
                    Err(e.into())
                }
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Sanitize and repair a recovered session.
    async fn sanitize_recovered_session(
        &self,
        mut session: PersistentChatSession,
        session_id: &str,
    ) -> PersistentChatSession {
        // Ensure session ID matches
        session.session_id = session_id.to_string();

        // Validate and fix timestamps
        if session.created_at > SystemTime::now() {
            session.created_at = SystemTime::now() - Duration::from_secs(86400);
            // Set to yesterday (24 hours ago)
        }

        if session.last_accessed > SystemTime::now() {
            session.last_accessed = SystemTime::now();
        }

        // Filter out potentially corrupted messages
        session.messages.retain(|msg| {
            !msg.id.is_empty()
                && !msg.content.to_string().is_empty()
                && msg.timestamp <= chrono::Utc::now()
        });

        // Ensure we have at least basic metrics
        if session.metrics.total_messages == 0 && !session.messages.is_empty() {
            session.metrics.total_messages = session.messages.len();
        }

        // Reset user preferences if they seem corrupted
        if session.user_preferences.len() > 1000 {
            warn!(
                "User preferences seem corrupted, resetting for session: {}",
                session_id
            );
            session.user_preferences = HashMap::new();
        }

        // Add recovery metadata
        session.user_preferences.insert(
            "recovery_performed".to_string(),
            serde_json::Value::String(
                serde_json::to_string(&chrono::Utc::now()).unwrap_or_default(),
            ),
        );

        session.user_preferences.insert(
            "recovery_type".to_string(),
            serde_json::Value::String("partial".to_string()),
        );

        session
    }
}
