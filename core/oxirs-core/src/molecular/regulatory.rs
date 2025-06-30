//! Regulatory systems for controlling molecular processes

use crate::error::OxirsResult;
use super::types::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Regulatory protein for controlling cellular processes
#[derive(Debug, Clone)]
pub struct RegulatoryProtein {
    /// Protein name
    pub name: String,
    /// Function
    pub function: RegulatoryFunction,
    /// Activity level
    pub activity_level: f64,
    /// Binding sites
    pub binding_sites: Vec<BindingSite>,
    /// Post-translational modifications
    pub modifications: Vec<PostTranslationalModification>,
    /// Half-life
    pub half_life: Duration,
    /// Last activity timestamp
    pub last_activity: Option<Instant>,
}

/// Binding site for protein interactions
#[derive(Debug, Clone)]
pub struct BindingSite {
    /// Site identifier
    pub id: String,
    /// Binding affinity
    pub affinity: f64,
    /// Specificity
    pub specificity: BindingSpecificity,
    /// Occupied status
    pub occupied: bool,
    /// Bound ligand
    pub bound_ligand: Option<String>,
}

/// Binding specificity
#[derive(Debug, Clone)]
pub enum BindingSpecificity {
    DNA,
    RNA,
    Protein,
    SmallMolecule,
    Metabolite,
}

/// Post-translational modification
#[derive(Debug, Clone)]
pub struct PostTranslationalModification {
    /// Modification type
    pub modification_type: ModificationType,
    /// Position
    pub position: usize,
    /// Intensity
    pub intensity: f64,
    /// Modification time
    pub timestamp: Instant,
    /// Enzyme responsible
    pub enzyme: Option<String>,
}

/// Comprehensive checkpoint system
#[derive(Debug, Clone)]
pub struct CheckpointSystem {
    /// Spindle checkpoint
    pub spindle_checkpoint: SpindleCheckpoint,
    /// DNA damage checkpoint
    pub dna_damage_checkpoint: DnaDamageCheckpoint,
    /// Replication checkpoint
    pub replication_checkpoint: ReplicationCheckpoint,
    /// Metabolic checkpoint
    pub metabolic_checkpoint: MetabolicCheckpoint,
    /// Quality control checkpoint
    pub quality_control_checkpoint: QualityControlCheckpoint,
}

/// Spindle checkpoint for mitosis control
#[derive(Debug, Clone)]
pub struct SpindleCheckpoint {
    /// Mad proteins
    pub mad_proteins: Vec<MadProtein>,
    /// Bub proteins
    pub bub_proteins: Vec<BubProtein>,
    /// APC/C regulation
    pub apc_c_regulation: ApcCRegulation,
    /// Kinetochore signals
    pub kinetochore_signals: Vec<KinetochoreSignal>,
}

/// Mad protein (Mitotic arrest deficient)
#[derive(Debug, Clone)]
pub struct MadProtein {
    /// Protein type
    pub protein_type: MadProteinType,
    /// Activity level
    pub activity_level: f64,
    /// Localization
    pub localization: ProteinLocalization,
    /// Binding partners
    pub binding_partners: Vec<String>,
}

/// Bub protein (Budding uninhibited by benzimidazoles)
#[derive(Debug, Clone)]
pub struct BubProtein {
    /// Protein type
    pub protein_type: BubProteinType,
    /// Activity level
    pub activity_level: f64,
    /// Kinetochore binding
    pub kinetochore_binding: bool,
    /// Phosphorylation state
    pub phosphorylation_state: PhosphorylationState,
}

/// APC/C regulation system
#[derive(Debug, Clone)]
pub struct ApcCRegulation {
    /// Cdc20 level
    pub cdc20_level: f64,
    /// Cdh1 level
    pub cdh1_level: f64,
    /// Activity state
    pub activity_state: ApcCActivityState,
    /// Substrate recognition
    pub substrate_recognition: SubstrateRecognition,
}

/// Kinetochore signal
#[derive(Debug, Clone)]
pub struct KinetochoreSignal {
    /// Signal type
    pub signal_type: KinetochoreSignalType,
    /// Strength
    pub strength: f64,
    /// Duration
    pub duration: Duration,
    /// Source kinetochore
    pub source: String,
}

/// DNA damage checkpoint
#[derive(Debug, Clone)]
pub struct DnaDamageCheckpoint {
    /// ATM activity
    pub atm_activity: f64,
    /// ATR activity
    pub atr_activity: f64,
    /// p53 level
    pub p53_level: f64,
    /// DNA repair mechanisms
    pub repair_mechanisms: Vec<DnaRepairMechanism>,
    /// Damage sensors
    pub damage_sensors: Vec<DamageSensor>,
}

/// DNA repair mechanism
#[derive(Debug, Clone)]
pub struct DnaRepairMechanism {
    /// Repair type
    pub repair_type: DnaRepairType,
    /// Efficiency
    pub efficiency: f64,
    /// Active proteins
    pub active_proteins: Vec<String>,
}

/// Damage sensor
#[derive(Debug, Clone)]
pub struct DamageSensor {
    /// Sensor type
    pub sensor_type: DamageSensorType,
    /// Sensitivity
    pub sensitivity: f64,
    /// Response time
    pub response_time: Duration,
}

/// Replication checkpoint
#[derive(Debug, Clone)]
pub struct ReplicationCheckpoint {
    /// Replication forks
    pub replication_forks: Vec<ReplicationFork>,
    /// Checkpoint active
    pub checkpoint_active: bool,
    /// Fork protection complex
    pub fork_protection_complex: ForkProtectionComplex,
}

/// Replication fork
#[derive(Debug, Clone)]
pub struct ReplicationFork {
    /// Position
    pub position: usize,
    /// Stalled
    pub stalled: bool,
    /// Associated proteins
    pub proteins: Vec<String>,
    /// Replication speed
    pub speed: f64,
    /// Fork integrity
    pub integrity: f64,
}

/// Fork protection complex
#[derive(Debug, Clone)]
pub struct ForkProtectionComplex {
    /// Complex components
    pub components: Vec<String>,
    /// Activity level
    pub activity: f64,
    /// Protection efficiency
    pub protection_efficiency: f64,
}

/// Metabolic checkpoint
#[derive(Debug, Clone)]
pub struct MetabolicCheckpoint {
    /// Energy levels
    pub energy_levels: EnergyLevels,
    /// Nutrient availability
    pub nutrient_availability: NutrientAvailability,
    /// Metabolic sensors
    pub metabolic_sensors: Vec<MetabolicSensor>,
}

/// Energy levels in the cell
#[derive(Debug, Clone)]
pub struct EnergyLevels {
    /// ATP concentration
    pub atp: f64,
    /// ADP concentration
    pub adp: f64,
    /// AMP concentration
    pub amp: f64,
    /// Energy charge
    pub energy_charge: f64,
}

/// Nutrient availability
#[derive(Debug, Clone)]
pub struct NutrientAvailability {
    /// Glucose level
    pub glucose: f64,
    /// Amino acid levels
    pub amino_acids: HashMap<String, f64>,
    /// Lipid levels
    pub lipids: f64,
    /// Vitamin levels
    pub vitamins: HashMap<String, f64>,
}

/// Metabolic sensor
#[derive(Debug, Clone)]
pub struct MetabolicSensor {
    /// Sensor type
    pub sensor_type: MetabolicSensorType,
    /// Sensitivity
    pub sensitivity: f64,
    /// Target metabolite
    pub target_metabolite: String,
}

/// Quality control checkpoint
#[derive(Debug, Clone)]
pub struct QualityControlCheckpoint {
    /// Protein quality control
    pub protein_qc: ProteinQualityControl,
    /// RNA quality control
    pub rna_qc: RnaQualityControl,
    /// Organelle quality control
    pub organelle_qc: OrganelleQualityControl,
}

/// Protein quality control
#[derive(Debug, Clone)]
pub struct ProteinQualityControl {
    /// Chaperone systems
    pub chaperones: Vec<ChaperoneSystem>,
    /// Proteasome activity
    pub proteasome_activity: f64,
    /// Autophagy activity
    pub autophagy_activity: f64,
}

/// Chaperone system
#[derive(Debug, Clone)]
pub struct ChaperoneSystem {
    /// Chaperone type
    pub chaperone_type: ChaperoneType,
    /// Activity level
    pub activity: f64,
    /// Client proteins
    pub client_proteins: Vec<String>,
}

/// RNA quality control
#[derive(Debug, Clone)]
pub struct RnaQualityControl {
    /// Nonsense-mediated decay
    pub nmd_activity: f64,
    /// RNA surveillance
    pub surveillance_activity: f64,
    /// Exosome activity
    pub exosome_activity: f64,
}

/// Organelle quality control
#[derive(Debug, Clone)]
pub struct OrganelleQualityControl {
    /// Mitochondrial quality
    pub mitochondrial_qc: MitochondrialQC,
    /// ER quality control
    pub er_qc: ErQualityControl,
}

/// Mitochondrial quality control
#[derive(Debug, Clone)]
pub struct MitochondrialQC {
    /// Mitophagy activity
    pub mitophagy_activity: f64,
    /// Membrane potential
    pub membrane_potential: f64,
    /// ROS levels
    pub ros_levels: f64,
}

/// ER quality control
#[derive(Debug, Clone)]
pub struct ErQualityControl {
    /// UPR activity
    pub upr_activity: f64,
    /// ERAD activity
    pub erad_activity: f64,
    /// ER stress level
    pub stress_level: f64,
}

// Supporting enums

#[derive(Debug, Clone)]
pub enum MadProteinType {
    Mad1,
    Mad2,
    Mad3,
    BubR1,
}

#[derive(Debug, Clone)]
pub enum BubProteinType {
    Bub1,
    Bub3,
    BubR1,
}

#[derive(Debug, Clone)]
pub enum ProteinLocalization {
    Kinetochore,
    SpindlePole,
    Cytoplasm,
    Nucleus,
    Centrosome,
    Membrane,
}

#[derive(Debug, Clone)]
pub enum ApcCActivityState {
    Active,
    Inactive,
    PartiallyActive(f64),
}

#[derive(Debug, Clone)]
pub struct SubstrateRecognition {
    pub destruction_box: bool,
    pub ken_box: bool,
    pub abba_motif: bool,
}

#[derive(Debug, Clone)]
pub enum KinetochoreSignalType {
    Tension,
    Attachment,
    Detachment,
    Error,
}

#[derive(Debug, Clone)]
pub enum PhosphorylationState {
    Phosphorylated,
    Dephosphorylated,
    PartiallyPhosphorylated(f64),
}

#[derive(Debug, Clone)]
pub enum DnaRepairType {
    HomologousRecombination,
    NonHomologousEndJoining,
    BaseExcisionRepair,
    NucleotideExcisionRepair,
    MismatchRepair,
}

#[derive(Debug, Clone)]
pub enum DamageSensorType {
    ATM,
    ATR,
    DNAPKcs,
    PARP1,
}

#[derive(Debug, Clone)]
pub enum MetabolicSensorType {
    AMPK,
    mTOR,
    SIRT1,
    HIF1α,
}

#[derive(Debug, Clone)]
pub enum ChaperoneType {
    Hsp70,
    Hsp90,
    Hsp60,
    SmallHSPs,
}

impl RegulatoryProtein {
    /// Create a new regulatory protein
    pub fn new(name: String, function: RegulatoryFunction) -> Self {
        Self {
            name,
            function,
            activity_level: 1.0,
            binding_sites: Vec::new(),
            modifications: Vec::new(),
            half_life: Duration::from_secs(3600), // 1 hour default
            last_activity: None,
        }
    }

    /// Activate the protein
    pub fn activate(&mut self) -> OxirsResult<()> {
        self.activity_level = 1.0;
        self.last_activity = Some(Instant::now());
        Ok(())
    }

    /// Deactivate the protein
    pub fn deactivate(&mut self) -> OxirsResult<()> {
        self.activity_level = 0.0;
        Ok(())
    }

    /// Add a post-translational modification
    pub fn add_modification(&mut self, mod_type: ModificationType, position: usize) {
        let modification = PostTranslationalModification {
            modification_type: mod_type,
            position,
            intensity: 1.0,
            timestamp: Instant::now(),
            enzyme: None,
        };
        self.modifications.push(modification);
    }

    /// Remove a modification
    pub fn remove_modification(&mut self, position: usize) {
        self.modifications.retain(|m| m.position != position);
    }

    /// Check if protein is active
    pub fn is_active(&self) -> bool {
        self.activity_level > 0.1
    }

    /// Calculate current activity based on modifications and time
    pub fn calculate_current_activity(&self) -> f64 {
        let base_activity = self.activity_level;
        
        // Apply modification effects
        let modification_factor: f64 = self.modifications
            .iter()
            .map(|m| match m.modification_type {
                ModificationType::Phosphorylation => 1.2, // Increase activity
                ModificationType::Methylation => 0.8,      // Decrease activity
                ModificationType::Acetylation => 1.1,     // Slight increase
                _ => 1.0,
            })
            .product();

        // Apply time decay
        let time_factor = if let Some(last_active) = self.last_activity {
            let elapsed = last_active.elapsed();
            if elapsed < self.half_life {
                1.0
            } else {
                0.5_f64.powf(elapsed.as_secs_f64() / self.half_life.as_secs_f64())
            }
        } else {
            1.0
        };

        base_activity * modification_factor * time_factor
    }
}

impl CheckpointSystem {
    /// Create new comprehensive checkpoint system
    pub fn new() -> Self {
        Self {
            spindle_checkpoint: SpindleCheckpoint::new(),
            dna_damage_checkpoint: DnaDamageCheckpoint::new(),
            replication_checkpoint: ReplicationCheckpoint::new(),
            metabolic_checkpoint: MetabolicCheckpoint::new(),
            quality_control_checkpoint: QualityControlCheckpoint::new(),
        }
    }

    /// Perform comprehensive checkpoint evaluation
    pub fn evaluate_checkpoints(&self) -> OxirsResult<CheckpointResult> {
        let spindle_ok = self.spindle_checkpoint.check_alignment()?;
        let dna_ok = self.dna_damage_checkpoint.check_integrity()?;
        let replication_ok = self.replication_checkpoint.check_completion()?;
        let metabolic_ok = self.metabolic_checkpoint.check_resources()?;
        let quality_ok = self.quality_control_checkpoint.check_quality()?;

        Ok(CheckpointResult {
            spindle_checkpoint: spindle_ok,
            dna_damage_checkpoint: dna_ok,
            replication_checkpoint: replication_ok,
            metabolic_checkpoint: metabolic_ok,
            quality_control_checkpoint: quality_ok,
            overall_pass: spindle_ok && dna_ok && replication_ok && metabolic_ok && quality_ok,
        })
    }

    /// Get checkpoint status summary
    pub fn get_status_summary(&self) -> CheckpointStatusSummary {
        CheckpointStatusSummary {
            active_checkpoints: vec![
                "Spindle".to_string(),
                "DNA Damage".to_string(),
                "Replication".to_string(),
                "Metabolic".to_string(),
                "Quality Control".to_string(),
            ],
            critical_issues: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

impl SpindleCheckpoint {
    /// Create new spindle checkpoint
    pub fn new() -> Self {
        Self {
            mad_proteins: vec![
                MadProtein {
                    protein_type: MadProteinType::Mad1,
                    activity_level: 1.0,
                    localization: ProteinLocalization::Kinetochore,
                    binding_partners: vec!["Mad2".to_string()],
                },
                MadProtein {
                    protein_type: MadProteinType::Mad2,
                    activity_level: 1.0,
                    localization: ProteinLocalization::Kinetochore,
                    binding_partners: vec!["Mad1".to_string(), "Cdc20".to_string()],
                },
            ],
            bub_proteins: vec![
                BubProtein {
                    protein_type: BubProteinType::Bub1,
                    activity_level: 1.0,
                    kinetochore_binding: true,
                    phosphorylation_state: PhosphorylationState::Phosphorylated,
                },
                BubProtein {
                    protein_type: BubProteinType::Bub3,
                    activity_level: 1.0,
                    kinetochore_binding: true,
                    phosphorylation_state: PhosphorylationState::Phosphorylated,
                },
            ],
            apc_c_regulation: ApcCRegulation {
                cdc20_level: 0.5,
                cdh1_level: 0.1,
                activity_state: ApcCActivityState::Inactive,
                substrate_recognition: SubstrateRecognition {
                    destruction_box: true,
                    ken_box: true,
                    abba_motif: false,
                },
            },
            kinetochore_signals: Vec::new(),
        }
    }

    /// Check chromosome alignment
    pub fn check_alignment(&self) -> OxirsResult<bool> {
        // Check Mad protein activity (should be low when aligned)
        let mad_activity: f64 = self.mad_proteins.iter().map(|p| p.activity_level).sum();
        
        // Check APC/C activity (should be active when aligned)
        let apc_active = matches!(self.apc_c_regulation.activity_state, ApcCActivityState::Active);
        
        Ok(mad_activity < 0.5 && apc_active)
    }
}

impl DnaDamageCheckpoint {
    /// Create new DNA damage checkpoint
    pub fn new() -> Self {
        Self {
            atm_activity: 0.0,
            atr_activity: 0.0,
            p53_level: 0.1,
            repair_mechanisms: vec![
                DnaRepairMechanism {
                    repair_type: DnaRepairType::HomologousRecombination,
                    efficiency: 0.95,
                    active_proteins: vec!["BRCA1".to_string(), "BRCA2".to_string(), "RAD51".to_string()],
                },
                DnaRepairMechanism {
                    repair_type: DnaRepairType::NonHomologousEndJoining,
                    efficiency: 0.85,
                    active_proteins: vec!["DNA-PKcs".to_string(), "Ku70".to_string(), "Ku80".to_string()],
                },
            ],
            damage_sensors: vec![
                DamageSensor {
                    sensor_type: DamageSensorType::ATM,
                    sensitivity: 0.99,
                    response_time: Duration::from_millis(100),
                },
                DamageSensor {
                    sensor_type: DamageSensorType::ATR,
                    sensitivity: 0.95,
                    response_time: Duration::from_millis(50),
                },
            ],
        }
    }

    /// Check DNA integrity
    pub fn check_integrity(&self) -> OxirsResult<bool> {
        let damage_level = self.atm_activity + self.atr_activity;
        Ok(damage_level < 0.1) // Low damage threshold
    }
}

impl ReplicationCheckpoint {
    /// Create new replication checkpoint
    pub fn new() -> Self {
        Self {
            replication_forks: Vec::new(),
            checkpoint_active: false,
            fork_protection_complex: ForkProtectionComplex {
                components: vec!["RPA".to_string(), "Rad9".to_string(), "Rad1".to_string()],
                activity: 1.0,
                protection_efficiency: 0.95,
            },
        }
    }

    /// Check replication completion
    pub fn check_completion(&self) -> OxirsResult<bool> {
        let stalled_forks = self.replication_forks.iter().filter(|f| f.stalled).count();
        Ok(stalled_forks == 0)
    }
}

impl MetabolicCheckpoint {
    /// Create new metabolic checkpoint
    pub fn new() -> Self {
        Self {
            energy_levels: EnergyLevels {
                atp: 5.0,
                adp: 1.0,
                amp: 0.1,
                energy_charge: 0.9,
            },
            nutrient_availability: NutrientAvailability {
                glucose: 5.0,
                amino_acids: HashMap::new(),
                lipids: 2.0,
                vitamins: HashMap::new(),
            },
            metabolic_sensors: vec![
                MetabolicSensor {
                    sensor_type: MetabolicSensorType::AMPK,
                    sensitivity: 0.95,
                    target_metabolite: "AMP".to_string(),
                },
                MetabolicSensor {
                    sensor_type: MetabolicSensorType::mTOR,
                    sensitivity: 0.90,
                    target_metabolite: "Amino acids".to_string(),
                },
            ],
        }
    }

    /// Check resource availability
    pub fn check_resources(&self) -> OxirsResult<bool> {
        Ok(self.energy_levels.energy_charge > 0.7 && self.nutrient_availability.glucose > 1.0)
    }
}

impl QualityControlCheckpoint {
    /// Create new quality control checkpoint
    pub fn new() -> Self {
        Self {
            protein_qc: ProteinQualityControl {
                chaperones: vec![
                    ChaperoneSystem {
                        chaperone_type: ChaperoneType::Hsp70,
                        activity: 1.0,
                        client_proteins: Vec::new(),
                    },
                    ChaperoneSystem {
                        chaperone_type: ChaperoneType::Hsp90,
                        activity: 1.0,
                        client_proteins: Vec::new(),
                    },
                ],
                proteasome_activity: 1.0,
                autophagy_activity: 0.8,
            },
            rna_qc: RnaQualityControl {
                nmd_activity: 1.0,
                surveillance_activity: 1.0,
                exosome_activity: 1.0,
            },
            organelle_qc: OrganelleQualityControl {
                mitochondrial_qc: MitochondrialQC {
                    mitophagy_activity: 0.8,
                    membrane_potential: 180.0,
                    ros_levels: 0.2,
                },
                er_qc: ErQualityControl {
                    upr_activity: 0.1,
                    erad_activity: 1.0,
                    stress_level: 0.1,
                },
            },
        }
    }

    /// Check overall quality
    pub fn check_quality(&self) -> OxirsResult<bool> {
        let protein_ok = self.protein_qc.proteasome_activity > 0.5;
        let rna_ok = self.rna_qc.nmd_activity > 0.5;
        let organelle_ok = self.organelle_qc.mitochondrial_qc.membrane_potential > 150.0;
        
        Ok(protein_ok && rna_ok && organelle_ok)
    }
}

/// Checkpoint evaluation result
#[derive(Debug, Clone)]
pub struct CheckpointResult {
    pub spindle_checkpoint: bool,
    pub dna_damage_checkpoint: bool,
    pub replication_checkpoint: bool,
    pub metabolic_checkpoint: bool,
    pub quality_control_checkpoint: bool,
    pub overall_pass: bool,
}

/// Checkpoint status summary
#[derive(Debug, Clone)]
pub struct CheckpointStatusSummary {
    pub active_checkpoints: Vec<String>,
    pub critical_issues: Vec<String>,
    pub warnings: Vec<String>,
}

impl Default for RegulatoryProtein {
    fn default() -> Self {
        Self::new("Unknown".to_string(), RegulatoryFunction::Loading)
    }
}

impl Default for CheckpointSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regulatory_protein_creation() {
        let protein = RegulatoryProtein::new("TestProtein".to_string(), RegulatoryFunction::Loading);
        assert_eq!(protein.name, "TestProtein");
        assert!(protein.is_active());
    }

    #[test]
    fn test_checkpoint_system() {
        let checkpoint = CheckpointSystem::new();
        let result = checkpoint.evaluate_checkpoints().unwrap();
        assert!(result.overall_pass);
    }

    #[test]
    fn test_protein_modification() {
        let mut protein = RegulatoryProtein::new("TestProtein".to_string(), RegulatoryFunction::Loading);
        protein.add_modification(ModificationType::Phosphorylation, 100);
        assert_eq!(protein.modifications.len(), 1);
        
        protein.remove_modification(100);
        assert_eq!(protein.modifications.len(), 0);
    }

    #[test]
    fn test_protein_activity_calculation() {
        let mut protein = RegulatoryProtein::new("TestProtein".to_string(), RegulatoryFunction::Loading);
        protein.activate().unwrap();
        
        let activity = protein.calculate_current_activity();
        assert!(activity > 0.0);
    }
}