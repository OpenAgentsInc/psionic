use psionic_collectives::{
    CollectiveMeshMember, CollectiveSyncCadencePolicy, ElasticCollectivePlanner,
};
use psionic_core::{DType, Device, DeviceKind, Shape, TensorSpec};
use psionic_eval::{
    PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF,
    PARAMETER_GOLF_DISTRIBUTED_8XH100_BENCHMARK_REF,
    PARAMETER_GOLF_DISTRIBUTED_8XH100_CLAIM_BOUNDARY,
    ParameterGolfDistributedChallengeThresholds,
    ParameterGolfDistributedCommunicationReceipt,
    ParameterGolfDistributedCommunicationStageReceipt,
    ParameterGolfDistributedLaneDisposition, ParameterGolfDistributedLaneRefusal,
    ParameterGolfDistributedLaneRefusalKind, ParameterGolfDistributedMemoryReceipt,
    ParameterGolfDistributedThroughputReceipt, ParameterGolfDistributedTimingReceipt,
    ParameterGolfDistributedTopologyReceipt,
};
use psionic_ir::GraphError;
use psionic_models::ParameterGolfModelDescriptor;
use psionic_runtime::{
    BackendSelection, ClusterCommunicationClass, ClusterExecutionCapabilityProfile,
    ClusterTransportClass, DeviceDescriptor, ExecutionTopologyPlan, ServedProductBackendPolicy,
    ServedProductFallbackTrigger, TrainingCollectiveKind, TrainingCollectiveQuantization,
    TrainingDeviceMeshAxis, TrainingDeviceMeshAxisKind, TrainingElasticMembershipContext,
};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use thiserror::Error;

use crate::{
    DistributedOptimizerContract, DistributedOptimizerError, DistributedOptimizerGroupContract,
    DistributedOptimizerRun, DistributedTrainingMemoryBudget, OptimizerStateResidency,
    ParameterGolfBatchGeometry, ParameterGolfOptimizerExecution, ParameterGolfTrainError,
    ParameterGolfTrainingHyperparameters, TrainingActivationCheckpointPolicy,
    TrainingDistributedOptimizerKind, TrainingGradientAccumulationPolicy,
    TrainingGradientAccumulationReduction, TrainingLoopBudget, TrainingOptimizerConfig,
    TrainingOptimizerResidencyPolicy, TrainingOptimizerShardResidency,
    TrainingOptimizerStateShardKind, TrainingOptimizerStateShardLayout, TrainingParameterGroupState,
    TrainingParameterShardKind, TrainingParameterShardLayout,
    TrainingPrecisionPolicy, TrainingShardPlacement, TrainingShardRange, TrainingTensorBuffer,
    builtin_parameter_golf_cuda_training_capability_report, parameter_golf_optimizer_plan,
};

/// Stable version identifier for the distributed `8xH100` receipt lane.
pub const PARAMETER_GOLF_DISTRIBUTED_8XH100_VERSION: &str =
    "2026.03.18.distributed_8xh100.v1";

const PARAMETER_GOLF_DISTRIBUTED_CHECKPOINT_FAMILY: &str =
    "train.parameter_golf.distributed_8xh100";

/// One measured optimizer-step observation used to build the distributed receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfDistributedStepObservation {
    /// One-based global step number.
    pub global_step: u64,
    /// Observed step start.
    pub started_at_ms: u64,
    /// Observed step finish.
    pub finished_at_ms: u64,
    /// Global train tokens processed by the step.
    pub train_tokens: u64,
}

impl ParameterGolfDistributedStepObservation {
    /// Creates one measured step observation.
    #[must_use]
    pub const fn new(
        global_step: u64,
        started_at_ms: u64,
        finished_at_ms: u64,
        train_tokens: u64,
    ) -> Self {
        Self {
            global_step,
            started_at_ms,
            finished_at_ms,
            train_tokens,
        }
    }

    const fn duration_ms(&self) -> u64 {
        self.finished_at_ms.saturating_sub(self.started_at_ms)
    }
}

/// Config for the distributed `8xH100` Parameter Golf receipt lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfDistributed8xH100Config {
    /// Stable run identifier.
    pub run_id: String,
    /// Stable mesh identifier.
    pub mesh_id: String,
    /// Public challenge batch geometry.
    pub geometry: ParameterGolfBatchGeometry,
    /// Declared challenge thresholds.
    pub thresholds: ParameterGolfDistributedChallengeThresholds,
    /// Ordered measured step observations.
    pub step_observations: Vec<ParameterGolfDistributedStepObservation>,
    /// Observed validation duration.
    pub validation_observed_ms: u64,
    /// Observed export or roundtrip duration.
    pub export_observed_ms: u64,
}

impl ParameterGolfDistributed8xH100Config {
    /// Returns the canonical public `8xH100` config skeleton.
    #[must_use]
    pub fn challenge_defaults() -> Self {
        Self {
            run_id: String::from("parameter-golf-distributed-8xh100"),
            mesh_id: String::from("mesh.parameter_golf.8xh100"),
            geometry: ParameterGolfBatchGeometry::challenge_distributed_8xh100_defaults(),
            thresholds: ParameterGolfDistributedChallengeThresholds::challenge_8xh100(),
            step_observations: Vec::new(),
            validation_observed_ms: 0,
            export_observed_ms: 0,
        }
    }

    fn validate(&self) -> Result<(), ParameterGolfDistributedLaneError> {
        if self.run_id.trim().is_empty() {
            return Err(ParameterGolfDistributedLaneError::MissingRunId);
        }
        validate_distributed_geometry(&self.geometry, &self.thresholds)?;
        for observation in &self.step_observations {
            if observation.global_step == 0 {
                return Err(ParameterGolfDistributedLaneError::InvalidStepObservation {
                    message: String::from("global_step must be positive"),
                });
            }
            if observation.finished_at_ms < observation.started_at_ms {
                return Err(ParameterGolfDistributedLaneError::InvalidStepObservation {
                    message: format!(
                        "step {} finished_at_ms={} is earlier than started_at_ms={}",
                        observation.global_step, observation.finished_at_ms, observation.started_at_ms
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Failure while building the distributed `8xH100` Parameter Golf lane.
#[derive(Debug, Error)]
pub enum ParameterGolfDistributedLaneError {
    #[error("distributed parameter golf config is missing `run_id`")]
    MissingRunId,
    #[error("distributed parameter golf config is invalid: {message}")]
    InvalidConfig { message: String },
    #[error("distributed parameter golf step observation is invalid: {message}")]
    InvalidStepObservation { message: String },
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    CollectivePlanning(#[from] psionic_collectives::CollectivePlanningError),
    #[error(transparent)]
    DistributedOptimizer(#[from] DistributedOptimizerError),
    #[error(transparent)]
    TrainingCore(#[from] crate::TrainingCoreError),
    #[error(transparent)]
    Train(#[from] ParameterGolfTrainError),
}

/// Builds the distributed `8xH100` Parameter Golf throughput receipt.
pub fn benchmark_parameter_golf_distributed_8xh100(
    descriptor: &ParameterGolfModelDescriptor,
    hyperparameters: &ParameterGolfTrainingHyperparameters,
    devices: &[DeviceDescriptor],
    capability_profile: &ClusterExecutionCapabilityProfile,
    config: &ParameterGolfDistributed8xH100Config,
) -> Result<ParameterGolfDistributedThroughputReceipt, ParameterGolfDistributedLaneError> {
    config.validate()?;
    let optimizer_plan = parameter_golf_optimizer_plan(descriptor, hyperparameters)?;
    let optimizer_plan_digest =
        stable_digest(b"psionic_parameter_golf_optimizer_plan|", &optimizer_plan);
    let cuda_coverage_report = builtin_parameter_golf_cuda_training_capability_report()?;
    let (backend_selection, all_devices_match_required_model, inventory_refusal) =
        distributed_backend_selection(devices, capability_profile, &config.thresholds);
    let topology_digest = backend_selection
        .execution_topology
        .as_ref()
        .map_or_else(String::new, ExecutionTopologyPlan::stable_digest);
    let topology = ParameterGolfDistributedTopologyReceipt {
        selected_device_names: devices
            .iter()
            .map(|device| {
                device
                    .device_name
                    .clone()
                    .unwrap_or_else(|| device.device.label().unwrap_or("unknown").to_string())
            })
            .collect(),
        backend_selection,
        topology_digest,
        all_devices_match_required_model,
    };
    let axes = vec![
        TrainingDeviceMeshAxis::new(
            "dp",
            TrainingDeviceMeshAxisKind::DataParallel,
            config.geometry.world_size,
        )
        .with_collective_group_size(config.geometry.world_size)
        .with_detail("matches WORLD_SIZE=8 DDP replica axis from train_gpt.py"),
    ];
    let matrix_parameter_count = optimizer_plan
        .groups
        .iter()
        .find(|group| matches!(group.execution, ParameterGolfOptimizerExecution::Muon { .. }))
        .map_or(0_u64, |group| group.parameter_count as u64);
    let total_parameter_count = optimizer_plan.total_parameter_count as u64;
    let ddp_gradient_payload_bytes = total_parameter_count.saturating_mul(2);
    let muon_update_payload_bytes = matrix_parameter_count.saturating_mul(2);
    let validation_payload_bytes = 3 * 8;
    let communication = build_communication_receipt(
        &config.mesh_id,
        &axes,
        devices,
        ddp_gradient_payload_bytes,
        muon_update_payload_bytes,
        validation_payload_bytes,
    )?;
    let memory = build_memory_receipt(
        descriptor,
        hyperparameters,
        &config.geometry,
        &config.thresholds,
        total_parameter_count,
    )?;
    let timing = build_timing_receipt(config);
    let mut boundary_notes = vec![
        String::from(
            "This receipt mirrors the public train_gpt.py 8xH100 posture: replicated DDP, WORLD_SIZE=8, grad_accum_steps=1, NCCL-style all-reduce for training and validation.",
        ),
    ];
    boundary_notes.extend(cuda_coverage_report.boundary_notes());
    boundary_notes.push(String::from(
        "The memory receipt is an analytic upper bound over the distributed optimizer contract; it is not a direct CUDA allocator trace.",
    ));

    let refusal = inventory_refusal
        .or_else(|| {
            if !memory.within_device_budget || !memory.within_host_budget {
                Some(ParameterGolfDistributedLaneRefusal {
                    refusal_kind: ParameterGolfDistributedLaneRefusalKind::MemoryBudgetExceeded,
                    reason: format!(
                        "planned memory exceeds the declared challenge bar: peak_device_bytes_per_worker={} peak_host_bytes_per_worker={}",
                        memory.peak_device_bytes_per_worker, memory.peak_host_bytes_per_worker
                    ),
                    fallback_benchmark_ref: String::from(
                        PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF,
                    ),
                })
            } else {
                None
            }
        })
        .or_else(|| {
            if config.step_observations.is_empty() {
                Some(ParameterGolfDistributedLaneRefusal {
                    refusal_kind: ParameterGolfDistributedLaneRefusalKind::MeasurementsMissing,
                    reason: String::from(
                        "no distributed optimizer-step timing observations were supplied for the 8xH100 lane",
                    ),
                    fallback_benchmark_ref: String::from(
                        PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF,
                    ),
                })
            } else {
                None
            }
        })
        .or_else(|| {
            timing.as_ref().and_then(|timing| {
                (!timing.within_wallclock_cap).then(|| ParameterGolfDistributedLaneRefusal {
                    refusal_kind: ParameterGolfDistributedLaneRefusalKind::WallclockExceeded,
                    reason: format!(
                        "observed distributed wallclock {}ms exceeds the declared cap {}ms",
                        timing.total_observed_ms, timing.wallclock_cap_ms
                    ),
                    fallback_benchmark_ref: String::from(
                        PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF,
                    ),
                })
            })
        });

    Ok(ParameterGolfDistributedThroughputReceipt {
        benchmark_ref: String::from(PARAMETER_GOLF_DISTRIBUTED_8XH100_BENCHMARK_REF),
        run_id: config.run_id.clone(),
        model_descriptor_digest: descriptor.stable_digest(),
        optimizer_plan_digest,
        thresholds: config.thresholds.clone(),
        topology,
        communication,
        training_capability_report_digest: cuda_coverage_report.report_digest.clone(),
        challenge_kernel_blockers: cuda_coverage_report.challenge_kernel_blockers().to_vec(),
        disposition: if refusal.is_some() {
            ParameterGolfDistributedLaneDisposition::Refused
        } else {
            ParameterGolfDistributedLaneDisposition::Measured
        },
        timing,
        memory: Some(memory),
        refusal,
        boundary_notes,
        claim_boundary: String::from(PARAMETER_GOLF_DISTRIBUTED_8XH100_CLAIM_BOUNDARY),
        receipt_digest: String::new(),
    }
    .with_stable_digest())
}

fn validate_distributed_geometry(
    geometry: &ParameterGolfBatchGeometry,
    thresholds: &ParameterGolfDistributedChallengeThresholds,
) -> Result<(), ParameterGolfDistributedLaneError> {
    if geometry.world_size != thresholds.required_world_size {
        return Err(ParameterGolfDistributedLaneError::InvalidConfig {
            message: format!(
                "geometry.world_size={} must match the required challenge world_size={}",
                geometry.world_size, thresholds.required_world_size
            ),
        });
    }
    if geometry.grad_accum_steps != 1 {
        return Err(ParameterGolfDistributedLaneError::InvalidConfig {
            message: format!(
                "the exact 8xH100 lane requires grad_accum_steps=1, found {}",
                geometry.grad_accum_steps
            ),
        });
    }
    if geometry.train_sequence_length == 0 {
        return Err(ParameterGolfDistributedLaneError::InvalidConfig {
            message: String::from("train_sequence_length must be positive"),
        });
    }
    let denom = geometry.world_size.saturating_mul(geometry.grad_accum_steps);
    if geometry.train_batch_tokens == 0
        || geometry.train_batch_tokens % denom != 0
        || geometry.local_train_batch_tokens() < geometry.train_sequence_length
        || geometry.local_train_batch_tokens() % geometry.train_sequence_length != 0
    {
        return Err(ParameterGolfDistributedLaneError::InvalidConfig {
            message: format!(
                "train_batch_tokens={} must divide cleanly across world_size={} and admit full sequences of length {}",
                geometry.train_batch_tokens, geometry.world_size, geometry.train_sequence_length
            ),
        });
    }
    if geometry.validation_batch_tokens == 0
        || geometry.validation_batch_tokens % denom != 0
        || geometry.local_validation_batch_tokens() < geometry.train_sequence_length
        || geometry.local_validation_batch_tokens() % geometry.train_sequence_length != 0
    {
        return Err(ParameterGolfDistributedLaneError::InvalidConfig {
            message: format!(
                "validation_batch_tokens={} must divide cleanly across world_size={} and admit full sequences of length {}",
                geometry.validation_batch_tokens,
                geometry.world_size,
                geometry.train_sequence_length
            ),
        });
    }
    Ok(())
}

fn distributed_backend_selection(
    devices: &[DeviceDescriptor],
    capability_profile: &ClusterExecutionCapabilityProfile,
    thresholds: &ParameterGolfDistributedChallengeThresholds,
) -> (
    BackendSelection,
    bool,
    Option<ParameterGolfDistributedLaneRefusal>,
) {
    let supported_ops = vec![String::from("parameter_golf_distributed_train")];
    let topology = (!devices.is_empty()).then(|| {
        ExecutionTopologyPlan::replicated(
            thresholds.required_backend.clone(),
            devices
                .iter()
                .map(DeviceDescriptor::inventory_qualifiers)
                .collect(),
        )
    });
    let all_devices_match_required_model =
        devices.iter().all(|device| device_matches_h100(device, thresholds));
    let rejection_reason = if capability_profile.runtime_backend != thresholds.required_backend {
        Some(format!(
            "cluster capability profile targets backend `{}` instead of required `{}`",
            capability_profile.runtime_backend, thresholds.required_backend
        ))
    } else if !capability_profile
        .supports_communication_class(ClusterCommunicationClass::TensorCollectiveMesh)
    {
        Some(String::from(
            "cluster capability profile does not advertise tensor_collective_mesh support required for NCCL-style all-reduce",
        ))
    } else if devices.len() != thresholds.required_world_size {
        Some(format!(
            "expected exactly {} devices for the 8xH100 lane, found {}",
            thresholds.required_world_size,
            devices.len()
        ))
    } else if !all_devices_match_required_model {
        Some(format!(
            "selected inventory is not an exact `{}` CUDA posture with non-MIG devices",
            thresholds.required_device_name
        ))
    } else {
        None
    };
    let selection = match rejection_reason.as_ref() {
        Some(reason) => BackendSelection::refused(
            thresholds.required_backend.clone(),
            devices.first().cloned(),
            supported_ops,
            ServedProductBackendPolicy::same_backend_only(),
            ServedProductFallbackTrigger::RequestedBackendUnavailable,
            reason.clone(),
        ),
        None => BackendSelection::direct(
            thresholds.required_backend.clone(),
            devices.first().cloned(),
            supported_ops,
        ),
    }
    .with_selected_devices(devices.to_vec())
    .with_execution_topology(topology)
    .with_cluster_execution_capability_profile(capability_profile.clone());
    let refusal = rejection_reason.map(|reason| ParameterGolfDistributedLaneRefusal {
        refusal_kind: if capability_profile.runtime_backend != thresholds.required_backend
            || !capability_profile
                .supports_communication_class(ClusterCommunicationClass::TensorCollectiveMesh)
        {
            ParameterGolfDistributedLaneRefusalKind::CapabilityMismatch
        } else {
            ParameterGolfDistributedLaneRefusalKind::DeviceInventoryMismatch
        },
        reason,
        fallback_benchmark_ref: String::from(PARAMETER_GOLF_CHALLENGE_REVIEW_BENCHMARK_REF),
    });
    (selection, all_devices_match_required_model, refusal)
}

fn device_matches_h100(
    device: &DeviceDescriptor,
    thresholds: &ParameterGolfDistributedChallengeThresholds,
) -> bool {
    if device.backend != thresholds.required_backend || device.device.kind() != DeviceKind::Cuda {
        return false;
    }
    let name_matches = device
        .device_name
        .as_deref()
        .is_some_and(|name| name.contains(thresholds.required_device_name.as_str()));
    let metadata = match &device.nvidia_metadata {
        Some(metadata) => metadata,
        None => return false,
    };
    name_matches
        && metadata.topology.mig_profile.is_none()
        && !metadata.risk.mig_partitioned
        && device
            .memory_capacity_bytes
            .is_some_and(|bytes| bytes >= thresholds.max_peak_device_bytes_per_worker)
}

fn build_communication_receipt(
    mesh_id: &str,
    axes: &[TrainingDeviceMeshAxis],
    devices: &[DeviceDescriptor],
    ddp_gradient_payload_bytes: u64,
    muon_update_payload_bytes: u64,
    validation_payload_bytes: u64,
) -> Result<ParameterGolfDistributedCommunicationReceipt, ParameterGolfDistributedLaneError> {
    let mut planner = ElasticCollectivePlanner::new(
        mesh_id,
        "cuda",
        ClusterCommunicationClass::TensorCollectiveMesh,
        axes.to_vec(),
    )
    .with_transport(ClusterTransportClass::Loopback);
    let members = devices
        .iter()
        .enumerate()
        .map(|(rank, device)| {
            let node_id = format!("rank-{rank}");
            let device_label = device.device.label().unwrap_or("cuda").to_string();
            CollectiveMeshMember::new(node_id, rank, rank, device_label)
        })
        .collect::<Vec<_>>();
    let elastic_membership = TrainingElasticMembershipContext::new(
        1,
        format!("{mesh_id}.cluster_state"),
        format!("{mesh_id}.topology"),
        (0..devices.len()).map(|rank| format!("rank-{rank}")).collect(),
    );
    planner.observe_mesh(elastic_membership, members)?;
    let ddp_collective = planner.plan_collective(
        TrainingCollectiveKind::AllReduce,
        ddp_gradient_payload_bytes,
        TrainingCollectiveQuantization::None,
    )?;
    let muon_collective = planner.plan_collective(
        TrainingCollectiveKind::AllReduce,
        muon_update_payload_bytes,
        TrainingCollectiveQuantization::None,
    )?;
    let validation_collective = planner.plan_collective(
        TrainingCollectiveKind::AllReduce,
        validation_payload_bytes,
        TrainingCollectiveQuantization::None,
    )?;
    Ok(ParameterGolfDistributedCommunicationReceipt {
        communication_class: ClusterCommunicationClass::TensorCollectiveMesh,
        transport: ClusterTransportClass::Loopback,
        mesh_id: String::from(mesh_id),
        axes: axes.to_vec(),
        stages: vec![
            ParameterGolfDistributedCommunicationStageReceipt {
                stage_id: String::from("ddp_gradient_all_reduce"),
                collective_kind: ddp_collective.collective.kind,
                quantization: ddp_collective.collective.quantization,
                payload_bytes: ddp_collective.collective.payload_bytes,
                estimated_wire_bytes: ddp_collective.collective.estimated_wire_bytes,
                worker_count: ddp_collective.collective.worker_count,
                detail: String::from(
                    "PyTorch DDP-style gradient synchronization over the full BF16 model gradient surface once per optimizer step",
                ),
            },
            ParameterGolfDistributedCommunicationStageReceipt {
                stage_id: String::from("muon_matrix_update_all_reduce"),
                collective_kind: muon_collective.collective.kind,
                quantization: muon_collective.collective.quantization,
                payload_bytes: muon_collective.collective.payload_bytes,
                estimated_wire_bytes: muon_collective.collective.estimated_wire_bytes,
                worker_count: muon_collective.collective.worker_count,
                detail: String::from(
                    "Muon rank-local matrix updates are flattened into one BF16 buffer and all-reduced across the full 8-way mesh",
                ),
            },
            ParameterGolfDistributedCommunicationStageReceipt {
                stage_id: String::from("validation_metric_all_reduce"),
                collective_kind: validation_collective.collective.kind,
                quantization: validation_collective.collective.quantization,
                payload_bytes: validation_collective.collective.payload_bytes,
                estimated_wire_bytes: validation_collective.collective.estimated_wire_bytes,
                worker_count: validation_collective.collective.worker_count,
                detail: String::from(
                    "Validation loss, token count, and byte count are reduced as three f64 scalars across the full mesh",
                ),
            },
        ],
    })
}

fn build_memory_receipt(
    descriptor: &ParameterGolfModelDescriptor,
    hyperparameters: &ParameterGolfTrainingHyperparameters,
    geometry: &ParameterGolfBatchGeometry,
    thresholds: &ParameterGolfDistributedChallengeThresholds,
    total_parameter_count: u64,
) -> Result<ParameterGolfDistributedMemoryReceipt, ParameterGolfDistributedLaneError> {
    let optimizer_plan = parameter_golf_optimizer_plan(descriptor, hyperparameters)?;
    let parameter_groups = optimizer_plan
        .groups
        .iter()
        .map(|group| {
            let optimizer = match &group.execution {
                ParameterGolfOptimizerExecution::Adam { optimizer } => optimizer.clone(),
                ParameterGolfOptimizerExecution::Muon { optimizer } => {
                    TrainingOptimizerConfig::sgd(optimizer.learning_rate)
                        .with_momentum(optimizer.momentum)
                }
            };
            TrainingParameterGroupState::new(
                group.group_id.clone(),
                group.parameter_class,
                TrainingTensorBuffer::from_f32(
                    group.group_id.clone(),
                    TensorSpec::new(
                        Shape::new(vec![group.parameter_count]),
                        DType::F32,
                        Device::cpu(),
                    ),
                    vec![0.0; group.parameter_count],
                )?,
                optimizer,
                TrainingOptimizerResidencyPolicy::new(
                    OptimizerStateResidency::DeviceResident,
                    OptimizerStateResidency::DeviceResident,
                ),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let collective_sync_plan = build_collective_sync_plan(total_parameter_count, geometry.world_size)?;
    let contract = DistributedOptimizerContract::new(
        "optimizer://openagents/parameter_golf/distributed_8xh100",
        TrainingDistributedOptimizerKind::DataParallel,
        TrainingPrecisionPolicy::bf16_master_fp32(TrainingCollectiveQuantization::None),
        TrainingGradientAccumulationPolicy::new(
            geometry.grad_accum_steps as u32,
            TrainingGradientAccumulationReduction::Mean,
            TrainingCollectiveKind::AllReduce,
        ),
        TrainingActivationCheckpointPolicy::Disabled {
            activation_peak_bytes: estimated_activation_peak_bytes(descriptor, geometry),
        },
        DistributedTrainingMemoryBudget::new(u64::MAX / 4, u64::MAX / 4, 0),
        collective_sync_plan,
        optimizer_plan
            .groups
            .iter()
            .map(|group| {
                let placements = (0..geometry.world_size)
                    .map(|rank| {
                        TrainingShardPlacement::new(
                            rank,
                            "dp",
                            format!("rank-{rank}"),
                            format!("cuda:{rank}"),
                            rank,
                            TrainingShardRange::new(0, group.parameter_count),
                        )
                    })
                    .collect::<Vec<_>>();
                DistributedOptimizerGroupContract::new(
                    group.group_id.clone(),
                    group.parameter_class,
                    TrainingParameterShardLayout::new(
                        TrainingParameterShardKind::Replicated,
                        placements.clone(),
                    )
                    .with_axis_id("dp"),
                    TrainingParameterShardLayout::new(
                        TrainingParameterShardKind::Replicated,
                        placements.clone(),
                    )
                    .with_axis_id("dp"),
                    TrainingOptimizerStateShardLayout::new(
                        TrainingOptimizerStateShardKind::Replicated,
                        TrainingOptimizerShardResidency::DeviceResident,
                        placements,
                    )
                    .with_axis_id("dp"),
                    OptimizerStateResidency::DeviceResident,
                )
            })
            .collect(),
    )?;
    let run = DistributedOptimizerRun::new(
        "parameter-golf-distributed-memory-plan",
        PARAMETER_GOLF_DISTRIBUTED_CHECKPOINT_FAMILY,
        TrainingLoopBudget::new(1, 1, 1)?,
        parameter_groups,
        contract,
    )?;
    let memory_plan = run.memory_plan;
    Ok(ParameterGolfDistributedMemoryReceipt {
        measurement_posture: String::from("analytic_optimizer_contract_upper_bound"),
        declared_device_budget_bytes_per_worker: thresholds.max_peak_device_bytes_per_worker,
        declared_host_budget_bytes_per_worker: thresholds.max_peak_host_bytes_per_worker,
        parameter_logical_bytes: memory_plan.parameter_logical_bytes,
        gradient_logical_bytes: memory_plan.gradient_logical_bytes,
        optimizer_state_logical_bytes: memory_plan.optimizer_state_logical_bytes,
        master_weight_logical_bytes: memory_plan.master_weight_logical_bytes,
        activation_peak_bytes: memory_plan.activation_peak_bytes,
        activation_bytes_saved: memory_plan.activation_bytes_saved,
        peak_device_bytes_per_worker: memory_plan.peak_device_bytes_per_worker,
        peak_host_bytes_per_worker: memory_plan.peak_host_bytes_per_worker,
        remote_offloaded_optimizer_bytes: memory_plan.remote_offloaded_optimizer_bytes,
        within_device_budget: memory_plan.peak_device_bytes_per_worker
            <= thresholds.max_peak_device_bytes_per_worker,
        within_host_budget: memory_plan.peak_host_bytes_per_worker
            <= thresholds.max_peak_host_bytes_per_worker,
        planner_receipt_digest: memory_plan.receipt_digest,
    })
}

fn build_collective_sync_plan(
    total_parameter_count: u64,
    world_size: usize,
) -> Result<psionic_collectives::CollectiveSyncExecutionPlan, ParameterGolfDistributedLaneError> {
    let mesh_id = "mesh.parameter_golf.8xh100.contract";
    let axes = vec![
        TrainingDeviceMeshAxis::new("dp", TrainingDeviceMeshAxisKind::DataParallel, world_size)
            .with_collective_group_size(world_size)
            .with_detail("single data-parallel axis for the public 8xH100 lane"),
    ];
    let mut planner = ElasticCollectivePlanner::new(
        mesh_id,
        "cuda",
        ClusterCommunicationClass::TensorCollectiveMesh,
        axes,
    )
    .with_transport(ClusterTransportClass::Loopback);
    let members = (0..world_size)
        .map(|rank| CollectiveMeshMember::new(format!("rank-{rank}"), rank, rank, format!("cuda:{rank}")))
        .collect::<Vec<_>>();
    planner.observe_mesh(
        TrainingElasticMembershipContext::new(
            1,
            "parameter-golf-contract-cluster-state",
            "parameter-golf-contract-topology",
            (0..world_size).map(|rank| format!("rank-{rank}")).collect(),
        ),
        members,
    )?;
    Ok(planner.plan_sync(
        1,
        TrainingCollectiveKind::AllReduce,
        total_parameter_count.saturating_mul(2),
        TrainingCollectiveQuantization::None,
        &CollectiveSyncCadencePolicy::new(),
    )?)
}

fn estimated_activation_peak_bytes(
    descriptor: &ParameterGolfModelDescriptor,
    geometry: &ParameterGolfBatchGeometry,
) -> u64 {
    let tokens_per_worker = geometry.local_train_batch_tokens() as u64;
    let model_dim = descriptor.config.model_dim as u64;
    let layer_factor = (descriptor.config.num_layers as u64)
        .saturating_mul((4 + descriptor.config.mlp_mult) as u64);
    tokens_per_worker
        .saturating_mul(model_dim)
        .saturating_mul(2)
        .saturating_mul(layer_factor.max(1))
}

fn build_timing_receipt(
    config: &ParameterGolfDistributed8xH100Config,
) -> Option<ParameterGolfDistributedTimingReceipt> {
    if config.step_observations.is_empty() {
        return None;
    }
    let step_count = config.step_observations.len() as u64;
    let total_train_tokens = config
        .step_observations
        .iter()
        .map(|observation| observation.train_tokens)
        .sum::<u64>();
    let training_step_observed_ms = config
        .step_observations
        .iter()
        .map(ParameterGolfDistributedStepObservation::duration_ms)
        .sum::<u64>();
    let total_observed_ms = training_step_observed_ms
        .saturating_add(config.validation_observed_ms)
        .saturating_add(config.export_observed_ms);
    let tail_step_duration_ms = config
        .step_observations
        .iter()
        .map(ParameterGolfDistributedStepObservation::duration_ms)
        .max()
        .unwrap_or_default();
    let mean_step_duration_ms = if step_count == 0 {
        0
    } else {
        training_step_observed_ms / step_count
    };
    let train_tokens_per_second = if training_step_observed_ms == 0 {
        0
    } else {
        total_train_tokens.saturating_mul(1_000) / training_step_observed_ms
    };
    Some(ParameterGolfDistributedTimingReceipt {
        measurement_posture: String::from("observed_step_wallclock_plus_validation_and_export"),
        step_count,
        total_train_tokens,
        training_step_observed_ms,
        validation_observed_ms: config.validation_observed_ms,
        export_observed_ms: config.export_observed_ms,
        total_observed_ms,
        mean_step_duration_ms,
        tail_step_duration_ms,
        train_tokens_per_second,
        wallclock_cap_ms: config.thresholds.max_total_wallclock_ms,
        within_wallclock_cap: total_observed_ms <= config.thresholds.max_total_wallclock_ms,
    })
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded = match serde_json::to_vec(value) {
        Ok(encoded) => encoded,
        Err(error) => error.to_string().into_bytes(),
    };
    let mut hasher = sha2::Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use psionic_core::{DType, Device, DeviceKind};
    use psionic_models::{
        ModelDescriptor, ParameterGolfConfig, ParameterGolfDeterministicInitializer,
        ParameterGolfReferenceModel, ParameterGolfWeights,
    };
    use psionic_runtime::{
        BackendSelectionState, ClusterCommunicationClass, ClusterExecutionCapabilityProfile,
        DeviceDescriptor, NvidiaDeviceMetadata, NvidiaRecoveryAction, NvidiaRecoveryProfile,
        NvidiaRiskLevel, NvidiaRiskProfile, NvidiaTopologyInfo,
    };

    use crate::{
        PARAMETER_GOLF_DISTRIBUTED_8XH100_VERSION, ParameterGolfBatchGeometry,
        ParameterGolfDistributed8xH100Config, ParameterGolfDistributedLaneError,
        ParameterGolfDistributedStepObservation, ParameterGolfTrainingHyperparameters,
        benchmark_parameter_golf_distributed_8xh100,
    };
    use psionic_eval::{
        ParameterGolfDistributedLaneDisposition, ParameterGolfDistributedLaneRefusalKind,
    };

    fn sample_model() -> Result<ParameterGolfReferenceModel, Box<dyn Error>> {
        let config = ParameterGolfConfig {
            vocab_size: 16,
            num_layers: 2,
            model_dim: 8,
            num_heads: 2,
            num_kv_heads: 1,
            mlp_mult: 2,
            max_context: 16,
            tie_embeddings: true,
            tied_embed_init_std: 0.005,
            logit_softcap: 30.0,
            rope_base: 10_000.0,
            qk_gain_init: 1.5,
        };
        let weights =
            ParameterGolfWeights::from_initializer(&config, ParameterGolfDeterministicInitializer::default())?;
        Ok(ParameterGolfReferenceModel::new(
            ModelDescriptor::new("parameter-golf-distributed-test", "parameter_golf_decoder", "test"),
            config,
            weights,
        )?)
    }

    fn sample_h100_device(ordinal: usize) -> DeviceDescriptor {
        DeviceDescriptor {
            backend: String::from("cuda"),
            device: Device::new(
                DeviceKind::Cuda,
                ordinal as u16,
                Some(format!("cuda:{ordinal}")),
            ),
            device_name: Some(String::from("NVIDIA H100 80GB HBM3")),
            supported_dtypes: vec![DType::F32, DType::BF16],
            supported_quantization: Vec::new(),
            memory_capacity_bytes: Some(80 * 1024 * 1024 * 1024),
            unified_memory: Some(false),
            feature_flags: vec![String::from("cuda_architecture_surface")],
            amd_metadata: None,
            nvidia_metadata: Some(NvidiaDeviceMetadata {
                topology: NvidiaTopologyInfo {
                    architecture: Some(String::from("hopper")),
                    compute_capability: Some(String::from("9.0")),
                    pci_bdf: Some(format!("00000000:{:02x}:00.0", ordinal + 1)),
                    sm_count: Some(132),
                    vram_bytes: Some(80 * 1024 * 1024 * 1024),
                    mig_profile: None,
                },
                risk: NvidiaRiskProfile {
                    level: NvidiaRiskLevel::Standard,
                    display_attached: Some(false),
                    mig_partitioned: false,
                    warnings: Vec::new(),
                },
                recovery: NvidiaRecoveryProfile {
                    supports_gpu_reset: Some(true),
                    expected_actions: vec![
                        NvidiaRecoveryAction::ProcessRestart,
                        NvidiaRecoveryAction::GpuReset,
                        NvidiaRecoveryAction::RebootHost,
                    ],
                },
            }),
        }
    }

    fn sample_non_h100_device(ordinal: usize) -> DeviceDescriptor {
        DeviceDescriptor {
            device_name: Some(String::from("NVIDIA A100 80GB")),
            ..sample_h100_device(ordinal)
        }
    }

    fn capability_profile() -> ClusterExecutionCapabilityProfile {
        ClusterExecutionCapabilityProfile::new("cuda")
            .with_supported_communication_classes(vec![ClusterCommunicationClass::TensorCollectiveMesh])
            .with_detail("single-node nccl-style all-reduce mesh")
    }

    fn measured_config() -> ParameterGolfDistributed8xH100Config {
        let mut config = ParameterGolfDistributed8xH100Config::challenge_defaults();
        config.run_id = format!("parameter-golf-distributed-test-{}", PARAMETER_GOLF_DISTRIBUTED_8XH100_VERSION);
        config.mesh_id = String::from("mesh.parameter_golf.test");
        config.geometry = ParameterGolfBatchGeometry {
            world_size: 8,
            train_batch_tokens: 128,
            validation_batch_tokens: 128,
            train_sequence_length: 4,
            grad_accum_steps: 1,
        };
        config.thresholds.max_total_wallclock_ms = 250;
        config.step_observations = vec![
            ParameterGolfDistributedStepObservation::new(1, 0, 40, 128),
            ParameterGolfDistributedStepObservation::new(2, 40, 85, 128),
        ];
        config.validation_observed_ms = 10;
        config.export_observed_ms = 5;
        config
    }

    #[test]
    fn parameter_golf_distributed_8xh100_lane_measures_ddp_posture() -> Result<(), Box<dyn Error>>
    {
        let model = sample_model()?;
        let devices = (0..8).map(sample_h100_device).collect::<Vec<_>>();
        let receipt = benchmark_parameter_golf_distributed_8xh100(
            model.descriptor(),
            &ParameterGolfTrainingHyperparameters::baseline_defaults(),
            devices.as_slice(),
            &capability_profile(),
            &measured_config(),
        )?;
        assert_eq!(receipt.disposition, ParameterGolfDistributedLaneDisposition::Measured);
        assert_eq!(
            receipt.topology.backend_selection.selection_state,
            BackendSelectionState::Direct
        );
        assert!(receipt.topology.all_devices_match_required_model);
        assert_eq!(receipt.communication.axes.len(), 1);
        assert_eq!(receipt.communication.axes[0].axis_id, "dp");
        assert_eq!(receipt.communication.axes[0].extent, 8);
        assert_eq!(receipt.communication.stages.len(), 3);
        assert_eq!(receipt.communication.stages[0].stage_id, "ddp_gradient_all_reduce");
        assert_eq!(
            receipt.communication.stages[1].stage_id,
            "muon_matrix_update_all_reduce"
        );
        assert!(!receipt.training_capability_report_digest.is_empty());
        assert!(receipt
            .challenge_kernel_blockers
            .contains(&String::from("cuda_bf16_train_precision_contract")));
        assert!(receipt.boundary_notes.iter().any(|note| {
            note.contains("cuda_rope_gqa_attention_block")
                || note.contains("cuda_bf16_train_precision_contract")
        }));
        assert!(receipt.timing.as_ref().is_some_and(|timing| timing.within_wallclock_cap));
        assert!(receipt.memory.as_ref().is_some_and(|memory| memory.within_device_budget));
        assert!(receipt.refusal.is_none());
        Ok(())
    }

    #[test]
    fn parameter_golf_distributed_8xh100_lane_refuses_non_h100_inventory()
    -> Result<(), Box<dyn Error>> {
        let model = sample_model()?;
        let devices = (0..8).map(sample_non_h100_device).collect::<Vec<_>>();
        let receipt = benchmark_parameter_golf_distributed_8xh100(
            model.descriptor(),
            &ParameterGolfTrainingHyperparameters::baseline_defaults(),
            devices.as_slice(),
            &capability_profile(),
            &measured_config(),
        )?;
        assert_eq!(receipt.disposition, ParameterGolfDistributedLaneDisposition::Refused);
        let refusal = receipt.refusal.expect("refusal should be present");
        assert_eq!(
            refusal.refusal_kind,
            ParameterGolfDistributedLaneRefusalKind::DeviceInventoryMismatch
        );
        Ok(())
    }

    #[test]
    fn parameter_golf_distributed_8xh100_lane_refuses_when_wallclock_exceeds_cap()
    -> Result<(), Box<dyn Error>> {
        let model = sample_model()?;
        let devices = (0..8).map(sample_h100_device).collect::<Vec<_>>();
        let mut config = measured_config();
        config.thresholds.max_total_wallclock_ms = 80;
        let receipt = benchmark_parameter_golf_distributed_8xh100(
            model.descriptor(),
            &ParameterGolfTrainingHyperparameters::baseline_defaults(),
            devices.as_slice(),
            &capability_profile(),
            &config,
        )?;
        assert_eq!(receipt.disposition, ParameterGolfDistributedLaneDisposition::Refused);
        assert!(receipt.timing.is_some());
        let refusal = receipt.refusal.expect("refusal should be present");
        assert_eq!(
            refusal.refusal_kind,
            ParameterGolfDistributedLaneRefusalKind::WallclockExceeded
        );
        Ok(())
    }

    #[test]
    fn parameter_golf_distributed_8xh100_lane_rejects_invalid_geometry()
    -> Result<(), Box<dyn Error>> {
        let model = sample_model()?;
        let devices = (0..8).map(sample_h100_device).collect::<Vec<_>>();
        let mut config = measured_config();
        config.geometry.grad_accum_steps = 2;
        let error = benchmark_parameter_golf_distributed_8xh100(
            model.descriptor(),
            &ParameterGolfTrainingHyperparameters::baseline_defaults(),
            devices.as_slice(),
            &capability_profile(),
            &config,
        )
        .expect_err("invalid geometry should be rejected");
        assert!(matches!(
            error,
            ParameterGolfDistributedLaneError::InvalidConfig { .. }
        ));
        Ok(())
    }
}
