use std::{
    cmp::Ordering,
    collections::BTreeMap,
    env,
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
    time::Instant,
};

use psionic_adapters::{
    AdapterArtifactIdentity, AdapterResidencyMode, AdapterServingBinding, AdapterTargetFamily,
    LmHeadLoraAdapterArtifact,
};
use psionic_backend_cpu::{decode_quantized_row_into, quantized_row_byte_len, quantized_row_dot};
use psionic_backend_cuda::{CudaBackend, CudaBuffer};
use psionic_backend_metal::{
    MetalBackend, MetalBuffer, MetalKvCacheMirror, MetalLogitsOutputMode,
    MetalQuantizedMatvecRequest,
};
use psionic_catalog::{BlobIntegrityPolicy, LocalBlobOpenOptions};
use psionic_core::{QuantizationMode, Shape};
use psionic_models::{
    DecoderModelDescriptor, GgufBlobArtifact, GgufDecoderAdapterLoader, GgufDecoderFamily,
    GgufDecoderFamilyMetadata, GgufDecoderLayerTensorLayout, GgufMetadataValue,
    GgufRuntimeTokenizer, ModelLoadError, PagedTensorStorage, TokenId, TokenSequence,
    TokenizerBoundary,
};
use psionic_runtime::DeviceDiscovery;
use psionic_train::{
    GemmaE4bCudaAdapterCheckpoint, GemmaE4bCudaAdapterExportedArtifact,
    GemmaE4bServedBaseModelBinding,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    CompiledWordGenerationModel, ContinuousBatchGenerationResult,
    CpuGgufGptOssTextGenerationService, GenerationEventStream, GenerationInput,
    GenerationModelHandle, GenerationRequest, GenerationResponse, GenerationStepOutput,
    GenerationStreamChunk, GenerationStreamEvent, GenerationStreamStatus, GenerationStreamTerminal,
    GenerationStreamingPolicy, InMemoryGenerationModelRegistry, InMemoryGenerationSessionStore,
    LoadedModelView, LoadedModelsObservation, LocalRuntimeObservability,
    ManagedTextGenerationRuntime, ReferenceTextGenerationError, ServedModelRevisionIdentity,
    SessionId, SharedPrefixStore, StreamingTextGenerationExecutor, TextGenerationExecutor,
    continuous_batch_text_generation_execution_profile, default_generation_scheduler_policy,
    default_generation_streaming_policy,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderAdapterRuntimeSupport {
    pub support_level: String,
    pub import_formats: Vec<String>,
    pub residency_modes: Vec<String>,
    pub batching_mode: String,
    pub unsupported_reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufDecoderRuntimeSupport {
    pub family: GgufDecoderFamily,
    pub supported_backends: Vec<String>,
    pub unsupported_backends: Vec<String>,
    pub unsupported_features: Vec<String>,
    pub quantization_modes: Vec<QuantizationMode>,
    pub adapter_runtime: DecoderAdapterRuntimeSupport,
}

#[derive(Clone, Debug)]
pub struct CpuGgufTextGenerationService {
    inner: CpuGgufServiceKind,
}

#[derive(Clone, Debug)]
enum CpuGgufServiceKind {
    GptOss(CpuGgufGptOssTextGenerationService),
    Dense(CpuDenseGgufTextGenerationService),
    Qwen35(CpuQwen35ProxyTextGenerationService),
}

#[derive(Clone, Debug)]
struct DenseAdapterRuntime {
    binding: AdapterServingBinding,
    adapter: LmHeadLoraAdapterArtifact,
    merged_delta: Option<DenseMatrix>,
}

impl DenseAdapterRuntime {
    fn new(
        binding: AdapterServingBinding,
        adapter: LmHeadLoraAdapterArtifact,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let merged_delta = matches!(binding.residency_mode, AdapterResidencyMode::MergedResident)
            .then(|| DenseMatrix {
                rows: adapter.vocab_size,
                columns: adapter.hidden_size,
                values: adapter.merged_output_delta(),
            });
        Ok(Self {
            binding,
            adapter,
            merged_delta,
        })
    }

    fn apply_to_logits(
        &self,
        hidden: &[f32],
        logits: &mut [f32],
    ) -> Result<(), ReferenceTextGenerationError> {
        if let Some(merged_delta) = self.merged_delta.as_ref() {
            merged_delta.matvec_add(hidden, logits)?;
            return Ok(());
        }
        self.adapter
            .apply_to_logits(hidden, logits)
            .map_err(
                |error| ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: self.binding.binding_id.clone(),
                    reason: error.to_string(),
                },
            )
    }
}

#[derive(Clone, Debug, Default)]
struct DenseAdapterRuntimeStore {
    runtimes: BTreeMap<String, DenseAdapterRuntime>,
}

impl DenseAdapterRuntimeStore {
    fn insert(&mut self, runtime: DenseAdapterRuntime) {
        self.runtimes
            .insert(runtime.binding.served_adapter_digest.clone(), runtime);
    }

    fn remove(&mut self, served_adapter_digest: &str) -> Option<DenseAdapterRuntime> {
        self.runtimes.remove(served_adapter_digest)
    }

    fn get(
        &self,
        binding: &AdapterServingBinding,
    ) -> Result<&DenseAdapterRuntime, ReferenceTextGenerationError> {
        self.runtimes
            .get(&binding.served_adapter_digest)
            .ok_or_else(|| ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: binding.binding_id.clone(),
                reason: format!(
                    "adapter binding `{}` is not registered on this CPU GGUF runtime",
                    binding.served_adapter_digest
                ),
            })
    }
}

#[derive(Clone, Debug)]
struct PromotedGemmaRevisionEntry {
    binding: AdapterServingBinding,
    identity: ServedModelRevisionIdentity,
}

#[derive(Clone, Debug, Default)]
struct PromotedGemmaRevisionState {
    entries: BTreeMap<String, PromotedGemmaRevisionEntry>,
    active_served_adapter_digest: Option<String>,
    last_known_good_served_adapter_digest: Option<String>,
}

impl PromotedGemmaRevisionState {
    fn promote(&mut self, entry: PromotedGemmaRevisionEntry) -> ServedModelRevisionIdentity {
        let digest = entry.binding.served_adapter_digest.clone();
        if let Some(active_digest) = self.active_served_adapter_digest.clone() {
            self.last_known_good_served_adapter_digest = Some(active_digest);
        } else {
            self.last_known_good_served_adapter_digest = Some(digest.clone());
        }
        self.entries.insert(digest.clone(), entry.clone());
        self.active_served_adapter_digest = Some(digest);
        entry.identity
    }

    fn rollback(&mut self) -> Option<ServedModelRevisionIdentity> {
        let rollback_digest = self.last_known_good_served_adapter_digest.clone()?;
        self.active_served_adapter_digest = Some(rollback_digest.clone());
        self.last_known_good_served_adapter_digest = Some(rollback_digest.clone());
        self.entries
            .get(rollback_digest.as_str())
            .map(|entry| entry.identity.clone())
    }

    fn active_binding(&self) -> Option<AdapterServingBinding> {
        self.active_entry().map(|entry| entry.binding.clone())
    }

    fn active_identity(&self) -> Option<ServedModelRevisionIdentity> {
        self.active_entry().map(|entry| entry.identity.clone())
    }

    fn last_known_good_identity(&self) -> Option<ServedModelRevisionIdentity> {
        self.last_known_good_served_adapter_digest
            .as_deref()
            .and_then(|digest| self.entries.get(digest))
            .map(|entry| entry.identity.clone())
    }

    fn entry_for_binding(
        &self,
        binding: &AdapterServingBinding,
    ) -> Option<&PromotedGemmaRevisionEntry> {
        self.entries.get(binding.served_adapter_digest.as_str())
    }

    fn attach_to_response(&self, mut response: GenerationResponse) -> GenerationResponse {
        let served_revision = response
            .provenance
            .as_ref()
            .and_then(|provenance| provenance.adapter_serving.as_ref())
            .and_then(|binding| self.entry_for_binding(binding))
            .map(|entry| entry.identity.clone());
        if let (Some(provenance), Some(served_revision)) =
            (response.provenance.as_mut(), served_revision)
        {
            provenance.served_revision = Some(served_revision);
        }
        response
    }

    fn active_entry(&self) -> Option<&PromotedGemmaRevisionEntry> {
        self.active_served_adapter_digest
            .as_deref()
            .and_then(|digest| self.entries.get(digest))
    }
}

impl CpuGgufTextGenerationService {
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, ReferenceTextGenerationError> {
        let family = GgufDecoderAdapterLoader
            .load_path(path.as_ref())?
            .family_metadata()
            .family;
        let inner = match family {
            GgufDecoderFamily::GptOss => CpuGgufServiceKind::GptOss(
                CpuGgufGptOssTextGenerationService::from_gguf_path(path)?,
            ),
            GgufDecoderFamily::Llama
            | GgufDecoderFamily::Qwen
            | GgufDecoderFamily::Mistral
            | GgufDecoderFamily::Gemma4 => {
                CpuGgufServiceKind::Dense(CpuDenseGgufTextGenerationService::from_gguf_path(path)?)
            }
            GgufDecoderFamily::Qwen35 => CpuGgufServiceKind::Qwen35(
                CpuQwen35ProxyTextGenerationService::from_gguf_path(path)?,
            ),
        };
        Ok(Self { inner })
    }

    pub fn load_model_from_gguf_path(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<(), ReferenceTextGenerationError> {
        *self = Self::from_gguf_path(path)?;
        Ok(())
    }

    #[must_use]
    pub fn model_descriptor(&self) -> &DecoderModelDescriptor {
        match &self.inner {
            CpuGgufServiceKind::GptOss(service) => service.model_descriptor(),
            CpuGgufServiceKind::Dense(service) => service.model_descriptor(),
            CpuGgufServiceKind::Qwen35(service) => service.model_descriptor(),
        }
    }

    #[must_use]
    pub fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        match &self.inner {
            CpuGgufServiceKind::GptOss(service) => runtime_support_for_descriptor(
                service.model_descriptor(),
                GgufDecoderFamily::GptOss,
                vec![
                    String::from("cpu"),
                    String::from("cuda"),
                    String::from("metal"),
                ],
                Vec::new(),
                unsupported_adapter_runtime_support(
                    "LM-head LoRA serving is currently implemented only on dense CPU GGUF families",
                ),
            ),
            CpuGgufServiceKind::Dense(service) => service.runtime_support(),
            CpuGgufServiceKind::Qwen35(service) => service.runtime_support(),
        }
    }

    #[must_use]
    pub fn plan_digest(&self, model_id: &str) -> Option<&str> {
        match &self.inner {
            CpuGgufServiceKind::GptOss(service) => service.plan_digest(model_id),
            CpuGgufServiceKind::Dense(service) => service.plan_digest(model_id),
            CpuGgufServiceKind::Qwen35(service) => service.plan_digest(model_id),
        }
    }

    pub fn create_session(
        &mut self,
        model_id: &str,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.create_session(model_id),
            CpuGgufServiceKind::Dense(service) => service.create_session(model_id),
            CpuGgufServiceKind::Qwen35(service) => service.create_session(model_id),
        }
    }

    pub fn reset_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.reset_session(session_id),
            CpuGgufServiceKind::Dense(service) => service.reset_session(session_id),
            CpuGgufServiceKind::Qwen35(service) => service.reset_session(session_id),
        }
    }

    pub fn close_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.close_session(session_id),
            CpuGgufServiceKind::Dense(service) => service.close_session(session_id),
            CpuGgufServiceKind::Qwen35(service) => service.close_session(session_id),
        }
    }

    #[must_use]
    pub fn loaded_model_views(&mut self) -> Vec<LoadedModelView> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.loaded_model_views(),
            CpuGgufServiceKind::Dense(service) => service.loaded_model_views(),
            CpuGgufServiceKind::Qwen35(service) => service.loaded_model_views(),
        }
    }

    pub fn generate_continuous_batch(
        &mut self,
        requests: Vec<GenerationRequest>,
    ) -> ContinuousBatchGenerationResult {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.generate_continuous_batch(requests),
            CpuGgufServiceKind::Dense(service) => service.generate_continuous_batch(requests),
            CpuGgufServiceKind::Qwen35(service) => service.generate_continuous_batch(requests),
        }
    }

    pub fn register_lm_head_lora_adapter(
        &mut self,
        binding_id: impl Into<String>,
        path: impl AsRef<Path>,
        identity: AdapterArtifactIdentity,
        alpha: f32,
        residency_mode: AdapterResidencyMode,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(_) => {
                Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: binding_id.into(),
                    reason: String::from(
                        "LM-head LoRA serving is currently supported only on dense CPU GGUF families",
                    ),
                })
            }
            CpuGgufServiceKind::Dense(service) => service.register_lm_head_lora_adapter(
                binding_id,
                path,
                identity,
                alpha,
                residency_mode,
            ),
            CpuGgufServiceKind::Qwen35(_) => {
                Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: binding_id.into(),
                    reason: String::from(
                        "LM-head LoRA serving is currently supported only on dense CPU GGUF families",
                    ),
                })
            }
        }
    }

    pub fn detach_adapter_binding(
        &mut self,
        served_adapter_digest: &str,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(_) => {
                Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: served_adapter_digest.to_string(),
                    reason: String::from(
                        "LM-head LoRA serving is currently supported only on dense CPU GGUF families",
                    ),
                })
            }
            CpuGgufServiceKind::Dense(service) => {
                service.detach_adapter_binding(served_adapter_digest)
            }
            CpuGgufServiceKind::Qwen35(_) => {
                Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: served_adapter_digest.to_string(),
                    reason: String::from(
                        "LM-head LoRA serving is currently supported only on dense CPU GGUF families",
                    ),
                })
            }
        }
    }

    pub fn merge_adapter_binding(
        &mut self,
        served_adapter_digest: &str,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(_) => {
                Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: served_adapter_digest.to_string(),
                    reason: String::from(
                        "LM-head LoRA serving is currently supported only on dense CPU GGUF families",
                    ),
                })
            }
            CpuGgufServiceKind::Dense(service) => {
                service.merge_adapter_binding(served_adapter_digest)
            }
            CpuGgufServiceKind::Qwen35(_) => {
                Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: served_adapter_digest.to_string(),
                    reason: String::from(
                        "LM-head LoRA serving is currently supported only on dense CPU GGUF families",
                    ),
                })
            }
        }
    }

    pub fn unmerge_adapter_binding(
        &mut self,
        served_adapter_digest: &str,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(_) => {
                Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: served_adapter_digest.to_string(),
                    reason: String::from(
                        "LM-head LoRA serving is currently supported only on dense CPU GGUF families",
                    ),
                })
            }
            CpuGgufServiceKind::Dense(service) => {
                service.unmerge_adapter_binding(served_adapter_digest)
            }
            CpuGgufServiceKind::Qwen35(_) => {
                Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: served_adapter_digest.to_string(),
                    reason: String::from(
                        "LM-head LoRA serving is currently supported only on dense CPU GGUF families",
                    ),
                })
            }
        }
    }
}

impl TextGenerationExecutor for CpuGgufTextGenerationService {
    type Error = ReferenceTextGenerationError;

    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResponse, Self::Error> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.generate(request),
            CpuGgufServiceKind::Dense(service) => service.generate(request),
            CpuGgufServiceKind::Qwen35(service) => service.generate(request),
        }
    }
}

impl StreamingTextGenerationExecutor for CpuGgufTextGenerationService {
    type Stream<'a> = Box<dyn GenerationEventStream + 'a>;

    fn generate_stream<'a>(
        &'a mut self,
        request: &GenerationRequest,
    ) -> Result<Self::Stream<'a>, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.generate_stream(request),
            CpuGgufServiceKind::Dense(service) => service.generate_stream(request),
            CpuGgufServiceKind::Qwen35(service) => service.generate_stream(request),
        }
    }
}

impl ManagedTextGenerationRuntime for CpuGgufTextGenerationService {
    fn isolation_policy(&self) -> psionic_runtime::LocalServingIsolationPolicy {
        match &self.inner {
            CpuGgufServiceKind::GptOss(service) => service.isolation_policy(),
            CpuGgufServiceKind::Dense(service) => service.isolation_policy(),
            CpuGgufServiceKind::Qwen35(service) => service.isolation_policy(),
        }
    }

    fn loaded_models(&mut self) -> LoadedModelsObservation {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.loaded_models(),
            CpuGgufServiceKind::Dense(service) => service.loaded_models(),
            CpuGgufServiceKind::Qwen35(service) => service.loaded_models(),
        }
    }

    fn observability(&mut self) -> LocalRuntimeObservability {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.observability(),
            CpuGgufServiceKind::Dense(service) => service.observability(),
            CpuGgufServiceKind::Qwen35(service) => service.observability(),
        }
    }

    fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.warm_model(model_id, keep_alive_millis),
            CpuGgufServiceKind::Dense(service) => service.warm_model(model_id, keep_alive_millis),
            CpuGgufServiceKind::Qwen35(service) => service.warm_model(model_id, keep_alive_millis),
        }
    }

    fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        match &mut self.inner {
            CpuGgufServiceKind::GptOss(service) => service.unload_model(model_id),
            CpuGgufServiceKind::Dense(service) => service.unload_model(model_id),
            CpuGgufServiceKind::Qwen35(service) => service.unload_model(model_id),
        }
    }
}

#[derive(Clone, Debug)]
struct CpuDenseGgufTextGenerationService {
    backend: super::CpuBackend,
    models: InMemoryGenerationModelRegistry<CpuDenseGgufGenerationModel>,
    sessions: InMemoryGenerationSessionStore,
    shared_prefixes: SharedPrefixStore,
    backend_health: super::BackendHealthTracker,
    model_descriptor: DecoderModelDescriptor,
    runtime_support: GgufDecoderRuntimeSupport,
    adapters: Arc<Mutex<DenseAdapterRuntimeStore>>,
}

impl CpuDenseGgufTextGenerationService {
    fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, ReferenceTextGenerationError> {
        let backend = super::CpuBackend::new();
        let adapters = Arc::new(Mutex::new(DenseAdapterRuntimeStore::default()));
        let model = CpuDenseGgufGenerationModel::from_gguf_path(path, Arc::clone(&adapters))?;
        let model_descriptor = model.descriptor().clone();
        let runtime_support = model.runtime_support();
        let mut models = InMemoryGenerationModelRegistry::new();
        models.warm_with_metadata(
            model,
            super::current_time_millis(),
            super::DEFAULT_MODEL_KEEPALIVE_MILLIS,
            None,
            Some(String::from("cpu")),
            None,
        )?;
        let mut backend_health = super::BackendHealthTracker::default();
        backend_health.observe("cpu", backend.health(), super::current_time_millis());
        Ok(Self {
            backend,
            models,
            sessions: InMemoryGenerationSessionStore::new(),
            shared_prefixes: SharedPrefixStore::default(),
            backend_health,
            model_descriptor,
            runtime_support,
            adapters,
        })
    }

    #[must_use]
    fn model_descriptor(&self) -> &DecoderModelDescriptor {
        &self.model_descriptor
    }

    #[must_use]
    fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        self.runtime_support.clone()
    }

    fn register_lm_head_lora_adapter(
        &mut self,
        binding_id: impl Into<String>,
        path: impl AsRef<Path>,
        identity: AdapterArtifactIdentity,
        alpha: f32,
        residency_mode: AdapterResidencyMode,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        let binding_id = binding_id.into();
        let served_artifact =
            crate::served_artifact_identity_for_decoder_backend(&self.model_descriptor, "cpu", &[]);
        validate_adapter_identity(
            &self.model_descriptor,
            self.runtime_support.family,
            served_artifact.served_artifact_digest.as_str(),
            &identity,
        )?;
        let adapter =
            LmHeadLoraAdapterArtifact::from_safetensors_path(path, identity.clone(), alpha)
                .map_err(
                    |error| ReferenceTextGenerationError::UnsupportedAdapterBinding {
                        binding_id: binding_id.clone(),
                        reason: error.to_string(),
                    },
                )?;
        validate_lm_head_lora_adapter(&self.model_descriptor, &adapter)?;
        let binding = AdapterServingBinding::new(
            binding_id,
            self.model_descriptor.model.model_id.clone(),
            self.model_descriptor.model.revision.clone(),
            served_artifact.served_artifact_digest,
            residency_mode,
            vec![identity],
        );
        self.adapters
            .lock()
            .map_err(
                |_| ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: binding.binding_id.clone(),
                    reason: String::from("adapter registry is poisoned"),
                },
            )?
            .insert(DenseAdapterRuntime::new(binding.clone(), adapter)?);
        Ok(binding)
    }

    fn detach_adapter_binding(
        &mut self,
        served_adapter_digest: &str,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        self.adapters
            .lock()
            .map_err(
                |_| ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: served_adapter_digest.to_string(),
                    reason: String::from("adapter registry is poisoned"),
                },
            )?
            .remove(served_adapter_digest)
            .map(|runtime| runtime.binding)
            .ok_or_else(|| ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: served_adapter_digest.to_string(),
                reason: String::from("adapter binding is not registered"),
            })
    }

    fn merge_adapter_binding(
        &mut self,
        served_adapter_digest: &str,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        self.rebind_adapter_residency(served_adapter_digest, AdapterResidencyMode::MergedResident)
    }

    fn unmerge_adapter_binding(
        &mut self,
        served_adapter_digest: &str,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        self.rebind_adapter_residency(served_adapter_digest, AdapterResidencyMode::HotSwapOverlay)
    }

    fn rebind_adapter_residency(
        &mut self,
        served_adapter_digest: &str,
        residency_mode: AdapterResidencyMode,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        let mut adapters = self.adapters.lock().map_err(|_| {
            ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: served_adapter_digest.to_string(),
                reason: String::from("adapter registry is poisoned"),
            }
        })?;
        let runtime = adapters.remove(served_adapter_digest).ok_or_else(|| {
            ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: served_adapter_digest.to_string(),
                reason: String::from("adapter binding is not registered"),
            }
        })?;
        let binding = AdapterServingBinding::new(
            runtime.binding.binding_id.clone(),
            runtime.binding.base_model_id.clone(),
            runtime.binding.base_model_revision.clone(),
            runtime.binding.base_served_artifact_digest.clone(),
            residency_mode,
            runtime.binding.adapters.clone(),
        );
        adapters.insert(DenseAdapterRuntime::new(binding.clone(), runtime.adapter)?);
        Ok(binding)
    }

    #[must_use]
    fn plan_digest(&self, model_id: &str) -> Option<&str> {
        self.models
            .active(model_id)
            .map(CpuDenseGgufGenerationModel::plan_digest)
    }

    fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Ok(self
            .models
            .warm_loaded(model_id, super::current_time_millis(), keep_alive_millis)?)
    }

    #[must_use]
    fn loaded_models(&mut self) -> LoadedModelsObservation {
        self.loaded_models_at(super::current_time_millis())
    }

    #[must_use]
    fn loaded_models_at(&mut self, now_millis: u64) -> LoadedModelsObservation {
        self.models.expire_idle(now_millis);
        self.models.loaded_models_observation()
    }

    #[must_use]
    fn observability(&mut self) -> LocalRuntimeObservability {
        self.observability_at(super::current_time_millis())
    }

    #[must_use]
    fn observability_at(&mut self, now_millis: u64) -> LocalRuntimeObservability {
        self.models.expire_idle(now_millis);
        self.backend_health
            .observe("cpu", self.backend.health(), now_millis);
        super::generation_runtime_observability(
            &self.models,
            &self.sessions,
            &self.backend_health,
            continuous_batch_text_generation_execution_profile(),
        )
    }

    #[must_use]
    fn loaded_model_views(&mut self) -> Vec<LoadedModelView> {
        self.loaded_model_views_at(super::current_time_millis())
    }

    #[must_use]
    fn loaded_model_views_at(&mut self, now_millis: u64) -> Vec<LoadedModelView> {
        self.models.expire_idle(now_millis);
        self.models.loaded_model_views()
    }

    fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Ok(self
            .models
            .unload_view(model_id, super::current_time_millis())?)
    }

    fn create_session(
        &mut self,
        model_id: &str,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        let model = self
            .models
            .active(model_id)
            .ok_or_else(|| ReferenceTextGenerationError::UnsupportedModel(model_id.to_string()))?;
        Ok(self.sessions.create(
            model,
            super::served_artifact_identity_for_decoder_backend(model.descriptor(), "cpu", &[])
                .served_artifact_digest,
        ))
    }

    fn reset_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        Ok(self.sessions.reset(session_id)?)
    }

    fn close_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        Ok(self.sessions.close(session_id)?)
    }

    fn generate_continuous_batch(
        &mut self,
        requests: Vec<GenerationRequest>,
    ) -> ContinuousBatchGenerationResult {
        super::run_continuous_batch_generation_requests(
            &mut self.backend,
            &mut self.models,
            &mut self.sessions,
            &mut self.shared_prefixes,
            requests,
            default_generation_scheduler_policy(),
        )
    }
}

impl TextGenerationExecutor for CpuDenseGgufTextGenerationService {
    type Error = ReferenceTextGenerationError;

    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResponse, Self::Error> {
        super::run_generation_request(
            &mut self.backend,
            &mut self.models,
            &mut self.sessions,
            &mut self.shared_prefixes,
            request,
        )
    }
}

impl StreamingTextGenerationExecutor for CpuDenseGgufTextGenerationService {
    type Stream<'a> = Box<dyn GenerationEventStream + 'a>;

    fn generate_stream<'a>(
        &'a mut self,
        request: &GenerationRequest,
    ) -> Result<Self::Stream<'a>, ReferenceTextGenerationError> {
        let response = self.generate(request)?;
        Ok(Box::new(CompletedGenerationStream::new(response)))
    }
}

impl ManagedTextGenerationRuntime for CpuDenseGgufTextGenerationService {
    fn loaded_models(&mut self) -> LoadedModelsObservation {
        Self::loaded_models(self)
    }

    fn observability(&mut self) -> LocalRuntimeObservability {
        Self::observability(self)
    }

    fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Self::warm_model(self, model_id, keep_alive_millis)
    }

    fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Self::unload_model(self, model_id)
    }
}

#[derive(Clone, Debug)]
struct CpuQwen35ProxyTextGenerationService {
    proxy: Arc<Qwen35LlamaCppProxyState>,
    model_descriptor: DecoderModelDescriptor,
    runtime_support: GgufDecoderRuntimeSupport,
    plan_digest: String,
    load_duration_ns: u64,
    sessions: InMemoryGenerationSessionStore,
    backend_health: crate::BackendHealthTracker,
    residency: psionic_runtime::LoadedModelResidency,
    memory_plan: psionic_runtime::ModelMemoryPlan,
    residency_policy: psionic_runtime::ModelResidencyPolicy,
}

#[derive(Debug)]
struct Qwen35LlamaCppProxyState {
    base_url: String,
    client: reqwest::blocking::Client,
    child: Mutex<Option<Child>>,
}

impl Drop for Qwen35LlamaCppProxyState {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.lock().ok().and_then(|mut child| child.take()) {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[derive(Debug, Deserialize)]
struct Qwen35ProxyCompletionResponse {
    content: String,
    #[serde(default)]
    tokens: Vec<u32>,
    #[serde(default)]
    stop_type: String,
    #[serde(default)]
    truncated: bool,
    #[serde(default)]
    tokens_evaluated: usize,
}

impl CpuQwen35ProxyTextGenerationService {
    fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, ReferenceTextGenerationError> {
        let load_start = Instant::now();
        let path = path.as_ref();
        let artifact = GgufBlobArtifact::open_path(path, gguf_local_blob_open_options())?;
        let adapter = GgufDecoderAdapterLoader.load_blob_artifact(&artifact)?;
        if !matches!(adapter.family_metadata().family, GgufDecoderFamily::Qwen35) {
            return Err(ModelLoadError::UnsupportedModel(
                adapter.descriptor().model.model_id.clone(),
            )
            .into());
        }

        let descriptor = adapter.descriptor().clone();
        let plan_digest = digest_qwen35_proxy_plan(&descriptor, adapter.family_metadata());
        let weight_bytes = std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or_default();
        let memory_plan = psionic_runtime::ModelMemoryPlan::host_only(weight_bytes, 0, 0);
        let residency_policy = psionic_runtime::ModelResidencyPolicy::default();
        let now_millis = crate::current_time_millis();
        let residency = psionic_runtime::LoadedModelResidency::ready(
            now_millis,
            crate::DEFAULT_MODEL_KEEPALIVE_MILLIS,
        );
        let runtime_support = qwen35_proxy_runtime_support(&descriptor);
        let mut backend_health = crate::BackendHealthTracker::default();
        backend_health.observe(
            "cpu",
            psionic_runtime::RuntimeHealth {
                status: psionic_runtime::HealthStatus::Ready,
                message: String::from("qwen35 llama.cpp proxy ready"),
            },
            now_millis,
        );
        Ok(Self {
            proxy: Qwen35LlamaCppProxyState::spawn(path, descriptor.config.max_context)?,
            model_descriptor: descriptor,
            runtime_support,
            plan_digest,
            load_duration_ns: load_start
                .elapsed()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
            sessions: InMemoryGenerationSessionStore::new(),
            backend_health,
            residency,
            memory_plan,
            residency_policy,
        })
    }

    #[must_use]
    fn model_descriptor(&self) -> &DecoderModelDescriptor {
        &self.model_descriptor
    }

    #[must_use]
    fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        self.runtime_support.clone()
    }

    #[must_use]
    fn plan_digest(&self, model_id: &str) -> Option<&str> {
        (model_id == self.model_descriptor.model.model_id).then_some(self.plan_digest.as_str())
    }

    fn create_session(
        &mut self,
        model_id: &str,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        if model_id != self.model_descriptor.model.model_id {
            return Err(ReferenceTextGenerationError::UnsupportedModel(
                model_id.to_string(),
            ));
        }
        Ok(self.sessions.create(
            &self.model_descriptor,
            crate::served_artifact_identity_for_decoder_backend(&self.model_descriptor, "cpu", &[])
                .served_artifact_digest,
        ))
    }

    fn reset_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        Ok(self.sessions.reset(session_id)?)
    }

    fn close_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        Ok(self.sessions.close(session_id)?)
    }

    #[must_use]
    fn loaded_model_views(&mut self) -> Vec<LoadedModelView> {
        vec![self.loaded_model_view()]
    }

    #[must_use]
    fn loaded_models(&mut self) -> LoadedModelsObservation {
        LoadedModelsObservation::new(vec![self.loaded_model_view().summary])
    }

    #[must_use]
    fn observability(&mut self) -> LocalRuntimeObservability {
        LocalRuntimeObservability {
            isolation_policy: psionic_runtime::LocalServingIsolationPolicy::subprocess_runtime(),
            cache_invalidation_policy: crate::cache_invalidation_policy(),
            execution_profile: continuous_batch_text_generation_execution_profile(),
            queue_depth: 0,
            queue_capacity: Some(
                continuous_batch_text_generation_execution_profile()
                    .queue_policy
                    .max_queued_requests,
            ),
            active_sessions: self.sessions.len(),
            active_requests: self.residency.active_requests,
            memory_footprint: self.residency_snapshot(),
            backend_health: self.backend_health.snapshot(),
            recent_transitions: self.backend_health.recent_changes(),
        }
    }

    fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        if model_id != self.model_descriptor.model.model_id {
            return Err(ReferenceTextGenerationError::UnsupportedModel(
                model_id.to_string(),
            ));
        }
        self.residency
            .refresh_keep_alive(keep_alive_millis, crate::current_time_millis());
        Ok(self.loaded_model_view())
    }

    fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        if model_id != self.model_descriptor.model.model_id {
            return Err(ReferenceTextGenerationError::UnsupportedModel(
                model_id.to_string(),
            ));
        }
        self.residency.expire_now(crate::current_time_millis());
        Ok(self.loaded_model_view())
    }

    fn generate_continuous_batch(
        &mut self,
        requests: Vec<GenerationRequest>,
    ) -> ContinuousBatchGenerationResult {
        let responses = requests
            .iter()
            .map(|request| self.generate(request))
            .collect::<Vec<_>>();
        ContinuousBatchGenerationResult {
            responses,
            scheduler_metrics: psionic_runtime::GenerationSchedulerMetrics::default(),
        }
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
    ) -> Result<GenerationResponse, ReferenceTextGenerationError> {
        if request.product_id != crate::TEXT_GENERATION_PRODUCT_ID {
            return Err(ReferenceTextGenerationError::UnsupportedProduct(
                request.product_id.clone(),
            ));
        }
        if request.model.model.model_id != self.model_descriptor.model.model_id {
            return Err(ReferenceTextGenerationError::UnsupportedModel(
                request.model.model.model_id.clone(),
            ));
        }
        if request.adapter_serving.is_some() {
            return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: request
                    .adapter_serving
                    .as_ref()
                    .map(|binding| binding.binding_id.clone())
                    .unwrap_or_else(|| String::from("unknown")),
                reason: String::from(
                    "LM-head LoRA serving is currently unsupported on the qwen35 proxy runtime",
                ),
            });
        }
        if request.session_id.is_some() || request.reset_session {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::UnsupportedStep(String::from(
                    "qwen35 proxy runtime does not implement session-bound KV reuse",
                )),
            ));
        }
        if request.options.structured_output.is_some() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::UnsupportedStep(String::from(
                    "qwen35 proxy runtime does not implement structured-output fallback",
                )),
            ));
        }

        let prompt =
            qwen35_proxy_prompt_json(&request.prompt, self.model_descriptor.config.vocab_size)?;
        let response_started = Instant::now();
        self.residency.begin_request(crate::current_time_millis());
        let upstream = self
            .proxy
            .complete(prompt, &request.options)
            .and_then(|response| {
                build_qwen35_proxy_generation_response(
                    request,
                    &self.model_descriptor,
                    &self.plan_digest,
                    &self.memory_plan,
                    self.residency_snapshot(),
                    self.load_duration_ns,
                    response_started
                        .elapsed()
                        .as_nanos()
                        .try_into()
                        .unwrap_or(u64::MAX),
                    response,
                )
            });
        self.residency.finish_request(crate::current_time_millis());
        upstream
    }
}

impl TextGenerationExecutor for CpuQwen35ProxyTextGenerationService {
    type Error = ReferenceTextGenerationError;

    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResponse, Self::Error> {
        Self::generate(self, request)
    }
}

impl StreamingTextGenerationExecutor for CpuQwen35ProxyTextGenerationService {
    type Stream<'a> = Box<dyn GenerationEventStream + 'a>;

    fn generate_stream<'a>(
        &'a mut self,
        request: &GenerationRequest,
    ) -> Result<Self::Stream<'a>, ReferenceTextGenerationError> {
        let response = self.generate(request)?;
        Ok(Box::new(CompletedGenerationStream::new(response)))
    }
}

impl ManagedTextGenerationRuntime for CpuQwen35ProxyTextGenerationService {
    fn isolation_policy(&self) -> psionic_runtime::LocalServingIsolationPolicy {
        psionic_runtime::LocalServingIsolationPolicy::subprocess_runtime()
    }

    fn loaded_models(&mut self) -> LoadedModelsObservation {
        Self::loaded_models(self)
    }

    fn observability(&mut self) -> LocalRuntimeObservability {
        Self::observability(self)
    }

    fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Self::warm_model(self, model_id, keep_alive_millis)
    }

    fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Self::unload_model(self, model_id)
    }
}

impl CpuQwen35ProxyTextGenerationService {
    fn loaded_model_view(&self) -> LoadedModelView {
        let mut summary = crate::LoadedModelSummary::from_decoder_descriptor(
            self.model_descriptor.model.model_id.clone(),
            &self.model_descriptor,
        );
        summary.size_bytes = Some(self.memory_plan.weights_bytes);
        summary.size_vram_bytes = Some(0);
        summary.backend = Some(String::from("cpu"));
        summary.fallback_state = Some(String::from("proxy_llama_cpp"));
        LoadedModelView {
            summary,
            residency: self.residency.clone(),
            memory_plan: self.memory_plan.clone(),
            residency_policy: self.residency_policy.clone(),
            residency_snapshot: self.residency_snapshot(),
        }
    }

    fn residency_snapshot(&self) -> psionic_runtime::MemoryResidencySnapshot {
        psionic_runtime::MemoryResidencySnapshot::from_loaded_models(&[
            psionic_runtime::LoadedModelMemoryState {
                model_id: self.model_descriptor.model.model_id.clone(),
                plan: self.memory_plan.clone(),
                active_requests: self.residency.active_requests,
                last_used_at_millis: self.residency.last_used_at_millis,
            },
        ])
    }
}

impl Qwen35LlamaCppProxyState {
    fn spawn(
        model_path: &Path,
        context_length: usize,
    ) -> Result<Arc<Self>, ReferenceTextGenerationError> {
        if let Ok(base_url) = env::var("PSIONIC_QWEN35_PROXY_BASE_URL") {
            let state = Arc::new(Self {
                base_url,
                client: reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(600))
                    .build()
                    .map_err(qwen35_proxy_runtime_error)?,
                child: Mutex::new(None),
            });
            state.wait_until_ready()?;
            return Ok(state);
        }

        let internal_port = reserve_proxy_port()?;
        let host = "127.0.0.1";
        let mut command = Command::new(qwen35_llama_server_bin());
        command
            .arg("-m")
            .arg(model_path)
            .arg("--host")
            .arg(host)
            .arg("--port")
            .arg(internal_port.to_string())
            .arg("-c")
            .arg(context_length.to_string())
            .arg("-ngl")
            .arg("0")
            .arg("--no-mmproj")
            .arg("--no-webui")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let child = command.spawn().map_err(qwen35_proxy_runtime_error)?;
        let state = Arc::new(Self {
            base_url: format!("http://{host}:{internal_port}"),
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(600))
                .build()
                .map_err(qwen35_proxy_runtime_error)?,
            child: Mutex::new(Some(child)),
        });
        state.wait_until_ready()?;
        Ok(state)
    }

    fn wait_until_ready(&self) -> Result<(), ReferenceTextGenerationError> {
        let health_url = format!("{}/health", self.base_url);
        let completion_url = format!("{}/completion", self.base_url);
        let probe = serde_json::json!({
            "prompt": "hello",
            "n_predict": 1,
            "temperature": 0.0,
            "cache_prompt": false,
            "return_tokens": true,
        });
        let health_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .map_err(qwen35_proxy_runtime_error)?;
        let completion_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(qwen35_proxy_runtime_error)?;
        for _ in 0..300 {
            let health_ready = matches!(
                health_client.get(health_url.as_str()).send(),
                Ok(response) if response.status().is_success()
            );
            if health_ready {
                match completion_client
                    .post(completion_url.as_str())
                    .json(&probe)
                    .send()
                {
                    Ok(response) if response.status().is_success() => return Ok(()),
                    Ok(response)
                        if response.status() != reqwest::StatusCode::SERVICE_UNAVAILABLE =>
                    {
                        return Err(ReferenceTextGenerationError::Runtime(
                            crate::RuntimeError::Backend(format!(
                                "qwen35 llama.cpp proxy readiness probe failed with status {}",
                                response.status()
                            )),
                        ));
                    }
                    Ok(_) | Err(_) => {}
                }
            }
            thread::sleep(Duration::from_millis(200));
        }
        Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "qwen35 llama.cpp proxy did not become ready: {completion_url}"
            )),
        ))
    }

    fn complete(
        &self,
        prompt: serde_json::Value,
        options: &crate::GenerationOptions,
    ) -> Result<Qwen35ProxyCompletionResponse, ReferenceTextGenerationError> {
        let mut body = serde_json::json!({
            "prompt": prompt,
            "n_predict": options.max_output_tokens,
            "cache_prompt": false,
            "return_tokens": true,
            "stream": false,
        });
        if matches!(options.decode_strategy, crate::DecodeStrategy::Greedy) {
            body["temperature"] = serde_json::json!(0.0_f32);
        } else if let Some(temperature) = options.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(top_k) = options.top_k {
            body["top_k"] = serde_json::json!(top_k);
        }
        if let Some(top_p) = options.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(min_p) = options.min_p {
            body["min_p"] = serde_json::json!(min_p);
        }
        if let Some(typical_p) = options.typical_p {
            body["typical_p"] = serde_json::json!(typical_p);
        }
        if let Some(mirostat) = options.mirostat {
            body["mirostat"] = serde_json::json!(mirostat);
        }
        if let Some(mirostat_tau) = options.mirostat_tau {
            body["mirostat_tau"] = serde_json::json!(mirostat_tau);
        }
        if let Some(mirostat_eta) = options.mirostat_eta {
            body["mirostat_eta"] = serde_json::json!(mirostat_eta);
        }
        if let Some(repeat_penalty) = options.repeat_penalty {
            body["repeat_penalty"] = serde_json::json!(repeat_penalty);
        }
        if let Some(repeat_last_n) = options.repeat_last_n {
            body["repeat_last_n"] = serde_json::json!(repeat_last_n);
        }
        if let Some(presence_penalty) = options.presence_penalty {
            body["presence_penalty"] = serde_json::json!(presence_penalty);
        }
        if let Some(frequency_penalty) = options.frequency_penalty {
            body["frequency_penalty"] = serde_json::json!(frequency_penalty);
        }
        if let Some(seed) = options.seed {
            body["seed"] = serde_json::json!(seed);
        }
        if !options.stop_sequences.is_empty() {
            body["stop"] = serde_json::json!(options.stop_sequences);
        }
        let response = self
            .client
            .post(format!("{}/completion", self.base_url))
            .json(&body)
            .send()
            .map_err(qwen35_proxy_runtime_error)?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 llama.cpp completion request failed with status {status}: {body}"
                )),
            ));
        }
        response.json().map_err(qwen35_proxy_runtime_error)
    }
}

#[derive(Clone, Debug)]
struct CpuDenseGgufGenerationModel {
    inner: Arc<DenseGgufModelInner>,
    adapters: Arc<Mutex<DenseAdapterRuntimeStore>>,
}

impl CpuDenseGgufGenerationModel {
    fn from_gguf_path(
        path: impl AsRef<Path>,
        adapters: Arc<Mutex<DenseAdapterRuntimeStore>>,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let artifact = GgufBlobArtifact::open_path(path, gguf_local_blob_open_options())?;
        Self::from_blob_artifact(artifact, adapters)
    }

    fn from_blob_artifact(
        artifact: GgufBlobArtifact,
        adapters: Arc<Mutex<DenseAdapterRuntimeStore>>,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let load_start = Instant::now();
        let adapter = GgufDecoderAdapterLoader.load_blob_artifact(&artifact)?;
        if matches!(adapter.family_metadata().family, GgufDecoderFamily::GptOss) {
            return Err(ModelLoadError::UnsupportedModel(
                adapter.descriptor().model.model_id.clone(),
            )
            .into());
        }
        let tokenizer = GgufRuntimeTokenizer::from_gguf(adapter.tokenizer()).map_err(|error| {
            ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!("failed to build runtime tokenizer: {error}"),
            }
        })?;
        let token_embedding =
            ProjectionMatrix::load(&artifact, adapter.tensor_layout().token_embedding.as_str())?;
        let output = if let Some(name) = adapter.tensor_layout().output.as_ref() {
            ProjectionMatrix::load(&artifact, name)?
        } else {
            token_embedding.clone()
        };
        let descriptor = adapter.descriptor().clone();
        let dense_gemma4_per_layer_inputs =
            if matches!(adapter.family_metadata().family, GgufDecoderFamily::Gemma4) {
                DenseGemma4PerLayerInputs::load(&artifact)?
            } else {
                None
            };
        let mut cache_offset = 0usize;
        let mut last_swa_cache_offset = None;
        let mut last_full_cache_offset = None;
        let mut last_swa_kv_layer_index = None;
        let mut last_full_kv_layer_index = None;
        let mut layers = Vec::with_capacity(adapter.tensor_layout().layers.len());
        for (layer_index, layout) in adapter.tensor_layout().layers.iter().enumerate() {
            let query_weight_name = required_tensor_name(
                layout.attention_query_weight.as_deref(),
                "attention_query_weight",
            )?;
            let query_rows = artifact
                .paged_tensor(query_weight_name)?
                .metadata()
                .shape
                .dims()[0];
            let layer_head_dim = query_rows
                .checked_div(descriptor.config.block.attention.head_count)
                .unwrap_or(0);
            let is_swa =
                gemma4_layer_is_swa(adapter.family_metadata(), layer_index, layer_head_dim);
            let has_kv = gemma4_layer_has_kv(adapter.family_metadata(), layout);
            let (cache_read_offset, cache_write_offset, reuse_layer_index) = if has_kv {
                (cache_offset, Some(cache_offset), None)
            } else if is_swa {
                (
                    last_swa_cache_offset.ok_or_else(|| ModelLoadError::ArtifactFormat {
                        format: String::from("gguf"),
                        message: format!(
                            "gemma4 swa layer {layer_index} reuses kv before any swa kv source was loaded"
                        ),
                    })?,
                    None,
                    Some(last_swa_kv_layer_index.ok_or_else(|| ModelLoadError::ArtifactFormat {
                        format: String::from("gguf"),
                        message: format!(
                            "gemma4 swa layer {layer_index} is missing a reused kv source layer index"
                        ),
                    })?),
                )
            } else {
                (
                    last_full_cache_offset.ok_or_else(|| ModelLoadError::ArtifactFormat {
                        format: String::from("gguf"),
                        message: format!(
                            "gemma4 full-attention layer {layer_index} reuses kv before any full-attention kv source was loaded"
                        ),
                    })?,
                    None,
                    Some(
                        last_full_kv_layer_index.ok_or_else(|| ModelLoadError::ArtifactFormat {
                            format: String::from("gguf"),
                            message: format!(
                                "gemma4 full-attention layer {layer_index} is missing a reused kv source layer index"
                            ),
                        })?,
                    ),
                )
            };
            let layer = DenseGgufLayer::load(
                &artifact,
                layout,
                &descriptor,
                adapter.family_metadata(),
                layer_index,
                cache_read_offset,
                cache_write_offset,
                reuse_layer_index,
            )?;
            if layer.attention_geometry.has_kv() {
                if is_swa {
                    last_swa_cache_offset = layer.attention_geometry.cache_write_offset;
                    last_swa_kv_layer_index = Some(layer_index);
                } else {
                    last_full_cache_offset = layer.attention_geometry.cache_write_offset;
                    last_full_kv_layer_index = Some(layer_index);
                }
            }
            cache_offset = cache_offset.saturating_add(layer.cache_write_width());
            layers.push(layer);
        }
        let inner = DenseGgufModelInner {
            descriptor: descriptor.clone(),
            family_metadata: adapter.family_metadata().clone(),
            tokenizer,
            token_embedding,
            gemma4_per_layer_inputs: dense_gemma4_per_layer_inputs,
            rope_freq_factors: load_named_optional_dense_vector(&artifact, "rope_freqs.weight")?,
            output_norm: load_dense_vector(
                &artifact,
                adapter.tensor_layout().output_norm.as_str(),
            )?,
            output,
            layers,
            plan_digest: digest_dense_gguf_plan(&descriptor, adapter.family_metadata()),
            load_duration_ns: load_start
                .elapsed()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
        };
        Ok(Self {
            inner: Arc::new(inner),
            adapters,
        })
    }

    #[must_use]
    fn plan_digest(&self) -> &str {
        self.inner.plan_digest.as_str()
    }

    #[must_use]
    fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        runtime_support_for_descriptor(
            &self.inner.descriptor,
            self.inner.family_metadata.family,
            vec![String::from("cpu")],
            vec![String::from("cuda"), String::from("metal")],
            dense_adapter_runtime_support(),
        )
    }
}

impl crate::GenerationModelHandle for CpuDenseGgufGenerationModel {
    fn descriptor(&self) -> &DecoderModelDescriptor {
        &self.inner.descriptor
    }

    fn cache_width(&self) -> usize {
        self.inner.cache_width()
    }
}

impl super::CompiledWordGenerationModel for CpuDenseGgufGenerationModel {
    type Backend = super::CpuBackend;

    fn tokenizer(&self) -> &dyn TokenizerBoundary {
        &self.inner.tokenizer
    }

    fn encode_prompt_input(
        &self,
        input: &GenerationInput,
    ) -> Result<TokenSequence, ReferenceTextGenerationError> {
        Ok(match input {
            GenerationInput::Text(text) => self.inner.tokenizer.encode_with_defaults(text),
            GenerationInput::Tokens(tokens) => tokens.clone(),
        })
    }

    fn is_end_of_sequence(&self, token: TokenId) -> bool {
        self.inner.tokenizer.is_end_of_sequence(token)
    }

    fn execute_step(
        &self,
        _backend: &mut Self::Backend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
    ) -> Result<GenerationStepOutput, ReferenceTextGenerationError> {
        let config = &self.inner.descriptor.config;
        if token.as_u32() as usize >= config.vocab_size {
            return Err(ReferenceTextGenerationError::InvalidToken {
                token: token.as_u32(),
                vocab_size: config.vocab_size,
            });
        }
        if position >= config.max_context {
            return Err(ReferenceTextGenerationError::InvalidPosition {
                position,
                max_context: config.max_context,
            });
        }
        if cache.width() != self.inner.cache_width() {
            return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
                expected_kv_width: self.inner.cache_width(),
                kv_width: cache.width(),
            });
        }
        let step = self.inner.forward_step(token, position, cache)?;
        Ok(GenerationStepOutput {
            key: step.key,
            value: step.value,
            logits: step.logits,
            hidden: Some(step.final_hidden),
            execution_plan_digest: Some(self.inner.plan_digest.clone()),
            compile_path: None,
            kernel_count: step.kernel_count,
            bytes_moved: step.bytes_moved,
            plan_cache_hits: 0,
            plan_cache_misses: 0,
            gpt_oss_perf: None,
        })
    }

    fn plan_digest(&self) -> &str {
        self.plan_digest()
    }

    fn load_duration_ns(&self) -> u64 {
        self.inner.load_duration_ns
    }

    fn backend_compatibility(&self) -> &'static str {
        "cpu"
    }

    fn adjust_step_output(
        &self,
        step: &mut GenerationStepOutput,
        request: &GenerationRequest,
    ) -> Result<(), ReferenceTextGenerationError> {
        let Some(binding) = request.adapter_serving.as_ref() else {
            return Ok(());
        };
        let hidden = step.hidden.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: binding.binding_id.clone(),
                reason: String::from(
                    "the active dense GGUF step does not expose the final hidden state needed for LM-head LoRA serving",
                ),
            }
        })?;
        let adapters = self.adapters.lock().map_err(|_| {
            ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: binding.binding_id.clone(),
                reason: String::from("adapter registry is poisoned"),
            }
        })?;
        let runtime = adapters.get(binding)?;
        runtime.apply_to_logits(hidden.as_slice(), step.logits.as_mut_slice())
    }
}

#[derive(Clone, Debug)]
struct DenseGgufModelInner {
    descriptor: DecoderModelDescriptor,
    family_metadata: GgufDecoderFamilyMetadata,
    tokenizer: GgufRuntimeTokenizer,
    token_embedding: ProjectionMatrix,
    gemma4_per_layer_inputs: Option<DenseGemma4PerLayerInputs>,
    rope_freq_factors: Option<Vec<f32>>,
    output_norm: Vec<f32>,
    output: ProjectionMatrix,
    layers: Vec<DenseGgufLayer>,
    plan_digest: String,
    load_duration_ns: u64,
}

impl DenseGgufModelInner {
    fn rope_freq_factors_for_layer(&self, layer_index: usize, head_dim: usize) -> Option<&[f32]> {
        (self.family_metadata.family == GgufDecoderFamily::Gemma4
            && !gemma4_layer_is_swa(&self.family_metadata, layer_index, head_dim))
        .then_some(self.rope_freq_factors.as_deref())
        .flatten()
    }

    fn cache_width(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.attention_geometry.cache_end())
            .max()
            .unwrap_or(0)
    }

    fn forward_step(
        &self,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
    ) -> Result<DenseGgufForwardStep, ReferenceTextGenerationError> {
        let mut bytes_moved = self.token_embedding.byte_length() as u64;
        let mut kernel_count = 1usize;
        let mut hidden = self.token_embedding.decode_row(token.as_u32() as usize)?;
        scale_in_place(
            &mut hidden,
            input_embedding_scale(&self.family_metadata, self.descriptor.config.hidden_size),
        );
        let gemma4_per_layer_inputs =
            if let Some(per_layer_inputs) = self.gemma4_per_layer_inputs.as_ref() {
                Some(per_layer_inputs.project(
                    token,
                    hidden.as_slice(),
                    self.layers.len(),
                    self.descriptor.config.hidden_size,
                    self.family_metadata.rms_norm_epsilon,
                )?)
            } else {
                None
            };
        let mut cache_key = vec![0.0; self.cache_width()];
        let mut cache_value = vec![0.0; self.cache_width()];
        let mut live_keys: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        let mut live_values: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            let hidden_norm = rms_norm(
                hidden.as_slice(),
                layer.attention_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );

            let mut q = Vec::new();
            layer.attention_query_weight.matvec(&hidden_norm, &mut q)?;
            if let Some(bias) = layer.attention_query_bias.as_ref() {
                add_bias_in_place(&mut q, bias.as_slice());
            }
            if let Some(norm) = layer.attention_query_norm.as_ref() {
                per_head_rms_norm_in_place(
                    q.as_mut_slice(),
                    layer.attention_geometry.head_count,
                    layer.attention_geometry.head_dim,
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }

            apply_rope_neox(
                &mut q,
                layer.attention_geometry.head_count,
                layer.attention_geometry.head_dim,
                layer.attention_geometry.rotary_dim,
                position,
                layer.attention_geometry.rope_theta,
                self.rope_freq_factors_for_layer(layer_index, layer.attention_geometry.head_dim),
                &self.family_metadata,
            );
            let (k, v) = if layer.attention_geometry.has_kv() {
                let mut k = Vec::new();
                layer.attention_key_weight.matvec(&hidden_norm, &mut k)?;
                if let Some(bias) = layer.attention_key_bias.as_ref() {
                    add_bias_in_place(&mut k, bias.as_slice());
                }
                if let Some(norm) = layer.attention_key_norm.as_ref() {
                    per_head_rms_norm_in_place(
                        k.as_mut_slice(),
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }

                let mut v = Vec::new();
                layer.attention_value_weight.matvec(&hidden_norm, &mut v)?;
                if let Some(bias) = layer.attention_value_bias.as_ref() {
                    add_bias_in_place(&mut v, bias.as_slice());
                }
                if self.family_metadata.family == GgufDecoderFamily::Gemma4 {
                    per_head_rms_norm_unit_in_place(
                        v.as_mut_slice(),
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        self.family_metadata.rms_norm_epsilon,
                    );
                }
                apply_rope_neox(
                    &mut k,
                    layer.attention_geometry.kv_head_count,
                    layer.attention_geometry.head_dim,
                    layer.attention_geometry.rotary_dim,
                    position,
                    layer.attention_geometry.rope_theta,
                    self.rope_freq_factors_for_layer(
                        layer_index,
                        layer.attention_geometry.head_dim,
                    ),
                    &self.family_metadata,
                );
                if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                    let kv_width = layer.attention_geometry.kv_width();
                    cache_key[cache_offset..cache_offset + kv_width].copy_from_slice(k.as_slice());
                    cache_value[cache_offset..cache_offset + kv_width]
                        .copy_from_slice(v.as_slice());
                }
                live_keys[layer_index] = Some(k.clone());
                live_values[layer_index] = Some(v.clone());
                (k, v)
            } else {
                let reuse_layer_index =
                    layer.attention_geometry.reuse_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} is missing reused kv source metadata"
                            ),
                        ))
                    })?;
                let reused_k = live_keys
                    .get(reuse_layer_index)
                    .and_then(|value| value.clone())
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} expected live key cache from layer {reuse_layer_index}"
                            ),
                        ))
                    })?;
                let reused_v = live_values
                    .get(reuse_layer_index)
                    .and_then(|value| value.clone())
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} expected live value cache from layer {reuse_layer_index}"
                            ),
                        ))
                    })?;
                (reused_k, reused_v)
            };

            let attention = attend_impl(
                layer_index,
                q.as_slice(),
                k.as_slice(),
                v.as_slice(),
                cache,
                layer.attention_geometry.head_count,
                layer.attention_geometry.kv_head_count,
                layer.attention_geometry.head_dim,
                layer.attention_geometry.cache_read_offset,
                layer.attention_geometry.sliding_window,
                attention_scale(&self.family_metadata, layer.attention_geometry.head_dim),
            );
            let mut attention_out = Vec::new();
            layer
                .attention_output_weight
                .matvec(attention.as_slice(), &mut attention_out)?;
            if let Some(bias) = layer.attention_output_bias.as_ref() {
                add_bias_in_place(&mut attention_out, bias.as_slice());
            }
            if let Some(norm) = layer.attention_post_norm.as_ref() {
                rms_norm_in_place(
                    attention_out.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(attention_out.as_mut_slice(), residual.as_slice())?;
            hidden = attention_out;

            let ffn_residual = hidden.clone();
            let ffn_input = rms_norm(
                hidden.as_slice(),
                layer.feed_forward_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let mut gate = Vec::new();
            layer
                .feed_forward_gate_weight
                .matvec(&ffn_input, &mut gate)?;
            let mut up = Vec::new();
            layer.feed_forward_up_weight.matvec(&ffn_input, &mut up)?;
            let activated =
                feed_forward_activation(&self.family_metadata, gate.as_slice(), up.as_slice());
            let mut ffn_out = Vec::new();
            layer
                .feed_forward_down_weight
                .matvec(activated.as_slice(), &mut ffn_out)?;
            if let Some(norm) = layer.feed_forward_post_norm.as_ref() {
                rms_norm_in_place(
                    ffn_out.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(ffn_out.as_mut_slice(), ffn_residual.as_slice())?;
            hidden = ffn_out;

            if let Some(per_layer_inputs) = gemma4_per_layer_inputs.as_ref() {
                if let (Some(input_gate), Some(proj), Some(post_norm)) = (
                    layer.per_layer_input_gate.as_ref(),
                    layer.per_layer_proj.as_ref(),
                    layer.per_layer_post_norm.as_ref(),
                ) {
                    let mut gated = Vec::new();
                    input_gate.matvec(hidden.as_slice(), &mut gated)?;
                    for value in &mut gated {
                        *value = approximate_gelu(*value);
                    }
                    let gated = multiply_vectors(
                        gated.as_slice(),
                        self.gemma4_per_layer_inputs
                            .as_ref()
                            .expect("per-layer config")
                            .layer_slice(per_layer_inputs.as_slice(), layer_index),
                    )?;
                    let mut projected = Vec::new();
                    proj.matvec(gated.as_slice(), &mut projected)?;
                    rms_norm_in_place(
                        projected.as_mut_slice(),
                        post_norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                    hidden = add_vectors(hidden.as_slice(), projected.as_slice())?;
                    bytes_moved = bytes_moved
                        .saturating_add(input_gate.byte_length() as u64)
                        .saturating_add(proj.byte_length() as u64);
                    kernel_count = kernel_count.saturating_add(2);
                }
            }
            if let Some(scale) = layer.layer_output_scale {
                scale_in_place(&mut hidden, scale);
            }

            bytes_moved = bytes_moved
                .saturating_add(layer.attention_query_weight.byte_length() as u64)
                .saturating_add(layer.attention_output_weight.byte_length() as u64)
                .saturating_add(layer.feed_forward_gate_weight.byte_length() as u64)
                .saturating_add(layer.feed_forward_up_weight.byte_length() as u64)
                .saturating_add(layer.feed_forward_down_weight.byte_length() as u64);
            if layer.attention_geometry.has_kv() {
                bytes_moved = bytes_moved
                    .saturating_add(layer.attention_key_weight.byte_length() as u64)
                    .saturating_add(layer.attention_value_weight.byte_length() as u64);
                kernel_count = kernel_count.saturating_add(7);
            } else {
                kernel_count = kernel_count.saturating_add(5);
            }
        }

        let final_hidden = rms_norm(
            hidden.as_slice(),
            self.output_norm.as_slice(),
            self.family_metadata.rms_norm_epsilon,
        );
        let mut logits = Vec::new();
        self.output.matvec(final_hidden.as_slice(), &mut logits)?;
        apply_final_logit_softcapping_in_place(
            logits.as_mut_slice(),
            self.family_metadata.final_logit_softcapping,
        );
        bytes_moved = bytes_moved.saturating_add(self.output.byte_length() as u64);
        kernel_count = kernel_count.saturating_add(1);

        Ok(DenseGgufForwardStep {
            key: cache_key,
            value: cache_value,
            logits,
            final_hidden,
            kernel_count,
            bytes_moved,
        })
    }
}

#[derive(Clone, Debug)]
struct DenseAttentionGeometry {
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    sliding_window: Option<usize>,
    cache_read_offset: usize,
    cache_write_offset: Option<usize>,
    reuse_layer_index: Option<usize>,
}

impl DenseAttentionGeometry {
    fn kv_width(&self) -> usize {
        self.kv_head_count.saturating_mul(self.head_dim)
    }

    fn cache_write_width(&self) -> usize {
        self.cache_write_offset
            .map(|_| self.kv_width())
            .unwrap_or(0)
    }

    fn cache_end(&self) -> usize {
        self.cache_read_offset.saturating_add(self.kv_width())
    }

    fn has_kv(&self) -> bool {
        self.cache_write_offset.is_some()
    }
}

#[derive(Clone, Debug)]
struct DenseGgufLayer {
    attention_geometry: DenseAttentionGeometry,
    attention_norm: Vec<f32>,
    attention_query_weight: ProjectionMatrix,
    attention_query_bias: Option<Vec<f32>>,
    attention_query_norm: Option<Vec<f32>>,
    attention_key_weight: ProjectionMatrix,
    attention_key_bias: Option<Vec<f32>>,
    attention_key_norm: Option<Vec<f32>>,
    attention_value_weight: ProjectionMatrix,
    attention_value_bias: Option<Vec<f32>>,
    attention_output_weight: ProjectionMatrix,
    attention_output_bias: Option<Vec<f32>>,
    attention_post_norm: Option<Vec<f32>>,
    feed_forward_norm: Vec<f32>,
    feed_forward_gate_weight: ProjectionMatrix,
    feed_forward_up_weight: ProjectionMatrix,
    feed_forward_down_weight: ProjectionMatrix,
    feed_forward_post_norm: Option<Vec<f32>>,
    layer_output_scale: Option<f32>,
    per_layer_input_gate: Option<ProjectionMatrix>,
    per_layer_proj: Option<ProjectionMatrix>,
    per_layer_post_norm: Option<Vec<f32>>,
}

impl DenseGgufLayer {
    fn load(
        artifact: &GgufBlobArtifact,
        layout: &GgufDecoderLayerTensorLayout,
        descriptor: &DecoderModelDescriptor,
        family_metadata: &GgufDecoderFamilyMetadata,
        layer_index: usize,
        cache_read_offset: usize,
        cache_write_offset: Option<usize>,
        reuse_layer_index: Option<usize>,
    ) -> Result<Self, ModelLoadError> {
        let has_kv = cache_write_offset.is_some();
        let attention_query_weight = ProjectionMatrix::load(
            artifact,
            required_tensor_name(
                layout.attention_query_weight.as_deref(),
                "attention_query_weight",
            )?,
        )?;
        let query_width = attention_query_weight.rows();
        let head_dim = query_width
            .checked_div(descriptor.config.block.attention.head_count)
            .unwrap_or(0);
        let attention_key_weight = if let Some(name) = layout.attention_key_weight.as_deref() {
            ProjectionMatrix::load(artifact, name)?
        } else if family_metadata.family == GgufDecoderFamily::Gemma4 && !has_kv {
            attention_query_weight.clone()
        } else {
            return Err(ModelLoadError::MissingTensor(String::from(
                "attention_key_weight",
            )));
        };
        let key_width = if layout.attention_key_weight.is_some() {
            attention_key_weight.rows()
        } else {
            gemma4_expected_kv_width(descriptor, family_metadata, layer_index, head_dim)
        };
        let attention_value_weight = if let Some(name) = layout.attention_value_weight.as_deref() {
            ProjectionMatrix::load(artifact, name)?
        } else if family_metadata.family == GgufDecoderFamily::Gemma4 {
            attention_key_weight.clone()
        } else if has_kv {
            return Err(ModelLoadError::MissingTensor(String::from(
                "attention_value_weight",
            )));
        } else {
            attention_key_weight.clone()
        };
        let value_width =
            if layout.attention_value_weight.is_some() || layout.attention_key_weight.is_some() {
                attention_value_weight.rows()
            } else {
                key_width
            };
        let attention_geometry = dense_attention_geometry(
            descriptor,
            family_metadata,
            layer_index,
            query_width,
            key_width,
            value_width,
            cache_read_offset,
            cache_write_offset,
            reuse_layer_index,
        )?;
        Ok(Self {
            attention_geometry,
            attention_norm: load_dense_vector(artifact, layout.attention_norm.as_str())?,
            attention_query_weight,
            attention_query_bias: load_optional_dense_vector(
                artifact,
                layout.attention_query_bias.as_deref(),
            )?,
            attention_query_norm: load_optional_dense_vector(
                artifact,
                layout.attention_query_norm.as_deref(),
            )?,
            attention_key_weight,
            attention_key_bias: load_optional_dense_vector(
                artifact,
                layout.attention_key_bias.as_deref(),
            )?,
            attention_key_norm: load_optional_dense_vector(
                artifact,
                layout.attention_key_norm.as_deref(),
            )?,
            attention_value_weight,
            attention_value_bias: load_optional_dense_vector(
                artifact,
                layout.attention_value_bias.as_deref(),
            )?,
            attention_output_weight: ProjectionMatrix::load(
                artifact,
                required_tensor_name(
                    layout.attention_output_weight.as_deref(),
                    "attention_output_weight",
                )?,
            )?,
            attention_output_bias: load_optional_dense_vector(
                artifact,
                layout.attention_output_bias.as_deref(),
            )?,
            attention_post_norm: load_optional_dense_vector(
                artifact,
                layout.attention_post_norm.as_deref(),
            )?,
            feed_forward_norm: load_dense_vector(
                artifact,
                required_tensor_name(layout.feed_forward_norm.as_deref(), "feed_forward_norm")?,
            )?,
            feed_forward_gate_weight: ProjectionMatrix::load(
                artifact,
                required_tensor_name(
                    layout.feed_forward_gate_weight.as_deref(),
                    "feed_forward_gate_weight",
                )?,
            )?,
            feed_forward_up_weight: ProjectionMatrix::load(
                artifact,
                required_tensor_name(
                    layout.feed_forward_up_weight.as_deref(),
                    "feed_forward_up_weight",
                )?,
            )?,
            feed_forward_down_weight: ProjectionMatrix::load(
                artifact,
                required_tensor_name(
                    layout.feed_forward_down_weight.as_deref(),
                    "feed_forward_down_weight",
                )?,
            )?,
            feed_forward_post_norm: load_optional_dense_vector(
                artifact,
                layout.feed_forward_post_norm.as_deref(),
            )?,
            layer_output_scale: load_optional_dense_scalar(
                artifact,
                layout.layer_output_scale.as_deref(),
            )?,
            per_layer_input_gate: load_named_optional_projection_matrix(
                artifact,
                format!("blk.{layer_index}.inp_gate.weight").as_str(),
            )?,
            per_layer_proj: load_named_optional_projection_matrix(
                artifact,
                format!("blk.{layer_index}.proj.weight").as_str(),
            )?,
            per_layer_post_norm: load_named_optional_dense_vector(
                artifact,
                format!("blk.{layer_index}.post_norm.weight").as_str(),
            )?,
        })
    }

    fn cache_write_width(&self) -> usize {
        self.attention_geometry.cache_write_width()
    }
}

#[derive(Clone, Debug)]
struct DenseGemma4PerLayerInputs {
    token_embedding: ProjectionMatrix,
    model_proj: ProjectionMatrix,
    proj_norm: Vec<f32>,
    width: usize,
}

impl DenseGemma4PerLayerInputs {
    fn load(artifact: &GgufBlobArtifact) -> Result<Option<Self>, ModelLoadError> {
        let token_embedding =
            load_named_optional_projection_matrix(artifact, "per_layer_token_embd.weight")?;
        let model_proj =
            load_named_optional_projection_matrix(artifact, "per_layer_model_proj.weight")?;
        let proj_norm = load_named_optional_dense_vector(artifact, "per_layer_proj_norm.weight")?;
        let (Some(token_embedding), Some(model_proj), Some(proj_norm)) =
            (token_embedding, model_proj, proj_norm)
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            token_embedding,
            model_proj,
            width: proj_norm.as_slice().len(),
            proj_norm,
        }))
    }

    fn project(
        &self,
        token: TokenId,
        hidden: &[f32],
        layer_count: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, ReferenceTextGenerationError> {
        let mut token_inputs = self.token_embedding.decode_row(token.as_u32() as usize)?;
        let expected_len = self.width.saturating_mul(layer_count);
        if token_inputs.len() != expected_len {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 per-layer token embedding width mismatch: expected {expected_len}, actual {}",
                    token_inputs.len()
                )),
            ));
        }
        scale_in_place(&mut token_inputs, (self.width as f32).sqrt());

        let mut projected = Vec::new();
        self.model_proj.matvec(hidden, &mut projected)?;
        if projected.len() != expected_len {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 per-layer model projection width mismatch: expected {expected_len}, actual {}",
                    projected.len()
                )),
            ));
        }
        scale_in_place(&mut projected, (hidden_size as f32).sqrt().recip());

        let layer_mix_scale = 2.0_f32.sqrt().recip();
        let mut combined = vec![0.0_f32; expected_len];
        for layer_index in 0..layer_count {
            let start = layer_index.saturating_mul(self.width);
            let end = start.saturating_add(self.width);
            let normalized = rms_norm(&projected[start..end], self.proj_norm.as_slice(), epsilon);
            for index in 0..self.width {
                combined[start + index] =
                    (normalized[index] + token_inputs[start + index]) * layer_mix_scale;
            }
        }
        Ok(combined)
    }

    fn layer_slice<'a>(&self, values: &'a [f32], layer_index: usize) -> &'a [f32] {
        let start = layer_index.saturating_mul(self.width);
        let end = start.saturating_add(self.width);
        &values[start..end]
    }
}

#[derive(Clone, Debug)]
enum ProjectionMatrix {
    Dense(DenseMatrix),
    Quantized(QuantizedMatrix),
}

impl ProjectionMatrix {
    fn load(artifact: &GgufBlobArtifact, name: &str) -> Result<Self, ModelLoadError> {
        let storage = artifact.paged_tensor(name)?;
        let metadata = storage.metadata();
        if let Some(layout) = metadata.quantized_layout {
            let dims = metadata.shape.dims().to_vec();
            let tensor_name = metadata.name.clone();
            let quantization = metadata.quantization;
            let [rows, columns] = dims.as_slice() else {
                return Err(ModelLoadError::InvalidTensorShape {
                    name: tensor_name,
                    expected: vec![0, 0],
                    actual: dims,
                });
            };
            let row_byte_len = quantized_row_byte_len(&metadata.shape, layout).map_err(|_| {
                ModelLoadError::InvalidQuantizedTensorShape {
                    quantization,
                    shape: metadata.shape.dims().to_vec(),
                }
            })?;
            return Ok(Self::Quantized(QuantizedMatrix {
                storage,
                mode: quantization,
                rows: *rows,
                columns: *columns,
                row_byte_len,
            }));
        }

        let tensor = artifact.load_tensor(name)?;
        let [rows, columns] = tensor.metadata().shape.dims() else {
            return Err(ModelLoadError::InvalidTensorShape {
                name: tensor.metadata().name.clone(),
                expected: vec![0, 0],
                actual: tensor.metadata().shape.dims().to_vec(),
            });
        };
        Ok(Self::Dense(DenseMatrix {
            rows: *rows,
            columns: *columns,
            values: tensor.values()?.into_owned(),
        }))
    }

    fn byte_length(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix
                .values
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
            Self::Quantized(matrix) => matrix.byte_length(),
        }
    }

    fn decode_row(&self, row_index: usize) -> Result<Vec<f32>, crate::RuntimeError> {
        match self {
            Self::Dense(matrix) => matrix.decode_row(row_index),
            Self::Quantized(matrix) => matrix.decode_row(row_index),
        }
    }

    fn matvec(&self, input: &[f32], output: &mut Vec<f32>) -> Result<(), crate::RuntimeError> {
        match self {
            Self::Dense(matrix) => matrix.matvec(input, output),
            Self::Quantized(matrix) => matrix.matvec(input, output),
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.rows,
            Self::Quantized(matrix) => matrix.rows,
        }
    }
}

#[derive(Clone, Debug)]
struct DenseMatrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
}

impl DenseMatrix {
    fn decode_row(&self, row_index: usize) -> Result<Vec<f32>, crate::RuntimeError> {
        if row_index >= self.rows {
            return Err(crate::RuntimeError::Backend(format!(
                "dense row index {row_index} exceeds row count {}",
                self.rows
            )));
        }
        let start = row_index.saturating_mul(self.columns);
        let end = start.saturating_add(self.columns);
        Ok(self.values[start..end].to_vec())
    }

    fn matvec(&self, input: &[f32], output: &mut Vec<f32>) -> Result<(), crate::RuntimeError> {
        if input.len() != self.columns {
            return Err(crate::RuntimeError::Backend(format!(
                "dense matvec width mismatch: expected {}, actual {}",
                self.columns,
                input.len()
            )));
        }
        output.clear();
        output.resize(self.rows, 0.0);
        for (row_index, row) in self.values.chunks_exact(self.columns).enumerate() {
            output[row_index] = dot(row, input);
        }
        Ok(())
    }

    fn matvec_add(&self, input: &[f32], output: &mut [f32]) -> Result<(), crate::RuntimeError> {
        if input.len() != self.columns {
            return Err(crate::RuntimeError::Backend(format!(
                "dense matvec width mismatch: expected {}, actual {}",
                self.columns,
                input.len()
            )));
        }
        if output.len() != self.rows {
            return Err(crate::RuntimeError::Backend(format!(
                "dense output row mismatch: expected {}, actual {}",
                self.rows,
                output.len()
            )));
        }
        for (row_index, row) in self.values.chunks_exact(self.columns).enumerate() {
            output[row_index] += dot(row, input);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct QuantizedMatrix {
    storage: PagedTensorStorage,
    mode: QuantizationMode,
    rows: usize,
    columns: usize,
    row_byte_len: usize,
}

impl QuantizedMatrix {
    fn byte_length(&self) -> usize {
        self.storage.byte_length()
    }

    fn decode_row(&self, row_index: usize) -> Result<Vec<f32>, crate::RuntimeError> {
        if row_index >= self.rows {
            return Err(crate::RuntimeError::Backend(format!(
                "quantized row index {row_index} exceeds row count {}",
                self.rows
            )));
        }
        let offset = row_index.saturating_mul(self.row_byte_len);
        let bytes = self
            .storage
            .read_range(offset, self.row_byte_len)
            .map_err(model_load_runtime_error)?;
        let mut output = Vec::new();
        decode_quantized_row_into(self.mode, bytes, &mut output)?;
        Ok(output)
    }

    fn matvec(&self, input: &[f32], output: &mut Vec<f32>) -> Result<(), crate::RuntimeError> {
        if input.len() != self.columns {
            return Err(crate::RuntimeError::Backend(format!(
                "quantized matvec width mismatch: expected {}, actual {}",
                self.columns,
                input.len()
            )));
        }
        output.clear();
        output.resize(self.rows, 0.0);
        for row_index in 0..self.rows {
            let offset = row_index.saturating_mul(self.row_byte_len);
            let bytes = self
                .storage
                .read_range(offset, self.row_byte_len)
                .map_err(model_load_runtime_error)?;
            output[row_index] = quantized_row_dot(input, self.mode, bytes)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct QuantizedExpertTensor {
    storage: PagedTensorStorage,
    mode: QuantizationMode,
    outer: usize,
    rows: usize,
    columns: usize,
    row_byte_len: usize,
}

impl QuantizedExpertTensor {
    fn byte_length(&self) -> usize {
        self.storage.byte_length()
    }

    fn expert_matvec(
        &self,
        expert_index: usize,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), crate::RuntimeError> {
        if expert_index >= self.outer {
            return Err(crate::RuntimeError::Backend(format!(
                "expert index {expert_index} exceeds expert count {}",
                self.outer
            )));
        }
        if input.len() != self.columns {
            return Err(crate::RuntimeError::Backend(format!(
                "expert matvec width mismatch: expected {}, actual {}",
                self.columns,
                input.len()
            )));
        }
        output.clear();
        output.resize(self.rows, 0.0);
        for row_index in 0..self.rows {
            let offset = (expert_index
                .saturating_mul(self.rows)
                .saturating_add(row_index))
            .saturating_mul(self.row_byte_len);
            let bytes = self
                .storage
                .read_range(offset, self.row_byte_len)
                .map_err(model_load_runtime_error)?;
            output[row_index] = quantized_row_dot(input, self.mode, bytes)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct DenseGgufForwardStep {
    key: Vec<f32>,
    value: Vec<f32>,
    logits: Vec<f32>,
    final_hidden: Vec<f32>,
    kernel_count: usize,
    bytes_moved: u64,
}

pub struct MetalGemma4TextGenerationService {
    backend: MetalBackend,
    models: InMemoryGenerationModelRegistry<MetalGemma4GenerationModel>,
    sessions: InMemoryGenerationSessionStore,
    shared_prefixes: SharedPrefixStore,
    backend_health: super::BackendHealthTracker,
    model_descriptor: DecoderModelDescriptor,
    runtime_support: GgufDecoderRuntimeSupport,
}

impl MetalGemma4TextGenerationService {
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, ReferenceTextGenerationError> {
        let mut backend = MetalBackend::new();
        let model = MetalGemma4GenerationModel::from_gguf_path(path, &mut backend)?;
        let model_descriptor = model.descriptor().clone();
        let runtime_support = model.runtime_support();
        let mut models = InMemoryGenerationModelRegistry::new();
        let now_millis = super::current_time_millis();
        models.warm_with_metadata(
            model,
            now_millis,
            super::DEFAULT_MODEL_KEEPALIVE_MILLIS,
            None,
            Some(String::from("metal")),
            None,
        )?;
        let mut backend_health = super::BackendHealthTracker::default();
        backend_health.observe("metal", backend.health(), now_millis);
        Ok(Self {
            backend,
            models,
            sessions: InMemoryGenerationSessionStore::new(),
            shared_prefixes: SharedPrefixStore::default(),
            backend_health,
            model_descriptor,
            runtime_support,
        })
    }

    #[must_use]
    pub fn model_descriptor(&self) -> &DecoderModelDescriptor {
        &self.model_descriptor
    }

    #[must_use]
    pub fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        self.runtime_support.clone()
    }

    #[must_use]
    pub fn loaded_model_views(&mut self) -> Vec<LoadedModelView> {
        self.loaded_model_views_at(super::current_time_millis())
    }

    #[must_use]
    pub fn loaded_models(&mut self) -> LoadedModelsObservation {
        self.loaded_models_at(super::current_time_millis())
    }

    #[must_use]
    pub fn observability(&mut self) -> LocalRuntimeObservability {
        self.observability_at(super::current_time_millis())
    }

    pub fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Ok(self
            .models
            .warm_loaded(model_id, super::current_time_millis(), keep_alive_millis)?)
    }

    pub fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Ok(self
            .models
            .unload_view(model_id, super::current_time_millis())?)
    }

    pub fn create_session(
        &mut self,
        model_id: &str,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        let model = self
            .models
            .active(model_id)
            .ok_or_else(|| ReferenceTextGenerationError::UnsupportedModel(model_id.to_string()))?;
        Ok(self.sessions.create(
            model,
            super::served_artifact_identity_for_decoder_backend(model.descriptor(), "metal", &[])
                .served_artifact_digest,
        ))
    }

    pub fn reset_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        Ok(self.sessions.reset(session_id)?)
    }

    pub fn close_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        Ok(self.sessions.close(session_id)?)
    }

    pub fn generate_continuous_batch(
        &mut self,
        requests: Vec<GenerationRequest>,
    ) -> ContinuousBatchGenerationResult {
        super::run_continuous_batch_generation_requests(
            &mut self.backend,
            &mut self.models,
            &mut self.sessions,
            &mut self.shared_prefixes,
            requests,
            default_generation_scheduler_policy(),
        )
    }

    #[must_use]
    fn loaded_models_at(&mut self, now_millis: u64) -> LoadedModelsObservation {
        self.models.expire_idle(now_millis);
        self.models.loaded_models_observation()
    }

    #[must_use]
    fn observability_at(&mut self, now_millis: u64) -> LocalRuntimeObservability {
        self.models.expire_idle(now_millis);
        self.backend_health
            .observe("metal", self.backend.health(), now_millis);
        super::generation_runtime_observability(
            &self.models,
            &self.sessions,
            &self.backend_health,
            super::default_text_generation_execution_profile(),
        )
    }

    #[must_use]
    fn loaded_model_views_at(&mut self, now_millis: u64) -> Vec<LoadedModelView> {
        self.models.expire_idle(now_millis);
        self.models.loaded_model_views()
    }
}

impl TextGenerationExecutor for MetalGemma4TextGenerationService {
    type Error = ReferenceTextGenerationError;

    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResponse, Self::Error> {
        run_metal_gemma4_generation_request(
            &mut self.backend,
            &mut self.models,
            &mut self.sessions,
            &mut self.shared_prefixes,
            request,
        )
    }
}

impl StreamingTextGenerationExecutor for MetalGemma4TextGenerationService {
    type Stream<'a> = Box<dyn GenerationEventStream + 'a>;

    fn generate_stream<'a>(
        &'a mut self,
        request: &GenerationRequest,
    ) -> Result<Self::Stream<'a>, ReferenceTextGenerationError> {
        let response = self.generate(request)?;
        Ok(Box::new(CompletedGenerationStream::new(response)))
    }
}

impl ManagedTextGenerationRuntime for MetalGemma4TextGenerationService {
    fn loaded_models(&mut self) -> LoadedModelsObservation {
        Self::loaded_models(self)
    }

    fn observability(&mut self) -> LocalRuntimeObservability {
        Self::observability(self)
    }

    fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Self::warm_model(self, model_id, keep_alive_millis)
    }

    fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Self::unload_model(self, model_id)
    }
}

#[derive(Debug)]
struct MetalGemma4LayerStepPlan {
    identity_cos_values: Vec<f32>,
    identity_sin_values: Vec<f32>,
    attention_values: Vec<f32>,
    rope_freq_factors_buffer: MetalBuffer,
    rope_rotary_half: usize,
    rope_theta_scale: f32,
    rope_freq_scale: f32,
    rope_corr_dims: [f32; 2],
    rope_ext_factor: f32,
    rope_yarn_mscale: f32,
    per_layer_multiplier_buffer: Option<MetalBuffer>,
    qkv_buffer: Option<MetalBuffer>,
    hidden_input_buffer: MetalBuffer,
    query_buffer: MetalBuffer,
    key_buffer: Option<MetalBuffer>,
    value_buffer: Option<MetalBuffer>,
    attention_buffer: MetalBuffer,
    projected_output_buffer: MetalBuffer,
    projected_output_values: Vec<f32>,
    live_key_values: Vec<f32>,
    live_value_values: Vec<f32>,
}

struct MetalGemma4StepPlan {
    layer_plans: Vec<Option<MetalGemma4LayerStepPlan>>,
    hidden_buffer: MetalBuffer,
    residual_buffer: MetalBuffer,
    norm_buffer: MetalBuffer,
    per_layer_inputs_token_buffer: Option<MetalBuffer>,
    per_layer_inputs_buffer: Option<MetalBuffer>,
    ffn_gate_up_buffer: Option<MetalBuffer>,
    ffn_gate_buffer: MetalBuffer,
    ffn_up_buffer: MetalBuffer,
    ffn_activated_buffer: MetalBuffer,
    per_layer_gate_buffer: Option<MetalBuffer>,
    per_layer_activated_buffer: Option<MetalBuffer>,
    per_layer_projected_buffer: Option<MetalBuffer>,
    logits_buffer: MetalBuffer,
    argmax_candidates_buffer: MetalBuffer,
    selected_token_buffer: MetalBuffer,
}

fn build_gemma4_metal_layer_caches_from_host_cache(
    backend: &mut MetalBackend,
    model: &MetalGemma4ModelInner,
    cache: &crate::InMemoryKvCache,
    reserve_tokens: usize,
    kv_cache_encoding_policy: &psionic_runtime::KvCacheEncodingPolicy,
) -> Result<Vec<Option<MetalKvCacheMirror>>, ReferenceTextGenerationError> {
    if cache.width() != model.cache_width() {
        return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
            expected_kv_width: model.cache_width(),
            kv_width: cache.width(),
        });
    }

    model
        .layers
        .iter()
        .map(|layer| {
            let Some(cache_offset) = layer.attention_geometry.cache_write_offset else {
                return Ok(None);
            };
            let layer_kv_width = layer.attention_geometry.kv_width();
            let mut keys = Vec::with_capacity(cache.len().saturating_mul(layer_kv_width));
            let mut values = Vec::with_capacity(cache.len().saturating_mul(layer_kv_width));
            for entry in cache.entries() {
                keys.extend_from_slice(&entry.key[cache_offset..cache_offset + layer_kv_width]);
                values.extend_from_slice(&entry.value[cache_offset..cache_offset + layer_kv_width]);
            }
            backend
                .kv_cache_mirror_from_host_rows(
                    layer_kv_width,
                    cache.max_context(),
                    cache.len(),
                    keys.as_slice(),
                    values.as_slice(),
                    reserve_tokens,
                    kv_cache_encoding_policy.clone(),
                )
                .map(Some)
                .map_err(ReferenceTextGenerationError::Runtime)
        })
        .collect()
}

fn materialize_metal_cache_entries_from_layer_caches(
    cache: &mut crate::InMemoryKvCache,
    model: &MetalGemma4ModelInner,
    layer_caches: &[Option<MetalKvCacheMirror>],
    start_token_index: usize,
) -> Result<(), ReferenceTextGenerationError> {
    if cache.width() != model.cache_width() {
        return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
            expected_kv_width: model.cache_width(),
            kv_width: cache.width(),
        });
    }
    for token_index in start_token_index..cache.len() {
        let mut key = vec![0.0; model.cache_width()];
        let mut value = vec![0.0; model.cache_width()];
        for (layer_index, layer) in model.layers.iter().enumerate() {
            let Some(cache_offset) = layer.attention_geometry.cache_write_offset else {
                continue;
            };
            let layer_cache = layer_caches
                .get(layer_index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "gemma4 metal layer {layer_index} is missing a device kv cache mirror during host cache materialization"
                    )))
                })?;
            let (layer_key, layer_value) = layer_cache
                .read_entry(token_index)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let kv_width = layer.attention_geometry.kv_width();
            key[cache_offset..cache_offset + kv_width].copy_from_slice(layer_key.as_slice());
            value[cache_offset..cache_offset + kv_width].copy_from_slice(layer_value.as_slice());
        }
        cache
            .set_entry_kv(token_index, key, value)
            .map_err(|error| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                    "failed to materialize gemma4 metal host cache entry {token_index}: {error}"
                )))
            })?;
    }
    Ok(())
}

fn gemma4_metal_layer_cache_state(
    layer_caches: &[Option<MetalKvCacheMirror>],
) -> psionic_runtime::KvCacheState {
    layer_caches
        .iter()
        .flatten()
        .next()
        .map(MetalKvCacheMirror::state)
        .unwrap_or_default()
}

fn run_metal_gemma4_generation_request(
    backend: &mut MetalBackend,
    models: &mut InMemoryGenerationModelRegistry<MetalGemma4GenerationModel>,
    sessions: &mut InMemoryGenerationSessionStore,
    shared_prefixes: &mut SharedPrefixStore,
    request: &GenerationRequest,
) -> Result<GenerationResponse, ReferenceTextGenerationError> {
    if !super::generation_product_supported(request) {
        return Err(ReferenceTextGenerationError::UnsupportedProduct(
            request.product_id.clone(),
        ));
    }

    let loaded_model = models
        .active(request.model.model.model_id.as_str())
        .ok_or_else(|| {
            ReferenceTextGenerationError::UnsupportedModel(request.model.model.model_id.clone())
        })?
        .clone();
    if loaded_model.descriptor() != &request.model {
        return Err(ReferenceTextGenerationError::UnsupportedModel(
            request.model.model.model_id.clone(),
        ));
    }

    let model_id = request.model.model.model_id.as_str();
    let load_state = models
        .load_state(model_id)
        .unwrap_or(super::GenerationLoadState::Warm);
    let request_start = super::current_time_millis();
    let generation_start = Instant::now();
    models.begin_request(model_id, request_start)?;
    let memory_plan = models.memory_plan(model_id).cloned();
    let residency_policy = Some(models.residency_policy().clone());
    let residency_snapshot = Some(models.memory_snapshot());
    let served_artifact = super::served_artifact_identity_for_decoder_backend(
        loaded_model.descriptor(),
        loaded_model.backend_compatibility(),
        &[],
    );
    let effective_served_artifact_digest =
        super::effective_generation_served_artifact_digest(&served_artifact, None);

    let result = (|| -> Result<GenerationResponse, ReferenceTextGenerationError> {
        let prompt_eval_start = Instant::now();
        let tokenizer = loaded_model.tokenizer();
        let prompt_tokens = loaded_model.encode_prompt_input(&request.prompt)?;
        if prompt_tokens.is_empty() {
            return Err(ReferenceTextGenerationError::EmptyPrompt);
        }

        let expected_kv_width = loaded_model.cache_width();
        let mut session_tokens = Vec::new();
        let compatibility = super::prefix_compatibility_for_request(&loaded_model, request);
        let prefix_policy = super::default_prefix_cache_policy();
        let mut prefix_state = crate::PrefixCacheState::None;
        let mut prefix_cache_refusal_reason = None;
        let mut prefix_cache_invalidation_trigger = None;
        let mut prefix_tokens_reused = 0usize;
        let mut prefix_identity = None;
        let mut shared_prefix_eligible = false;
        let previous_kv_state = if let Some(session_id) = &request.session_id {
            if request.reset_session {
                sessions.reset(session_id)?;
            }
            let state = sessions.state(session_id)?;
            super::validate_session_model(
                state,
                session_id,
                loaded_model.descriptor(),
                effective_served_artifact_digest.as_str(),
            )?;
            session_tokens = state.tokens().to_vec();
            if state.cache().is_empty() {
                shared_prefix_eligible = true;
            } else {
                prefix_state = crate::PrefixCacheState::Bypassed;
                prefix_cache_refusal_reason =
                    Some(crate::PrefixCacheRefusalReason::SessionBoundState);
            }
            state.cache().state()
        } else {
            shared_prefix_eligible = true;
            psionic_runtime::KvCacheState::default()
        };
        let preserve_prefix_tokens = usize::from(
            prompt_tokens.as_slice().first().copied() == Some(tokenizer.vocabulary().bos_id()),
        );
        let (prompt_tokens, context_window) = super::apply_context_window(
            &prompt_tokens,
            loaded_model.descriptor().config.max_context,
            previous_kv_state.tokens,
            request.options.max_output_tokens,
            request.options.context_overflow_policy,
            preserve_prefix_tokens,
        )?;
        let mut prompt_logits = Vec::new();
        let mut last_logits = Vec::new();
        let mut cache = if shared_prefix_eligible {
            let lookup = super::controlled_prefix_lookup(
                shared_prefixes,
                &compatibility,
                &prompt_tokens,
                request,
            );
            prefix_state = lookup.state;
            prefix_cache_refusal_reason = lookup.refusal_reason;
            prefix_cache_invalidation_trigger = lookup.invalidation_trigger;
            prefix_tokens_reused = lookup.reused_tokens;
            prefix_identity = lookup.identity;
            prompt_logits = lookup.prompt_logits;
            last_logits = if lookup.last_logits.is_empty() {
                prompt_logits.last().cloned().unwrap_or_default()
            } else {
                lookup.last_logits
            };
            lookup.cache.unwrap_or_else(|| {
                crate::InMemoryKvCache::new(
                    loaded_model.descriptor().config.max_context,
                    expected_kv_width,
                )
            })
        } else if let Some(session_id) = &request.session_id {
            sessions.state(session_id)?.cache().clone()
        } else {
            crate::InMemoryKvCache::new(
                loaded_model.descriptor().config.max_context,
                expected_kv_width,
            )
        };
        if cache.width() != expected_kv_width {
            return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
                expected_kv_width,
                kv_width: cache.width(),
            });
        }
        cache.bind_owner(super::request_kv_owner(
            request,
            psionic_runtime::BatchExecutionPosture::SingleRequestOnly,
            None,
        ));
        let request_kv_checkpoint = cache.checkpoint();
        let kv_cache_encoding_policy = super::default_generation_kv_cache_encoding_policy(
            &loaded_model,
            loaded_model.backend_compatibility(),
        );
        let mut layer_caches = build_gemma4_metal_layer_caches_from_host_cache(
            backend,
            &loaded_model.inner,
            &cache,
            request.options.max_output_tokens.saturating_add(1),
            &kv_cache_encoding_policy,
        )?;
        let mut step_plan = loaded_model.inner.build_decode_step_plan(backend)?;
        let logits_output_mode = metal_gemma4_decode_logits_output_mode(request);
        let prompt_logits_output_mode = if shared_prefix_eligible {
            MetalLogitsOutputMode::RawLogits
        } else {
            logits_output_mode
        };
        let use_greedy_selected_token = logits_output_mode == MetalLogitsOutputMode::GreedyToken;
        let mut last_selected_token = None;
        let mut kernel_count = 0usize;
        let mut bytes_moved = 0u64;
        let mut materialized_host_tokens = cache.len();
        if use_greedy_selected_token && !last_logits.is_empty() {
            last_selected_token = super::select_argmax_token(last_logits.as_slice()).map(TokenId);
        }

        for (relative_prompt_index, token) in prompt_tokens.as_slice()[prefix_tokens_reused..]
            .iter()
            .enumerate()
        {
            let prompt_index = prefix_tokens_reused + relative_prompt_index;
            let produce_prompt_logits = prompt_index + 1 == prompt_tokens.len();
            let step = loaded_model.inner.forward_step_with_layer_caches(
                backend,
                *token,
                cache.len(),
                &cache,
                layer_caches.as_mut_slice(),
                &mut step_plan,
                0,
                loaded_model.inner.layers.len(),
                None,
                None,
                None,
                produce_prompt_logits,
                prompt_logits_output_mode,
            )?;
            kernel_count = kernel_count.saturating_add(step.kernel_count);
            bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
            cache.append_token_without_kv(*token)?;
            if produce_prompt_logits {
                if use_greedy_selected_token {
                    if step.logits.is_empty() {
                        last_selected_token = step.selected_token;
                    } else {
                        last_logits = step.logits;
                        prompt_logits.push(last_logits.clone());
                        last_selected_token =
                            super::select_argmax_token(last_logits.as_slice()).map(TokenId);
                    }
                } else {
                    last_logits = step.logits;
                    prompt_logits.push(last_logits.clone());
                }
            }
        }

        if shared_prefix_eligible
            && prefix_tokens_reused != prompt_tokens.len()
            && cache.len() > materialized_host_tokens
        {
            materialize_metal_cache_entries_from_layer_caches(
                &mut cache,
                &loaded_model.inner,
                layer_caches.as_slice(),
                materialized_host_tokens,
            )?;
            materialized_host_tokens = cache.len();
        }

        let prefill_handoff_state = gemma4_metal_layer_cache_state(layer_caches.as_slice());
        let prompt_cache = (shared_prefix_eligible && prefix_tokens_reused != prompt_tokens.len())
            .then(|| cache.clone());
        let mut sampler = super::GenerationSampler::new(&request.options)?;
        let structured_output_report = sampler.structured_output_report();
        let mut generated_tokens = Vec::new();
        let mut first_token_emitted_at = None;
        let mut last_token_emitted_at = None;
        let termination = loop {
            if generated_tokens.len() >= request.options.max_output_tokens {
                break super::TerminationReason::MaxOutputTokens;
            }
            if cache.len() >= cache.max_context() {
                break super::TerminationReason::ContextLimit;
            }

            let next_token = if use_greedy_selected_token {
                last_selected_token.ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                        String::from(
                            "metal gemma4 greedy decode path did not produce a selected token",
                        ),
                    ))
                })?
            } else {
                match sampler.select_next_token(
                    loaded_model.tokenizer(),
                    &last_logits,
                    &cache,
                    generated_tokens.as_slice(),
                )? {
                    super::GenerationSelection::Token(token) => token,
                    super::GenerationSelection::Terminate => {
                        break super::TerminationReason::EndOfSequence;
                    }
                }
            };
            if loaded_model.is_end_of_sequence(next_token) {
                break super::TerminationReason::EndOfSequence;
            }

            generated_tokens.push(next_token);
            let step = loaded_model.inner.forward_step_with_layer_caches(
                backend,
                next_token,
                cache.len(),
                &cache,
                layer_caches.as_mut_slice(),
                &mut step_plan,
                0,
                loaded_model.inner.layers.len(),
                None,
                None,
                None,
                true,
                logits_output_mode,
            )?;
            kernel_count = kernel_count.saturating_add(step.kernel_count);
            bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
            cache.append_token_without_kv(next_token)?;
            if use_greedy_selected_token {
                last_selected_token = step.selected_token;
            } else {
                last_logits = step.logits;
            }
            let emitted_at = Instant::now();
            if first_token_emitted_at.is_none() {
                first_token_emitted_at = Some(emitted_at);
            }
            last_token_emitted_at = Some(emitted_at);

            if super::truncate_generated_text(
                loaded_model.tokenizer(),
                &mut generated_tokens,
                &request.options.stop_sequences,
            )
            .is_some()
            {
                break super::TerminationReason::EndOfSequence;
            }
        };

        if shared_prefix_eligible && prefix_tokens_reused != prompt_tokens.len() {
            if let Some(prompt_cache) = prompt_cache.as_ref() {
                let recorded_identity = shared_prefixes.record(
                    compatibility,
                    &prompt_tokens,
                    &prompt_logits,
                    prompt_cache,
                );
                if prefix_state != crate::PrefixCacheState::Hit || prefix_identity.is_none() {
                    prefix_identity = Some(recorded_identity);
                }
            }
        }

        let prompt_eval_duration_ns = super::elapsed_ns(prompt_eval_start);
        let generated = TokenSequence::new(generated_tokens);
        if let Some(session_id) = &request.session_id {
            if cache.len() > materialized_host_tokens {
                materialize_metal_cache_entries_from_layer_caches(
                    &mut cache,
                    &loaded_model.inner,
                    layer_caches.as_slice(),
                    materialized_host_tokens,
                )?;
            }
            session_tokens.extend_from_slice(prompt_tokens.as_slice());
            session_tokens.extend_from_slice(generated.as_slice());
            sessions.replace_cache(
                session_id,
                loaded_model.descriptor(),
                effective_served_artifact_digest.as_str(),
                cache.clone(),
                TokenSequence::new(session_tokens),
            )?;
        }
        let text = loaded_model.tokenizer().decode(generated.as_slice());
        let usage = super::GenerationUsage {
            input_tokens: prompt_tokens.len(),
            output_tokens: generated.len(),
            cache_tokens: cache.len(),
        };
        let kv_cache =
            psionic_runtime::KvCacheAccounting::from_states(&previous_kv_state, cache.state());
        let total_duration_ns = super::elapsed_ns(generation_start);
        let time_to_first_token_ns = first_token_emitted_at
            .map(|first_token_at| first_token_at.duration_since(generation_start))
            .and_then(|duration| duration.as_nanos().try_into().ok());
        let inter_token_latency_ns = super::average_inter_token_latency_ns(
            first_token_emitted_at,
            last_token_emitted_at,
            usage.output_tokens,
        );
        let prefill_decode_handoff = super::local_prefill_decode_handoff(&prefill_handoff_state);
        let kv_residency = super::host_only_kv_residency(cache.policy(), cache.state());
        let metrics = crate::GenerationMetrics {
            total_duration_ns: Some(total_duration_ns),
            load_duration_ns: Some(match load_state {
                super::GenerationLoadState::Cold => loaded_model.load_duration_ns(),
                super::GenerationLoadState::Warm => 0,
            }),
            prompt_eval_count: Some(usage.input_tokens),
            prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
            context_window: Some(context_window),
            eval_count: Some(usage.output_tokens),
            eval_duration_ns: Some(total_duration_ns.saturating_sub(prompt_eval_duration_ns)),
            time_to_first_token_ns,
            inter_token_latency_ns,
            kv_cache: Some(kv_cache),
            kv_residency: kv_residency.clone(),
            kv_cache_encoding: Some(psionic_runtime::KvCacheEncodingAccounting::active(
                kv_cache_encoding_policy.clone(),
            )),
            prefix_tokens_reused: Some(prefix_tokens_reused),
            termination_detail: None,
            qwen35_cuda_decode: None,
            gpt_oss_perf: None,
        };
        let delivery_plan_digest = loaded_model.plan_digest().to_string();
        let provenance = crate::GenerationProvenance {
            served_artifact,
            adapter_serving: request.adapter_serving.clone(),
            served_revision: None,
            execution_plan_digest: delivery_plan_digest.clone(),
            cluster_execution: None,
            load_state,
            isolation_policy: crate::LocalServingIsolationPolicy::in_process_runtime(),
            streaming_policy: None,
            memory_plan,
            residency_policy,
            residency_snapshot,
            kv_cache_policy: Some(cache.policy().clone()),
            kv_cache_encoding_policy: Some(kv_cache_encoding_policy),
            kv_ownership: cache.ownership_since(&request_kv_checkpoint),
            prefix_cache_control: Some(request.prefix_cache_control.clone()),
            prefix_cache_state: Some(prefix_state),
            prefix_cache_refusal_reason,
            prefix_cache_policy: Some(prefix_policy),
            prefix_cache_identity: prefix_identity,
            compile_path: None,
            delivery_proof: crate::build_delivery_proof(
                delivery_plan_digest,
                kernel_count,
                bytes_moved,
                0,
                0,
                metrics.kv_cache.as_ref().map(|value| value.growth.clone()),
                prefill_decode_handoff,
                kv_residency,
            ),
            cache_observations: crate::generation_cache_observations(
                loaded_model.descriptor(),
                None,
                load_state,
                request.session_id.as_ref(),
                request.reset_session,
                &previous_kv_state,
                prefix_state,
                prefix_cache_invalidation_trigger,
            ),
            scheduler: None,
            structured_output: structured_output_report,
            psion_served_evidence: None,
            psion_served_output_claim_posture: None,
        };
        let structured_output_value = sampler.structured_output_value(text.as_str())?;
        let response = GenerationResponse::new(
            request,
            request.session_id.clone(),
            generated,
            text,
            usage.input_tokens,
            usage.cache_tokens,
            termination,
        )
        .with_metrics_and_provenance(metrics, provenance);
        Ok(if let Some(value) = structured_output_value {
            response.with_structured_output_value(value)
        } else {
            response
        })
    })();

    let _ = models.finish_request(model_id, super::current_time_millis());
    result
}

#[derive(Clone, Debug)]
struct MetalGemma4GenerationModel {
    inner: Arc<MetalGemma4ModelInner>,
}

impl MetalGemma4GenerationModel {
    fn from_gguf_path(
        path: impl AsRef<Path>,
        backend: &mut MetalBackend,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let artifact = GgufBlobArtifact::open_path(path, gguf_local_blob_open_options())?;
        Self::from_blob_artifact(artifact, backend)
    }

    fn from_blob_artifact(
        artifact: GgufBlobArtifact,
        backend: &mut MetalBackend,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let load_start = Instant::now();
        let adapter = GgufDecoderAdapterLoader.load_blob_artifact(&artifact)?;
        if adapter.family_metadata().family != GgufDecoderFamily::Gemma4 {
            return Err(ModelLoadError::UnsupportedModel(
                adapter.descriptor().model.model_id.clone(),
            )
            .into());
        }
        let tokenizer = GgufRuntimeTokenizer::from_gguf(adapter.tokenizer()).map_err(|error| {
            ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!("failed to build runtime tokenizer: {error}"),
            }
        })?;
        let token_embedding =
            ProjectionMatrix::load(&artifact, adapter.tensor_layout().token_embedding.as_str())?;
        let output_name = adapter
            .tensor_layout()
            .output
            .as_deref()
            .unwrap_or(adapter.tensor_layout().token_embedding.as_str());
        let output = MetalQuantizedProjectionMatrix::load(backend, &artifact, output_name)?;
        let descriptor = adapter.descriptor().clone();
        let gemma4_per_layer_inputs = MetalGemma4PerLayerInputs::load(backend, &artifact)?;
        let mut cache_offset = 0usize;
        let mut last_swa_cache_offset = None;
        let mut last_full_cache_offset = None;
        let mut last_swa_kv_layer_index = None;
        let mut last_full_kv_layer_index = None;
        let mut layers = Vec::with_capacity(adapter.tensor_layout().layers.len());
        for (layer_index, layout) in adapter.tensor_layout().layers.iter().enumerate() {
            let query_weight_name = required_tensor_name(
                layout.attention_query_weight.as_deref(),
                "attention_query_weight",
            )?;
            let query_rows = artifact
                .paged_tensor(query_weight_name)?
                .metadata()
                .shape
                .dims()[0];
            let layer_head_dim = query_rows
                .checked_div(descriptor.config.block.attention.head_count)
                .unwrap_or(0);
            let is_swa =
                gemma4_layer_is_swa(adapter.family_metadata(), layer_index, layer_head_dim);
            let has_kv = gemma4_layer_has_kv(adapter.family_metadata(), layout);
            let (cache_read_offset, cache_write_offset, reuse_layer_index) = if has_kv {
                (cache_offset, Some(cache_offset), None)
            } else if is_swa {
                (
                    last_swa_cache_offset.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 swa layer {layer_index} reuses kv before any swa kv source was loaded"
                            ),
                        ))
                    })?,
                    None,
                    Some(last_swa_kv_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 swa layer {layer_index} is missing a reused kv source layer index"
                            ),
                        ))
                    })?),
                )
            } else {
                (
                    last_full_cache_offset.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 full-attention layer {layer_index} reuses kv before any full-attention kv source was loaded"
                            ),
                        ))
                    })?,
                    None,
                    Some(last_full_kv_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 full-attention layer {layer_index} is missing a reused kv source layer index"
                            ),
                        ))
                    })?),
                )
            };
            let layer = MetalGemma4Layer::load(
                backend,
                &artifact,
                layout,
                &descriptor,
                adapter.family_metadata(),
                layer_index,
                cache_read_offset,
                cache_write_offset,
                reuse_layer_index,
            )?;
            if layer.attention_geometry.has_kv() {
                if is_swa {
                    last_swa_cache_offset = layer.attention_geometry.cache_write_offset;
                    last_swa_kv_layer_index = Some(layer_index);
                } else {
                    last_full_cache_offset = layer.attention_geometry.cache_write_offset;
                    last_full_kv_layer_index = Some(layer_index);
                }
            }
            cache_offset = cache_offset.saturating_add(layer.cache_write_width());
            layers.push(layer);
        }
        let inner = MetalGemma4ModelInner {
            descriptor: descriptor.clone(),
            family_metadata: adapter.family_metadata().clone(),
            tokenizer,
            token_embedding,
            gemma4_per_layer_inputs,
            rope_freq_factors: load_named_optional_dense_vector(&artifact, "rope_freqs.weight")?,
            output_norm: load_metal_static_vector(
                backend,
                &artifact,
                adapter.tensor_layout().output_norm.as_str(),
            )?,
            output,
            layers,
            plan_digest: digest_gemma4_metal_plan(&descriptor, adapter.family_metadata()),
            load_duration_ns: load_start
                .elapsed()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[must_use]
    fn plan_digest(&self) -> &str {
        self.inner.plan_digest.as_str()
    }

    #[must_use]
    fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        let mut support = runtime_support_for_descriptor(
            &self.inner.descriptor,
            GgufDecoderFamily::Gemma4,
            vec![String::from("metal")],
            vec![String::from("cpu"), String::from("cuda")],
            unsupported_adapter_runtime_support(
                "LM-head LoRA serving is currently unsupported on the metal gemma4 runtime",
            ),
        );
        support.unsupported_features = vec![
            String::from("image_inputs"),
            String::from("video_inputs"),
            String::from("audio_inputs"),
        ];
        support
    }
}

fn metal_gemma4_can_use_greedy_selected_token(request: &GenerationRequest) -> bool {
    request.adapter_serving.is_none()
        && request.options.structured_output.is_none()
        && request.options.repeat_penalty.is_none()
        && request.options.presence_penalty.is_none()
        && request.options.frequency_penalty.is_none()
        && (request.options.decode_strategy == crate::DecodeStrategy::Greedy
            || request.options.sampling_policy().effective_temperature() <= 1e-6)
}

fn metal_gemma4_decode_logits_output_mode(request: &GenerationRequest) -> MetalLogitsOutputMode {
    if metal_gemma4_can_use_greedy_selected_token(request) {
        MetalLogitsOutputMode::GreedyToken
    } else {
        MetalLogitsOutputMode::RawLogits
    }
}

impl crate::GenerationModelHandle for MetalGemma4GenerationModel {
    fn descriptor(&self) -> &DecoderModelDescriptor {
        &self.inner.descriptor
    }

    fn cache_width(&self) -> usize {
        self.inner.cache_width()
    }
}

impl super::CompiledWordGenerationModel for MetalGemma4GenerationModel {
    type Backend = MetalBackend;

    fn tokenizer(&self) -> &dyn TokenizerBoundary {
        &self.inner.tokenizer
    }

    fn encode_prompt_input(
        &self,
        input: &GenerationInput,
    ) -> Result<TokenSequence, ReferenceTextGenerationError> {
        Ok(match input {
            GenerationInput::Text(text) => self.inner.tokenizer.encode_with_defaults(text),
            GenerationInput::Tokens(tokens) => tokens.clone(),
        })
    }

    fn is_end_of_sequence(&self, token: TokenId) -> bool {
        self.inner.tokenizer.is_end_of_sequence(token)
    }

    fn execute_step(
        &self,
        backend: &mut Self::Backend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
    ) -> Result<GenerationStepOutput, ReferenceTextGenerationError> {
        let config = &self.inner.descriptor.config;
        if token.as_u32() as usize >= config.vocab_size {
            return Err(ReferenceTextGenerationError::InvalidToken {
                token: token.as_u32(),
                vocab_size: config.vocab_size,
            });
        }
        if position >= config.max_context {
            return Err(ReferenceTextGenerationError::InvalidPosition {
                position,
                max_context: config.max_context,
            });
        }
        if cache.width() != self.inner.cache_width() {
            return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
                expected_kv_width: self.inner.cache_width(),
                kv_width: cache.width(),
            });
        }
        let step = self.inner.forward_step(
            backend,
            token,
            position,
            cache,
            MetalLogitsOutputMode::RawLogits,
        )?;
        Ok(GenerationStepOutput {
            key: step.key,
            value: step.value,
            logits: step.logits,
            hidden: Some(step.final_hidden),
            execution_plan_digest: Some(self.inner.plan_digest.clone()),
            compile_path: None,
            kernel_count: step.kernel_count,
            bytes_moved: step.bytes_moved,
            plan_cache_hits: 0,
            plan_cache_misses: 0,
            gpt_oss_perf: None,
        })
    }

    fn plan_digest(&self) -> &str {
        self.plan_digest()
    }

    fn load_duration_ns(&self) -> u64 {
        self.inner.load_duration_ns
    }

    fn backend_compatibility(&self) -> &'static str {
        "metal"
    }
}

#[derive(Clone, Debug)]
struct MetalGemma4ModelInner {
    descriptor: DecoderModelDescriptor,
    family_metadata: GgufDecoderFamilyMetadata,
    tokenizer: GgufRuntimeTokenizer,
    token_embedding: ProjectionMatrix,
    gemma4_per_layer_inputs: Option<MetalGemma4PerLayerInputs>,
    rope_freq_factors: Option<Vec<f32>>,
    output_norm: MetalStaticVector,
    output: MetalQuantizedProjectionMatrix,
    layers: Vec<MetalGemma4Layer>,
    plan_digest: String,
    load_duration_ns: u64,
}

impl MetalGemma4ModelInner {
    fn rope_freq_factors_for_layer(&self, layer_index: usize, head_dim: usize) -> Option<&[f32]> {
        (!gemma4_layer_is_swa(&self.family_metadata, layer_index, head_dim))
            .then_some(self.rope_freq_factors.as_deref())
            .flatten()
    }

    fn cache_width(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.attention_geometry.cache_end())
            .max()
            .unwrap_or(0)
    }

    fn build_decode_step_plan(
        &self,
        backend: &mut MetalBackend,
    ) -> Result<MetalGemma4StepPlan, ReferenceTextGenerationError> {
        let hidden_size = self.descriptor.config.hidden_size;
        let max_ffn_width = self
            .layers
            .iter()
            .filter_map(|layer| {
                layer
                    .feed_forward_gate_weight
                    .as_ref()
                    .map(MetalQuantizedProjectionMatrix::rows)
            })
            .max()
            .unwrap_or(hidden_size);
        let max_ffn_gate_up_width = self
            .layers
            .iter()
            .filter_map(|layer| {
                layer
                    .feed_forward_gate_up_weight
                    .as_ref()
                    .map(MetalQuantizedProjectionMatrix::rows)
            })
            .max()
            .unwrap_or(0);
        let max_per_layer_width = self
            .layers
            .iter()
            .filter_map(|layer| layer.per_layer_input_gate.as_ref().map(|gate| gate.rows()))
            .max()
            .unwrap_or(0);
        let total_per_layer_input_width = self
            .gemma4_per_layer_inputs
            .as_ref()
            .map(|inputs| inputs.width.saturating_mul(self.layers.len()))
            .unwrap_or(0);
        let per_layer_input_width = self
            .gemma4_per_layer_inputs
            .as_ref()
            .map(|inputs| inputs.width);
        let per_layer_inputs_buffer = (total_per_layer_input_width > 0)
            .then(|| backend.zeros_buffer(Shape::new(vec![total_per_layer_input_width])))
            .transpose()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let layer_plans = self
            .layers
            .iter()
            .enumerate()
            .map(|(layer_index, layer)| {
                let q_rows = layer
                    .attention_geometry
                    .head_count
                    .saturating_mul(layer.attention_geometry.head_dim);
                let kv_rows = layer.attention_geometry.kv_width();
                let half_dim = layer.attention_geometry.head_dim / 2;
                let hidden_size = self.descriptor.config.hidden_size;
                let rope_freq_factors = metal_gemma4_rope_freq_factor_values(
                    layer.attention_geometry.head_dim,
                    layer.attention_geometry.rotary_dim,
                    self.rope_freq_factors_for_layer(layer_index, layer.attention_geometry.head_dim),
                );
                let (
                    rope_rotary_half,
                    rope_theta_scale,
                    rope_freq_scale,
                    rope_corr_dims,
                    rope_ext_factor,
                    rope_yarn_mscale,
                ) = metal_gemma4_rope_kernel_params(
                    layer.attention_geometry.head_dim,
                    layer.attention_geometry.rotary_dim,
                    layer.attention_geometry.rope_theta,
                    &self.family_metadata,
                );
                let qkv_buffer = layer
                    .attention_qkv_weight
                    .as_ref()
                    .map(|weight| backend.zeros_buffer(Shape::new(vec![weight.rows()])))
                    .transpose()
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                let query_buffer = if let Some(buffer) = qkv_buffer.as_ref() {
                    buffer
                        .dense_f32_view(0, q_rows)
                        .map_err(ReferenceTextGenerationError::Runtime)?
                } else {
                    backend
                        .zeros_buffer(Shape::new(vec![q_rows]))
                        .map_err(ReferenceTextGenerationError::Runtime)?
                };
                let key_buffer = if layer.attention_geometry.has_kv() {
                    if let Some(buffer) = qkv_buffer.as_ref() {
                        Some(
                            buffer
                                .dense_f32_view(q_rows, kv_rows)
                                .map_err(ReferenceTextGenerationError::Runtime)?,
                        )
                    } else {
                        Some(
                            backend
                                .zeros_buffer(Shape::new(vec![kv_rows]))
                                .map_err(ReferenceTextGenerationError::Runtime)?,
                        )
                    }
                } else {
                    None
                };
                let value_buffer = if layer.attention_geometry.has_kv() {
                    if let Some(buffer) = qkv_buffer.as_ref() {
                        Some(
                            buffer
                                .dense_f32_view(q_rows.saturating_add(kv_rows), kv_rows)
                                .map_err(ReferenceTextGenerationError::Runtime)?,
                        )
                    } else {
                        Some(
                            backend
                                .zeros_buffer(Shape::new(vec![kv_rows]))
                                .map_err(ReferenceTextGenerationError::Runtime)?,
                        )
                    }
                } else {
                    None
                };
                Ok(Some(MetalGemma4LayerStepPlan {
                    identity_cos_values: vec![1.0; half_dim],
                    identity_sin_values: vec![0.0; half_dim],
                    attention_values: vec![0.0; q_rows],
                    rope_freq_factors_buffer: backend
                        .input_buffer(Shape::new(vec![half_dim]), rope_freq_factors)
                        .map_err(ReferenceTextGenerationError::Runtime)?,
                    rope_rotary_half,
                    rope_theta_scale,
                    rope_freq_scale,
                    rope_corr_dims,
                    rope_ext_factor,
                    rope_yarn_mscale,
                    per_layer_multiplier_buffer: match (
                        layer.per_layer_input_gate.as_ref(),
                        per_layer_inputs_buffer.as_ref(),
                        per_layer_input_width,
                    ) {
                        (Some(gate), Some(buffer), Some(width)) => {
                            if gate.rows() != width {
                                return Err(ReferenceTextGenerationError::Runtime(
                                    crate::RuntimeError::Backend(format!(
                                        "gemma4 metal per-layer multiplier width mismatch at layer {layer_index}: gate rows {}, input width {width}",
                                        gate.rows()
                                    )),
                                ));
                            }
                            Some(
                                buffer
                                    .dense_f32_view(
                                        layer_index.saturating_mul(width),
                                        width,
                                    )
                                    .map_err(ReferenceTextGenerationError::Runtime)?,
                            )
                        }
                        _ => None,
                    },
                    qkv_buffer,
                    hidden_input_buffer: backend
                        .zeros_buffer(Shape::new(vec![hidden_size]))
                        .map_err(ReferenceTextGenerationError::Runtime)?,
                    query_buffer,
                    key_buffer,
                    value_buffer,
                    attention_buffer: backend
                        .zeros_buffer(Shape::new(vec![q_rows]))
                        .map_err(ReferenceTextGenerationError::Runtime)?,
                    projected_output_buffer: backend
                        .zeros_buffer(Shape::new(vec![hidden_size]))
                        .map_err(ReferenceTextGenerationError::Runtime)?,
                    projected_output_values: vec![0.0; hidden_size],
                    live_key_values: vec![0.0; kv_rows],
                    live_value_values: vec![0.0; kv_rows],
                }))
            })
            .collect::<Result<Vec<_>, ReferenceTextGenerationError>>()?;
        let ffn_gate_up_buffer = (max_ffn_gate_up_width > 0)
            .then(|| backend.zeros_buffer(Shape::new(vec![max_ffn_gate_up_width])))
            .transpose()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let ffn_gate_buffer = if let Some(buffer) = ffn_gate_up_buffer.as_ref() {
            buffer
                .dense_f32_view(0, max_ffn_width)
                .map_err(ReferenceTextGenerationError::Runtime)?
        } else {
            backend
                .zeros_buffer(Shape::new(vec![max_ffn_width]))
                .map_err(ReferenceTextGenerationError::Runtime)?
        };
        let ffn_up_buffer = if let Some(buffer) = ffn_gate_up_buffer.as_ref() {
            buffer
                .dense_f32_view(max_ffn_width, max_ffn_width)
                .map_err(ReferenceTextGenerationError::Runtime)?
        } else {
            backend
                .zeros_buffer(Shape::new(vec![max_ffn_width]))
                .map_err(ReferenceTextGenerationError::Runtime)?
        };
        Ok(MetalGemma4StepPlan {
            layer_plans,
            hidden_buffer: backend
                .zeros_buffer(Shape::new(vec![hidden_size]))
                .map_err(ReferenceTextGenerationError::Runtime)?,
            residual_buffer: backend
                .zeros_buffer(Shape::new(vec![hidden_size]))
                .map_err(ReferenceTextGenerationError::Runtime)?,
            norm_buffer: backend
                .zeros_buffer(Shape::new(vec![hidden_size]))
                .map_err(ReferenceTextGenerationError::Runtime)?,
            per_layer_inputs_token_buffer: (total_per_layer_input_width > 0)
                .then(|| backend.zeros_buffer(Shape::new(vec![total_per_layer_input_width])))
                .transpose()
                .map_err(ReferenceTextGenerationError::Runtime)?,
            per_layer_inputs_buffer,
            ffn_gate_up_buffer,
            ffn_gate_buffer,
            ffn_up_buffer,
            ffn_activated_buffer: backend
                .zeros_buffer(Shape::new(vec![max_ffn_width]))
                .map_err(ReferenceTextGenerationError::Runtime)?,
            per_layer_gate_buffer: (max_per_layer_width > 0)
                .then(|| backend.zeros_buffer(Shape::new(vec![max_per_layer_width])))
                .transpose()
                .map_err(ReferenceTextGenerationError::Runtime)?,
            per_layer_activated_buffer: (max_per_layer_width > 0)
                .then(|| backend.zeros_buffer(Shape::new(vec![max_per_layer_width])))
                .transpose()
                .map_err(ReferenceTextGenerationError::Runtime)?,
            per_layer_projected_buffer: (max_per_layer_width > 0)
                .then(|| backend.zeros_buffer(Shape::new(vec![hidden_size])))
                .transpose()
                .map_err(ReferenceTextGenerationError::Runtime)?,
            logits_buffer: backend
                .zeros_buffer(Shape::new(vec![self.output.rows()]))
                .map_err(ReferenceTextGenerationError::Runtime)?,
            argmax_candidates_buffer: backend
                .zeros_i32_buffer(Shape::new(vec![
                    psionic_backend_metal::quantized_argmax_candidate_count(self.output.rows())
                        .saturating_mul(2),
                ]))
                .map_err(ReferenceTextGenerationError::Runtime)?,
            selected_token_buffer: backend
                .zeros_i32_buffer(Shape::new(vec![2]))
                .map_err(ReferenceTextGenerationError::Runtime)?,
        })
    }

    fn forward_step(
        &self,
        backend: &mut MetalBackend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
        logits_output_mode: MetalLogitsOutputMode,
    ) -> Result<MetalGemma4ForwardStep, ReferenceTextGenerationError> {
        let step = self.forward_stage_step(
            backend,
            token,
            position,
            cache,
            0,
            self.layers.len(),
            None,
            None,
            None,
            true,
            logits_output_mode,
        )?;
        let final_hidden = step.lm_head_hidden.ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "metal gemma4 full forward step did not produce lm-head hidden",
            )))
        })?;
        Ok(MetalGemma4ForwardStep {
            key: step.key,
            value: step.value,
            logits: step.logits,
            selected_token: step.selected_token,
            final_hidden,
            kernel_count: step.kernel_count,
            bytes_moved: step.bytes_moved,
        })
    }

    fn forward_local_step_with_layer_caches(
        &self,
        backend: &mut MetalBackend,
        token: TokenId,
        position: usize,
        layer_caches: &mut [Option<MetalKvCacheMirror>],
        step_plan: &mut MetalGemma4StepPlan,
        start_layer: usize,
        end_layer: usize,
        input_hidden: Option<&[f32]>,
        forwarded_key: Option<&[f32]>,
        forwarded_value: Option<&[f32]>,
        materialize_stage_outputs: bool,
        produce_logits: bool,
        logits_output_mode: MetalLogitsOutputMode,
    ) -> Result<MetalGemma4StageStep, ReferenceTextGenerationError> {
        if start_layer > end_layer || end_layer > self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "invalid gemma4 metal stage range [{start_layer}..{end_layer}) for {} layers",
                    self.layers.len()
                )),
            ));
        }
        if layer_caches.len() != self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 metal layer cache count mismatch: expected {}, actual {}",
                    self.layers.len(),
                    layer_caches.len()
                )),
            ));
        }
        if step_plan.layer_plans.len() != self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 metal step-plan layer count mismatch: expected {}, actual {}",
                    self.layers.len(),
                    step_plan.layer_plans.len()
                )),
            ));
        }

        let hidden_size = self.descriptor.config.hidden_size;
        let mut embedding_hidden = self
            .token_embedding
            .decode_row(token.as_u32() as usize)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        scale_in_place(
            &mut embedding_hidden,
            input_embedding_scale(&self.family_metadata, hidden_size),
        );
        let hidden = if let Some(input_hidden) = input_hidden {
            if input_hidden.len() != hidden_size {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "gemma4 metal stage input hidden width mismatch: expected {hidden_size}, actual {}",
                        input_hidden.len()
                    )),
                ));
            }
            input_hidden.to_vec()
        } else {
            embedding_hidden.clone()
        };
        step_plan
            .hidden_buffer
            .write_f32(hidden.as_slice())
            .map_err(ReferenceTextGenerationError::Runtime)?;

        let mut cache_key = forwarded_key
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        let mut cache_value = forwarded_value
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        if cache_key.len() != self.cache_width() || cache_value.len() != self.cache_width() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 metal forwarded kv width mismatch: expected {}, actual key={} value={}",
                    self.cache_width(),
                    cache_key.len(),
                    cache_value.len()
                )),
            ));
        }
        if start_layer > 0
            && materialize_stage_outputs
            && (forwarded_key.is_none() || forwarded_value.is_none())
        {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(String::from(
                    "gemma4 metal staged execution requires forwarded kv for earlier layers",
                )),
            ));
        }
        if start_layer > 0 {
            for (layer_index, layer) in self.layers.iter().enumerate().take(start_layer) {
                let Some(cache_offset) = layer.attention_geometry.cache_write_offset else {
                    continue;
                };
                let kv_width = layer.attention_geometry.kv_width();
                let layer_key = &cache_key[cache_offset..cache_offset + kv_width];
                let layer_value = &cache_value[cache_offset..cache_offset + kv_width];
                let layer_cache = layer_caches
                    .get_mut(layer_index)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 metal layer {layer_index} is missing a device kv cache mirror"
                            ),
                        ))
                    })?;
                match layer_cache.len() {
                    len if len == position => {
                        layer_cache
                            .append_entry(backend, layer_key, layer_value)
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                    }
                    len if len == position.saturating_add(1) => {}
                    len => {
                        return Err(ReferenceTextGenerationError::Runtime(
                            crate::RuntimeError::Backend(format!(
                                "gemma4 metal stage layer {layer_index} cache length mismatch: expected {position} or {}, actual {len}",
                                position.saturating_add(1)
                            )),
                        ));
                    }
                }
            }
        }

        let gemma4_per_layer_input_shape = if let Some(per_layer_inputs) =
            self.gemma4_per_layer_inputs.as_ref()
        {
            let mut token_inputs = per_layer_inputs
                .token_embedding
                .decode_row(token.as_u32() as usize)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let total_width = per_layer_inputs.width.saturating_mul(self.layers.len());
            if token_inputs.len() != total_width {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "gemma4 per-layer token embedding width mismatch: expected {total_width}, actual {}",
                        token_inputs.len()
                    )),
                ));
            }
            scale_in_place(&mut token_inputs, (per_layer_inputs.width as f32).sqrt());
            let token_buffer = step_plan
                .per_layer_inputs_token_buffer
                .as_mut()
                .ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                        String::from(
                            "gemma4 metal step-plan is missing the per-layer token input buffer",
                        ),
                    ))
                })?;
            token_buffer
                .write_f32(token_inputs.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            Some((per_layer_inputs.width, total_width))
        } else {
            None
        };

        let mut submission = backend
            .begin_submission(format!(
                "psionic.gemma4.metal.local.layerstack.token{position}"
            ))
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let mut bytes_moved = self.token_embedding.byte_length() as u64;

        if let (
            Some(per_layer_inputs),
            Some((per_layer_input_width, total_per_layer_input_width)),
        ) = (
            self.gemma4_per_layer_inputs.as_ref(),
            gemma4_per_layer_input_shape,
        ) {
            let token_buffer = step_plan
                .per_layer_inputs_token_buffer
                .as_ref()
                .ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                        String::from(
                            "gemma4 metal step-plan is missing the per-layer token input buffer",
                        ),
                    ))
                })?;
            let projected_buffer = step_plan.per_layer_inputs_buffer.as_mut().ok_or_else(|| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                    "gemma4 metal step-plan is missing the per-layer input projection buffer",
                )))
            })?;
            match &per_layer_inputs.model_proj {
                MetalGemma4PerLayerProjectionMatrix::Quantized(model_proj)
                    if model_proj.is_native() =>
                {
                    backend
                        .encode_quantized_matvec_submission(
                            &mut submission,
                            model_proj.native_weights()?,
                            0,
                            model_proj.mode,
                            model_proj.rows,
                            model_proj.columns,
                            &step_plan.hidden_buffer,
                            projected_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    backend
                        .encode_scale_inplace_submission(
                            &mut submission,
                            projected_buffer,
                            (hidden_size as f32).sqrt().recip(),
                            total_per_layer_input_width,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    backend
                        .encode_per_head_rms_norm_submission(
                            &mut submission,
                            projected_buffer,
                            per_layer_inputs.proj_norm.device(),
                            self.layers.len(),
                            per_layer_input_width,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    backend
                        .encode_add_inplace_submission(
                            &mut submission,
                            projected_buffer,
                            token_buffer,
                            total_per_layer_input_width,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    backend
                        .encode_scale_inplace_submission(
                            &mut submission,
                            projected_buffer,
                            2.0_f32.sqrt().recip(),
                            total_per_layer_input_width,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                MetalGemma4PerLayerProjectionMatrix::Host(_)
                | MetalGemma4PerLayerProjectionMatrix::Quantized(_) => {
                    let projected = per_layer_inputs.project(
                        backend,
                        token,
                        embedding_hidden.as_slice(),
                        self.layers.len(),
                        hidden_size,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    projected_buffer
                        .write_f32(projected.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
            }
            bytes_moved =
                bytes_moved.saturating_add(per_layer_inputs.model_proj.byte_length() as u64);
        }

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_plan = step_plan
                .layer_plans
                .get_mut(layer_index)
                .and_then(Option::as_mut)
                .ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "gemma4 metal layer {layer_index} is missing a decode step plan",
                    )))
                })?;
            backend
                .encode_per_head_rms_norm_to_output_submission(
                    &mut submission,
                    &step_plan.hidden_buffer,
                    layer.attention_norm.device(),
                    &step_plan.norm_buffer,
                    1,
                    hidden_size,
                    self.family_metadata.rms_norm_epsilon,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;

            if layer.attention_geometry.has_kv() {
                let key_buffer = layer_plan.key_buffer.as_ref().ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "gemma4 metal layer {layer_index} is missing a key buffer",
                    )))
                })?;
                let value_buffer = layer_plan.value_buffer.as_ref().ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "gemma4 metal layer {layer_index} is missing a value buffer",
                    )))
                })?;
                let layer_cache = layer_caches
                    .get_mut(layer_index)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 metal layer {layer_index} is missing a device kv cache mirror",
                            ),
                        ))
                    })?;
                if let (Some(qkv_weight), Some(qkv_buffer)) = (
                    layer.attention_qkv_weight.as_ref(),
                    layer_plan.qkv_buffer.as_ref(),
                ) {
                    backend
                        .encode_quantized_matvec_submission(
                            &mut submission,
                            qkv_weight.native_weights()?,
                            0,
                            qkv_weight.mode,
                            qkv_weight.rows,
                            qkv_weight.columns,
                            &step_plan.norm_buffer,
                            qkv_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                } else {
                    backend
                        .encode_quantized_matvec_submission(
                            &mut submission,
                            layer.attention_query_weight.native_weights()?,
                            0,
                            layer.attention_query_weight.mode,
                            layer.attention_query_weight.rows,
                            layer.attention_query_weight.columns,
                            &step_plan.norm_buffer,
                            &layer_plan.query_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    backend
                        .encode_quantized_matvec_submission(
                            &mut submission,
                            layer.attention_key_weight.native_weights()?,
                            0,
                            layer.attention_key_weight.mode,
                            layer.attention_key_weight.rows,
                            layer.attention_key_weight.columns,
                            &step_plan.norm_buffer,
                            key_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    backend
                        .encode_quantized_matvec_submission(
                            &mut submission,
                            layer.attention_value_weight.native_weights()?,
                            0,
                            layer.attention_value_weight.mode,
                            layer.attention_value_weight.rows,
                            layer.attention_value_weight.columns,
                            &step_plan.norm_buffer,
                            value_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                if let Some(bias) = layer.attention_query_bias.as_ref() {
                    backend
                        .encode_add_inplace_submission(
                            &mut submission,
                            &layer_plan.query_buffer,
                            bias.device(),
                            layer.attention_query_weight.rows,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                if let Some(norm) = layer.attention_query_norm.as_ref() {
                    backend
                        .encode_per_head_rms_norm_submission(
                            &mut submission,
                            &layer_plan.query_buffer,
                            norm.device(),
                            layer.attention_geometry.head_count,
                            layer.attention_geometry.head_dim,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                backend
                    .encode_rope_neox_position_submission(
                        &mut submission,
                        &layer_plan.query_buffer,
                        &layer_plan.rope_freq_factors_buffer,
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.head_dim,
                        layer_plan.rope_rotary_half,
                        position,
                        layer_plan.rope_theta_scale,
                        layer_plan.rope_freq_scale,
                        layer_plan.rope_corr_dims,
                        layer_plan.rope_ext_factor,
                        layer_plan.rope_yarn_mscale,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                if let Some(bias) = layer.attention_key_bias.as_ref() {
                    backend
                        .encode_add_inplace_submission(
                            &mut submission,
                            key_buffer,
                            bias.device(),
                            layer.attention_key_weight.rows,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                if let Some(norm) = layer.attention_key_norm.as_ref() {
                    backend
                        .encode_per_head_rms_norm_submission(
                            &mut submission,
                            key_buffer,
                            norm.device(),
                            layer.attention_geometry.kv_head_count,
                            layer.attention_geometry.head_dim,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                backend
                    .encode_rope_neox_position_submission(
                        &mut submission,
                        key_buffer,
                        &layer_plan.rope_freq_factors_buffer,
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        layer_plan.rope_rotary_half,
                        position,
                        layer_plan.rope_theta_scale,
                        layer_plan.rope_freq_scale,
                        layer_plan.rope_corr_dims,
                        layer_plan.rope_ext_factor,
                        layer_plan.rope_yarn_mscale,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                if let Some(bias) = layer.attention_value_bias.as_ref() {
                    backend
                        .encode_add_inplace_submission(
                            &mut submission,
                            value_buffer,
                            bias.device(),
                            layer.attention_value_weight.rows,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                backend
                    .encode_per_head_rms_norm_unit_submission(
                        &mut submission,
                        value_buffer,
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        self.family_metadata.rms_norm_epsilon,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_append_dense_kv_submission(
                        &mut submission,
                        layer_cache,
                        key_buffer,
                        value_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_decode_attention_dense_submission(
                        &mut submission,
                        &layer_plan.query_buffer,
                        layer_cache,
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        attention_scale(&self.family_metadata, layer.attention_geometry.head_dim),
                        &layer_plan.attention_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            } else {
                let reuse_layer_index =
                    layer.attention_geometry.reuse_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} is missing reused kv source metadata",
                            ),
                        ))
                    })?;
                let source_layer_cache = layer_caches
                    .get(reuse_layer_index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} expected a device kv cache mirror from layer {reuse_layer_index}",
                            ),
                        ))
                    })?;
                backend
                    .encode_quantized_matvec_submission(
                        &mut submission,
                        layer.attention_query_weight.native_weights()?,
                        0,
                        layer.attention_query_weight.mode,
                        layer.attention_query_weight.rows,
                        layer.attention_query_weight.columns,
                        &step_plan.norm_buffer,
                        &layer_plan.query_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                if let Some(bias) = layer.attention_query_bias.as_ref() {
                    backend
                        .encode_add_inplace_submission(
                            &mut submission,
                            &layer_plan.query_buffer,
                            bias.device(),
                            layer.attention_query_weight.rows,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                if let Some(norm) = layer.attention_query_norm.as_ref() {
                    backend
                        .encode_per_head_rms_norm_submission(
                            &mut submission,
                            &layer_plan.query_buffer,
                            norm.device(),
                            layer.attention_geometry.head_count,
                            layer.attention_geometry.head_dim,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                backend
                    .encode_rope_neox_position_submission(
                        &mut submission,
                        &layer_plan.query_buffer,
                        &layer_plan.rope_freq_factors_buffer,
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.head_dim,
                        layer_plan.rope_rotary_half,
                        position,
                        layer_plan.rope_theta_scale,
                        layer_plan.rope_freq_scale,
                        layer_plan.rope_corr_dims,
                        layer_plan.rope_ext_factor,
                        layer_plan.rope_yarn_mscale,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_decode_attention_dense_submission(
                        &mut submission,
                        &layer_plan.query_buffer,
                        source_layer_cache,
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        attention_scale(&self.family_metadata, layer.attention_geometry.head_dim),
                        &layer_plan.attention_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            }

            backend
                .encode_quantized_matvec_submission(
                    &mut submission,
                    layer.attention_output_weight.native_weights()?,
                    0,
                    layer.attention_output_weight.mode,
                    layer.attention_output_weight.rows,
                    layer.attention_output_weight.columns,
                    &layer_plan.attention_buffer,
                    &step_plan.residual_buffer,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;
            if let Some(bias) = layer.attention_output_bias.as_ref() {
                backend
                    .encode_add_inplace_submission(
                        &mut submission,
                        &step_plan.residual_buffer,
                        bias.device(),
                        hidden_size,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            }
            if let Some(norm) = layer.attention_post_norm.as_ref() {
                backend
                    .encode_per_head_rms_norm_submission(
                        &mut submission,
                        &step_plan.residual_buffer,
                        norm.device(),
                        1,
                        hidden_size,
                        self.family_metadata.rms_norm_epsilon,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            }
            backend
                .encode_add_inplace_submission(
                    &mut submission,
                    &step_plan.residual_buffer,
                    &step_plan.hidden_buffer,
                    hidden_size,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;

            backend
                .encode_per_head_rms_norm_to_output_submission(
                    &mut submission,
                    &step_plan.residual_buffer,
                    layer.feed_forward_norm.device(),
                    &step_plan.norm_buffer,
                    1,
                    hidden_size,
                    self.family_metadata.rms_norm_epsilon,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let dense_gate_weight = layer.feed_forward_gate_weight.as_ref().ok_or_else(|| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                    "missing gemma4 dense feed_forward_gate_weight in local metal decode path",
                )))
            })?;
            let dense_up_weight = layer.feed_forward_up_weight.as_ref().ok_or_else(|| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                    "missing gemma4 dense feed_forward_up_weight in local metal decode path",
                )))
            })?;
            let dense_down_weight = layer.feed_forward_down_weight.as_ref().ok_or_else(|| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                    "missing gemma4 dense feed_forward_down_weight in local metal decode path",
                )))
            })?;
            if let (Some(gate_up_weight), Some(gate_up_buffer)) = (
                layer.feed_forward_gate_up_weight.as_ref(),
                step_plan.ffn_gate_up_buffer.as_ref(),
            ) {
                backend
                    .encode_quantized_matvec_submission(
                        &mut submission,
                        gate_up_weight.native_weights()?,
                        0,
                        gate_up_weight.mode,
                        gate_up_weight.rows,
                        gate_up_weight.columns,
                        &step_plan.norm_buffer,
                        gate_up_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            } else {
                backend
                    .encode_quantized_matvec_submission(
                        &mut submission,
                        dense_gate_weight.native_weights()?,
                        0,
                        dense_gate_weight.mode,
                        dense_gate_weight.rows,
                        dense_gate_weight.columns,
                        &step_plan.norm_buffer,
                        &step_plan.ffn_gate_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_quantized_matvec_submission(
                        &mut submission,
                        dense_up_weight.native_weights()?,
                        0,
                        dense_up_weight.mode,
                        dense_up_weight.rows,
                        dense_up_weight.columns,
                        &step_plan.norm_buffer,
                        &step_plan.ffn_up_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            }
            backend
                .encode_gelu_glu_submission(
                    &mut submission,
                    &step_plan.ffn_gate_buffer,
                    &step_plan.ffn_up_buffer,
                    &step_plan.ffn_activated_buffer,
                    dense_gate_weight.rows,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;
            backend
                .encode_quantized_matvec_submission(
                    &mut submission,
                    dense_down_weight.native_weights()?,
                    0,
                    dense_down_weight.mode,
                    dense_down_weight.rows,
                    dense_down_weight.columns,
                    &step_plan.ffn_activated_buffer,
                    &step_plan.hidden_buffer,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;
            if let Some(norm) = layer.feed_forward_post_norm.as_ref() {
                backend
                    .encode_per_head_rms_norm_submission(
                        &mut submission,
                        &step_plan.hidden_buffer,
                        norm.device(),
                        1,
                        hidden_size,
                        self.family_metadata.rms_norm_epsilon,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            }
            backend
                .encode_add_inplace_submission(
                    &mut submission,
                    &step_plan.hidden_buffer,
                    &step_plan.residual_buffer,
                    hidden_size,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;

            if let (
                Some(input_gate),
                Some(proj),
                Some(post_norm),
                Some(multiplier_buffer),
                Some(per_layer_gate_buffer),
                Some(per_layer_activated_buffer),
                Some(per_layer_projected_buffer),
            ) = (
                layer.per_layer_input_gate.as_ref(),
                layer.per_layer_proj.as_ref(),
                layer.per_layer_post_norm.as_ref(),
                layer_plan.per_layer_multiplier_buffer.as_ref(),
                step_plan.per_layer_gate_buffer.as_ref(),
                step_plan.per_layer_activated_buffer.as_ref(),
                step_plan.per_layer_projected_buffer.as_ref(),
            ) {
                backend
                    .encode_quantized_matvec_submission(
                        &mut submission,
                        input_gate.native_weights()?,
                        0,
                        input_gate.mode,
                        input_gate.rows,
                        input_gate.columns,
                        &step_plan.hidden_buffer,
                        per_layer_gate_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_gelu_glu_submission(
                        &mut submission,
                        per_layer_gate_buffer,
                        multiplier_buffer,
                        per_layer_activated_buffer,
                        input_gate.rows,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_quantized_matvec_submission(
                        &mut submission,
                        proj.native_weights()?,
                        0,
                        proj.mode,
                        proj.rows,
                        proj.columns,
                        per_layer_activated_buffer,
                        per_layer_projected_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_per_head_rms_norm_submission(
                        &mut submission,
                        per_layer_projected_buffer,
                        post_norm.device(),
                        1,
                        hidden_size,
                        self.family_metadata.rms_norm_epsilon,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_add_inplace_submission(
                        &mut submission,
                        &step_plan.hidden_buffer,
                        per_layer_projected_buffer,
                        hidden_size,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                bytes_moved = bytes_moved
                    .saturating_add(input_gate.byte_length() as u64)
                    .saturating_add(proj.byte_length() as u64);
            }

            if let Some(scale) = layer.layer_output_scale {
                backend
                    .encode_scale_inplace_submission(
                        &mut submission,
                        &step_plan.hidden_buffer,
                        scale,
                        hidden_size,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            }

            bytes_moved = bytes_moved
                .saturating_add(layer.attention_query_weight.byte_length() as u64)
                .saturating_add(layer.attention_output_weight.byte_length() as u64)
                .saturating_add(dense_gate_weight.byte_length() as u64)
                .saturating_add(dense_up_weight.byte_length() as u64)
                .saturating_add(dense_down_weight.byte_length() as u64);
            if layer.attention_geometry.has_kv() {
                bytes_moved = bytes_moved
                    .saturating_add(layer.attention_key_weight.byte_length() as u64)
                    .saturating_add(layer.attention_value_weight.byte_length() as u64);
            }
        }

        let mut logits = Vec::new();
        let mut selected_token = None;
        match (produce_logits, logits_output_mode) {
            (false, _) => {}
            (true, MetalLogitsOutputMode::GreedyToken) => {
                backend
                    .encode_per_head_rms_norm_to_output_submission(
                        &mut submission,
                        &step_plan.hidden_buffer,
                        self.output_norm.device(),
                        &step_plan.norm_buffer,
                        1,
                        hidden_size,
                        self.family_metadata.rms_norm_epsilon,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                if materialize_stage_outputs {
                    submission
                        .synchronize_buffer(&step_plan.norm_buffer)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                backend
                    .encode_quantized_matvec_argmax_submission(
                        &mut submission,
                        self.output.native_weights()?,
                        0,
                        self.output.mode,
                        self.output.rows,
                        self.output.columns,
                        &step_plan.norm_buffer,
                        &step_plan.argmax_candidates_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                backend
                    .encode_argmax_candidates_submission(
                        &mut submission,
                        &step_plan.argmax_candidates_buffer,
                        &step_plan.selected_token_buffer,
                        psionic_backend_metal::quantized_argmax_candidate_count(self.output.rows),
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                submission
                    .synchronize_buffer(&step_plan.selected_token_buffer)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                bytes_moved = bytes_moved.saturating_add(self.output.byte_length() as u64);
            }
            (true, MetalLogitsOutputMode::RawLogits) => {
                backend
                    .encode_per_head_rms_norm_to_output_submission(
                        &mut submission,
                        &step_plan.hidden_buffer,
                        self.output_norm.device(),
                        &step_plan.norm_buffer,
                        1,
                        hidden_size,
                        self.family_metadata.rms_norm_epsilon,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                if materialize_stage_outputs {
                    submission
                        .synchronize_buffer(&step_plan.norm_buffer)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                backend
                    .encode_quantized_matvec_submission(
                        &mut submission,
                        self.output.native_weights()?,
                        0,
                        self.output.mode,
                        self.output.rows,
                        self.output.columns,
                        &step_plan.norm_buffer,
                        &step_plan.logits_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                submission
                    .synchronize_buffer(&step_plan.logits_buffer)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                bytes_moved = bytes_moved.saturating_add(self.output.byte_length() as u64);
            }
            (true, MetalLogitsOutputMode::TopKCandidates(_)) => {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(String::from(
                        "metal gemma4 top-k logits output mode is not yet wired into the local device-resident generation loop",
                    )),
                ));
            }
        }
        if materialize_stage_outputs {
            submission
                .synchronize_buffer(&step_plan.hidden_buffer)
                .map_err(ReferenceTextGenerationError::Runtime)?;
        }

        let report = submission
            .commit(psionic_backend_metal::MetalCommandWait::Completed)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let hidden = if materialize_stage_outputs {
            step_plan
                .hidden_buffer
                .read_f32()
                .map_err(ReferenceTextGenerationError::Runtime)?
        } else {
            Vec::new()
        };
        let lm_head_hidden = if produce_logits && materialize_stage_outputs {
            Some(
                step_plan
                    .norm_buffer
                    .read_f32()
                    .map_err(ReferenceTextGenerationError::Runtime)?,
            )
        } else {
            None
        };
        if produce_logits {
            match logits_output_mode {
                MetalLogitsOutputMode::GreedyToken => {
                    selected_token = Some(TokenId(
                        psionic_backend_metal::read_argmax_candidate_index(
                            &step_plan.selected_token_buffer,
                            "metal gemma4 greedy token",
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?,
                    ));
                }
                MetalLogitsOutputMode::RawLogits => {
                    logits = step_plan
                        .logits_buffer
                        .read_f32()
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    apply_final_logit_softcapping_in_place(
                        logits.as_mut_slice(),
                        self.family_metadata.final_logit_softcapping,
                    );
                }
                MetalLogitsOutputMode::TopKCandidates(_) => {}
            }
        }
        if materialize_stage_outputs {
            for (layer_index, layer) in self
                .layers
                .iter()
                .enumerate()
                .skip(start_layer)
                .take(end_layer.saturating_sub(start_layer))
            {
                let Some(cache_offset) = layer.attention_geometry.cache_write_offset else {
                    continue;
                };
                let layer_cache = layer_caches
                    .get(layer_index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 metal layer {layer_index} is missing a device kv cache mirror"
                            ),
                        ))
                    })?;
                let (layer_key, layer_value) = layer_cache
                    .read_entry(position)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                let kv_width = layer.attention_geometry.kv_width();
                cache_key[cache_offset..cache_offset + kv_width]
                    .copy_from_slice(layer_key.as_slice());
                cache_value[cache_offset..cache_offset + kv_width]
                    .copy_from_slice(layer_value.as_slice());
            }
        }

        Ok(MetalGemma4StageStep {
            key: cache_key,
            value: cache_value,
            logits,
            selected_token,
            hidden,
            lm_head_hidden,
            kernel_count: report.encoded_operations,
            bytes_moved,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step_with_layer_caches(
        &self,
        backend: &mut MetalBackend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
        layer_caches: &mut [Option<MetalKvCacheMirror>],
        step_plan: &mut MetalGemma4StepPlan,
        start_layer: usize,
        end_layer: usize,
        input_hidden: Option<&[f32]>,
        forwarded_key: Option<&[f32]>,
        forwarded_value: Option<&[f32]>,
        produce_logits: bool,
        logits_output_mode: MetalLogitsOutputMode,
    ) -> Result<MetalGemma4StageStep, ReferenceTextGenerationError> {
        if start_layer > end_layer || end_layer > self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "invalid gemma4 metal stage range [{start_layer}..{end_layer}) for {} layers",
                    self.layers.len()
                )),
            ));
        }
        if layer_caches.len() != self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 metal layer cache count mismatch: expected {}, actual {}",
                    self.layers.len(),
                    layer_caches.len()
                )),
            ));
        }
        if step_plan.layer_plans.len() != self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 metal step-plan layer count mismatch: expected {}, actual {}",
                    self.layers.len(),
                    step_plan.layer_plans.len()
                )),
            ));
        }
        if self.layers[start_layer..end_layer].iter().all(|layer| {
            backend.supports_dense_decode_attention(layer.attention_geometry.head_dim)
                && layer.attention_query_weight.is_native()
                && layer.attention_key_weight.is_native()
                && layer.attention_value_weight.is_native()
                && layer.attention_output_weight.is_native()
                && layer
                    .feed_forward_gate_weight
                    .as_ref()
                    .map(MetalQuantizedProjectionMatrix::is_native)
                    .unwrap_or(true)
                && layer
                    .feed_forward_gate_up_weight
                    .as_ref()
                    .map(MetalQuantizedProjectionMatrix::is_native)
                    .unwrap_or(true)
                && layer
                    .feed_forward_up_weight
                    .as_ref()
                    .map(MetalQuantizedProjectionMatrix::is_native)
                    .unwrap_or(true)
                && layer
                    .feed_forward_down_weight
                    .as_ref()
                    .map(MetalQuantizedProjectionMatrix::is_native)
                    .unwrap_or(true)
                && layer
                    .per_layer_input_gate
                    .as_ref()
                    .map(MetalQuantizedProjectionMatrix::is_native)
                    .unwrap_or(true)
                && layer
                    .per_layer_proj
                    .as_ref()
                    .map(MetalQuantizedProjectionMatrix::is_native)
                    .unwrap_or(true)
                && layer.feed_forward_router_weight.is_none()
                && layer.feed_forward_gate_experts_weight.is_none()
                && layer.feed_forward_up_experts_weight.is_none()
                && layer.feed_forward_down_experts_weight.is_none()
        }) && self.output.is_native()
            && self
                .gemma4_per_layer_inputs
                .as_ref()
                .map(|inputs| {
                    inputs.model_proj.is_native()
                        || matches!(
                            inputs.model_proj,
                            MetalGemma4PerLayerProjectionMatrix::Host(_)
                        )
                })
                .unwrap_or(true)
        {
            return self.forward_local_step_with_layer_caches(
                backend,
                token,
                position,
                layer_caches,
                step_plan,
                start_layer,
                end_layer,
                input_hidden,
                forwarded_key,
                forwarded_value,
                true,
                produce_logits,
                logits_output_mode,
            );
        }

        let mut bytes_moved = self.token_embedding.byte_length() as u64;
        let mut kernel_count = 1usize;
        let mut embedding_hidden = self
            .token_embedding
            .decode_row(token.as_u32() as usize)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        scale_in_place(
            &mut embedding_hidden,
            input_embedding_scale(&self.family_metadata, self.descriptor.config.hidden_size),
        );
        let mut hidden = if let Some(input_hidden) = input_hidden {
            if input_hidden.len() != self.descriptor.config.hidden_size {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "gemma4 metal stage input hidden width mismatch: expected {}, actual {}",
                        self.descriptor.config.hidden_size,
                        input_hidden.len()
                    )),
                ));
            }
            input_hidden.to_vec()
        } else {
            embedding_hidden.clone()
        };
        let gemma4_per_layer_inputs =
            if let Some(per_layer_inputs) = self.gemma4_per_layer_inputs.as_ref() {
                Some(per_layer_inputs.project(
                    backend,
                    token,
                    embedding_hidden.as_slice(),
                    self.layers.len(),
                    self.descriptor.config.hidden_size,
                    self.family_metadata.rms_norm_epsilon,
                )?)
            } else {
                None
            };
        let mut cache_key = forwarded_key
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        let mut cache_value = forwarded_value
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        if cache_key.len() != self.cache_width() || cache_value.len() != self.cache_width() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 metal forwarded kv width mismatch: expected {}, actual key={} value={}",
                    self.cache_width(),
                    cache_key.len(),
                    cache_value.len()
                )),
            ));
        }
        let mut live_keys: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        let mut live_values: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        if start_layer > 0 {
            for (layer_index, layer) in self.layers.iter().enumerate().take(start_layer) {
                if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                    let kv_width = layer.attention_geometry.kv_width();
                    live_keys[layer_index] =
                        Some(cache_key[cache_offset..cache_offset + kv_width].to_vec());
                    live_values[layer_index] =
                        Some(cache_value[cache_offset..cache_offset + kv_width].to_vec());
                }
            }
        }

        for (layer_index, layer) in self
            .layers
            .iter()
            .enumerate()
            .skip(start_layer)
            .take(end_layer.saturating_sub(start_layer))
        {
            let hidden_norm = rms_norm(
                hidden.as_slice(),
                layer.attention_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let residual = std::mem::take(&mut hidden);

            let mut kv_kernel_count = 0usize;
            let mut kv_bytes_moved = 0u64;
            let mut q_kernel_count = 0usize;
            let mut q_bytes_moved = 0u64;
            let mut device_attention: Option<MetalDeviceAttentionStep> = None;
            let (fallback_q, k_values, v_values, fresh_attention_values) = if layer
                .attention_geometry
                .has_kv()
            {
                let layer_step_plan = step_plan
                    .layer_plans
                    .get_mut(layer_index)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 metal layer {layer_index} is missing a decode step plan"
                            ),
                        ))
                    })?;
                let layer_cache = layer_caches
                        .get_mut(layer_index)
                        .and_then(Option::as_mut)
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                format!(
                                    "gemma4 metal layer {layer_index} is missing a device kv cache mirror"
                                ),
                            ))
                        })?;
                if let Some(device_step) = metal_gemma4_device_attention_with_kv(
                    backend,
                    layer,
                    hidden_norm.as_slice(),
                    position,
                    layer_index,
                    &self.family_metadata,
                    layer_cache,
                    layer_step_plan,
                )? {
                    if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                        let kv_width = layer.attention_geometry.kv_width();
                        let key_values = device_step
                            .key_values
                            .as_deref()
                            .expect("device attention with kv returns key values");
                        let value_values = device_step
                            .value_values
                            .as_deref()
                            .expect("device attention with kv returns value values");
                        cache_key[cache_offset..cache_offset + kv_width]
                            .copy_from_slice(key_values);
                        cache_value[cache_offset..cache_offset + kv_width]
                            .copy_from_slice(value_values);
                        live_keys[layer_index] = Some(key_values.to_vec());
                        live_values[layer_index] = Some(value_values.to_vec());
                    }
                    device_attention = Some(device_step);
                    (None, &[][..], &[][..], None)
                } else {
                    let (mut q, mut k, mut v) = MetalQuantizedProjectionMatrix::matvec_triplet(
                        &layer.attention_query_weight,
                        &layer.attention_key_weight,
                        &layer.attention_value_weight,
                        backend,
                        &hidden_norm,
                    )?;
                    if let Some(bias) = layer.attention_query_bias.as_ref() {
                        add_bias_in_place(&mut q.values, bias.as_slice());
                    }
                    if let Some(norm) = layer.attention_query_norm.as_ref() {
                        per_head_rms_norm_in_place(
                            q.values.as_mut_slice(),
                            layer.attention_geometry.head_count,
                            layer.attention_geometry.head_dim,
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    if let Some(bias) = layer.attention_key_bias.as_ref() {
                        add_bias_in_place(&mut k.values, bias.as_slice());
                    }
                    if let Some(norm) = layer.attention_key_norm.as_ref() {
                        per_head_rms_norm_in_place(
                            k.values.as_mut_slice(),
                            layer.attention_geometry.kv_head_count,
                            layer.attention_geometry.head_dim,
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    if let Some(bias) = layer.attention_value_bias.as_ref() {
                        add_bias_in_place(&mut v.values, bias.as_slice());
                    }
                    per_head_rms_norm_unit_in_place(
                        v.values.as_mut_slice(),
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        self.family_metadata.rms_norm_epsilon,
                    );
                    let mut cache_layer_key = k.values.clone();
                    apply_rope_neox(
                        &mut cache_layer_key,
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        layer.attention_geometry.rotary_dim,
                        position,
                        layer.attention_geometry.rope_theta,
                        self.rope_freq_factors_for_layer(
                            layer_index,
                            layer.attention_geometry.head_dim,
                        ),
                        &self.family_metadata,
                    );
                    if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                        let kv_width = layer.attention_geometry.kv_width();
                        cache_key[cache_offset..cache_offset + kv_width]
                            .copy_from_slice(cache_layer_key.as_slice());
                        cache_value[cache_offset..cache_offset + kv_width]
                            .copy_from_slice(v.values.as_slice());
                    }
                    kv_kernel_count = kv_kernel_count
                        .saturating_add(k.kernel_count)
                        .saturating_add(v.kernel_count)
                        .saturating_add(1);
                    kv_bytes_moved = kv_bytes_moved
                        .saturating_add(k.bytes_moved)
                        .saturating_add(v.bytes_moved)
                        .saturating_add(layer.attention_geometry.kv_width() as u64);
                    live_keys[layer_index] = Some(cache_layer_key);
                    live_values[layer_index] = Some(v.values.clone());

                    let (cos_values, sin_values) = metal_gemma4_rope_cos_sin_values(
                        position,
                        layer.attention_geometry.head_dim,
                        layer.attention_geometry.rotary_dim,
                        layer.attention_geometry.rope_theta,
                        self.rope_freq_factors_for_layer(
                            layer_index,
                            layer.attention_geometry.head_dim,
                        ),
                        &self.family_metadata,
                    );
                    let attention = backend
                        .decode_attention_values_f32(
                            q.values.as_slice(),
                            layer.attention_geometry.head_count,
                            k.values.as_slice(),
                            layer.attention_geometry.kv_head_count,
                            v.values.as_slice(),
                            cos_values.as_slice(),
                            sin_values.as_slice(),
                            layer_cache,
                            attention_scale(
                                &self.family_metadata,
                                layer.attention_geometry.head_dim,
                            ),
                            true,
                            false,
                            true,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    layer_step_plan.attention_values = attention.output_values;
                    q_kernel_count = q.kernel_count;
                    q_bytes_moved = q.bytes_moved;
                    (
                            Some(q),
                            live_keys[layer_index].as_deref().ok_or_else(|| {
                                ReferenceTextGenerationError::Runtime(
                                    crate::RuntimeError::Backend(format!(
                                        "gemma4 layer {layer_index} lost live key cache after publication"
                                    )),
                                )
                            })?,
                            live_values[layer_index].as_deref().ok_or_else(|| {
                                ReferenceTextGenerationError::Runtime(
                                    crate::RuntimeError::Backend(format!(
                                        "gemma4 layer {layer_index} lost live value cache after publication"
                                    )),
                                )
                            })?,
                            Some(layer_step_plan.attention_values.as_slice()),
                        )
                }
            } else {
                let layer_step_plan = step_plan
                        .layer_plans
                        .get_mut(layer_index)
                        .and_then(Option::as_mut)
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                format!(
                                    "gemma4 metal layer {layer_index} is missing a reused-kv decode step plan"
                                ),
                            ))
                        })?;
                let reuse_layer_index =
                    layer.attention_geometry.reuse_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} is missing reused kv source metadata"
                            ),
                        ))
                    })?;
                let source_layer_cache = layer_caches
                        .get(reuse_layer_index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                format!(
                                    "gemma4 layer {layer_index} expected a device kv cache mirror from layer {reuse_layer_index}"
                                ),
                            ))
                        })?;
                let source_cache_view = source_layer_cache.clone();
                if let Some(device_step) = metal_gemma4_device_attention_reuse(
                    backend,
                    layer,
                    hidden_norm.as_slice(),
                    position,
                    layer_index,
                    &self.family_metadata,
                    &source_cache_view,
                    layer_step_plan,
                )? {
                    device_attention = Some(device_step);
                    (None, &[][..], &[][..], None)
                } else {
                    let mut q = layer.attention_query_weight.matvec(backend, &hidden_norm)?;
                    if let Some(bias) = layer.attention_query_bias.as_ref() {
                        add_bias_in_place(&mut q.values, bias.as_slice());
                    }
                    if let Some(norm) = layer.attention_query_norm.as_ref() {
                        per_head_rms_norm_in_place(
                            q.values.as_mut_slice(),
                            layer.attention_geometry.head_count,
                            layer.attention_geometry.head_dim,
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    apply_rope_neox(
                        &mut q.values,
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.head_dim,
                        layer.attention_geometry.rotary_dim,
                        position,
                        layer.attention_geometry.rope_theta,
                        self.rope_freq_factors_for_layer(
                            layer_index,
                            layer.attention_geometry.head_dim,
                        ),
                        &self.family_metadata,
                    );
                    let reused_k = live_keys
                            .get(reuse_layer_index)
                            .and_then(Option::as_deref)
                            .ok_or_else(|| {
                                ReferenceTextGenerationError::Runtime(
                                    crate::RuntimeError::Backend(format!(
                                        "gemma4 layer {layer_index} expected live key cache from layer {reuse_layer_index}"
                                    )),
                                )
                            })?;
                    let reused_v = live_values
                            .get(reuse_layer_index)
                            .and_then(Option::as_deref)
                            .ok_or_else(|| {
                                ReferenceTextGenerationError::Runtime(
                                    crate::RuntimeError::Backend(format!(
                                        "gemma4 layer {layer_index} expected live value cache from layer {reuse_layer_index}"
                                    )),
                                )
                            })?;
                    let mut source_cache_view = source_layer_cache.clone();
                    let attention = backend
                        .decode_attention_values_f32(
                            q.values.as_slice(),
                            layer.attention_geometry.head_count,
                            reused_k,
                            layer.attention_geometry.kv_head_count,
                            reused_v,
                            layer_step_plan.identity_cos_values.as_slice(),
                            layer_step_plan.identity_sin_values.as_slice(),
                            &mut source_cache_view,
                            attention_scale(
                                &self.family_metadata,
                                layer.attention_geometry.head_dim,
                            ),
                            true,
                            false,
                            true,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    layer_step_plan.attention_values = attention.output_values;
                    q_kernel_count = q.kernel_count;
                    q_bytes_moved = q.bytes_moved;
                    kernel_count = kernel_count.saturating_add(1);
                    (
                        Some(q),
                        reused_k,
                        reused_v,
                        Some(layer_step_plan.attention_values.as_slice()),
                    )
                }
            };

            let mut attention_out = if let Some(device_step) = device_attention {
                MetalProjectionStep {
                    values: device_step.projected_output,
                    kernel_count: device_step.kernel_count,
                    bytes_moved: device_step.bytes_moved,
                }
            } else {
                let q = fallback_q.as_ref().ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "gemma4 metal layer {layer_index} lost fallback query state",
                    )))
                })?;
                let attention_values = if let Some(values) = fresh_attention_values {
                    values.to_vec()
                } else {
                    attend_impl(
                        layer_index,
                        q.values.as_slice(),
                        k_values,
                        v_values,
                        cache,
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        layer.attention_geometry.cache_read_offset,
                        layer.attention_geometry.sliding_window,
                        attention_scale(&self.family_metadata, layer.attention_geometry.head_dim),
                    )
                };
                q_kernel_count = q.kernel_count;
                q_bytes_moved = q.bytes_moved;
                layer
                    .attention_output_weight
                    .matvec(backend, attention_values.as_slice())?
            };
            if let Some(bias) = layer.attention_output_bias.as_ref() {
                add_bias_in_place(&mut attention_out.values, bias.as_slice());
            }
            if let Some(norm) = layer.attention_post_norm.as_ref() {
                rms_norm_in_place(
                    attention_out.values.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(attention_out.values.as_mut_slice(), residual.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            hidden = attention_out.values;

            let ffn_input = rms_norm(
                hidden.as_slice(),
                layer.feed_forward_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let ffn_residual = std::mem::take(&mut hidden);
            let mut ffn_out = if self.family_metadata.family == GgufDecoderFamily::Gemma4
                && layer.feed_forward_router_weight.is_some()
                && layer.feed_forward_gate_experts_weight.is_some()
                && layer.feed_forward_down_experts_weight.is_some()
            {
                let mut shared_ffn =
                    MetalQuantizedProjectionMatrix::matvec_pair_gelu_glu_projected(
                        layer.feed_forward_gate_weight.as_ref().ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_gate_weight"),
                            ))
                        })?,
                        layer.feed_forward_up_weight.as_ref().ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_up_weight"),
                            ))
                        })?,
                        layer.feed_forward_down_weight.as_ref().ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_down_weight"),
                            ))
                        })?,
                        backend,
                        &ffn_input,
                    )?;
                if let Some(norm) = layer.feed_forward_post_norm_1.as_ref() {
                    rms_norm_in_place(
                        shared_ffn.values.as_mut_slice(),
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }
                let mut router_input = rms_norm_unit(
                    ffn_residual.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
                scale_in_place(
                    router_input.as_mut_slice(),
                    (self.descriptor.config.hidden_size as f32).sqrt().recip(),
                );
                if let Some(scale) = layer.feed_forward_router_input_scale.as_ref() {
                    multiply_vectors_in_place(router_input.as_mut_slice(), scale.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                let expert_input = if let Some(norm) = layer.feed_forward_pre_norm_2.as_ref() {
                    rms_norm(
                        ffn_residual.as_slice(),
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    )
                } else {
                    ffn_input.clone()
                };
                let mut moe_ffn = evaluate_sparse_moe_host(
                    &self.family_metadata,
                    router_input.as_slice(),
                    expert_input.as_slice(),
                    layer
                        .feed_forward_router_weight
                        .as_ref()
                        .expect("router checked"),
                    layer.feed_forward_router_bias.as_deref(),
                    layer
                        .feed_forward_gate_experts_weight
                        .as_ref()
                        .expect("gate experts checked"),
                    layer.feed_forward_gate_experts_bias.as_deref(),
                    layer.feed_forward_up_experts_weight.as_ref(),
                    layer.feed_forward_up_experts_bias.as_deref(),
                    layer
                        .feed_forward_down_experts_weight
                        .as_ref()
                        .expect("down experts checked"),
                    layer.feed_forward_down_experts_bias.as_deref(),
                    layer.feed_forward_down_experts_scale.as_deref(),
                    self.family_metadata.expert_used_count.unwrap_or(1),
                )?;
                if let Some(norm) = layer.feed_forward_post_norm_2.as_ref() {
                    rms_norm_in_place(
                        moe_ffn.values.as_mut_slice(),
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }
                MetalProjectionStep {
                    values: add_vectors(shared_ffn.values.as_slice(), moe_ffn.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?,
                    kernel_count: shared_ffn.kernel_count + moe_ffn.kernel_count,
                    bytes_moved: shared_ffn.bytes_moved + moe_ffn.bytes_moved,
                }
            } else if let (Some(router), Some(gate_experts), Some(up_experts), Some(down_experts)) = (
                layer.feed_forward_router_weight.as_ref(),
                layer.feed_forward_gate_experts_weight.as_ref(),
                layer.feed_forward_up_experts_weight.as_ref(),
                layer.feed_forward_down_experts_weight.as_ref(),
            ) {
                let moe_ffn = evaluate_sparse_moe_host(
                    &self.family_metadata,
                    ffn_input.as_slice(),
                    ffn_input.as_slice(),
                    router,
                    layer.feed_forward_router_bias.as_deref(),
                    gate_experts,
                    layer.feed_forward_gate_experts_bias.as_deref(),
                    Some(up_experts),
                    layer.feed_forward_up_experts_bias.as_deref(),
                    down_experts,
                    layer.feed_forward_down_experts_bias.as_deref(),
                    layer.feed_forward_down_experts_scale.as_deref(),
                    self.family_metadata.expert_used_count.unwrap_or(1),
                )?;
                MetalProjectionStep {
                    values: moe_ffn.values,
                    kernel_count: moe_ffn.kernel_count,
                    bytes_moved: moe_ffn.bytes_moved,
                }
            } else if self.family_metadata.family == GgufDecoderFamily::Gemma4 {
                MetalQuantizedProjectionMatrix::matvec_pair_gelu_glu_projected(
                    layer.feed_forward_gate_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing gemma4 dense feed_forward_gate_weight"),
                        ))
                    })?,
                    layer.feed_forward_up_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing gemma4 dense feed_forward_up_weight"),
                        ))
                    })?,
                    layer.feed_forward_down_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing gemma4 dense feed_forward_down_weight"),
                        ))
                    })?,
                    backend,
                    &ffn_input,
                )?
            } else {
                let (mut gate, up) = MetalQuantizedProjectionMatrix::matvec_pair(
                    layer.feed_forward_gate_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing dense feed_forward_gate_weight"),
                        ))
                    })?,
                    layer.feed_forward_up_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing dense feed_forward_up_weight"),
                        ))
                    })?,
                    backend,
                    &ffn_input,
                )?;
                feed_forward_activation_in_place(
                    &self.family_metadata,
                    gate.values.as_mut_slice(),
                    up.values.as_slice(),
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;
                layer
                    .feed_forward_down_weight
                    .as_ref()
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing dense feed_forward_down_weight"),
                        ))
                    })?
                    .matvec(backend, gate.values.as_slice())?
            };
            if let Some(norm) = layer.feed_forward_post_norm.as_ref() {
                rms_norm_in_place(
                    ffn_out.values.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(ffn_out.values.as_mut_slice(), ffn_residual.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            hidden = ffn_out.values;

            if let Some(per_layer_inputs) = gemma4_per_layer_inputs.as_ref() {
                if let (Some(input_gate), Some(proj), Some(post_norm)) = (
                    layer.per_layer_input_gate.as_ref(),
                    layer.per_layer_proj.as_ref(),
                    layer.per_layer_post_norm.as_ref(),
                ) {
                    let multiplier = self
                        .gemma4_per_layer_inputs
                        .as_ref()
                        .expect("per-layer config")
                        .layer_slice(per_layer_inputs.values.as_slice(), layer_index);
                    let mut projected = if input_gate.is_native() && proj.is_native() {
                        MetalProjectionStep {
                            values: backend
                                .quantized_gelu_mul_projected(
                                    input_gate.native_weights()?,
                                    input_gate.mode,
                                    input_gate.rows,
                                    input_gate.columns,
                                    multiplier,
                                    proj.native_weights()?,
                                    proj.mode,
                                    proj.rows,
                                    proj.columns,
                                    hidden.as_slice(),
                                )
                                .map_err(ReferenceTextGenerationError::Runtime)?,
                            kernel_count: 3,
                            bytes_moved: (input_gate.byte_length() + proj.byte_length()) as u64,
                        }
                    } else {
                        let mut gate = input_gate.matvec(backend, hidden.as_slice())?;
                        if gate.values.len() != multiplier.len() {
                            return Err(ReferenceTextGenerationError::Runtime(
                                crate::RuntimeError::Backend(format!(
                                    "gemma4 per-layer multiplier width mismatch: gate={} multiplier={}",
                                    gate.values.len(),
                                    multiplier.len()
                                )),
                            ));
                        }
                        for (gate_value, multiplier_value) in
                            gate.values.iter_mut().zip(multiplier.iter().copied())
                        {
                            *gate_value = approximate_gelu(*gate_value) * multiplier_value;
                        }
                        let mut projected = proj.matvec(backend, gate.values.as_slice())?;
                        projected.kernel_count =
                            projected.kernel_count.saturating_add(gate.kernel_count);
                        projected.bytes_moved =
                            projected.bytes_moved.saturating_add(gate.bytes_moved);
                        projected
                    };
                    rms_norm_in_place(
                        projected.values.as_mut_slice(),
                        post_norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                    add_vectors_in_place(hidden.as_mut_slice(), projected.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    bytes_moved = bytes_moved.saturating_add(projected.bytes_moved);
                    kernel_count = kernel_count.saturating_add(projected.kernel_count);
                }
            }
            if let Some(scale) = layer.layer_output_scale {
                scale_in_place(&mut hidden, scale);
            }

            bytes_moved = bytes_moved
                .saturating_add(q_bytes_moved)
                .saturating_add(attention_out.bytes_moved)
                .saturating_add(ffn_out.bytes_moved);
            kernel_count = kernel_count
                .saturating_add(q_kernel_count)
                .saturating_add(attention_out.kernel_count)
                .saturating_add(ffn_out.kernel_count);
            if layer.attention_geometry.has_kv() {
                bytes_moved = bytes_moved.saturating_add(kv_bytes_moved);
                kernel_count = kernel_count.saturating_add(kv_kernel_count);
            }
        }

        let (logits, selected_token, lm_head_hidden) = if produce_logits {
            let final_hidden = rms_norm(
                hidden.as_slice(),
                self.output_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            match logits_output_mode {
                MetalLogitsOutputMode::GreedyToken => {
                    let selection = self.output.select_logits_output(
                        backend,
                        final_hidden.as_slice(),
                        logits_output_mode,
                    )?;
                    bytes_moved = bytes_moved.saturating_add(self.output.byte_length() as u64);
                    kernel_count = kernel_count.saturating_add(1);
                    let token = selection.selected_tokens.first().copied().ok_or_else(|| {
                        ReferenceTextGenerationError::MissingOutput("metal gemma4 greedy token")
                    })?;
                    (Vec::new(), Some(TokenId(token)), Some(final_hidden))
                }
                MetalLogitsOutputMode::RawLogits => {
                    let mut logits = self.output.matvec(backend, final_hidden.as_slice())?;
                    apply_final_logit_softcapping_in_place(
                        logits.values.as_mut_slice(),
                        self.family_metadata.final_logit_softcapping,
                    );
                    bytes_moved = bytes_moved.saturating_add(logits.bytes_moved);
                    kernel_count = kernel_count.saturating_add(logits.kernel_count);
                    (logits.values, None, Some(final_hidden))
                }
                MetalLogitsOutputMode::TopKCandidates(_) => {
                    return Err(ReferenceTextGenerationError::Runtime(
                        crate::RuntimeError::Backend(String::from(
                            "metal gemma4 top-k logits output mode is not yet wired into the custom generation loop",
                        )),
                    ));
                }
            }
        } else {
            (Vec::new(), None, None)
        };

        Ok(MetalGemma4StageStep {
            key: cache_key,
            value: cache_value,
            logits,
            selected_token,
            hidden,
            lm_head_hidden,
            kernel_count,
            bytes_moved,
        })
    }

    fn forward_stage_step(
        &self,
        backend: &mut MetalBackend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
        start_layer: usize,
        end_layer: usize,
        input_hidden: Option<&[f32]>,
        forwarded_key: Option<&[f32]>,
        forwarded_value: Option<&[f32]>,
        produce_logits: bool,
        logits_output_mode: MetalLogitsOutputMode,
    ) -> Result<MetalGemma4StageStep, ReferenceTextGenerationError> {
        if start_layer > end_layer || end_layer > self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "invalid gemma4 metal stage range [{start_layer}..{end_layer}) for {} layers",
                    self.layers.len()
                )),
            ));
        }
        let mut bytes_moved = self.token_embedding.byte_length() as u64;
        let mut kernel_count = 1usize;
        let mut embedding_hidden = self
            .token_embedding
            .decode_row(token.as_u32() as usize)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        scale_in_place(
            &mut embedding_hidden,
            input_embedding_scale(&self.family_metadata, self.descriptor.config.hidden_size),
        );
        let mut hidden = if let Some(input_hidden) = input_hidden {
            if input_hidden.len() != self.descriptor.config.hidden_size {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "gemma4 metal stage input hidden width mismatch: expected {}, actual {}",
                        self.descriptor.config.hidden_size,
                        input_hidden.len()
                    )),
                ));
            }
            input_hidden.to_vec()
        } else {
            embedding_hidden.clone()
        };
        let gemma4_per_layer_inputs =
            if let Some(per_layer_inputs) = self.gemma4_per_layer_inputs.as_ref() {
                Some(per_layer_inputs.project(
                    backend,
                    token,
                    embedding_hidden.as_slice(),
                    self.layers.len(),
                    self.descriptor.config.hidden_size,
                    self.family_metadata.rms_norm_epsilon,
                )?)
            } else {
                None
            };
        let mut cache_key = forwarded_key
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        let mut cache_value = forwarded_value
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        if cache_key.len() != self.cache_width() || cache_value.len() != self.cache_width() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 metal forwarded kv width mismatch: expected {}, actual key={} value={}",
                    self.cache_width(),
                    cache_key.len(),
                    cache_value.len()
                )),
            ));
        }
        let mut live_keys: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        let mut live_values: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        if start_layer > 0 {
            for (layer_index, layer) in self.layers.iter().enumerate().take(start_layer) {
                if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                    let kv_width = layer.attention_geometry.kv_width();
                    live_keys[layer_index] =
                        Some(cache_key[cache_offset..cache_offset + kv_width].to_vec());
                    live_values[layer_index] =
                        Some(cache_value[cache_offset..cache_offset + kv_width].to_vec());
                }
            }
        }
        for (layer_index, layer) in self
            .layers
            .iter()
            .enumerate()
            .skip(start_layer)
            .take(end_layer.saturating_sub(start_layer))
        {
            let hidden_norm = rms_norm(
                hidden.as_slice(),
                layer.attention_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let residual = std::mem::take(&mut hidden);

            let mut kv_kernel_count = 0usize;
            let mut kv_bytes_moved = 0u64;
            let (q, k_values, v_values) = if layer.attention_geometry.has_kv() {
                let (mut q, mut k, mut v) = MetalQuantizedProjectionMatrix::matvec_triplet(
                    &layer.attention_query_weight,
                    &layer.attention_key_weight,
                    &layer.attention_value_weight,
                    backend,
                    &hidden_norm,
                )?;
                if let Some(bias) = layer.attention_query_bias.as_ref() {
                    add_bias_in_place(&mut q.values, bias.as_slice());
                }
                if let Some(norm) = layer.attention_query_norm.as_ref() {
                    per_head_rms_norm_in_place(
                        q.values.as_mut_slice(),
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.head_dim,
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }
                apply_rope_neox(
                    &mut q.values,
                    layer.attention_geometry.head_count,
                    layer.attention_geometry.head_dim,
                    layer.attention_geometry.rotary_dim,
                    position,
                    layer.attention_geometry.rope_theta,
                    self.rope_freq_factors_for_layer(
                        layer_index,
                        layer.attention_geometry.head_dim,
                    ),
                    &self.family_metadata,
                );
                if let Some(bias) = layer.attention_key_bias.as_ref() {
                    add_bias_in_place(&mut k.values, bias.as_slice());
                }
                if let Some(norm) = layer.attention_key_norm.as_ref() {
                    per_head_rms_norm_in_place(
                        k.values.as_mut_slice(),
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }

                if let Some(bias) = layer.attention_value_bias.as_ref() {
                    add_bias_in_place(&mut v.values, bias.as_slice());
                }
                per_head_rms_norm_unit_in_place(
                    v.values.as_mut_slice(),
                    layer.attention_geometry.kv_head_count,
                    layer.attention_geometry.head_dim,
                    self.family_metadata.rms_norm_epsilon,
                );
                apply_rope_neox(
                    &mut k.values,
                    layer.attention_geometry.kv_head_count,
                    layer.attention_geometry.head_dim,
                    layer.attention_geometry.rotary_dim,
                    position,
                    layer.attention_geometry.rope_theta,
                    self.rope_freq_factors_for_layer(
                        layer_index,
                        layer.attention_geometry.head_dim,
                    ),
                    &self.family_metadata,
                );
                if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                    let kv_width = layer.attention_geometry.kv_width();
                    cache_key[cache_offset..cache_offset + kv_width]
                        .copy_from_slice(k.values.as_slice());
                    cache_value[cache_offset..cache_offset + kv_width]
                        .copy_from_slice(v.values.as_slice());
                }
                kv_kernel_count = kv_kernel_count
                    .saturating_add(k.kernel_count)
                    .saturating_add(v.kernel_count);
                kv_bytes_moved = kv_bytes_moved
                    .saturating_add(k.bytes_moved)
                    .saturating_add(v.bytes_moved);
                live_keys[layer_index] = Some(k.values);
                live_values[layer_index] = Some(v.values);
                (
                    q,
                    live_keys[layer_index].as_deref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} lost live key cache after publication"
                            ),
                        ))
                    })?,
                    live_values[layer_index].as_deref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} lost live value cache after publication"
                            ),
                        ))
                    })?,
                )
            } else {
                let mut q = layer.attention_query_weight.matvec(backend, &hidden_norm)?;
                if let Some(bias) = layer.attention_query_bias.as_ref() {
                    add_bias_in_place(&mut q.values, bias.as_slice());
                }
                if let Some(norm) = layer.attention_query_norm.as_ref() {
                    per_head_rms_norm_in_place(
                        q.values.as_mut_slice(),
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.head_dim,
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }
                apply_rope_neox(
                    &mut q.values,
                    layer.attention_geometry.head_count,
                    layer.attention_geometry.head_dim,
                    layer.attention_geometry.rotary_dim,
                    position,
                    layer.attention_geometry.rope_theta,
                    self.rope_freq_factors_for_layer(
                        layer_index,
                        layer.attention_geometry.head_dim,
                    ),
                    &self.family_metadata,
                );
                let reuse_layer_index =
                    layer.attention_geometry.reuse_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} is missing reused kv source metadata"
                            ),
                        ))
                    })?;
                let reused_k = live_keys
                    .get(reuse_layer_index)
                    .and_then(Option::as_deref)
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} expected live key cache from layer {reuse_layer_index}"
                            ),
                        ))
                    })?;
                let reused_v = live_values
                    .get(reuse_layer_index)
                    .and_then(Option::as_deref)
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} expected live value cache from layer {reuse_layer_index}"
                            ),
                        ))
                    })?;
                (q, reused_k, reused_v)
            };

            let attention = attend_impl(
                layer_index,
                q.values.as_slice(),
                k_values,
                v_values,
                cache,
                layer.attention_geometry.head_count,
                layer.attention_geometry.kv_head_count,
                layer.attention_geometry.head_dim,
                layer.attention_geometry.cache_read_offset,
                layer.attention_geometry.sliding_window,
                attention_scale(&self.family_metadata, layer.attention_geometry.head_dim),
            );
            let mut attention_out = layer
                .attention_output_weight
                .matvec(backend, attention.as_slice())?;
            if let Some(bias) = layer.attention_output_bias.as_ref() {
                add_bias_in_place(&mut attention_out.values, bias.as_slice());
            }
            if let Some(norm) = layer.attention_post_norm.as_ref() {
                rms_norm_in_place(
                    attention_out.values.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(attention_out.values.as_mut_slice(), residual.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            hidden = attention_out.values;

            let ffn_input = rms_norm(
                hidden.as_slice(),
                layer.feed_forward_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let ffn_residual = std::mem::take(&mut hidden);
            let mut ffn_out = if self.family_metadata.family == GgufDecoderFamily::Gemma4
                && layer.feed_forward_router_weight.is_some()
                && layer.feed_forward_gate_experts_weight.is_some()
                && layer.feed_forward_down_experts_weight.is_some()
            {
                let mut shared_ffn =
                    MetalQuantizedProjectionMatrix::matvec_pair_gelu_glu_projected(
                        layer.feed_forward_gate_weight.as_ref().ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_gate_weight"),
                            ))
                        })?,
                        layer.feed_forward_up_weight.as_ref().ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_up_weight"),
                            ))
                        })?,
                        layer.feed_forward_down_weight.as_ref().ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_down_weight"),
                            ))
                        })?,
                        backend,
                        &ffn_input,
                    )?;
                if let Some(norm) = layer.feed_forward_post_norm_1.as_ref() {
                    rms_norm_in_place(
                        shared_ffn.values.as_mut_slice(),
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }
                let mut router_input = rms_norm_unit(
                    ffn_residual.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
                scale_in_place(
                    router_input.as_mut_slice(),
                    (self.descriptor.config.hidden_size as f32).sqrt().recip(),
                );
                if let Some(scale) = layer.feed_forward_router_input_scale.as_ref() {
                    multiply_vectors_in_place(router_input.as_mut_slice(), scale.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                let expert_input = if let Some(norm) = layer.feed_forward_pre_norm_2.as_ref() {
                    rms_norm(
                        ffn_residual.as_slice(),
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    )
                } else {
                    ffn_input.clone()
                };
                let mut moe_ffn = evaluate_sparse_moe_host(
                    &self.family_metadata,
                    router_input.as_slice(),
                    expert_input.as_slice(),
                    layer
                        .feed_forward_router_weight
                        .as_ref()
                        .expect("router checked"),
                    layer.feed_forward_router_bias.as_deref(),
                    layer
                        .feed_forward_gate_experts_weight
                        .as_ref()
                        .expect("gate experts checked"),
                    layer.feed_forward_gate_experts_bias.as_deref(),
                    layer.feed_forward_up_experts_weight.as_ref(),
                    layer.feed_forward_up_experts_bias.as_deref(),
                    layer
                        .feed_forward_down_experts_weight
                        .as_ref()
                        .expect("down experts checked"),
                    layer.feed_forward_down_experts_bias.as_deref(),
                    layer.feed_forward_down_experts_scale.as_deref(),
                    self.family_metadata.expert_used_count.unwrap_or(1),
                )?;
                if let Some(norm) = layer.feed_forward_post_norm_2.as_ref() {
                    rms_norm_in_place(
                        moe_ffn.values.as_mut_slice(),
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }
                MetalProjectionStep {
                    values: add_vectors(shared_ffn.values.as_slice(), moe_ffn.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?,
                    kernel_count: shared_ffn.kernel_count + moe_ffn.kernel_count,
                    bytes_moved: shared_ffn.bytes_moved + moe_ffn.bytes_moved,
                }
            } else if let (Some(router), Some(gate_experts), Some(up_experts), Some(down_experts)) = (
                layer.feed_forward_router_weight.as_ref(),
                layer.feed_forward_gate_experts_weight.as_ref(),
                layer.feed_forward_up_experts_weight.as_ref(),
                layer.feed_forward_down_experts_weight.as_ref(),
            ) {
                let moe_ffn = evaluate_sparse_moe_host(
                    &self.family_metadata,
                    ffn_input.as_slice(),
                    ffn_input.as_slice(),
                    router,
                    layer.feed_forward_router_bias.as_deref(),
                    gate_experts,
                    layer.feed_forward_gate_experts_bias.as_deref(),
                    Some(up_experts),
                    layer.feed_forward_up_experts_bias.as_deref(),
                    down_experts,
                    layer.feed_forward_down_experts_bias.as_deref(),
                    layer.feed_forward_down_experts_scale.as_deref(),
                    self.family_metadata.expert_used_count.unwrap_or(1),
                )?;
                MetalProjectionStep {
                    values: moe_ffn.values,
                    kernel_count: moe_ffn.kernel_count,
                    bytes_moved: moe_ffn.bytes_moved,
                }
            } else if self.family_metadata.family == GgufDecoderFamily::Gemma4 {
                MetalQuantizedProjectionMatrix::matvec_pair_gelu_glu_projected(
                    layer.feed_forward_gate_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing gemma4 dense feed_forward_gate_weight"),
                        ))
                    })?,
                    layer.feed_forward_up_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing gemma4 dense feed_forward_up_weight"),
                        ))
                    })?,
                    layer.feed_forward_down_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing gemma4 dense feed_forward_down_weight"),
                        ))
                    })?,
                    backend,
                    &ffn_input,
                )?
            } else {
                let (mut gate, up) = MetalQuantizedProjectionMatrix::matvec_pair(
                    layer.feed_forward_gate_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing dense feed_forward_gate_weight"),
                        ))
                    })?,
                    layer.feed_forward_up_weight.as_ref().ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing dense feed_forward_up_weight"),
                        ))
                    })?,
                    backend,
                    &ffn_input,
                )?;
                feed_forward_activation_in_place(
                    &self.family_metadata,
                    gate.values.as_mut_slice(),
                    up.values.as_slice(),
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;
                layer
                    .feed_forward_down_weight
                    .as_ref()
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            String::from("missing dense feed_forward_down_weight"),
                        ))
                    })?
                    .matvec(backend, gate.values.as_slice())?
            };
            if let Some(norm) = layer.feed_forward_post_norm.as_ref() {
                rms_norm_in_place(
                    ffn_out.values.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(ffn_out.values.as_mut_slice(), ffn_residual.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            hidden = ffn_out.values;

            if let Some(per_layer_inputs) = gemma4_per_layer_inputs.as_ref() {
                if let (Some(input_gate), Some(proj), Some(post_norm)) = (
                    layer.per_layer_input_gate.as_ref(),
                    layer.per_layer_proj.as_ref(),
                    layer.per_layer_post_norm.as_ref(),
                ) {
                    let multiplier = self
                        .gemma4_per_layer_inputs
                        .as_ref()
                        .expect("per-layer config")
                        .layer_slice(per_layer_inputs.values.as_slice(), layer_index);
                    let mut projected = if input_gate.is_native() && proj.is_native() {
                        MetalProjectionStep {
                            values: backend
                                .quantized_gelu_mul_projected(
                                    input_gate.native_weights()?,
                                    input_gate.mode,
                                    input_gate.rows,
                                    input_gate.columns,
                                    multiplier,
                                    proj.native_weights()?,
                                    proj.mode,
                                    proj.rows,
                                    proj.columns,
                                    hidden.as_slice(),
                                )
                                .map_err(ReferenceTextGenerationError::Runtime)?,
                            kernel_count: 3,
                            bytes_moved: (input_gate.byte_length() + proj.byte_length()) as u64,
                        }
                    } else {
                        let mut gate = input_gate.matvec(backend, hidden.as_slice())?;
                        if gate.values.len() != multiplier.len() {
                            return Err(ReferenceTextGenerationError::Runtime(
                                crate::RuntimeError::Backend(format!(
                                    "gemma4 per-layer multiplier width mismatch: gate={} multiplier={}",
                                    gate.values.len(),
                                    multiplier.len()
                                )),
                            ));
                        }
                        for (gate_value, multiplier_value) in
                            gate.values.iter_mut().zip(multiplier.iter().copied())
                        {
                            *gate_value = approximate_gelu(*gate_value) * multiplier_value;
                        }
                        let mut projected = proj.matvec(backend, gate.values.as_slice())?;
                        projected.kernel_count =
                            projected.kernel_count.saturating_add(gate.kernel_count);
                        projected.bytes_moved =
                            projected.bytes_moved.saturating_add(gate.bytes_moved);
                        projected
                    };
                    rms_norm_in_place(
                        projected.values.as_mut_slice(),
                        post_norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                    add_vectors_in_place(hidden.as_mut_slice(), projected.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    bytes_moved = bytes_moved.saturating_add(projected.bytes_moved);
                    kernel_count = kernel_count.saturating_add(projected.kernel_count);
                }
            }
            if let Some(scale) = layer.layer_output_scale {
                scale_in_place(&mut hidden, scale);
            }

            bytes_moved = bytes_moved
                .saturating_add(q.bytes_moved)
                .saturating_add(attention_out.bytes_moved)
                .saturating_add(ffn_out.bytes_moved);
            kernel_count = kernel_count
                .saturating_add(q.kernel_count)
                .saturating_add(attention_out.kernel_count)
                .saturating_add(ffn_out.kernel_count);
            if layer.attention_geometry.has_kv() {
                bytes_moved = bytes_moved.saturating_add(kv_bytes_moved);
                kernel_count = kernel_count.saturating_add(kv_kernel_count);
            }
        }

        let (logits, selected_token, lm_head_hidden) = if produce_logits {
            let final_hidden = rms_norm(
                hidden.as_slice(),
                self.output_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            match logits_output_mode {
                MetalLogitsOutputMode::GreedyToken => {
                    let selection = self.output.select_logits_output(
                        backend,
                        final_hidden.as_slice(),
                        logits_output_mode,
                    )?;
                    bytes_moved = bytes_moved.saturating_add(self.output.byte_length() as u64);
                    kernel_count = kernel_count.saturating_add(1);
                    let token = selection.selected_tokens.first().copied().ok_or_else(|| {
                        ReferenceTextGenerationError::MissingOutput("metal gemma4 greedy token")
                    })?;
                    (Vec::new(), Some(TokenId(token)), Some(final_hidden))
                }
                MetalLogitsOutputMode::RawLogits => {
                    let mut logits = self.output.matvec(backend, final_hidden.as_slice())?;
                    apply_final_logit_softcapping_in_place(
                        logits.values.as_mut_slice(),
                        self.family_metadata.final_logit_softcapping,
                    );
                    bytes_moved = bytes_moved.saturating_add(logits.bytes_moved);
                    kernel_count = kernel_count.saturating_add(logits.kernel_count);
                    (logits.values, None, Some(final_hidden))
                }
                MetalLogitsOutputMode::TopKCandidates(_) => {
                    return Err(ReferenceTextGenerationError::Runtime(
                        crate::RuntimeError::Backend(String::from(
                            "metal gemma4 top-k logits output mode is not yet wired into the generic generation loop",
                        )),
                    ));
                }
            }
        } else {
            (Vec::new(), None, None)
        };

        Ok(MetalGemma4StageStep {
            key: cache_key,
            value: cache_value,
            logits,
            selected_token,
            hidden,
            lm_head_hidden,
            kernel_count,
            bytes_moved,
        })
    }
}

#[derive(Clone, Debug)]
struct MetalGemma4Layer {
    attention_geometry: DenseAttentionGeometry,
    attention_norm: MetalStaticVector,
    attention_query_weight: MetalQuantizedProjectionMatrix,
    attention_qkv_weight: Option<MetalQuantizedProjectionMatrix>,
    attention_query_bias: Option<MetalStaticVector>,
    attention_query_norm: Option<MetalStaticVector>,
    attention_key_weight: MetalQuantizedProjectionMatrix,
    attention_key_bias: Option<MetalStaticVector>,
    attention_key_norm: Option<MetalStaticVector>,
    attention_value_weight: MetalQuantizedProjectionMatrix,
    attention_value_bias: Option<MetalStaticVector>,
    attention_output_weight: MetalQuantizedProjectionMatrix,
    attention_output_bias: Option<MetalStaticVector>,
    attention_post_norm: Option<MetalStaticVector>,
    feed_forward_norm: MetalStaticVector,
    feed_forward_gate_weight: Option<MetalQuantizedProjectionMatrix>,
    feed_forward_gate_up_weight: Option<MetalQuantizedProjectionMatrix>,
    feed_forward_up_weight: Option<MetalQuantizedProjectionMatrix>,
    feed_forward_down_weight: Option<MetalQuantizedProjectionMatrix>,
    feed_forward_router_weight: Option<DenseMatrix>,
    feed_forward_router_bias: Option<Vec<f32>>,
    feed_forward_router_input_scale: Option<Vec<f32>>,
    feed_forward_gate_experts_weight: Option<QuantizedExpertTensor>,
    feed_forward_gate_experts_bias: Option<Vec<f32>>,
    feed_forward_up_experts_weight: Option<QuantizedExpertTensor>,
    feed_forward_up_experts_bias: Option<Vec<f32>>,
    feed_forward_down_experts_weight: Option<QuantizedExpertTensor>,
    feed_forward_down_experts_bias: Option<Vec<f32>>,
    feed_forward_down_experts_scale: Option<Vec<f32>>,
    feed_forward_post_norm_1: Option<Vec<f32>>,
    feed_forward_pre_norm_2: Option<Vec<f32>>,
    feed_forward_post_norm_2: Option<Vec<f32>>,
    feed_forward_post_norm: Option<MetalStaticVector>,
    layer_output_scale: Option<f32>,
    per_layer_input_gate: Option<MetalQuantizedProjectionMatrix>,
    per_layer_proj: Option<MetalQuantizedProjectionMatrix>,
    per_layer_post_norm: Option<MetalStaticVector>,
}

#[derive(Clone, Debug)]
struct MetalStaticVector {
    host: Vec<f32>,
    device: MetalBuffer,
}

impl MetalStaticVector {
    fn from_values(
        backend: &mut MetalBackend,
        values: Vec<f32>,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let device = backend
            .input_buffer(Shape::new(vec![values.len()]), values.clone())
            .map_err(ReferenceTextGenerationError::Runtime)?;
        Ok(Self {
            host: values,
            device,
        })
    }

    fn as_slice(&self) -> &[f32] {
        self.host.as_slice()
    }

    fn len(&self) -> usize {
        self.host.len()
    }

    fn device(&self) -> &MetalBuffer {
        &self.device
    }
}

fn load_metal_static_vector(
    backend: &mut MetalBackend,
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<MetalStaticVector, ReferenceTextGenerationError> {
    MetalStaticVector::from_values(backend, load_dense_vector(artifact, name)?)
}

fn load_optional_metal_static_vector(
    backend: &mut MetalBackend,
    artifact: &GgufBlobArtifact,
    name: Option<&str>,
) -> Result<Option<MetalStaticVector>, ReferenceTextGenerationError> {
    let Some(values) = load_optional_dense_vector(artifact, name)? else {
        return Ok(None);
    };
    Ok(Some(MetalStaticVector::from_values(backend, values)?))
}

impl MetalGemma4Layer {
    fn load(
        backend: &mut MetalBackend,
        artifact: &GgufBlobArtifact,
        layout: &GgufDecoderLayerTensorLayout,
        descriptor: &DecoderModelDescriptor,
        family_metadata: &GgufDecoderFamilyMetadata,
        layer_index: usize,
        cache_read_offset: usize,
        cache_write_offset: Option<usize>,
        reuse_layer_index: Option<usize>,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let has_kv = cache_write_offset.is_some();
        let attention_query_weight = MetalQuantizedProjectionMatrix::load(
            backend,
            artifact,
            required_tensor_name(
                layout.attention_query_weight.as_deref(),
                "attention_query_weight",
            )?,
        )?;
        let query_width = attention_query_weight.rows();
        let head_dim = query_width
            .checked_div(descriptor.config.block.attention.head_count)
            .unwrap_or(0);
        let attention_key_weight = if let Some(name) = layout.attention_key_weight.as_deref() {
            MetalQuantizedProjectionMatrix::load(backend, artifact, name)?
        } else if family_metadata.family == GgufDecoderFamily::Gemma4 && !has_kv {
            attention_query_weight.clone()
        } else {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(String::from("missing attention_key_weight")),
            ));
        };
        let key_width = if layout.attention_key_weight.is_some() {
            attention_key_weight.rows()
        } else {
            gemma4_expected_kv_width(descriptor, family_metadata, layer_index, head_dim)
        };
        let attention_value_weight = if let Some(name) = layout.attention_value_weight.as_deref() {
            MetalQuantizedProjectionMatrix::load(backend, artifact, name)?
        } else if family_metadata.family == GgufDecoderFamily::Gemma4 {
            attention_key_weight.clone()
        } else if has_kv {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(String::from("missing attention_value_weight")),
            ));
        } else {
            attention_key_weight.clone()
        };
        let value_width =
            if layout.attention_value_weight.is_some() || layout.attention_key_weight.is_some() {
                attention_value_weight.rows()
            } else {
                key_width
            };
        let attention_qkv_weight =
            if let Some(value_weight_name) = layout.attention_value_weight.as_deref() {
                MetalQuantizedProjectionMatrix::load_fused(
                    backend,
                    artifact,
                    &[
                        required_tensor_name(
                            layout.attention_query_weight.as_deref(),
                            "attention_query_weight",
                        )?,
                        required_tensor_name(
                            layout.attention_key_weight.as_deref(),
                            "attention_key_weight",
                        )?,
                        value_weight_name,
                    ],
                )?
            } else {
                None
            };
        let attention_geometry = dense_attention_geometry(
            descriptor,
            family_metadata,
            layer_index,
            query_width,
            key_width,
            value_width,
            cache_read_offset,
            cache_write_offset,
            reuse_layer_index,
        )?;
        Ok(Self {
            attention_geometry,
            attention_norm: load_metal_static_vector(
                backend,
                artifact,
                layout.attention_norm.as_str(),
            )?,
            attention_query_weight,
            attention_qkv_weight,
            attention_query_bias: load_optional_metal_static_vector(
                backend,
                artifact,
                layout.attention_query_bias.as_deref(),
            )?,
            attention_query_norm: load_optional_metal_static_vector(
                backend,
                artifact,
                layout.attention_query_norm.as_deref(),
            )?,
            attention_key_weight,
            attention_key_bias: load_optional_metal_static_vector(
                backend,
                artifact,
                layout.attention_key_bias.as_deref(),
            )?,
            attention_key_norm: load_optional_metal_static_vector(
                backend,
                artifact,
                layout.attention_key_norm.as_deref(),
            )?,
            attention_value_weight,
            attention_value_bias: load_optional_metal_static_vector(
                backend,
                artifact,
                layout.attention_value_bias.as_deref(),
            )?,
            attention_output_weight: MetalQuantizedProjectionMatrix::load(
                backend,
                artifact,
                required_tensor_name(
                    layout.attention_output_weight.as_deref(),
                    "attention_output_weight",
                )?,
            )?,
            attention_output_bias: load_optional_metal_static_vector(
                backend,
                artifact,
                layout.attention_output_bias.as_deref(),
            )?,
            attention_post_norm: load_optional_metal_static_vector(
                backend,
                artifact,
                layout.attention_post_norm.as_deref(),
            )?,
            feed_forward_norm: load_metal_static_vector(
                backend,
                artifact,
                required_tensor_name(layout.feed_forward_norm.as_deref(), "feed_forward_norm")?,
            )?,
            feed_forward_gate_weight: layout
                .feed_forward_gate_weight
                .as_deref()
                .map(|name| MetalQuantizedProjectionMatrix::load(backend, artifact, name))
                .transpose()?,
            feed_forward_gate_up_weight: if let (Some(gate), Some(up)) = (
                layout.feed_forward_gate_weight.as_deref(),
                layout.feed_forward_up_weight.as_deref(),
            ) {
                MetalQuantizedProjectionMatrix::load_fused(backend, artifact, &[gate, up])?
            } else {
                None
            },
            feed_forward_up_weight: layout
                .feed_forward_up_weight
                .as_deref()
                .map(|name| MetalQuantizedProjectionMatrix::load(backend, artifact, name))
                .transpose()?,
            feed_forward_down_weight: layout
                .feed_forward_down_weight
                .as_deref()
                .map(|name| MetalQuantizedProjectionMatrix::load(backend, artifact, name))
                .transpose()?,
            feed_forward_router_weight: layout
                .feed_forward_router_weight
                .as_deref()
                .map(|name| load_dense_matrix(artifact, name))
                .transpose()?,
            feed_forward_router_bias: layout
                .feed_forward_router_bias
                .as_deref()
                .map(|name| load_dense_vector(artifact, name))
                .transpose()?,
            feed_forward_router_input_scale: load_first_named_optional_dense_vector(
                artifact,
                &[format!("blk.{layer_index}.ffn_gate_inp.scale")],
            )?,
            feed_forward_gate_experts_weight: layout
                .feed_forward_gate_experts_weight
                .as_deref()
                .map(|name| load_quantized_expert_tensor(artifact, name))
                .transpose()?,
            feed_forward_gate_experts_bias: layout
                .feed_forward_gate_experts_bias
                .as_deref()
                .map(|name| load_dense_rank2_flat(artifact, name))
                .transpose()?,
            feed_forward_up_experts_weight: layout
                .feed_forward_up_experts_weight
                .as_deref()
                .map(|name| load_quantized_expert_tensor(artifact, name))
                .transpose()?,
            feed_forward_up_experts_bias: layout
                .feed_forward_up_experts_bias
                .as_deref()
                .map(|name| load_dense_rank2_flat(artifact, name))
                .transpose()?,
            feed_forward_down_experts_weight: layout
                .feed_forward_down_experts_weight
                .as_deref()
                .map(|name| load_quantized_expert_tensor(artifact, name))
                .transpose()?,
            feed_forward_down_experts_bias: layout
                .feed_forward_down_experts_bias
                .as_deref()
                .map(|name| load_dense_rank2_flat(artifact, name))
                .transpose()?,
            feed_forward_down_experts_scale: load_first_named_optional_dense_vector(
                artifact,
                &[format!("blk.{layer_index}.ffn_down_exps.scale")],
            )?,
            feed_forward_post_norm_1: load_first_named_optional_dense_vector(
                artifact,
                &[
                    format!("blk.{layer_index}.post_ffw_norm_1.weight"),
                    format!("blk.{layer_index}.ffn_post_norm_1.weight"),
                ],
            )?,
            feed_forward_pre_norm_2: load_first_named_optional_dense_vector(
                artifact,
                &[
                    format!("blk.{layer_index}.pre_ffw_norm_2.weight"),
                    format!("blk.{layer_index}.ffn_pre_norm_2.weight"),
                ],
            )?,
            feed_forward_post_norm_2: load_first_named_optional_dense_vector(
                artifact,
                &[
                    format!("blk.{layer_index}.post_ffw_norm_2.weight"),
                    format!("blk.{layer_index}.ffn_post_norm_2.weight"),
                ],
            )?,
            feed_forward_post_norm: load_optional_metal_static_vector(
                backend,
                artifact,
                layout.feed_forward_post_norm.as_deref(),
            )?,
            layer_output_scale: load_optional_dense_scalar(
                artifact,
                layout.layer_output_scale.as_deref(),
            )?,
            per_layer_input_gate: load_named_optional_metal_quantized_projection_matrix(
                backend,
                artifact,
                format!("blk.{layer_index}.inp_gate.weight").as_str(),
            )?,
            per_layer_proj: load_named_optional_metal_quantized_projection_matrix(
                backend,
                artifact,
                format!("blk.{layer_index}.proj.weight").as_str(),
            )?,
            per_layer_post_norm: load_named_optional_metal_static_vector(
                backend,
                artifact,
                format!("blk.{layer_index}.post_norm.weight").as_str(),
            )?,
        })
    }

    fn cache_write_width(&self) -> usize {
        self.attention_geometry.cache_write_width()
    }
}

#[derive(Clone, Debug)]
struct MetalGemma4PerLayerInputs {
    token_embedding: ProjectionMatrix,
    model_proj: MetalGemma4PerLayerProjectionMatrix,
    proj_norm: MetalStaticVector,
    width: usize,
}

impl MetalGemma4PerLayerInputs {
    fn load(
        backend: &mut MetalBackend,
        artifact: &GgufBlobArtifact,
    ) -> Result<Option<Self>, ReferenceTextGenerationError> {
        let token_embedding =
            load_named_optional_projection_matrix(artifact, "per_layer_token_embd.weight")?;
        let model_proj = load_named_optional_metal_gemma4_projection_matrix(
            backend,
            artifact,
            "per_layer_model_proj.weight",
        )?;
        let proj_norm = load_named_optional_metal_static_vector(
            backend,
            artifact,
            "per_layer_proj_norm.weight",
        )?;
        let (Some(token_embedding), Some(model_proj), Some(proj_norm)) =
            (token_embedding, model_proj, proj_norm)
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            token_embedding,
            model_proj,
            width: proj_norm.len(),
            proj_norm,
        }))
    }

    fn project(
        &self,
        backend: &mut MetalBackend,
        token: TokenId,
        hidden: &[f32],
        layer_count: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<MetalProjectionStep, ReferenceTextGenerationError> {
        let mut token_inputs = self
            .token_embedding
            .decode_row(token.as_u32() as usize)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let expected_len = self.width.saturating_mul(layer_count);
        if token_inputs.len() != expected_len {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 per-layer token embedding width mismatch: expected {expected_len}, actual {}",
                    token_inputs.len()
                )),
            ));
        }
        scale_in_place(&mut token_inputs, (self.width as f32).sqrt());

        let mut projected = self.model_proj.matvec(backend, hidden)?;
        if projected.values.len() != expected_len {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 per-layer model projection width mismatch: expected {expected_len}, actual {}",
                    projected.values.len()
                )),
            ));
        }
        scale_in_place(&mut projected.values, (hidden_size as f32).sqrt().recip());

        let layer_mix_scale = 2.0_f32.sqrt().recip();
        let mut combined = vec![0.0_f32; expected_len];
        for layer_index in 0..layer_count {
            let start = layer_index.saturating_mul(self.width);
            let end = start.saturating_add(self.width);
            let normalized = rms_norm(
                &projected.values[start..end],
                self.proj_norm.as_slice(),
                epsilon,
            );
            for index in 0..self.width {
                combined[start + index] =
                    (normalized[index] + token_inputs[start + index]) * layer_mix_scale;
            }
        }
        projected.values = combined;
        Ok(projected)
    }

    fn layer_slice<'a>(&self, values: &'a [f32], layer_index: usize) -> &'a [f32] {
        let start = layer_index.saturating_mul(self.width);
        let end = start.saturating_add(self.width);
        &values[start..end]
    }
}

#[derive(Clone, Debug)]
struct MetalQuantizedProjectionMatrix {
    mode: QuantizationMode,
    rows: usize,
    columns: usize,
    native_weights: Option<MetalBuffer>,
    host_projection: Option<ProjectionMatrix>,
}

impl MetalQuantizedProjectionMatrix {
    fn is_native(&self) -> bool {
        self.native_weights.is_some()
    }

    fn native_weights(&self) -> Result<&MetalBuffer, ReferenceTextGenerationError> {
        self.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                "metal-native projection is unavailable for quantized mode {:?}",
                self.mode
            )))
        })
    }

    fn load(
        backend: &mut MetalBackend,
        artifact: &GgufBlobArtifact,
        name: &str,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let storage = artifact.paged_tensor(name)?;
        let metadata = storage.metadata();
        let Some(layout) = metadata.quantized_layout else {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "native gemma4 metal runtime currently requires quantized projection tensor `{name}`",
                )),
            ));
        };
        let [rows, columns] = metadata.shape.dims() else {
            return Err(ModelLoadError::InvalidTensorShape {
                name: metadata.name.clone(),
                expected: vec![0, 0],
                actual: metadata.shape.dims().to_vec(),
            }
            .into());
        };
        let _ = quantized_row_byte_len(&metadata.shape, layout).map_err(|_| {
            ModelLoadError::InvalidQuantizedTensorShape {
                quantization: metadata.quantization,
                shape: metadata.shape.dims().to_vec(),
            }
        })?;
        let (native_weights, host_projection) =
            if supports_native_metal_quantized_projection(metadata.quantization) {
                let keepalive: Arc<PagedTensorStorage> = Arc::new(storage.clone());
                let bytes_owner = Arc::clone(&keepalive);
                let bytes = bytes_owner.bytes()?;
                let keepalive: Arc<dyn std::any::Any> = keepalive;
                let weights = backend
                    .quantized_buffer_from_slice(
                        metadata.shape.clone(),
                        metadata.quantization,
                        bytes,
                        Some(keepalive),
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                (Some(weights), None)
            } else {
                (None, Some(ProjectionMatrix::load(artifact, name)?))
            };
        Ok(Self {
            mode: metadata.quantization,
            rows: *rows,
            columns: *columns,
            native_weights,
            host_projection,
        })
    }

    fn load_fused(
        backend: &mut MetalBackend,
        artifact: &GgufBlobArtifact,
        names: &[&str],
    ) -> Result<Option<Self>, ReferenceTextGenerationError> {
        if names.is_empty() {
            return Ok(None);
        }

        let mut combined = Vec::new();
        let mut total_rows = 0usize;
        let mut expected_mode = None;
        let mut expected_columns = None;

        for name in names {
            let storage = artifact.paged_tensor(name)?;
            let metadata = storage.metadata();
            let Some(layout) = metadata.quantized_layout else {
                return Ok(None);
            };
            let [rows, columns] = metadata.shape.dims() else {
                return Err(ModelLoadError::InvalidTensorShape {
                    name: metadata.name.clone(),
                    expected: vec![0, 0],
                    actual: metadata.shape.dims().to_vec(),
                }
                .into());
            };
            let row_byte_len = quantized_row_byte_len(&metadata.shape, layout).map_err(|_| {
                ModelLoadError::InvalidQuantizedTensorShape {
                    quantization: metadata.quantization,
                    shape: metadata.shape.dims().to_vec(),
                }
            })?;
            if let Some(mode) = expected_mode {
                if mode != metadata.quantization || expected_columns != Some(*columns) {
                    return Ok(None);
                }
            } else {
                expected_mode = Some(metadata.quantization);
                expected_columns = Some(*columns);
            }
            if !supports_native_metal_quantized_projection(metadata.quantization) {
                return Ok(None);
            }
            let bytes = storage.bytes()?;
            let expected_bytes = row_byte_len.saturating_mul(*rows);
            if bytes.len() != expected_bytes {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "fused metal projection `{name}` byte length mismatch: expected {expected_bytes}, actual {}",
                        bytes.len()
                    )),
                ));
            }
            combined.extend_from_slice(bytes);
            total_rows = total_rows.saturating_add(*rows);
        }

        let Some(mode) = expected_mode else {
            return Ok(None);
        };
        let columns = expected_columns.unwrap_or(0);
        let weights = backend
            .quantized_buffer_from_slice(
                Shape::new(vec![total_rows, columns]),
                mode,
                combined.as_slice(),
                None,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
        Ok(Some(Self {
            mode,
            rows: total_rows,
            columns,
            native_weights: Some(weights),
            host_projection: None,
        }))
    }

    fn matvec(
        &self,
        backend: &mut MetalBackend,
        input: &[f32],
    ) -> Result<MetalProjectionStep, ReferenceTextGenerationError> {
        if let Some(matrix) = &self.host_projection {
            let mut values = Vec::new();
            matrix
                .matvec(input, &mut values)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            return Ok(MetalProjectionStep {
                values,
                kernel_count: 0,
                bytes_moved: matrix.byte_length() as u64,
            });
        }
        let weights = self.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native or host metal projection weights",
            )))
        })?;
        let values = backend
            .quantized_matvec(weights, self.mode, self.rows, self.columns, input)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        Ok(MetalProjectionStep {
            values,
            kernel_count: 1,
            bytes_moved: self.byte_length() as u64,
        })
    }

    fn matvec_pair(
        left: &Self,
        right: &Self,
        backend: &mut MetalBackend,
        input: &[f32],
    ) -> Result<(MetalProjectionStep, MetalProjectionStep), ReferenceTextGenerationError> {
        if left.host_projection.is_some() || right.host_projection.is_some() {
            return Ok((left.matvec(backend, input)?, right.matvec(backend, input)?));
        }
        let left_weights = left.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native left projection weights",
            )))
        })?;
        let right_weights = right.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native right projection weights",
            )))
        })?;
        let results = backend
            .quantized_matvec_batch(
                &[
                    MetalQuantizedMatvecRequest {
                        weights: left_weights,
                        byte_offset: 0,
                        mode: left.mode,
                        rows: left.rows,
                        columns: left.columns,
                    },
                    MetalQuantizedMatvecRequest {
                        weights: right_weights,
                        byte_offset: 0,
                        mode: right.mode,
                        rows: right.rows,
                        columns: right.columns,
                    },
                ],
                input,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let mut results = results.into_iter();
        let left_values = results.next().ok_or_else(|| {
            ReferenceTextGenerationError::MissingOutput("metal quantized matvec pair left")
        })?;
        let right_values = results.next().ok_or_else(|| {
            ReferenceTextGenerationError::MissingOutput("metal quantized matvec pair right")
        })?;
        Ok((
            MetalProjectionStep {
                values: left_values.values,
                kernel_count: 1,
                bytes_moved: left.byte_length() as u64,
            },
            MetalProjectionStep {
                values: right_values.values,
                kernel_count: 1,
                bytes_moved: right.byte_length() as u64,
            },
        ))
    }

    fn matvec_pair_gelu_glu_projected(
        gate: &Self,
        up: &Self,
        down: &Self,
        backend: &mut MetalBackend,
        input: &[f32],
    ) -> Result<MetalProjectionStep, ReferenceTextGenerationError> {
        if gate.host_projection.is_some()
            || up.host_projection.is_some()
            || down.host_projection.is_some()
        {
            let mut gate_step = gate.matvec(backend, input)?;
            let up_step = up.matvec(backend, input)?;
            if gate_step.values.len() != up_step.values.len() {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "metal host fallback gate/up width mismatch: gate={} up={}",
                        gate_step.values.len(),
                        up_step.values.len()
                    )),
                ));
            }
            for (gate_value, up_value) in gate_step.values.iter_mut().zip(up_step.values.iter()) {
                *gate_value = approximate_gelu(*gate_value) * *up_value;
            }
            let mut projected = down.matvec(backend, gate_step.values.as_slice())?;
            projected.kernel_count = projected
                .kernel_count
                .saturating_add(gate_step.kernel_count)
                .saturating_add(up_step.kernel_count);
            projected.bytes_moved = projected
                .bytes_moved
                .saturating_add(gate_step.bytes_moved)
                .saturating_add(up_step.bytes_moved);
            return Ok(projected);
        }
        let gate_weights = gate.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native gate projection weights",
            )))
        })?;
        let up_weights = up.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native up projection weights",
            )))
        })?;
        let down_weights = down.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native down projection weights",
            )))
        })?;
        let values = backend
            .quantized_gelu_glu_projected(
                gate_weights,
                gate.mode,
                gate.rows,
                gate.columns,
                up_weights,
                up.mode,
                up.rows,
                up.columns,
                down_weights,
                down.mode,
                down.rows,
                down.columns,
                input,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
        Ok(MetalProjectionStep {
            values,
            kernel_count: 4,
            bytes_moved: (gate.byte_length() + up.byte_length() + down.byte_length()) as u64,
        })
    }

    fn matvec_triplet(
        first: &Self,
        second: &Self,
        third: &Self,
        backend: &mut MetalBackend,
        input: &[f32],
    ) -> Result<
        (
            MetalProjectionStep,
            MetalProjectionStep,
            MetalProjectionStep,
        ),
        ReferenceTextGenerationError,
    > {
        if first.host_projection.is_some()
            || second.host_projection.is_some()
            || third.host_projection.is_some()
        {
            return Ok((
                first.matvec(backend, input)?,
                second.matvec(backend, input)?,
                third.matvec(backend, input)?,
            ));
        }
        let first_weights = first.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native first projection weights",
            )))
        })?;
        let second_weights = second.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native second projection weights",
            )))
        })?;
        let third_weights = third.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native third projection weights",
            )))
        })?;
        let results = backend
            .quantized_matvec_batch(
                &[
                    MetalQuantizedMatvecRequest {
                        weights: first_weights,
                        byte_offset: 0,
                        mode: first.mode,
                        rows: first.rows,
                        columns: first.columns,
                    },
                    MetalQuantizedMatvecRequest {
                        weights: second_weights,
                        byte_offset: 0,
                        mode: second.mode,
                        rows: second.rows,
                        columns: second.columns,
                    },
                    MetalQuantizedMatvecRequest {
                        weights: third_weights,
                        byte_offset: 0,
                        mode: third.mode,
                        rows: third.rows,
                        columns: third.columns,
                    },
                ],
                input,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let mut results = results.into_iter();
        let first_values = results.next().ok_or_else(|| {
            ReferenceTextGenerationError::MissingOutput("metal quantized matvec triplet first")
        })?;
        let second_values = results.next().ok_or_else(|| {
            ReferenceTextGenerationError::MissingOutput("metal quantized matvec triplet second")
        })?;
        let third_values = results.next().ok_or_else(|| {
            ReferenceTextGenerationError::MissingOutput("metal quantized matvec triplet third")
        })?;
        Ok((
            MetalProjectionStep {
                values: first_values.values,
                kernel_count: 1,
                bytes_moved: first.byte_length() as u64,
            },
            MetalProjectionStep {
                values: second_values.values,
                kernel_count: 1,
                bytes_moved: second.byte_length() as u64,
            },
            MetalProjectionStep {
                values: third_values.values,
                kernel_count: 1,
                bytes_moved: third.byte_length() as u64,
            },
        ))
    }

    fn select_logits_output(
        &self,
        backend: &mut MetalBackend,
        input: &[f32],
        output_mode: MetalLogitsOutputMode,
    ) -> Result<psionic_backend_metal::MetalLogitsSelectionResult, ReferenceTextGenerationError>
    {
        if let Some(matrix) = &self.host_projection {
            return host_select_logits_output(matrix, input, output_mode);
        }
        let weights = self.native_weights.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "missing native logits projection weights",
            )))
        })?;
        backend
            .quantized_matvec_select_logits_output(
                weights,
                0,
                self.mode,
                self.rows,
                self.columns,
                input,
                output_mode,
            )
            .map_err(ReferenceTextGenerationError::Runtime)
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn byte_length(&self) -> usize {
        if let Some(matrix) = &self.host_projection {
            matrix.byte_length()
        } else {
            self.native_weights
                .as_ref()
                .map(MetalBuffer::byte_len)
                .unwrap_or(0)
        }
    }
}

fn supports_native_metal_quantized_projection(mode: QuantizationMode) -> bool {
    matches!(
        mode,
        QuantizationMode::GgmlQ4K
            | QuantizationMode::GgmlQ6K
            | QuantizationMode::GgmlQ8_0
            | QuantizationMode::GgmlMxfp4
    )
}

fn host_select_logits_output(
    matrix: &ProjectionMatrix,
    input: &[f32],
    output_mode: MetalLogitsOutputMode,
) -> Result<psionic_backend_metal::MetalLogitsSelectionResult, ReferenceTextGenerationError> {
    let mut logits = Vec::new();
    matrix
        .matvec(input, &mut logits)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    if logits.is_empty() {
        return Err(ReferenceTextGenerationError::MissingOutput(
            "host logits selection requires non-empty logits",
        ));
    }
    let greedy_index = logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.partial_cmp(right.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| u32::try_from(index).unwrap_or(u32::MAX))
        .ok_or(ReferenceTextGenerationError::MissingOutput(
            "host logits selection missing greedy token",
        ))?;
    match output_mode {
        MetalLogitsOutputMode::GreedyToken => {
            Ok(psionic_backend_metal::MetalLogitsSelectionResult {
                selected_tokens: vec![greedy_index],
                candidates: None,
                logits: None,
                metrics: psionic_backend_metal::MetalLogitsSelectionMetrics {
                    output_mode,
                    readback_bytes: std::mem::size_of::<u32>() as u64,
                    raw_logits_materialized: false,
                },
            })
        }
        MetalLogitsOutputMode::TopKCandidates(top_k) => {
            let keep = top_k.max(1).min(logits.len());
            let mut indexed = logits
                .iter()
                .copied()
                .enumerate()
                .map(|(index, value)| (u32::try_from(index).unwrap_or(u32::MAX), value))
                .collect::<Vec<_>>();
            indexed.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
            indexed.truncate(keep);
            let indices = indexed.iter().map(|(index, _)| *index).collect::<Vec<_>>();
            let values = indexed.iter().map(|(_, value)| *value).collect::<Vec<_>>();
            Ok(psionic_backend_metal::MetalLogitsSelectionResult {
                selected_tokens: vec![indices[0]],
                candidates: Some(psionic_backend_metal::MetalTopKResult {
                    row_count: 1,
                    top_k: keep,
                    indices,
                    values,
                }),
                logits: None,
                metrics: psionic_backend_metal::MetalLogitsSelectionMetrics {
                    output_mode,
                    readback_bytes: (keep
                        .saturating_mul(std::mem::size_of::<u32>())
                        .saturating_add(keep.saturating_mul(std::mem::size_of::<f32>())))
                        as u64,
                    raw_logits_materialized: false,
                },
            })
        }
        MetalLogitsOutputMode::RawLogits => Ok(psionic_backend_metal::MetalLogitsSelectionResult {
            selected_tokens: vec![greedy_index],
            candidates: None,
            logits: Some(logits.clone()),
            metrics: psionic_backend_metal::MetalLogitsSelectionMetrics {
                output_mode,
                readback_bytes: logits.len().saturating_mul(std::mem::size_of::<f32>()) as u64,
                raw_logits_materialized: true,
            },
        }),
    }
}

#[derive(Clone, Debug)]
enum MetalGemma4PerLayerProjectionMatrix {
    Host(ProjectionMatrix),
    Quantized(MetalQuantizedProjectionMatrix),
}

impl MetalGemma4PerLayerProjectionMatrix {
    fn is_native(&self) -> bool {
        matches!(self, Self::Quantized(matrix) if matrix.is_native())
    }

    fn load(
        backend: &mut MetalBackend,
        artifact: &GgufBlobArtifact,
        name: &str,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let storage = artifact.paged_tensor(name)?;
        if storage.metadata().quantized_layout.is_some() {
            return Ok(Self::Quantized(MetalQuantizedProjectionMatrix::load(
                backend, artifact, name,
            )?));
        }
        Ok(Self::Host(ProjectionMatrix::load(artifact, name)?))
    }

    fn matvec(
        &self,
        backend: &mut MetalBackend,
        input: &[f32],
    ) -> Result<MetalProjectionStep, ReferenceTextGenerationError> {
        match self {
            Self::Host(matrix) => {
                let mut values = Vec::new();
                matrix
                    .matvec(input, &mut values)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                Ok(MetalProjectionStep {
                    values,
                    kernel_count: 0,
                    bytes_moved: matrix.byte_length() as u64,
                })
            }
            Self::Quantized(matrix) => matrix.matvec(backend, input),
        }
    }

    fn byte_length(&self) -> usize {
        match self {
            Self::Host(matrix) => matrix.byte_length(),
            Self::Quantized(matrix) => matrix.byte_length(),
        }
    }
}

#[derive(Clone, Debug)]
struct MetalProjectionStep {
    values: Vec<f32>,
    kernel_count: usize,
    bytes_moved: u64,
}

#[derive(Clone, Debug)]
struct MetalDeviceAttentionStep {
    projected_output: Vec<f32>,
    key_values: Option<Vec<f32>>,
    value_values: Option<Vec<f32>>,
    kernel_count: usize,
    bytes_moved: u64,
}

#[allow(clippy::too_many_arguments)]
fn metal_gemma4_device_attention_with_kv(
    backend: &mut MetalBackend,
    layer: &MetalGemma4Layer,
    hidden_norm: &[f32],
    position: usize,
    layer_index: usize,
    family_metadata: &GgufDecoderFamilyMetadata,
    layer_cache: &mut MetalKvCacheMirror,
    layer_step_plan: &mut MetalGemma4LayerStepPlan,
) -> Result<Option<MetalDeviceAttentionStep>, ReferenceTextGenerationError> {
    if !backend.supports_dense_decode_attention(layer.attention_geometry.head_dim) {
        return Ok(None);
    }
    if !layer.attention_query_weight.is_native()
        || !layer.attention_key_weight.is_native()
        || !layer.attention_value_weight.is_native()
        || !layer.attention_output_weight.is_native()
    {
        return Ok(None);
    }
    let Some(key_buffer) = layer_step_plan.key_buffer.as_ref() else {
        return Ok(None);
    };
    let Some(value_buffer) = layer_step_plan.value_buffer.as_ref() else {
        return Ok(None);
    };

    layer_step_plan
        .hidden_input_buffer
        .write_f32(hidden_norm)
        .map_err(ReferenceTextGenerationError::Runtime)?;

    let mut submission = backend
        .begin_submission(format!("psionic.gemma4.metal.attn.layer{layer_index}"))
        .map_err(ReferenceTextGenerationError::Runtime)?;
    backend
        .encode_quantized_matvec_submission(
            &mut submission,
            layer.attention_query_weight.native_weights()?,
            0,
            layer.attention_query_weight.mode,
            layer.attention_query_weight.rows,
            layer.attention_query_weight.columns,
            &layer_step_plan.hidden_input_buffer,
            &layer_step_plan.query_buffer,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    backend
        .encode_quantized_matvec_submission(
            &mut submission,
            layer.attention_key_weight.native_weights()?,
            0,
            layer.attention_key_weight.mode,
            layer.attention_key_weight.rows,
            layer.attention_key_weight.columns,
            &layer_step_plan.hidden_input_buffer,
            key_buffer,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    backend
        .encode_quantized_matvec_submission(
            &mut submission,
            layer.attention_value_weight.native_weights()?,
            0,
            layer.attention_value_weight.mode,
            layer.attention_value_weight.rows,
            layer.attention_value_weight.columns,
            &layer_step_plan.hidden_input_buffer,
            value_buffer,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;

    if let Some(bias) = layer.attention_query_bias.as_ref() {
        backend
            .encode_add_inplace_submission(
                &mut submission,
                &layer_step_plan.query_buffer,
                bias.device(),
                layer.attention_query_weight.rows,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
    }
    if let Some(norm) = layer.attention_query_norm.as_ref() {
        backend
            .encode_per_head_rms_norm_submission(
                &mut submission,
                &layer_step_plan.query_buffer,
                norm.device(),
                layer.attention_geometry.head_count,
                layer.attention_geometry.head_dim,
                family_metadata.rms_norm_epsilon,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
    }
    backend
        .encode_rope_neox_position_submission(
            &mut submission,
            &layer_step_plan.query_buffer,
            &layer_step_plan.rope_freq_factors_buffer,
            layer.attention_geometry.head_count,
            layer.attention_geometry.head_dim,
            layer_step_plan.rope_rotary_half,
            position,
            layer_step_plan.rope_theta_scale,
            layer_step_plan.rope_freq_scale,
            layer_step_plan.rope_corr_dims,
            layer_step_plan.rope_ext_factor,
            layer_step_plan.rope_yarn_mscale,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;

    if let Some(bias) = layer.attention_key_bias.as_ref() {
        backend
            .encode_add_inplace_submission(
                &mut submission,
                key_buffer,
                bias.device(),
                layer.attention_key_weight.rows,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
    }
    if let Some(norm) = layer.attention_key_norm.as_ref() {
        backend
            .encode_per_head_rms_norm_submission(
                &mut submission,
                key_buffer,
                norm.device(),
                layer.attention_geometry.kv_head_count,
                layer.attention_geometry.head_dim,
                family_metadata.rms_norm_epsilon,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
    }
    backend
        .encode_rope_neox_position_submission(
            &mut submission,
            key_buffer,
            &layer_step_plan.rope_freq_factors_buffer,
            layer.attention_geometry.kv_head_count,
            layer.attention_geometry.head_dim,
            layer_step_plan.rope_rotary_half,
            position,
            layer_step_plan.rope_theta_scale,
            layer_step_plan.rope_freq_scale,
            layer_step_plan.rope_corr_dims,
            layer_step_plan.rope_ext_factor,
            layer_step_plan.rope_yarn_mscale,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;

    if let Some(bias) = layer.attention_value_bias.as_ref() {
        backend
            .encode_add_inplace_submission(
                &mut submission,
                value_buffer,
                bias.device(),
                layer.attention_value_weight.rows,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
    }
    backend
        .encode_per_head_rms_norm_unit_submission(
            &mut submission,
            value_buffer,
            layer.attention_geometry.kv_head_count,
            layer.attention_geometry.head_dim,
            family_metadata.rms_norm_epsilon,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;

    backend
        .encode_append_dense_kv_submission(&mut submission, layer_cache, key_buffer, value_buffer)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    backend
        .encode_decode_attention_dense_submission(
            &mut submission,
            &layer_step_plan.query_buffer,
            layer_cache,
            layer.attention_geometry.head_count,
            layer.attention_geometry.kv_head_count,
            layer.attention_geometry.head_dim,
            attention_scale(family_metadata, layer.attention_geometry.head_dim),
            &layer_step_plan.attention_buffer,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    backend
        .encode_quantized_matvec_submission(
            &mut submission,
            layer.attention_output_weight.native_weights()?,
            0,
            layer.attention_output_weight.mode,
            layer.attention_output_weight.rows,
            layer.attention_output_weight.columns,
            &layer_step_plan.attention_buffer,
            &layer_step_plan.projected_output_buffer,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    submission
        .synchronize_buffer(&layer_step_plan.projected_output_buffer)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    submission
        .synchronize_buffer(key_buffer)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    submission
        .synchronize_buffer(value_buffer)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    let report = submission
        .commit(psionic_backend_metal::MetalCommandWait::Completed)
        .map_err(ReferenceTextGenerationError::Runtime)?;

    layer_step_plan
        .projected_output_buffer
        .read_f32_prefix_into(
            layer.attention_output_weight.rows,
            &mut layer_step_plan.projected_output_values,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    key_buffer
        .read_f32_prefix_into(
            layer.attention_geometry.kv_width(),
            &mut layer_step_plan.live_key_values,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    value_buffer
        .read_f32_prefix_into(
            layer.attention_geometry.kv_width(),
            &mut layer_step_plan.live_value_values,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;

    Ok(Some(MetalDeviceAttentionStep {
        projected_output: layer_step_plan.projected_output_values.clone(),
        key_values: Some(layer_step_plan.live_key_values.clone()),
        value_values: Some(layer_step_plan.live_value_values.clone()),
        kernel_count: report.encoded_operations,
        bytes_moved: (layer.attention_query_weight.byte_length()
            + layer.attention_key_weight.byte_length()
            + layer.attention_value_weight.byte_length()
            + layer.attention_output_weight.byte_length()) as u64,
    }))
}

#[allow(clippy::too_many_arguments)]
fn metal_gemma4_device_attention_reuse(
    backend: &mut MetalBackend,
    layer: &MetalGemma4Layer,
    hidden_norm: &[f32],
    position: usize,
    layer_index: usize,
    family_metadata: &GgufDecoderFamilyMetadata,
    source_cache: &MetalKvCacheMirror,
    layer_step_plan: &mut MetalGemma4LayerStepPlan,
) -> Result<Option<MetalDeviceAttentionStep>, ReferenceTextGenerationError> {
    if !backend.supports_dense_decode_attention(layer.attention_geometry.head_dim) {
        return Ok(None);
    }
    if !layer.attention_query_weight.is_native() || !layer.attention_output_weight.is_native() {
        return Ok(None);
    }
    layer_step_plan
        .hidden_input_buffer
        .write_f32(hidden_norm)
        .map_err(ReferenceTextGenerationError::Runtime)?;

    let mut submission = backend
        .begin_submission(format!(
            "psionic.gemma4.metal.attn.reuse.layer{layer_index}"
        ))
        .map_err(ReferenceTextGenerationError::Runtime)?;
    backend
        .encode_quantized_matvec_submission(
            &mut submission,
            layer.attention_query_weight.native_weights()?,
            0,
            layer.attention_query_weight.mode,
            layer.attention_query_weight.rows,
            layer.attention_query_weight.columns,
            &layer_step_plan.hidden_input_buffer,
            &layer_step_plan.query_buffer,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    if let Some(bias) = layer.attention_query_bias.as_ref() {
        backend
            .encode_add_inplace_submission(
                &mut submission,
                &layer_step_plan.query_buffer,
                bias.device(),
                layer.attention_query_weight.rows,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
    }
    if let Some(norm) = layer.attention_query_norm.as_ref() {
        backend
            .encode_per_head_rms_norm_submission(
                &mut submission,
                &layer_step_plan.query_buffer,
                norm.device(),
                layer.attention_geometry.head_count,
                layer.attention_geometry.head_dim,
                family_metadata.rms_norm_epsilon,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
    }
    backend
        .encode_rope_neox_position_submission(
            &mut submission,
            &layer_step_plan.query_buffer,
            &layer_step_plan.rope_freq_factors_buffer,
            layer.attention_geometry.head_count,
            layer.attention_geometry.head_dim,
            layer_step_plan.rope_rotary_half,
            position,
            layer_step_plan.rope_theta_scale,
            layer_step_plan.rope_freq_scale,
            layer_step_plan.rope_corr_dims,
            layer_step_plan.rope_ext_factor,
            layer_step_plan.rope_yarn_mscale,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    backend
        .encode_decode_attention_dense_submission(
            &mut submission,
            &layer_step_plan.query_buffer,
            source_cache,
            layer.attention_geometry.head_count,
            layer.attention_geometry.kv_head_count,
            layer.attention_geometry.head_dim,
            attention_scale(family_metadata, layer.attention_geometry.head_dim),
            &layer_step_plan.attention_buffer,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    backend
        .encode_quantized_matvec_submission(
            &mut submission,
            layer.attention_output_weight.native_weights()?,
            0,
            layer.attention_output_weight.mode,
            layer.attention_output_weight.rows,
            layer.attention_output_weight.columns,
            &layer_step_plan.attention_buffer,
            &layer_step_plan.projected_output_buffer,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    submission
        .synchronize_buffer(&layer_step_plan.projected_output_buffer)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    let report = submission
        .commit(psionic_backend_metal::MetalCommandWait::Completed)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    layer_step_plan
        .projected_output_buffer
        .read_f32_prefix_into(
            layer.attention_output_weight.rows,
            &mut layer_step_plan.projected_output_values,
        )
        .map_err(ReferenceTextGenerationError::Runtime)?;
    Ok(Some(MetalDeviceAttentionStep {
        projected_output: layer_step_plan.projected_output_values.clone(),
        key_values: None,
        value_values: None,
        kernel_count: report.encoded_operations,
        bytes_moved: (layer.attention_query_weight.byte_length()
            + layer.attention_output_weight.byte_length()) as u64,
    }))
}

#[derive(Clone, Debug)]
struct MetalGemma4StageStep {
    key: Vec<f32>,
    value: Vec<f32>,
    logits: Vec<f32>,
    selected_token: Option<TokenId>,
    hidden: Vec<f32>,
    lm_head_hidden: Option<Vec<f32>>,
    kernel_count: usize,
    bytes_moved: u64,
}

#[derive(Clone, Debug)]
struct MetalGemma4ForwardStep {
    key: Vec<f32>,
    value: Vec<f32>,
    logits: Vec<f32>,
    selected_token: Option<TokenId>,
    final_hidden: Vec<f32>,
    kernel_count: usize,
    bytes_moved: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedGemma4PeerConfig {
    pub peer_base_url: String,
    pub split_layer: usize,
    pub shared_key: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct DistributedGemma4PeerStatus {
    #[serde(default)]
    nodes: Vec<DistributedGemma4PeerNode>,
    #[serde(default)]
    routes: Vec<DistributedGemma4PeerPublishedModel>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct DistributedGemma4PeerNode {
    #[serde(default)]
    models: Vec<DistributedGemma4PeerPublishedModel>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct DistributedGemma4PeerPublishedModel {
    model_key: String,
    canonical_name: String,
    family: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistributedGemma4RemoteStepRequest {
    pub request_id: String,
    pub model_id: String,
    pub token: u32,
    pub position: usize,
    pub split_layer: usize,
    pub input_hidden: Vec<f32>,
    pub forwarded_key: Vec<f32>,
    pub forwarded_value: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistributedGemma4RemoteStepResponse {
    pub logits: Vec<f32>,
    pub kernel_count: usize,
    pub bytes_moved: u64,
}

pub(crate) fn encode_distributed_gemma4_remote_step_request(
    request: &DistributedGemma4RemoteStepRequest,
) -> Result<Vec<u8>, ReferenceTextGenerationError> {
    let request_id = request.request_id.as_bytes();
    let model_id = request.model_id.as_bytes();
    let request_id_len = u64::try_from(request_id.len()).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step request id exceeds u64 length",
        )))
    })?;
    let model_id_len = u64::try_from(model_id.len()).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step model id exceeds u64 length",
        )))
    })?;
    let position = u64::try_from(request.position).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step position exceeds u64",
        )))
    })?;
    let split_layer = u64::try_from(request.split_layer).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step split layer exceeds u64",
        )))
    })?;
    let input_hidden_len = u64::try_from(request.input_hidden.len()).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step hidden length exceeds u64",
        )))
    })?;
    let forwarded_key_len = u64::try_from(request.forwarded_key.len()).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step forwarded key length exceeds u64",
        )))
    })?;
    let forwarded_value_len = u64::try_from(request.forwarded_value.len()).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step forwarded value length exceeds u64",
        )))
    })?;
    let floats_len = request
        .input_hidden
        .len()
        .saturating_add(request.forwarded_key.len())
        .saturating_add(request.forwarded_value.len());
    let mut encoded = Vec::with_capacity(
        60usize
            .saturating_add(request_id.len())
            .saturating_add(model_id.len())
            .saturating_add(floats_len.saturating_mul(std::mem::size_of::<f32>())),
    );
    encoded.extend_from_slice(&request_id_len.to_le_bytes());
    encoded.extend_from_slice(&model_id_len.to_le_bytes());
    encoded.extend_from_slice(&request.token.to_le_bytes());
    encoded.extend_from_slice(&position.to_le_bytes());
    encoded.extend_from_slice(&split_layer.to_le_bytes());
    encoded.extend_from_slice(&input_hidden_len.to_le_bytes());
    encoded.extend_from_slice(&forwarded_key_len.to_le_bytes());
    encoded.extend_from_slice(&forwarded_value_len.to_le_bytes());
    encoded.extend_from_slice(request_id);
    encoded.extend_from_slice(model_id);
    for value in &request.input_hidden {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    for value in &request.forwarded_key {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    for value in &request.forwarded_value {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    Ok(encoded)
}

pub(crate) fn decode_distributed_gemma4_remote_step_request(
    bytes: &[u8],
) -> Result<DistributedGemma4RemoteStepRequest, ReferenceTextGenerationError> {
    const HEADER_LEN: usize = 60;
    if bytes.len() < HEADER_LEN {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "distributed gemma4 remote step request is truncated: expected at least {HEADER_LEN} bytes, got {}",
                bytes.len()
            )),
        ));
    }
    let request_id_len = u64::from_le_bytes(bytes[0..8].try_into().map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step request id length header is malformed",
        )))
    })?);
    let model_id_len = u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step model id length header is malformed",
        )))
    })?);
    let token = u32::from_le_bytes(bytes[16..20].try_into().map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step token header is malformed",
        )))
    })?);
    let position = u64::from_le_bytes(bytes[20..28].try_into().map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step position header is malformed",
        )))
    })?);
    let split_layer = u64::from_le_bytes(bytes[28..36].try_into().map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step split-layer header is malformed",
        )))
    })?);
    let input_hidden_len = u64::from_le_bytes(bytes[36..44].try_into().map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step hidden-length header is malformed",
        )))
    })?);
    let forwarded_key_len = u64::from_le_bytes(bytes[44..52].try_into().map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step key-length header is malformed",
        )))
    })?);
    let forwarded_value_len = u64::from_le_bytes(bytes[52..60].try_into().map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step value-length header is malformed",
        )))
    })?);
    let request_id_len = usize::try_from(request_id_len).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step request id length exceeds usize",
        )))
    })?;
    let model_id_len = usize::try_from(model_id_len).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step model id length exceeds usize",
        )))
    })?;
    let position = usize::try_from(position).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step position exceeds usize",
        )))
    })?;
    let split_layer = usize::try_from(split_layer).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step split layer exceeds usize",
        )))
    })?;
    let input_hidden_len = usize::try_from(input_hidden_len).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step hidden length exceeds usize",
        )))
    })?;
    let forwarded_key_len = usize::try_from(forwarded_key_len).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step key length exceeds usize",
        )))
    })?;
    let forwarded_value_len = usize::try_from(forwarded_value_len).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step value length exceeds usize",
        )))
    })?;
    let string_bytes = request_id_len.checked_add(model_id_len).ok_or_else(|| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote step string section overflow",
        )))
    })?;
    let float_count = input_hidden_len
        .checked_add(forwarded_key_len)
        .and_then(|value| value.checked_add(forwarded_value_len))
        .ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "distributed gemma4 remote step float section overflow",
            )))
        })?;
    let expected_len = HEADER_LEN
        .checked_add(string_bytes)
        .and_then(|value| value.checked_add(float_count.saturating_mul(std::mem::size_of::<f32>())))
        .ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "distributed gemma4 remote step total payload length overflow",
            )))
        })?;
    if bytes.len() != expected_len {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "distributed gemma4 remote step request length mismatch: expected {expected_len} bytes, got {}",
                bytes.len()
            )),
        ));
    }
    let request_id_start = HEADER_LEN;
    let request_id_end = request_id_start + request_id_len;
    let model_id_end = request_id_end + model_id_len;
    let request_id = std::str::from_utf8(&bytes[request_id_start..request_id_end])
        .map_err(|error| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                "distributed gemma4 remote step request id is not valid utf-8: {error}",
            )))
        })?
        .to_owned();
    let model_id = std::str::from_utf8(&bytes[request_id_end..model_id_end])
        .map_err(|error| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                "distributed gemma4 remote step model id is not valid utf-8: {error}",
            )))
        })?
        .to_owned();
    let mut cursor = model_id_end;
    let mut read_f32_vec = |len: usize| -> Result<Vec<f32>, ReferenceTextGenerationError> {
        let byte_len = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "distributed gemma4 remote step float byte length overflow",
            )))
        })?;
        let end = cursor.checked_add(byte_len).ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "distributed gemma4 remote step cursor overflow",
            )))
        })?;
        let mut values = Vec::with_capacity(len);
        for chunk in bytes[cursor..end].chunks_exact(std::mem::size_of::<f32>()) {
            values.push(f32::from_le_bytes(
                chunk.try_into().expect("f32 chunk width"),
            ));
        }
        cursor = end;
        Ok(values)
    };
    let input_hidden = read_f32_vec(input_hidden_len)?;
    let forwarded_key = read_f32_vec(forwarded_key_len)?;
    let forwarded_value = read_f32_vec(forwarded_value_len)?;
    Ok(DistributedGemma4RemoteStepRequest {
        request_id,
        model_id,
        token,
        position,
        split_layer,
        input_hidden,
        forwarded_key,
        forwarded_value,
    })
}

pub(crate) fn encode_distributed_gemma4_remote_step_response(
    response: &DistributedGemma4RemoteStepResponse,
) -> Result<Vec<u8>, ReferenceTextGenerationError> {
    let logits_len = u64::try_from(response.logits.len()).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote suffix response exceeds u64 logits length",
        )))
    })?;
    let kernel_count = u64::try_from(response.kernel_count).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote suffix response exceeds u64 kernel count",
        )))
    })?;
    let mut encoded = Vec::with_capacity(24 + response.logits.len() * std::mem::size_of::<f32>());
    encoded.extend_from_slice(&logits_len.to_le_bytes());
    encoded.extend_from_slice(&kernel_count.to_le_bytes());
    encoded.extend_from_slice(&response.bytes_moved.to_le_bytes());
    for logit in &response.logits {
        encoded.extend_from_slice(&logit.to_le_bytes());
    }
    Ok(encoded)
}

pub(crate) fn decode_distributed_gemma4_remote_step_response(
    bytes: &[u8],
) -> Result<DistributedGemma4RemoteStepResponse, ReferenceTextGenerationError> {
    if bytes.len() < 24 {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "distributed gemma4 remote suffix response is truncated: expected at least 24 bytes, got {}",
                bytes.len()
            )),
        ));
    }
    let mut logits_len_bytes = [0u8; 8];
    logits_len_bytes.copy_from_slice(&bytes[0..8]);
    let logits_len = u64::from_le_bytes(logits_len_bytes);
    let mut kernel_count_bytes = [0u8; 8];
    kernel_count_bytes.copy_from_slice(&bytes[8..16]);
    let kernel_count = u64::from_le_bytes(kernel_count_bytes);
    let mut bytes_moved_bytes = [0u8; 8];
    bytes_moved_bytes.copy_from_slice(&bytes[16..24]);
    let bytes_moved = u64::from_le_bytes(bytes_moved_bytes);
    let logits_len = usize::try_from(logits_len).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote suffix response logits length exceeds usize",
        )))
    })?;
    let kernel_count = usize::try_from(kernel_count).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
            "distributed gemma4 remote suffix response kernel count exceeds usize",
        )))
    })?;
    let expected_len = 24usize
        .checked_add(
            logits_len
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                        String::from(
                            "distributed gemma4 remote suffix response logits byte length overflow",
                        ),
                    ))
                })?,
        )
        .ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "distributed gemma4 remote suffix response total byte length overflow",
            )))
        })?;
    if bytes.len() != expected_len {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "distributed gemma4 remote suffix response has invalid size: expected {expected_len} bytes, got {}",
                bytes.len()
            )),
        ));
    }
    let mut logits = Vec::with_capacity(logits_len);
    for chunk in bytes[24..].chunks_exact(4) {
        let mut value = [0u8; 4];
        value.copy_from_slice(chunk);
        logits.push(f32::from_le_bytes(value));
    }
    Ok(DistributedGemma4RemoteStepResponse {
        logits,
        kernel_count,
        bytes_moved,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistributedGemma4RemoteResetRequest {
    pub request_id: String,
    pub model_id: String,
}

struct DistributedGemma4FrontBackend {
    metal: MetalBackend,
    client: reqwest::blocking::Client,
    peer: DistributedGemma4PeerConfig,
    peer_model_key: String,
    active_request_id: Option<String>,
}

impl DistributedGemma4FrontBackend {
    fn new(
        peer: DistributedGemma4PeerConfig,
        canonical_name: &str,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|error| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                    "failed to build distributed gemma4 peer client: {error}"
                )))
            })?;
        let peer_model_key = Self::resolve_peer_model_key(&client, &peer, canonical_name)?;
        Ok(Self {
            metal: MetalBackend::new(),
            client,
            peer,
            peer_model_key,
            active_request_id: None,
        })
    }

    fn resolve_peer_model_key(
        client: &reqwest::blocking::Client,
        peer: &DistributedGemma4PeerConfig,
        canonical_name: &str,
    ) -> Result<String, ReferenceTextGenerationError> {
        let response = client
            .get(format!("{}/psionic/management/status", peer.peer_base_url))
            .send()
            .map_err(|error| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                    "failed to fetch distributed gemma4 peer status: {error}"
                )))
            })?
            .error_for_status()
            .map_err(|error| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                    "distributed gemma4 peer status request failed with status: {error}"
                )))
            })?;
        let status = response
            .json::<DistributedGemma4PeerStatus>()
            .map_err(|error| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                    "failed to decode distributed gemma4 peer status: {error}"
                )))
            })?;
        let mut gemma_models = status
            .routes
            .into_iter()
            .chain(
                status
                    .nodes
                    .into_iter()
                    .flat_map(|node| node.models.into_iter()),
            )
            .filter(|model| model.family == "gemma4")
            .collect::<Vec<_>>();
        if let Some(model) = gemma_models
            .iter()
            .find(|model| {
                model.canonical_name == canonical_name || model.model_key == canonical_name
            })
            .or_else(|| (gemma_models.len() == 1).then(|| &gemma_models[0]))
        {
            return Ok(model.model_key.clone());
        }
        gemma_models.sort_by(|left, right| left.model_key.cmp(&right.model_key));
        let available = gemma_models
            .into_iter()
            .map(|model| format!("{}=>{}", model.canonical_name, model.model_key))
            .collect::<Vec<_>>()
            .join(", ");
        Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "failed to resolve distributed gemma4 peer model key for `{canonical_name}`; available gemma4 routes: {available}"
            )),
        ))
    }

    fn health(&mut self) -> psionic_runtime::RuntimeHealth {
        self.metal.health()
    }

    fn begin_request(&mut self, request_id: &str) {
        self.active_request_id = Some(request_id.to_string());
    }

    fn finish_request(&mut self, _model_id: &str) {
        let Some(request_id) = self.active_request_id.take() else {
            return;
        };
        let client = self.client.clone();
        let peer = self.peer.clone();
        let model_id = self.peer_model_key.clone();
        let _ = std::thread::Builder::new()
            .name(String::from("psionic-gemma4-distributed-reset"))
            .spawn(move || {
                let mut request = client.post(format!(
                    "{}/psionic/internal/gemma4/pipeline/reset",
                    peer.peer_base_url
                ));
                if let Some(shared_key) = peer.shared_key.as_deref() {
                    request = request.header("x-psionic-distributed-key", shared_key);
                }
                let _ = request
                    .json(&DistributedGemma4RemoteResetRequest {
                        request_id,
                        model_id,
                    })
                    .send();
            });
    }

    fn remote_suffix_step(
        &mut self,
        _model_id: &str,
        token: TokenId,
        position: usize,
        input_hidden: &[f32],
        forwarded_key: &[f32],
        forwarded_value: &[f32],
        split_layer: usize,
    ) -> Result<DistributedGemma4RemoteStepResponse, ReferenceTextGenerationError> {
        let request_id = self.active_request_id.clone().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "distributed gemma4 backend attempted a remote step without an active request id",
            )))
        })?;
        let body =
            encode_distributed_gemma4_remote_step_request(&DistributedGemma4RemoteStepRequest {
                request_id: request_id.clone(),
                model_id: self.peer_model_key.clone(),
                token: token.as_u32(),
                position,
                split_layer,
                input_hidden: input_hidden.to_vec(),
                forwarded_key: forwarded_key.to_vec(),
                forwarded_value: forwarded_value.to_vec(),
            })?;
        let mut request = self.client.post(format!(
            "{}/psionic/internal/gemma4/pipeline/step",
            self.peer.peer_base_url
        ));
        if let Some(shared_key) = self.peer.shared_key.as_deref() {
            request = request.header("x-psionic-distributed-key", shared_key);
        }
        let response = request
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .body(body)
            .send()
            .map_err(|error| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                    "distributed gemma4 remote suffix request failed: {error}"
                )))
            })?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            let detail = if body.trim().is_empty() {
                String::new()
            } else {
                format!(" body={body}")
            };
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "distributed gemma4 remote suffix request failed with status: {status}{detail}"
                )),
            ));
        }
        let body = response.bytes().map_err(|error| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                "failed to read distributed gemma4 remote suffix response body: {error}"
            )))
        })?;
        decode_distributed_gemma4_remote_step_response(body.as_ref())
    }
}

pub struct DistributedGemma4TextGenerationService {
    backend: DistributedGemma4FrontBackend,
    models: InMemoryGenerationModelRegistry<DistributedGemma4GenerationModel>,
    sessions: InMemoryGenerationSessionStore,
    shared_prefixes: SharedPrefixStore,
    backend_health: super::BackendHealthTracker,
    model_descriptor: DecoderModelDescriptor,
    runtime_support: GgufDecoderRuntimeSupport,
    peer: DistributedGemma4PeerConfig,
}

impl DistributedGemma4TextGenerationService {
    pub fn from_gguf_path(
        path: impl AsRef<Path>,
        peer: DistributedGemma4PeerConfig,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let canonical_name = path
            .as_ref()
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .map(String::from)
            .unwrap_or_else(|| String::from("gemma4"));
        let mut backend =
            DistributedGemma4FrontBackend::new(peer.clone(), canonical_name.as_str())?;
        let metal_model = MetalGemma4GenerationModel::from_gguf_path(path, &mut backend.metal)?;
        let model =
            DistributedGemma4GenerationModel::from_metal_model(metal_model, peer.split_layer)?;
        let model_descriptor = model.descriptor().clone();
        let runtime_support = model.runtime_support();
        let mut models = InMemoryGenerationModelRegistry::new();
        let now_millis = super::current_time_millis();
        models.warm_with_metadata(
            model,
            now_millis,
            super::DEFAULT_MODEL_KEEPALIVE_MILLIS,
            None,
            Some(String::from("metal")),
            None,
        )?;
        let mut backend_health = super::BackendHealthTracker::default();
        backend_health.observe("metal", backend.health(), now_millis);
        Ok(Self {
            backend,
            models,
            sessions: InMemoryGenerationSessionStore::new(),
            shared_prefixes: SharedPrefixStore::default(),
            backend_health,
            model_descriptor,
            runtime_support,
            peer,
        })
    }

    #[must_use]
    pub fn model_descriptor(&self) -> &DecoderModelDescriptor {
        &self.model_descriptor
    }

    #[must_use]
    pub fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        self.runtime_support.clone()
    }

    pub fn generate_continuous_batch(
        &mut self,
        requests: Vec<GenerationRequest>,
    ) -> ContinuousBatchGenerationResult {
        ContinuousBatchGenerationResult {
            responses: requests
                .into_iter()
                .map(|request| self.generate(&request))
                .collect(),
            scheduler_metrics: psionic_runtime::GenerationSchedulerMetrics::default(),
        }
    }

    fn attach_cluster_execution(&self, response: &mut GenerationResponse) {
        let Some(provenance) = response.provenance.as_mut() else {
            return;
        };
        let mut digest = Sha256::new();
        digest.update(self.model_descriptor.model.model_id.as_bytes());
        digest.update(b"|gemma4_distributed_front|");
        digest.update(self.peer.peer_base_url.as_bytes());
        digest.update(format!("|split:{}|", self.peer.split_layer).as_bytes());
        let topology_digest = format!("{:x}", digest.finalize());
        let cluster_id = format!("gemma4-distributed-{}", &topology_digest[..12]);
        let local = psionic_runtime::DeviceInventoryQualifiers {
            stable_device_id: String::from("local-metal"),
            topology_key: None,
            performance_class: psionic_runtime::DevicePerformanceClass::IntegratedAccelerator,
            memory_class: psionic_runtime::DeviceMemoryClass::SharedHostDevice,
            total_memory_bytes: None,
            free_memory_bytes: None,
        };
        let remote = psionic_runtime::DeviceInventoryQualifiers {
            stable_device_id: format!("remote-cuda@{}", self.peer.peer_base_url),
            topology_key: None,
            performance_class: psionic_runtime::DevicePerformanceClass::DiscreteAccelerator,
            memory_class: psionic_runtime::DeviceMemoryClass::DedicatedDevice,
            total_memory_bytes: None,
            free_memory_bytes: None,
        };
        let capability_profile =
            psionic_runtime::ClusterExecutionCapabilityProfile::new("metal+cuda")
                .with_supported_lanes(vec![psionic_runtime::ClusterExecutionLane::PipelineSharded])
                .with_serving_semantics_capability(
                    psionic_runtime::ClusterServingSemantics::new(
                        psionic_runtime::ClusterExecutionLane::PipelineSharded,
                        super::default_text_generation_execution_profile(),
                        psionic_runtime::ClusterWarmRoutePosture::TopologyPinned,
                    )
                    .with_detail(
                        "one Mac prefix stage hands each token to one CUDA suffix stage over the distributed proof path",
                    ),
                );
        let cluster_execution = psionic_runtime::ClusterExecutionContext::new(
            cluster_id.clone(),
            topology_digest.clone(),
            topology_digest.clone(),
            "distributed-front",
            psionic_runtime::ClusterTransportClass::WiderNetworkStream,
            psionic_runtime::ClusterExecutionDisposition::Sharded,
        )
        .with_communication_eligibility(
            capability_profile.lane_communication_eligibility(
                psionic_runtime::ClusterExecutionLane::PipelineSharded,
            ),
        )
        .with_serving_semantics(
            capability_profile
                .serving_semantics_capability(
                    psionic_runtime::ClusterExecutionLane::PipelineSharded,
                )
                .cloned()
                .unwrap_or_else(|| {
                    psionic_runtime::ClusterServingSemantics::new(
                        psionic_runtime::ClusterExecutionLane::PipelineSharded,
                        super::default_text_generation_execution_profile(),
                        psionic_runtime::ClusterWarmRoutePosture::TopologyPinned,
                    )
                }),
        )
        .with_execution_topology(psionic_runtime::ExecutionTopologyPlan::pipeline_sharded(
            "metal+cuda",
            vec![
                (local.clone(), 0, self.peer.split_layer),
                (
                    remote.clone(),
                    self.peer.split_layer,
                    self.model_descriptor.config.layer_count,
                ),
            ],
        ))
        .with_selected_nodes(vec![
            psionic_runtime::ClusterSelectedNode::new("local-metal", "metal").with_role("front"),
            psionic_runtime::ClusterSelectedNode::new("remote-cuda", "cuda").with_role("suffix"),
        ])
        .with_pipeline_stages(vec![
            psionic_runtime::ClusterPipelineStage::new(
                0,
                "local-metal",
                psionic_runtime::ClusterPipelineStageRole::Entry,
                0,
                self.peer.split_layer,
                0,
                0,
                0,
            )
            .with_handoff(
                psionic_runtime::ClusterTransportClass::WiderNetworkStream,
                None,
                None,
            )
            .with_detail("Mac Metal prefix stage"),
            psionic_runtime::ClusterPipelineStage::new(
                1,
                "remote-cuda",
                psionic_runtime::ClusterPipelineStageRole::Exit,
                self.peer.split_layer,
                self.model_descriptor.config.layer_count,
                0,
                0,
                0,
            )
            .with_detail("remote CUDA suffix stage"),
        ]);
        provenance.cluster_execution = Some(cluster_execution);
    }
}

impl TextGenerationExecutor for DistributedGemma4TextGenerationService {
    type Error = ReferenceTextGenerationError;

    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResponse, Self::Error> {
        self.backend.begin_request(request.request_id.as_str());
        let result = run_distributed_gemma4_generation_request(
            &mut self.backend,
            &mut self.models,
            &mut self.sessions,
            &mut self.shared_prefixes,
            request,
        );
        self.backend
            .finish_request(request.model.model.model_id.as_str());
        let mut response = result?;
        self.attach_cluster_execution(&mut response);
        Ok(response)
    }
}

impl StreamingTextGenerationExecutor for DistributedGemma4TextGenerationService {
    type Stream<'a> = Box<dyn GenerationEventStream + 'a>;

    fn generate_stream<'a>(
        &'a mut self,
        request: &GenerationRequest,
    ) -> Result<Self::Stream<'a>, ReferenceTextGenerationError> {
        let response = self.generate(request)?;
        Ok(Box::new(CompletedGenerationStream::new(response)))
    }
}

impl ManagedTextGenerationRuntime for DistributedGemma4TextGenerationService {
    fn loaded_models(&mut self) -> LoadedModelsObservation {
        self.models.loaded_models_observation()
    }

    fn observability(&mut self) -> LocalRuntimeObservability {
        let now_millis = super::current_time_millis();
        self.models.expire_idle(now_millis);
        self.backend_health
            .observe("metal", self.backend.health(), now_millis);
        super::generation_runtime_observability(
            &self.models,
            &self.sessions,
            &self.backend_health,
            super::default_text_generation_execution_profile(),
        )
    }

    fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Ok(self
            .models
            .warm_loaded(model_id, super::current_time_millis(), keep_alive_millis)?)
    }

    fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Ok(self
            .models
            .unload_view(model_id, super::current_time_millis())?)
    }
}

#[derive(Clone, Debug)]
struct DistributedGemma4GenerationModel {
    inner: Arc<MetalGemma4ModelInner>,
    split_layer: usize,
}

impl DistributedGemma4GenerationModel {
    fn from_metal_model(
        model: MetalGemma4GenerationModel,
        split_layer: usize,
    ) -> Result<Self, ReferenceTextGenerationError> {
        if split_layer == 0 || split_layer >= model.inner.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "distributed gemma4 split layer must be within (0, {}), got {split_layer}",
                    model.inner.layers.len()
                )),
            ));
        }
        Ok(Self {
            inner: Arc::clone(&model.inner),
            split_layer,
        })
    }

    fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        let mut support = runtime_support_for_descriptor(
            &self.inner.descriptor,
            GgufDecoderFamily::Gemma4,
            vec![String::from("metal"), String::from("cuda")],
            vec![String::from("cpu")],
            unsupported_adapter_runtime_support(
                "LM-head LoRA serving is currently unsupported on the distributed gemma4 proof path",
            ),
        );
        support.unsupported_features = vec![
            String::from("image_inputs"),
            String::from("video_inputs"),
            String::from("audio_inputs"),
        ];
        support
    }
}

impl crate::GenerationModelHandle for DistributedGemma4GenerationModel {
    fn descriptor(&self) -> &DecoderModelDescriptor {
        &self.inner.descriptor
    }

    fn cache_width(&self) -> usize {
        self.inner.cache_width()
    }
}

impl super::CompiledWordGenerationModel for DistributedGemma4GenerationModel {
    type Backend = DistributedGemma4FrontBackend;

    fn tokenizer(&self) -> &dyn TokenizerBoundary {
        &self.inner.tokenizer
    }

    fn encode_prompt_input(
        &self,
        input: &GenerationInput,
    ) -> Result<TokenSequence, ReferenceTextGenerationError> {
        Ok(match input {
            GenerationInput::Text(text) => self.inner.tokenizer.encode_with_defaults(text),
            GenerationInput::Tokens(tokens) => tokens.clone(),
        })
    }

    fn is_end_of_sequence(&self, token: TokenId) -> bool {
        self.inner.tokenizer.is_end_of_sequence(token)
    }

    fn execute_step(
        &self,
        backend: &mut Self::Backend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
    ) -> Result<GenerationStepOutput, ReferenceTextGenerationError> {
        let config = &self.inner.descriptor.config;
        if token.as_u32() as usize >= config.vocab_size {
            return Err(ReferenceTextGenerationError::InvalidToken {
                token: token.as_u32(),
                vocab_size: config.vocab_size,
            });
        }
        if position >= config.max_context {
            return Err(ReferenceTextGenerationError::InvalidPosition {
                position,
                max_context: config.max_context,
            });
        }
        if cache.width() != self.inner.cache_width() {
            return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
                expected_kv_width: self.inner.cache_width(),
                kv_width: cache.width(),
            });
        }
        let prefix_step = self.inner.forward_stage_step(
            &mut backend.metal,
            token,
            position,
            cache,
            0,
            self.split_layer,
            None,
            None,
            None,
            false,
            MetalLogitsOutputMode::RawLogits,
        )?;
        let remote_step = backend.remote_suffix_step(
            self.inner.descriptor.model.model_id.as_str(),
            token,
            position,
            prefix_step.hidden.as_slice(),
            prefix_step.key.as_slice(),
            prefix_step.value.as_slice(),
            self.split_layer,
        )?;
        Ok(GenerationStepOutput {
            key: prefix_step.key,
            value: prefix_step.value,
            logits: remote_step.logits,
            hidden: None,
            execution_plan_digest: Some(self.inner.plan_digest.clone()),
            compile_path: None,
            kernel_count: prefix_step
                .kernel_count
                .saturating_add(remote_step.kernel_count),
            bytes_moved: prefix_step
                .bytes_moved
                .saturating_add(remote_step.bytes_moved),
            plan_cache_hits: 0,
            plan_cache_misses: 0,
            gpt_oss_perf: None,
        })
    }

    fn plan_digest(&self) -> &str {
        self.inner.plan_digest.as_str()
    }

    fn load_duration_ns(&self) -> u64 {
        self.inner.load_duration_ns
    }

    fn backend_compatibility(&self) -> &'static str {
        "metal"
    }
}

fn run_distributed_gemma4_generation_request(
    backend: &mut DistributedGemma4FrontBackend,
    models: &mut InMemoryGenerationModelRegistry<DistributedGemma4GenerationModel>,
    sessions: &mut InMemoryGenerationSessionStore,
    shared_prefixes: &mut SharedPrefixStore,
    request: &GenerationRequest,
) -> Result<GenerationResponse, ReferenceTextGenerationError> {
    if !super::generation_product_supported(request) {
        return Err(ReferenceTextGenerationError::UnsupportedProduct(
            request.product_id.clone(),
        ));
    }

    let loaded_model = models
        .active(request.model.model.model_id.as_str())
        .ok_or_else(|| {
            ReferenceTextGenerationError::UnsupportedModel(request.model.model.model_id.clone())
        })?
        .clone();
    if loaded_model.descriptor() != &request.model {
        return Err(ReferenceTextGenerationError::UnsupportedModel(
            request.model.model.model_id.clone(),
        ));
    }

    let model_id = request.model.model.model_id.as_str();
    let load_state = models
        .load_state(model_id)
        .unwrap_or(super::GenerationLoadState::Warm);
    let request_start = super::current_time_millis();
    let generation_start = Instant::now();
    models.begin_request(model_id, request_start)?;
    let memory_plan = models.memory_plan(model_id).cloned();
    let residency_policy = Some(models.residency_policy().clone());
    let residency_snapshot = Some(models.memory_snapshot());
    let served_artifact = super::served_artifact_identity_for_decoder_backend(
        loaded_model.descriptor(),
        loaded_model.backend_compatibility(),
        &[],
    );
    let effective_served_artifact_digest =
        super::effective_generation_served_artifact_digest(&served_artifact, None);

    let result = (|| -> Result<GenerationResponse, ReferenceTextGenerationError> {
        let prompt_eval_start = Instant::now();
        let tokenizer = loaded_model.tokenizer();
        let prompt_tokens = loaded_model.encode_prompt_input(&request.prompt)?;
        if prompt_tokens.is_empty() {
            return Err(ReferenceTextGenerationError::EmptyPrompt);
        }

        let expected_kv_width = loaded_model.cache_width();
        let mut session_tokens = Vec::new();
        let compatibility = super::prefix_compatibility_for_request(&loaded_model, request);
        let prefix_policy = super::default_prefix_cache_policy();
        let mut prefix_state = crate::PrefixCacheState::None;
        let mut prefix_cache_refusal_reason = None;
        let mut prefix_cache_invalidation_trigger = None;
        let mut prefix_tokens_reused = 0usize;
        let mut prefix_identity = None;
        let mut shared_prefix_eligible = false;
        let previous_kv_state = if let Some(session_id) = &request.session_id {
            if request.reset_session {
                sessions.reset(session_id)?;
            }
            let state = sessions.state(session_id)?;
            super::validate_session_model(
                state,
                session_id,
                loaded_model.descriptor(),
                effective_served_artifact_digest.as_str(),
            )?;
            session_tokens = state.tokens().to_vec();
            if state.cache().is_empty() {
                shared_prefix_eligible = true;
            } else {
                prefix_state = crate::PrefixCacheState::Bypassed;
                prefix_cache_refusal_reason =
                    Some(crate::PrefixCacheRefusalReason::SessionBoundState);
            }
            state.cache().state()
        } else {
            shared_prefix_eligible = true;
            psionic_runtime::KvCacheState::default()
        };
        let preserve_prefix_tokens = usize::from(
            prompt_tokens.as_slice().first().copied() == Some(tokenizer.vocabulary().bos_id()),
        );
        let (prompt_tokens, context_window) = super::apply_context_window(
            &prompt_tokens,
            loaded_model.descriptor().config.max_context,
            previous_kv_state.tokens,
            request.options.max_output_tokens,
            request.options.context_overflow_policy,
            preserve_prefix_tokens,
        )?;
        let mut prompt_logits = Vec::new();
        let mut last_logits = Vec::new();
        let mut cache = if shared_prefix_eligible {
            let lookup = super::controlled_prefix_lookup(
                shared_prefixes,
                &compatibility,
                &prompt_tokens,
                request,
            );
            prefix_state = lookup.state;
            prefix_cache_refusal_reason = lookup.refusal_reason;
            prefix_cache_invalidation_trigger = lookup.invalidation_trigger;
            prefix_tokens_reused = lookup.reused_tokens;
            prefix_identity = lookup.identity;
            prompt_logits = lookup.prompt_logits;
            last_logits = if lookup.last_logits.is_empty() {
                prompt_logits.last().cloned().unwrap_or_default()
            } else {
                lookup.last_logits
            };
            lookup.cache.unwrap_or_else(|| {
                crate::InMemoryKvCache::new(
                    loaded_model.descriptor().config.max_context,
                    expected_kv_width,
                )
            })
        } else if let Some(session_id) = &request.session_id {
            sessions.state(session_id)?.cache().clone()
        } else {
            crate::InMemoryKvCache::new(
                loaded_model.descriptor().config.max_context,
                expected_kv_width,
            )
        };
        if cache.width() != expected_kv_width {
            return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
                expected_kv_width,
                kv_width: cache.width(),
            });
        }
        cache.bind_owner(super::request_kv_owner(
            request,
            psionic_runtime::BatchExecutionPosture::SingleRequestOnly,
            None,
        ));
        let request_kv_checkpoint = cache.checkpoint();
        let kv_cache_encoding_policy = super::default_generation_kv_cache_encoding_policy(
            &loaded_model,
            loaded_model.backend_compatibility(),
        );
        let mut layer_caches = build_gemma4_metal_layer_caches_from_host_cache(
            &mut backend.metal,
            &loaded_model.inner,
            &cache,
            request.options.max_output_tokens.saturating_add(1),
            &kv_cache_encoding_policy,
        )?;
        let mut step_plan = loaded_model
            .inner
            .build_decode_step_plan(&mut backend.metal)?;
        let mut kernel_count = 0usize;
        let mut bytes_moved = 0u64;

        for token in &prompt_tokens.as_slice()[prefix_tokens_reused..] {
            let prefix_step = loaded_model.inner.forward_step_with_layer_caches(
                &mut backend.metal,
                *token,
                cache.len(),
                &cache,
                layer_caches.as_mut_slice(),
                &mut step_plan,
                0,
                loaded_model.split_layer,
                None,
                None,
                None,
                false,
                MetalLogitsOutputMode::RawLogits,
            )?;
            let remote_step = backend.remote_suffix_step(
                loaded_model.descriptor().model.model_id.as_str(),
                *token,
                cache.len(),
                prefix_step.hidden.as_slice(),
                prefix_step.key.as_slice(),
                prefix_step.value.as_slice(),
                loaded_model.split_layer,
            )?;
            cache.append(*token, prefix_step.key, prefix_step.value)?;
            kernel_count = kernel_count
                .saturating_add(prefix_step.kernel_count)
                .saturating_add(remote_step.kernel_count);
            bytes_moved = bytes_moved
                .saturating_add(prefix_step.bytes_moved)
                .saturating_add(remote_step.bytes_moved);
            last_logits = remote_step.logits;
            prompt_logits.push(last_logits.clone());
        }

        let prefill_handoff_state = gemma4_metal_layer_cache_state(layer_caches.as_slice());
        let prompt_cache = (shared_prefix_eligible && prefix_tokens_reused != prompt_tokens.len())
            .then(|| cache.clone());
        let mut sampler = super::GenerationSampler::new(&request.options)?;
        let structured_output_report = sampler.structured_output_report();
        let mut generated_tokens = Vec::new();
        let mut first_token_emitted_at = None;
        let mut last_token_emitted_at = None;
        let termination = loop {
            if generated_tokens.len() >= request.options.max_output_tokens {
                break super::TerminationReason::MaxOutputTokens;
            }
            if cache.len() >= cache.max_context() {
                break super::TerminationReason::ContextLimit;
            }

            let next_token = match sampler.select_next_token(
                loaded_model.tokenizer(),
                &last_logits,
                &cache,
                generated_tokens.as_slice(),
            )? {
                super::GenerationSelection::Token(token) => token,
                super::GenerationSelection::Terminate => {
                    break super::TerminationReason::EndOfSequence;
                }
            };
            if loaded_model.is_end_of_sequence(next_token) {
                break super::TerminationReason::EndOfSequence;
            }

            generated_tokens.push(next_token);
            let prefix_step = loaded_model.inner.forward_step_with_layer_caches(
                &mut backend.metal,
                next_token,
                cache.len(),
                &cache,
                layer_caches.as_mut_slice(),
                &mut step_plan,
                0,
                loaded_model.split_layer,
                None,
                None,
                None,
                false,
                MetalLogitsOutputMode::RawLogits,
            )?;
            let remote_step = backend.remote_suffix_step(
                loaded_model.descriptor().model.model_id.as_str(),
                next_token,
                cache.len(),
                prefix_step.hidden.as_slice(),
                prefix_step.key.as_slice(),
                prefix_step.value.as_slice(),
                loaded_model.split_layer,
            )?;
            cache.append(next_token, prefix_step.key, prefix_step.value)?;
            kernel_count = kernel_count
                .saturating_add(prefix_step.kernel_count)
                .saturating_add(remote_step.kernel_count);
            bytes_moved = bytes_moved
                .saturating_add(prefix_step.bytes_moved)
                .saturating_add(remote_step.bytes_moved);
            last_logits = remote_step.logits;
            let emitted_at = Instant::now();
            if first_token_emitted_at.is_none() {
                first_token_emitted_at = Some(emitted_at);
            }
            last_token_emitted_at = Some(emitted_at);

            if super::truncate_generated_text(
                loaded_model.tokenizer(),
                &mut generated_tokens,
                &request.options.stop_sequences,
            )
            .is_some()
            {
                break super::TerminationReason::EndOfSequence;
            }
        };

        if shared_prefix_eligible && prefix_tokens_reused != prompt_tokens.len() {
            if let Some(prompt_cache) = prompt_cache.as_ref() {
                let recorded_identity = shared_prefixes.record(
                    compatibility,
                    &prompt_tokens,
                    &prompt_logits,
                    prompt_cache,
                );
                if prefix_state != crate::PrefixCacheState::Hit || prefix_identity.is_none() {
                    prefix_identity = Some(recorded_identity);
                }
            }
        }

        let prompt_eval_duration_ns = super::elapsed_ns(prompt_eval_start);
        let generated = TokenSequence::new(generated_tokens);
        if let Some(session_id) = &request.session_id {
            session_tokens.extend_from_slice(prompt_tokens.as_slice());
            session_tokens.extend_from_slice(generated.as_slice());
            sessions.replace_cache(
                session_id,
                loaded_model.descriptor(),
                effective_served_artifact_digest.as_str(),
                cache.clone(),
                TokenSequence::new(session_tokens),
            )?;
        }
        let text = loaded_model.tokenizer().decode(generated.as_slice());
        let usage = super::GenerationUsage {
            input_tokens: prompt_tokens.len(),
            output_tokens: generated.len(),
            cache_tokens: cache.len(),
        };
        let kv_cache =
            psionic_runtime::KvCacheAccounting::from_states(&previous_kv_state, cache.state());
        let total_duration_ns = super::elapsed_ns(generation_start);
        let time_to_first_token_ns = first_token_emitted_at
            .map(|first_token_at| first_token_at.duration_since(generation_start))
            .and_then(|duration| duration.as_nanos().try_into().ok());
        let inter_token_latency_ns = super::average_inter_token_latency_ns(
            first_token_emitted_at,
            last_token_emitted_at,
            usage.output_tokens,
        );
        let prefill_decode_handoff = super::local_prefill_decode_handoff(&prefill_handoff_state);
        let kv_residency = super::host_only_kv_residency(cache.policy(), cache.state());
        let metrics = crate::GenerationMetrics {
            total_duration_ns: Some(total_duration_ns),
            load_duration_ns: Some(match load_state {
                super::GenerationLoadState::Cold => loaded_model.load_duration_ns(),
                super::GenerationLoadState::Warm => 0,
            }),
            prompt_eval_count: Some(usage.input_tokens),
            prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
            context_window: Some(context_window),
            eval_count: Some(usage.output_tokens),
            eval_duration_ns: Some(total_duration_ns.saturating_sub(prompt_eval_duration_ns)),
            time_to_first_token_ns,
            inter_token_latency_ns,
            kv_cache: Some(kv_cache),
            kv_residency: kv_residency.clone(),
            kv_cache_encoding: Some(psionic_runtime::KvCacheEncodingAccounting::active(
                kv_cache_encoding_policy.clone(),
            )),
            prefix_tokens_reused: Some(prefix_tokens_reused),
            termination_detail: None,
            qwen35_cuda_decode: None,
            gpt_oss_perf: None,
        };
        let delivery_plan_digest = loaded_model.plan_digest().to_string();
        let provenance = crate::GenerationProvenance {
            served_artifact,
            adapter_serving: request.adapter_serving.clone(),
            served_revision: None,
            execution_plan_digest: delivery_plan_digest.clone(),
            cluster_execution: None,
            load_state,
            isolation_policy: crate::LocalServingIsolationPolicy::in_process_runtime(),
            streaming_policy: None,
            memory_plan,
            residency_policy,
            residency_snapshot,
            kv_cache_policy: Some(cache.policy().clone()),
            kv_cache_encoding_policy: Some(kv_cache_encoding_policy),
            kv_ownership: cache.ownership_since(&request_kv_checkpoint),
            prefix_cache_control: Some(request.prefix_cache_control.clone()),
            prefix_cache_state: Some(prefix_state),
            prefix_cache_refusal_reason,
            prefix_cache_policy: Some(prefix_policy),
            prefix_cache_identity: prefix_identity,
            compile_path: None,
            delivery_proof: crate::build_delivery_proof(
                delivery_plan_digest,
                kernel_count,
                bytes_moved,
                0,
                0,
                metrics.kv_cache.as_ref().map(|value| value.growth.clone()),
                prefill_decode_handoff,
                kv_residency,
            ),
            cache_observations: crate::generation_cache_observations(
                loaded_model.descriptor(),
                None,
                load_state,
                request.session_id.as_ref(),
                request.reset_session,
                &previous_kv_state,
                prefix_state,
                prefix_cache_invalidation_trigger,
            ),
            scheduler: None,
            structured_output: structured_output_report,
            psion_served_evidence: None,
            psion_served_output_claim_posture: None,
        };
        let structured_output_value = sampler.structured_output_value(text.as_str())?;
        let response = GenerationResponse::new(
            request,
            request.session_id.clone(),
            generated,
            text,
            usage.input_tokens,
            usage.cache_tokens,
            termination,
        )
        .with_metrics_and_provenance(metrics, provenance);
        Ok(if let Some(value) = structured_output_value {
            response.with_structured_output_value(value)
        } else {
            response
        })
    })();

    let _ = models.finish_request(model_id, super::current_time_millis());
    result
}

fn build_gemma4_cuda_layer_caches_from_host_cache(
    backend: &mut CudaBackend,
    model: &CudaGemma4ModelInner,
    cache: &crate::InMemoryKvCache,
    reserve_tokens: usize,
) -> Result<Vec<Option<CudaGemma4LayerCacheState>>, ReferenceTextGenerationError> {
    if cache.width() != model.cache_width() {
        return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
            expected_kv_width: model.cache_width(),
            kv_width: cache.width(),
        });
    }

    model
        .layers
        .iter()
        .map(|layer| {
            let Some(cache_offset) = layer.attention_geometry.cache_write_offset else {
                return Ok(None);
            };
            let layer_kv_width = layer.attention_geometry.kv_width();
            let mut state = CudaGemma4LayerCacheState::new(
                backend,
                layer_kv_width,
                cache.len().saturating_add(reserve_tokens),
            )?;
            if cache.len() > 0 {
                let mut keys = Vec::with_capacity(cache.len().saturating_mul(layer_kv_width));
                let mut values = Vec::with_capacity(cache.len().saturating_mul(layer_kv_width));
                for entry in cache.entries() {
                    keys.extend_from_slice(&entry.key[cache_offset..cache_offset + layer_kv_width]);
                    values.extend_from_slice(
                        &entry.value[cache_offset..cache_offset + layer_kv_width],
                    );
                }
                state
                    .key_cache
                    .write_f32(keys.as_slice())
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                state
                    .value_cache
                    .write_f32(values.as_slice())
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                state.len = cache.len();
            }
            Ok(Some(state))
        })
        .collect()
}

fn gemma4_cuda_layer_cache_state(
    layer_caches: &[Option<CudaGemma4LayerCacheState>],
) -> psionic_runtime::KvCacheState {
    layer_caches
        .iter()
        .flatten()
        .next()
        .map(CudaGemma4LayerCacheState::state)
        .unwrap_or_default()
}

fn run_cuda_gemma4_generation_request(
    backend: &mut CudaBackend,
    models: &mut InMemoryGenerationModelRegistry<CudaGemma4GenerationModel>,
    sessions: &mut InMemoryGenerationSessionStore,
    shared_prefixes: &mut SharedPrefixStore,
    request: &GenerationRequest,
) -> Result<GenerationResponse, ReferenceTextGenerationError> {
    if !super::generation_product_supported(request) {
        return Err(ReferenceTextGenerationError::UnsupportedProduct(
            request.product_id.clone(),
        ));
    }

    let loaded_model = models
        .active(request.model.model.model_id.as_str())
        .ok_or_else(|| {
            ReferenceTextGenerationError::UnsupportedModel(request.model.model.model_id.clone())
        })?
        .clone();
    if loaded_model.descriptor() != &request.model {
        return Err(ReferenceTextGenerationError::UnsupportedModel(
            request.model.model.model_id.clone(),
        ));
    }

    let model_id = request.model.model.model_id.as_str();
    let load_state = models
        .load_state(model_id)
        .unwrap_or(super::GenerationLoadState::Warm);
    let request_start = super::current_time_millis();
    let generation_start = Instant::now();
    models.begin_request(model_id, request_start)?;
    let memory_plan = models.memory_plan(model_id).cloned();
    let residency_policy = Some(models.residency_policy().clone());
    let residency_snapshot = Some(models.memory_snapshot());
    let served_artifact = super::served_artifact_identity_for_decoder_backend(
        loaded_model.descriptor(),
        loaded_model.backend_compatibility(),
        &[],
    );
    let effective_served_artifact_digest = super::effective_generation_served_artifact_digest(
        &served_artifact,
        request.adapter_serving.as_ref(),
    );

    let result = (|| -> Result<GenerationResponse, ReferenceTextGenerationError> {
        let prompt_eval_start = Instant::now();
        let tokenizer = loaded_model.tokenizer();
        let prompt_tokens = loaded_model.encode_prompt_input(&request.prompt)?;
        if prompt_tokens.is_empty() {
            return Err(ReferenceTextGenerationError::EmptyPrompt);
        }

        let expected_kv_width = loaded_model.cache_width();
        let mut session_tokens = Vec::new();
        let compatibility = super::prefix_compatibility_for_request(&loaded_model, request);
        let prefix_policy = super::default_prefix_cache_policy();
        let mut prefix_state = crate::PrefixCacheState::None;
        let mut prefix_cache_refusal_reason = None;
        let mut prefix_cache_invalidation_trigger = None;
        let mut prefix_tokens_reused = 0usize;
        let mut prefix_identity = None;
        let mut shared_prefix_eligible = false;
        let previous_kv_state = if let Some(session_id) = &request.session_id {
            if request.reset_session {
                sessions.reset(session_id)?;
            }
            let state = sessions.state(session_id)?;
            super::validate_session_model(
                state,
                session_id,
                loaded_model.descriptor(),
                effective_served_artifact_digest.as_str(),
            )?;
            session_tokens = state.tokens().to_vec();
            if state.cache().is_empty() {
                shared_prefix_eligible = true;
            } else {
                prefix_state = crate::PrefixCacheState::Bypassed;
                prefix_cache_refusal_reason =
                    Some(crate::PrefixCacheRefusalReason::SessionBoundState);
            }
            state.cache().state()
        } else {
            shared_prefix_eligible = true;
            psionic_runtime::KvCacheState::default()
        };
        let preserve_prefix_tokens = usize::from(
            prompt_tokens.as_slice().first().copied() == Some(tokenizer.vocabulary().bos_id()),
        );
        let (prompt_tokens, context_window) = super::apply_context_window(
            &prompt_tokens,
            loaded_model.descriptor().config.max_context,
            previous_kv_state.tokens,
            request.options.max_output_tokens,
            request.options.context_overflow_policy,
            preserve_prefix_tokens,
        )?;
        let mut prompt_logits = Vec::new();
        let mut last_logits = Vec::new();
        let mut cache = if shared_prefix_eligible {
            let lookup = super::controlled_prefix_lookup(
                shared_prefixes,
                &compatibility,
                &prompt_tokens,
                request,
            );
            prefix_state = lookup.state;
            prefix_cache_refusal_reason = lookup.refusal_reason;
            prefix_cache_invalidation_trigger = lookup.invalidation_trigger;
            prefix_tokens_reused = lookup.reused_tokens;
            prefix_identity = lookup.identity;
            prompt_logits = lookup.prompt_logits;
            last_logits = if lookup.last_logits.is_empty() {
                prompt_logits.last().cloned().unwrap_or_default()
            } else {
                lookup.last_logits
            };
            lookup.cache.unwrap_or_else(|| {
                crate::InMemoryKvCache::with_policy(
                    loaded_model.descriptor().config.max_context,
                    expected_kv_width,
                    super::default_generation_kv_cache_policy(&loaded_model),
                )
            })
        } else if let Some(session_id) = &request.session_id {
            sessions.state(session_id)?.cache().clone()
        } else {
            crate::InMemoryKvCache::with_policy(
                loaded_model.descriptor().config.max_context,
                expected_kv_width,
                super::default_generation_kv_cache_policy(&loaded_model),
            )
        };
        if cache.width() != expected_kv_width {
            return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
                expected_kv_width,
                kv_width: cache.width(),
            });
        }
        cache.bind_owner(super::request_kv_owner(
            request,
            psionic_runtime::BatchExecutionPosture::SingleRequestOnly,
            None,
        ));
        let request_kv_checkpoint = cache.checkpoint();
        let mut layer_caches = build_gemma4_cuda_layer_caches_from_host_cache(
            backend,
            &loaded_model.inner,
            &cache,
            request.options.max_output_tokens.saturating_add(1),
        )?;
        let mut step_plan = loaded_model.inner.build_decode_step_plan(backend)?;
        let mut kernel_count = 0usize;
        let mut bytes_moved = 0u64;

        for token in &prompt_tokens.as_slice()[prefix_tokens_reused..] {
            let step = loaded_model.inner.forward_step_with_layer_caches(
                backend,
                *token,
                cache.len(),
                &cache,
                layer_caches.as_mut_slice(),
                &mut step_plan,
                0,
                loaded_model.inner.layers.len(),
                None,
                None,
                None,
                true,
            )?;
            let mut step_output = GenerationStepOutput {
                key: step.key,
                value: step.value,
                logits: step.logits,
                hidden: step.lm_head_hidden.clone(),
                execution_plan_digest: Some(loaded_model.inner.plan_digest.clone()),
                compile_path: None,
                kernel_count: step.kernel_count,
                bytes_moved: step.bytes_moved,
                plan_cache_hits: 0,
                plan_cache_misses: 0,
                gpt_oss_perf: None,
            };
            loaded_model.adjust_step_output(&mut step_output, request)?;
            kernel_count = kernel_count.saturating_add(step_output.kernel_count);
            bytes_moved = bytes_moved.saturating_add(step_output.bytes_moved);
            cache.append(*token, step_output.key, step_output.value)?;
            last_logits = step_output.logits;
            prompt_logits.push(last_logits.clone());
        }

        let prefill_handoff_state = gemma4_cuda_layer_cache_state(layer_caches.as_slice());
        let prompt_cache = (shared_prefix_eligible && prefix_tokens_reused != prompt_tokens.len())
            .then(|| cache.clone());
        let mut sampler = super::GenerationSampler::new(&request.options)?;
        let structured_output_report = sampler.structured_output_report();
        let mut generated_tokens = Vec::new();
        let mut first_token_emitted_at = None;
        let mut last_token_emitted_at = None;
        let termination = loop {
            if generated_tokens.len() >= request.options.max_output_tokens {
                break super::TerminationReason::MaxOutputTokens;
            }
            if cache.len() >= cache.max_context() {
                break super::TerminationReason::ContextLimit;
            }

            let next_token = match sampler.select_next_token(
                loaded_model.tokenizer(),
                &last_logits,
                &cache,
                generated_tokens.as_slice(),
            )? {
                super::GenerationSelection::Token(token) => token,
                super::GenerationSelection::Terminate => {
                    break super::TerminationReason::EndOfSequence;
                }
            };
            if loaded_model.is_end_of_sequence(next_token) {
                break super::TerminationReason::EndOfSequence;
            }

            generated_tokens.push(next_token);
            let step = loaded_model.inner.forward_step_with_layer_caches(
                backend,
                next_token,
                cache.len(),
                &cache,
                layer_caches.as_mut_slice(),
                &mut step_plan,
                0,
                loaded_model.inner.layers.len(),
                None,
                None,
                None,
                true,
            )?;
            let mut step_output = GenerationStepOutput {
                key: step.key,
                value: step.value,
                logits: step.logits,
                hidden: step.lm_head_hidden.clone(),
                execution_plan_digest: Some(loaded_model.inner.plan_digest.clone()),
                compile_path: None,
                kernel_count: step.kernel_count,
                bytes_moved: step.bytes_moved,
                plan_cache_hits: 0,
                plan_cache_misses: 0,
                gpt_oss_perf: None,
            };
            loaded_model.adjust_step_output(&mut step_output, request)?;
            kernel_count = kernel_count.saturating_add(step_output.kernel_count);
            bytes_moved = bytes_moved.saturating_add(step_output.bytes_moved);
            cache.append(next_token, step_output.key, step_output.value)?;
            last_logits = step_output.logits;
            let emitted_at = Instant::now();
            if first_token_emitted_at.is_none() {
                first_token_emitted_at = Some(emitted_at);
            }
            last_token_emitted_at = Some(emitted_at);

            if super::truncate_generated_text(
                loaded_model.tokenizer(),
                &mut generated_tokens,
                &request.options.stop_sequences,
            )
            .is_some()
            {
                break super::TerminationReason::EndOfSequence;
            }
        };

        if shared_prefix_eligible && prefix_tokens_reused != prompt_tokens.len() {
            if let Some(prompt_cache) = prompt_cache.as_ref() {
                let recorded_identity = shared_prefixes.record(
                    compatibility,
                    &prompt_tokens,
                    &prompt_logits,
                    prompt_cache,
                );
                if prefix_state != crate::PrefixCacheState::Hit || prefix_identity.is_none() {
                    prefix_identity = Some(recorded_identity);
                }
            }
        }

        let prompt_eval_duration_ns = super::elapsed_ns(prompt_eval_start);
        let generated = TokenSequence::new(generated_tokens);
        if let Some(session_id) = &request.session_id {
            session_tokens.extend_from_slice(prompt_tokens.as_slice());
            session_tokens.extend_from_slice(generated.as_slice());
            sessions.replace_cache(
                session_id,
                loaded_model.descriptor(),
                effective_served_artifact_digest.as_str(),
                cache.clone(),
                TokenSequence::new(session_tokens),
            )?;
        }
        let text = loaded_model.tokenizer().decode(generated.as_slice());
        let usage = super::GenerationUsage {
            input_tokens: prompt_tokens.len(),
            output_tokens: generated.len(),
            cache_tokens: cache.len(),
        };
        let current_host_state = cache.state();
        let device_state = gemma4_cuda_layer_cache_state(layer_caches.as_slice());
        let kv_cache = psionic_runtime::KvCacheAccounting::from_states(
            &previous_kv_state,
            current_host_state.clone(),
        );
        let total_duration_ns = super::elapsed_ns(generation_start);
        let time_to_first_token_ns = first_token_emitted_at
            .map(|first_token_at| first_token_at.duration_since(generation_start))
            .and_then(|duration| duration.as_nanos().try_into().ok());
        let inter_token_latency_ns = super::average_inter_token_latency_ns(
            first_token_emitted_at,
            last_token_emitted_at,
            usage.output_tokens,
        );
        let prefill_decode_handoff = super::local_prefill_decode_handoff(&prefill_handoff_state);
        let write_back_growth = psionic_runtime::KvCacheGrowth {
            tokens: current_host_state
                .tokens
                .saturating_sub(previous_kv_state.tokens),
            bytes: current_host_state
                .bytes
                .saturating_sub(previous_kv_state.bytes),
            pages: current_host_state
                .pages
                .saturating_sub(previous_kv_state.pages),
        };
        let kv_residency = super::host_device_kv_residency(
            cache.policy(),
            current_host_state,
            device_state,
            previous_kv_state.tokens > 0,
            Some(write_back_growth),
        );
        let metrics = crate::GenerationMetrics {
            total_duration_ns: Some(total_duration_ns),
            load_duration_ns: Some(match load_state {
                super::GenerationLoadState::Cold => loaded_model.load_duration_ns(),
                super::GenerationLoadState::Warm => 0,
            }),
            prompt_eval_count: Some(usage.input_tokens),
            prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
            context_window: Some(context_window),
            eval_count: Some(usage.output_tokens),
            eval_duration_ns: Some(total_duration_ns.saturating_sub(prompt_eval_duration_ns)),
            time_to_first_token_ns,
            inter_token_latency_ns,
            kv_cache: Some(kv_cache),
            kv_residency: kv_residency.clone(),
            kv_cache_encoding: None,
            prefix_tokens_reused: Some(prefix_tokens_reused),
            termination_detail: None,
            qwen35_cuda_decode: None,
            gpt_oss_perf: None,
        };
        let delivery_plan_digest = loaded_model.plan_digest().to_string();
        let provenance = crate::GenerationProvenance {
            served_artifact,
            adapter_serving: request.adapter_serving.clone(),
            served_revision: None,
            execution_plan_digest: delivery_plan_digest.clone(),
            cluster_execution: None,
            load_state,
            isolation_policy: crate::LocalServingIsolationPolicy::in_process_runtime(),
            streaming_policy: None,
            memory_plan,
            residency_policy,
            residency_snapshot,
            kv_cache_policy: Some(cache.policy().clone()),
            kv_cache_encoding_policy: None,
            kv_ownership: cache.ownership_since(&request_kv_checkpoint),
            prefix_cache_control: Some(request.prefix_cache_control.clone()),
            prefix_cache_state: Some(prefix_state),
            prefix_cache_refusal_reason,
            prefix_cache_policy: Some(prefix_policy),
            prefix_cache_identity: prefix_identity,
            compile_path: None,
            delivery_proof: crate::build_delivery_proof(
                delivery_plan_digest,
                kernel_count,
                bytes_moved,
                0,
                0,
                metrics.kv_cache.as_ref().map(|value| value.growth.clone()),
                prefill_decode_handoff,
                kv_residency.clone(),
            ),
            cache_observations: crate::generation_cache_observations(
                loaded_model.descriptor(),
                None,
                load_state,
                request.session_id.as_ref(),
                request.reset_session,
                &previous_kv_state,
                prefix_state,
                prefix_cache_invalidation_trigger,
            ),
            scheduler: None,
            structured_output: structured_output_report,
            psion_served_evidence: None,
            psion_served_output_claim_posture: None,
        };
        let structured_output_value = sampler.structured_output_value(text.as_str())?;
        let response = GenerationResponse::new(
            request,
            request.session_id.clone(),
            generated,
            text,
            usage.input_tokens,
            usage.cache_tokens,
            termination,
        )
        .with_metrics_and_provenance(metrics, provenance);
        Ok(if let Some(value) = structured_output_value {
            response.with_structured_output_value(value)
        } else {
            response
        })
    })();

    let finish_at = super::current_time_millis();
    let _ = models.finish_request(model_id, finish_at);
    result
}

pub struct CudaGemma4TextGenerationService {
    backend: CudaBackend,
    models: InMemoryGenerationModelRegistry<CudaGemma4GenerationModel>,
    sessions: InMemoryGenerationSessionStore,
    shared_prefixes: SharedPrefixStore,
    backend_health: super::BackendHealthTracker,
    model_descriptor: DecoderModelDescriptor,
    runtime_support: GgufDecoderRuntimeSupport,
    adapters: Arc<Mutex<DenseAdapterRuntimeStore>>,
    promoted_revisions: PromotedGemmaRevisionState,
    distributed_requests: BTreeMap<String, DistributedCudaGemma4RequestState>,
}

impl CudaGemma4TextGenerationService {
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, ReferenceTextGenerationError> {
        let mut backend = CudaBackend::new();
        let adapters = Arc::new(Mutex::new(DenseAdapterRuntimeStore::default()));
        let model =
            CudaGemma4GenerationModel::from_gguf_path(path, &mut backend, Arc::clone(&adapters))?;
        let model_descriptor = model.descriptor().clone();
        let runtime_support = model.runtime_support();
        let mut models = InMemoryGenerationModelRegistry::new();
        let now_millis = super::current_time_millis();
        models.warm_with_metadata(
            model,
            now_millis,
            super::DEFAULT_MODEL_KEEPALIVE_MILLIS,
            None,
            Some(String::from("cuda")),
            None,
        )?;
        let mut backend_health = super::BackendHealthTracker::default();
        backend_health.observe("cuda", backend.health(), now_millis);
        Ok(Self {
            backend,
            models,
            sessions: InMemoryGenerationSessionStore::new(),
            shared_prefixes: SharedPrefixStore::default(),
            backend_health,
            model_descriptor,
            runtime_support,
            adapters,
            promoted_revisions: PromotedGemmaRevisionState::default(),
            distributed_requests: BTreeMap::new(),
        })
    }

    #[must_use]
    pub fn model_descriptor(&self) -> &DecoderModelDescriptor {
        &self.model_descriptor
    }

    #[must_use]
    pub fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        self.runtime_support.clone()
    }

    #[must_use]
    pub fn loaded_model_views(&mut self) -> Vec<LoadedModelView> {
        self.loaded_model_views_at(super::current_time_millis())
    }

    #[must_use]
    pub fn loaded_models(&mut self) -> LoadedModelsObservation {
        self.loaded_models_at(super::current_time_millis())
    }

    #[must_use]
    pub fn observability(&mut self) -> LocalRuntimeObservability {
        self.observability_at(super::current_time_millis())
    }

    pub fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Ok(self
            .models
            .warm_loaded(model_id, super::current_time_millis(), keep_alive_millis)?)
    }

    pub fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Ok(self
            .models
            .unload_view(model_id, super::current_time_millis())?)
    }

    pub fn create_session(
        &mut self,
        model_id: &str,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        let model = self
            .models
            .active(model_id)
            .ok_or_else(|| ReferenceTextGenerationError::UnsupportedModel(model_id.to_string()))?;
        let served_artifact =
            super::served_artifact_identity_for_decoder_backend(model.descriptor(), "cuda", &[]);
        let effective_served_artifact_digest = self
            .promoted_revisions
            .active_binding()
            .map(|binding| binding.served_adapter_digest)
            .unwrap_or(served_artifact.served_artifact_digest);
        Ok(self
            .sessions
            .create(model, effective_served_artifact_digest))
    }

    pub fn reset_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        Ok(self.sessions.reset(session_id)?)
    }

    pub fn close_session(
        &mut self,
        session_id: &SessionId,
    ) -> Result<crate::GenerationSession, ReferenceTextGenerationError> {
        Ok(self.sessions.close(session_id)?)
    }

    pub fn generate_continuous_batch(
        &mut self,
        requests: Vec<GenerationRequest>,
    ) -> ContinuousBatchGenerationResult {
        let requests = requests
            .into_iter()
            .map(|request| self.request_with_active_revision(request))
            .collect();
        let mut result = super::run_continuous_batch_generation_requests(
            &mut self.backend,
            &mut self.models,
            &mut self.sessions,
            &mut self.shared_prefixes,
            requests,
            default_generation_scheduler_policy(),
        );
        result.responses = result
            .responses
            .into_iter()
            .map(|response| {
                response.map(|response| self.promoted_revisions.attach_to_response(response))
            })
            .collect();
        result
    }

    #[must_use]
    fn loaded_models_at(&mut self, now_millis: u64) -> LoadedModelsObservation {
        self.models.expire_idle(now_millis);
        self.models.loaded_models_observation()
    }

    #[must_use]
    fn observability_at(&mut self, now_millis: u64) -> LocalRuntimeObservability {
        self.models.expire_idle(now_millis);
        self.backend_health
            .observe("cuda", self.backend.health(), now_millis);
        super::generation_runtime_observability(
            &self.models,
            &self.sessions,
            &self.backend_health,
            super::default_text_generation_execution_profile(),
        )
    }

    #[must_use]
    fn loaded_model_views_at(&mut self, now_millis: u64) -> Vec<LoadedModelView> {
        self.models.expire_idle(now_millis);
        self.models.loaded_model_views()
    }

    pub fn active_revision_identity(&self) -> Option<ServedModelRevisionIdentity> {
        self.promoted_revisions.active_identity()
    }

    pub fn last_known_good_revision_identity(&self) -> Option<ServedModelRevisionIdentity> {
        self.promoted_revisions.last_known_good_identity()
    }

    pub fn promote_exported_revision(
        &mut self,
        base_binding: &GemmaE4bServedBaseModelBinding,
        exported_artifact: &GemmaE4bCudaAdapterExportedArtifact,
        checkpoint: &GemmaE4bCudaAdapterCheckpoint,
    ) -> Result<ServedModelRevisionIdentity, ReferenceTextGenerationError> {
        let served_artifact = crate::served_artifact_identity_for_decoder_backend(
            &self.model_descriptor,
            "cuda",
            &[],
        );
        let binding = self.register_exported_revision(
            base_binding,
            &served_artifact,
            exported_artifact,
            checkpoint,
        )?;
        let identity = gemma_served_revision_identity(
            &binding,
            exported_artifact,
            checkpoint,
            crate::current_time_millis(),
        );
        Ok(self
            .promoted_revisions
            .promote(PromotedGemmaRevisionEntry { binding, identity }))
    }

    pub fn rollback_to_last_known_good_revision(
        &mut self,
    ) -> Result<Option<ServedModelRevisionIdentity>, ReferenceTextGenerationError> {
        Ok(self.promoted_revisions.rollback())
    }

    fn request_with_active_revision(&self, mut request: GenerationRequest) -> GenerationRequest {
        if request.adapter_serving.is_none() {
            request.adapter_serving = self.promoted_revisions.active_binding();
        }
        request
    }

    fn register_exported_revision(
        &mut self,
        base_binding: &GemmaE4bServedBaseModelBinding,
        served_artifact: &psionic_runtime::ServedArtifactIdentity,
        exported_artifact: &GemmaE4bCudaAdapterExportedArtifact,
        checkpoint: &GemmaE4bCudaAdapterCheckpoint,
    ) -> Result<AdapterServingBinding, ReferenceTextGenerationError> {
        let adapter = validate_gemma_exported_revision(
            &self.model_descriptor,
            base_binding,
            served_artifact,
            exported_artifact,
            checkpoint,
        )?;
        let binding = AdapterServingBinding::new(
            format!(
                "gemma4-promoted:{}:{}",
                checkpoint.checkpoint_id, exported_artifact.adapter_identity.adapter_revision
            ),
            base_binding.model_id.clone(),
            base_binding.base_model_revision.clone(),
            base_binding.base_served_artifact_digest.clone(),
            AdapterResidencyMode::HotSwapOverlay,
            vec![exported_artifact.adapter_identity.clone()],
        );
        self.adapters
            .lock()
            .map_err(
                |_| ReferenceTextGenerationError::UnsupportedAdapterBinding {
                    binding_id: binding.binding_id.clone(),
                    reason: String::from("adapter registry is poisoned"),
                },
            )?
            .insert(DenseAdapterRuntime::new(binding.clone(), adapter)?);
        Ok(binding)
    }

    pub fn distributed_worker_step(
        &mut self,
        request: &DistributedGemma4RemoteStepRequest,
    ) -> Result<DistributedGemma4RemoteStepResponse, ReferenceTextGenerationError> {
        if request.model_id != self.model_descriptor.model.model_id {
            return Err(ReferenceTextGenerationError::UnsupportedModel(
                request.model_id.clone(),
            ));
        }
        let model = self
            .models
            .active(request.model_id.as_str())
            .ok_or_else(|| {
                ReferenceTextGenerationError::UnsupportedModel(request.model_id.clone())
            })?
            .clone();
        if !self
            .distributed_requests
            .contains_key(request.request_id.as_str())
        {
            let cache = crate::InMemoryKvCache::with_policy(
                model.descriptor().config.max_context,
                model.cache_width(),
                super::default_generation_kv_cache_policy(&model),
            );
            let layer_caches = build_gemma4_cuda_layer_caches_from_host_cache(
                &mut self.backend,
                &model.inner,
                &cache,
                1,
            )?;
            let step_plan = model.inner.build_decode_step_plan(&mut self.backend)?;
            self.distributed_requests.insert(
                request.request_id.clone(),
                DistributedCudaGemma4RequestState {
                    cache,
                    layer_caches,
                    step_plan,
                },
            );
        }
        let state = self
            .distributed_requests
            .get_mut(request.request_id.as_str())
            .expect("distributed gemma4 request state");
        if state.cache.width() != model.cache_width() {
            return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
                expected_kv_width: model.cache_width(),
                kv_width: state.cache.width(),
            });
        }
        if state.cache.len() != request.position {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "distributed gemma4 remote cache position mismatch: expected {}, actual {}",
                    request.position,
                    state.cache.len()
                )),
            ));
        }
        let step = model.inner.forward_step_with_layer_caches(
            &mut self.backend,
            TokenId(request.token),
            request.position,
            &state.cache,
            state.layer_caches.as_mut_slice(),
            &mut state.step_plan,
            request.split_layer,
            model.inner.layers.len(),
            Some(request.input_hidden.as_slice()),
            Some(request.forwarded_key.as_slice()),
            Some(request.forwarded_value.as_slice()),
            true,
        )?;
        state
            .cache
            .append(TokenId(request.token), step.key, step.value)?;
        Ok(DistributedGemma4RemoteStepResponse {
            logits: step.logits,
            kernel_count: step.kernel_count,
            bytes_moved: step.bytes_moved,
        })
    }

    pub fn distributed_worker_reset(&mut self, request_id: &str) {
        self.distributed_requests.remove(request_id);
    }
}

impl TextGenerationExecutor for CudaGemma4TextGenerationService {
    type Error = ReferenceTextGenerationError;

    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResponse, Self::Error> {
        let request = self.request_with_active_revision(request.clone());
        let response = run_cuda_gemma4_generation_request(
            &mut self.backend,
            &mut self.models,
            &mut self.sessions,
            &mut self.shared_prefixes,
            &request,
        )?;
        Ok(self.promoted_revisions.attach_to_response(response))
    }
}

impl StreamingTextGenerationExecutor for CudaGemma4TextGenerationService {
    type Stream<'a> = Box<dyn GenerationEventStream + 'a>;

    fn generate_stream<'a>(
        &'a mut self,
        request: &GenerationRequest,
    ) -> Result<Self::Stream<'a>, ReferenceTextGenerationError> {
        let response = self.generate(request)?;
        Ok(Box::new(CompletedGenerationStream::new(response)))
    }
}

impl ManagedTextGenerationRuntime for CudaGemma4TextGenerationService {
    fn loaded_models(&mut self) -> LoadedModelsObservation {
        Self::loaded_models(self)
    }

    fn observability(&mut self) -> LocalRuntimeObservability {
        Self::observability(self)
    }

    fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Self::warm_model(self, model_id, keep_alive_millis)
    }

    fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        Self::unload_model(self, model_id)
    }
}

#[derive(Clone, Debug)]
struct CudaGemma4GenerationModel {
    inner: Arc<CudaGemma4ModelInner>,
    adapters: Arc<Mutex<DenseAdapterRuntimeStore>>,
}

impl CudaGemma4GenerationModel {
    fn from_gguf_path(
        path: impl AsRef<Path>,
        backend: &mut CudaBackend,
        adapters: Arc<Mutex<DenseAdapterRuntimeStore>>,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let artifact = GgufBlobArtifact::open_path(path, gguf_local_blob_open_options())?;
        Self::from_blob_artifact(artifact, backend, adapters)
    }

    fn from_blob_artifact(
        artifact: GgufBlobArtifact,
        backend: &mut CudaBackend,
        adapters: Arc<Mutex<DenseAdapterRuntimeStore>>,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let load_start = Instant::now();
        let adapter = GgufDecoderAdapterLoader.load_blob_artifact(&artifact)?;
        if adapter.family_metadata().family != GgufDecoderFamily::Gemma4 {
            return Err(ModelLoadError::UnsupportedModel(
                adapter.descriptor().model.model_id.clone(),
            )
            .into());
        }
        let tokenizer = GgufRuntimeTokenizer::from_gguf(adapter.tokenizer()).map_err(|error| {
            ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!("failed to build runtime tokenizer: {error}"),
            }
        })?;
        let token_embedding =
            ProjectionMatrix::load(&artifact, adapter.tensor_layout().token_embedding.as_str())?;
        let output_name = adapter
            .tensor_layout()
            .output
            .as_deref()
            .unwrap_or(adapter.tensor_layout().token_embedding.as_str());
        let output = CudaQuantizedProjectionMatrix::load(backend, &artifact, output_name)?;
        let descriptor = adapter.descriptor().clone();
        let gemma4_per_layer_inputs = CudaGemma4PerLayerInputs::load(backend, &artifact)?;
        let mut cache_offset = 0usize;
        let mut last_swa_cache_offset = None;
        let mut last_full_cache_offset = None;
        let mut last_swa_kv_layer_index = None;
        let mut last_full_kv_layer_index = None;
        let mut layers = Vec::with_capacity(adapter.tensor_layout().layers.len());
        for (layer_index, layout) in adapter.tensor_layout().layers.iter().enumerate() {
            let query_weight_name = required_tensor_name(
                layout.attention_query_weight.as_deref(),
                "attention_query_weight",
            )?;
            let query_rows = artifact
                .paged_tensor(query_weight_name)?
                .metadata()
                .shape
                .dims()[0];
            let layer_head_dim = query_rows
                .checked_div(descriptor.config.block.attention.head_count)
                .unwrap_or(0);
            let is_swa =
                gemma4_layer_is_swa(adapter.family_metadata(), layer_index, layer_head_dim);
            let has_kv = gemma4_layer_has_kv(adapter.family_metadata(), layout);
            let (cache_read_offset, cache_write_offset, reuse_layer_index) = if has_kv {
                (cache_offset, Some(cache_offset), None)
            } else if is_swa {
                (
                    last_swa_cache_offset.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 swa layer {layer_index} reuses kv before any swa kv source was loaded"
                            ),
                        ))
                    })?,
                    None,
                    Some(last_swa_kv_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 swa layer {layer_index} is missing a reused kv source layer index"
                            ),
                        ))
                    })?),
                )
            } else {
                (
                    last_full_cache_offset.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 full-attention layer {layer_index} reuses kv before any full-attention kv source was loaded"
                            ),
                        ))
                    })?,
                    None,
                    Some(last_full_kv_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 full-attention layer {layer_index} is missing a reused kv source layer index"
                            ),
                        ))
                    })?),
                )
            };
            let layer = CudaGemma4Layer::load(
                backend,
                &artifact,
                layout,
                &descriptor,
                adapter.family_metadata(),
                layer_index,
                cache_read_offset,
                cache_write_offset,
                reuse_layer_index,
            )?;
            if layer.attention_geometry.has_kv() {
                if is_swa {
                    last_swa_cache_offset = layer.attention_geometry.cache_write_offset;
                    last_swa_kv_layer_index = Some(layer_index);
                } else {
                    last_full_cache_offset = layer.attention_geometry.cache_write_offset;
                    last_full_kv_layer_index = Some(layer_index);
                }
            }
            cache_offset = cache_offset.saturating_add(layer.cache_write_width());
            layers.push(layer);
        }
        let inner = CudaGemma4ModelInner {
            descriptor: descriptor.clone(),
            family_metadata: adapter.family_metadata().clone(),
            tokenizer,
            token_embedding,
            gemma4_per_layer_inputs,
            rope_freq_factors: load_named_optional_dense_vector(&artifact, "rope_freqs.weight")?,
            output_norm: load_dense_vector(
                &artifact,
                adapter.tensor_layout().output_norm.as_str(),
            )?,
            output,
            layers,
            plan_digest: digest_gemma4_cuda_plan(&descriptor, adapter.family_metadata()),
            load_duration_ns: load_start
                .elapsed()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
        };
        Ok(Self {
            inner: Arc::new(inner),
            adapters,
        })
    }

    #[must_use]
    fn plan_digest(&self) -> &str {
        self.inner.plan_digest.as_str()
    }

    #[must_use]
    fn runtime_support(&self) -> GgufDecoderRuntimeSupport {
        let mut support = runtime_support_for_descriptor(
            &self.inner.descriptor,
            GgufDecoderFamily::Gemma4,
            vec![String::from("cuda")],
            vec![String::from("cpu"), String::from("metal")],
            gemma_cuda_adapter_runtime_support(),
        );
        support.unsupported_features = vec![
            String::from("image_inputs"),
            String::from("video_inputs"),
            String::from("audio_inputs"),
            String::from("tool_calling"),
            String::from("response_state"),
        ];
        support
    }
}

impl crate::GenerationModelHandle for CudaGemma4GenerationModel {
    fn descriptor(&self) -> &DecoderModelDescriptor {
        &self.inner.descriptor
    }

    fn cache_width(&self) -> usize {
        self.inner.cache_width()
    }
}

impl super::CompiledWordGenerationModel for CudaGemma4GenerationModel {
    type Backend = CudaBackend;

    fn tokenizer(&self) -> &dyn TokenizerBoundary {
        &self.inner.tokenizer
    }

    fn encode_prompt_input(
        &self,
        input: &GenerationInput,
    ) -> Result<TokenSequence, ReferenceTextGenerationError> {
        Ok(match input {
            GenerationInput::Text(text) => self.inner.tokenizer.encode_with_defaults(text),
            GenerationInput::Tokens(tokens) => tokens.clone(),
        })
    }

    fn is_end_of_sequence(&self, token: TokenId) -> bool {
        self.inner.tokenizer.is_end_of_sequence(token)
    }

    fn execute_step(
        &self,
        backend: &mut Self::Backend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
    ) -> Result<GenerationStepOutput, ReferenceTextGenerationError> {
        let config = &self.inner.descriptor.config;
        if token.as_u32() as usize >= config.vocab_size {
            return Err(ReferenceTextGenerationError::InvalidToken {
                token: token.as_u32(),
                vocab_size: config.vocab_size,
            });
        }
        if position >= config.max_context {
            return Err(ReferenceTextGenerationError::InvalidPosition {
                position,
                max_context: config.max_context,
            });
        }
        if cache.width() != self.inner.cache_width() {
            return Err(ReferenceTextGenerationError::UnsupportedCacheGeometry {
                expected_kv_width: self.inner.cache_width(),
                kv_width: cache.width(),
            });
        }
        let step = self.inner.forward_step(backend, token, position, cache)?;
        Ok(GenerationStepOutput {
            key: step.key,
            value: step.value,
            logits: step.logits,
            hidden: Some(step.final_hidden),
            execution_plan_digest: Some(self.inner.plan_digest.clone()),
            compile_path: None,
            kernel_count: step.kernel_count,
            bytes_moved: step.bytes_moved,
            plan_cache_hits: 0,
            plan_cache_misses: 0,
            gpt_oss_perf: None,
        })
    }

    fn plan_digest(&self) -> &str {
        self.plan_digest()
    }

    fn load_duration_ns(&self) -> u64 {
        self.inner.load_duration_ns
    }

    fn backend_compatibility(&self) -> &'static str {
        "cuda"
    }

    fn adjust_step_output(
        &self,
        step: &mut GenerationStepOutput,
        request: &GenerationRequest,
    ) -> Result<(), ReferenceTextGenerationError> {
        let Some(binding) = request.adapter_serving.as_ref() else {
            return Ok(());
        };
        let hidden = step.hidden.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: binding.binding_id.clone(),
                reason: String::from(
                    "the active gemma4 cuda step does not expose the final hidden state needed for LM-head LoRA serving",
                ),
            }
        })?;
        let adapters = self.adapters.lock().map_err(|_| {
            ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: binding.binding_id.clone(),
                reason: String::from("adapter registry is poisoned"),
            }
        })?;
        let runtime = adapters.get(binding)?;
        runtime.apply_to_logits(hidden.as_slice(), step.logits.as_mut_slice())
    }
}

#[derive(Clone, Debug)]
struct CudaGemma4ModelInner {
    descriptor: DecoderModelDescriptor,
    family_metadata: GgufDecoderFamilyMetadata,
    tokenizer: GgufRuntimeTokenizer,
    token_embedding: ProjectionMatrix,
    gemma4_per_layer_inputs: Option<CudaGemma4PerLayerInputs>,
    rope_freq_factors: Option<Vec<f32>>,
    output_norm: Vec<f32>,
    output: CudaQuantizedProjectionMatrix,
    layers: Vec<CudaGemma4Layer>,
    plan_digest: String,
    load_duration_ns: u64,
}

impl CudaGemma4ModelInner {
    fn rope_freq_factors_for_layer(&self, layer_index: usize, head_dim: usize) -> Option<&[f32]> {
        (!gemma4_layer_is_swa(&self.family_metadata, layer_index, head_dim))
            .then_some(self.rope_freq_factors.as_deref())
            .flatten()
    }

    fn cache_width(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.attention_geometry.cache_end())
            .max()
            .unwrap_or(0)
    }

    fn forward_step(
        &self,
        backend: &mut CudaBackend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
    ) -> Result<CudaGemma4ForwardStep, ReferenceTextGenerationError> {
        let step = self.forward_stage_step(
            backend,
            token,
            position,
            cache,
            0,
            self.layers.len(),
            None,
            None,
            None,
            true,
        )?;
        let final_hidden = step.lm_head_hidden.ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "cuda gemma4 full forward step did not produce lm-head hidden",
            )))
        })?;
        Ok(CudaGemma4ForwardStep {
            key: step.key,
            value: step.value,
            logits: step.logits,
            final_hidden,
            kernel_count: step.kernel_count,
            bytes_moved: step.bytes_moved,
        })
    }

    fn build_decode_step_plan(
        &self,
        backend: &mut CudaBackend,
    ) -> Result<CudaGemma4StepPlan, ReferenceTextGenerationError> {
        let head_count = self.descriptor.config.block.attention.head_count;
        let head_dim = self.descriptor.config.block.attention.head_dim;
        let q_rows = head_count.saturating_mul(head_dim);
        let kv_rows = self
            .layers
            .iter()
            .filter(|layer| layer.attention_geometry.has_kv())
            .map(|layer| layer.attention_geometry.kv_width())
            .max()
            .unwrap_or(self.descriptor.config.kv_width());
        Ok(CudaGemma4StepPlan {
            qkv_buffer: backend
                .input_buffer(
                    Shape::new(vec![q_rows.saturating_add(kv_rows.saturating_mul(2))]),
                    vec![0.0; q_rows.saturating_add(kv_rows.saturating_mul(2))],
                )
                .map_err(ReferenceTextGenerationError::Runtime)?,
            query_buffer: backend
                .input_buffer(Shape::new(vec![q_rows]), vec![0.0; q_rows])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            key_buffer: backend
                .input_buffer(Shape::new(vec![kv_rows]), vec![0.0; kv_rows])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            value_buffer: backend
                .input_buffer(Shape::new(vec![kv_rows]), vec![0.0; kv_rows])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            attention_buffer: backend
                .input_buffer(Shape::new(vec![q_rows]), vec![0.0; q_rows])
                .map_err(ReferenceTextGenerationError::Runtime)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step_with_layer_caches(
        &self,
        backend: &mut CudaBackend,
        token: TokenId,
        position: usize,
        _cache: &crate::InMemoryKvCache,
        layer_caches: &mut [Option<CudaGemma4LayerCacheState>],
        step_plan: &mut CudaGemma4StepPlan,
        start_layer: usize,
        end_layer: usize,
        input_hidden: Option<&[f32]>,
        forwarded_key: Option<&[f32]>,
        forwarded_value: Option<&[f32]>,
        produce_logits: bool,
    ) -> Result<CudaGemma4StageStep, ReferenceTextGenerationError> {
        if start_layer > end_layer || end_layer > self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "invalid gemma4 cuda stage range [{start_layer}..{end_layer}) for {} layers",
                    self.layers.len()
                )),
            ));
        }
        if layer_caches.len() != self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 cuda layer cache count mismatch: expected {}, actual {}",
                    self.layers.len(),
                    layer_caches.len()
                )),
            ));
        }

        let mut bytes_moved = self.token_embedding.byte_length() as u64;
        let mut kernel_count = 1usize;
        let mut embedding_hidden = self
            .token_embedding
            .decode_row(token.as_u32() as usize)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        scale_in_place(
            &mut embedding_hidden,
            input_embedding_scale(&self.family_metadata, self.descriptor.config.hidden_size),
        );
        let mut hidden = if let Some(input_hidden) = input_hidden {
            if input_hidden.len() != self.descriptor.config.hidden_size {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "gemma4 cuda stage input hidden width mismatch: expected {}, actual {}",
                        self.descriptor.config.hidden_size,
                        input_hidden.len()
                    )),
                ));
            }
            input_hidden.to_vec()
        } else {
            embedding_hidden.clone()
        };
        let gemma4_per_layer_inputs =
            if let Some(per_layer_inputs) = self.gemma4_per_layer_inputs.as_ref() {
                Some(per_layer_inputs.project(
                    backend,
                    token,
                    embedding_hidden.as_slice(),
                    self.layers.len(),
                    self.descriptor.config.hidden_size,
                    self.family_metadata.rms_norm_epsilon,
                )?)
            } else {
                None
            };
        let mut cache_key = forwarded_key
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        let mut cache_value = forwarded_value
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        if cache_key.len() != self.cache_width() || cache_value.len() != self.cache_width() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 cuda forwarded kv width mismatch: expected {}, actual key={} value={}",
                    self.cache_width(),
                    cache_key.len(),
                    cache_value.len()
                )),
            ));
        }

        let mut live_keys: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        let mut live_values: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        let mut pending_cache_writes: Vec<Option<(Vec<f32>, Vec<f32>)>> =
            vec![None; self.layers.len()];
        if start_layer > 0 {
            for (layer_index, layer) in self.layers.iter().enumerate().take(start_layer) {
                if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                    let kv_width = layer.attention_geometry.kv_width();
                    let layer_key = cache_key[cache_offset..cache_offset + kv_width].to_vec();
                    let layer_value = cache_value[cache_offset..cache_offset + kv_width].to_vec();
                    live_keys[layer_index] = Some(layer_key.clone());
                    live_values[layer_index] = Some(layer_value.clone());
                    pending_cache_writes[layer_index] = Some((layer_key, layer_value));
                }
            }
        }

        for (layer_index, layer) in self
            .layers
            .iter()
            .enumerate()
            .skip(start_layer)
            .take(end_layer.saturating_sub(start_layer))
        {
            let residual = hidden.clone();
            let hidden_norm = rms_norm(
                hidden.as_slice(),
                layer.attention_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );

            let (attention_values, q_kernel_count, q_bytes_moved, kv_kernel_count, kv_bytes_moved) =
                if layer.attention_geometry.has_kv() {
                    let mut q = layer.attention_query_weight.matvec(backend, &hidden_norm)?;
                    if let Some(bias) = layer.attention_query_bias.as_ref() {
                        add_bias_in_place(&mut q.values, bias.as_slice());
                    }
                    if let Some(norm) = layer.attention_query_norm.as_ref() {
                        per_head_rms_norm_in_place(
                            q.values.as_mut_slice(),
                            layer.attention_geometry.head_count,
                            layer.attention_geometry.head_dim,
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    apply_rope_neox(
                        &mut q.values,
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.head_dim,
                        layer.attention_geometry.rotary_dim,
                        position,
                        layer.attention_geometry.rope_theta,
                        self.rope_freq_factors_for_layer(
                            layer_index,
                            layer.attention_geometry.head_dim,
                        ),
                        &self.family_metadata,
                    );

                    let mut k = layer.attention_key_weight.matvec(backend, &hidden_norm)?;
                    if let Some(bias) = layer.attention_key_bias.as_ref() {
                        add_bias_in_place(&mut k.values, bias.as_slice());
                    }
                    if let Some(norm) = layer.attention_key_norm.as_ref() {
                        per_head_rms_norm_in_place(
                            k.values.as_mut_slice(),
                            layer.attention_geometry.kv_head_count,
                            layer.attention_geometry.head_dim,
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    apply_rope_neox(
                        &mut k.values,
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        layer.attention_geometry.rotary_dim,
                        position,
                        layer.attention_geometry.rope_theta,
                        self.rope_freq_factors_for_layer(
                            layer_index,
                            layer.attention_geometry.head_dim,
                        ),
                        &self.family_metadata,
                    );

                    let mut v = layer.attention_value_weight.matvec(backend, &hidden_norm)?;
                    if let Some(bias) = layer.attention_value_bias.as_ref() {
                        add_bias_in_place(&mut v.values, bias.as_slice());
                    }
                    if self.family_metadata.family == GgufDecoderFamily::Gemma4 {
                        per_head_rms_norm_unit_in_place(
                            v.values.as_mut_slice(),
                            layer.attention_geometry.kv_head_count,
                            layer.attention_geometry.head_dim,
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }

                    if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                        let kv_width = layer.attention_geometry.kv_width();
                        cache_key[cache_offset..cache_offset + kv_width]
                            .copy_from_slice(k.values.as_slice());
                        cache_value[cache_offset..cache_offset + kv_width]
                            .copy_from_slice(v.values.as_slice());
                    }
                    live_keys[layer_index] = Some(k.values.clone());
                    live_values[layer_index] = Some(v.values.clone());
                    pending_cache_writes[layer_index] = Some((k.values.clone(), v.values.clone()));

                    let layer_cache = layer_caches
                        .get_mut(layer_index)
                        .and_then(Option::as_mut)
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                format!(
                                    "gemma4 cuda layer {layer_index} is missing a device kv cache state"
                                ),
                            ))
                        })?;
                    layer_cache.ensure_capacity(backend, position.saturating_add(1))?;
                    step_plan
                        .query_buffer
                        .write_f32_at_offset(0, q.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    step_plan
                        .key_buffer
                        .write_f32_at_offset(0, k.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    step_plan
                        .value_buffer
                        .write_f32_at_offset(0, v.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    let mut submission = backend
                        .begin_submission()
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .attention_decode(
                            &step_plan.query_buffer,
                            0,
                            &step_plan.key_buffer,
                            0,
                            &step_plan.value_buffer,
                            0,
                            &layer_cache.key_cache,
                            &layer_cache.value_cache,
                            layer_cache.width,
                            0,
                            position,
                            layer.attention_geometry.sliding_window.unwrap_or(0),
                            layer.attention_geometry.head_count,
                            layer.attention_geometry.kv_head_count,
                            layer.attention_geometry.head_dim,
                            None,
                            &step_plan.attention_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .commit(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    (
                        step_plan
                            .attention_buffer
                            .read_f32()
                            .map_err(ReferenceTextGenerationError::Runtime)?,
                        q.kernel_count,
                        q.bytes_moved,
                        k.kernel_count
                            .saturating_add(v.kernel_count)
                            .saturating_add(1),
                        k.bytes_moved
                            .saturating_add(v.bytes_moved)
                            .saturating_add(step_plan.attention_buffer.byte_len() as u64),
                    )
                } else {
                    let mut q = layer.attention_query_weight.matvec(backend, &hidden_norm)?;
                    if let Some(bias) = layer.attention_query_bias.as_ref() {
                        add_bias_in_place(&mut q.values, bias.as_slice());
                    }
                    if let Some(norm) = layer.attention_query_norm.as_ref() {
                        per_head_rms_norm_in_place(
                            q.values.as_mut_slice(),
                            layer.attention_geometry.head_count,
                            layer.attention_geometry.head_dim,
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    apply_rope_neox(
                        &mut q.values,
                        layer.attention_geometry.head_count,
                        layer.attention_geometry.head_dim,
                        layer.attention_geometry.rotary_dim,
                        position,
                        layer.attention_geometry.rope_theta,
                        self.rope_freq_factors_for_layer(
                            layer_index,
                            layer.attention_geometry.head_dim,
                        ),
                        &self.family_metadata,
                    );
                    let reuse_layer_index =
                        layer.attention_geometry.reuse_layer_index.ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                format!(
                                    "gemma4 layer {layer_index} is missing reused kv source metadata"
                                ),
                            ))
                        })?;
                    let reused_k = live_keys
                        .get(reuse_layer_index)
                        .and_then(Option::as_deref)
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                format!(
                                    "gemma4 layer {layer_index} expected live key cache from layer {reuse_layer_index}"
                                ),
                            ))
                        })?;
                    let reused_v = live_values
                        .get(reuse_layer_index)
                        .and_then(Option::as_deref)
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                format!(
                                    "gemma4 layer {layer_index} expected live value cache from layer {reuse_layer_index}"
                                ),
                            ))
                        })?;
                    let source_cache = layer_caches
                        .get_mut(reuse_layer_index)
                        .and_then(Option::as_mut)
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                format!(
                                    "gemma4 layer {layer_index} expected device kv cache state from layer {reuse_layer_index}"
                                ),
                            ))
                        })?;
                    source_cache.ensure_capacity(backend, position.saturating_add(1))?;
                    step_plan
                        .query_buffer
                        .write_f32_at_offset(0, q.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    step_plan
                        .key_buffer
                        .write_f32_at_offset(0, reused_k)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    step_plan
                        .value_buffer
                        .write_f32_at_offset(0, reused_v)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    let mut submission = backend
                        .begin_submission()
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .attention_decode(
                            &step_plan.query_buffer,
                            0,
                            &step_plan.key_buffer,
                            0,
                            &step_plan.value_buffer,
                            0,
                            &source_cache.key_cache,
                            &source_cache.value_cache,
                            source_cache.width,
                            0,
                            position,
                            layer.attention_geometry.sliding_window.unwrap_or(0),
                            layer.attention_geometry.head_count,
                            layer.attention_geometry.kv_head_count,
                            layer.attention_geometry.head_dim,
                            None,
                            &step_plan.attention_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .commit(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    (
                        step_plan
                            .attention_buffer
                            .read_f32()
                            .map_err(ReferenceTextGenerationError::Runtime)?,
                        q.kernel_count,
                        q.bytes_moved,
                        1,
                        step_plan.attention_buffer.byte_len() as u64,
                    )
                };

            let mut attention_out = layer
                .attention_output_weight
                .matvec(backend, attention_values.as_slice())?;
            if let Some(bias) = layer.attention_output_bias.as_ref() {
                add_bias_in_place(&mut attention_out.values, bias.as_slice());
            }
            if let Some(norm) = layer.attention_post_norm.as_ref() {
                rms_norm_in_place(
                    attention_out.values.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(attention_out.values.as_mut_slice(), residual.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            hidden = attention_out.values;

            let ffn_residual = hidden.clone();
            let ffn_input = rms_norm(
                hidden.as_slice(),
                layer.feed_forward_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let (mut ffn_out, gate_kernel_count, gate_bytes_moved, up_kernel_count, up_bytes_moved) =
                if self.family_metadata.family == GgufDecoderFamily::Gemma4
                    && layer.feed_forward_router_weight.is_some()
                    && layer.feed_forward_gate_experts_weight.is_some()
                    && layer.feed_forward_down_experts_weight.is_some()
                {
                    let gate = layer
                        .feed_forward_gate_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_gate_weight"),
                            ))
                        })?
                        .matvec(backend, &ffn_input)?;
                    let up = layer
                        .feed_forward_up_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_up_weight"),
                            ))
                        })?
                        .matvec(backend, &ffn_input)?;
                    let activated = feed_forward_activation(
                        &self.family_metadata,
                        gate.values.as_slice(),
                        up.values.as_slice(),
                    );
                    let mut shared_ffn = layer
                        .feed_forward_down_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_down_weight"),
                            ))
                        })?
                        .matvec(backend, activated.as_slice())?;
                    if let Some(norm) = layer.feed_forward_post_norm_1.as_ref() {
                        rms_norm_in_place(
                            shared_ffn.values.as_mut_slice(),
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    let mut router_input = rms_norm_unit(
                        ffn_residual.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                    scale_in_place(
                        router_input.as_mut_slice(),
                        (self.descriptor.config.hidden_size as f32).sqrt().recip(),
                    );
                    if let Some(scale) = layer.feed_forward_router_input_scale.as_ref() {
                        multiply_vectors_in_place(router_input.as_mut_slice(), scale.as_slice())
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                    }
                    let expert_input = if let Some(norm) = layer.feed_forward_pre_norm_2.as_ref() {
                        rms_norm(
                            ffn_residual.as_slice(),
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        )
                    } else {
                        ffn_input.clone()
                    };
                    let mut moe_ffn = evaluate_sparse_moe_host(
                        &self.family_metadata,
                        router_input.as_slice(),
                        expert_input.as_slice(),
                        layer
                            .feed_forward_router_weight
                            .as_ref()
                            .expect("router checked"),
                        layer.feed_forward_router_bias.as_deref(),
                        layer
                            .feed_forward_gate_experts_weight
                            .as_ref()
                            .expect("gate experts checked"),
                        layer.feed_forward_gate_experts_bias.as_deref(),
                        layer.feed_forward_up_experts_weight.as_ref(),
                        layer.feed_forward_up_experts_bias.as_deref(),
                        layer
                            .feed_forward_down_experts_weight
                            .as_ref()
                            .expect("down experts checked"),
                        layer.feed_forward_down_experts_bias.as_deref(),
                        layer.feed_forward_down_experts_scale.as_deref(),
                        self.family_metadata.expert_used_count.unwrap_or(1),
                    )?;
                    if let Some(norm) = layer.feed_forward_post_norm_2.as_ref() {
                        rms_norm_in_place(
                            moe_ffn.values.as_mut_slice(),
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    (
                        CudaProjectionStep {
                            values: add_vectors(
                                shared_ffn.values.as_slice(),
                                moe_ffn.values.as_slice(),
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?,
                            kernel_count: shared_ffn.kernel_count + moe_ffn.kernel_count,
                            bytes_moved: shared_ffn.bytes_moved + moe_ffn.bytes_moved,
                        },
                        gate.kernel_count,
                        gate.bytes_moved,
                        up.kernel_count,
                        up.bytes_moved,
                    )
                } else if let (Some(router), Some(gate_experts), Some(down_experts)) = (
                    layer.feed_forward_router_weight.as_ref(),
                    layer.feed_forward_gate_experts_weight.as_ref(),
                    layer.feed_forward_down_experts_weight.as_ref(),
                ) {
                    let moe_ffn = evaluate_sparse_moe_host(
                        &self.family_metadata,
                        ffn_input.as_slice(),
                        ffn_input.as_slice(),
                        router,
                        layer.feed_forward_router_bias.as_deref(),
                        gate_experts,
                        layer.feed_forward_gate_experts_bias.as_deref(),
                        layer.feed_forward_up_experts_weight.as_ref(),
                        layer.feed_forward_up_experts_bias.as_deref(),
                        down_experts,
                        layer.feed_forward_down_experts_bias.as_deref(),
                        layer.feed_forward_down_experts_scale.as_deref(),
                        self.family_metadata.expert_used_count.unwrap_or(1),
                    )?;
                    (
                        CudaProjectionStep {
                            values: moe_ffn.values,
                            kernel_count: moe_ffn.kernel_count,
                            bytes_moved: moe_ffn.bytes_moved,
                        },
                        0,
                        0,
                        0,
                        0,
                    )
                } else {
                    let gate = layer
                        .feed_forward_gate_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing dense feed_forward_gate_weight"),
                            ))
                        })?
                        .matvec(backend, &ffn_input)?;
                    let up = layer
                        .feed_forward_up_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing dense feed_forward_up_weight"),
                            ))
                        })?
                        .matvec(backend, &ffn_input)?;
                    let activated = feed_forward_activation(
                        &self.family_metadata,
                        gate.values.as_slice(),
                        up.values.as_slice(),
                    );
                    (
                        layer
                            .feed_forward_down_weight
                            .as_ref()
                            .ok_or_else(|| {
                                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                    String::from("missing dense feed_forward_down_weight"),
                                ))
                            })?
                            .matvec(backend, activated.as_slice())?,
                        gate.kernel_count,
                        gate.bytes_moved,
                        up.kernel_count,
                        up.bytes_moved,
                    )
                };
            if let Some(norm) = layer.feed_forward_post_norm.as_ref() {
                rms_norm_in_place(
                    ffn_out.values.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(ffn_out.values.as_mut_slice(), ffn_residual.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            hidden = ffn_out.values;

            if let Some(per_layer_inputs) = gemma4_per_layer_inputs.as_ref() {
                if let (Some(input_gate), Some(proj), Some(post_norm)) = (
                    layer.per_layer_input_gate.as_ref(),
                    layer.per_layer_proj.as_ref(),
                    layer.per_layer_post_norm.as_ref(),
                ) {
                    let mut gated = input_gate.matvec(backend, hidden.as_slice())?;
                    for value in &mut gated.values {
                        *value = approximate_gelu(*value);
                    }
                    let gated_values = multiply_vectors(
                        gated.values.as_slice(),
                        self.gemma4_per_layer_inputs
                            .as_ref()
                            .expect("per-layer config")
                            .layer_slice(per_layer_inputs.values.as_slice(), layer_index),
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                    let mut projected = proj.matvec(backend, gated_values.as_slice())?;
                    rms_norm_in_place(
                        projected.values.as_mut_slice(),
                        post_norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                    hidden = add_vectors(hidden.as_slice(), projected.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    bytes_moved = bytes_moved
                        .saturating_add(gated.bytes_moved)
                        .saturating_add(projected.bytes_moved);
                    kernel_count = kernel_count
                        .saturating_add(gated.kernel_count)
                        .saturating_add(projected.kernel_count);
                }
            }
            if let Some(scale) = layer.layer_output_scale {
                scale_in_place(&mut hidden, scale);
            }

            bytes_moved = bytes_moved
                .saturating_add(q_bytes_moved)
                .saturating_add(kv_bytes_moved)
                .saturating_add(attention_out.bytes_moved)
                .saturating_add(gate_bytes_moved)
                .saturating_add(up_bytes_moved)
                .saturating_add(ffn_out.bytes_moved);
            kernel_count = kernel_count
                .saturating_add(q_kernel_count)
                .saturating_add(kv_kernel_count)
                .saturating_add(attention_out.kernel_count)
                .saturating_add(gate_kernel_count)
                .saturating_add(up_kernel_count)
                .saturating_add(ffn_out.kernel_count);
        }

        for (layer_index, pending) in pending_cache_writes.into_iter().enumerate() {
            let Some((pending_key, pending_value)) = pending else {
                continue;
            };
            let layer_cache = layer_caches
                .get_mut(layer_index)
                .and_then(Option::as_mut)
                .ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "gemma4 cuda layer {layer_index} is missing a device kv cache state during writeback"
                    )))
                })?;
            layer_cache.ensure_capacity(backend, layer_cache.len.saturating_add(1))?;
            let write_offset = layer_cache.len.saturating_mul(layer_cache.width);
            layer_cache
                .key_cache
                .write_f32_at_offset(write_offset, pending_key.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            layer_cache
                .value_cache
                .write_f32_at_offset(write_offset, pending_value.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            layer_cache.len = layer_cache.len.saturating_add(1);
        }

        let (logits, lm_head_hidden) = if produce_logits {
            let final_hidden = rms_norm(
                hidden.as_slice(),
                self.output_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let mut logits = self.output.matvec(backend, final_hidden.as_slice())?;
            apply_final_logit_softcapping_in_place(
                logits.values.as_mut_slice(),
                self.family_metadata.final_logit_softcapping,
            );
            bytes_moved = bytes_moved.saturating_add(logits.bytes_moved);
            kernel_count = kernel_count.saturating_add(logits.kernel_count);
            (logits.values, Some(final_hidden))
        } else {
            (Vec::new(), None)
        };

        Ok(CudaGemma4StageStep {
            key: cache_key,
            value: cache_value,
            logits,
            hidden,
            lm_head_hidden,
            kernel_count,
            bytes_moved,
        })
    }

    fn forward_stage_step(
        &self,
        backend: &mut CudaBackend,
        token: TokenId,
        position: usize,
        cache: &crate::InMemoryKvCache,
        start_layer: usize,
        end_layer: usize,
        input_hidden: Option<&[f32]>,
        forwarded_key: Option<&[f32]>,
        forwarded_value: Option<&[f32]>,
        produce_logits: bool,
    ) -> Result<CudaGemma4StageStep, ReferenceTextGenerationError> {
        if start_layer > end_layer || end_layer > self.layers.len() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "invalid gemma4 cuda stage range [{start_layer}..{end_layer}) for {} layers",
                    self.layers.len()
                )),
            ));
        }
        let mut bytes_moved = self.token_embedding.byte_length() as u64;
        let mut kernel_count = 1usize;
        let mut embedding_hidden = self
            .token_embedding
            .decode_row(token.as_u32() as usize)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        scale_in_place(
            &mut embedding_hidden,
            input_embedding_scale(&self.family_metadata, self.descriptor.config.hidden_size),
        );
        let mut hidden = if let Some(input_hidden) = input_hidden {
            if input_hidden.len() != self.descriptor.config.hidden_size {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "gemma4 cuda stage input hidden width mismatch: expected {}, actual {}",
                        self.descriptor.config.hidden_size,
                        input_hidden.len()
                    )),
                ));
            }
            input_hidden.to_vec()
        } else {
            embedding_hidden.clone()
        };
        let gemma4_per_layer_inputs =
            if let Some(per_layer_inputs) = self.gemma4_per_layer_inputs.as_ref() {
                Some(per_layer_inputs.project(
                    backend,
                    token,
                    embedding_hidden.as_slice(),
                    self.layers.len(),
                    self.descriptor.config.hidden_size,
                    self.family_metadata.rms_norm_epsilon,
                )?)
            } else {
                None
            };
        let mut cache_key = forwarded_key
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        let mut cache_value = forwarded_value
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; self.cache_width()]);
        if cache_key.len() != self.cache_width() || cache_value.len() != self.cache_width() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 cuda forwarded kv width mismatch: expected {}, actual key={} value={}",
                    self.cache_width(),
                    cache_key.len(),
                    cache_value.len()
                )),
            ));
        }
        let mut live_keys: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        let mut live_values: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
        if start_layer > 0 {
            for (layer_index, layer) in self.layers.iter().enumerate().take(start_layer) {
                if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                    let kv_width = layer.attention_geometry.kv_width();
                    live_keys[layer_index] =
                        Some(cache_key[cache_offset..cache_offset + kv_width].to_vec());
                    live_values[layer_index] =
                        Some(cache_value[cache_offset..cache_offset + kv_width].to_vec());
                }
            }
        }
        for (layer_index, layer) in self
            .layers
            .iter()
            .enumerate()
            .skip(start_layer)
            .take(end_layer.saturating_sub(start_layer))
        {
            let residual = hidden.clone();
            let hidden_norm = rms_norm(
                hidden.as_slice(),
                layer.attention_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );

            let mut q = layer.attention_query_weight.matvec(backend, &hidden_norm)?;
            if let Some(bias) = layer.attention_query_bias.as_ref() {
                add_bias_in_place(&mut q.values, bias.as_slice());
            }
            if let Some(norm) = layer.attention_query_norm.as_ref() {
                per_head_rms_norm_in_place(
                    q.values.as_mut_slice(),
                    layer.attention_geometry.head_count,
                    layer.attention_geometry.head_dim,
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }

            apply_rope_neox(
                &mut q.values,
                layer.attention_geometry.head_count,
                layer.attention_geometry.head_dim,
                layer.attention_geometry.rotary_dim,
                position,
                layer.attention_geometry.rope_theta,
                self.rope_freq_factors_for_layer(layer_index, layer.attention_geometry.head_dim),
                &self.family_metadata,
            );
            let (k, v) = if layer.attention_geometry.has_kv() {
                let mut k = layer.attention_key_weight.matvec(backend, &hidden_norm)?;
                if let Some(bias) = layer.attention_key_bias.as_ref() {
                    add_bias_in_place(&mut k.values, bias.as_slice());
                }
                if let Some(norm) = layer.attention_key_norm.as_ref() {
                    per_head_rms_norm_in_place(
                        k.values.as_mut_slice(),
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                }

                let mut v = layer.attention_value_weight.matvec(backend, &hidden_norm)?;
                if let Some(bias) = layer.attention_value_bias.as_ref() {
                    add_bias_in_place(&mut v.values, bias.as_slice());
                }
                if self.family_metadata.family == GgufDecoderFamily::Gemma4 {
                    per_head_rms_norm_unit_in_place(
                        v.values.as_mut_slice(),
                        layer.attention_geometry.kv_head_count,
                        layer.attention_geometry.head_dim,
                        self.family_metadata.rms_norm_epsilon,
                    );
                }
                apply_rope_neox(
                    &mut k.values,
                    layer.attention_geometry.kv_head_count,
                    layer.attention_geometry.head_dim,
                    layer.attention_geometry.rotary_dim,
                    position,
                    layer.attention_geometry.rope_theta,
                    self.rope_freq_factors_for_layer(
                        layer_index,
                        layer.attention_geometry.head_dim,
                    ),
                    &self.family_metadata,
                );
                if let Some(cache_offset) = layer.attention_geometry.cache_write_offset {
                    let kv_width = layer.attention_geometry.kv_width();
                    cache_key[cache_offset..cache_offset + kv_width]
                        .copy_from_slice(k.values.as_slice());
                    cache_value[cache_offset..cache_offset + kv_width]
                        .copy_from_slice(v.values.as_slice());
                }
                live_keys[layer_index] = Some(k.values.clone());
                live_values[layer_index] = Some(v.values.clone());
                (k, v)
            } else {
                let reuse_layer_index =
                    layer.attention_geometry.reuse_layer_index.ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} is missing reused kv source metadata"
                            ),
                        ))
                    })?;
                let reused_k = live_keys
                    .get(reuse_layer_index)
                    .and_then(|value| value.clone())
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} expected live key cache from layer {reuse_layer_index}"
                            ),
                        ))
                    })?;
                let reused_v = live_values
                    .get(reuse_layer_index)
                    .and_then(|value| value.clone())
                    .ok_or_else(|| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "gemma4 layer {layer_index} expected live value cache from layer {reuse_layer_index}"
                            ),
                        ))
                    })?;
                (
                    CudaProjectionStep {
                        values: reused_k,
                        kernel_count: 0,
                        bytes_moved: 0,
                    },
                    CudaProjectionStep {
                        values: reused_v,
                        kernel_count: 0,
                        bytes_moved: 0,
                    },
                )
            };

            let attention = attend_impl(
                layer_index,
                q.values.as_slice(),
                k.values.as_slice(),
                v.values.as_slice(),
                cache,
                layer.attention_geometry.head_count,
                layer.attention_geometry.kv_head_count,
                layer.attention_geometry.head_dim,
                layer.attention_geometry.cache_read_offset,
                layer.attention_geometry.sliding_window,
                attention_scale(&self.family_metadata, layer.attention_geometry.head_dim),
            );
            let mut attention_out = layer
                .attention_output_weight
                .matvec(backend, attention.as_slice())?;
            if let Some(bias) = layer.attention_output_bias.as_ref() {
                add_bias_in_place(&mut attention_out.values, bias.as_slice());
            }
            if let Some(norm) = layer.attention_post_norm.as_ref() {
                rms_norm_in_place(
                    attention_out.values.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(attention_out.values.as_mut_slice(), residual.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            hidden = attention_out.values;

            let ffn_residual = hidden.clone();
            let ffn_input = rms_norm(
                hidden.as_slice(),
                layer.feed_forward_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let (mut ffn_out, gate_kernel_count, gate_bytes_moved, up_kernel_count, up_bytes_moved) =
                if self.family_metadata.family == GgufDecoderFamily::Gemma4
                    && layer.feed_forward_router_weight.is_some()
                    && layer.feed_forward_gate_experts_weight.is_some()
                    && layer.feed_forward_down_experts_weight.is_some()
                {
                    let gate = layer
                        .feed_forward_gate_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_gate_weight"),
                            ))
                        })?
                        .matvec(backend, &ffn_input)?;
                    let up = layer
                        .feed_forward_up_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_up_weight"),
                            ))
                        })?
                        .matvec(backend, &ffn_input)?;
                    let activated = feed_forward_activation(
                        &self.family_metadata,
                        gate.values.as_slice(),
                        up.values.as_slice(),
                    );
                    let mut shared_ffn = layer
                        .feed_forward_down_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing gemma4 dense feed_forward_down_weight"),
                            ))
                        })?
                        .matvec(backend, activated.as_slice())?;
                    if let Some(norm) = layer.feed_forward_post_norm_1.as_ref() {
                        rms_norm_in_place(
                            shared_ffn.values.as_mut_slice(),
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    let mut router_input = rms_norm_unit(
                        ffn_residual.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                    scale_in_place(
                        router_input.as_mut_slice(),
                        (self.descriptor.config.hidden_size as f32).sqrt().recip(),
                    );
                    if let Some(scale) = layer.feed_forward_router_input_scale.as_ref() {
                        multiply_vectors_in_place(router_input.as_mut_slice(), scale.as_slice())
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                    }
                    let expert_input = if let Some(norm) = layer.feed_forward_pre_norm_2.as_ref() {
                        rms_norm(
                            ffn_residual.as_slice(),
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        )
                    } else {
                        ffn_input.clone()
                    };
                    let mut moe_ffn = evaluate_sparse_moe_host(
                        &self.family_metadata,
                        router_input.as_slice(),
                        expert_input.as_slice(),
                        layer
                            .feed_forward_router_weight
                            .as_ref()
                            .expect("router checked"),
                        layer.feed_forward_router_bias.as_deref(),
                        layer
                            .feed_forward_gate_experts_weight
                            .as_ref()
                            .expect("gate experts checked"),
                        layer.feed_forward_gate_experts_bias.as_deref(),
                        layer.feed_forward_up_experts_weight.as_ref(),
                        layer.feed_forward_up_experts_bias.as_deref(),
                        layer
                            .feed_forward_down_experts_weight
                            .as_ref()
                            .expect("down experts checked"),
                        layer.feed_forward_down_experts_bias.as_deref(),
                        layer.feed_forward_down_experts_scale.as_deref(),
                        self.family_metadata.expert_used_count.unwrap_or(1),
                    )?;
                    if let Some(norm) = layer.feed_forward_post_norm_2.as_ref() {
                        rms_norm_in_place(
                            moe_ffn.values.as_mut_slice(),
                            norm.as_slice(),
                            self.family_metadata.rms_norm_epsilon,
                        );
                    }
                    (
                        CudaProjectionStep {
                            values: add_vectors(
                                shared_ffn.values.as_slice(),
                                moe_ffn.values.as_slice(),
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?,
                            kernel_count: shared_ffn.kernel_count + moe_ffn.kernel_count,
                            bytes_moved: shared_ffn.bytes_moved + moe_ffn.bytes_moved,
                        },
                        gate.kernel_count,
                        gate.bytes_moved,
                        up.kernel_count,
                        up.bytes_moved,
                    )
                } else if let (Some(router), Some(gate_experts), Some(down_experts)) = (
                    layer.feed_forward_router_weight.as_ref(),
                    layer.feed_forward_gate_experts_weight.as_ref(),
                    layer.feed_forward_down_experts_weight.as_ref(),
                ) {
                    let moe_ffn = evaluate_sparse_moe_host(
                        &self.family_metadata,
                        ffn_input.as_slice(),
                        ffn_input.as_slice(),
                        router,
                        layer.feed_forward_router_bias.as_deref(),
                        gate_experts,
                        layer.feed_forward_gate_experts_bias.as_deref(),
                        layer.feed_forward_up_experts_weight.as_ref(),
                        layer.feed_forward_up_experts_bias.as_deref(),
                        down_experts,
                        layer.feed_forward_down_experts_bias.as_deref(),
                        layer.feed_forward_down_experts_scale.as_deref(),
                        self.family_metadata.expert_used_count.unwrap_or(1),
                    )?;
                    (
                        CudaProjectionStep {
                            values: moe_ffn.values,
                            kernel_count: moe_ffn.kernel_count,
                            bytes_moved: moe_ffn.bytes_moved,
                        },
                        0,
                        0,
                        0,
                        0,
                    )
                } else {
                    let gate = layer
                        .feed_forward_gate_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing dense feed_forward_gate_weight"),
                            ))
                        })?
                        .matvec(backend, &ffn_input)?;
                    let up = layer
                        .feed_forward_up_weight
                        .as_ref()
                        .ok_or_else(|| {
                            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                String::from("missing dense feed_forward_up_weight"),
                            ))
                        })?
                        .matvec(backend, &ffn_input)?;
                    let activated = feed_forward_activation(
                        &self.family_metadata,
                        gate.values.as_slice(),
                        up.values.as_slice(),
                    );
                    (
                        layer
                            .feed_forward_down_weight
                            .as_ref()
                            .ok_or_else(|| {
                                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                                    String::from("missing dense feed_forward_down_weight"),
                                ))
                            })?
                            .matvec(backend, activated.as_slice())?,
                        gate.kernel_count,
                        gate.bytes_moved,
                        up.kernel_count,
                        up.bytes_moved,
                    )
                };
            if let Some(norm) = layer.feed_forward_post_norm.as_ref() {
                rms_norm_in_place(
                    ffn_out.values.as_mut_slice(),
                    norm.as_slice(),
                    self.family_metadata.rms_norm_epsilon,
                );
            }
            add_vectors_in_place(ffn_out.values.as_mut_slice(), ffn_residual.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            hidden = ffn_out.values;

            if let Some(per_layer_inputs) = gemma4_per_layer_inputs.as_ref() {
                if let (Some(input_gate), Some(proj), Some(post_norm)) = (
                    layer.per_layer_input_gate.as_ref(),
                    layer.per_layer_proj.as_ref(),
                    layer.per_layer_post_norm.as_ref(),
                ) {
                    let mut gated = input_gate.matvec(backend, hidden.as_slice())?;
                    for value in &mut gated.values {
                        *value = approximate_gelu(*value);
                    }
                    let gated_values = multiply_vectors(
                        gated.values.as_slice(),
                        self.gemma4_per_layer_inputs
                            .as_ref()
                            .expect("per-layer config")
                            .layer_slice(per_layer_inputs.values.as_slice(), layer_index),
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                    let mut projected = proj.matvec(backend, gated_values.as_slice())?;
                    rms_norm_in_place(
                        projected.values.as_mut_slice(),
                        post_norm.as_slice(),
                        self.family_metadata.rms_norm_epsilon,
                    );
                    hidden = add_vectors(hidden.as_slice(), projected.values.as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    bytes_moved = bytes_moved
                        .saturating_add(gated.bytes_moved)
                        .saturating_add(projected.bytes_moved);
                    kernel_count = kernel_count
                        .saturating_add(gated.kernel_count)
                        .saturating_add(projected.kernel_count);
                }
            }
            if let Some(scale) = layer.layer_output_scale {
                scale_in_place(&mut hidden, scale);
            }

            bytes_moved = bytes_moved
                .saturating_add(q.bytes_moved)
                .saturating_add(attention_out.bytes_moved)
                .saturating_add(gate_bytes_moved)
                .saturating_add(up_bytes_moved)
                .saturating_add(ffn_out.bytes_moved);
            kernel_count = kernel_count
                .saturating_add(q.kernel_count)
                .saturating_add(attention_out.kernel_count)
                .saturating_add(gate_kernel_count)
                .saturating_add(up_kernel_count)
                .saturating_add(ffn_out.kernel_count);
            if layer.attention_geometry.has_kv() {
                bytes_moved = bytes_moved
                    .saturating_add(k.bytes_moved)
                    .saturating_add(v.bytes_moved);
                kernel_count = kernel_count
                    .saturating_add(k.kernel_count)
                    .saturating_add(v.kernel_count);
            }
        }

        let (logits, lm_head_hidden) = if produce_logits {
            let final_hidden = rms_norm(
                hidden.as_slice(),
                self.output_norm.as_slice(),
                self.family_metadata.rms_norm_epsilon,
            );
            let mut logits = self.output.matvec(backend, final_hidden.as_slice())?;
            apply_final_logit_softcapping_in_place(
                logits.values.as_mut_slice(),
                self.family_metadata.final_logit_softcapping,
            );
            bytes_moved = bytes_moved.saturating_add(logits.bytes_moved);
            kernel_count = kernel_count.saturating_add(logits.kernel_count);
            (logits.values, Some(final_hidden))
        } else {
            (Vec::new(), None)
        };

        Ok(CudaGemma4StageStep {
            key: cache_key,
            value: cache_value,
            logits,
            hidden,
            lm_head_hidden,
            kernel_count,
            bytes_moved,
        })
    }
}

#[derive(Clone, Debug)]
struct CudaGemma4Layer {
    attention_geometry: DenseAttentionGeometry,
    attention_norm: Vec<f32>,
    attention_query_weight: CudaQuantizedProjectionMatrix,
    attention_query_bias: Option<Vec<f32>>,
    attention_query_norm: Option<Vec<f32>>,
    attention_key_weight: CudaQuantizedProjectionMatrix,
    attention_key_bias: Option<Vec<f32>>,
    attention_key_norm: Option<Vec<f32>>,
    attention_value_weight: CudaQuantizedProjectionMatrix,
    attention_value_bias: Option<Vec<f32>>,
    attention_output_weight: CudaQuantizedProjectionMatrix,
    attention_output_bias: Option<Vec<f32>>,
    attention_post_norm: Option<Vec<f32>>,
    feed_forward_norm: Vec<f32>,
    feed_forward_gate_weight: Option<CudaQuantizedProjectionMatrix>,
    feed_forward_up_weight: Option<CudaQuantizedProjectionMatrix>,
    feed_forward_down_weight: Option<CudaQuantizedProjectionMatrix>,
    feed_forward_router_weight: Option<DenseMatrix>,
    feed_forward_router_bias: Option<Vec<f32>>,
    feed_forward_router_input_scale: Option<Vec<f32>>,
    feed_forward_gate_experts_weight: Option<QuantizedExpertTensor>,
    feed_forward_gate_experts_bias: Option<Vec<f32>>,
    feed_forward_up_experts_weight: Option<QuantizedExpertTensor>,
    feed_forward_up_experts_bias: Option<Vec<f32>>,
    feed_forward_down_experts_weight: Option<QuantizedExpertTensor>,
    feed_forward_down_experts_bias: Option<Vec<f32>>,
    feed_forward_down_experts_scale: Option<Vec<f32>>,
    feed_forward_post_norm_1: Option<Vec<f32>>,
    feed_forward_pre_norm_2: Option<Vec<f32>>,
    feed_forward_post_norm_2: Option<Vec<f32>>,
    feed_forward_post_norm: Option<Vec<f32>>,
    layer_output_scale: Option<f32>,
    per_layer_input_gate: Option<CudaQuantizedProjectionMatrix>,
    per_layer_proj: Option<CudaQuantizedProjectionMatrix>,
    per_layer_post_norm: Option<Vec<f32>>,
}

impl CudaGemma4Layer {
    fn load(
        backend: &mut CudaBackend,
        artifact: &GgufBlobArtifact,
        layout: &GgufDecoderLayerTensorLayout,
        descriptor: &DecoderModelDescriptor,
        family_metadata: &GgufDecoderFamilyMetadata,
        layer_index: usize,
        cache_read_offset: usize,
        cache_write_offset: Option<usize>,
        reuse_layer_index: Option<usize>,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let has_kv = cache_write_offset.is_some();
        let attention_query_weight = CudaQuantizedProjectionMatrix::load(
            backend,
            artifact,
            required_tensor_name(
                layout.attention_query_weight.as_deref(),
                "attention_query_weight",
            )?,
        )?;
        let query_width = attention_query_weight.rows();
        let head_dim = query_width
            .checked_div(descriptor.config.block.attention.head_count)
            .unwrap_or(0);
        let attention_key_weight = if let Some(name) = layout.attention_key_weight.as_deref() {
            CudaQuantizedProjectionMatrix::load(backend, artifact, name)?
        } else if family_metadata.family == GgufDecoderFamily::Gemma4 && !has_kv {
            attention_query_weight.clone()
        } else {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(String::from("missing attention_key_weight")),
            ));
        };
        let key_width = if layout.attention_key_weight.is_some() {
            attention_key_weight.rows()
        } else {
            gemma4_expected_kv_width(descriptor, family_metadata, layer_index, head_dim)
        };
        let attention_value_weight = if let Some(name) = layout.attention_value_weight.as_deref() {
            CudaQuantizedProjectionMatrix::load(backend, artifact, name)?
        } else if family_metadata.family == GgufDecoderFamily::Gemma4 {
            attention_key_weight.clone()
        } else if has_kv {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(String::from("missing attention_value_weight")),
            ));
        } else {
            attention_key_weight.clone()
        };
        let value_width =
            if layout.attention_value_weight.is_some() || layout.attention_key_weight.is_some() {
                attention_value_weight.rows()
            } else {
                key_width
            };
        let attention_geometry = dense_attention_geometry(
            descriptor,
            family_metadata,
            layer_index,
            query_width,
            key_width,
            value_width,
            cache_read_offset,
            cache_write_offset,
            reuse_layer_index,
        )?;
        Ok(Self {
            attention_geometry,
            attention_norm: load_dense_vector(artifact, layout.attention_norm.as_str())?,
            attention_query_weight,
            attention_query_bias: load_optional_dense_vector(
                artifact,
                layout.attention_query_bias.as_deref(),
            )?,
            attention_query_norm: load_optional_dense_vector(
                artifact,
                layout.attention_query_norm.as_deref(),
            )?,
            attention_key_weight,
            attention_key_bias: load_optional_dense_vector(
                artifact,
                layout.attention_key_bias.as_deref(),
            )?,
            attention_key_norm: load_optional_dense_vector(
                artifact,
                layout.attention_key_norm.as_deref(),
            )?,
            attention_value_weight,
            attention_value_bias: load_optional_dense_vector(
                artifact,
                layout.attention_value_bias.as_deref(),
            )?,
            attention_output_weight: CudaQuantizedProjectionMatrix::load(
                backend,
                artifact,
                required_tensor_name(
                    layout.attention_output_weight.as_deref(),
                    "attention_output_weight",
                )?,
            )?,
            attention_output_bias: load_optional_dense_vector(
                artifact,
                layout.attention_output_bias.as_deref(),
            )?,
            attention_post_norm: load_optional_dense_vector(
                artifact,
                layout.attention_post_norm.as_deref(),
            )?,
            feed_forward_norm: load_dense_vector(
                artifact,
                required_tensor_name(layout.feed_forward_norm.as_deref(), "feed_forward_norm")?,
            )?,
            feed_forward_gate_weight: layout
                .feed_forward_gate_weight
                .as_deref()
                .map(|name| CudaQuantizedProjectionMatrix::load(backend, artifact, name))
                .transpose()?,
            feed_forward_up_weight: layout
                .feed_forward_up_weight
                .as_deref()
                .map(|name| CudaQuantizedProjectionMatrix::load(backend, artifact, name))
                .transpose()?,
            feed_forward_down_weight: layout
                .feed_forward_down_weight
                .as_deref()
                .map(|name| CudaQuantizedProjectionMatrix::load(backend, artifact, name))
                .transpose()?,
            feed_forward_router_weight: layout
                .feed_forward_router_weight
                .as_deref()
                .map(|name| load_dense_matrix(artifact, name))
                .transpose()?,
            feed_forward_router_bias: layout
                .feed_forward_router_bias
                .as_deref()
                .map(|name| load_dense_vector(artifact, name))
                .transpose()?,
            feed_forward_router_input_scale: load_first_named_optional_dense_vector(
                artifact,
                &[format!("blk.{layer_index}.ffn_gate_inp.scale")],
            )?,
            feed_forward_gate_experts_weight: layout
                .feed_forward_gate_experts_weight
                .as_deref()
                .map(|name| load_quantized_expert_tensor(artifact, name))
                .transpose()?,
            feed_forward_gate_experts_bias: layout
                .feed_forward_gate_experts_bias
                .as_deref()
                .map(|name| load_dense_rank2_flat(artifact, name))
                .transpose()?,
            feed_forward_up_experts_weight: layout
                .feed_forward_up_experts_weight
                .as_deref()
                .map(|name| load_quantized_expert_tensor(artifact, name))
                .transpose()?,
            feed_forward_up_experts_bias: layout
                .feed_forward_up_experts_bias
                .as_deref()
                .map(|name| load_dense_rank2_flat(artifact, name))
                .transpose()?,
            feed_forward_down_experts_weight: layout
                .feed_forward_down_experts_weight
                .as_deref()
                .map(|name| load_quantized_expert_tensor(artifact, name))
                .transpose()?,
            feed_forward_down_experts_bias: layout
                .feed_forward_down_experts_bias
                .as_deref()
                .map(|name| load_dense_rank2_flat(artifact, name))
                .transpose()?,
            feed_forward_down_experts_scale: load_first_named_optional_dense_vector(
                artifact,
                &[format!("blk.{layer_index}.ffn_down_exps.scale")],
            )?,
            feed_forward_post_norm_1: load_first_named_optional_dense_vector(
                artifact,
                &[
                    format!("blk.{layer_index}.post_ffw_norm_1.weight"),
                    format!("blk.{layer_index}.ffn_post_norm_1.weight"),
                ],
            )?,
            feed_forward_pre_norm_2: load_first_named_optional_dense_vector(
                artifact,
                &[
                    format!("blk.{layer_index}.pre_ffw_norm_2.weight"),
                    format!("blk.{layer_index}.ffn_pre_norm_2.weight"),
                ],
            )?,
            feed_forward_post_norm_2: load_first_named_optional_dense_vector(
                artifact,
                &[
                    format!("blk.{layer_index}.post_ffw_norm_2.weight"),
                    format!("blk.{layer_index}.ffn_post_norm_2.weight"),
                ],
            )?,
            feed_forward_post_norm: load_optional_dense_vector(
                artifact,
                layout.feed_forward_post_norm.as_deref(),
            )?,
            layer_output_scale: load_optional_dense_scalar(
                artifact,
                layout.layer_output_scale.as_deref(),
            )?,
            per_layer_input_gate: load_named_optional_cuda_quantized_projection_matrix(
                backend,
                artifact,
                format!("blk.{layer_index}.inp_gate.weight").as_str(),
            )?,
            per_layer_proj: load_named_optional_cuda_quantized_projection_matrix(
                backend,
                artifact,
                format!("blk.{layer_index}.proj.weight").as_str(),
            )?,
            per_layer_post_norm: load_named_optional_dense_vector(
                artifact,
                format!("blk.{layer_index}.post_norm.weight").as_str(),
            )?,
        })
    }

    fn cache_write_width(&self) -> usize {
        self.attention_geometry.cache_write_width()
    }
}

#[derive(Clone, Debug)]
struct CudaGemma4PerLayerInputs {
    token_embedding: ProjectionMatrix,
    model_proj: CudaGemma4PerLayerProjectionMatrix,
    proj_norm: Vec<f32>,
    width: usize,
}

impl CudaGemma4PerLayerInputs {
    fn load(
        backend: &mut CudaBackend,
        artifact: &GgufBlobArtifact,
    ) -> Result<Option<Self>, ReferenceTextGenerationError> {
        let token_embedding =
            load_named_optional_projection_matrix(artifact, "per_layer_token_embd.weight")?;
        let model_proj = load_named_optional_cuda_gemma4_projection_matrix(
            backend,
            artifact,
            "per_layer_model_proj.weight",
        )?;
        let proj_norm = load_named_optional_dense_vector(artifact, "per_layer_proj_norm.weight")?;
        let (Some(token_embedding), Some(model_proj), Some(proj_norm)) =
            (token_embedding, model_proj, proj_norm)
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            token_embedding,
            model_proj,
            width: proj_norm.len(),
            proj_norm,
        }))
    }

    fn project(
        &self,
        backend: &mut CudaBackend,
        token: TokenId,
        hidden: &[f32],
        layer_count: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<CudaProjectionStep, ReferenceTextGenerationError> {
        let mut token_inputs = self
            .token_embedding
            .decode_row(token.as_u32() as usize)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let expected_len = self.width.saturating_mul(layer_count);
        if token_inputs.len() != expected_len {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 per-layer token embedding width mismatch: expected {expected_len}, actual {}",
                    token_inputs.len()
                )),
            ));
        }
        scale_in_place(&mut token_inputs, (self.width as f32).sqrt());

        let mut projected = self.model_proj.matvec(backend, hidden)?;
        if projected.values.len() != expected_len {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "gemma4 per-layer model projection width mismatch: expected {expected_len}, actual {}",
                    projected.values.len()
                )),
            ));
        }
        scale_in_place(&mut projected.values, (hidden_size as f32).sqrt().recip());

        let layer_mix_scale = 2.0_f32.sqrt().recip();
        let mut combined = vec![0.0_f32; expected_len];
        for layer_index in 0..layer_count {
            let start = layer_index.saturating_mul(self.width);
            let end = start.saturating_add(self.width);
            let normalized = rms_norm(
                &projected.values[start..end],
                self.proj_norm.as_slice(),
                epsilon,
            );
            for index in 0..self.width {
                combined[start + index] =
                    (normalized[index] + token_inputs[start + index]) * layer_mix_scale;
            }
        }
        projected.values = combined;
        Ok(projected)
    }

    fn layer_slice<'a>(&self, values: &'a [f32], layer_index: usize) -> &'a [f32] {
        let start = layer_index.saturating_mul(self.width);
        let end = start.saturating_add(self.width);
        &values[start..end]
    }
}

#[derive(Clone, Debug)]
struct CudaQuantizedProjectionMatrix {
    mode: QuantizationMode,
    rows: usize,
    columns: usize,
    weights: Arc<CudaBuffer>,
}

impl CudaQuantizedProjectionMatrix {
    fn load(
        backend: &mut CudaBackend,
        artifact: &GgufBlobArtifact,
        name: &str,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let storage = artifact.paged_tensor(name)?;
        let metadata = storage.metadata();
        if metadata.quantized_layout.is_none() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "native gemma4 cuda runtime currently requires quantized projection tensor `{name}`",
                )),
            ));
        }
        let [rows, columns] = metadata.shape.dims() else {
            return Err(ModelLoadError::InvalidTensorShape {
                name: metadata.name.clone(),
                expected: vec![0, 0],
                actual: metadata.shape.dims().to_vec(),
            }
            .into());
        };
        let weights = backend.byte_buffer(storage.bytes()?).map_err(|error| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                "failed to upload `{name}` to cuda: {error}",
            )))
        })?;
        Ok(Self {
            mode: metadata.quantization,
            rows: *rows,
            columns: *columns,
            weights: Arc::new(weights),
        })
    }

    fn matvec(
        &self,
        backend: &mut CudaBackend,
        input: &[f32],
    ) -> Result<CudaProjectionStep, ReferenceTextGenerationError> {
        let result = backend
            .quantized_matvec_profiled(
                self.weights.as_ref(),
                self.mode,
                self.rows,
                self.columns,
                input,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
        Ok(CudaProjectionStep {
            values: result.values,
            kernel_count: result.stats.kernel_launches as usize,
            bytes_moved: result
                .stats
                .host_to_device_bytes
                .saturating_add(result.stats.device_to_host_bytes),
        })
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn byte_length(&self) -> usize {
        self.weights.byte_len()
    }
}

#[derive(Clone, Debug)]
enum CudaGemma4PerLayerProjectionMatrix {
    Host(ProjectionMatrix),
    Quantized(CudaQuantizedProjectionMatrix),
}

impl CudaGemma4PerLayerProjectionMatrix {
    fn load(
        backend: &mut CudaBackend,
        artifact: &GgufBlobArtifact,
        name: &str,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let storage = artifact.paged_tensor(name)?;
        if storage.metadata().quantized_layout.is_some() {
            return Ok(Self::Quantized(CudaQuantizedProjectionMatrix::load(
                backend, artifact, name,
            )?));
        }
        Ok(Self::Host(ProjectionMatrix::load(artifact, name)?))
    }

    fn matvec(
        &self,
        backend: &mut CudaBackend,
        input: &[f32],
    ) -> Result<CudaProjectionStep, ReferenceTextGenerationError> {
        match self {
            Self::Host(matrix) => {
                let mut values = Vec::new();
                matrix
                    .matvec(input, &mut values)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                Ok(CudaProjectionStep {
                    values,
                    kernel_count: 0,
                    bytes_moved: matrix.byte_length() as u64,
                })
            }
            Self::Quantized(matrix) => matrix.matvec(backend, input),
        }
    }

    fn byte_length(&self) -> usize {
        match self {
            Self::Host(matrix) => matrix.byte_length(),
            Self::Quantized(matrix) => matrix.byte_length(),
        }
    }
}

#[derive(Clone, Debug)]
struct CudaProjectionStep {
    values: Vec<f32>,
    kernel_count: usize,
    bytes_moved: u64,
}

#[derive(Clone, Debug)]
struct CudaGemma4StageStep {
    key: Vec<f32>,
    value: Vec<f32>,
    logits: Vec<f32>,
    hidden: Vec<f32>,
    lm_head_hidden: Option<Vec<f32>>,
    kernel_count: usize,
    bytes_moved: u64,
}

#[derive(Clone, Debug)]
struct CudaGemma4ForwardStep {
    key: Vec<f32>,
    value: Vec<f32>,
    logits: Vec<f32>,
    final_hidden: Vec<f32>,
    kernel_count: usize,
    bytes_moved: u64,
}

#[derive(Clone, Debug)]
struct CudaGemma4LayerCacheState {
    key_cache: CudaBuffer,
    value_cache: CudaBuffer,
    width: usize,
    len: usize,
    capacity_tokens: usize,
}

impl CudaGemma4LayerCacheState {
    fn new(
        backend: &mut CudaBackend,
        width: usize,
        capacity_tokens: usize,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let cache_len = width.saturating_mul(capacity_tokens);
        Ok(Self {
            key_cache: backend
                .input_buffer(Shape::new(vec![cache_len]), vec![0.0; cache_len])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            value_cache: backend
                .input_buffer(Shape::new(vec![cache_len]), vec![0.0; cache_len])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            width,
            len: 0,
            capacity_tokens,
        })
    }

    fn ensure_capacity(
        &mut self,
        backend: &mut CudaBackend,
        required_tokens: usize,
    ) -> Result<(), ReferenceTextGenerationError> {
        if required_tokens <= self.capacity_tokens {
            return Ok(());
        }
        let new_capacity = required_tokens
            .max(self.capacity_tokens.saturating_mul(2))
            .checked_next_power_of_two()
            .unwrap_or(required_tokens);
        let new_len = self.width.saturating_mul(new_capacity);
        let new_key_cache = backend
            .input_buffer(Shape::new(vec![new_len]), vec![0.0; new_len])
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let new_value_cache = backend
            .input_buffer(Shape::new(vec![new_len]), vec![0.0; new_len])
            .map_err(ReferenceTextGenerationError::Runtime)?;
        if self.len > 0 {
            let copy_bytes = self
                .len
                .saturating_mul(self.width)
                .saturating_mul(std::mem::size_of::<f32>());
            let mut submission = backend
                .begin_submission()
                .map_err(ReferenceTextGenerationError::Runtime)?;
            submission
                .copy_buffer_region(&self.key_cache, 0, &new_key_cache, 0, copy_bytes)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            submission
                .copy_buffer_region(&self.value_cache, 0, &new_value_cache, 0, copy_bytes)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            submission
                .commit(psionic_backend_cuda::CudaCommandWait::Completed)
                .map_err(ReferenceTextGenerationError::Runtime)?;
        }
        self.key_cache = new_key_cache;
        self.value_cache = new_value_cache;
        self.capacity_tokens = new_capacity;
        Ok(())
    }

    fn state(&self) -> psionic_runtime::KvCacheState {
        let bytes = self
            .len
            .saturating_mul(self.width)
            .saturating_mul(std::mem::size_of::<f32>());
        psionic_runtime::KvCacheState {
            tokens: self.len,
            bytes: bytes as u64,
            pages: self.len,
        }
    }
}

#[derive(Clone, Debug)]
struct CudaGemma4StepPlan {
    qkv_buffer: CudaBuffer,
    query_buffer: CudaBuffer,
    key_buffer: CudaBuffer,
    value_buffer: CudaBuffer,
    attention_buffer: CudaBuffer,
}

#[derive(Debug)]
struct DistributedCudaGemma4RequestState {
    cache: crate::InMemoryKvCache,
    layer_caches: Vec<Option<CudaGemma4LayerCacheState>>,
    step_plan: CudaGemma4StepPlan,
}

fn gguf_local_blob_open_options() -> LocalBlobOpenOptions {
    LocalBlobOpenOptions::default().with_integrity_policy(BlobIntegrityPolicy::LocalUnverifiedLabel)
}

fn runtime_support_for_descriptor(
    descriptor: &DecoderModelDescriptor,
    family: GgufDecoderFamily,
    supported_backends: Vec<String>,
    unsupported_backends: Vec<String>,
    adapter_runtime: DecoderAdapterRuntimeSupport,
) -> GgufDecoderRuntimeSupport {
    GgufDecoderRuntimeSupport {
        family,
        supported_backends,
        unsupported_backends,
        unsupported_features: Vec::new(),
        quantization_modes: descriptor.weights.quantization_modes.clone(),
        adapter_runtime,
    }
}

fn qwen35_proxy_runtime_support(descriptor: &DecoderModelDescriptor) -> GgufDecoderRuntimeSupport {
    GgufDecoderRuntimeSupport {
        family: GgufDecoderFamily::Qwen35,
        supported_backends: vec![String::from("cpu")],
        unsupported_backends: vec![String::from("cuda"), String::from("metal")],
        unsupported_features: vec![
            String::from("multimodal_inputs"),
            String::from("video_inputs"),
            String::from("tool_calling"),
            String::from("structured_output_fallback"),
            String::from("adapter_serving"),
        ],
        quantization_modes: descriptor.weights.quantization_modes.clone(),
        adapter_runtime: unsupported_adapter_runtime_support(
            "LM-head LoRA serving is currently unsupported on the qwen35 proxy runtime",
        ),
    }
}

fn unsupported_adapter_runtime_support(reason: impl Into<String>) -> DecoderAdapterRuntimeSupport {
    DecoderAdapterRuntimeSupport {
        support_level: String::from("unsupported"),
        import_formats: Vec::new(),
        residency_modes: Vec::new(),
        batching_mode: String::from("not_available"),
        unsupported_reasons: vec![reason.into()],
    }
}

fn dense_adapter_runtime_support() -> DecoderAdapterRuntimeSupport {
    DecoderAdapterRuntimeSupport {
        support_level: String::from("lm_head_lora_cpu"),
        import_formats: vec![String::from("safetensors")],
        residency_modes: vec![
            String::from("hot_swap_overlay"),
            String::from("merged_resident"),
        ],
        batching_mode: String::from("mixed_adapter_bindings_per_request"),
        unsupported_reasons: Vec::new(),
    }
}

fn gemma_cuda_adapter_runtime_support() -> DecoderAdapterRuntimeSupport {
    DecoderAdapterRuntimeSupport {
        support_level: String::from("lm_head_lora_cuda"),
        import_formats: vec![String::from("safetensors")],
        residency_modes: vec![
            String::from("hot_swap_overlay"),
            String::from("merged_resident"),
        ],
        batching_mode: String::from("one_promoted_or_explicit_binding_per_request"),
        unsupported_reasons: Vec::new(),
    }
}

fn validate_gemma_exported_revision(
    descriptor: &DecoderModelDescriptor,
    base_binding: &GemmaE4bServedBaseModelBinding,
    served_artifact: &psionic_runtime::ServedArtifactIdentity,
    exported_artifact: &GemmaE4bCudaAdapterExportedArtifact,
    checkpoint: &GemmaE4bCudaAdapterCheckpoint,
) -> Result<LmHeadLoraAdapterArtifact, ReferenceTextGenerationError> {
    if descriptor.model.family != "gemma4" {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: format!(
                "promoted revision loading requires a gemma4 base model, but the runtime loaded `{}`",
                descriptor.model.family
            ),
        });
    }
    if base_binding.model_id != descriptor.model.model_id {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: format!(
                "promotion targets model `{}`, but the runtime loaded `{}`",
                base_binding.model_id, descriptor.model.model_id
            ),
        });
    }
    if base_binding.base_served_artifact_digest != served_artifact.served_artifact_digest {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: format!(
                "promotion targets served artifact `{}`, but the runtime loaded `{}`",
                base_binding.base_served_artifact_digest, served_artifact.served_artifact_digest
            ),
        });
    }
    if base_binding.hidden_size != descriptor.config.hidden_size {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: format!(
                "promotion targets hidden width {}, but the runtime loaded {}",
                base_binding.hidden_size, descriptor.config.hidden_size
            ),
        });
    }
    if base_binding.vocab_size() != descriptor.config.vocab_size {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: format!(
                "promotion targets vocabulary width {}, but the runtime loaded {}",
                base_binding.vocab_size(),
                descriptor.config.vocab_size
            ),
        });
    }

    let tokenizer_contract_digest = base_binding.tokenizer.stable_digest();
    if exported_artifact.tokenizer_contract_digest != tokenizer_contract_digest {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "exported adapter tokenizer contract does not match the served base binding",
            ),
        });
    }
    if checkpoint.tokenizer_contract_digest != tokenizer_contract_digest {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "checkpoint tokenizer contract does not match the served base binding",
            ),
        });
    }
    if exported_artifact.contract_digest != checkpoint.contract_digest {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "exported adapter contract digest does not match the checkpoint contract digest",
            ),
        });
    }
    if exported_artifact.compatibility_digest != checkpoint.compatibility_digest {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "exported adapter compatibility digest does not match the checkpoint compatibility digest",
            ),
        });
    }
    if checkpoint.base_served_artifact_digest != base_binding.base_served_artifact_digest {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "checkpoint base served artifact digest does not match the served base binding",
            ),
        });
    }
    if checkpoint.checkpoint_digest != checkpoint.stable_digest() {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "checkpoint digest does not match the serialized checkpoint payload",
            ),
        });
    }
    if exported_artifact.adapter_identity.base_model_id != base_binding.model_id
        || exported_artifact.adapter_identity.base_model_revision
            != base_binding.base_model_revision
        || exported_artifact
            .adapter_identity
            .base_served_artifact_digest
            != base_binding.base_served_artifact_digest
    {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "exported adapter base-model identity does not match the served base binding",
            ),
        });
    }
    if exported_artifact.adapter_artifact_digest
        != exported_artifact.adapter_identity.artifact_digest
    {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "exported adapter artifact digest does not match the embedded adapter identity",
            ),
        });
    }
    if exported_artifact.adapter_identity_digest
        != exported_artifact.adapter_identity.stable_digest()
    {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
            reason: String::from(
                "exported adapter identity digest does not match the adapter identity payload",
            ),
        });
    }

    let adapter = exported_artifact
        .load_lm_head_lora_artifact()
        .map_err(
            |error| ReferenceTextGenerationError::UnsupportedAdapterBinding {
                binding_id: exported_artifact.adapter_identity.adapter_id.clone(),
                reason: error.to_string(),
            },
        )?;
    validate_lm_head_lora_adapter(descriptor, &adapter)?;
    Ok(adapter)
}

fn gemma_served_revision_identity(
    binding: &AdapterServingBinding,
    exported_artifact: &GemmaE4bCudaAdapterExportedArtifact,
    checkpoint: &GemmaE4bCudaAdapterCheckpoint,
    activated_at_ms: u64,
) -> ServedModelRevisionIdentity {
    let revision_id = stable_digest(
        b"psionic_gemma_served_revision|",
        &(
            binding.served_adapter_digest.as_str(),
            checkpoint.checkpoint_id.as_str(),
            checkpoint.checkpoint_digest.as_str(),
            exported_artifact.adapter_identity_digest.as_str(),
            exported_artifact.adapter_artifact_digest.as_str(),
        ),
    );
    ServedModelRevisionIdentity {
        revision_id,
        binding_id: binding.binding_id.clone(),
        served_adapter_digest: binding.served_adapter_digest.clone(),
        base_model_id: binding.base_model_id.clone(),
        base_model_revision: binding.base_model_revision.clone(),
        base_served_artifact_digest: binding.base_served_artifact_digest.clone(),
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        checkpoint_digest: checkpoint.checkpoint_digest.clone(),
        contract_digest: exported_artifact.contract_digest.clone(),
        compatibility_digest: exported_artifact.compatibility_digest.clone(),
        adapter_id: exported_artifact.adapter_identity.adapter_id.clone(),
        adapter_revision: exported_artifact.adapter_identity.adapter_revision.clone(),
        adapter_identity_digest: exported_artifact.adapter_identity_digest.clone(),
        adapter_artifact_digest: exported_artifact.adapter_artifact_digest.clone(),
        activated_at_ms,
    }
}

fn validate_adapter_identity(
    descriptor: &DecoderModelDescriptor,
    family: GgufDecoderFamily,
    served_artifact_digest: &str,
    identity: &AdapterArtifactIdentity,
) -> Result<(), ReferenceTextGenerationError> {
    if !matches!(
        family,
        GgufDecoderFamily::Llama | GgufDecoderFamily::Qwen | GgufDecoderFamily::Mistral
    ) {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: identity.adapter_id.clone(),
            reason: format!("decoder family `{family:?}` does not support LM-head LoRA serving"),
        });
    }
    if identity.base_model_id != descriptor.model.model_id {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: identity.adapter_id.clone(),
            reason: format!(
                "adapter targets base model `{}`, but the loaded model is `{}`",
                identity.base_model_id, descriptor.model.model_id
            ),
        });
    }
    if identity.base_model_revision != descriptor.model.revision {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: identity.adapter_id.clone(),
            reason: format!(
                "adapter targets base revision `{}`, but the loaded model is `{}`",
                identity.base_model_revision, descriptor.model.revision
            ),
        });
    }
    if identity.base_served_artifact_digest != served_artifact_digest {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: identity.adapter_id.clone(),
            reason: format!(
                "adapter targets served artifact `{}`, but the loaded model is `{served_artifact_digest}`",
                identity.base_served_artifact_digest
            ),
        });
    }
    if identity.target_family != AdapterTargetFamily::DecoderComposite {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: identity.adapter_id.clone(),
            reason: format!(
                "adapter target family `{:?}` is unsupported; only `decoder_composite` LM-head LoRA bindings are implemented",
                identity.target_family
            ),
        });
    }
    Ok(())
}

fn validate_lm_head_lora_adapter(
    descriptor: &DecoderModelDescriptor,
    adapter: &LmHeadLoraAdapterArtifact,
) -> Result<(), ReferenceTextGenerationError> {
    if adapter.hidden_size != descriptor.config.hidden_size {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: adapter.identity.adapter_id.clone(),
            reason: format!(
                "adapter hidden width {} does not match model hidden width {}",
                adapter.hidden_size, descriptor.config.hidden_size
            ),
        });
    }
    if adapter.vocab_size != descriptor.config.vocab_size {
        return Err(ReferenceTextGenerationError::UnsupportedAdapterBinding {
            binding_id: adapter.identity.adapter_id.clone(),
            reason: format!(
                "adapter vocab width {} does not match model vocab width {}",
                adapter.vocab_size, descriptor.config.vocab_size
            ),
        });
    }
    Ok(())
}

fn digest_dense_gguf_plan(
    descriptor: &DecoderModelDescriptor,
    metadata: &GgufDecoderFamilyMetadata,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(descriptor.model.model_id.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.model.revision.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.weights.digest.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.model.family.as_bytes());
    hasher.update(b"|");
    hasher.update(metadata.architecture.as_bytes());
    hasher.update(b"|dense-gguf-cpu|v1");
    hex::encode(hasher.finalize())
}

fn digest_gemma4_cuda_plan(
    descriptor: &DecoderModelDescriptor,
    metadata: &GgufDecoderFamilyMetadata,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(descriptor.model.model_id.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.model.revision.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.weights.digest.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.model.family.as_bytes());
    hasher.update(b"|");
    hasher.update(metadata.architecture.as_bytes());
    hasher.update(b"|gemma4-native-cuda|v1");
    hex::encode(hasher.finalize())
}

fn digest_gemma4_metal_plan(
    descriptor: &DecoderModelDescriptor,
    metadata: &GgufDecoderFamilyMetadata,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(descriptor.model.model_id.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.model.revision.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.weights.digest.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.model.family.as_bytes());
    hasher.update(b"|");
    hasher.update(metadata.architecture.as_bytes());
    hasher.update(b"|gemma4-native-metal|v1");
    hex::encode(hasher.finalize())
}

fn digest_qwen35_proxy_plan(
    descriptor: &DecoderModelDescriptor,
    metadata: &GgufDecoderFamilyMetadata,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(descriptor.model.model_id.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.model.revision.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.weights.digest.as_bytes());
    hasher.update(b"|");
    hasher.update(descriptor.model.family.as_bytes());
    hasher.update(b"|");
    hasher.update(metadata.architecture.as_bytes());
    hasher.update(b"|qwen35-llama-cpp-proxy-cpu|v1");
    hex::encode(hasher.finalize())
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(
        serde_json::to_vec(value)
            .expect("served revision digest serialization should not fail for stable structs"),
    );
    hex::encode(hasher.finalize())
}

fn qwen35_proxy_prompt_json(
    prompt: &GenerationInput,
    vocab_size: usize,
) -> Result<serde_json::Value, ReferenceTextGenerationError> {
    match prompt {
        GenerationInput::Text(text) => {
            if text.is_empty() {
                Err(ReferenceTextGenerationError::EmptyPrompt)
            } else {
                Ok(serde_json::Value::String(text.clone()))
            }
        }
        GenerationInput::Tokens(tokens) => {
            if tokens.as_slice().is_empty() {
                return Err(ReferenceTextGenerationError::EmptyPrompt);
            }
            let mut values = Vec::with_capacity(tokens.as_slice().len());
            for token in tokens.as_slice() {
                let raw = token.as_u32();
                if raw as usize >= vocab_size {
                    return Err(ReferenceTextGenerationError::InvalidToken {
                        token: raw,
                        vocab_size,
                    });
                }
                values.push(serde_json::json!(raw));
            }
            Ok(serde_json::Value::Array(values))
        }
    }
}

fn build_qwen35_proxy_generation_response(
    request: &GenerationRequest,
    descriptor: &DecoderModelDescriptor,
    plan_digest: &str,
    memory_plan: &psionic_runtime::ModelMemoryPlan,
    residency_snapshot: psionic_runtime::MemoryResidencySnapshot,
    load_duration_ns: u64,
    total_duration_ns: u64,
    upstream: Qwen35ProxyCompletionResponse,
) -> Result<GenerationResponse, ReferenceTextGenerationError> {
    let output_tokens =
        TokenSequence::new(upstream.tokens.into_iter().map(TokenId).collect::<Vec<_>>());
    let termination = if upstream.truncated {
        crate::TerminationReason::ContextLimit
    } else {
        match upstream.stop_type.as_str() {
            "limit" => crate::TerminationReason::MaxOutputTokens,
            "eos" | "word" | "none" | "" => crate::TerminationReason::EndOfSequence,
            _ => crate::TerminationReason::EndOfSequence,
        }
    };
    let metrics = crate::GenerationMetrics {
        total_duration_ns: Some(total_duration_ns),
        load_duration_ns: Some(load_duration_ns),
        prompt_eval_count: Some(upstream.tokens_evaluated),
        prompt_eval_duration_ns: None,
        context_window: None,
        eval_count: Some(output_tokens.len()),
        eval_duration_ns: None,
        time_to_first_token_ns: None,
        inter_token_latency_ns: None,
        kv_cache: None,
        kv_residency: None,
        kv_cache_encoding: None,
        prefix_tokens_reused: None,
        termination_detail: None,
        gpt_oss_perf: None,
        qwen35_cuda_decode: None,
    };
    let provenance = crate::GenerationProvenance {
        served_artifact: crate::served_artifact_identity_for_decoder_backend(
            descriptor,
            "cpu",
            &[],
        ),
        adapter_serving: None,
        served_revision: None,
        execution_plan_digest: plan_digest.to_string(),
        cluster_execution: None,
        load_state: crate::GenerationLoadState::Warm,
        isolation_policy: psionic_runtime::LocalServingIsolationPolicy::subprocess_runtime(),
        streaming_policy: None,
        memory_plan: Some(memory_plan.clone()),
        residency_policy: Some(psionic_runtime::ModelResidencyPolicy::default()),
        residency_snapshot: Some(residency_snapshot),
        kv_cache_policy: None,
        kv_cache_encoding_policy: None,
        kv_ownership: None,
        prefix_cache_control: Some(request.prefix_cache_control.clone()),
        prefix_cache_state: None,
        prefix_cache_refusal_reason: None,
        prefix_cache_policy: None,
        prefix_cache_identity: None,
        compile_path: None,
        delivery_proof: None,
        cache_observations: Vec::new(),
        scheduler: None,
        structured_output: None,
        psion_served_evidence: None,
        psion_served_output_claim_posture: None,
    };
    Ok(GenerationResponse::new(
        request,
        None,
        output_tokens,
        upstream.content,
        metrics.prompt_eval_count.unwrap_or_default(),
        0,
        termination,
    )
    .with_metrics_and_provenance(metrics, provenance))
}

fn reserve_proxy_port() -> Result<u16, ReferenceTextGenerationError> {
    let listener =
        std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(qwen35_proxy_runtime_error)?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(qwen35_proxy_runtime_error)
}

fn qwen35_llama_server_bin() -> String {
    env::var("PSIONIC_LLAMA_SERVER_BIN").unwrap_or_else(|_| {
        if cfg!(target_os = "macos") {
            String::from("/Users/christopherdavid/code/llama.cpp/build/bin/llama-server")
        } else {
            String::from("/home/christopherdavid/code/llama.cpp/build/bin/llama-server")
        }
    })
}

fn qwen35_proxy_runtime_error(error: impl std::fmt::Display) -> ReferenceTextGenerationError {
    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(error.to_string()))
}

fn load_dense_vector(artifact: &GgufBlobArtifact, name: &str) -> Result<Vec<f32>, ModelLoadError> {
    artifact
        .load_tensor(name)?
        .values()
        .map(|values| values.into_owned())
}

fn load_dense_matrix(
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<DenseMatrix, ModelLoadError> {
    let tensor = artifact.load_tensor(name)?;
    let [rows, columns] = tensor.metadata().shape.dims() else {
        return Err(ModelLoadError::InvalidTensorShape {
            name: tensor.metadata().name.clone(),
            expected: vec![0, 0],
            actual: tensor.metadata().shape.dims().to_vec(),
        });
    };
    Ok(DenseMatrix {
        rows: *rows,
        columns: *columns,
        values: tensor.values()?.into_owned(),
    })
}

fn load_dense_rank2_flat(
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<Vec<f32>, ModelLoadError> {
    artifact
        .load_tensor(name)?
        .values()
        .map(|values| values.into_owned())
}

fn load_quantized_expert_tensor(
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<QuantizedExpertTensor, ModelLoadError> {
    let storage = artifact.paged_tensor(name)?;
    let metadata = storage.metadata();
    let tensor_name = metadata.name.clone();
    let dims = metadata.shape.dims().to_vec();
    let quantization = metadata.quantization;
    let layout = metadata.quantized_layout;
    let [outer, rows, columns] = dims.as_slice() else {
        return Err(ModelLoadError::InvalidTensorShape {
            name: tensor_name,
            expected: vec![0, 0, 0],
            actual: dims,
        });
    };
    let layout = layout.ok_or_else(|| ModelLoadError::UnsupportedTensorDType {
        name: tensor_name,
        dtype: String::from("quantized"),
    })?;
    let row_byte_len = quantized_row_byte_len(&metadata.shape, layout).map_err(|_| {
        ModelLoadError::InvalidQuantizedTensorShape {
            quantization,
            shape: dims.clone(),
        }
    })?;
    Ok(QuantizedExpertTensor {
        storage,
        mode: quantization,
        outer: *outer,
        rows: *rows,
        columns: *columns,
        row_byte_len,
    })
}

fn load_optional_dense_vector(
    artifact: &GgufBlobArtifact,
    name: Option<&str>,
) -> Result<Option<Vec<f32>>, ModelLoadError> {
    name.map(|name| load_dense_vector(artifact, name))
        .transpose()
}

fn load_optional_dense_scalar(
    artifact: &GgufBlobArtifact,
    name: Option<&str>,
) -> Result<Option<f32>, ModelLoadError> {
    load_optional_dense_vector(artifact, name)?
        .map(|values| match values.as_slice() {
            [value] => Ok(*value),
            actual => Err(ModelLoadError::InvalidTensorShape {
                name: name.expect("scalar tensor name").to_string(),
                expected: vec![1],
                actual: vec![actual.len()],
            }),
        })
        .transpose()
}

fn load_named_optional_dense_vector(
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<Option<Vec<f32>>, ModelLoadError> {
    artifact
        .content()
        .tensor_info(name)
        .map(|_| load_dense_vector(artifact, name))
        .transpose()
}

fn load_first_named_optional_dense_vector(
    artifact: &GgufBlobArtifact,
    names: &[String],
) -> Result<Option<Vec<f32>>, ModelLoadError> {
    for name in names {
        if let Some(values) = load_named_optional_dense_vector(artifact, name.as_str())? {
            return Ok(Some(values));
        }
    }
    Ok(None)
}

fn add_expert_bias_in_place(values: &mut [f32], bias: &[f32], expert_index: usize, width: usize) {
    let start = expert_index.saturating_mul(width);
    let end = start.saturating_add(width).min(bias.len());
    add_bias_in_place(values, &bias[start..end]);
}

fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    indices.truncate(k.min(indices.len()));
    indices
}

fn softmax_selected(values: &[f32], indices: &[usize]) -> Vec<f32> {
    let selected = indices
        .iter()
        .copied()
        .map(|index| values[index])
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Vec::new();
    }
    let max_value = selected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = selected
        .iter()
        .copied()
        .map(|value| (value - max_value).exp())
        .collect::<Vec<_>>();
    let denom = weights.iter().copied().sum::<f32>();
    if denom != 0.0 {
        for weight in &mut weights {
            *weight /= denom;
        }
    }
    weights
}

#[derive(Clone, Debug)]
struct SparseMoeHostProjection {
    values: Vec<f32>,
    kernel_count: usize,
    bytes_moved: u64,
}

fn evaluate_sparse_moe_host(
    family_metadata: &GgufDecoderFamilyMetadata,
    router_input: &[f32],
    expert_input: &[f32],
    router: &DenseMatrix,
    router_bias: Option<&[f32]>,
    gate_experts: &QuantizedExpertTensor,
    gate_bias: Option<&[f32]>,
    up_experts: Option<&QuantizedExpertTensor>,
    up_bias: Option<&[f32]>,
    down_experts: &QuantizedExpertTensor,
    down_bias: Option<&[f32]>,
    down_scale: Option<&[f32]>,
    active_expert_count: usize,
) -> Result<SparseMoeHostProjection, ReferenceTextGenerationError> {
    let mut router_logits = Vec::new();
    router
        .matvec(router_input, &mut router_logits)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    if let Some(bias) = router_bias {
        add_bias_in_place(&mut router_logits, bias);
    }
    let selected = top_k_indices(router_logits.as_slice(), active_expert_count.max(1));
    let routing = softmax_selected(router_logits.as_slice(), selected.as_slice());
    let mut moe_out = vec![0.0; down_experts.rows];
    let mut gate = Vec::new();
    let mut up = Vec::new();
    let mut gate_up = Vec::new();
    let mut expert = Vec::new();
    let mut per_expert_kernels = 3usize;
    let mut bytes_moved = (router
        .values
        .len()
        .saturating_mul(std::mem::size_of::<f32>())
        + gate_experts.byte_length()
        + down_experts.byte_length()) as u64;
    if let Some(up_experts) = up_experts {
        bytes_moved = bytes_moved.saturating_add(up_experts.byte_length() as u64);
    } else {
        per_expert_kernels = 2;
    }
    for (selected_index, expert_index) in selected.iter().copied().enumerate() {
        let activated = if let Some(up_experts) = up_experts {
            gate_experts
                .expert_matvec(expert_index, expert_input, &mut gate)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            if let Some(bias) = gate_bias {
                add_expert_bias_in_place(
                    gate.as_mut_slice(),
                    bias,
                    expert_index,
                    gate_experts.rows,
                );
            }
            up_experts
                .expert_matvec(expert_index, expert_input, &mut up)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            if let Some(bias) = up_bias {
                add_expert_bias_in_place(up.as_mut_slice(), bias, expert_index, up_experts.rows);
            }
            feed_forward_activation(family_metadata, gate.as_slice(), up.as_slice())
        } else {
            gate_experts
                .expert_matvec(expert_index, expert_input, &mut gate_up)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            if let Some(bias) = gate_bias {
                add_expert_bias_in_place(
                    gate_up.as_mut_slice(),
                    bias,
                    expert_index,
                    gate_experts.rows,
                );
            }
            if gate_up.len() % 2 != 0 || gate_up.len() / 2 != down_experts.columns {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "sparse fused gate_up width {} does not match expected twice expert width {}",
                        gate_up.len(),
                        down_experts.columns
                    )),
                ));
            }
            let split = gate_up.len() / 2;
            let (gate, up) = gate_up.split_at(split);
            feed_forward_activation(family_metadata, gate, up)
        };
        down_experts
            .expert_matvec(expert_index, activated.as_slice(), &mut expert)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        if let Some(bias) = down_bias {
            add_expert_bias_in_place(expert.as_mut_slice(), bias, expert_index, down_experts.rows);
        }
        if let Some(scales) = down_scale {
            scale_in_place(
                expert.as_mut_slice(),
                scales.get(expert_index).copied().unwrap_or(1.0),
            );
        }
        let route = routing[selected_index];
        for (dst, value) in moe_out.iter_mut().zip(expert.iter().copied()) {
            *dst += value * route;
        }
    }
    Ok(SparseMoeHostProjection {
        values: moe_out,
        kernel_count: 1 + selected.len().saturating_mul(per_expert_kernels),
        bytes_moved,
    })
}

fn load_named_optional_projection_matrix(
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<Option<ProjectionMatrix>, ModelLoadError> {
    artifact
        .content()
        .tensor_info(name)
        .map(|_| ProjectionMatrix::load(artifact, name))
        .transpose()
}

fn load_named_optional_cuda_quantized_projection_matrix(
    backend: &mut CudaBackend,
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<Option<CudaQuantizedProjectionMatrix>, ReferenceTextGenerationError> {
    artifact
        .content()
        .tensor_info(name)
        .map(|_| CudaQuantizedProjectionMatrix::load(backend, artifact, name))
        .transpose()
}

fn load_named_optional_cuda_gemma4_projection_matrix(
    backend: &mut CudaBackend,
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<Option<CudaGemma4PerLayerProjectionMatrix>, ReferenceTextGenerationError> {
    artifact
        .content()
        .tensor_info(name)
        .map(|_| CudaGemma4PerLayerProjectionMatrix::load(backend, artifact, name))
        .transpose()
}

fn load_named_optional_metal_quantized_projection_matrix(
    backend: &mut MetalBackend,
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<Option<MetalQuantizedProjectionMatrix>, ReferenceTextGenerationError> {
    artifact
        .content()
        .tensor_info(name)
        .map(|_| MetalQuantizedProjectionMatrix::load(backend, artifact, name))
        .transpose()
}

fn load_named_optional_metal_gemma4_projection_matrix(
    backend: &mut MetalBackend,
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<Option<MetalGemma4PerLayerProjectionMatrix>, ReferenceTextGenerationError> {
    artifact
        .content()
        .tensor_info(name)
        .map(|_| MetalGemma4PerLayerProjectionMatrix::load(backend, artifact, name))
        .transpose()
}

fn load_named_optional_metal_static_vector(
    backend: &mut MetalBackend,
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<Option<MetalStaticVector>, ReferenceTextGenerationError> {
    artifact
        .content()
        .tensor_info(name)
        .map(|_| load_metal_static_vector(backend, artifact, name))
        .transpose()
}

fn family_fact_usize(family_metadata: &GgufDecoderFamilyMetadata, key: &str) -> Option<usize> {
    family_metadata
        .family_facts
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn family_fact_f32(family_metadata: &GgufDecoderFamilyMetadata, key: &str) -> Option<f32> {
    family_metadata
        .family_facts
        .get(key)
        .and_then(GgufMetadataValue::as_f32)
}

fn required_tensor_name<'a>(name: Option<&'a str>, field: &str) -> Result<&'a str, ModelLoadError> {
    name.ok_or_else(|| ModelLoadError::ArtifactFormat {
        format: String::from("gguf"),
        message: format!("missing required dense gguf tensor layout field `{field}`"),
    })
}

fn model_load_runtime_error(error: ModelLoadError) -> crate::RuntimeError {
    crate::RuntimeError::Backend(error.to_string())
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    #[cfg(target_os = "macos")]
    let mean_square = macos_sum_squares(input) / input.len() as f32;
    #[cfg(not(target_os = "macos"))]
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + epsilon).sqrt().recip();
    input
        .iter()
        .zip(weight.iter())
        .map(|(value, weight)| value * scale * weight)
        .collect()
}

fn rms_norm_unit(input: &[f32], epsilon: f32) -> Vec<f32> {
    #[cfg(target_os = "macos")]
    let mean_square = macos_sum_squares(input) / input.len() as f32;
    #[cfg(not(target_os = "macos"))]
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + epsilon).sqrt().recip();
    input.iter().map(|value| value * scale).collect()
}

fn rms_norm_in_place(values: &mut [f32], weight: &[f32], epsilon: f32) {
    #[cfg(target_os = "macos")]
    {
        let mean_square = macos_sum_squares(values) / values.len() as f32;
        let scale = (mean_square + epsilon).sqrt().recip();
        unsafe {
            vDSP_vsmul(
                values.as_ptr(),
                1,
                &scale,
                values.as_mut_ptr(),
                1,
                values.len(),
            );
            vDSP_vmul(
                values.as_ptr(),
                1,
                weight.as_ptr(),
                1,
                values.as_mut_ptr(),
                1,
                values.len(),
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mean_square =
            values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
        let scale = (mean_square + epsilon).sqrt().recip();
        for (value, weight) in values.iter_mut().zip(weight.iter().copied()) {
            *value *= scale * weight;
        }
    }
}

fn per_head_rms_norm(
    input: &[f32],
    head_count: usize,
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let mut normalized = vec![0.0_f32; input.len()];
    for head_index in 0..head_count {
        let start = head_index.saturating_mul(head_dim);
        let end = start.saturating_add(head_dim);
        if end > input.len() {
            break;
        }
        let input_head = &input[start..end];
        let output_head = &mut normalized[start..end];
        let mean_square =
            input_head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let scale = (mean_square + epsilon).sqrt().recip();
        for ((out, value), norm) in output_head
            .iter_mut()
            .zip(input_head.iter().copied())
            .zip(weight.iter().copied())
        {
            *out = value * scale * norm;
        }
    }
    normalized
}

fn per_head_rms_norm_in_place(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
) {
    for head in values.chunks_exact_mut(head_dim).take(head_count) {
        #[cfg(target_os = "macos")]
        {
            let mean_square = macos_sum_squares(head) / head_dim as f32;
            let scale = (mean_square + epsilon).sqrt().recip();
            unsafe {
                vDSP_vsmul(head.as_ptr(), 1, &scale, head.as_mut_ptr(), 1, head.len());
                vDSP_vmul(
                    head.as_ptr(),
                    1,
                    weight.as_ptr(),
                    1,
                    head.as_mut_ptr(),
                    1,
                    head.len(),
                );
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let mean_square = head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
            let scale = (mean_square + epsilon).sqrt().recip();
            for (value, norm) in head.iter_mut().zip(weight.iter().copied()) {
                *value *= scale * norm;
            }
        }
    }
}

fn per_head_rms_norm_unit(
    input: &[f32],
    head_count: usize,
    head_dim: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut normalized = vec![0.0_f32; input.len()];
    for head_index in 0..head_count {
        let start = head_index.saturating_mul(head_dim);
        let end = start.saturating_add(head_dim);
        if end > input.len() {
            break;
        }
        let input_head = &input[start..end];
        let output_head = &mut normalized[start..end];
        let mean_square =
            input_head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let scale = (mean_square + epsilon).sqrt().recip();
        for (out, value) in output_head.iter_mut().zip(input_head.iter().copied()) {
            *out = value * scale;
        }
    }
    normalized
}

fn per_head_rms_norm_unit_in_place(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    epsilon: f32,
) {
    for head in values.chunks_exact_mut(head_dim).take(head_count) {
        #[cfg(target_os = "macos")]
        {
            let mean_square = macos_sum_squares(head) / head_dim as f32;
            let scale = (mean_square + epsilon).sqrt().recip();
            unsafe {
                vDSP_vsmul(head.as_ptr(), 1, &scale, head.as_mut_ptr(), 1, head.len());
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let mean_square = head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
            let scale = (mean_square + epsilon).sqrt().recip();
            for value in head {
                *value *= scale;
            }
        }
    }
}

fn add_vectors(left: &[f32], right: &[f32]) -> Result<Vec<f32>, crate::RuntimeError> {
    if left.len() != right.len() {
        return Err(crate::RuntimeError::Backend(format!(
            "vector width mismatch: left={} right={}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left + right)
        .collect())
}

fn add_vectors_in_place(left: &mut [f32], right: &[f32]) -> Result<(), crate::RuntimeError> {
    if left.len() != right.len() {
        return Err(crate::RuntimeError::Backend(format!(
            "vector width mismatch: left={} right={}",
            left.len(),
            right.len()
        )));
    }
    #[cfg(target_os = "macos")]
    {
        unsafe {
            vDSP_vadd(
                left.as_ptr(),
                1,
                right.as_ptr(),
                1,
                left.as_mut_ptr(),
                1,
                left.len(),
            );
        }
        return Ok(());
    }
    #[cfg(not(target_os = "macos"))]
    {
        for (left, right) in left.iter_mut().zip(right.iter().copied()) {
            *left += right;
        }
        Ok(())
    }
}

fn add_bias_in_place(values: &mut [f32], bias: &[f32]) {
    for (value, bias) in values.iter_mut().zip(bias.iter().copied()) {
        *value += bias;
    }
}

fn multiply_vectors(left: &[f32], right: &[f32]) -> Result<Vec<f32>, crate::RuntimeError> {
    if left.len() != right.len() {
        return Err(crate::RuntimeError::Backend(format!(
            "vector width mismatch: left={} right={}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .collect())
}

fn multiply_vectors_in_place(left: &mut [f32], right: &[f32]) -> Result<(), crate::RuntimeError> {
    if left.len() != right.len() {
        return Err(crate::RuntimeError::Backend(format!(
            "vector width mismatch: left={} right={}",
            left.len(),
            right.len()
        )));
    }
    #[cfg(target_os = "macos")]
    {
        unsafe {
            vDSP_vmul(
                left.as_ptr(),
                1,
                right.as_ptr(),
                1,
                left.as_mut_ptr(),
                1,
                left.len(),
            );
        }
        return Ok(());
    }
    #[cfg(not(target_os = "macos"))]
    {
        for (left, right) in left.iter_mut().zip(right.iter().copied()) {
            *left *= right;
        }
        Ok(())
    }
}

fn scale_in_place(values: &mut [f32], scale: f32) {
    if (scale - 1.0).abs() <= f32::EPSILON {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        unsafe {
            vDSP_vsmul(
                values.as_ptr(),
                1,
                &scale,
                values.as_mut_ptr(),
                1,
                values.len(),
            );
        }
        return;
    }
    #[cfg(not(target_os = "macos"))]
    {
        for value in values {
            *value *= scale;
        }
    }
}

fn input_embedding_scale(family_metadata: &GgufDecoderFamilyMetadata, hidden_size: usize) -> f32 {
    match family_metadata.family {
        GgufDecoderFamily::Gemma4 => (hidden_size as f32).sqrt(),
        _ => 1.0,
    }
}

fn attention_scale(family_metadata: &GgufDecoderFamilyMetadata, head_dim: usize) -> f32 {
    match family_metadata.family {
        GgufDecoderFamily::Gemma4 => 1.0,
        _ => 1.0 / (head_dim as f32).sqrt(),
    }
}

const APPROXIMATE_GELU_TANH_COEFFICIENT: f32 = 0.797_884_6;

#[cfg(target_os = "macos")]
const MACOS_VFORCE_CHUNK_LEN: usize = 1024;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_svesq(input: *const f32, input_stride: isize, output: *mut f32, count: usize);
    fn vDSP_dotpr(
        left: *const f32,
        left_stride: isize,
        right: *const f32,
        right_stride: isize,
        output: *mut f32,
        count: usize,
    );
    fn vDSP_vsmul(
        input: *const f32,
        input_stride: isize,
        scale: *const f32,
        output: *mut f32,
        output_stride: isize,
        count: usize,
    );
    fn vDSP_vmul(
        left: *const f32,
        left_stride: isize,
        right: *const f32,
        right_stride: isize,
        output: *mut f32,
        output_stride: isize,
        count: usize,
    );
    fn vDSP_vadd(
        left: *const f32,
        left_stride: isize,
        right: *const f32,
        right_stride: isize,
        output: *mut f32,
        output_stride: isize,
        count: usize,
    );
    fn vvtanhf(output: *mut f32, input: *const f32, count: *const i32);
}

#[cfg(target_os = "macos")]
fn macos_sum_squares(values: &[f32]) -> f32 {
    let mut output = 0.0_f32;
    // SAFETY: the input and output pointers are valid for the specified element counts.
    unsafe {
        vDSP_svesq(values.as_ptr(), 1, &mut output, values.len());
    }
    output
}

fn approximate_gelu(value: f32) -> f32 {
    let cubic = value * value * value;
    let inner = APPROXIMATE_GELU_TANH_COEFFICIENT * (value + 0.044_715 * cubic);
    0.5 * value * (1.0 + inner.tanh())
}

#[cfg(target_os = "macos")]
fn feed_forward_activation_in_place_macos_gemma4(
    gate: &mut [f32],
    up: &[f32],
) -> Result<(), crate::RuntimeError> {
    if gate.len() != up.len() {
        return Err(crate::RuntimeError::Backend(format!(
            "vector width mismatch: left={} right={}",
            gate.len(),
            up.len()
        )));
    }

    let mut tanh_input = [0.0_f32; MACOS_VFORCE_CHUNK_LEN];
    let mut tanh_output = [0.0_f32; MACOS_VFORCE_CHUNK_LEN];
    for (gate_chunk, up_chunk) in gate
        .chunks_mut(MACOS_VFORCE_CHUNK_LEN)
        .zip(up.chunks(MACOS_VFORCE_CHUNK_LEN))
    {
        let chunk_len = i32::try_from(gate_chunk.len()).map_err(|_| {
            crate::RuntimeError::Backend(String::from(
                "gemma4 macos GELU chunk length does not fit vForce count type",
            ))
        })?;
        for (slot, value) in tanh_input.iter_mut().zip(gate_chunk.iter().copied()) {
            let cubic = value * value * value;
            *slot = APPROXIMATE_GELU_TANH_COEFFICIENT * (value + 0.044_715 * cubic);
        }
        // SAFETY: the buffers are valid for `chunk_len` contiguous `f32` values.
        unsafe {
            vvtanhf(tanh_output.as_mut_ptr(), tanh_input.as_ptr(), &chunk_len);
        }
        for ((gate, up), tanh_value) in gate_chunk
            .iter_mut()
            .zip(up_chunk.iter().copied())
            .zip(tanh_output.iter().copied())
        {
            *gate = 0.5 * *gate * (1.0 + tanh_value) * up;
        }
    }
    Ok(())
}

fn gelu_glu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(gate, up)| approximate_gelu(*gate) * *up)
        .collect()
}

fn silu_glu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(gate, up)| {
            let activated = *gate / (1.0 + (-*gate).exp());
            activated * *up
        })
        .collect()
}

fn feed_forward_activation(
    family_metadata: &GgufDecoderFamilyMetadata,
    gate: &[f32],
    up: &[f32],
) -> Vec<f32> {
    match family_metadata.family {
        GgufDecoderFamily::Gemma4 => gelu_glu(gate, up),
        _ => silu_glu(gate, up),
    }
}

fn feed_forward_activation_in_place(
    family_metadata: &GgufDecoderFamilyMetadata,
    gate: &mut [f32],
    up: &[f32],
) -> Result<(), crate::RuntimeError> {
    if gate.len() != up.len() {
        return Err(crate::RuntimeError::Backend(format!(
            "vector width mismatch: left={} right={}",
            gate.len(),
            up.len()
        )));
    }
    match family_metadata.family {
        GgufDecoderFamily::Gemma4 => {
            #[cfg(target_os = "macos")]
            {
                feed_forward_activation_in_place_macos_gemma4(gate, up)
            }
            #[cfg(not(target_os = "macos"))]
            {
                for (gate, up) in gate.iter_mut().zip(up.iter().copied()) {
                    *gate = approximate_gelu(*gate) * up;
                }
                Ok(())
            }
        }
        _ => {
            for (gate, up) in gate.iter_mut().zip(up.iter().copied()) {
                let activated = *gate / (1.0 + (-*gate).exp());
                *gate = activated * up;
            }
            Ok(())
        }
    }
}

fn apply_final_logit_softcapping_in_place(logits: &mut [f32], softcap: Option<f32>) {
    let Some(softcap) = softcap.filter(|softcap| *softcap > 0.0) else {
        return;
    };
    for logit in logits {
        *logit = (*logit / softcap).tanh() * softcap;
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let mut output = 0.0_f32;
            vDSP_dotpr(left.as_ptr(), 1, right.as_ptr(), 1, &mut output, left.len());
            output
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        left.iter()
            .zip(right.iter())
            .map(|(left, right)| left * right)
            .sum()
    }
}

fn axpy(destination: &mut [f32], source: &[f32], alpha: f32) {
    for (destination, source) in destination.iter_mut().zip(source.iter().copied()) {
        *destination += source * alpha;
    }
}

fn accumulate_online_softmax_value(
    destination: &mut [f32],
    source: &[f32],
    logit: f32,
    max_logit: &mut f32,
    denom: &mut f32,
) {
    if *max_logit == f32::NEG_INFINITY {
        *max_logit = logit;
        *denom = 1.0;
        destination.copy_from_slice(source);
        return;
    }

    if logit <= *max_logit {
        let weight = (logit - *max_logit).exp();
        *denom += weight;
        axpy(destination, source, weight);
        return;
    }

    let rescale = (*max_logit - logit).exp();
    scale_in_place(destination, rescale);
    *denom = *denom * rescale + 1.0;
    *max_logit = logit;
    axpy(destination, source, 1.0);
}

fn attend_impl(
    _layer_index: usize,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    cache: &crate::InMemoryKvCache,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    layer_offset: usize,
    sliding_window: Option<usize>,
    attention_scale: f32,
) -> Vec<f32> {
    let group_size = head_count / kv_head_count.max(1);

    let cached_entries = if let Some(window) = sliding_window {
        let retained = window.saturating_sub(1);
        let start = cache.entries().len().saturating_sub(retained);
        &cache.entries()[start..]
    } else {
        cache.entries()
    };

    let mut output = vec![0.0; head_count.saturating_mul(head_dim)];
    for head_index in 0..head_count {
        let kv_head_index = if group_size == 0 {
            0
        } else {
            head_index / group_size
        }
        .min(kv_head_count.saturating_sub(1));
        let q = &query[head_index * head_dim..(head_index + 1) * head_dim];
        let local_key = &key[kv_head_index * head_dim..(kv_head_index + 1) * head_dim];
        let local_value = &value[kv_head_index * head_dim..(kv_head_index + 1) * head_dim];

        let output_slice = &mut output[head_index * head_dim..(head_index + 1) * head_dim];
        let mut max_logit = f32::NEG_INFINITY;
        let mut denom = 0.0;
        for entry in cached_entries {
            let start = layer_offset + kv_head_index * head_dim;
            let end = start + head_dim;
            let logit = dot(q, &entry.key[start..end]) * attention_scale;
            accumulate_online_softmax_value(
                output_slice,
                &entry.value[start..end],
                logit,
                &mut max_logit,
                &mut denom,
            );
        }
        let local_logit = dot(q, local_key) * attention_scale;
        accumulate_online_softmax_value(
            output_slice,
            local_value,
            local_logit,
            &mut max_logit,
            &mut denom,
        );
        scale_in_place(output_slice, 1.0 / denom.max(f32::EPSILON));
    }
    output
}

fn dense_attention_geometry(
    descriptor: &DecoderModelDescriptor,
    family_metadata: &GgufDecoderFamilyMetadata,
    layer_index: usize,
    query_width: usize,
    key_width: usize,
    value_width: usize,
    cache_read_offset: usize,
    cache_write_offset: Option<usize>,
    reuse_layer_index: Option<usize>,
) -> Result<DenseAttentionGeometry, ModelLoadError> {
    let head_count = descriptor.config.block.attention.head_count;
    if head_count == 0 || query_width == 0 || query_width % head_count != 0 {
        return Err(ModelLoadError::InvalidGgufMetadata {
            key: format!("{}.attention.head_count", family_metadata.architecture),
            message: format!(
                "layer {layer_index} query width {query_width} is not divisible by attention head count {head_count}"
            ),
        });
    }
    let head_dim = query_width / head_count;
    if key_width == 0 || key_width % head_dim != 0 {
        return Err(ModelLoadError::InvalidTensorShape {
            name: format!("blk.{layer_index}.attn_k.weight"),
            expected: vec![0, descriptor.config.hidden_size],
            actual: vec![key_width, descriptor.config.hidden_size],
        });
    }
    if value_width != key_width {
        return Err(ModelLoadError::UnsupportedGgufDecoderFamilyFeature {
            family: descriptor.model.family.clone(),
            feature: String::from("distinct_layer_value_width"),
        });
    }
    let kv_head_count = key_width / head_dim;
    let sliding_window = layer_sliding_window(family_metadata, layer_index, head_dim);
    let is_swa = gemma4_layer_is_swa(family_metadata, layer_index, head_dim);
    Ok(DenseAttentionGeometry {
        head_count,
        kv_head_count,
        head_dim,
        rotary_dim: layer_rotary_dim(descriptor, family_metadata, is_swa, head_dim),
        rope_theta: layer_rope_theta(family_metadata, is_swa),
        sliding_window,
        cache_read_offset,
        cache_write_offset,
        reuse_layer_index,
    })
}

fn gemma4_layer_is_swa(
    family_metadata: &GgufDecoderFamilyMetadata,
    layer_index: usize,
    head_dim: usize,
) -> bool {
    if family_metadata.family != GgufDecoderFamily::Gemma4 {
        return false;
    }
    let pattern_key = format!(
        "{}.attention.sliding_window_pattern",
        family_metadata.architecture
    );
    if let Some(enabled) = family_metadata
        .family_facts
        .get(pattern_key.as_str())
        .and_then(GgufMetadataValue::as_array)
        .and_then(|values| values.get(layer_index))
        .and_then(GgufMetadataValue::as_bool)
    {
        return enabled;
    }
    family_metadata
        .attention_key_length
        .is_some_and(|full_head_dim| head_dim < full_head_dim)
}

fn gemma4_layer_has_kv(
    family_metadata: &GgufDecoderFamilyMetadata,
    layout: &GgufDecoderLayerTensorLayout,
) -> bool {
    if family_metadata.family != GgufDecoderFamily::Gemma4 {
        return true;
    }
    layout.attention_key_weight.is_some()
}

fn gemma4_layer_kv_head_count(
    descriptor: &DecoderModelDescriptor,
    family_metadata: &GgufDecoderFamilyMetadata,
    layer_index: usize,
) -> usize {
    let default = descriptor
        .config
        .block
        .attention
        .kv_head_count
        .max(1)
        .min(descriptor.config.block.attention.head_count.max(1));
    if family_metadata.family != GgufDecoderFamily::Gemma4 {
        return default;
    }
    let key = format!("{}.attention.head_count_kv", family_metadata.architecture);
    if let Some(kv_head_count) = family_metadata
        .family_facts
        .get(key.as_str())
        .and_then(GgufMetadataValue::as_array)
        .and_then(|values| values.get(layer_index))
        .and_then(GgufMetadataValue::as_u64)
        .filter(|value| *value > 0)
        .and_then(|value| usize::try_from(value).ok())
    {
        return kv_head_count;
    }
    family_fact_usize(family_metadata, key.as_str())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn gemma4_expected_kv_width(
    descriptor: &DecoderModelDescriptor,
    family_metadata: &GgufDecoderFamilyMetadata,
    layer_index: usize,
    head_dim: usize,
) -> usize {
    gemma4_layer_kv_head_count(descriptor, family_metadata, layer_index).saturating_mul(head_dim)
}

fn layer_rope_theta(family_metadata: &GgufDecoderFamilyMetadata, is_swa: bool) -> f32 {
    if family_metadata.family == GgufDecoderFamily::Gemma4 && is_swa {
        return family_fact_f32(
            family_metadata,
            format!("{}.rope.freq_base_swa", family_metadata.architecture).as_str(),
        )
        .unwrap_or(family_metadata.rope_theta);
    }
    family_metadata.rope_theta
}

fn layer_rotary_dim(
    descriptor: &DecoderModelDescriptor,
    family_metadata: &GgufDecoderFamilyMetadata,
    is_swa: bool,
    head_dim: usize,
) -> usize {
    if family_metadata.family == GgufDecoderFamily::Gemma4 {
        let key = if is_swa {
            format!("{}.rope.dimension_count_swa", family_metadata.architecture)
        } else {
            format!("{}.rope.dimension_count", family_metadata.architecture)
        };
        return family_fact_usize(family_metadata, key.as_str())
            .unwrap_or(descriptor.config.block.attention.rotary_dim)
            .min(head_dim);
    }
    descriptor.config.block.attention.rotary_dim.min(head_dim)
}

fn layer_sliding_window(
    family_metadata: &GgufDecoderFamilyMetadata,
    layer_index: usize,
    head_dim: usize,
) -> Option<usize> {
    let default_window = family_metadata.sliding_window?;
    if family_metadata.family != GgufDecoderFamily::Gemma4 {
        return Some(default_window);
    }
    let pattern_key = format!(
        "{}.attention.sliding_window_pattern",
        family_metadata.architecture
    );
    if let Some(enabled) = family_metadata
        .family_facts
        .get(pattern_key.as_str())
        .and_then(GgufMetadataValue::as_array)
        .and_then(|values| values.get(layer_index))
        .and_then(GgufMetadataValue::as_bool)
    {
        return enabled.then_some(default_window);
    }
    family_metadata
        .attention_key_length
        .and_then(|full_head_dim| (head_dim < full_head_dim).then_some(default_window))
}

fn apply_rope_neox(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    metadata: &GgufDecoderFamilyMetadata,
) {
    let rotary_dim = rotary_dim.min(head_dim).max(2);
    let rotary_half = rotary_dim / 2;
    let freq_scale = metadata
        .rope_scaling_factor
        .filter(|value| *value > 0.0)
        .map_or(1.0, |value| 1.0 / value);
    let ext_factor = metadata
        .rope_scaling_factor
        .zip(metadata.rope_original_context_length)
        .filter(|(factor, original)| *factor > 1.0 && *original > 0)
        .map_or(0.0, |_| 1.0);
    let corr_dims = metadata
        .rope_original_context_length
        .map(|original| rope_yarn_corr_dims(rotary_dim, original, rope_theta))
        .unwrap_or([0.0, rotary_dim as f32 - 1.0]);
    let theta_scale = rope_theta.powf(-2.0 / rotary_dim as f32);
    for pair in 0..rotary_half {
        let i0 = pair * 2;
        let freq_factor = freq_factors
            .and_then(|factors| factors.get(pair))
            .copied()
            .filter(|value| *value > 0.0)
            .unwrap_or(1.0);
        let theta_base = position as f32 * theta_scale.powf(pair as f32);
        let (cos_theta, sin_theta) = rope_yarn(
            theta_base / freq_factor,
            freq_scale,
            corr_dims,
            i0,
            ext_factor,
            1.0,
        );
        for head in values.chunks_exact_mut(head_dim).take(head_count) {
            let x0 = head[pair];
            let x1 = head[pair + rotary_half];
            head[pair] = x0 * cos_theta - x1 * sin_theta;
            head[pair + rotary_half] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

fn metal_gemma4_rope_cos_sin_values(
    position: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    freq_factors: Option<&[f32]>,
    metadata: &GgufDecoderFamilyMetadata,
) -> (Vec<f32>, Vec<f32>) {
    let rotary_dim = rotary_dim.min(head_dim).max(2);
    let half_dim = head_dim / 2;
    let freq_scale = metadata
        .rope_scaling_factor
        .filter(|value| *value > 0.0)
        .map_or(1.0, |value| 1.0 / value);
    let ext_factor = metadata
        .rope_scaling_factor
        .zip(metadata.rope_original_context_length)
        .filter(|(factor, original)| *factor > 1.0 && *original > 0)
        .map_or(0.0, |_| 1.0);
    let corr_dims = metadata
        .rope_original_context_length
        .map(|original| rope_yarn_corr_dims(rotary_dim, original, rope_theta))
        .unwrap_or([0.0, rotary_dim as f32 - 1.0]);
    let theta_scale = rope_theta.powf(-2.0 / rotary_dim as f32);
    let mut cos = vec![1.0; half_dim];
    let mut sin = vec![0.0; half_dim];
    for pair in 0..(rotary_dim / 2) {
        let i0 = pair * 2;
        let freq_factor = freq_factors
            .and_then(|factors| factors.get(pair))
            .copied()
            .filter(|value| *value > 0.0)
            .unwrap_or(1.0);
        let theta_base = position as f32 * theta_scale.powf(pair as f32);
        let (cos_theta, sin_theta) = rope_yarn(
            theta_base / freq_factor,
            freq_scale,
            corr_dims,
            i0,
            ext_factor,
            1.0,
        );
        cos[pair] = cos_theta;
        sin[pair] = sin_theta;
    }
    (cos, sin)
}

fn metal_gemma4_rope_freq_factor_values(
    head_dim: usize,
    rotary_dim: usize,
    freq_factors: Option<&[f32]>,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let rotary_half = rotary_dim.min(head_dim).max(2) / 2;
    let mut values = vec![1.0; half_dim];
    for pair in 0..rotary_half {
        values[pair] = freq_factors
            .and_then(|factors| factors.get(pair))
            .copied()
            .filter(|value| *value > 0.0)
            .unwrap_or(1.0);
    }
    values
}

fn metal_gemma4_rope_kernel_params(
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    metadata: &GgufDecoderFamilyMetadata,
) -> (usize, f32, f32, [f32; 2], f32, f32) {
    let rotary_dim = rotary_dim.min(head_dim).max(2);
    let rotary_half = rotary_dim / 2;
    let freq_scale = metadata
        .rope_scaling_factor
        .filter(|value| *value > 0.0)
        .map_or(1.0, |value| 1.0 / value);
    let ext_factor = metadata
        .rope_scaling_factor
        .zip(metadata.rope_original_context_length)
        .filter(|(factor, original)| *factor > 1.0 && *original > 0)
        .map_or(0.0, |_| 1.0);
    let corr_dims = metadata
        .rope_original_context_length
        .map(|original| rope_yarn_corr_dims(rotary_dim, original, rope_theta))
        .unwrap_or([0.0, rotary_dim as f32 - 1.0]);
    let theta_scale = rope_theta.powf(-2.0 / rotary_dim as f32);
    let yarn_mscale = if ext_factor != 0.0 {
        1.0 + 0.1 * (1.0 / freq_scale).ln()
    } else {
        1.0
    };
    (
        rotary_half,
        theta_scale,
        freq_scale,
        corr_dims,
        ext_factor,
        yarn_mscale,
    )
}

fn rope_yarn_corr_dims(n_dims: usize, n_ctx_orig: usize, freq_base: f32) -> [f32; 2] {
    let corr_dim = |n_rot: f32| {
        n_dims as f32
            * ((n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * freq_base.ln()))
    };
    let start = corr_dim(32.0).floor().max(0.0);
    let end = corr_dim(1.0).ceil().min(n_dims.saturating_sub(1) as f32);
    [start, end]
}

fn rope_yarn(
    theta_extrap: f32,
    freq_scale: f32,
    corr_dims: [f32; 2],
    i0: usize,
    ext_factor: f32,
    mscale: f32,
) -> (f32, f32) {
    let theta_interp = freq_scale * theta_extrap;
    let mut theta = theta_interp;
    let mut mscale = mscale;
    if ext_factor != 0.0 {
        let ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }
    (theta.cos() * mscale, theta.sin() * mscale)
}

fn rope_yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = ((i0 / 2) as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

#[derive(Clone, Debug)]
struct CompletedGenerationStream {
    policy: GenerationStreamingPolicy,
    chunk: Option<GenerationStreamChunk>,
    terminal: Option<GenerationStreamTerminal>,
}

impl CompletedGenerationStream {
    fn new(response: GenerationResponse) -> Self {
        let chunk = GenerationStreamChunk {
            request_id: response.request_id.clone(),
            model_id: response.model_id.clone(),
            session_id: response.session_id.clone(),
            output: response.output.clone(),
            cumulative_output_tokens: response.output.tokens.len(),
        };
        let terminal = GenerationStreamTerminal {
            status: GenerationStreamStatus::Succeeded,
            response,
            failure_reason: None,
            diagnostic: None,
        };
        Self {
            policy: default_generation_streaming_policy(),
            chunk: Some(chunk),
            terminal: Some(terminal),
        }
    }
}

impl GenerationEventStream for CompletedGenerationStream {
    fn policy(&self) -> &GenerationStreamingPolicy {
        &self.policy
    }

    fn next_event(&mut self) -> Option<GenerationStreamEvent> {
        if let Some(chunk) = self.chunk.take() {
            return Some(GenerationStreamEvent::Chunk(chunk));
        }
        self.terminal.take().map(GenerationStreamEvent::Terminal)
    }

    fn cancel(&mut self) -> Option<GenerationStreamTerminal> {
        self.chunk.take();
        self.terminal.take()
    }

    fn disconnect(&mut self) -> Option<GenerationStreamTerminal> {
        self.cancel()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CpuGgufServiceKind, CpuGgufTextGenerationService, DistributedGemma4RemoteStepRequest,
        DistributedGemma4RemoteStepResponse, PromotedGemmaRevisionEntry,
        PromotedGemmaRevisionState, attend_impl, axpy,
        decode_distributed_gemma4_remote_step_request,
        decode_distributed_gemma4_remote_step_response, dot,
        encode_distributed_gemma4_remote_step_request,
        encode_distributed_gemma4_remote_step_response, gemma_served_revision_identity,
        validate_gemma_exported_revision,
    };
    use crate::{
        AdapterServingBinding, CompiledWordGenerationModel, GenerationLoadState,
        GenerationModelHandle, GenerationOptions, GenerationProvenance, GenerationRequest,
        GenerationResponse, GenerationStreamEvent, InMemoryKvCache, ReferenceTextGenerationError,
        StreamingTextGenerationExecutor, TerminationReason, TextGenerationExecutor,
    };
    use psionic_adapters::{
        AdapterArtifactFormat, AdapterArtifactIdentity, AdapterArtifactKind, AdapterResidencyMode,
        AdapterTargetFamily,
    };
    use psionic_core::{DType, Device, QuantizationMode, Shape, TensorSpec};
    use psionic_data::{TokenizerDigest, TokenizerFamily};
    use psionic_models::{
        DecoderModelDescriptor, GgufDecoderFamily, GgufMetadataValue, GgufTensorType, TokenId,
        TokenSequence,
    };
    use psionic_runtime::LocalServingIsolationPolicy;
    use psionic_train::{
        FixedBudgetTrainingRun, GEMMA_E4B_CUDA_ADAPTER_CHECKPOINT_SCHEMA_VERSION,
        GEMMA_E4B_CUDA_ADAPTER_TARGET_SET_ID, GEMMA_E4B_FINETUNING_MVP_TRAINING_FAMILY_ID,
        GemmaE4bCudaAdapterCheckpoint, GemmaE4bCudaAdapterExportedArtifact,
        GemmaE4bServedBaseModelBinding, TrainingLoopBudget, TrainingOptimizerConfig,
        TrainingOptimizerResidencyPolicy, TrainingParameterClass, TrainingParameterGroupState,
        TrainingTensorBuffer,
    };
    use safetensors::{Dtype as SafeTensorsDType, serialize, tensor::TensorView};
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        path::Path,
        sync::{Mutex, OnceLock},
    };
    use tempfile::tempdir;

    #[derive(Clone, Debug)]
    struct TestGgufTensor {
        name: String,
        shape: Vec<usize>,
        tensor_type: GgufTensorType,
        bytes: Vec<u8>,
    }

    impl TestGgufTensor {
        fn new(
            name: impl Into<String>,
            shape: Vec<usize>,
            tensor_type: GgufTensorType,
            bytes: Vec<u8>,
        ) -> Self {
            Self {
                name: name.into(),
                shape,
                tensor_type,
                bytes,
            }
        }
    }

    fn reference_dense_attend_impl(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        cache: &crate::InMemoryKvCache,
        head_count: usize,
        kv_head_count: usize,
        head_dim: usize,
        layer_offset: usize,
        sliding_window: Option<usize>,
        attention_scale: f32,
    ) -> Vec<f32> {
        let group_size = head_count / kv_head_count.max(1);
        let cached_entries = if let Some(window) = sliding_window {
            let retained = window.saturating_sub(1);
            let start = cache.entries().len().saturating_sub(retained);
            &cache.entries()[start..]
        } else {
            cache.entries()
        };
        let mut output = vec![0.0; head_count.saturating_mul(head_dim)];
        for head_index in 0..head_count {
            let kv_head_index = if group_size == 0 {
                0
            } else {
                head_index / group_size
            }
            .min(kv_head_count.saturating_sub(1));
            let q = &query[head_index * head_dim..(head_index + 1) * head_dim];
            let local_key = &key[kv_head_index * head_dim..(kv_head_index + 1) * head_dim];
            let local_value = &value[kv_head_index * head_dim..(kv_head_index + 1) * head_dim];

            let mut weights = Vec::with_capacity(cached_entries.len().saturating_add(1));
            for entry in cached_entries {
                let start = layer_offset + kv_head_index * head_dim;
                let end = start + head_dim;
                weights.push(dot(q, &entry.key[start..end]) * attention_scale);
            }
            weights.push(dot(q, local_key) * attention_scale);

            let max_weight = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for weight in &mut weights {
                *weight = (*weight - max_weight).exp();
            }
            let denom = weights.iter().copied().sum::<f32>().max(f32::EPSILON);
            for weight in &mut weights {
                *weight /= denom;
            }

            let output_slice = &mut output[head_index * head_dim..(head_index + 1) * head_dim];
            for (entry, weight) in cached_entries.iter().zip(weights.iter().copied()) {
                let start = layer_offset + kv_head_index * head_dim;
                let end = start + head_dim;
                axpy(output_slice, &entry.value[start..end], weight);
            }
            axpy(output_slice, local_value, *weights.last().unwrap_or(&0.0));
        }
        output
    }

    #[test]
    fn dense_attend_impl_matches_reference_for_grouped_kv_heads() {
        let mut cache = crate::InMemoryKvCache::new(8, 4);
        cache
            .append(
                TokenId(11),
                vec![0.1, -0.2, 0.3, 0.4],
                vec![1.0, 2.0, -1.0, 0.5],
            )
            .expect("first cache append");
        cache
            .append(
                TokenId(12),
                vec![0.6, 0.2, -0.3, 0.8],
                vec![0.25, -0.5, 1.25, 0.75],
            )
            .expect("second cache append");
        let query = vec![0.5, -0.25, -0.1, 0.75, 0.3, 0.2, -0.4, 0.6];
        let key = vec![0.15, 0.45, -0.35, 0.55];
        let value = vec![0.8, -0.3, 1.1, 0.9];

        let expected = reference_dense_attend_impl(
            query.as_slice(),
            key.as_slice(),
            value.as_slice(),
            &cache,
            4,
            2,
            2,
            0,
            None,
            0.5,
        );
        let actual = attend_impl(
            0,
            query.as_slice(),
            key.as_slice(),
            value.as_slice(),
            &cache,
            4,
            2,
            2,
            0,
            None,
            0.5,
        );

        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert!((*expected - *actual).abs() < 1e-6_f32);
        }
    }

    #[test]
    fn dense_attend_impl_matches_reference_with_sliding_window() {
        let mut cache = crate::InMemoryKvCache::new(8, 2);
        cache
            .append(TokenId(21), vec![0.2, -0.1], vec![0.5, 1.5])
            .expect("append cache entry 1");
        cache
            .append(TokenId(22), vec![0.8, 0.4], vec![-0.25, 0.75])
            .expect("append cache entry 2");
        cache
            .append(TokenId(23), vec![-0.3, 0.9], vec![1.25, -0.5])
            .expect("append cache entry 3");
        let query = vec![0.25, -0.4, 0.8, 0.1];
        let key = vec![0.5, -0.25];
        let value = vec![1.0, 0.75];

        let expected = reference_dense_attend_impl(
            query.as_slice(),
            key.as_slice(),
            value.as_slice(),
            &cache,
            2,
            1,
            2,
            0,
            Some(2),
            1.0,
        );
        let actual = attend_impl(
            0,
            query.as_slice(),
            key.as_slice(),
            value.as_slice(),
            &cache,
            2,
            1,
            2,
            0,
            Some(2),
            1.0,
        );

        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert!((*expected - *actual).abs() < 1e-6_f32);
        }
    }

    #[test]
    fn distributed_gemma4_remote_step_response_binary_roundtrips() {
        let response = DistributedGemma4RemoteStepResponse {
            logits: vec![1.25, -3.5, 0.0, 9.75],
            kernel_count: 17,
            bytes_moved: 8192,
        };
        let encoded = encode_distributed_gemma4_remote_step_response(&response)
            .expect("response should encode");
        let decoded = decode_distributed_gemma4_remote_step_response(encoded.as_slice())
            .expect("response should decode");
        assert_eq!(decoded.logits, response.logits);
        assert_eq!(decoded.kernel_count, response.kernel_count);
        assert_eq!(decoded.bytes_moved, response.bytes_moved);
    }

    #[test]
    fn distributed_gemma4_remote_step_request_binary_roundtrips() {
        let request = DistributedGemma4RemoteStepRequest {
            request_id: String::from("req-123"),
            model_id: String::from("gemma4-e4b"),
            token: 42,
            position: 7,
            split_layer: 11,
            input_hidden: vec![1.25, -3.5, 0.0, 9.75],
            forwarded_key: vec![0.5, 1.5, 2.5],
            forwarded_value: vec![-1.0, 3.0],
        };
        let encoded =
            encode_distributed_gemma4_remote_step_request(&request).expect("request should encode");
        let decoded = decode_distributed_gemma4_remote_step_request(encoded.as_slice())
            .expect("request should decode");
        assert_eq!(decoded.request_id, request.request_id);
        assert_eq!(decoded.model_id, request.model_id);
        assert_eq!(decoded.token, request.token);
        assert_eq!(decoded.position, request.position);
        assert_eq!(decoded.split_layer, request.split_layer);
        assert_eq!(decoded.input_hidden, request.input_hidden);
        assert_eq!(decoded.forwarded_key, request.forwarded_key);
        assert_eq!(decoded.forwarded_value, request.forwarded_value);
    }

    #[test]
    fn distributed_gemma4_remote_step_request_binary_rejects_truncated_header() {
        let error = decode_distributed_gemma4_remote_step_request(&[1, 2, 3])
            .expect_err("truncated header should fail");
        assert!(matches!(error, ReferenceTextGenerationError::Runtime(_)));
        assert!(
            error
                .to_string()
                .contains("distributed gemma4 remote step request is truncated")
        );
    }

    #[test]
    fn distributed_gemma4_remote_step_request_binary_rejects_size_mismatch() {
        let mut encoded =
            encode_distributed_gemma4_remote_step_request(&DistributedGemma4RemoteStepRequest {
                request_id: String::from("req"),
                model_id: String::from("model"),
                token: 1,
                position: 2,
                split_layer: 3,
                input_hidden: vec![1.0, 2.0],
                forwarded_key: vec![3.0],
                forwarded_value: vec![4.0],
            })
            .expect("request should encode");
        encoded.pop();
        let error = decode_distributed_gemma4_remote_step_request(encoded.as_slice())
            .expect_err("truncated payload should fail");
        assert!(matches!(error, ReferenceTextGenerationError::Runtime(_)));
        assert!(
            error
                .to_string()
                .contains("distributed gemma4 remote step request length mismatch")
        );
    }

    #[test]
    fn distributed_gemma4_remote_step_response_binary_rejects_truncated_header() {
        let error = decode_distributed_gemma4_remote_step_response(&[1, 2, 3])
            .expect_err("truncated header should fail");
        assert!(matches!(error, ReferenceTextGenerationError::Runtime(_)));
        assert!(
            error
                .to_string()
                .contains("distributed gemma4 remote suffix response is truncated")
        );
    }

    #[test]
    fn distributed_gemma4_remote_step_response_binary_rejects_size_mismatch() {
        let mut encoded =
            encode_distributed_gemma4_remote_step_response(&DistributedGemma4RemoteStepResponse {
                logits: vec![1.0, 2.0],
                kernel_count: 1,
                bytes_moved: 64,
            })
            .expect("response should encode");
        encoded.pop();
        let error = decode_distributed_gemma4_remote_step_response(encoded.as_slice())
            .expect_err("truncated logits should fail");
        assert!(matches!(error, ReferenceTextGenerationError::Runtime(_)));
        assert!(
            error
                .to_string()
                .contains("distributed gemma4 remote suffix response has invalid size")
        );
    }

    #[test]
    fn cpu_gguf_service_executes_llama_family() -> Result<(), Box<dyn std::error::Error>> {
        run_dense_family_case(
            "llama",
            GgufDecoderFamily::Llama,
            dense_llama_metadata("tiny psionic llama"),
            false,
            3,
            4,
            "hello",
        )
    }

    #[test]
    fn cpu_gguf_service_executes_qwen_family() -> Result<(), Box<dyn std::error::Error>> {
        run_dense_family_case(
            "qwen2",
            GgufDecoderFamily::Qwen,
            dense_qwen_metadata("tiny psionic qwen"),
            true,
            2,
            3,
            "hello",
        )
    }

    #[test]
    fn cpu_gguf_service_executes_mistral_family() -> Result<(), Box<dyn std::error::Error>> {
        run_dense_family_case(
            "mistral",
            GgufDecoderFamily::Mistral,
            dense_mistral_metadata("tiny psionic mistral"),
            false,
            3,
            4,
            "hello",
        )
    }

    #[test]
    fn cpu_gguf_service_executes_gemma4_family() -> Result<(), Box<dyn std::error::Error>> {
        run_dense_family_case(
            "gemma4",
            GgufDecoderFamily::Gemma4,
            dense_gemma4_metadata("tiny psionic gemma4"),
            false,
            3,
            4,
            "hello",
        )
    }

    #[test]
    fn cpu_dense_generation_handle_uses_whole_model_cache_width_for_multi_layer_models()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = temp.path().join("gemma4_two_layer_cache_width.gguf");
        write_test_gguf(
            &path,
            dense_gemma4_metadata_with_block_count("tiny psionic gemma4 two layer", 2).as_slice(),
            dense_decoder_tensors_with_layers(false, 3, 4, 2).as_slice(),
        )?;

        let mut service = CpuGgufTextGenerationService::from_gguf_path(&path)?;
        let descriptor = service.model_descriptor().clone();
        {
            let CpuGgufServiceKind::Dense(dense) = &mut service.inner else {
                panic!("expected dense GGUF runtime");
            };
            let model_id = dense.model_descriptor.model.model_id.clone();
            let loaded = dense
                .models
                .active(model_id.as_str())
                .expect("dense model should be active")
                .clone();
            let per_layer_width = loaded.descriptor().config.kv_width();
            assert!(
                loaded.descriptor().config.layer_count > 1,
                "regression fixture must stay multi-layer"
            );
            assert_eq!(
                loaded.cache_width(),
                loaded.descriptor().config.layer_count * per_layer_width
            );
            assert!(loaded.cache_width() > per_layer_width);
        }

        let response = service.generate(&GenerationRequest::new_text(
            String::from("gguf-gemma4-two-layer"),
            descriptor,
            None,
            "hello",
            GenerationOptions::greedy(1),
        ))?;
        assert_eq!(response.output.text, "world");
        Ok(())
    }

    #[test]
    fn cpu_gguf_service_executes_qwen35_proxy_family() -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = qwen35_proxy_test_lock()
            .lock()
            .expect("qwen35 proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let (base_url, shutdown_tx) = runtime.block_on(start_qwen35_proxy_test_server())?;
        let temp = tempdir()?;
        let path = temp.path().join("tiny_qwen35.gguf");
        write_test_gguf(
            &path,
            qwen35_decoder_metadata("tiny psionic qwen35 proxy").as_slice(),
            qwen35_decoder_tensors().as_slice(),
        )?;
        let _proxy_env = ScopedEnvVar::set("PSIONIC_QWEN35_PROXY_BASE_URL", base_url.as_str());

        let mut service = CpuGgufTextGenerationService::from_gguf_path(&path)?;
        let descriptor = service.model_descriptor().clone();
        let request = GenerationRequest::new_text(
            String::from("gguf-qwen35"),
            descriptor.clone(),
            None,
            "hello",
            GenerationOptions::greedy(2),
        );

        let response = service.generate(&request)?;
        let support = service.runtime_support();
        let loaded = service.loaded_model_views();

        assert_eq!(descriptor.model.family, "qwen35");
        assert_eq!(response.output.text, "proxy world");
        assert_eq!(support.family, GgufDecoderFamily::Qwen35);
        assert_eq!(support.supported_backends, vec![String::from("cpu")]);
        assert_eq!(
            support.unsupported_features,
            vec![
                String::from("multimodal_inputs"),
                String::from("video_inputs"),
                String::from("tool_calling"),
                String::from("structured_output_fallback"),
                String::from("adapter_serving"),
            ]
        );
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].summary.family.as_deref(), Some("qwen35"));

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn cpu_gguf_service_streams_generic_family_output() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = temp.path().join("llama_stream.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny psionic llama stream").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let mut service = CpuGgufTextGenerationService::from_gguf_path(&path)?;
        let request = GenerationRequest::new_text(
            String::from("gguf-llama-stream"),
            service.model_descriptor().clone(),
            None,
            "hello",
            GenerationOptions::greedy(1),
        );
        let mut stream = service.generate_stream(&request)?;

        let Some(GenerationStreamEvent::Chunk(chunk)) = stream.next_event() else {
            panic!("expected streamed chunk");
        };
        assert_eq!(chunk.output.text, "world");
        assert_eq!(chunk.cumulative_output_tokens, 1);

        let Some(GenerationStreamEvent::Terminal(terminal)) = stream.next_event() else {
            panic!("expected terminal stream event");
        };
        assert_eq!(terminal.response.output.text, "world");
        assert!(stream.next_event().is_none());
        Ok(())
    }

    #[test]
    fn cpu_gguf_service_serves_lm_head_lora_overlay_and_merge_modes()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = temp.path().join("llama_lora.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny psionic llama lora").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let mut service = CpuGgufTextGenerationService::from_gguf_path(&path)?;
        let descriptor = service.model_descriptor().clone();
        let hidden = final_hidden_for_prompt(&mut service, "hello")?;
        let adapter_path = temp.path().join("lm_head_lora.safetensors");
        write_prompt_specific_lora_adapter(&adapter_path, hidden.as_slice(), 5, 32.0)?;
        let binding = service.register_lm_head_lora_adapter(
            "adapter-psionic",
            &adapter_path,
            sample_lora_identity(&descriptor, AdapterTargetFamily::DecoderComposite),
            1.0,
            AdapterResidencyMode::HotSwapOverlay,
        )?;
        let support = service.runtime_support();
        assert_eq!(support.adapter_runtime.support_level, "lm_head_lora_cpu");
        assert_eq!(
            support.adapter_runtime.import_formats,
            vec![String::from("safetensors")]
        );

        let baseline = service.generate(&GenerationRequest::new_text(
            String::from("llama-baseline"),
            descriptor.clone(),
            None,
            "hello",
            GenerationOptions::greedy(1),
        ))?;
        assert_eq!(baseline.output.text, "world");

        let overlay_request = GenerationRequest::new_text(
            String::from("llama-overlay"),
            descriptor.clone(),
            None,
            "hello",
            GenerationOptions::greedy(1),
        )
        .with_adapter_serving(binding.clone());
        let overlay = service.generate(&overlay_request)?;
        assert_eq!(overlay.output.text, "psionic");
        assert_eq!(
            overlay
                .provenance
                .as_ref()
                .and_then(|value| value.adapter_serving.clone()),
            Some(binding.clone())
        );

        let merged_binding =
            service.merge_adapter_binding(binding.served_adapter_digest.as_str())?;
        let merged = service.generate(
            &GenerationRequest::new_text(
                String::from("llama-merged"),
                descriptor.clone(),
                None,
                "hello",
                GenerationOptions::greedy(1),
            )
            .with_adapter_serving(merged_binding.clone()),
        )?;
        assert_eq!(merged.output.text, "psionic");
        assert_eq!(
            merged_binding.residency_mode,
            AdapterResidencyMode::MergedResident
        );

        let unmerged_binding =
            service.unmerge_adapter_binding(merged_binding.served_adapter_digest.as_str())?;
        let unmerged = service.generate(
            &GenerationRequest::new_text(
                String::from("llama-unmerged"),
                descriptor.clone(),
                None,
                "hello",
                GenerationOptions::greedy(1),
            )
            .with_adapter_serving(unmerged_binding.clone()),
        )?;
        assert_eq!(unmerged.output.text, "psionic");
        assert_eq!(
            unmerged_binding.residency_mode,
            AdapterResidencyMode::HotSwapOverlay
        );

        let detached =
            service.detach_adapter_binding(unmerged_binding.served_adapter_digest.as_str())?;
        assert_eq!(
            detached.served_adapter_digest,
            unmerged_binding.served_adapter_digest
        );
        let error = service
            .generate(
                &GenerationRequest::new_text(
                    String::from("llama-detached"),
                    descriptor,
                    None,
                    "hello",
                    GenerationOptions::greedy(1),
                )
                .with_adapter_serving(unmerged_binding),
            )
            .expect_err("detached adapter binding should be refused");
        assert!(matches!(
            error,
            ReferenceTextGenerationError::UnsupportedAdapterBinding { .. }
        ));
        Ok(())
    }

    #[test]
    fn cpu_gguf_service_refuses_incompatible_lm_head_lora_binding()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = temp.path().join("llama_lora_refusal.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny psionic llama refusal").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let mut service = CpuGgufTextGenerationService::from_gguf_path(&path)?;
        let descriptor = service.model_descriptor().clone();
        let hidden = final_hidden_for_prompt(&mut service, "hello")?;
        let adapter_path = temp.path().join("lm_head_lora_bad.safetensors");
        write_prompt_specific_lora_adapter(&adapter_path, hidden.as_slice(), 5, 32.0)?;
        let error = service
            .register_lm_head_lora_adapter(
                "adapter-bad",
                &adapter_path,
                sample_lora_identity(&descriptor, AdapterTargetFamily::DecoderAttention),
                1.0,
                AdapterResidencyMode::HotSwapOverlay,
            )
            .expect_err("unsupported target family should be refused");
        assert!(matches!(
            error,
            ReferenceTextGenerationError::UnsupportedAdapterBinding { .. }
        ));
        assert!(
            error.to_string().contains("decoder_composite"),
            "refusal should describe the supported adapter target"
        );
        Ok(())
    }

    #[test]
    fn gemma_promoted_revision_validation_accepts_matching_export_and_checkpoint()
    -> Result<(), Box<dyn std::error::Error>> {
        let descriptor = tiny_gemma4_descriptor()?;
        let (served_artifact, base_binding) = sample_gemma_served_base_binding(&descriptor);
        let (exported_artifact, checkpoint) = sample_gemma_exported_revision(
            &descriptor,
            &base_binding,
            "gemma4-e4b-helpdesk",
            "r1",
            5,
        )?;

        let adapter = validate_gemma_exported_revision(
            &descriptor,
            &base_binding,
            &served_artifact,
            &exported_artifact,
            &checkpoint,
        )?;
        let mut logits = vec![0.0; descriptor.config.vocab_size];
        adapter.apply_to_logits(&[1.0, 0.0, 0.0, 0.0], logits.as_mut_slice())?;

        assert_eq!(adapter.hidden_size, descriptor.config.hidden_size);
        assert_eq!(adapter.vocab_size, descriptor.config.vocab_size);
        assert!(
            logits[5] > 0.0,
            "validated adapter should steer the target token"
        );
        Ok(())
    }

    #[test]
    fn gemma_promoted_revision_validation_refuses_tokenizer_and_artifact_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let descriptor = tiny_gemma4_descriptor()?;
        let (served_artifact, base_binding) = sample_gemma_served_base_binding(&descriptor);
        let (exported_artifact, checkpoint) = sample_gemma_exported_revision(
            &descriptor,
            &base_binding,
            "gemma4-e4b-helpdesk",
            "r1",
            5,
        )?;

        let mut stale_checkpoint = checkpoint.clone();
        stale_checkpoint.saved_at_ms += 1;
        let checkpoint_error = validate_gemma_exported_revision(
            &descriptor,
            &base_binding,
            &served_artifact,
            &exported_artifact,
            &stale_checkpoint,
        )
        .expect_err("stale checkpoint digests must be refused");
        assert!(
            checkpoint_error
                .to_string()
                .contains("checkpoint digest does not match"),
            "checkpoint refusal should describe digest drift"
        );

        let mut wrong_tokenizer = exported_artifact.clone();
        wrong_tokenizer.tokenizer_contract_digest = String::from("drifted-tokenizer-contract");
        let tokenizer_error = validate_gemma_exported_revision(
            &descriptor,
            &base_binding,
            &served_artifact,
            &wrong_tokenizer,
            &checkpoint,
        )
        .expect_err("tokenizer drift must be refused");
        assert!(
            tokenizer_error
                .to_string()
                .contains("tokenizer contract does not match"),
            "tokenizer refusal should describe the contract mismatch"
        );

        let mut wrong_identity = exported_artifact.clone();
        wrong_identity.adapter_identity.base_served_artifact_digest =
            String::from("sha256:stale-base-artifact");
        wrong_identity.adapter_identity_digest = wrong_identity.adapter_identity.stable_digest();
        let artifact_error = validate_gemma_exported_revision(
            &descriptor,
            &base_binding,
            &served_artifact,
            &wrong_identity,
            &checkpoint,
        )
        .expect_err("artifact identity drift must be refused");
        assert!(
            artifact_error
                .to_string()
                .contains("base-model identity does not match"),
            "artifact refusal should describe the base-lane mismatch"
        );

        Ok(())
    }

    #[test]
    fn gemma_promoted_revision_state_surfaces_revision_identity_and_rolls_back()
    -> Result<(), Box<dyn std::error::Error>> {
        let descriptor = tiny_gemma4_descriptor()?;
        let (_, base_binding) = sample_gemma_served_base_binding(&descriptor);
        let entry_one = sample_promoted_gemma_revision_entry(
            &descriptor,
            &base_binding,
            "gemma4-e4b-helpdesk",
            "r1",
            5,
            1_000,
        )?;
        let entry_two = sample_promoted_gemma_revision_entry(
            &descriptor,
            &base_binding,
            "gemma4-e4b-helpdesk",
            "r2",
            4,
            2_000,
        )?;
        let mut state = PromotedGemmaRevisionState::default();

        let first_identity = state.promote(entry_one.clone());
        assert_eq!(first_identity, entry_one.identity);
        assert_eq!(state.active_identity(), Some(entry_one.identity.clone()));
        assert_eq!(
            state.last_known_good_identity(),
            Some(entry_one.identity.clone())
        );

        let first_response = state.attach_to_response(response_with_adapter_binding(
            &descriptor,
            &entry_one.binding,
        ));
        assert_eq!(
            first_response
                .provenance
                .as_ref()
                .and_then(|provenance| provenance.served_revision.clone()),
            Some(entry_one.identity.clone())
        );

        let second_identity = state.promote(entry_two.clone());
        assert_eq!(second_identity, entry_two.identity);
        assert_eq!(state.active_identity(), Some(entry_two.identity.clone()));
        assert_eq!(
            state.last_known_good_identity(),
            Some(entry_one.identity.clone())
        );

        let rolled_back = state
            .rollback()
            .expect("rollback should restore the last known-good revision");
        assert_eq!(rolled_back, entry_one.identity);
        assert_eq!(state.active_identity(), Some(entry_one.identity.clone()));
        assert_eq!(
            state.last_known_good_identity(),
            Some(entry_one.identity.clone())
        );

        Ok(())
    }

    fn run_dense_family_case(
        family_label: &str,
        expected_family: GgufDecoderFamily,
        metadata: Vec<(String, GgufMetadataValue)>,
        include_qkv_bias: bool,
        hello_token_index: usize,
        world_token_index: usize,
        prompt: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = temp.path().join(format!("{family_label}.gguf"));
        write_test_gguf(
            &path,
            metadata.as_slice(),
            dense_decoder_tensors(include_qkv_bias, hello_token_index, world_token_index)
                .as_slice(),
        )?;

        let mut service = CpuGgufTextGenerationService::from_gguf_path(&path)?;
        let descriptor = service.model_descriptor().clone();
        let request = GenerationRequest::new_text(
            format!("gguf-{family_label}"),
            descriptor.clone(),
            None,
            prompt,
            GenerationOptions::greedy(1),
        );

        let response = service.generate(&request)?;
        let support = service.runtime_support();
        let loaded = service.loaded_model_views();
        let expected_family_label = family_label_for(expected_family);

        assert_eq!(descriptor.model.family, expected_family_label);
        assert_eq!(response.output.text, "world");
        assert_eq!(response.output.tokens.as_slice().len(), 1);
        assert_eq!(
            response
                .provenance
                .as_ref()
                .map(|value| value.served_artifact.quantization_family),
            Some(QuantizationMode::None)
        );
        assert_eq!(support.family, expected_family);
        assert_eq!(support.supported_backends, vec![String::from("cpu")]);
        assert_eq!(
            support.unsupported_backends,
            vec![String::from("cuda"), String::from("metal")]
        );
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded[0].summary.family.as_deref(),
            Some(expected_family_label)
        );
        Ok(())
    }

    fn family_label_for(family: GgufDecoderFamily) -> &'static str {
        match family {
            GgufDecoderFamily::Llama => "llama",
            GgufDecoderFamily::Qwen => "qwen",
            GgufDecoderFamily::Qwen35 => "qwen35",
            GgufDecoderFamily::Gemma4 => "gemma4",
            GgufDecoderFamily::Mistral => "mistral",
            GgufDecoderFamily::GptOss => "gpt_oss",
        }
    }

    fn final_hidden_for_prompt(
        service: &mut CpuGgufTextGenerationService,
        prompt: &str,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let CpuGgufServiceKind::Dense(dense) = &mut service.inner else {
            panic!("expected dense GGUF runtime");
        };
        let model_id = dense.model_descriptor.model.model_id.clone();
        let loaded = dense
            .models
            .active(model_id.as_str())
            .expect("dense model should be active")
            .clone();
        let prompt_tokens =
            loaded.encode_prompt_input(&crate::GenerationInput::Text(prompt.into()))?;
        let mut cache =
            InMemoryKvCache::new(loaded.descriptor().config.max_context, loaded.cache_width());
        let mut final_hidden = None;
        for token in prompt_tokens.as_slice() {
            let step = loaded.execute_step(&mut dense.backend, *token, cache.len(), &cache)?;
            cache.append(*token, step.key, step.value)?;
            final_hidden = step.hidden;
        }
        Ok(final_hidden.expect("prompt evaluation should produce hidden state"))
    }

    fn sample_lora_identity(
        descriptor: &psionic_models::DecoderModelDescriptor,
        target_family: AdapterTargetFamily,
    ) -> AdapterArtifactIdentity {
        AdapterArtifactIdentity::new(
            "adapter-psionic",
            "r1",
            AdapterArtifactKind::Lora,
            AdapterArtifactFormat::Safetensors,
            descriptor.model.model_id.clone(),
            descriptor.model.revision.clone(),
            crate::served_artifact_identity_for_decoder_backend(descriptor, "cpu", &[])
                .served_artifact_digest,
            "adapter-artifact-digest",
            QuantizationMode::None,
            target_family,
            10,
        )
    }

    fn write_prompt_specific_lora_adapter(
        path: &Path,
        hidden: &[f32],
        target_token_index: usize,
        strength: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(
            path,
            prompt_specific_lora_adapter_bytes(hidden, 6, target_token_index, strength)?,
        )?;
        Ok(())
    }

    fn prompt_specific_lora_adapter_bytes(
        hidden: &[f32],
        vocab_size: usize,
        target_token_index: usize,
        strength: f32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let norm = hidden.iter().map(|value| value * value).sum::<f32>();
        let lora_a: Vec<f32> = hidden.iter().map(|value| value / norm.max(1e-6)).collect();
        let mut lora_b = vec![0.0_f32; vocab_size];
        lora_b[target_token_index] = strength;
        let lora_a_bytes = encode_f32_bytes(lora_a.as_slice());
        let lora_b_bytes = encode_f32_bytes(lora_b.as_slice());
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "lm_head.lora_A.weight".to_string(),
            TensorView::new(SafeTensorsDType::F32, vec![1, hidden.len()], &lora_a_bytes)?,
        );
        tensors.insert(
            "lm_head.lora_B.weight".to_string(),
            TensorView::new(SafeTensorsDType::F32, vec![vocab_size, 1], &lora_b_bytes)?,
        );
        Ok(serialize(
            tensors
                .iter()
                .map(|(name, view)| (name.as_str(), view.clone())),
            None,
        )?)
    }

    fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn tiny_gemma4_descriptor() -> Result<DecoderModelDescriptor, Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = temp.path().join("tiny_gemma4_promoted_revision.gguf");
        write_test_gguf(
            &path,
            dense_gemma4_metadata("tiny psionic gemma4 promoted revision").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        Ok(CpuGgufTextGenerationService::from_gguf_path(&path)?
            .model_descriptor()
            .clone())
    }

    fn sample_gemma_served_base_binding(
        descriptor: &DecoderModelDescriptor,
    ) -> (
        psionic_runtime::ServedArtifactIdentity,
        GemmaE4bServedBaseModelBinding,
    ) {
        let served_artifact =
            crate::served_artifact_identity_for_decoder_backend(descriptor, "cuda", &[]);
        let tokenizer = TokenizerDigest::new(
            TokenizerFamily::SentencePiece,
            "tiny-gemma4-tokenizer",
            descriptor.config.vocab_size as u32,
        )
        .with_special_tokens_digest("tiny-gemma4-specials")
        .with_template_digest("tiny-gemma4-template");
        let binding = GemmaE4bServedBaseModelBinding {
            model_id: descriptor.model.model_id.clone(),
            base_model_revision: descriptor.model.revision.clone(),
            base_served_artifact_digest: served_artifact.served_artifact_digest.clone(),
            tokenizer,
            hidden_size: descriptor.config.hidden_size,
        };
        (served_artifact, binding)
    }

    fn sample_training_run(
        hidden_size: usize,
    ) -> Result<FixedBudgetTrainingRun, Box<dyn std::error::Error>> {
        let parameter = TrainingTensorBuffer::from_f32(
            "lm_head",
            TensorSpec::new(Shape::new(vec![hidden_size]), DType::F32, Device::cpu()),
            vec![0.0; hidden_size],
        )?;
        let group = TrainingParameterGroupState::new(
            "lm_head",
            TrainingParameterClass::Head,
            parameter,
            TrainingOptimizerConfig::adamw(0.05, 0.9, 0.99, 1e-8),
            TrainingOptimizerResidencyPolicy::host_only(),
        )?;
        Ok(FixedBudgetTrainingRun::new(
            "gemma4-promoted-run",
            "gemma4-e4b-mesh-checkpoint-family",
            TrainingLoopBudget::new(1, 1, 1)?,
            vec![group],
        )?)
    }

    fn sample_gemma_exported_revision(
        descriptor: &DecoderModelDescriptor,
        base_binding: &GemmaE4bServedBaseModelBinding,
        adapter_id: &str,
        adapter_revision: &str,
        target_token_index: usize,
    ) -> Result<
        (
            GemmaE4bCudaAdapterExportedArtifact,
            GemmaE4bCudaAdapterCheckpoint,
        ),
        Box<dyn std::error::Error>,
    > {
        let adapter_bytes = prompt_specific_lora_adapter_bytes(
            &vec![1.0, 0.0, 0.0, 0.0],
            descriptor.config.vocab_size,
            target_token_index,
            24.0,
        )?;
        let adapter_artifact_digest = format!("sha256:{}", sha256_hex(&adapter_bytes));
        let adapter_identity = AdapterArtifactIdentity::new(
            adapter_id,
            adapter_revision,
            AdapterArtifactKind::Lora,
            AdapterArtifactFormat::Safetensors,
            base_binding.model_id.clone(),
            base_binding.base_model_revision.clone(),
            base_binding.base_served_artifact_digest.clone(),
            adapter_artifact_digest.clone(),
            QuantizationMode::None,
            AdapterTargetFamily::DecoderComposite,
            10,
        );
        let contract_digest = String::from("gemma4-e4b-contract-r1");
        let compatibility_digest = String::from("gemma4-e4b-compatibility-r1");
        let tokenizer_contract_digest = base_binding.tokenizer.stable_digest();
        let exported_artifact = GemmaE4bCudaAdapterExportedArtifact {
            contract_digest: contract_digest.clone(),
            compatibility_digest: compatibility_digest.clone(),
            tokenizer_contract_digest: tokenizer_contract_digest.clone(),
            adapter_identity: adapter_identity.clone(),
            adapter_identity_digest: adapter_identity.stable_digest(),
            adapter_artifact_digest,
            adapter_alpha: 1.0,
            adapter_bytes,
        };
        let mut checkpoint = GemmaE4bCudaAdapterCheckpoint {
            schema_version: String::from(GEMMA_E4B_CUDA_ADAPTER_CHECKPOINT_SCHEMA_VERSION),
            checkpoint_id: format!("{adapter_id}-{adapter_revision}-checkpoint"),
            training_family_id: String::from(GEMMA_E4B_FINETUNING_MVP_TRAINING_FAMILY_ID),
            contract_digest,
            compatibility_digest,
            target_set_id: String::from(GEMMA_E4B_CUDA_ADAPTER_TARGET_SET_ID),
            base_served_artifact_digest: base_binding.base_served_artifact_digest.clone(),
            tokenizer_contract_digest,
            saved_at_ms: 1_250,
            run: sample_training_run(descriptor.config.hidden_size)?,
            checkpoint_digest: String::new(),
        };
        checkpoint.checkpoint_digest = checkpoint.stable_digest();
        Ok((exported_artifact, checkpoint))
    }

    fn sample_promoted_gemma_revision_entry(
        descriptor: &DecoderModelDescriptor,
        base_binding: &GemmaE4bServedBaseModelBinding,
        adapter_id: &str,
        adapter_revision: &str,
        target_token_index: usize,
        activated_at_ms: u64,
    ) -> Result<PromotedGemmaRevisionEntry, Box<dyn std::error::Error>> {
        let (served_artifact, exported_artifact, checkpoint) = {
            let (served_artifact, _) = sample_gemma_served_base_binding(descriptor);
            let (exported_artifact, checkpoint) = sample_gemma_exported_revision(
                descriptor,
                base_binding,
                adapter_id,
                adapter_revision,
                target_token_index,
            )?;
            (served_artifact, exported_artifact, checkpoint)
        };
        validate_gemma_exported_revision(
            descriptor,
            base_binding,
            &served_artifact,
            &exported_artifact,
            &checkpoint,
        )?;
        let binding = AdapterServingBinding::new(
            format!(
                "gemma4-promoted:{}:{}",
                checkpoint.checkpoint_id, exported_artifact.adapter_identity.adapter_revision
            ),
            base_binding.model_id.clone(),
            base_binding.base_model_revision.clone(),
            base_binding.base_served_artifact_digest.clone(),
            AdapterResidencyMode::HotSwapOverlay,
            vec![exported_artifact.adapter_identity.clone()],
        );
        let identity = gemma_served_revision_identity(
            &binding,
            &exported_artifact,
            &checkpoint,
            activated_at_ms,
        );
        Ok(PromotedGemmaRevisionEntry { binding, identity })
    }

    fn response_with_adapter_binding(
        descriptor: &DecoderModelDescriptor,
        binding: &psionic_adapters::AdapterServingBinding,
    ) -> GenerationResponse {
        let request = GenerationRequest::new_text(
            format!("{}-response", binding.binding_id),
            descriptor.clone(),
            None,
            "hello",
            GenerationOptions::greedy(1),
        )
        .with_adapter_serving(binding.clone());
        let mut response = GenerationResponse::new(
            &request,
            None,
            TokenSequence::new(vec![TokenId(4)]),
            "world",
            1,
            0,
            TerminationReason::EndOfSequence,
        );
        response.provenance = Some(GenerationProvenance {
            served_artifact: crate::served_artifact_identity_for_decoder_backend(
                descriptor,
                "cuda",
                &[],
            ),
            adapter_serving: Some(binding.clone()),
            served_revision: None,
            execution_plan_digest: String::from("tiny-gemma4-cuda-plan"),
            cluster_execution: None,
            load_state: GenerationLoadState::Warm,
            isolation_policy: LocalServingIsolationPolicy::in_process_runtime(),
            streaming_policy: None,
            memory_plan: None,
            residency_policy: None,
            residency_snapshot: None,
            kv_cache_policy: None,
            kv_cache_encoding_policy: None,
            kv_ownership: None,
            prefix_cache_control: None,
            prefix_cache_state: None,
            prefix_cache_refusal_reason: None,
            prefix_cache_policy: None,
            prefix_cache_identity: None,
            compile_path: None,
            delivery_proof: None,
            cache_observations: Vec::new(),
            scheduler: None,
            structured_output: None,
            psion_served_evidence: None,
            psion_served_output_claim_posture: None,
        });
        response
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }

    fn dense_llama_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("llama", name);
        metadata.extend(sentencepiece_tokenizer_metadata_entries());
        metadata
    }

    fn dense_mistral_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("mistral", name);
        metadata.push((
            String::from("mistral.attention.sliding_window"),
            GgufMetadataValue::U32(16),
        ));
        metadata.extend(sentencepiece_tokenizer_metadata_entries());
        metadata
    }

    fn dense_qwen_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("qwen2", name);
        metadata.extend(qwen_tokenizer_metadata_entries());
        metadata
    }

    fn dense_gemma4_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("gemma4", name);
        metadata.push((
            String::from("tokenizer.ggml.pre"),
            GgufMetadataValue::String(String::from("gemma4")),
        ));
        metadata.extend(sentencepiece_tokenizer_metadata_entries());
        metadata
    }

    fn dense_gemma4_metadata_with_block_count(
        name: &str,
        block_count: u32,
    ) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header_with_block_count("gemma4", name, block_count);
        metadata.push((
            String::from("tokenizer.ggml.pre"),
            GgufMetadataValue::String(String::from("gemma4")),
        ));
        metadata.extend(sentencepiece_tokenizer_metadata_entries());
        metadata
    }

    fn qwen35_chat_template() -> &'static str {
        include_str!("../../psionic-models/src/testdata/qwen35_chat_template.jinja")
            .trim_end_matches('\n')
    }

    fn qwen35_decoder_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = vec![
            (
                String::from("general.architecture"),
                GgufMetadataValue::String(String::from("qwen35")),
            ),
            (
                String::from("general.name"),
                GgufMetadataValue::String(name.to_string()),
            ),
            (
                String::from("qwen35.context_length"),
                GgufMetadataValue::U32(256),
            ),
            (
                String::from("qwen35.embedding_length"),
                GgufMetadataValue::U32(8),
            ),
            (
                String::from("qwen35.feed_forward_length"),
                GgufMetadataValue::U32(16),
            ),
            (
                String::from("qwen35.block_count"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.attention.head_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.attention.head_count_kv"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::U32(0),
                    GgufMetadataValue::U32(0),
                    GgufMetadataValue::U32(0),
                    GgufMetadataValue::U32(1),
                ]),
            ),
            (
                String::from("qwen35.attention.key_length"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.attention.value_length"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::F32(1e-6),
            ),
            (
                String::from("qwen35.rope.dimension_count"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.rope.freq_base"),
                GgufMetadataValue::F32(10_000_000.0),
            ),
            (
                String::from("qwen35.full_attention_interval"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.ssm.conv_kernel"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.ssm.group_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.ssm.inner_size"),
                GgufMetadataValue::U32(8),
            ),
            (
                String::from("qwen35.ssm.state_size"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.ssm.time_step_rank"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.vision.block_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.vision.embedding_length"),
                GgufMetadataValue::U32(6),
            ),
            (
                String::from("qwen35.vision_start_token_id"),
                GgufMetadataValue::U32(900),
            ),
            (
                String::from("qwen35.vision_end_token_id"),
                GgufMetadataValue::U32(901),
            ),
            (
                String::from("qwen35.image_token_id"),
                GgufMetadataValue::U32(902),
            ),
            (
                String::from("tokenizer.chat_template"),
                GgufMetadataValue::String(qwen35_chat_template().to_string()),
            ),
        ];
        metadata.extend(qwen35_tokenizer_metadata_entries());
        metadata
    }

    fn dense_family_header(architecture: &str, name: &str) -> Vec<(String, GgufMetadataValue)> {
        dense_family_header_with_block_count(architecture, name, 1)
    }

    fn dense_family_header_with_block_count(
        architecture: &str,
        name: &str,
        block_count: u32,
    ) -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("general.architecture"),
                GgufMetadataValue::String(architecture.to_string()),
            ),
            (
                String::from("general.name"),
                GgufMetadataValue::String(name.to_string()),
            ),
            (
                format!("{architecture}.context_length"),
                GgufMetadataValue::U32(32),
            ),
            (
                format!("{architecture}.embedding_length"),
                GgufMetadataValue::U32(4),
            ),
            (
                format!("{architecture}.feed_forward_length"),
                GgufMetadataValue::U32(8),
            ),
            (
                format!("{architecture}.block_count"),
                GgufMetadataValue::U32(block_count),
            ),
            (
                format!("{architecture}.attention.head_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                format!("{architecture}.attention.head_count_kv"),
                GgufMetadataValue::U32(1),
            ),
            (
                format!("{architecture}.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::F32(1e-5),
            ),
            (
                format!("{architecture}.rope.freq_base"),
                GgufMetadataValue::F32(10_000.0),
            ),
        ]
    }

    fn sentencepiece_tokenizer_metadata_entries() -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("tokenizer.ggml.model"),
                GgufMetadataValue::String(String::from("llama")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("<unk>")),
                    GgufMetadataValue::String(String::from("<s>")),
                    GgufMetadataValue::String(String::from("</s>")),
                    GgufMetadataValue::String(String::from("hello")),
                    GgufMetadataValue::String(String::from("world")),
                    GgufMetadataValue::String(String::from("psionic")),
                ]),
            ),
            (
                String::from("tokenizer.ggml.bos_token_id"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("tokenizer.ggml.eos_token_id"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("tokenizer.ggml.unknown_token_id"),
                GgufMetadataValue::U32(0),
            ),
            (
                String::from("tokenizer.ggml.add_bos_token"),
                GgufMetadataValue::Bool(false),
            ),
            (
                String::from("tokenizer.ggml.add_eos_token"),
                GgufMetadataValue::Bool(false),
            ),
        ]
    }

    fn qwen_tokenizer_metadata_entries() -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("tokenizer.ggml.model"),
                GgufMetadataValue::String(String::from("gpt2")),
            ),
            (
                String::from("tokenizer.ggml.pre"),
                GgufMetadataValue::String(String::from("qwen2")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("<|bos|>")),
                    GgufMetadataValue::String(String::from("<|eos|>")),
                    GgufMetadataValue::String(String::from("hello")),
                    GgufMetadataValue::String(String::from("world")),
                    GgufMetadataValue::String(String::from("psionic")),
                    GgufMetadataValue::String(String::from("agent")),
                ]),
            ),
            (
                String::from("tokenizer.ggml.merges"),
                GgufMetadataValue::Array(vec![GgufMetadataValue::String(String::from(
                    "hello world",
                ))]),
            ),
            (
                String::from("tokenizer.ggml.bos_token_id"),
                GgufMetadataValue::U32(0),
            ),
            (
                String::from("tokenizer.ggml.eos_token_id"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("tokenizer.ggml.unknown_token_id"),
                GgufMetadataValue::U32(0),
            ),
            (
                String::from("tokenizer.ggml.add_bos_token"),
                GgufMetadataValue::Bool(false),
            ),
            (
                String::from("tokenizer.ggml.add_eos_token"),
                GgufMetadataValue::Bool(false),
            ),
        ]
    }

    fn qwen35_tokenizer_metadata_entries() -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("tokenizer.ggml.model"),
                GgufMetadataValue::String(String::from("gpt2")),
            ),
            (
                String::from("tokenizer.ggml.pre"),
                GgufMetadataValue::String(String::from("qwen35")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("<|bos|>")),
                    GgufMetadataValue::String(String::from("<|eos|>")),
                    GgufMetadataValue::String(String::from("<|im_start|>")),
                    GgufMetadataValue::String(String::from("<|im_end|>")),
                    GgufMetadataValue::String(String::from("<think>")),
                    GgufMetadataValue::String(String::from("</think>")),
                    GgufMetadataValue::String(String::from("hello")),
                    GgufMetadataValue::String(String::from("world")),
                    GgufMetadataValue::String(String::from("proxy")),
                    GgufMetadataValue::String(String::from("qwen35")),
                ]),
            ),
            (
                String::from("tokenizer.ggml.merges"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("hello world")),
                    GgufMetadataValue::String(String::from("proxy qwen35")),
                ]),
            ),
            (
                String::from("tokenizer.ggml.bos_token_id"),
                GgufMetadataValue::U32(0),
            ),
            (
                String::from("tokenizer.ggml.eos_token_id"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("tokenizer.ggml.add_bos_token"),
                GgufMetadataValue::Bool(false),
            ),
            (
                String::from("tokenizer.ggml.add_eos_token"),
                GgufMetadataValue::Bool(false),
            ),
        ]
    }

    fn dense_decoder_tensors(
        include_qkv_bias: bool,
        hello_token_index: usize,
        world_token_index: usize,
    ) -> Vec<TestGgufTensor> {
        dense_decoder_tensors_with_layers(include_qkv_bias, hello_token_index, world_token_index, 1)
    }

    fn dense_decoder_tensors_with_layers(
        include_qkv_bias: bool,
        hello_token_index: usize,
        world_token_index: usize,
        layer_count: usize,
    ) -> Vec<TestGgufTensor> {
        let mut tensors = vec![
            dense_tensor(
                "token_embd.weight",
                vec![6, 4],
                token_embedding_values(hello_token_index),
            ),
            dense_tensor("output_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            dense_tensor(
                "output.weight",
                vec![6, 4],
                output_values(world_token_index),
            ),
        ];
        for layer_index in 0..layer_count {
            let prefix = format!("blk.{layer_index}");
            tensors.push(dense_tensor(
                &format!("{prefix}.attn_norm.weight"),
                vec![4],
                vec![1.0, 1.0, 1.0, 1.0],
            ));
            tensors.push(dense_tensor(
                &format!("{prefix}.attn_q.weight"),
                vec![4, 4],
                vec![0.0; 16],
            ));
            tensors.push(dense_tensor(
                &format!("{prefix}.attn_k.weight"),
                vec![2, 4],
                vec![0.0; 8],
            ));
            tensors.push(dense_tensor(
                &format!("{prefix}.attn_v.weight"),
                vec![2, 4],
                vec![0.0; 8],
            ));
            tensors.push(dense_tensor(
                &format!("{prefix}.attn_output.weight"),
                vec![4, 4],
                vec![0.0; 16],
            ));
            tensors.push(dense_tensor(
                &format!("{prefix}.ffn_gate.weight"),
                vec![8, 4],
                vec![0.0; 32],
            ));
            tensors.push(dense_tensor(
                &format!("{prefix}.ffn_down.weight"),
                vec![4, 8],
                vec![0.0; 32],
            ));
            tensors.push(dense_tensor(
                &format!("{prefix}.ffn_up.weight"),
                vec![8, 4],
                vec![0.0; 32],
            ));
            tensors.push(dense_tensor(
                &format!("{prefix}.ffn_norm.weight"),
                vec![4],
                vec![1.0, 1.0, 1.0, 1.0],
            ));
            if include_qkv_bias {
                tensors.push(dense_tensor(
                    &format!("{prefix}.attn_q.bias"),
                    vec![4],
                    vec![0.0; 4],
                ));
                tensors.push(dense_tensor(
                    &format!("{prefix}.attn_k.bias"),
                    vec![2],
                    vec![0.0; 2],
                ));
                tensors.push(dense_tensor(
                    &format!("{prefix}.attn_v.bias"),
                    vec![2],
                    vec![0.0; 2],
                ));
            }
        }
        tensors
    }

    fn qwen35_decoder_tensors() -> Vec<TestGgufTensor> {
        let mut tensors = vec![
            dense_f32_tensor("token_embd.weight", vec![10, 8]),
            dense_f32_tensor("output_norm.weight", vec![8]),
            dense_f32_tensor("output.weight", vec![10, 8]),
        ];

        for layer_index in 0..4 {
            let prefix = format!("blk.{layer_index}");
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.attn_norm.weight"),
                vec![8],
            ));
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.ffn_gate.weight"),
                vec![16, 8],
            ));
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.ffn_up.weight"),
                vec![16, 8],
            ));
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.ffn_down.weight"),
                vec![8, 16],
            ));
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.post_attention_norm.weight"),
                vec![8],
            ));

            if layer_index < 3 {
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_qkv.weight"),
                    vec![24, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_gate.weight"),
                    vec![8, 8],
                ));
                tensors.push(dense_f32_tensor(&format!("{prefix}.ssm_a"), vec![2]));
                tensors.push(dense_f32_tensor(&format!("{prefix}.ssm_dt"), vec![2]));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_alpha.weight"),
                    vec![2, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_beta.weight"),
                    vec![2, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_conv1d.weight"),
                    vec![24, 4],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_norm.weight"),
                    vec![4],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_out.weight"),
                    vec![8, 8],
                ));
            } else {
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_q.weight"),
                    vec![16, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_k.weight"),
                    vec![4, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_v.weight"),
                    vec![4, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_output.weight"),
                    vec![8, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_q_norm.weight"),
                    vec![4],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_k_norm.weight"),
                    vec![4],
                ));
            }
        }

        tensors
    }

    fn token_embedding_values(hello_token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; 6 * 4];
        values[hello_token_index.saturating_mul(4)] = 2.0;
        values
    }

    fn output_values(world_token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; 6 * 4];
        values[world_token_index.saturating_mul(4)] = 1.0;
        values
    }

    fn dense_tensor(name: &str, shape: Vec<usize>, values: Vec<f32>) -> TestGgufTensor {
        TestGgufTensor::new(
            name,
            shape,
            GgufTensorType::F32,
            encode_f32_bytes(values.as_slice()),
        )
    }

    fn dense_f32_tensor(name: &str, shape: Vec<usize>) -> TestGgufTensor {
        let element_count = shape.iter().product::<usize>();
        dense_tensor(name, shape, vec![0.0; element_count])
    }

    struct ScopedEnvVar {
        key: &'static str,
        previous: Option<String>,
    }

    impl ScopedEnvVar {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }
    }

    impl Drop for ScopedEnvVar {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.as_deref() {
                unsafe {
                    std::env::set_var(self.key, previous);
                }
            } else {
                unsafe {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

    fn qwen35_proxy_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    async fn start_qwen35_proxy_test_server()
    -> Result<(String, tokio::sync::oneshot::Sender<()>), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let router = axum::Router::new()
            .route(
                "/health",
                axum::routing::get(|| async { axum::http::StatusCode::OK }),
            )
            .route(
                "/completion",
                axum::routing::post(|_body: axum::Json<serde_json::Value>| async move {
                    axum::Json(serde_json::json!({
                        "content": "proxy world",
                        "tokens": [7, 8],
                        "stop_type": "eos",
                        "truncated": false,
                        "tokens_evaluated": 3
                    }))
                }),
            );
        tokio::spawn(async move {
            let _ = axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        Ok((format!("http://{address}"), shutdown_tx))
    }

    fn write_test_gguf(
        path: &Path,
        metadata: &[(String, GgufMetadataValue)],
        tensors: &[TestGgufTensor],
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(path, build_test_gguf(metadata, tensors)?)?;
        Ok(())
    }

    fn build_test_gguf(
        metadata: &[(String, GgufMetadataValue)],
        tensors: &[TestGgufTensor],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let alignment = metadata
            .iter()
            .find(|(key, _)| key == "general.alignment")
            .and_then(|(_, value)| match value {
                GgufMetadataValue::U64(value) => Some(*value as usize),
                GgufMetadataValue::U32(value) => Some(*value as usize),
                _ => None,
            })
            .unwrap_or(32)
            .max(1);

        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, u64::try_from(tensors.len())?);
        push_u64(&mut bytes, u64::try_from(metadata.len())?);

        for (key, value) in metadata {
            push_gguf_string(&mut bytes, key)?;
            push_u32(&mut bytes, gguf_metadata_value_type(value));
            push_gguf_value(&mut bytes, value)?;
        }

        let mut next_offset = 0usize;
        let mut tensor_offsets = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            tensor_offsets.push(next_offset);
            next_offset = align_usize(next_offset + tensor.bytes.len(), alignment);
        }

        for (tensor, offset) in tensors.iter().zip(&tensor_offsets) {
            push_gguf_string(&mut bytes, tensor.name.as_str())?;
            push_u32(&mut bytes, u32::try_from(tensor.shape.len())?);
            for dimension in tensor.shape.iter().rev() {
                push_u64(&mut bytes, u64::try_from(*dimension)?);
            }
            push_u32(&mut bytes, gguf_tensor_type_code(tensor.tensor_type));
            push_u64(&mut bytes, u64::try_from(*offset)?);
        }

        let tensor_data_offset = align_usize(bytes.len(), alignment);
        bytes.resize(tensor_data_offset, 0);

        for (tensor, offset) in tensors.iter().zip(&tensor_offsets) {
            let start = tensor_data_offset + offset;
            if bytes.len() < start {
                bytes.resize(start, 0);
            }
            bytes.extend_from_slice(tensor.bytes.as_slice());
            bytes.resize(align_usize(bytes.len(), alignment), 0);
        }

        Ok(bytes)
    }

    fn align_usize(value: usize, alignment: usize) -> usize {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    fn gguf_metadata_value_type(value: &GgufMetadataValue) -> u32 {
        match value {
            GgufMetadataValue::U8(_) => 0,
            GgufMetadataValue::I8(_) => 1,
            GgufMetadataValue::U16(_) => 2,
            GgufMetadataValue::I16(_) => 3,
            GgufMetadataValue::U32(_) => 4,
            GgufMetadataValue::I32(_) => 5,
            GgufMetadataValue::F32(_) => 6,
            GgufMetadataValue::Bool(_) => 7,
            GgufMetadataValue::String(_) => 8,
            GgufMetadataValue::Array(_) => 9,
            GgufMetadataValue::U64(_) => 10,
            GgufMetadataValue::I64(_) => 11,
            GgufMetadataValue::F64(_) => 12,
        }
    }

    fn gguf_tensor_type_code(tensor_type: GgufTensorType) -> u32 {
        match tensor_type {
            GgufTensorType::F32 => 0,
            other => panic!("unsupported synthetic gguf tensor type: {other:?}"),
        }
    }

    fn push_gguf_string(
        bytes: &mut Vec<u8>,
        value: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        push_u64(bytes, u64::try_from(value.len())?);
        bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn push_gguf_value(
        bytes: &mut Vec<u8>,
        value: &GgufMetadataValue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match value {
            GgufMetadataValue::U8(value) => bytes.push(*value),
            GgufMetadataValue::I8(value) => bytes.push(value.to_le_bytes()[0]),
            GgufMetadataValue::U16(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I16(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::U32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::U64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::F32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::F64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::Bool(value) => bytes.push(u8::from(*value)),
            GgufMetadataValue::String(value) => push_gguf_string(bytes, value)?,
            GgufMetadataValue::Array(values) => {
                let value_type = values.first().map_or(4, gguf_metadata_value_type);
                push_u32(bytes, value_type);
                push_u64(bytes, u64::try_from(values.len())?);
                for value in values {
                    push_gguf_value(bytes, value)?;
                }
            }
        }
        Ok(())
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend(value.to_le_bytes());
    }
}
