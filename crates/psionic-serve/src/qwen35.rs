use std::{path::Path, sync::Arc, time::Instant};

use psionic_backend_cpu::{decode_quantized_row_into, quantized_row_byte_len};
use psionic_backend_cuda::{
    ggml_q8_1_storage_bytes, CudaBackend, CudaBuffer, CudaHostBuffer, CudaQuantizedMatvecStats,
};
use psionic_catalog::{BlobIntegrityPolicy, LocalBlobOpenOptions};
use psionic_core::QuantizationMode;
use psionic_models::{
    DecoderModelDescriptor, GgufBlobArtifact, GgufDecoderAdapterLoader, GgufDecoderFamily,
    GgufDecoderFamilyMetadata, GgufDecoderLayerKind, GgufMetadataValue, GgufRuntimeTokenizer,
    ModelLoadError, PagedTensorStorage, TokenId, TokenSequence, TokenizerBoundary,
};
use psionic_runtime::{
    BackendHealthTracker, DeviceDiscovery, LoadedModelResidency, LocalRuntimeObservability,
};
use sha2::{Digest, Sha256};

use crate::{
    current_time_millis, default_generation_streaming_policy, ContinuousBatchGenerationResult,
    GenerationEventStream, GenerationInput, GenerationMetrics, GenerationOptions,
    GenerationProvenance, GenerationRequest, GenerationResponse, GenerationStreamChunk,
    GenerationStreamEvent, GenerationStreamStatus, GenerationStreamTerminal,
    GenerationStreamingPolicy, LoadedModelView, LoadedModelsObservation, LocalRuntimeDiagnostic,
    ManagedTextGenerationRuntime, ReferenceTextGenerationError, StreamingTextGenerationExecutor,
    TerminationReason, TextGenerationExecutor,
};

pub struct CudaGgufQwen35TextGenerationService {
    backend: CudaBackend,
    backend_selection: psionic_runtime::BackendSelection,
    model: Arc<CudaQwen35Model>,
    step_plan: Qwen35CudaStepPlan,
    backend_health: BackendHealthTracker,
    residency: LoadedModelResidency,
    memory_plan: psionic_runtime::ModelMemoryPlan,
    residency_policy: psionic_runtime::ModelResidencyPolicy,
}

impl CudaGgufQwen35TextGenerationService {
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, ReferenceTextGenerationError> {
        let mut backend = CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(String::from(
                    "cuda quantized kernels are unavailable in this build",
                )),
            ));
        }
        let backend_selection = backend
            .backend_selection(&["input", "constant", "quantized_matmul", "rms_norm"])
            .map_err(|error| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                    error.to_string(),
                ))
            })?;
        let model = Arc::new(CudaQwen35Model::from_gguf_path(path, &mut backend)?);
        let step_plan = model.build_step_plan(&mut backend)?;
        let mut backend_health = BackendHealthTracker::default();
        let now_millis = current_time_millis();
        backend_health.observe("cuda", backend.health(), now_millis);
        let residency =
            LoadedModelResidency::ready(now_millis, crate::DEFAULT_MODEL_KEEPALIVE_MILLIS);
        Ok(Self {
            memory_plan: model.memory_plan.clone(),
            residency_policy: psionic_runtime::ModelResidencyPolicy::default(),
            backend,
            backend_selection,
            model,
            step_plan,
            backend_health,
            residency,
        })
    }

    #[must_use]
    pub fn model_descriptor(&self) -> &DecoderModelDescriptor {
        &self.model.descriptor
    }

    #[must_use]
    pub fn runtime_support(&self) -> crate::GgufDecoderRuntimeSupport {
        crate::GgufDecoderRuntimeSupport {
            family: GgufDecoderFamily::Qwen35,
            supported_backends: vec![String::from("cuda")],
            unsupported_backends: vec![String::from("cpu"), String::from("metal")],
            unsupported_features: vec![
                String::from("multimodal_inputs"),
                String::from("video_inputs"),
                String::from("tool_calling"),
                String::from("structured_output_fallback"),
                String::from("adapter_serving"),
                String::from("session_reuse"),
                String::from("prefix_cache"),
            ],
            quantization_modes: self.model.descriptor.weights.quantization_modes.clone(),
            adapter_runtime: crate::DecoderAdapterRuntimeSupport {
                support_level: String::from("unsupported"),
                import_formats: Vec::new(),
                residency_modes: Vec::new(),
                batching_mode: String::from("not_available"),
                unsupported_reasons: vec![String::from(
                    "LM-head LoRA serving is not implemented on the native qwen35 cuda runtime",
                )],
            },
        }
    }

    #[must_use]
    pub fn plan_digest(&self, model_id: &str) -> Option<&str> {
        (model_id == self.model.descriptor.model.model_id)
            .then_some(self.model.plan_digest.as_str())
    }

    #[must_use]
    pub fn loaded_model_views(&mut self) -> Vec<LoadedModelView> {
        vec![self.loaded_model_view()]
    }

    #[must_use]
    pub fn loaded_models(&mut self) -> LoadedModelsObservation {
        LoadedModelsObservation::new(vec![self.loaded_model_view().summary])
    }

    #[must_use]
    pub fn observability(&mut self) -> LocalRuntimeObservability {
        self.backend_health
            .observe("cuda", self.backend.health(), current_time_millis());
        LocalRuntimeObservability {
            isolation_policy: psionic_runtime::LocalServingIsolationPolicy::in_process_runtime(),
            cache_invalidation_policy: crate::cache_invalidation_policy(),
            execution_profile: crate::default_text_generation_execution_profile(),
            queue_depth: 0,
            queue_capacity: Some(
                crate::default_text_generation_execution_profile()
                    .queue_policy
                    .max_queued_requests,
            ),
            active_sessions: 0,
            active_requests: self.residency.active_requests,
            memory_footprint: self.residency_snapshot(),
            backend_health: self.backend_health.snapshot(),
            recent_transitions: self.backend_health.recent_changes(),
        }
    }

    pub fn warm_model(
        &mut self,
        model_id: &str,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        if model_id != self.model.descriptor.model.model_id {
            return Err(ReferenceTextGenerationError::UnsupportedModel(
                model_id.to_string(),
            ));
        }
        self.residency
            .refresh_keep_alive(keep_alive_millis, current_time_millis());
        Ok(self.loaded_model_view())
    }

    pub fn unload_model(
        &mut self,
        model_id: &str,
    ) -> Result<LoadedModelView, ReferenceTextGenerationError> {
        if model_id != self.model.descriptor.model.model_id {
            return Err(ReferenceTextGenerationError::UnsupportedModel(
                model_id.to_string(),
            ));
        }
        self.residency.expire_now(current_time_millis());
        Ok(self.loaded_model_view())
    }

    pub fn generate_continuous_batch(
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

    fn loaded_model_view(&self) -> LoadedModelView {
        let mut summary = crate::LoadedModelSummary::from_decoder_descriptor(
            self.model.descriptor.model.model_id.clone(),
            &self.model.descriptor,
        );
        summary.size_bytes = Some(self.memory_plan.weights_bytes);
        summary.size_vram_bytes = Some(self.memory_plan.resident_device_bytes);
        summary.backend = Some(String::from("cuda"));
        summary.fallback_state = crate::backend_selection_fallback_state(&self.backend_selection);
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
                model_id: self.model.descriptor.model.model_id.clone(),
                plan: self.memory_plan.clone(),
                active_requests: self.residency.active_requests,
                last_used_at_millis: self.residency.last_used_at_millis,
            },
        ])
    }
}

impl TextGenerationExecutor for CudaGgufQwen35TextGenerationService {
    type Error = ReferenceTextGenerationError;

    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResponse, Self::Error> {
        if request.product_id != crate::TEXT_GENERATION_PRODUCT_ID {
            return Err(ReferenceTextGenerationError::UnsupportedProduct(
                request.product_id.clone(),
            ));
        }
        if request.model != self.model.descriptor {
            return Err(ReferenceTextGenerationError::UnsupportedModel(
                request.model.model.model_id.clone(),
            ));
        }
        if request.session_id.is_some() || request.reset_session {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::UnsupportedStep(String::from(
                    "native qwen35 cuda runtime does not implement session reuse yet",
                )),
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
                    "LM-head LoRA serving is not implemented on the native qwen35 cuda runtime",
                ),
            });
        }
        if request.options.structured_output.is_some() {
            return Err(ReferenceTextGenerationError::Runtime(crate::RuntimeError::UnsupportedStep(
                String::from(
                    "structured-output fallback is not implemented on the native qwen35 cuda runtime",
                ),
            )));
        }

        let generation_start = Instant::now();
        self.residency.begin_request(current_time_millis());
        let result = self.generate_inner(request);
        self.residency.finish_request(current_time_millis());
        result.map(|mut response| {
            response.metrics.total_duration_ns = Some(
                generation_start
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            response
        })
    }
}

impl StreamingTextGenerationExecutor for CudaGgufQwen35TextGenerationService {
    type Stream<'a> = Box<dyn GenerationEventStream + 'a>;

    fn generate_stream<'a>(
        &'a mut self,
        request: &GenerationRequest,
    ) -> Result<Self::Stream<'a>, ReferenceTextGenerationError> {
        let response = self.generate(request)?;
        Ok(Box::new(CompletedQwen35Stream::new(response)))
    }
}

impl ManagedTextGenerationRuntime for CudaGgufQwen35TextGenerationService {
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

impl CudaGgufQwen35TextGenerationService {
    fn generate_inner(
        &mut self,
        request: &GenerationRequest,
    ) -> Result<GenerationResponse, ReferenceTextGenerationError> {
        let prompt_eval_start = Instant::now();
        let prompt_tokens = match &request.prompt {
            GenerationInput::Text(text) => self.model.tokenizer.encode_with_defaults(text),
            GenerationInput::Tokens(tokens) => tokens.clone(),
        };
        if prompt_tokens.is_empty() {
            return Err(ReferenceTextGenerationError::EmptyPrompt);
        }
        let (prompt_tokens, context_window) = psionic_models::apply_context_window(
            &prompt_tokens,
            self.model.descriptor.config.max_context,
            0,
            request.options.max_output_tokens,
            request.options.context_overflow_policy,
            usize::from(
                prompt_tokens.as_slice().first().copied()
                    == Some(self.model.tokenizer.vocabulary().bos_id()),
            ),
        )?;

        let cache_capacity_tokens = qwen35_cache_capacity_tokens(
            prompt_tokens.len(),
            request.options.max_output_tokens,
            self.model.descriptor.config.max_context,
        );
        let mut state = self
            .model
            .initial_state(&mut self.backend, cache_capacity_tokens)?;
        let mut kernel_count = 0usize;
        let mut bytes_moved = 0u64;
        let fast_greedy = qwen35_fast_greedy_path(&request.options);
        let mut last_logits = Vec::new();
        let mut pending_greedy_token = None;

        if fast_greedy {
            let (last_prompt_token, prompt_prefix) = prompt_tokens
                .as_slice()
                .split_last()
                .expect("validated non-empty prompt token list");
            for token in prompt_prefix {
                let step = self.model.forward_token(
                    &mut self.backend,
                    &mut self.step_plan,
                    &mut state,
                    *token,
                    CudaStepOutputMode::NoOutput,
                )?;
                kernel_count = kernel_count.saturating_add(step.kernel_count);
                bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
            }
            let step = self.model.forward_token(
                &mut self.backend,
                &mut self.step_plan,
                &mut state,
                *last_prompt_token,
                CudaStepOutputMode::ArgmaxOnly,
            )?;
            pending_greedy_token = step.selected_token;
            kernel_count = kernel_count.saturating_add(step.kernel_count);
            bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
        } else {
            for token in prompt_tokens.as_slice() {
                let step = self.model.forward_token(
                    &mut self.backend,
                    &mut self.step_plan,
                    &mut state,
                    *token,
                    CudaStepOutputMode::FullLogits,
                )?;
                last_logits = step.logits;
                kernel_count = kernel_count.saturating_add(step.kernel_count);
                bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
            }
        }

        let prompt_eval_duration_ns = prompt_eval_start
            .elapsed()
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);
        let mut sampler = crate::GenerationSampler::new(&request.options)?;
        let structured_output_report = sampler.structured_output_report();
        let mut generated_tokens = Vec::new();
        let mut generated_text_terminated = None;
        let first_token_started = Instant::now();
        let mut first_token_emitted_at = None;
        let mut last_token_emitted_at = None;

        let termination = loop {
            if generated_tokens.len() >= request.options.max_output_tokens {
                break TerminationReason::MaxOutputTokens;
            }
            if prompt_tokens.len().saturating_add(generated_tokens.len())
                >= self.model.descriptor.config.max_context
            {
                break TerminationReason::ContextLimit;
            }

            let next_token = if fast_greedy {
                let step = if let Some(selected) = pending_greedy_token.take() {
                    Qwen35ForwardStep {
                        selected_token: Some(selected),
                        logits: Vec::new(),
                        kernel_count: 0,
                        bytes_moved: 0,
                    }
                } else {
                    self.model.forward_token(
                        &mut self.backend,
                        &mut self.step_plan,
                        &mut state,
                        *generated_tokens
                            .last()
                            .expect("generated token should exist"),
                        CudaStepOutputMode::ArgmaxOnly,
                    )?
                };
                kernel_count = kernel_count.saturating_add(step.kernel_count);
                bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
                step.selected_token.ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                        String::from("qwen35 argmax decode did not return a selected token"),
                    ))
                })?
            } else {
                if !generated_tokens.is_empty() {
                    let step = self.model.forward_token(
                        &mut self.backend,
                        &mut self.step_plan,
                        &mut state,
                        *generated_tokens
                            .last()
                            .expect("generated token should exist"),
                        CudaStepOutputMode::FullLogits,
                    )?;
                    last_logits = step.logits;
                    kernel_count = kernel_count.saturating_add(step.kernel_count);
                    bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
                }
                match sampler.select_next_token(
                    &self.model.tokenizer,
                    &last_logits,
                    &crate::InMemoryKvCache::new(1, 1),
                    generated_tokens.as_slice(),
                )? {
                    crate::GenerationSelection::Token(token) => token,
                    crate::GenerationSelection::Terminate => {
                        break TerminationReason::EndOfSequence
                    }
                }
            };

            if self.model.tokenizer.is_end_of_sequence(next_token) {
                break TerminationReason::EndOfSequence;
            }

            if first_token_emitted_at.is_none() {
                first_token_emitted_at = Some(first_token_started.elapsed());
            }
            last_token_emitted_at = Some(first_token_started.elapsed());
            generated_tokens.push(next_token);
            if crate::truncate_generated_text(
                &self.model.tokenizer,
                &mut generated_tokens,
                &request.options.stop_sequences,
            )
            .is_some()
            {
                generated_text_terminated = Some(TerminationReason::EndOfSequence);
                break TerminationReason::EndOfSequence;
            }
        };

        let generated = TokenSequence::new(generated_tokens);
        let text = self.model.tokenizer.decode(generated.as_slice());
        let metrics = GenerationMetrics {
            total_duration_ns: None,
            load_duration_ns: Some(self.model.load_duration_ns),
            prompt_eval_count: Some(prompt_tokens.len()),
            prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
            context_window: Some(context_window),
            eval_count: Some(generated.len()),
            eval_duration_ns: Some(
                first_token_started
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX),
            ),
            time_to_first_token_ns: first_token_emitted_at
                .map(|duration| duration.as_nanos().try_into().unwrap_or(u64::MAX)),
            inter_token_latency_ns: average_inter_token_latency_ns(
                first_token_emitted_at,
                last_token_emitted_at,
                generated.len(),
            ),
            kv_cache: None,
            kv_residency: None,
            kv_cache_encoding: None,
            prefix_tokens_reused: Some(0),
            gpt_oss_perf: None,
        };
        let provenance = GenerationProvenance {
            served_artifact: crate::served_artifact_identity_for_decoder_backend(
                &self.model.descriptor,
                "cuda",
                &[],
            ),
            adapter_serving: None,
            execution_plan_digest: self.model.plan_digest.clone(),
            cluster_execution: None,
            load_state: crate::GenerationLoadState::Warm,
            isolation_policy: psionic_runtime::LocalServingIsolationPolicy::in_process_runtime(),
            streaming_policy: None,
            memory_plan: Some(self.memory_plan.clone()),
            residency_policy: Some(self.residency_policy.clone()),
            residency_snapshot: Some(self.residency_snapshot()),
            kv_cache_policy: None,
            kv_cache_encoding_policy: None,
            kv_ownership: None,
            prefix_cache_control: Some(request.prefix_cache_control.clone()),
            prefix_cache_state: None,
            prefix_cache_refusal_reason: None,
            prefix_cache_policy: None,
            prefix_cache_identity: None,
            compile_path: None,
            delivery_proof: Some(psionic_runtime::ExecutionDeliveryProof {
                execution_plan_digest: self.model.plan_digest.clone(),
                kernel_count,
                bytes_moved,
                plan_cache_hits: 0,
                plan_cache_misses: 0,
                kv_growth: None,
                prefill_decode_handoff: None,
                kv_residency: None,
            }),
            cache_observations: Vec::new(),
            scheduler: None,
            structured_output: structured_output_report,
            psion_served_evidence: None,
            psion_served_output_claim_posture: None,
        };
        let response = GenerationResponse::new(
            request,
            None,
            generated,
            text,
            prompt_tokens.len(),
            0,
            generated_text_terminated.unwrap_or(termination),
        )
        .with_metrics_and_provenance(metrics, provenance);
        Ok(response)
    }
}

fn qwen35_fast_greedy_path(options: &GenerationOptions) -> bool {
    matches!(options.decode_strategy, crate::DecodeStrategy::Greedy)
        && options.temperature.is_none()
        && options.top_k.is_none()
        && options.top_p.is_none()
        && options.repeat_penalty.is_none()
        && options.presence_penalty.is_none()
        && options.frequency_penalty.is_none()
}

fn can_use_q8_1_mmvq(mode: QuantizationMode) -> bool {
    matches!(
        mode,
        QuantizationMode::GgmlQ8_0 | QuantizationMode::GgmlMxfp4
    )
}

fn initial_cuda_argmax_pair_bytes() -> [u8; std::mem::size_of::<u64>()] {
    let packed = (u64::from(i32::MAX as u32) << 32) | u64::from(f32::NEG_INFINITY.to_bits());
    packed.to_ne_bytes()
}

fn cuda_argmax_token_id(token: i32) -> Result<TokenId, ReferenceTextGenerationError> {
    u32::try_from(token).map(TokenId).map_err(|_| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
            "cuda argmax returned a negative token id {token}",
        )))
    })
}

fn cuda_argmax_token_from_packed_host_buffer(
    host_buffer: &CudaHostBuffer,
) -> Result<TokenId, ReferenceTextGenerationError> {
    let bytes = host_buffer.read_bytes().map_err(|error| {
        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
            "cuda argmax returned an invalid packed host buffer: {error}",
        )))
    })?;
    let packed = u64::from_ne_bytes(bytes[..std::mem::size_of::<u64>()].try_into().map_err(
        |_| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "cuda argmax returned invalid packed argmax bytes",
            )))
        },
    )?);
    cuda_argmax_token_id((packed >> 32) as i32)
}

fn select_argmax(logits: &[f32]) -> Result<TokenId, ReferenceTextGenerationError> {
    let Some((index, _)) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
    else {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(String::from(
                "argmax selection requires non-empty logits",
            )),
        ));
    };
    Ok(TokenId(index as u32))
}

fn average_inter_token_latency_ns(
    first: Option<std::time::Duration>,
    last: Option<std::time::Duration>,
    token_count: usize,
) -> Option<u64> {
    if token_count < 2 {
        return None;
    }
    let first = first?;
    let last = last?;
    last.checked_sub(first)
        .and_then(|delta| delta.as_nanos().checked_div((token_count - 1) as u128))
        .and_then(|average| average.try_into().ok())
}

fn qwen35_cache_capacity_tokens(
    current_tokens: usize,
    reserve_tokens: usize,
    max_context: usize,
) -> usize {
    let requested = current_tokens
        .saturating_add(reserve_tokens)
        .max(64)
        .min(max_context.max(1));
    requested
        .checked_next_power_of_two()
        .unwrap_or(max_context.max(1))
        .min(max_context.max(1))
}

#[derive(Clone, Debug)]
struct CudaQwen35Model {
    descriptor: DecoderModelDescriptor,
    family_metadata: GgufDecoderFamilyMetadata,
    tokenizer: GgufRuntimeTokenizer,
    token_embedding: HostMatrix,
    output_norm: Vec<f32>,
    output_norm_device: CudaBuffer,
    output: CudaQuantizedMatrix,
    layers: Vec<Qwen35Layer>,
    plan_digest: String,
    load_duration_ns: u64,
    memory_plan: psionic_runtime::ModelMemoryPlan,
}

impl CudaQwen35Model {
    fn from_gguf_path(
        path: impl AsRef<Path>,
        backend: &mut CudaBackend,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let load_start = Instant::now();
        let artifact = GgufBlobArtifact::open_path(&path, gguf_local_blob_open_options())?;
        let adapter = GgufDecoderAdapterLoader.load_blob_artifact(&artifact)?;
        if adapter.family_metadata().family != GgufDecoderFamily::Qwen35 {
            return Err(ModelLoadError::UnsupportedModel(
                adapter.descriptor().model.model_id.clone(),
            )
            .into());
        }
        let tokenizer = GgufRuntimeTokenizer::from_gguf(adapter.tokenizer()).map_err(|error| {
            ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!("failed to build qwen35 tokenizer: {error}"),
            }
        })?;
        let token_embedding =
            HostMatrix::load(&artifact, adapter.tensor_layout().token_embedding.as_str())?;
        let output = if let Some(name) = adapter.tensor_layout().output.as_ref() {
            load_cuda_quantized_matrix(backend, &artifact, name.as_str())?
        } else {
            load_cuda_quantized_matrix(
                backend,
                &artifact,
                adapter.tensor_layout().token_embedding.as_str(),
            )?
        };
        let layers = adapter
            .tensor_layout()
            .layers
            .iter()
            .map(|layout| Qwen35Layer::load(backend, &artifact, layout, adapter.family_metadata()))
            .collect::<Result<Vec<_>, _>>()?;
        let output_norm =
            load_dense_vector(&artifact, adapter.tensor_layout().output_norm.as_str())?;
        let output_norm_device =
            upload_f32_buffer(backend, output_norm.as_slice(), "qwen35_output_norm")?;
        let weights_bytes = std::fs::metadata(path.as_ref())
            .map(|metadata| metadata.len())
            .unwrap_or_default();
        let device_bytes = output
            .device_residency_bytes()
            .saturating_add(vec_f32_bytes(output_norm.as_slice()))
            .saturating_add(
                layers
                    .iter()
                    .map(Qwen35Layer::device_residency_bytes)
                    .sum::<usize>(),
            )
            .try_into()
            .unwrap_or(u64::MAX);
        let host_bytes = token_embedding
            .host_residency_bytes()
            .saturating_add(vec_f32_bytes(output_norm.as_slice()))
            .saturating_add(
                layers
                    .iter()
                    .map(Qwen35Layer::host_residency_bytes)
                    .sum::<usize>(),
            )
            .try_into()
            .unwrap_or(u64::MAX);
        Ok(Self {
            descriptor: adapter.descriptor().clone(),
            family_metadata: adapter.family_metadata().clone(),
            tokenizer,
            token_embedding,
            output_norm,
            output_norm_device,
            output,
            layers,
            plan_digest: digest_qwen35_cuda_plan(adapter.descriptor(), adapter.family_metadata()),
            load_duration_ns: load_start
                .elapsed()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
            memory_plan: psionic_runtime::ModelMemoryPlan::split_residency(
                weights_bytes,
                0,
                0,
                host_bytes,
                device_bytes,
            ),
        })
    }

    fn initial_state(
        &self,
        backend: &mut CudaBackend,
        cache_capacity_tokens: usize,
    ) -> Result<Qwen35State, ReferenceTextGenerationError> {
        Ok(Qwen35State {
            position: 0,
            layers: self
                .layers
                .iter()
                .map(|layer| layer.initial_state(backend, cache_capacity_tokens))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn build_step_plan(
        &self,
        backend: &mut CudaBackend,
    ) -> Result<Qwen35CudaStepPlan, ReferenceTextGenerationError> {
        Qwen35CudaStepPlan::new(
            backend,
            self.descriptor.config.hidden_size,
            self.max_projection_input_columns(),
            self.max_projection_output_rows(),
            self.descriptor.config.vocab_size,
        )
    }

    fn max_projection_input_columns(&self) -> usize {
        self.layers
            .iter()
            .map(Qwen35Layer::max_matvec_input_columns)
            .fold(self.output.host.columns, usize::max)
    }

    fn max_projection_output_rows(&self) -> usize {
        self.layers
            .iter()
            .map(Qwen35Layer::max_matvec_output_rows)
            .max()
            .unwrap_or(self.descriptor.config.hidden_size)
    }

    fn forward_token(
        &self,
        backend: &mut CudaBackend,
        plan: &mut Qwen35CudaStepPlan,
        state: &mut Qwen35State,
        token: TokenId,
        output_mode: CudaStepOutputMode,
    ) -> Result<Qwen35ForwardStep, ReferenceTextGenerationError> {
        if token.as_u32() as usize >= self.descriptor.config.vocab_size {
            return Err(ReferenceTextGenerationError::InvalidToken {
                token: token.as_u32(),
                vocab_size: self.descriptor.config.vocab_size,
            });
        }
        let hidden = self.token_embedding.decode_row(token.as_u32() as usize)?;
        plan.current_hidden_buffer
            .write_f32_at_offset(0, hidden.as_slice())
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let mut bytes_moved = 0u64;
        let mut kernel_count = 1usize;
        let position = state.position;
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            match (&layer.kind, layer_state) {
                (Qwen35LayerKind::Hybrid(_), Qwen35LayerState::Hybrid(hybrid_state)) => {
                    layer.forward_hybrid_device(
                        backend,
                        plan,
                        self,
                        hybrid_state,
                        &mut kernel_count,
                        &mut bytes_moved,
                    )?;
                }
                (
                    Qwen35LayerKind::FullAttention(full_attention),
                    Qwen35LayerState::FullAttention(full_attention_state),
                ) => {
                    layer.forward_full_attention_device(
                        backend,
                        plan,
                        self,
                        full_attention,
                        full_attention_state,
                        position,
                        &mut kernel_count,
                        &mut bytes_moved,
                    )?;
                }
                _ => {
                    return Err(ReferenceTextGenerationError::Runtime(
                        crate::RuntimeError::Backend(String::from(
                            "qwen35 layer state kind mismatch",
                        )),
                    ));
                }
            }
        }

        let current_hidden_buffer = plan.current_hidden_buffer.clone();
        let output_norm_device = self.output_norm_device.clone();
        let (logits, selected_token, output_stats) = match output_mode {
            CudaStepOutputMode::NoOutput => (Vec::new(), None, zero_cuda_matvec_stats()),
            CudaStepOutputMode::FullLogits => {
                let (logits, stats) = plan
                    .run_output_logits_from_device(
                        backend,
                        &current_hidden_buffer,
                        &output_norm_device,
                        self.family_metadata.rms_norm_epsilon,
                        &self.output.storage,
                        self.output.host.mode,
                        self.output.host.rows,
                        self.output.host.columns,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                (logits, None, stats)
            }
            CudaStepOutputMode::ArgmaxOnly => {
                let (selected, stats) = plan
                    .run_output_argmax_from_device(
                        backend,
                        &current_hidden_buffer,
                        &output_norm_device,
                        self.family_metadata.rms_norm_epsilon,
                        &self.output.storage,
                        self.output.host.mode,
                        self.output.host.rows,
                        self.output.host.columns,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                (Vec::new(), Some(selected), stats)
            }
        };
        bytes_moved = bytes_moved.saturating_add(cuda_stats_bytes(output_stats));
        kernel_count = kernel_count.saturating_add(output_stats.kernel_launches);
        state.position = state.position.saturating_add(1);
        Ok(Qwen35ForwardStep {
            logits,
            selected_token,
            kernel_count,
            bytes_moved,
        })
    }
}

#[derive(Clone, Debug)]
struct Qwen35Layer {
    attention_norm: Vec<f32>,
    attention_norm_device: CudaBuffer,
    post_attention_norm: Vec<f32>,
    post_attention_norm_device: CudaBuffer,
    ffn_gate_up: CudaQuantizedProjectionGroup,
    ffn_down: CudaQuantizedMatrix,
    kind: Qwen35LayerKind,
}

impl Qwen35Layer {
    fn load(
        backend: &mut CudaBackend,
        artifact: &GgufBlobArtifact,
        layout: &psionic_models::GgufDecoderLayerTensorLayout,
        metadata: &GgufDecoderFamilyMetadata,
    ) -> Result<Self, ModelLoadError> {
        let attention_norm = load_dense_vector(artifact, layout.attention_norm.as_str())?;
        let post_attention_norm = load_dense_vector(
            artifact,
            required_tensor_name(layout.attention_post_norm.as_deref(), "post_attention_norm")?,
        )?;
        let ffn_gate_up = load_cuda_quantized_projection_group(
            backend,
            artifact,
            &[
                required_tensor_name(layout.feed_forward_gate_weight.as_deref(), "ffn_gate")?,
                required_tensor_name(layout.feed_forward_up_weight.as_deref(), "ffn_up")?,
            ],
        )?;
        let kind = match layout.layer_kind {
            GgufDecoderLayerKind::Qwen35Hybrid => Qwen35LayerKind::Hybrid(Qwen35HybridLayer::load(
                backend, artifact, layout, metadata,
            )?),
            GgufDecoderLayerKind::Qwen35FullAttention => Qwen35LayerKind::FullAttention(
                Qwen35FullAttentionLayer::load(backend, artifact, layout, metadata)?,
            ),
            other => {
                return Err(ModelLoadError::ArtifactFormat {
                    format: String::from("gguf"),
                    message: format!("qwen35 cuda runtime does not support layer kind `{other:?}`"),
                })
            }
        };
        Ok(Self {
            attention_norm_device: upload_f32_buffer(
                backend,
                attention_norm.as_slice(),
                "qwen35_attention_norm",
            )?,
            attention_norm,
            post_attention_norm_device: upload_f32_buffer(
                backend,
                post_attention_norm.as_slice(),
                "qwen35_post_attention_norm",
            )?,
            post_attention_norm,
            ffn_gate_up,
            ffn_down: load_cuda_quantized_matrix(
                backend,
                artifact,
                required_tensor_name(layout.feed_forward_down_weight.as_deref(), "ffn_down")?,
            )?,
            kind,
        })
    }

    fn initial_state(
        &self,
        backend: &mut CudaBackend,
        cache_capacity_tokens: usize,
    ) -> Result<Qwen35LayerState, ReferenceTextGenerationError> {
        match &self.kind {
            Qwen35LayerKind::Hybrid(layer) => {
                Ok(Qwen35LayerState::Hybrid(layer.initial_state(backend)?))
            }
            Qwen35LayerKind::FullAttention(layer) => Ok(Qwen35LayerState::FullAttention(
                layer.initial_state(backend, cache_capacity_tokens)?,
            )),
        }
    }

    fn device_residency_bytes(&self) -> usize {
        let aux_bytes = vec_f32_bytes(self.attention_norm.as_slice())
            .saturating_add(vec_f32_bytes(self.post_attention_norm.as_slice()));
        self.ffn_gate_up
            .device_residency_bytes()
            .saturating_add(self.ffn_down.device_residency_bytes())
            .saturating_add(aux_bytes)
            .saturating_add(match &self.kind {
                Qwen35LayerKind::Hybrid(layer) => layer.device_residency_bytes(),
                Qwen35LayerKind::FullAttention(layer) => layer.device_residency_bytes(),
            })
    }

    fn host_residency_bytes(&self) -> usize {
        vec_f32_bytes(self.attention_norm.as_slice())
            .saturating_add(vec_f32_bytes(self.post_attention_norm.as_slice()))
            .saturating_add(match &self.kind {
                Qwen35LayerKind::Hybrid(layer) => layer.host_residency_bytes(),
                Qwen35LayerKind::FullAttention(layer) => layer.host_residency_bytes(),
            })
    }

    fn max_matvec_input_columns(&self) -> usize {
        usize::max(
            usize::max(self.ffn_gate_up.columns, self.ffn_down.host.columns),
            match &self.kind {
                Qwen35LayerKind::Hybrid(layer) => layer.max_matvec_input_columns(),
                Qwen35LayerKind::FullAttention(layer) => layer.max_matvec_input_columns(),
            },
        )
    }

    fn max_matvec_output_rows(&self) -> usize {
        usize::max(
            usize::max(self.ffn_gate_up.total_rows(), self.ffn_down.host.rows),
            match &self.kind {
                Qwen35LayerKind::Hybrid(layer) => layer.max_matvec_output_rows(),
                Qwen35LayerKind::FullAttention(layer) => layer.max_matvec_output_rows(),
            },
        )
    }

    fn forward_host(
        &self,
        _backend: &mut CudaBackend,
        _plan: &mut Qwen35CudaStepPlan,
        _model: &CudaQwen35Model,
        _position: usize,
        _hidden: Vec<f32>,
        _state: &mut Qwen35LayerState,
        _kernel_count: &mut usize,
        _bytes_moved: &mut u64,
    ) -> Result<Vec<f32>, ReferenceTextGenerationError> {
        Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(String::from(
                "qwen35 host layer path is disabled; use the cuda-native device path",
            )),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_full_attention_device(
        &self,
        backend: &mut CudaBackend,
        plan: &mut Qwen35CudaStepPlan,
        model: &CudaQwen35Model,
        full_attention: &Qwen35FullAttentionLayer,
        state: &mut Qwen35FullAttentionState,
        position: usize,
        kernel_count: &mut usize,
        _bytes_moved: &mut u64,
    ) -> Result<(), ReferenceTextGenerationError> {
        let hidden_size = model.descriptor.config.hidden_size;
        let epsilon = model.family_metadata.rms_norm_epsilon;
        let head_count = model.descriptor.config.block.attention.head_count;
        let head_dim = model.descriptor.config.block.attention.head_dim;
        let rotary_dim = model.descriptor.config.block.attention.rotary_dim;
        let query_width = head_count.saturating_mul(head_dim);
        let kv_head_count = full_attention.kv_width / head_dim.max(1);
        let query_gate_rows = full_attention.qkv.rows_per_projection[0];
        let key_rows = full_attention.qkv.rows_per_projection[1];
        let value_rows = full_attention.qkv.rows_per_projection[2];
        if query_gate_rows != query_width.saturating_mul(2) {
            return Err(ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                format!(
                    "qwen35 full-attention cuda path requires query/gate rows={} for query width {}, actual {}",
                    query_width.saturating_mul(2),
                    query_width,
                    query_gate_rows,
                ),
            )));
        }
        if key_rows != full_attention.kv_width || value_rows != full_attention.kv_width {
            return Err(ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                format!(
                    "qwen35 full-attention cuda path requires key/value rows={} and {} to match kv width {}, actual key={} value={}",
                    full_attention.kv_width,
                    full_attention.kv_width,
                    full_attention.kv_width,
                    key_rows,
                    value_rows,
                ),
            )));
        }
        if full_attention.output.host.columns != query_width {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 full-attention output width mismatch: expected {}, actual {}",
                    query_width, full_attention.output.host.columns,
                )),
            ));
        }
        state
            .ensure_capacity(backend, state.len.saturating_add(1))
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let (freq_scale, ext_factor, corr_dims, theta_scale) =
            qwen35_rope_runtime_parameters(rotary_dim, &model.family_metadata);
        let query_bytes = query_width.saturating_mul(std::mem::size_of::<f32>());
        let kv_bytes = full_attention
            .kv_width
            .saturating_mul(std::mem::size_of::<f32>());
        let debug_attention = std::env::var_os("PSIONIC_QWEN35_DEBUG_ATTENTION").is_some();

        if debug_attention {
            eprintln!(
                "qwen35_debug_layout position={} state_len={} query_width={} kv_width={} query_gate_rows={} key_rows={} value_rows={} q_buffer_bytes={} k_buffer_bytes={} gate_buffer_bytes={} qkv_norm_bytes={} hidden_norm_bytes={} matvec_output_bytes={} key_cache_bytes={} value_cache_bytes={}",
                position,
                state.len,
                query_width,
                full_attention.kv_width,
                query_gate_rows,
                key_rows,
                value_rows,
                plan.q_buffer.byte_len(),
                plan.k_buffer.byte_len(),
                plan.gate_buffer.byte_len(),
                plan.qkv_norm_buffer.byte_len(),
                plan.hidden_norm_buffer.byte_len(),
                plan.matvec_output_buffer.byte_len(),
                state.key_cache.byte_len(),
                state.value_cache.byte_len(),
            );
            let mut prep = backend
                .begin_submission()
                .map_err(ReferenceTextGenerationError::Runtime)?;
            prep.rms_norm_q8_1(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.matvec_input_q8_1_buffer,
                hidden_size,
                epsilon,
            )?;
            prep.quantized_matvec_q8_1(
                &full_attention.qkv.storage,
                0,
                full_attention.qkv.mode,
                full_attention.qkv.total_rows(),
                full_attention.qkv.columns,
                &plan.matvec_input_q8_1_buffer,
                None,
                &plan.matvec_output_buffer,
            )?;
            prep.split_interleaved_query_gate_f32(
                &plan.matvec_output_buffer,
                head_count,
                head_dim,
                &plan.q_buffer,
                &plan.gate_buffer,
            )?;
            prep.copy_buffer_region(
                &plan.matvec_output_buffer,
                query_gate_rows.saturating_mul(std::mem::size_of::<f32>()),
                &plan.k_buffer,
                0,
                kv_bytes,
            )?;
            prep.rms_norm(
                &plan.q_buffer,
                &full_attention.query_norm_device,
                &plan.q_buffer,
                query_width,
                epsilon,
            )?;
            prep.rms_norm(
                &plan.k_buffer,
                &full_attention.key_norm_device,
                &plan.k_buffer,
                full_attention.kv_width,
                epsilon,
            )?;
            prep.copy_buffer_region(&plan.q_buffer, 0, &plan.qkv_norm_buffer, 0, query_bytes)?;
            prep.copy_buffer_region(
                &plan.k_buffer,
                0,
                &plan.qkv_norm_buffer,
                query_bytes,
                kv_bytes,
            )?;
            prep.copy_buffer_region(
                &plan.matvec_output_buffer,
                query_gate_rows
                    .saturating_add(key_rows)
                    .saturating_mul(std::mem::size_of::<f32>()),
                &plan.qkv_norm_buffer,
                query_bytes.saturating_add(kv_bytes),
                kv_bytes,
            )?;
            let prep_report = prep.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
            *kernel_count = kernel_count.saturating_add(prep_report.encoded_operations);

            let q_host = plan
                .q_buffer
                .read_f32_at_offset(0, query_width)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let gate_host = plan
                .gate_buffer
                .read_f32_at_offset(0, query_width)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let k_host = plan
                .k_buffer
                .read_f32_at_offset(0, full_attention.kv_width)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let v_host = plan
                .qkv_norm_buffer
                .read_f32_at_offset(
                    query_width.saturating_add(full_attention.kv_width),
                    full_attention.kv_width,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let cache_key_host = if state.len > 0 {
                f16_bytes_to_f32_vec(
                    state
                        .key_cache
                        .read_bytes_at_offset(
                            0,
                            state
                                .len
                                .saturating_mul(state.width)
                                .saturating_mul(state.element_bytes),
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?
                        .as_slice(),
                )
                .map_err(ReferenceTextGenerationError::Runtime)?
            } else {
                Vec::new()
            };
            let cache_value_host = if state.len > 0 {
                f16_bytes_to_f32_vec(
                    state
                        .value_cache
                        .read_bytes_at_offset(
                            0,
                            state
                                .len
                                .saturating_mul(state.width)
                                .saturating_mul(state.element_bytes),
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?
                        .as_slice(),
                )
                .map_err(ReferenceTextGenerationError::Runtime)?
            } else {
                Vec::new()
            };
            let mut q_rot = q_host.clone();
            let mut k_rot = k_host.clone();
            apply_rope_neox(
                q_rot.as_mut_slice(),
                head_count,
                head_dim,
                rotary_dim,
                position,
                &model.family_metadata,
            );
            apply_rope_neox(
                k_rot.as_mut_slice(),
                kv_head_count,
                head_dim,
                rotary_dim,
                position,
                &model.family_metadata,
            );
            let cache_entries = cache_key_host
                .chunks_exact(state.width)
                .zip(cache_value_host.chunks_exact(state.width))
                .map(|(key, value)| Qwen35FullAttentionEntry {
                    key: key.to_vec(),
                    value: value.to_vec(),
                })
                .collect::<Vec<_>>();
            let host_attention = attend_full_attention(
                q_rot.as_slice(),
                k_rot.as_slice(),
                v_host.as_slice(),
                cache_entries.as_slice(),
                head_count,
                kv_head_count,
                head_dim,
            );
            let host_gated = host_attention
                .iter()
                .copied()
                .zip(gate_host.iter().copied())
                .map(|(value, gate)| value * sigmoid(gate))
                .collect::<Vec<_>>();

            let mut attention = backend
                .begin_submission()
                .map_err(ReferenceTextGenerationError::Runtime)?;
            attention.attention_decode_rope_cache_f16_kv(
                &plan.qkv_norm_buffer,
                0,
                query_width,
                query_width.saturating_add(full_attention.kv_width),
                &state.key_cache,
                &state.value_cache,
                state.width,
                0,
                state.len,
                model.family_metadata.sliding_window.unwrap_or(0),
                head_count,
                kv_head_count,
                head_dim,
                rotary_dim,
                position,
                freq_scale,
                ext_factor,
                corr_dims,
                theta_scale,
                None,
                &plan.gated_delta_buffer,
            )?;
            attention.sigmoid_mul_f32(
                &plan.gated_delta_buffer,
                0,
                &plan.gate_buffer,
                0,
                query_width,
                &plan.gated_delta_buffer,
            )?;
            let attention_report =
                attention.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
            *kernel_count = kernel_count.saturating_add(attention_report.encoded_operations);
            let device_gated = plan
                .gated_delta_buffer
                .read_f32_at_offset(0, query_width)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let max_diff = host_gated
                .iter()
                .copied()
                .zip(device_gated.iter().copied())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "qwen35_debug position={} state_len={} max_gated_diff={max_diff:.6}",
                position, state.len
            );

            let mut tail = backend
                .begin_submission()
                .map_err(ReferenceTextGenerationError::Runtime)?;
            tail.quantize_f32_to_q8_1(
                &plan.gated_delta_buffer,
                1,
                query_width,
                &plan.activated_q8_1_buffer,
            )?;
            tail.quantized_matvec_q8_1(
                &full_attention.output.storage,
                0,
                full_attention.output.host.mode,
                full_attention.output.host.rows,
                full_attention.output.host.columns,
                &plan.activated_q8_1_buffer,
                None,
                &plan.projected_buffer,
            )?;
            tail.add_residual_rms_norm_q8_1(
                &plan.projected_buffer,
                &plan.current_hidden_buffer,
                None,
                &self.post_attention_norm_device,
                &plan.current_hidden_buffer,
                &plan.hidden_norm_buffer,
                &plan.matvec_input_q8_1_buffer,
                hidden_size,
                epsilon,
            )?;
            tail.quantized_matvec_q8_1(
                &self.ffn_gate_up.storage,
                0,
                self.ffn_gate_up.mode,
                self.ffn_gate_up.total_rows(),
                self.ffn_gate_up.columns,
                &plan.matvec_input_q8_1_buffer,
                None,
                &plan.matvec_output_buffer,
            )?;
            let gate_rows = self.ffn_gate_up.rows_per_projection[0];
            let up_rows = self.ffn_gate_up.rows_per_projection[1];
            if gate_rows != up_rows {
                return Err(ReferenceTextGenerationError::Runtime(
                    crate::RuntimeError::Backend(format!(
                        "qwen35 dense ffn gate/up width mismatch: gate={} up={}",
                        gate_rows, up_rows
                    )),
                ));
            }
            tail.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.hybrid_gated_buffer,
            )?;
            tail.quantize_f32_to_q8_1(
                &plan.hybrid_gated_buffer,
                1,
                gate_rows,
                &plan.activated_q8_1_buffer,
            )?;
            tail.quantized_matvec_q8_1(
                &self.ffn_down.storage,
                0,
                self.ffn_down.host.mode,
                self.ffn_down.host.rows,
                self.ffn_down.host.columns,
                &plan.activated_q8_1_buffer,
                None,
                &plan.projected_buffer,
            )?;
            tail.add_f32_in_place(
                &plan.current_hidden_buffer,
                0,
                &plan.projected_buffer,
                hidden_size,
            )?;
            let tail_report = tail.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
            *kernel_count = kernel_count.saturating_add(tail_report.encoded_operations);
            state.len = state.len.saturating_add(1);
            return Ok(());
        }

        let mut submission = backend
            .begin_submission()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        submission.rms_norm_q8_1(
            &plan.current_hidden_buffer,
            &self.attention_norm_device,
            &plan.matvec_input_q8_1_buffer,
            hidden_size,
            epsilon,
        )?;
        submission.quantized_matvec_q8_1(
            &full_attention.qkv.storage,
            0,
            full_attention.qkv.mode,
            full_attention.qkv.total_rows(),
            full_attention.qkv.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.matvec_output_buffer,
        )?;
        submission.split_interleaved_query_gate_f32(
            &plan.matvec_output_buffer,
            head_count,
            head_dim,
            &plan.q_buffer,
            &plan.gate_buffer,
        )?;
        submission.copy_buffer_region(
            &plan.matvec_output_buffer,
            query_gate_rows.saturating_mul(std::mem::size_of::<f32>()),
            &plan.k_buffer,
            0,
            kv_bytes,
        )?;
        submission.rms_norm(
            &plan.q_buffer,
            &full_attention.query_norm_device,
            &plan.q_buffer,
            query_width,
            epsilon,
        )?;
        submission.rms_norm(
            &plan.k_buffer,
            &full_attention.key_norm_device,
            &plan.k_buffer,
            full_attention.kv_width,
            epsilon,
        )?;
        submission.copy_buffer_region(&plan.q_buffer, 0, &plan.qkv_norm_buffer, 0, query_bytes)?;
        submission.copy_buffer_region(
            &plan.k_buffer,
            0,
            &plan.qkv_norm_buffer,
            query_bytes,
            kv_bytes,
        )?;
        submission.copy_buffer_region(
            &plan.matvec_output_buffer,
            query_gate_rows
                .saturating_add(key_rows)
                .saturating_mul(std::mem::size_of::<f32>()),
            &plan.qkv_norm_buffer,
            query_bytes.saturating_add(kv_bytes),
            kv_bytes,
        )?;
        submission.attention_decode_rope_cache_f16_kv(
            &plan.qkv_norm_buffer,
            0,
            query_width,
            query_width.saturating_add(full_attention.kv_width),
            &state.key_cache,
            &state.value_cache,
            state.width,
            0,
            state.len,
            model.family_metadata.sliding_window.unwrap_or(0),
            head_count,
            kv_head_count,
            head_dim,
            rotary_dim,
            position,
            freq_scale,
            ext_factor,
            corr_dims,
            theta_scale,
            None,
            &plan.gated_delta_buffer,
        )?;
        submission.sigmoid_mul_f32(
            &plan.gated_delta_buffer,
            0,
            &plan.gate_buffer,
            0,
            query_width,
            &plan.gated_delta_buffer,
        )?;
        submission.quantize_f32_to_q8_1(
            &plan.gated_delta_buffer,
            1,
            query_width,
            &plan.activated_q8_1_buffer,
        )?;
        submission.quantized_matvec_q8_1(
            &full_attention.output.storage,
            0,
            full_attention.output.host.mode,
            full_attention.output.host.rows,
            full_attention.output.host.columns,
            &plan.activated_q8_1_buffer,
            None,
            &plan.projected_buffer,
        )?;
        submission.add_residual_rms_norm_q8_1(
            &plan.projected_buffer,
            &plan.current_hidden_buffer,
            None,
            &self.post_attention_norm_device,
            &plan.current_hidden_buffer,
            &plan.hidden_norm_buffer,
            &plan.matvec_input_q8_1_buffer,
            hidden_size,
            epsilon,
        )?;
        submission.quantized_matvec_q8_1(
            &self.ffn_gate_up.storage,
            0,
            self.ffn_gate_up.mode,
            self.ffn_gate_up.total_rows(),
            self.ffn_gate_up.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.matvec_output_buffer,
        )?;
        let gate_rows = self.ffn_gate_up.rows_per_projection[0];
        let up_rows = self.ffn_gate_up.rows_per_projection[1];
        if gate_rows != up_rows {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 dense ffn gate/up width mismatch: gate={} up={}",
                    gate_rows, up_rows
                )),
            ));
        }
        submission.silu_mul_f32(
            &plan.matvec_output_buffer,
            0,
            &plan.matvec_output_buffer,
            gate_rows,
            gate_rows,
            &plan.hybrid_gated_buffer,
        )?;
        submission.quantize_f32_to_q8_1(
            &plan.hybrid_gated_buffer,
            1,
            gate_rows,
            &plan.activated_q8_1_buffer,
        )?;
        submission.quantized_matvec_q8_1(
            &self.ffn_down.storage,
            0,
            self.ffn_down.host.mode,
            self.ffn_down.host.rows,
            self.ffn_down.host.columns,
            &plan.activated_q8_1_buffer,
            None,
            &plan.projected_buffer,
        )?;
        submission.add_f32_in_place(
            &plan.current_hidden_buffer,
            0,
            &plan.projected_buffer,
            hidden_size,
        )?;
        let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
        *kernel_count = kernel_count.saturating_add(report.encoded_operations);
        state.len = state.len.saturating_add(1);
        Ok(())
    }

    fn forward_hybrid_device(
        &self,
        backend: &mut CudaBackend,
        plan: &mut Qwen35CudaStepPlan,
        model: &CudaQwen35Model,
        state: &mut Qwen35HybridState,
        kernel_count: &mut usize,
        bytes_moved: &mut u64,
    ) -> Result<(), ReferenceTextGenerationError> {
        let Qwen35LayerKind::Hybrid(hybrid) = &self.kind else {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(String::from(
                    "qwen35 cuda hybrid path requires a hybrid layer",
                )),
            ));
        };
        let hidden_size = model.descriptor.config.hidden_size;
        let epsilon = model.family_metadata.rms_norm_epsilon;
        let qkv_rows = hybrid.qkv_gate_alpha_beta.rows_per_projection[0];
        let z_rows = hybrid.qkv_gate_alpha_beta.rows_per_projection[1];
        let alpha_rows = hybrid.qkv_gate_alpha_beta.rows_per_projection[2];
        let beta_rows = hybrid.qkv_gate_alpha_beta.rows_per_projection[3];
        if alpha_rows != beta_rows
            || alpha_rows != hybrid.ssm_a.len()
            || alpha_rows != hybrid.ssm_dt.len()
        {
            return Err(ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                format!(
                    "qwen35 hybrid cuda path requires aligned alpha/beta widths, got alpha={} beta={} ssm_a={} ssm_dt={}",
                    alpha_rows,
                    beta_rows,
                    hybrid.ssm_a.len(),
                    hybrid.ssm_dt.len()
                ),
            )));
        }
        let q_size = hybrid.group_count.saturating_mul(hybrid.state_size);
        let k_size = q_size;
        let v_size = hybrid.inner_size;
        let v_offset = q_size.saturating_add(k_size);
        let z_offset = qkv_rows;
        let alpha_offset = z_offset.saturating_add(z_rows);
        let beta_offset = alpha_offset.saturating_add(alpha_rows);
        let q_bytes = q_size.saturating_mul(std::mem::size_of::<f32>());
        let k_bytes = k_size.saturating_mul(std::mem::size_of::<f32>());
        let v_bytes = v_size.saturating_mul(std::mem::size_of::<f32>());

        let mut submission = backend
            .begin_submission()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        submission.rms_norm_q8_1(
            &plan.current_hidden_buffer,
            &self.attention_norm_device,
            &plan.matvec_input_q8_1_buffer,
            hidden_size,
            epsilon,
        )?;
        submission.quantized_matvec_q8_1(
            &hybrid.qkv_gate_alpha_beta.storage,
            0,
            hybrid.qkv_gate_alpha_beta.mode,
            hybrid.qkv_gate_alpha_beta.total_rows(),
            hybrid.qkv_gate_alpha_beta.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.matvec_output_buffer,
        )?;
        submission.depthwise_causal_conv1d_step_f32(
            &plan.matvec_output_buffer,
            &state.conv_state,
            &hybrid.ssm_conv1d_device,
            qkv_rows,
            hybrid.conv_kernel,
            &plan.conv_buffer,
        )?;
        submission.silu_mul_f32(
            &plan.conv_buffer,
            0,
            &plan.ones_buffer,
            0,
            qkv_rows,
            &plan.conv_buffer,
        )?;
        let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
        *kernel_count = kernel_count.saturating_add(report.encoded_operations);

        plan.decay_host = plan
            .matvec_output_buffer
            .read_f32_at_offset(alpha_offset, alpha_rows)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        plan.beta_host = plan
            .matvec_output_buffer
            .read_f32_at_offset(beta_offset, beta_rows)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        for (index, decay) in plan.decay_host.iter_mut().enumerate() {
            *decay = (softplus(*decay + hybrid.ssm_dt[index]) * hybrid.ssm_a[index]).exp();
        }
        for beta in &mut plan.beta_host {
            *beta = sigmoid(*beta);
        }
        plan.decay_buffer
            .write_f32_at_offset(0, plan.decay_host.as_slice())
            .map_err(ReferenceTextGenerationError::Runtime)?;
        plan.beta_buffer
            .write_f32_at_offset(0, plan.beta_host.as_slice())
            .map_err(ReferenceTextGenerationError::Runtime)?;
        *bytes_moved = bytes_moved.saturating_add(
            alpha_rows
                .saturating_add(beta_rows)
                .saturating_add(plan.decay_host.len())
                .saturating_add(plan.beta_host.len())
                .saturating_mul(std::mem::size_of::<f32>())
                .try_into()
                .unwrap_or(u64::MAX),
        );

        let mut submission = backend
            .begin_submission()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        submission.copy_buffer_region(&plan.conv_buffer, 0, &plan.q_buffer, 0, q_bytes)?;
        submission.copy_buffer_region(&plan.conv_buffer, q_bytes, &plan.k_buffer, 0, k_bytes)?;
        submission.rms_norm(
            &plan.q_buffer,
            &hybrid.q_scale_device,
            &plan.q_buffer,
            q_size,
            1e-6,
        )?;
        submission.rms_norm(
            &plan.k_buffer,
            &hybrid.k_scale_device,
            &plan.k_buffer,
            k_size,
            1e-6,
        )?;
        submission.copy_buffer_region(&plan.q_buffer, 0, &plan.qkv_norm_buffer, 0, q_bytes)?;
        submission.copy_buffer_region(
            &plan.k_buffer,
            0,
            &plan.qkv_norm_buffer,
            q_bytes,
            k_bytes,
        )?;
        submission.copy_buffer_region(
            &plan.conv_buffer,
            v_offset.saturating_mul(std::mem::size_of::<f32>()),
            &plan.qkv_norm_buffer,
            v_offset.saturating_mul(std::mem::size_of::<f32>()),
            v_bytes,
        )?;
        submission.gated_delta_step_f32(
            &plan.qkv_norm_buffer,
            0,
            q_size,
            v_offset,
            &plan.decay_buffer,
            &plan.beta_buffer,
            &state.delta_state,
            hybrid.group_count,
            hybrid.time_step_rank,
            hybrid.state_size,
            hybrid.state_size,
            &plan.gated_delta_buffer,
        )?;
        submission.rms_norm(
            &plan.gated_delta_buffer,
            &hybrid.ssm_norm_device,
            &plan.hybrid_norm_buffer,
            v_size,
            epsilon,
        )?;
        submission.silu_mul_f32(
            &plan.matvec_output_buffer,
            z_offset,
            &plan.hybrid_norm_buffer,
            0,
            v_size,
            &plan.hybrid_gated_buffer,
        )?;
        submission.quantize_f32_to_q8_1(
            &plan.hybrid_gated_buffer,
            1,
            v_size,
            &plan.activated_q8_1_buffer,
        )?;
        submission.quantized_matvec_q8_1(
            &hybrid.ssm_out.storage,
            0,
            hybrid.ssm_out.host.mode,
            hybrid.ssm_out.host.rows,
            hybrid.ssm_out.host.columns,
            &plan.activated_q8_1_buffer,
            None,
            &plan.projected_buffer,
        )?;
        submission.add_residual_rms_norm_q8_1(
            &plan.projected_buffer,
            &plan.current_hidden_buffer,
            None,
            &self.post_attention_norm_device,
            &plan.current_hidden_buffer,
            &plan.hidden_norm_buffer,
            &plan.matvec_input_q8_1_buffer,
            hidden_size,
            epsilon,
        )?;
        submission.quantized_matvec_q8_1(
            &self.ffn_gate_up.storage,
            0,
            self.ffn_gate_up.mode,
            self.ffn_gate_up.total_rows(),
            self.ffn_gate_up.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.matvec_output_buffer,
        )?;
        let gate_rows = self.ffn_gate_up.rows_per_projection[0];
        let up_rows = self.ffn_gate_up.rows_per_projection[1];
        if gate_rows != up_rows {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 dense ffn gate/up width mismatch: gate={} up={}",
                    gate_rows, up_rows
                )),
            ));
        }
        submission.silu_mul_f32(
            &plan.matvec_output_buffer,
            0,
            &plan.matvec_output_buffer,
            gate_rows,
            gate_rows,
            &plan.hybrid_gated_buffer,
        )?;
        submission.quantize_f32_to_q8_1(
            &plan.hybrid_gated_buffer,
            1,
            gate_rows,
            &plan.activated_q8_1_buffer,
        )?;
        submission.quantized_matvec_q8_1(
            &self.ffn_down.storage,
            0,
            self.ffn_down.host.mode,
            self.ffn_down.host.rows,
            self.ffn_down.host.columns,
            &plan.activated_q8_1_buffer,
            None,
            &plan.projected_buffer,
        )?;
        submission.add_f32_in_place(
            &plan.current_hidden_buffer,
            0,
            &plan.projected_buffer,
            hidden_size,
        )?;
        let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
        *kernel_count = kernel_count.saturating_add(report.encoded_operations);
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum Qwen35LayerKind {
    Hybrid(Qwen35HybridLayer),
    FullAttention(Qwen35FullAttentionLayer),
}

#[derive(Clone, Debug)]
struct Qwen35HybridLayer {
    qkv_gate_alpha_beta: CudaQuantizedProjectionGroup,
    ssm_conv1d: DenseMatrix,
    ssm_conv1d_device: CudaBuffer,
    ssm_a: Vec<f32>,
    ssm_dt: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_norm_device: CudaBuffer,
    q_scale_device: CudaBuffer,
    k_scale_device: CudaBuffer,
    ssm_out: CudaQuantizedMatrix,
    inner_size: usize,
    state_size: usize,
    group_count: usize,
    time_step_rank: usize,
    conv_kernel: usize,
}

impl Qwen35HybridLayer {
    fn load(
        backend: &mut CudaBackend,
        artifact: &GgufBlobArtifact,
        layout: &psionic_models::GgufDecoderLayerTensorLayout,
        metadata: &GgufDecoderFamilyMetadata,
    ) -> Result<Self, ModelLoadError> {
        let ssm_conv1d = load_dense_matrix(
            artifact,
            required_tensor_name(layout.ssm_conv1d_weight.as_deref(), "ssm_conv1d")?,
        )?;
        let ssm_a = load_dense_vector(
            artifact,
            required_tensor_name(layout.ssm_a.as_deref(), "ssm_a")?,
        )?;
        let ssm_dt = load_dense_vector(
            artifact,
            required_tensor_name(layout.ssm_dt.as_deref(), "ssm_dt")?,
        )?;
        let ssm_norm = load_dense_vector(
            artifact,
            required_tensor_name(layout.ssm_norm_weight.as_deref(), "ssm_norm")?,
        )?;
        let state_size = family_fact_usize(metadata, "qwen35.ssm.state_size")?;
        let q_scale = 1.0_f32 / state_size as f32;
        let k_scale = 1.0_f32 / (state_size as f32).sqrt();
        Ok(Self {
            qkv_gate_alpha_beta: load_cuda_quantized_projection_group(
                backend,
                artifact,
                &[
                    required_tensor_name(layout.attention_qkv_weight.as_deref(), "attn_qkv")?,
                    required_tensor_name(layout.attention_gate_weight.as_deref(), "attn_gate")?,
                    required_tensor_name(layout.ssm_alpha_weight.as_deref(), "ssm_alpha")?,
                    required_tensor_name(layout.ssm_beta_weight.as_deref(), "ssm_beta")?,
                ],
            )?,
            ssm_conv1d_device: upload_f32_buffer(
                backend,
                ssm_conv1d.values.as_slice(),
                "qwen35_ssm_conv1d",
            )?,
            ssm_conv1d,
            ssm_a,
            ssm_dt,
            ssm_norm_device: upload_f32_buffer(backend, ssm_norm.as_slice(), "qwen35_ssm_norm")?,
            q_scale_device: upload_f32_buffer(
                backend,
                &vec![q_scale; state_size],
                "qwen35_ssm_q_scale",
            )?,
            k_scale_device: upload_f32_buffer(
                backend,
                &vec![k_scale; state_size],
                "qwen35_ssm_k_scale",
            )?,
            ssm_norm,
            ssm_out: load_cuda_quantized_matrix(
                backend,
                artifact,
                required_tensor_name(layout.ssm_out_weight.as_deref(), "ssm_out")?,
            )?,
            inner_size: family_fact_usize(metadata, "qwen35.ssm.inner_size")?,
            state_size,
            group_count: family_fact_usize(metadata, "qwen35.ssm.group_count")?,
            time_step_rank: family_fact_usize(metadata, "qwen35.ssm.time_step_rank")?,
            conv_kernel: family_fact_usize(metadata, "qwen35.ssm.conv_kernel")?,
        })
    }

    fn initial_state(
        &self,
        backend: &mut CudaBackend,
    ) -> Result<Qwen35HybridState, ReferenceTextGenerationError> {
        Ok(Qwen35HybridState {
            conv_state: backend
                .f32_buffer(
                    self.qkv_gate_alpha_beta.rows_per_projection[0]
                        .saturating_mul(self.conv_kernel.saturating_sub(1)),
                )
                .map_err(ReferenceTextGenerationError::Runtime)?,
            delta_state: backend
                .f32_buffer(
                    self.time_step_rank
                        .saturating_mul(self.state_size)
                        .saturating_mul(self.state_size),
                )
                .map_err(ReferenceTextGenerationError::Runtime)?,
        })
    }

    fn device_residency_bytes(&self) -> usize {
        let aux_bytes = self
            .ssm_conv1d
            .host_residency_bytes()
            .saturating_add(vec_f32_bytes(self.ssm_norm.as_slice()))
            .saturating_add(self.state_size.saturating_mul(std::mem::size_of::<f32>()))
            .saturating_add(self.state_size.saturating_mul(std::mem::size_of::<f32>()));
        self.qkv_gate_alpha_beta
            .device_residency_bytes()
            .saturating_add(self.ssm_out.device_residency_bytes())
            .saturating_add(aux_bytes)
    }

    fn host_residency_bytes(&self) -> usize {
        self.ssm_conv1d
            .host_residency_bytes()
            .saturating_add(vec_f32_bytes(self.ssm_a.as_slice()))
            .saturating_add(vec_f32_bytes(self.ssm_dt.as_slice()))
            .saturating_add(vec_f32_bytes(self.ssm_norm.as_slice()))
    }

    fn max_matvec_input_columns(&self) -> usize {
        usize::max(self.qkv_gate_alpha_beta.columns, self.ssm_out.host.columns)
    }

    fn max_matvec_output_rows(&self) -> usize {
        usize::max(
            self.qkv_gate_alpha_beta.total_rows(),
            self.ssm_out.host.rows,
        )
    }

    fn forward(
        &self,
        _backend: &mut CudaBackend,
        _plan: &mut Qwen35CudaStepPlan,
        _model: &CudaQwen35Model,
        _position: usize,
        _hidden: &[f32],
        _state: &mut Qwen35HybridState,
        _kernel_count: &mut usize,
        _bytes_moved: &mut u64,
    ) -> Result<Vec<f32>, ReferenceTextGenerationError> {
        Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(String::from(
                "qwen35 hybrid host path is disabled; use forward_hybrid_device",
            )),
        ))
    }
}

#[derive(Clone, Debug)]
struct Qwen35HybridState {
    conv_state: CudaBuffer,
    delta_state: CudaBuffer,
}

#[derive(Clone, Debug)]
struct Qwen35FullAttentionLayer {
    qkv: CudaQuantizedProjectionGroup,
    query_norm: Vec<f32>,
    query_norm_device: CudaBuffer,
    key_norm: Vec<f32>,
    key_norm_device: CudaBuffer,
    output: CudaQuantizedMatrix,
    kv_width: usize,
}

impl Qwen35FullAttentionLayer {
    fn load(
        backend: &mut CudaBackend,
        artifact: &GgufBlobArtifact,
        layout: &psionic_models::GgufDecoderLayerTensorLayout,
        _metadata: &GgufDecoderFamilyMetadata,
    ) -> Result<Self, ModelLoadError> {
        let qkv = load_cuda_quantized_projection_group(
            backend,
            artifact,
            &[
                required_tensor_name(layout.attention_query_weight.as_deref(), "attn_q")?,
                required_tensor_name(layout.attention_key_weight.as_deref(), "attn_k")?,
                required_tensor_name(layout.attention_value_weight.as_deref(), "attn_v")?,
            ],
        )?;
        let query_norm = load_dense_vector(
            artifact,
            required_tensor_name(layout.attention_query_norm.as_deref(), "attn_q_norm")?,
        )?;
        let key_norm = load_dense_vector(
            artifact,
            required_tensor_name(layout.attention_key_norm.as_deref(), "attn_k_norm")?,
        )?;
        Ok(Self {
            kv_width: qkv.rows_per_projection[1],
            qkv,
            query_norm_device: upload_f32_buffer(
                backend,
                query_norm.as_slice(),
                "qwen35_attention_query_norm",
            )?,
            query_norm,
            key_norm_device: upload_f32_buffer(
                backend,
                key_norm.as_slice(),
                "qwen35_attention_key_norm",
            )?,
            key_norm,
            output: load_cuda_quantized_matrix(
                backend,
                artifact,
                required_tensor_name(layout.attention_output_weight.as_deref(), "attn_output")?,
            )?,
        })
    }

    fn initial_state(
        &self,
        backend: &mut CudaBackend,
        cache_capacity_tokens: usize,
    ) -> Result<Qwen35FullAttentionState, ReferenceTextGenerationError> {
        let cache_bytes = cache_capacity_tokens
            .saturating_mul(self.kv_width)
            .saturating_mul(std::mem::size_of::<u16>());
        Ok(Qwen35FullAttentionState {
            key_cache: backend
                .byte_buffer(&vec![0_u8; cache_bytes])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            value_cache: backend
                .byte_buffer(&vec![0_u8; cache_bytes])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            width: self.kv_width,
            element_bytes: std::mem::size_of::<u16>(),
            len: 0,
            capacity_tokens: cache_capacity_tokens,
        })
    }

    fn device_residency_bytes(&self) -> usize {
        self.qkv
            .device_residency_bytes()
            .saturating_add(vec_f32_bytes(self.query_norm.as_slice()))
            .saturating_add(vec_f32_bytes(self.key_norm.as_slice()))
            .saturating_add(self.output.device_residency_bytes())
    }

    fn host_residency_bytes(&self) -> usize {
        vec_f32_bytes(self.query_norm.as_slice())
            .saturating_add(vec_f32_bytes(self.key_norm.as_slice()))
    }

    fn max_matvec_input_columns(&self) -> usize {
        usize::max(self.qkv.columns, self.output.host.columns)
    }

    fn max_matvec_output_rows(&self) -> usize {
        usize::max(self.qkv.total_rows(), self.output.host.rows)
    }

    fn forward(
        &self,
        _backend: &mut CudaBackend,
        _plan: &mut Qwen35CudaStepPlan,
        _model: &CudaQwen35Model,
        _position: usize,
        _hidden: &[f32],
        _state: &mut Qwen35FullAttentionState,
        _kernel_count: &mut usize,
        _bytes_moved: &mut u64,
    ) -> Result<Vec<f32>, ReferenceTextGenerationError> {
        Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(String::from(
                "qwen35 full-attention host path is disabled; use forward_full_attention_device",
            )),
        ))
    }
}

#[derive(Clone, Debug)]
struct Qwen35FullAttentionState {
    key_cache: CudaBuffer,
    value_cache: CudaBuffer,
    width: usize,
    element_bytes: usize,
    len: usize,
    capacity_tokens: usize,
}

impl Qwen35FullAttentionState {
    fn ensure_capacity(
        &mut self,
        backend: &mut CudaBackend,
        required_tokens: usize,
    ) -> Result<(), crate::RuntimeError> {
        if required_tokens <= self.capacity_tokens {
            return Ok(());
        }
        let new_capacity = required_tokens
            .max(self.capacity_tokens.saturating_mul(2))
            .checked_next_power_of_two()
            .unwrap_or(required_tokens);
        let token_bytes = self.width.saturating_mul(self.element_bytes);
        let new_cache_bytes = new_capacity.saturating_mul(token_bytes);
        let new_key_cache = backend.byte_buffer(&vec![0_u8; new_cache_bytes])?;
        let new_value_cache = backend.byte_buffer(&vec![0_u8; new_cache_bytes])?;
        if self.len > 0 {
            let copy_bytes = self.len.saturating_mul(token_bytes);
            let mut submission = backend.begin_submission()?;
            submission.copy_buffer_region(&self.key_cache, 0, &new_key_cache, 0, copy_bytes)?;
            submission.copy_buffer_region(&self.value_cache, 0, &new_value_cache, 0, copy_bytes)?;
            submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
        }
        self.key_cache = new_key_cache;
        self.value_cache = new_value_cache;
        self.capacity_tokens = new_capacity;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Qwen35FullAttentionEntry {
    key: Vec<f32>,
    value: Vec<f32>,
}

#[derive(Clone, Debug)]
struct Qwen35State {
    position: usize,
    layers: Vec<Qwen35LayerState>,
}

#[derive(Clone, Debug)]
enum Qwen35LayerState {
    Hybrid(Qwen35HybridState),
    FullAttention(Qwen35FullAttentionState),
}

#[derive(Clone, Copy, Debug)]
enum CudaStepOutputMode {
    NoOutput,
    FullLogits,
    ArgmaxOnly,
}

#[derive(Clone, Debug)]
struct Qwen35ForwardStep {
    logits: Vec<f32>,
    selected_token: Option<TokenId>,
    kernel_count: usize,
    bytes_moved: u64,
}

#[derive(Debug)]
struct Qwen35CudaStepPlan {
    matvec_input_buffer: CudaBuffer,
    matvec_input_q8_1_buffer: CudaBuffer,
    matvec_output_buffer: CudaBuffer,
    current_hidden_buffer: CudaBuffer,
    hidden_norm_buffer: CudaBuffer,
    gate_buffer: CudaBuffer,
    q_buffer: CudaBuffer,
    k_buffer: CudaBuffer,
    qkv_norm_buffer: CudaBuffer,
    conv_buffer: CudaBuffer,
    gated_delta_buffer: CudaBuffer,
    hybrid_norm_buffer: CudaBuffer,
    hybrid_gated_buffer: CudaBuffer,
    projected_buffer: CudaBuffer,
    activated_q8_1_buffer: CudaBuffer,
    decay_buffer: CudaBuffer,
    beta_buffer: CudaBuffer,
    ones_buffer: CudaBuffer,
    decay_host: Vec<f32>,
    beta_host: Vec<f32>,
    logits_buffer: CudaBuffer,
    next_token_host_buffer: CudaHostBuffer,
    next_token_buffer: CudaBuffer,
    argmax_state_host_buffer: CudaHostBuffer,
    argmax_state_buffer: CudaBuffer,
}

impl Qwen35CudaStepPlan {
    fn new(
        backend: &mut CudaBackend,
        hidden_size: usize,
        max_input_columns: usize,
        max_output_rows: usize,
        vocab_size: usize,
    ) -> Result<Self, ReferenceTextGenerationError> {
        let q8_1_bytes = ggml_q8_1_storage_bytes(1, max_input_columns)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let activated_q8_1_bytes = ggml_q8_1_storage_bytes(1, max_output_rows)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let mut ones_buffer = backend
            .f32_buffer(max_output_rows)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        ones_buffer
            .write_f32(&vec![1.0_f32; max_output_rows])
            .map_err(ReferenceTextGenerationError::Runtime)?;
        Ok(Self {
            matvec_input_buffer: backend
                .f32_buffer(max_input_columns)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            matvec_input_q8_1_buffer: backend
                .byte_buffer(&vec![0_u8; q8_1_bytes])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            matvec_output_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            current_hidden_buffer: backend
                .f32_buffer(hidden_size)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            hidden_norm_buffer: backend
                .f32_buffer(hidden_size)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            gate_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            q_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            k_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            qkv_norm_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            conv_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            gated_delta_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            hybrid_norm_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            hybrid_gated_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            projected_buffer: backend
                .f32_buffer(hidden_size)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            activated_q8_1_buffer: backend
                .byte_buffer(&vec![0_u8; activated_q8_1_bytes])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            decay_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            beta_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            ones_buffer,
            decay_host: Vec::new(),
            beta_host: Vec::new(),
            logits_buffer: backend
                .f32_buffer(vocab_size)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            next_token_host_buffer: backend
                .host_buffer(std::mem::size_of::<i32>())
                .map_err(ReferenceTextGenerationError::Runtime)?,
            next_token_buffer: backend
                .i32_buffer(1)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            argmax_state_host_buffer: backend
                .host_buffer(std::mem::size_of::<u64>())
                .map_err(ReferenceTextGenerationError::Runtime)?,
            argmax_state_buffer: backend
                .byte_buffer(&vec![0_u8; std::mem::size_of::<u64>()])
                .map_err(ReferenceTextGenerationError::Runtime)?,
        })
    }

    fn run_projection_matvec(
        &mut self,
        backend: &mut CudaBackend,
        weights: &CudaBuffer,
        byte_offset: usize,
        mode: QuantizationMode,
        rows: usize,
        cols: usize,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<CudaQuantizedMatvecStats, crate::RuntimeError> {
        if input.len() != cols {
            return Err(crate::RuntimeError::Backend(format!(
                "qwen35 cuda matvec input width mismatch: expected {cols}, actual {}",
                input.len()
            )));
        }
        self.matvec_input_buffer.write_f32_at_offset(0, input)?;
        let mut submission = backend.begin_submission()?;
        if can_use_q8_1_mmvq(mode) {
            submission.quantize_f32_to_q8_1(
                &self.matvec_input_buffer,
                1,
                cols,
                &self.matvec_input_q8_1_buffer,
            )?;
            submission.quantized_matvec_q8_1(
                weights,
                byte_offset,
                mode,
                rows,
                cols,
                &self.matvec_input_q8_1_buffer,
                None,
                &self.matvec_output_buffer,
            )?;
        } else {
            submission.quantized_matvec(
                weights,
                byte_offset,
                mode,
                rows,
                cols,
                &self.matvec_input_buffer,
                &self.matvec_output_buffer,
            )?;
        }
        let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
        *output = self.matvec_output_buffer.read_f32_at_offset(0, rows)?;
        Ok(CudaQuantizedMatvecStats {
            host_to_device_bytes: input
                .len()
                .saturating_mul(std::mem::size_of::<f32>())
                .try_into()
                .unwrap_or(u64::MAX),
            device_to_host_bytes: rows
                .saturating_mul(std::mem::size_of::<f32>())
                .try_into()
                .unwrap_or(u64::MAX),
            submission_count: 1,
            sync_count: 1,
            kernel_launches: report.encoded_operations,
        })
    }

    fn run_output_logits(
        &mut self,
        backend: &mut CudaBackend,
        weights: &CudaBuffer,
        mode: QuantizationMode,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<(Vec<f32>, CudaQuantizedMatvecStats), crate::RuntimeError> {
        if input.len() != cols {
            return Err(crate::RuntimeError::Backend(format!(
                "qwen35 cuda logits input width mismatch: expected {cols}, actual {}",
                input.len()
            )));
        }
        self.matvec_input_buffer.write_f32_at_offset(0, input)?;
        let mut submission = backend.begin_submission()?;
        if can_use_q8_1_mmvq(mode) {
            submission.quantize_f32_to_q8_1(
                &self.matvec_input_buffer,
                1,
                cols,
                &self.matvec_input_q8_1_buffer,
            )?;
            submission.quantized_matvec_q8_1(
                weights,
                0,
                mode,
                rows,
                cols,
                &self.matvec_input_q8_1_buffer,
                None,
                &self.logits_buffer,
            )?;
        } else {
            submission.quantized_matvec(
                weights,
                0,
                mode,
                rows,
                cols,
                &self.matvec_input_buffer,
                &self.logits_buffer,
            )?;
        }
        let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
        let logits = self.logits_buffer.read_f32_at_offset(0, rows)?;
        Ok((
            logits,
            CudaQuantizedMatvecStats {
                host_to_device_bytes: input
                    .len()
                    .saturating_mul(std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap_or(u64::MAX),
                device_to_host_bytes: rows
                    .saturating_mul(std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap_or(u64::MAX),
                submission_count: 1,
                sync_count: 1,
                kernel_launches: report.encoded_operations,
            },
        ))
    }

    fn run_output_logits_from_device(
        &mut self,
        backend: &mut CudaBackend,
        input: &CudaBuffer,
        norm_weight: &CudaBuffer,
        epsilon: f32,
        weights: &CudaBuffer,
        mode: QuantizationMode,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<f32>, CudaQuantizedMatvecStats), crate::RuntimeError> {
        let mut submission = backend.begin_submission()?;
        if can_use_q8_1_mmvq(mode) {
            submission.rms_norm_q8_1(
                input,
                norm_weight,
                &self.matvec_input_q8_1_buffer,
                cols,
                epsilon,
            )?;
            submission.quantized_matvec_q8_1(
                weights,
                0,
                mode,
                rows,
                cols,
                &self.matvec_input_q8_1_buffer,
                None,
                &self.logits_buffer,
            )?;
        } else {
            submission.rms_norm(input, norm_weight, &self.matvec_input_buffer, cols, epsilon)?;
            submission.quantized_matvec(
                weights,
                0,
                mode,
                rows,
                cols,
                &self.matvec_input_buffer,
                &self.logits_buffer,
            )?;
        }
        let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
        let logits = self.logits_buffer.read_f32_at_offset(0, rows)?;
        Ok((
            logits,
            CudaQuantizedMatvecStats {
                host_to_device_bytes: 0,
                device_to_host_bytes: rows
                    .saturating_mul(std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap_or(u64::MAX),
                submission_count: 1,
                sync_count: 1,
                kernel_launches: report.encoded_operations,
            },
        ))
    }

    fn run_output_argmax(
        &mut self,
        backend: &mut CudaBackend,
        weights: &CudaBuffer,
        mode: QuantizationMode,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<(TokenId, CudaQuantizedMatvecStats), crate::RuntimeError> {
        if input.len() != cols {
            return Err(crate::RuntimeError::Backend(format!(
                "qwen35 cuda argmax input width mismatch: expected {cols}, actual {}",
                input.len()
            )));
        }
        self.matvec_input_buffer.write_f32_at_offset(0, input)?;
        let mut submission = backend.begin_submission()?;
        let (selected, device_to_host_bytes, kernel_launches) = if can_use_q8_1_mmvq(mode) {
            self.argmax_state_host_buffer
                .write_bytes(initial_cuda_argmax_pair_bytes().as_slice())?;
            submission.quantize_f32_to_q8_1(
                &self.matvec_input_buffer,
                1,
                cols,
                &self.matvec_input_q8_1_buffer,
            )?;
            submission
                .copy_host_to_device(&self.argmax_state_host_buffer, &self.argmax_state_buffer)?;
            submission.quantized_matvec_q8_1_argmax(
                weights,
                0,
                mode,
                rows,
                cols,
                &self.matvec_input_q8_1_buffer,
                None,
                &self.argmax_state_buffer,
            )?;
            submission
                .copy_device_to_host(&self.argmax_state_buffer, &self.argmax_state_host_buffer)?;
            let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
            (
                cuda_argmax_token_from_packed_host_buffer(&self.argmax_state_host_buffer).map_err(
                    |error| match error {
                        ReferenceTextGenerationError::Runtime(runtime) => runtime,
                        other => crate::RuntimeError::Backend(other.to_string()),
                    },
                )?,
                std::mem::size_of::<u64>().try_into().unwrap_or(u64::MAX),
                report.encoded_operations,
            )
        } else {
            submission.quantized_matvec(
                weights,
                0,
                mode,
                rows,
                cols,
                &self.matvec_input_buffer,
                &self.logits_buffer,
            )?;
            submission.argmax_f32(&self.logits_buffer, 1, rows, &self.next_token_buffer)?;
            submission
                .copy_device_to_host(&self.next_token_buffer, &self.next_token_host_buffer)?;
            let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
            (
                cuda_argmax_token_id(self.next_token_host_buffer.read_i32()?).map_err(|error| {
                    match error {
                        ReferenceTextGenerationError::Runtime(runtime) => runtime,
                        other => crate::RuntimeError::Backend(other.to_string()),
                    }
                })?,
                std::mem::size_of::<i32>().try_into().unwrap_or(u64::MAX),
                report.encoded_operations,
            )
        };
        Ok((
            selected,
            CudaQuantizedMatvecStats {
                host_to_device_bytes: input
                    .len()
                    .saturating_mul(std::mem::size_of::<f32>())
                    .saturating_add(std::mem::size_of::<u64>())
                    .try_into()
                    .unwrap_or(u64::MAX),
                device_to_host_bytes,
                submission_count: 1,
                sync_count: 1,
                kernel_launches,
            },
        ))
    }

    fn run_output_argmax_from_device(
        &mut self,
        backend: &mut CudaBackend,
        input: &CudaBuffer,
        norm_weight: &CudaBuffer,
        epsilon: f32,
        weights: &CudaBuffer,
        mode: QuantizationMode,
        rows: usize,
        cols: usize,
    ) -> Result<(TokenId, CudaQuantizedMatvecStats), crate::RuntimeError> {
        let mut submission = backend.begin_submission()?;
        let (selected, host_to_device_bytes, device_to_host_bytes, kernel_launches) =
            if can_use_q8_1_mmvq(mode) {
                self.argmax_state_host_buffer
                    .write_bytes(initial_cuda_argmax_pair_bytes().as_slice())?;
                submission.rms_norm_q8_1(
                    input,
                    norm_weight,
                    &self.matvec_input_q8_1_buffer,
                    cols,
                    epsilon,
                )?;
                submission.copy_host_to_device(
                    &self.argmax_state_host_buffer,
                    &self.argmax_state_buffer,
                )?;
                submission.quantized_matvec_q8_1_argmax(
                    weights,
                    0,
                    mode,
                    rows,
                    cols,
                    &self.matvec_input_q8_1_buffer,
                    None,
                    &self.argmax_state_buffer,
                )?;
                submission.copy_device_to_host(
                    &self.argmax_state_buffer,
                    &self.argmax_state_host_buffer,
                )?;
                let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
                (
                    cuda_argmax_token_from_packed_host_buffer(&self.argmax_state_host_buffer)
                        .map_err(|error| match error {
                            ReferenceTextGenerationError::Runtime(runtime) => runtime,
                            other => crate::RuntimeError::Backend(other.to_string()),
                        })?,
                    std::mem::size_of::<u64>().try_into().unwrap_or(u64::MAX),
                    std::mem::size_of::<u64>().try_into().unwrap_or(u64::MAX),
                    report.encoded_operations,
                )
            } else {
                submission.rms_norm(
                    input,
                    norm_weight,
                    &self.matvec_input_buffer,
                    cols,
                    epsilon,
                )?;
                submission.quantized_matvec(
                    weights,
                    0,
                    mode,
                    rows,
                    cols,
                    &self.matvec_input_buffer,
                    &self.logits_buffer,
                )?;
                submission.argmax_f32(&self.logits_buffer, 1, rows, &self.next_token_buffer)?;
                submission
                    .copy_device_to_host(&self.next_token_buffer, &self.next_token_host_buffer)?;
                let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
                (
                    cuda_argmax_token_id(self.next_token_host_buffer.read_i32()?).map_err(
                        |error| match error {
                            ReferenceTextGenerationError::Runtime(runtime) => runtime,
                            other => crate::RuntimeError::Backend(other.to_string()),
                        },
                    )?,
                    0,
                    std::mem::size_of::<i32>().try_into().unwrap_or(u64::MAX),
                    report.encoded_operations,
                )
            };
        Ok((
            selected,
            CudaQuantizedMatvecStats {
                host_to_device_bytes,
                device_to_host_bytes,
                submission_count: 1,
                sync_count: 1,
                kernel_launches,
            },
        ))
    }
}

#[derive(Clone, Debug)]
struct CompletedQwen35Stream {
    policy: GenerationStreamingPolicy,
    chunk: Option<GenerationStreamChunk>,
    terminal: Option<GenerationStreamTerminal>,
}

impl CompletedQwen35Stream {
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

impl GenerationEventStream for CompletedQwen35Stream {
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
        self.terminal.take().map(|mut terminal| {
            terminal.status = GenerationStreamStatus::Cancelled;
            terminal.failure_reason = Some(String::from("stream cancelled by caller"));
            terminal.diagnostic = Some(
                LocalRuntimeDiagnostic::new(
                    psionic_runtime::LocalRuntimeErrorCode::Cancelled,
                    499,
                    "stream cancelled by caller",
                )
                .with_backend("cuda"),
            );
            terminal
        })
    }

    fn disconnect(&mut self) -> Option<GenerationStreamTerminal> {
        self.chunk.take();
        self.terminal.take().map(|mut terminal| {
            terminal.status = GenerationStreamStatus::Disconnected;
            terminal.failure_reason = Some(String::from("stream disconnected by caller"));
            terminal.diagnostic = Some(
                LocalRuntimeDiagnostic::new(
                    psionic_runtime::LocalRuntimeErrorCode::Disconnected,
                    499,
                    "stream disconnected by caller",
                )
                .with_backend("cuda"),
            );
            terminal
        })
    }
}

#[derive(Clone, Debug)]
struct HostMatrix {
    kind: HostMatrixKind,
}

#[derive(Clone, Debug)]
enum HostMatrixKind {
    Dense(DenseMatrix),
    Quantized(QuantizedMatrix),
}

impl HostMatrix {
    fn load(artifact: &GgufBlobArtifact, name: &str) -> Result<Self, ModelLoadError> {
        let storage = artifact.paged_tensor(name)?;
        if storage.metadata().quantized_layout.is_some() {
            let matrix = load_quantized_matrix(artifact, name)?;
            return Ok(Self {
                kind: HostMatrixKind::Quantized(matrix),
            });
        }
        let matrix = load_dense_matrix(artifact, name)?;
        Ok(Self {
            kind: HostMatrixKind::Dense(matrix),
        })
    }

    fn decode_row(&self, row_index: usize) -> Result<Vec<f32>, ReferenceTextGenerationError> {
        match &self.kind {
            HostMatrixKind::Dense(matrix) => matrix
                .decode_row(row_index)
                .map_err(ReferenceTextGenerationError::Runtime),
            HostMatrixKind::Quantized(matrix) => matrix
                .decode_row(row_index)
                .map_err(ReferenceTextGenerationError::Runtime),
        }
    }

    fn host_residency_bytes(&self) -> usize {
        match &self.kind {
            HostMatrixKind::Dense(matrix) => matrix.host_residency_bytes(),
            HostMatrixKind::Quantized(matrix) => matrix.byte_length(),
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

    fn host_residency_bytes(&self) -> usize {
        vec_f32_bytes(self.values.as_slice())
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
}

#[derive(Clone, Debug)]
struct CudaQuantizedMatrix {
    storage: CudaBuffer,
    host: QuantizedMatrix,
}

impl CudaQuantizedMatrix {
    fn device_residency_bytes(&self) -> usize {
        self.storage.byte_len()
    }

    fn matvec_profiled(
        &self,
        backend: &mut CudaBackend,
        plan: &mut Qwen35CudaStepPlan,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<CudaQuantizedMatvecStats, crate::RuntimeError> {
        plan.run_projection_matvec(
            backend,
            &self.storage,
            0,
            self.host.mode,
            self.host.rows,
            self.host.columns,
            input,
            output,
        )
    }
}

#[derive(Clone, Debug)]
struct CudaQuantizedProjectionGroup {
    storage: CudaBuffer,
    rows_per_projection: Vec<usize>,
    columns: usize,
    mode: QuantizationMode,
}

#[derive(Clone, Debug)]
struct ProjectionOutputs {
    values: Vec<f32>,
    spans: Vec<(usize, usize)>,
}

impl ProjectionOutputs {
    fn slice(&self, index: usize) -> Result<&[f32], crate::RuntimeError> {
        let Some((start, end)) = self.spans.get(index).copied() else {
            return Err(crate::RuntimeError::Backend(format!(
                "projection output index {index} exceeds projection count {}",
                self.spans.len()
            )));
        };
        Ok(&self.values[start..end])
    }
}

impl CudaQuantizedProjectionGroup {
    fn total_rows(&self) -> usize {
        self.rows_per_projection
            .iter()
            .copied()
            .fold(0usize, usize::saturating_add)
    }

    fn device_residency_bytes(&self) -> usize {
        self.storage.byte_len()
    }

    fn matvec_profiled(
        &self,
        backend: &mut CudaBackend,
        plan: &mut Qwen35CudaStepPlan,
        input: &[f32],
    ) -> Result<(ProjectionOutputs, CudaQuantizedMatvecStats), crate::RuntimeError> {
        let mut values = Vec::new();
        let stats = plan.run_projection_matvec(
            backend,
            &self.storage,
            0,
            self.mode,
            self.total_rows(),
            self.columns,
            input,
            &mut values,
        )?;
        Ok((
            ProjectionOutputs::new(self.rows_per_projection.as_slice(), values)?,
            stats,
        ))
    }
}

impl ProjectionOutputs {
    fn new(rows_per_projection: &[usize], values: Vec<f32>) -> Result<Self, crate::RuntimeError> {
        let expected = rows_per_projection
            .iter()
            .copied()
            .fold(0usize, usize::saturating_add);
        if values.len() != expected {
            return Err(crate::RuntimeError::Backend(format!(
                "packed projection output mismatch: expected {expected} values, actual {}",
                values.len()
            )));
        }
        let mut spans = Vec::with_capacity(rows_per_projection.len());
        let mut offset = 0usize;
        for rows in rows_per_projection {
            let end = offset.saturating_add(*rows);
            spans.push((offset, end));
            offset = end;
        }
        Ok(Self { values, spans })
    }
}

fn load_quantized_matrix(
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<QuantizedMatrix, ModelLoadError> {
    let storage = artifact.paged_tensor(name)?;
    let metadata = storage.metadata().clone();
    let dims = metadata.shape.dims().to_vec();
    let [rows, columns] = dims.as_slice() else {
        return Err(ModelLoadError::InvalidTensorShape {
            name: metadata.name.clone(),
            expected: vec![0, 0],
            actual: dims,
        });
    };
    let layout =
        metadata
            .quantized_layout
            .ok_or_else(|| ModelLoadError::UnsupportedTensorDType {
                name: metadata.name.clone(),
                dtype: String::from("quantized"),
            })?;
    let row_byte_len = quantized_row_byte_len(&metadata.shape, layout).map_err(|_| {
        ModelLoadError::InvalidQuantizedTensorShape {
            quantization: metadata.quantization,
            shape: metadata.shape.dims().to_vec(),
        }
    })?;
    Ok(QuantizedMatrix {
        storage,
        mode: metadata.quantization,
        rows: *rows,
        columns: *columns,
        row_byte_len,
    })
}

fn load_cuda_quantized_matrix(
    backend: &mut CudaBackend,
    artifact: &GgufBlobArtifact,
    name: &str,
) -> Result<CudaQuantizedMatrix, ModelLoadError> {
    let host = load_quantized_matrix(artifact, name)?;
    let storage = backend
        .byte_buffer(host.storage.bytes()?)
        .map_err(|error| ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!("failed to upload `{name}` to cuda: {error}"),
        })?;
    Ok(CudaQuantizedMatrix { storage, host })
}

fn load_cuda_quantized_projection_group(
    backend: &mut CudaBackend,
    artifact: &GgufBlobArtifact,
    names: &[&str],
) -> Result<CudaQuantizedProjectionGroup, ModelLoadError> {
    let mut mode = None;
    let mut columns = None;
    let mut row_byte_len = None;
    let mut rows_per_projection = Vec::with_capacity(names.len());
    let mut projection_bytes = Vec::with_capacity(names.len());
    for name in names {
        let projection = load_quantized_matrix(artifact, name)?;
        if let Some(expected_mode) = mode {
            if projection.mode != expected_mode {
                return Err(ModelLoadError::ArtifactFormat {
                    format: String::from("gguf"),
                    message: format!(
                        "packed qwen35 cuda projection requires matching quantization; `{name}` had {:?} but expected {:?}",
                        projection.mode, expected_mode
                    ),
                });
            }
        } else {
            mode = Some(projection.mode);
        }
        if let Some(expected_columns) = columns {
            if projection.columns != expected_columns {
                return Err(ModelLoadError::ArtifactFormat {
                    format: String::from("gguf"),
                    message: format!(
                        "packed qwen35 cuda projection requires matching input width; `{name}` had {} but expected {}",
                        projection.columns, expected_columns
                    ),
                });
            }
        } else {
            columns = Some(projection.columns);
        }
        if let Some(expected_row_byte_len) = row_byte_len {
            if projection.row_byte_len != expected_row_byte_len {
                return Err(ModelLoadError::ArtifactFormat {
                    format: String::from("gguf"),
                    message: format!(
                        "packed qwen35 cuda projection requires matching row layout; `{name}` had {} but expected {}",
                        projection.row_byte_len, expected_row_byte_len
                    ),
                });
            }
        } else {
            row_byte_len = Some(projection.row_byte_len);
        }
        rows_per_projection.push(projection.rows);
        projection_bytes.push(projection.storage.bytes()?.to_vec());
    }
    let packed = pack_quantized_projection_bytes(
        projection_bytes
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let resolved_mode = mode.expect("projection group should resolve quantization mode");
    let resolved_columns = columns.expect("projection group should resolve columns");
    let _resolved_row_byte_len =
        row_byte_len.expect("projection group should resolve row byte length");
    let storage =
        backend
            .byte_buffer(packed.as_slice())
            .map_err(|error| ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "failed to upload packed qwen35 cuda projection `{}`: {error}",
                    names.join(", ")
                ),
            })?;
    Ok(CudaQuantizedProjectionGroup {
        storage,
        rows_per_projection,
        columns: resolved_columns,
        mode: resolved_mode,
    })
}

fn pack_quantized_projection_bytes(projections: &[&[u8]]) -> Vec<u8> {
    let total = projections
        .iter()
        .copied()
        .fold(0usize, |sum, bytes| sum.saturating_add(bytes.len()));
    let mut packed = Vec::with_capacity(total);
    for bytes in projections {
        packed.extend_from_slice(bytes);
    }
    packed
}

fn family_fact_usize(
    metadata: &GgufDecoderFamilyMetadata,
    key: &str,
) -> Result<usize, ModelLoadError> {
    metadata
        .family_facts
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!("missing required qwen35 family fact `{key}`"),
        })
}

fn gguf_local_blob_open_options() -> LocalBlobOpenOptions {
    LocalBlobOpenOptions::default().with_integrity_policy(BlobIntegrityPolicy::LocalUnverifiedLabel)
}

fn digest_qwen35_cuda_plan(
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
    hasher.update(metadata.architecture.as_bytes());
    hasher.update(b"|qwen35-native-cuda|v1");
    hex::encode(hasher.finalize())
}

fn required_tensor_name<'a>(name: Option<&'a str>, field: &str) -> Result<&'a str, ModelLoadError> {
    name.ok_or_else(|| ModelLoadError::ArtifactFormat {
        format: String::from("gguf"),
        message: format!("missing required qwen35 tensor `{field}`"),
    })
}

fn load_dense_vector(artifact: &GgufBlobArtifact, name: &str) -> Result<Vec<f32>, ModelLoadError> {
    artifact
        .load_tensor(name)?
        .values()
        .map(|values| values.into_owned())
}

fn upload_f32_buffer(
    backend: &mut CudaBackend,
    values: &[f32],
    name: &str,
) -> Result<CudaBuffer, ModelLoadError> {
    let mut buffer =
        backend
            .f32_buffer(values.len())
            .map_err(|error| ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!("failed to allocate cuda f32 buffer for `{name}`: {error}"),
            })?;
    buffer
        .write_f32(values)
        .map_err(|error| ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!("failed to upload cuda f32 buffer for `{name}`: {error}"),
        })?;
    Ok(buffer)
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

fn model_load_runtime_error(error: ModelLoadError) -> crate::RuntimeError {
    crate::RuntimeError::Backend(error.to_string())
}

fn zero_cuda_matvec_stats() -> CudaQuantizedMatvecStats {
    CudaQuantizedMatvecStats {
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        submission_count: 0,
        sync_count: 0,
        kernel_launches: 0,
    }
}

fn cuda_stats_bytes(stats: CudaQuantizedMatvecStats) -> u64 {
    stats
        .host_to_device_bytes
        .saturating_add(stats.device_to_host_bytes)
}

fn vec_f32_bytes(values: &[f32]) -> usize {
    values.len().saturating_mul(std::mem::size_of::<f32>())
}

fn qwen35_rope_runtime_parameters(
    rotary_dim: usize,
    metadata: &GgufDecoderFamilyMetadata,
) -> (f32, f32, [f32; 2], f32) {
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
        .map(|original| rope_yarn_corr_dims(rotary_dim, original, metadata.rope_theta))
        .unwrap_or([0.0, rotary_dim as f32 - 1.0]);
    let theta_scale = metadata.rope_theta.powf(-2.0 / rotary_dim as f32);
    (freq_scale, ext_factor, corr_dims, theta_scale)
}

fn f16_bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, crate::RuntimeError> {
    if bytes.len() % std::mem::size_of::<u16>() != 0 {
        return Err(crate::RuntimeError::Backend(format!(
            "f16 byte buffer length must be divisible by 2, actual {}",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(bytes.len() / std::mem::size_of::<u16>());
    for chunk in bytes.chunks_exact(std::mem::size_of::<u16>()) {
        values.push(f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    Ok(values)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    let value = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut normalized = u32::from(mantissa);
            let mut shift = 0_u32;
            while (normalized & 0x0400) == 0 {
                normalized <<= 1;
                shift = shift.saturating_add(1);
            }
            normalized &= 0x03ff;
            sign | ((113_u32.saturating_sub(shift)) << 23) | (normalized << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (u32::from(mantissa) << 13)
    } else {
        sign | ((u32::from(exponent) + 112) << 23) | (u32::from(mantissa) << 13)
    };
    f32::from_bits(value)
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + epsilon).sqrt().recip();
    input
        .iter()
        .zip(weight.iter())
        .map(|(value, weight)| value * scale * weight)
        .collect()
}

fn per_head_rms_norm(
    input: &[f32],
    head_count: usize,
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let mut normalized = vec![0.0_f32; input.len()];
    per_head_rms_norm_into(
        input,
        head_count,
        head_dim,
        weight,
        epsilon,
        &mut normalized,
    );
    normalized
}

fn per_head_rms_norm_into(
    input: &[f32],
    head_count: usize,
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) {
    for head_index in 0..head_count {
        let start = head_index.saturating_mul(head_dim);
        let end = start.saturating_add(head_dim);
        let input_head = &input[start..end];
        let output_head = &mut output[start..end];
        let mean_square =
            input_head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let scale = (mean_square + epsilon).sqrt().recip();
        for ((out, value), weight) in output_head
            .iter_mut()
            .zip(input_head.iter().copied())
            .zip(weight.iter().copied())
        {
            *out = value * scale * weight;
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

fn silu_glu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(gate, up)| silu_scalar(*gate) * *up)
        .collect()
}

fn silu_forward_in_place(values: &mut [f32]) {
    for value in values {
        *value = silu_scalar(*value);
    }
}

fn silu_scalar(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn axpy(destination: &mut [f32], source: &[f32], alpha: f32) {
    for (destination, source) in destination.iter_mut().zip(source.iter().copied()) {
        *destination += source * alpha;
    }
}

fn l2_normalize_into(values: &[f32], epsilon: f32, output: &mut [f32]) {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(epsilon);
    for (out, value) in output.iter_mut().zip(values.iter().copied()) {
        *out = value / norm;
    }
}

fn delta_net_autoregressive_step_in_place(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: f32,
    beta: f32,
    state: &mut [f32],
    norm_q: &mut [f32],
    norm_k: &mut [f32],
    kv_mem: &mut [f32],
    delta: &mut [f32],
    output: &mut [f32],
) {
    let head_dim = q.len();
    l2_normalize_into(q, 1e-6, &mut norm_q[..head_dim]);
    let scale = (head_dim as f32).sqrt().recip();
    for value in &mut norm_q[..head_dim] {
        *value *= scale;
    }
    l2_normalize_into(k, 1e-6, &mut norm_k[..head_dim]);
    let gate = gate.exp();

    for value in state.iter_mut() {
        *value *= gate;
    }
    for row in 0..head_dim {
        let row_slice = &state[row * head_dim..(row + 1) * head_dim];
        kv_mem[row] = dot(row_slice, &norm_k[..head_dim]);
    }
    for row in 0..head_dim {
        delta[row] = (v[row] - kv_mem[row]) * beta;
    }
    for row in 0..head_dim {
        let row_delta = delta[row];
        for column in 0..head_dim {
            state[row * head_dim + column] += row_delta * norm_k[column];
        }
    }
    for row in 0..head_dim {
        let row_slice = &state[row * head_dim..(row + 1) * head_dim];
        output[row] = dot(row_slice, &norm_q[..head_dim]);
    }
}

fn causal_depthwise_conv1d_step_in_place(
    input: &[f32],
    state: &mut [f32],
    weights: &DenseMatrix,
    kernel_size: usize,
    output: &mut [f32],
) -> Result<(), ReferenceTextGenerationError> {
    if weights.columns != kernel_size || weights.rows != input.len() {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "qwen35 conv1d shape mismatch: weights=[{}, {}] input={}",
                weights.rows,
                weights.columns,
                input.len()
            )),
        ));
    }
    let state_tokens = kernel_size.saturating_sub(1);
    if state.len() != input.len().saturating_mul(state_tokens) {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "qwen35 conv1d state length mismatch: expected {}, actual {}",
                input.len().saturating_mul(state_tokens),
                state.len()
            )),
        ));
    }
    if output.len() != input.len() {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "qwen35 conv1d output length mismatch: expected {}, actual {}",
                input.len(),
                output.len()
            )),
        ));
    }
    for row in 0..input.len() {
        let row_state = &state[row * state_tokens..(row + 1) * state_tokens];
        let row_weights = &weights.values[row * kernel_size..(row + 1) * kernel_size];
        output[row] =
            dot(row_state, &row_weights[..state_tokens]) + input[row] * row_weights[state_tokens];
    }
    if state_tokens > 0 {
        for row in 0..input.len() {
            let row_state = &mut state[row * state_tokens..(row + 1) * state_tokens];
            row_state.rotate_left(1);
            row_state[state_tokens - 1] = input[row];
        }
    }
    Ok(())
}

fn attend_full_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    cache: &[Qwen35FullAttentionEntry],
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
) -> Vec<f32> {
    let group_size = head_count / kv_head_count.max(1);
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0_f32; head_count.saturating_mul(head_dim)];
    for head_index in 0..head_count {
        let kv_head_index = (head_index / group_size.max(1)).min(kv_head_count.saturating_sub(1));
        let q = &query[head_index * head_dim..(head_index + 1) * head_dim];
        let current_key = &key[kv_head_index * head_dim..(kv_head_index + 1) * head_dim];
        let current_value = &value[kv_head_index * head_dim..(kv_head_index + 1) * head_dim];
        let mut logits = Vec::with_capacity(cache.len().saturating_add(1));
        for entry in cache {
            let cached_key = &entry.key[kv_head_index * head_dim..(kv_head_index + 1) * head_dim];
            logits.push(dot(q, cached_key) * scale);
        }
        logits.push(dot(q, current_key) * scale);
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = logits
            .iter()
            .map(|logit| (logit - max_logit).exp())
            .collect::<Vec<_>>();
        let denom = weights.iter().copied().sum::<f32>().max(f32::MIN_POSITIVE);
        for weight in &mut weights {
            *weight /= denom;
        }
        let destination = &mut output[head_index * head_dim..(head_index + 1) * head_dim];
        for (entry_index, entry) in cache.iter().enumerate() {
            let cached_value =
                &entry.value[kv_head_index * head_dim..(kv_head_index + 1) * head_dim];
            axpy(destination, cached_value, weights[entry_index]);
        }
        axpy(destination, current_value, *weights.last().unwrap_or(&0.0));
    }
    output
}

fn apply_rope_neox(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    metadata: &GgufDecoderFamilyMetadata,
) {
    let rotary_dim = rotary_dim.min(head_dim).max(2);
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
        .map(|original| rope_yarn_corr_dims(rotary_dim, original, metadata.rope_theta))
        .unwrap_or([0.0, rotary_dim as f32 - 1.0]);
    let theta_scale = metadata.rope_theta.powf(-2.0 / rotary_dim as f32);
    for head_index in 0..head_count {
        let head_base = head_index.saturating_mul(head_dim);
        for i0 in (0..rotary_dim).step_by(2) {
            let pair = i0 / 2;
            let index0 = head_base + pair;
            let index1 = head_base + pair + rotary_dim / 2;
            if index1 >= head_base + head_dim || index1 >= values.len() {
                continue;
            }
            let theta_base = position as f32 * theta_scale.powf(pair as f32);
            let (cos_theta, sin_theta) =
                rope_yarn(theta_base, freq_scale, corr_dims, i0, ext_factor, 1.0);
            let x0 = values[index0];
            let x1 = values[index1];
            values[index0] = x0 * cos_theta - x1 * sin_theta;
            values[index1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
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
