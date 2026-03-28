use std::{path::Path, sync::Arc, time::Instant};

use psionic_backend_cpu::{decode_quantized_row_into, quantized_row_byte_len};
use psionic_backend_cuda::{
    CudaBackend, CudaBuffer, CudaGraphExec, CudaHostBuffer, CudaQuantizedMatvecStats,
    CudaSubmission, CudaTopKResult, ggml_q8_1_storage_bytes,
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
    ContinuousBatchGenerationResult, GenerationEventStream, GenerationInput, GenerationMetrics,
    GenerationOptions, GenerationProvenance, GenerationRequest, GenerationResponse,
    GenerationStreamChunk, GenerationStreamEvent, GenerationStreamStatus, GenerationStreamTerminal,
    GenerationStreamingPolicy, LoadedModelView, LoadedModelsObservation, LocalRuntimeDiagnostic,
    ManagedTextGenerationRuntime, Qwen35CudaDecodeOutputMetrics, Qwen35CudaDecodeOutputMode,
    ReferenceTextGenerationError, StreamingTextGenerationExecutor, TerminationReason,
    TextGenerationExecutor, current_time_millis, default_generation_streaming_policy,
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
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::UnsupportedStep(String::from(
                    "structured-output fallback is not implemented on the native qwen35 cuda runtime",
                )),
            ));
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
        let output_mode = qwen35_cuda_output_mode(&request.options);
        let mut last_logits = Vec::new();
        let mut pending_greedy_token = None;
        let mut last_candidates = None;
        let mut decode_output_metrics = Qwen35CudaDecodeOutputMetrics::default();

        if matches!(
            output_mode,
            CudaStepOutputMode::ArgmaxOnly | CudaStepOutputMode::TopKCandidates(_)
        ) {
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
                output_mode,
            )?;
            pending_greedy_token = step.selected_token;
            last_candidates = step.candidates;
            kernel_count = kernel_count.saturating_add(step.kernel_count);
            bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
            accumulate_qwen35_decode_output_metrics(
                &mut decode_output_metrics,
                step.output_metrics.as_ref(),
            );
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
                accumulate_qwen35_decode_output_metrics(
                    &mut decode_output_metrics,
                    step.output_metrics.as_ref(),
                );
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

            let next_token = if matches!(output_mode, CudaStepOutputMode::ArgmaxOnly) {
                let step = if let Some(selected) = pending_greedy_token.take() {
                    Qwen35ForwardStep {
                        selected_token: Some(selected),
                        logits: Vec::new(),
                        candidates: None,
                        kernel_count: 0,
                        bytes_moved: 0,
                        output_metrics: None,
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
                accumulate_qwen35_decode_output_metrics(
                    &mut decode_output_metrics,
                    step.output_metrics.as_ref(),
                );
                step.selected_token.ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                        String::from("qwen35 argmax decode did not return a selected token"),
                    ))
                })?
            } else if let CudaStepOutputMode::TopKCandidates(top_k) = output_mode {
                if !generated_tokens.is_empty() {
                    let step = self.model.forward_token(
                        &mut self.backend,
                        &mut self.step_plan,
                        &mut state,
                        *generated_tokens
                            .last()
                            .expect("generated token should exist"),
                        CudaStepOutputMode::TopKCandidates(top_k),
                    )?;
                    last_candidates = step.candidates;
                    kernel_count = kernel_count.saturating_add(step.kernel_count);
                    bytes_moved = bytes_moved.saturating_add(step.bytes_moved);
                    accumulate_qwen35_decode_output_metrics(
                        &mut decode_output_metrics,
                        step.output_metrics.as_ref(),
                    );
                }
                let candidates = last_candidates.as_ref().ok_or_else(|| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                        String::from("qwen35 bounded candidate decode did not return candidates"),
                    ))
                })?;
                match sampler.select_next_token_from_candidates(
                    candidates.indices.as_slice(),
                    candidates.values.as_slice(),
                    self.model.descriptor.config.vocab_size,
                )? {
                    crate::GenerationSelection::Token(token) => token,
                    crate::GenerationSelection::Terminate => {
                        break TerminationReason::EndOfSequence;
                    }
                }
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
                    accumulate_qwen35_decode_output_metrics(
                        &mut decode_output_metrics,
                        step.output_metrics.as_ref(),
                    );
                }
                match sampler.select_next_token(
                    &self.model.tokenizer,
                    &last_logits,
                    &crate::InMemoryKvCache::new(1, 1),
                    generated_tokens.as_slice(),
                )? {
                    crate::GenerationSelection::Token(token) => token,
                    crate::GenerationSelection::Terminate => {
                        break TerminationReason::EndOfSequence;
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
            qwen35_cuda_decode: (!decode_output_metrics.is_zero()).then_some(decode_output_metrics),
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

const QWEN35_CUDA_MAX_TOP_K: usize = 128;
const QWEN35_CUDA_PARTITIONED_TOP_K_THRESHOLD: usize = 40;
const QWEN35_CUDA_PARTITIONED_TOP_K_BLOCKS: usize = 8;
fn qwen35_fast_greedy_path_enabled() -> bool {
    std::env::var_os("PSIONIC_QWEN35_DISABLE_FAST_GREEDY").is_none()
}

fn qwen35_cuda_output_mode(options: &GenerationOptions) -> CudaStepOutputMode {
    let policy = options.sampling_policy();
    let mirostat = policy.effective_mirostat();
    if options.structured_output.is_none()
        && qwen35_fast_greedy_path_enabled()
        && !qwen35_sampling_penalties_active(options)
        && mirostat.is_none()
        && (matches!(options.decode_strategy, crate::DecodeStrategy::Greedy)
            || policy.effective_temperature() <= 1e-6
            || policy.effective_top_k() == Some(1))
    {
        return CudaStepOutputMode::ArgmaxOnly;
    }
    if options.structured_output.is_none()
        && matches!(options.decode_strategy, crate::DecodeStrategy::Sample)
        && !qwen35_sampling_penalties_active(options)
        && policy.effective_temperature() > 1e-6
        && mirostat.is_none()
        && policy.effective_temperature() > 1e-6
        && let Some(top_k) = policy.effective_top_k()
        && top_k > 1
        && top_k <= QWEN35_CUDA_MAX_TOP_K
    {
        return CudaStepOutputMode::TopKCandidates(top_k);
    }
    CudaStepOutputMode::FullLogits
}

fn qwen35_sampling_penalties_active(options: &GenerationOptions) -> bool {
    let policy = options.sampling_policy();
    (policy.effective_repeat_penalty() - 1.0).abs() > f32::EPSILON
        || policy.effective_presence_penalty().abs() > f32::EPSILON
        || policy.effective_frequency_penalty().abs() > f32::EPSILON
}

fn qwen35_decode_output_metrics(
    output_mode: Qwen35CudaDecodeOutputMode,
    readback_bytes: u64,
    raw_logits_materialized: bool,
) -> Qwen35CudaDecodeOutputMetrics {
    Qwen35CudaDecodeOutputMetrics {
        step_count: 1,
        output_modes: vec![output_mode],
        readback_bytes,
        raw_logits_materialized,
    }
}

fn accumulate_qwen35_decode_output_metrics(
    total: &mut Qwen35CudaDecodeOutputMetrics,
    step_metrics: Option<&Qwen35CudaDecodeOutputMetrics>,
) {
    if let Some(step_metrics) = step_metrics {
        total.accumulate(step_metrics);
    }
}

fn can_use_q8_1_quantized_matvec(mode: QuantizationMode) -> bool {
    matches!(
        mode,
        QuantizationMode::GgmlQ8_0
            | QuantizationMode::GgmlQ4K
            | QuantizationMode::GgmlQ6K
            | QuantizationMode::GgmlMxfp4
    )
}

fn can_use_q8_1_argmax(mode: QuantizationMode) -> bool {
    matches!(
        mode,
        QuantizationMode::GgmlQ8_0 | QuantizationMode::GgmlQ4K | QuantizationMode::GgmlMxfp4
    )
}

fn can_use_cuda_quantized_matvec(mode: QuantizationMode) -> bool {
    matches!(
        mode,
        QuantizationMode::GgmlQ8_0
            | QuantizationMode::GgmlMxfp4
            | QuantizationMode::GgmlQ4K
            | QuantizationMode::GgmlQ6K
    )
}

fn qwen35_requires_dense_f16_mirror(mode: QuantizationMode) -> bool {
    !can_use_cuda_quantized_matvec(mode)
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

fn cuda_top_k_result_from_host_buffers(
    indices_host_buffer: &CudaHostBuffer,
    values_host_buffer: &CudaHostBuffer,
    row_count: usize,
    top_k: usize,
) -> Result<CudaTopKResult, crate::RuntimeError> {
    let expected_indices_bytes = row_count
        .saturating_mul(top_k)
        .saturating_mul(std::mem::size_of::<i32>());
    let expected_values_bytes = row_count
        .saturating_mul(top_k)
        .saturating_mul(std::mem::size_of::<f32>());
    if indices_host_buffer.byte_len() < expected_indices_bytes
        || values_host_buffer.byte_len() < expected_values_bytes
    {
        return Err(crate::RuntimeError::Backend(format!(
            "qwen35 cuda top-k host buffers are too small: need {} index bytes and {} value bytes, have {} and {}",
            expected_indices_bytes,
            expected_values_bytes,
            indices_host_buffer.byte_len(),
            values_host_buffer.byte_len()
        )));
    }

    let index_bytes = indices_host_buffer.read_bytes().map_err(|error| {
        crate::RuntimeError::Backend(format!(
            "failed to read qwen35 cuda top-k index host buffer: {error}",
        ))
    })?;
    let value_bytes = values_host_buffer.read_bytes().map_err(|error| {
        crate::RuntimeError::Backend(format!(
            "failed to read qwen35 cuda top-k value host buffer: {error}",
        ))
    })?;

    let mut indices = Vec::with_capacity(row_count.saturating_mul(top_k));
    for chunk in index_bytes[..expected_indices_bytes].chunks_exact(std::mem::size_of::<i32>()) {
        let index = i32::from_ne_bytes(chunk.try_into().map_err(|_| {
            crate::RuntimeError::Backend(String::from(
                "qwen35 cuda top-k returned invalid index bytes",
            ))
        })?);
        indices.push(u32::try_from(index).map_err(|_| {
            crate::RuntimeError::Backend(format!(
                "qwen35 cuda top-k returned a negative token index {index}",
            ))
        })?);
    }

    let mut values = Vec::with_capacity(row_count.saturating_mul(top_k));
    for chunk in value_bytes[..expected_values_bytes].chunks_exact(std::mem::size_of::<f32>()) {
        values.push(f32::from_ne_bytes(chunk.try_into().map_err(|_| {
            crate::RuntimeError::Backend(String::from(
                "qwen35 cuda top-k returned invalid value bytes",
            ))
        })?));
    }

    Ok(CudaTopKResult {
        row_count,
        top_k,
        indices,
        values,
    })
}

fn cuda_f32_vec_from_host_buffer(
    host_buffer: &CudaHostBuffer,
    element_count: usize,
) -> Result<Vec<f32>, crate::RuntimeError> {
    let expected_bytes = element_count.saturating_mul(std::mem::size_of::<f32>());
    if host_buffer.byte_len() < expected_bytes {
        return Err(crate::RuntimeError::Backend(format!(
            "qwen35 cuda logits host buffer is too small: need {} bytes, have {}",
            expected_bytes,
            host_buffer.byte_len()
        )));
    }
    let bytes = host_buffer.read_bytes().map_err(|error| {
        crate::RuntimeError::Backend(format!(
            "failed to read qwen35 cuda logits host buffer: {error}",
        ))
    })?;
    let mut values = Vec::with_capacity(element_count);
    for chunk in bytes[..expected_bytes].chunks_exact(std::mem::size_of::<f32>()) {
        values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
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
    token_embedding_f16: Option<CudaBuffer>,
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
        let token_embedding_name = adapter.tensor_layout().token_embedding.as_str();
        let token_embedding = HostMatrix::load(&artifact, token_embedding_name)?;
        let token_embedding_f16 = None;
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
            .saturating_add(
                token_embedding_f16
                    .as_ref()
                    .map(CudaBuffer::byte_len)
                    .unwrap_or(0),
            )
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
            token_embedding_f16,
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

    fn encode_token_embedding_lookup(
        &self,
        submission: &mut CudaSubmission,
        plan: &mut Qwen35CudaStepPlan,
        token: TokenId,
        position: usize,
    ) -> Result<u64, ReferenceTextGenerationError> {
        let Some(token_embedding_f16) = self.token_embedding_f16.as_ref() else {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(String::from(
                    "qwen35 cuda token embedding lookup requires a device f16 mirror",
                )),
            ));
        };
        let decode_params = [
            0_i32,
            i32::try_from(position).map_err(|_| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                    "qwen35 decode position {position} exceeds i32 decode parameter limits",
                )))
            })?,
            i32::try_from(token.as_u32()).map_err(|_| {
                ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                    "qwen35 token {} exceeds i32 decode parameter limits",
                    token.as_u32(),
                )))
            })?,
        ];
        plan.decode_params_host_buffer
            .write_i32(&decode_params)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        submission
            .copy_host_to_device(&plan.decode_params_host_buffer, &plan.decode_params_buffer)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        submission
            .gather_f16_row_to_f32(
                token_embedding_f16,
                self.token_embedding.rows(),
                self.token_embedding.columns(),
                &plan.decode_params_buffer,
                &plan.current_hidden_buffer,
            )
            .map_err(ReferenceTextGenerationError::Runtime)?;
        Ok((decode_params.len() * std::mem::size_of::<i32>())
            .try_into()
            .unwrap_or(u64::MAX))
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
        if std::env::var_os("PSIONIC_QWEN35_DEBUG_ATTENTION").is_none() {
            return self.forward_token_fused(backend, plan, state, token, output_mode);
        }
        let mut bytes_moved = 0u64;
        let mut kernel_count = 0usize;
        let position = state.position;
        if self.token_embedding_f16.is_none() {
            let hidden = self.token_embedding.decode_row(token.as_u32() as usize)?;
            plan.current_hidden_buffer
                .write_f32_at_offset(0, hidden.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            bytes_moved = bytes_moved.saturating_add(
                vec_f32_bytes(hidden.as_slice())
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            kernel_count = kernel_count.saturating_add(1);
        }
        for (layer_index, (layer, layer_state)) in
            self.layers.iter().zip(state.layers.iter_mut()).enumerate()
        {
            let initial_token = (layer_index == 0)
                .then_some(token)
                .filter(|_| self.token_embedding_f16.is_some());
            match (&layer.kind, layer_state) {
                (Qwen35LayerKind::Hybrid(_), Qwen35LayerState::Hybrid(hybrid_state)) => {
                    layer.forward_hybrid_device(
                        backend,
                        plan,
                        self,
                        layer_index,
                        hybrid_state,
                        position,
                        initial_token,
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
                        layer_index,
                        full_attention,
                        full_attention_state,
                        initial_token,
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
        let (logits, selected_token, candidates, output_stats, output_metrics) = match output_mode {
            CudaStepOutputMode::NoOutput => {
                (Vec::new(), None, None, zero_cuda_matvec_stats(), None)
            }
            CudaStepOutputMode::FullLogits => {
                let (logits, stats) = plan
                    .run_output_logits_from_device(
                        backend,
                        &current_hidden_buffer,
                        &output_norm_device,
                        self.family_metadata.rms_norm_epsilon,
                        self.output.transposed_f16.as_ref(),
                        &self.output.storage,
                        self.output.host.mode,
                        self.output.host.rows,
                        self.output.host.columns,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                let readback_bytes = self
                    .output
                    .host
                    .rows
                    .saturating_mul(std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap_or(u64::MAX);
                (
                    logits,
                    None,
                    None,
                    stats,
                    Some(qwen35_decode_output_metrics(
                        Qwen35CudaDecodeOutputMode::RawLogits,
                        readback_bytes,
                        true,
                    )),
                )
            }
            CudaStepOutputMode::ArgmaxOnly => {
                let (selected, stats) = plan
                    .run_output_argmax_from_device(
                        backend,
                        &current_hidden_buffer,
                        &output_norm_device,
                        self.family_metadata.rms_norm_epsilon,
                        self.output.transposed_f16.as_ref(),
                        &self.output.storage,
                        self.output.host.mode,
                        self.output.host.rows,
                        self.output.host.columns,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                let readback_bytes = stats.device_to_host_bytes;
                (
                    Vec::new(),
                    Some(selected),
                    None,
                    stats,
                    Some(qwen35_decode_output_metrics(
                        Qwen35CudaDecodeOutputMode::ArgmaxOnly,
                        readback_bytes,
                        false,
                    )),
                )
            }
            CudaStepOutputMode::TopKCandidates(top_k) => {
                let (candidates, stats) = plan
                    .run_output_top_k_from_device(
                        backend,
                        &current_hidden_buffer,
                        &output_norm_device,
                        self.family_metadata.rms_norm_epsilon,
                        self.output.transposed_f16.as_ref(),
                        &self.output.storage,
                        self.output.host.mode,
                        self.output.host.rows,
                        self.output.host.columns,
                        top_k,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                let readback_bytes = stats.device_to_host_bytes;
                (
                    Vec::new(),
                    None,
                    Some(candidates),
                    stats,
                    Some(qwen35_decode_output_metrics(
                        Qwen35CudaDecodeOutputMode::TopKCandidates { top_k },
                        readback_bytes,
                        false,
                    )),
                )
            }
        };
        bytes_moved = bytes_moved.saturating_add(cuda_stats_bytes(output_stats));
        kernel_count = kernel_count.saturating_add(output_stats.kernel_launches);
        state.position = state.position.saturating_add(1);
        Ok(Qwen35ForwardStep {
            logits,
            selected_token,
            candidates,
            kernel_count,
            bytes_moved,
            output_metrics,
        })
    }

    fn forward_token_fused(
        &self,
        backend: &mut CudaBackend,
        plan: &mut Qwen35CudaStepPlan,
        state: &mut Qwen35State,
        token: TokenId,
        output_mode: CudaStepOutputMode,
    ) -> Result<Qwen35ForwardStep, ReferenceTextGenerationError> {
        let mut bytes_moved = 0u64;
        let mut kernel_count = 0usize;
        let position = state.position;
        if self.token_embedding_f16.is_none() {
            let hidden = self.token_embedding.decode_row(token.as_u32() as usize)?;
            plan.current_hidden_buffer
                .write_f32_at_offset(0, hidden.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            bytes_moved = bytes_moved.saturating_add(
                vec_f32_bytes(hidden.as_slice())
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            kernel_count = kernel_count.saturating_add(1);
        }

        if output_mode == CudaStepOutputMode::NoOutput && self.token_embedding_f16.is_none() {
            let decode_params = [
                i32::try_from(position).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 past-token count {position} exceeds i32 decode parameter limits",
                    )))
                })?,
                i32::try_from(position).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 decode position {position} exceeds i32 decode parameter limits",
                    )))
                })?,
                i32::try_from(token.as_u32()).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 token {} exceeds i32 decode parameter limits",
                        token.as_u32(),
                    )))
                })?,
            ];
            plan.decode_params_host_buffer
                .write_i32(&decode_params)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let decode_params_bytes = (decode_params.len() * std::mem::size_of::<i32>())
                .try_into()
                .unwrap_or(u64::MAX);
            bytes_moved = bytes_moved.saturating_add(decode_params_bytes);
            let no_output_graph_cache_identity = qwen35_decode_graph_cache_identity(state);
            let mut reused_graph_exec = false;
            let report = if plan.no_output_graph_cache_identity.as_ref()
                == Some(&no_output_graph_cache_identity)
            {
                if let Some(graph_exec) = plan.no_output_graph_exec.as_ref() {
                    reused_graph_exec = true;
                    graph_exec
                        .launch(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?
                } else {
                    let mut submission = backend
                        .begin_captured_submission()
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .copy_host_to_device(
                            &plan.decode_params_host_buffer,
                            &plan.decode_params_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
                        match (&layer.kind, layer_state) {
                            (
                                Qwen35LayerKind::Hybrid(_),
                                Qwen35LayerState::Hybrid(hybrid_state),
                            ) => {
                                layer.encode_hybrid_device_submission(
                                    backend,
                                    &mut submission,
                                    plan,
                                    self,
                                    hybrid_state,
                                    position,
                                    None,
                                    &mut bytes_moved,
                                )?;
                            }
                            (
                                Qwen35LayerKind::FullAttention(full_attention),
                                Qwen35LayerState::FullAttention(full_attention_state),
                            ) => {
                                layer.encode_full_attention_device_submission(
                                    backend,
                                    &mut submission,
                                    plan,
                                    self,
                                    full_attention,
                                    full_attention_state,
                                    position,
                                    None,
                                    true,
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
                    let (report, graph_exec) = submission
                        .commit_captured(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    plan.no_output_graph_exec = Some(graph_exec);
                    plan.no_output_graph_cache_identity = Some(no_output_graph_cache_identity);
                    report
                }
            } else {
                plan.no_output_graph_exec = None;
                plan.no_output_graph_cache_identity = None;
                let mut submission = backend
                    .begin_captured_submission()
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                submission
                    .copy_host_to_device(
                        &plan.decode_params_host_buffer,
                        &plan.decode_params_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
                    match (&layer.kind, layer_state) {
                        (Qwen35LayerKind::Hybrid(_), Qwen35LayerState::Hybrid(hybrid_state)) => {
                            layer.encode_hybrid_device_submission(
                                backend,
                                &mut submission,
                                plan,
                                self,
                                hybrid_state,
                                position,
                                None,
                                &mut bytes_moved,
                            )?;
                        }
                        (
                            Qwen35LayerKind::FullAttention(full_attention),
                            Qwen35LayerState::FullAttention(full_attention_state),
                        ) => {
                            layer.encode_full_attention_device_submission(
                                backend,
                                &mut submission,
                                plan,
                                self,
                                full_attention,
                                full_attention_state,
                                position,
                                None,
                                true,
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
                let (report, graph_exec) = submission
                    .commit_captured(psionic_backend_cuda::CudaCommandWait::Completed)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                plan.no_output_graph_exec = Some(graph_exec);
                plan.no_output_graph_cache_identity = Some(no_output_graph_cache_identity);
                report
            };
            kernel_count = kernel_count.saturating_add(report.encoded_operations);
            if reused_graph_exec {
                for layer_state in &mut state.layers {
                    if let Qwen35LayerState::FullAttention(full_attention) = layer_state {
                        full_attention.len = full_attention.len.saturating_add(1);
                    }
                }
            }
            state.position = state.position.saturating_add(1);
            return Ok(Qwen35ForwardStep {
                logits: Vec::new(),
                selected_token: None,
                candidates: None,
                kernel_count,
                bytes_moved,
                output_metrics: None,
            });
        }

        let output_uses_q8_1_argmax =
            self.output.transposed_f16.is_none() && can_use_q8_1_argmax(self.output.host.mode);
        let output_uses_q8_1_matvec = self.output.transposed_f16.is_none()
            && can_use_q8_1_quantized_matvec(self.output.host.mode);
        let output_uses_f16_argmax = self.output.transposed_f16.is_some();

        if output_mode == CudaStepOutputMode::FullLogits && self.token_embedding_f16.is_none() {
            let decode_params = [
                i32::try_from(position).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 past-token count {position} exceeds i32 decode parameter limits",
                    )))
                })?,
                i32::try_from(position).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 decode position {position} exceeds i32 decode parameter limits",
                    )))
                })?,
                i32::try_from(token.as_u32()).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 token {} exceeds i32 decode parameter limits",
                        token.as_u32(),
                    )))
                })?,
            ];
            plan.decode_params_host_buffer
                .write_i32(&decode_params)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let decode_params_bytes = (decode_params.len() * std::mem::size_of::<i32>())
                .try_into()
                .unwrap_or(u64::MAX);
            let logits_bytes = self
                .output
                .host
                .rows
                .saturating_mul(std::mem::size_of::<f32>())
                .try_into()
                .unwrap_or(u64::MAX);
            bytes_moved = bytes_moved
                .saturating_add(decode_params_bytes)
                .saturating_add(logits_bytes);
            let full_logits_graph_cache_identity = qwen35_decode_graph_cache_identity(state);
            let mut reused_graph_exec = false;
            let report = if plan.full_logits_graph_cache_identity.as_ref()
                == Some(&full_logits_graph_cache_identity)
            {
                if let Some(graph_exec) = plan.full_logits_graph_exec.as_ref() {
                    reused_graph_exec = true;
                    graph_exec
                        .launch(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?
                } else {
                    let mut submission = backend
                        .begin_captured_submission()
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .copy_host_to_device(
                            &plan.decode_params_host_buffer,
                            &plan.decode_params_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
                        match (&layer.kind, layer_state) {
                            (
                                Qwen35LayerKind::Hybrid(_),
                                Qwen35LayerState::Hybrid(hybrid_state),
                            ) => {
                                layer.encode_hybrid_device_submission(
                                    backend,
                                    &mut submission,
                                    plan,
                                    self,
                                    hybrid_state,
                                    position,
                                    None,
                                    &mut bytes_moved,
                                )?;
                            }
                            (
                                Qwen35LayerKind::FullAttention(full_attention),
                                Qwen35LayerState::FullAttention(full_attention_state),
                            ) => {
                                layer.encode_full_attention_device_submission(
                                    backend,
                                    &mut submission,
                                    plan,
                                    self,
                                    full_attention,
                                    full_attention_state,
                                    position,
                                    None,
                                    true,
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
                    if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                        submission
                            .rms_norm(
                                &plan.current_hidden_buffer,
                                &self.output_norm_device,
                                &plan.matvec_input_buffer,
                                self.output.host.columns,
                                self.family_metadata.rms_norm_epsilon,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .cast_f32_to_f16(
                                &plan.matvec_input_buffer,
                                &plan.vector_f16_buffer,
                                self.output.host.columns,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .matmul_f16_to_f32(
                                &plan.vector_f16_buffer,
                                transposed_f16,
                                &plan.logits_buffer,
                                1,
                                self.output.host.columns,
                                self.output.host.rows,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                    } else if output_uses_q8_1_matvec {
                        submission
                            .rms_norm_q8_1(
                                &plan.current_hidden_buffer,
                                &self.output_norm_device,
                                &plan.matvec_input_q8_1_buffer,
                                self.output.host.columns,
                                self.family_metadata.rms_norm_epsilon,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission.quantized_matvec_q8_1(
                            &self.output.storage,
                            0,
                            self.output.host.mode,
                            self.output.host.rows,
                            self.output.host.columns,
                            &plan.matvec_input_q8_1_buffer,
                            None,
                            &plan.logits_buffer,
                        )?;
                    } else {
                        submission
                            .rms_norm(
                                &plan.current_hidden_buffer,
                                &self.output_norm_device,
                                &plan.matvec_input_buffer,
                                self.output.host.columns,
                                self.family_metadata.rms_norm_epsilon,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission.quantized_matvec(
                            &self.output.storage,
                            0,
                            self.output.host.mode,
                            self.output.host.rows,
                            self.output.host.columns,
                            &plan.matvec_input_buffer,
                            &plan.logits_buffer,
                        )?;
                    }
                    submission
                        .copy_device_to_host(&plan.logits_buffer, &plan.logits_host_buffer)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    let (report, graph_exec) = submission
                        .commit_captured(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    plan.full_logits_graph_exec = Some(graph_exec);
                    plan.full_logits_graph_cache_identity = Some(full_logits_graph_cache_identity);
                    report
                }
            } else {
                plan.full_logits_graph_exec = None;
                plan.full_logits_graph_cache_identity = None;
                let mut submission = backend
                    .begin_captured_submission()
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                submission
                    .copy_host_to_device(
                        &plan.decode_params_host_buffer,
                        &plan.decode_params_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
                    match (&layer.kind, layer_state) {
                        (Qwen35LayerKind::Hybrid(_), Qwen35LayerState::Hybrid(hybrid_state)) => {
                            layer.encode_hybrid_device_submission(
                                backend,
                                &mut submission,
                                plan,
                                self,
                                hybrid_state,
                                position,
                                None,
                                &mut bytes_moved,
                            )?;
                        }
                        (
                            Qwen35LayerKind::FullAttention(full_attention),
                            Qwen35LayerState::FullAttention(full_attention_state),
                        ) => {
                            layer.encode_full_attention_device_submission(
                                backend,
                                &mut submission,
                                plan,
                                self,
                                full_attention,
                                full_attention_state,
                                position,
                                None,
                                true,
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
                if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                    submission
                        .rms_norm(
                            &plan.current_hidden_buffer,
                            &self.output_norm_device,
                            &plan.matvec_input_buffer,
                            self.output.host.columns,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .cast_f32_to_f16(
                            &plan.matvec_input_buffer,
                            &plan.vector_f16_buffer,
                            self.output.host.columns,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .matmul_f16_to_f32(
                            &plan.vector_f16_buffer,
                            transposed_f16,
                            &plan.logits_buffer,
                            1,
                            self.output.host.columns,
                            self.output.host.rows,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                } else if output_uses_q8_1_matvec {
                    submission
                        .rms_norm_q8_1(
                            &plan.current_hidden_buffer,
                            &self.output_norm_device,
                            &plan.matvec_input_q8_1_buffer,
                            self.output.host.columns,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission.quantized_matvec_q8_1(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        self.output.host.rows,
                        self.output.host.columns,
                        &plan.matvec_input_q8_1_buffer,
                        None,
                        &plan.logits_buffer,
                    )?;
                } else {
                    submission
                        .rms_norm(
                            &plan.current_hidden_buffer,
                            &self.output_norm_device,
                            &plan.matvec_input_buffer,
                            self.output.host.columns,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission.quantized_matvec(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        self.output.host.rows,
                        self.output.host.columns,
                        &plan.matvec_input_buffer,
                        &plan.logits_buffer,
                    )?;
                }
                submission
                    .copy_device_to_host(&plan.logits_buffer, &plan.logits_host_buffer)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                let (report, graph_exec) = submission
                    .commit_captured(psionic_backend_cuda::CudaCommandWait::Completed)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                plan.full_logits_graph_exec = Some(graph_exec);
                plan.full_logits_graph_cache_identity = Some(full_logits_graph_cache_identity);
                report
            };
            kernel_count = kernel_count.saturating_add(report.encoded_operations);
            if reused_graph_exec {
                for layer_state in &mut state.layers {
                    if let Qwen35LayerState::FullAttention(full_attention) = layer_state {
                        full_attention.len = full_attention.len.saturating_add(1);
                    }
                }
            }
            state.position = state.position.saturating_add(1);
            let logits =
                cuda_f32_vec_from_host_buffer(&plan.logits_host_buffer, self.output.host.rows)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            return Ok(Qwen35ForwardStep {
                logits,
                selected_token: None,
                candidates: None,
                kernel_count,
                bytes_moved,
                output_metrics: Some(qwen35_decode_output_metrics(
                    Qwen35CudaDecodeOutputMode::RawLogits,
                    logits_bytes,
                    true,
                )),
            });
        }

        if let CudaStepOutputMode::TopKCandidates(top_k) = output_mode {
            if self.token_embedding_f16.is_none() {
                let decode_params = [
                    i32::try_from(position).map_err(|_| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "qwen35 past-token count {position} exceeds i32 decode parameter limits",
                            ),
                        ))
                    })?,
                    i32::try_from(position).map_err(|_| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "qwen35 decode position {position} exceeds i32 decode parameter limits",
                            ),
                        ))
                    })?,
                    i32::try_from(token.as_u32()).map_err(|_| {
                        ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(
                            format!(
                                "qwen35 token {} exceeds i32 decode parameter limits",
                                token.as_u32(),
                            ),
                        ))
                    })?,
                ];
                plan.decode_params_host_buffer
                    .write_i32(&decode_params)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                let decode_params_bytes = (decode_params.len() * std::mem::size_of::<i32>())
                    .try_into()
                    .unwrap_or(u64::MAX);
                let top_k_bytes = top_k
                    .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap_or(u64::MAX);
                bytes_moved = bytes_moved
                    .saturating_add(decode_params_bytes)
                    .saturating_add(top_k_bytes);
                let top_k_graph_cache_identity = (top_k, qwen35_decode_graph_cache_identity(state));
                let mut reused_graph_exec = false;
                let report = if plan.top_k_graph_cache_identity.as_ref()
                    == Some(&top_k_graph_cache_identity)
                {
                    if let Some(graph_exec) = plan.top_k_graph_exec.as_ref() {
                        reused_graph_exec = true;
                        graph_exec
                            .launch(psionic_backend_cuda::CudaCommandWait::Completed)
                            .map_err(ReferenceTextGenerationError::Runtime)?
                    } else {
                        let mut submission = backend
                            .begin_captured_submission()
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .copy_host_to_device(
                                &plan.decode_params_host_buffer,
                                &plan.decode_params_buffer,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut())
                        {
                            match (&layer.kind, layer_state) {
                                (
                                    Qwen35LayerKind::Hybrid(_),
                                    Qwen35LayerState::Hybrid(hybrid_state),
                                ) => {
                                    layer.encode_hybrid_device_submission(
                                        backend,
                                        &mut submission,
                                        plan,
                                        self,
                                        hybrid_state,
                                        position,
                                        None,
                                        &mut bytes_moved,
                                    )?;
                                }
                                (
                                    Qwen35LayerKind::FullAttention(full_attention),
                                    Qwen35LayerState::FullAttention(full_attention_state),
                                ) => {
                                    layer.encode_full_attention_device_submission(
                                        backend,
                                        &mut submission,
                                        plan,
                                        self,
                                        full_attention,
                                        full_attention_state,
                                        position,
                                        None,
                                        true,
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
                        if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                            submission
                                .rms_norm(
                                    &plan.current_hidden_buffer,
                                    &self.output_norm_device,
                                    &plan.matvec_input_buffer,
                                    self.output.host.columns,
                                    self.family_metadata.rms_norm_epsilon,
                                )
                                .map_err(ReferenceTextGenerationError::Runtime)?;
                            submission
                                .cast_f32_to_f16(
                                    &plan.matvec_input_buffer,
                                    &plan.vector_f16_buffer,
                                    self.output.host.columns,
                                )
                                .map_err(ReferenceTextGenerationError::Runtime)?;
                            submission
                                .matmul_f16_to_f32(
                                    &plan.vector_f16_buffer,
                                    transposed_f16,
                                    &plan.logits_buffer,
                                    1,
                                    self.output.host.columns,
                                    self.output.host.rows,
                                )
                                .map_err(ReferenceTextGenerationError::Runtime)?;
                        } else if output_uses_q8_1_matvec {
                            submission
                                .rms_norm_q8_1(
                                    &plan.current_hidden_buffer,
                                    &self.output_norm_device,
                                    &plan.matvec_input_q8_1_buffer,
                                    self.output.host.columns,
                                    self.family_metadata.rms_norm_epsilon,
                                )
                                .map_err(ReferenceTextGenerationError::Runtime)?;
                            submission.quantized_matvec_q8_1(
                                &self.output.storage,
                                0,
                                self.output.host.mode,
                                self.output.host.rows,
                                self.output.host.columns,
                                &plan.matvec_input_q8_1_buffer,
                                None,
                                &plan.logits_buffer,
                            )?;
                        } else {
                            submission
                                .rms_norm(
                                    &plan.current_hidden_buffer,
                                    &self.output_norm_device,
                                    &plan.matvec_input_buffer,
                                    self.output.host.columns,
                                    self.family_metadata.rms_norm_epsilon,
                                )
                                .map_err(ReferenceTextGenerationError::Runtime)?;
                            submission.quantized_matvec(
                                &self.output.storage,
                                0,
                                self.output.host.mode,
                                self.output.host.rows,
                                self.output.host.columns,
                                &plan.matvec_input_buffer,
                                &plan.logits_buffer,
                            )?;
                        }
                        plan.encode_top_k_from_logits(
                            &mut submission,
                            self.output.host.rows,
                            top_k,
                        )?;
                        submission
                            .copy_device_to_host(
                                &plan.top_k_indices_buffer,
                                &plan.top_k_indices_host_buffer,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .copy_device_to_host(
                                &plan.top_k_values_buffer,
                                &plan.top_k_values_host_buffer,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        let (report, graph_exec) = submission
                            .commit_captured(psionic_backend_cuda::CudaCommandWait::Completed)
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        plan.top_k_graph_exec = Some(graph_exec);
                        plan.top_k_graph_cache_identity = Some(top_k_graph_cache_identity);
                        report
                    }
                } else {
                    plan.top_k_graph_exec = None;
                    plan.top_k_graph_cache_identity = None;
                    let mut submission = backend
                        .begin_captured_submission()
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .copy_host_to_device(
                            &plan.decode_params_host_buffer,
                            &plan.decode_params_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
                        match (&layer.kind, layer_state) {
                            (
                                Qwen35LayerKind::Hybrid(_),
                                Qwen35LayerState::Hybrid(hybrid_state),
                            ) => {
                                layer.encode_hybrid_device_submission(
                                    backend,
                                    &mut submission,
                                    plan,
                                    self,
                                    hybrid_state,
                                    position,
                                    None,
                                    &mut bytes_moved,
                                )?;
                            }
                            (
                                Qwen35LayerKind::FullAttention(full_attention),
                                Qwen35LayerState::FullAttention(full_attention_state),
                            ) => {
                                layer.encode_full_attention_device_submission(
                                    backend,
                                    &mut submission,
                                    plan,
                                    self,
                                    full_attention,
                                    full_attention_state,
                                    position,
                                    None,
                                    true,
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
                    if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                        submission
                            .rms_norm(
                                &plan.current_hidden_buffer,
                                &self.output_norm_device,
                                &plan.matvec_input_buffer,
                                self.output.host.columns,
                                self.family_metadata.rms_norm_epsilon,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .cast_f32_to_f16(
                                &plan.matvec_input_buffer,
                                &plan.vector_f16_buffer,
                                self.output.host.columns,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .matmul_f16_to_f32(
                                &plan.vector_f16_buffer,
                                transposed_f16,
                                &plan.logits_buffer,
                                1,
                                self.output.host.columns,
                                self.output.host.rows,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                    } else if output_uses_q8_1_matvec {
                        submission
                            .rms_norm_q8_1(
                                &plan.current_hidden_buffer,
                                &self.output_norm_device,
                                &plan.matvec_input_q8_1_buffer,
                                self.output.host.columns,
                                self.family_metadata.rms_norm_epsilon,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission.quantized_matvec_q8_1(
                            &self.output.storage,
                            0,
                            self.output.host.mode,
                            self.output.host.rows,
                            self.output.host.columns,
                            &plan.matvec_input_q8_1_buffer,
                            None,
                            &plan.logits_buffer,
                        )?;
                    } else {
                        submission
                            .rms_norm(
                                &plan.current_hidden_buffer,
                                &self.output_norm_device,
                                &plan.matvec_input_buffer,
                                self.output.host.columns,
                                self.family_metadata.rms_norm_epsilon,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission.quantized_matvec(
                            &self.output.storage,
                            0,
                            self.output.host.mode,
                            self.output.host.rows,
                            self.output.host.columns,
                            &plan.matvec_input_buffer,
                            &plan.logits_buffer,
                        )?;
                    }
                    plan.encode_top_k_from_logits(&mut submission, self.output.host.rows, top_k)?;
                    submission
                        .copy_device_to_host(
                            &plan.top_k_indices_buffer,
                            &plan.top_k_indices_host_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .copy_device_to_host(
                            &plan.top_k_values_buffer,
                            &plan.top_k_values_host_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    let (report, graph_exec) = submission
                        .commit_captured(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    plan.top_k_graph_exec = Some(graph_exec);
                    plan.top_k_graph_cache_identity = Some(top_k_graph_cache_identity);
                    report
                };
                kernel_count = kernel_count.saturating_add(report.encoded_operations);
                if reused_graph_exec {
                    for layer_state in &mut state.layers {
                        if let Qwen35LayerState::FullAttention(full_attention) = layer_state {
                            full_attention.len = full_attention.len.saturating_add(1);
                        }
                    }
                }
                state.position = state.position.saturating_add(1);
                let candidates = cuda_top_k_result_from_host_buffers(
                    &plan.top_k_indices_host_buffer,
                    &plan.top_k_values_host_buffer,
                    1,
                    top_k,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?;
                return Ok(Qwen35ForwardStep {
                    logits: Vec::new(),
                    selected_token: None,
                    candidates: Some(candidates),
                    kernel_count,
                    bytes_moved,
                    output_metrics: Some(qwen35_decode_output_metrics(
                        Qwen35CudaDecodeOutputMode::TopKCandidates { top_k },
                        top_k_bytes,
                        false,
                    )),
                });
            }
        }

        if output_mode == CudaStepOutputMode::ArgmaxOnly
            && self.token_embedding_f16.is_none()
            && (output_uses_q8_1_argmax || output_uses_f16_argmax)
        {
            let decode_params = [
                i32::try_from(position).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 past-token count {position} exceeds i32 decode parameter limits",
                    )))
                })?,
                i32::try_from(position).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 decode position {position} exceeds i32 decode parameter limits",
                    )))
                })?,
                i32::try_from(token.as_u32()).map_err(|_| {
                    ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(format!(
                        "qwen35 token {} exceeds i32 decode parameter limits",
                        token.as_u32(),
                    )))
                })?,
            ];
            plan.decode_params_host_buffer
                .write_i32(&decode_params)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            plan.argmax_state_host_buffer
                .write_bytes(initial_cuda_argmax_pair_bytes().as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let decode_params_bytes = (decode_params.len() * std::mem::size_of::<i32>())
                .try_into()
                .unwrap_or(u64::MAX);
            if output_uses_q8_1_argmax {
                let argmax_bytes = std::mem::size_of::<u64>().try_into().unwrap_or(u64::MAX);
                bytes_moved = bytes_moved
                    .saturating_add(decode_params_bytes)
                    .saturating_add(argmax_bytes)
                    .saturating_add(argmax_bytes);
            } else {
                bytes_moved = bytes_moved
                    .saturating_add(decode_params_bytes)
                    .saturating_add(std::mem::size_of::<i32>().try_into().unwrap_or(u64::MAX));
            }
            let decode_graph_cache_identity = qwen35_decode_graph_cache_identity(state);
            let mut reused_graph_exec = false;
            let report = if plan.decode_graph_cache_identity.as_ref()
                == Some(&decode_graph_cache_identity)
            {
                if let Some(graph_exec) = plan.decode_graph_exec.as_ref() {
                    reused_graph_exec = true;
                    graph_exec
                        .launch(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?
                } else {
                    let mut submission = backend
                        .begin_captured_submission()
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .copy_host_to_device(
                            &plan.decode_params_host_buffer,
                            &plan.decode_params_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
                        match (&layer.kind, layer_state) {
                            (
                                Qwen35LayerKind::Hybrid(_),
                                Qwen35LayerState::Hybrid(hybrid_state),
                            ) => {
                                layer.encode_hybrid_device_submission(
                                    backend,
                                    &mut submission,
                                    plan,
                                    self,
                                    hybrid_state,
                                    position,
                                    None,
                                    &mut bytes_moved,
                                )?;
                            }
                            (
                                Qwen35LayerKind::FullAttention(full_attention),
                                Qwen35LayerState::FullAttention(full_attention_state),
                            ) => {
                                layer.encode_full_attention_device_submission(
                                    backend,
                                    &mut submission,
                                    plan,
                                    self,
                                    full_attention,
                                    full_attention_state,
                                    position,
                                    None,
                                    true,
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
                    if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                        submission
                            .rms_norm(
                                &plan.current_hidden_buffer,
                                &self.output_norm_device,
                                &plan.matvec_input_buffer,
                                self.output.host.columns,
                                self.family_metadata.rms_norm_epsilon,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .cast_f32_to_f16(
                                &plan.matvec_input_buffer,
                                &plan.vector_f16_buffer,
                                self.output.host.columns,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .matmul_f16_to_f32(
                                &plan.vector_f16_buffer,
                                transposed_f16,
                                &plan.logits_buffer,
                                1,
                                self.output.host.columns,
                                self.output.host.rows,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .argmax_f32(
                                &plan.logits_buffer,
                                1,
                                self.output.host.rows,
                                &plan.next_token_buffer,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .copy_device_to_host(
                                &plan.next_token_buffer,
                                &plan.next_token_host_buffer,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                    } else {
                        submission
                            .rms_norm_q8_1(
                                &plan.current_hidden_buffer,
                                &self.output_norm_device,
                                &plan.matvec_input_q8_1_buffer,
                                self.output.host.columns,
                                self.family_metadata.rms_norm_epsilon,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission
                            .copy_host_to_device(
                                &plan.argmax_state_host_buffer,
                                &plan.argmax_state_buffer,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        submission.quantized_matvec_q8_1_argmax(
                            &self.output.storage,
                            0,
                            self.output.host.mode,
                            self.output.host.rows,
                            self.output.host.columns,
                            &plan.matvec_input_q8_1_buffer,
                            None,
                            &plan.argmax_state_buffer,
                        )?;
                        submission
                            .copy_device_to_host(
                                &plan.argmax_state_buffer,
                                &plan.argmax_state_host_buffer,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                    }
                    let (report, graph_exec) = submission
                        .commit_captured(psionic_backend_cuda::CudaCommandWait::Completed)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    plan.decode_graph_exec = Some(graph_exec);
                    plan.decode_graph_cache_identity = Some(decode_graph_cache_identity);
                    report
                }
            } else {
                plan.decode_graph_exec = None;
                plan.decode_graph_cache_identity = None;
                let mut submission = backend
                    .begin_captured_submission()
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                submission
                    .copy_host_to_device(
                        &plan.decode_params_host_buffer,
                        &plan.decode_params_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
                    match (&layer.kind, layer_state) {
                        (Qwen35LayerKind::Hybrid(_), Qwen35LayerState::Hybrid(hybrid_state)) => {
                            layer.encode_hybrid_device_submission(
                                backend,
                                &mut submission,
                                plan,
                                self,
                                hybrid_state,
                                position,
                                None,
                                &mut bytes_moved,
                            )?;
                        }
                        (
                            Qwen35LayerKind::FullAttention(full_attention),
                            Qwen35LayerState::FullAttention(full_attention_state),
                        ) => {
                            layer.encode_full_attention_device_submission(
                                backend,
                                &mut submission,
                                plan,
                                self,
                                full_attention,
                                full_attention_state,
                                position,
                                None,
                                true,
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
                if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                    submission
                        .rms_norm(
                            &plan.current_hidden_buffer,
                            &self.output_norm_device,
                            &plan.matvec_input_buffer,
                            self.output.host.columns,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .cast_f32_to_f16(
                            &plan.matvec_input_buffer,
                            &plan.vector_f16_buffer,
                            self.output.host.columns,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .matmul_f16_to_f32(
                            &plan.vector_f16_buffer,
                            transposed_f16,
                            &plan.logits_buffer,
                            1,
                            self.output.host.columns,
                            self.output.host.rows,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .argmax_f32(
                            &plan.logits_buffer,
                            1,
                            self.output.host.rows,
                            &plan.next_token_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .copy_device_to_host(&plan.next_token_buffer, &plan.next_token_host_buffer)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                } else {
                    submission
                        .rms_norm_q8_1(
                            &plan.current_hidden_buffer,
                            &self.output_norm_device,
                            &plan.matvec_input_q8_1_buffer,
                            self.output.host.columns,
                            self.family_metadata.rms_norm_epsilon,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission
                        .copy_host_to_device(
                            &plan.argmax_state_host_buffer,
                            &plan.argmax_state_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission.quantized_matvec_q8_1_argmax(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        self.output.host.rows,
                        self.output.host.columns,
                        &plan.matvec_input_q8_1_buffer,
                        None,
                        &plan.argmax_state_buffer,
                    )?;
                    submission
                        .copy_device_to_host(
                            &plan.argmax_state_buffer,
                            &plan.argmax_state_host_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                }
                let (report, graph_exec) = submission
                    .commit_captured(psionic_backend_cuda::CudaCommandWait::Completed)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                plan.decode_graph_exec = Some(graph_exec);
                plan.decode_graph_cache_identity = Some(decode_graph_cache_identity);
                report
            };
            kernel_count = kernel_count.saturating_add(report.encoded_operations);
            if reused_graph_exec {
                for layer_state in &mut state.layers {
                    if let Qwen35LayerState::FullAttention(full_attention) = layer_state {
                        full_attention.len = full_attention.len.saturating_add(1);
                    }
                }
            }
            state.position = state.position.saturating_add(1);
            let selected_token = if output_uses_q8_1_argmax {
                cuda_argmax_token_from_packed_host_buffer(&plan.argmax_state_host_buffer)?
            } else {
                cuda_argmax_token_id(
                    plan.next_token_host_buffer
                        .read_i32()
                        .map_err(ReferenceTextGenerationError::Runtime)?,
                )?
            };
            return Ok(Qwen35ForwardStep {
                logits: Vec::new(),
                selected_token: Some(selected_token),
                candidates: None,
                kernel_count,
                bytes_moved,
                output_metrics: Some(qwen35_decode_output_metrics(
                    Qwen35CudaDecodeOutputMode::ArgmaxOnly,
                    if output_uses_q8_1_argmax {
                        std::mem::size_of::<u64>().try_into().unwrap_or(u64::MAX)
                    } else {
                        std::mem::size_of::<i32>().try_into().unwrap_or(u64::MAX)
                    },
                    false,
                )),
            });
        }

        if std::env::var_os("PSIONIC_QWEN35_DEBUG_FUSED_LAYERS").is_some() {
            for (layer_index, (layer, layer_state)) in
                self.layers.iter().zip(state.layers.iter_mut()).enumerate()
            {
                let initial_token = (layer_index == 0)
                    .then_some(token)
                    .filter(|_| self.token_embedding_f16.is_some());
                let mut submission = backend
                    .begin_submission()
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                match (&layer.kind, &mut *layer_state) {
                    (Qwen35LayerKind::Hybrid(_), Qwen35LayerState::Hybrid(hybrid_state)) => {
                        layer.encode_hybrid_device_submission(
                            backend,
                            &mut submission,
                            plan,
                            self,
                            hybrid_state,
                            position,
                            initial_token,
                            &mut bytes_moved,
                        )?;
                    }
                    (
                        Qwen35LayerKind::FullAttention(full_attention),
                        Qwen35LayerState::FullAttention(full_attention_state),
                    ) => {
                        layer.encode_full_attention_device_submission(
                            backend,
                            &mut submission,
                            plan,
                            self,
                            full_attention,
                            full_attention_state,
                            position,
                            initial_token,
                            false,
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
                let report = submission
                    .commit(psionic_backend_cuda::CudaCommandWait::Completed)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                kernel_count = kernel_count.saturating_add(report.encoded_operations);
                if let (Qwen35LayerKind::Hybrid(hybrid), Qwen35LayerState::Hybrid(hybrid_state)) =
                    (&layer.kind, &*layer_state)
                {
                    emit_qwen35_hybrid_intermediate_debug(
                        position,
                        layer_index,
                        hybrid,
                        hybrid_state,
                        plan,
                        self.descriptor.config.hidden_size,
                    )?;
                }
                emit_qwen35_hidden_debug(
                    position,
                    layer_index,
                    &layer.kind,
                    &plan.current_hidden_buffer,
                    self.descriptor.config.hidden_size,
                )?;
            }
            let current_hidden_buffer = plan.current_hidden_buffer.clone();
            let output_norm_device = self.output_norm_device.clone();
            let (logits, selected_token, candidates, output_stats, output_metrics) =
                match output_mode {
                    CudaStepOutputMode::NoOutput => {
                        (Vec::new(), None, None, zero_cuda_matvec_stats(), None)
                    }
                    CudaStepOutputMode::FullLogits => {
                        let (logits, stats) = plan
                            .run_output_logits_from_device(
                                backend,
                                &current_hidden_buffer,
                                &output_norm_device,
                                self.family_metadata.rms_norm_epsilon,
                                self.output.transposed_f16.as_ref(),
                                &self.output.storage,
                                self.output.host.mode,
                                self.output.host.rows,
                                self.output.host.columns,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        let readback_bytes = self
                            .output
                            .host
                            .rows
                            .saturating_mul(std::mem::size_of::<f32>())
                            .try_into()
                            .unwrap_or(u64::MAX);
                        (
                            logits,
                            None,
                            None,
                            stats,
                            Some(qwen35_decode_output_metrics(
                                Qwen35CudaDecodeOutputMode::RawLogits,
                                readback_bytes,
                                true,
                            )),
                        )
                    }
                    CudaStepOutputMode::ArgmaxOnly => {
                        let (selected, stats) = plan
                            .run_output_argmax_from_device(
                                backend,
                                &current_hidden_buffer,
                                &output_norm_device,
                                self.family_metadata.rms_norm_epsilon,
                                self.output.transposed_f16.as_ref(),
                                &self.output.storage,
                                self.output.host.mode,
                                self.output.host.rows,
                                self.output.host.columns,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        let readback_bytes = stats.device_to_host_bytes;
                        (
                            Vec::new(),
                            Some(selected),
                            None,
                            stats,
                            Some(qwen35_decode_output_metrics(
                                Qwen35CudaDecodeOutputMode::ArgmaxOnly,
                                readback_bytes,
                                false,
                            )),
                        )
                    }
                    CudaStepOutputMode::TopKCandidates(top_k) => {
                        let (candidates, stats) = plan
                            .run_output_top_k_from_device(
                                backend,
                                &current_hidden_buffer,
                                &output_norm_device,
                                self.family_metadata.rms_norm_epsilon,
                                self.output.transposed_f16.as_ref(),
                                &self.output.storage,
                                self.output.host.mode,
                                self.output.host.rows,
                                self.output.host.columns,
                                top_k,
                            )
                            .map_err(ReferenceTextGenerationError::Runtime)?;
                        let readback_bytes = stats.device_to_host_bytes;
                        (
                            Vec::new(),
                            None,
                            Some(candidates),
                            stats,
                            Some(qwen35_decode_output_metrics(
                                Qwen35CudaDecodeOutputMode::TopKCandidates { top_k },
                                readback_bytes,
                                false,
                            )),
                        )
                    }
                };
            bytes_moved = bytes_moved.saturating_add(cuda_stats_bytes(output_stats));
            kernel_count = kernel_count.saturating_add(output_stats.kernel_launches);
            state.position = state.position.saturating_add(1);
            return Ok(Qwen35ForwardStep {
                logits,
                selected_token,
                candidates,
                kernel_count,
                bytes_moved,
                output_metrics,
            });
        }

        let mut submission = backend
            .begin_submission()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        for (layer_index, (layer, layer_state)) in
            self.layers.iter().zip(state.layers.iter_mut()).enumerate()
        {
            let initial_token = (layer_index == 0)
                .then_some(token)
                .filter(|_| self.token_embedding_f16.is_some());
            match (&layer.kind, layer_state) {
                (Qwen35LayerKind::Hybrid(_), Qwen35LayerState::Hybrid(hybrid_state)) => {
                    layer.encode_hybrid_device_submission(
                        backend,
                        &mut submission,
                        plan,
                        self,
                        hybrid_state,
                        position,
                        initial_token,
                        &mut bytes_moved,
                    )?;
                }
                (
                    Qwen35LayerKind::FullAttention(full_attention),
                    Qwen35LayerState::FullAttention(full_attention_state),
                ) => {
                    layer.encode_full_attention_device_submission(
                        backend,
                        &mut submission,
                        plan,
                        self,
                        full_attention,
                        full_attention_state,
                        position,
                        initial_token,
                        false,
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

        let output_rows = self.output.host.rows;
        let output_cols = self.output.host.columns;
        let output_mode_q8_1_projection = self.output.transposed_f16.is_none()
            && can_use_q8_1_quantized_matvec(self.output.host.mode);
        let output_mode_q8_1_argmax =
            output_mode_q8_1_projection && can_use_q8_1_argmax(self.output.host.mode);
        match output_mode {
            CudaStepOutputMode::NoOutput => {}
            CudaStepOutputMode::FullLogits => {
                if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                    submission.rms_norm(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.cast_f32_to_f16(
                        &plan.matvec_input_buffer,
                        &plan.vector_f16_buffer,
                        output_cols,
                    )?;
                    submission.matmul_f16_to_f32(
                        &plan.vector_f16_buffer,
                        transposed_f16,
                        &plan.logits_buffer,
                        1,
                        output_cols,
                        output_rows,
                    )?;
                } else if output_mode_q8_1_projection {
                    submission.rms_norm_q8_1(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_q8_1_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.quantized_matvec_q8_1(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        output_rows,
                        output_cols,
                        &plan.matvec_input_q8_1_buffer,
                        None,
                        &plan.logits_buffer,
                    )?;
                } else {
                    submission.rms_norm(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.quantized_matvec(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        output_rows,
                        output_cols,
                        &plan.matvec_input_buffer,
                        &plan.logits_buffer,
                    )?;
                }
                bytes_moved = bytes_moved.saturating_add(
                    output_rows
                        .saturating_mul(std::mem::size_of::<f32>())
                        .try_into()
                        .unwrap_or(u64::MAX),
                );
            }
            CudaStepOutputMode::ArgmaxOnly => {
                if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                    submission.rms_norm(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.cast_f32_to_f16(
                        &plan.matvec_input_buffer,
                        &plan.vector_f16_buffer,
                        output_cols,
                    )?;
                    submission.matmul_f16_to_f32(
                        &plan.vector_f16_buffer,
                        transposed_f16,
                        &plan.logits_buffer,
                        1,
                        output_cols,
                        output_rows,
                    )?;
                    submission.argmax_f32(
                        &plan.logits_buffer,
                        1,
                        output_rows,
                        &plan.next_token_buffer,
                    )?;
                    submission
                        .copy_device_to_host(&plan.next_token_buffer, &plan.next_token_host_buffer)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    bytes_moved = bytes_moved
                        .saturating_add(std::mem::size_of::<i32>().try_into().unwrap_or(u64::MAX));
                } else if output_mode_q8_1_argmax {
                    plan.argmax_state_host_buffer
                        .write_bytes(initial_cuda_argmax_pair_bytes().as_slice())
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission.rms_norm_q8_1(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_q8_1_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission
                        .copy_host_to_device(
                            &plan.argmax_state_host_buffer,
                            &plan.argmax_state_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    submission.quantized_matvec_q8_1_argmax(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        output_rows,
                        output_cols,
                        &plan.matvec_input_q8_1_buffer,
                        None,
                        &plan.argmax_state_buffer,
                    )?;
                    submission
                        .copy_device_to_host(
                            &plan.argmax_state_buffer,
                            &plan.argmax_state_host_buffer,
                        )
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    let argmax_bytes = std::mem::size_of::<u64>().try_into().unwrap_or(u64::MAX);
                    bytes_moved = bytes_moved
                        .saturating_add(argmax_bytes)
                        .saturating_add(argmax_bytes);
                } else if output_mode_q8_1_projection {
                    submission.rms_norm_q8_1(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_q8_1_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.quantized_matvec_q8_1(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        output_rows,
                        output_cols,
                        &plan.matvec_input_q8_1_buffer,
                        None,
                        &plan.logits_buffer,
                    )?;
                    submission.argmax_f32(
                        &plan.logits_buffer,
                        1,
                        output_rows,
                        &plan.next_token_buffer,
                    )?;
                    submission
                        .copy_device_to_host(&plan.next_token_buffer, &plan.next_token_host_buffer)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    bytes_moved = bytes_moved
                        .saturating_add(std::mem::size_of::<i32>().try_into().unwrap_or(u64::MAX));
                } else {
                    submission.rms_norm(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.quantized_matvec(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        output_rows,
                        output_cols,
                        &plan.matvec_input_buffer,
                        &plan.logits_buffer,
                    )?;
                    submission.argmax_f32(
                        &plan.logits_buffer,
                        1,
                        output_rows,
                        &plan.next_token_buffer,
                    )?;
                    submission
                        .copy_device_to_host(&plan.next_token_buffer, &plan.next_token_host_buffer)
                        .map_err(ReferenceTextGenerationError::Runtime)?;
                    bytes_moved = bytes_moved
                        .saturating_add(std::mem::size_of::<i32>().try_into().unwrap_or(u64::MAX));
                }
            }
            CudaStepOutputMode::TopKCandidates(top_k) => {
                if let Some(transposed_f16) = self.output.transposed_f16.as_ref() {
                    submission.rms_norm(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.cast_f32_to_f16(
                        &plan.matvec_input_buffer,
                        &plan.vector_f16_buffer,
                        output_cols,
                    )?;
                    submission.matmul_f16_to_f32(
                        &plan.vector_f16_buffer,
                        transposed_f16,
                        &plan.logits_buffer,
                        1,
                        output_cols,
                        output_rows,
                    )?;
                } else if output_mode_q8_1_projection {
                    submission.rms_norm_q8_1(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_q8_1_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.quantized_matvec_q8_1(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        output_rows,
                        output_cols,
                        &plan.matvec_input_q8_1_buffer,
                        None,
                        &plan.logits_buffer,
                    )?;
                } else {
                    submission.rms_norm(
                        &plan.current_hidden_buffer,
                        &self.output_norm_device,
                        &plan.matvec_input_buffer,
                        output_cols,
                        self.family_metadata.rms_norm_epsilon,
                    )?;
                    submission.quantized_matvec(
                        &self.output.storage,
                        0,
                        self.output.host.mode,
                        output_rows,
                        output_cols,
                        &plan.matvec_input_buffer,
                        &plan.logits_buffer,
                    )?;
                }
                plan.encode_top_k_from_logits(&mut submission, output_rows, top_k)?;
                submission
                    .copy_device_to_host(
                        &plan.top_k_indices_buffer,
                        &plan.top_k_indices_host_buffer,
                    )
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                submission
                    .copy_device_to_host(&plan.top_k_values_buffer, &plan.top_k_values_host_buffer)
                    .map_err(ReferenceTextGenerationError::Runtime)?;
                bytes_moved = bytes_moved.saturating_add(
                    top_k
                        .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
                        .try_into()
                        .unwrap_or(u64::MAX),
                );
            }
        }

        let report = submission
            .commit(psionic_backend_cuda::CudaCommandWait::Completed)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        kernel_count = kernel_count.saturating_add(report.encoded_operations);
        state.position = state.position.saturating_add(1);

        let logits = match output_mode {
            CudaStepOutputMode::FullLogits => plan
                .logits_buffer
                .read_f32_at_offset(0, output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            _ => Vec::new(),
        };
        let candidates = match output_mode {
            CudaStepOutputMode::TopKCandidates(top_k) => Some(
                cuda_top_k_result_from_host_buffers(
                    &plan.top_k_indices_host_buffer,
                    &plan.top_k_values_host_buffer,
                    1,
                    top_k,
                )
                .map_err(ReferenceTextGenerationError::Runtime)?,
            ),
            _ => None,
        };
        let selected_token = match output_mode {
            CudaStepOutputMode::ArgmaxOnly if output_mode_q8_1_argmax => Some(
                cuda_argmax_token_from_packed_host_buffer(&plan.argmax_state_host_buffer)?,
            ),
            CudaStepOutputMode::ArgmaxOnly => Some(cuda_argmax_token_id(
                plan.next_token_host_buffer
                    .read_i32()
                    .map_err(ReferenceTextGenerationError::Runtime)?,
            )?),
            _ => None,
        };
        let output_metrics = match output_mode {
            CudaStepOutputMode::NoOutput => None,
            CudaStepOutputMode::FullLogits => Some(qwen35_decode_output_metrics(
                Qwen35CudaDecodeOutputMode::RawLogits,
                output_rows
                    .saturating_mul(std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap_or(u64::MAX),
                true,
            )),
            CudaStepOutputMode::ArgmaxOnly => Some(qwen35_decode_output_metrics(
                Qwen35CudaDecodeOutputMode::ArgmaxOnly,
                if output_mode_q8_1_argmax {
                    std::mem::size_of::<u64>().try_into().unwrap_or(u64::MAX)
                } else {
                    std::mem::size_of::<i32>().try_into().unwrap_or(u64::MAX)
                },
                false,
            )),
            CudaStepOutputMode::TopKCandidates(top_k) => Some(qwen35_decode_output_metrics(
                Qwen35CudaDecodeOutputMode::TopKCandidates { top_k },
                top_k
                    .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap_or(u64::MAX),
                false,
            )),
        };
        Ok(Qwen35ForwardStep {
            logits,
            selected_token,
            candidates,
            kernel_count,
            bytes_moved,
            output_metrics,
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
    fn encode_full_attention_qkv_native_submission(
        &self,
        submission: &mut CudaSubmission,
        plan: &mut Qwen35CudaStepPlan,
        full_attention: &Qwen35FullAttentionLayer,
        hidden_size: usize,
        epsilon: f32,
        head_count: usize,
        head_dim: usize,
        query_width: usize,
    ) -> Result<(), ReferenceTextGenerationError> {
        let native_qkv = full_attention.native_qkv.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "qwen35 full-attention native qkv path requested without native matrices",
            )))
        })?;
        let query_bytes = query_width.saturating_mul(std::mem::size_of::<f32>());
        let kv_bytes = full_attention
            .kv_width
            .saturating_mul(std::mem::size_of::<f32>());
        submission.rms_norm_q8_1(
            &plan.current_hidden_buffer,
            &self.attention_norm_device,
            &plan.matvec_input_q8_1_buffer,
            hidden_size,
            epsilon,
        )?;
        submission.quantized_matvec_q8_1(
            &native_qkv.query_gate.storage,
            0,
            native_qkv.query_gate.host.mode,
            native_qkv.query_gate.host.rows,
            native_qkv.query_gate.host.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.matvec_output_buffer,
        )?;
        submission.split_interleaved_query_gate_rms_norm_f32(
            &plan.matvec_output_buffer,
            head_count,
            head_dim,
            &full_attention.query_norm_device,
            epsilon,
            &plan.qkv_norm_buffer,
            &plan.gate_buffer,
        )?;
        submission.quantized_matvec_q8_1(
            &native_qkv.key.storage,
            0,
            native_qkv.key.host.mode,
            native_qkv.key.host.rows,
            native_qkv.key.host.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.k_buffer,
        )?;
        submission.rms_norm(
            &plan.k_buffer,
            &full_attention.key_norm_device,
            &plan.q_buffer,
            full_attention.kv_width,
            epsilon,
        )?;
        submission.copy_buffer_region(
            &plan.q_buffer,
            0,
            &plan.qkv_norm_buffer,
            query_bytes,
            kv_bytes,
        )?;
        submission.quantized_matvec_q8_1(
            &native_qkv.value.storage,
            0,
            native_qkv.value.host.mode,
            native_qkv.value.host.rows,
            native_qkv.value.host.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.k_buffer,
        )?;
        submission.copy_buffer_region(
            &plan.k_buffer,
            0,
            &plan.qkv_norm_buffer,
            query_bytes.saturating_add(kv_bytes),
            kv_bytes,
        )?;
        Ok(())
    }

    fn encode_full_attention_qkv_native_debug_submission(
        &self,
        submission: &mut CudaSubmission,
        plan: &mut Qwen35CudaStepPlan,
        full_attention: &Qwen35FullAttentionLayer,
        hidden_size: usize,
        epsilon: f32,
        head_count: usize,
        head_dim: usize,
        query_width: usize,
    ) -> Result<(), ReferenceTextGenerationError> {
        let native_qkv = full_attention.native_qkv.as_ref().ok_or_else(|| {
            ReferenceTextGenerationError::Runtime(crate::RuntimeError::Backend(String::from(
                "qwen35 full-attention native debug qkv path requested without native matrices",
            )))
        })?;
        let query_bytes = query_width.saturating_mul(std::mem::size_of::<f32>());
        let kv_bytes = full_attention
            .kv_width
            .saturating_mul(std::mem::size_of::<f32>());
        submission.rms_norm_q8_1(
            &plan.current_hidden_buffer,
            &self.attention_norm_device,
            &plan.matvec_input_q8_1_buffer,
            hidden_size,
            epsilon,
        )?;
        submission.quantized_matvec_q8_1(
            &native_qkv.query_gate.storage,
            0,
            native_qkv.query_gate.host.mode,
            native_qkv.query_gate.host.rows,
            native_qkv.query_gate.host.columns,
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
        submission.rms_norm(
            &plan.q_buffer,
            &full_attention.query_norm_device,
            &plan.q_buffer,
            query_width,
            epsilon,
        )?;
        submission.quantized_matvec_q8_1(
            &native_qkv.key.storage,
            0,
            native_qkv.key.host.mode,
            native_qkv.key.host.rows,
            native_qkv.key.host.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.k_buffer,
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
        submission.quantized_matvec_q8_1(
            &native_qkv.value.storage,
            0,
            native_qkv.value.host.mode,
            native_qkv.value.host.rows,
            native_qkv.value.host.columns,
            &plan.matvec_input_q8_1_buffer,
            None,
            &plan.matvec_output_buffer,
        )?;
        submission.copy_buffer_region(
            &plan.matvec_output_buffer,
            0,
            &plan.qkv_norm_buffer,
            query_bytes.saturating_add(kv_bytes),
            kv_bytes,
        )?;
        Ok(())
    }

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
                });
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

    #[allow(clippy::too_many_arguments)]
    fn encode_full_attention_device_submission(
        &self,
        backend: &mut CudaBackend,
        submission: &mut CudaSubmission,
        plan: &mut Qwen35CudaStepPlan,
        model: &CudaQwen35Model,
        full_attention: &Qwen35FullAttentionLayer,
        state: &mut Qwen35FullAttentionState,
        position: usize,
        initial_token: Option<TokenId>,
        use_graph_attention: bool,
        bytes_moved: &mut u64,
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
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 full-attention cuda path requires query/gate rows={} for query width {}, actual {}",
                    query_width.saturating_mul(2),
                    query_width,
                    query_gate_rows,
                )),
            ));
        }
        if key_rows != full_attention.kv_width || value_rows != full_attention.kv_width {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 full-attention cuda path requires key/value rows={} and {} to match kv width {}, actual key={} value={}",
                    full_attention.kv_width,
                    full_attention.kv_width,
                    full_attention.kv_width,
                    key_rows,
                    value_rows,
                )),
            ));
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
        if let Some(token) = initial_token {
            *bytes_moved = bytes_moved.saturating_add(
                model.encode_token_embedding_lookup(submission, plan, token, position)?,
            );
        }
        if full_attention.native_qkv.is_some() {
            self.encode_full_attention_qkv_native_submission(
                submission,
                plan,
                full_attention,
                hidden_size,
                epsilon,
                head_count,
                head_dim,
                query_width,
            )?;
        } else if let Some(transposed_f16) = full_attention.qkv.transposed_f16.as_ref() {
            submission.rms_norm(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.hidden_norm_buffer,
                hidden_size,
                epsilon,
            )?;
            submission.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                full_attention.qkv.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                full_attention.qkv.columns,
                full_attention.qkv.total_rows(),
            )?;
        } else if can_use_q8_1_quantized_matvec(full_attention.qkv.mode) {
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
        } else {
            submission.rms_norm(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.hidden_norm_buffer,
                hidden_size,
                epsilon,
            )?;
            submission.quantized_matvec(
                &full_attention.qkv.storage,
                0,
                full_attention.qkv.mode,
                full_attention.qkv.total_rows(),
                full_attention.qkv.columns,
                &plan.hidden_norm_buffer,
                &plan.matvec_output_buffer,
            )?;
        }
        if full_attention.native_qkv.is_none() {
            submission.split_interleaved_query_gate_rms_norm_f32(
                &plan.matvec_output_buffer,
                head_count,
                head_dim,
                &full_attention.query_norm_device,
                epsilon,
                &plan.qkv_norm_buffer,
                &plan.gate_buffer,
            )?;
            submission.pack_qwen35_key_value_rms_norm_f32(
                &plan.matvec_output_buffer,
                query_gate_rows,
                query_gate_rows.saturating_add(key_rows),
                kv_head_count,
                head_dim,
                &full_attention.key_norm_device,
                epsilon,
                &plan.qkv_norm_buffer,
                query_width,
                query_width.saturating_add(full_attention.kv_width),
            )?;
        }
        if use_graph_attention {
            submission.attention_decode_rope_cache_f16_kv_graph(
                &plan.qkv_norm_buffer,
                0,
                query_width,
                query_width.saturating_add(full_attention.kv_width),
                &state.key_cache,
                &state.value_cache,
                state.width,
                0,
                &plan.decode_params_buffer,
                model.family_metadata.sliding_window.unwrap_or(0),
                head_count,
                kv_head_count,
                head_dim,
                rotary_dim,
                freq_scale,
                ext_factor,
                corr_dims,
                theta_scale,
                None,
                &plan.gated_delta_buffer,
            )?;
        } else {
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
        }
        if let Some(transposed_f16) = full_attention.output.transposed_f16.as_ref() {
            submission.sigmoid_mul_f32(
                &plan.gated_delta_buffer,
                0,
                &plan.gate_buffer,
                0,
                query_width,
                &plan.gated_delta_buffer,
            )?;
            submission.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                full_attention.output.host.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                full_attention.output.host.columns,
                full_attention.output.host.rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(full_attention.output.host.mode) {
            submission.sigmoid_mul_q8_1(
                &plan.gated_delta_buffer,
                0,
                &plan.gate_buffer,
                0,
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
        } else {
            submission.sigmoid_mul_f32(
                &plan.gated_delta_buffer,
                0,
                &plan.gate_buffer,
                0,
                query_width,
                &plan.gated_delta_buffer,
            )?;
            submission.quantized_matvec(
                &full_attention.output.storage,
                0,
                full_attention.output.host.mode,
                full_attention.output.host.rows,
                full_attention.output.host.columns,
                &plan.gated_delta_buffer,
                &plan.projected_buffer,
            )?;
        }
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
        if let Some(transposed_f16) = self.ffn_gate_up.transposed_f16.as_ref() {
            submission.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                self.ffn_gate_up.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                self.ffn_gate_up.columns,
                self.ffn_gate_up.total_rows(),
            )?;
        } else if can_use_q8_1_quantized_matvec(self.ffn_gate_up.mode) {
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
        } else {
            submission.quantized_matvec(
                &self.ffn_gate_up.storage,
                0,
                self.ffn_gate_up.mode,
                self.ffn_gate_up.total_rows(),
                self.ffn_gate_up.columns,
                &plan.hidden_norm_buffer,
                &plan.matvec_output_buffer,
            )?;
        }
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
        if let Some(transposed_f16) = self.ffn_down.transposed_f16.as_ref() {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            submission.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                self.ffn_down.host.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                self.ffn_down.host.columns,
                self.ffn_down.host.rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(self.ffn_down.host.mode) {
            submission.silu_mul_q8_1(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
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
        } else {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            submission.quantized_matvec(
                &self.ffn_down.storage,
                0,
                self.ffn_down.host.mode,
                self.ffn_down.host.rows,
                self.ffn_down.host.columns,
                &plan.gated_delta_buffer,
                &plan.projected_buffer,
            )?;
        }
        submission.add_f32_in_place(
            &plan.current_hidden_buffer,
            0,
            &plan.projected_buffer,
            hidden_size,
        )?;
        state.len = state.len.saturating_add(1);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_hybrid_device_submission(
        &self,
        _backend: &mut CudaBackend,
        submission: &mut CudaSubmission,
        plan: &mut Qwen35CudaStepPlan,
        model: &CudaQwen35Model,
        state: &mut Qwen35HybridState,
        position: usize,
        initial_token: Option<TokenId>,
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
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 hybrid cuda path requires aligned alpha/beta widths, got alpha={} beta={} ssm_a={} ssm_dt={}",
                    alpha_rows,
                    beta_rows,
                    hybrid.ssm_a.len(),
                    hybrid.ssm_dt.len()
                )),
            ));
        }
        let q_size = hybrid.group_count.saturating_mul(hybrid.state_size);
        let k_size = q_size;
        let v_size = hybrid.inner_size;
        let v_offset = q_size.saturating_add(k_size);
        let z_offset = qkv_rows;
        let alpha_offset = z_offset.saturating_add(z_rows);
        let beta_offset = alpha_offset.saturating_add(alpha_rows);

        if let Some(token) = initial_token {
            *bytes_moved = bytes_moved.saturating_add(
                model.encode_token_embedding_lookup(submission, plan, token, position)?,
            );
        }
        if let Some(transposed_f16) = hybrid.qkv_gate_alpha_beta.transposed_f16.as_ref() {
            submission.rms_norm(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.hidden_norm_buffer,
                hidden_size,
                epsilon,
            )?;
            submission.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                hybrid.qkv_gate_alpha_beta.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                hybrid.qkv_gate_alpha_beta.columns,
                hybrid.qkv_gate_alpha_beta.total_rows(),
            )?;
        } else if can_use_q8_1_quantized_matvec(hybrid.qkv_gate_alpha_beta.mode) {
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
        } else {
            submission.rms_norm(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.hidden_norm_buffer,
                hidden_size,
                epsilon,
            )?;
            submission.quantized_matvec(
                &hybrid.qkv_gate_alpha_beta.storage,
                0,
                hybrid.qkv_gate_alpha_beta.mode,
                hybrid.qkv_gate_alpha_beta.total_rows(),
                hybrid.qkv_gate_alpha_beta.columns,
                &plan.hidden_norm_buffer,
                &plan.matvec_output_buffer,
            )?;
        }
        submission.depthwise_causal_conv1d_step_silu_f32(
            &plan.matvec_output_buffer,
            &state.conv_state,
            &hybrid.ssm_conv1d_device,
            qkv_rows,
            hybrid.conv_kernel,
            &plan.conv_buffer,
        )?;
        submission.qwen35_ssm_decay_beta_f32(
            &plan.matvec_output_buffer,
            alpha_offset,
            beta_offset,
            &hybrid.ssm_a_device,
            &hybrid.ssm_dt_device,
            alpha_rows,
            &plan.decay_buffer,
            &plan.beta_buffer,
        )?;
        submission.pack_qwen35_hybrid_qkv_rms_norm_f32(
            &plan.conv_buffer,
            0,
            q_size,
            v_offset,
            hybrid.group_count,
            hybrid.state_size,
            v_size,
            &hybrid.q_scale_device,
            &hybrid.k_scale_device,
            1e-6,
            &plan.qkv_norm_buffer,
            0,
            q_size,
            v_offset,
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
            hybrid.v_head_reordered,
            &plan.gated_delta_buffer,
        )?;
        submission.rms_norm_region(
            &plan.gated_delta_buffer,
            0,
            &hybrid.ssm_norm_device,
            &plan.hybrid_norm_buffer,
            0,
            v_size,
            epsilon,
        )?;
        if let Some(transposed_f16) = hybrid.ssm_out.transposed_f16.as_ref() {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                z_offset,
                &plan.hybrid_norm_buffer,
                0,
                v_size,
                &plan.gated_delta_buffer,
            )?;
            submission.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                hybrid.ssm_out.host.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                hybrid.ssm_out.host.columns,
                hybrid.ssm_out.host.rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(hybrid.ssm_out.host.mode) {
            submission.silu_mul_q8_1(
                &plan.matvec_output_buffer,
                z_offset,
                &plan.hybrid_norm_buffer,
                0,
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
        } else {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                z_offset,
                &plan.hybrid_norm_buffer,
                0,
                v_size,
                &plan.gated_delta_buffer,
            )?;
            submission.quantized_matvec(
                &hybrid.ssm_out.storage,
                0,
                hybrid.ssm_out.host.mode,
                hybrid.ssm_out.host.rows,
                hybrid.ssm_out.host.columns,
                &plan.gated_delta_buffer,
                &plan.projected_buffer,
            )?;
        }
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
        if let Some(transposed_f16) = self.ffn_gate_up.transposed_f16.as_ref() {
            submission.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                self.ffn_gate_up.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                self.ffn_gate_up.columns,
                self.ffn_gate_up.total_rows(),
            )?;
        } else if can_use_q8_1_quantized_matvec(self.ffn_gate_up.mode) {
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
        } else {
            submission.quantized_matvec(
                &self.ffn_gate_up.storage,
                0,
                self.ffn_gate_up.mode,
                self.ffn_gate_up.total_rows(),
                self.ffn_gate_up.columns,
                &plan.hidden_norm_buffer,
                &plan.matvec_output_buffer,
            )?;
        }
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
        if let Some(transposed_f16) = self.ffn_down.transposed_f16.as_ref() {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            submission.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                self.ffn_down.host.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                self.ffn_down.host.columns,
                self.ffn_down.host.rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(self.ffn_down.host.mode) {
            submission.silu_mul_q8_1(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
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
        } else {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            submission.quantized_matvec(
                &self.ffn_down.storage,
                0,
                self.ffn_down.host.mode,
                self.ffn_down.host.rows,
                self.ffn_down.host.columns,
                &plan.gated_delta_buffer,
                &plan.projected_buffer,
            )?;
        }
        submission.add_f32_in_place(
            &plan.current_hidden_buffer,
            0,
            &plan.projected_buffer,
            hidden_size,
        )?;
        Ok(())
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
        layer_index: usize,
        full_attention: &Qwen35FullAttentionLayer,
        state: &mut Qwen35FullAttentionState,
        initial_token: Option<TokenId>,
        position: usize,
        kernel_count: &mut usize,
        bytes_moved: &mut u64,
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
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 full-attention cuda path requires query/gate rows={} for query width {}, actual {}",
                    query_width.saturating_mul(2),
                    query_width,
                    query_gate_rows,
                )),
            ));
        }
        if key_rows != full_attention.kv_width || value_rows != full_attention.kv_width {
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 full-attention cuda path requires key/value rows={} and {} to match kv width {}, actual key={} value={}",
                    full_attention.kv_width,
                    full_attention.kv_width,
                    full_attention.kv_width,
                    key_rows,
                    value_rows,
                )),
            ));
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
            if let Some(token) = initial_token {
                *bytes_moved = bytes_moved.saturating_add(
                    model.encode_token_embedding_lookup(&mut prep, plan, token, position)?,
                );
            }
            if full_attention.native_qkv.is_some() {
                self.encode_full_attention_qkv_native_debug_submission(
                    &mut prep,
                    plan,
                    full_attention,
                    hidden_size,
                    epsilon,
                    head_count,
                    head_dim,
                    query_width,
                )?;
            } else if let Some(transposed_f16) = full_attention.qkv.transposed_f16.as_ref() {
                prep.rms_norm(
                    &plan.current_hidden_buffer,
                    &self.attention_norm_device,
                    &plan.hidden_norm_buffer,
                    hidden_size,
                    epsilon,
                )?;
                prep.cast_f32_to_f16(
                    &plan.hidden_norm_buffer,
                    &plan.vector_f16_buffer,
                    full_attention.qkv.columns,
                )?;
                prep.matmul_f16_to_f32(
                    &plan.vector_f16_buffer,
                    transposed_f16,
                    &plan.matvec_output_buffer,
                    1,
                    full_attention.qkv.columns,
                    full_attention.qkv.total_rows(),
                )?;
            } else {
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
            }
            if full_attention.native_qkv.is_none() {
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
            }
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
            let input_hidden = plan
                .current_hidden_buffer
                .read_f32_at_offset(0, hidden_size)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let host_projected = full_attention
                .output
                .host_matvec(host_gated.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let host_post_attention =
                add_vectors(host_projected.as_slice(), input_hidden.as_slice())
                    .map_err(ReferenceTextGenerationError::Runtime)?;
            let host_post_attention_norm = rms_norm(
                host_post_attention.as_slice(),
                self.post_attention_norm.as_slice(),
                epsilon,
            );
            let host_gate_up = self
                .ffn_gate_up
                .host_matvec(host_post_attention_norm.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let host_ffn = silu_glu(
                host_gate_up
                    .slice(0)
                    .map_err(ReferenceTextGenerationError::Runtime)?,
                host_gate_up
                    .slice(1)
                    .map_err(ReferenceTextGenerationError::Runtime)?,
            );
            let host_ffn_down = self
                .ffn_down
                .host_matvec(host_ffn.as_slice())
                .map_err(ReferenceTextGenerationError::Runtime)?;
            let host_final_hidden =
                add_vectors(host_post_attention.as_slice(), host_ffn_down.as_slice())
                    .map_err(ReferenceTextGenerationError::Runtime)?;

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
            if let Some(transposed_f16) = full_attention.output.transposed_f16.as_ref() {
                tail.cast_f32_to_f16(
                    &plan.gated_delta_buffer,
                    &plan.vector_f16_buffer,
                    full_attention.output.host.columns,
                )?;
                tail.matmul_f16_to_f32(
                    &plan.vector_f16_buffer,
                    transposed_f16,
                    &plan.projected_buffer,
                    1,
                    full_attention.output.host.columns,
                    full_attention.output.host.rows,
                )?;
            } else {
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
            }
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
            if let Some(transposed_f16) = self.ffn_gate_up.transposed_f16.as_ref() {
                tail.cast_f32_to_f16(
                    &plan.hidden_norm_buffer,
                    &plan.vector_f16_buffer,
                    self.ffn_gate_up.columns,
                )?;
                tail.matmul_f16_to_f32(
                    &plan.vector_f16_buffer,
                    transposed_f16,
                    &plan.matvec_output_buffer,
                    1,
                    self.ffn_gate_up.columns,
                    self.ffn_gate_up.total_rows(),
                )?;
            } else {
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
            }
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
            if let Some(transposed_f16) = self.ffn_down.transposed_f16.as_ref() {
                tail.silu_mul_f32(
                    &plan.matvec_output_buffer,
                    0,
                    &plan.matvec_output_buffer,
                    gate_rows,
                    gate_rows,
                    &plan.gated_delta_buffer,
                )?;
                tail.cast_f32_to_f16(
                    &plan.gated_delta_buffer,
                    &plan.vector_f16_buffer,
                    self.ffn_down.host.columns,
                )?;
                tail.matmul_f16_to_f32(
                    &plan.vector_f16_buffer,
                    transposed_f16,
                    &plan.projected_buffer,
                    1,
                    self.ffn_down.host.columns,
                    self.ffn_down.host.rows,
                )?;
            } else {
                tail.silu_mul_q8_1(
                    &plan.matvec_output_buffer,
                    0,
                    &plan.matvec_output_buffer,
                    gate_rows,
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
            }
            tail.add_f32_in_place(
                &plan.current_hidden_buffer,
                0,
                &plan.projected_buffer,
                hidden_size,
            )?;
            let tail_report = tail.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
            *kernel_count = kernel_count.saturating_add(tail_report.encoded_operations);
            let device_final_hidden = plan
                .current_hidden_buffer
                .read_f32_at_offset(0, hidden_size)
                .map_err(ReferenceTextGenerationError::Runtime)?;
            state.len = state.len.saturating_add(1);
            eprintln!(
                "qwen35_debug layer={} position={} state_len={} max_gated_diff={max_diff:.6} final_hidden_diff={:.6}",
                layer_index,
                position,
                state.len.saturating_sub(1),
                max_abs_diff(device_final_hidden.as_slice(), host_final_hidden.as_slice())?,
            );
            return Ok(());
        }

        let mut submission = backend
            .begin_submission()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        if let Some(token) = initial_token {
            *bytes_moved = bytes_moved.saturating_add(model.encode_token_embedding_lookup(
                &mut submission,
                plan,
                token,
                position,
            )?);
        }
        if full_attention.native_qkv.is_some() {
            self.encode_full_attention_qkv_native_submission(
                &mut submission,
                plan,
                full_attention,
                hidden_size,
                epsilon,
                head_count,
                head_dim,
                query_width,
            )?;
        } else if can_use_q8_1_quantized_matvec(full_attention.qkv.mode) {
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
        } else {
            submission.rms_norm(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.hidden_norm_buffer,
                hidden_size,
                epsilon,
            )?;
            submission.quantized_matvec(
                &full_attention.qkv.storage,
                0,
                full_attention.qkv.mode,
                full_attention.qkv.total_rows(),
                full_attention.qkv.columns,
                &plan.hidden_norm_buffer,
                &plan.matvec_output_buffer,
            )?;
        }
        if full_attention.native_qkv.is_none() {
            submission.split_interleaved_query_gate_rms_norm_f32(
                &plan.matvec_output_buffer,
                head_count,
                head_dim,
                &full_attention.query_norm_device,
                epsilon,
                &plan.qkv_norm_buffer,
                &plan.gate_buffer,
            )?;
            submission.pack_qwen35_key_value_rms_norm_f32(
                &plan.matvec_output_buffer,
                query_gate_rows,
                query_gate_rows.saturating_add(key_rows),
                kv_head_count,
                head_dim,
                &full_attention.key_norm_device,
                epsilon,
                &plan.qkv_norm_buffer,
                query_width,
                query_width.saturating_add(full_attention.kv_width),
            )?;
        }
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
        if can_use_q8_1_quantized_matvec(full_attention.output.host.mode) {
            submission.sigmoid_mul_q8_1(
                &plan.gated_delta_buffer,
                0,
                &plan.gate_buffer,
                0,
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
        } else {
            submission.sigmoid_mul_f32(
                &plan.gated_delta_buffer,
                0,
                &plan.gate_buffer,
                0,
                query_width,
                &plan.gated_delta_buffer,
            )?;
            submission.quantized_matvec(
                &full_attention.output.storage,
                0,
                full_attention.output.host.mode,
                full_attention.output.host.rows,
                full_attention.output.host.columns,
                &plan.gated_delta_buffer,
                &plan.projected_buffer,
            )?;
        }
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
        if let Some(transposed_f16) = self.ffn_gate_up.transposed_f16.as_ref() {
            submission.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                self.ffn_gate_up.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                self.ffn_gate_up.columns,
                self.ffn_gate_up.total_rows(),
            )?;
        } else if can_use_q8_1_quantized_matvec(self.ffn_gate_up.mode) {
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
        } else {
            submission.quantized_matvec(
                &self.ffn_gate_up.storage,
                0,
                self.ffn_gate_up.mode,
                self.ffn_gate_up.total_rows(),
                self.ffn_gate_up.columns,
                &plan.hidden_norm_buffer,
                &plan.matvec_output_buffer,
            )?;
        }
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
        if let Some(transposed_f16) = self.ffn_down.transposed_f16.as_ref() {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            submission.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                self.ffn_down.host.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                self.ffn_down.host.columns,
                self.ffn_down.host.rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(self.ffn_down.host.mode) {
            submission.silu_mul_q8_1(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
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
        } else {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            submission.quantized_matvec(
                &self.ffn_down.storage,
                0,
                self.ffn_down.host.mode,
                self.ffn_down.host.rows,
                self.ffn_down.host.columns,
                &plan.gated_delta_buffer,
                &plan.projected_buffer,
            )?;
        }
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
        layer_index: usize,
        state: &mut Qwen35HybridState,
        position: usize,
        initial_token: Option<TokenId>,
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
        if qwen35_hybrid_compare_enabled(layer_index, position) {
            return self.forward_hybrid_device_debug_compare(
                backend,
                plan,
                model,
                layer_index,
                hybrid,
                state,
                position,
                initial_token,
                kernel_count,
                bytes_moved,
            );
        }
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
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 hybrid cuda path requires aligned alpha/beta widths, got alpha={} beta={} ssm_a={} ssm_dt={}",
                    alpha_rows,
                    beta_rows,
                    hybrid.ssm_a.len(),
                    hybrid.ssm_dt.len()
                )),
            ));
        }
        let q_size = hybrid.group_count.saturating_mul(hybrid.state_size);
        let k_size = q_size;
        let v_size = hybrid.inner_size;
        let v_offset = q_size.saturating_add(k_size);
        let z_offset = qkv_rows;
        let alpha_offset = z_offset.saturating_add(z_rows);
        let beta_offset = alpha_offset.saturating_add(alpha_rows);
        let v_bytes = v_size.saturating_mul(std::mem::size_of::<f32>());

        let mut submission = backend
            .begin_submission()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        if let Some(token) = initial_token {
            *bytes_moved = bytes_moved.saturating_add(model.encode_token_embedding_lookup(
                &mut submission,
                plan,
                token,
                position,
            )?);
        }
        if let Some(transposed_f16) = hybrid.qkv_gate_alpha_beta.transposed_f16.as_ref() {
            submission.rms_norm(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.hidden_norm_buffer,
                hidden_size,
                epsilon,
            )?;
            submission.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                hybrid.qkv_gate_alpha_beta.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                hybrid.qkv_gate_alpha_beta.columns,
                hybrid.qkv_gate_alpha_beta.total_rows(),
            )?;
        } else if can_use_q8_1_quantized_matvec(hybrid.qkv_gate_alpha_beta.mode) {
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
        } else {
            submission.rms_norm(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.hidden_norm_buffer,
                hidden_size,
                epsilon,
            )?;
            submission.quantized_matvec(
                &hybrid.qkv_gate_alpha_beta.storage,
                0,
                hybrid.qkv_gate_alpha_beta.mode,
                hybrid.qkv_gate_alpha_beta.total_rows(),
                hybrid.qkv_gate_alpha_beta.columns,
                &plan.hidden_norm_buffer,
                &plan.matvec_output_buffer,
            )?;
        }
        submission.depthwise_causal_conv1d_step_silu_f32(
            &plan.matvec_output_buffer,
            &state.conv_state,
            &hybrid.ssm_conv1d_device,
            qkv_rows,
            hybrid.conv_kernel,
            &plan.conv_buffer,
        )?;
        submission.qwen35_ssm_decay_beta_f32(
            &plan.matvec_output_buffer,
            alpha_offset,
            beta_offset,
            &hybrid.ssm_a_device,
            &hybrid.ssm_dt_device,
            alpha_rows,
            &plan.decay_buffer,
            &plan.beta_buffer,
        )?;
        submission.rms_norm_region(
            &plan.conv_buffer,
            0,
            &hybrid.q_scale_device,
            &plan.qkv_norm_buffer,
            0,
            q_size,
            1e-6,
        )?;
        submission.rms_norm_region(
            &plan.conv_buffer,
            q_size,
            &hybrid.k_scale_device,
            &plan.qkv_norm_buffer,
            q_size,
            k_size,
            1e-6,
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
            hybrid.v_head_reordered,
            &plan.gated_delta_buffer,
        )?;
        submission.rms_norm_region(
            &plan.gated_delta_buffer,
            0,
            &hybrid.ssm_norm_device,
            &plan.hybrid_norm_buffer,
            0,
            v_size,
            epsilon,
        )?;
        if let Some(transposed_f16) = hybrid.ssm_out.transposed_f16.as_ref() {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                z_offset,
                &plan.hybrid_norm_buffer,
                0,
                v_size,
                &plan.gated_delta_buffer,
            )?;
            submission.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                hybrid.ssm_out.host.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                hybrid.ssm_out.host.columns,
                hybrid.ssm_out.host.rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(hybrid.ssm_out.host.mode) {
            submission.silu_mul_q8_1(
                &plan.matvec_output_buffer,
                z_offset,
                &plan.hybrid_norm_buffer,
                0,
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
        } else {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                z_offset,
                &plan.hybrid_norm_buffer,
                0,
                v_size,
                &plan.gated_delta_buffer,
            )?;
            submission.quantized_matvec(
                &hybrid.ssm_out.storage,
                0,
                hybrid.ssm_out.host.mode,
                hybrid.ssm_out.host.rows,
                hybrid.ssm_out.host.columns,
                &plan.gated_delta_buffer,
                &plan.projected_buffer,
            )?;
        }
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
        if let Some(transposed_f16) = self.ffn_gate_up.transposed_f16.as_ref() {
            submission.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                self.ffn_gate_up.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                self.ffn_gate_up.columns,
                self.ffn_gate_up.total_rows(),
            )?;
        } else if can_use_q8_1_quantized_matvec(self.ffn_gate_up.mode) {
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
        } else {
            submission.quantized_matvec(
                &self.ffn_gate_up.storage,
                0,
                self.ffn_gate_up.mode,
                self.ffn_gate_up.total_rows(),
                self.ffn_gate_up.columns,
                &plan.hidden_norm_buffer,
                &plan.matvec_output_buffer,
            )?;
        }
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
        if let Some(transposed_f16) = self.ffn_down.transposed_f16.as_ref() {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            submission.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                self.ffn_down.host.columns,
            )?;
            submission.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                self.ffn_down.host.columns,
                self.ffn_down.host.rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(self.ffn_down.host.mode) {
            submission.silu_mul_q8_1(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
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
        } else {
            submission.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            submission.quantized_matvec(
                &self.ffn_down.storage,
                0,
                self.ffn_down.host.mode,
                self.ffn_down.host.rows,
                self.ffn_down.host.columns,
                &plan.gated_delta_buffer,
                &plan.projected_buffer,
            )?;
        }
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

    #[allow(clippy::too_many_arguments)]
    fn forward_hybrid_device_debug_compare(
        &self,
        backend: &mut CudaBackend,
        plan: &mut Qwen35CudaStepPlan,
        model: &CudaQwen35Model,
        layer_index: usize,
        hybrid: &Qwen35HybridLayer,
        state: &mut Qwen35HybridState,
        position: usize,
        initial_token: Option<TokenId>,
        kernel_count: &mut usize,
        bytes_moved: &mut u64,
    ) -> Result<(), ReferenceTextGenerationError> {
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
            return Err(ReferenceTextGenerationError::Runtime(
                crate::RuntimeError::Backend(format!(
                    "qwen35 hybrid cuda path requires aligned alpha/beta widths, got alpha={} beta={} ssm_a={} ssm_dt={}",
                    alpha_rows,
                    beta_rows,
                    hybrid.ssm_a.len(),
                    hybrid.ssm_dt.len()
                )),
            ));
        }
        let q_size = hybrid.group_count.saturating_mul(hybrid.state_size);
        let k_size = q_size;
        let v_size = hybrid.inner_size;
        let v_offset = q_size.saturating_add(k_size);
        let z_offset = qkv_rows;
        let alpha_offset = z_offset.saturating_add(z_rows);
        let beta_offset = alpha_offset.saturating_add(alpha_rows);

        let input_hidden = if let Some(token) = initial_token {
            model.token_embedding.decode_row(token.as_u32() as usize)?
        } else {
            plan.current_hidden_buffer
                .read_f32_at_offset(0, hidden_size)
                .map_err(ReferenceTextGenerationError::Runtime)?
        };
        let conv_state_before = state
            .conv_state
            .read_f32()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let delta_state_before = state
            .delta_state
            .read_f32()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let host = self.compute_qwen35_hybrid_host_debug(
            hybrid,
            input_hidden.as_slice(),
            conv_state_before.as_slice(),
            delta_state_before.as_slice(),
            epsilon,
        )?;

        let mut attention = backend
            .begin_submission()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        if let Some(token) = initial_token {
            *bytes_moved = bytes_moved.saturating_add(model.encode_token_embedding_lookup(
                &mut attention,
                plan,
                token,
                position,
            )?);
        }
        if let Some(transposed_f16) = hybrid.qkv_gate_alpha_beta.transposed_f16.as_ref() {
            attention.rms_norm(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.hidden_norm_buffer,
                hidden_size,
                epsilon,
            )?;
            attention.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                hybrid.qkv_gate_alpha_beta.columns,
            )?;
            attention.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                hybrid.qkv_gate_alpha_beta.columns,
                hybrid.qkv_gate_alpha_beta.total_rows(),
            )?;
        } else {
            attention.rms_norm_q8_1(
                &plan.current_hidden_buffer,
                &self.attention_norm_device,
                &plan.matvec_input_q8_1_buffer,
                hidden_size,
                epsilon,
            )?;
            attention.quantized_matvec_q8_1(
                &hybrid.qkv_gate_alpha_beta.storage,
                0,
                hybrid.qkv_gate_alpha_beta.mode,
                hybrid.qkv_gate_alpha_beta.total_rows(),
                hybrid.qkv_gate_alpha_beta.columns,
                &plan.matvec_input_q8_1_buffer,
                None,
                &plan.matvec_output_buffer,
            )?;
        }
        attention.depthwise_causal_conv1d_step_silu_f32(
            &plan.matvec_output_buffer,
            &state.conv_state,
            &hybrid.ssm_conv1d_device,
            qkv_rows,
            hybrid.conv_kernel,
            &plan.conv_buffer,
        )?;
        attention.qwen35_ssm_decay_beta_f32(
            &plan.matvec_output_buffer,
            alpha_offset,
            beta_offset,
            &hybrid.ssm_a_device,
            &hybrid.ssm_dt_device,
            alpha_rows,
            &plan.decay_buffer,
            &plan.beta_buffer,
        )?;
        attention.pack_qwen35_hybrid_qkv_rms_norm_f32(
            &plan.conv_buffer,
            0,
            q_size,
            v_offset,
            hybrid.group_count,
            hybrid.state_size,
            v_size,
            &hybrid.q_scale_device,
            &hybrid.k_scale_device,
            1e-6,
            &plan.qkv_norm_buffer,
            0,
            q_size,
            v_offset,
        )?;
        attention.gated_delta_step_f32(
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
            hybrid.v_head_reordered,
            &plan.gated_delta_buffer,
        )?;
        attention.rms_norm_region(
            &plan.gated_delta_buffer,
            0,
            &hybrid.ssm_norm_device,
            &plan.hybrid_norm_buffer,
            0,
            v_size,
            epsilon,
        )?;
        if let Some(transposed_f16) = hybrid.ssm_out.transposed_f16.as_ref() {
            attention.silu_mul_f32(
                &plan.matvec_output_buffer,
                z_offset,
                &plan.hybrid_norm_buffer,
                0,
                v_size,
                &plan.gated_delta_buffer,
            )?;
            attention.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                hybrid.ssm_out.host.columns,
            )?;
            attention.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                hybrid.ssm_out.host.columns,
                hybrid.ssm_out.host.rows,
            )?;
        } else {
            attention.silu_mul_q8_1(
                &plan.matvec_output_buffer,
                z_offset,
                &plan.hybrid_norm_buffer,
                0,
                v_size,
                &plan.activated_q8_1_buffer,
            )?;
            attention.quantized_matvec_q8_1(
                &hybrid.ssm_out.storage,
                0,
                hybrid.ssm_out.host.mode,
                hybrid.ssm_out.host.rows,
                hybrid.ssm_out.host.columns,
                &plan.activated_q8_1_buffer,
                None,
                &plan.projected_buffer,
            )?;
        }
        let attention_report = attention
            .commit(psionic_backend_cuda::CudaCommandWait::Completed)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        *kernel_count = kernel_count.saturating_add(attention_report.encoded_operations);

        let device_conv = plan
            .conv_buffer
            .read_f32_at_offset(0, qkv_rows)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let device_decay = plan
            .decay_buffer
            .read_f32_at_offset(0, alpha_rows)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let device_beta = plan
            .beta_buffer
            .read_f32_at_offset(0, alpha_rows)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let device_qkv_norm = plan
            .qkv_norm_buffer
            .read_f32_at_offset(0, v_offset.saturating_add(v_size))
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let device_gated_delta = plan
            .gated_delta_buffer
            .read_f32_at_offset(0, v_size)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let device_hybrid_norm = plan
            .hybrid_norm_buffer
            .read_f32_at_offset(0, v_size)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let device_projected = plan
            .projected_buffer
            .read_f32_at_offset(0, hidden_size)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let device_conv_state = state
            .conv_state
            .read_f32()
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let device_delta_state = state
            .delta_state
            .read_f32()
            .map_err(ReferenceTextGenerationError::Runtime)?;

        let mut tail = backend
            .begin_submission()
            .map_err(ReferenceTextGenerationError::Runtime)?;
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
        if let Some(transposed_f16) = self.ffn_gate_up.transposed_f16.as_ref() {
            tail.cast_f32_to_f16(
                &plan.hidden_norm_buffer,
                &plan.vector_f16_buffer,
                self.ffn_gate_up.columns,
            )?;
            tail.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.matvec_output_buffer,
                1,
                self.ffn_gate_up.columns,
                self.ffn_gate_up.total_rows(),
            )?;
        } else {
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
        }
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
        if let Some(transposed_f16) = self.ffn_down.transposed_f16.as_ref() {
            tail.silu_mul_f32(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
                gate_rows,
                &plan.gated_delta_buffer,
            )?;
            tail.cast_f32_to_f16(
                &plan.gated_delta_buffer,
                &plan.vector_f16_buffer,
                self.ffn_down.host.columns,
            )?;
            tail.matmul_f16_to_f32(
                &plan.vector_f16_buffer,
                transposed_f16,
                &plan.projected_buffer,
                1,
                self.ffn_down.host.columns,
                self.ffn_down.host.rows,
            )?;
        } else {
            tail.silu_mul_q8_1(
                &plan.matvec_output_buffer,
                0,
                &plan.matvec_output_buffer,
                gate_rows,
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
        }
        tail.add_f32_in_place(
            &plan.current_hidden_buffer,
            0,
            &plan.projected_buffer,
            hidden_size,
        )?;
        let tail_report = tail
            .commit(psionic_backend_cuda::CudaCommandWait::Completed)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        *kernel_count = kernel_count.saturating_add(tail_report.encoded_operations);

        let device_final_hidden = plan
            .current_hidden_buffer
            .read_f32_at_offset(0, hidden_size)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        eprintln!(
            "qwen35_hybrid_compare position={} layer={} conv_diff={:.6} decay_diff={:.6} beta_diff={:.6} qkv_norm_diff={:.6} gated_diff={:.6} hybrid_norm_diff={:.6} projected_diff={:.6} conv_state_diff={:.6} delta_state_diff={:.6} final_hidden_diff={:.6}",
            position,
            layer_index,
            max_abs_diff(device_conv.as_slice(), host.conv.as_slice())?,
            max_abs_diff(device_decay.as_slice(), host.decay.as_slice())?,
            max_abs_diff(device_beta.as_slice(), host.beta.as_slice())?,
            max_abs_diff(device_qkv_norm.as_slice(), host.qkv_norm.as_slice())?,
            max_abs_diff(device_gated_delta.as_slice(), host.gated_delta.as_slice())?,
            max_abs_diff(device_hybrid_norm.as_slice(), host.hybrid_norm.as_slice())?,
            max_abs_diff(device_projected.as_slice(), host.projected.as_slice())?,
            max_abs_diff(
                device_conv_state.as_slice(),
                host.next_conv_state.as_slice()
            )?,
            max_abs_diff(
                device_delta_state.as_slice(),
                host.next_delta_state.as_slice()
            )?,
            max_abs_diff(device_final_hidden.as_slice(), host.final_hidden.as_slice())?,
        );
        Ok(())
    }

    fn compute_qwen35_hybrid_host_debug(
        &self,
        hybrid: &Qwen35HybridLayer,
        input_hidden: &[f32],
        conv_state: &[f32],
        delta_state: &[f32],
        epsilon: f32,
    ) -> Result<Qwen35HybridHostDebug, ReferenceTextGenerationError> {
        let hidden_norm = rms_norm(input_hidden, self.attention_norm.as_slice(), epsilon);
        let projected = hybrid
            .qkv_gate_alpha_beta
            .host_matvec(hidden_norm.as_slice())?;
        let qkv = projected
            .slice(0)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let z = projected
            .slice(1)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let alpha = projected
            .slice(2)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let beta = projected
            .slice(3)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let q_size = hybrid.group_count.saturating_mul(hybrid.state_size);
        let k_size = q_size;
        let v_size = hybrid.inner_size;
        let v_offset = q_size.saturating_add(k_size);

        let mut next_conv_state = conv_state.to_vec();
        let mut conv = vec![0.0_f32; qkv.len()];
        causal_depthwise_conv1d_step_in_place(
            qkv,
            next_conv_state.as_mut_slice(),
            &hybrid.ssm_conv1d,
            hybrid.conv_kernel,
            conv.as_mut_slice(),
        )?;
        silu_forward_in_place(conv.as_mut_slice());

        let mut gate_preexp = vec![0.0_f32; alpha.len()];
        let mut decay = vec![0.0_f32; alpha.len()];
        let mut beta_sigmoid = vec![0.0_f32; beta.len()];
        for index in 0..alpha.len() {
            let gate = softplus(alpha[index] + hybrid.ssm_dt[index]) * hybrid.ssm_a[index];
            gate_preexp[index] = gate;
            decay[index] = gate.exp();
            beta_sigmoid[index] = sigmoid(beta[index]);
        }

        let q_scale = vec![1.0_f32 / hybrid.state_size as f32; hybrid.state_size];
        let k_scale = vec![1.0_f32 / (hybrid.state_size as f32).sqrt(); hybrid.state_size];
        let mut qkv_norm = vec![0.0_f32; v_offset.saturating_add(v_size)];
        per_head_rms_norm_into(
            &conv[..q_size],
            hybrid.group_count,
            hybrid.state_size,
            q_scale.as_slice(),
            1e-6,
            &mut qkv_norm[..q_size],
        );
        per_head_rms_norm_into(
            &conv[q_size..q_size + k_size],
            hybrid.group_count,
            hybrid.state_size,
            k_scale.as_slice(),
            1e-6,
            &mut qkv_norm[q_size..q_size + k_size],
        );
        qkv_norm[v_offset..v_offset + v_size].copy_from_slice(&conv[v_offset..v_offset + v_size]);

        let mut next_delta_state = delta_state.to_vec();
        let mut gated_delta = vec![0.0_f32; v_size];
        let mut norm_q = vec![0.0_f32; hybrid.state_size];
        let mut norm_k = vec![0.0_f32; hybrid.state_size];
        let mut kv_mem = vec![0.0_f32; hybrid.state_size];
        let mut delta = vec![0.0_f32; hybrid.state_size];
        let repeat_factor = hybrid.time_step_rank / hybrid.group_count.max(1);
        for value_head_index in 0..hybrid.time_step_rank {
            let key_head_index = if hybrid.v_head_reordered {
                value_head_index % hybrid.group_count.max(1)
            } else if repeat_factor > 0 {
                value_head_index / repeat_factor
            } else {
                0
            };
            let q = &qkv_norm
                [key_head_index * hybrid.state_size..(key_head_index + 1) * hybrid.state_size];
            let k = &qkv_norm[q_size + key_head_index * hybrid.state_size
                ..q_size + (key_head_index + 1) * hybrid.state_size];
            let v = &qkv_norm[v_offset + value_head_index * hybrid.state_size
                ..v_offset + (value_head_index + 1) * hybrid.state_size];
            let state_slice = &mut next_delta_state[value_head_index
                .saturating_mul(hybrid.state_size)
                .saturating_mul(hybrid.state_size)
                ..(value_head_index + 1)
                    .saturating_mul(hybrid.state_size)
                    .saturating_mul(hybrid.state_size)];
            let output_slice = &mut gated_delta
                [value_head_index * hybrid.state_size..(value_head_index + 1) * hybrid.state_size];
            delta_net_autoregressive_step_in_place(
                q,
                k,
                v,
                gate_preexp[value_head_index],
                beta_sigmoid[value_head_index],
                state_slice,
                norm_q.as_mut_slice(),
                norm_k.as_mut_slice(),
                kv_mem.as_mut_slice(),
                delta.as_mut_slice(),
                output_slice,
            );
        }

        let hybrid_norm = per_head_rms_norm(
            gated_delta.as_slice(),
            hybrid.time_step_rank,
            hybrid.state_size,
            hybrid.ssm_norm.as_slice(),
            epsilon,
        );
        let activated = hybrid_norm
            .iter()
            .copied()
            .zip(z.iter().copied())
            .map(|(value, gate)| value * silu_scalar(gate))
            .collect::<Vec<_>>();
        let projected = hybrid
            .ssm_out
            .host_matvec(activated.as_slice())
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let post_attention = add_vectors(projected.as_slice(), input_hidden)
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let post_attention_norm = rms_norm(
            post_attention.as_slice(),
            self.post_attention_norm.as_slice(),
            epsilon,
        );
        let gate_up = self
            .ffn_gate_up
            .host_matvec(post_attention_norm.as_slice())
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let ffn = silu_glu(
            gate_up
                .slice(0)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            gate_up
                .slice(1)
                .map_err(ReferenceTextGenerationError::Runtime)?,
        );
        let ffn_down = self
            .ffn_down
            .host_matvec(ffn.as_slice())
            .map_err(ReferenceTextGenerationError::Runtime)?;
        let final_hidden = add_vectors(post_attention.as_slice(), ffn_down.as_slice())
            .map_err(ReferenceTextGenerationError::Runtime)?;
        Ok(Qwen35HybridHostDebug {
            conv,
            decay,
            beta: beta_sigmoid,
            qkv_norm,
            gated_delta,
            hybrid_norm,
            projected,
            next_conv_state,
            next_delta_state,
            final_hidden,
        })
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
    ssm_a_device: CudaBuffer,
    ssm_dt: Vec<f32>,
    ssm_dt_device: CudaBuffer,
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
    v_head_reordered: bool,
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
        // Ollama applies L2 normalization to q/k, then scales q by
        // `1 / sqrt(head_v_dim)`. Our CUDA kernels use RMS-style normalization
        // with constant weights, so the weights must absorb the
        // `sqrt(state_size)` conversion from RMS to L2 space:
        // q = RMSNorm(x) * (1 / state_size), k = RMSNorm(x) * (1 / sqrt(state_size)).
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
            ssm_a_device: upload_f32_buffer(backend, ssm_a.as_slice(), "qwen35_ssm_a")?,
            ssm_a,
            ssm_dt_device: upload_f32_buffer(backend, ssm_dt.as_slice(), "qwen35_ssm_dt")?,
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
            v_head_reordered: family_fact_bool(metadata, "qwen35.ssm.v_head_reordered")?,
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
            .saturating_add(vec_f32_bytes(self.ssm_a.as_slice()))
            .saturating_add(vec_f32_bytes(self.ssm_dt.as_slice()))
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
    native_qkv: Option<CudaQwen35FullAttentionQkv>,
    query_norm: Vec<f32>,
    query_norm_device: CudaBuffer,
    key_norm: Vec<f32>,
    key_norm_device: CudaBuffer,
    output: CudaQuantizedMatrix,
    kv_width: usize,
}

#[derive(Clone, Debug)]
struct CudaQwen35FullAttentionQkv {
    query_gate: CudaQuantizedMatrix,
    key: CudaQuantizedMatrix,
    value: CudaQuantizedMatrix,
}

impl Qwen35FullAttentionLayer {
    fn load(
        backend: &mut CudaBackend,
        artifact: &GgufBlobArtifact,
        layout: &psionic_models::GgufDecoderLayerTensorLayout,
        _metadata: &GgufDecoderFamilyMetadata,
    ) -> Result<Self, ModelLoadError> {
        let query_name = required_tensor_name(layout.attention_query_weight.as_deref(), "attn_q")?;
        let key_name = required_tensor_name(layout.attention_key_weight.as_deref(), "attn_k")?;
        let value_name = required_tensor_name(layout.attention_value_weight.as_deref(), "attn_v")?;
        let mut qkv = load_cuda_quantized_projection_group(
            backend,
            artifact,
            &[query_name, key_name, value_name],
        )?;
        let native_qkv = if qkv
            .host_parts
            .iter()
            .all(|part| can_use_q8_1_quantized_matvec(part.mode))
            && qkv
                .host_parts
                .windows(2)
                .any(|window| window[0].mode != window[1].mode)
        {
            Some(CudaQwen35FullAttentionQkv {
                query_gate: load_cuda_quantized_matrix(backend, artifact, query_name)?,
                key: load_cuda_quantized_matrix(backend, artifact, key_name)?,
                value: load_cuda_quantized_matrix(backend, artifact, value_name)?,
            })
        } else {
            None
        };
        if native_qkv.is_some() {
            qkv.transposed_f16 = None;
        }
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
            native_qkv,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CudaStepOutputMode {
    NoOutput,
    FullLogits,
    ArgmaxOnly,
    TopKCandidates(usize),
}

#[derive(Clone, Debug)]
struct Qwen35ForwardStep {
    logits: Vec<f32>,
    selected_token: Option<TokenId>,
    candidates: Option<CudaTopKResult>,
    kernel_count: usize,
    bytes_moved: u64,
    output_metrics: Option<Qwen35CudaDecodeOutputMetrics>,
}

#[derive(Debug)]
struct Qwen35CudaStepPlan {
    matvec_input_buffer: CudaBuffer,
    matvec_input_q8_1_buffer: CudaBuffer,
    vector_f16_buffer: CudaBuffer,
    matvec_output_buffer: CudaBuffer,
    current_hidden_buffer: CudaBuffer,
    decode_params_host_buffer: CudaHostBuffer,
    decode_params_buffer: CudaBuffer,
    hidden_norm_buffer: CudaBuffer,
    gate_buffer: CudaBuffer,
    q_buffer: CudaBuffer,
    k_buffer: CudaBuffer,
    qkv_norm_buffer: CudaBuffer,
    conv_buffer: CudaBuffer,
    gated_delta_buffer: CudaBuffer,
    hybrid_norm_buffer: CudaBuffer,
    projected_buffer: CudaBuffer,
    activated_q8_1_buffer: CudaBuffer,
    decay_buffer: CudaBuffer,
    beta_buffer: CudaBuffer,
    logits_buffer: CudaBuffer,
    logits_host_buffer: CudaHostBuffer,
    top_k_indices_buffer: CudaBuffer,
    top_k_values_buffer: CudaBuffer,
    top_k_partial_indices_buffer: CudaBuffer,
    top_k_partial_values_buffer: CudaBuffer,
    next_token_host_buffer: CudaHostBuffer,
    next_token_buffer: CudaBuffer,
    argmax_state_host_buffer: CudaHostBuffer,
    argmax_state_buffer: CudaBuffer,
    top_k_indices_host_buffer: CudaHostBuffer,
    top_k_values_host_buffer: CudaHostBuffer,
    no_output_graph_exec: Option<CudaGraphExec>,
    no_output_graph_cache_identity: Option<Vec<(usize, usize)>>,
    decode_graph_exec: Option<CudaGraphExec>,
    decode_graph_cache_identity: Option<Vec<(usize, usize)>>,
    full_logits_graph_exec: Option<CudaGraphExec>,
    full_logits_graph_cache_identity: Option<Vec<(usize, usize)>>,
    top_k_graph_exec: Option<CudaGraphExec>,
    top_k_graph_cache_identity: Option<(usize, Vec<(usize, usize)>)>,
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
        let top_k_partial_len =
            QWEN35_CUDA_MAX_TOP_K.saturating_mul(QWEN35_CUDA_PARTITIONED_TOP_K_BLOCKS);
        Ok(Self {
            matvec_input_buffer: backend
                .f32_buffer(max_input_columns)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            matvec_input_q8_1_buffer: backend
                .byte_buffer(&vec![0_u8; q8_1_bytes])
                .map_err(ReferenceTextGenerationError::Runtime)?,
            vector_f16_buffer: backend
                .f16_buffer(max_input_columns)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            matvec_output_buffer: backend
                .f32_buffer(max_output_rows)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            current_hidden_buffer: backend
                .f32_buffer(hidden_size)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            decode_params_host_buffer: backend
                .host_buffer(3 * std::mem::size_of::<i32>())
                .map_err(ReferenceTextGenerationError::Runtime)?,
            decode_params_buffer: backend
                .i32_buffer(3)
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
            logits_buffer: backend
                .f32_buffer(vocab_size)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            logits_host_buffer: backend
                .host_buffer(vocab_size * std::mem::size_of::<f32>())
                .map_err(ReferenceTextGenerationError::Runtime)?,
            top_k_indices_buffer: backend
                .i32_buffer(QWEN35_CUDA_MAX_TOP_K)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            top_k_values_buffer: backend
                .f32_buffer(QWEN35_CUDA_MAX_TOP_K)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            top_k_partial_indices_buffer: backend
                .i32_buffer(top_k_partial_len)
                .map_err(ReferenceTextGenerationError::Runtime)?,
            top_k_partial_values_buffer: backend
                .f32_buffer(top_k_partial_len)
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
            top_k_indices_host_buffer: backend
                .host_buffer(QWEN35_CUDA_MAX_TOP_K * std::mem::size_of::<i32>())
                .map_err(ReferenceTextGenerationError::Runtime)?,
            top_k_values_host_buffer: backend
                .host_buffer(QWEN35_CUDA_MAX_TOP_K * std::mem::size_of::<f32>())
                .map_err(ReferenceTextGenerationError::Runtime)?,
            no_output_graph_exec: None,
            no_output_graph_cache_identity: None,
            decode_graph_exec: None,
            decode_graph_cache_identity: None,
            full_logits_graph_exec: None,
            full_logits_graph_cache_identity: None,
            top_k_graph_exec: None,
            top_k_graph_cache_identity: None,
        })
    }

    fn encode_top_k_from_logits(
        &self,
        submission: &mut CudaSubmission,
        logit_count: usize,
        top_k: usize,
    ) -> Result<(), crate::RuntimeError> {
        if top_k > QWEN35_CUDA_PARTITIONED_TOP_K_THRESHOLD {
            return submission.top_k_f32_one_row_partitioned(
                &self.logits_buffer,
                logit_count,
                top_k,
                QWEN35_CUDA_PARTITIONED_TOP_K_BLOCKS,
                &self.top_k_partial_indices_buffer,
                &self.top_k_partial_values_buffer,
                &self.top_k_indices_buffer,
                &self.top_k_values_buffer,
            );
        }
        submission.top_k_f32(
            &self.logits_buffer,
            1,
            logit_count,
            top_k,
            &self.top_k_indices_buffer,
            &self.top_k_values_buffer,
        )
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
        if can_use_q8_1_quantized_matvec(mode) {
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
        if can_use_q8_1_quantized_matvec(mode) {
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
        transposed_f16: Option<&CudaBuffer>,
        weights: &CudaBuffer,
        mode: QuantizationMode,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<f32>, CudaQuantizedMatvecStats), crate::RuntimeError> {
        let mut submission = backend.begin_submission()?;
        if let Some(transposed_f16) = transposed_f16 {
            submission.rms_norm(input, norm_weight, &self.matvec_input_buffer, cols, epsilon)?;
            submission.cast_f32_to_f16(&self.matvec_input_buffer, &self.vector_f16_buffer, cols)?;
            submission.matmul_f16_to_f32(
                &self.vector_f16_buffer,
                transposed_f16,
                &self.logits_buffer,
                1,
                cols,
                rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(mode) {
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

    fn run_output_top_k_from_device(
        &mut self,
        backend: &mut CudaBackend,
        input: &CudaBuffer,
        norm_weight: &CudaBuffer,
        epsilon: f32,
        transposed_f16: Option<&CudaBuffer>,
        weights: &CudaBuffer,
        mode: QuantizationMode,
        rows: usize,
        cols: usize,
        top_k: usize,
    ) -> Result<(CudaTopKResult, CudaQuantizedMatvecStats), crate::RuntimeError> {
        if top_k == 0 || top_k > QWEN35_CUDA_MAX_TOP_K {
            return Err(crate::RuntimeError::Backend(format!(
                "qwen35 cuda top-k width must be in 1..={}, actual {}",
                QWEN35_CUDA_MAX_TOP_K, top_k
            )));
        }
        let mut submission = backend.begin_submission()?;
        if let Some(transposed_f16) = transposed_f16 {
            submission.rms_norm(input, norm_weight, &self.matvec_input_buffer, cols, epsilon)?;
            submission.cast_f32_to_f16(&self.matvec_input_buffer, &self.vector_f16_buffer, cols)?;
            submission.matmul_f16_to_f32(
                &self.vector_f16_buffer,
                transposed_f16,
                &self.logits_buffer,
                1,
                cols,
                rows,
            )?;
        } else if can_use_q8_1_quantized_matvec(mode) {
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
        self.encode_top_k_from_logits(&mut submission, rows, top_k)?;
        submission
            .copy_device_to_host(&self.top_k_indices_buffer, &self.top_k_indices_host_buffer)?;
        submission
            .copy_device_to_host(&self.top_k_values_buffer, &self.top_k_values_host_buffer)?;
        let report = submission.commit(psionic_backend_cuda::CudaCommandWait::Completed)?;
        let result = cuda_top_k_result_from_host_buffers(
            &self.top_k_indices_host_buffer,
            &self.top_k_values_host_buffer,
            1,
            top_k,
        )?;
        Ok((
            result,
            CudaQuantizedMatvecStats {
                host_to_device_bytes: 0,
                device_to_host_bytes: top_k
                    .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
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
        let (selected, device_to_host_bytes, kernel_launches) = if can_use_q8_1_argmax(mode) {
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
        } else if can_use_q8_1_quantized_matvec(mode) {
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
        transposed_f16: Option<&CudaBuffer>,
        weights: &CudaBuffer,
        mode: QuantizationMode,
        rows: usize,
        cols: usize,
    ) -> Result<(TokenId, CudaQuantizedMatvecStats), crate::RuntimeError> {
        let mut submission = backend.begin_submission()?;
        let (selected, host_to_device_bytes, device_to_host_bytes, kernel_launches) =
            if let Some(transposed_f16) = transposed_f16 {
                submission.rms_norm(
                    input,
                    norm_weight,
                    &self.matvec_input_buffer,
                    cols,
                    epsilon,
                )?;
                submission.cast_f32_to_f16(
                    &self.matvec_input_buffer,
                    &self.vector_f16_buffer,
                    cols,
                )?;
                submission.matmul_f16_to_f32(
                    &self.vector_f16_buffer,
                    transposed_f16,
                    &self.logits_buffer,
                    1,
                    cols,
                    rows,
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
            } else if can_use_q8_1_argmax(mode) {
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
            } else if can_use_q8_1_quantized_matvec(mode) {
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

    fn rows(&self) -> usize {
        match &self.kind {
            HostMatrixKind::Dense(matrix) => matrix.rows,
            HostMatrixKind::Quantized(matrix) => matrix.rows,
        }
    }

    fn columns(&self) -> usize {
        match &self.kind {
            HostMatrixKind::Dense(matrix) => matrix.columns,
            HostMatrixKind::Quantized(matrix) => matrix.columns,
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

    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, crate::RuntimeError> {
        if input.len() != self.columns {
            return Err(crate::RuntimeError::Backend(format!(
                "dense matvec input width mismatch: expected {}, actual {}",
                self.columns,
                input.len()
            )));
        }
        Ok(self
            .values
            .chunks_exact(self.columns)
            .map(|row| dot(row, input))
            .collect())
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

    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, crate::RuntimeError> {
        if input.len() != self.columns {
            return Err(crate::RuntimeError::Backend(format!(
                "quantized matvec input width mismatch: expected {}, actual {}",
                self.columns,
                input.len()
            )));
        }
        let bytes = self.storage.bytes().map_err(model_load_runtime_error)?;
        let mut decoded = Vec::with_capacity(self.columns);
        let mut output = Vec::with_capacity(self.rows);
        for row_bytes in bytes.chunks_exact(self.row_byte_len) {
            decoded.clear();
            decode_quantized_row_into(self.mode, row_bytes, &mut decoded)?;
            output.push(dot(decoded.as_slice(), input));
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
struct CudaQuantizedMatrix {
    storage: CudaBuffer,
    host: QuantizedMatrix,
    transposed_f16: Option<CudaBuffer>,
}

impl CudaQuantizedMatrix {
    fn device_residency_bytes(&self) -> usize {
        self.storage.byte_len().saturating_add(
            self.transposed_f16
                .as_ref()
                .map(CudaBuffer::byte_len)
                .unwrap_or(0),
        )
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

    fn host_matvec(&self, input: &[f32]) -> Result<Vec<f32>, crate::RuntimeError> {
        self.host.matvec(input)
    }
}

#[derive(Clone, Debug)]
struct CudaQuantizedProjectionGroup {
    storage: CudaBuffer,
    host_parts: Vec<QuantizedMatrix>,
    rows_per_projection: Vec<usize>,
    columns: usize,
    mode: QuantizationMode,
    transposed_f16: Option<CudaBuffer>,
}

#[derive(Clone, Debug)]
struct ProjectionOutputs {
    values: Vec<f32>,
    spans: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct Qwen35HybridHostDebug {
    conv: Vec<f32>,
    decay: Vec<f32>,
    beta: Vec<f32>,
    qkv_norm: Vec<f32>,
    gated_delta: Vec<f32>,
    hybrid_norm: Vec<f32>,
    projected: Vec<f32>,
    next_conv_state: Vec<f32>,
    next_delta_state: Vec<f32>,
    final_hidden: Vec<f32>,
}

struct LoadedQuantizedProjectionPart {
    name: String,
    mode: QuantizationMode,
    rows: usize,
    columns: usize,
    row_byte_len: usize,
    bytes: Vec<u8>,
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
        self.storage.byte_len().saturating_add(
            self.transposed_f16
                .as_ref()
                .map(CudaBuffer::byte_len)
                .unwrap_or(0),
        )
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

    fn host_matvec(&self, input: &[f32]) -> Result<ProjectionOutputs, crate::RuntimeError> {
        if input.len() != self.columns {
            return Err(crate::RuntimeError::Backend(format!(
                "packed projection host matvec input width mismatch: expected {}, actual {}",
                self.columns,
                input.len()
            )));
        }
        let mut values = Vec::with_capacity(self.total_rows());
        for matrix in &self.host_parts {
            values.extend(matrix.matvec(input)?);
        }
        ProjectionOutputs::new(self.rows_per_projection.as_slice(), values)
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
    let transposed_f16 = if qwen35_requires_dense_f16_mirror(host.mode) {
        Some(
            try_build_cuda_transposed_f16_mirror(
                backend,
                name,
                host.mode,
                host.rows,
                host.columns,
                host.row_byte_len,
                host.storage.bytes()?,
            )?
            .ok_or_else(|| ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "failed to build required qwen35 cuda f16 transpose mirror for `{name}`"
                ),
            })?,
        )
    } else {
        None
    };
    let storage = backend
        .byte_buffer(host.storage.bytes()?)
        .map_err(|error| ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!("failed to upload `{name}` to cuda: {error}"),
        })?;
    Ok(CudaQuantizedMatrix {
        storage,
        host,
        transposed_f16,
    })
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
    let mut host_parts = Vec::with_capacity(names.len());
    let mut projections = Vec::with_capacity(names.len());
    let mut mixed_quantization = false;
    for name in names {
        let projection = load_quantized_matrix(artifact, name)?;
        if let Some(expected_mode) = mode {
            if projection.mode != expected_mode {
                mixed_quantization = true;
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
                mixed_quantization = true;
            }
        } else {
            row_byte_len = Some(projection.row_byte_len);
        }
        rows_per_projection.push(projection.rows);
        host_parts.push(projection.clone());
        projections.push(LoadedQuantizedProjectionPart {
            name: String::from(*name),
            mode: projection.mode,
            rows: projection.rows,
            columns: projection.columns,
            row_byte_len: projection.row_byte_len,
            bytes: projection.storage.bytes()?.to_vec(),
        });
    }
    let packed = pack_quantized_projection_bytes(
        projections
            .iter()
            .map(|projection| projection.bytes.as_slice())
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let resolved_mode = mode.expect("projection group should resolve quantization mode");
    let resolved_columns = columns.expect("projection group should resolve columns");
    let _resolved_row_byte_len =
        row_byte_len.expect("projection group should resolve row byte length");
    let total_rows = rows_per_projection
        .iter()
        .copied()
        .fold(0usize, usize::saturating_add);
    let transposed_f16 = if mixed_quantization || qwen35_requires_dense_f16_mirror(resolved_mode) {
        Some(
            try_build_cuda_projection_group_transposed_f16_mirror(
                backend,
                names.join(", ").as_str(),
                projections.as_slice(),
                total_rows,
                resolved_columns,
            )?
            .ok_or_else(|| ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "failed to build required qwen35 cuda f16 transpose mirror for packed projection `{}`",
                    names.join(", ")
                ),
            })?,
        )
    } else {
        None
    };
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
        host_parts,
        rows_per_projection,
        columns: resolved_columns,
        mode: resolved_mode,
        transposed_f16,
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

fn decode_quantized_projection_group_bytes_transposed_f16(
    projections: &[LoadedQuantizedProjectionPart],
    total_rows: usize,
    columns: usize,
    name: &str,
) -> Result<Vec<u8>, ModelLoadError> {
    let mut transposed = vec![
        0_u8;
        total_rows
            .saturating_mul(columns)
            .saturating_mul(std::mem::size_of::<u16>())
    ];
    let mut decoded_row = Vec::with_capacity(columns);
    let mut row_offset = 0usize;
    for projection in projections {
        if projection.columns != columns {
            return Err(ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "packed projection `{name}` width mismatch while building f16 transpose: expected {columns}, actual {} for `{}`",
                    projection.columns, projection.name
                ),
            });
        }
        let expected_bytes = projection.rows.saturating_mul(projection.row_byte_len);
        if projection.bytes.len() != expected_bytes {
            return Err(ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "packed projection `{name}` byte length mismatch while building f16 transpose for `{}`: expected {expected_bytes}, actual {}",
                    projection.name,
                    projection.bytes.len()
                ),
            });
        }
        for (row_index, row_bytes) in projection
            .bytes
            .chunks_exact(projection.row_byte_len)
            .enumerate()
        {
            decoded_row.clear();
            decode_quantized_row_into(projection.mode, row_bytes, &mut decoded_row).map_err(
                |error| ModelLoadError::ArtifactFormat {
                    format: String::from("gguf"),
                    message: format!(
                        "failed to decode quantized tensor `{}` while building packed projection f16 transpose `{name}`: {error}",
                        projection.name
                    ),
                },
            )?;
            if decoded_row.len() != columns {
                return Err(ModelLoadError::ArtifactFormat {
                    format: String::from("gguf"),
                    message: format!(
                        "packed projection `{name}` decode width mismatch for `{}` while building f16 transpose: expected {columns}, actual {}",
                        projection.name,
                        decoded_row.len()
                    ),
                });
            }
            let packed_row_index = row_offset.saturating_add(row_index);
            for (column_index, value) in decoded_row.iter().copied().enumerate() {
                let offset = column_index
                    .saturating_mul(total_rows)
                    .saturating_add(packed_row_index)
                    .saturating_mul(std::mem::size_of::<u16>());
                transposed[offset..offset + std::mem::size_of::<u16>()]
                    .copy_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
        }
        row_offset = row_offset.saturating_add(projection.rows);
    }
    if row_offset != total_rows {
        return Err(ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!(
                "packed projection `{name}` row count mismatch while building f16 transpose: expected {total_rows}, actual {row_offset}"
            ),
        });
    }
    Ok(transposed)
}

fn try_build_cuda_projection_group_transposed_f16_mirror(
    backend: &mut CudaBackend,
    name: &str,
    projections: &[LoadedQuantizedProjectionPart],
    total_rows: usize,
    columns: usize,
) -> Result<Option<CudaBuffer>, ModelLoadError> {
    let transposed = if projections.windows(2).all(|window| {
        window[0].mode == window[1].mode && window[0].row_byte_len == window[1].row_byte_len
    }) && !projections.is_empty()
    {
        try_build_cuda_transposed_f16_mirror(
            backend,
            name,
            projections[0].mode,
            total_rows,
            columns,
            projections[0].row_byte_len,
            pack_quantized_projection_bytes(
                projections
                    .iter()
                    .map(|projection| projection.bytes.as_slice())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .as_slice(),
        )?
        .map(Some)
        .unwrap_or(None)
    } else {
        let transposed = decode_quantized_projection_group_bytes_transposed_f16(
            projections,
            total_rows,
            columns,
            name,
        )?;
        match backend.byte_buffer(transposed.as_slice()) {
            Ok(buffer) => Some(buffer),
            Err(error) if error.to_string().contains("out of memory") => None,
            Err(error) => {
                return Err(ModelLoadError::ArtifactFormat {
                    format: String::from("gguf"),
                    message: format!(
                        "failed to upload packed projection f16 transpose mirror for `{name}` to cuda: {error}"
                    ),
                });
            }
        }
    };
    Ok(transposed)
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

fn family_fact_bool(
    metadata: &GgufDecoderFamilyMetadata,
    key: &str,
) -> Result<bool, ModelLoadError> {
    metadata
        .family_facts
        .get(key)
        .and_then(GgufMetadataValue::as_bool)
        .ok_or_else(|| ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!("missing required qwen35 family fact `{key}`"),
        })
}

fn gguf_local_blob_open_options() -> LocalBlobOpenOptions {
    LocalBlobOpenOptions::default().with_integrity_policy(BlobIntegrityPolicy::LocalUnverifiedLabel)
}

fn emit_qwen35_hidden_debug(
    position: usize,
    layer_index: usize,
    kind: &Qwen35LayerKind,
    buffer: &CudaBuffer,
    element_count: usize,
) -> Result<(), ReferenceTextGenerationError> {
    let values = buffer
        .read_f32_at_offset(0, element_count)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut max_abs = 0.0_f32;
    for value in values.iter().copied() {
        if value.is_nan() {
            nan_count = nan_count.saturating_add(1);
            continue;
        }
        if !value.is_finite() {
            inf_count = inf_count.saturating_add(1);
            continue;
        }
        finite_count = finite_count.saturating_add(1);
        max_abs = max_abs.max(value.abs());
    }
    let kind = match kind {
        Qwen35LayerKind::Hybrid(_) => "hybrid",
        Qwen35LayerKind::FullAttention(_) => "full_attention",
    };
    eprintln!(
        "qwen35_fused_debug position={} layer={} kind={} finite={} nan={} inf={} max_abs={:.6}",
        position, layer_index, kind, finite_count, nan_count, inf_count, max_abs
    );
    Ok(())
}

fn emit_qwen35_buffer_debug(
    position: usize,
    layer_index: usize,
    label: &str,
    buffer: &CudaBuffer,
    element_offset: usize,
    element_count: usize,
) -> Result<(), ReferenceTextGenerationError> {
    if element_count == 0 {
        return Ok(());
    }
    let values = buffer
        .read_f32_at_offset(element_offset, element_count)
        .map_err(ReferenceTextGenerationError::Runtime)?;
    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut max_abs = 0.0_f32;
    for value in values.iter().copied() {
        if value.is_nan() {
            nan_count = nan_count.saturating_add(1);
            continue;
        }
        if !value.is_finite() {
            inf_count = inf_count.saturating_add(1);
            continue;
        }
        finite_count = finite_count.saturating_add(1);
        max_abs = max_abs.max(value.abs());
    }
    eprintln!(
        "qwen35_hybrid_debug position={} layer={} buffer={} offset={} count={} finite={} nan={} inf={} max_abs={:.6}",
        position,
        layer_index,
        label,
        element_offset,
        element_count,
        finite_count,
        nan_count,
        inf_count,
        max_abs
    );
    Ok(())
}

fn emit_qwen35_hybrid_intermediate_debug(
    position: usize,
    layer_index: usize,
    hybrid: &Qwen35HybridLayer,
    hybrid_state: &Qwen35HybridState,
    plan: &Qwen35CudaStepPlan,
    hidden_size: usize,
) -> Result<(), ReferenceTextGenerationError> {
    let debug_layer = std::env::var("PSIONIC_QWEN35_DEBUG_HYBRID_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    if debug_layer.is_some() && debug_layer != Some(layer_index) {
        return Ok(());
    }
    let debug_position_min = std::env::var("PSIONIC_QWEN35_DEBUG_HYBRID_POSITION_MIN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if position < debug_position_min {
        return Ok(());
    }
    let qkv_rows = hybrid.qkv_gate_alpha_beta.rows_per_projection[0];
    let z_rows = hybrid.qkv_gate_alpha_beta.rows_per_projection[1];
    let alpha_rows = hybrid.qkv_gate_alpha_beta.rows_per_projection[2];
    let q_size = hybrid.group_count.saturating_mul(hybrid.state_size);
    let k_size = q_size;
    let v_size = hybrid.inner_size;
    let v_offset = q_size.saturating_add(k_size);
    let alpha_offset = qkv_rows.saturating_add(z_rows);
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "matvec_output_qkv",
        &plan.matvec_output_buffer,
        0,
        qkv_rows,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "matvec_output_z",
        &plan.matvec_output_buffer,
        qkv_rows,
        z_rows,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "matvec_output_alpha",
        &plan.matvec_output_buffer,
        alpha_offset,
        alpha_rows,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "conv",
        &plan.conv_buffer,
        0,
        qkv_rows,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "decay",
        &plan.decay_buffer,
        0,
        alpha_rows,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "beta",
        &plan.beta_buffer,
        0,
        alpha_rows,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "qkv_norm_q",
        &plan.qkv_norm_buffer,
        0,
        q_size,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "qkv_norm_k",
        &plan.qkv_norm_buffer,
        q_size,
        k_size,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "qkv_norm_v",
        &plan.qkv_norm_buffer,
        v_offset,
        v_size,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "gated_delta",
        &plan.gated_delta_buffer,
        0,
        v_size,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "hybrid_norm",
        &plan.hybrid_norm_buffer,
        0,
        v_size,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "projected",
        &plan.projected_buffer,
        0,
        hidden_size,
    )?;
    emit_qwen35_buffer_debug(
        position,
        layer_index,
        "delta_state",
        &hybrid_state.delta_state,
        0,
        hybrid
            .time_step_rank
            .saturating_mul(hybrid.state_size)
            .saturating_mul(hybrid.state_size),
    )?;
    Ok(())
}

fn qwen35_hybrid_compare_enabled(layer_index: usize, position: usize) -> bool {
    if std::env::var_os("PSIONIC_QWEN35_DEBUG_HYBRID_COMPARE").is_none() {
        return false;
    }
    let debug_layer = std::env::var("PSIONIC_QWEN35_DEBUG_HYBRID_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    if debug_layer.is_some() && debug_layer != Some(layer_index) {
        return false;
    }
    let debug_position_min = std::env::var("PSIONIC_QWEN35_DEBUG_HYBRID_POSITION_MIN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    position >= debug_position_min
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> Result<f32, ReferenceTextGenerationError> {
    if left.len() != right.len() {
        return Err(ReferenceTextGenerationError::Runtime(
            crate::RuntimeError::Backend(format!(
                "vector width mismatch during qwen35 diff: left={} right={}",
                left.len(),
                right.len()
            )),
        ));
    }
    Ok(left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max))
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

fn try_build_cuda_host_matrix_row_major_f16_mirror(
    backend: &mut CudaBackend,
    name: &str,
    matrix: &HostMatrix,
) -> Result<Option<CudaBuffer>, ModelLoadError> {
    let dense = match &matrix.kind {
        HostMatrixKind::Dense(matrix) => dense_matrix_bytes_row_major_f16(matrix),
        HostMatrixKind::Quantized(matrix) => decode_quantized_matrix_bytes_row_major_f16(
            matrix.mode,
            matrix.rows,
            matrix.columns,
            matrix.row_byte_len,
            matrix.storage.bytes()?,
            name,
        )?,
    };
    match backend.byte_buffer(dense.as_slice()) {
        Ok(buffer) => Ok(Some(buffer)),
        Err(error) if error.to_string().contains("out of memory") => Ok(None),
        Err(error) => Err(ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!("failed to upload row-major f16 mirror for `{name}` to cuda: {error}"),
        }),
    }
}

fn decode_quantized_matrix_bytes_transposed_f16(
    mode: QuantizationMode,
    rows: usize,
    columns: usize,
    row_byte_len: usize,
    bytes: &[u8],
    name: &str,
) -> Result<Vec<u8>, ModelLoadError> {
    let expected_bytes = rows.saturating_mul(row_byte_len);
    if bytes.len() != expected_bytes {
        return Err(ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!(
                "quantized tensor `{name}` byte length mismatch while building f16 transpose: expected {expected_bytes}, actual {}",
                bytes.len()
            ),
        });
    }
    let mut transposed = vec![
        0_u8;
        rows.saturating_mul(columns)
            .saturating_mul(std::mem::size_of::<u16>())
    ];
    let mut decoded_row = Vec::with_capacity(columns);
    for (row_index, row_bytes) in bytes.chunks_exact(row_byte_len).enumerate() {
        decoded_row.clear();
        decode_quantized_row_into(mode, row_bytes, &mut decoded_row).map_err(|error| {
            ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "failed to decode quantized tensor `{name}` while building f16 transpose: {error}"
                ),
            }
        })?;
        if decoded_row.len() != columns {
            return Err(ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "quantized tensor `{name}` decode width mismatch while building f16 transpose: expected {columns}, actual {}",
                    decoded_row.len()
                ),
            });
        }
        for (column_index, value) in decoded_row.iter().copied().enumerate() {
            let offset = column_index
                .saturating_mul(rows)
                .saturating_add(row_index)
                .saturating_mul(std::mem::size_of::<u16>());
            transposed[offset..offset + std::mem::size_of::<u16>()]
                .copy_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }
    }
    Ok(transposed)
}

fn try_build_cuda_transposed_f16_mirror(
    backend: &mut CudaBackend,
    name: &str,
    mode: QuantizationMode,
    rows: usize,
    columns: usize,
    row_byte_len: usize,
    bytes: &[u8],
) -> Result<Option<CudaBuffer>, ModelLoadError> {
    let transposed = decode_quantized_matrix_bytes_transposed_f16(
        mode,
        rows,
        columns,
        row_byte_len,
        bytes,
        name,
    )?;
    match backend.byte_buffer(transposed.as_slice()) {
        Ok(buffer) => Ok(Some(buffer)),
        Err(error) if error.to_string().contains("out of memory") => Ok(None),
        Err(error) => Err(ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!("failed to upload f16 transpose mirror for `{name}` to cuda: {error}"),
        }),
    }
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

fn dense_matrix_bytes_row_major_f16(matrix: &DenseMatrix) -> Vec<u8> {
    let mut dense = Vec::with_capacity(
        matrix
            .values
            .len()
            .saturating_mul(std::mem::size_of::<u16>()),
    );
    for value in &matrix.values {
        dense.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
    }
    dense
}

fn decode_quantized_matrix_bytes_row_major_f16(
    mode: QuantizationMode,
    rows: usize,
    columns: usize,
    row_byte_len: usize,
    bytes: &[u8],
    name: &str,
) -> Result<Vec<u8>, ModelLoadError> {
    let expected_bytes = rows.saturating_mul(row_byte_len);
    if bytes.len() != expected_bytes {
        return Err(ModelLoadError::ArtifactFormat {
            format: String::from("gguf"),
            message: format!(
                "quantized tensor `{name}` byte length mismatch while building row-major f16 mirror: expected {expected_bytes}, actual {}",
                bytes.len()
            ),
        });
    }
    let mut dense = vec![
        0_u8;
        rows.saturating_mul(columns)
            .saturating_mul(std::mem::size_of::<u16>())
    ];
    let mut decoded_row = Vec::with_capacity(columns);
    for (row_index, row_bytes) in bytes.chunks_exact(row_byte_len).enumerate() {
        decoded_row.clear();
        decode_quantized_row_into(mode, row_bytes, &mut decoded_row).map_err(|error| {
            ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "failed to decode quantized tensor `{name}` while building row-major f16 mirror: {error}"
                ),
            }
        })?;
        if decoded_row.len() != columns {
            return Err(ModelLoadError::ArtifactFormat {
                format: String::from("gguf"),
                message: format!(
                    "quantized tensor `{name}` decode width mismatch while building row-major f16 mirror: expected {columns}, actual {}",
                    decoded_row.len()
                ),
            });
        }
        for (column_index, value) in decoded_row.iter().copied().enumerate() {
            let offset = row_index
                .saturating_mul(columns)
                .saturating_add(column_index)
                .saturating_mul(std::mem::size_of::<u16>());
            dense[offset..offset + std::mem::size_of::<u16>()]
                .copy_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }
    }
    Ok(dense)
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

fn qwen35_decode_graph_cache_identity(state: &Qwen35State) -> Vec<(usize, usize)> {
    state
        .layers
        .iter()
        .filter_map(|layer| match layer {
            Qwen35LayerState::Hybrid(_) => None,
            Qwen35LayerState::FullAttention(full_attention) => Some((
                full_attention.key_cache.allocation_identity(),
                full_attention.value_cache.allocation_identity(),
            )),
        })
        .collect()
}

fn vec_f32_bytes(values: &[f32]) -> usize {
    values.len().saturating_mul(std::mem::size_of::<f32>())
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mantissa >> 13) as u16) | 1;
        return sign | 0x7c00 | payload;
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half_mantissa = (mantissa >> shift) as u16;
        let remainder_mask = (1_u32 << shift) - 1;
        let remainder = mantissa & remainder_mask;
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half_mantissa & 1) != 0) {
            half_mantissa = half_mantissa.wrapping_add(1);
        }
        return sign | half_mantissa;
    }

    let mut half = sign | (((half_exponent as u16) & 0x1f) << 10) | ((mantissa >> 13) as u16);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) != 0) {
        half = half.wrapping_add(1);
    }
    half
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
