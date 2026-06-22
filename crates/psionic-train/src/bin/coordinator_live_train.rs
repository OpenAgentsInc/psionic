//! Live coordinator-evolution training driver (Khala M6, issue #6014 / EPIC #6017).
//!
//! This is the bounded **live training run** wiring on top of the merged P1-P5
//! substrate. It does two things, in order:
//!
//! 1. **No-spend simulated validation pass** (`--validate`, the default). Proves
//!    the full loop end-to-end on CPU with a REAL frozen cs336 backbone
//!    (`forward_with_hidden`) feeding a real `CoordinatorHead`, sep-CMA-ES
//!    optimizing it, the P5 capability-filtered worker pool, and the
//!    `DailySpendCap` wired in. The verdict source is the deterministic,
//!    zero-spend `SimulatedVerdictSource`, so this moves ZERO sats while
//!    exercising the cap-debit path. It is labeled a SMOKE, not an ML result.
//!
//! 2. **Bounded real run** (`--real`). Held by default: a real run needs a live
//!    Pylon verdict source (dispatch each trajectory as a buy-mode eval job)
//!    and the Tassadar `training.verification_classes.v1` verdict. Those move
//!    sats and MUST debit the shared autonomous daily cap in the `openagents`
//!    Worker (see `coordinator_live_training.rs` module docs). Without a
//!    reachable Pylon endpoint + spend-enabled buy-mode campaign this binary
//!    refuses to fabricate an ML result and prints exactly what is needed,
//!    while still proving the cap would fail closed.
//!
//! Run:
//!   cargo run -q -p psionic-train --bin coordinator_live_train            # validate (no spend)
//!   cargo run -q -p psionic-train --bin coordinator_live_train -- --real  # bounded real run (held)

use std::process::ExitCode;

use psionic_core::Shape;
use psionic_models::{CoordinatorHead, CoordinatorHeadConfig, Cs336A1ReferenceConfig, Cs336A1TransformerLm};
use psionic_nn::ModuleStateLoadMode;
use psionic_core::TensorData;
use psionic_train::{
    DailySpendCap, EvalSample, LiveCoordinatorFitness, LiveRunLane, LiveRunReport,
    SepCmaEs, SepCmaEsConfig, SimulatedVerdictSource, TerminalRewardAdapter, WorkerKind,
    WorkerPoolBinding, WorkerPoolMember, CoordinatorLiveTrainingError,
    OWNER_DAILY_CAP_MSATS,
};

/// Builds a deterministic frozen cs336 backbone with `d_model == hidden_dim`.
/// On the real lane this is replaced by a frozen Qwen3-0.6B; the loop is
/// identical because the head only sees `forward_with_hidden`'s `[1, d_model]`
/// hidden state.
fn frozen_backbone(d_model: usize, vocab: usize) -> Cs336A1TransformerLm {
    let config = Cs336A1ReferenceConfig {
        vocab_size: vocab,
        context_length: 16,
        d_model,
        num_layers: 2,
        num_heads: 2,
        d_ff: d_model * 2,
    };
    let mut model =
        Cs336A1TransformerLm::new("frozen_backbone", config, 10_000.0, 1e-5).expect("backbone");
    let mut weights = model.state_dict();
    // Deterministic but well-spread frozen weights. The token embedding must be
    // strongly token-dependent so different prompts produce well-separated
    // last-token hidden states (otherwise no linear head can route them apart).
    for (index, (path, entry)) in weights.entries.iter_mut().enumerate() {
        let len = entry.spec.shape().element_count();
        let values: Vec<f32> = if path.contains("ln") || path.ends_with("norm.weight") {
            vec![1.0; len]
        } else if path == "token_embeddings.weight" {
            // Distinct, spread embeddings per (token, dim): a 2D-ish pattern so
            // different token ids land in clearly different directions.
            (0..len)
                .map(|i| {
                    let row = (i / d_model) as f32;
                    let col = (i % d_model) as f32;
                    (0.7 * (row * 1.3 + col * 0.9).sin()) + 0.3 * ((row - col) * 0.5).cos()
                })
                .collect()
        } else {
            let base = ((index % 7) as f32) * 0.03 + 0.02;
            (0..len)
                .map(|i| base + ((i % 5) as f32 * 0.05 - 0.1) + 0.04 * ((i as f32) * 0.7).sin())
                .collect()
        };
        entry.data = TensorData::F32(values);
    }
    model
        .load_state_dict(&weights, ModuleStateLoadMode::Strict)
        .expect("frozen weights");
    model
}

fn run_validation() -> Result<LiveRunReport, CoordinatorLiveTrainingError> {
    let d_model = 8;
    let backbone = frozen_backbone(d_model, 32);

    // P5: capability-filtered eligible pool (3 workers for `rust_build`).
    let candidates = vec![
        WorkerPoolMember {
            worker_id: "frontier-claude".to_string(),
            kind: WorkerKind::Frontier,
            receipted_capabilities: ["rust_build".to_string(), "python".to_string()]
                .into_iter()
                .collect(),
        },
        WorkerPoolMember {
            worker_id: "open-pylon-a".to_string(),
            kind: WorkerKind::Open,
            receipted_capabilities: ["rust_build".to_string()].into_iter().collect(),
        },
        WorkerPoolMember {
            worker_id: "open-pylon-b".to_string(),
            kind: WorkerKind::Open,
            receipted_capabilities: ["rust_build".to_string()].into_iter().collect(),
        },
        // Filtered out: lacks rust_build.
        WorkerPoolMember {
            worker_id: "open-python-only".to_string(),
            kind: WorkerKind::Open,
            receipted_capabilities: ["python".to_string()].into_iter().collect(),
        },
    ];
    let pool = WorkerPoolBinding::from_candidates(candidates, "rust_build")?;
    println!(
        "P5 worker pool: {} eligible for `{}` -> {:?}",
        pool.len(),
        pool.required_capability(),
        pool.workers().iter().map(|w| w.worker_id.as_str()).collect::<Vec<_>>()
    );

    let head_config = CoordinatorHeadConfig {
        hidden_dim: d_model,
        num_workers: pool.len(),
        num_roles: 3,
    };
    let seed_head = CoordinatorHead::zeros(head_config)?;
    println!(
        "P2 head: hidden_dim={} num_workers={} num_roles={} -> {} params",
        head_config.hidden_dim,
        head_config.num_workers,
        head_config.num_roles,
        head_config.parameter_count()
    );

    // Three eval samples, each must route to a specific eligible worker to
    // Verify. The "correct" worker varies per sample so the head must learn a
    // real h -> worker mapping (not a constant).
    let samples = vec![
        EvalSample { sample_id: "task-0".to_string(), token_ids: vec![1, 5, 9, 2] },
        EvalSample { sample_id: "task-1".to_string(), token_ids: vec![3, 7, 1, 8] },
        EvalSample { sample_id: "task-2".to_string(), token_ids: vec![2, 2, 6, 4] },
    ];
    let source = SimulatedVerdictSource::new(vec![
        ("task-0".to_string(), 0),
        ("task-1".to_string(), 2),
        ("task-2".to_string(), 1),
    ]);

    let hidden = move |tokens: &[usize]| -> Result<Vec<f32>, CoordinatorLiveTrainingError> {
        let (_, h) = backbone
            .forward_with_hidden(Shape::new(vec![1, tokens.len()]), tokens)
            .map_err(|e| CoordinatorLiveTrainingError::VerdictSource { detail: e.to_string() })?;
        h.as_f32_slice()
            .map(<[f32]>::to_vec)
            .map_err(|e| CoordinatorLiveTrainingError::VerdictSource { detail: e.to_string() })
    };

    // Owner cap wired in (no spend on this lane, but the path is exercised).
    let cap = DailySpendCap::owner_default("2026-06-22");
    let fitness = LiveCoordinatorFitness::new(
        seed_head.clone(),
        pool,
        TerminalRewardAdapter::offline(),
        samples,
        hidden,
        source,
        cap,
    )?;

    let dimension = head_config.parameter_count();
    let optimizer = SepCmaEs::new(SepCmaEsConfig {
        dimension,
        population_size: 40,
        generations: 200,
        initial_sigma: 0.8,
        seed: 0x6014_6017,
    })?;
    let initial = seed_head.flatten_parameters()?;
    let outcome = optimizer.optimize(&fitness, &initial)?;

    let cap_after = fitness.cap_snapshot();
    Ok(LiveRunReport {
        lane: LiveRunLane::SimulatedNoSpend,
        initial_fitness: outcome.initial_fitness,
        best_fitness: outcome.best_fitness,
        improved: outcome.improved(),
        evaluations: outcome.evaluations,
        spent_msats: cap_after.spent_today_msats(),
        cap_msats: cap_after.cap_msats(),
        halted_on_cap: fitness.halted_on_cap(),
        day_key: cap_after.day_key().to_string(),
    })
}

fn main() -> ExitCode {
    let real = std::env::args().any(|a| a == "--real");

    println!("== Coordinator LIVE training driver (Khala M6, #6014 / EPIC #6017) ==");
    println!(
        "Owner HARD budget cap: {} sats/day ({} msats/day). Fails closed.\n",
        OWNER_DAILY_CAP_MSATS / 1_000,
        OWNER_DAILY_CAP_MSATS
    );

    // Always run the no-spend validation pass first.
    println!("-- Phase 1: no-spend simulated validation (proves loop + cap enforcement) --");
    let report = match run_validation() {
        Ok(report) => report,
        Err(error) => {
            eprintln!("validation failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("\nLiveRunReport (simulated, no-spend):");
    println!("  lane            : {:?}", report.lane);
    println!("  initial fitness : {:.4}", report.initial_fitness);
    println!("  best fitness    : {:.4}", report.best_fitness);
    println!("  improved        : {}", report.improved);
    println!("  evaluations     : {}", report.evaluations);
    println!("  spent (msats)   : {}  (must be 0 on this lane)", report.spent_msats);
    println!("  cap   (msats)   : {}", report.cap_msats);
    println!("  halted on cap   : {}", report.halted_on_cap);
    println!("  within cap      : {}", report.within_cap());

    // Hard gates on the validation pass.
    if report.spent_msats != 0 {
        eprintln!("\nFAIL: simulated lane must spend zero sats.");
        return ExitCode::FAILURE;
    }
    if !report.within_cap() {
        eprintln!("\nFAIL: cap invariant violated.");
        return ExitCode::FAILURE;
    }
    if !report.improved || (report.best_fitness - 1.0).abs() > 1e-6 {
        eprintln!(
            "\nFAIL: ES did not drive the real-backbone head to a verified routing (got {}).",
            report.best_fitness
        );
        return ExitCode::FAILURE;
    }
    println!("\nVALIDATION PASS: real frozen backbone -> head -> sep-CMA-ES reached verified");
    println!("routing (fitness 1.0) over the capability-filtered pool, zero sats spent, cap");
    println!("enforcement path exercised. This is a SMOKE, not a frontier ML result.");

    if !real {
        println!(
            "\n(Pass --real to attempt a bounded real run. Held by default; see below.)"
        );
        return ExitCode::SUCCESS;
    }

    // -- Phase 2: bounded real run -----------------------------------------
    println!("\n-- Phase 2: bounded REAL run (--real) --");
    println!("HELD. A real (sat-moving) run requires, and this binary will NOT fabricate:");
    println!("  1. A live EvalVerdictSource that dispatches each trajectory to the Pylon");
    println!("     network as a buy-mode eval job (reuse qwen_legal_pylon_dispatch path).");
    println!("  2. The Tassadar training.verification_classes.v1 verdict as the reward");
    println!("     (exact_trace_replay @ 1.0 for deterministic work), NOT a prompted judge.");
    println!("  3. A spend-enabled buy-mode campaign row in the openagents Worker so each");
    println!("     Pylon eval debits the SAME daily_cap_msats / spent_today_msats / day_key");
    println!("     ledger this binary's DailySpendCap mirrors. Clamp: 10,000 sats/day.");
    println!("  4. A frozen Qwen3-0.6B backbone for forward_with_hidden (swap the cs336 stub).");
    println!(
        "\nUntil (1)-(3) are reachable, launching here would either spend with no shared-cap\n\
         authority or fabricate verdicts. Both are refused. The cap is proven to fail closed\n\
         (see coordinator_live_training::tests::live_fitness_fails_closed_at_the_cap).\n\
         Total sats spent by this invocation: 0."
    );
    ExitCode::SUCCESS
}
