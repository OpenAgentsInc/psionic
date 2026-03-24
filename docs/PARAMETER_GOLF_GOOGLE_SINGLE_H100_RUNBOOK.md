# Parameter Golf Google Single-H100 Runbook

> Status: canonical bounded Google operator runbook for the Parameter Golf
> single-H100 lane, written on 2026-03-23 after the repo-owned PGOLF launch
> profile, immutable input-package contract, and local rehearsal script landed.

This runbook defines the current honest Google posture for the Rust-only
Parameter Golf single-H100 baseline lane.

It is narrower than a completed Google run, narrower than the later `8xH100`
RunPod lane, and narrower than a record-track claim.

## Canonical Artifacts

- launch profile authority:
  `fixtures/parameter_golf/google/parameter_golf_google_single_node_launch_profiles_v1.json`
- quota and cost guardrail:
  `fixtures/parameter_golf/google/parameter_golf_google_billing_guardrails_v1.json`
- local operator preflight policy:
  `fixtures/parameter_golf/google/parameter_golf_google_operator_preflight_policy_v1.json`
- immutable input-package policy:
  `fixtures/parameter_golf/google/parameter_golf_google_input_package_policy_v1.json`
- committed input contract:
  `fixtures/parameter_golf/google/parameter_golf_google_input_contract_v1.json`
- committed input package manifest:
  `fixtures/parameter_golf/google/parameter_golf_google_input_package_manifest_v1.json`
- committed input package descriptor:
  `fixtures/parameter_golf/google/parameter_golf_google_input_package_descriptor_v1.json`
- committed input package archive:
  `fixtures/parameter_golf/google/parameter_golf_google_input_package_v1.tar.gz`
- first real bucket-upload receipt:
  `fixtures/parameter_golf/reports/parameter_golf_google_input_package_upload_v1.json`
- public-cache materializer:
  `scripts/parameter-golf-materialize-public-cache.sh`
- local package builder:
  `scripts/parameter-golf-google-package-inputs.sh`
- quota preflight wrapper:
  `scripts/parameter-golf-google-quota-preflight.sh`
- operator preflight wrapper:
  `scripts/parameter-golf-google-operator-preflight.sh`
- launch wrapper:
  `scripts/parameter-golf-google-launch-single-node.sh`
- local rehearsal:
  `scripts/check-parameter-golf-google-single-h100-lane.sh`
- committed local rehearsal report:
  `fixtures/parameter_golf/reports/parameter_golf_google_single_h100_operator_rehearsal.json`

## Profile Contract

The committed Google lane is:

- profile id: `a3_h100_single_node_parameter_golf`
- machine type: `a3-highgpu-1g`
- accelerator type: `nvidia-h100-80gb`
- accelerator count: `1`
- expected execution backend: `cuda`
- zone fallback order: `us-central1-a`, `us-central1-b`, `us-central1-c`
- declared run cost ceiling: `150.0` USD

Google's current A3 High documentation requires `a3-highgpu-1g` instances to
use Spot or Flex-start provisioning. The committed profile therefore keeps the
non-standard provisioning model explicit:

- provisioning model: `FLEX_START`

The launch manifest is also explicit about the Rust-owned trainer command:

- pre-training command:
  materialize the public Parameter Golf cache, validate it with the committed
  bring-up checker, then build `parameter_golf_single_h100_train` in `--release`
- training command:
  run `parameter_golf_single_h100_train` directly against the materialized
  `fineweb10B_sp1024` cache and `fineweb_1024_bpe.model`

## Immutable Input Package Contract

The committed package does not pretend to embed the full challenge cache.

Instead, it binds:

- dataset identity:
  `dataset://parameter-golf/fineweb-sp1024 @ 2026.03.18`
- dataset manifest digest from the committed single-H100 bring-up report
- tokenizer file SHA-256 from the committed single-H100 bring-up report
- fixed validation identity:
  `fineweb_val_* fixed first-50k-doc validation split`
- the exact public `parameter-golf` repo revision used for the cache bootstrap

The package archive remains small because it carries the immutable contract, not
the raw training shards themselves. The Google pre-training command uses that
contract to clone the public `parameter-golf` repo and materialize the public
challenge cache onto the VM before the Rust trainer runs.

The first real bucket materialization for this contract now exists in:

- `gs://openagentsgemini-psion-train-us-central1/manifests/parameter_golf_google_input_package_v1.json`
- `gs://openagentsgemini-psion-train-us-central1/runs/staging/input_packages/parameter_golf_google_input_package_v1.tar.gz`

The committed receipt that records the observed object generations, retention,
and local artifact digests is:

- `fixtures/parameter_golf/reports/parameter_golf_google_input_package_upload_v1.json`

## Local Rehearsal

The repo-owned local rehearsal is:

```bash
bash scripts/check-parameter-golf-google-single-h100-lane.sh \
  --report /tmp/parameter_golf_google_single_h100_operator_rehearsal.json
```

That rehearsal uses mocked `gcloud` and `bq` commands to verify two narrow
things:

- the local operator preflight accepts the PGOLF H100 profile under a valid
  authority context
- the manifest-only launcher emits a launch manifest whose machine, input
  package, and trainer command are all bound to the committed PGOLF lane

The committed first rehearsal report is:

- `fixtures/parameter_golf/reports/parameter_golf_google_single_h100_operator_rehearsal.json`

## Honest Boundary

This runbook does not claim:

- a successful live Google single-H100 run
- final `val_loss`, `val_bpb`, or compressed-model bytes from Google
- `8xH100` distributed closure
- record-track readiness

The immutable input package is now materialized to the real bucket authority.
Live Google execution is still blocked until the project has usable H100 quota
and one real launch is audited.
