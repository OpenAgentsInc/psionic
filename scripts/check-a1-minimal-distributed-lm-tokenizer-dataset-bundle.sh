#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
bundle_path="${repo_root}/fixtures/psion/tokenized/a1_minimal_distributed_lm_tokenizer_dataset_bundle_v1.json"
corpus_path="${repo_root}/fixtures/training/a1_minimal_distributed_lm_corpus.txt"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/a1_minimal_distributed_lm_tokenizer_dataset.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

cargo run -q -p psionic-train --example a1_minimal_distributed_lm_tokenizer_dataset_bundle_fixture -- "${tmpdir}" >/dev/null

generated_bundle_path="${tmpdir}/fixtures/psion/tokenized/a1_minimal_distributed_lm_tokenizer_dataset_bundle_v1.json"
generated_corpus_path="${tmpdir}/fixtures/training/a1_minimal_distributed_lm_corpus.txt"

python3 - "${bundle_path}" "${generated_bundle_path}" "${corpus_path}" "${generated_corpus_path}" <<'PY'
import json
import sys
from pathlib import Path

bundle_path = Path(sys.argv[1])
generated_bundle_path = Path(sys.argv[2])
corpus_path = Path(sys.argv[3])
generated_corpus_path = Path(sys.argv[4])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

committed = json.loads(bundle_path.read_text(encoding="utf-8"))
generated = json.loads(generated_bundle_path.read_text(encoding="utf-8"))

if committed != generated:
    fail("A1 minimal distributed LM tokenizer/dataset bundle drifted from generator output")
if corpus_path.read_text(encoding="utf-8") != generated_corpus_path.read_text(encoding="utf-8"):
    fail("A1 minimal distributed LM corpus fixture drifted from generator output")

if committed["lane_id"] != "a1_minimal_distributed_lm_001":
    fail("bundle lost the A1 minimal distributed LM lane id")
if not committed["tokenizer_digest"].startswith("sha256:"):
    fail("tokenizer digest must use sha256:<hex> form")
if not committed["training_dataset_digest"].startswith("sha256:"):
    fail("training dataset digest must use sha256:<hex> form")
if not committed["validation_dataset_digest"].startswith("sha256:"):
    fail("validation dataset digest must use sha256:<hex> form")

tokenizer_digest = committed["tokenizer_digest"]
for shard in committed["training_shards"] + committed["validation_shards"]:
    if shard["tokenizer_digest"] != tokenizer_digest:
        fail(f"shard {shard['shard_id']} tokenizer digest drifted")
    if not shard["source_shard_digest"].startswith("sha256:"):
        fail(f"shard {shard['shard_id']} source shard digest must use sha256:<hex> form")
    if shard["token_count"] != len(shard["tokens"]):
        fail(f"shard {shard['shard_id']} token count does not match tokens")

validation_replays = [
    sample for sample in committed["replay_samples"]
    if sample["split_name"] == "validation"
]
if not validation_replays:
    fail("bundle must keep validation replay samples")
for sample in validation_replays:
    if not sample["round_trip_matches"] or sample["raw_text"] != sample["decoded_text"]:
        fail(f"validation replay {sample['sample_id']} does not round-trip")

summary = {
    "verdict": "verified",
    "lane_id": committed["lane_id"],
    "tokenizer_digest": committed["tokenizer_digest"],
    "training_dataset_digest": committed["training_dataset_digest"],
    "validation_dataset_digest": committed["validation_dataset_digest"],
    "bundle_digest": committed["bundle_digest"],
    "training_tokens": sum(shard["token_count"] for shard in committed["training_shards"]),
    "validation_tokens": sum(shard["token_count"] for shard in committed["validation_shards"]),
}
print(json.dumps(summary, indent=2))
PY
