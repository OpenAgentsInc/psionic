#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixtures_dir="${repo_root}/fixtures/psion/instruct"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/psion_instruct_sft_lane.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

cargo run -q -p psionic-train --example psion_instruct_sft_lane_fixtures -- "${tmpdir}" >/dev/null

python3 - "${fixtures_dir}" "${tmpdir}/fixtures/psion/instruct" <<'PY'
import json
import sys
from pathlib import Path

committed_dir = Path(sys.argv[1])
generated_dir = Path(sys.argv[2])


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def load(directory: Path, name: str):
    path = directory / name
    if not path.exists():
        fail(f"missing fixture {path}")
    return path.read_text(encoding="utf-8")


names = [
    "psion_chat_template_v1.json",
    "psion_instruct_example_corpus_v1.jsonl",
    "psion_instruct_corpus_manifest_v1.json",
    "psion_instruct_generation_mask_fixture_v1.json",
    "psion_instruct_sft_lane_report_v1.json",
]
for name in names:
    committed = load(committed_dir, name)
    generated = load(generated_dir, name)
    if committed != generated:
        fail(f"instruct SFT lane fixture {name} drifted from generator output")

template = json.loads(load(committed_dir, "psion_chat_template_v1.json"))
manifest = json.loads(load(committed_dir, "psion_instruct_corpus_manifest_v1.json"))
masks = json.loads(load(committed_dir, "psion_instruct_generation_mask_fixture_v1.json"))
report = json.loads(load(committed_dir, "psion_instruct_sft_lane_report_v1.json"))

if template["schema_version"] != "psion.chat_template.v1":
    fail("chat template lost its schema version")
if not template["template_digest"].startswith("sha256:"):
    fail("chat template digest is not pinned")

if manifest["schema_version"] != "psion.instruct_corpus_manifest.v1":
    fail("corpus manifest lost its schema version")
if manifest["example_corpus"] is not True:
    fail("example corpus must stay labeled as example material")
if manifest["template_digest"] != template["template_digest"]:
    fail("corpus manifest is pinned to a different template digest")
record_ids = [row["record_id"] for row in manifest["records"]]
if len(record_ids) != len(set(record_ids)):
    fail("corpus manifest carries duplicate record ids")
for row in manifest["records"]:
    if not row["content_digest"].startswith("sha256:"):
        fail(f"record {row['record_id']} lost its content digest")
    if row["rights_posture"] != "repo_owned_example_text":
        fail(f"record {row['record_id']} lost its example rights posture")
    if row["trainable_token_count"] == 0:
        fail(f"record {row['record_id']} has no trainable tokens")

for rendered in masks:
    tokens = rendered["tokens"]
    mask = rendered["loss_mask"]
    if len(tokens) != len(mask):
        fail(f"mask length drifted for {rendered['conversation_id']}")
    if tokens[-1]["segment"] != "eos" or mask[-1] != 1:
        fail(f"{rendered['conversation_id']} lost its trainable eos token")
    for token, flag in zip(tokens, mask):
        trains = token["role"] == "assistant" and token["segment"] != "role_open"
        if bool(flag) != trains:
            fail(
                f"{rendered['conversation_id']} generation mask drifted at token "
                f"{token['text']!r}"
            )

if report["schema_version"] != "psion.instruct_sft_lane_report.v1":
    fail("lane report lost its schema version")
if report["lane_id"] != "psion_instruct_sft_v1":
    fail("lane report lost its lane id")
if report["corpus_manifest_digest"] != manifest["manifest_digest"]:
    fail("lane report is not pinned to the committed corpus manifest")
ratio = report["learning_rate"] / report["pretraining_reference_learning_rate"]
if abs(ratio - 0.1) > 1e-6:
    fail("instruct learning rate is no longer ten times below the pretraining reference")
if not report["loss_improved"]:
    fail("bounded smoke run no longer improves loss")
if report["resume_drill"]["resume_bit_exact"] is not True:
    fail("checkpoint/resume drill is no longer bit exact")
if report["resume_drill"]["post_resume_receipt_digests_match"] is not True:
    fail("post-resume receipts no longer match the uninterrupted run")
if len(report["learning_rate_schedule"]) != report["completed_steps"]:
    fail("learning-rate schedule rows do not cover the completed steps")
if "no real training-run claim" not in report["claim_boundary"]:
    fail("lane report lost its claim boundary")

summary = {
    "verdict": "verified",
    "lane_id": report["lane_id"],
    "template_digest": template["template_digest"],
    "corpus_manifest_digest": manifest["manifest_digest"],
    "report_digest": report["report_digest"],
}
print(json.dumps(summary, indent=2))
PY
