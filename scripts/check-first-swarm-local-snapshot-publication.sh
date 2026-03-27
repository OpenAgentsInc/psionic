#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

fixture_root="${repo_root}/fixtures/swarm/publications"
fixture_report="${fixture_root}/first_swarm_local_snapshot_publication_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/first_swarm_local_snapshot_publication.XXXXXX")"
trap 'rm -rf "${tmpdir}"' EXIT

generated_root="${tmpdir}/generated"

cargo run -q -p psionic-mlx-workflows --bin first_swarm_local_snapshot_publication -- "${generated_root}" >/dev/null

python3 - <<'PY' "${fixture_root}" "${generated_root}" "${fixture_report}"
import hashlib
import json
import pathlib
import sys

fixture_root = pathlib.Path(sys.argv[1])
generated_root = pathlib.Path(sys.argv[2])
fixture_report_path = pathlib.Path(sys.argv[3])
generated_report_path = generated_root / "first_swarm_local_snapshot_publication_v1.json"


def fail(message: str) -> None:
    raise SystemExit(message)


def load_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_tree(root: pathlib.Path):
    entries = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries[rel] = digest
    return entries


if not fixture_report_path.is_file():
    fail("first swarm local snapshot publication check: missing retained report fixture")
if not generated_report_path.is_file():
    fail("first swarm local snapshot publication check: generator did not emit a report")

fixture_report = load_json(fixture_report_path)
generated_report = load_json(generated_report_path)

if fixture_report != generated_report:
    fail("first swarm local snapshot publication check: committed report drifted from generator output")

fixture_tree = file_tree(fixture_root)
generated_tree = file_tree(generated_root)
if fixture_tree != generated_tree:
    fail("first swarm local snapshot publication check: retained publication tree drifted from generator output")

if fixture_report["publish_id"] != "first-swarm-local-snapshot":
    fail("first swarm local snapshot publication check: publish_id drifted")
if fixture_report["repo_id"] != "openagents/swarm-local-open-adapter":
    fail("first swarm local snapshot publication check: repo_id drifted")
if fixture_report["target"] != "hugging_face_snapshot":
    fail("first swarm local snapshot publication check: target drifted")

snapshot_root = fixture_root / fixture_report["published_snapshot_root"]
if not snapshot_root.joinpath("publish_manifest.json").is_file():
    fail("first swarm local snapshot publication check: retained snapshot lost publish_manifest.json")
if not snapshot_root.joinpath("model.safetensors").is_file():
    fail("first swarm local snapshot publication check: retained snapshot lost model.safetensors")

print(
    json.dumps(
        {
            "report_digest": fixture_report["report_digest"],
            "publish_manifest_digest": fixture_report["publish_manifest_digest"],
            "file_count": len(fixture_tree),
            "snapshot_root": fixture_report["published_snapshot_root"],
        },
        indent=2,
    )
)
PY
