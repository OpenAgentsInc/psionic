# Mesh Lane Service Mode

> Status: implemented on 2026-04-02 as the published install and service-mode
> entrypoint for durable inference-mesh lanes.

`psionic-mesh-lane` is the supported operator entrypoint for installing one
durable pooled-inference lane on one machine.

It does four things that the lower-level server binaries do not:

1. materializes one stable on-disk root for config, model paths, logs, node
   identity, and durable network state
2. writes platform service artifacts for macOS `launchd` and Linux `systemd`
3. starts the normal OpenAI-compatible Psionic server above file-backed mesh
   identity and join state
4. preserves node identity and trusted membership state across restart and
   binary upgrades

The current published service mode is intentionally bounded:

- shared-admission join bundles are supported
- signed-introduction bootstrap is not published here yet
- relative model paths resolve inside the mesh-lane root

## Published Binary

Build the supported binary:

```bash
cargo build -p psionic-serve --bin psionic-mesh-lane --release
```

The binary exposes three commands:

```bash
./target/release/psionic-mesh-lane install --root <dir> -m <model> ...
./target/release/psionic-mesh-lane run --root <dir>
./target/release/psionic-mesh-lane export-join-bundle --root <dir> --out <path>
```

## Durable Root Layout

One installed lane root has this layout:

- `bin/`
  - generated wrapper script used by `launchd` and `systemd`
- `config/mesh-lane.json`
  - durable operator config for serving, transport, and join material
- `state/node.identity.json`
  - file-backed Psionic node identity
- `state/network-state.json`
  - durable join state including last imported bundle and last joined mesh
    preference
- `logs/stdout.log`
- `logs/stderr.log`
- `models/`
  - relative `-m` or `--model` paths land here
- `run/`
- `service/`
  - generated `systemd` unit and `launchd` plist

## Install One Host

Example single-host install:

```bash
./target/release/psionic-mesh-lane install \
  --root ~/psionic/mesh-lanes/gemma4-e4b-a \
  -m gemma4-e4b.gguf \
  --backend cuda \
  --host 0.0.0.0 \
  --port 8080 \
  --mesh-bind 0.0.0.0:47470 \
  --service-name gemma4-e4b-a \
  --namespace gemma4-e4b-home \
  --node-role mixed
```

That command writes the durable root, generates a shared admission token if
none was supplied, and emits:

- `bin/gemma4-e4b-a.sh`
- `service/gemma4-e4b-a.service`
- `service/com.openagents.psionic.gemma4-e4b-a.plist`

Start the installed lane:

```bash
launchctl bootstrap gui/$(id -u) \
  ~/psionic/mesh-lanes/gemma4-e4b-a/service/com.openagents.psionic.gemma4-e4b-a.plist
```

```bash
systemctl --user enable --now \
  ~/psionic/mesh-lanes/gemma4-e4b-a/service/gemma4-e4b-a.service
```

Then verify it:

```bash
curl -s http://127.0.0.1:8080/psionic/management/status | jq
open http://127.0.0.1:8080/psionic/management/console
```

## Join A Second Host

Export a shared-admission join bundle from the first machine:

```bash
./target/release/psionic-mesh-lane export-join-bundle \
  --root ~/psionic/mesh-lanes/gemma4-e4b-a \
  --out /tmp/gemma4-e4b-home.join.json \
  --mesh-label gemma4-e4b-home \
  --advertise 10.0.0.10:47470
```

Install the second machine against that join bundle:

```bash
./target/release/psionic-mesh-lane install \
  --root ~/psionic/mesh-lanes/gemma4-e4b-b \
  -m gemma4-e4b.gguf \
  --backend cuda \
  --mesh-bind 0.0.0.0:47470 \
  --service-name gemma4-e4b-b \
  --join-bundle /tmp/gemma4-e4b-home.join.json \
  --node-role mixed
```

On first `run`, the lane imports the configured bundle into durable network
state and publishes the resulting join posture on
`/psionic/management/status`.

## Upgrade And Restart Contract

Reinstalling against the same `--root` is the supported upgrade path:

```bash
./target/release/psionic-mesh-lane install \
  --root ~/psionic/mesh-lanes/gemma4-e4b-a \
  -m gemma4-e4b.gguf \
  --backend cuda
```

That refreshes config and service artifacts without rotating the lane
identity.

The durable contract is:

- `state/node.identity.json` is reused, so the node comes back with the same
  `node_id`
- `state/network-state.json` is reused, so imported join material and last
  joined mesh preference survive restart
- node epoch advances on restart, which keeps session generation explicit
- changing `namespace` or `admission_token` is refused once durable identity or
  network-state files exist

If you need a different mesh identity, use a different `--root`.

## Service Notes

- wrapper scripts append to `logs/stdout.log` and `logs/stderr.log`
- service units restart automatically
- the service-mode root is the source of truth for future upgrades and
  restarts, not ad hoc shell flags
- management join state is populated from durable Psionic network state before
  the HTTP server starts
