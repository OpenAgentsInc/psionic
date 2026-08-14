#!/usr/bin/env bash
set -euo pipefail

readonly PINNED_LLAMA_CPP_REVISION="9b05354ec6fb58b4e665e9a39ebc40285c015638"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 LLAMA_CPP_DIR MODEL_GGUF OUTPUT_DIR" >&2
  exit 2
fi

readonly LLAMA_CPP_DIR="$(realpath "$1")"
readonly MODEL_GGUF="$(realpath "$2")"
readonly OUTPUT_DIR="$(realpath -m "$3")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE="$SCRIPT_DIR/qwen38-llama-cpp-intermediate-trace.cpp"
readonly BUILD_DIR="$LLAMA_CPP_DIR/build"
readonly TARGET_DIR="$BUILD_DIR/examples/eval-callback"
readonly LINK_FILE="$TARGET_DIR/CMakeFiles/llama-eval-callback.dir/link.txt"

actual_revision="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
if [[ "$actual_revision" != "$PINNED_LLAMA_CPP_REVISION" ]]; then
  echo "refusing unpinned llama.cpp revision: $actual_revision" >&2
  exit 1
fi
if [[ -n "$(git -C "$LLAMA_CPP_DIR" status --short)" ]]; then
  echo "refusing dirty llama.cpp checkout: $LLAMA_CPP_DIR" >&2
  exit 1
fi

cmake --build "$BUILD_DIR" --target llama-eval-callback -j1
if [[ ! -f "$LINK_FILE" ]]; then
  echo "missing llama-eval-callback link command: $LINK_FILE" >&2
  exit 1
fi

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT
readonly OBJECT="$temp_dir/qwen38-llama-cpp-intermediate-trace.o"
readonly BINARY="$temp_dir/qwen38-llama-cpp-intermediate-trace"

link_command="$(<"$LINK_FILE")"
compiler="${link_command%% *}"
"$compiler" \
  -std=c++20 \
  -O3 \
  -DNDEBUG \
  -DGGML_USE_CPU \
  -DGGML_USE_CUDA \
  -I"$LLAMA_CPP_DIR/include" \
  -I"$LLAMA_CPP_DIR/ggml/include" \
  -c "$SOURCE" \
  -o "$OBJECT"

link_command="${link_command//\"CMakeFiles\/llama-eval-callback.dir\/eval-callback.cpp.o\"/\"$OBJECT\"}"
link_command="${link_command//-o ..\/..\/bin\/llama-eval-callback/-o \"$BINARY\"}"
(
  cd "$TARGET_DIR"
  eval "$link_command"
)

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
"$BINARY" "$MODEL_GGUF" "$OUTPUT_DIR"
printf 'artifact_byte_length\t%s\nartifact_sha256\t%s\n' \
  "$(stat -c '%s' "$MODEL_GGUF")" \
  "$(sha256sum "$MODEL_GGUF" | cut -d' ' -f1)" \
  >>"$OUTPUT_DIR/metadata.tsv"
