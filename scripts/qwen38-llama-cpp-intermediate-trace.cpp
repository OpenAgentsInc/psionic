#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr const char *kSchemaVersion = "qwen38_llama_cpp_recurrent_trace_v1";
constexpr const char *kLlamaCppRevision =
    "9b05354ec6fb58b4e665e9a39ebc40285c015638";

constexpr std::array<const char *, 14> kStages = {
    "attn_norm",       "linear_attn_qkv_mixed",
    "conv_output_raw", "conv_output_silu",
    "a_softplus",      "gate",
    "beta_sigmoid",    "q_conv_predelta",
    "k_conv_predelta", "v_conv_predelta",
    "new_state",       "attn_output",
    "final_output",    "linear_attn_out",
};

struct Capture {
  std::array<int64_t, 4> shape{};
  std::vector<float> values;
};

struct CallbackData {
  std::string phase;
  std::map<std::string, Capture> captures;
  std::string error;
};

bool host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t *>(&value) == 1;
}

void discard_llama_log(ggml_log_level, const char *, void *) {}

const char *matching_stage(const char *tensor_name) {
  for (const char *stage : kStages) {
    const std::string expected = std::string(stage) + "-0";
    if (expected == tensor_name) {
      return stage;
    }
  }
  return nullptr;
}

bool trace_callback(ggml_tensor *tensor, bool ask, void *user_data) {
  auto *data = static_cast<CallbackData *>(user_data);
  const char *stage = matching_stage(tensor->name);
  if (ask) {
    return stage != nullptr;
  }
  if (stage == nullptr || !data->error.empty()) {
    return data->error.empty();
  }
  if (tensor->type != GGML_TYPE_F32) {
    data->error = std::string("selected tensor is not F32: ") + tensor->name;
    return false;
  }

  const std::string key = data->phase + "/" + stage;
  if (data->captures.contains(key)) {
    data->error = "duplicate selected tensor capture: " + key;
    return false;
  }

  if (!ggml_backend_buffer_is_host(tensor->buffer)) {
    data->error =
        std::string("selected tensor is not host-resident: ") + tensor->name;
    return false;
  }
  const uint8_t *base = static_cast<const uint8_t *>(tensor->data);

  Capture capture;
  for (size_t index = 0; index < capture.shape.size(); ++index) {
    capture.shape[index] = tensor->ne[index];
  }
  capture.values.reserve(static_cast<size_t>(ggml_nelements(tensor)));
  for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
    for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
      for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
          const size_t offset = static_cast<size_t>(i0) * tensor->nb[0] +
                                static_cast<size_t>(i1) * tensor->nb[1] +
                                static_cast<size_t>(i2) * tensor->nb[2] +
                                static_cast<size_t>(i3) * tensor->nb[3];
          float value = 0.0F;
          std::memcpy(&value, base + offset, sizeof(value));
          capture.values.push_back(value);
        }
      }
    }
  }
  data->captures.emplace(key, std::move(capture));
  return true;
}

bool decode(llama_context *context, std::vector<llama_token> &tokens,
            const char *phase) {
  const int32_t status = llama_decode(
      context,
      llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size())));
  if (status != 0) {
    std::cerr << "llama_decode failed for " << phase << " with status "
              << status << '\n';
    return false;
  }
  return true;
}

bool write_outputs(const std::filesystem::path &output_dir,
                   const CallbackData &data) {
  std::filesystem::create_directories(output_dir);
  std::ofstream metadata(output_dir / "metadata.tsv", std::ios::trunc);
  std::ofstream manifest(output_dir / "manifest.tsv", std::ios::trunc);
  if (!metadata || !manifest) {
    std::cerr << "failed to create trace manifests under " << output_dir
              << '\n';
    return false;
  }
  metadata << "key\tvalue\n"
           << "schema_version\t" << kSchemaVersion << '\n'
           << "llama_cpp_revision\t" << kLlamaCppRevision << '\n'
           << "prefill_tokens\t9419,11\n"
           << "decode_tokens\t353\n"
           << "layer_index\t0\n"
           << "byte_order\tlittle_endian\n";
  manifest << "phase\tstage\tne0\tne1\tne2\tne3\telement_count\tfile\n";

  for (const char *phase : {"prefill", "decode"}) {
    for (const char *stage : kStages) {
      const std::string key = std::string(phase) + "/" + stage;
      const auto capture = data.captures.find(key);
      if (capture == data.captures.end()) {
        std::cerr << "missing selected tensor capture: " << key << '\n';
        return false;
      }
      const std::string filename = std::string(phase) + "." + stage + ".f32le";
      std::ofstream output(output_dir / filename,
                           std::ios::binary | std::ios::trunc);
      if (!output) {
        std::cerr << "failed to create " << filename << '\n';
        return false;
      }
      output.write(
          reinterpret_cast<const char *>(capture->second.values.data()),
          static_cast<std::streamsize>(capture->second.values.size() *
                                       sizeof(float)));
      if (!output) {
        std::cerr << "failed to write " << filename << '\n';
        return false;
      }
      const auto &shape = capture->second.shape;
      manifest << phase << '\t' << stage << '\t' << shape[0] << '\t' << shape[1]
               << '\t' << shape[2] << '\t' << shape[3] << '\t'
               << capture->second.values.size() << '\t' << filename << '\n';
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " MODEL_GGUF OUTPUT_DIR\n";
    return 2;
  }
  if (!host_is_little_endian()) {
    std::cerr << "the retained trace writer requires a little-endian host\n";
    return 2;
  }

  CallbackData callback_data;
  llama_log_set(discard_llama_log, nullptr);
  llama_backend_init();

  std::array<ggml_backend_dev_t, 2> devices = {
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
  if (devices[0] == nullptr) {
    std::cerr << "failed to resolve the GGML CPU device\n";
    llama_backend_free();
    return 1;
  }
  llama_model_params model_params = llama_model_default_params();
  model_params.devices = devices.data();
  model_params.n_gpu_layers = 0;
  model_params.load_mtp = false;
  llama_model *model = llama_model_load_from_file(argv[1], model_params);
  if (model == nullptr) {
    std::cerr << "failed to load model " << argv[1] << '\n';
    llama_backend_free();
    return 1;
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = 128;
  context_params.n_batch = 2;
  context_params.n_ubatch = 2;
  context_params.n_threads = 16;
  context_params.n_threads_batch = 16;
  context_params.cb_eval = trace_callback;
  context_params.cb_eval_user_data = &callback_data;
  context_params.offload_kqv = false;
  context_params.op_offload = false;
  context_params.no_perf = true;
  llama_context *context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    std::cerr << "failed to create llama context\n";
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  std::vector<llama_token> prefill = {9419, 11};
  std::vector<llama_token> decode_token = {353};
  callback_data.phase = "prefill";
  bool ok = decode(context, prefill, "prefill");
  if (ok && callback_data.error.empty()) {
    callback_data.phase = "decode";
    ok = decode(context, decode_token, "decode");
  }
  if (!callback_data.error.empty()) {
    std::cerr << callback_data.error << '\n';
    ok = false;
  }
  if (ok) {
    ok = write_outputs(argv[2], callback_data);
  }

  llama_free(context);
  llama_model_free(model);
  llama_backend_free();
  return ok ? 0 : 1;
}
