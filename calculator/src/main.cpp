#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kOneE9 = 1000000000.0;
constexpr double kOneE12 = 1000000000000.0;

struct Hardware {
  std::string name;
  int cards_per_node{};
  double fp8_matrix{};
  double bf16_matrix{};
  double fp8_vector{};
  double bf16_vector{};
  double fp8_sparse{};
  double bf16_sparse{};
  double sfu{};
  double sfu_eff{};
  double hbm_bw{};
  double mem_eff{};
  double d2d_bw{};
  double nvlink_bw{};
  double rdma_bw{};
  double pcie_bw{};
  double vram{};
  double comp_eff{};
  int dies_per_card{};
};

struct Model {
  std::string name;
  double total_params_b{};
  double active_params_b{};
  double layers{};
  double hidden{};
  double q_heads{};
  double kv_heads{};
  bool use_mla{};
  double mla_kv_factor{};
  bool moe{};
  double experts_total{};
  double experts_active{};
  double default_bpp{};
};

struct ModelInput {
  std::string hardware = "H20";
  std::string model = "Llama-8B";
  std::string stage = "Decode";
  double bytes_per_param = 1.0;
  long batch = 1;
  long context = 4096;
  long tp = 1;
  long pp = 1;
  long dp = 1;
  long ep = 1;
  long nodes = 1;
  long micro_batch = 1;  // prefill PP: micro-batch SIZE (count = ceil(B*S / size))
  std::string pp_placement = "Intra-node";
};

struct ModelOutput {
  bool valid_config{true};
  std::string config_validity{"OK"};
  std::string warning;
  std::string bottleneck;
  std::string bottleneck_detail;
  double e4_tf_mat{};
  double e5_bw_hbm{};
  double e6_mem_eff{};
  double e7_comp_eff{};
  double e8_vram{};
  double e9_total_params{};
  double e10_active_params{};
  double e11_layers{};
  double e12_hidden{};
  double e13_q_heads{};
  double e14_kv_heads{};
  bool e17_is_moe{};
  double e18_weight_total_gb{};
  double e19_weight_active_gb{};
  double e20_kv_gb_per_token{};
  double e21_bytes_per_token_gb{};
  double e22_flops_per_token{};
  double e23_pf_comp{};
  double e24_pf_mem{};
  double e25_tps_comp_raw{};
  double e26_tps_mem{};
  double e27_tps_prefill_raw{};
  double e29_bw_tp{};
  double e30_bw_pp{};
  double e31_bw_ep{};
  double e32_pp_eff{};
  double e33_pp_bubble_pct{};
  double e34_tp_comm_overhead_pct{};
  double e35_tps_comp_before_comm{};
  double e36_tps_comp_after_comm{};
  double b23_total_cards{};
  double b24_available_cards{};
  double b25_recommended_cards{};
  double b27_per_gpu_mem_gb{};
  double b28_vram_gb{};
  std::string b29_per_gpu_fit;
  std::string b30_cluster_check;
  double b35_tps_roof{};
  double b37_tps_effective{};
  double b38_tps_per_gpu{};
  double b39_tps_cluster{};
  double b40_latency_ms{};
};

struct OperatorDef {
  std::string name;
  std::string category;
  std::string compute_unit;
  double flops_per_elem{};
  double read_factor{};
  double write_factor{};
  bool use_sparse{};
};

struct OperatorInput {
  std::string hardware = "H20";
  std::string precision = "FP8";
  std::string operator_name = "GEMM_QK";
  double sparsity_pct = 0;
  double b = 1;
  double s = 4096;
  double h = 4096;
  double n = 32;
  double d = 128;
  double i = 11008;
  double v = 128256;
  double l = 32;
  double e = 256;
  double k = 8;
  double w = 1024;
  double tp = 1;
  double dp = 1;
};

struct OperatorOutput {
  std::string category;
  std::string compute_unit;
  double flops_per_elem{};
  double read_factor{};
  double write_factor{};
  double total_elements{};
  double total_flops{};
  double total_bytes{};
  double ai{};
  double tflops_used{};
  double hbm_bw{};
  double parallel_factor{};
  double compute_bound{};
  double mem_bound{};
  double roofline_invoc_s{};
  std::string bottleneck;
  double efficiency{};
  double effective_invoc_s{};
  double bw_util_gbps{};
  double bw_pct{};
  double per_gpu_invoc_s{};
  double cluster_invoc_s{};
  double batch_pass_ms{};
  double invoc_duration_ns{};
  double duration_compute_ns{};
  double duration_comm_ns{};
  double duration_total_ns{};
  double comm_fraction_pct{};
};

std::map<std::string, Hardware> BuildHardwareDb() {
  return {
      {"H20", {"H20", 8, 148, 74, 30, 15, 296, 148, 7.5, 0.85, 4000, 0.35, 0, 900, 400, 128, 96, 0.45, 1}},
      {"B200", {"B200", 8, 2250, 1125, 450, 225, 4500, 2250, 112.5, 0.85, 8000, 0.35, 10000, 1800, 800, 128, 192, 0.45, 2}},
      {"GB300", {"GB300", 8, 2500, 1250, 500, 250, 5000, 2500, 125, 0.85, 8500, 0.35, 10000, 1800, 800, 128, 192, 0.45, 2}},
      {"Custom", {"Custom", 8, 307, 153, 38, 19, 307, 153, 7.5, 0.85, 4000, 0.7, 3400, 800, 100, 128, 144, 0.45, 2}},
  };
}

std::map<std::string, Model> BuildModelDb() {
  return {
      {"Llama-8B", {"Llama-8B", 8, 8, 32, 4096, 32, 8, false, 1, false, 0, 0, 1}},
      {"Llama-70B", {"Llama-70B", 70, 70, 80, 8192, 64, 8, false, 1, false, 0, 0, 1}},
      {"Llama-405B", {"Llama-405B", 405, 405, 128, 16384, 128, 8, false, 1, false, 0, 0, 1}},
      {"DeepSeek-V3", {"DeepSeek-V3", 671, 37, 61, 7168, 128, 16, true, 0.25, true, 256, 8, 1}},
      {"Qwen2.5-72B", {"Qwen2.5-72B", 72, 72, 80, 8192, 64, 8, false, 1, false, 0, 0, 1}},
      {"Mixtral-8x7B", {"Mixtral-8x7B", 47, 13, 32, 4096, 32, 8, false, 1, true, 8, 2, 1}},
  };
}

std::vector<OperatorDef> BuildOperatorDb() {
  return {
      {"GEMM_QK", "Compute-Intensive", "Matrix", 2, 2, 1, false},
      {"GEMM_AV", "Compute-Intensive", "Matrix", 2, 2, 1, false},
      {"GEMM_FFN_Up", "Compute-Intensive", "Matrix", 2, 2, 1, false},
      {"GEMM_FFN_Down", "Compute-Intensive", "Matrix", 2, 2, 1, false},
      {"GEMM_Gate", "Compute-Intensive", "Matrix", 2, 2, 1, false},
      {"GEMM_Output", "Compute-Intensive", "Matrix", 2, 2, 1, false},
      {"SiLU", "Memory-Bound", "SFU", 4, 1, 1, false},
      {"GELU", "Memory-Bound", "SFU", 8, 1, 1, false},
      {"ReLU", "Memory-Bound", "Vector", 1, 1, 1, false},
      {"RMSNorm", "Memory-Bound", "SFU", 4, 1, 1, false},
      {"LayerNorm", "Memory-Bound", "SFU", 6, 1, 1, false},
      {"Softmax", "Memory-Bound", "SFU", 5, 1, 1, false},
      {"RoPE", "Memory-Bound", "Vector", 3, 2, 1, false},
      {"KV_Cache_Read", "Memory-Bound", "Vector", 0, 1, 0, false},
      {"KV_Cache_Write", "Memory-Bound", "Vector", 0, 0, 1, false},
      {"MoE_Router", "Sparse", "SFU", 64, 16, 8, true},
      {"MoE_Sparse_GEMM", "Sparse", "Matrix_Sparse", 2, 2, 1, true},
      {"Sparse_Attention", "Sparse", "Matrix_Sparse", 2, 2, 1, true},
  };
}

double CeilDiv(double a, double b) {
  if (b <= 0) return 1;
  return std::ceil(a / b);
}

double SelectInterconnectBW(long parallel_degree, long cards_per_node, long dies_per_card,
                            double bw_d2d, double bw_nvlink, double bw_rdma, double bw_pcie) {
  if (parallel_degree <= dies_per_card && bw_d2d > 0) return bw_d2d;
  if (parallel_degree <= cards_per_node) return bw_nvlink > 0 ? bw_nvlink : bw_pcie;
  return bw_rdma > 0 ? bw_rdma : bw_pcie;
}

std::string CsvQuote(const std::string& in) {
  if (in.find_first_of(",\"") == std::string::npos) return in;
  std::string out = "\"";
  for (char c : in) out += (c == '"' ? "\"\"" : std::string(1, c));
  out += "\"";
  return out;
}

double CalculateOperatorElements(const std::string& op, const OperatorInput& in) {
  if (op == "GEMM_QK" || op == "GEMM_AV") return in.b * in.n * in.s * in.s * in.d;
  if (op == "GEMM_FFN_Up" || op == "GEMM_FFN_Down" || op == "GEMM_Gate") return in.b * in.s * in.h * in.i;
  if (op == "GEMM_Output") return in.b * in.s * in.h * in.v;
  if (op == "SiLU" || op == "GELU") return in.b * in.s * in.i;
  if (op == "ReLU" || op == "RMSNorm" || op == "LayerNorm") return in.b * in.s * in.h;
  if (op == "Softmax") return in.b * in.n * in.s * in.s;
  if (op == "RoPE") return in.b * in.s * in.n * in.d;
  if (op == "KV_Cache_Read" || op == "KV_Cache_Write") return in.b * in.l * in.s * in.d;
  if (op == "MoE_Router") return in.b * in.s * in.e;
  if (op == "MoE_Sparse_GEMM") return in.b * in.s * in.k * in.d * in.i;
  if (op == "Sparse_Attention") return in.b * in.n * in.s * in.w * in.d;
  return 0;
}

double CalculateOperatorBytes(const std::string& op, const OperatorInput& in,
                              double bpe, double read_factor, double write_factor,
                              double sparsity_factor) {
  if (op == "GEMM_QK") return (in.b * in.n * in.s * in.d + in.b * in.n * in.d * in.s + in.b * in.n * in.s * in.s) * bpe;
  if (op == "GEMM_AV") return (in.b * in.n * in.s * in.s + in.b * in.n * in.s * in.d + in.b * in.n * in.s * in.d) * bpe;
  if (op == "GEMM_FFN_Up" || op == "GEMM_FFN_Down" || op == "GEMM_Gate") return (in.h * in.i + in.b * in.s * in.h + in.b * in.s * in.i) * bpe;
  if (op == "GEMM_Output") return (in.h * in.v + in.b * in.s * in.h + in.b * in.s * in.v) * bpe;
  if (op == "MoE_Sparse_GEMM") {
    const double w = in.k * in.d * in.i * bpe * sparsity_factor;
    const double x = in.b * in.s * in.d * bpe;
    const double y = in.b * in.s * in.k * in.i * bpe;
    return w + x + y;
  }
  if (op == "Sparse_Attention") return (in.b * in.n * in.s * in.d + in.b * in.n * in.w * in.d + in.b * in.n * in.s * in.w) * bpe;
  const double elems = CalculateOperatorElements(op, in);
  return elems * bpe * (read_factor + write_factor);
}

ModelOutput CalculateModel(const ModelInput& input, const std::map<std::string, Hardware>& hw_db,
                           const std::map<std::string, Model>& model_db) {
  ModelOutput out;
  const auto hw_it = hw_db.find(input.hardware);
  const auto m_it = model_db.find(input.model);
  if (hw_it == hw_db.end() || m_it == model_db.end()) {
    out.valid_config = false;
    out.config_validity = "INVALID: unknown hardware/model";
    return out;
  }
  const Hardware& hw = hw_it->second;
  const Model& mdl = m_it->second;
  const long batch = std::max(1L, input.batch);
  const long ctx = std::max(1L, input.context);
  const long tp = std::max(1L, input.tp);
  const long pp = std::max(1L, input.pp);
  const long dp = std::max(1L, input.dp);
  const long ep = std::max(1L, input.ep);
  const long nodes = std::max(1L, input.nodes);
  const long micro_batch = std::max(1L, input.micro_batch);
  const long dies = std::max(1, hw.dies_per_card);

  out.e4_tf_mat = hw.fp8_matrix;
  out.e5_bw_hbm = hw.hbm_bw;
  out.e6_mem_eff = hw.mem_eff;
  out.e7_comp_eff = hw.comp_eff;
  out.e8_vram = hw.vram;
  out.e9_total_params = mdl.total_params_b;
  out.e10_active_params = mdl.active_params_b;
  out.e11_layers = mdl.layers;
  out.e12_hidden = mdl.hidden;
  out.e13_q_heads = mdl.q_heads;
  out.e14_kv_heads = mdl.kv_heads;
  out.e17_is_moe = mdl.moe;
  out.e18_weight_total_gb = mdl.total_params_b * input.bytes_per_param;
  out.e19_weight_active_gb = mdl.active_params_b * input.bytes_per_param;
  out.e20_kv_gb_per_token = mdl.use_mla
                                ? (2.0 * mdl.layers * mdl.hidden * mdl.mla_kv_factor * input.bytes_per_param / kOneE9)
                                : (2.0 * mdl.layers * (mdl.hidden / std::max(1.0, mdl.q_heads)) * mdl.kv_heads * input.bytes_per_param / kOneE9);
  // Decode traffic: PP does not divide this. A token still walks every stage, so
  // the replica still reads all weights and all KV. PP only shards capacity
  // (see b27) and cuts compute via pp_eff = 1/PP.
  out.e21_bytes_per_token_gb = std::max(
      1e-12, out.e19_weight_active_gb / (std::max(1L, tp) * std::max(1L, batch)) +
                 out.e20_kv_gb_per_token * ctx / std::max(1L, tp));
  out.e22_flops_per_token = 2.0 * out.e10_active_params * kOneE9;
  out.e23_pf_comp = mdl.moe ? tp * ep : tp;
  out.e24_pf_mem = 1.0;
  out.e25_tps_comp_raw = (out.e22_flops_per_token > 0)
                             ? (out.e4_tf_mat * kOneE12 * out.e23_pf_comp / out.e22_flops_per_token / dies)
                             : 0.0;
  out.e26_tps_mem = (out.e21_bytes_per_token_gb > 0)
                        ? (out.e5_bw_hbm / dies * out.e24_pf_mem / out.e21_bytes_per_token_gb)
                        : 0.0;
  out.e27_tps_prefill_raw = out.e25_tps_comp_raw;

  out.b23_total_cards = static_cast<double>(tp) * pp * dp * ep;
  out.b24_available_cards = static_cast<double>(nodes) * hw.cards_per_node;
  const double min_cards = std::ceil((out.e18_weight_total_gb + out.e20_kv_gb_per_token * ctx * batch) / (std::max(1.0, out.e8_vram) * 0.8));
  const double rec_log = std::ceil(std::log2(std::max(1.0, min_cards)));
  out.b25_recommended_cards = std::max(1.0, std::pow(2.0, rec_log));
  out.b27_per_gpu_mem_gb = out.e18_weight_total_gb / (std::max(1L, tp) * std::max(1L, pp) * std::max(1L, ep)) +
                           out.e20_kv_gb_per_token * ctx * batch / (std::max(1L, tp) * std::max(1L, pp));
  out.b28_vram_gb = out.e8_vram;
  if (out.b27_per_gpu_mem_gb <= out.b28_vram_gb * 0.8) out.b29_per_gpu_fit = "OK";
  else if (out.b27_per_gpu_mem_gb <= out.b28_vram_gb * 2.0) out.b29_per_gpu_fit = "RISK";
  else out.b29_per_gpu_fit = "NOT_OK";
  out.b30_cluster_check = (out.b24_available_cards >= out.b25_recommended_cards) ? "OK" : "NOT_OK";
  if (!mdl.moe && ep > 1) {
    out.config_validity = "INVALID: EP>1 on non-MoE model";
  } else if (out.b23_total_cards > out.b24_available_cards) {
    std::ostringstream oss;
    oss << "INVALID: need " << static_cast<long>(out.b23_total_cards) << " cards but only "
        << static_cast<long>(out.b24_available_cards) << " available";
    out.config_validity = oss.str();
  } else if (out.b30_cluster_check == "NOT_OK") {
    std::ostringstream oss;
    oss << "WARNING: available cards below recommended (" << static_cast<long>(out.b25_recommended_cards) << ")";
    out.config_validity = oss.str();
  } else if (out.b29_per_gpu_fit == "NOT_OK") {
    out.config_validity = "WARNING: per-GPU mem exceeds 2x VRAM (impossible config)";
  } else if (out.b29_per_gpu_fit == "RISK") {
    out.config_validity = "WARNING: per-GPU mem exceeds 80% VRAM";
  } else {
    out.config_validity = "OK";
  }
  out.valid_config = out.config_validity.rfind("INVALID", 0) != 0;
  if (tp > 8 && out.e10_active_params < 70) out.warning = "Possible comm-bound: TP>8 on small model";
  else if (ep > 8 && mdl.moe) out.warning = "Possible comm-bound: EP>8 on MoE";
  else if (out.b23_total_cards > 8 && nodes > 1) out.warning = "Check inter-node network bandwidth";

  if (!out.valid_config) {
    out.bottleneck = "N/A";
    return out;
  }

  double pp_eff = 1.0;
  if (std::max(1L, pp) > 0) {
    if (input.stage == "Decode") pp_eff = 1.0 / pp;
    else {
      const double micro_batches = std::max(1.0, CeilDiv(static_cast<double>(batch) * ctx, micro_batch));
      pp_eff = micro_batches / (micro_batches + pp - 1.0);
    }
  }
  const double tps_after_pp = out.e25_tps_comp_raw * pp_eff;
  const double pp_bubble_pct = (1.0 - pp_eff) * 100.0;
  const double total_tokens = input.stage == "Prefill" ? std::max(1.0, static_cast<double>(batch) * ctx) : std::max(1.0, static_cast<double>(batch));
  const double bw_tp = SelectInterconnectBW(tp, hw.cards_per_node, dies, hw.d2d_bw, hw.nvlink_bw, hw.rdma_bw, hw.pcie_bw);

  double tp_comm_time = 0.0;
  double tp_comm_pct = 0.0;
  if (tp > 1) {
    const double all_reduce_per_layer = (input.stage == "Decode")
                                            ? (2.0 * (tp - 1.0) / tp * mdl.hidden * batch * input.bytes_per_param)
                                            : (2.0 * (tp - 1.0) / tp * mdl.hidden * batch * ctx * input.bytes_per_param);
    const double layers_per_stage = mdl.layers / std::max(1L, pp);
    const double all_reduce_total = layers_per_stage * 2.0 * all_reduce_per_layer;
    if (bw_tp > 0) tp_comm_time = all_reduce_total / (bw_tp * kOneE9);
    if (tps_after_pp > 0 && tp_comm_time > 0) {
      const double compute_time = total_tokens / tps_after_pp;
      if (compute_time > 0) tp_comm_pct = tp_comm_time / compute_time * 100.0;
    }
  }
  double tps_with_comm = tps_after_pp;
  if (tps_after_pp > 0 && tp_comm_time > 0) tps_with_comm = total_tokens / (total_tokens / tps_after_pp + tp_comm_time);

  const double bw_ep = SelectInterconnectBW(ep, hw.cards_per_node, dies, hw.d2d_bw, hw.nvlink_bw, hw.rdma_bw, hw.pcie_bw);
  double tps_with_ep = tps_with_comm;
  if (mdl.moe && ep > 1) {
    const double k_experts = mdl.experts_active;
    const double moe_layers = mdl.layers;
    const double a2a_bytes = (input.stage == "Decode")
                                 ? (2.0 * batch * mdl.hidden * k_experts * input.bytes_per_param * moe_layers)
                                 : (2.0 * batch * ctx * mdl.hidden * k_experts * input.bytes_per_param * moe_layers);
    double ep_comm_time = 0.0;
    if (bw_ep > 0) ep_comm_time = a2a_bytes / (bw_ep * kOneE9);
    if (ep_comm_time > 0 && tps_with_comm > 0) tps_with_ep = total_tokens / (total_tokens / tps_with_comm + ep_comm_time);
  }

  double bw_pp = (input.pp_placement == "Inter-node") ? (hw.rdma_bw > 0 ? hw.rdma_bw : hw.pcie_bw) : (hw.nvlink_bw > 0 ? hw.nvlink_bw : hw.pcie_bw);
  double tps_comp_final = tps_with_ep;
  if (pp > 1) {
    const double activation_bytes = (input.stage == "Decode")
                                        ? (batch * mdl.hidden * input.bytes_per_param * (pp - 1.0))
                                        : (batch * ctx * mdl.hidden * input.bytes_per_param * (pp - 1.0));
    double pp_transfer_time = 0.0;
    if (bw_pp > 0) pp_transfer_time = activation_bytes / (bw_pp * kOneE9);
    if (pp_transfer_time > 0 && tps_with_ep > 0) tps_comp_final = total_tokens / (total_tokens / tps_with_ep + pp_transfer_time);
  }

  double tps_roof = 0.0;
  std::string bottleneck;
  std::string bottleneck_detail;
  double tps_eff = 0.0;
  if (input.stage == "Prefill") {
    tps_roof = tps_comp_final;
    bottleneck = "comp";
    if (pp_eff < 0.5) bottleneck_detail = "PP_bubble";
    tps_eff = tps_roof * out.e7_comp_eff;
  } else {
    tps_roof = std::min(tps_comp_final, out.e26_tps_mem);
    // Memory wall first: comm is hidden behind t_mem, so never label "comm"
    // (or apply η_comp) when the roof is memory.
    if (out.e26_tps_mem < tps_comp_final) {
      bottleneck = "mem";
      if (pp_eff < 0.5) bottleneck_detail = "PP_bubble";
    } else if (tp_comm_pct > 30) {
      bottleneck = "comm";
      bottleneck_detail = "TP_comm";
    } else {
      bottleneck = "comp";
      if (pp_eff < 0.5) bottleneck_detail = "PP_bubble";
    }
    tps_eff = tps_roof * (bottleneck == "mem" ? out.e6_mem_eff : out.e7_comp_eff);
  }

  out.e29_bw_tp = bw_tp;
  out.e30_bw_pp = bw_pp;
  out.e31_bw_ep = bw_ep;
  out.e32_pp_eff = pp_eff;
  out.e33_pp_bubble_pct = pp_bubble_pct;
  out.e34_tp_comm_overhead_pct = tp_comm_pct;
  out.e35_tps_comp_before_comm = tps_after_pp;
  out.e36_tps_comp_after_comm = tps_comp_final;
  out.b35_tps_roof = tps_roof;
  out.bottleneck = bottleneck;
  out.bottleneck_detail = bottleneck_detail;
  out.b37_tps_effective = tps_eff;
  out.b39_tps_cluster = tps_eff * dp;
  out.b38_tps_per_gpu = out.b23_total_cards > 0 ? out.b39_tps_cluster / out.b23_total_cards : 0;
  out.b40_latency_ms = (tps_eff > 0)
                           ? ((input.stage == "Prefill") ? (batch * ctx * 1000.0 / tps_eff) : (batch * 1000.0 / tps_eff))
                           : 0.0;
  return out;
}

OperatorOutput CalculateOperator(const OperatorInput& input, const std::map<std::string, Hardware>& hw_db,
                                 const std::vector<OperatorDef>& op_defs) {
  OperatorOutput out;
  const auto hw_it = hw_db.find(input.hardware);
  if (hw_it == hw_db.end()) return out;
  const Hardware& hw = hw_it->second;
  OperatorDef op = op_defs.front();
  bool found = false;
  for (const auto& d : op_defs) {
    if (d.name == input.operator_name) {
      op = d;
      found = true;
      break;
    }
  }
  if (!found) return out;

  const double bpe = input.precision == "BF16" ? 2.0 : 1.0;
  double flops_per_elem = op.flops_per_elem;
  if (input.operator_name == "MoE_Router" && input.e > 1 && input.k > 0) flops_per_elem = input.k * (std::log(input.e) / std::log(2.0));
  const double sparsity = 1.0 - input.sparsity_pct / 100.0;
  const int dies = std::max(1, hw.dies_per_card);

  double tflops = 0.0;
  if (op.compute_unit == "Matrix") tflops = (input.precision == "BF16") ? hw.bf16_matrix : hw.fp8_matrix;
  else if (op.compute_unit == "Matrix_Sparse") tflops = (input.precision == "BF16") ? hw.bf16_sparse : hw.fp8_sparse;
  else if (op.compute_unit == "Vector") tflops = (input.precision == "BF16") ? hw.bf16_vector : hw.fp8_vector;
  else if (op.compute_unit == "SFU") tflops = hw.sfu;
  tflops /= dies;
  const double hbm = hw.hbm_bw / dies;
  const double mem_eff = hw.mem_eff == 0 ? 0.35 : hw.mem_eff;
  const double comp_eff = hw.comp_eff == 0 ? 0.45 : hw.comp_eff;
  const double sfu_eff = hw.sfu_eff == 0 ? 0.85 : hw.sfu_eff;

  out.category = op.category;
  out.compute_unit = op.compute_unit;
  out.flops_per_elem = flops_per_elem;
  out.read_factor = op.read_factor;
  out.write_factor = op.write_factor;
  out.total_elements = CalculateOperatorElements(input.operator_name, input);
  out.total_flops = out.total_elements * flops_per_elem * sparsity;
  out.total_bytes = CalculateOperatorBytes(input.operator_name, input, bpe, op.read_factor, op.write_factor, sparsity);
  out.ai = out.total_bytes > 0 ? out.total_flops / out.total_bytes : 0.0;
  out.tflops_used = tflops;
  out.hbm_bw = hbm;
  out.parallel_factor = std::max(1.0, input.tp);
  out.compute_bound = (out.total_flops > 0 && tflops > 0) ? (tflops * kOneE12 * out.parallel_factor / out.total_flops) : 0.0;
  out.mem_bound = (out.total_bytes > 0 && hbm > 0) ? (hbm * kOneE9 * out.parallel_factor / out.total_bytes) : 0.0;
  if (out.compute_bound == 0 && out.mem_bound == 0) {
    out.roofline_invoc_s = 0;
    out.bottleneck = "N/A";
  } else if (out.compute_bound == 0) {
    out.roofline_invoc_s = out.mem_bound;
    out.bottleneck = "mem";
  } else if (out.mem_bound == 0) {
    out.roofline_invoc_s = out.compute_bound;
    out.bottleneck = "comp";
  } else {
    out.roofline_invoc_s = std::min(out.compute_bound, out.mem_bound);
    out.bottleneck = out.mem_bound < out.compute_bound ? "mem" : "comp";
  }
  if (out.bottleneck == "mem") out.efficiency = mem_eff;
  else if (op.compute_unit == "SFU") out.efficiency = sfu_eff;
  else out.efficiency = comp_eff;
  out.effective_invoc_s = out.roofline_invoc_s * out.efficiency;
  out.bw_util_gbps = (out.effective_invoc_s > 0 && out.total_bytes > 0) ? (out.effective_invoc_s * out.total_bytes / kOneE9) : 0.0;
  out.bw_pct = hbm > 0 ? (out.bw_util_gbps / hbm * 100.0) : 0.0;
  out.per_gpu_invoc_s = out.parallel_factor > 0 ? (out.effective_invoc_s / out.parallel_factor) : 0.0;
  out.cluster_invoc_s = out.effective_invoc_s * std::max(1.0, input.dp);
  out.batch_pass_ms = out.effective_invoc_s > 0 ? (input.b * 1000.0 / out.effective_invoc_s) : 0.0;
  if (out.total_flops > 0) {
    if (out.effective_invoc_s > 0) out.invoc_duration_ns = 1.0 / out.effective_invoc_s * 1e9;
  } else if (out.total_bytes > 0) {
    const double effective_bw = hbm * mem_eff * kOneE9;
    if (effective_bw > 0) out.invoc_duration_ns = out.total_bytes / effective_bw * 1e9;
  }

  double comm_ns = 0.0;
  if (input.tp > 1) {
    const bool is_gemm = (input.operator_name == "GEMM_QK" || input.operator_name == "GEMM_AV" ||
                          input.operator_name == "GEMM_FFN_Up" || input.operator_name == "GEMM_FFN_Down" ||
                          input.operator_name == "GEMM_Gate" || input.operator_name == "GEMM_Output" ||
                          input.operator_name == "MoE_Sparse_GEMM");
    if (is_gemm) {
      double output_size = 0;
      if (input.operator_name == "GEMM_QK") output_size = input.b * input.n * input.s * input.s;
      else if (input.operator_name == "GEMM_AV") output_size = input.b * input.n * input.s * input.d;
      else if (input.operator_name == "GEMM_FFN_Up" || input.operator_name == "GEMM_Gate") output_size = input.b * input.s * input.i;
      else if (input.operator_name == "GEMM_FFN_Down") output_size = input.b * input.s * input.h;
      else if (input.operator_name == "GEMM_Output") output_size = input.b * input.s * input.v;
      else if (input.operator_name == "MoE_Sparse_GEMM") output_size = input.b * input.s * input.k * input.i;
      const double all_reduce_bytes = 2.0 * output_size * (input.tp - 1.0) / input.tp * bpe;
      const double bw_tp = hw.nvlink_bw > 0 ? hw.nvlink_bw : (hw.rdma_bw > 0 ? hw.rdma_bw : hw.pcie_bw);
      if (bw_tp > 0) comm_ns = all_reduce_bytes / (bw_tp * kOneE9) * 1e9;
    }
  }
  out.duration_compute_ns = out.invoc_duration_ns;
  out.duration_comm_ns = comm_ns;
  out.duration_total_ns = out.duration_compute_ns + out.duration_comm_ns;
  out.comm_fraction_pct = out.duration_total_ns > 0 ? (out.duration_comm_ns / out.duration_total_ns * 100.0) : 0.0;
  return out;
}

void ExportModelMatrix(const std::filesystem::path& out_file, const std::map<std::string, Hardware>& hw_db,
                       const std::map<std::string, Model>& model_db) {
  std::ofstream os(out_file);
  os << "Hardware,Model,Stage,Batch,Context,TP,PP,DP,EP,BytesPerParam,FP8_TFLOPS,HBM_BW_GBps,VRAM_GB,"
        "TotalParams_B,ActiveParams_B,IsMoE,KV_GB_per_token,Decode_BytesPerToken_GB,FLOPs_per_token,"
        "PF_Comp,PF_Mem,TPS_Comp_Theo,TPS_Mem_Theo,TPS_Roof_Theo,Bottleneck,MemEff,CompEff,"
        "TPS_Effective,TPS_perGPU,TPS_Cluster,TPOT_ms,PerGPU_Mem_GB,PerGPU_VRAM_GB,PerGPU_Fit,"
        "RecommendedCards,ConfiguredCards,ClusterCardCheck\n";
  const std::vector<std::string> hw = {"H20", "B200", "GB300", "Custom"};
  const std::vector<std::string> mdl = {"Llama-8B", "Llama-70B", "Llama-405B", "DeepSeek-V3", "Qwen2.5-72B", "Mixtral-8x7B"};
  const std::vector<std::string> stage = {"Decode", "Prefill"};
  const std::vector<int> batch = {1, 200};
  const std::vector<int> ctx = {4096, 16384};
  for (const auto& h : hw) for (const auto& m : mdl) for (const auto& s : stage) for (int b : batch) for (int c : ctx) {
    ModelInput in;
    in.hardware = h;
    in.model = m;
    in.stage = s;
    in.batch = b;
    in.context = c;
    in.bytes_per_param = model_db.at(m).default_bpp;
    const ModelOutput o = CalculateModel(in, hw_db, model_db);
    os << h << ',' << m << ',' << s << ',' << b << ',' << c << ',' << in.tp << ',' << in.pp << ',' << in.dp << ',' << in.ep << ','
       << in.bytes_per_param << ',' << o.e4_tf_mat << ',' << o.e5_bw_hbm << ',' << o.e8_vram << ','
       << o.e9_total_params << ',' << o.e10_active_params << ',' << (o.e17_is_moe ? "Yes" : "No") << ','
       << o.e20_kv_gb_per_token << ',' << o.e21_bytes_per_token_gb << ',' << o.e22_flops_per_token << ','
       << o.e23_pf_comp << ',' << o.e24_pf_mem << ',' << o.e25_tps_comp_raw << ',' << o.e26_tps_mem << ',' << o.b35_tps_roof << ','
       << o.bottleneck << ',' << o.e6_mem_eff << ',' << o.e7_comp_eff << ','
       << o.b37_tps_effective << ',' << o.b38_tps_per_gpu << ',' << o.b39_tps_cluster << ',' << o.b40_latency_ms << ','
       << o.b27_per_gpu_mem_gb << ',' << o.b28_vram_gb << ',' << o.b29_per_gpu_fit << ','
       << o.b25_recommended_cards << ',' << o.b23_total_cards << ',' << o.b30_cluster_check << '\n';
  }
}

void ExportOperatorMatrix(const std::filesystem::path& out_dir, const std::map<std::string, Hardware>& hw_db,
                          const std::vector<OperatorDef>& op_defs) {
  const long chunk = 2000;
  const std::vector<std::string> hw = {"H20", "B200", "GB300", "Custom"};
  const std::vector<std::string> prec = {"FP8", "BF16"};
  const std::vector<int> batch = {1, 64};
  const std::vector<int> seq = {1024, 4096, 16384};
  const std::vector<int> tp = {1, 2, 4, 8};
  const std::vector<int> dp = {1, 2, 4, 8};
  long part = 1;
  long rows = 0;
  long total = 0;
  std::ofstream os(out_dir / "operator_test_part01.csv");
  const std::string hdr = "Hardware,Precision,Category,Operator,Sparsity,B,S,TP,DP,TotalElements,TotalFLOPs,TotalBytes,AI,"
                          "ComputeInvoc_s,MemInvoc_s,RoofInvoc_s,Bottleneck,Efficiency,EffInvoc_s,BW_Util_GBps,BatchPass_ms,InvocDuration_ns";
  os << hdr << '\n';
  for (const auto& h : hw) for (const auto& p : prec) for (const auto& op : op_defs) {
    const std::vector<int> spar = op.use_sparse ? std::vector<int>{0, 50} : std::vector<int>{0};
    for (int sp : spar) for (int b : batch) for (int s : seq) for (int t : tp) for (int d : dp) {
      if (rows >= chunk) {
        os.close();
        part++;
        rows = 0;
        std::ostringstream fn;
        fn << "operator_test_part" << std::setfill('0') << std::setw(2) << part << ".csv";
        os.open(out_dir / fn.str());
        os << hdr << '\n';
      }
      OperatorInput in;
      in.hardware = h;
      in.precision = p;
      in.operator_name = op.name;
      in.sparsity_pct = sp;
      in.b = b;
      in.s = s;
      in.tp = t;
      in.dp = d;
      const OperatorOutput o = CalculateOperator(in, hw_db, op_defs);
      os << h << ',' << p << ',' << op.category << ',' << op.name << ',' << sp << ',' << b << ',' << s << ',' << t << ',' << d << ','
         << o.total_elements << ',' << o.total_flops << ',' << o.total_bytes << ',' << o.ai << ','
         << o.compute_bound << ',' << o.mem_bound << ',' << o.roofline_invoc_s << ',' << o.bottleneck << ','
         << o.efficiency << ',' << o.effective_invoc_s << ',' << o.bw_util_gbps << ',' << o.batch_pass_ms << ',' << o.invoc_duration_ns << '\n';
      rows++;
      total++;
    }
  }
  os.close();
  std::cout << "Operator matrix done. rows=" << total << ", files=" << part << '\n';
}

void RunModelTests(const std::map<std::string, Hardware>& hw_db, const std::map<std::string, Model>& model_db) {
  struct T {
    std::string id;
    ModelInput in;
    std::function<double(const ModelOutput&)> read;
    double expected;
  };
  std::vector<T> tests;
  ModelInput a;
  a.hardware = "H20"; a.model = "Llama-8B"; a.stage = "Decode"; a.batch = 1; a.context = 4096;
  tests.push_back({"A1", a, [](const ModelOutput& o) { return o.e26_tps_mem; }, 484});
  a.batch = 64;
  tests.push_back({"A2", a, [](const ModelOutput& o) { return o.e21_bytes_per_token_gb; }, 0.393});
  tests.push_back({"A2b", a, [](const ModelOutput& o) { return o.e26_tps_mem; }, 10178});
  tests.push_back({"A2c", a, [](const ModelOutput& o) { return o.b40_latency_ms; }, 15.4});
  a.batch = 1; a.model = "Llama-70B"; a.tp = 8;
  tests.push_back({"E2", a, [](const ModelOutput& o) { return o.e34_tp_comm_overhead_pct; }, 2.16});
  tests.push_back({"E2b", a, [](const ModelOutput& o) { return o.bottleneck == "mem" ? 1.0 : 0.0; }, 1.0});
  tests.push_back({"E2c", a, [](const ModelOutput& o) { return o.e26_tps_mem; }, 453});
  a.batch = 256;
  tests.push_back({"E3", a, [](const ModelOutput& o) { return o.e34_tp_comm_overhead_pct; }, 2.16});
  tests.push_back({"E3b", a, [](const ModelOutput& o) { return o.e36_tps_comp_after_comm; }, 8278});
  tests.push_back({"E3c", a, [](const ModelOutput& o) { return o.bottleneck == "comp" || o.bottleneck == "comm" ? 1.0 : 0.0; }, 1.0});

  ModelInput a1 = {};
  a1.hardware = "H20"; a1.model = "Llama-8B"; a1.stage = "Decode"; a1.batch = 1; a1.context = 4096;
  tests.push_back({"A1b", a1, [](const ModelOutput& o) { return o.b37_tps_effective; }, 169.3});
  tests.push_back({"A1c", a1, [](const ModelOutput& o) { return o.b40_latency_ms; }, 5.91});

  ModelInput prefill = a1;
  prefill.stage = "Prefill";
  tests.push_back({"P1", prefill, [](const ModelOutput& o) { return o.b40_latency_ms; }, 984.0});
  tests.push_back({"P2", prefill, [](const ModelOutput& o) { return o.bottleneck == "comp" ? 1.0 : 0.0; }, 1.0});

  ModelInput pp = a1;
  pp.pp = 4;
  tests.push_back({"PP1", pp, [](const ModelOutput& o) { return o.e32_pp_eff; }, 0.25});
  tests.push_back({"PP2", pp, [](const ModelOutput& o) { return o.e21_bytes_per_token_gb; }, 8.268});
  tests.push_back({"PP3", pp, [](const ModelOutput& o) { return o.e26_tps_mem; }, 484});
  tests.push_back({"PP4", pp, [](const ModelOutput& o) { return o.b37_tps_effective; }, 169.3});

  ModelInput dp = a1;
  dp.dp = 4;
  tests.push_back({"D1", dp, [](const ModelOutput& o) { return o.b40_latency_ms; }, 5.91});
  tests.push_back({"D2", dp, [](const ModelOutput& o) { return o.b39_tps_cluster; }, 677.3});

  ModelInput mla;
  mla.hardware = "H20"; mla.model = "DeepSeek-V3"; mla.stage = "Decode";
  tests.push_back({"M1", mla, [](const ModelOutput& o) { return o.e20_kv_gb_per_token; }, 0.0002186});

  ModelInput dies;
  dies.hardware = "B200"; dies.model = "Llama-8B"; dies.stage = "Decode";
  tests.push_back({"DIES1", dies, [](const ModelOutput& o) { return o.e25_tps_comp_raw; }, 70312.5});

  long pass = 0;
  for (const auto& t : tests) {
    ModelInput in = t.in;
    in.bytes_per_param = model_db.at(in.model).default_bpp;
    const ModelOutput o = CalculateModel(in, hw_db, model_db);
    const double actual = t.read(o);
    const double delta = std::abs(t.expected) > 1e-10 ? std::abs((actual - t.expected) / t.expected) * 100.0 : 0.0;
    const bool ok = delta <= 1.0;
    std::cout << t.id << " expected=" << t.expected << " actual=" << actual << " delta%=" << delta
              << " => " << (ok ? "PASS" : "FAIL") << '\n';
    if (ok) pass++;
  }
  std::cout << "Model tests pass " << pass << "/" << tests.size() << '\n';
}

void PrintUsage() {
  std::cout << "tool1_roofline <command>\n"
               "  model                run one default model case\n"
               "  operator             run one default operator case\n"
               "  export-model [file]  export test_results_from_calculator.csv\n"
               "  export-operator [dir] export operator_test_partXX.csv\n"
               "  run-model-tests      run the built-in checks against the pocket-note numbers\n";
}

}  // namespace

int main(int argc, char** argv) {
  const auto hw_db = BuildHardwareDb();
  const auto model_db = BuildModelDb();
  const auto op_defs = BuildOperatorDb();

  if (argc < 2) {
    PrintUsage();
    return 0;
  }
  const std::string cmd = argv[1];
  if (cmd == "model") {
    ModelInput in;
    in.bytes_per_param = model_db.at(in.model).default_bpp;
    const ModelOutput o = CalculateModel(in, hw_db, model_db);
    std::cout << "Model: " << in.model << " on " << in.hardware << '\n';
    std::cout << "BytesPerToken_GB=" << o.e21_bytes_per_token_gb
              << ", TPS_comp_final=" << o.e36_tps_comp_after_comm
              << ", TPS_mem=" << o.e26_tps_mem
              << ", TPS_roof=" << o.b35_tps_roof << '\n';
    std::cout << "Bottleneck=" << o.bottleneck << ", TPS_Effective=" << o.b37_tps_effective
              << ", TPS_Cluster=" << o.b39_tps_cluster << ", Latency_ms=" << o.b40_latency_ms << '\n';
    return 0;
  }
  if (cmd == "operator") {
    OperatorInput in;
    const OperatorOutput o = CalculateOperator(in, hw_db, op_defs);
    std::cout << "Operator: " << in.operator_name << " on " << in.hardware << '\n';
    std::cout << "Bottleneck=" << o.bottleneck << ", Effective_invoc_s=" << o.effective_invoc_s
              << ", BatchPass_ms=" << o.batch_pass_ms << ", InvocDuration_ns=" << o.invoc_duration_ns << '\n';
    return 0;
  }
  if (cmd == "export-model") {
    std::filesystem::path p = (argc >= 3) ? std::filesystem::path(argv[2]) : std::filesystem::path("test_results_from_calculator.csv");
    ExportModelMatrix(p, hw_db, model_db);
    std::cout << "Wrote " << p.string() << '\n';
    return 0;
  }
  if (cmd == "export-operator") {
    std::filesystem::path p = (argc >= 3) ? std::filesystem::path(argv[2]) : std::filesystem::path(".");
    ExportOperatorMatrix(p, hw_db, op_defs);
    return 0;
  }
  if (cmd == "run-model-tests") {
    RunModelTests(hw_db, model_db);
    return 0;
  }
  PrintUsage();
  return 0;
}
