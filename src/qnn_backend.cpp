#include "dragonedge/qnn_backend.h"

#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpMem.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpSystemContext.h"
#include "QnnInterface.h"
#include "QnnOpDef.h"
#include "QnnTypes.h"
#include "System/QnnSystemInterface.h"
#include "dragonedge/clock.h"
#include "dragonedge/file_io.h"
#include "dragonedge/log.h"
#include "dragonedge/mem_accounting.h"

namespace dragonedge {

namespace {

constexpr int kRpcmemHeapIdSystem = 25;
constexpr uint32_t kRpcmemDefaultFlags = 1;

void log_callback(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  char msg[2048];
  std::vsnprintf(msg, sizeof(msg), fmt, args);
  size_t n = std::strlen(msg);
  while (n > 0 && (msg[n - 1] == '\n' || msg[n - 1] == '\r')) msg[--n] = 0;
  switch (level) {
    case QNN_LOG_LEVEL_ERROR: DE_ERROR("qnn", "%s", msg); break;
    case QNN_LOG_LEVEL_WARN: DE_WARN("qnn", "%s", msg); break;
    case QNN_LOG_LEVEL_INFO: DE_INFO("qnn", "%s", msg); break;
    default: DE_DEBUG("qnn", "%s", msg); break;
  }
}

std::string normalize(const std::string& s) {
  std::string out = s;
  for (char& c : out)
    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
  return out;
}

template <class F>
auto tensor_fields(Qnn_Tensor_t& t, F f) {
  if (t.version == QNN_TENSOR_VERSION_2) return f(t.v2);
  return f(t.v1);
}

template <class F>
auto tensor_fields(const Qnn_Tensor_t& t, F f) {
  if (t.version == QNN_TENSOR_VERSION_2) return f(t.v2);
  return f(t.v1);
}

Encoding encoding_of(const Qnn_Tensor_t& t, const std::string& who) {
  return tensor_fields(t, [&](const auto& v) {
    Encoding e;
    switch (v.dataType) {
      case QNN_DATATYPE_FLOAT_16: e.dtype = DType::F16; break;
      case QNN_DATATYPE_FLOAT_32: e.dtype = DType::F32; break;
      case QNN_DATATYPE_UFIXED_POINT_16: e.dtype = DType::UFixed16; break;
      case QNN_DATATYPE_UFIXED_POINT_8: e.dtype = DType::UFixed8; break;
      default:
        throw std::runtime_error("qnn: " + who + " tensor " + v.name + " has unsupported data type " +
                                 std::to_string(static_cast<int>(v.dataType)));
    }
    if (e.dtype == DType::UFixed16 || e.dtype == DType::UFixed8) {
      const auto& q = v.quantizeParams;
      if (q.encodingDefinition != QNN_DEFINITION_DEFINED ||
          q.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
        throw std::runtime_error("qnn: " + who + " tensor " + v.name + " is not per-tensor scale/offset quantized");
      e.scale = q.scaleOffsetEncoding.scale;
      e.offset = q.scaleOffsetEncoding.offset;
    }
    return e;
  });
}

struct GraphView {
  std::string name;
  std::vector<Qnn_Tensor_t> inputs;
  std::vector<Qnn_Tensor_t> outputs;
  uint64_t spill_fill = 0;
  uint64_t vtcm_bytes = 0;
};

}

struct RpcMemAllocator::Impl {
  void* lib = nullptr;
  void* (*alloc)(int, uint32_t, int) = nullptr;
  void (*free)(void*) = nullptr;
  int (*to_fd)(void*) = nullptr;
  std::mutex mu;
  std::map<void*, uint64_t> sizes;
};

RpcMemAllocator::RpcMemAllocator() : impl_(std::make_unique<Impl>()) {
  impl_->lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (!impl_->lib) throw std::runtime_error(std::string("rpcmem: cannot load libcdsprpc.so: ") + dlerror());
  impl_->alloc = reinterpret_cast<void* (*)(int, uint32_t, int)>(dlsym(impl_->lib, "rpcmem_alloc"));
  impl_->free = reinterpret_cast<void (*)(void*)>(dlsym(impl_->lib, "rpcmem_free"));
  impl_->to_fd = reinterpret_cast<int (*)(void*)>(dlsym(impl_->lib, "rpcmem_to_fd"));
  if (!impl_->alloc || !impl_->free || !impl_->to_fd) throw std::runtime_error("rpcmem: missing symbols");
}

RpcMemAllocator::~RpcMemAllocator() {
  if (impl_->lib) dlclose(impl_->lib);
}

void* RpcMemAllocator::allocate(uint64_t bytes) {
  if (bytes > 0x7FFFFFFFull) throw std::runtime_error("rpcmem: allocation above 2 GB");
  void* p = impl_->alloc(kRpcmemHeapIdSystem, kRpcmemDefaultFlags, static_cast<int>(bytes));
  if (!p) throw std::runtime_error("rpcmem: rpcmem_alloc failed for " + std::to_string(bytes) + " bytes");
  std::lock_guard<std::mutex> l(impl_->mu);
  impl_->sizes[p] = bytes;
  npu_tracked_bytes() += static_cast<int64_t>(bytes);
  return p;
}

void RpcMemAllocator::release(void* p) {
  if (!p) return;
  {
    std::lock_guard<std::mutex> l(impl_->mu);
    auto it = impl_->sizes.find(p);
    if (it != impl_->sizes.end()) {
      npu_tracked_bytes() -= static_cast<int64_t>(it->second);
      impl_->sizes.erase(it);
    }
  }
  impl_->free(p);
}

int RpcMemAllocator::fd(void* p) const { return impl_->to_fd(p); }

struct SharedContext {
  std::string path;
  QnnSystemContext_Handle_t sys = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  std::vector<GraphView> graphs;
  int refs = 0;
};

struct QnnPiece : LoadedPiece {
  using LoadedPiece::LoadedPiece;
  std::shared_ptr<SharedContext> ctx;
  Qnn_GraphHandle_t graph = nullptr;
  std::vector<Qnn_Tensor_t> inputs;
  std::vector<Qnn_Tensor_t> outputs;
  std::vector<Qnn_MemHandle_t> handles;
};

struct QnnBackend::Impl {
  Arena& arena;
  const Manifest& manifest;
  QnnOptions opts;
  void* backend_lib = nullptr;
  void* system_lib = nullptr;
  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE sys{};
  Qnn_LogHandle_t log = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  uint32_t power_id = 0;
  bool have_power_id = false;
  QnnHtpDevice_PerfInfrastructure_t perf{};
  QnnDeviceInfo device_info;
  std::map<std::string, uint64_t> spill_fill;
  uint64_t max_spill_fill = 0;

  std::mutex group_mu;
  Qnn_ContextHandle_t anchor = nullptr;
  int live_contexts = 0;
  std::map<std::string, std::shared_ptr<SharedContext>> contexts;
  Qnn_Version_t core_api{};
  Qnn_Version_t backend_api{};
  std::string build_id = "?";

  Impl(Arena& a, const Manifest& m, QnnOptions o) : arena(a), manifest(m), opts(std::move(o)) {}

  void check(Qnn_ErrorHandle_t err, const std::string& what) const {
    if (err == QNN_SUCCESS) return;
    std::string msg;
    const char* detail = nullptr;
    if (qnn.errorGetMessage && qnn.errorGetMessage(err, &detail) == QNN_SUCCESS && detail) msg = detail;
    throw std::runtime_error("qnn: " + what + " failed (error " + std::to_string(static_cast<uint64_t>(err)) + ")" +
                             (msg.empty() ? "" : ": " + msg));
  }

  void open_interfaces() {
    backend_lib = dlopen(opts.backend_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!backend_lib) throw std::runtime_error("qnn: cannot load " + opts.backend_lib + ": " + dlerror());
    using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    auto get = reinterpret_cast<GetProviders>(dlsym(backend_lib, "QnnInterface_getProviders"));
    if (!get) throw std::runtime_error("qnn: QnnInterface_getProviders not found");
    const QnnInterface_t** providers = nullptr;
    uint32_t n = 0;
    if (get(&providers, &n) != QNN_SUCCESS || n == 0) throw std::runtime_error("qnn: no backend providers");
    bool found = false;
    for (uint32_t i = 0; i < n; ++i) {
      const auto& v = providers[i]->apiVersion.coreApiVersion;
      if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
        qnn = providers[i]->QNN_INTERFACE_VER_NAME;
        core_api = v;
        backend_api = providers[i]->apiVersion.backendApiVersion;
        found = true;
        break;
      }
    }
    if (!found) throw std::runtime_error("qnn: backend library API version does not match the SDK headers");

    system_lib = dlopen(opts.system_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!system_lib) throw std::runtime_error("qnn: cannot load " + opts.system_lib + ": " + dlerror());
    using GetSysProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
    auto get_sys = reinterpret_cast<GetSysProviders>(dlsym(system_lib, "QnnSystemInterface_getProviders"));
    if (!get_sys) throw std::runtime_error("qnn: QnnSystemInterface_getProviders not found");
    const QnnSystemInterface_t** sys_providers = nullptr;
    uint32_t ns = 0;
    if (get_sys(&sys_providers, &ns) != QNN_SUCCESS || ns == 0) throw std::runtime_error("qnn: no system providers");
    found = false;
    for (uint32_t i = 0; i < ns; ++i) {
      const auto& v = sys_providers[i]->systemApiVersion;
      if (v.major == QNN_SYSTEM_API_VERSION_MAJOR && v.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
        sys = sys_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
        found = true;
        break;
      }
    }
    if (!found) throw std::runtime_error("qnn: system library API version does not match the SDK headers");
  }

  void probe_device() {
    if (!qnn.deviceGetPlatformInfo) return;
    const QnnDevice_PlatformInfo_t* info = nullptr;
    if (qnn.deviceGetPlatformInfo(log, &info) != QNN_SUCCESS || !info) return;
    if (info->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1 && info->v1.numHwDevices > 0) {
      const auto& hw = info->v1.hwDevices[0];
      if (hw.version == QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) {
        auto* ext = reinterpret_cast<QnnHtpDevice_DeviceInfoExtension_t*>(hw.v1.deviceInfoExtension);
        if (ext && ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
          device_info.valid = true;
          device_info.soc_model = ext->onChipDevice.socModel;
          device_info.arch = static_cast<uint32_t>(ext->onChipDevice.arch);
          device_info.vtcm_size = ext->onChipDevice.vtcmSize;
        }
      }
    }
    if (qnn.deviceFreePlatformInfo) qnn.deviceFreePlatformInfo(log, info);
  }

  void set_burst() {
    QnnDevice_Infrastructure_t infra = nullptr;
    check(qnn.deviceGetInfrastructure(&infra), "deviceGetInfrastructure");
    auto* htp = static_cast<QnnHtpDevice_Infrastructure_t*>(infra);
    if (!htp || htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    perf = htp->perfInfra;
    check(perf.createPowerConfigId(0, 0, &power_id), "createPowerConfigId");
    have_power_id = true;

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& v3 = dcvs.dcvsV3Config;
    v3.contextId = power_id;
    v3.setDcvsEnable = 1;
    v3.dcvsEnable = 0;
    v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    v3.setSleepLatency = 1;
    v3.sleepLatency = 40;
    v3.setSleepDisable = 0;
    v3.sleepDisable = 0;
    v3.setBusParams = 1;
    v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    v3.setCoreParams = 1;
    v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    const QnnHtpPerfInfrastructure_PowerConfig_t* dcvs_cfg[] = {&dcvs, nullptr};
    check(perf.setPowerConfig(power_id, dcvs_cfg), "setPowerConfig(dcvs)");

    QnnHtpPerfInfrastructure_PowerConfig_t poll{};
    poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    poll.rpcPollingTimeConfig = 9999;
    QnnHtpPerfInfrastructure_PowerConfig_t latency{};
    latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    latency.rpcControlLatencyConfig = 100;
    const QnnHtpPerfInfrastructure_PowerConfig_t* rpc_cfg[] = {&poll, &latency, nullptr};
    check(perf.setPowerConfig(power_id, rpc_cfg), "setPowerConfig(rpc)");
  }

  std::vector<GraphView> read_binary_info(QnnSystemContext_Handle_t ctx, uint8_t* data, uint64_t size,
                                          const std::string& who) const {
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t info_size = 0;
    check(sys.systemContextGetBinaryInfo(ctx, data, size, &info, &info_size), "systemContextGetBinaryInfo(" + who + ")");

    uint32_t n_graphs = 0;
    QnnSystemContext_GraphInfo_t* graphs = nullptr;
    if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
      n_graphs = info->contextBinaryInfoV1.numGraphs;
      graphs = info->contextBinaryInfoV1.graphs;
    } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
      n_graphs = info->contextBinaryInfoV2.numGraphs;
      graphs = info->contextBinaryInfoV2.graphs;
    } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
      n_graphs = info->contextBinaryInfoV3.numGraphs;
      graphs = info->contextBinaryInfoV3.graphs;
    } else {
      throw std::runtime_error("qnn: unsupported context binary info version in " + who);
    }
    if (n_graphs == 0) throw std::runtime_error("qnn: " + who + " contains no graphs");

    std::vector<GraphView> out(n_graphs);
    for (uint32_t i = 0; i < n_graphs; ++i) {
      GraphView& g = out[i];
      const QnnSystemContext_GraphInfo_t& gi = graphs[i];
      auto take = [&](const auto& v) {
        g.name = v.graphName;
        g.inputs.assign(v.graphInputs, v.graphInputs + v.numGraphInputs);
        g.outputs.assign(v.graphOutputs, v.graphOutputs + v.numGraphOutputs);
      };
      if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        take(gi.graphInfoV1);
      } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        take(gi.graphInfoV2);
      } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        take(gi.graphInfoV3);
        auto* blob = reinterpret_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(gi.graphInfoV3.graphBlobInfo);
        if (blob && blob->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
          g.spill_fill = blob->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
          g.vtcm_bytes = blob->contextBinaryGraphBlobInfoV1.vtcmSize;
        }
      } else {
        throw std::runtime_error("qnn: unsupported graph info version in " + who);
      }
    }
    return out;
  }

  static const GraphView& graph_for(const std::vector<GraphView>& graphs, const PieceSpec& spec) {
    for (const auto& g : graphs)
      if (g.name == spec.name) return g;
    if (graphs.size() == 1) return graphs[0];
    throw std::runtime_error("qnn: " + spec.context_path + " has no graph named " + spec.name);
  }

  std::shared_ptr<SharedContext> acquire_context(const PieceSpec& spec, LoadTiming& timing) {
    std::lock_guard<std::mutex> lock(group_mu);
    auto it = contexts.find(spec.context_path);
    if (it != contexts.end()) {
      ++it->second->refs;
      return it->second;
    }
    auto c = std::make_shared<SharedContext>();
    c->path = spec.context_path;
    const double t0 = now_ms();
    MappedFile file(spec.context_path);
    file.touch();
    timing.file_bytes = file.size();
    const double t1 = now_ms();
    try {
      check(sys.systemContextCreate(&c->sys), "systemContextCreate");
      c->graphs = read_binary_info(c->sys, file.data(), file.size(), spec.context_path);

      QnnHtpContext_CustomConfig_t group{};
      QnnContext_Config_t group_cfg{};
      const QnnContext_Config_t* cfgs[] = {nullptr, nullptr};
      const bool grouped = opts.share_spill_fill && max_spill_fill > 0;
      if (grouped) {
        group.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
        group.groupRegistration.firstGroupHandle = anchor;
        group.groupRegistration.maxSpillFillBuffer = max_spill_fill;
        group_cfg.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
        group_cfg.customConfig = &group;
        cfgs[0] = &group_cfg;
      }
      const Qnn_ErrorHandle_t err = qnn.contextCreateFromBinary(backend, device, grouped ? cfgs : nullptr, file.data(),
                                                                file.size(), &c->context, nullptr);
      if (err != QNN_SUCCESS) {
        const MemSnapshot s = MemAccounting::snapshot();
        DE_ERROR("qnn", "%s: context create failed with %zu other contexts live; process dmabuf %.1f MB, rss %.1f MB, "
                 "MemAvailable %.1f MB, binary %.1f MB",
                 spec.name.c_str(), contexts.size(), s.dmabuf_fd_bytes / 1048576.0, s.rss / 1048576.0,
                 s.mem_available / 1048576.0, file.size() / 1048576.0);
      }
      check(err, "contextCreateFromBinary(" + spec.name + ")");
      if (grouped && !anchor) anchor = c->context;
    } catch (...) {
      if (c->sys) sys.systemContextFree(c->sys);
      throw;
    }
    timing.read_ms = t1 - t0;
    timing.create_ms = now_ms() - t1;
    c->refs = 1;
    contexts[c->path] = c;
    ++live_contexts;
    return c;
  }

  void release_context(const std::shared_ptr<SharedContext>& c, const std::string& who) {
    std::lock_guard<std::mutex> lock(group_mu);
    if (--c->refs > 0) return;
    if (c->context == anchor && live_contexts > 1)
      throw std::logic_error("qnn: spill-fill anchor " + who + " freed while other contexts still use it");
    qnn.contextFree(c->context, nullptr);
    if (c->context == anchor) anchor = nullptr;
    if (c->sys) sys.systemContextFree(c->sys);
    contexts.erase(c->path);
    --live_contexts;
  }

  Slot& slot_for(const Qnn_Tensor_t& t, const PieceSpec& spec, bool is_input) {
    std::string tname = tensor_fields(t, [](const auto& v) { return std::string(v.name); });
    const auto& names = is_input ? spec.inputs : spec.outputs;
    for (const auto& n : names)
      if (n == tname || normalize(n) == normalize(tname)) return arena.slot(n);
    throw std::runtime_error("qnn: " + spec.name + " graph " + (is_input ? "input " : "output ") + tname +
                             " is not in the manifest");
  }

  Encoding layout_for(const Qnn_Tensor_t& t, const Slot& s, Encoding e, const std::string& who) {
    std::vector<int64_t> dims = tensor_fields(t, [](const auto& v) {
      return std::vector<int64_t>(v.dimensions, v.dimensions + v.rank);
    });
    const auto& shape = s.spec.shape;
    if (dims == shape) return e;
    if (shape.size() == 4 && dims == std::vector<int64_t>{shape[0], shape[2], shape[3], shape[1]}) {
      e.nhwc = true;
      return e;
    }
    std::string got, want;
    for (auto d : dims) got += std::to_string(d) + ",";
    for (auto d : shape) want += std::to_string(d) + ",";
    throw std::runtime_error("qnn: " + who + " tensor " + s.spec.name + " has dims [" + got + "], manifest says [" +
                             want + "]");
  }

  Qnn_MemHandle_t register_shared(Qnn_ContextHandle_t ctx, Qnn_Tensor_t& t, int fd, uint64_t total, uint64_t offset,
                                  const std::string& what) {
    Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
    tensor_fields(t, [&](auto& v) {
      desc.memShape.numDim = v.rank;
      desc.memShape.dimSize = v.dimensions;
      desc.memShape.shapeConfig = nullptr;
      desc.dataType = v.dataType;
    });
    desc.memType = QNN_MEM_TYPE_CUSTOM;
    QnnMemHtp_Descriptor_t htp{};
    htp.type = QNN_HTP_MEM_SHARED_BUFFER;
    htp.size = total;
    htp.sharedBufferConfig.fd = fd;
    htp.sharedBufferConfig.offset = offset;
    desc.customInfo = &htp;
    Qnn_MemHandle_t h = nullptr;
    check(qnn.memRegister(ctx, &desc, 1, &h), "memRegister(" + what + ")");
    tensor_fields(t, [&](auto& v) {
      v.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
      v.memHandle = h;
    });
    return h;
  }

  Qnn_MemHandle_t register_slot(Qnn_ContextHandle_t ctx, Qnn_Tensor_t& t, const Slot& s) {
    return register_shared(ctx, t, arena.fd(), arena.size(), s.offset, s.spec.name);
  }
};

QnnBackend::QnnBackend(Arena& arena, const Manifest& m, QnnOptions opts)
    : impl_(std::make_unique<Impl>(arena, m, std::move(opts))) {
  Impl& I = *impl_;
  if (arena.fd() < 0) throw std::runtime_error("qnn: the arena must be allocated with RpcMemAllocator");
  I.open_interfaces();
  if (I.qnn.logCreate) I.qnn.logCreate(log_callback, static_cast<QnnLog_Level_t>(I.opts.log_level), &I.log);
  I.check(I.qnn.backendCreate(I.log, nullptr, &I.backend), "backendCreate");
  const char* build_id = nullptr;
  if (I.qnn.backendGetBuildId && I.qnn.backendGetBuildId(&build_id) == QNN_SUCCESS && build_id)
    I.build_id = build_id;
  DE_INFO("qnn", "backend %s loaded: core API %u.%u.%u, HTP backend API %u.%u.%u, build %s", I.opts.backend_lib.c_str(),
          I.core_api.major, I.core_api.minor, I.core_api.patch, I.backend_api.major, I.backend_api.minor,
          I.backend_api.patch, I.build_id.c_str());
  I.check(I.qnn.deviceCreate(I.log, nullptr, &I.device), "deviceCreate");
  I.probe_device();
  if (I.device_info.valid)
    DE_INFO("qnn", "device: soc_model %u, Hexagon v%u, VTCM %llu MB", I.device_info.soc_model, I.device_info.arch,
            static_cast<unsigned long long>(I.device_info.vtcm_size));
  else
    DE_WARN("qnn", "device platform info not available");
  if (I.opts.burst) {
    I.set_burst();
    DE_INFO("qnn", "performance mode: burst (DCVS off, max voltage corners, RPC polling 9999 us)");
  }
  DE_INFO("qnn", "arena: %.2f MB rpcmem, fd %d, %zu tensors", arena.size() / 1048576.0, arena.fd(), arena.slots().size());

  if (I.opts.max_spill_fill_override > 0) {
    I.max_spill_fill = I.opts.max_spill_fill_override;
  } else {
    size_t present = 0;
    std::map<std::string, std::vector<QnnBinaryInfo>> seen;
    for (const auto& p : m.pieces) {
      if (!file_exists(p.context_path)) {
        DE_DEBUG("qnn", "%s: no context binary at %s", p.name.c_str(), p.context_path.c_str());
        continue;
      }
      ++present;
      auto it = seen.find(p.context_path);
      if (it == seen.end()) it = seen.emplace(p.context_path, inspect(p.context_path)).first;
      const QnnBinaryInfo* info = nullptr;
      for (const auto& g : it->second)
        if (g.graph_name == p.name || it->second.size() == 1) info = &g;
      if (!info) throw std::runtime_error("qnn: " + p.context_path + " has no graph named " + p.name);
      uint64_t ctx_spill = 0;
      for (const auto& g : it->second) ctx_spill = std::max(ctx_spill, g.spill_fill_bytes);
      I.spill_fill[p.name] = ctx_spill;
      I.max_spill_fill = std::max(I.max_spill_fill, ctx_spill);
      const std::string file = p.context_path.substr(p.context_path.find_last_of("/\\") + 1);
      DE_INFO("qnn", "%-10s %-14s graph %-12s spill-fill %7.2f MB  vtcm %.1f MB  (%zu in / %zu out, read in %.1f ms)",
              p.name.c_str(), file.c_str(), info->graph_name.c_str(), info->spill_fill_bytes / 1048576.0,
              info->vtcm_bytes / 1048576.0, info->inputs.size(), info->outputs.size(), info->inspect_ms);
    }
    DE_INFO("qnn", "context binaries present for %zu of %zu pieces (%zu files); largest spill-fill %.2f MB", present,
            m.pieces.size(), seen.size(), I.max_spill_fill / 1048576.0);
  }
}

QnnBackend::~QnnBackend() {
  Impl& I = *impl_;
  if (I.have_power_id) I.perf.destroyPowerConfigId(I.power_id);
  if (I.device && I.qnn.deviceFree) I.qnn.deviceFree(I.device);
  if (I.backend && I.qnn.backendFree) I.qnn.backendFree(I.backend);
  if (I.log && I.qnn.logFree) I.qnn.logFree(I.log);
  if (I.system_lib) dlclose(I.system_lib);
  if (I.backend_lib) dlclose(I.backend_lib);
}

const QnnDeviceInfo& QnnBackend::device_info() const { return impl_->device_info; }
uint64_t QnnBackend::max_spill_fill() const { return impl_->max_spill_fill; }

uint64_t QnnBackend::spill_fill(const std::string& piece) const {
  auto it = impl_->spill_fill.find(piece);
  return it == impl_->spill_fill.end() ? 0 : it->second;
}

std::vector<QnnBinaryInfo> QnnBackend::inspect(const std::string& context_path) const {
  const Impl& I = *impl_;
  const double t0 = now_ms();
  MappedFile file(context_path);
  QnnSystemContext_Handle_t ctx = nullptr;
  I.check(I.sys.systemContextCreate(&ctx), "systemContextCreate");
  std::vector<QnnBinaryInfo> out;
  try {
    for (auto& g : I.read_binary_info(ctx, file.data(), file.size(), context_path)) {
      QnnBinaryInfo b;
      b.graph_name = g.name;
      b.spill_fill_bytes = g.spill_fill;
      b.vtcm_bytes = g.vtcm_bytes;
      for (auto& t : g.inputs) b.inputs.push_back(tensor_fields(t, [](const auto& v) { return std::string(v.name); }));
      for (auto& t : g.outputs) b.outputs.push_back(tensor_fields(t, [](const auto& v) { return std::string(v.name); }));
      out.push_back(std::move(b));
    }
  } catch (...) {
    I.sys.systemContextFree(ctx);
    throw;
  }
  I.sys.systemContextFree(ctx);
  for (auto& b : out) b.inspect_ms = now_ms() - t0;
  return out;
}

QnnSelfTest QnnBackend::self_test(uint64_t elems, int reps) {
  Impl& I = *impl_;
  QnnSelfTest r;
  const uint32_t C = 64, W = 64;
  const uint32_t H = static_cast<uint32_t>(std::max<uint64_t>(1, elems / (C * W)));
  r.elems = static_cast<uint64_t>(H) * W * C;
  const uint64_t bytes = r.elems * 2;
  uint32_t dims[4] = {1, H, W, C};

  double t = now_ms();
  RpcMemAllocator io;
  auto* buf = static_cast<uint8_t*>(io.allocate(bytes * 3));
  const int fd = io.fd(buf);
  r.alloc_ms = now_ms() - t;

  Qnn_ContextHandle_t ctx = nullptr;
  std::vector<Qnn_MemHandle_t> handles;
  auto cleanup = [&] {
    if (!handles.empty()) I.qnn.memDeRegister(handles.data(), static_cast<uint32_t>(handles.size()));
    handles.clear();
    if (ctx) I.qnn.contextFree(ctx, nullptr);
    ctx = nullptr;
    if (buf) io.release(buf);
    buf = nullptr;
  };
  try {
    t = now_ms();
    I.check(I.qnn.contextCreate(I.backend, I.device, nullptr, &ctx), "contextCreate");
    Qnn_GraphHandle_t graph = nullptr;
    I.check(I.qnn.graphCreate(ctx, "dragonedge_selftest", nullptr, &graph), "graphCreate");

    const Encoding enc_a{DType::UFixed16, 2.0f / 65535.0f, -32768};
    const Encoding enc_b{DType::UFixed16, 1.0f / 65535.0f, -32768};
    const Encoding enc_c{DType::UFixed16, 3.0f / 65535.0f, -32768};
    auto make = [&](const char* name, Qnn_TensorType_t type, const Encoding& e) {
      Qnn_Tensor_t x = QNN_TENSOR_INIT;
      x.v1.name = name;
      x.v1.type = type;
      x.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
      x.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
      x.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
      x.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      x.v1.quantizeParams.scaleOffsetEncoding.scale = e.scale;
      x.v1.quantizeParams.scaleOffsetEncoding.offset = e.offset;
      x.v1.rank = 4;
      x.v1.dimensions = dims;
      x.v1.memType = QNN_TENSORMEMTYPE_RAW;
      x.v1.clientBuf.data = nullptr;
      x.v1.clientBuf.dataSize = 0;
      I.check(I.qnn.tensorCreateGraphTensor(graph, &x), std::string("tensorCreateGraphTensor(") + name + ")");
      return x;
    };
    Qnn_Tensor_t ins[2] = {make("a", QNN_TENSOR_TYPE_APP_WRITE, enc_a), make("b", QNN_TENSOR_TYPE_APP_WRITE, enc_b)};
    Qnn_Tensor_t outs[1] = {make("sum", QNN_TENSOR_TYPE_APP_READ, enc_c)};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "add";
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_ELEMENT_WISE_ADD;
    op.v1.numOfInputs = 2;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = outs;
    I.check(I.qnn.graphAddNode(graph, op), "graphAddNode(ElementWiseAdd)");
    r.build_ms = now_ms() - t;

    t = now_ms();
    I.check(I.qnn.graphFinalize(graph, nullptr, nullptr), "graphFinalize");
    r.finalize_ms = now_ms() - t;

    t = now_ms();
    handles.push_back(I.register_shared(ctx, ins[0], fd, bytes * 3, 0, "selftest.a"));
    handles.push_back(I.register_shared(ctx, ins[1], fd, bytes * 3, bytes, "selftest.b"));
    handles.push_back(I.register_shared(ctx, outs[0], fd, bytes * 3, bytes * 2, "selftest.sum"));
    r.register_ms = now_ms() - t;

    auto* a = reinterpret_cast<uint16_t*>(buf);
    auto* b = reinterpret_cast<uint16_t*>(buf + bytes);
    auto* c = reinterpret_cast<uint16_t*>(buf + bytes * 2);
    {
      std::vector<float> va(r.elems), vb(r.elems);
      for (uint64_t i = 0; i < r.elems; ++i) {
        va[i] = std::sin(0.001f * static_cast<float>(i % 100000));
        vb[i] = 0.5f * std::cos(0.0007f * static_cast<float>(i % 70000));
        c[i] = 0;
      }
      encode_f32(va.data(), a, r.elems, enc_a);
      encode_f32(vb.data(), b, r.elems, enc_b);
    }

    t = now_ms();
    I.check(I.qnn.graphExecute(graph, ins, 2, outs, 1, nullptr, nullptr), "graphExecute(selftest)");
    r.first_exec_ms = now_ms() - t;
    for (uint64_t i = 0; i < r.elems; ++i) {
      float fa, fb, fc;
      decode_f32(a + i, &fa, 1, enc_a);
      decode_f32(b + i, &fb, 1, enc_b);
      decode_f32(c + i, &fc, 1, enc_c);
      r.max_abs_err = std::max(r.max_abs_err, std::fabs(fc - (fa + fb)));
    }
    t = now_ms();
    for (int i = 0; i < reps; ++i)
      I.check(I.qnn.graphExecute(graph, ins, 2, outs, 1, nullptr, nullptr), "graphExecute(selftest)");
    r.exec_ms = reps > 0 ? (now_ms() - t) / reps : 0;
    r.gb_per_s = r.exec_ms > 0 ? (3.0 * bytes / 1e9) / (r.exec_ms / 1000.0) : 0;
    r.pass = r.max_abs_err <= 3.0f * enc_c.scale;
    DE_INFO("selftest", "HTP finalize %.1f ms | first execute %.2f ms | steady %.3f ms (%.2f GB/s) | max |npu-cpu| %.6f -> %s",
            r.finalize_ms, r.first_exec_ms, r.exec_ms, r.gb_per_s, r.max_abs_err, r.pass ? "PASS" : "FAIL");

    t = now_ms();
    for (const PieceSpec& p : I.manifest.pieces) {
      std::vector<std::string> names = p.inputs;
      names.insert(names.end(), p.outputs.begin(), p.outputs.end());
      std::vector<Qnn_MemHandle_t> handles_p;
      std::vector<std::vector<uint32_t>> dims_p;
      dims_p.reserve(names.size());
      for (const auto& n : names) {
        const Slot& s = I.arena.slot(n);
        dims_p.emplace_back();
        for (int64_t d : s.spec.shape) dims_p.back().push_back(static_cast<uint32_t>(d));
        Qnn_Tensor_t x = QNN_TENSOR_INIT;
        x.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
        x.v1.rank = static_cast<uint32_t>(dims_p.back().size());
        x.v1.dimensions = dims_p.back().data();
        handles_p.push_back(I.register_slot(ctx, x, s));
      }
      I.check(I.qnn.memDeRegister(handles_p.data(), static_cast<uint32_t>(handles_p.size())),
              "memDeRegister(" + p.name + ")");
      r.arena_slots += static_cast<uint32_t>(handles_p.size());
    }
    r.arena_register_ms = now_ms() - t;
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
  return r;
}

std::unique_ptr<LoadedPiece> QnnBackend::load(const PieceSpec& spec) {
  Impl& I = *impl_;
  auto piece = std::make_unique<QnnPiece>(spec);
  const double t0 = now_ms();
  try {
    piece->ctx = I.acquire_context(spec, piece->timing);
    const double t1 = now_ms();
    const GraphView& g = Impl::graph_for(piece->ctx->graphs, spec);
    I.check(I.qnn.graphRetrieve(piece->ctx->context, g.name.c_str(), &piece->graph), "graphRetrieve(" + spec.name + ")");
    const double t2 = now_ms();
    if (piece->timing.create_ms == 0) piece->timing.create_ms = t2 - t1;

    piece->inputs = g.inputs;
    piece->outputs = g.outputs;
    for (int pass = 0; pass < 2; ++pass) {
      auto& tensors = pass == 0 ? piece->inputs : piece->outputs;
      for (auto& t : tensors) {
        Slot& s = I.slot_for(t, spec, pass == 0);
        Encoding e = I.layout_for(t, s, encoding_of(t, spec.name), spec.name);
        I.arena.bind_encoding(s.spec.name, e, spec.name);
        DE_DEBUG("qnn", "%s %s %-14s %s scale %.6g offset %d%s @ arena+%llu", spec.name.c_str(), pass == 0 ? "in " : "out",
                 s.spec.name.c_str(), dtype_name(e.dtype), e.scale, e.offset, e.nhwc ? " nhwc" : "",
                 static_cast<unsigned long long>(s.offset));
        piece->handles.push_back(I.register_slot(piece->ctx->context, t, s));
      }
    }
    if (piece->inputs.size() != spec.inputs.size() || piece->outputs.size() != spec.outputs.size())
      throw std::runtime_error("qnn: " + spec.name + " graph I/O count differs from the manifest");
    piece->timing.register_ms = now_ms() - t2;
    piece->timing.total_ms = now_ms() - t0;
  } catch (...) {
    unload(std::move(piece));
    throw;
  }
  return piece;
}

void QnnBackend::execute(LoadedPiece& lp) {
  Impl& I = *impl_;
  auto& p = static_cast<QnnPiece&>(lp);
  I.check(I.qnn.graphExecute(p.graph, p.inputs.data(), static_cast<uint32_t>(p.inputs.size()), p.outputs.data(),
                             static_cast<uint32_t>(p.outputs.size()), nullptr, nullptr),
          "graphExecute(" + p.spec->name + ")");
}

void QnnBackend::unload(std::unique_ptr<LoadedPiece> lp) {
  if (!lp) return;
  Impl& I = *impl_;
  auto* p = static_cast<QnnPiece*>(lp.get());
  if (!p->handles.empty()) I.qnn.memDeRegister(p->handles.data(), static_cast<uint32_t>(p->handles.size()));
  if (p->ctx) I.release_context(p->ctx, p->spec->name);
}

}
