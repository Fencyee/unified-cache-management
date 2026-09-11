#include "compressor.h"
#include <sched.h>
#include "logger/logger.h"
#include "trans_manager.h"

namespace UC::Compressor {
namespace {

Status ValidateTransferTask(const Detail::TaskDesc& task)
{
    if (task.empty()) {
        return Status::InvalidParam("compress task must contain at least one shard");
    }
    for (const auto& shard : task) {
        if (shard.addrs.size() != 1 || shard.addrs.front() == nullptr) {
            return Status::InvalidParam(
                "compress shard({}) must contain exactly one non-null host address", shard.index);
        }
    }
    return Status::OK();
}

}  // namespace

class CompressorImpl {
public:
    StoreV1* backend{nullptr};
    bool transEnable{false};
    TransManager transMgr;

public:
    Status Setup(const Config& config)
    {
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        backend = config.storeBackend;
        transEnable = config.deviceId >= 0;
        if (transEnable) {
            s = transMgr.Setup(config);
            if (s.Failure()) [[unlikely]] { return s; }
        }
        ShowConfig(config);
        return Status::OK();
    }

private:
    Status CheckConfig(const Config& config)
    {
        if (!config.storeBackend) { return Status::InvalidParam("invalid store backend"); }
        if (config.deviceId < -1) {
            return Status::InvalidParam("invalid device({})", config.deviceId);
        }
        if (config.deviceId >= 0) {
            if (config.shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
            if (config.compressedShardSize == 0) {
                return Status::InvalidParam("invalid compressed shard size");
            }
            if (config.compressThreadNum == 0) {
                return Status::InvalidParam("invalid compress thread number");
            }
            if (config.decompressThreadNum == 0) {
                return Status::InvalidParam("invalid decompress thread number");
            }
        }
        if (config.metricsLevel != "off" && config.metricsLevel != "basic" &&
            config.metricsLevel != "detailed") {
            return Status::InvalidParam("invalid compress_metrics_level({})", config.metricsLevel);
        }
        for (const auto core : config.cpuAffinityCores) {
            if (core < 0 || core >= CPU_SETSIZE) {
                return Status::InvalidParam("invalid cpu core({})", core);
            }
        }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "Compressor";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::StoreBackend to {}.", ns, backend->Readme());
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::CompressedShardSize to {}.", ns, config.compressedShardSize);
        UC_INFO("Set {}::CompressRatio to {}.", ns, config.compressRatio);
        UC_INFO("Set {}::DataType to {}.", ns, static_cast<int32_t>(config.dataType));
        UC_INFO("Set {}::CompressThreadNum to {}.", ns, config.compressThreadNum);
        UC_INFO("Set {}::DecompressThreadNum to {}.", ns, config.decompressThreadNum);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
        UC_INFO("Set {}::CpuAffinityCores to {}.", ns, config.cpuAffinityCores);
        UC_INFO("Set {}::MetricsLevel to {}.", ns, config.metricsLevel);
    }
};

Compressor::~Compressor() = default;

Status Compressor::Setup(const Detail::Dictionary& config)
{
    Config param;
    config.Get("store_backend", param.storeBackend);
    config.GetNumber("device_id", param.deviceId);
    config.GetNumber("shard_size", param.shardSize);
    config.GetNumber("compressed_shard_size", param.compressedShardSize);
    config.GetNumber("compress_ratio", param.compressRatio);
    config.GetNumber("data_type", param.dataType);
    config.Get("compress_metrics_level", param.metricsLevel);
    if (config.Contains("compress_thread_num")) {
        config.GetNumber("compress_thread_num", param.compressThreadNum);
    } else {
        size_t legacyStreamNumber = 8;
        config.GetNumber("stream_number", legacyStreamNumber);
        param.compressThreadNum = legacyStreamNumber >> 1;
    }
    config.GetNumber("decompress_thread_num", param.decompressThreadNum);
    config.GetNumber("timeout_ms", param.timeoutMs);
    config.Get("cpu_affinity_cores", param.cpuAffinityCores);

    try {
        auto impl = std::make_shared<CompressorImpl>();
        auto status = impl->Setup(param);
        if (status.Failure()) { return status; }
        impl_ = std::move(impl);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to setup Compressor.", e.what());
        return Status::Error(e.what());
    } catch (...) {
        UC_ERROR("Failed to setup Compressor with an unknown exception.");
        return Status::Error("unknown compressor setup exception");
    }
    return Status::OK();
}

std::string Compressor::Readme() const { return "Compressor"; }

Expected<std::vector<uint8_t>> Compressor::Lookup(const Detail::BlockId* blocks, size_t num)
{
    auto res = impl_->backend->Lookup(blocks, num);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
    return res;
}

Expected<ssize_t> Compressor::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    auto res = impl_->backend->LookupOnPrefix(blocks, num);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
    return res;
}

Expected<ssize_t> Compressor::LookupOnReverse(const Detail::BlockId* blocks, size_t num)
{
    auto res = impl_->backend->LookupOnReverse(blocks, num);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
    return res;
}

void Compressor::Prefetch(const Detail::BlockId* blocks, size_t num)
{
    impl_->backend->Prefetch(blocks, num);
}

Status Compressor::CheckHealth() { return impl_->backend->CheckHealth(); }

Expected<Detail::TaskHandle> Compressor::Load(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto status = ValidateTransferTask(task);
    if (status.Failure()) { return status; }
    const auto brief = task.brief;
    auto res = impl_->transMgr.Submit({TransTask::Type::LOAD, std::move(task)});
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to submit load task({}).", res.Error(), brief); }
    return res;
}

Expected<Detail::TaskHandle> Compressor::Dump(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto status = ValidateTransferTask(task);
    if (status.Failure()) { return status; }
    const auto brief = task.brief;
    auto res = impl_->transMgr.Submit({TransTask::Type::DUMP, std::move(task)});
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to submit dump task({}).", res.Error(), brief); }
    return res;
}

Expected<bool> Compressor::Check(Detail::TaskHandle taskId)
{
    auto res = impl_->transMgr.Check(taskId);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to check task({}).", res.Error(), taskId); }
    return res;
}

Status Compressor::Wait(Detail::TaskHandle taskId)
{
    auto s = impl_->transMgr.Wait(taskId);
    if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to wait task({}).", s, taskId); }
    return s;
}

}  // namespace UC::Compressor

extern "C" UC::StoreV1* MakeCompressStore() { return new UC::Compressor::Compressor(); }
