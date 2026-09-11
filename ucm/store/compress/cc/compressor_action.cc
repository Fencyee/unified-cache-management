#include "compressor_action.h"
#include <chrono>
#include <exception>
#include <pthread.h>
#include <queue>
#include <vector>
#include "logger/logger.h"
#include "metrics_api.h"

namespace UC::Compressor {
namespace {

using SteadyClock = std::chrono::steady_clock;

class ActivityGuard {
public:
    explicit ActivityGuard(std::atomic<size_t>* active) : active_(active)
    {
        if (active_) { active_->fetch_add(1, std::memory_order_relaxed); }
    }
    ~ActivityGuard()
    {
        if (active_) { active_->fetch_sub(1, std::memory_order_relaxed); }
    }

    ActivityGuard(const ActivityGuard&) = delete;
    ActivityGuard& operator=(const ActivityGuard&) = delete;

private:
    std::atomic<size_t>* active_;
};

double SecondsSince(const SteadyClock::time_point& start)
{
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

void UpdateHighWatermark(std::atomic<size_t>& highWatermark, size_t value)
{
    size_t current = highWatermark.load(std::memory_order_relaxed);
    while (current < value &&
           !highWatermark.compare_exchange_weak(current, value, std::memory_order_relaxed)) {}
}

struct R160ModeCounts {
    size_t highPrecision{0};
    size_t quantized{0};
};

struct R200ModeCounts {
    size_t tunstall{0};
    size_t fp8Fallback{0};
};

struct CodecModeCounts {
    R160ModeCounts r160;
    R200ModeCounts r200;

    void Add(CodecPayloadMode mode)
    {
        switch (mode) {
            case CodecPayloadMode::R160_HIGH_PRECISION: ++r160.highPrecision; break;
            case CodecPayloadMode::R160_QUANTIZED: ++r160.quantized; break;
            case CodecPayloadMode::R200_TUNSTALL: ++r200.tunstall; break;
            case CodecPayloadMode::R200_FP8_FALLBACK: ++r200.fp8Fallback; break;
            case CodecPayloadMode::INVALID: break;
            case CodecPayloadMode::NOT_APPLICABLE: break;
        }
    }
};

enum class CodecStatsStage {
    LOAD,
    DUMP,
};

void ReportR160ModeStats(Detail::TaskHandle taskId, CodecStatsStage stage,
                         const R160ModeCounts& counts)
{
    const size_t valid = counts.highPrecision + counts.quantized;
    if (valid == 0) { return; }

    const double highRatio =
        100.0 * static_cast<double>(counts.highPrecision) / static_cast<double>(valid);
    const double quantizedRatio =
        100.0 * static_cast<double>(counts.quantized) / static_cast<double>(valid);
    UC_DEBUG(
        "R160 {} MODE | task_id: {}, high_precision: {}, quantized: {}, high_ratio: {:.2f}%, "
        "quantized_ratio: {:.2f}%",
        stage == CodecStatsStage::LOAD ? "LOAD" : "DUMP", taskId, counts.highPrecision,
        counts.quantized, highRatio, quantizedRatio);
}

void ReportR200ModeStats(Detail::TaskHandle taskId, CodecStatsStage stage,
                         const R200ModeCounts& counts)
{
    const size_t valid = counts.tunstall + counts.fp8Fallback;
    if (valid == 0) { return; }

    const double tunstallRatio =
        100.0 * static_cast<double>(counts.tunstall) / static_cast<double>(valid);
    const double fallbackRatio =
        100.0 * static_cast<double>(counts.fp8Fallback) / static_cast<double>(valid);
    UC_DEBUG(
        "R200 {} MODE | task_id: {}, tunstall: {}, fp8_fallback: {}, tunstall_ratio: {:.2f}%, "
        "fallback_ratio: {:.2f}%",
        stage == CodecStatsStage::LOAD ? "LOAD" : "DUMP", taskId, counts.tunstall,
        counts.fp8Fallback, tunstallRatio, fallbackRatio);
}

void ReportCodecModeStats(Detail::TaskHandle taskId, CodecStatsStage stage,
                          const CodecModeCounts& counts)
{
    ReportR160ModeStats(taskId, stage, counts.r160);
    ReportR200ModeStats(taskId, stage, counts.r200);
}

}  // namespace

CompressorAction::~CompressorAction()
{
    metricsStop_.store(true, std::memory_order_relaxed);
    metricsCv_.notify_all();
    if (metricsThread_.joinable()) { metricsThread_.join(); }
}

Status CompressorAction::Setup(const Config& config, HashSet<Detail::TaskHandle>* failureSet)
{
    backend_ = config.storeBackend;
    failureSet_ = failureSet;
    shardSize_ = config.shardSize;
    compressedShardSize_ = config.compressedShardSize;
    decompressThreadNum = config.decompressThreadNum;
    metricsLevel_ = config.metricsLevel == "off"     ? MetricsLevel::OFF
                    : config.metricsLevel == "basic" ? MetricsLevel::BASIC
                                                     : MetricsLevel::DETAILED;

    const auto ratio = static_cast<FixedRatio>(config.compressRatio);
    const auto dataType = static_cast<DataType>(config.dataType);
    codec_ = ratio == R160 && dataType == DT_BF16
                 ? MakeR160BaseCodec(compressedShardSize_)
                 : MakeCodec(ratio, dataType, compressedShardSize_);
    if (!codec_) {
        return Status::InvalidParam("Unsupported codec combo (ratio={}, dtype={})",
                                    config.compressRatio, config.dataType);
    }

    if ((shardSize_ & 1U) != 0) {
        return Status::InvalidParam("BF16 shardSize({}) must be even", shardSize_);
    }
    if (compressedShardSize_ == 0) {
        return Status::InvalidParam("compressed_shard_size must be provided by pipeline builder");
    }
    const size_t codecCompressedSize = codec_->CompressedSize(shardSize_);
    if (codecCompressedSize != compressedShardSize_) {
        return Status::InvalidParam(
            "compressed shard size({}) is invalid for shardSize({}), ratio({}) and dtype({})",
            compressedShardSize_, shardSize_, config.compressRatio, config.dataType);
    }
    if (codec_->NeedsCompress() && compressedShardSize_ % 4096 != 0) {
        return Status::InvalidParam(
            "compressed shard size({}) must be 4096-byte aligned for shardSize({}) and ratio({})",
            compressedShardSize_, shardSize_, config.compressRatio);
    }

    dump_pool_.SetNWorker(config.streamNumber >> 1)
        .SetCpuAffinity(config.cpuAffinityCores)
        .SetWorkerFn([this](auto& ct, auto&) { Compress_Dump(ct); })
        .Run();

    load_pool_.SetNWorker(decompressThreadNum)
        .SetCpuAffinity(config.cpuAffinityCores)
        .SetWorkerFn([this](auto& ct, auto&) { Compress_Load(ct); })
        .Run();

    UC_DEBUG("Compressor Setup OK | load_threads: {}, shard_size: {} B, stored_shard_size: {} B",
             decompressThreadNum, shardSize_, compressedShardSize_);

    threadBuf_ = std::make_unique<uint8_t[]>(shardSize_);
    if (MetricsEnabled()) {
        static Metrics::CachedMetric threadCount{"compress_decompress_thread_count"};
        Metrics::UpdateStats(threadCount, static_cast<double>(decompressThreadNum));
    }
    if (!DetailedMetricsEnabled()) { return Status::OK(); }
    try {
        metricsThread_ = std::thread([this] { MetricsLoop(); });
    } catch (const std::exception& e) {
        return Status::Error(
            fmt::format("failed to start compressor metrics thread: {}", e.what()));
    }
    return Status::OK();
}

void CompressorAction::MetricsLoop()
{
    constexpr auto kReportInterval = std::chrono::milliseconds(100);
    (void)pthread_setname_np(pthread_self(), "ucm_cmp_metric");

    static Metrics::CachedMetric queueDepth{"compress_load_queue_depth"};
    static Metrics::CachedMetric queueHighWatermark{"compress_load_queue_high_watermark"};
    static Metrics::CachedMetric loadActive{"compress_load_active_workers"};
    static Metrics::CachedMetric backendWaitActive{"compress_backend_wait_active_workers"};
    static Metrics::CachedMetric decodeActive{"compress_decode_active_workers"};
    static Metrics::CachedMetric threadCount{"compress_decompress_thread_count"};

    while (!metricsStop_.load(std::memory_order_relaxed)) {
        Metrics::UpdateStats(queueDepth,
                             static_cast<double>(loadQueueDepth_.load(std::memory_order_relaxed)));
        Metrics::UpdateStats(
            queueHighWatermark,
            static_cast<double>(loadQueueHighWatermark_.load(std::memory_order_relaxed)));
        Metrics::UpdateStats(
            loadActive, static_cast<double>(loadActiveWorkers_.load(std::memory_order_relaxed)));
        Metrics::UpdateStats(
            backendWaitActive,
            static_cast<double>(backendWaitActiveWorkers_.load(std::memory_order_relaxed)));
        Metrics::UpdateStats(
            decodeActive,
            static_cast<double>(decodeActiveWorkers_.load(std::memory_order_relaxed)));
        Metrics::UpdateStats(threadCount, static_cast<double>(decompressThreadNum));

        std::unique_lock<std::mutex> lock(metricsMtx_);
        metricsCv_.wait_for(lock, kReportInterval,
                            [this] { return metricsStop_.load(std::memory_order_relaxed); });
    }
}

Status CompressorAction::WaitLoadBackend(Detail::TaskHandle taskHandle)
{
    if (!DetailedMetricsEnabled()) { return backend_->Wait(taskHandle); }

    static Metrics::CachedMetric waitSeconds{"compress_backend_wait_seconds_total"};
    static Metrics::CachedMetric waitDuration{"compress_backend_wait_duration_ms"};

    const auto start = SteadyClock::now();
    Status status = Status::OK();
    {
        ActivityGuard guard(&backendWaitActiveWorkers_);
        status = backend_->Wait(taskHandle);
    }
    const double seconds = SecondsSince(start);
    Metrics::UpdateStats(waitSeconds, seconds);
    Metrics::UpdateStats(waitDuration, seconds * 1e3);
    return status;
}

void CompressorAction::Push(TaskPtr task, WaiterPtr waiter)
{
    const char* type = (task->type == TransTask::Type::DUMP) ? "DUMP" : "LOAD";
    UC_DEBUG("Task Pushed | id: {}, type: {}, shards: {}", task->id, type, task->desc.size());

    waiter->Set(1);
    if (task->type == TransTask::Type::DUMP) {
        dump_pool_.Push(CompressTask{task, waiter, {}});
    } else if (task->type == TransTask::Type::LOAD) {
        SteadyClock::time_point enqueueTp{};
        if (DetailedMetricsEnabled()) {
            const size_t depth = loadQueueDepth_.fetch_add(1, std::memory_order_relaxed) + 1;
            UpdateHighWatermark(loadQueueHighWatermark_, depth);
            enqueueTp = SteadyClock::now();
        }
        load_pool_.Push(CompressTask{task, waiter, enqueueTp});
    }
}

void CompressorAction::Compress_Load(CompressTask& ct)
{
    static Metrics::CachedMetric loadTasks{"compress_load_tasks_total"};
    static Metrics::CachedMetric loadShards{"compress_load_shards_total"};
    static Metrics::CachedMetric queueWait{"compress_load_queue_wait_duration_ms"};
    static Metrics::CachedMetric decodeBytes{"compress_decode_bytes_total"};
    static Metrics::CachedMetric decodeBusySeconds{"compress_decode_busy_seconds_total"};
    static Metrics::CachedMetric decodeDuration{"compress_decode_duration_ms"};
    static Metrics::CachedMetric decodeBandwidth{"compress_decode_bandwidth_gbps"};

    const bool metricsEnabled = MetricsEnabled();
    const bool detailedMetricsEnabled = DetailedMetricsEnabled();
    if (detailedMetricsEnabled) {
        const auto pickedTp = SteadyClock::now();
        const size_t previousDepth = loadQueueDepth_.fetch_sub(1, std::memory_order_relaxed);
        if (previousDepth == 0) [[unlikely]] {
            loadQueueDepth_.store(0, std::memory_order_relaxed);
            UC_WARN("Compressor load queue depth underflow for task({}).", ct.task->id);
        }
        if (ct.enqueueTp != SteadyClock::time_point{}) {
            const double queuedSeconds =
                std::chrono::duration<double>(pickedTp - ct.enqueueTp).count();
            Metrics::UpdateStats(queueWait, queuedSeconds * 1e3);
        }
    }
    if (metricsEnabled) {
        Metrics::UpdateStats(loadTasks, 1.0);
        Metrics::UpdateStats(loadShards, static_cast<double>(ct.task->desc.size()));
    }
    ActivityGuard loadGuard(detailedMetricsEnabled ? &loadActiveWorkers_ : nullptr);

    UC_DEBUG("COMPRESS LOAD START | task_id: {}", ct.task->id);
    auto fail = [this, &ct](const char* stage, const Status& status) {
        UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, stage: {}, status: {}", ct.task->id, stage,
                 status);
        failureSet_->Insert(ct.task->id);
    };
    if (!codec_->NeedsDecompress()) {
        auto result = backend_->Load(std::move(ct.task->desc));
        if (!result) {
            fail("backend submit", result.Error());
        } else {
            auto status = WaitLoadBackend(result.Value());
            if (status.Failure()) { fail("backend wait", status); }
        }
        ct.waiter->Done();
        UC_DEBUG("COMPRESS LOAD END | task_id: {}", ct.task->id);
        return;
    }

    auto result = backend_->Load(ct.task->desc);
    if (!result) {
        fail("backend submit", result.Error());
        ct.waiter->Done();
        return;
    }
    if (result.Value() > 0) {
        auto status = WaitLoadBackend(result.Value());
        if (status.Failure()) {
            fail("backend wait", status);
            ct.waiter->Done();
            return;
        }
    }

    const auto& shards = ct.task->desc;
    UC_DEBUG("COMPRESS LOAD | shards_count: {}", shards.size());

    CodecModeCounts modeCounts;
    size_t decodedShards = 0;
    const auto decodeStart =
        detailedMetricsEnabled ? SteadyClock::now() : SteadyClock::time_point{};
    {
        ActivityGuard decodeGuard(detailedMetricsEnabled ? &decodeActiveWorkers_ : nullptr);
        for (const auto& shard : shards) {
            const CodecPayloadMode payloadMode =
                codec_->GetPayloadMode(shard.addrs[0], compressedShardSize_, shardSize_);
            const int err = codec_->DecompressInplace(shard.addrs[0], shardSize_);
            if (err != 0) {
                UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard: {}, error: {} ({})",
                         ct.task->id, shard.index, err, CodecErrorName(err));
                failureSet_->Insert(ct.task->id);
                continue;
            }
            if (metricsEnabled) { ++decodedShards; }
            modeCounts.Add(payloadMode);
            UC_DEBUG("COMPRESS LOAD | shard: {}, done, decompressed_size: {}", shard.index,
                     shardSize_);
        }
    }
    if (metricsEnabled) {
        const double decodedBytesValue = static_cast<double>(decodedShards) * shardSize_;
        Metrics::UpdateStats(decodeBytes, decodedBytesValue);
        if (detailedMetricsEnabled) {
            const double decodeSeconds = SecondsSince(decodeStart);
            Metrics::UpdateStats(decodeBusySeconds, decodeSeconds);
            Metrics::UpdateStats(decodeDuration, decodeSeconds * 1e3);
            if (decodeSeconds > 0.0) {
                Metrics::UpdateStats(decodeBandwidth, decodedBytesValue / decodeSeconds / 1e9);
            }
        }
    }
    ReportCodecModeStats(ct.task->id, CodecStatsStage::LOAD, modeCounts);

    ct.waiter->Done();
    UC_DEBUG("COMPRESS LOAD END | task_id: {}", ct.task->id);
}

void CompressorAction::Compress_Dump(CompressTask& ct)
{
    UC_DEBUG("COMPRESS DUMP START | task_id: {}", ct.task->id);
    auto fail = [this, &ct](const char* stage, const Status& status) {
        UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, stage: {}, status: {}", ct.task->id, stage,
                 status);
        failureSet_->Insert(ct.task->id);
    };

    if (!codec_->NeedsCompress()) {
        const auto n = ct.task->desc.size();
        if (n > 0) {
            auto result = backend_->Dump(std::move(ct.task->desc));
            if (!result) {
                fail("backend submit", result.Error());
            } else {
                auto status = backend_->Wait(result.Value());
                if (status.Failure()) { fail("backend wait", status); }
            }
        }
        ct.waiter->Done();
        UC_DEBUG("COMPRESS DUMP END | task_id: {}", ct.task->id);
        return;
    }

    const auto& desc = ct.task->desc;
    if (desc.empty()) {
        UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, desc is empty", ct.task->id);
        failureSet_->Insert(ct.task->id);
        ct.waiter->Done();
        return;
    }

    const size_t scratchSize = codec_->CompressScratchSize(shardSize_);
    Detail::TaskDesc backendDesc;
    backendDesc.brief = ct.task->desc.brief;
    std::vector<void*> blockToFree;
    auto dumpMemoryPool = std::make_unique<MemoryPool>(scratchSize, desc.size());

    CodecModeCounts modeCounts;
    for (const UC::Detail::Shard& shard : desc) {
        UC_DEBUG("COMPRESS DUMP | task_id: {}, shard: {}, compressing...", ct.task->id,
                 shard.index);

        auto* compressed = static_cast<uint8_t*>(dumpMemoryPool->allocate());
        const size_t compressedBytes = codec_->Compress(compressed, shard.addrs[0], shardSize_);
        if (compressedBytes != compressedShardSize_) [[unlikely]] {
            UC_ERROR(
                "COMPRESS DUMP FAILED | task_id: {}, shard: {}, expected {} B but codec produced "
                "{} B",
                ct.task->id, shard.index, compressedShardSize_, compressedBytes);
            dumpMemoryPool->deallocate({compressed});
            continue;
        }
        modeCounts.Add(codec_->GetPayloadMode(compressed, compressedBytes, shardSize_));

        std::vector<void*> addrs{static_cast<void*>(compressed)};
        backendDesc.push_back(Detail::Shard{shard.owner, shard.index, addrs});
        blockToFree.push_back(static_cast<void*>(compressed));
        UC_DEBUG("COMPRESS DUMP | shard: {}, done, stored_size: {}", shard.index, compressedBytes);
    }
    if (backendDesc.size() != desc.size()) {
        UC_ERROR(
            "COMPRESS DUMP FAILED | task_id: {}, only {}/{} shards met the compression budget; "
            "the whole dump is aborted",
            ct.task->id, backendDesc.size(), desc.size());
        failureSet_->Insert(ct.task->id);
        if (!blockToFree.empty()) { dumpMemoryPool->deallocate(blockToFree); }
        ct.waiter->Done();
        return;
    }

    bool backendDumpSucceeded = false;
    auto result = backend_->Dump(std::move(backendDesc));
    if (!result) {
        fail("backend submit", result.Error());
    } else if (result.Value() > 0) {
        auto status = backend_->Wait(result.Value());
        if (status.Failure()) {
            fail("backend wait", status);
        } else {
            backendDumpSucceeded = true;
        }
    } else {
        backendDumpSucceeded = true;
    }
    if (backendDumpSucceeded) {
        ReportCodecModeStats(ct.task->id, CodecStatsStage::DUMP, modeCounts);
    }
    if (!blockToFree.empty()) { dumpMemoryPool->deallocate(blockToFree); }

    ct.waiter->Done();
    UC_DEBUG("COMPRESS DUMP END | task_id: {}", ct.task->id);
}

}  // namespace UC::Compressor
