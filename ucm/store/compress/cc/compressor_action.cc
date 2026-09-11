#include "compressor_action.h"
#include <chrono>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <pthread.h>
#include "logger/logger.h"
#include "metrics_api.h"

namespace UC::Compressor {
namespace {

using SteadyClock = std::chrono::steady_clock;

class CompletionGuard {
public:
    explicit CompletionGuard(std::shared_ptr<Latch> waiter) : waiter_(std::move(waiter)) {}
    ~CompletionGuard() { waiter_->Done(); }

    CompletionGuard(const CompletionGuard&) = delete;
    CompletionGuard& operator=(const CompletionGuard&) = delete;

private:
    std::shared_ptr<Latch> waiter_;
};

class AlignedBlocks {
public:
    AlignedBlocks(size_t blockSize, size_t blockCount)
        : data_(nullptr, &std::free), stride_(AlignedSize(blockSize))
    {
        if (blockCount == 0 || stride_ > std::numeric_limits<size_t>::max() / blockCount) {
            throw std::bad_alloc();
        }
        void* data = nullptr;
        if (posix_memalign(&data, ALIGNMENT, stride_ * blockCount) != 0) { throw std::bad_alloc(); }
        data_.reset(data);
    }

    void* At(size_t index) const noexcept
    {
        return static_cast<uint8_t*>(data_.get()) + index * stride_;
    }

private:
    static constexpr size_t ALIGNMENT = 4096;
    using Buffer = std::unique_ptr<void, decltype(&std::free)>;

    static size_t AlignedSize(size_t size)
    {
        if (size == 0 || size > std::numeric_limits<size_t>::max() - (ALIGNMENT - 1)) {
            throw std::bad_alloc();
        }
        return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    Buffer data_;
    size_t stride_;
};

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
    compressThreadNum_ = config.compressThreadNum;
    decompressThreadNum_ = config.decompressThreadNum;
    timeoutMs_ = config.timeoutMs;
    metricsLevel_ = config.metricsLevel == "off"     ? MetricsLevel::OFF
                    : config.metricsLevel == "basic" ? MetricsLevel::BASIC
                                                     : MetricsLevel::DETAILED;

    const auto ratio = static_cast<FixedRatio>(config.compressRatio);
    const auto dataType = config.dataType;
    codec_ = ratio == R160 && dataType == DT_BF16
                 ? MakeR160BaseCodec(compressedShardSize_)
                 : MakeCodec(ratio, dataType, compressedShardSize_);
    if (!codec_) {
        return Status::InvalidParam("Unsupported codec combo (ratio={}, dtype={})",
                                    config.compressRatio, static_cast<int32_t>(config.dataType));
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
            compressedShardSize_, shardSize_, config.compressRatio,
            static_cast<int32_t>(config.dataType));
    }
    if (codec_->NeedsCompress() && compressedShardSize_ % 4096 != 0) {
        return Status::InvalidParam(
            "compressed shard size({}) must be 4096-byte aligned for shardSize({}) and ratio({})",
            compressedShardSize_, shardSize_, config.compressRatio);
    }

    const bool dumpPoolStarted = dumpPool_.SetNWorker(compressThreadNum_)
                                     .SetCpuAffinity(config.cpuAffinityCores)
                                     .SetWorkerFn([this](auto& task, auto&) { DumpWorker(task); })
                                     .Run();
    if (!dumpPoolStarted) {
        return Status::Error(fmt::format("compress workers({}) start failed", compressThreadNum_));
    }

    const bool loadPoolStarted = loadPool_.SetNWorker(decompressThreadNum_)
                                     .SetCpuAffinity(config.cpuAffinityCores)
                                     .SetWorkerFn([this](auto& task, auto&) { LoadWorker(task); })
                                     .Run();
    if (!loadPoolStarted) {
        return Status::Error(
            fmt::format("decompress workers({}) start failed", decompressThreadNum_));
    }

    UC_DEBUG(
        "Compressor Setup OK | dump_threads: {}, load_threads: {}, shard_size: {} B, "
        "stored_shard_size: {} B",
        compressThreadNum_, decompressThreadNum_, shardSize_, compressedShardSize_);

    if (MetricsEnabled()) {
        static Metrics::CachedMetric threadCount{"compress_decompress_thread_count"};
        Metrics::UpdateStats(threadCount, static_cast<double>(decompressThreadNum_));
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
    const bool isLoad = task->type == TransTask::Type::LOAD;
    SteadyClock::time_point enqueueTp{};
    if (isLoad && DetailedMetricsEnabled()) {
        const size_t depth = loadQueueDepth_.fetch_add(1, std::memory_order_relaxed) + 1;
        UpdateHighWatermark(loadQueueHighWatermark_, depth);
        enqueueTp = SteadyClock::now();
    }
    try {
        if (isLoad) {
            loadPool_.Push(CompressTask{task, waiter, enqueueTp});
        } else {
            dumpPool_.Push(CompressTask{task, waiter, {}});
        }
    } catch (const std::exception& e) {
        if (isLoad) { DecrementLoadQueueDepth(task->id); }
        FailTask(task, type, "queue submit", Status::Error(e.what()));
        waiter->Done();
    } catch (...) {
        if (isLoad) { DecrementLoadQueueDepth(task->id); }
        FailTask(task, type, "queue submit", Status::Error("unknown exception"));
        waiter->Done();
    }
}

void CompressorAction::Cancel(TaskPtr task)
{
    auto& pool = task->type == TransTask::Type::LOAD ? loadPool_ : dumpPool_;
    pool.TraverseWaitQueue(
        [this, id = task->id](CompressTask& queued) {
            return queued.task->id == id || queued.waiter->IsTimeout(timeoutMs_);
        },
        [this](CompressTask& queued) { FinishCancelledTask(queued); }, nullptr);
}

void CompressorAction::FailTask(const TaskPtr& task, const char* operation, const char* stage,
                                const Status& status)
{
    UC_ERROR("COMPRESS {} FAILED | task_id: {}, stage: {}, status: {}", operation, task->id, stage,
             status);
    task->Fail(status);
    failureSet_->Insert(task->id);
}

void CompressorAction::FinishCancelledTask(CompressTask& task)
{
    if (task.task->type == TransTask::Type::LOAD) { DecrementLoadQueueDepth(task.task->id); }
    task.task->Fail(Status::Timeout());
    failureSet_->Insert(task.task->id);
    task.waiter->Done();
}

void CompressorAction::DecrementLoadQueueDepth(Detail::TaskHandle taskHandle)
{
    if (!DetailedMetricsEnabled()) { return; }
    const size_t previousDepth = loadQueueDepth_.fetch_sub(1, std::memory_order_relaxed);
    if (previousDepth == 0) [[unlikely]] {
        loadQueueDepth_.store(0, std::memory_order_relaxed);
        UC_WARN("Compressor load queue depth underflow for task({}).", taskHandle);
    }
}

void CompressorAction::LoadWorker(CompressTask& ct)
{
    CompletionGuard completion{ct.waiter};
    try {
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
            DecrementLoadQueueDepth(ct.task->id);
            if (ct.enqueueTp != SteadyClock::time_point{}) {
                const double queuedSeconds =
                    std::chrono::duration<double>(pickedTp - ct.enqueueTp).count();
                Metrics::UpdateStats(queueWait, queuedSeconds * 1e3);
            }
        }
        if (failureSet_->Contains(ct.task->id)) {
            ct.task->Fail(Status::Timeout());
            return;
        }
        if (metricsEnabled) {
            Metrics::UpdateStats(loadTasks, 1.0);
            Metrics::UpdateStats(loadShards, static_cast<double>(ct.task->desc.size()));
        }
        ActivityGuard loadGuard(detailedMetricsEnabled ? &loadActiveWorkers_ : nullptr);

        UC_DEBUG("COMPRESS LOAD START | task_id: {}", ct.task->id);
        if (!codec_->NeedsDecompress()) {
            auto result = backend_->Load(std::move(ct.task->desc));
            if (!result) {
                FailTask(ct.task, "LOAD", "backend submit", result.Error());
                return;
            }
            if (result.Value() > 0) {
                auto status = WaitLoadBackend(result.Value());
                if (status.Failure()) {
                    FailTask(ct.task, "LOAD", "backend wait", status);
                    return;
                }
            }
            UC_DEBUG("COMPRESS LOAD END | task_id: {}", ct.task->id);
            return;
        }

        auto result = backend_->Load(ct.task->desc);
        if (!result) {
            FailTask(ct.task, "LOAD", "backend submit", result.Error());
            return;
        }
        if (result.Value() > 0) {
            auto status = WaitLoadBackend(result.Value());
            if (status.Failure()) {
                FailTask(ct.task, "LOAD", "backend wait", status);
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
                    detailedMetricsEnabled
                        ? codec_->GetPayloadMode(shard.addrs.front(), compressedShardSize_,
                                                 shardSize_)
                        : CodecPayloadMode::NOT_APPLICABLE;
                const int err = codec_->DecompressInplace(shard.addrs.front(), shardSize_);
                if (err != 0) {
                    FailTask(ct.task, "LOAD", "decompress",
                             Status::Error(fmt::format("shard({}) codec error {} ({})", shard.index,
                                                       err, CodecErrorName(err))));
                    return;
                }
                if (metricsEnabled) { ++decodedShards; }
                if (detailedMetricsEnabled) { modeCounts.Add(payloadMode); }
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
        if (detailedMetricsEnabled) {
            ReportCodecModeStats(ct.task->id, CodecStatsStage::LOAD, modeCounts);
        }
        UC_DEBUG("COMPRESS LOAD END | task_id: {}", ct.task->id);
    } catch (const std::exception& e) {
        FailTask(ct.task, "LOAD", "worker exception", Status::Error(e.what()));
    } catch (...) {
        FailTask(ct.task, "LOAD", "worker exception", Status::Error("unknown exception"));
    }
}

void CompressorAction::DumpWorker(CompressTask& ct)
{
    CompletionGuard completion{ct.waiter};
    try {
        UC_DEBUG("COMPRESS DUMP START | task_id: {}", ct.task->id);
        if (failureSet_->Contains(ct.task->id)) {
            ct.task->Fail(Status::Timeout());
            return;
        }
        if (!codec_->NeedsCompress()) {
            auto result = backend_->Dump(std::move(ct.task->desc));
            if (!result) {
                FailTask(ct.task, "DUMP", "backend submit", result.Error());
                return;
            }
            if (result.Value() > 0) {
                auto status = backend_->Wait(result.Value());
                if (status.Failure()) {
                    FailTask(ct.task, "DUMP", "backend wait", status);
                    return;
                }
            }
            UC_DEBUG("COMPRESS DUMP END | task_id: {}", ct.task->id);
            return;
        }

        const auto& desc = ct.task->desc;
        const size_t scratchSize = codec_->CompressScratchSize(shardSize_);
        AlignedBlocks buffers{scratchSize, desc.size()};
        Detail::TaskDesc backendDesc;
        backendDesc.brief = desc.brief;
        backendDesc.reserve(desc.size());

        CodecModeCounts modeCounts;
        for (size_t i = 0; i < desc.size(); ++i) {
            const auto& shard = desc[i];
            UC_DEBUG("COMPRESS DUMP | task_id: {}, shard: {}, compressing...", ct.task->id,
                     shard.index);

            void* compressed = buffers.At(i);
            const size_t compressedBytes =
                codec_->Compress(compressed, shard.addrs.front(), shardSize_);
            if (compressedBytes != compressedShardSize_) [[unlikely]] {
                FailTask(
                    ct.task, "DUMP", "compress",
                    Status::Error(fmt::format("shard({}) expected {} B but codec produced {} B",
                                              shard.index, compressedShardSize_, compressedBytes)));
                return;
            }
            if (DetailedMetricsEnabled()) {
                modeCounts.Add(codec_->GetPayloadMode(compressed, compressedBytes, shardSize_));
            }

            backendDesc.push_back(Detail::Shard{shard.owner, shard.index, {compressed}});
            UC_DEBUG("COMPRESS DUMP | shard: {}, done, stored_size: {}", shard.index,
                     compressedBytes);
        }
        auto result = backend_->Dump(std::move(backendDesc));
        if (!result) {
            FailTask(ct.task, "DUMP", "backend submit", result.Error());
            return;
        }
        if (result.Value() > 0) {
            auto status = backend_->Wait(result.Value());
            if (status.Failure()) {
                FailTask(ct.task, "DUMP", "backend wait", status);
                return;
            }
        }
        if (DetailedMetricsEnabled()) {
            ReportCodecModeStats(ct.task->id, CodecStatsStage::DUMP, modeCounts);
        }
        UC_DEBUG("COMPRESS DUMP END | task_id: {}", ct.task->id);
    } catch (const std::exception& e) {
        FailTask(ct.task, "DUMP", "worker exception", Status::Error(e.what()));
    } catch (...) {
        FailTask(ct.task, "DUMP", "worker exception", Status::Error("unknown exception"));
    }
}

}  // namespace UC::Compressor
