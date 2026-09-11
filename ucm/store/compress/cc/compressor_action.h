#ifndef UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H
#define UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "codec.h"
#include "global_config.h"
#include "template/hashset.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::Compressor {

class CompressorAction {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;

private:
    enum class MetricsLevel { OFF, BASIC, DETAILED };

    StoreV1* backend_{nullptr};
    HashSet<Detail::TaskHandle>* failureSet_{nullptr};
    size_t shardSize_{0};
    size_t compressedShardSize_{0};
    size_t compressThreadNum_{4};
    size_t decompressThreadNum_{6};
    size_t timeoutMs_{30000};
    MetricsLevel metricsLevel_{MetricsLevel::BASIC};
    std::unique_ptr<Codec> codec_;

    struct CompressTask {
        std::shared_ptr<TransTask> task;
        std::shared_ptr<Latch> waiter;
        std::chrono::steady_clock::time_point enqueueTp{};
    };
    ThreadPool<CompressTask> dumpPool_;
    ThreadPool<CompressTask> loadPool_;

    std::atomic<size_t> loadQueueDepth_{0};
    std::atomic<size_t> loadQueueHighWatermark_{0};
    std::atomic<size_t> loadActiveWorkers_{0};
    std::atomic<size_t> backendWaitActiveWorkers_{0};
    std::atomic<size_t> decodeActiveWorkers_{0};
    std::atomic_bool metricsStop_{false};
    std::mutex metricsMtx_;
    std::condition_variable metricsCv_;
    std::thread metricsThread_;

public:
    ~CompressorAction();
    Status Setup(const Config& config, HashSet<Detail::TaskHandle>* failureSet);
    void Push(TaskPtr task, WaiterPtr waiter);
    void Cancel(TaskPtr task);

private:
    void DumpWorker(CompressTask& task);
    void LoadWorker(CompressTask& task);
    Status WaitLoadBackend(Detail::TaskHandle taskHandle);
    void FailTask(const TaskPtr& task, const char* operation, const char* stage,
                  const Status& status);
    void FinishCancelledTask(CompressTask& task);
    void DecrementLoadQueueDepth(Detail::TaskHandle taskHandle);
    void MetricsLoop();
    bool MetricsEnabled() const noexcept { return metricsLevel_ != MetricsLevel::OFF; }
    bool DetailedMetricsEnabled() const noexcept { return metricsLevel_ == MetricsLevel::DETAILED; }
};

}  // namespace UC::Compressor

#endif
