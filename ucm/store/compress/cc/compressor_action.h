#ifndef UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H
#define UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unistd.h>
#include "codec.h"
#include "global_config.h"
#include "memory_pool.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::Compressor {

class CompressorAction {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;

    struct ShardTask {
        Detail::TaskHandle taskHandle;
        Detail::Shard* shard;
        Detail::TaskHandle backendTaskHandle;
        WaiterPtr waiter;
    };

private:
    enum class MetricsLevel { OFF, BASIC, DETAILED };

    StoreV1* backend_{nullptr};
    HashSet<Detail::TaskHandle>* failureSet_{nullptr};
    size_t shardSize_{0};
    size_t compressedShardSize_{0};
    size_t decompressThreadNum{6};
    MetricsLevel metricsLevel_{MetricsLevel::BASIC};
    std::unique_ptr<Codec> codec_;

    struct CompressTask {
        std::shared_ptr<TransTask> task;
        std::shared_ptr<Latch> waiter;
        std::chrono::steady_clock::time_point enqueueTp{};
    };
    ThreadPool<CompressTask> dump_pool_;
    ThreadPool<CompressTask> load_pool_;
    std::unique_ptr<uint8_t[]> threadBuf_{0};

    std::atomic<size_t> loadQueueDepth_{0};
    std::atomic<size_t> loadQueueHighWatermark_{0};
    std::atomic<size_t> loadActiveWorkers_{0};
    std::atomic<size_t> backendWaitActiveWorkers_{0};
    std::atomic<size_t> decodeActiveWorkers_{0};
    std::atomic_bool metricsStop_{false};
    std::mutex metricsMtx_;
    std::condition_variable metricsCv_;
    std::thread metricsThread_;

    alignas(64) std::atomic_bool stop_{false};
    Detail::TaskHandle finishedBackendTaskHandle_{0};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<ShardTask> running_;
    std::thread dispatcher_;
    std::thread transfer_;

    std::mutex waiterMtx_;
    std::mutex backendMtx_;
    std::atomic<Detail::TaskHandle> backendTaskHandle_{0};
    std::condition_variable cv_;

public:
    ~CompressorAction();
    Status Setup(const Config& config, HashSet<Detail::TaskHandle>* failureSet);
    void Push(TaskPtr task, WaiterPtr waiter);

private:
    void Compress_Dump(CompressTask& ios);
    void Compress_Load(CompressTask& ios);
    Status WaitLoadBackend(Detail::TaskHandle taskHandle);
    void MetricsLoop();
    bool MetricsEnabled() const noexcept { return metricsLevel_ != MetricsLevel::OFF; }
    bool DetailedMetricsEnabled() const noexcept { return metricsLevel_ == MetricsLevel::DETAILED; }

    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    void TransferOneTask(ShardTask&& task);
};

}  // namespace UC::Compressor

#endif
