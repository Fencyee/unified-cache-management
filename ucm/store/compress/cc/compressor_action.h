#ifndef UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H
#define UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include "codec.h"
#include "global_config.h"
#include "memory_pool.h"
#include "template/hashset.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::Compressor {

class CompressorAction {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using FailureSet = HashSet<Detail::TaskHandle>;

    struct CompressTask {
        TaskPtr task;
        WaiterPtr waiter;
    };

    struct LoadAggregate {
        std::atomic<size_t> remaining;
        std::atomic<size_t> r160HighPrecision{0};
        std::atomic<size_t> r160Quantized{0};
        std::atomic<size_t> r200Tunstall{0};
        std::atomic<size_t> r200Fp8Fallback{0};

        explicit LoadAggregate(size_t nShard) : remaining{nShard} {}
    };
    using LoadAggregatePtr = std::shared_ptr<LoadAggregate>;

    struct LoadShardCtx {
        TaskPtr task;
        WaiterPtr waiter;
        LoadAggregatePtr aggregate;
        size_t shardIndex{0};
    };

    struct LoadWaitCtx {
        LoadShardCtx shard;
        Detail::TaskHandle backendHandle{0};
    };

private:
    StoreV1* backend_{nullptr};
    FailureSet* failureSet_{nullptr};
    size_t shardSize_{0};
    size_t compressedShardSize_{0};
    size_t decompressThreadNum_{6};
    static constexpr size_t kMaxActiveLoads = 128;
    static constexpr size_t kMaxOutstandingWork = 8192;
    std::unique_ptr<Codec> codec_;

    // Declaration order is intentional. Once the destructor drains all accepted work,
    // ThreadPool members are destroyed in reverse pipeline order: submit, wait, decode, dump.
    ThreadPool<CompressTask> dumpPool_;
    ThreadPool<LoadShardCtx> decodePool_;
    ThreadPool<LoadWaitCtx> waitPool_;
    ThreadPool<LoadShardCtx> submitPool_;

    std::mutex lifecycleMtx_;
    std::condition_variable lifecycleCv_;
    size_t outstandingWork_{0};
    bool accepting_{true};

    std::mutex activeLoadMtx_;
    std::condition_variable activeLoadCv_;
    size_t activeLoads_{0};

public:
    ~CompressorAction();
    Status Setup(const Config& config, FailureSet* failureSet);
    void Push(TaskPtr task, WaiterPtr waiter);

private:
    void ProcessDump(CompressTask& task) noexcept;
    void CompressDump(CompressTask& task);

    void SubmitLoadShard(LoadShardCtx& ctx) noexcept;
    void WaitLoadShard(LoadWaitCtx& ctx) noexcept;
    void DecodeLoadShard(LoadShardCtx& ctx) noexcept;

    bool RegisterWork(size_t count);
    void CompleteWork() noexcept;
    void AcquireLoadSlot();
    void ReleaseLoadSlot() noexcept;
    void FinishLoadShard(const LoadShardCtx& ctx, CodecPayloadMode mode,
                         bool holdsLoadSlot) noexcept;
    void RecordPayloadMode(const LoadAggregatePtr& aggregate, CodecPayloadMode mode) noexcept;
    void ReportLoadModeStats(const LoadShardCtx& ctx, CodecPayloadMode mode) const noexcept;
    Status WaitBackendHandle(Detail::TaskHandle backendHandle, Detail::TaskHandle taskId,
                             const char* stage) noexcept;
    bool IsTaskFailed(Detail::TaskHandle taskId) const noexcept;
    void MarkTaskFailed(Detail::TaskHandle taskId) noexcept;
};

}  // namespace UC::Compressor

#endif
