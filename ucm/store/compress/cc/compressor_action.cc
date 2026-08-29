#include "compressor_action.h"
#include <algorithm>
#include <exception>
#include <vector>
#include "logger/logger.h"

namespace UC::Compressor {
namespace {

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
    // ThreadPool stops immediately in its destructor and may leave queued work behind. Keep every
    // stage alive until all tasks accepted by Push have reached their terminal state.
    std::unique_lock<std::mutex> lock(lifecycleMtx_);
    accepting_ = false;
    lifecycleCv_.wait(lock, [this] { return outstandingWork_ == 0; });
}

Status CompressorAction::Setup(const Config& config, FailureSet* failureSet)
{
    backend_ = config.storeBackend;
    failureSet_ = failureSet;
    shardSize_ = config.shardSize;
    compressedShardSize_ = config.compressedShardSize;
    decompressThreadNum_ = config.decompressThreadNum;

    if (backend_ == nullptr) { return Status::InvalidParam("invalid store backend"); }
    if (decompressThreadNum_ == 0) {
        return Status::InvalidParam("decompress_thread_num must be greater than zero");
    }
    codec_ = MakeCodec(static_cast<FixedRatio>(config.compressRatio),
                       static_cast<DataType>(config.dataType), compressedShardSize_);
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

    const size_t dumpThreadNum = std::max<size_t>(1, config.streamNumber >> 1U);
    auto success = dumpPool_.SetNWorker(dumpThreadNum)
                       .SetCpuAffinity(config.cpuAffinityCores)
                       .SetWorkerFn([this](auto& task, auto&) { ProcessDump(task); })
                       .Run();
    if (!success) { return Status::Error("Failed to start compress dump worker pool"); }

    success = decodePool_.SetNWorker(decompressThreadNum_)
                  .SetCpuAffinity(config.cpuAffinityCores)
                  .SetWorkerFn([this](auto& ctx, auto&) { DecodeLoadShard(ctx); })
                  .Run();
    if (!success) { return Status::Error("Failed to start decompress worker pool"); }

    // Posix AIO performs the actual reads concurrently. A single ordered Wait stage mirrors the
    // Cache transfer stage and avoids creating another decompressThreadNum blocking threads.
    success = waitPool_.SetNWorker(1)
                  .SetCpuAffinity(config.cpuAffinityCores)
                  .SetWorkerFn([this](auto& ctx, auto&) { WaitLoadShard(ctx); })
                  .Run();
    if (!success) { return Status::Error("Failed to start backend wait worker pool"); }

    success = submitPool_.SetNWorker(1)
                  .SetCpuAffinity(config.cpuAffinityCores)
                  .SetWorkerFn([this](auto& ctx, auto&) { SubmitLoadShard(ctx); })
                  .Run();
    if (!success) { return Status::Error("Failed to start backend load submit worker pool"); }

    UC_INFO(
        "Compressor Setup | load_pipeline=submit(1)->wait(1)->decompress({}), "
        "max_active_loads={}, max_outstanding_work={}, shard_size={} B, stored_shard_size={} B",
        decompressThreadNum_, kMaxActiveLoads, kMaxOutstandingWork, shardSize_,
        compressedShardSize_);
    return Status::OK();
}

void CompressorAction::Push(TaskPtr task, WaiterPtr waiter)
{
    const char* type = (task->type == TransTask::Type::DUMP) ? "DUMP" : "LOAD";
    UC_DEBUG("Task Pushed | id: {}, type: {}, shards: {}", task->id, type, task->desc.size());

    if (task->type == TransTask::Type::DUMP) {
        waiter->Set(1);
        if (!RegisterWork(1)) {
            MarkTaskFailed(task->id);
            waiter->Done();
            return;
        }
        try {
            dumpPool_.Push(CompressTask{task, waiter});
        } catch (const std::exception& e) {
            UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, stage: enqueue, error: {}", task->id,
                     e.what());
            MarkTaskFailed(task->id);
            waiter->Done();
            CompleteWork();
        } catch (...) {
            UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, stage: enqueue, unknown error", task->id);
            MarkTaskFailed(task->id);
            waiter->Done();
            CompleteWork();
        }
        return;
    }

    const size_t nShard = task->desc.size();
    if (nShard == 0) [[unlikely]] {
        waiter->Set(1);
        UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, desc is empty", task->id);
        MarkTaskFailed(task->id);
        waiter->Done();
        return;
    }

    waiter->Set(nShard);
    LoadAggregatePtr aggregate;
    if (nShard > 1) {
        try {
            aggregate = std::make_shared<LoadAggregate>(nShard);
        } catch (const std::exception& e) {
            UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, stage: state allocation, error: {}",
                     task->id, e.what());
            MarkTaskFailed(task->id);
            for (size_t i = 0; i < nShard; ++i) { waiter->Done(); }
            return;
        }
    }

    if (!RegisterWork(nShard)) {
        UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, pipeline is stopping or queue is full",
                 task->id);
        MarkTaskFailed(task->id);
        for (size_t i = 0; i < nShard; ++i) { waiter->Done(); }
        return;
    }

    for (size_t i = 0; i < nShard; ++i) {
        try {
            submitPool_.Push(LoadShardCtx{task, waiter, aggregate, i});
        } catch (const std::exception& e) {
            UC_ERROR(
                "COMPRESS LOAD FAILED | task_id: {}, shard: {}, stage: enqueue submit, "
                "error: {}",
                task->id, task->desc[i].index, e.what());
            MarkTaskFailed(task->id);
            for (size_t pending = i; pending < nShard; ++pending) {
                FinishLoadShard(LoadShardCtx{task, waiter, aggregate, pending},
                                CodecPayloadMode::NOT_APPLICABLE, false);
            }
            return;
        } catch (...) {
            UC_ERROR(
                "COMPRESS LOAD FAILED | task_id: {}, shard: {}, stage: enqueue submit, "
                "unknown error",
                task->id, task->desc[i].index);
            MarkTaskFailed(task->id);
            for (size_t pending = i; pending < nShard; ++pending) {
                FinishLoadShard(LoadShardCtx{task, waiter, aggregate, pending},
                                CodecPayloadMode::NOT_APPLICABLE, false);
            }
            return;
        }
    }
}

void CompressorAction::ProcessDump(CompressTask& task) noexcept
{
    try {
        CompressDump(task);
    } catch (const std::exception& e) {
        UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, unhandled error: {}", task.task->id,
                 e.what());
        MarkTaskFailed(task.task->id);
        task.waiter->Done();
    } catch (...) {
        UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, unknown unhandled error", task.task->id);
        MarkTaskFailed(task.task->id);
        task.waiter->Done();
    }
    CompleteWork();
}

void CompressorAction::CompressDump(CompressTask& ct)
{
    UC_DEBUG("COMPRESS DUMP START | task_id: {}", ct.task->id);
    auto fail = [this, &ct](const char* stage, const Status& status) {
        UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, stage: {}, status: {}", ct.task->id, stage,
                 status);
        MarkTaskFailed(ct.task->id);
    };

    if (!codec_->NeedsCompress()) {
        if (!ct.task->desc.empty()) {
            auto result = backend_->Dump(std::move(ct.task->desc));
            if (!result) {
                fail("backend submit", result.Error());
            } else {
                auto status = WaitBackendHandle(result.Value(), ct.task->id, "DumpWorker");
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
        MarkTaskFailed(ct.task->id);
        ct.waiter->Done();
        return;
    }

    const size_t scratchSize = codec_->CompressScratchSize(shardSize_);
    Detail::TaskDesc backendDesc;
    backendDesc.brief = desc.brief;
    std::vector<void*> blockToFree;
    auto dumpMemoryPool = std::make_unique<MemoryPool>(scratchSize, desc.size());

    CodecModeCounts modeCounts;
    for (const Detail::Shard& shard : desc) {
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
        backendDesc.push_back(Detail::Shard{shard.owner, shard.index, {compressed}});
        blockToFree.push_back(compressed);
    }

    if (backendDesc.size() != desc.size()) {
        UC_ERROR(
            "COMPRESS DUMP FAILED | task_id: {}, only {}/{} shards met the compression budget; "
            "the whole dump is aborted",
            ct.task->id, backendDesc.size(), desc.size());
        MarkTaskFailed(ct.task->id);
        if (!blockToFree.empty()) { dumpMemoryPool->deallocate(blockToFree); }
        ct.waiter->Done();
        return;
    }

    bool backendDumpSucceeded = false;
    auto result = backend_->Dump(std::move(backendDesc));
    if (!result) {
        fail("backend submit", result.Error());
    } else {
        auto status = WaitBackendHandle(result.Value(), ct.task->id, "DumpWorker");
        if (status.Failure()) {
            fail("backend wait", status);
        } else {
            backendDumpSucceeded = true;
        }
    }
    if (backendDumpSucceeded) {
        ReportCodecModeStats(ct.task->id, CodecStatsStage::DUMP, modeCounts);
    }
    if (!blockToFree.empty()) { dumpMemoryPool->deallocate(blockToFree); }

    ct.waiter->Done();
    UC_DEBUG("COMPRESS DUMP END | task_id: {}", ct.task->id);
}

void CompressorAction::SubmitLoadShard(LoadShardCtx& ctx) noexcept
{
    const auto taskId = ctx.task->id;
    Detail::TaskHandle backendHandle{0};
    bool holdsLoadSlot = false;

    try {
        if (IsTaskFailed(taskId)) {
            FinishLoadShard(ctx, CodecPayloadMode::NOT_APPLICABLE, false);
            return;
        }

        AcquireLoadSlot();
        holdsLoadSlot = true;
        if (IsTaskFailed(taskId)) {
            FinishLoadShard(ctx, CodecPayloadMode::NOT_APPLICABLE, true);
            return;
        }

        if (ctx.shardIndex >= ctx.task->desc.size()) [[unlikely]] {
            UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard offset: {}, invalid descriptor",
                     taskId, ctx.shardIndex);
            MarkTaskFailed(taskId);
            FinishLoadShard(ctx, CodecPayloadMode::INVALID, true);
            return;
        }

        const auto& shard = ctx.task->desc[ctx.shardIndex];
        if (shard.addrs.empty() || shard.addrs[0] == nullptr) [[unlikely]] {
            UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard: {}, invalid destination", taskId,
                     shard.index);
            MarkTaskFailed(taskId);
            FinishLoadShard(ctx, CodecPayloadMode::INVALID, true);
            return;
        }

        Detail::TaskDesc backendDesc{shard};
        backendDesc.brief = ctx.task->desc.brief;
        backendDesc.prerequisiteHandle = ctx.task->desc.prerequisiteHandle;
        auto result = backend_->Load(std::move(backendDesc));
        if (!result) [[unlikely]] {
            UC_ERROR(
                "COMPRESS LOAD FAILED | task_id: {}, shard: {}, stage: backend submit, "
                "status: {}",
                taskId, shard.index, result.Error());
            MarkTaskFailed(taskId);
            FinishLoadShard(ctx, CodecPayloadMode::NOT_APPLICABLE, true);
            return;
        }

        backendHandle = result.Value();
        if (backendHandle == 0) [[unlikely]] {
            UC_ERROR(
                "COMPRESS LOAD FAILED | task_id: {}, shard: {}, stage: backend submit, "
                "invalid task handle 0",
                taskId, shard.index);
            MarkTaskFailed(taskId);
            FinishLoadShard(ctx, CodecPayloadMode::NOT_APPLICABLE, true);
            return;
        }

        waitPool_.Push(LoadWaitCtx{ctx, backendHandle});
        return;
    } catch (const std::exception& e) {
        UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard offset: {}, stage: submit, error: {}",
                 taskId, ctx.shardIndex, e.what());
    } catch (...) {
        UC_ERROR(
            "COMPRESS LOAD FAILED | task_id: {}, shard offset: {}, stage: submit, "
            "unknown error",
            taskId, ctx.shardIndex);
    }

    MarkTaskFailed(taskId);
    if (backendHandle > 0) {
        // The handle was created but could not be handed to the Wait stage. Reap it before the
        // Cache buffer can be released or reused.
        (void)WaitBackendHandle(backendHandle, taskId, "LoadSubmitWorker recovery");
    }
    FinishLoadShard(ctx, CodecPayloadMode::NOT_APPLICABLE, holdsLoadSlot);
}

void CompressorAction::WaitLoadShard(LoadWaitCtx& ctx) noexcept
{
    const auto& shardCtx = ctx.shard;
    const auto taskId = shardCtx.task->id;
    try {
        // Always reap a submitted handle, even if another shard has already failed or the outer
        // task has timed out. The backend may still be writing the shared Cache buffer.
        const auto status = WaitBackendHandle(ctx.backendHandle, taskId, "LoadWaitWorker");
        if (status.Failure()) [[unlikely]] {
            const auto shardIndex = shardCtx.task->desc[shardCtx.shardIndex].index;
            UC_ERROR(
                "COMPRESS LOAD FAILED | task_id: {}, shard: {}, stage: backend wait, "
                "status: {}",
                taskId, shardIndex, status);
            MarkTaskFailed(taskId);
            FinishLoadShard(shardCtx, CodecPayloadMode::NOT_APPLICABLE, true);
            return;
        }

        if (IsTaskFailed(taskId) || !codec_->NeedsDecompress()) {
            FinishLoadShard(shardCtx, CodecPayloadMode::NOT_APPLICABLE, true);
            return;
        }

        decodePool_.Push(LoadShardCtx{shardCtx});
        return;
    } catch (const std::exception& e) {
        UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard offset: {}, stage: wait, error: {}",
                 taskId, shardCtx.shardIndex, e.what());
    } catch (...) {
        UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard offset: {}, stage: wait, unknown error",
                 taskId, shardCtx.shardIndex);
    }

    MarkTaskFailed(taskId);
    FinishLoadShard(shardCtx, CodecPayloadMode::NOT_APPLICABLE, true);
}

void CompressorAction::DecodeLoadShard(LoadShardCtx& ctx) noexcept
{
    const auto taskId = ctx.task->id;
    CodecPayloadMode payloadMode = CodecPayloadMode::NOT_APPLICABLE;
    try {
        if (IsTaskFailed(taskId)) {
            FinishLoadShard(ctx, payloadMode, true);
            return;
        }

        if (ctx.shardIndex >= ctx.task->desc.size()) [[unlikely]] {
            UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard offset: {}, invalid descriptor",
                     taskId, ctx.shardIndex);
            MarkTaskFailed(taskId);
            FinishLoadShard(ctx, CodecPayloadMode::INVALID, true);
            return;
        }

        const auto& shard = ctx.task->desc[ctx.shardIndex];
        if (shard.addrs.empty() || shard.addrs[0] == nullptr) [[unlikely]] {
            UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard: {}, invalid destination", taskId,
                     shard.index);
            MarkTaskFailed(taskId);
            FinishLoadShard(ctx, CodecPayloadMode::INVALID, true);
            return;
        }

        payloadMode = codec_->GetPayloadMode(shard.addrs[0], compressedShardSize_, shardSize_);
        const int err = codec_->DecompressInplace(shard.addrs[0], shardSize_);
        if (err != 0) [[unlikely]] {
            UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard: {}, error: {} ({})", taskId,
                     shard.index, err, CodecErrorName(err));
            MarkTaskFailed(taskId);
            FinishLoadShard(ctx, CodecPayloadMode::INVALID, true);
            return;
        }

        UC_DEBUG("COMPRESS LOAD | task_id: {}, shard: {}, decompressed_size: {}", taskId,
                 shard.index, shardSize_);
        FinishLoadShard(ctx, payloadMode, true);
        return;
    } catch (const std::exception& e) {
        UC_ERROR(
            "COMPRESS LOAD FAILED | task_id: {}, shard offset: {}, stage: decompress, "
            "error: {}",
            taskId, ctx.shardIndex, e.what());
    } catch (...) {
        UC_ERROR(
            "COMPRESS LOAD FAILED | task_id: {}, shard offset: {}, stage: decompress, "
            "unknown error",
            taskId, ctx.shardIndex);
    }

    MarkTaskFailed(taskId);
    FinishLoadShard(ctx, CodecPayloadMode::INVALID, true);
}

bool CompressorAction::RegisterWork(size_t count)
{
    std::lock_guard<std::mutex> lock(lifecycleMtx_);
    const size_t outstanding = outstandingWork_;
    if (!accepting_ || outstanding > kMaxOutstandingWork ||
        count > kMaxOutstandingWork - outstanding) {
        return false;
    }
    outstandingWork_ += count;
    return true;
}

void CompressorAction::CompleteWork() noexcept
{
    std::lock_guard<std::mutex> lock(lifecycleMtx_);
    if (outstandingWork_ == 0) [[unlikely]] {
        UC_ERROR("Compressor lifecycle counter underflow");
        return;
    }
    --outstandingWork_;
    if (outstandingWork_ == 0) { lifecycleCv_.notify_all(); }
}

void CompressorAction::AcquireLoadSlot()
{
    std::unique_lock<std::mutex> lock(activeLoadMtx_);
    activeLoadCv_.wait(lock, [this] { return activeLoads_ < kMaxActiveLoads; });
    ++activeLoads_;
}

void CompressorAction::ReleaseLoadSlot() noexcept
{
    bool wasFull = false;
    {
        std::lock_guard<std::mutex> lock(activeLoadMtx_);
        if (activeLoads_ == 0) [[unlikely]] {
            UC_ERROR("Compressor load pipeline counter underflow");
            return;
        }
        wasFull = activeLoads_ == kMaxActiveLoads;
        --activeLoads_;
    }
    if (wasFull) { activeLoadCv_.notify_one(); }
}

void CompressorAction::FinishLoadShard(const LoadShardCtx& ctx, CodecPayloadMode mode,
                                       bool holdsLoadSlot) noexcept
{
    bool taskFinished = true;
    if (ctx.aggregate) {
        RecordPayloadMode(ctx.aggregate, mode);
        const size_t previous = ctx.aggregate->remaining.fetch_sub(1, std::memory_order_acq_rel);
        if (previous == 0) [[unlikely]] {
            ctx.aggregate->remaining.store(0, std::memory_order_release);
            UC_ERROR("COMPRESS LOAD duplicate completion | task_id: {}", ctx.task->id);
            return;
        }
        taskFinished = previous == 1;
    }
    if (taskFinished) {
        ReportLoadModeStats(ctx, mode);
        UC_DEBUG("COMPRESS LOAD END | task_id: {}", ctx.task->id);
    }

    if (holdsLoadSlot) { ReleaseLoadSlot(); }
    ctx.waiter->Done();

    // This must be the final access through CompressorAction. It may wake the destructor, which
    // then starts destroying the now-empty worker pools.
    CompleteWork();
}

void CompressorAction::RecordPayloadMode(const LoadAggregatePtr& aggregate,
                                         CodecPayloadMode mode) noexcept
{
    switch (mode) {
        case CodecPayloadMode::R160_HIGH_PRECISION:
            aggregate->r160HighPrecision.fetch_add(1, std::memory_order_relaxed);
            break;
        case CodecPayloadMode::R160_QUANTIZED:
            aggregate->r160Quantized.fetch_add(1, std::memory_order_relaxed);
            break;
        case CodecPayloadMode::R200_TUNSTALL:
            aggregate->r200Tunstall.fetch_add(1, std::memory_order_relaxed);
            break;
        case CodecPayloadMode::R200_FP8_FALLBACK:
            aggregate->r200Fp8Fallback.fetch_add(1, std::memory_order_relaxed);
            break;
        case CodecPayloadMode::INVALID: break;
        case CodecPayloadMode::NOT_APPLICABLE: break;
    }
}

void CompressorAction::ReportLoadModeStats(const LoadShardCtx& ctx,
                                           CodecPayloadMode mode) const noexcept
{
    CodecModeCounts counts;
    if (ctx.aggregate) {
        counts.r160.highPrecision =
            ctx.aggregate->r160HighPrecision.load(std::memory_order_acquire);
        counts.r160.quantized = ctx.aggregate->r160Quantized.load(std::memory_order_acquire);
        counts.r200.tunstall = ctx.aggregate->r200Tunstall.load(std::memory_order_acquire);
        counts.r200.fp8Fallback = ctx.aggregate->r200Fp8Fallback.load(std::memory_order_acquire);
    } else {
        counts.Add(mode);
    }
    ReportCodecModeStats(ctx.task->id, CodecStatsStage::LOAD, counts);
}

Status CompressorAction::WaitBackendHandle(Detail::TaskHandle backendHandle,
                                           Detail::TaskHandle taskId, const char* stage) noexcept
{
    if (backendHandle == 0) { return Status::InvalidParam("invalid backend task handle 0"); }
    try {
        return backend_->Wait(backendHandle);
    } catch (const std::exception& e) {
        UC_ERROR("{}: backend Wait threw for task {} handle {}: {}", stage, taskId, backendHandle,
                 e.what());
    } catch (...) {
        UC_ERROR("{}: backend Wait threw for task {} handle {}: unknown error", stage, taskId,
                 backendHandle);
    }

    // StoreV1::Wait reports operational failures through Status. Once a backend throws, there is
    // no contract guaranteeing that the I/O has stopped or that the erased handle can be waited
    // again. Releasing the Load buffer or Dump scratch memory would therefore be unsafe.
    std::terminate();
}

bool CompressorAction::IsTaskFailed(Detail::TaskHandle taskId) const noexcept
{
    if (failureSet_ == nullptr) { return false; }
    try {
        return failureSet_->Contains(taskId);
    } catch (const std::exception& e) {
        UC_ERROR("LoadPipeline: failed to query task {} state: {}", taskId, e.what());
    } catch (...) {
        UC_ERROR("LoadPipeline: failed to query task {} state: unknown error", taskId);
    }
    return true;
}

void CompressorAction::MarkTaskFailed(Detail::TaskHandle taskId) noexcept
{
    if (failureSet_ == nullptr) { return; }
    try {
        failureSet_->Insert(taskId);
    } catch (const std::exception& e) {
        UC_ERROR("LoadPipeline: failed to mark task {} as failed: {}", taskId, e.what());
    } catch (...) {
        UC_ERROR("LoadPipeline: failed to mark task {} as failed: unknown error", taskId);
    }
}

}  // namespace UC::Compressor
