#include "compress/cc/compressor_action.h"
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <tuple>
#include <gtest/gtest.h>
#include "detail/mock_store.h"

namespace {
using namespace std::chrono_literals;
using UC::Compressor::CompressorAction;
using UC::Compressor::TransTask;

// Hold the first read indefinitely, but complete the second immediately. Use
// the real R160 codec so a completed latch also proves decoding has finished.
class CompletionTest : public testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(CompletionTest, ReadyShardBypassesUnfinishedFirstRead)
{
    constexpr size_t rawBytes = 65536;
    constexpr size_t storedBytes = 40960;
    auto codec = UC::Compressor::MakeCodec(R160, DT_BF16, storedBytes);
    ASSERT_NE(codec, nullptr);
    std::vector<uint16_t> source(rawBytes/2, 0x3f80);
    std::vector<uint16_t> compressed(rawBytes/2, 0);
    ASSERT_EQ(codec->Compress(compressed.data(), source.data(), rawBytes), storedBytes);
    auto expected = compressed;
    ASSERT_EQ(codec->DecompressInplace(expected.data(), rawBytes), 0);
    std::vector<uint16_t> slow(rawBytes/2), fast(rawBytes/2);
    testing::NiceMock<UC::Test::Detail::MockStore> backend;
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::atomic<size_t> slowWaits{0}, fastWaits{0};
    std::promise<void> drainStarted;
    auto drainFuture = drainStarted.get_future();
    const int mode = std::get<0>(GetParam()); // 0: pending, 1: Check error, 2: timeout, 3: failed ready I/O
    ON_CALL(backend, Load(testing::_)).WillByDefault([&](UC::Detail::TaskDesc task)
        -> UC::Expected<UC::Detail::TaskHandle> {
        std::memcpy(task[0].addrs[0], compressed.data(), storedBytes);
        return static_cast<UC::Detail::TaskHandle>(task[0].index + 1);
    });
    ON_CALL(backend, Check(testing::_)).WillByDefault([&](auto id) -> UC::Expected<bool> {
        if (id == 2) { return true; }
        if (mode == 1) { return UC::Status::Error(); }
        std::lock_guard<std::mutex> lock(mutex);
        return bool(release);
    });
    ON_CALL(backend, Wait(testing::_)).WillByDefault([&](auto id) {
        if (id == 2) {
            ++fastWaits;
            return mode == 3 ? UC::Status::Error() : UC::Status::OK();
        }
        if (slowWaits.fetch_add(1) == 0) { drainStarted.set_value(); }
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return release; });
        return mode == 2 ? UC::Status::Timeout() : UC::Status::OK();
    });
    UC::HashSet<UC::Detail::TaskHandle> failures;
    auto action = std::make_unique<CompressorAction>();
    UC::Compressor::Config config;
    config.storeBackend = &backend;
    config.shardSize = rawBytes;
    config.compressedShardSize = storedBytes;
    config.compressRatio = R160;
    config.dataType = DT_BF16;
    config.streamNumber = 2;
    config.decompressThreadNum = 1;
    config.timeoutMs = mode == 2 ? 10 : 0;
    const int experiment = std::get<1>(GetParam());
    config.readyPollUs = experiment == 1 ? 0 : 10;
    config.maxActiveLoads = experiment == 2 ? 24 : 128;
    ASSERT_EQ(action->Setup(config, &failures), UC::Status::OK());
    auto makeTask = [](void* data, size_t index) {
        UC::Detail::TaskDesc desc{{UC::Detail::BlockId{}, index, {data}}};
        return std::make_shared<TransTask>(TransTask::Type::LOAD, std::move(desc));
    };
    auto slowTask = makeTask(slow.data(), 0);
    auto fastTask = makeTask(fast.data(), 1);
    auto slowLatch = std::make_shared<UC::Latch>();
    auto fastLatch = std::make_shared<UC::Latch>();
    action->Push(slowTask, slowLatch);
    if (mode == 1 || mode == 2) {
        // Verify even a blocking recovery drain cannot stall healthy completions.
        EXPECT_EQ(drainFuture.wait_for(2s), std::future_status::ready);
    }
    action->Push(fastTask, fastLatch);
    const bool fastFinished = fastLatch->WaitForDuration(2000);
    EXPECT_TRUE(fastFinished);
    EXPECT_FALSE(slowLatch->Check());
    if (fastFinished) {
        if (mode == 3) {
            EXPECT_TRUE(failures.Contains(fastTask->id));
            EXPECT_EQ(fast, compressed); // Failed I/O must never enter the decoder.
        } else {
            EXPECT_EQ(fast, expected);
        }
    }
    if (mode == 0 || mode == 3) { EXPECT_EQ(slowWaits.load(), 0U); }
    auto shutdown = std::async(std::launch::async, [&] { action.reset(); });
    EXPECT_EQ(shutdown.wait_for(20ms), std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
    }
    cv.notify_all();
    EXPECT_EQ(shutdown.wait_for(2s), std::future_status::ready);
    shutdown.get();
    EXPECT_TRUE(slowLatch->Check());
    EXPECT_EQ(slowWaits.load(), 1U);
    EXPECT_EQ(fastWaits.load(), 1U);
    if (mode == 2) { EXPECT_TRUE(failures.Contains(slowTask->id)); }
}
INSTANTIATE_TEST_SUITE_P(CompletionPaths, CompletionTest, testing::Combine(testing::Values(0, 1, 2, 3), testing::Values(0, 1, 2)));
TEST(CompletionConfigTest, RejectsInvalidPollingLimits)
{
    testing::NiceMock<UC::Test::Detail::MockStore> backend;
    UC::HashSet<UC::Detail::TaskHandle> failures;
    UC::Compressor::Config config;
    config.storeBackend = &backend;
    for (size_t limit : {size_t(0), size_t(8193), size_t(-1)}) {
        CompressorAction action;
        config.maxActiveLoads = limit;
        EXPECT_EQ(action.Setup(config, &failures), UC::Status::InvalidParam());
    }
    config.maxActiveLoads = 128;
    config.readyPollUs = 1000001;
    CompressorAction action;
    EXPECT_EQ(action.Setup(config, &failures), UC::Status::InvalidParam());
}
} // namespace
