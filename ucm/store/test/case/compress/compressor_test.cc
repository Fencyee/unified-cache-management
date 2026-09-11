/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "compress/cc/compressor.h"
#include <cstring>
#include <future>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>
#include "detail/mock_store.h"
#include "detail/types_helper.h"

namespace {

using UC::Compressor::Compressor;
using UC::Test::Detail::MockStore;

UC::Detail::Dictionary MakeConfig(UC::StoreV1* backend)
{
    UC::Detail::Dictionary config;
    config.Set("store_backend", backend);
    config.SetNumber("device_id", 0);
    config.SetNumber("shard_size", 4096);
    config.SetNumber("compressed_shard_size", 4096);
    config.SetNumber("compress_ratio", R100);
    config.SetNumber("data_type", DT_BF16);
    config.SetNumber("compress_thread_num", 1);
    config.SetNumber("decompress_thread_num", 1);
    config.SetNumber("timeout_ms", 1000);
    config.Set("compress_metrics_level", std::string("off"));
    return config;
}

UC::Detail::TaskDesc MakeTask(void* data)
{
    UC::Detail::TaskDesc task{
        {UC::Test::Detail::TypesHelper::MakeBlockIdRandomly(), 0, {data}}
    };
    task.brief = "compress-test";
    return task;
}

TEST(UCCompressorTest, RejectsMissingBackend)
{
    Compressor compressor;
    UC::Detail::Dictionary config;
    EXPECT_EQ(compressor.Setup(config), UC::Status::InvalidParam());
}

TEST(UCCompressorTest, RejectsZeroWorkerCounts)
{
    MockStore backend;
    {
        Compressor compressor;
        auto config = MakeConfig(&backend);
        config.SetNumber("compress_thread_num", 0);
        EXPECT_EQ(compressor.Setup(config), UC::Status::InvalidParam());
    }
    {
        Compressor compressor;
        auto config = MakeConfig(&backend);
        config.SetNumber("decompress_thread_num", 0);
        EXPECT_EQ(compressor.Setup(config), UC::Status::InvalidParam());
    }
}

TEST(UCCompressorTest, RejectsInvalidTransferTasks)
{
    testing::NiceMock<MockStore> backend;
    Compressor compressor;
    ASSERT_EQ(compressor.Setup(MakeConfig(&backend)), UC::Status::OK());

    EXPECT_FALSE(compressor.Load({}).HasValue());
    EXPECT_FALSE(compressor.Dump({}).HasValue());

    auto nullTask = MakeTask(nullptr);
    EXPECT_FALSE(compressor.Load(nullTask).HasValue());

    std::vector<uint8_t> data(4096);
    auto multiAddressTask = MakeTask(data.data());
    multiAddressTask.front().addrs.push_back(data.data());
    EXPECT_FALSE(compressor.Dump(std::move(multiAddressTask)).HasValue());
}

TEST(UCCompressorTest, PreservesBackendFailureStatus)
{
    testing::NiceMock<MockStore> backend;
    Compressor compressor;
    ASSERT_EQ(compressor.Setup(MakeConfig(&backend)), UC::Status::OK());

    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>{UC::Detail::TaskHandle{17}};
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::NotFound()));

    std::vector<uint8_t> data(4096);
    auto handle = compressor.Load(MakeTask(data.data()));
    ASSERT_TRUE(handle.HasValue());
    EXPECT_EQ(compressor.Wait(handle.Value()), UC::Status::NotFound());
}

TEST(UCCompressorTest, R160DumpThenLoad)
{
    constexpr size_t shardSize = 131072;
    constexpr size_t compressedSize = 86016;
    testing::NiceMock<MockStore> backend;
    auto config = MakeConfig(&backend);
    config.SetNumber("shard_size", shardSize);
    config.SetNumber("compressed_shard_size", compressedSize);
    config.SetNumber("compress_ratio", R160);
    Compressor compressor;
    ASSERT_EQ(compressor.Setup(config), UC::Status::OK());

    std::vector<uint8_t> stored(compressedSize);
    EXPECT_CALL(backend, Dump).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
        if (task.size() != 1 || task.front().addrs.size() != 1) {
            return UC::Expected<UC::Detail::TaskHandle>{UC::Status::InvalidParam()};
        }
        std::memcpy(stored.data(), task.front().addrs.front(), stored.size());
        return UC::Expected<UC::Detail::TaskHandle>{UC::Detail::TaskHandle{0}};
    }));

    std::vector<uint16_t> input(shardSize / sizeof(uint16_t), 0x3f80);
    auto dump = compressor.Dump(MakeTask(input.data()));
    ASSERT_TRUE(dump.HasValue());
    ASSERT_EQ(compressor.Wait(dump.Value()), UC::Status::OK());

    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
        if (task.size() != 1 || task.front().addrs.size() != 1) {
            return UC::Expected<UC::Detail::TaskHandle>{UC::Status::InvalidParam()};
        }
        std::memcpy(task.front().addrs.front(), stored.data(), stored.size());
        return UC::Expected<UC::Detail::TaskHandle>{UC::Detail::TaskHandle{0}};
    }));

    std::vector<uint16_t> output(shardSize / sizeof(uint16_t), 0xffff);
    auto load = compressor.Load(MakeTask(output.data()));
    ASSERT_TRUE(load.HasValue());
    ASSERT_EQ(compressor.Wait(load.Value()), UC::Status::OK());
    EXPECT_EQ(output, input);
}

TEST(UCCompressorTest, WorkerExceptionCompletesTask)
{
    testing::NiceMock<MockStore> backend;
    Compressor compressor;
    ASSERT_EQ(compressor.Setup(MakeConfig(&backend)), UC::Status::OK());

    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        throw std::runtime_error("backend exception");
        return UC::Expected<UC::Detail::TaskHandle>{UC::Detail::TaskHandle{0}};
    }));

    std::vector<uint8_t> data(4096);
    auto handle = compressor.Load(MakeTask(data.data()));
    ASSERT_TRUE(handle.HasValue());
    EXPECT_EQ(compressor.Wait(handle.Value()), UC::Status::Error());
}

TEST(UCCompressorTest, CancelsQueuedTaskOnTimeout)
{
    testing::NiceMock<MockStore> backend;
    auto config = MakeConfig(&backend);
    config.SetNumber("timeout_ms", 30);
    Compressor compressor;
    ASSERT_EQ(compressor.Setup(config), UC::Status::OK());

    std::promise<void> backendWaitEntered;
    auto entered = backendWaitEntered.get_future();
    std::promise<void> releaseBackendWait;
    auto release = releaseBackendWait.get_future();
    EXPECT_CALL(backend, Load).Times(1).WillOnce(testing::Invoke([](UC::Detail::TaskDesc) {
        return UC::Expected<UC::Detail::TaskHandle>{UC::Detail::TaskHandle{19}};
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Invoke([&](UC::Detail::TaskHandle) {
        backendWaitEntered.set_value();
        release.wait();
        return UC::Status::OK();
    }));

    std::vector<uint8_t> firstData(4096);
    std::vector<uint8_t> secondData(4096);
    auto first = compressor.Load(MakeTask(firstData.data()));
    ASSERT_TRUE(first.HasValue());
    entered.wait();
    auto second = compressor.Load(MakeTask(secondData.data()));
    ASSERT_TRUE(second.HasValue());

    EXPECT_EQ(compressor.Wait(second.Value()), UC::Status::Timeout());
    releaseBackendWait.set_value();
    for (size_t i = 0; i < 100 && !compressor.Check(first.Value()).Value(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(compressor.Check(first.Value()).Value());
    EXPECT_EQ(compressor.Wait(first.Value()), UC::Status::OK());
}

TEST(UCCompressorTest, ForwardsHealthCheck)
{
    testing::NiceMock<MockStore> backend;
    Compressor compressor;
    ASSERT_EQ(compressor.Setup(MakeConfig(&backend)), UC::Status::OK());

    EXPECT_CALL(backend, CheckHealth).WillOnce(testing::Return(UC::Status::StoreUnhealthy()));
    EXPECT_EQ(compressor.CheckHealth(), UC::Status::StoreUnhealthy());
}

}  // namespace
