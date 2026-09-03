// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <type_traits>
#include <vector>
#include "r160_base_bf16.h"
#include "tunstall.h"
#include "tunstall_bf16_r160.h"
#include "tunstall_bf16_r200.h"

#if defined(__GNUC__)
#define KVFOLD_LONGBENCH_EXPORT __attribute__((visibility("default")))
#else
#define KVFOLD_LONGBENCH_EXPORT
#endif

#ifndef KVFOLD_R160_BASE_SOURCE_ID
#define KVFOLD_R160_BASE_SOURCE_ID "unknown"
#endif

#ifndef KVFOLD_TUNSTALL_R160_SOURCE_ID
#define KVFOLD_TUNSTALL_R160_SOURCE_ID "unknown"
#endif

#ifndef KVFOLD_TUNSTALL_R200_SOURCE_ID
#define KVFOLD_TUNSTALL_R200_SOURCE_ID "unknown"
#endif

namespace {

constexpr uint32_t kAbiVersion = 2;
constexpr uint32_t kCodecR160BaseBf16 = 1;
constexpr uint32_t kCodecTunstallBf16R200 = 2;
constexpr uint32_t kCodecTunstallBf16R160 = 3;
constexpr size_t kRecordAlignment = 4096;
constexpr size_t kScratchTailBytes = 4096;
constexpr int kBridgeException = -1000;

struct KvfoldLongbenchStats {
    uint32_t structSize;
    uint32_t codecId;
    uint64_t blocks;
    uint64_t values;
    uint64_t rawBytes;
    uint64_t recordBytes;
    uint64_t primaryBlocks;
    uint64_t fallbackBlocks;
    uint64_t exceptions;
    uint64_t m1Groups;
    uint64_t m0Groups;
};

static_assert(std::is_standard_layout<KvfoldLongbenchStats>::value,
              "bridge stats must have a stable C layout");

using Workspace = std::vector<uint16_t>;
using RecordBytesFunction = size_t (*)(size_t);
using RoundtripFunction = int (*)(const uint16_t*, uint16_t*, size_t, Workspace&,
                                  KvfoldLongbenchStats&);

struct CodecOps {
    uint32_t id;
    const char* name;
    const char* sourceId;
    RecordBytesFunction recordBytes;
    RoundtripFunction roundtrip;
};

size_t AlignUp(size_t value, size_t alignment)
{
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) { return 0; }
    return (value + alignment - 1) / alignment * alignment;
}

bool RawBytes(size_t nBf16, size_t& rawBytes)
{
    if (nBf16 == 0 || nBf16 > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
        return false;
    }
    rawBytes = nBf16 * sizeof(uint16_t);
    return true;
}

uint8_t* ResizeWorkspace(Workspace& workspace, size_t rawBytes)
{
    if (rawBytes > std::numeric_limits<size_t>::max() - kScratchTailBytes) { return nullptr; }
    const size_t bytes = rawBytes + kScratchTailBytes;
    workspace.resize((bytes + sizeof(uint16_t) - 1) / sizeof(uint16_t));
    return reinterpret_cast<uint8_t*>(workspace.data());
}

size_t R160BaseRecordBytes(size_t nBf16)
{
    size_t rawBytes = 0;
    if (!RawBytes(nBf16, rawBytes)) { return 0; }
    const size_t minimum = R160BaseMinimumBytes(nBf16);
    if (minimum == 0) { return 0; }
    const size_t record = AlignUp(minimum, kRecordAlignment);
    return record != 0 && record <= rawBytes ? record : 0;
}

size_t TunstallR160RecordBytes(size_t nBf16)
{
    if (nBf16 < 64 || nBf16 % 32 != 0 || nBf16 > std::numeric_limits<size_t>::max() / 5) {
        return 0;
    }
    const size_t maximum = nBf16 * 5 / 4;
    if (maximum > std::numeric_limits<uint32_t>::max()) { return 0; }
    const size_t record = maximum / kRecordAlignment * kRecordAlignment;
    return record >= nBf16 ? record : 0;
}

size_t TunstallR200RecordBytes(size_t nBf16)
{
    size_t rawBytes = 0;
    if (!RawBytes(nBf16, rawBytes) || nBf16 > std::numeric_limits<uint32_t>::max()) { return 0; }
    return nBf16;
}

int RoundtripR160Base(const uint16_t* src, uint16_t* dst, size_t nBf16, Workspace& workspace,
                      KvfoldLongbenchStats& stats)
{
    size_t rawBytes = 0;
    const size_t record = R160BaseRecordBytes(nBf16);
    if (record == 0 || !RawBytes(nBf16, rawBytes)) { return R_ERR_UNSUPPORT; }
    uint8_t* data = ResizeWorkspace(workspace, rawBytes);
    if (data == nullptr) { return R_ERR_UNSUPPORT; }

    const int compressError = R160BaseCompressBF16(data, record, src, nBf16);
    if (compressError != R_TS_OK) { return compressError; }

    size_t exceptions = 0;
    for (size_t i = 0; i < nBf16; ++i) { exceptions += static_cast<size_t>((data[i] >> 4) == 0); }
    const size_t fixed = R160BaseMinimumBytes(nBf16);
    if (fixed == 0 || exceptions > record - fixed) { return R_ERR_SYNTAX; }
    const size_t optionalBytes = record - fixed - exceptions;
    const size_t planeBytes = nBf16 / 8;
    const size_t m1 = std::min(planeBytes, optionalBytes);
    const size_t m0 = std::min(planeBytes, optionalBytes - m1);

    const int decompressError = R160BaseDecompressBF16Inplace(data, nBf16, record);
    if (decompressError != R_TS_OK) { return decompressError; }
    std::memcpy(dst, data, rawBytes);

    ++stats.primaryBlocks;
    stats.exceptions += exceptions;
    stats.m1Groups += m1;
    stats.m0Groups += m0;
    return R_TS_OK;
}

int RoundtripTunstallR160(const uint16_t* src, uint16_t* dst, size_t nBf16, Workspace& workspace,
                          KvfoldLongbenchStats& stats)
{
    size_t rawBytes = 0;
    const size_t record = TunstallR160RecordBytes(nBf16);
    if (record == 0 || !RawBytes(nBf16, rawBytes)) { return R_ERR_UNSUPPORT; }
    uint8_t* data = ResizeWorkspace(workspace, rawBytes);
    if (data == nullptr) { return R_ERR_UNSUPPORT; }

    size_t payloadBytes = record;
    const int compressError = TunstallCompressBF16R160(data, &payloadBytes, src, nBf16);
    if (compressError != R_TS_OK) { return compressError; }
    if (payloadBytes != record) { return R_ERR_SYNTAX; }

    const R160PayloadMode mode = TunstallGetBF16R160Mode(data, record, nBf16);
    if (mode == R160PayloadMode::INVALID) { return R_ERR_R160_E8_TAG; }
    if (mode == R160PayloadMode::HIGH_PRECISION) {
        ++stats.primaryBlocks;
    } else {
        ++stats.fallbackBlocks;
    }

    const int decompressError = TunstallDecompressBF16R160Inplace(data, nBf16, record);
    if (decompressError != R_TS_OK) { return decompressError; }
    std::memcpy(dst, data, rawBytes);
    return R_TS_OK;
}

int RoundtripTunstallR200(const uint16_t* src, uint16_t* dst, size_t nBf16, Workspace& workspace,
                          KvfoldLongbenchStats& stats)
{
    size_t rawBytes = 0;
    const size_t record = TunstallR200RecordBytes(nBf16);
    if (record == 0 || !RawBytes(nBf16, rawBytes)) { return R_ERR_UNSUPPORT; }
    uint8_t* data = ResizeWorkspace(workspace, rawBytes);
    if (data == nullptr) { return R_ERR_UNSUPPORT; }

    const int compressError = TunstallCompressBF16(data, src, nBf16);
    if (compressError != R_TS_OK) { return compressError; }

    const R200PayloadMode mode = TunstallGetBF16R200Mode(data, record);
    if (mode == R200PayloadMode::INVALID) { return R_ERR_SYNTAX; }
    if (mode == R200PayloadMode::TUNSTALL) {
        ++stats.primaryBlocks;
    } else {
        ++stats.fallbackBlocks;
    }

    const int decompressError = TunstallDecompressBF16Inplace(data, nBf16);
    if (decompressError != R_TS_OK) { return decompressError; }
    std::memcpy(dst, data, rawBytes);
    return R_TS_OK;
}

const CodecOps kCodecRegistry[] = {
    {kCodecR160BaseBf16,     "r160_base_bf16",     KVFOLD_R160_BASE_SOURCE_ID,     R160BaseRecordBytes,
     RoundtripR160Base                                                                                                       },
    {kCodecTunstallBf16R200, "tunstall_bf16_r200", KVFOLD_TUNSTALL_R200_SOURCE_ID,
     TunstallR200RecordBytes,                                                                           RoundtripTunstallR200},
    {kCodecTunstallBf16R160, "tunstall_bf16_r160", KVFOLD_TUNSTALL_R160_SOURCE_ID,
     TunstallR160RecordBytes,                                                                           RoundtripTunstallR160},
};

const CodecOps* FindCodec(uint32_t codecId)
{
    for (const CodecOps& codec : kCodecRegistry) {
        if (codec.id == codecId) { return &codec; }
    }
    return nullptr;
}

const char* ErrorName(int error)
{
    switch (error) {
        case R_TS_OK: return "OK";
        case R_ERR_UNSUPPORT: return "unsupported input, codec, or record size";
        case R_ERR_SYNTAX: return "invalid compressed stream";
        case R_ERR_SYMB_RANGE: return "symbol out of range";
        case R_ERR_SYMB_RANGE_PREDEF: return "preset symbol out of range";
        case R_ERR_DST_OVERFLOW: return "destination overflow";
        case R_ERR_SRC_OVERFLOW: return "source overflow";
        case R_ERR_LUT_BUILD: return "Tunstall LUT build failed";
        case R_ERR_LUT_CHECK: return "Tunstall LUT validation failed";
        case R_ERR_LUT_MISMATCH: return "Tunstall LUT mismatch";
        case R_ERR_NORMALIZE: return "Tunstall normalization failed";
        case R_ERR_PREDEF_UNINIT: return "Tunstall preset table is not initialized";
        case R_ERR_LARGER: return "R160 exception stream does not fit the fixed record";
        case R_ERR_R160_STREAM_SIZE: return "invalid Tunstall R160 stream size";
        case R_ERR_R160_E8_TAG: return "invalid Tunstall R160 mode tag";
        case R_ERR_R160_E8_METADATA: return "invalid Tunstall R160 E8 metadata";
        case R_ERR_R160_E8_EXPANSION: return "invalid Tunstall R160 E8 expansion";
        case kBridgeException: return "C++ exception inside the KVfold bridge";
        default: return "KVfold codec error";
    }
}

}  // namespace

extern "C" {

KVFOLD_LONGBENCH_EXPORT uint32_t kvfold_longbench_abi_version() { return kAbiVersion; }

KVFOLD_LONGBENCH_EXPORT const char* kvfold_longbench_codec_name(uint32_t codecId)
{
    const CodecOps* codec = FindCodec(codecId);
    return codec == nullptr ? "invalid" : codec->name;
}

KVFOLD_LONGBENCH_EXPORT const char* kvfold_longbench_codec_source_id(uint32_t codecId)
{
    const CodecOps* codec = FindCodec(codecId);
    return codec == nullptr ? "invalid" : codec->sourceId;
}

KVFOLD_LONGBENCH_EXPORT size_t kvfold_longbench_record_bytes(uint32_t codecId, size_t nBf16)
{
    const CodecOps* codec = FindCodec(codecId);
    return codec == nullptr ? 0 : codec->recordBytes(nBf16);
}

KVFOLD_LONGBENCH_EXPORT const char* kvfold_longbench_error_name(int error)
{
    return ErrorName(error);
}

KVFOLD_LONGBENCH_EXPORT int kvfold_longbench_roundtrip_bf16_blocks(uint32_t codecId,
                                                                   const uint16_t* src,
                                                                   uint16_t* dst, size_t blockCount,
                                                                   size_t valuesPerBlock,
                                                                   KvfoldLongbenchStats* stats)
{
    try {
        const CodecOps* codec = FindCodec(codecId);
        const size_t recordBytes = codec == nullptr ? 0 : codec->recordBytes(valuesPerBlock);
        if (src == nullptr || dst == nullptr || stats == nullptr ||
            reinterpret_cast<uintptr_t>(src) % alignof(uint16_t) != 0 ||
            reinterpret_cast<uintptr_t>(dst) % alignof(uint16_t) != 0 ||
            stats->structSize != sizeof(KvfoldLongbenchStats) || blockCount == 0 ||
            valuesPerBlock == 0 ||
            valuesPerBlock > std::numeric_limits<size_t>::max() / blockCount || recordBytes == 0 ||
            blockCount > std::numeric_limits<size_t>::max() / recordBytes) {
            return R_ERR_UNSUPPORT;
        }

        const size_t totalValues = blockCount * valuesPerBlock;
        if (totalValues > std::numeric_limits<uint64_t>::max() / sizeof(uint16_t) ||
            blockCount > std::numeric_limits<uint64_t>::max() / recordBytes) {
            return R_ERR_UNSUPPORT;
        }

        KvfoldLongbenchStats result{};
        result.structSize = sizeof(KvfoldLongbenchStats);
        result.codecId = codecId;
        result.blocks = blockCount;
        result.values = totalValues;
        result.rawBytes = static_cast<uint64_t>(totalValues) * sizeof(uint16_t);
        result.recordBytes = static_cast<uint64_t>(blockCount) * recordBytes;

        thread_local Workspace workspace;
        for (size_t block = 0; block < blockCount; ++block) {
            const size_t offset = block * valuesPerBlock;
            const int error =
                codec->roundtrip(src + offset, dst + offset, valuesPerBlock, workspace, result);
            if (error != R_TS_OK) { return error; }
        }

        *stats = result;
        return R_TS_OK;
    } catch (const std::exception&) {
        return kBridgeException;
    } catch (...) {
        return kBridgeException;
    }
}

}  // extern "C"

#undef KVFOLD_LONGBENCH_EXPORT
