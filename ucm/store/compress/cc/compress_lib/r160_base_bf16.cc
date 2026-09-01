#include "r160_base_bf16.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "tunstall.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr size_t kValuesPerTile = 32;
constexpr size_t kValuesPerOptionalGroup = 8;
constexpr size_t kRangeLength = 15;
constexpr size_t kMetadataBytes = 1;
constexpr size_t kRecordAlignment = 4096;

using M1ExpandLut = std::array<std::array<uint16_t, 8>, 256>;

M1ExpandLut BuildM1ExpandLut()
{
    M1ExpandLut lut{};
    for (size_t packed = 0; packed < lut.size(); ++packed) {
        for (size_t lane = 0; lane < kValuesPerOptionalGroup; ++lane) {
            lut[packed][lane] = static_cast<uint16_t>(((packed >> (7 - lane)) & 1U) << 1);
        }
    }
    return lut;
}

alignas(64) const M1ExpandLut M1_EXPAND_LUT = BuildM1ExpandLut();
static_assert(sizeof(M1ExpandLut) == 4096, "M1 expansion LUT must remain 4 KiB");

class AlignedScratch {
public:
    ~AlignedScratch() { std::free(data_); }

    bool Ensure(size_t bytes)
    {
        if (capacity_ >= bytes) { return true; }
        void* replacement = nullptr;
        if (posix_memalign(&replacement, kRecordAlignment, bytes) != 0) { return false; }
        std::free(data_);
        data_ = static_cast<uint8_t*>(replacement);
        capacity_ = bytes;
        return true;
    }

    uint8_t* Data() { return data_; }

private:
    uint8_t* data_{nullptr};
    size_t capacity_{0};
};

bool FixedBytes(size_t n_bf16, size_t& bytes)
{
    if (n_bf16 == 0 || n_bf16 % kValuesPerTile != 0 ||
        n_bf16 > (std::numeric_limits<size_t>::max() - kMetadataBytes) / 5 * 4) {
        return false;
    }
    bytes = n_bf16 + n_bf16 / 4 + kMetadataBytes;
    return true;
}

bool ValidRecordSize(size_t n_bf16, size_t stored_bytes, size_t& fixed_bytes)
{
    return FixedBytes(n_bf16, fixed_bytes) && stored_bytes >= fixed_bytes &&
           n_bf16 <= std::numeric_limits<size_t>::max() / sizeof(uint16_t) &&
           stored_bytes <= n_bf16 * sizeof(uint16_t);
}

uint8_t Exponent(uint16_t value) { return static_cast<uint8_t>(value >> 7); }

uint8_t SignMantissa4(uint16_t value)
{
    return static_cast<uint8_t>(((value >> 12) & 0x08U) | ((value >> 4) & 0x07U));
}

uint16_t LoadLE16(const uint8_t* src)
{
    return static_cast<uint16_t>(src[0]) | static_cast<uint16_t>(src[1]) << 8;
}

void StoreLE16(uint8_t* dst, uint16_t value)
{
    dst[0] = static_cast<uint8_t>(value);
    dst[1] = static_cast<uint8_t>(value >> 8);
}

size_t OptionalPlaneBytes(size_t n_bf16) { return n_bf16 / kValuesPerOptionalGroup; }

void OptionalCoverage(size_t n_bf16, size_t optional_bytes, size_t& m1_groups, size_t& m0_groups)
{
    const size_t plane_bytes = OptionalPlaneBytes(n_bf16);
    m1_groups = std::min(plane_bytes, optional_bytes);
    m0_groups = std::min(plane_bytes, optional_bytes - m1_groups);
}

uint8_t PackM1Group(const uint16_t* src)
{
    uint8_t packed = 0;
    for (size_t lane = 0; lane < kValuesPerOptionalGroup; ++lane) {
        packed = static_cast<uint8_t>(packed |
                                      static_cast<uint8_t>(((src[lane] >> 1) & 1U) << (7 - lane)));
    }
    return packed;
}

// Prefix groups with both M1 and M0 use lane-major M10 tiles. The rest of
// the covered prefix stores one M1 bit-plane byte per eight BF16 values.
void PackOptionalBits(uint8_t* dst, const uint16_t* src, size_t m1_groups, size_t m0_groups)
{
    const size_t full_m10_groups = m0_groups & ~size_t{3};
    for (size_t group = 0; group < full_m10_groups; group += 4) {
        uint8_t* const tile = dst + group * 2;
        for (size_t lane = 0; lane < kValuesPerOptionalGroup; ++lane) {
            uint8_t packed = 0;
            for (size_t tile_group = 0; tile_group < 4; ++tile_group) {
                packed = static_cast<uint8_t>(
                    packed | static_cast<uint8_t>(
                                 src[(group + tile_group) * kValuesPerOptionalGroup + lane] & 0x03U)
                                 << (tile_group * 2));
            }
            tile[lane] = packed;
        }
    }
    for (size_t group = full_m10_groups; group < m0_groups; ++group) {
        uint16_t packed = 0;
        for (size_t lane = 0; lane < kValuesPerOptionalGroup; ++lane) {
            packed = static_cast<uint16_t>(
                packed | static_cast<uint16_t>(src[group * kValuesPerOptionalGroup + lane] & 0x03U)
                             << (lane * 2));
        }
        StoreLE16(dst + group * 2, packed);
    }

    uint8_t* const m1_only = dst + m0_groups * 2;
    for (size_t group = m0_groups; group < m1_groups; ++group) {
        m1_only[group - m0_groups] = PackM1Group(src + group * kValuesPerOptionalGroup);
    }
}

void ApplyM10Tile(uint16_t* dst, const uint8_t* optional)
{
#if defined(__aarch64__)
    const uint16x8_t packed = vmovl_u8(vld1_u8(optional));
    const uint16x8_t mask = vdupq_n_u16(0x0003U);
    const uint16x8_t value0 = vorrq_u16(vld1q_u16(dst), vandq_u16(packed, mask));
    const uint16x8_t value1 =
        vorrq_u16(vld1q_u16(dst + 8), vandq_u16(vshrq_n_u16(packed, 2), mask));
    const uint16x8_t value2 =
        vorrq_u16(vld1q_u16(dst + 16), vandq_u16(vshrq_n_u16(packed, 4), mask));
    const uint16x8_t value3 = vorrq_u16(vld1q_u16(dst + 24), vshrq_n_u16(packed, 6));
    vst1q_u16(dst, value0);
    vst1q_u16(dst + 8, value1);
    vst1q_u16(dst + 16, value2);
    vst1q_u16(dst + 24, value3);
#elif defined(__AVX2__)
    const __m128i packed8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(optional));
    const __m128i packed = _mm_cvtepu8_epi16(packed8);
    const __m128i mask = _mm_set1_epi16(0x0003);
    const __m128i value0 = _mm_or_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dst)),
                                        _mm_and_si128(packed, mask));
    const __m128i value1 = _mm_or_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + 8)),
                                        _mm_and_si128(_mm_srli_epi16(packed, 2), mask));
    const __m128i value2 = _mm_or_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + 16)),
                                        _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    const __m128i value3 = _mm_or_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + 24)),
                                        _mm_srli_epi16(packed, 6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8), value1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), value2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), value3);
#else
    for (size_t group = 0; group < 4; ++group) {
        for (size_t lane = 0; lane < kValuesPerOptionalGroup; ++lane) {
            dst[group * kValuesPerOptionalGroup + lane] =
                static_cast<uint16_t>(dst[group * kValuesPerOptionalGroup + lane] |
                                      ((optional[lane] >> (group * 2)) & 0x03U));
        }
    }
#endif
}

void ApplyM1Group(uint16_t* dst, uint8_t packed)
{
#if defined(__aarch64__)
    uint16x8_t value = vld1q_u16(dst);
    value = vorrq_u16(value, vld1q_u16(M1_EXPAND_LUT[packed].data()));
    vst1q_u16(dst, value);
#elif defined(__AVX2__)
    __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst));
    value = _mm_or_si128(
        value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(M1_EXPAND_LUT[packed].data())));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value);
#else
    for (size_t lane = 0; lane < kValuesPerOptionalGroup; ++lane) {
        dst[lane] = static_cast<uint16_t>(dst[lane] | M1_EXPAND_LUT[packed][lane]);
    }
#endif
}

void ApplyOptionalBits(uint16_t* dst, const uint8_t* optional, size_t m1_groups, size_t m0_groups)
{
    size_t group = 0;
    const size_t full_m10_groups = m0_groups & ~size_t{3};
    for (; group < full_m10_groups; group += 4) {
        ApplyM10Tile(dst + group * kValuesPerOptionalGroup, optional + group * 2);
    }
    for (; group < m0_groups; ++group) {
        const uint16_t packed = LoadLE16(optional + group * 2);
        for (size_t lane = 0; lane < kValuesPerOptionalGroup; ++lane) {
            dst[group * kValuesPerOptionalGroup + lane] = static_cast<uint16_t>(
                dst[group * kValuesPerOptionalGroup + lane] | ((packed >> (lane * 2)) & 0x03U));
        }
    }

    const uint8_t* const m1_only = optional + m0_groups * 2;
    for (; group < m1_groups; ++group) {
        ApplyM1Group(dst + group * kValuesPerOptionalGroup, m1_only[group - m0_groups]);
    }
}

uint8_t SelectBase(const uint16_t* src, size_t n_bf16, size_t& exception_bytes)
{
    std::array<size_t, 256> frequency{};
    for (size_t i = 0; i < n_bf16; ++i) { ++frequency[Exponent(src[i])]; }

    size_t window = 0;
    for (size_t exponent = 1; exponent <= kRangeLength; ++exponent) {
        window += frequency[exponent];
    }
    size_t best_count = window;
    uint8_t best_base = 1;
    // Keep zero/subnormal and Inf/NaN classes explicit. The selected normal
    // range is always wholly contained in [1, 254].
    for (size_t base = 2; base + kRangeLength - 1 <= 254; ++base) {
        window -= frequency[base - 1];
        window += frequency[base + kRangeLength - 1];
        if (window > best_count) {
            best_count = window;
            best_base = static_cast<uint8_t>(base);
        }
    }
    exception_bytes = n_bf16 - best_count;
    return best_base;
}

void PackCoreAndMantissa(uint8_t* core, uint8_t* m32, uint8_t* exceptions, const uint16_t* src,
                         size_t n_bf16, uint8_t base)
{
    uint8_t* exception = exceptions;
    const unsigned limit = static_cast<unsigned>(base) + kRangeLength;
    for (size_t tile = 0; tile < n_bf16; tile += kValuesPerTile) {
        for (size_t group = 0; group < 4; ++group) {
            for (size_t lane = 0; lane < 8; ++lane) {
                const size_t index = tile + group * 8 + lane;
                const uint16_t value = src[index];
                const uint8_t exponent = Exponent(value);
                uint8_t code = 0;
                if (exponent >= base && exponent < limit) {
                    code = static_cast<uint8_t>(exponent - base + 1);
                } else {
                    *exception++ = exponent;
                }
                core[index] = static_cast<uint8_t>((code << 4) | SignMantissa4(value));
            }
        }
        for (size_t lane = 0; lane < 8; ++lane) {
            uint8_t packed = 0;
            for (size_t group = 0; group < 4; ++group) {
                packed = static_cast<uint8_t>(
                    packed | static_cast<uint8_t>(((src[tile + group * 8 + lane] >> 2) & 0x03U)
                                                  << (group * 2)));
            }
            m32[tile / 4 + lane] = packed;
        }
    }
}

bool ParseStream(const uint8_t* src, size_t src_bytes, size_t n_bf16, uint8_t& base,
                 const uint8_t*& core, const uint8_t*& m32, const uint8_t*& exceptions,
                 const uint8_t*& exceptions_end)
{
    size_t fixed_bytes = 0;
    if (src == nullptr || !ValidRecordSize(n_bf16, src_bytes, fixed_bytes)) { return false; }

    core = src;
    m32 = core + n_bf16;
    const uint8_t* const metadata = m32 + n_bf16 / 4;
    base = metadata[0];
    if (base == 0 || static_cast<unsigned>(base) + kRangeLength - 1 > 254 ||
        src_bytes - fixed_bytes > n_bf16) {
        return false;
    }
    exceptions = metadata + kMetadataBytes;
    exceptions_end = src + src_bytes;
    return true;
}

#if !defined(__aarch64__) && !defined(__AVX2__)
uint16_t JoinScalar(uint8_t exponent, uint8_t sm4, uint8_t low)
{
    return static_cast<uint16_t>(
        (static_cast<uint16_t>(exponent) << 7) | (static_cast<uint16_t>(sm4 & 0x08U) << 12) |
        (static_cast<uint16_t>(sm4 & 0x07U) << 4) | (static_cast<uint16_t>(low) << 2));
}

bool DecompressScalar(const uint8_t* core, const uint8_t* m32, const uint8_t*& exceptions,
                      const uint8_t* exceptions_end, uint8_t base, uint16_t* dst, size_t n_bf16)
{
    for (size_t tile = 0; tile < n_bf16; tile += kValuesPerTile) {
        for (size_t group = 0; group < 4; ++group) {
            for (size_t lane = 0; lane < 8; ++lane) {
                const size_t index = tile + group * 8 + lane;
                const uint8_t packed = core[index];
                const uint8_t code = packed >> 4;
                uint8_t exponent = 0;
                if (code == 0) {
                    if (exceptions == exceptions_end) { return false; }
                    exponent = *exceptions++;
                } else {
                    exponent = static_cast<uint8_t>(base + code - 1);
                }
                const uint8_t low =
                    static_cast<uint8_t>((m32[tile / 4 + lane] >> (group * 2)) & 0x03U);
                dst[index] = JoinScalar(exponent, packed & 0x0fU, low);
            }
        }
    }
    return true;
}
#endif

#if defined(__aarch64__)
inline uint16x8_t Join(uint8x8_t exponent, uint8x8_t sm4, uint16x8_t mantissa_low)
{
    const uint16x8_t sm = vmovl_u8(sm4);
    uint16x8_t value = vshll_n_u8(exponent, 7);
    value = vorrq_u16(value, vshlq_n_u16(vandq_u16(sm, vdupq_n_u16(0x08U)), 12));
    value = vorrq_u16(value, vshlq_n_u16(vandq_u16(sm, vdupq_n_u16(0x07U)), 4));
    return vorrq_u16(value, mantissa_low);
}

bool DecompressNEON(const uint8_t* core, const uint8_t* m32, const uint8_t*& exceptions,
                    const uint8_t* exceptions_end, uint8_t base, uint16_t* dst, size_t n_bf16)
{
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0fU);
    const uint8x16_t base_minus_one = vdupq_n_u8(base - 1);
    const uint16x8_t m32_mask = vdupq_n_u16(0x000cU);
    for (size_t tile = 0; tile < n_bf16; tile += kValuesPerTile) {
        const uint8x16_t core0 = vld1q_u8(core + tile);
        const uint8x16_t core1 = vld1q_u8(core + tile + 16);
        const uint8x16_t index0 = vshrq_n_u8(core0, 4);
        const uint8x16_t index1 = vshrq_n_u8(core1, 4);
        const uint8x16_t exp0 = vaddq_u8(base_minus_one, index0);
        const uint8x16_t exp1 = vaddq_u8(base_minus_one, index1);
        const uint8x16_t sm0 = vandq_u8(core0, nibble_mask);
        const uint8x16_t sm1 = vandq_u8(core1, nibble_mask);
        const uint16x8_t packed = vmovl_u8(vld1_u8(m32 + tile / 4));
        const uint16x8_t m0 = vandq_u16(vshlq_n_u16(packed, 2), m32_mask);
        const uint16x8_t m1 = vandq_u16(packed, m32_mask);
        const uint16x8_t m2 = vandq_u16(vshrq_n_u16(packed, 2), m32_mask);
        const uint16x8_t m3 = vandq_u16(vshrq_n_u16(packed, 4), m32_mask);
        vst1q_u16(dst + tile, Join(vget_low_u8(exp0), vget_low_u8(sm0), m0));
        vst1q_u16(dst + tile + 8, Join(vget_high_u8(exp0), vget_high_u8(sm0), m1));
        vst1q_u16(dst + tile + 16, Join(vget_low_u8(exp1), vget_low_u8(sm1), m2));
        vst1q_u16(dst + tile + 24, Join(vget_high_u8(exp1), vget_high_u8(sm1), m3));

        const uint8x16_t zero0 = vceqq_u8(index0, vdupq_n_u8(0));
        const uint8x16_t zero1 = vceqq_u8(index1, vdupq_n_u8(0));
        const uint64x2_t bits0 = vreinterpretq_u64_u8(zero0);
        const uint64x2_t bits1 = vreinterpretq_u64_u8(zero1);
        const uint64_t any = vgetq_lane_u64(bits0, 0) | vgetq_lane_u64(bits0, 1) |
                             vgetq_lane_u64(bits1, 0) | vgetq_lane_u64(bits1, 1);
        if (any == 0) { continue; }

        for (size_t lane = 0; lane < kValuesPerTile; ++lane) {
            if ((core[tile + lane] >> 4) != 0) { continue; }
            if (exceptions == exceptions_end) { return false; }
            dst[tile + lane] = static_cast<uint16_t>((dst[tile + lane] & 0x807fU) |
                                                     (static_cast<uint16_t>(*exceptions++) << 7));
        }
    }
    return true;
}
#elif defined(__AVX2__)
inline __m128i Join(__m128i exponent, __m128i sm4, __m128i mantissa_low)
{
    const __m128i exp16 = _mm_cvtepu8_epi16(exponent);
    const __m128i sm16 = _mm_cvtepu8_epi16(sm4);
    __m128i value = _mm_slli_epi16(exp16, 7);
    value = _mm_or_si128(value, _mm_slli_epi16(_mm_and_si128(sm16, _mm_set1_epi16(0x08)), 12));
    value = _mm_or_si128(value, _mm_slli_epi16(_mm_and_si128(sm16, _mm_set1_epi16(0x07)), 4));
    return _mm_or_si128(value, mantissa_low);
}

bool DecompressAVX2(const uint8_t* core, const uint8_t* m32, const uint8_t*& exceptions,
                    const uint8_t* exceptions_end, uint8_t base, uint16_t* dst, size_t n_bf16)
{
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    const __m256i base_minus_one = _mm256_set1_epi8(static_cast<char>(base - 1));
    const __m256i zero = _mm256_setzero_si256();
    const __m128i m_mask = _mm_set1_epi16(0x000c);
    for (size_t tile = 0; tile < n_bf16; tile += kValuesPerTile) {
        const __m256i packed_core =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(core + tile));
        const __m256i index = _mm256_and_si256(_mm256_srli_epi16(packed_core, 4), nibble_mask);
        const __m256i exponent = _mm256_add_epi8(base_minus_one, index);
        const __m256i sm = _mm256_and_si256(packed_core, nibble_mask);
        const __m128i packed8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(m32 + tile / 4));
        const __m128i packed = _mm_cvtepu8_epi16(packed8);
        const __m128i m0 = _mm_and_si128(_mm_slli_epi16(packed, 2), m_mask);
        const __m128i m1 = _mm_and_si128(packed, m_mask);
        const __m128i m2 = _mm_and_si128(_mm_srli_epi16(packed, 2), m_mask);
        const __m128i m3 = _mm_and_si128(_mm_srli_epi16(packed, 4), m_mask);
        const __m128i exp0 = _mm256_castsi256_si128(exponent);
        const __m128i exp1 = _mm256_extracti128_si256(exponent, 1);
        const __m128i sm0 = _mm256_castsi256_si128(sm);
        const __m128i sm1 = _mm256_extracti128_si256(sm, 1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + tile), Join(exp0, sm0, m0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + tile + 8),
                         Join(_mm_srli_si128(exp0, 8), _mm_srli_si128(sm0, 8), m1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + tile + 16), Join(exp1, sm1, m2));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + tile + 24),
                         Join(_mm_srli_si128(exp1, 8), _mm_srli_si128(sm1, 8), m3));

        uint32_t escape_mask =
            static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(index, zero)));
        while (escape_mask != 0) {
            if (exceptions == exceptions_end) { return false; }
            const unsigned lane = static_cast<unsigned>(__builtin_ctz(escape_mask));
            dst[tile + lane] = static_cast<uint16_t>((dst[tile + lane] & 0x807fU) |
                                                     (static_cast<uint16_t>(*exceptions++) << 7));
            escape_mask &= escape_mask - 1;
        }
    }
    return true;
}
#endif

int Decompress(const uint8_t* src, size_t src_bytes, uint16_t* dst, size_t n_bf16)
{
    uint8_t base = 0;
    const uint8_t* core = nullptr;
    const uint8_t* m32 = nullptr;
    const uint8_t* exceptions = nullptr;
    const uint8_t* exceptions_end = nullptr;
    if (!ParseStream(src, src_bytes, n_bf16, base, core, m32, exceptions, exceptions_end)) {
        return R_ERR_SYNTAX;
    }

    bool ok = false;
#if defined(__aarch64__)
    ok = DecompressNEON(core, m32, exceptions, exceptions_end, base, dst, n_bf16);
#elif defined(__AVX2__)
    ok = DecompressAVX2(core, m32, exceptions, exceptions_end, base, dst, n_bf16);
#else
    ok = DecompressScalar(core, m32, exceptions, exceptions_end, base, dst, n_bf16);
#endif
    if (!ok) { return R_ERR_SYNTAX; }

    // Base15 decoding advances exceptions to the first optional byte. The
    // remaining fixed-record space first restores an M1 prefix and, once M1
    // is complete, an M0 prefix. Any uncovered low bits remain zero.
    const size_t optional_bytes = static_cast<size_t>(exceptions_end - exceptions);
    size_t m1_groups = 0;
    size_t m0_groups = 0;
    OptionalCoverage(n_bf16, optional_bytes, m1_groups, m0_groups);
    ApplyOptionalBits(dst, exceptions, m1_groups, m0_groups);
    return R_TS_OK;
}

}  // namespace

size_t R160BaseMinimumBytes(size_t n_bf16)
{
    size_t bytes = 0;
    return FixedBytes(n_bf16, bytes) ? bytes : 0;
}

bool R160BaseIsValid(const uint8_t* src, size_t src_bytes, size_t n_bf16)
{
    uint8_t base = 0;
    const uint8_t* core = nullptr;
    const uint8_t* m32 = nullptr;
    const uint8_t* exceptions = nullptr;
    const uint8_t* exceptions_end = nullptr;
    return ParseStream(src, src_bytes, n_bf16, base, core, m32, exceptions, exceptions_end);
}

int R160BaseCompressBF16(uint8_t* dst, size_t stored_bytes, const uint16_t* src, size_t n_bf16)
{
    size_t fixed_bytes = 0;
    if (dst == nullptr || src == nullptr || !ValidRecordSize(n_bf16, stored_bytes, fixed_bytes)) {
        return R_ERR_UNSUPPORT;
    }

    size_t exception_bytes = 0;
    const uint8_t base = SelectBase(src, n_bf16, exception_bytes);
    if (exception_bytes > stored_bytes - fixed_bytes) { return R_ERR_LARGER; }

    uint8_t* const core = dst;
    uint8_t* const m32 = core + n_bf16;
    uint8_t* const metadata = m32 + n_bf16 / 4;
    metadata[0] = base;
    uint8_t* const exceptions = metadata + kMetadataBytes;
    PackCoreAndMantissa(core, m32, exceptions, src, n_bf16, base);

    uint8_t* const optional = exceptions + exception_bytes;
    const size_t optional_bytes = stored_bytes - fixed_bytes - exception_bytes;
    size_t m1_groups = 0;
    size_t m0_groups = 0;
    OptionalCoverage(n_bf16, optional_bytes, m1_groups, m0_groups);
    PackOptionalBits(optional, src, m1_groups, m0_groups);

    const size_t used_bytes = fixed_bytes + exception_bytes + m1_groups + m0_groups;
    std::memset(dst + used_bytes, 0, stored_bytes - used_bytes);
    return R_TS_OK;
}

int R160BaseDecompressBF16(const uint8_t* src, size_t src_bytes, uint16_t* dst, size_t n_bf16)
{
    if (src == nullptr || dst == nullptr ||
        reinterpret_cast<uintptr_t>(dst) % alignof(uint16_t) != 0) {
        return R_ERR_SRC_OVERFLOW;
    }
    return Decompress(src, src_bytes, dst, n_bf16);
}

int R160BaseDecompressBF16Inplace(uint8_t* data, size_t n_bf16, size_t src_bytes)
{
    if (data == nullptr || reinterpret_cast<uintptr_t>(data) % alignof(uint16_t) != 0) {
        return R_ERR_SRC_OVERFLOW;
    }
    size_t fixed_bytes = 0;
    if (!ValidRecordSize(n_bf16, src_bytes, fixed_bytes)) { return R_ERR_UNSUPPORT; }

    // Forward expansion overwrites core bytes that have not yet been decoded.
    // Keep one thread-local copy of the fixed record and decode back into the
    // caller's original, 4-KiB-aligned shard buffer.
    thread_local AlignedScratch compressed_scratch;
    if (!compressed_scratch.Ensure(src_bytes)) { return R_ERR_DST_OVERFLOW; }
    std::memcpy(compressed_scratch.Data(), data, src_bytes);
    return Decompress(compressed_scratch.Data(), src_bytes, reinterpret_cast<uint16_t*>(data),
                      n_bf16);
}
