#ifndef R160_BASE_BF16_H
#define R160_BASE_BF16_H

#include <cstddef>
#include <cstdint>

// Fixed-record BF16 Base15 codec.
//
// Stream layout for N BF16 values:
//   core8[N] | lane-major M32[N/4] | base[1] | exception_e8[K]
//   | optional M1/M0 | padding
//
// The caller supplies the fixed, 4-KiB-aligned record size. Code zero in
// core8 marks an exception, so K is inferred while decoding and is not stored.
// Sign, the complete E8 exponent, and M6:M2 are always retained. Remaining
// record space first stores an M1 prefix; after all M1 groups fit, it stores
// an M0 prefix. Uncovered low bits are reconstructed as zero.

// Minimum bytes before exceptions and fixed-record padding. Returns zero for
// unsupported value counts.
size_t R160BaseMinimumBytes(size_t n_bf16);

// Compresses into exactly stored_bytes. Returns R_ERR_LARGER if the exception
// stream does not fit the fixed record.
int R160BaseCompressBF16(uint8_t* dst, size_t stored_bytes, const uint16_t* src, size_t n_bf16);

// Out-of-place decompression. src_bytes may include trailing record padding.
int R160BaseDecompressBF16(const uint8_t* src, size_t src_bytes, uint16_t* dst, size_t n_bf16);

// In-place decompression. data has n_bf16*sizeof(uint16_t) bytes of capacity
// and initially contains src_bytes bytes of compressed data.
int R160BaseDecompressBF16Inplace(uint8_t* data, size_t n_bf16, size_t src_bytes);

// Lightweight stream validation used for payload-mode accounting.
bool R160BaseIsValid(const uint8_t* src, size_t src_bytes, size_t n_bf16);

#endif  // R160_BASE_BF16_H
